"""
Полная диагностика: загрузка буфера + модулей + прогон accuracy.
python check_buffer_accuracy.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pickle

from torch_geometric.nn import GATv2Conv, global_mean_pool
from skrl.utils.spaces.torch import unflatten_tensorized_space, flatten_tensorized_space
from skrl.resources.preprocessors.torch.running_standard_scaler import RunningStandardScaler

device = torch.device("cuda")

# ═══════════════════════════════════════════════════════════
# ПУТИ — ПОМЕНЯЙ
# ═══════════════════════════════════════════════════════════
BUF_PATH = "replay_buffer_500.pt"
CHECKPOINT_PATH = "/home/xiso/IsaacLab/logs/skrl/aloha_ppo_orientation/new_gt_1o/checkpoints/agent_25000.pt"
EMBEDDINGS_PATH = "source/isaaclab_tasks/isaaclab_tasks/direct/aloha/text_embeddings.pt"
OBS_SPACE_PKL = "obs_space.pkl"  # если есть, иначе поднимем env

# ═══════════════════════════════════════════════════════════
# КОНСТАНТЫ
# ═══════════════════════════════════════════════════════════
NUM_NODES = 17
PER_OBJ = 24
TEXT_EMB_DIM = 16
GRAPH_EMB_DIM = 128
ORIENTATION_BINS = 36
GOAL_NODE_INDEX = 0

# ═══════════════════════════════════════════════════════════
# ОПРЕДЕЛЕНИЯ МОДУЛЕЙ
# ═══════════════════════════════════════════════════════════

def build_star_chain_edge_index(num_nodes, batch_size, device, add_self_loops=True):
    N = num_nodes
    src, dst = [], []
    for i in range(1, N):
        src += [0, i]; dst += [i, 0]
    for i in range(N - 1):
        src += [i, i + 1]; dst += [i + 1, i]
    if add_self_loops:
        for i in range(N):
            src.append(i); dst.append(i)
    ei = torch.tensor([src, dst], device=device, dtype=torch.long)
    return torch.cat([ei + b * N for b in range(batch_size)], dim=1)


class SceneGraphGATEncoder(nn.Module):
    def __init__(self, num_nodes, node_in_dim, hidden_dim=128, out_dim=GRAPH_EMB_DIM,
                 num_layers=2, heads=2, dropout=0.1):
        super().__init__()
        self.num_nodes = num_nodes
        self.dropout = dropout
        self.node_mlp = nn.Sequential(nn.Linear(node_in_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim))
        convs, norms = [], []
        in_ch = hidden_dim
        for _ in range(num_layers):
            convs.append(GATv2Conv(in_ch, hidden_dim // heads, heads=heads, edge_dim=None, dropout=dropout, concat=True))
            norms.append(nn.LayerNorm(hidden_dim))
            in_ch = hidden_dim
        self.convs = nn.ModuleList(convs)
        self.norms = nn.ModuleList(norms)
        self.head = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, out_dim))
        self._edge_cache = {}

    def _get_edge_index(self, B, device):
        key = (B, device.index if device.type == "cuda" else -1)
        ei = self._edge_cache.get(key)
        if ei is None or ei.device != device:
            ei = build_star_chain_edge_index(self.num_nodes, B, device)
            self._edge_cache[key] = ei
        return ei

    def forward(self, node_feats, batch_size):
        B, N = int(batch_size), self.num_nodes
        x = self.node_mlp(node_feats)
        edge_index = self._get_edge_index(B, node_feats.device)
        batch = torch.repeat_interleave(torch.arange(B, device=node_feats.device), repeats=N)
        for conv, norm in zip(self.convs, self.norms):
            x = torch.relu(norm(conv(x, edge_index)))
            x = F.dropout(x, p=self.dropout, training=self.training)
        return self.head(global_mean_pool(x, batch))


class FrozenCLIPNameColorEncoder(nn.Module):
    def __init__(self, embeddings_path, text_dim=TEXT_EMB_DIM):
        super().__init__()
        payload = torch.load(embeddings_path, map_location="cpu")
        self.register_buffer("name_embs", payload["name_embs"].float(), persistent=False)
        self.register_buffer("color_embs", payload["color_embs"].float(), persistent=False)
        self.proj = nn.Sequential(nn.Linear(self.name_embs.shape[-1], 128), nn.ReLU(), nn.Linear(128, text_dim))

    def forward(self, name_idx, color_bits_or_idx):
        if name_idx.dim() == 3: name_idx = name_idx.argmax(dim=-1)
        name_idx = name_idx.long()
        if color_bits_or_idx.dim() == 3 and color_bits_or_idx.size(-1) == 3:
            bits = color_bits_or_idx.round().long().clamp(0, 1)
            color_idx = (bits[..., 0] * 4 + bits[..., 1] * 2 + bits[..., 2]) - 1
        else:
            color_idx = color_bits_or_idx.round().long()
        name_idx = name_idx.clamp(0, self.name_embs.shape[0] - 1)
        color_idx = color_idx.clamp(0, self.color_embs.shape[0] - 1)
        return self.proj(0.5 * (self.name_embs[name_idx] + self.color_embs[color_idx]))


class SharedGraphModule(nn.Module):
    def __init__(self, embeddings_path, num_nodes=NUM_NODES, per_object_dim=PER_OBJ, text_dim=TEXT_EMB_DIM):
        super().__init__()
        self.num_nodes = num_nodes
        self.per_object_dim = per_object_dim
        self.text_encoder = FrozenCLIPNameColorEncoder(embeddings_path, text_dim)
        self.graph_encoder = SceneGraphGATEncoder(num_nodes, per_object_dim + text_dim)

    def _build_node_features(self, graph_flat):
        B = graph_flat.shape[0]
        node_raw = graph_flat.view(B, self.num_nodes, self.per_object_dim)
        text_emb = self.text_encoder(node_raw[..., 20], node_raw[..., 21:24])
        return torch.cat([node_raw, text_emb], dim=-1).view(B * self.num_nodes, -1)

    def forward(self, graph_flat):
        B = graph_flat.shape[0]
        return self.graph_encoder(self._build_node_features(graph_flat), B)


class OrientationModule(nn.Module):
    def __init__(self, img_dim, graph_emb_dim, num_bins=36, device=None):
        super().__init__()
        self.num_bins = num_bins
        self.net = nn.Sequential(
            nn.Linear(img_dim + graph_emb_dim, 256), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(256, 128), nn.ReLU(), nn.Linear(128, num_bins)
        ).to(device)

    def forward(self, img, graph_emb, ground_truth_yaw=None):
        logits = self.net(torch.cat([img, graph_emb], dim=-1))
        probs = F.softmax(logits, dim=-1)
        outputs = {'orientation_logits': logits, 'orientation_probs': probs}
        if ground_truth_yaw is not None:
            if ground_truth_yaw.dim() == 2:
                ground_truth_yaw = ground_truth_yaw.squeeze(-1)
            bin_size = (2 * torch.pi) / self.num_bins
            gt_norm = torch.atan2(torch.sin(ground_truth_yaw), torch.cos(ground_truth_yaw))
            bin_centers = torch.linspace(-torch.pi, torch.pi, self.num_bins + 1, device=logits.device)[:-1] + bin_size / 2
            labels = ((gt_norm + torch.pi) / bin_size).long().clamp(0, self.num_bins - 1)
            pred_bins = logits.argmax(dim=-1)
            bin_diff = torch.abs(pred_bins - labels)
            bin_diff = torch.minimum(bin_diff, self.num_bins - bin_diff)
            pred_angles = bin_centers[pred_bins]
            outputs.update({
                'orientation_label': labels,
                'orientation_accuracy': (bin_diff <= 1).float().mean(),
                'orientation_accuracy_strict': (pred_bins == labels).float().mean(),
                'orientation_confidence': probs.max(dim=-1)[0].mean(),
                'orientation_entropy': (-(probs * torch.log(probs + 1e-8)).sum(-1) / np.log(self.num_bins)).mean(),
                'orientation_mean_error_deg': torch.abs(torch.atan2(
                    torch.sin(gt_norm - pred_angles), torch.cos(gt_norm - pred_angles)
                )).mean() * 180 / torch.pi,
                'bin_diff': bin_diff,
            })
        return probs, outputs


class DictRunningStandardScaler(nn.Module):
    def __init__(self, size, img_space, device=None, epsilon=1e-8, clip_threshold=5.0):
        super().__init__()
        self.full_space = size
        self.img_scaler = RunningStandardScaler(size=img_space, epsilon=epsilon, clip_threshold=clip_threshold, device=device)

    def forward(self, x, train=False, inverse=False, no_grad=True):
        s = unflatten_tensorized_space(self.full_space, x)
        s["img"] = self.img_scaler(s["img"], train=train, inverse=inverse, no_grad=no_grad)
        return flatten_tensorized_space(s)


# ═══════════════════════════════════════════════════════════
# 1. OBSERVATION SPACE
# ═══════════════════════════════════════════════════════════
import os
if os.path.exists(OBS_SPACE_PKL):
    with open(OBS_SPACE_PKL, "rb") as f:
        observation_space = pickle.load(f)
    print(f"✅ Loaded obs_space from {OBS_SPACE_PKL}")
else:
    from skrl.envs.loaders.torch import load_isaaclab_env
    from skrl.envs.wrappers.torch import wrap_env
    env = load_isaaclab_env(task_name="Isaac-Aloha-Direct-v0", num_envs=1, headless=True, cli_args=["--enable_cameras"])
    env = wrap_env(env)
    observation_space = env.observation_space
    with open(OBS_SPACE_PKL, "wb") as f:
        pickle.dump(observation_space, f)
    print(f"✅ Created obs_space, saved to {OBS_SPACE_PKL}")

print(f"Observation space keys:")
for k, v in observation_space.spaces.items():
    print(f"  {k}: shape={v.shape}")
img_dim = observation_space["img"].shape[0]

# ═══════════════════════════════════════════════════════════
# 2. ЗАГРУЗКА МОДУЛЕЙ
# ═══════════════════════════════════════════════════════════
checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)

# Shared graph
shared_graph = SharedGraphModule(EMBEDDINGS_PATH).to(device)
critic1_state = checkpoint["critic_1"]
sg_sd = {k[13:]: v for k, v in critic1_state.items() if k.startswith("shared_graph.")}
shared_graph.load_state_dict(sg_sd, strict=False)
shared_graph.eval()
print(f"✅ shared_graph loaded ({len(sg_sd)} keys)")

# Orientation module
orientation_module = OrientationModule(img_dim, GRAPH_EMB_DIM, ORIENTATION_BINS, device=device)
om_sd = {k[19:]: v for k, v in critic1_state.items() if k.startswith("orientation_module.")}
orientation_module.load_state_dict(om_sd, strict=False)
orientation_module.eval()
print(f"✅ orientation_module loaded ({len(om_sd)} keys)")

# Preprocessor
preprocessor = DictRunningStandardScaler(size=observation_space, img_space=observation_space["img"], device=device)
preprocessor.load_state_dict(checkpoint["state_preprocessor"])
preprocessor.eval()
print(f"✅ preprocessor loaded (count={preprocessor.img_scaler.current_count.item():.0f})")

# ═══════════════════════════════════════════════════════════
# 3. ЗАГРУЗКА БУФЕРА
# ═══════════════════════════════════════════════════════════
buf = torch.load(BUF_PATH, map_location=device)
states = buf["states"]
meta = buf["_meta"]
valid_rows = meta["valid_rows"]
n_envs = meta["num_envs"]
sample_2d = states[:valid_rows].reshape(-1, states.shape[-1])
num_samples = sample_2d.shape[0]
print(f"\n✅ Buffer: {valid_rows} rows × {n_envs} envs = {num_samples} samples")

# ═══════════════════════════════════════════════════════════
# 4. ПРОГОН
# ═══════════════════════════════════════════════════════════
batch_size = 256
all_acc_relaxed, all_acc_strict, all_errors_deg = [], [], []
all_confidences, all_entropies = [], []
all_bin_diffs = []

with torch.no_grad():
    for i in range(0, num_samples, batch_size):
        batch = sample_2d[i : i + batch_size]
        batch_proc = preprocessor(batch, train=False)
        s = unflatten_tensorized_space(observation_space, batch_proc)

        graph_emb = shared_graph(s["graph"])
        _, out = orientation_module(s["img"], graph_emb, ground_truth_yaw=s["orientation"])

        all_acc_relaxed.append(out["orientation_accuracy"].item())
        all_acc_strict.append(out["orientation_accuracy_strict"].item())
        all_errors_deg.append(out["orientation_mean_error_deg"].item())
        all_confidences.append(out["orientation_confidence"].item())
        all_entropies.append(out["orientation_entropy"].item())
        all_bin_diffs.append(out["bin_diff"].cpu())

all_bin_diffs = torch.cat(all_bin_diffs)

# ═══════════════════════════════════════════════════════════
# 5. РЕЗУЛЬТАТЫ
# ═══════════════════════════════════════════════════════════
print(f"\n{'='*60}")
print(f"RESULTS ({num_samples} samples)")
print(f"{'='*60}")

for name, arr in [
    ("Accuracy ±10° (relaxed)", all_acc_relaxed),
    ("Accuracy (strict)", all_acc_strict),
    ("Angular error (deg)", all_errors_deg),
    ("Confidence", all_confidences),
    ("Entropy (norm)", all_entropies),
]:
    print(f"\n  {name}:")
    print(f"    mean={np.mean(arr):.4f}  std={np.std(arr):.4f}  min={np.min(arr):.4f}  max={np.max(arr):.4f}")

print(f"\n  Bin diff cumulative:")
for thr in [0, 1, 2, 3, 5, 9, 18]:
    pct = (all_bin_diffs <= thr).float().mean().item() * 100
    print(f"    ≤{thr} bins (±{thr*10}°): {pct:.1f}%")

print(f"\n  Bin diff: mean={all_bin_diffs.float().mean():.2f}, median={all_bin_diffs.float().median():.0f}, std={all_bin_diffs.float().std():.2f}")

# ═══════════════════════════════════════════════════════════
# 6. ПО ВРЕМЕНИ
# ═══════════════════════════════════════════════════════════
print(f"\n{'='*60}")
print("BY TIME:")
print(f"{'='*60}")

for label, r0, r1 in [("First 50", 0, 50), ("Middle", valid_rows//2-25, valid_rows//2+25), ("Last 50", valid_rows-50, valid_rows)]:
    chunk = states[r0:r1].reshape(-1, states.shape[-1])
    with torch.no_grad():
        sc = unflatten_tensorized_space(observation_space, preprocessor(chunk, train=False))
        _, o = orientation_module(sc["img"], shared_graph(sc["graph"]), ground_truth_yaw=sc["orientation"])
    print(f"  {label} (rows {r0}-{r1}, {chunk.shape[0]} samples):")
    print(f"    acc±10°={o['orientation_accuracy']:.4f}  strict={o['orientation_accuracy_strict']:.4f}  err={o['orientation_mean_error_deg']:.1f}°  conf={o['orientation_confidence']:.4f}")

print(f"\n{'='*60}")
print("DONE")
print(f"{'='*60}")