"""
Supervised training: OrientationModule (+ опционально SharedGraphModule)
на данных из replay buffer.

python train_orientation_supervised.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import pickle
import argparse
from torch.utils.data import Dataset, DataLoader, random_split

from torch_geometric.nn import GATv2Conv, global_mean_pool
from skrl.utils.spaces.torch import unflatten_tensorized_space, flatten_tensorized_space
from skrl.resources.preprocessors.torch.running_standard_scaler import RunningStandardScaler

# ═══════════════════════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════════════════════

# Пути — ПОМЕНЯЙ
BUF_PATH = "replay_buffer_500.pt"
CHECKPOINT_PATH = "/home/xiso/IsaacLab/logs/skrl/aloha_ppo_orientation/new_gt_1o/checkpoints/agent_25000.pt"
EMBEDDINGS_PATH = "source/isaaclab_tasks/isaaclab_tasks/direct/aloha/text_embeddings.pt"
OBS_SPACE_PKL = "obs_space.pkl"
SAVE_DIR = "orientation_supervised_checkpoints"

# Гиперпараметры
TRAIN_GRAPH = True          # Обучать ли SharedGraphModule вместе с OrientationModule
EPOCHS = 500
BATCH_SIZE = 256
LR_ORIENT = 1e-3
LR_GRAPH = 3e-5             # Меньше для графа (fine-tune)
WEIGHT_DECAY = 1e-5
VAL_SPLIT = 0.15
PATIENCE = 1500               # Early stopping patience
KAPPA = 70.0                # Von Mises kappa для soft labels
SCHEDULER_PATIENCE = 5      # ReduceLROnPlateau patience

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.makedirs(SAVE_DIR, exist_ok=True)

# ═══════════════════════════════════════════════════════════
# CONSTANTS
# ═══════════════════════════════════════════════════════════
NUM_NODES = 17
PER_OBJ = 24
TEXT_EMB_DIM = 16
GRAPH_EMB_DIM = 128
NUM_BINS = 36

# ═══════════════════════════════════════════════════════════
# MODEL DEFINITIONS
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

    def _get_edge_index(self, B, dev):
        key = (B, dev.index if dev.type == "cuda" else -1)
        ei = self._edge_cache.get(key)
        if ei is None or ei.device != dev:
            ei = build_star_chain_edge_index(self.num_nodes, B, dev)
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
        print("std ", img.std(), graph_emb.std())
        self.net = nn.Sequential(
            nn.Linear(img_dim + graph_emb_dim, 256), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(256, 128), nn.ReLU(), nn.Linear(128, num_bins)
        ).to(device)

    def forward(self, img, graph_emb):
        logits = self.net(torch.cat([img, graph_emb], dim=-1))
        return logits


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
# DATASET
# ═══════════════════════════════════════════════════════════

class OrientationDataset(Dataset):
    """Dataset: предобработанные (img, graph, orientation) из replay buffer."""

    def __init__(self, img: torch.Tensor, graph: torch.Tensor, orientation: torch.Tensor):
        """
        img:         (N, img_dim) — уже через preprocessor
        graph:       (N, graph_dim) — сырой
        orientation: (N, 1) — ground truth yaw
        """
        self.img = img
        self.graph = graph
        self.orientation = orientation

    def __len__(self):
        return self.img.shape[0]

    def __getitem__(self, idx):
        return self.img[idx], self.graph[idx], self.orientation[idx]


# ═══════════════════════════════════════════════════════════
# METRICS
# ═══════════════════════════════════════════════════════════

def compute_metrics(logits, gt_yaw, num_bins=NUM_BINS):
    """Вычисляет все метрики по батчу."""
    bin_size = (2 * torch.pi) / num_bins
    gt_norm = torch.atan2(torch.sin(gt_yaw), torch.cos(gt_yaw))
    bin_centers = torch.linspace(-torch.pi, torch.pi, num_bins + 1, device=logits.device)[:-1] + bin_size / 2
    labels = ((gt_norm + torch.pi) / bin_size).long().clamp(0, num_bins - 1)

    pred_bins = logits.argmax(dim=-1)
    bin_diff = torch.abs(pred_bins - labels)
    bin_diff = torch.minimum(bin_diff, num_bins - bin_diff)

    probs = F.softmax(logits, dim=-1)
    pred_angles = bin_centers[pred_bins]
    angular_error = torch.abs(torch.atan2(
        torch.sin(gt_norm - pred_angles),
        torch.cos(gt_norm - pred_angles)
    )) * 180 / torch.pi

    return {
        'acc_relaxed': (bin_diff <= 1).float().mean().item(),
        'acc_strict': (pred_bins == labels).float().mean().item(),
        'mean_error_deg': angular_error.mean().item(),
        'confidence': probs.max(dim=-1)[0].mean().item(),
        'entropy_norm': (-(probs * torch.log(probs + 1e-8)).sum(-1) / np.log(num_bins)).mean().item(),
        'labels': labels,
    }


def compute_loss(logits, gt_yaw, num_bins=NUM_BINS, kappa=KAPPA):
    """Cross-entropy с soft labels (Von Mises) — но kappa поменьше для начала."""
    bin_size = (2 * torch.pi) / num_bins
    gt_norm = torch.atan2(torch.sin(gt_yaw), torch.cos(gt_yaw))
    labels = ((gt_norm + torch.pi) / bin_size).long().clamp(0, num_bins - 1)
    
    return F.cross_entropy(logits, labels, label_smoothing=0.1)


# ═══════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════

def main():
    print(f"{'='*70}")
    print("SUPERVISED ORIENTATION TRAINING")
    print(f"{'='*70}")
    print(f"  Device: {device}")
    print(f"  Train graph: {TRAIN_GRAPH}")
    print(f"  Epochs: {EPOCHS}, Batch: {BATCH_SIZE}")
    print(f"  LR orient: {LR_ORIENT}, LR graph: {LR_GRAPH}")
    print(f"  Kappa: {KAPPA}, Val split: {VAL_SPLIT}")
    print()

    # ── 1. Observation space ──
    if os.path.exists(OBS_SPACE_PKL):
        with open(OBS_SPACE_PKL, "rb") as f:
            observation_space = pickle.load(f)
        print(f"✅ obs_space from {OBS_SPACE_PKL}")
    else:
        from skrl.envs.loaders.torch import load_isaaclab_env
        from skrl.envs.wrappers.torch import wrap_env
        env = load_isaaclab_env(task_name="Isaac-Aloha-Direct-v0", num_envs=1, headless=True, cli_args=["--enable_cameras"])
        env = wrap_env(env)
        observation_space = env.observation_space
        with open(OBS_SPACE_PKL, "wb") as f:
            pickle.dump(observation_space, f)
        print("sucess load OBS_SPACE_PKL")

    for k, v in observation_space.spaces.items():
        print(f"  {k}: shape={v.shape}")
    img_dim = observation_space["img"].shape[0]

    # ── 2. Preprocessor (frozen) ──
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)

    preprocessor = DictRunningStandardScaler(
        size=observation_space, img_space=observation_space["img"], device=device
    )
    preprocessor.load_state_dict(checkpoint["state_preprocessor"])
    preprocessor.eval()
    for p in preprocessor.parameters():
        p.requires_grad = False
    print(f"✅ preprocessor loaded & frozen")

    # ── 3. Подготовка данных из буфера ──
    print(f"\n📦 Loading buffer: {BUF_PATH}")
    buf = torch.load(BUF_PATH, map_location=device)
    states = buf["states"]
    meta = buf["_meta"]
    valid_rows = meta["valid_rows"]
    n_envs = meta["num_envs"]

    # Flatten: (rows, envs, dim) → (N, dim)
    raw_flat = states[:valid_rows].reshape(-1, states.shape[-1])
    N = raw_flat.shape[0]
    print(f"   {valid_rows} rows × {n_envs} envs = {N} samples")

    # Применяем preprocessor ко всему буферу один раз
    print(f"   Applying preprocessor...")
    all_img, all_graph, all_orient = [], [], []

    with torch.no_grad():
        bs = 1024
        for i in range(0, N, bs):
            batch = raw_flat[i : i + bs]
            batch_proc = preprocessor(batch, train=False)
            s = unflatten_tensorized_space(observation_space, batch_proc)
            all_img.append(s["img"].cpu())
            all_graph.append(s["graph"].cpu())
            all_orient.append(s["orientation"].cpu())

    all_img = torch.cat(all_img, dim=0)         # (N, img_dim)
    all_graph = torch.cat(all_graph, dim=0)      # (N, graph_dim)
    all_orient = torch.cat(all_orient, dim=0)    # (N, 1)

    print(f"   img: {all_img.shape}, graph: {all_graph.shape}, orient: {all_orient.shape}")
    print(f"   img range: [{all_img.min():.3f}, {all_img.max():.3f}]")
    print(f"   orient range: [{all_orient.min():.3f}, {all_orient.max():.3f}]")

    # Проверка: ориентация покрывает [-π, π]
    orient_flat = all_orient.squeeze()
    print(f"   orient mean: {orient_flat.mean():.3f}, std: {orient_flat.std():.3f}")

    # Гистограмма GT по бинам
    bin_size = (2 * torch.pi) / NUM_BINS
    gt_norm = torch.atan2(torch.sin(orient_flat), torch.cos(orient_flat))
    gt_bins = ((gt_norm + torch.pi) / bin_size).long().clamp(0, NUM_BINS - 1)
    bin_counts = torch.bincount(gt_bins, minlength=NUM_BINS)
    print(f"\n   GT bin distribution:")
    print(f"     min count: {bin_counts.min().item()}, max count: {bin_counts.max().item()}")
    print(f"     std: {bin_counts.float().std():.1f}, mean: {bin_counts.float().mean():.1f}")
    empty_bins = (bin_counts == 0).sum().item()
    if empty_bins > 0:
        print(f"     ⚠️  {empty_bins} empty bins!")

    # ── 4. Train/Val split ──
    dataset = OrientationDataset(all_img, all_graph, all_orient)
    val_size = int(N * VAL_SPLIT)
    train_size = N - val_size

    train_ds, val_ds = random_split(
        dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(SEED)
    )

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=0, pin_memory=True, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=0, pin_memory=True)

    print(f"\n   Train: {train_size}, Val: {val_size}")
    print(f"   Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")

    # ── 5. Модели ──
    print(f"\n🔧 Creating models...")

    shared_graph = SharedGraphModule(EMBEDDINGS_PATH).to(device)
    orientation_module = OrientationModule(img_dim, GRAPH_EMB_DIM, NUM_BINS, device=device)

    # Загрузка предобученных весов
    critic1_state = checkpoint["critic_1"]
    # sg_sd = {k[13:]: v for k, v in critic1_state.items() if k.startswith("shared_graph.")}
    # om_sd = {k[19:]: v for k, v in critic1_state.items() if k.startswith("orientation_module.")}

    # shared_graph.load_state_dict(sg_sd, strict=False)
    # orientation_module.load_state_dict(om_sd, strict=False)
    # print(f"   ✅ Loaded pretrained weights (graph: {len(sg_sd)} keys, orient: {len(om_sd)} keys)")

    # ── 6. Optimizer ──
    if TRAIN_GRAPH:
        shared_graph.train()
        orientation_module.train()
        param_groups = [
            {'params': orientation_module.parameters(), 'lr': LR_ORIENT},
            {'params': shared_graph.parameters(), 'lr': LR_GRAPH},
        ]
        total_params = sum(p.numel() for g in param_groups for p in g['params'])
    else:
        shared_graph.eval()
        for p in shared_graph.parameters():
            p.requires_grad = False
        orientation_module.train()
        param_groups = [
            {'params': orientation_module.parameters(), 'lr': LR_ORIENT},
        ]
        total_params = sum(p.numel() for p in orientation_module.parameters())

    optimizer = torch.optim.AdamW(param_groups, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)


    print(f"   Trainable params: {total_params:,}")
    print(f"   Optimizer: AdamW, Scheduler: ReduceLROnPlateau")

    # ── 7. Training loop ──
    print(f"\n{'='*70}")
    print("TRAINING")
    print(f"{'='*70}\n")

    best_val_acc = 0.0
    best_epoch = 0
    patience_counter = 0

    for epoch in range(1, EPOCHS + 1):

        # ── Train ──
        if TRAIN_GRAPH:
            shared_graph.train()
        orientation_module.train()

        train_loss_sum = 0.0
        train_acc_sum = 0.0
        train_acc_strict_sum = 0.0
        train_samples = 0

        for img_b, graph_b, orient_b in train_loader:
            img_b = img_b.to(device)
            graph_b = graph_b.to(device)
            orient_b = orient_b.to(device).squeeze(-1)  # (B,)

            # Forward
            if TRAIN_GRAPH:
                graph_emb = shared_graph(graph_b)
            else:
                with torch.no_grad():
                    graph_emb = shared_graph(graph_b)

            logits = orientation_module(img_b, graph_emb)
            loss = compute_loss(logits, orient_b)

            # Backward
            optimizer.zero_grad()
            loss.backward()
            if epoch == 1 and train_samples == 0:
                for name, p in orientation_module.named_parameters():
                    if p.grad is not None:
                        print(f"  {name}: grad norm={p.grad.norm():.6f}")
                    else:
                        print(f"  {name}: NO GRAD!")
            torch.nn.utils.clip_grad_norm_(
                [p for g in param_groups for p in g['params'] if p.requires_grad],
                max_norm=1.0
            )
            optimizer.step()

            # Metrics
            with torch.no_grad():
                m = compute_metrics(logits, orient_b)

            B = img_b.shape[0]
            train_loss_sum += loss.item() * B
            train_acc_sum += m['acc_relaxed'] * B
            train_acc_strict_sum += m['acc_strict'] * B
            train_samples += B

        train_loss = train_loss_sum / train_samples
        train_acc = train_acc_sum / train_samples
        train_acc_strict = train_acc_strict_sum / train_samples

        # ── Validation ──
        shared_graph.eval()
        orientation_module.eval()

        val_loss_sum = 0.0
        val_acc_sum = 0.0
        val_acc_strict_sum = 0.0
        val_error_sum = 0.0
        val_conf_sum = 0.0
        val_samples = 0

        with torch.no_grad():
            for img_b, graph_b, orient_b in val_loader:
                img_b = img_b.to(device)
                graph_b = graph_b.to(device)
                orient_b = orient_b.to(device).squeeze(-1)

                graph_emb = shared_graph(graph_b)
                logits = orientation_module(img_b, graph_emb)
                loss = compute_loss(logits, orient_b)
                m = compute_metrics(logits, orient_b)

                B = img_b.shape[0]
                val_loss_sum += loss.item() * B
                val_acc_sum += m['acc_relaxed'] * B
                val_acc_strict_sum += m['acc_strict'] * B
                val_error_sum += m['mean_error_deg'] * B
                val_conf_sum += m['confidence'] * B
                val_samples += B

        val_loss = val_loss_sum / val_samples
        val_acc = val_acc_sum / val_samples
        val_acc_strict = val_acc_strict_sum / val_samples
        val_error = val_error_sum / val_samples
        val_conf = val_conf_sum / val_samples

        # Scheduler step on val accuracy
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']

        # ── Logging ──
        marker = ""
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
            patience_counter = 0
            marker = " ★ BEST"

            # Save best checkpoint
            save_dict = {
                'epoch': epoch,
                'orientation_module': orientation_module.state_dict(),
                'shared_graph': shared_graph.state_dict(),
                'optimizer': optimizer.state_dict(),
                'val_acc': val_acc,
                'val_acc_strict': val_acc_strict,
                'val_error_deg': val_error,
            }
            torch.save(save_dict, os.path.join(SAVE_DIR, "best.pt"))
        else:
            patience_counter += 1

        print(
            f"Epoch {epoch:3d}/{EPOCHS} │ "
            f"Train: loss={train_loss:.4f} acc±10°={train_acc:.4f} strict={train_acc_strict:.4f} │ "
            f"Val: loss={val_loss:.4f} acc±10°={val_acc:.4f} strict={val_acc_strict:.4f} "
            f"err={val_error:.1f}° conf={val_conf:.3f} │ "
            f"lr={current_lr:.1e}{marker}"
        )

        # Early stopping
        if patience_counter >= PATIENCE:
            print(f"\n⏹  Early stopping at epoch {epoch} (no improvement for {PATIENCE} epochs)")
            break

        # Periodic save
        if epoch % 20 == 0:
            torch.save({
                'epoch': epoch,
                'orientation_module': orientation_module.state_dict(),
                'shared_graph': shared_graph.state_dict(),
                'optimizer': optimizer.state_dict(),
            }, os.path.join(SAVE_DIR, f"epoch_{epoch}.pt"))

    # ── 8. Final evaluation ──
    print(f"\n{'='*70}")
    print("FINAL EVALUATION (best model)")
    print(f"{'='*70}")

    # Load best
    best_ckpt = torch.load(os.path.join(SAVE_DIR, "best.pt"), map_location=device)
    orientation_module.load_state_dict(best_ckpt['orientation_module'])
    shared_graph.load_state_dict(best_ckpt['shared_graph'])
    orientation_module.eval()
    shared_graph.eval()

    print(f"  Best epoch: {best_ckpt['epoch']}")
    print(f"  Val acc ±10°: {best_ckpt['val_acc']:.4f}")
    print(f"  Val acc strict: {best_ckpt['val_acc_strict']:.4f}")
    print(f"  Val error: {best_ckpt['val_error_deg']:.1f}°")

    # Full dataset evaluation
    all_bin_diffs = []
    total_acc, total_strict, total_err, total_n = 0, 0, 0, 0

    with torch.no_grad():
        for img_b, graph_b, orient_b in DataLoader(dataset, batch_size=512, shuffle=False):
            img_b = img_b.to(device)
            graph_b = graph_b.to(device)
            orient_b = orient_b.to(device).squeeze(-1)

            graph_emb = shared_graph(graph_b)
            logits = orientation_module(img_b, graph_emb)
            m = compute_metrics(logits, orient_b)

            B = img_b.shape[0]
            total_acc += m['acc_relaxed'] * B
            total_strict += m['acc_strict'] * B
            total_err += m['mean_error_deg'] * B
            total_n += B

            # Bin diff
            pred = logits.argmax(-1)
            bd = torch.abs(pred - m['labels'])
            bd = torch.minimum(bd, NUM_BINS - bd)
            all_bin_diffs.append(bd.cpu())

    all_bin_diffs = torch.cat(all_bin_diffs)

    print(f"\n  Full dataset ({total_n} samples):")
    print(f"    Accuracy ±10°: {total_acc / total_n:.4f}")
    print(f"    Accuracy strict: {total_strict / total_n:.4f}")
    print(f"    Mean error: {total_err / total_n:.1f}°")

    print(f"\n  Bin diff cumulative:")
    for thr in [0, 1, 2, 3, 5, 9, 18]:
        pct = (all_bin_diffs <= thr).float().mean().item() * 100
        print(f"    ≤{thr} bins (±{thr*10}°): {pct:.1f}%")

    print(f"\n  Saved: {SAVE_DIR}/best.pt")
    print(f"{'='*70}")
    print("DONE")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()