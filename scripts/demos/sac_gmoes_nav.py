# -*- coding: utf-8 -*-
"""
SAC + отдельно обучаемые GraphEncoder и OrientationModule
---------------------------------------------------------
Архитектура:
  - GraphEncoder: CLIP text lookup + GATv2 → graph_emb (128)
  - OrientationModule: img + graph_emb → orientation angle
  - Actor/Critic: используют graph_emb и orientation через no_grad
  - GraphEncoder и OrientationModule имеют СВОИ оптимизаторы,
    обучаются из replay buffer по orientation loss
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from skrl.agents.torch.sac import SAC, SAC_DEFAULT_CONFIG
from skrl.envs.loaders.torch import load_isaaclab_env
from skrl.envs.wrappers.torch import wrap_env
from skrl.memories.torch import RandomMemory
from skrl.models.torch import DeterministicMixin, GaussianMixin, Model
from skrl.resources.preprocessors.torch import RunningStandardScaler
from skrl.trainers.torch import SequentialTrainer
from skrl.utils import set_seed
from skrl.utils.spaces.torch import unflatten_tensorized_space, flatten_tensorized_space

from torch_geometric.nn import GATv2Conv, global_mean_pool

# =====================================================================
# Константы
# =====================================================================
NUM_GRAPH_NODES = 21
PER_OBJECT_DIM = 24
TEXT_EMB_DIM = 16
GRAPH_EMB_DIM = 128
NUM_ORIENT_BINS = 36
GOAL_NODE_INDEX = 0

set_seed(42)

# =====================================================================
# Edge builder
# =====================================================================
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


# =====================================================================
# GraphEncoder (объединяет SharedGraphModule + SceneGraphGATEncoder)
# =====================================================================
class GraphEncoder(nn.Module):
    """
    graph_flat [B, N*24] → graph_emb [B, 128]

    Внутри:
      1. CLIP text lookup (frozen embeddings) → text_emb per node
      2. Node MLP: (24 + text_dim) → hidden
      3. GATv2 × num_layers
      4. Global mean pool → head → graph_emb
    """
    def __init__(
        self,
        embeddings_path: str,
        num_nodes: int = NUM_GRAPH_NODES,
        per_object_dim: int = PER_OBJECT_DIM,
        text_dim: int = TEXT_EMB_DIM,
        hidden_dim: int = 128,
        out_dim: int = GRAPH_EMB_DIM,
        num_layers: int = 2,
        heads: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.num_nodes = num_nodes
        self.per_object_dim = per_object_dim
        self.text_dim = text_dim
        self.out_dim = out_dim
        self.dropout = dropout

        # --- Frozen CLIP text lookup ---
        payload = torch.load(embeddings_path, map_location="cpu")
        self.register_buffer("name_embs", payload["name_embs"].float(), persistent=False)
        self.register_buffer("color_embs", payload["color_embs"].float(), persistent=False)
        clip_dim = self.name_embs.shape[-1]  # 512

        self.text_proj = nn.Sequential(
            nn.Linear(clip_dim, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, text_dim),
        )

        # --- GATv2 ---
        node_in = per_object_dim + text_dim
        self.node_mlp = nn.Sequential(
            nn.Linear(node_in, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
        )

        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        for _ in range(num_layers):
            self.convs.append(GATv2Conv(
                hidden_dim, hidden_dim // heads,
                heads=heads, edge_dim=None, dropout=dropout, concat=True,
            ))
            self.norms.append(nn.LayerNorm(hidden_dim))

        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, out_dim),
        )

        self._edge_cache = {}

    # --- text encoding helpers ---
    def _encode_text(self, name_idx, color_bits_or_idx):
        """name_idx: [B,N], color_bits_or_idx: [B,N] or [B,N,3] → [B,N,text_dim]"""
        if name_idx.dim() == 3:
            name_idx = name_idx.argmax(-1)
        name_idx = name_idx.long().clamp(0, self.name_embs.shape[0] - 1)

        if color_bits_or_idx.dim() == 3 and color_bits_or_idx.size(-1) == 3:
            bits = color_bits_or_idx.round().long().clamp(0, 1)
            color_idx = (bits[..., 0] * 4 + bits[..., 1] * 2 + bits[..., 2]) - 1
        else:
            color_idx = color_bits_or_idx.round().long()
        color_idx = color_idx.clamp(0, self.color_embs.shape[0] - 1)

        emb = 0.5 * (self.name_embs[name_idx] + self.color_embs[color_idx])
        return self.text_proj(emb)

    # --- edge index with cache ---
    def _get_edge_index(self, B, device):
        key = (B, device.index if device.type == "cuda" else -1)
        ei = self._edge_cache.get(key)
        if ei is None or ei.device != device:
            ei = build_star_chain_edge_index(self.num_nodes, B, device)
            self._edge_cache[key] = ei
        return ei

    def forward(self, graph_flat: torch.Tensor) -> torch.Tensor:
        """graph_flat: [B, N*24] → [B, out_dim]"""
        B = graph_flat.shape[0]
        N = self.num_nodes

        node_raw = graph_flat.view(B, N, self.per_object_dim)
        text_emb = self._encode_text(node_raw[..., 20], node_raw[..., 21:24])
        x = torch.cat([node_raw, text_emb], dim=-1).view(B * N, -1)

        x = self.node_mlp(x)
        edge_index = self._get_edge_index(B, x.device)
        batch_vec = torch.repeat_interleave(torch.arange(B, device=x.device), N)

        for conv, norm in zip(self.convs, self.norms):
            x = conv(x, edge_index)
            x = norm(x)
            x = torch.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        g = global_mean_pool(x, batch_vec)
        return self.head(g)


# =====================================================================
# OrientationModule
# =====================================================================
class OrientationModule(nn.Module):
    """img + graph_emb → orientation logits (36 bins)"""
    def __init__(self, img_dim: int, graph_emb_dim: int = GRAPH_EMB_DIM,
                 num_bins: int = NUM_ORIENT_BINS):
        super().__init__()
        self.num_bins = num_bins
        self.net = nn.Sequential(
            nn.Linear(img_dim + graph_emb_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, num_bins),
        )

        # Предвычисленные bin centers (регистрируем как buffer)
        bin_size = 2 * torch.pi / num_bins
        centers = torch.linspace(-torch.pi, torch.pi, num_bins + 1)[:-1] + bin_size / 2
        self.register_buffer("bin_centers", centers)

    def forward(self, img, graph_emb):
        """Только forward без loss. Возвращает (pred_angle [B,1], probs [B,36])."""
        logits = self.net(torch.cat([img, graph_emb], dim=-1))
        probs = F.softmax(logits, dim=-1)
        pred_angle = self.bin_centers[probs.argmax(-1)].unsqueeze(-1)
        return pred_angle, probs, logits

    def compute_loss(self, logits, probs, gt_yaw):
        """Считает loss + метрики. gt_yaw: [B] или [B,1]."""
        if gt_yaw.dim() == 2:
            gt_yaw = gt_yaw.squeeze(-1)

        gt_norm = torch.atan2(torch.sin(gt_yaw), torch.cos(gt_yaw))
        bin_size = 2 * torch.pi / self.num_bins
        labels = ((gt_norm + torch.pi) / bin_size).long().clamp(0, self.num_bins - 1)

        # Cross-entropy (стабильнее Von Mises KL)
        loss = F.cross_entropy(logits, labels, label_smoothing=0.05)

        # Метрики
        with torch.no_grad():
            pred_bins = logits.argmax(-1)
            bd = torch.abs(pred_bins - labels)
            bd = torch.minimum(bd, self.num_bins - bd)

            pred_angles = self.bin_centers[pred_bins]
            ang_err = torch.atan2(torch.sin(gt_norm - pred_angles), torch.cos(gt_norm - pred_angles))

            metrics = {
                "orient/loss": loss.item(),
                "orient/acc_relaxed": (bd <= 1).float().mean().item(),
                "orient/acc_strict": (pred_bins == labels).float().mean().item(),
                "orient/mean_error_deg": (ang_err.abs().mean() * 180 / torch.pi).item(),
                "orient/confidence": probs.max(-1)[0].mean().item(),
            }
        return loss, metrics


# =====================================================================
# Preprocessor
# =====================================================================
class DictRunningStandardScaler(nn.Module):
    """Нормализует только img, graph и orientation оставляет как есть."""
    def __init__(self, size, img_space, device=None, epsilon=1e-8, clip_threshold=5.0):
        super().__init__()
        self.full_space = size
        self.img_scaler = RunningStandardScaler(
            size=img_space, epsilon=epsilon,
            clip_threshold=clip_threshold, device=device,
        )

    def forward(self, x, train=False, inverse=False, no_grad=True):
        s = unflatten_tensorized_space(self.full_space, x)
        s["img"] = self.img_scaler(s["img"], train=train, inverse=inverse, no_grad=no_grad)
        return flatten_tensorized_space(s)


# =====================================================================
# Actor & Critic
# =====================================================================
class StochasticActor(GaussianMixin, Model):
    def __init__(self, observation_space, action_space, device,
                 graph_encoder, orient_module,
                 clip_actions=False, clip_log_std=True, min_log_std=-5, max_log_std=2):
        Model.__init__(self, observation_space, action_space, device)
        GaussianMixin.__init__(self, clip_actions, clip_log_std, min_log_std, max_log_std)

        # Не регистрируем — SAC optimizer их не увидит
        self.__dict__["graph_encoder"] = graph_encoder
        self.__dict__["orient_module"] = orient_module

        img_dim = int(observation_space["img"].shape[0])
        goal_dim = int(observation_space["goal"].shape[0])
        mlp_in = img_dim + goal_dim + GRAPH_EMB_DIM + 1  # img + graph_emb + pred_angle

        self.net = nn.Sequential(
            nn.Linear(mlp_in, 512), nn.ReLU(),
            nn.Linear(512, 256), nn.ReLU(),
            nn.Linear(256, self.num_actions), nn.Tanh(),
        )
        self.log_std_parameter = nn.Parameter(torch.zeros(self.num_actions))

    def compute(self, inputs, role):
        states = unflatten_tensorized_space(self.observation_space, inputs["states"])
        img = states["img"]
        goal = states["goal"]
        graph_flat = states["graph"]

        with torch.no_grad():
            graph_emb = self.graph_encoder(graph_flat)
            pred_angle, _, _ = self.orient_module(img, graph_emb)

        x = torch.cat([img, goal, graph_emb, pred_angle], dim=-1)
        return self.net(x), self.log_std_parameter, {}


class Critic(DeterministicMixin, Model):
    def __init__(self, observation_space, action_space, device,
                 graph_encoder, orient_module, clip_actions=False):
        Model.__init__(self, observation_space, action_space, device)
        DeterministicMixin.__init__(self, clip_actions)

        # Не регистрируем — SAC optimizer их не увидит
        self.__dict__["graph_encoder"] = graph_encoder
        self.__dict__["orient_module"] = orient_module

        img_dim = int(observation_space["img"].shape[0])
        goal_dim = int(observation_space["goal"].shape[0])
        mlp_in = img_dim + goal_dim + GRAPH_EMB_DIM + 1 + self.num_actions

        self.net = nn.Sequential(
            nn.Linear(mlp_in, 512), nn.ReLU(),
            nn.Linear(512, 256), nn.ReLU(),
            nn.Linear(256, 1),
        )

    def compute(self, inputs, role):
        states = unflatten_tensorized_space(self.observation_space, inputs["states"])
        img = states["img"]
        goal = states["goal"]
        graph_flat = states["graph"]
        actions = inputs["taken_actions"]

        with torch.no_grad():
            graph_emb = self.graph_encoder(graph_flat)
            pred_angle, _, _ = self.orient_module(img, graph_emb)

        x = torch.cat([img, goal, graph_emb, pred_angle, actions], dim=-1)
        return self.net(x), {}


# =====================================================================
# Auxiliary trainer: обучает GraphEncoder + OrientationModule из buffer
# =====================================================================
class AuxModuleTrainer:
    """
    Отдельный trainer для GraphEncoder и OrientationModule.
    Обучается из replay buffer агента, не трогает actor/critic.
    """
    def __init__(self, graph_encoder, orient_module, agent,
                 obs_space, device,
                 lr_graph=3e-4, lr_orient=1e-3,
                 batch_size=512, train_steps_per_call=2,
                 log_interval=500):
        self.graph_encoder = graph_encoder
        self.orient_module = orient_module
        self.agent = agent
        self.obs_space = obs_space
        self.device = device
        self.batch_size = batch_size
        self.train_steps = train_steps_per_call
        self.log_interval = log_interval

        # Отдельные оптимизаторы
        self.graph_optimizer = torch.optim.AdamW(
            graph_encoder.parameters(), lr=lr_graph, weight_decay=1e-4
        )
        self.orient_optimizer = torch.optim.AdamW(
            orient_module.parameters(), lr=lr_orient, weight_decay=1e-4
        )

        # Аккумулятор метрик
        self._metrics = {}
        self._metric_count = 0

    def step(self, timestep):
        """Один вызов = train_steps обновлений из replay buffer."""
        mem = self.agent.memory
        if not mem.filled and mem.memory_index < self.batch_size:
            return  # мало данных

        self.graph_encoder.train()
        self.orient_module.train()

        for _ in range(self.train_steps):
            # Сэмплируем из буфера
            sample = mem.sample(names=["states"], batch_size=self.batch_size)[0]
            raw_states = sample[0]

            # Preprocessor (без обновления статистик)
            with torch.no_grad():
                processed = self.agent._state_preprocessor(raw_states, train=False)

            s = unflatten_tensorized_space(self.obs_space, processed)
            img = s["img"]
            graph_flat = s["graph"]
            gt_yaw = s["orientation"]

            # Forward (с градиентами для обоих модулей)
            graph_emb = self.graph_encoder(graph_flat)
            pred_angle, probs, logits = self.orient_module(img, graph_emb)

            # Loss (градиенты текут и в orient, и в graph)
            loss, metrics = self.orient_module.compute_loss(logits, probs, gt_yaw)

            # Backward + step для обоих оптимизаторов
            self.graph_optimizer.zero_grad()
            self.orient_optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.graph_encoder.parameters(), 1.0)
            torch.nn.utils.clip_grad_norm_(self.orient_module.parameters(), 1.0)
            self.graph_optimizer.step()
            self.orient_optimizer.step()

            # Accumulate metrics
            for k, v in metrics.items():
                self._metrics[k] = self._metrics.get(k, 0.0) + v
            self._metric_count += 1

        # Actor/Critic используют eval-режим модулей
        self.graph_encoder.eval()
        self.orient_module.eval()

        # Логируем
        if timestep % self.log_interval == 0 and self._metric_count > 0:
            n = self._metric_count
            line = " | ".join(f"{k}: {v/n:.4f}" for k, v in sorted(self._metrics.items()))
            print(f"🧭 [{timestep}] AuxTrain ({n} steps): {line}")
            self._metrics.clear()
            self._metric_count = 0


# =====================================================================
# Environment
# =====================================================================
EVAL = False

if EVAL:
    env = load_isaaclab_env(
        task_name="Aloha_nav", num_envs=4,
        cli_args=["--enable_cameras"],
    )
else:
    env = load_isaaclab_env(
        task_name="Aloha_nav", num_envs=4,
        headless=True, cli_args=["--enable_cameras"],
    )

env = wrap_env(env)
device = env.device

print(f"Device: {device} | Envs: {env.num_envs}")
print(f"Obs space: {env.observation_space}")
print(f"Act space: {env.action_space}")

# =====================================================================
# Shared modules (один экземпляр каждого)
# =====================================================================
graph_encoder = GraphEncoder(
    embeddings_path="source/isaaclab_tasks/isaaclab_tasks/direct/aloha/text_embeddings.pt",
).to(device)

orient_module = OrientationModule(
    img_dim=env.observation_space["img"].shape[0],
).to(device)

# Начинаем в eval (actor/critic используют no_grad)
graph_encoder.eval()
orient_module.eval()

print(f"GraphEncoder params: {sum(p.numel() for p in graph_encoder.parameters()):,}")
print(f"OrientModule params: {sum(p.numel() for p in orient_module.parameters()):,}")


# =====================================================================
# Models
# =====================================================================
memory = RandomMemory(memory_size=10000, num_envs=env.num_envs, device=device)

models = {
    "policy": StochasticActor(
        env.observation_space, env.action_space, device,
        graph_encoder=graph_encoder, orient_module=orient_module,
    ),
    "critic_1": Critic(
        env.observation_space, env.action_space, device,
        graph_encoder=graph_encoder, orient_module=orient_module,
    ),
    "critic_2": Critic(
        env.observation_space, env.action_space, device,
        graph_encoder=graph_encoder, orient_module=orient_module,
    ),
    "target_critic_1": Critic(
        env.observation_space, env.action_space, device,
        graph_encoder=graph_encoder, orient_module=orient_module,
    ),
    "target_critic_2": Critic(
        env.observation_space, env.action_space, device,
        graph_encoder=graph_encoder, orient_module=orient_module,
    ),
}

# =====================================================================
# SAC config
# =====================================================================
cfg = SAC_DEFAULT_CONFIG.copy()
cfg["gradient_steps"] = 4
cfg["batch_size"] = 512
cfg["discount_factor"] = 0.99
cfg["polyak"] = 0.005
cfg["actor_learning_rate"] = 3e-4
cfg["critic_learning_rate"] = 3e-4
cfg["random_timesteps"] = 0
cfg["learning_starts"] = 100
cfg["grad_norm_clip"] = 0
cfg["learn_entropy"] = True
cfg["entropy_learning_rate"] = 5e-3
cfg["initial_entropy_value"] = 1.0

cfg["state_preprocessor"] = DictRunningStandardScaler
cfg["state_preprocessor_kwargs"] = {
    "size": env.observation_space,
    "img_space": env.observation_space["img"],
    "device": device,
}
# TODO: delete state_preprocessor and make image net image statistics

cfg["experiment"]["write_interval"] = 10
cfg["experiment"]["checkpoint_interval"] = 5000
cfg["experiment"]["directory"] = "logs/skrl/aloha_sac"

agent = SAC(
    models=models, memory=memory, cfg=cfg,
    observation_space=env.observation_space,
    action_space=env.action_space, device=device,
)

# =====================================================================
# Auxiliary trainer + callback
# =====================================================================
aux_trainer = AuxModuleTrainer(
    graph_encoder=graph_encoder,
    orient_module=orient_module,
    agent=agent,
    obs_space=env.observation_space,
    device=device,
    lr_graph=3e-4,
    lr_orient=1e-3,
    batch_size=512,
    train_steps_per_call=2,
    log_interval=50,
)

# Подключаем к post_interaction
_original_post = agent.post_interaction

def _post_with_aux(timestep, timesteps):
    _original_post(timestep, timesteps)
    if timestep > cfg["learning_starts"]:
        aux_trainer.step(timestep)

agent.post_interaction = _post_with_aux

# =====================================================================
# Checkpoint loading (опционально)
# =====================================================================
# checkpoint_path = "logs/skrl/.../checkpoints/agent_XXXX.pt"
# agent.load(checkpoint_path)

# =====================================================================
# Train / Eval
# =====================================================================
if not EVAL:
    trainer = SequentialTrainer(cfg={"timesteps": 330000}, env=env, agents=agent)
    trainer.train()
else:
    checkpoint_path = "..."
    agent.load(checkpoint_path)
    trainer = SequentialTrainer(cfg={"timesteps": 1500}, env=env, agents=agent)
    trainer.eval()