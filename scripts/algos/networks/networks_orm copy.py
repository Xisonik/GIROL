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
from skrl.utils.spaces.torch import unflatten_tensorized_space, flatten_tensorized_space
from skrl.models.torch import DeterministicMixin, GaussianMixin, Model
from skrl.resources.preprocessors.torch import RunningStandardScaler
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

# for accuracy
# Внешние массивы для сбора данных при EVAL
eval_gt_angles = []
eval_pred_angles = []
eval_step_counter = 0
step = 0

def collect_orientation_data(gt, pred):
    """Внешняя функция для сбора данных об ориентации"""
    global eval_gt_angles, eval_pred_angles, eval_step_counter
    eval_gt_angles.append(gt.detach().cpu())
    eval_pred_angles.append(pred.detach().cpu())
    eval_step_counter += 1

def print_orientation_accuracy(peep=False):
    """Внешняя функция для подсчета и вывода accuracy"""
    global eval_gt_angles, eval_pred_angles, eval_step_counter, step
    step += 1

    if step > 3000 or peep:
        if not peep:
            step = 0
        if len(eval_gt_angles) == 0:
            print("No orientation data collected")
            return
        
        gt = torch.cat(eval_gt_angles, dim=0)
        pred = torch.cat(eval_pred_angles, dim=0)
        
        # Обработка углов в радианах
        error = torch.abs(gt - pred)
        error = torch.minimum(error, 2*torch.pi - error)
        
        # Accuracy при допустимой ошибке < 5 градусов (0.087 rad)
       
        # print(f"\n{'='*50}")
        # print(f"EVAL COMPLETED")
        # print(f"Total steps evaluated: {len(eval_gt_angles)}")
        # print(f"Mean error: {error.mean().item()*180/torch.pi:.2f} degrees")
        # print(f"Std error: {error.std().item()*180/torch.pi:.2f} degrees")
        # print(f"Min error: {error.min().item()*180/torch.pi:.2f} degrees")
        # print(f"Max error: {error.max().item()*180/torch.pi:.2f} degrees")
        threshold = 10.0 * torch.pi / 180.0
        accuracy_10 = (error < threshold).float().mean().item()
        # print(f"Orientation accuracy (<10°): {accuracy_10*100:.2f}%")
        threshold = 20.0 * torch.pi / 180.0
        accuracy_20 = (error < threshold).float().mean().item()
        # print(f"Orientation accuracy (<20°): {accuracy_20*100:.2f}%")
        threshold = 30.0 * torch.pi / 180.0
        accuracy_30 = (error < threshold).float().mean().item()
        # print(f"Orientation accuracy (<30°): {accuracy_30*100:.2f}%")
        # print(f"{'='*50}\n")
        
        # Очищаем данные после вывода
        if not peep:
            eval_gt_angles.clear()
            eval_pred_angles.clear()
            eval_step_counter = 0
        return accuracy_10, accuracy_20, accuracy_30

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
        env,
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

        global step
        if step > 2999:    
            with torch.no_grad():
                pred_bins = probs.argmax(-1)
                print(f"\n=== OrientationModule Debug ===")
                print(f"logits shape: {logits.shape}")
                print(f"logits mean: {logits.mean().item():.4f}, std: {logits.std().item():.4f}")
                print(f"probs max: {probs.max().item():.4f}, min: {probs.min().item():.4f}")
                print(f"predicted bins unique: {torch.unique(pred_bins)}")
                pred_angle = self.bin_centers[probs.argmax(-1)].unsqueeze(-1)
                print(f"pred_angle sample: {pred_angle[:5].flatten()}")
                print(f"bin_centers[:5]: {self.bin_centers[:5]}")
                print(f"===============================\n")

        gt_norm = torch.atan2(torch.sin(gt_yaw), torch.cos(gt_yaw))
        bin_size = 2 * torch.pi / self.num_bins
        labels = ((gt_norm + torch.pi) / bin_size).long().clamp(0, self.num_bins - 1)

        # Cross-entropy (стабильнее Von Mises KL)
        loss = F.cross_entropy(logits, labels, label_smoothing=0.05)
        
        if step > 2999:
            unique_labels, counts = torch.unique(labels, return_counts=True)
            print(f"Label distribution: {dict(zip(unique_labels.cpu().numpy(), counts.cpu().numpy()))}")
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
        mlp_in = img_dim + goal_dim + GRAPH_EMB_DIM + 1 # img + graph_emb + pred_angle

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
        gt_orientation = states["orientation"]
        

        with torch.no_grad():
            graph_emb = self.graph_encoder(graph_flat)
            pred_angle, _, _ = self.orient_module(img, graph_emb)

            if True:
                collect_orientation_data(gt_orientation, pred_angle)
                print_orientation_accuracy()

        # print("gt angle: ", gt_orientation)
        # print("angle: ", pred_angle)
        # random_orientation = (torch.rand_like(gt_orientation) * 2 * torch.pi) - torch.pi
        x = torch.cat([img, goal, graph_emb, gt_orientation], dim=-1)
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
        mlp_in = img_dim + goal_dim + GRAPH_EMB_DIM + self.num_actions + 1

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
        gt_orientation = states["orientation"]

        with torch.no_grad():
            graph_emb = self.graph_encoder(graph_flat)
            pred_angle, _, _ = self.orient_module(img, graph_emb)

        x = torch.cat([img, goal, graph_emb, gt_orientation, actions], dim=-1)
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
                 log_interval=1000):
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