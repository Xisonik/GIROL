import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F

# skrl / Isaac Lab imports
from skrl.agents.torch.sac import SAC, SAC_DEFAULT_CONFIG
from skrl.envs.loaders.torch import load_isaaclab_env
from skrl.envs.wrappers.torch import wrap_env
from skrl.memories.torch import RandomMemory
from skrl.models.torch import DeterministicMixin, GaussianMixin, Model
from skrl.resources.preprocessors.torch import RunningStandardScaler, PartialRunningStandardScaler
from skrl.trainers.torch import SequentialTrainer
from skrl.utils import set_seed
from skrl.utils.spaces.torch import unflatten_tensorized_space, flatten_tensorized_space

# GNN
from torch_geometric.nn import GATv2Conv, global_mean_pool

# ---------------------------------------------------------------------
# Глобальные настройки сцены / графа
# ---------------------------------------------------------------------

# Должно совпадать с количеством объектов в SceneManager / encode_scene_graph
NUM_GRAPH_NODES = 17          # M
PER_OBJECT_DIM = 24           # столько фич на объект из encode_scene_graph
TEXT_EMB_DIM = 16             # размер текстового эмбеддинга (имя+цвет)
GRAPH_EMB_DIM = 128           # выход графового энкодера
ORIENTATION_EMB_DIM = 4      # выход orientation модуля

# В графе "цель ↔ объекты" нужно знать индекс узла цели.
# Если цель у тебя всегда первая в encode_scene_graph — оставляй 0.
GOAL_NODE_INDEX = 0
DEBUG = True

# ---------------------------------------------------------------------
# Edge builders
# ---------------------------------------------------------------------
def build_star_chain_edge_index(
    num_nodes: int,
    batch_size: int,
    device: torch.device,
    add_self_loops: bool = True
) -> torch.Tensor:
    """
    Рёбра:
      - звезда: 0 <-> i для i=1..N-1
      - цепочка: i <-> i+1 для i=0..N-2
      - (опц.) self-loops
    """
    N = num_nodes
    src, dst = [], []

    # star 0 <-> i
    for i in range(1, N):
        src += [0, i]
        dst += [i, 0]

    # chain i <-> i+1
    for i in range(N - 1):
        src += [i, i + 1]
        dst += [i + 1, i]

    if add_self_loops:
        for i in range(N):
            src.append(i)
            dst.append(i)

    edge_index_single = torch.tensor([src, dst], device=device, dtype=torch.long)  # [2, E_single]

    edge_indices = [edge_index_single + b * N for b in range(batch_size)]
    return torch.cat(edge_indices, dim=1)  # [2, B*E_single]


def build_goal_star_edge_index(num_nodes: int, batch_size: int, device: torch.device, goal_index: int = 0,
                               add_self_loops: bool = True) -> torch.Tensor:
    """Граф 'звезда': двунаправленные рёбра между goal и каждым узлом.

    Рёбра:
      goal -> i
      i -> goal
    + (опционально) self-loop i->i для всех i.

    Возвращает edge_index формы [2, E_total] для батча из B графов.
    """
    assert 0 <= goal_index < num_nodes
    N = num_nodes
    g = goal_index

    # edges within single graph
    src = []
    dst = []
    for i in range(N):
        if i == g:
            continue
        src += [g, i]
        dst += [i, g]
    if add_self_loops:
        for i in range(N):
            src.append(i)
            dst.append(i)

    edge_index_single = torch.tensor([src, dst], device=device, dtype=torch.long)  # [2, E_single]

    # batch shift
    edge_indices = []
    for b in range(batch_size):
        edge_indices.append(edge_index_single + b * N)
    return torch.cat(edge_indices, dim=1)  # [2, B*E_single]

# ---------------------------------------------------------------------
# Encoders
# ---------------------------------------------------------------------

class SceneGraphGATEncoder(nn.Module):
    """GATv2 encoder для сценового графа без edge_attr.

    ВНИМАНИЕ: edge_index строится как 'цель ↔ объекты' (звезда), поэтому память/время намного меньше,
    чем у полносвязного графа.
    """
    def __init__(
        self,
        num_nodes: int,
        node_in_dim: int,
        hidden_dim: int = 128,
        out_dim: int = GRAPH_EMB_DIM,
        num_layers: int = 2,
        heads: int = 2,
        dropout: float = 0.1,
        goal_index: int = GOAL_NODE_INDEX,
    ):
        super().__init__()
        self.num_nodes = num_nodes
        self.out_dim = out_dim
        self.dropout = dropout
        self.goal_index = goal_index

        self.node_mlp = nn.Sequential(
            nn.Linear(node_in_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
        )

        convs = []
        norms = []
        in_ch = hidden_dim
        for _ in range(num_layers):
            convs.append(
                GATv2Conv(
                    in_channels=in_ch,
                    out_channels=hidden_dim // heads,
                    heads=heads,
                    edge_dim=None,
                    dropout=dropout,
                    concat=True,
                )
            )
            norms.append(nn.LayerNorm(hidden_dim))
            in_ch = hidden_dim

        self.convs = nn.ModuleList(convs)
        self.norms = nn.ModuleList(norms)

        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, out_dim),
        )

        # маленький кэш edge_index по (B, device)
        self._edge_cache = {}

    def prune_edges_by_distance(self, edge_index: torch.Tensor, node_pos_flat: torch.Tensor, max_dist: float) -> torch.Tensor:
        """
        edge_index: [2, E]
        node_pos_flat: [B*N, 3]
        """
        src, dst = edge_index[0], edge_index[1]
        d = node_pos_flat[src] - node_pos_flat[dst]
        dist = torch.norm(d, dim=-1)
        mask = dist <= max_dist
        return edge_index[:, mask]


    def _get_edge_index(self, B: int, device: torch.device) -> torch.Tensor:
        key = (B, device.index if device.type == "cuda" else -1)
        ei = self._edge_cache.get(key, None)
        if ei is None or ei.device != device:
            ei = build_star_chain_edge_index(self.num_nodes, B, device, add_self_loops=True)

            self._edge_cache[key] = ei
        return ei

    def forward(self, node_feats: torch.Tensor, batch_size: int) -> torch.Tensor:
        """
        node_feats: [B * N, node_in_dim]
        return:    [B, out_dim]
        """
        device = node_feats.device
        B = int(batch_size)
        N = self.num_nodes

        x = self.node_mlp(node_feats)  # [B*N, hidden]

        edge_index = self._get_edge_index(B, device)
        batch = torch.repeat_interleave(torch.arange(B, device=device), repeats=N)  # [B*N]

        for conv, norm in zip(self.convs, self.norms):
            x = conv(x, edge_index)      # [B*N, hidden]
            x = norm(x)
            x = torch.relu(x)
            x = nn.functional.dropout(x, p=self.dropout, training=self.training)

        g = global_mean_pool(x, batch)   # [B, hidden]
        return self.head(g)              # [B, out_dim]


class FrozenCLIPNameColorEncoder(nn.Module):
    """Lookup оффлайн CLIP-эмбеддингов и общий обучаемый проектор 512 -> TEXT_EMB_DIM.

    embeddings_path должен содержать:
      - name_embs:  [N_names, 512]
      - color_embs: [N_colors, 512]
    """
    def __init__(self, embeddings_path: str, text_dim: int = TEXT_EMB_DIM):
        super().__init__()
        self.text_dim = text_dim

        payload = torch.load(embeddings_path, map_location="cpu")
        name_embs = payload.get("name_embs", None)
        color_embs = payload.get("color_embs", None)
        if name_embs is None or color_embs is None:
            raise ValueError(f"Bad embeddings file: expected keys 'name_embs' and 'color_embs' in {embeddings_path}")

        self.register_buffer("name_embs", name_embs.float(), persistent=False)    # [N_names, 512]
        self.register_buffer("color_embs", color_embs.float(), persistent=False) # [N_colors, 512]

        self.proj = nn.Sequential(
            nn.Linear(self.name_embs.shape[-1], 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, text_dim),
        )

    def forward(self, name_idx: torch.Tensor, color_bits_or_idx: torch.Tensor) -> torch.Tensor:
        """
        name_idx:
          - [B, N] (индекс)  или [B, N, K] (one-hot/logits)
        color_bits_or_idx:
          - [B, N] (индекс)  или [B, N, 3] (биты 0/1 -> id 0..6)

        return: [B, N, text_dim]
        """
        if name_idx.dim() == 3:
            name_idx = name_idx.argmax(dim=-1)
        name_idx = name_idx.long()

        if color_bits_or_idx.dim() == 3 and color_bits_or_idx.size(-1) == 3:
            bits = color_bits_or_idx.round().long().clamp(0, 1)
            # (4,2,1) -> 1..7, затем -1 -> 0..6
            color_idx = (bits[..., 0] * 4 + bits[..., 1] * 2 + bits[..., 2]) - 1
        else:
            color_idx = color_bits_or_idx.round().long()

        name_idx = name_idx.clamp(0, self.name_embs.shape[0] - 1)
        color_idx = color_idx.clamp(0, self.color_embs.shape[0] - 1)

        emb_name = self.name_embs[name_idx]       # [B,N,512]
        emb_color = self.color_embs[color_idx]    # [B,N,512]
        emb = 0.5 * (emb_name + emb_color)        # [B,N,512]
        return self.proj(emb)                     # [B,N,text_dim]


class SharedGraphModule(nn.Module):
    """Общий графовый энкодер: node_raw (24) + text_emb (16) -> GAT -> graph_emb (128)."""
    def __init__(self, embeddings_path: str, num_nodes: int = NUM_GRAPH_NODES,
                 per_object_dim: int = PER_OBJECT_DIM, text_dim: int = TEXT_EMB_DIM):
        super().__init__()
        self.num_nodes = num_nodes
        self.per_object_dim = per_object_dim
        self.text_dim = text_dim

        self.text_encoder = FrozenCLIPNameColorEncoder(embeddings_path=embeddings_path, text_dim=text_dim)
        self.graph_encoder = SceneGraphGATEncoder(
            num_nodes=num_nodes,
            node_in_dim=per_object_dim + text_dim,
            hidden_dim=128,
            out_dim=GRAPH_EMB_DIM,
            num_layers=2,
            heads=2,
            dropout=0.1,
            goal_index=GOAL_NODE_INDEX,
        )

    def _build_node_features(self, graph_flat: torch.Tensor) -> torch.Tensor:
        """
        graph_flat: [B, N * 24]
        return:    [B*N, 24+text_dim]
        """
        B = graph_flat.shape[0]
        N = self.num_nodes
        node_raw = graph_flat.view(B, N, self.per_object_dim)  # [B,N,24]

        # По твоему encode_scene_graph:
        #  - name_code: index 20
        #  - color_code: 3 бита: 21:24  (или может быть индекс — encoder поддерживает оба)
        name_idx = node_raw[..., 20]
        color_bits = node_raw[..., 21:24]

        text_emb = self.text_encoder(name_idx, color_bits)     # [B,N,text_dim]
        full_node = torch.cat([node_raw, text_emb], dim=-1)    # [B,N,24+text_dim]
        return full_node.view(B * N, -1)

    def forward(self, graph_flat: torch.Tensor) -> torch.Tensor:
        """
        graph_flat: [B, N*24]
        return:    [B, GRAPH_EMB_DIM]
        """
        B = graph_flat.shape[0]
        node_feats = self._build_node_features(graph_flat)
        return self.graph_encoder(node_feats, batch_size=B)


# ---------------------------------------------------------------------
# Orientation Module (НОВЫЙ)
# ---------------------------------------------------------------------

class OrientationModule(nn.Module):
    """Предсказывает ориентацию робота через регрессию sin/cos и выдаёт embedding для Actor/Critic."""
    def __init__(self, img_dim: int, graph_emb_dim: int, 
                 emb_dim: int = ORIENTATION_EMB_DIM, device=None):
        super().__init__()
        self.device = device
        self.emb_dim = emb_dim
        
        # Основная сеть для предсказания sin и cos
        self.net = nn.Sequential(
            nn.Linear(img_dim + graph_emb_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 2)  # 2 выхода: [sin(yaw), cos(yaw)]
        ).to(device)
        
        # Проекция для embedding (используем нормализованные предсказания)
        self.embedding_proj = nn.Linear(2, emb_dim).to(device)
    
    def forward(self, img, graph_emb, ground_truth_yaw=None):
        x = torch.cat([img, graph_emb], dim=-1)
        # Предсказываем синус и косинус
        sin_cos_pred = self.net(x)
        
        # Нормализуем предсказания (приводим к единичной окружности)
        sin_cos_norm = F.normalize(sin_cos_pred, p=2, dim=-1)
        sin_pred, cos_pred = sin_cos_norm[:, 0], sin_cos_norm[:, 1]
        
        # Вычисляем предсказанный угол через арктангенс
        pred_yaw = torch.atan2(sin_pred, cos_pred)
        
        # ✅ DEBUG: Информация о предсказании
        if not hasattr(self, '_debug_counter'):
            self._debug_counter = 0
        
        self._debug_counter += 1
        if self._debug_counter % 100 == 0:  # Каждые 100 вызовов
            pred_deg = pred_yaw[0] * 180 / torch.pi
            norm = torch.sqrt(sin_pred[0]**2 + cos_pred[0]**2).item()
            print(f"[Orient] Predicted: {pred_deg:.1f}°, Norm: {norm:.3f}, "
                  f"Sin: {sin_pred[0]:.3f}, Cos: {cos_pred[0]:.3f}")
        
        # Создаём embedding из нормализованных sin/cos
        orientation_emb = self.embedding_proj(sin_cos_norm)
        
        outputs = {
            'orientation_sin_cos': sin_cos_pred,
            'orientation_sin_cos_norm': sin_cos_norm,
            'orientation_yaw': pred_yaw
        }
        
        if ground_truth_yaw is not None:
            if ground_truth_yaw.dim() == 2:
                ground_truth_yaw = ground_truth_yaw.squeeze(-1)
            
            # Нормализуем ground truth углы
            gt_normalized = torch.atan2(
                torch.sin(ground_truth_yaw),
                torch.cos(ground_truth_yaw)
            )
            
            # Ground truth синус и косинус
            gt_sin = torch.sin(gt_normalized)
            gt_cos = torch.cos(gt_normalized)
            gt_sin_cos = torch.stack([gt_sin, gt_cos], dim=-1)
            
            # Потери: MSE для синуса и косинуса
            mse_loss = F.mse_loss(sin_cos_norm, gt_sin_cos)
            
            # Дополнительная потеря на отклонение от единичной нормы
            norm_loss = F.mse_loss(
                torch.sqrt(sin_cos_pred[:, 0]**2 + sin_cos_pred[:, 1]**2),
                torch.ones_like(sin_cos_pred[:, 0])
            )
            
            # Потеря на угловую ошибку (важно для ориентации)
            angular_diff = torch.atan2(
                torch.sin(gt_normalized - pred_yaw),
                torch.cos(gt_normalized - pred_yaw)
            )
            angular_loss = torch.mean(1 - torch.cos(angular_diff))
            
            # Общая потеря (можно настроить веса)
            total_loss = mse_loss + 0.1 * norm_loss + angular_loss
            
            # ✅ МЕТРИКИ
            # Угловая ошибка в градусах
            angular_error_rad = torch.abs(angular_diff)
            mean_angular_error = angular_error_rad.mean() * 180 / torch.pi
            
            # Accuracy: считаем правильными предсказания в пределах ±10°
            accuracy_10deg = (angular_error_rad <= (10 * torch.pi / 180)).float().mean()
            
            # Accuracy: ±30°
            accuracy_30deg = (angular_error_rad <= (30 * torch.pi / 180)).float().mean()
            
            # Accuracy: ±5°
            accuracy_5deg = (angular_error_rad <= (5 * torch.pi / 180)).float().mean()
            
            # Confidence: норма предсказания (близость к единичной окружности)
            confidence = torch.sqrt(sin_pred**2 + cos_pred**2).mean()
            
            outputs.update({
                'orientation_loss': total_loss,
                'orientation_loss_mse': mse_loss,
                'orientation_loss_norm': norm_loss,
                'orientation_loss_angular': angular_loss,
                'orientation_gt_yaw': gt_normalized,
                'orientation_accuracy_10deg': accuracy_10deg,
                'orientation_accuracy_30deg': accuracy_30deg,
                'orientation_accuracy_5deg': accuracy_5deg,
                'orientation_mean_error_deg': mean_angular_error,
                'orientation_confidence': confidence,
                'orientation_angular_error_rad': angular_error_rad.mean(),
            })
        
        return orientation_emb, outputs
    
    def predict_angle(self, img, graph_emb):
        """Вспомогательный метод для инференса угла"""
        with torch.no_grad():
            x = torch.cat([img, graph_emb], dim=-1)
            sin_cos_pred = self.net(x)
            sin_cos_norm = F.normalize(sin_cos_pred, p=2, dim=-1)
            pred_yaw = torch.atan2(sin_cos_norm[:, 0], sin_cos_norm[:, 1])
            return pred_yaw, sin_cos_norm

# ---------------------------------------------------------------------
# CLI аргументы / seed
# ---------------------------------------------------------------------
EVAL = False
# EVAL = True

set_seed(42)

# ---------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------

class StochasticActor(GaussianMixin, Model):
    """Actor: использует orientation embedding (без обучения модуля)."""
    def __init__(self, observation_space, action_space, device, shared_graph: SharedGraphModule,
                 orientation_module: OrientationModule,
                 clip_actions=False, clip_log_std=True, min_log_std=-5, max_log_std=2):
        Model.__init__(self, observation_space, action_space, device)
        GaussianMixin.__init__(self, clip_actions, clip_log_std, min_log_std, max_log_std)
        self.device = device

        # НЕ обучаем shared modules
        self.__dict__["shared_graph"] = shared_graph
        self.__dict__["orientation_module"] = orientation_module

        self.img_dim = int(observation_space["img"].shape[0])

        # Policy: img + graph_emb + orientation_emb 
        mlp_in = self.img_dim + GRAPH_EMB_DIM
        self.net = nn.Sequential(
            nn.Linear(mlp_in, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, self.num_actions),
            nn.Tanh()
        ).to(device)

        self.log_std_parameter = nn.Parameter(torch.zeros(self.num_actions, device=device))
        
        if DEBUG:
            print(f"\n[StochasticActor] Initialized:")
            print(f"  img_dim={self.img_dim}")
            print(f"  mlp_in={mlp_in} (img + graph + orient)")
            print(f"  num_actions={self.num_actions}")

    def compute(self, inputs, role):
        B = inputs["states"].shape[0]
        states = unflatten_tensorized_space(self.observation_space, inputs["states"])
        img = states["img"].to(self.device)
        graph_flat = states["graph"].to(self.device)
        gt_orientation = states["orientation"].to(self.device)
        # Все модули в no_grad для actor
        with torch.no_grad():
            graph_emb = self.shared_graph(graph_flat)
            orientation_emb, _ = self.orientation_module(img, graph_emb, ground_truth_yaw=None)

        x = torch.cat([img, graph_emb], dim=-1)
        mu = self.net(x)
        
        if DEBUG and not hasattr(self, '_debug_compute_printed'):
            print(f"\n[StochasticActor.compute] First call:")
            print(f"  B={B}")
            print(f"  img: {img.shape}")
            print(f"  graph_emb: {graph_emb.shape}")
            print(f"  orientation_emb: {orientation_emb.shape}")
            print(f"  concatenated: {x.shape}")
            print(f"  mu: {mu.shape}")
            self._debug_compute_printed = True
        
        return mu, self.log_std_parameter, {}


class Critic(DeterministicMixin, Model):
    """Critic: Q(s,a) + orientation learning."""
    def __init__(self, observation_space, action_space, device, shared_graph: SharedGraphModule,
                 orientation_module: OrientationModule,
                 clip_actions=False, train_graph: bool = True, train_orientation: bool = True):
        Model.__init__(self, observation_space, action_space, device)
        DeterministicMixin.__init__(self, clip_actions)
        self.device = device

        self.train_graph = bool(train_graph)
        self.train_orientation = bool(train_orientation)

        # Shared modules registration
        if train_graph:
            # Регистрируем чтобы параметры были в optimizer
            self.shared_graph = shared_graph
        else:
            self.__dict__["shared_graph"] = shared_graph
        
        if train_orientation:
            # Регистрируем чтобы параметры были в optimizer
            self.orientation_module = orientation_module
        else:
            self.__dict__["orientation_module"] = orientation_module

        self.img_dim = int(observation_space["img"].shape[0])

        # Q-network: img + graph_emb + orientation_emb + action
        mlp_in = self.img_dim + GRAPH_EMB_DIM + self.num_actions
        self.net = nn.Sequential(
            nn.Linear(mlp_in, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        ).to(device)
        
        if DEBUG:
            print(f"\n[Critic] Initialized:")
            print(f"  train_graph={train_graph}, train_orientation={train_orientation}")
            print(f"  img_dim={self.img_dim}")
            print(f"  mlp_in={mlp_in} (img + graph + orient + action)")
            
            # Проверяем параметры
            critic_params = list(self.net.parameters())
            print(f"  Critic net params: {sum(p.numel() for p in critic_params):,}")
            
            if train_graph:
                graph_params = list(self.shared_graph.parameters())
                print(f"  Graph params (trainable): {sum(p.numel() for p in graph_params):,}")
            
            if train_orientation:
                orient_params = list(self.orientation_module.parameters())
                print(f"  Orientation params (trainable): {sum(p.numel() for p in orient_params):,}")

    def compute(self, inputs, role):
        B = inputs["states"].shape[0]
        states = unflatten_tensorized_space(self.observation_space, inputs["states"])
        img = states["img"].to(self.device)
        graph_flat = states["graph"].to(self.device)
        gt_orientation = states["orientation"].to(self.device)
        actions = inputs["taken_actions"].to(self.device)
        
        # Ground truth orientation
        ground_truth_yaw = states.get("orientation", None)
        if ground_truth_yaw is not None:
            ground_truth_yaw = ground_truth_yaw.to(self.device)
        
        # Graph encoding
        if self.train_graph:
            graph_emb = self.shared_graph(graph_flat)
        else:
            with torch.no_grad():
                graph_emb = self.shared_graph(graph_flat)
        
        # Orientation encoding
        if self.train_orientation:
            orientation_emb, orient_outputs = self.orientation_module(
                img, graph_emb, ground_truth_yaw
            )
        else:
            with torch.no_grad():
                orientation_emb, orient_outputs = self.orientation_module(
                    img, graph_emb, ground_truth_yaw
                )
        
        # Q-value
        x = torch.cat([img, graph_emb, actions], dim=-1)
        q = self.net(x)
        
        if DEBUG and not hasattr(self, '_debug_compute_printed'):
            print(f"\n[Critic.compute] First call (role={role}):")
            print(f"  B={B}")
            print(f"  img: {img.shape}")
            print(f"  graph_flat: {graph_flat.shape}")
            print(f"  graph_emb: {graph_emb.shape}")
            print(f"  orientation_emb: {orientation_emb.shape}")
            print(f"  actions: {actions.shape}")
            print(f"  concatenated: {x.shape}")
            print(f"  q: {q.shape}")
            
            if ground_truth_yaw is not None:
                print(f"  ground_truth_yaw: {ground_truth_yaw.shape}")
                if 'orientation_loss' in orient_outputs:
                    print(f"  orientation_loss: {orient_outputs['orientation_loss'].item():.4f}")
                    print(f"  orientation_accuracy_10deg: {orient_outputs['orientation_accuracy_10deg'].item():.4f}")
                    print(f"  orientation_accuracy_30deg: {orient_outputs['orientation_accuracy_30deg'].item():.4f}")
                    print(f"  orientation_accuracy_5deg: {orient_outputs['orientation_accuracy_5deg'].item():.4f}")
            
            self._debug_compute_printed = True
        
        return q, orient_outputs


from skrl.resources.preprocessors.torch.running_standard_scaler import RunningStandardScaler


class DictRunningStandardScaler(nn.Module):
    """
    Нормализует только states["img"], states["graph"] оставляет как есть.
    Работает независимо от порядка flatten'а.
    """
    def __init__(self, size, img_space, device=None, epsilon=1e-8, clip_threshold=5.0):
        super().__init__()
        self.full_space = size

        self.img_scaler = RunningStandardScaler(
            size=img_space,
            epsilon=epsilon,
            clip_threshold=clip_threshold,
            device=device,
        )

    def forward(self, x: torch.Tensor, train: bool = False, inverse: bool = False, no_grad: bool = True) -> torch.Tensor:
        # 1) развернуть в dict
        s = unflatten_tensorized_space(self.full_space, x)

        # 2) нормализовать только img
        s["img"] = self.img_scaler(s["img"], train=train, inverse=inverse, no_grad=no_grad)

        # 3) свернуть обратно
        return flatten_tensorized_space(s)


# ---------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------
if EVAL:
    from gymnasium.wrappers import RecordVideo
    print("[INFO] Running evaluation...")
    env = load_isaaclab_env(
        task_name="Isaac-Aloha-Direct-v0",
        num_envs=1,
        # headless=True,
        # cli_args=["--enable_cameras", "--video", "--livestream", "2",],
        cli_args=["--enable_cameras"],
    )
    # env = RecordVideo(
    #     env,
    #     video_folder="logs/skrl/aloha/videos",
    #     name_prefix="aloha_eval",
    #     episode_trigger=lambda ep: True,
    # )
else:
    print("[INFO] Running training...")
    env = load_isaaclab_env(
        task_name="Isaac-Aloha-Direct-v0",
        num_envs=32,
        headless=True,
        cli_args=["--enable_cameras"],
    )

env = wrap_env(env)
device = env.device

if DEBUG:
    print(f"\n{'='*60}")
    print("ENVIRONMENT INFO")
    print(f"{'='*60}")
    print(f"Device: {device}")
    print(f"Num envs: {env.num_envs}")
    print(f"Observation space: {env.observation_space}")
    print(f"Action space: {env.action_space}")
    
    # Проверяем что orientation есть в observation space
    if hasattr(env.observation_space, 'spaces') and isinstance(env.observation_space.spaces, dict):
        if "orientation" in env.observation_space.spaces:
            print(f"✓ 'orientation' found in observation space: {env.observation_space.spaces['orientation']}")
        else:
            print(f"⚠️  WARNING: 'orientation' NOT in observation space!")
            print(f"   Available keys: {list(env.observation_space.spaces.keys())}")
    print(f"{'='*60}\n")

# ---------------------------------------------------------------------
# Memory
# ---------------------------------------------------------------------
memory = RandomMemory(memory_size=10000, num_envs=env.num_envs, device=device)

# ---------------------------------------------------------------------
# Shared graph module (one instance)
# ---------------------------------------------------------------------
shared_graph = SharedGraphModule(
    embeddings_path="source/isaaclab_tasks/isaaclab_tasks/direct/aloha/text_embeddings.pt",
    num_nodes=NUM_GRAPH_NODES,
    per_object_dim=PER_OBJECT_DIM,
    text_dim=TEXT_EMB_DIM,
).to(device)

# ---------------------------------------------------------------------
# Orientation module (one instance)
# ---------------------------------------------------------------------
orientation_module = OrientationModule(
    img_dim=env.observation_space["img"].shape[0],
    graph_emb_dim=GRAPH_EMB_DIM,
    emb_dim=ORIENTATION_EMB_DIM,
    device=device
)

# ---------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------
models = {
    "policy": StochasticActor(
        env.observation_space, env.action_space, device,
        shared_graph=shared_graph,
        orientation_module=orientation_module
    ),
    
    # Только critic_1 обучает оба модуля
    "critic_1": Critic(
        env.observation_space, env.action_space, device,
        shared_graph=shared_graph,
        orientation_module=orientation_module,
        train_graph=True,
        train_orientation=True
    ),
    
    # critic_2 НЕ обучает модули (чтобы избежать двойного шага)
    "critic_2": Critic(
        env.observation_space, env.action_space, device,
        shared_graph=shared_graph,
        orientation_module=orientation_module,
        train_graph=False,
        train_orientation=False
    ),
    
    # targets НЕ обучают модули
    "target_critic_1": Critic(
        env.observation_space, env.action_space, device,
        shared_graph=shared_graph,
        orientation_module=orientation_module,
        train_graph=False,
        train_orientation=False
    ),
    "target_critic_2": Critic(
        env.observation_space, env.action_space, device,
        shared_graph=shared_graph,
        orientation_module=orientation_module,
        train_graph=False,
        train_orientation=False
    ),
}

if DEBUG:
    print(f"\n{'='*60}")
    print("PARAMETER CHECK")
    print(f"{'='*60}")
    
    # Проверяем что параметры правильно зарегистрированы
    print("\n1. Shared graph parameters:")
    graph_params = list(shared_graph.parameters())
    print(f"   Total: {sum(p.numel() for p in graph_params):,} parameters")
    
    print("\n2. Orientation module parameters:")
    orient_params = list(orientation_module.parameters())
    print(f"   Total: {sum(p.numel() for p in orient_params):,} parameters")
    
    print("\n3. Critic_1 parameters (should include graph + orientation):")
    critic1_params = list(models["critic_1"].parameters())
    print(f"   Total: {sum(p.numel() for p in critic1_params):,} parameters")
    
    # Проверяем что graph/orientation параметры есть в critic_1
    graph_param_ids = {id(p) for p in graph_params}
    orient_param_ids = {id(p) for p in orient_params}
    critic1_param_ids = {id(p) for p in critic1_params}
    
    graph_in_critic1 = bool(graph_param_ids & critic1_param_ids)
    orient_in_critic1 = bool(orient_param_ids & critic1_param_ids)
    
    print(f"\n   ✓ Graph params in Critic_1: {graph_in_critic1}")
    print(f"   ✓ Orientation params in Critic_1: {orient_in_critic1}")
    
    if not graph_in_critic1:
        print(f"   ⚠️  WARNING: Graph parameters NOT found in Critic_1!")
    if not orient_in_critic1:
        print(f"   ⚠️  WARNING: Orientation parameters NOT found in Critic_1!")
    
    print(f"\n4. Critic_2 parameters (should NOT include graph + orientation):")
    critic2_params = list(models["critic_2"].parameters())
    print(f"   Total: {sum(p.numel() for p in critic2_params):,} parameters")
    
    print(f"{'='*60}\n")

# ---------------------------------------------------------------------
# SAC config
# ---------------------------------------------------------------------
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
# cfg["state_preprocessor"] = None  
# cfg["state_preprocessor_kwargs"] = {}

cfg["experiment"]["write_interval"] = 50
cfg["experiment"]["checkpoint_interval"] = 5000
cfg["experiment"]["directory"] = "logs/skrl/aloha_ppo_orientation"

agent = SAC(
    models=models,
    memory=memory,
    cfg=cfg,
    observation_space=env.observation_space,
    action_space=env.action_space,
    device=device,
)
# После строки:
# agent = SAC(models=models, memory=memory, cfg=cfg, ...)

print(f"\n{'='*60}")
print("OPTIMIZER CHECK")
print(f"{'='*60}")

# 1. Проверяем параметры orientation_module
orient_params = list(orientation_module.parameters())
print(f"\nOrientation module:")
print(f"  Total params: {sum(p.numel() for p in orient_params):,}")
for i, p in enumerate(orient_params):
    print(f"  Param {i}: shape={p.shape}, requires_grad={p.requires_grad}")

# 2. Проверяем параметры critic_1
critic1_params = list(models["critic_1"].parameters())
print(f"\nCritic_1:")
print(f"  Total params: {sum(p.numel() for p in critic1_params):,}")

# 3. Проверяем что orientation_module В critic_1.parameters()
critic1_param_ids = {id(p) for p in critic1_params}
orient_param_ids = {id(p) for p in orient_params}
overlap = critic1_param_ids & orient_param_ids

print(f"\nOverlap check:")
print(f"  Orientation params in Critic_1: {len(overlap)}/{len(orient_params)}")
if len(overlap) == len(orient_params):
    print(f"  ✅ ALL orientation params are in Critic_1")
else:
    print(f"  ❌ ERROR: Only {len(overlap)}/{len(orient_params)} orientation params in Critic_1!")

# 4. Проверяем что они В OPTIMIZER
optimizer_param_ids = {id(p) for group in agent.critic_optimizer.param_groups for p in group['params']}
orient_in_optimizer = orient_param_ids & optimizer_param_ids

print(f"\nOptimizer check:")
print(f"  Orientation params in critic_optimizer: {len(orient_in_optimizer)}/{len(orient_params)}")
if len(orient_in_optimizer) == len(orient_params):
    print(f"  ✅ ALL orientation params are in optimizer")
else:
    print(f"  ❌ ERROR: Only {len(orient_in_optimizer)}/{len(orient_params)} orientation params in optimizer!")
    print(f"  Missing params: {len(orient_params) - len(orient_in_optimizer)}")

# 5. Проверяем learning rate
for i, group in enumerate(agent.critic_optimizer.param_groups):
    print(f"\nParam group {i}:")
    print(f"  Num params: {len(group['params'])}")
    print(f"  Learning rate: {group['lr']}")

print(f"{'='*60}\n")
# ---------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------
if not EVAL:
    cfg_trainer = {"timesteps": 330000}
    trainer = SequentialTrainer(cfg=cfg_trainer, env=env, agents=agent)
    
    if DEBUG:
        print(f"\n{'='*60}")
        print("STARTING TRAINING")
        print(f"{'='*60}")
        print("Debug mode ON - will print detailed info on first forward passes")
        print(f"{'='*60}\n")
    # ПЕРЕД строкой:
    # trainer.train()

    # Добавьте:
    print(f"\n{'='*60}")
    print("OPTIMIZER CHECK")
    print(f"{'='*60}")

    # 1. Проверяем параметры orientation_module
    orient_params = list(orientation_module.parameters())
    print(f"\nOrientation module:")
    print(f"  Total params: {sum(p.numel() for p in orient_params):,}")
    for i, (name, p) in enumerate(orientation_module.named_parameters()):
        print(f"  {name}: shape={p.shape}, requires_grad={p.requires_grad}")

    # 2. Проверяем параметры critic_1
    critic1_params = list(models["critic_1"].parameters())
    print(f"\nCritic_1:")
    print(f"  Total params: {sum(p.numel() for p in critic1_params):,}")

    # 3. Проверяем что orientation_module В critic_1.parameters()
    critic1_param_ids = {id(p) for p in critic1_params}
    orient_param_ids = {id(p) for p in orient_params}
    overlap = critic1_param_ids & orient_param_ids

    print(f"\nOverlap check:")
    print(f"  Orientation params in Critic_1: {len(overlap)}/{len(orient_params)}")
    if len(overlap) == len(orient_params):
        print(f"  ✅ ALL orientation params are in Critic_1")
    elif len(overlap) == 0:
        print(f"  ❌ ERROR: NONE of the orientation params are in Critic_1!")
        print(f"  → Orientation module will NOT be optimized!")
    else:
        print(f"  ⚠️  WARNING: Only {len(overlap)}/{len(orient_params)} params in Critic_1")

    # 4. Проверяем что они В OPTIMIZER
    optimizer_params = []
    for group in agent.critic_optimizer.param_groups:
        optimizer_params.extend(group['params'])
    optimizer_param_ids = {id(p) for p in optimizer_params}
    orient_in_optimizer = orient_param_ids & optimizer_param_ids

    print(f"\nOptimizer check:")
    print(f"  Orientation params in critic_optimizer: {len(orient_in_optimizer)}/{len(orient_params)}")
    if len(orient_in_optimizer) == len(orient_params):
        print(f"  ✅ ALL orientation params are in optimizer")
    elif len(orient_in_optimizer) == 0:
        print(f"  ❌ ERROR: NONE of the orientation params are in optimizer!")
        print(f"  → Orientation module will NOT receive gradient updates!")
    else:
        print(f"  ⚠️  WARNING: Only {len(orient_in_optimizer)}/{len(orient_params)} params in optimizer")

    # 5. Показываем optimizer param groups
    print(f"\nOptimizer structure:")
    for i, group in enumerate(agent.critic_optimizer.param_groups):
        print(f"  Param group {i}:")
        print(f"    Num params: {len(group['params'])}")
        print(f"    Learning rate: {group['lr']}")
        print(f"    Total elements: {sum(p.numel() for p in group['params']):,}")

    print(f"{'='*60}\n")
    trainer.train()
else:
    cfg_trainer = {"timesteps": 1500}
    trainer = SequentialTrainer(cfg=cfg_trainer, env=env, agents=agent)

    checkpoint_path = "/home/xiso/IsaacLab/logs/skrl/aloha_ppo_orientation/ 26-02-05_15-47-22-451517_SAC /checkpoints/agent_120000.pt"
    agent.load(checkpoint_path)

    trainer.eval()
