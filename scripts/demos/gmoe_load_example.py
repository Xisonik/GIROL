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
ORIENTATION_PROBS_DIM = 36    # количество бинов для ориентации (используем probs напрямую)

# В графе "цель ↔ объекты" нужно знать индекс узла цели.
# Если цель у тебя всегда первая в encode_scene_graph — оставляй 0.
GOAL_NODE_INDEX = 0
DEBUG = True
USE_PRETRAINED = False
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
    """Предсказывает ориентацию робота и выдаёт probs для Actor/Critic."""
    def __init__(self, img_dim: int, graph_emb_dim: int, num_bins: int = 36, 
                 emb_dim: int = 32, device=None):  # emb_dim не используется в новой архитектуре
        super().__init__()
        self.device = device
        self.num_bins = num_bins
        
        self.net = nn.Sequential(
            nn.Linear(img_dim + graph_emb_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, num_bins)
        ).to(device)
        
    def forward(self, img, graph_emb, ground_truth_yaw=None):
        x = torch.cat([img, graph_emb], dim=-1)
        logits = self.net(x)
        probs = F.softmax(logits, dim=-1)
        
        # ✅ DEBUG: Top-3 (только иногда, чтобы не спамить)
        if not hasattr(self, '_debug_counter'):
            self._debug_counter = 0
        
        self._debug_counter += 1
        if self._debug_counter % 1 == 0:  # Каждые 100 вызовов
            bin_size = (2 * torch.pi) / self.num_bins
            top3_probs, top3_indices = torch.topk(probs[0], k=3)
            angles_str = ', '.join([
                f'{((idx*bin_size + bin_size/2 - torch.pi)*180/torch.pi):.1f}° ({prob:.3f})'
                for prob, idx in zip(top3_probs, top3_indices)
            ])
            print(f"[Orient] Top-3: {angles_str}, MaxProb: {probs[0].max():.3f}")

        # Возвращаем probs вместо orientation_emb
        orientation_probs = probs
        
        outputs = {
            'orientation_logits': logits,
            'orientation_probs': probs  # Добавляем probs в outputs
        }
        
        if ground_truth_yaw is not None:
            if ground_truth_yaw.dim() == 2:
                ground_truth_yaw = ground_truth_yaw.squeeze(-1)
            
            bin_size = (2 * torch.pi) / self.num_bins
            
            gt_normalized = torch.atan2(
                torch.sin(ground_truth_yaw),
                torch.cos(ground_truth_yaw)
            )
            
            bin_centers = torch.linspace(-torch.pi, torch.pi, self.num_bins+1, 
                                        device=logits.device)[:-1]
            bin_centers = bin_centers + bin_size / 2
            
            labels = ((gt_normalized + torch.pi) / bin_size).long().clamp(0, self.num_bins - 1)
            
            # ✅ УВЕЛИЧЕННЫЙ KAPPA для более острого пика
            kappa = 70.0  # Было 10.0 → стало 50.0
            
            # Von Mises distribution (target)
            angle_diff = bin_centers.unsqueeze(0) - gt_normalized.unsqueeze(1)
            target_probs = torch.exp(kappa * torch.cos(angle_diff))
            target_probs = target_probs / target_probs.sum(dim=-1, keepdim=True)
            
            # KL divergence
            loss = F.kl_div(
                torch.log(probs + 1e-8),
                target_probs,
                reduction='batchmean'
            )
            
            # ✅ ДОБАВЛЕНЫ МЕТРИКИ
            pred_bins = torch.argmax(logits, dim=-1)
            bin_diff = torch.abs(pred_bins - labels)
            bin_diff = torch.minimum(bin_diff, self.num_bins - bin_diff)
            
            # Relaxed accuracy (±1 бин, то есть ±10°)
            accuracy_relaxed = (bin_diff <= 1).float().mean()
            
            # ✅ Strict accuracy (точное совпадение)
            accuracy_strict = (pred_bins == labels).float().mean()
            
            # ✅ Confidence (средняя максимальная вероятность)
            max_probs = probs.max(dim=-1)[0]
            confidence = max_probs.mean()
            
            # ✅ Entropy (нормализованная)
            entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=-1)
            max_entropy = torch.log(torch.tensor(self.num_bins, device=probs.device, dtype=torch.float32))
            normalized_entropy = (entropy / max_entropy).mean()
            
            # ✅ Angular error в градусах
            pred_angles = bin_centers[pred_bins]
            angular_error = torch.atan2(
                torch.sin(gt_normalized - pred_angles),
                torch.cos(gt_normalized - pred_angles)
            )
            mean_angular_error = torch.abs(angular_error).mean() * 180 / torch.pi
            
            outputs.update({
                'orientation_loss': loss,
                'orientation_label': labels,
                'orientation_accuracy': accuracy_relaxed,  # Основная метрика (для совместимости)
                'orientation_accuracy_strict': accuracy_strict,
                'orientation_confidence': confidence,  
                'orientation_entropy': normalized_entropy, 
                'orientation_mean_error_deg': mean_angular_error, 
            })
        
        return orientation_probs, outputs  # Возвращаем probs вместо embedding


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
    """Actor: использует orientation probs (без обучения модуля)."""
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

        # Policy: img + graph_emb + orientation_probs 
        mlp_in = self.img_dim + GRAPH_EMB_DIM + 1 # ORIENTATION_PROBS_DIM
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
            print(f"  mlp_in={mlp_in} (img + graph + orientation_probs)")
            print(f"  orientation_probs_dim={ORIENTATION_PROBS_DIM}")
            print(f"  num_actions={self.num_actions}")

    def compute(self, inputs, role):
        B = inputs["states"].shape[0]
        states = unflatten_tensorized_space(self.observation_space, inputs["states"])
        img = states["img"].to(self.device)
        graph_flat = states["graph"].to(self.device)
        gt_orientation = states["orientation"].to(self.device)  # Для мониторинга
        print("gt_orientation ", gt_orientation)
        # Все модули в no_grad для actor
        with torch.no_grad():
            graph_emb = self.shared_graph(graph_flat)
            orientation_probs, _ = self.orientation_module(img, graph_emb, ground_truth_yaw=None)
        # ПРЕОБРАЗУЕМ orientation_probs в угол
        bin_size = (2 * torch.pi) / ORIENTATION_PROBS_DIM
        bin_centers = torch.linspace(-torch.pi, torch.pi, ORIENTATION_PROBS_DIM+1, 
                                    device=orientation_probs.device)[:-1]
        bin_centers = bin_centers + bin_size / 2
        
        # Вычисляем ожидаемый угол (expected value)
        # Способ 1: По максимальной вероятности
        pred_bins = orientation_probs.argmax(dim=-1)  # [B]
        pred_angles = bin_centers[pred_bins]  # [B]
        
        # Или способ 2: Взвешенное среднее (более плавно)
        # pred_angles = torch.sum(orientation_probs * bin_centers.unsqueeze(0), dim=-1)  # [B]
        
        # Добавляем размерность для конкатенации
        pred_angles = pred_angles.unsqueeze(-1)  # [B, 1]
        # Используем orientation_probs вместо gt_orientation
        x = torch.cat([img, graph_emb, pred_angles], dim=-1) # orientation_probs
        mu = self.net(x)
        
        if DEBUG and not hasattr(self, '_debug_compute_printed'):
            print(f"\n[StochasticActor.compute] First call:")
            print(f"  B={B}")
            print(f"  img: {img.shape}")
            print(f"  graph_emb: {graph_emb.shape}")
            print(f"  orientation_probs: {orientation_probs.shape}")
            print(f"  concatenated: {x.shape}")
            print(f"  mu: {mu.shape}")
            self._debug_compute_printed = True
        
        return mu, self.log_std_parameter, {}


class Critic(DeterministicMixin, Model):
    """Critic: Q(s,a) + orientation monitoring (не обучение)."""
    def __init__(self, observation_space, action_space, device, shared_graph: SharedGraphModule,
                 orientation_module: OrientationModule,
                 clip_actions=False, train_graph: bool = False, train_orientation: bool = False):
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

        # Q-network: img + graph_emb + orientation_probs + action
        mlp_in = self.img_dim + GRAPH_EMB_DIM + 1 + self.num_actions
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
            print(f"  mlp_in={mlp_in} (img + graph + orientation_probs + action)")
            print(f"  orientation_probs_dim={ORIENTATION_PROBS_DIM}")
            
            # Проверяем параметры
            critic_params = list(self.net.parameters())
            print(f"  Critic net params: {sum(p.numel() for p in critic_params):,}")

    def compute(self, inputs, role):
        B = inputs["states"].shape[0]
        states = unflatten_tensorized_space(self.observation_space, inputs["states"])
        img = states["img"].to(self.device)
        graph_flat = states["graph"].to(self.device)
        gt_orientation = states["orientation"].to(self.device)  # Для мониторинга
        actions = inputs["taken_actions"].to(self.device)
        
        # Graph encoding
        if self.train_graph:
            graph_emb = self.shared_graph(graph_flat)
        else:
            with torch.no_grad():
                graph_emb = self.shared_graph(graph_flat)
        
        # Orientation encoding (для мониторинга используем gt_orientation)
        # Но для Q-функции используем предсказанные probs
        if self.train_orientation:
            orientation_probs, orient_outputs = self.orientation_module(
                img, graph_emb, ground_truth_yaw=gt_orientation  # gt для мониторинга
            )
        else:
            with torch.no_grad():
                orientation_probs, orient_outputs = self.orientation_module(
                    img, graph_emb, ground_truth_yaw=gt_orientation  # gt для мониторинга
                )
        # ПРЕОБРАЗУЕМ orientation_probs в угол
        bin_size = (2 * torch.pi) / ORIENTATION_PROBS_DIM
        bin_centers = torch.linspace(-torch.pi, torch.pi, ORIENTATION_PROBS_DIM+1, 
                                    device=orientation_probs.device)[:-1]
        bin_centers = bin_centers + bin_size / 2
        
        # Вычисляем ожидаемый угол (expected value)
        # Способ 1: По максимальной вероятности
        pred_bins = orientation_probs.argmax(dim=-1)  # [B]
        pred_angles = bin_centers[pred_bins]  # [B]
        
        # Или способ 2: Взвешенное среднее (более плавно)
        # pred_angles = torch.sum(orientation_probs * bin_centers.unsqueeze(0), dim=-1)  # [B]
        
        # Добавляем размерность для конкатенации
        pred_angles = pred_angles.unsqueeze(-1)  # [B, 1]
        # Q-value: используем orientation_probs вместо gt_orientation
        x = torch.cat([img, graph_emb, pred_angles, actions], dim=-1)
        q = self.net(x)
        
        if DEBUG and not hasattr(self, '_debug_compute_printed'):
            print(f"\n[Critic.compute] First call (role={role}):")
            print(f"  B={B}")
            print(f"  img: {img.shape}")
            print(f"  graph_flat: {graph_flat.shape}")
            print(f"  graph_emb: {graph_emb.shape}")
            print(f"  orientation_probs: {orientation_probs.shape}")
            print(f"  actions: {actions.shape}")
            print(f"  concatenated: {x.shape}")
            print(f"  q: {q.shape}")
            
            if 'orientation_loss' in orient_outputs:
                print(f"  orientation_loss: {orient_outputs['orientation_loss'].item():.4f}")
                print(f"  orientation_accuracy: {orient_outputs['orientation_accuracy'].item():.4f}")
                print(f"  orientation_confidence: {orient_outputs['orientation_confidence'].item():.4f}")
            
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
# Shared graph module (one instance) - ЗАГРУЖАЕМ И ЗАМОРАЖИВАЕМ
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
    num_bins=ORIENTATION_PROBS_DIM,
    emb_dim=32,  # Не используется, но оставляем для совместимости
    device=device
)

models = {
    "policy": StochasticActor(
        env.observation_space, env.action_space, device,
        shared_graph=shared_graph,
        orientation_module=orientation_module
    ),
    
    # ВСЕ критики используют frozen модули
    "critic_1": Critic(
        env.observation_space, env.action_space, device,
        shared_graph=shared_graph,
        orientation_module=orientation_module,
        train_graph=True,      # НЕ обучаем граф
        train_orientation=True # НЕ обучаем ориентацию
    ),
    
    "critic_2": Critic(
        env.observation_space, env.action_space, device,
        shared_graph=shared_graph,
        orientation_module=orientation_module,
        train_graph=False,
        train_orientation=False
    ),
    
    # targets тоже frozen
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


cfg["experiment"]["write_interval"] = 10
cfg["experiment"]["checkpoint_interval"] = 500
cfg["experiment"]["directory"] = "logs/skrl/aloha_ppo_orientation"

agent = SAC(
    models=models,
    memory=memory,
    cfg=cfg,
    observation_space=env.observation_space,
    action_space=env.action_space,
    device=device,
)

if USE_PRETRAINED:
    ORIENTATION_CHECKPOINT = "/home/xiso/IsaacLab/logs/skrl/aloha_ppo_orientation/test_baseline_gt_theta/checkpoints/agent_5000.pt"
    checkpoint = torch.load(ORIENTATION_CHECKPOINT, map_location=device)
    analyze_skrl_checkpoint(checkpoint)
    # 1. Загружаем shared_graph
    graph_weights = {}
    for key, value in checkpoint["critic_1"].items():
        if "shared_graph" in key:
            new_key = key.replace("shared_graph.", "")
            graph_weights[new_key] = value
    
    if graph_weights:
        try:
            shared_graph.load_state_dict(graph_weights)
            print(f"✅ Загружен shared_graph ({len(graph_weights)} параметров)")
        except Exception as e:
            print(f"❌ Ошибка загрузки shared_graph: {e}")
            print("   Используем случайную инициализацию")
    else:
        print("⚠️  shared_graph не найден в checkpoint")
    
    # 2. Загружаем orientation_module
    orientation_weights = {}
    for key, value in checkpoint["critic_1"].items():
        if "orientation_module" in key:
            new_key = key.replace("orientation_module.", "")
            orientation_weights[new_key] = value
    
    if orientation_weights:
        try:
            orientation_module.load_state_dict(orientation_weights)
            print(f"✅ Загружен orientation_module ({len(orientation_weights)} параметров)")
        except Exception as e:
            print(f"❌ Ошибка загрузки orientation_module: {e}")
            print("   Используем случайную инициализацию")
    else:
        print("⚠️  orientation_module не найден в checkpoint")
    
    for param in shared_graph.parameters():
        param.requires_grad = False
    shared_graph.eval()
    
    for param in orientation_module.parameters():
        param.requires_grad = False
    orientation_module.eval()
    
    print(f"✅ Оба модуля заморожены")
    print(f"{'='*80}\n")

    try:
        # Проверяем наличие state_preprocessor в чекпоинте
        if 'state_preprocessor' in checkpoint:
            agent._state_preprocessor.load_state_dict(checkpoint['state_preprocessor'])
            print(f"\n✅ Загружен state_preprocessor из checkpoint")
            
            # Проверка загрузки
            if hasattr(agent._state_preprocessor, 'img_scaler'):
                mean = agent._state_preprocessor.img_scaler.running_mean[:3]
                var = agent._state_preprocessor.img_scaler.running_variance[:3]
                count = agent._state_preprocessor.img_scaler.current_count
                print(f"   Статистики загружены:")
                print(f"   - Количество образцов: {count.item():.0f}")
                print(f"   - Средние (первые 3): {mean}")
                print(f"   - Дисперсии (первые 3): {var}")
        else:
            print(f"\nℹ️  state_preprocessor не найден в checkpoint, используем новый")
    except Exception as e:
        print(f"\n❌ Ошибка загрузки state_preprocessor: {e}")
        print("   Продолжаем с новым препроцессором")

# ---------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------
if not EVAL:
    cfg_trainer = {"timesteps": 330000}
    trainer = SequentialTrainer(cfg=cfg_trainer, env=env, agents=agent)
    checkpoint_path = "/home/xiso/IsaacLab/logs/skrl/aloha_ppo_orientation/test_baseline_gt_theta/checkpoints/agent_5000.pt"
    agent.load(checkpoint_path)
    if DEBUG:
        print(f"\n{'='*60}")
        print("STARTING TRAINING WITH FROZEN MODULES")
        print(f"{'='*60}")
        print("Key changes:")
        print("1. Both shared_graph and orientation_module are frozen")
        print("2. Using orientation_probs (36-dim) instead of gt_orientation")
        print("3. No graph training, no orientation training")
        print(f"{'='*60}\n")
    
    trainer.train()
else:
    cfg_trainer = {"timesteps": 1500}
    trainer = SequentialTrainer(cfg=cfg_trainer, env=env, agents=agent)

    checkpoint_path = "/home/xiso/IsaacLab/logs/skrl/aloha_ppo_orientation/test_baseline_gt_theta/checkpoints/agent_5000.pt"
    agent.load(checkpoint_path)

    trainer.eval()


if FREEZE:
    print(f"\n{'='*80}")
    print("FREEZING SHARED MODULES")
    print(f"{'='*80}")
    
    # Список всех моделей, в которых есть shared modules
    models_with_shared = [
        ("policy", agent.policy),
        ("critic_1", agent.critic_1),
        ("critic_2", agent.critic_2),
        ("target_critic_1", agent.target_critic_1),
        ("target_critic_2", agent.target_critic_2),
    ]
    
    frozen_count = 0
    
    for model_name, model in models_with_shared:
        # Заморозка shared_graph
        if hasattr(model, 'shared_graph'):
            for param in model.shared_graph.parameters():
                param.requires_grad = False
            model.shared_graph.eval()
            print(f"✅ {model_name}.shared_graph frozen")
            frozen_count += 1
        
        # Заморозка orientation_module
        if hasattr(model, 'orientation_module'):
            for param in model.orientation_module.parameters():
                param.requires_grad = False
            model.orientation_module.eval()
            print(f"✅ {model_name}.orientation_module frozen")
            frozen_count += 1
    
    print(f"\nTotal frozen modules: {frozen_count}")
    print(f"{'='*80}\n")