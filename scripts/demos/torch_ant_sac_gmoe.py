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
        # print("probs ", probs)
        # ✅ DEBUG: Top-3 (только иногда, чтобы не спамить)
        if not hasattr(self, '_debug_counter'):
            self._debug_counter = 0
        
        self._debug_counter += 1

        # Возвращаем probs вместо orientation_emb
        orientation_probs = probs
        
        outputs = {
            'orientation_logits': logits,
            'orientation_probs': probs  # Добавляем probs в outputs
        }
        
        if ground_truth_yaw is not None:
            if ground_truth_yaw.dim() == 2:
                ground_truth_yaw = ground_truth_yaw.squeeze(-1)
            # print("ground_truth_yaw ", ground_truth_yaw)
            bin_size = (2 * torch.pi) / self.num_bins
            
            gt_normalized = torch.atan2(
                torch.sin(ground_truth_yaw),
                torch.cos(ground_truth_yaw)
            )
            # print("gt_normalized ", gt_normalized)
            bin_centers = torch.linspace(-torch.pi, torch.pi, self.num_bins+1, 
                                        device=logits.device)[:-1]
            bin_centers = bin_centers + bin_size / 2
            
            labels = ((gt_normalized + torch.pi) / bin_size).long().clamp(0, self.num_bins - 1)
            # print("labels ", labels)
            kappa = 70.0
            
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
            # print("loss ", loss)
            # ✅ ДОБАВЛЕНЫ МЕТРИКИ
            pred_bins = torch.argmax(logits, dim=-1)
            # print("pred_bins ", pred_bins)
            bin_diff = torch.abs(pred_bins - labels)
            bin_diff = torch.minimum(bin_diff, self.num_bins - bin_diff)
            # print("bin_diff ", bin_diff)
            bad_labels = labels.clone()
            bad_labels[bin_diff < 3] = 0
            # print("bad_labels ", bad_labels)
            bad_ground_truth_yaw = ground_truth_yaw.clone()
            bad_ground_truth_yaw[bin_diff < 3] = 0
            # print("bad ground_truth_yaw ", bad_ground_truth_yaw)
            # Relaxed accuracy (±1 бин, то есть ±10°)
            accuracy_relaxed = (bin_diff <= 1).float().mean()
            # print("accuracy_relaxed ", accuracy_relaxed)
            # print("acc: ", accuracy_relaxed)
            # ✅ Strict accuracy (точное совпадение)
            accuracy_strict = (pred_bins == labels).float().mean()
            # print("accuracy_strict ", accuracy_strict)
            # ✅ Confidence (средняя максимальная вероятность)
            max_probs = probs.max(dim=-1)[0]
            confidence = max_probs.mean()
            
            # ✅ Entropy (нормализованная)
            entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=-1)
            max_entropy = torch.log(torch.tensor(self.num_bins, device=probs.device, dtype=torch.float32))
            normalized_entropy = (entropy / max_entropy).mean()
            
            # ✅ Angular error в градусах
            pred_angles = bin_centers[pred_bins]
            # print("pred_angles ", pred_angles)
            angular_error = torch.atan2(
                torch.sin(gt_normalized - pred_angles),
                torch.cos(gt_normalized - pred_angles)
            )
            # print("angular_error ", angular_error)
            mean_angular_error = torch.abs(angular_error).mean() * 180 / torch.pi
            # print("mean_angular_error ", mean_angular_error)
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
# CLI аргументы / seed args
# ---------------------------------------------------------------------

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
        # print("gt_orientation ", gt_orientation)
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
        x = torch.cat([img, graph_emb, gt_orientation], dim=-1) # orientation_probs
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
        x = torch.cat([img, graph_emb, gt_orientation, actions], dim=-1)
        q = self.net(x)
        
        # print(f"  orientation_accuracy: {orient_outputs['orientation_accuracy'].item():.4f}")
        # print(f"  orientation_confidence: {orient_outputs['orientation_confidence'].item():.4f}")
                    
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
EVAL = False
# EVAL = True
USE_PRETRAINED = False
FREEZE = False
KEEP_ON_TRAIN = True

if EVAL:
    from gymnasium.wrappers import RecordVideo
    print("[INFO] Running evaluation...")
    env = load_isaaclab_env(
        task_name="Isaac-Aloha-Direct-v0",
        num_envs=4,
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
        train_graph=KEEP_ON_TRAIN,      # НЕ обучаем граф
        train_orientation=KEEP_ON_TRAIN # НЕ обучаем ориентацию
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

if USE_PRETRAINED:
    CHECKPOINT_PATH = "/home/xiso/IsaacLab/logs/skrl/aloha_ppo_orientation/base/checkpoints/best_agent.pt"
    
    print(f"\n{'='*80}")
    print("ЗАГРУЗКА SHARED МОДУЛЕЙ ИЗ ЧЕКПОИНТА")
    print(f"{'='*80}")
    
    # 1. Загрузите чекпоинт
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)
    
    if 'critic_1' in checkpoint and isinstance(checkpoint['critic_1'], dict):
        critic1_state = checkpoint['critic_1']
        
        # Ищем ключи shared модулей
        shared_keys = [k for k in critic1_state.keys() if k.startswith('shared_graph.')]
        orient_keys = [k for k in critic1_state.keys() if k.startswith('orientation_module.')]
        
        print(f"\n📦 В critic_1 найдено:")
        print(f"   - shared_graph ключей: {len(shared_keys)}")
        print(f"   - orientation_module ключей: {len(orient_keys)}")
        
        # ========================================================================
        # 3. ЗАГРУЖАЕМ SHARED_GRAPH
        # ========================================================================
        if shared_keys:
            print(f"\n🔧 Загрузка shared_graph...")
            print(f"   Примеры ключей: {shared_keys[:3]}")
            
            # Создаем state_dict, убирая префикс 'shared_graph.'
            shared_state_dict = {}
            for key in shared_keys:
                new_key = key[13:]
                shared_state_dict[new_key] = critic1_state[key]
            
            # Загружаем
            try:
                missing, unexpected = shared_graph.load_state_dict(shared_state_dict, strict=False)
                print(f"   ✅ Загружено {len(shared_state_dict)} параметров")
                
                if missing:
                    print(f"   ⚠️  Пропущенные ключи: {missing[:5]}")
                if unexpected:
                    print(f"   ⚠️  Неожиданные ключи: {unexpected[:5]}")
                    
            except Exception as e:
                print(f"   ❌ Ошибка: {e}")
        else:
            print(f"   ⚠️  WARNING: shared_graph не найден в critic_1!")
        
        # ========================================================================
        # 4. ЗАГРУЖАЕМ ORIENTATION_MODULE
        # ========================================================================
        if orient_keys:
            print(f"\n🔧 Загрузка orientation_module...")
            print(f"   Примеры ключей: {orient_keys[:3]}")
            
            # Создаем state_dict, убирая префикс 'orientation_module.'
            orient_state_dict = {}
            for key in orient_keys:
                new_key = key[19:]  # убираем 'orientation_module.'
                orient_state_dict[new_key] = critic1_state[key]
            
            # Загружаем
            try:
                missing, unexpected = orientation_module.load_state_dict(orient_state_dict, strict=False)
                print(f"   ✅ Загружено {len(orient_state_dict)} параметров")
                
                if missing:
                    print(f"   ⚠️  Пропущенные ключи: {missing[:5]}")
                if unexpected:
                    print(f"   ⚠️  Неожиданные ключи: {unexpected[:5]}")
                    
            except Exception as e:
                print(f"   ❌ Ошибка: {e}")
        else:
            print(f"   ⚠️  WARNING: orientation_module не найден в critic_1!")
        
        # ========================================================================
        # 5. ЗАГРУЖАЕМ STATE_PREPROCESSOR (ВАЖНО!)
        # ========================================================================
        if 'state_preprocessor' in checkpoint:
            print(f"\n🔧 Загрузка state_preprocessor...")
            try:
                agent._state_preprocessor.load_state_dict(checkpoint['state_preprocessor'])
                print(f"   ✅ State preprocessor загружен")
                agent._state_preprocessor.eval()
                agent._state_preprocessor.img_scaler.eval()
                # Проверка статистик
                if hasattr(agent._state_preprocessor, 'img_scaler'):
                    scaler = agent._state_preprocessor.img_scaler
                    mean = scaler.running_mean[:3]
                    var = scaler.running_variance[:3]
                    print(f"   📊 Mean (first 3): {mean.cpu().numpy()}")
                    print(f"   📊 Var (first 3):  {var.cpu().numpy()}")
                    
            except Exception as e:
                print(f"   ❌ Ошибка загрузки preprocessor: {e}")
    
    # ========================================================================
    # 6. ПЕРЕВОДИМ В EVAL РЕЖИМ
    # ========================================================================
    print(f"\n{'='*80}")
    print("ПЕРЕВОД В EVAL РЕЖИМ")
    print(f"{'='*80}")
    
    shared_graph.eval()
    orientation_module.eval()
    
    print("✅ shared_graph → eval mode")
    print("✅ orientation_module → eval mode")
    
    # ========================================================================
    # 7. ЗАМОРАЖИВАЕМ ПАРАМЕТРЫ
    # ========================================================================
    for param in shared_graph.parameters():
        param.requires_grad = False
    
    for param in orientation_module.parameters():
        param.requires_grad = False
    
    # Подсчет frozen параметров
    shared_frozen = sum(p.numel() for p in shared_graph.parameters())
    orient_frozen = sum(p.numel() for p in orientation_module.parameters())
    
    print(f"\n🔒 ЗАМОРОЖЕНО:")
    print(f"   shared_graph: {shared_frozen:,} параметров")
    print(f"   orientation_module: {orient_frozen:,} параметров")
    print(f"   TOTAL: {shared_frozen + orient_frozen:,} параметров")
    print(f"{'='*80}\n")

# ---------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------
def save_replay_buffer(memory, path="replay_buffer.pt"):
    """Сохраняет валидную часть RandomMemory."""
    # Количество заполненных строк
    valid_rows = memory.memory_size if memory.filled else memory.memory_index
    
    data = {}
    for name, tensor in memory.tensors.items():
        # tensor shape: (memory_size, num_envs, data_size)
        data[name] = tensor[:valid_rows].cpu().clone()
    
    data["_meta"] = {
        "valid_rows": valid_rows,
        "num_envs": memory.num_envs,
        "filled": memory.filled,
        "memory_index": memory.memory_index,
    }
    torch.save(data, path)
    total = valid_rows * memory.num_envs
    print(f"✅ Saved {total} transitions ({valid_rows} rows × {memory.num_envs} envs) → {path}")

if not EVAL:
    cfg_trainer = {"timesteps": 330000}
    trainer = SequentialTrainer(cfg=cfg_trainer, env=env, agents=agent)
    # checkpoint_path = "/home/xiso/IsaacLab/logs/skrl/aloha_ppo_orientation/26-02-13_15-45-47-276076_SAC/checkpoints/agent_3000.pt"
    # agent.load(checkpoint_path)
    if FREEZE:
        print(f"\n{'='*80}")
        print("ФИНАЛЬНАЯ ЗАМОРОЗКА ПЕРЕД ОБУЧЕНИЕМ")
        print(f"{'='*80}")

        # 1. Shared модули
        shared_graph.eval()
        orientation_module.eval()
        for param in shared_graph.parameters():
            param.requires_grad = False
        for param in orientation_module.parameters():
            param.requires_grad = False

        # 2. Preprocessor (КРИТИЧНО!)
        agent._state_preprocessor.eval()
        if hasattr(agent._state_preprocessor, 'img_scaler'):
            agent._state_preprocessor.img_scaler.eval()
            
            # Замораживаем buffers (running_mean, running_variance)
            for name, buffer in agent._state_preprocessor.img_scaler.named_buffers():
                buffer.requires_grad = False

        print("✅ Все shared модули и preprocessor заморожены")
        print(f"{'='*80}\n")

        print(f"\n{'='*80}")
        print("ДИАГНОСТИКА: Сохраняем исходные веса")
        print(f"{'='*80}")

        import copy

        # Сохраняем копии весов
        initial_shared_graph = copy.deepcopy(shared_graph.state_dict())
        initial_orientation = copy.deepcopy(orientation_module.state_dict())
        initial_preprocessor = copy.deepcopy(agent._state_preprocessor.state_dict())

        # Сохраняем статистики preprocessor отдельно
        initial_mean = agent._state_preprocessor.img_scaler.running_mean.clone()
        initial_var = agent._state_preprocessor.img_scaler.running_variance.clone()
        initial_count = agent._state_preprocessor.img_scaler.current_count.clone()

        print("✅ Исходные веса сохранены")
        print(f"   Initial mean (first 3): {initial_mean[:3]}")
        print(f"   Initial var (first 3): {initial_var[:3]}")
        print(f"   Initial count: {initial_count.item()}")
        print(f"{'='*80}\n")

        # ========================================================================
        # ХАКЕРСКИЙ КОЛБЭК: Проверяем изменения весов каждые N шагов
        # ========================================================================
        original_post_interaction = agent.post_interaction

        check_interval = 2500  # Проверяем каждые 500 шагов

        def diagnostic_post_interaction(timestep, timesteps):
            # Вызываем оригинальный метод
            original_post_interaction(timestep, timesteps)
            
            # Проверяем изменения каждые check_interval шагов
            if timestep % check_interval == 0 and timestep > 0:
                print(f"\n🔍 [{timestep}] ДИАГНОСТИКА ИЗМЕНЕНИЙ:")
                save_replay_buffer(agent.memory, f"replay_buffer_{timestep}.pt")
                # 1. Проверяем shared_graph
                current_shared = shared_graph.state_dict()
                shared_changed = False
                max_diff = 0.0
                
                for key in initial_shared_graph.keys():
                    diff = (current_shared[key] - initial_shared_graph[key]).abs().max().item()
                    if diff > 1e-7:
                        shared_changed = True
                        max_diff = max(max_diff, diff)
                
                if shared_changed:
                    print(f"   ❌ shared_graph ИЗМЕНИЛСЯ! Max diff: {max_diff:.2e}")
                else:
                    print(f"   ✅ shared_graph НЕ изменился")
                
                # 2. Проверяем orientation_module
                current_orient = orientation_module.state_dict()
                orient_changed = False
                max_diff = 0.0
                
                for key in initial_orientation.keys():
                    diff = (current_orient[key] - initial_orientation[key]).abs().max().item()
                    if diff > 1e-7:
                        orient_changed = True
                        max_diff = max(max_diff, diff)
                
                if orient_changed:
                    print(f"   ❌ orientation_module ИЗМЕНИЛСЯ! Max diff: {max_diff:.2e}")
                else:
                    print(f"   ✅ orientation_module НЕ изменился")
                
                # 3. Проверяем preprocessor (САМОЕ ВАЖНОЕ!)
                current_mean = agent._state_preprocessor.img_scaler.running_mean
                current_var = agent._state_preprocessor.img_scaler.running_variance
                current_count = agent._state_preprocessor.img_scaler.current_count
                
                mean_diff = (current_mean - initial_mean).abs().max().item()
                var_diff = (current_var - initial_var).abs().max().item()
                count_diff = (current_count - initial_count).abs().item()
                
                if mean_diff > 1e-7 or var_diff > 1e-7 or count_diff > 0:
                    print(f"   ❌ PREPROCESSOR ИЗМЕНИЛСЯ!")
                    print(f"      Mean diff: {mean_diff:.2e} (max change: {(current_mean - initial_mean).abs().max():.4f})")
                    print(f"      Var diff: {var_diff:.2e}")
                    print(f"      Count diff: {count_diff:.0f}")
                    print(f"      Current mean (first 3): {current_mean[:3]}")
                    print(f"      Initial mean (first 3): {initial_mean[:3]}")
                else:
                    print(f"   ✅ preprocessor НЕ изменился")
                
                # 4. Проверяем requires_grad
                shared_trainable = sum(p.numel() for p in shared_graph.parameters() if p.requires_grad)
                orient_trainable = sum(p.numel() for p in orientation_module.parameters() if p.requires_grad)
                
                if shared_trainable > 0:
                    print(f"   ⚠️  shared_graph имеет {shared_trainable:,} обучаемых параметров!")
                if orient_trainable > 0:
                    print(f"   ⚠️  orientation_module имеет {orient_trainable:,} обучаемых параметров!")

        agent.post_interaction = diagnostic_post_interaction

        print("✅ Диагностический колбэк установлен (проверка каждые 500 шагов)")
        print(f"{'='*80}\n")
        # ========================================================================
        # ЖЁСТКАЯ ЗАМОРОЗКА PREPROCESSOR - ПЕРЕОПРЕДЕЛЯЕМ МЕТОД _update
        # ========================================================================
        print(f"\n{'='*80}")
        print("ЖЁСТКАЯ ЗАМОРОЗКА PREPROCESSOR В _update")
        print(f"{'='*80}")

        # Сохраняем оригинальный _update
        original_update = agent._update

        def frozen_update(timestep: int, timesteps: int) -> None:
            """Обёртка над _update, которая заменяет train=True на train=False для preprocessor"""
            
            # Временно переопределяем _state_preprocessor
            original_preprocessor = agent._state_preprocessor
            
            # Создаём обёртку, которая игнорирует train=True
            class FrozenPreprocessorWrapper:
                def __init__(self, inner):
                    self.inner = inner
                
                def __call__(self, x, train=False, inverse=False, no_grad=True):
                    # ВСЕГДА вызываем с train=False
                    return self.inner(x, train=False, inverse=inverse, no_grad=no_grad)
                
                def __getattr__(self, name):
                    # Проксируем все остальные атрибуты к inner
                    return getattr(self.inner, name)
            
            # Подменяем preprocessor на обёртку
            agent._state_preprocessor = FrozenPreprocessorWrapper(original_preprocessor)
            
            try:
                # Вызываем оригинальный _update (он будет использовать нашу обёртку)
                original_update(timestep, timesteps)
            finally:
                # Восстанавливаем оригинальный preprocessor
                agent._state_preprocessor = original_preprocessor

        # Заменяем метод
        agent._update = frozen_update

        print("✅ Метод _update переопределён:")
        print("   - train=True автоматически заменяется на train=False")
        print("   - Preprocessor статистики НЕ будут обновляться")
        print(f"{'='*80}\n")
        # ========================================================================
        # КРИТИЧЕСКАЯ ДИАГНОСТИКА: Ground Truth + Режим модулей
        # ========================================================================
        print(f"\n{'='*80}")
        print("ДИАГНОСТИКА: Ground Truth и режимы модулей")
        print(f"{'='*80}")

        # 1. Проверяем BatchNorm/LayerNorm/Dropout в orientation_module
        print("\n🔍 Проверка слоёв в orientation_module:")
        has_problematic_layers = False
        for name, module in orientation_module.named_modules():
            if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.LayerNorm)):
                print(f"   ⚠️  Найден {type(module).__name__} в '{name}'")
                print(f"      training={module.training}")
                has_problematic_layers = True
                
                # Замораживаем
                module.eval()
                if hasattr(module, 'track_running_stats'):
                    module.track_running_stats = False
                    print(f"      ✅ track_running_stats отключен")
            
            if isinstance(module, nn.Dropout):
                print(f"   Найден Dropout(p={module.p}) в '{name}'")
                print(f"      training={module.training} (должно быть False)")

        if not has_problematic_layers:
            print("   ✅ Нет проблемных слоёв (BatchNorm/LayerNorm)")

        # 2. Оборачиваем orientation_module.forward для диагностики
        original_orient_forward = orientation_module.forward

        gt_history = {'first_batch': None, 'recent_batches': [], 'call_count': 0}

        def diagnostic_orient_forward(img, graph_emb, ground_truth_yaw=None):
            """Диагностическая обёртка для отслеживания GT и режима"""
            
            gt_history['call_count'] += 1
            
            # Проверяем режим модуля
            if orientation_module.training:
                print(f"❌❌❌ [{gt_history['call_count']}] orientation_module в TRAIN режиме!")
                orientation_module.eval()  # Принудительно переводим в eval
            
            # Сохраняем ground truth для анализа
            if ground_truth_yaw is not None:
                if gt_history['first_batch'] is None:
                    gt_history['first_batch'] = ground_truth_yaw.clone().detach().cpu()
                    print(f"\n📊 First batch GT saved:")
                    print(f"   Shape: {ground_truth_yaw.shape}")
                    print(f"   Mean: {gt_history['first_batch'].mean():.4f}")
                    print(f"   Std: {gt_history['first_batch'].std():.4f}")
                    print(f"   Min: {gt_history['first_batch'].min():.4f}")
                    print(f"   Max: {gt_history['first_batch'].max():.4f}")
                
                # Сохраняем последние 10 батчей
                gt_history['recent_batches'].append(ground_truth_yaw.clone().detach().cpu())
                if len(gt_history['recent_batches']) > 10:
                    gt_history['recent_batches'].pop(0)
            
            # Вызываем оригинальный forward
            return original_orient_forward(img, graph_emb, ground_truth_yaw)

        orientation_module.forward = diagnostic_orient_forward

        # 3. Добавляем анализ в post_interaction
        original_post = agent.post_interaction

        def diagnostic_post_interaction(timestep, timesteps):
            # Вызываем оригинальный
            original_post(timestep, timesteps)
            
            # Анализ каждые 500 шагов
            if timestep % 500 == 0 and timestep > 0 and len(gt_history['recent_batches']) > 0:
                first = gt_history['first_batch']
                recent = torch.cat(gt_history['recent_batches'][-5:], dim=0)
                
                print(f"\n{'='*60}")
                print(f"🔍 [{timestep}] GROUND TRUTH АНАЛИЗ")
                print(f"{'='*60}")
                
                print(f"\n📊 Statistics:")
                print(f"   First batch:")
                print(f"      Mean: {first.mean():.4f}, Std: {first.std():.4f}")
                print(f"      Range: [{first.min():.4f}, {first.max():.4f}]")
                
                print(f"   Recent batches (last 5):")
                print(f"      Mean: {recent.mean():.4f}, Std: {recent.std():.4f}")
                print(f"      Range: [{recent.min():.4f}, {recent.max():.4f}]")
                
                # КРИТИЧНО: Проверяем изменение диапазона
                mean_diff = abs(first.mean() - recent.mean())
                std_diff = abs(first.std() - recent.std())
                min_diff = abs(first.min() - recent.min())
                max_diff = abs(first.max() - recent.max())
                
                print(f"\n📈 Changes from first:")
                print(f"   Mean diff: {mean_diff:.4f}")
                print(f"   Std diff: {std_diff:.4f}")
                print(f"   Min diff: {min_diff:.4f}")
                print(f"   Max diff: {max_diff:.4f}")
                
                # Предупреждения
                if min_diff > 0.5 or max_diff > 0.5:
                    print(f"\n   ⚠️⚠️⚠️ GROUND TRUTH ДИАПАЗОН СИЛЬНО ИЗМЕНИЛСЯ!")
                    print(f"   Это объясняет падение accuracy!")
                elif mean_diff > 0.3 or std_diff > 0.3:
                    print(f"\n   ⚠️  Ground truth распределение изменилось")
                else:
                    print(f"\n   ✅ Ground truth стабилен")
                
                # Проверяем режимы модулей
                print(f"\n🔧 Module modes:")
                print(f"   shared_graph.training: {shared_graph.training}")
                print(f"   orientation_module.training: {orientation_module.training}")
                print(f"   orientation_module.net.training: {orientation_module.net.training}")
                
                print(f"{'='*60}\n")

        agent.post_interaction = diagnostic_post_interaction

        print("\n✅ Диагностика установлена:")
        print("   - Отслеживание ground_truth изменений")
        print("   - Проверка режима orientation_module")
        print("   - Анализ каждые 500 шагов")
        print(f"{'='*80}\n")
    trainer.train()
else:
    cfg_trainer = {"timesteps": 1500}
    trainer = SequentialTrainer(cfg=cfg_trainer, env=env, agents=agent)

    checkpoint_path = "/home/xiso/IsaacLab/logs/skrl/aloha_ppo_orientation/26-02-12_18-08-06-880417_SAC/checkpoints/agent_500.pt"
    agent.load(checkpoint_path)

    trainer.eval()