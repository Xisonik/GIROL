import argparse
import torch
import torch.nn as nn

# skrl / Isaac Lab imports
from skrl.agents.torch.ppo import PPO_RNN as PPO, PPO_DEFAULT_CONFIG
from skrl.envs.loaders.torch import load_isaaclab_env
from skrl.envs.wrappers.torch import wrap_env
from skrl.memories.torch import RandomMemory
from skrl.models.torch import DeterministicMixin, GaussianMixin, Model
from skrl.resources.preprocessors.torch import RunningStandardScaler
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

# LSTM settings
LSTM_HIDDEN_SIZE = 256
LSTM_NUM_LAYERS = 1
LSTM_SEQUENCE_LENGTH = 16     # длина последовательности для LSTM

# В графе "цель ↔ объекты" нужно знать индекс узла цели.
# Если цель у тебя всегда первая в encode_scene_graph — оставляй 0.
GOAL_NODE_INDEX = 0

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
# CLI аргументы / seed
# ---------------------------------------------------------------------
EVAL = False
# EVAL = True

set_seed(42)

# ---------------------------------------------------------------------
# Models для PPO с LSTM
# ---------------------------------------------------------------------

class Policy(GaussianMixin, Model):
    """Policy (Actor) с LSTM: принимает dict-obs: {img, graph}. 
    В PPO графовый энкодер ОБУЧАЕТСЯ вместе с policy."""
    def __init__(self, observation_space, action_space, device, shared_graph: SharedGraphModule,
                 clip_actions=False, clip_log_std=True, min_log_std=-20, max_log_std=2,
                 reduction="sum"):
        Model.__init__(self, observation_space, action_space, device)
        GaussianMixin.__init__(self, clip_actions, clip_log_std, min_log_std, max_log_std, reduction)
        self.device = device

        # В PPO shared_graph регистрируем как submodule policy - он будет обучаться!
        self.shared_graph = shared_graph

        # img — это "нормализуемая" часть
        self.img_dim = int(observation_space["img"].shape[0])

        # Feature extractor: img + graph -> embedding
        feature_dim = self.img_dim + GRAPH_EMB_DIM
        self.feature_net = nn.Sequential(
            nn.Linear(feature_dim, 512),
            nn.ELU(),
            nn.Linear(512, LSTM_HIDDEN_SIZE),
            nn.ELU()
        ).to(device)

        # LSTM layer
        # ВАЖНО: PPO_RNN автоматически определит тип по классу слоя (nn.LSTM)
        # Если нужен GRU - используйте nn.GRU, для vanilla RNN - nn.RNN
        self.lstm = nn.LSTM(
            input_size=LSTM_HIDDEN_SIZE,
            hidden_size=LSTM_HIDDEN_SIZE,
            num_layers=LSTM_NUM_LAYERS,
            batch_first=True  # input shape: (batch, seq, feature)
        ).to(device)

        # Output head
        self.output_net = nn.Sequential(
            nn.Linear(LSTM_HIDDEN_SIZE, 256),
            nn.ELU(),
            nn.Linear(256, self.num_actions),
            nn.Tanh()
        ).to(device)

        self.log_std_parameter = nn.Parameter(torch.zeros(self.num_actions, device=device))

    # def get_specification(self):
    #     """Указываем, что модель использует RNN.
        
    #     PPO_RNN автоматически определит тип RNN (LSTM/GRU/RNN) по классу слоя в модели:
    #     - nn.LSTM → использует LSTM логику (как в этой модели)
    #     - nn.GRU → использует GRU логику  
    #     - nn.RNN → использует vanilla RNN логику
        
    #     Ключ "rnn" в спецификации нужен только чтобы указать sequence_length и sizes.
    #     """
    #     return {
    #         "rnn": {
    #             "sequence_length": LSTM_SEQUENCE_LENGTH,
    #             "sizes": [LSTM_HIDDEN_SIZE, LSTM_NUM_LAYERS],  # [hidden_size, num_layers]
    #         }
    #     }

    def compute(self, inputs, role):
        # Получаем состояния
        states = inputs["states"]
        terminated = inputs.get("terminated", None)
        hidden_states = inputs.get("rnn", [None, None])
        
        # Unflatten dict observation
        B = states.shape[0]
        states_dict = unflatten_tensorized_space(self.observation_space, states)
        img = states_dict["img"].to(self.device)          # [B, img_dim]
        graph_flat = states_dict["graph"].to(self.device) # [B, N*24]

        # Encode graph (с градиентами для policy)
        graph_emb = self.shared_graph(graph_flat)  # [B, 128]

        # Combine features
        x = torch.cat([img, graph_emb], dim=-1)
        features = self.feature_net(x)  # [B, LSTM_HIDDEN_SIZE]

        # Проверяем формат hidden states
        # hidden_states может быть:
        # 1. [h, c] где h и c это тензоры
        # 2. None для обоих
        if hidden_states[0] is None or hidden_states[1] is None:
            # Инициализируем hidden states
            h = torch.zeros(LSTM_NUM_LAYERS, B, LSTM_HIDDEN_SIZE, 
                          device=self.device, dtype=features.dtype)
            c = torch.zeros(LSTM_NUM_LAYERS, B, LSTM_HIDDEN_SIZE,
                          device=self.device, dtype=features.dtype)
        else:
            h, c = hidden_states
            # Убеждаемся, что они правильной формы [num_layers, B, hidden_size]
            if h.dim() == 2:  # [B, hidden_size]
                h = h.unsqueeze(0)  # [1, B, hidden_size]
            if c.dim() == 2:
                c = c.unsqueeze(0)

        # LSTM forward
        # features: [B, LSTM_HIDDEN_SIZE] -> unsqueeze -> [B, 1, LSTM_HIDDEN_SIZE]
        features = features.unsqueeze(1)  # добавляем sequence dimension
        
        rnn_output, (h_new, c_new) = self.lstm(features, (h, c))
        # rnn_output: [B, 1, LSTM_HIDDEN_SIZE]
        
        rnn_output = rnn_output.squeeze(1)  # [B, LSTM_HIDDEN_SIZE]

        # Сбрасываем hidden states на terminated эпизодах
        if terminated is not None:
            terminated = terminated.view(-1, 1)  # [B, 1]
            # h_new, c_new: [num_layers, B, hidden_size]
            h_new = h_new * (1 - terminated).transpose(0, 1).unsqueeze(0)
            c_new = c_new * (1 - terminated).transpose(0, 1).unsqueeze(0)

        # Output
        mu = self.output_net(rnn_output)

        return mu, self.log_std_parameter, {"rnn": [h_new, c_new]}


class Value(DeterministicMixin, Model):
    """Value function с LSTM: V(s). Использует тот же shared_graph через no_grad."""
    def __init__(self, observation_space, action_space, device, shared_graph: SharedGraphModule,
                 clip_actions=False):
        Model.__init__(self, observation_space, action_space, device)
        DeterministicMixin.__init__(self, clip_actions)
        self.device = device

        # Value function НЕ обучает shared_graph - использует его через no_grad
        self.__dict__["shared_graph"] = shared_graph

        self.img_dim = int(observation_space["img"].shape[0])

        # Feature extractor
        feature_dim = self.img_dim + GRAPH_EMB_DIM
        self.feature_net = nn.Sequential(
            nn.Linear(feature_dim, 512),
            nn.ELU(),
            nn.Linear(512, LSTM_HIDDEN_SIZE),
            nn.ELU()
        ).to(device)

        # LSTM layer
        # ВАЖНО: PPO_RNN автоматически определит тип по классу слоя (nn.LSTM)
        self.lstm = nn.LSTM(
            input_size=LSTM_HIDDEN_SIZE,
            hidden_size=LSTM_HIDDEN_SIZE,
            num_layers=LSTM_NUM_LAYERS,
            batch_first=True
        ).to(device)

        # Output head
        self.output_net = nn.Sequential(
            nn.Linear(LSTM_HIDDEN_SIZE, 256),
            nn.ELU(),
            nn.Linear(256, 1)
        ).to(device)

    # def get_specification(self):
    #     """Указываем, что модель использует RNN.
        
    #     PPO_RNN автоматически определит тип RNN (LSTM/GRU/RNN) по классу слоя в модели.
    #     """
    #     return {
    #         "rnn": {
    #             "sequence_length": LSTM_SEQUENCE_LENGTH,
    #             "sizes": [LSTM_HIDDEN_SIZE, LSTM_NUM_LAYERS],
    #         }
    #     }

    def compute(self, inputs, role):
        states = inputs["states"]
        terminated = inputs.get("terminated", None)
        hidden_states = inputs.get("rnn", [None, None])
        
        B = states.shape[0]
        states_dict = unflatten_tensorized_space(self.observation_space, states)
        img = states_dict["img"].to(self.device)
        graph_flat = states_dict["graph"].to(self.device)

        # Value function не обучает graph
        with torch.no_grad():
            graph_emb = self.shared_graph(graph_flat)

        x = torch.cat([img, graph_emb], dim=-1)
        features = self.feature_net(x)

        # Handle hidden states
        if hidden_states[0] is None or hidden_states[1] is None:
            h = torch.zeros(LSTM_NUM_LAYERS, B, LSTM_HIDDEN_SIZE,
                          device=self.device, dtype=features.dtype)
            c = torch.zeros(LSTM_NUM_LAYERS, B, LSTM_HIDDEN_SIZE,
                          device=self.device, dtype=features.dtype)
        else:
            h, c = hidden_states
            if h.dim() == 2:
                h = h.unsqueeze(0)
            if c.dim() == 2:
                c = c.unsqueeze(0)

        features = features.unsqueeze(1)
        rnn_output, (h_new, c_new) = self.lstm(features, (h, c))
        rnn_output = rnn_output.squeeze(1)

        # Reset on terminated
        if terminated is not None:
            terminated = terminated.view(-1, 1)
            h_new = h_new * (1 - terminated).transpose(0, 1).unsqueeze(0)
            c_new = c_new * (1 - terminated).transpose(0, 1).unsqueeze(0)

        v = self.output_net(rnn_output)

        return v, {"rnn": [h_new, c_new]}


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
        headless=True,
        cli_args=["--enable_cameras", "--video"],
    )
    env = RecordVideo(
        env,
        video_folder="logs/skrl/aloha/videos",
        name_prefix="aloha_eval",
        episode_trigger=lambda ep: True,
    )
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

# ---------------------------------------------------------------------
# Memory для PPO с LSTM
# ---------------------------------------------------------------------
# Важно: memory_size должен быть кратен sequence_length
memory = RandomMemory(memory_size=LSTM_SEQUENCE_LENGTH * 3, num_envs=env.num_envs, device=device)

# ---------------------------------------------------------------------
# Shared graph module (one instance)
# ---------------------------------------------------------------------
shared_graph = SharedGraphModule(
    embeddings_path="/home/xiso/IsaacLab/source/isaaclab_tasks/isaaclab_tasks/direct/aloha/text_embeddings.pt",
    num_nodes=NUM_GRAPH_NODES,
    per_object_dim=PER_OBJECT_DIM,
    text_dim=TEXT_EMB_DIM,
).to(device)

# ---------------------------------------------------------------------
# Models для PPO с LSTM
# ---------------------------------------------------------------------
models = {
    "policy": Policy(env.observation_space, env.action_space, device, shared_graph=shared_graph),
    "value": Value(env.observation_space, env.action_space, device, shared_graph=shared_graph),
}

# ---------------------------------------------------------------------
# PPO config для LSTM
# ---------------------------------------------------------------------
cfg = PPO_DEFAULT_CONFIG.copy()

# Memory - золотая середина для LSTM
cfg["rollouts"] = 64  # было 48→128, теперь 64 (компромисс)
cfg["learning_epochs"] = 8  # было 5 → увеличили, чтобы лучше использовать rollout
cfg["mini_batches"] = 8  # было 4 → увеличили

cfg["discount_factor"] = 0.99
cfg["lambda"] = 0.95

# Learning rates
cfg["learning_rate"] = 3e-4  
cfg["learning_rate_scheduler"] = None

# PPO-specific
cfg["ratio_clip"] = 0.2
cfg["value_clip"] = 0.2
cfg["clip_predicted_values"] = True

# Regularization - ПОСТЕПЕННЫЙ EXPLORATION
cfg["entropy_loss_scale"] = 0.02  # было 0.01→0.05, теперь 0.02 (средне)
cfg["value_loss_scale"] = 1.0     # было 0.5 → увеличили, чтобы Value лучше училась
cfg["grad_norm_clip"] = 1.0       # было 0.5 → ОК

# State preprocessor
cfg["state_preprocessor"] = DictRunningStandardScaler
cfg["state_preprocessor_kwargs"] = {
    "size": env.observation_space,
    "img_space": env.observation_space["img"],
    "device": device,
}

cfg["experiment"]["write_interval"] = 100
cfg["experiment"]["checkpoint_interval"] = 1000
cfg["experiment"]["directory"] = "logs/skrl/aloha_sac_graph"

agent = PPO(
    models=models,
    memory=memory,
    cfg=cfg,
    observation_space=env.observation_space,
    action_space=env.action_space,
    device=device,
)

# ---------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------
if not EVAL:
    cfg_trainer = {"timesteps": 330000, "headless": True}
    trainer = SequentialTrainer(cfg=cfg_trainer, env=env, agents=agent)
    # Если есть чекпоинт - раскомментируй:
    # checkpoint_path = "/home/xiso/IsaacLab/logs/skrl/aloha_ppo_lstm_graph/CHECKPOINT.pt"
    # agent.load(checkpoint_path)
    trainer.train()
else:
    cfg_trainer = {"timesteps": 1000, "headless": True}
    trainer = SequentialTrainer(cfg=cfg_trainer, env=env, agents=agent)

    checkpoint_path = "/home/xiso/IsaacLab/logs/skrl/aloha_ppo_lstm_graph/CHECKPOINT.pt"
    agent.load(checkpoint_path)

    trainer.eval()