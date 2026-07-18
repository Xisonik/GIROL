import argparse
import torch
import torch.nn as nn

# skrl / Isaac Lab imports
from skrl.agents.torch.sac import SAC, SAC_DEFAULT_CONFIG
from skrl.envs.loaders.torch import load_isaaclab_env
from skrl.envs.wrappers.torch import wrap_env
from skrl.memories.torch import RandomMemory
from skrl.models.torch import DeterministicMixin, GaussianMixin, Model
from skrl.resources.preprocessors.torch import RunningStandardScaler, PartialRunningStandardScaler
from skrl.trainers.torch import SequentialTrainer
from skrl.utils import set_seed
from skrl.utils.spaces.torch import unflatten_tensorized_space

# GNN
from torch_geometric.nn import GATv2Conv, global_mean_pool

# ---------------------------------------------------------------------
# Глобальные настройки сцены / графа
# ---------------------------------------------------------------------

# Должно совпадать с количеством объектов в SceneManager / encode_scene_graph
NUM_GRAPH_NODES = 17          # M
PER_OBJECT_DIM = 24           # столько фич на объект из encode_scene_graph
TEXT_EMB_DIM = 512             # размер текстового эмбеддинга (имя+цвет)
GRAPH_EMB_DIM = 128           # выход графового энкодера

# В графе "цель ↔ объекты" нужно знать индекс узла цели.
# Если цель у тебя всегда первая в encode_scene_graph — оставляй 0.
GOAL_NODE_INDEX = 0

# ---------------------------------------------------------------------
# Edge builders
# ---------------------------------------------------------------------

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

    def _get_edge_index(self, B: int, device: torch.device) -> torch.Tensor:
        key = (B, device.index if device.type == "cuda" else -1)
        ei = self._edge_cache.get(key, None)
        if ei is None or ei.device != device:
            ei = build_goal_star_edge_index(
                num_nodes=self.num_nodes,
                batch_size=B,
                device=device,
                goal_index=self.goal_index,
                add_self_loops=True
            )
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
    """Lookup оффлайн CLIP-эмбеддингов (512). Без проекции.

    embeddings_path должен содержать:
      - name_embs:  [N_names, 512]
      - color_embs: [N_colors, 512]
    """
    def __init__(self, embeddings_path: str):
        super().__init__()

        payload = torch.load(embeddings_path, map_location="cpu")
        name_embs = payload.get("name_embs", None)
        color_embs = payload.get("color_embs", None)
        if name_embs is None or color_embs is None:
            raise ValueError(f"Bad embeddings file: expected keys 'name_embs' and 'color_embs' in {embeddings_path}")

        self.register_buffer("name_embs", name_embs.float(), persistent=False)    # [N_names, 512]
        self.register_buffer("color_embs", color_embs.float(), persistent=False) # [N_colors, 512]

    def forward(self, name_idx: torch.Tensor, color_bits_or_idx: torch.Tensor) -> torch.Tensor:
        """
        return: [B, N, 512]
        """
        if name_idx.dim() == 3:
            name_idx = name_idx.argmax(dim=-1)
        name_idx = name_idx.long()

        if color_bits_or_idx.dim() == 3 and color_bits_or_idx.size(-1) == 3:
            bits = color_bits_or_idx.round().long().clamp(0, 1)
            color_idx = (bits[..., 0] * 4 + bits[..., 1] * 2 + bits[..., 2]) - 1
        else:
            color_idx = color_bits_or_idx.round().long()

        name_idx = name_idx.clamp(0, self.name_embs.shape[0] - 1)
        color_idx = color_idx.clamp(0, self.color_embs.shape[0] - 1)

        emb_name = self.name_embs[name_idx]       # [B,N,512]
        emb_color = self.color_embs[color_idx]    # [B,N,512]
        emb = 0.5 * (emb_name + emb_color)        # [B,N,512]
        return emb



class SharedGraphModule(nn.Module):
    """Общий графовый энкодер: node_raw (24) + text_emb (16) -> GAT -> graph_emb (128)."""
    def __init__(self, embeddings_path: str, num_nodes: int = NUM_GRAPH_NODES,
                 per_object_dim: int = PER_OBJECT_DIM, text_dim: int = TEXT_EMB_DIM):
        super().__init__()
        self.num_nodes = num_nodes
        self.per_object_dim = per_object_dim
        self.text_dim = text_dim

        self.text_encoder = FrozenCLIPNameColorEncoder(embeddings_path=embeddings_path)
        self.graph_encoder = SceneGraphGATEncoder(
            num_nodes=num_nodes,
            node_in_dim=per_object_dim + 512,   # <-- 24 + 512 = 536
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
# Models
# ---------------------------------------------------------------------

class StochasticActor(GaussianMixin, Model):
    """Actor: принимает dict-obs: {img, graph}. Графовый энкодер НЕ обучается актором."""
    def __init__(self, observation_space, action_space, device, shared_graph: SharedGraphModule,
                 clip_actions=False, clip_log_std=True, min_log_std=-5, max_log_std=2):
        Model.__init__(self, observation_space, action_space, device)
        GaussianMixin.__init__(self, clip_actions, clip_log_std, min_log_std, max_log_std)
        self.device = device

        # ВАЖНО: shared_graph не регистрируем как submodule в акторе,
        # чтобы актор-оптимизатор не трогал его параметры.
        self.__dict__["shared_graph"] = shared_graph

        # img — это “нормализуемая” часть
        self.img_dim = int(observation_space["img"].shape[0])

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

    def compute(self, inputs, role):
        B = inputs["states"].shape[0]
        states = unflatten_tensorized_space(self.observation_space, inputs["states"])
        img = states["img"].to(self.device)          # [B, img_dim]
        graph_flat = states["graph"].to(self.device) # [B, N*24]
        print("graph_flat ", graph_flat)

        # Энкодер обучает критик -> для актора no_grad
        with torch.no_grad():
            graph_emb = self.shared_graph(graph_flat)  # [B, 128]

        x = torch.cat([img, graph_emb], dim=-1)
        mu = self.net(x)
        return mu, self.log_std_parameter, {}


class Critic(DeterministicMixin, Model):
    """Critic: Q(s,a). Здесь shared_graph обучается (градиенты идут в него)."""
    def __init__(self, observation_space, action_space, device, shared_graph: SharedGraphModule,
                 clip_actions=False, train_graph: bool = True):
        Model.__init__(self, observation_space, action_space, device)
        DeterministicMixin.__init__(self, clip_actions)
        self.device = device

        # В critic_1 train_graph=True: shared_graph обучается.
        # В critic_2 и target critics train_graph=False: shared_graph используется в no_grad().
        self.train_graph = bool(train_graph)

        # НЕ регистрируем как submodule, чтобы избежать двойного шага оптимизатора,
        # если shared_graph будет привязан к нескольким критикам.
        self.__dict__["shared_graph"] = shared_graph

        self.img_dim = int(observation_space["img"].shape[0])

        mlp_in = self.img_dim + GRAPH_EMB_DIM + self.num_actions
        self.net = nn.Sequential(
            nn.Linear(mlp_in, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        ).to(device)

    def compute(self, inputs, role):
        B = inputs["states"].shape[0]
        states = unflatten_tensorized_space(self.observation_space, inputs["states"])
        img = states["img"].to(self.device)            # [B, img_dim]
        graph_flat = states["graph"].to(self.device)   # [B, N*24]
        actions = inputs["taken_actions"].to(self.device)

        if self.train_graph:
            graph_emb = self.shared_graph(graph_flat)
        else:
            with torch.no_grad():
                graph_emb = self.shared_graph(graph_flat)

        x = torch.cat([img, graph_emb, actions], dim=-1)
        q = self.net(x)
        return q, {}


from skrl.resources.preprocessors.torch.running_standard_scaler import RunningStandardScaler
from skrl.utils.spaces.torch import unflatten_tensorized_space, flatten_tensorized_space


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

        # 3) свернуть обратно (в твоей версии skrl — только 1 аргумент)
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
# Memory
# ---------------------------------------------------------------------
memory = RandomMemory(memory_size=6000, num_envs=env.num_envs, device=device)

# ---------------------------------------------------------------------
# Shared graph module (one instance)
# ---------------------------------------------------------------------
shared_graph = SharedGraphModule(
    embeddings_path="/home/xiso/IsaacLab/source/isaaclab_tasks/isaaclab_tasks/direct/aloha/text_embeddings.pt",
    num_nodes=NUM_GRAPH_NODES,
    per_object_dim=PER_OBJECT_DIM,
).to(device)

# ---------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------
models = {
    "policy": StochasticActor(env.observation_space, env.action_space, device, shared_graph=shared_graph),
    # критик 1 обучает shared_graph
    "critic_1": Critic(env.observation_space, env.action_space, device, shared_graph=shared_graph, train_graph=True),
    # критик 2 НЕ обучает shared_graph (чтобы не делать двойной шаг оптимизатора)
    "critic_2": Critic(env.observation_space, env.action_space, device, shared_graph=shared_graph, train_graph=False),
    # таргеты никогда не обучают shared_graph
    "target_critic_1": Critic(env.observation_space, env.action_space, device, shared_graph=shared_graph, train_graph=False),
    "target_critic_2": Critic(env.observation_space, env.action_space, device, shared_graph=shared_graph, train_graph=False),
}

# ---------------------------------------------------------------------
# SAC config
# ---------------------------------------------------------------------
cfg = SAC_DEFAULT_CONFIG.copy()
cfg["gradient_steps"] = 4
cfg["batch_size"] = 256
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
    "size": env.observation_space,              # ВАЖНО: полный dict-space
    "img_space": env.observation_space["img"],  # Нормализуем только img
    "device": device,
}



cfg["experiment"]["write_interval"] = 100
cfg["experiment"]["checkpoint_interval"] = 1000
cfg["experiment"]["directory"] = "logs/skrl/aloha_sac_graph"

agent = SAC(
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
    cfg_trainer = {"timesteps": 330000}
    trainer = SequentialTrainer(cfg=cfg_trainer, env=env, agents=agent)
    trainer.train()
else:
    cfg_trainer = {"timesteps": 1000}
    trainer = SequentialTrainer(cfg=cfg_trainer, env=env, agents=agent)

    checkpoint_path = "/home/xiso/IsaacLab/logs/skrl/aloha_sac_graph/25-12-27_01-45-04-706276_SAC/checkpoints/best_agent.pt"
    agent.load(checkpoint_path)

    trainer.eval()
