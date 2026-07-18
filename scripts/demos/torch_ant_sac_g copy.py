import argparse
import torch
import torch.nn as nn

# skrl / Isaac Lab imports
from skrl.agents.torch.sac import SAC, SAC_DEFAULT_CONFIG
from skrl.envs.loaders.torch import load_isaaclab_env
from skrl.envs.wrappers.torch import wrap_env
from skrl.memories.torch import RandomMemory
from skrl.models.torch import DeterministicMixin, GaussianMixin, Model
from skrl.resources.preprocessors.torch import RunningStandardScaler
from skrl.trainers.torch import SequentialTrainer
from skrl.utils import set_seed

# GNN
from torch_geometric.nn import GATv2Conv, global_mean_pool

# ---------------------------------------------------------------------
# Глобальные настройки сцены / графа
# ---------------------------------------------------------------------

# Должно совпадать с количеством объектов в SceneManager / encode_scene_graph
NUM_GRAPH_NODES = 17          # M
PER_OBJECT_DIM = 24           # столько фич на объект из encode_scene_graph
TEXT_EMB_DIM = 16             # размер текстового эмбеддинга (имя+цвет)
NAME_VOCAB_SIZE = 100         # сколько разных кодов имён (0..99)
COLOR_VOCAB_SIZE = 7          # 7 трёхбитовых кодов (001..111)
GRAPH_EMB_DIM = 128           # выход графового энкодера

# ---------------------------------------------------------------------
# Вспомогательные функции / модули
# ---------------------------------------------------------------------

def build_fully_connected_edge_index(num_nodes, batch_size, device):
    """Полносвязный граф для батча из B сцен."""
    row, col = torch.meshgrid(
        torch.arange(num_nodes, device=device),
        torch.arange(num_nodes, device=device),
        indexing="ij"
    )
    edge_index_single = torch.stack([row.flatten(), col.flatten()], dim=0)  # [2, N^2]
    edge_indices = []
    for b in range(batch_size):
        edge_indices.append(edge_index_single + b * num_nodes)
    edge_index = torch.cat(edge_indices, dim=1)  # [2, B * N^2]
    return edge_index


class SceneGraphGATEncoder(nn.Module):
    """Простой GATv2-encoder для сценового графа без edge_attr."""
    def __init__(
        self,
        num_nodes: int,
        node_in_dim: int,
        hidden_dim: int = 256,
        out_dim: int = GRAPH_EMB_DIM,
        num_layers: int = 3,
        heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.num_nodes = num_nodes
        self.out_dim = out_dim
        self.dropout = dropout

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

    def forward(self, node_feats, batch_size: int):
        """
        node_feats: [B * N, node_in_dim]
        возвращает: [B, out_dim]
        """
        device = node_feats.device
        B = batch_size
        N = self.num_nodes

        # [B*N, d] -> [B*N, hidden]
        x = self.node_mlp(node_feats)

        # edge_index и batch
        edge_index = build_fully_connected_edge_index(N, B, device)
        batch = torch.repeat_interleave(
            torch.arange(B, device=device),
            repeats=N
        )  # [B*N]

        for conv, norm in zip(self.convs, self.norms):
            x = conv(x, edge_index)      # [B*N, hidden]
            x = norm(x)
            x = torch.relu(x)
            x = nn.functional.dropout(x, p=self.dropout, training=self.training)

        g = global_mean_pool(x, batch)   # [B, hidden]
        return self.head(g)              # [B, out_dim]



class FrozenCLIPNameColorEncoder(nn.Module):
    """Семантический энкодер (name_idx + color_idx) на основе оффлайн CLIP-эмбеддингов.

    - CLIP-эмбеддинги (512) для всех уникальных имён и цветов посчитаны оффлайн и сохранены в .pt
    - Здесь мы делаем lookup по индексам и прогоняем через один общий обучаемый MLP 512 -> TEXT_EMB_DIM
    """
    def __init__(self, embeddings_path: str, text_dim: int = TEXT_EMB_DIM):
        super().__init__()
        self.text_dim = text_dim

        payload = torch.load(embeddings_path, map_location="cpu")
        name_embs = payload.get("name_embs", None)
        color_embs = payload.get("color_embs", None)
        if name_embs is None or color_embs is None:
            raise ValueError(f"Bad embeddings file: expected keys 'name_embs' and 'color_embs' in {embeddings_path}")

        # buffers: не обучаем, но .to(device) их переносит
        self.register_buffer("name_embs", name_embs.float(), persistent=False)    # [N_names, 512]
        self.register_buffer("color_embs", color_embs.float(), persistent=False) # [N_colors, 512]

        # Один общий обучаемый проектор: 512 -> TEXT_EMB_DIM
        self.proj = nn.Sequential(
            nn.Linear(self.name_embs.shape[-1], 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, text_dim),
        )

    def forward(self, name_idx: torch.Tensor, color_idx: torch.Tensor) -> torch.Tensor:
        """
        Поддерживаем 2 формата входа:
        name_idx:
            - [B, N] (индекс)
            - [B, N, K] (one-hot / logits) -> argmax
        color_idx:
            - [B, N] (индекс)
            - [B, N, 3] (биты 0/1) -> id 0..6
        Возврат: [B, N, text_dim]
        """

        # --- name: если one-hot/логиты -> индекс
        if name_idx.dim() == 3:
            # [B,N,K] -> [B,N]
            name_idx = name_idx.argmax(dim=-1)
        name_idx = name_idx.long()

        # --- color: если биты -> индекс 0..6
        if color_idx.dim() == 3 and color_idx.size(-1) == 3:
            bits = color_idx.round().long().clamp(0, 1)  # [B,N,3]
            # (4,2,1) даёт 1..7, затем -1 -> 0..6
            color_idx = (bits[..., 0] * 4 + bits[..., 1] * 2 + bits[..., 2]) - 1
        color_idx = color_idx.long()

        # clamp
        name_idx = name_idx.clamp(0, self.name_embs.shape[0] - 1)
        color_idx = color_idx.clamp(0, self.color_embs.shape[0] - 1)

        emb_name = self.name_embs[name_idx]     # [B,N,512]
        emb_color = self.color_embs[color_idx]  # [B,N,512]
        emb = 0.5 * (emb_name + emb_color)      # [B,N,512]

        return self.proj(emb)                   # [B,N,text_dim]



# ---------------------------------------------------------------------
# CLI аргументы / seed
# ---------------------------------------------------------------------
EVAL = False
set_seed(42)

# ---------------------------------------------------------------------
# Модели
# ---------------------------------------------------------------------
class StochasticActor(GaussianMixin, Model):
    """
    Actor:
      - вход: полный вектор obs (флэт)
      - последние NUM_GRAPH_NODES * PER_OBJECT_DIM фич = граф сцены
      - из графа достаём:
          * сырые 24 фичи на объект
          * name_code (index 20)
          * color_bits (21:24)
        -> через общий NameColorTextEncoder получаем семантический emb
        -> конкатим с 24 фичами и кормим в GATv2
      - результат GNN + "обычная" часть obs идут в MLP политики.
    """
    def __init__(self, observation_space, action_space, device, clip_actions=False,
                 clip_log_std=True, min_log_std=-5, max_log_std=2):
        Model.__init__(self, observation_space, action_space, device)
        GaussianMixin.__init__(self, clip_actions, clip_log_std, min_log_std, max_log_std)

        self.device = device

        self.num_nodes = NUM_GRAPH_NODES
        self.node_dim = PER_OBJECT_DIM
        self.text_dim = TEXT_EMB_DIM

        total_obs_dim = self.num_observations
        self.graph_dim = self.num_nodes * self.node_dim
        assert total_obs_dim >= self.graph_dim, \
            f"num_observations={total_obs_dim} меньше чем graph_dim={self.graph_dim}"

        # часть obs, не относящаяся к графу (embedding, скорости, целевая позиция и т.п.)
        self.base_obs_dim = total_obs_dim - self.graph_dim

        # семантический encoder: lookup оффлайн-CLIP (512) -> обучаемая проекция 512->TEXT_EMB_DIM
        self.text_encoder = FrozenCLIPNameColorEncoder(
            embeddings_path="/home/xiso/IsaacLab/source/isaaclab_tasks/isaaclab_tasks/direct/aloha/text_embeddings.pt",
            text_dim=self.text_dim,
        ).to(device)

        # графовый encoder
        node_in_dim = self.node_dim + self.text_dim
        self.graph_encoder = SceneGraphGATEncoder(
            num_nodes=self.num_nodes,
            node_in_dim=node_in_dim,
            hidden_dim=256,
            out_dim=GRAPH_EMB_DIM,
            num_layers=3,
            heads=4,
            dropout=0.1,
        ).to(device)

        mlp_in = self.base_obs_dim + GRAPH_EMB_DIM
        self.net = nn.Sequential(
            nn.Linear(mlp_in, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, self.num_actions),
            nn.Tanh()
        ).to(device)

        self.log_std_parameter = nn.Parameter(torch.zeros(self.num_actions, device=device))

    def _split_obs(self, states: torch.Tensor):
        """
        Разделяем obs на:
          - base_obs: всё, кроме графа
          - graph_flat: последние graph_dim фич
        """
        base_obs = states[:, :-self.graph_dim]         # [B, base_obs_dim]
        graph_flat = states[:, -self.graph_dim:]       # [B, graph_dim]
        return base_obs, graph_flat

    def _build_node_features(self, graph_flat: torch.Tensor):
        """
        graph_flat: [B, graph_dim] = [B, N * 24]
        Возвращает:
          node_feats: [B * N, node_in_dim]
        """
        B = graph_flat.shape[0]
        N = self.num_nodes

        node_raw = graph_flat.view(B, N, self.node_dim)     # [B, N, 24]

        # name_code и color_bits внутри 24-фичового вектора
        name_idx = node_raw[..., 20].round().long()       # [B, N] индекс имени в name_embs
        color_idx = node_raw[..., 21].round().long()      # [B, N] индекс цвета в color_embs

        # семантический emb: lookup CLIP (512) + обучаемая проекция 512->text_dim
        text_emb = self.text_encoder(name_idx, color_idx)    # [B, N, text_dim]
# финальный вектор узла = сырые 24 фичи + текстовый emb
        full_node = torch.cat([node_raw, text_emb], dim=-1)     # [B, N, 24 + text_dim]

        return full_node.view(B * N, -1)                        # [B*N, node_in_dim]

    def compute(self, inputs, role):
        states = inputs["states"].to(self.device)   # [B, num_observations]
        B = states.shape[0]

        base_obs, graph_flat = self._split_obs(states)          # [B, base_obs_dim], [B, graph_dim]
        node_feats = self._build_node_features(graph_flat)      # [B*N, node_in_dim]

        graph_emb = self.graph_encoder(node_feats, batch_size=B)  # [B, GRAPH_EMB_DIM]

        x = torch.cat([base_obs, graph_emb], dim=-1)            # [B, base_obs_dim + GRAPH_EMB_DIM]
        mu = self.net(x)

        return mu, self.log_std_parameter, {}


class Critic(DeterministicMixin, Model):
    """
    Critic — то же самое, что Actor, но на выходе Q(s, a).
    """
    def __init__(self, observation_space, action_space, device, clip_actions=False):
        Model.__init__(self, observation_space, action_space, device)
        DeterministicMixin.__init__(self, clip_actions)

        self.device = device

        self.num_nodes = NUM_GRAPH_NODES
        self.node_dim = PER_OBJECT_DIM
        self.text_dim = TEXT_EMB_DIM

        total_obs_dim = self.num_observations
        self.graph_dim = self.num_nodes * self.node_dim
        assert total_obs_dim >= self.graph_dim, \
            f"num_observations={total_obs_dim} меньше чем graph_dim={self.graph_dim}"

        self.base_obs_dim = total_obs_dim - self.graph_dim

        self.text_encoder = FrozenCLIPNameColorEncoder(
            embeddings_path="/home/xiso/IsaacLab/source/isaaclab_tasks/isaaclab_tasks/direct/aloha/text_embeddings.pt",
            text_dim=self.text_dim,
        ).to(device)

        node_in_dim = self.node_dim + self.text_dim
        self.graph_encoder = SceneGraphGATEncoder(
            num_nodes=self.num_nodes,
            node_in_dim=node_in_dim,
            hidden_dim=256,
            out_dim=GRAPH_EMB_DIM,
            num_layers=3,
            heads=4,
            dropout=0.1,
        ).to(device)

        mlp_in = self.base_obs_dim + GRAPH_EMB_DIM + self.num_actions
        self.net = nn.Sequential(
            nn.Linear(mlp_in, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        ).to(device)

    def _split_obs(self, states: torch.Tensor):
        base_obs = states[:, :-self.graph_dim]
        graph_flat = states[:, -self.graph_dim:]
        return base_obs, graph_flat

    def _build_node_features(self, graph_flat: torch.Tensor):
        B = graph_flat.shape[0]
        N = self.num_nodes

        node_raw = graph_flat.view(B, N, self.node_dim)     # [B, N, 24]
        name_codes = node_raw[..., 20].round().long()       # [B, N]
        color_bits = node_raw[..., 21:24]                   # [B, N, 3]

        text_emb = self.text_encoder(name_codes, color_bits)    # [B, N, text_dim]
        full_node = torch.cat([node_raw, text_emb], dim=-1)     # [B, N, 24 + text_dim]

        return full_node.view(B * N, -1)                        # [B*N, node_in_dim]

    def compute(self, inputs, role):
        states = inputs["states"].to(self.device)
        actions = inputs["taken_actions"].to(self.device)
        B = states.shape[0]

        base_obs, graph_flat = self._split_obs(states)
        node_feats = self._build_node_features(graph_flat)
        graph_emb = self.graph_encoder(node_feats, batch_size=B)

        x = torch.cat([base_obs, graph_emb, actions], dim=-1)
        q = self.net(x)
        return q, {}


# ---------------------------------------------------------------------
# Environment: train vs eval
# ---------------------------------------------------------------------
if EVAL:
    from gymnasium.wrappers import RecordVideo

    print("[INFO] Running evaluation...")
    env = load_isaaclab_env(
        task_name="Isaac-Aloha-Direct-v0",
        num_envs=1,
        headless=True,
        cli_args=[
            "--enable_cameras",
            "--video",
        ],
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
        num_envs=16,
        headless=True,
        cli_args=[
            "--enable_cameras",
        ],
    )

env = wrap_env(env)
device = env.device

# ---------------------------------------------------------------------
# Память
# ---------------------------------------------------------------------
memory = RandomMemory(memory_size=6000, num_envs=env.num_envs, device=device)

# ---------------------------------------------------------------------
# Модели агента
# ---------------------------------------------------------------------
models = {
    "policy": StochasticActor(env.observation_space, env.action_space, device),
    "critic_1": Critic(env.observation_space, env.action_space, device),
    "critic_2": Critic(env.observation_space, env.action_space, device),
    "target_critic_1": Critic(env.observation_space, env.action_space, device),
    "target_critic_2": Critic(env.observation_space, env.action_space, device),
}

# ---------------------------------------------------------------------
# Конфиг SAC
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

cfg["state_preprocessor"] = RunningStandardScaler
cfg["state_preprocessor_kwargs"] = {"size": env.observation_space, "device": device}

cfg["experiment"]["write_interval"] = 100
cfg["experiment"]["checkpoint_interval"] = 1000
cfg["experiment"]["directory"] = "logs/skrl/aloha_ppo"

agent = SAC(
    models=models,
    memory=memory,
    cfg=cfg,
    observation_space=env.observation_space,
    action_space=env.action_space,
    device=device,
)

# ---------------------------------------------------------------------
# Trainer: train / eval
# ---------------------------------------------------------------------
if not EVAL:
    cfg_trainer = {"timesteps": 33000}
    trainer = SequentialTrainer(cfg=cfg_trainer, env=env, agents=agent)
    trainer.train()
else:
    cfg_trainer = {"timesteps": 1000}
    trainer = SequentialTrainer(cfg=cfg_trainer, env=env, agents=agent)

    checkpoint_path = "/home/xiso/IsaacLab/logs/skrl/aloha_ppo/25-12-22_16-02-20-982242_SAC/checkpoints/agent_7000.pt"
    agent.load(checkpoint_path)

    trainer.eval()
