"""
Адаптированная версия основного скрипта тренировки для Ray Tune.
Принимает гиперпараметры через config dict вместо hardcoded значений.
"""

import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import os

# skrl / Isaac Lab imports
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

# Custom PPO
from skrl.agents.torch.ppo import PPO, PPO_DEFAULT_CONFIG

# ---------------------------------------------------------------------
# Глобальные настройки сцены / графа
# ---------------------------------------------------------------------

NUM_GRAPH_NODES = 17
PER_OBJECT_DIM = 24
TEXT_EMB_DIM = 16
GRAPH_EMB_DIM = 128  # Пока фиксированный, но можно тоже тюнить

GOAL_NODE_INDEX = 0
DEBUG = False  # Выключаем для tuning, чтобы не засорять логи

# ---------------------------------------------------------------------
# [ВСЕ ВАШИ КЛАССЫ БЕЗ ИЗМЕНЕНИЙ]
# ---------------------------------------------------------------------

def build_star_chain_edge_index(
    num_nodes: int,
    batch_size: int,
    device: torch.device,
    add_self_loops: bool = True
) -> torch.Tensor:
    """Рёбра: звезда + цепочка"""
    N = num_nodes
    src, dst = [], []

    for i in range(1, N):
        src += [0, i]
        dst += [i, 0]

    for i in range(N - 1):
        src += [i, i + 1]
        dst += [i + 1, i]

    if add_self_loops:
        for i in range(N):
            src.append(i)
            dst.append(i)

    edge_index_single = torch.tensor([src, dst], device=device, dtype=torch.long)
    edge_indices = [edge_index_single + b * N for b in range(batch_size)]
    return torch.cat(edge_indices, dim=1)


class SceneGraphGATEncoder(nn.Module):
    """GATv2 encoder для сценового графа - ТЕПЕРЬ С ПАРАМЕТРАМИ ИЗ CONFIG"""
    def __init__(
        self,
        num_nodes: int,
        node_in_dim: int,
        hidden_dim: int = 128,  # <- Будет приходить из config
        out_dim: int = GRAPH_EMB_DIM,
        num_layers: int = 2,  # <- Будет приходить из config
        heads: int = 2,  # <- Будет приходить из config
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

        self._edge_cache = {}

    def _get_edge_index(self, B: int, device: torch.device) -> torch.Tensor:
        key = (B, device.index if device.type == "cuda" else -1)
        ei = self._edge_cache.get(key, None)
        if ei is None or ei.device != device:
            ei = build_star_chain_edge_index(self.num_nodes, B, device, add_self_loops=True)
            self._edge_cache[key] = ei
        return ei

    def forward(self, node_feats: torch.Tensor, batch_size: int) -> torch.Tensor:
        device = node_feats.device
        B = int(batch_size)
        N = self.num_nodes

        x = self.node_mlp(node_feats)
        edge_index = self._get_edge_index(B, device)
        batch = torch.repeat_interleave(torch.arange(B, device=device), repeats=N)

        for conv, norm in zip(self.convs, self.norms):
            x = conv(x, edge_index)
            x = norm(x)
            x = torch.relu(x)
            x = nn.functional.dropout(x, p=self.dropout, training=self.training)

        g = global_mean_pool(x, batch)
        return self.head(g)


class FrozenCLIPNameColorEncoder(nn.Module):
    """Lookup оффлайн CLIP-эмбеддингов"""
    def __init__(self, embeddings_path: str, text_dim: int = TEXT_EMB_DIM):
        super().__init__()
        self.text_dim = text_dim

        payload = torch.load(embeddings_path, map_location="cpu")
        name_embs = payload.get("name_embs", None)
        color_embs = payload.get("color_embs", None)
        if name_embs is None or color_embs is None:
            raise ValueError(f"Bad embeddings file")

        self.register_buffer("name_embs", name_embs.float(), persistent=False)
        self.register_buffer("color_embs", color_embs.float(), persistent=False)

        self.proj = nn.Sequential(
            nn.Linear(self.name_embs.shape[-1], 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, text_dim),
        )

    def forward(self, name_idx: torch.Tensor, color_bits_or_idx: torch.Tensor) -> torch.Tensor:
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

        emb_name = self.name_embs[name_idx]
        emb_color = self.color_embs[color_idx]
        emb = 0.5 * (emb_name + emb_color)
        return self.proj(emb)


class SharedGraphModule(nn.Module):
    """Общий графовый энкодер - ПРИНИМАЕТ ПАРАМЕТРЫ ИЗ CONFIG"""
    def __init__(
        self, 
        embeddings_path: str, 
        num_nodes: int = NUM_GRAPH_NODES,
        per_object_dim: int = PER_OBJECT_DIM, 
        text_dim: int = TEXT_EMB_DIM,
        # НОВЫЕ параметры из config:
        graph_hidden_dim: int = 128,
        graph_num_layers: int = 2,
        graph_heads: int = 2,
    ):
        super().__init__()
        self.num_nodes = num_nodes
        self.per_object_dim = per_object_dim
        self.text_dim = text_dim

        self.text_encoder = FrozenCLIPNameColorEncoder(
            embeddings_path=embeddings_path, 
            text_dim=text_dim
        )
        
        # Передаём параметры из config в graph encoder
        self.graph_encoder = SceneGraphGATEncoder(
            num_nodes=num_nodes,
            node_in_dim=per_object_dim + text_dim,
            hidden_dim=graph_hidden_dim,  # <- ИЗ CONFIG
            out_dim=GRAPH_EMB_DIM,
            num_layers=graph_num_layers,  # <- ИЗ CONFIG
            heads=graph_heads,  # <- ИЗ CONFIG
            dropout=0.1,
            goal_index=GOAL_NODE_INDEX,
        )

    def _build_node_features(self, graph_flat: torch.Tensor) -> torch.Tensor:
        B = graph_flat.shape[0]
        N = self.num_nodes
        node_raw = graph_flat.view(B, N, self.per_object_dim)

        name_idx = node_raw[..., 20]
        color_bits = node_raw[..., 21:24]

        text_emb = self.text_encoder(name_idx, color_bits)
        full_node = torch.cat([node_raw, text_emb], dim=-1)
        return full_node.view(B * N, -1)

    def forward(self, graph_flat: torch.Tensor) -> torch.Tensor:
        B = graph_flat.shape[0]
        node_feats = self._build_node_features(graph_flat)
        return self.graph_encoder(node_feats, batch_size=B)


class OrientationModule(nn.Module):
    """Предсказывает ориентацию робота - ПРИНИМАЕТ ПАРАМЕТРЫ ИЗ CONFIG"""
    def __init__(
        self, 
        img_dim: int, 
        graph_emb_dim: int, 
        num_bins: int = 36,  # <- ИЗ CONFIG
        emb_dim: int = 32,  # <- ИЗ CONFIG (ORIENTATION_EMB_DIM)
        device=None
    ):
        super().__init__()
        self.device = device
        self.num_bins = num_bins
        self.emb_dim = emb_dim
        
        # Predictor: img + graph → logits
        self.orientation_predictor = nn.Sequential(
            nn.Linear(img_dim + graph_emb_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, num_bins)  # <- num_bins из config
        ).to(device)
        
        # Embedding projector: probs → continuous embedding
        self.embedding_proj = nn.Sequential(
            nn.Linear(num_bins, 64),  # <- num_bins из config
            nn.ReLU(),
            nn.Linear(64, emb_dim)  # <- emb_dim из config
        ).to(device)
    
    def forward(self, img: torch.Tensor, graph_emb: torch.Tensor, 
                ground_truth_yaw: torch.Tensor = None):
        x = torch.cat([img, graph_emb], dim=-1)
        logits = self.orientation_predictor(x)
        
        probs = F.softmax(logits, dim=-1)
        orientation_emb = self.embedding_proj(probs)
        
        outputs = {'orientation_logits': logits}
        
        if ground_truth_yaw is not None:
            if ground_truth_yaw.dim() == 2:
                ground_truth_yaw = ground_truth_yaw.squeeze(-1)
            
            normalized = (ground_truth_yaw + torch.pi) % (2 * torch.pi)
            bin_size = (2 * torch.pi) / self.num_bins
            labels = (normalized / bin_size).long()
            labels = torch.clamp(labels, 0, self.num_bins - 1)
            
            loss = F.cross_entropy(logits, labels)
            pred_bins = torch.argmax(logits, dim=-1)
            accuracy = (pred_bins == labels).float().mean()
            
            outputs['orientation_loss'] = loss
            outputs['orientation_label'] = labels
            outputs['orientation_accuracy'] = accuracy
        
        return orientation_emb, outputs


class Policy(GaussianMixin, Model):
    """Policy (Actor) для PPO"""
    def __init__(
        self, 
        observation_space, 
        action_space, 
        device, 
        shared_graph: SharedGraphModule,
        orientation_module: OrientationModule,
        orientation_emb_dim: int = 32,  # <- ИЗ CONFIG
        clip_actions=False, 
        clip_log_std=True, 
        min_log_std=-20, 
        max_log_std=2
    ):
        Model.__init__(self, observation_space, action_space, device)
        GaussianMixin.__init__(self, clip_actions, clip_log_std, min_log_std, max_log_std)
        self.device = device

        self.shared_graph = shared_graph
        self.__dict__["orientation_module"] = orientation_module

        self.img_dim = int(observation_space["img"].shape[0])

        # mlp_in зависит от orientation_emb_dim из config
        mlp_in = self.img_dim + GRAPH_EMB_DIM + orientation_emb_dim
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
        states = unflatten_tensorized_space(self.observation_space, inputs["states"])
        img = states["img"].to(self.device)
        graph_flat = states["graph"].to(self.device)

        graph_emb = self.shared_graph(graph_flat)
        
        # Orientation НЕ обучается через policy
        with torch.no_grad():
            orientation_emb, _ = self.orientation_module(img, graph_emb, ground_truth_yaw=None)
        
        x = torch.cat([img, graph_emb, orientation_emb], dim=-1)
        mu = self.net(x)
        
        return mu, self.log_std_parameter, {}


class Value(DeterministicMixin, Model):
    """Value function для PPO: V(s) + orientation learning"""
    def __init__(
        self, 
        observation_space, 
        action_space, 
        device, 
        shared_graph: SharedGraphModule,
        orientation_module: OrientationModule,
        orientation_emb_dim: int = 32,  # <- ИЗ CONFIG
        clip_actions=False, 
        train_graph: bool = False, 
        train_orientation: bool = True
    ):
        Model.__init__(self, observation_space, action_space, device)
        DeterministicMixin.__init__(self, clip_actions)
        self.device = device

        self.train_graph = bool(train_graph)
        self.train_orientation = bool(train_orientation)

        if train_graph:
            self.shared_graph = shared_graph
        else:
            self.__dict__["shared_graph"] = shared_graph
        
        if train_orientation:
            self.orientation_module = orientation_module
        else:
            self.__dict__["orientation_module"] = orientation_module

        self.img_dim = int(observation_space["img"].shape[0])

        # mlp_in зависит от orientation_emb_dim из config
        mlp_in = self.img_dim + GRAPH_EMB_DIM + orientation_emb_dim
        self.net = nn.Sequential(
            nn.Linear(mlp_in, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        ).to(device)

    def compute(self, inputs, role):
        states = unflatten_tensorized_space(self.observation_space, inputs["states"])
        img = states["img"].to(self.device)
        graph_flat = states["graph"].to(self.device)
        
        ground_truth_yaw = states.get("orientation", None)
        if ground_truth_yaw is not None:
            ground_truth_yaw = ground_truth_yaw.to(self.device)
        
        # Graph НЕ обучается через value
        if self.train_graph:
            graph_emb = self.shared_graph(graph_flat)
        else:
            with torch.no_grad():
                graph_emb = self.shared_graph(graph_flat)
        
        # Orientation ОБУЧАЕТСЯ через value
        if self.train_orientation:
            orientation_emb, orient_outputs = self.orientation_module(
                img, graph_emb, ground_truth_yaw
            )
        else:
            with torch.no_grad():
                orientation_emb, orient_outputs = self.orientation_module(
                    img, graph_emb, ground_truth_yaw
                )
        
        x = torch.cat([img, graph_emb, orientation_emb], dim=-1)
        v = self.net(x)
        
        return v, orient_outputs


class DictRunningStandardScaler(nn.Module):
    """Нормализует только states["img"]"""
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
        s = unflatten_tensorized_space(self.full_space, x)
        s["img"] = self.img_scaler(s["img"], train=train, inverse=inverse, no_grad=no_grad)
        return flatten_tensorized_space(s)


# ---------------------------------------------------------------------
# ГЛАВНАЯ ФУНКЦИЯ ДЛЯ RAY TUNE
# ---------------------------------------------------------------------

def train_with_config(config: dict):
    """
    Основная функция тренировки, вызываемая Ray Tune для каждого trial.
    
    Args:
        config: dict с гиперпараметрами от Optuna/Ray Tune
    """
    # ============= SEED ДЛЯ ВОСПРОИЗВОДИМОСТИ =============
    set_seed(42)
    
    # ============= СОЗДАНИЕ ОКРУЖЕНИЯ =============
    env = load_isaaclab_env(
        task_name=config.get("task_name", "Isaac-Aloha-Direct-v0"),
        num_envs=config.get("num_envs", 32),
        headless=config.get("headless", True),
        cli_args=["--enable_cameras"] if config.get("enable_cameras", True) else [],
    )
    env = wrap_env(env)
    device = env.device
    
    # ============= MEMORY =============
    # Размер памяти берём из config (это один из тюнимых параметров)
    memory = RandomMemory(
        memory_size=config.get("rollouts", 48),
        num_envs=env.num_envs,
        device=device
    )
    
    # ============= SHARED MODULES С ПАРАМЕТРАМИ ИЗ CONFIG =============
    shared_graph = SharedGraphModule(
        embeddings_path=config.get(
            "embeddings_path",
            "/home/xiso/IsaacLab/source/isaaclab_tasks/isaaclab_tasks/direct/aloha/text_embeddings.pt"
        ),
        num_nodes=NUM_GRAPH_NODES,
        per_object_dim=PER_OBJECT_DIM,
        text_dim=TEXT_EMB_DIM,
        # ПАРАМЕТРЫ ИЗ CONFIG:
        graph_hidden_dim=config.get("graph_hidden_dim", 128),
        graph_num_layers=config.get("graph_num_layers", 2),
        graph_heads=config.get("graph_heads", 2),
    ).to(device)
    
    orientation_module = OrientationModule(
        img_dim=env.observation_space["img"].shape[0],
        graph_emb_dim=GRAPH_EMB_DIM,
        # ПАРАМЕТРЫ ИЗ CONFIG:
        num_bins=config.get("orientation_num_bins", 36),
        emb_dim=config.get("orientation_emb_dim", 32),
        device=device
    )
    
    # ============= MODELS =============
    # Передаём orientation_emb_dim, чтобы Policy и Value знали размерность
    orientation_emb_dim = config.get("orientation_emb_dim", 32)
    
    models = {
        "policy": Policy(
            env.observation_space, 
            env.action_space, 
            device,
            shared_graph=shared_graph,
            orientation_module=orientation_module,
            orientation_emb_dim=orientation_emb_dim,  # <- ИЗ CONFIG
        ),
        "value": Value(
            env.observation_space, 
            env.action_space, 
            device,
            shared_graph=shared_graph,
            orientation_module=orientation_module,
            orientation_emb_dim=orientation_emb_dim,  # <- ИЗ CONFIG
            train_graph=False,  # Graph обучается только через policy
            train_orientation=True  # Orientation обучается только через value
        ),
    }
    
    # ============= PPO CONFIG С ПАРАМЕТРАМИ ИЗ TUNING =============
    cfg = PPO_DEFAULT_CONFIG.copy()
    
    # ТЮНИМЫЕ параметры из config:
    cfg["rollouts"] = config.get("rollouts", 48)
    cfg["learning_epochs"] = config.get("learning_epochs", 5)
    cfg["mini_batches"] = config.get("mini_batches", 8)
    cfg["learning_rate"] = config.get("learning_rate", 3e-4)
    cfg["entropy_loss_scale"] = config.get("entropy_loss_scale", 0.05)
    cfg["value_loss_scale"] = config.get("value_loss_scale", 0.5)
    cfg["orientation_loss_weight"] = config.get("orientation_loss_weight", 0.01)  # <- КЛЮЧЕВОЙ!
    
    # ФИКСИРОВАННЫЕ параметры из config:
    cfg["discount_factor"] = config.get("discount_factor", 0.99)
    cfg["lambda"] = config.get("lambda_gae", 0.95)
    cfg["ratio_clip"] = config.get("ratio_clip", 0.2)
    cfg["value_clip"] = config.get("value_clip", 0.2)
    cfg["clip_predicted_values"] = config.get("clip_predicted_values", True)
    cfg["grad_norm_clip"] = config.get("grad_norm_clip", 0.5)
    cfg["learning_rate_scheduler"] = None
    
    # ============= PREPROCESSOR =============
    cfg["state_preprocessor"] = DictRunningStandardScaler
    cfg["state_preprocessor_kwargs"] = {
        "size": env.observation_space,
        "img_space": env.observation_space["img"],
        "device": device,
    }
    
    # ============= ЭКСПЕРИМЕНТ (ЛОГИ) =============
    # Ray Tune управляет директориями автоматически, но можно указать базовую
    cfg["experiment"]["write_interval"] = config.get("write_interval", 50)
    cfg["experiment"]["checkpoint_interval"] = config.get("checkpoint_interval", 500)
    cfg["experiment"]["directory"] = "logs/skrl/aloha_tune"  # Ray создаст поддиректории
    
    # ============= PPO AGENT =============
    agent = PPO(
        models=models,
        memory=memory,
        cfg=cfg,
        observation_space=env.observation_space,
        action_space=env.action_space,
        device=device,
    )
    
    # ============= TRAINER =============
    cfg_trainer = {
        "timesteps": config.get("timesteps", 1000),  # Из config
        "headless": config.get("headless", True)
    }
    trainer = SequentialTrainer(cfg=cfg_trainer, env=env, agents=agent)
    
    # ============= ЗАПУСК ТРЕНИРОВКИ =============
    print(f"\n{'='*60}")
    print(f"STARTING TRIAL WITH CONFIG:")
    print(f"  learning_rate: {config.get('learning_rate')}")
    print(f"  orientation_loss_weight: {config.get('orientation_loss_weight')}")
    print(f"  orientation_num_bins: {config.get('orientation_num_bins')}")
    print(f"  graph_hidden_dim: {config.get('graph_hidden_dim')}")
    print(f"{'='*60}\n")
    
    trainer.train()


# ---------------------------------------------------------------------
# MAIN (для standalone запуска с аргументами)
# ---------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Aloha Orientation PPO Training (Tunable)")
    
    # ============= ГИПЕРПАРАМЕТРЫ (для standalone) =============
    parser.add_argument("--learning_rate", type=float, default=3e-4)
    parser.add_argument("--rollouts", type=int, default=48)
    parser.add_argument("--mini_batches", type=int, default=8)
    parser.add_argument("--learning_epochs", type=int, default=5)
    parser.add_argument("--orientation_loss_weight", type=float, default=0.01)
    parser.add_argument("--orientation_num_bins", type=int, default=36)
    parser.add_argument("--orientation_emb_dim", type=int, default=32)
    parser.add_argument("--graph_hidden_dim", type=int, default=128)
    parser.add_argument("--graph_num_layers", type=int, default=2)
    parser.add_argument("--graph_heads", type=int, default=2)
    parser.add_argument("--timesteps", type=int, default=1000)
    
    args = parser.parse_args()
    
    # Конвертируем аргументы в config dict
    config = {
        "learning_rate": args.learning_rate,
        "rollouts": args.rollouts,
        "mini_batches": args.mini_batches,
        "learning_epochs": args.learning_epochs,
        "orientation_loss_weight": args.orientation_loss_weight,
        "orientation_num_bins": args.orientation_num_bins,
        "orientation_emb_dim": args.orientation_emb_dim,
        "graph_hidden_dim": args.graph_hidden_dim,
        "graph_num_layers": args.graph_num_layers,
        "graph_heads": args.graph_heads,
        "timesteps": args.timesteps,
        # Фиксированные:
        "task_name": "Isaac-Aloha-Direct-v0",
        "num_envs": 32,
        "headless": True,
        "enable_cameras": True,
        "embeddings_path": "/home/xiso/IsaacLab/source/isaaclab_tasks/isaaclab_tasks/direct/aloha/text_embeddings.pt",
    }
    
    # Запуск тренировки
    train_with_config(config)