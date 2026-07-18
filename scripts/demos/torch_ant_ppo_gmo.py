import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F

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
GRAPH_EMB_DIM = 128
ORIENTATION_EMB_DIM = 32

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
    """Рёбра: звезда + цепочка"""
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

    edge_index_single = torch.tensor([src, dst], device=device, dtype=torch.long)
    edge_indices = [edge_index_single + b * N for b in range(batch_size)]
    return torch.cat(edge_indices, dim=1)


# ---------------------------------------------------------------------
# Encoders
# ---------------------------------------------------------------------

class SceneGraphGATEncoder(nn.Module):
    """GATv2 encoder для сценового графа"""
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

        self._edge_cache = {}

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
    """Общий графовый энкодер"""
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


# ---------------------------------------------------------------------
# Orientation Module
# ---------------------------------------------------------------------

class OrientationModule(nn.Module):
    """Предсказывает ориентацию робота и выдаёт embedding"""
    def __init__(self, img_dim: int, graph_emb_dim: int, num_bins: int = 36, 
                 emb_dim: int = ORIENTATION_EMB_DIM, device=None):
        super().__init__()
        self.device = device
        self.num_bins = num_bins
        self.emb_dim = emb_dim
        
        if DEBUG:
            print(f"\n[OrientationModule] Initializing:")
            print(f"  img_dim={img_dim}, graph_emb_dim={graph_emb_dim}")
            print(f"  num_bins={num_bins}, emb_dim={emb_dim}")
        
        # Predictor: img + graph → logits
        self.orientation_predictor = nn.Sequential(
            nn.Linear(img_dim + graph_emb_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, num_bins)
        ).to(device)
        
        # Embedding projector: probs → continuous embedding
        self.embedding_proj = nn.Sequential(
            nn.Linear(num_bins, 64),
            nn.ReLU(),
            nn.Linear(64, emb_dim)
        ).to(device)
        
        if DEBUG:
            total_params = sum(p.numel() for p in self.parameters())
            print(f"  Total params: {total_params:,}")
    
    def forward(self, img: torch.Tensor, graph_emb: torch.Tensor, 
                ground_truth_yaw: torch.Tensor = None):
        """
        img: [B, img_dim]
        graph_emb: [B, graph_emb_dim]
        ground_truth_yaw: [B, 1] или [B] (опционально)
        
        Returns:
            orientation_emb: [B, emb_dim]
            outputs: dict с logits, loss, accuracy
        """
        x = torch.cat([img, graph_emb], dim=-1)
        logits = self.orientation_predictor(x)
        
        # Soft embedding через softmax (differentiable)
        probs = F.softmax(logits, dim=-1)
        orientation_emb = self.embedding_proj(probs)
        
        outputs = {'orientation_logits': logits}
        
        # Если есть ground truth - вычисляем loss и accuracy
        # print("ground_truth_yaw ORM", ground_truth_yaw)
        if ground_truth_yaw is not None:
            if ground_truth_yaw.dim() == 2:
                ground_truth_yaw = ground_truth_yaw.squeeze(-1)
            
            # Конвертируем yaw → bin label
            normalized = (ground_truth_yaw + torch.pi) % (2 * torch.pi)
            bin_size = (2 * torch.pi) / self.num_bins
            labels = (normalized / bin_size).long()
            labels = torch.clamp(labels, 0, self.num_bins - 1)
            
            # Loss
            loss = F.cross_entropy(logits, labels)
            
            # Accuracy
            pred_bins = torch.argmax(logits, dim=-1)
            accuracy = (pred_bins == labels).float().mean()
            
            outputs['orientation_loss'] = loss
            outputs['orientation_label'] = labels
            outputs['orientation_accuracy'] = accuracy
            
            if DEBUG and not hasattr(self, '_debug_forward_printed'):
                print(f"\n[OrientationModule.forward] First call:")
                print(f"  ground_truth_yaw range: [{ground_truth_yaw.min():.3f}, {ground_truth_yaw.max():.3f}]")
                print(f"  labels range: [{labels.min()}, {labels.max()}]")
                print(f"  loss: {loss.item():.4f}, accuracy: {accuracy.item():.4f}")
                self._debug_forward_printed = True
        
        return orientation_emb, outputs


# ---------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------
def print_orientation_info(orient_outputs, batch_idx=0):
    """Быстрый вывод информации об ориентации"""
    logits = orient_outputs['orientation_logits']
    probs = F.softmax(logits, dim=-1)
    
    num_bins = logits.shape[-1]
    bin_size = 2 * torch.pi / num_bins
    
    # Топ-3 вероятности и их бины
    top_probs, top_bins = torch.topk(probs[batch_idx], 3)
    
    # Конвертируем бины в углы
    top_angles = -torch.pi + (top_bins + 0.5) * bin_size
    
    # Неопределённость
    entropy = -(probs[batch_idx] * torch.log(probs[batch_idx] + 1e-10)).sum()
    max_entropy = torch.log(torch.tensor(num_bins))
    
    print(f"Top-3 predicted angles:")
    for i in range(3):
        angle_deg = torch.rad2deg(top_angles[i]).item()
        prob = top_probs[i].item()
        bin_idx = top_bins[i].item()
        print(f"  {i+1}. {angle_deg:6.1f}° (bin {bin_idx:2d}): prob={prob:.3f}")
    
    print(f"\nEntropy: {entropy:.3f}/{max_entropy:.3f}")
    
    return top_angles, top_probs
    

class Policy(GaussianMixin, Model):
    """Policy (Actor) для PPO"""
    def __init__(self, observation_space, action_space, device, shared_graph: SharedGraphModule,
                 orientation_module: OrientationModule,
                 clip_actions=False, clip_log_std=True, min_log_std=-20, max_log_std=2):
        Model.__init__(self, observation_space, action_space, device)
        GaussianMixin.__init__(self, clip_actions, clip_log_std, min_log_std, max_log_std)
        self.device = device

        # В PPO графовый энкодер ОБУЧАЕТСЯ через policy
        self.shared_graph = shared_graph
        
        # Orientation НЕ обучается через policy (только через value)
        self.__dict__["orientation_module"] = orientation_module

        self.img_dim = int(observation_space["img"].shape[0])

        mlp_in = self.img_dim + GRAPH_EMB_DIM + ORIENTATION_EMB_DIM
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
            print(f"\n[Policy] Initialized:")
            print(f"  mlp_in={mlp_in} (img + graph + orient)")

    def compute(self, inputs, role):
        B = inputs["states"].shape[0]
        states = unflatten_tensorized_space(self.observation_space, inputs["states"])
        img = states["img"].to(self.device)
        graph_flat = states["graph"].to(self.device)

        # Graph обучается
        graph_emb = self.shared_graph(graph_flat)
        
        # Orientation НЕ обучается через policy
        # print("policy")
        with torch.no_grad():
            orientation_emb, orient_outputs = self.orientation_module(img, graph_emb, ground_truth_yaw=None)
        # print("or outputs: ", orient_outputs)
        # angle, prob = print_orientation_info(orient_outputs)
        x = torch.cat([img, graph_emb, orientation_emb], dim=-1)
        mu = self.net(x)
        
        return mu, self.log_std_parameter, {}


class Value(DeterministicMixin, Model):
    """Value function для PPO: V(s) + orientation learning"""
    def __init__(self, observation_space, action_space, device, shared_graph: SharedGraphModule,
                 orientation_module: OrientationModule,
                 clip_actions=False, train_graph: bool = False, train_orientation: bool = True):
        Model.__init__(self, observation_space, action_space, device)
        DeterministicMixin.__init__(self, clip_actions)
        self.device = device

        self.train_graph = bool(train_graph)
        self.train_orientation = bool(train_orientation)

        # Graph НЕ обучается через value (только через policy)
        if train_graph:
            self.shared_graph = shared_graph
        else:
            self.__dict__["shared_graph"] = shared_graph
        
        # Orientation ОБУЧАЕТСЯ через value!
        if train_orientation:
            self.orientation_module = orientation_module
        else:
            self.__dict__["orientation_module"] = orientation_module

        self.img_dim = int(observation_space["img"].shape[0])

        # V-network: img + graph_emb + orientation_emb (БЕЗ action!)
        mlp_in = self.img_dim + GRAPH_EMB_DIM + ORIENTATION_EMB_DIM
        self.net = nn.Sequential(
            nn.Linear(mlp_in, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        ).to(device)
        
        if DEBUG:
            print(f"\n[Value] Initialized:")
            print(f"  train_graph={train_graph}, train_orientation={train_orientation}")
            print(f"  mlp_in={mlp_in}")

    def compute(self, inputs, role):
        B = inputs["states"].shape[0]
        states = unflatten_tensorized_space(self.observation_space, inputs["states"])
        img = states["img"].to(self.device)
        graph_flat = states["graph"].to(self.device)
        
        # Ground truth orientation
        ground_truth_yaw = states.get("orientation", None)
        # print("ground_truth_yaw VALUE", ground_truth_yaw)
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
        
        # V-value
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
# Main
# ---------------------------------------------------------------------
if __name__ == "__main__":
    EVAL = False
    # EVAL = True
    
    set_seed(42)

    # Environment
    if EVAL:
        from gymnasium.wrappers import RecordVideo
        print("[INFO] Running evaluation...")
        env = load_isaaclab_env(
            task_name="Isaac-Aloha-Direct-v0",
            num_envs=1,
            # headless=True,
            cli_args=["--enable_cameras", "--video"],
        )
        env = RecordVideo(
            env,
            video_folder="logs/skrl/videos",
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

    if DEBUG:
        print(f"\n{'='*60}")
        print("ENVIRONMENT INFO")
        print(f"{'='*60}")
        print(f"Device: {device}")
        print(f"Num envs: {env.num_envs}")
        print(f"Observation space: {env.observation_space}")
        
        # Проверяем что orientation есть
        if hasattr(env.observation_space, 'spaces') and isinstance(env.observation_space.spaces, dict):
            if "orientation" in env.observation_space.spaces:
                print(f"✓ 'orientation' found: {env.observation_space.spaces['orientation']}")
            else:
                print(f"⚠️  WARNING: 'orientation' NOT in observation space!")
        print(f"{'='*60}\n")

    # Memory
    memory = RandomMemory(memory_size=48, num_envs=env.num_envs, device=device)

    # Shared modules
    shared_graph = SharedGraphModule(
        embeddings_path="/home/xiso/IsaacLab/source/isaaclab_tasks/isaaclab_tasks/direct/aloha/text_embeddings.pt",
        num_nodes=NUM_GRAPH_NODES,
        per_object_dim=PER_OBJECT_DIM,
        text_dim=TEXT_EMB_DIM,
    ).to(device)

    orientation_module = OrientationModule(
        img_dim=env.observation_space["img"].shape[0],
        graph_emb_dim=GRAPH_EMB_DIM,
        num_bins=36,
        emb_dim=ORIENTATION_EMB_DIM,
        device=device
    )

    # Models
    models = {
        "policy": Policy(
            env.observation_space, env.action_space, device,
            shared_graph=shared_graph,
            orientation_module=orientation_module
        ),
        "value": Value(
            env.observation_space, env.action_space, device,
            shared_graph=shared_graph,
            orientation_module=orientation_module,
            train_graph=False,        # Graph обучается через policy!
            train_orientation=True    # Orientation обучается через value!
        ),
    }

    if DEBUG:
        print(f"\n{'='*60}")
        print("PARAMETER CHECK")
        print(f"{'='*60}")
        
        graph_params = list(shared_graph.parameters())
        orient_params = list(orientation_module.parameters())
        policy_params = list(models["policy"].parameters())
        value_params = list(models["value"].parameters())
        
        print(f"\n1. Shared graph: {sum(p.numel() for p in graph_params):,}")
        print(f"2. Orientation: {sum(p.numel() for p in orient_params):,}")
        print(f"3. Policy total: {sum(p.numel() for p in policy_params):,}")
        print(f"4. Value total: {sum(p.numel() for p in value_params):,}")
        
        # Проверяем регистрацию
        graph_param_ids = {id(p) for p in graph_params}
        orient_param_ids = {id(p) for p in orient_params}
        policy_param_ids = {id(p) for p in policy_params}
        value_param_ids = {id(p) for p in value_params}
        
        graph_in_policy = bool(graph_param_ids & policy_param_ids)
        orient_in_value = bool(orient_param_ids & value_param_ids)
        
        print(f"\n✓ Graph in Policy: {graph_in_policy}")
        print(f"✓ Orientation in Value: {orient_in_value}")
        print(f"{'='*60}\n")

    # PPO Config
    cfg = PPO_DEFAULT_CONFIG.copy()
    cfg["rollouts"] = 64 #48 1536
    cfg["learning_epochs"] = 5
    cfg["mini_batches"] = 4

    cfg["discount_factor"] = 0.99
    cfg["lambda"] = 0.95

    cfg["learning_rate"] = 5.0e-04
    from skrl.resources.schedulers.torch import KLAdaptiveLR
    cfg["learning_rate_scheduler"] = KLAdaptiveLR 
    cfg["learning_rate_scheduler_kwargs"] = {"kl_threshold": 0.016}

    cfg["ratio_clip"] = 0.2
    cfg["value_clip"] = 0.2
    cfg["clip_predicted_values"] = True

    cfg["entropy_loss_scale"] = 0.0
    cfg["value_loss_scale"] = 2.0
    cfg["grad_norm_clip"] = 1.0

    # ВАЖНО: Вес для orientation loss
    cfg["orientation_loss_weight"] = 0.05

    cfg["state_preprocessor"] = DictRunningStandardScaler
    cfg["state_preprocessor_kwargs"] = {
        "size": env.observation_space,
        "img_space": env.observation_space["img"],
        "device": device,
    }

    cfg["experiment"]["write_interval"] = 100
    cfg["experiment"]["checkpoint_interval"] = 1000
    cfg["experiment"]["directory"] = "logs/skrl/aloha_ppo_orientation"

    # Agent (кастомный PPO с auxiliary loss)
    agent = PPO(
        models=models,
        memory=memory,
        cfg=cfg,
        observation_space=env.observation_space,
        action_space=env.action_space,
        device=device,
    )

    # Trainer
    if not EVAL:
        cfg_trainer = {"timesteps": 330000, "headless": True}
        trainer = SequentialTrainer(cfg=cfg_trainer, env=env, agents=agent)
        
        if DEBUG:
            print(f"\n{'='*60}")
            print("STARTING TRAINING")
            print(f"{'='*60}\n")
        
        trainer.train()
    else:
        cfg_trainer = {"timesteps": 1000, "headless": True}
        trainer = SequentialTrainer(cfg=cfg_trainer, env=env, agents=agent)
        
        checkpoint_path = "/home/xiso/IsaacLab/logs/skrl/aloha_ppo_orientation/26-01-19_17-33-54-140845_PPO/checkpoints/agent_7000.pt"
        agent.load(checkpoint_path)
        
        trainer.eval()