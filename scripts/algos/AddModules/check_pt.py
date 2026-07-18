# train_orientation.py
import torch
import numpy as np
import gymnasium as gym
from torch.utils.data import DataLoader, TensorDataset
from skrl.memories.torch import RandomMemory
from skrl.utils.spaces.torch import unflatten_tensorized_space
from networks.networks_orm import GraphEncoder, OrientationModule, DictRunningStandardScaler

DEVICE = torch.device("cuda")
MEMORY_DIR      = "logs/skrl/memory/memories/26-02-24_20-46-34-792695_memory_0x734a9c708ca0.pt"
AGENT_CKPT_PATH = "logs/skrl/aloha_sac/26-02-23_16-49-28-377932_SAC/checkpoints/agent_80000.pt"
EMBEDDINGS_PATH = "source/isaaclab_tasks/isaaclab_tasks/direct/aloha/text_embeddings.pt"
GRAPH_ENCODER_INIT = None  # или путь к .pt для старта с предобученных весов
ORIENT_MODULE_INIT = None
BATCH_SIZE = 1024
EPOCHS     = 50000
LR_GRAPH   = 3e-4
LR_ORIENT  = 3e-3

NUM_TOTAL_OBJECTS = 21  # NUM_GRAPH_NODES из networks.py

# ── 1. Observation space вручную ─────────────────────────────────────
observation_space = gym.spaces.Dict({
    "img":         gym.spaces.Box(low=-float("inf"), high=float("inf"), shape=(512,),                    dtype=np.float32),
    "memory":      gym.spaces.Box(low=-float("inf"), high=float("inf"), shape=(2056,),                   dtype=np.float32),
    "goal":        gym.spaces.Box(low=-float("inf"), high=float("inf"), shape=(2,),                      dtype=np.float32),
    "orientation": gym.spaces.Box(low=-float("inf"), high=float("inf"), shape=(1,),                      dtype=np.float32),
    "graph":       gym.spaces.Box(low=-float("inf"), high=float("inf"), shape=(24 * NUM_TOTAL_OBJECTS,), dtype=np.float32),
})

# ── 2. Загружаем буфер ───────────────────────────────────────────────
MEMORY_DIRS = [
    "logs/skrl/memory/memories/26-02-25_12-12-28-281761_memory_0x7d4aa536f1f0.pt",
    ]

all_states_list = []
for path in MEMORY_DIRS:
    mem = torch.load(path, map_location=DEVICE)
    all_states_list.append(mem["states"])
    print(f"Loaded {path}: {mem['states'].shape}")

# Конкатенируем по оси времени [memory_size, num_envs, flat_dim]
all_states = torch.cat(all_states_list, dim=0)
print(f"Combined shape: {all_states.shape}")

# Разворачиваем как обычно
all_states = all_states.reshape(-1, all_states.shape[-1])
print(f"Flat shape: {all_states.shape}")

# ── 3. Препроцессор из агентского чекпоинта ──────────────────────────
checkpoint = torch.load(AGENT_CKPT_PATH, map_location=DEVICE)
preprocessor = DictRunningStandardScaler(
    size=observation_space, img_space=observation_space["img"], device=DEVICE
)
preprocessor.load_state_dict(checkpoint["state_preprocessor"])
preprocessor.eval()
print(f"Preprocessor loaded (count={preprocessor.img_scaler.current_count.item():.0f})")

# ── 4. Вытаскиваем данные ────────────────────────────────────────────
# Разворачиваем в [N, flat_dim]
# ── 4. Train/Val split ───────────────────────────────────────────────
all_states = all_states.reshape(-1, all_states.shape[-1])

# Перемешиваем перед сплитом
perm = torch.randperm(all_states.shape[0])
all_states = all_states[perm]

with torch.no_grad():
    processed = preprocessor(all_states, train=False)
    s = unflatten_tensorized_space(observation_space, processed)

img_data    = s["img"].cpu()
graph_data  = s["graph"].cpu()
orient_data = s["orientation"].cpu()

# 80/20 split
n_total = img_data.shape[0]
n_train = int(n_total * 0.8)

train_dataset = TensorDataset(img_data[:n_train], graph_data[:n_train], orient_data[:n_train])
val_dataset   = TensorDataset(img_data[n_train:], graph_data[n_train:], orient_data[n_train:])

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,  drop_last=True)
val_loader   = DataLoader(val_dataset,   batch_size=BATCH_SIZE, shuffle=False, drop_last=False)

print(f"Train: {n_train} | Val: {n_total - n_train} | Batches/epoch: {len(train_loader)}")

# ── 5. Модули ────────────────────────────────────────────────────────
graph_encoder = GraphEncoder(embeddings_path=EMBEDDINGS_PATH).to(DEVICE)
orient_module = OrientationModule(img_dim=observation_space["img"].shape[0]).to(DEVICE)

if GRAPH_ENCODER_INIT:
    graph_encoder.load_state_dict(torch.load(GRAPH_ENCODER_INIT, map_location=DEVICE))
    print("Loaded graph_encoder from checkpoint")
if ORIENT_MODULE_INIT:
    orient_module.load_state_dict(torch.load(ORIENT_MODULE_INIT, map_location=DEVICE))
    print("Loaded orient_module from checkpoint")

# ── 6. Оптимизаторы ──────────────────────────────────────────────────
graph_opt  = torch.optim.AdamW(graph_encoder.parameters(), lr=LR_GRAPH,  weight_decay=1e-4)
orient_opt = torch.optim.AdamW(orient_module.parameters(), lr=LR_ORIENT, weight_decay=1e-4)
# Меняем scheduler
scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(orient_opt, T_0=100)

best_val_acc = 0.0
patience = 200  # early stopping
no_improve = 0

for epoch in range(EPOCHS):
    # ── Train ──
    graph_encoder.train()
    orient_module.train()
    total_loss, total_acc_relaxed = 0.0, 0.0

    for img, graph_flat, gt_yaw in train_loader:
        img, graph_flat, gt_yaw = img.to(DEVICE), graph_flat.to(DEVICE), gt_yaw.to(DEVICE)
        graph_emb = graph_encoder(graph_flat)
        _, probs, logits = orient_module(img, graph_emb)
        loss, metrics = orient_module.compute_loss(logits, probs, gt_yaw)

        graph_opt.zero_grad()
        orient_opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(graph_encoder.parameters(), 1.0)
        torch.nn.utils.clip_grad_norm_(orient_module.parameters(), 1.0)
        graph_opt.step()
        orient_opt.step()

        total_loss        += metrics["orient/loss"]
        total_acc_relaxed += metrics["orient/acc_relaxed"]

    scheduler.step()

    # ── Val ──
    graph_encoder.eval()
    orient_module.eval()
    val_loss, val_acc = 0.0, 0.0

    with torch.no_grad():
        for img, graph_flat, gt_yaw in val_loader:
            img, graph_flat, gt_yaw = img.to(DEVICE), graph_flat.to(DEVICE), gt_yaw.to(DEVICE)
            graph_emb = graph_encoder(graph_flat)
            _, probs, logits = orient_module(img, graph_emb)
            _, metrics = orient_module.compute_loss(logits, probs, gt_yaw)
            val_loss += metrics["orient/loss"]
            val_acc  += metrics["orient/acc_relaxed"]

    n_train_b = len(train_loader)
    n_val_b   = len(val_loader)
    val_acc_mean = val_acc / n_val_b

    print(
        f"Epoch {epoch+1:4d} | "
        f"train_loss: {total_loss/n_train_b:.4f} | train_acc: {total_acc_relaxed/n_train_b:.4f} | "
        f"val_loss: {val_loss/n_val_b:.4f} | val_acc: {val_acc_mean:.4f}"
    )

    # Early stopping по val
    if val_acc_mean > best_val_acc:
        best_val_acc = val_acc_mean
        no_improve = 0
        torch.save(graph_encoder.state_dict(), "logs/skrl/aloha_sac/added/graph_encoder_pretrained_best.pt")
        torch.save(orient_module.state_dict(),  "logs/skrl/aloha_sac/added/orient_module_pretrained_best.pt")
        print(f"  ✓ Best val saved (val_acc={best_val_acc:.4f})")
    else:
        no_improve += 1
        if no_improve >= patience:
            print(f"Early stopping at epoch {epoch+1}")
            break