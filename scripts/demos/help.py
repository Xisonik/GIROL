# ═══════════════════════════════════════════════════════════
# ОТДЕЛЬНАЯ СЕТЬ ДЛЯ ОРИЕНТАЦИИ (никак не связана с actor/critic)
# ═══════════════════════════════════════════════════════════

class StandaloneOrientationNet(nn.Module):
    """Полностью отдельная сеть: img+graph -> orientation bins."""
    def __init__(self, img_dim, graph_dim, num_bins=36):
        super().__init__()
        self.num_bins = num_bins
        self.net = nn.Sequential(
            nn.Linear(img_dim + graph_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, num_bins),
        )

    def forward(self, img, graph_flat):
        return self.net(torch.cat([img, graph_flat], dim=-1))


# Создаём
standalone_orient = StandaloneOrientationNet(
    img_dim=env.observation_space["img"].shape[0],
    graph_dim=env.observation_space["graph"].shape[0],  # сырой граф, без GATv2
    num_bins=36,
).to(device)

standalone_optimizer = torch.optim.Adam(standalone_orient.parameters(), lr=1e-3)
standalone_orient.train()

# Счётчики для логирования
orient_train_stats = {
    'step': 0,
    'loss_sum': 0.0,
    'acc_sum': 0.0,
    'acc_strict_sum': 0.0,
    'count': 0,
}

NUM_BINS = 36

def train_standalone_orientation(memory, batch_size=256, train_steps=1):
    """Один шаг обучения на данных из replay buffer."""
    if not memory.filled and memory.memory_index < batch_size:
        return  # мало данных

    for _ in range(train_steps):
        # Сэмплируем из буфера (так же как SAC)
        sample = memory.sample(names=["states"], batch_size=batch_size)[0]
        raw_states = sample[0]  # (B, flat_dim)

        # Preprocessor (train=False — не обновляем статистики)
        with torch.no_grad():
            processed = agent._state_preprocessor(raw_states, train=False)

        s = unflatten_tensorized_space(env.observation_space, processed)
        img = s["img"]
        graph_flat = s["graph"]
        gt_orient = s["orientation"].squeeze(-1)  # (B,)

        # Forward
        logits = standalone_orient(img, graph_flat)

        # Loss: cross-entropy
        bin_size = (2 * torch.pi) / NUM_BINS
        gt_norm = torch.atan2(torch.sin(gt_orient), torch.cos(gt_orient))
        labels = ((gt_norm + torch.pi) / bin_size).long().clamp(0, NUM_BINS - 1)
        loss = F.cross_entropy(logits, labels, label_smoothing=0.05)

        # Backward
        standalone_optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(standalone_orient.parameters(), 1.0)
        standalone_optimizer.step()

        # Metrics
        with torch.no_grad():
            pred = logits.argmax(-1)
            bd = torch.abs(pred - labels)
            bd = torch.minimum(bd, NUM_BINS - bd)
            acc_relaxed = (bd <= 1).float().mean().item()
            acc_strict = (pred == labels).float().mean().item()

        orient_train_stats['step'] += 1
        orient_train_stats['loss_sum'] += loss.item()
        orient_train_stats['acc_sum'] += acc_relaxed
        orient_train_stats['acc_strict_sum'] += acc_strict
        orient_train_stats['count'] += 1



        # В diagnostic_post_interaction (или создай новый wrapper):
original_post_final = agent.post_interaction

def full_post_interaction(timestep, timesteps):
    original_post_final(timestep, timesteps)

    # Обучаем standalone сеть каждый шаг (после learning_starts)
    if timestep > 100:
        train_standalone_orientation(agent.memory, batch_size=256, train_steps=1)

    # Логируем каждые 500 шагов
    if timestep % 500 == 0 and orient_train_stats['count'] > 0:
        n = orient_train_stats['count']
        avg_loss = orient_train_stats['loss_sum'] / n
        avg_acc = orient_train_stats['acc_sum'] / n
        avg_strict = orient_train_stats['acc_strict_sum'] / n

        print(f"\n🧭 [{timestep}] Standalone Orientation:")
        print(f"   loss={avg_loss:.4f}  acc±10°={avg_acc:.4f}  strict={avg_strict:.4f}  ({n} steps)")

        # Сброс
        orient_train_stats['loss_sum'] = 0.0
        orient_train_stats['acc_sum'] = 0.0
        orient_train_stats['acc_strict_sum'] = 0.0
        orient_train_stats['count'] = 0

        # Сохраняем чекпоинт
        if timestep % 5000 == 0:
            torch.save({
                'model': standalone_orient.state_dict(),
                'optimizer': standalone_optimizer.state_dict(),
                'timestep': timestep,
            }, f"standalone_orient_{timestep}.pt")

agent.post_interaction = full_post_interaction