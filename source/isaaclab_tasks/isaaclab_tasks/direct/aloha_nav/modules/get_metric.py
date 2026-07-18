import os
import torch
import pandas as pd
import numpy as np

def compute_seed_means(log_tensor):
    """
    log_tensor: [num_episodes, 6] (success, collision, timeout, episode_time, episode_length, optimal_length)
    Возвращает словарь mean-метрик по этому сиду, включая SPL.
    """
    arr = log_tensor.numpy()
    success = arr[:, 0]
    episode_length = arr[:, 4]
    optimal_length = arr[:, 5]

    # Вычисляем SPL
    spl = np.where(optimal_length > 0,
                   success * (optimal_length / np.maximum(optimal_length, episode_length)),
                   0.0)

    return {
        "success_mean":          success.mean(),
        "collision_mean":        arr[:, 1].mean(),
        "timeout_mean":          arr[:, 2].mean(),
        "episode_time_mean":     arr[:, 3].mean(),
        "episode_length_mean":   episode_length.mean(),
        "optimal_length_mean":   optimal_length.mean(),
        "spl_mean":              spl.mean(),
    }

def aggregate_logs_across_seeds(log_dir: str):
    """
    Считает mean/std/CI95_delta по сидов.
    
    Returns:
        df_seeds: все сиды и их средние метрики
        stats: mean/std/CI95_delta across seeds
    """
    seed_stats = []

    # --- Читаем все .pt файлы ---
    for fname in os.listdir(log_dir):
        if fname.endswith(".pt"):
            path = os.path.join(log_dir, fname)
            data = torch.load(path)  # [episodes, 6]
            seed_stats.append(compute_seed_means(data))

    if not seed_stats:
        raise ValueError("В папке нет .pt логов!")

    # --- Таблица средних по каждому сидy ---
    df_seeds = pd.DataFrame(seed_stats)

    # --- Mean/std/count across seeds ---
    mean = df_seeds.mean()
    std  = df_seeds.std()
    count = df_seeds.count()

    # --- CI delta (95%) ---
    se = std / np.sqrt(count)
    ci95_delta = 1.96 * se

    # Итоги
    stats = pd.DataFrame({
        "mean": mean,
        "std": std,
        "CI95_delta": ci95_delta,
        "num_seeds": count
    })

    return df_seeds, stats


if __name__ == "__main__":
    log_dir = "/home/xiso/IsaacLab/logs/skrl/logs/base"  # поменяй путь
    df_seeds, stats = aggregate_logs_across_seeds(log_dir)

    print("=== Per-seed metrics ===")
    print(df_seeds)

    print("\n=== Final metrics (mean of means with 95% CI delta) ===")
    print(stats)