import os
import torch
import pandas as pd
import numpy as np

def aggregate_eval_logs(log_dir: str):
    """
    Считает mean-of-means и 95% CI по сидам.
    
    Для каждого seed-файла считается среднее по его эпизодам.
    Затем по этим per-seed средним считается итоговое среднее, std и CI95.
    
    Файлы:
      eval_log_{seed}.pt    — [num_episodes, 6], эпизодные метрики
      orient_acc_{seed}.pt  — [3],               acc_10, acc_20, acc_30
    """
    episode_columns = [
        "success", "collision", "timeout",
        "episode_time_s", "episode_length_m", "optimal_length_m"
    ]
    orient_columns = ["acc_10_deg", "acc_20_deg", "acc_30_deg"]

    # --- Собираем per-seed средние ---
    per_seed_episode = []   # list of Series (mean по эпизодам одного seed)
    per_seed_orient  = []   # list of [3] array (одно значение на seed)

    for fname in sorted(os.listdir(log_dir)):
        if fname.startswith("eval_log_") and fname.endswith(".pt"):
            seed = fname.replace("eval_log_", "").replace(".pt", "")
            ep_path = os.path.join(log_dir, fname)
            or_path = os.path.join(log_dir, f"orient_acc_{seed}.pt")

            # Эпизодные метрики — среднее по эпизодам этого seed
            ep_data = torch.load(ep_path)   # [N, 6]
            ep_df = pd.DataFrame(ep_data.numpy(), columns=episode_columns)

            # SPL = success * optimal / max(optimal, traveled)
            denom = ep_df[["episode_length_m", "optimal_length_m"]].max(axis=1).clip(lower=1e-3)
            ep_df["spl"] = ep_df["success"] * ep_df["optimal_length_m"] / denom

            per_seed_episode.append(ep_df.mean())

            # Orientation accuracy
            if os.path.exists(or_path):
                or_data = torch.load(or_path)   # [3]
                per_seed_orient.append(or_data.numpy())
            else:
                print(f"[WARN] orient_acc_{seed}.pt not found, skipping orient for this seed")

    if not per_seed_episode:
        raise ValueError("Нет eval_log_*.pt файлов в папке!")

    n_seeds = len(per_seed_episode)
    print(f"Found {n_seeds} seeds")

    # --- Агрегация эпизодных метрик ---
    ep_seeds_df = pd.DataFrame(per_seed_episode)   # [n_seeds, 6]
    ep_stats    = _compute_stats(ep_seeds_df, n_seeds)

    # --- Агрегация orientation accuracy ---
    or_stats = None
    if per_seed_orient:
        or_seeds_df = pd.DataFrame(
            np.stack(per_seed_orient, axis=0),
            columns=orient_columns
        )   # [n_seeds, 3]
        or_stats = _compute_stats(or_seeds_df, len(per_seed_orient))

    return ep_seeds_df, ep_stats, or_seeds_df if per_seed_orient else None, or_stats


def _compute_stats(df: pd.DataFrame, n: int) -> pd.DataFrame:
    """Mean, std, CI95 по строкам df (каждая строка = один seed)."""
    mean      = df.mean()
    std       = df.std(ddof=1)          # несмещённое std
    sem       = std / np.sqrt(n)
    from scipy import stats
    t_crit     = stats.t.ppf(0.975, df=n - 1)  # двусторонний 95% CI
    ci95_delta = t_crit * sem
    return pd.DataFrame({
        "mean":       mean,
        "std":        std,
        "n_seeds":    n,
        "CI95_delta": ci95_delta,
        "CI95_low":   mean - ci95_delta,
        "CI95_high":  mean + ci95_delta,
    })


if __name__ == "__main__":
    log_dir = "/home/xiso/IsaacLab/logs/skrl/logs/old/bbq"

    ep_seeds_df, ep_stats, or_seeds_df, or_stats = aggregate_eval_logs(log_dir)

    print("\n=== Per-seed episode means ===")
    print(ep_seeds_df.to_string())

    print("\n=== Episode metrics: mean-of-means ± 95% CI ===")
    print(ep_stats.to_string())

    if or_stats is not None:
        print("\n=== Per-seed orientation accuracy ===")
        print(or_seeds_df.to_string())
        print("\n=== Orientation accuracy: mean-of-means ± 95% CI ===")
        print(or_stats.to_string())