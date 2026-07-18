from pathlib import Path
import math

import numpy as np
import matplotlib.pyplot as plt

from tensorboard.backend.event_processing.event_accumulator import (
    EventAccumulator,
    SCALARS,
    TENSORS,
)
from tensorboard.util import tensor_util


# ============================================================
# GLOBAL PARAMETERS — МЕНЯЙ ТОЛЬКО ЭТОТ БЛОК
# ============================================================

# Папка, куда ты скопировал runs
ROOT_DIR = Path("s")

# Один нужный TensorBoard tag.
# В UI это видно как "Metrics / success_rate_percent",
# но реальное имя почти всегда такое:
TARGET_TAG = "Metrics / success_rate_percent"

# Runs для отрисовки.
# Первый элемент — подпись в легенде.
# Второй элемент — путь относительно ROOT_DIR.
RUNS = [
    (
        "GIROL без ориентации",
        "/home/xiso/IsaacLab/logs/t/1",
    ),
    (
        "GIROL",
        "/home/xiso/IsaacLab/logs/t/2",
    ),
    (
        "AKGVP",
        "/home/xiso/IsaacLab/logs/t/3",
    ),
    (
        "GIROL без эксперта",
        "/home/xiso/IsaacLab/logs/t/4",
    ),
    # (
    #     "5",
    #     "/home/xiso/IsaacLab/logs/t/5",
    # ),
    (
        "PPO+LSTM",
        "/home/xiso/IsaacLab/logs/t/6",
    ),
]

# Куда сохранить PNG
OUTPUT_PNG = Path("/home/xiso/IsaacLab/logs/success_rate_percent.png")

# Привести максимум X каждого графика к N тысячам.
# Например 100 => каждый график будет растянут/сжат до 100_000 по X.
# Если X менять не нужно, поставь None.
TARGET_X_MAX_K = None

# Дополнительный множитель X.
X_SCALE_EXTRA = 1.0

# Смещение графика по Y.
# Например +10 поднимет success_rate на 10 процентных пунктов.
Y_SHIFT = 0.0

# Масштабирование самого графика по Y.
# 1.0 — без изменения
# 0.5 — сжать амплитуду в 2 раза
# 2.0 — растянуть амплитуду в 2 раза
Y_SCALE = 1.0

# Центр сжатия/растяжения по Y.
# Формула: y_new = Y_CENTER + (y_old - Y_CENTER) * Y_SCALE + Y_SHIFT
Y_CENTER = 0.0

# Для success_rate_percent обычно нужно оставить значения в [0, 100]
CLIP_Y = True
CLIP_Y_MIN = 0.0
CLIP_Y_MAX = 100.0

# Сглаживание, похожее по смыслу на TensorBoard smoothing.
# 0.0 — без сглаживания
# 0.6–0.9 — заметное сглаживание
SMOOTHING = 0.0

# Размер и качество PNG
FIGSIZE = (10, 6)
DPI = 300

# Заголовок и подписи
TITLE = "Success rate"
X_LABEL = "Environment steps"
Y_LABEL = "Success rate, %"

# Диапазоны осей.
# None => автоматически.
X_LIM = None
Y_LIM = (0, 100)

# Толщина линий
LINE_WIDTH = 2.2

# Показывать маркеры на точках.
SHOW_MARKERS = False

# ============================================================
# CODE
# ============================================================

EVENT_GLOB = "events.out.tfevents.*"


def is_scalar_tensor_event(event) -> bool:
    arr = tensor_util.make_ndarray(event.tensor_proto)
    return arr.size == 1


def tensor_event_to_float(event) -> float:
    arr = tensor_util.make_ndarray(event.tensor_proto)
    return float(arr.reshape(-1)[0])


def find_event_dirs(run_dir: Path) -> list[Path]:
    dirs = set()

    for event_file in run_dir.rglob(EVENT_GLOB):
        dirs.add(event_file.parent)

    return sorted(dirs)


def read_target_records_from_event_dir(event_dir: Path, target_tag: str) -> list[dict]:
    acc = EventAccumulator(
        str(event_dir),
        size_guidance={
            SCALARS: 0,
            TENSORS: 0,
        },
    )
    acc.Reload()

    records = []

    if target_tag in acc.Tags().get("scalars", []):
        for e in acc.Scalars(target_tag):
            value = float(e.value)

            if math.isfinite(value):
                records.append(
                    {
                        "step": int(e.step),
                        "wall_time": float(e.wall_time),
                        "value": value,
                    }
                )

    if target_tag in acc.Tags().get("tensors", []):
        for e in acc.Tensors(target_tag):
            if not is_scalar_tensor_event(e):
                continue

            value = tensor_event_to_float(e)

            if math.isfinite(value):
                records.append(
                    {
                        "step": int(e.step),
                        "wall_time": float(e.wall_time),
                        "value": value,
                    }
                )

    return records


def list_available_tags(run_dir: Path) -> set[str]:
    tags = set()

    for event_dir in find_event_dirs(run_dir):
        acc = EventAccumulator(
            str(event_dir),
            size_guidance={
                SCALARS: 0,
                TENSORS: 0,
            },
        )
        acc.Reload()

        tags.update(acc.Tags().get("scalars", []))
        tags.update(acc.Tags().get("tensors", []))

    return tags


def deduplicate_by_step(records: list[dict]) -> list[dict]:
    """
    Если в нескольких event-файлах есть одинаковый step,
    оставляем последнее значение по wall_time.
    """
    by_step = {}

    for r in records:
        step = r["step"]

        if step not in by_step:
            by_step[step] = r
        elif r["wall_time"] >= by_step[step]["wall_time"]:
            by_step[step] = r

    return [by_step[step] for step in sorted(by_step)]


def smooth_ema(values: np.ndarray, smoothing: float) -> np.ndarray:
    if smoothing <= 0:
        return values

    if not 0 <= smoothing < 1:
        raise ValueError("SMOOTHING must be in [0, 1).")

    smoothed = np.empty_like(values, dtype=float)
    smoothed[0] = values[0]

    for i in range(1, len(values)):
        smoothed[i] = smoothing * smoothed[i - 1] + (1.0 - smoothing) * values[i]

    return smoothed


def transform_y(values: np.ndarray) -> np.ndarray:
    y = Y_CENTER + (values - Y_CENTER) * Y_SCALE + Y_SHIFT

    if CLIP_Y:
        y = np.clip(y, CLIP_Y_MIN, CLIP_Y_MAX)

    return y


def read_series(run_dir: Path, target_tag: str) -> tuple[np.ndarray, np.ndarray]:
    records = []

    event_dirs = find_event_dirs(run_dir)

    if not event_dirs:
        raise RuntimeError(f"No event files found in: {run_dir}")

    for event_dir in event_dirs:
        records.extend(read_target_records_from_event_dir(event_dir, target_tag))

    if not records:
        tags = sorted(list_available_tags(run_dir))

        print()
        print(f"Target tag not found in: {run_dir}")
        print(f"Requested tag: {target_tag}")
        print("Available tags:")
        for tag in tags:
            print(f"  {tag}")

        raise RuntimeError("TARGET_TAG was not found. See printed available tags.")

    records = deduplicate_by_step(records)

    x = np.array([r["step"] for r in records], dtype=float)
    y = np.array([r["value"] for r in records], dtype=float)

    if TARGET_X_MAX_K is not None:
        old_max_step = float(np.max(x))

        if old_max_step <= 0:
            raise RuntimeError(f"Invalid max step in {run_dir}: {old_max_step}")

        target_max_step = float(TARGET_X_MAX_K * 1000)
        x = x * (target_max_step / old_max_step) * X_SCALE_EXTRA
    else:
        x = x * X_SCALE_EXTRA

    y = transform_y(y)
    y = smooth_ema(y, SMOOTHING)

    return x, y


def main():
    fig, ax = plt.subplots(figsize=FIGSIZE, dpi=DPI)

    # Белый фон вместо тёмного
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    plotted = 0

    for label, rel_path in RUNS:
        run_dir = ROOT_DIR / rel_path

        if not run_dir.exists():
            print(f"WARNING: run directory not found, skipped: {run_dir}")
            continue

        x, y = read_series(run_dir, TARGET_TAG)

        marker = "o" if SHOW_MARKERS else None

        ax.plot(
            x,
            y,
            label=label,
            linewidth=LINE_WIDTH,
            marker=marker,
            markersize=3 if SHOW_MARKERS else None,
        )

        plotted += 1

    if plotted == 0:
        raise RuntimeError("No runs were plotted.")

    ax.set_xlabel(X_LABEL, fontsize=13)
    ax.set_ylabel(Y_LABEL, fontsize=13)

    if X_LIM is not None:
        ax.set_xlim(*X_LIM)

    if Y_LIM is not None:
        ax.set_ylim(*Y_LIM)

    ax.grid(True, linewidth=0.8, alpha=0.3)

    # Нормальная легенда
    ax.legend(
        loc="best",
        frameon=True,
        fontsize=10,
    )

    ax.tick_params(axis="both", labelsize=11)

    # Убрать лишние рамки сверху/справа
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()

    fig.savefig(
        OUTPUT_PNG,
        dpi=DPI,
        facecolor="white",
        bbox_inches="tight",
    )

    print(f"Saved: {OUTPUT_PNG.resolve()}")


if __name__ == "__main__":
    main()