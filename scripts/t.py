from pathlib import Path
import shutil
import math

from tensorboard.backend.event_processing.event_accumulator import (
    EventAccumulator,
    SCALARS,
    TENSORS,
)
from tensorboard.util import tensor_util
from torch.utils.tensorboard import SummaryWriter


# ============================================================
# GLOBAL PARAMETERS — МЕНЯЙ ТОЛЬКО ЭТОТ БЛОК
# ============================================================

# Папка run-а skrl, где лежат events.out.tfevents.* и tensorboard/
i = 6
INPUT_DIR = Path(f"/home/xiso/IsaacLab/logs/s/{i}")

# Куда сохранить новый TensorBoard run только с одним графиком
OUTPUT_DIR = Path(f"/home/xiso/IsaacLab/logs/t/{i}")

# TensorBoard UI показывает "Metrics / success_rate_percent",
# но реальный tag почти наверняка такой:
TARGET_TAG = "Metrics / success_rate_percent"

# Привести максимальный step к N тысячам.
# Например 100 => максимум X станет 100_000.
# Если X менять не нужно, поставь None.
TARGET_X_MAX_K = 100

# Дополнительный множитель X.
# Обычно оставь 1.0.
X_SCALE_EXTRA = 1.0

# Смещение графика по Y.
# Например +10 поднимет success_rate на 10 процентных пунктов.
Y_SHIFT = -0.0

# Сжатие/растяжение самого графика по Y.
# 1.0 — без изменения
# 0.5 — сжать амплитуду в 2 раза
# 2.0 — растянуть амплитуду в 2 раза
Y_SCALE = 0.3

# Центр, относительно которого сжимается/растягивается Y.
# Для success_rate_percent чаще всего логично 0.0.
Y_CENTER = 0.0

# Ограничивать ли success_rate диапазоном [0, 100].
# Для percent-метрики обычно лучше True.
CLIP_Y = True
CLIP_Y_MIN = 0.0
CLIP_Y_MAX = 100.0

# Если в разных event-файлах есть одинаковые step для этого tag,
# оставить только последнее значение по wall_time.
DEDUPLICATE_BY_STEP = True

# Если OUTPUT_DIR уже существует — удалить и создать заново.
OVERWRITE_OUTPUT = True


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


def find_event_dirs(root: Path) -> list[Path]:
    dirs = set()

    for event_file in root.rglob(EVENT_GLOB):
        dirs.add(event_file.parent)

    return sorted(dirs)


def list_all_tags(event_dirs: list[Path]) -> set[str]:
    tags = set()

    for event_dir in event_dirs:
        acc = EventAccumulator(
            str(event_dir),
            size_guidance={SCALARS: 0, TENSORS: 0},
        )
        acc.Reload()

        for tag in acc.Tags().get("scalars", []):
            tags.add(tag)

        for tag in acc.Tags().get("tensors", []):
            tags.add(tag)

    return tags


def read_target_records(event_dir: Path, target_tag: str) -> list[dict]:
    acc = EventAccumulator(
        str(event_dir),
        size_guidance={
            SCALARS: 0,
            TENSORS: 0,
        },
    )
    acc.Reload()

    records = []

    # Обычный scalar формат
    if target_tag in acc.Tags().get("scalars", []):
        for e in acc.Scalars(target_tag):
            value = float(e.value)

            if math.isfinite(value):
                records.append(
                    {
                        "tag": target_tag,
                        "step": int(e.step),
                        "wall_time": float(e.wall_time),
                        "value": value,
                        "source_dir": str(event_dir),
                    }
                )

    # Tensor-based scalar формат
    if target_tag in acc.Tags().get("tensors", []):
        for e in acc.Tensors(target_tag):
            if not is_scalar_tensor_event(e):
                continue

            value = tensor_event_to_float(e)

            if math.isfinite(value):
                records.append(
                    {
                        "tag": target_tag,
                        "step": int(e.step),
                        "wall_time": float(e.wall_time),
                        "value": value,
                        "source_dir": str(event_dir),
                    }
                )

    return records


def deduplicate_records_by_step(records: list[dict]) -> list[dict]:
    """
    Если есть несколько значений на одном step, оставляем последнее по wall_time.
    """
    by_step = {}

    for r in records:
        step = r["step"]

        if step not in by_step:
            by_step[step] = r
        else:
            if r["wall_time"] >= by_step[step]["wall_time"]:
                by_step[step] = r

    return [by_step[step] for step in sorted(by_step)]


def transform_step(step: int, x_scale: float) -> int:
    return int(round(step * x_scale))


def transform_value(value: float) -> float:
    new_value = Y_CENTER + (value - Y_CENTER) * Y_SCALE + Y_SHIFT

    if CLIP_Y:
        new_value = max(CLIP_Y_MIN, min(CLIP_Y_MAX, new_value))

    return float(new_value)


def main():
    input_dir = INPUT_DIR.resolve()
    output_dir = OUTPUT_DIR.resolve()

    if not input_dir.exists():
        raise FileNotFoundError(f"INPUT_DIR does not exist: {input_dir}")

    if output_dir.exists():
        if OVERWRITE_OUTPUT:
            shutil.rmtree(output_dir)
        else:
            raise FileExistsError(f"OUTPUT_DIR already exists: {output_dir}")

    event_dirs = find_event_dirs(input_dir)

    if not event_dirs:
        raise RuntimeError(f"No TensorBoard event files found under: {input_dir}")

    records = []

    for event_dir in event_dirs:
        records.extend(read_target_records(event_dir, TARGET_TAG))

    if not records:
        available_tags = sorted(list_all_tags(event_dirs))

        print("Target tag not found.")
        print(f"Requested tag: {TARGET_TAG}")
        print()
        print("Available tags:")
        for tag in available_tags:
            print(f"  {tag}")

        raise RuntimeError(
            "TARGET_TAG was not found. "
            "Check exact spelling in the printed available tags."
        )

    if DEDUPLICATE_BY_STEP:
        records = deduplicate_records_by_step(records)
    else:
        records = sorted(records, key=lambda r: (r["step"], r["wall_time"]))

    old_max_step = max(r["step"] for r in records)

    if TARGET_X_MAX_K is None:
        x_scale = X_SCALE_EXTRA
        target_max_step = None
    else:
        target_max_step = int(TARGET_X_MAX_K * 1000)

        if old_max_step <= 0:
            raise RuntimeError(f"Invalid old_max_step: {old_max_step}")

        x_scale = (target_max_step / old_max_step) * X_SCALE_EXTRA

    output_dir.mkdir(parents=True, exist_ok=True)

    writer = SummaryWriter(str(output_dir))

    for r in records:
        new_step = transform_step(r["step"], x_scale)
        new_value = transform_value(r["value"])

        writer.add_scalar(
            tag=TARGET_TAG,
            scalar_value=new_value,
            global_step=new_step,
            walltime=r["wall_time"],
        )

    writer.flush()
    writer.close()

    print("Done.")
    print(f"Input dir:       {input_dir}")
    print(f"Output dir:      {output_dir}")
    print(f"Target tag:      {TARGET_TAG}")
    print(f"Records written: {len(records)}")
    print(f"Old max step:    {old_max_step}")
    print(f"X scale:         {x_scale}")

    if target_max_step is not None:
        print(f"Target max step: {target_max_step}")

    print(
        "Y transform:     "
        f"y_new = {Y_CENTER} + (y_old - {Y_CENTER}) * {Y_SCALE} + {Y_SHIFT}"
    )

    if CLIP_Y:
        print(f"Y clipping:      [{CLIP_Y_MIN}, {CLIP_Y_MAX}]")


if __name__ == "__main__":
    main()