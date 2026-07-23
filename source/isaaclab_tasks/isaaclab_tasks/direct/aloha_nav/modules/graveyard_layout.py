from __future__ import annotations

from collections.abc import Sequence

# One navigation scene occupies env-local coordinates [-10, 10] on X and Y.
SCENE_HALF_EXTENT = 10.0

# Must match InteractiveSceneCfg.env_spacing in the environments.
ENV_SPACING = 30.0

# Graveyard layout:
# - outside the right wall of the current scene (x > 10);
# - two lines parallel to that wall;
# - exactly 1 metre between neighboring grid anchors both along Y and across X.
GRAVEYARD_WALL_X = SCENE_HALF_EXTENT
GRAVEYARD_FIRST_LINE_OFFSET = 1.5
GRAVEYARD_LINES = 2
GRAVEYARD_POSITIONS_PER_LINE = 18
GRAVEYARD_SPACING = 1.0
GRAVEYARD_Y_START = -8.5
GRAVEYARD_GROUND_CLEARANCE = 0.20


def graveyard_capacity() -> int:
    return GRAVEYARD_LINES * GRAVEYARD_POSITIONS_PER_LINE


def validate_graveyard_layout() -> None:
    x_start = GRAVEYARD_WALL_X + GRAVEYARD_FIRST_LINE_OFFSET
    x_end = x_start + (GRAVEYARD_LINES - 1) * GRAVEYARD_SPACING
    y_end = (
        GRAVEYARD_Y_START
        + (GRAVEYARD_POSITIONS_PER_LINE - 1) * GRAVEYARD_SPACING
    )

    next_scene_left_wall = ENV_SPACING - SCENE_HALF_EXTENT

    if GRAVEYARD_SPACING != 1.0:
        raise RuntimeError(
            "Graveyard spacing must be exactly 1 metre, got "
            f"{GRAVEYARD_SPACING}"
        )
    if x_start <= SCENE_HALF_EXTENT:
        raise RuntimeError(
            "Graveyard must start outside the current scene: "
            f"x_start={x_start}, scene_half_extent={SCENE_HALF_EXTENT}"
        )
    if x_end >= next_scene_left_wall:
        raise RuntimeError(
            "Graveyard reaches the neighboring scene: "
            f"x_end={x_end}, next_scene_left_wall={next_scene_left_wall}"
        )
    if GRAVEYARD_Y_START <= -SCENE_HALF_EXTENT or y_end >= SCENE_HALF_EXTENT:
        raise RuntimeError(
            "Graveyard lines must stay inside the wall span on Y: "
            f"y_range=[{GRAVEYARD_Y_START}, {y_end}]"
        )


def graveyard_position(
    global_index: int,
    *,
    scaled_height: float,
    offset: Sequence[float] = (0.0, 0.0, 0.0),
) -> tuple[float, float, float]:
    """Return one env-local root position in lines along the right wall.

    Instances are filled along Y on the first line and then continue on the
    next parallel line. ``global_index`` is shared across all object types, so
    different types cannot receive the same graveyard anchor.
    """
    validate_graveyard_layout()

    if global_index < 0:
        raise ValueError(f"global_index must be non-negative, got {global_index}")
    if global_index >= graveyard_capacity():
        raise RuntimeError(
            "Graveyard capacity exceeded: "
            f"index={global_index}, capacity={graveyard_capacity()}"
        )
    if len(offset) != 3:
        raise ValueError(f"offset must contain 3 values, got {offset!r}")
    if scaled_height <= 0.0:
        raise ValueError(
            f"scaled_height must be positive, got {scaled_height}"
        )

    line_index = global_index // GRAVEYARD_POSITIONS_PER_LINE
    position_in_line = global_index % GRAVEYARD_POSITIONS_PER_LINE

    anchor_x = (
        GRAVEYARD_WALL_X
        + GRAVEYARD_FIRST_LINE_OFFSET
        + line_index * GRAVEYARD_SPACING
    )
    anchor_y = (
        GRAVEYARD_Y_START
        + position_in_line * GRAVEYARD_SPACING
    )

    # Place the object's logical bounding box above the floor. The same object
    # root offset is applied here and in normal scene placement.
    anchor_z = 0.5 * float(scaled_height) + GRAVEYARD_GROUND_CLEARANCE

    return (
        anchor_x + float(offset[0]),
        anchor_y + float(offset[1]),
        anchor_z + float(offset[2]),
    )
