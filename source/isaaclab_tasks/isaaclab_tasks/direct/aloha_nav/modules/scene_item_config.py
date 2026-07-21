from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Mapping


@dataclass(frozen=True)
class ObjectTransform:
    """Static transform declared for one object type in scene_items.json.

    Quaternion order is Isaac Lab's (w, x, y, z).
    Euler angles use degrees and XYZ roll-pitch-yaw semantics:
    rotation = Rz(z) @ Ry(y) @ Rx(x).
    """

    rotation_deg: tuple[float, float, float]
    rotation_quat_wxyz: tuple[float, float, float, float]
    scale: tuple[float, float, float]
    offset: tuple[float, float, float]


def _finite_float(value: Any, *, field: str, object_name: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"Object {object_name!r}: {field} must contain numbers, got {value!r}"
        ) from exc

    if not math.isfinite(result):
        raise ValueError(
            f"Object {object_name!r}: {field} must be finite, got {result!r}"
        )
    return result


def _vector3(
    value: Any,
    *,
    field: str,
    object_name: str,
    allow_scalar: bool,
) -> tuple[float, float, float]:
    if allow_scalar and isinstance(value, (int, float)):
        scalar = _finite_float(value, field=field, object_name=object_name)
        return (scalar, scalar, scalar)

    if not isinstance(value, (list, tuple)) or len(value) != 3:
        scalar_hint = " or a scalar" if allow_scalar else ""
        raise ValueError(
            f"Object {object_name!r}: {field} must be [x, y, z]"
            f"{scalar_hint}, got {value!r}"
        )

    return tuple(
        _finite_float(component, field=field, object_name=object_name)
        for component in value
    )


def euler_xyz_deg_to_quat_wxyz(
    rotation_deg: tuple[float, float, float],
) -> tuple[float, float, float, float]:
    """Convert XYZ roll-pitch-yaw angles in degrees to (w, x, y, z)."""
    roll, pitch, yaw = (math.radians(angle) for angle in rotation_deg)

    cr = math.cos(roll * 0.5)
    sr = math.sin(roll * 0.5)
    cp = math.cos(pitch * 0.5)
    sp = math.sin(pitch * 0.5)
    cy = math.cos(yaw * 0.5)
    sy = math.sin(yaw * 0.5)

    quat = (
        cr * cp * cy + sr * sp * sy,
        sr * cp * cy - cr * sp * sy,
        cr * sp * cy + sr * cp * sy,
        cr * cp * sy - sr * sp * cy,
    )

    norm = math.sqrt(sum(component * component for component in quat))
    if norm <= 0.0:
        raise ValueError("Computed a zero-length quaternion")
    return tuple(component / norm for component in quat)


def read_object_transform(obj_cfg: Mapping[str, Any]) -> ObjectTransform:
    """Read and validate an object's optional ``transform`` section.

    Supported JSON forms:

        "transform": {
          "rotation_deg": [x, y, z],
          "scale": [sx, sy, sz],
          "offset": [dx, dy, dz]
        }

    ``scale`` may also be a positive scalar. ``offset`` is expressed in the
    scene-local XYZ frame and is added to every runtime placement position.
    Missing values are identity rotation, unit scale and zero offset.
    """
    object_name = str(obj_cfg.get("name", "<unnamed>"))
    raw_transform = obj_cfg.get("transform", {}) or {}
    if not isinstance(raw_transform, Mapping):
        raise ValueError(
            f"Object {object_name!r}: transform must be a JSON object"
        )

    allowed_keys = {"rotation_deg", "scale", "offset"}
    unknown_keys = set(raw_transform).difference(allowed_keys)
    if unknown_keys:
        raise ValueError(
            f"Object {object_name!r}: unknown transform keys "
            f"{sorted(unknown_keys)}; allowed keys are {sorted(allowed_keys)}"
        )

    rotation_deg = _vector3(
        raw_transform.get("rotation_deg", (0.0, 0.0, 0.0)),
        field="transform.rotation_deg",
        object_name=object_name,
        allow_scalar=False,
    )
    scale = _vector3(
        raw_transform.get("scale", (1.0, 1.0, 1.0)),
        field="transform.scale",
        object_name=object_name,
        allow_scalar=True,
    )
    offset = _vector3(
        raw_transform.get("offset", (0.0, 0.0, 0.0)),
        field="transform.offset",
        object_name=object_name,
        allow_scalar=False,
    )

    if any(component <= 0.0 for component in scale):
        raise ValueError(
            f"Object {object_name!r}: transform.scale must be strictly positive, "
            f"got {scale}"
        )

    return ObjectTransform(
        rotation_deg=rotation_deg,
        rotation_quat_wxyz=euler_xyz_deg_to_quat_wxyz(rotation_deg),
        scale=scale,
        offset=offset,
    )
