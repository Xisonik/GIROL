from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import torch


@dataclass(frozen=True)
class WallOrientationConfig:
    """Configuration for room-wall-aware object orientation.

    ``wall_distance`` is measured from the object's final root position to a
    room boundary in scene-local XY coordinates. The object's corrected local
    forward direction is assumed to be +X before the placement-dependent yaw
    is applied.
    """

    enabled: bool = False
    wall_distance: float = 1.25

    @classmethod
    def from_mapping(
        cls,
        raw: Mapping[str, Any] | None,
    ) -> "WallOrientationConfig":
        raw = raw or {}
        if not isinstance(raw, Mapping):
            raise ValueError("object_orientation must be a JSON object")

        allowed_keys = {
            "enabled",
            "wall_distance",
            "forward_axis",
            "corner_mode",
            "_help",
        }
        unknown = set(raw).difference(allowed_keys)
        if unknown:
            raise ValueError(
                "Unknown object_orientation keys: "
                f"{sorted(unknown)}; allowed keys are {sorted(allowed_keys)}"
            )

        forward_axis = str(raw.get("forward_axis", "+x")).lower()
        if forward_axis != "+x":
            raise ValueError(
                "object_orientation.forward_axis currently supports only '+x'"
            )

        corner_mode = str(raw.get("corner_mode", "towards_room_center"))
        if corner_mode != "towards_room_center":
            raise ValueError(
                "object_orientation.corner_mode currently supports only "
                "'towards_room_center'"
            )

        wall_distance = float(raw.get("wall_distance", 1.25))
        if not torch.isfinite(torch.tensor(wall_distance)):
            raise ValueError("object_orientation.wall_distance must be finite")
        if wall_distance < 0.0:
            raise ValueError(
                "object_orientation.wall_distance must be non-negative"
            )

        return cls(
            enabled=bool(raw.get("enabled", False)),
            wall_distance=wall_distance,
        )


def _quat_mul_wxyz(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    """Hamilton product for quaternions stored as (w, x, y, z)."""
    lw, lx, ly, lz = left.unbind(dim=-1)
    rw, rx, ry, rz = right.unbind(dim=-1)
    return torch.stack(
        (
            lw * rw - lx * rx - ly * ry - lz * rz,
            lw * rx + lx * rw + ly * rz - lz * ry,
            lw * ry - lx * rz + ly * rw + lz * rx,
            lw * rz + lx * ry - ly * rx + lz * rw,
        ),
        dim=-1,
    )


def _yaw_quaternion_wxyz(yaw: torch.Tensor) -> torch.Tensor:
    """Return world-Z yaw quaternions in (w, x, y, z) order."""
    half = yaw * 0.5
    result = torch.zeros(
        (*yaw.shape, 4),
        device=yaw.device,
        dtype=yaw.dtype,
    )
    result[..., 0] = torch.cos(half)
    result[..., 3] = torch.sin(half)
    return result


class WallAwareObjectOrienter:
    """Compute placement-dependent object orientations from room geometry.

    Responsibility is deliberately limited to orientation. The class does not
    place objects and does not write poses to the simulator.

    Rules:
    - outside the configured wall zone: keep the base USD orientation;
    - near exactly one wall: point corrected +X along that wall's inward normal;
    - near two or more walls: point corrected +X towards the room center.
    """

    def __init__(
        self,
        *,
        device: str | torch.device,
        room_centers: torch.Tensor,
        room_half_extent: float,
        config: Mapping[str, Any] | None,
    ) -> None:
        self.device = torch.device(device)
        self.config = WallOrientationConfig.from_mapping(config)

        centers = torch.as_tensor(
            room_centers,
            device=self.device,
            dtype=torch.float32,
        )
        if centers.ndim != 2 or centers.shape[1] < 2:
            raise ValueError(
                "room_centers must have shape [num_rooms, 2 or 3], "
                f"got {tuple(centers.shape)}"
            )
        self.room_centers_xy = centers[:, :2].contiguous()

        self.room_half_extent = float(room_half_extent)
        if self.room_half_extent <= 0.0:
            raise ValueError("room_half_extent must be positive")
        if self.config.wall_distance > self.room_half_extent:
            raise ValueError(
                "object_orientation.wall_distance cannot exceed the room "
                f"half extent ({self.room_half_extent})"
            )

        # Order matches distances: x_min, x_max, y_min, y_max.
        self.inward_normals = torch.tensor(
            [
                [1.0, 0.0],
                [-1.0, 0.0],
                [0.0, 1.0],
                [0.0, -1.0],
            ],
            device=self.device,
            dtype=torch.float32,
        )

    @torch.no_grad()
    def compute(
        self,
        *,
        positions: torch.Tensor,
        active: torch.Tensor,
        room_ids: torch.Tensor,
        base_orientations: torch.Tensor,
    ) -> torch.Tensor:
        """Return final orientations for a selected environment batch.

        Shapes:
            positions:         [E, M, 3]
            active:            [E, M]
            room_ids:          [E, M]
            base_orientations: [E, M, 4]
        """
        if positions.ndim != 3 or positions.shape[-1] != 3:
            raise ValueError(
                f"positions must have shape [E, M, 3], got {positions.shape}"
            )
        if active.shape != positions.shape[:2]:
            raise ValueError("active must have shape [E, M]")
        if room_ids.shape != positions.shape[:2]:
            raise ValueError("room_ids must have shape [E, M]")
        if base_orientations.shape != (*positions.shape[:2], 4):
            raise ValueError(
                "base_orientations must have shape [E, M, 4]"
            )

        result = base_orientations.clone()
        if not self.config.enabled:
            return result

        num_rooms = self.room_centers_xy.shape[0]
        valid = (
            active.bool()
            & (room_ids >= 0)
            & (room_ids < num_rooms)
        )
        if not valid.any():
            return result

        safe_room_ids = room_ids.clamp(0, num_rooms - 1).long()
        centers = self.room_centers_xy[safe_room_ids]
        relative_xy = positions[..., :2] - centers
        half = self.room_half_extent

        # Positive distance from the object root to each room boundary.
        distances = torch.stack(
            (
                relative_xy[..., 0] + half,  # x_min / left wall
                half - relative_xy[..., 0],  # x_max / right wall
                relative_xy[..., 1] + half,  # y_min / bottom wall
                half - relative_xy[..., 1],  # y_max / top wall
            ),
            dim=-1,
        )
        near_walls = distances <= self.config.wall_distance
        near_count = near_walls.sum(dim=-1)

        single_wall = valid & (near_count == 1)
        corner = valid & (near_count >= 2)
        orient_mask = single_wall | corner
        if not orient_mask.any():
            return result

        direction = torch.zeros_like(relative_xy)

        nearest_wall = distances.argmin(dim=-1)
        direction[single_wall] = self.inward_normals[
            nearest_wall[single_wall]
        ].to(dtype=direction.dtype)

        # For corners this naturally points diagonally into the room.
        direction[corner] = centers[corner] - positions[..., :2][corner]

        direction_norm = torch.linalg.norm(direction, dim=-1)
        orient_mask &= direction_norm > 1e-8
        if not orient_mask.any():
            return result

        yaw = torch.atan2(direction[..., 1], direction[..., 0])
        yaw_quat = _yaw_quaternion_wxyz(yaw)

        # Apply world-Z placement yaw after the static USD correction rotation.
        composed = _quat_mul_wxyz(yaw_quat, base_orientations)
        composed = composed / torch.linalg.norm(
            composed,
            dim=-1,
            keepdim=True,
        ).clamp_min(1e-8)

        result[orient_mask] = composed[orient_mask]
        return result
