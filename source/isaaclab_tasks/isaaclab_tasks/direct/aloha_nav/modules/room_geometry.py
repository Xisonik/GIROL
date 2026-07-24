from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import torch


@dataclass(frozen=True)
class RoomGeometryConfig:
    """Geometry and activation state of the four physical subrooms.

    Public room numbers are 1-based and follow the visual layout::

        [1, 2]
        [3, 4]

    Internal tensor indices remain 0-based.
    """

    outer_size: float = 20.0
    subroom_size: float = 10.0
    inner_wall_clearance: float = 0.5
    allow_inner_corner_cells: bool = True
    passage_center: float = 5.0
    passage_width: float = 1.2
    room_centers: tuple[tuple[float, float, float], ...] = (
        (-5.0, 5.0, 0.0),
        (5.0, 5.0, 0.0),
        (-5.0, -5.0, 0.0),
        (5.0, -5.0, 0.0),
    )
    active_rooms: tuple[int, ...] = (1, 2, 3, 4)

    @classmethod
    def from_mapping(cls, cfg: Mapping[str, Any] | None) -> "RoomGeometryConfig":
        if not cfg:
            return cls()

        raw_centers: Sequence[Sequence[float]] = cfg.get(
            "room_centers", cls.room_centers
        )
        centers: list[tuple[float, float, float]] = []
        for center in raw_centers:
            if len(center) == 2:
                centers.append((float(center[0]), float(center[1]), 0.0))
            elif len(center) == 3:
                centers.append(
                    (float(center[0]), float(center[1]), float(center[2]))
                )
            else:
                raise ValueError(
                    "Each room center must have 2 or 3 coordinates, "
                    f"got {center!r}"
                )

        if len(centers) != 4:
            raise ValueError(
                f"Exactly four physical room centers are required, got {len(centers)}"
            )

        raw_active = cfg.get("active_rooms", [1, 2, 3, 4])
        if not isinstance(raw_active, list) or not raw_active:
            raise ValueError("room_layout.active_rooms must be a non-empty list")
        active_rooms = tuple(int(room) for room in raw_active)
        if len(set(active_rooms)) != len(active_rooms):
            raise ValueError(
                f"room_layout.active_rooms contains duplicates: {active_rooms}"
            )
        invalid = [room for room in active_rooms if room not in (1, 2, 3, 4)]
        if invalid:
            raise ValueError(
                "Room numbers are 1-based and must belong to {1,2,3,4}; "
                f"invalid values: {invalid}"
            )

        numbering = cfg.get("room_numbering")
        if numbering is not None and numbering != [[1, 2], [3, 4]]:
            raise ValueError(
                "room_numbering is documentation and must equal [[1, 2], [3, 4]]"
            )

        return cls(
            outer_size=float(cfg.get("outer_size", 20.0)),
            subroom_size=float(cfg.get("subroom_size", 10.0)),
            inner_wall_clearance=float(cfg.get("inner_wall_clearance", 0.5)),
            allow_inner_corner_cells=bool(
                cfg.get("allow_inner_corner_cells", True)
            ),
            passage_center=float(cfg.get("passage_center", 5.0)),
            passage_width=float(cfg.get("passage_width", 1.2)),
            room_centers=tuple(centers),
            active_rooms=active_rooms,
        )


class RoomCoordinateMapper:
    """Coordinate transforms and active-room geometry for one Isaac Lab env."""

    ROOM_NEIGHBOURS = {
        1: (2, 3),
        2: (1, 4),
        3: (1, 4),
        4: (2, 3),
    }

    # Fixed navigation geometry in the common env-local/global frame.
    # Robot center is valid inside +/-9.5 m. The internal cross walls occupy
    # +/-0.5 m around each axis. All four doors are always active and each
    # door is a 1.2 x 1.2 m square centered at (+/-5, 0) or (0, +/-5).
    NAVIGATION_OUTER_LIMIT = 9.5
    INNER_WALL_HALF_WIDTH = 0.5
    PASSAGE_CENTER = 5.0
    PASSAGE_HALF_SIZE = 0.6

    def __init__(
        self,
        device: str | torch.device,
        config: Mapping[str, Any] | None = None,
    ):
        self.device = torch.device(device)
        self.config = RoomGeometryConfig.from_mapping(config)
        self.centers = torch.tensor(
            self.config.room_centers,
            device=self.device,
            dtype=torch.float32,
        )
        self.active_room_numbers = tuple(self.config.active_rooms)
        self.active_room_ids = tuple(room - 1 for room in self.active_room_numbers)
        self.active_room_mask = torch.zeros(
            4, dtype=torch.bool, device=self.device
        )
        self.active_room_mask[list(self.active_room_ids)] = True

    @property
    def num_rooms(self) -> int:
        return int(self.centers.shape[0])

    @property
    def num_active_rooms(self) -> int:
        return len(self.active_room_ids)

    @property
    def outer_half_extent(self) -> float:
        return self.config.outer_size * 0.5

    @property
    def subroom_half_extent(self) -> float:
        return self.config.subroom_size * 0.5

    @property
    def room_bounds(self) -> dict[str, float]:
        h = self.outer_half_extent
        return {"x_min": -h, "x_max": h, "y_min": -h, "y_max": h}

    def center(
        self, room_id: int, *, dtype: torch.dtype | None = None
    ) -> torch.Tensor:
        center = self.centers[int(room_id)]
        return center if dtype is None else center.to(dtype=dtype)

    def is_room_active(self, room_id: int) -> bool:
        return bool(self.active_room_mask[int(room_id)].item())

    def local_to_global(
        self,
        local_positions: torch.Tensor,
        room_id: int | torch.Tensor,
    ) -> torch.Tensor:
        local_positions = torch.as_tensor(
            local_positions, device=self.device, dtype=torch.float32
        )
        if isinstance(room_id, int):
            return local_positions + self.center(
                room_id, dtype=local_positions.dtype
            )
        room_id = torch.as_tensor(
            room_id, device=self.device, dtype=torch.long
        )
        return local_positions + self.centers[room_id].to(
            dtype=local_positions.dtype
        )

    def global_to_local(
        self,
        global_positions: torch.Tensor,
        room_id: int | torch.Tensor,
    ) -> torch.Tensor:
        global_positions = torch.as_tensor(
            global_positions, device=self.device, dtype=torch.float32
        )
        if isinstance(room_id, int):
            return global_positions - self.center(
                room_id, dtype=global_positions.dtype
            )
        room_id = torch.as_tensor(
            room_id, device=self.device, dtype=torch.long
        )
        return global_positions - self.centers[room_id].to(
            dtype=global_positions.dtype
        )

    def room_ids_from_positions(self, positions: torch.Tensor) -> torch.Tensor:
        positions = torch.as_tensor(
            positions, device=self.device, dtype=torch.float32
        )
        x_positive = positions[..., 0] >= 0.0
        y_positive = positions[..., 1] >= 0.0
        room_ids = torch.empty_like(x_positive, dtype=torch.long)
        room_ids[y_positive & ~x_positive] = 0
        room_ids[y_positive & x_positive] = 1
        room_ids[~y_positive & ~x_positive] = 2
        room_ids[~y_positive & x_positive] = 3
        return room_ids

    def positions_in_active_rooms(self, positions: torch.Tensor) -> torch.Tensor:
        room_ids = self.room_ids_from_positions(positions)
        return self.active_room_mask[room_ids]

    def positions_in_active_room_interiors(
        self,
        positions: torch.Tensor,
        margin: float = 0.0,
    ) -> torch.Tensor:
        positions = torch.as_tensor(
            positions, device=self.device, dtype=torch.float32
        )
        xy = positions[..., :2]
        result = torch.zeros(xy.shape[:-1], dtype=torch.bool, device=self.device)
        half = self.subroom_half_extent - float(margin)
        if half <= 0:
            raise ValueError(
                f"Room interior margin {margin} leaves no usable area"
            )
        for room_id in self.active_room_ids:
            center = self.centers[room_id, :2]
            result |= ((xy - center).abs() <= half).all(dim=-1)
        return result

    def vertical_passage_open(self, positive_y: bool) -> bool:
        pair = (1, 2) if positive_y else (3, 4)
        return all(room in self.active_room_numbers for room in pair)

    def horizontal_passage_open(self, positive_x: bool) -> bool:
        pair = (2, 4) if positive_x else (1, 3)
        return all(room in self.active_room_numbers for room in pair)

    def positions_in_active_navigation_area(
        self,
        positions: torch.Tensor,
    ) -> torch.Tensor:
        """Return whether robot centers are inside the allowed navigation area.

        The rule is intentionally direct:
        - reject points outside +/-9.5 m on either axis;
        - the two internal wall strips are abs(x) <= 0.5 or abs(y) <= 0.5;
        - a point in a wall strip is valid only inside one of four 1.2 m doors;
        - all four doors are always active.
        """
        positions = torch.as_tensor(
            positions, device=self.device, dtype=torch.float32
        )
        x = positions[..., 0]
        y = positions[..., 1]

        outer = self.NAVIGATION_OUTER_LIMIT
        wall = self.INNER_WALL_HALF_WIDTH
        door_center = self.PASSAGE_CENTER
        door_half = self.PASSAGE_HALF_SIZE

        inside_outer = (x.abs() <= outer) & (y.abs() <= outer)
        inside_wall_strip = (x.abs() <= wall) | (y.abs() <= wall)

        inside_door = (
            ((x.abs() <= door_half) & ((y - door_center).abs() <= door_half))
            | ((x.abs() <= door_half) & ((y + door_center).abs() <= door_half))
            | ((y.abs() <= door_half) & ((x - door_center).abs() <= door_half))
            | ((y.abs() <= door_half) & ((x + door_center).abs() <= door_half))
        )

        return inside_outer & (~inside_wall_strip | inside_door)

    def inner_wall_masks(
        self, global_positions: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        positions = torch.as_tensor(
            global_positions, device=self.device, dtype=torch.float32
        )
        clearance = float(self.config.inner_wall_clearance)
        near_x_wall = positions[..., 0].abs() <= clearance
        near_y_wall = positions[..., 1].abs() <= clearance
        inner_corner = near_x_wall & near_y_wall
        return near_x_wall, near_y_wall, inner_corner

    def forbidden_inner_wall_mask(self, global_positions: torch.Tensor) -> torch.Tensor:
        near_x_wall, near_y_wall, inner_corner = self.inner_wall_masks(
            global_positions
        )
        near_any_inner_wall = near_x_wall | near_y_wall
        if self.config.allow_inner_corner_cells:
            return near_any_inner_wall & ~inner_corner
        return near_any_inner_wall

    def allowed_cell_indices(
        self, local_grid: torch.Tensor, room_id: int
    ) -> list[int]:
        if not self.is_room_active(room_id):
            return []
        global_grid = self.local_to_global(local_grid, room_id)
        forbidden = self.forbidden_inner_wall_mask(global_grid)
        return torch.nonzero(~forbidden, as_tuple=False).view(-1).tolist()