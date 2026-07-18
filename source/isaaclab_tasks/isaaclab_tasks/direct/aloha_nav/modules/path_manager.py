from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch

GridNode = Tuple[int, int]


class Path_manager:
    """Load fixed four-room paths indexed only by target and start position.

    New database format:
        {target_node: {start_node: path}}

    ``active_obstacles_by_type_list`` remains in ``get_paths`` only to preserve
    compatibility with the existing environment call. It is intentionally ignored.
    """

    def __init__(
        self,
        scene_manager,
        ratio: float = 4.0,
        shift: Optional[List[float]] = None,
        device: str = "cpu",
        paths_file: str = "data/all_paths.json",
        max_path_length: int = 128,
        **_legacy_kwargs,
    ) -> None:
        self.scene_manager = scene_manager
        self.device = torch.device(device)
        self.ratio = float(ratio)
        self.max_path_length = int(max_path_length)
        self.paths_file = str(paths_file)

        if self.ratio <= 0:
            raise ValueError("ratio must be positive")
        if self.max_path_length < 2:
            raise ValueError("max_path_length must be at least 2")

        bounds = getattr(scene_manager, "room_bounds", None)
        if bounds is not None:
            derived_shift = [-float(bounds["x_min"]), -float(bounds["y_min"])]
        else:
            derived_shift = [10.0, 10.0]

        if shift is not None and any(abs(float(a) - float(b)) > 1e-6 for a, b in zip(shift, derived_shift)):
            print(
                f"[Path_manager] Ignoring legacy shift={shift}; "
                f"using room-derived shift={derived_shift}."
            )
        self.shift = torch.tensor(derived_shift, device=self.device, dtype=torch.float32)

        self.all_paths: Dict[GridNode, Dict[GridNode, List[GridNode]]] = {}
        self._load_paths()

    @staticmethod
    def _parse_node(value: str) -> GridNode:
        x, y = value.split(",", 1)
        return int(x), int(y)

    def _load_paths(self) -> None:
        path = Path(self.paths_file)
        if not path.exists():
            raise FileNotFoundError(f"Path database not found: {path}")

        with path.open("r", encoding="utf-8") as f:
            loaded = json.load(f)

        if not isinstance(loaded, dict) or not loaded:
            raise RuntimeError(f"Path database is empty or malformed: {path}")

        first_key = next(iter(loaded))
        if ":" in first_key and "," not in first_key:
            raise RuntimeError(
                "Legacy config-indexed all_paths.json detected. Regenerate it with "
                "path_generator_fixed_four_rooms.py."
            )

        parsed: Dict[GridNode, Dict[GridNode, List[GridNode]]] = {}
        for target_str, starts in loaded.items():
            target = self._parse_node(target_str)
            parsed[target] = {}
            for start_str, path_nodes in starts.items():
                start = self._parse_node(start_str)
                parsed[target][start] = [
                    (int(node[0]), int(node[1])) for node in path_nodes
                ]

        self.all_paths = parsed
        total_starts = sum(len(starts) for starts in parsed.values())
        print(
            f"Loaded fixed path database: {len(parsed)} targets, "
            f"{total_starts} target/start entries from {path}"
        )

    def real_to_grid(self, real_point: torch.Tensor) -> torch.Tensor:
        point = real_point.to(device=self.device, dtype=torch.float32)
        grid_x = torch.round((point[:, 0] + self.shift[0]) * self.ratio).to(torch.int32)
        grid_y = torch.round((point[:, 1] + self.shift[1]) * self.ratio).to(torch.int32)
        return torch.stack([grid_x, grid_y], dim=-1)

    def grid_to_real(self, grid_point: torch.Tensor) -> torch.Tensor:
        point = grid_point.to(dtype=torch.float32)
        real_x = point[..., 0] / self.ratio - self.shift[0].to(point.device)
        real_y = point[..., 1] / self.ratio - self.shift[1].to(point.device)
        return torch.stack([real_x, real_y], dim=-1)

    @staticmethod
    def _nearest_node(target: GridNode, nodes) -> Optional[GridNode]:
        nodes = list(nodes)
        if not nodes:
            return None
        tx, ty = target
        return min(nodes, key=lambda n: abs(tx - n[0]) + abs(ty - n[1]))

    @staticmethod
    def _validate_path_length(
        path: List[GridNode],
        max_length: int,
        *,
        start: GridNode,
        target: GridNode,
    ) -> List[GridNode]:
        if len(path) > max_length:
            raise RuntimeError(
                f"Dense path {start}->{target} contains {len(path)} nodes, "
                f"but max_path_length={max_length}. Increase max_path_length; "
                "do not resample because that creates unsafe shortcuts."
            )
        return path

    def _lookup_path(self, start: GridNode, target: GridNode) -> List[GridNode]:
        if target not in self.all_paths:
            raise KeyError(
                f"Target grid node {target} is absent from the path database. "
                "The database is stale or was generated for another active_rooms config."
            )

        starts = self.all_paths[target]
        start_key = start if start in starts else self._nearest_node(start, starts.keys())
        if start_key is None:
            raise RuntimeError(f"No reachable start nodes stored for target {target}")
        return starts[start_key]

    def get_paths(
        self,
        env_ids: torch.Tensor,
        start_positions: torch.Tensor,
        target_positions: torch.Tensor,
        active_obstacles_by_type_list=None,
        device: Optional[str] = None,
        **_ignored,
    ) -> torch.Tensor:
        """Return paths for current start/target positions.

        Scene obstacle configuration is deliberately ignored because the database
        assumes all four chair slots in every room are occupied.
        """
        del active_obstacles_by_type_list

        output_device = torch.device(device) if device is not None else self.device
        starts_grid = self.real_to_grid(start_positions).cpu().tolist()
        targets_grid = self.real_to_grid(target_positions).cpu().tolist()

        paths: List[List[GridNode]] = []
        for start, target in zip(starts_grid, targets_grid):
            start_node = (int(start[0]), int(start[1]))
            target_node = (int(target[0]), int(target[1]))
            path = self._lookup_path(start_node, target_node)
            paths.append(
                self._validate_path_length(
                    path,
                    self.max_path_length,
                    start=start_node,
                    target=target_node,
                )
            )

        path_tensor = torch.full(
            (len(paths), self.max_path_length, 2),
            float("nan"),
            device=output_device,
            dtype=torch.float32,
        )

        for row, path in enumerate(paths):
            path_t = torch.tensor(path, device=output_device, dtype=torch.float32)
            path_tensor[row, :len(path)] = path_t

        return self.grid_to_real(path_tensor)

    def debug_print_targets(self) -> None:
        for index, target in enumerate(sorted(self.all_paths), 1):
            print(f"{index:3d}: target={target}, starts={len(self.all_paths[target])}")