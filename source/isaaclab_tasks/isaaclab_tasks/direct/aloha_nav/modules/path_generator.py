from __future__ import annotations

import argparse
import json
import math
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import networkx as nx

GridNode = Tuple[int, int]
Point3 = Tuple[float, float, float]
PathMap = Dict[str, Dict[str, List[GridNode]]]


class FixedFourRoomPathGenerator:
    """Generate one path database for a fixed four-room geometry.

    The generated JSON has no scene-configuration key. Its format is:

        {target_node: {start_node: path}}

    All four local chair cells are treated as occupied in every room, so the
    database is conservative when the runtime scene contains fewer chairs.
    Random staff objects are intentionally ignored.
    """

    def __init__(
        self,
        scene_items_path: str,
        layout_rules_path: str,
        *,
        ratio: int = 4,
        robot_radius: float = 0.18,
        collision_margin: float = 0.03,
        tracking_margin: float = 0.12,
        passage_center: float | None = None,
        passage_width: float = 1.0,
        save_dir: str = "data",
        paths_filename: str = "all_paths.json",
        graphs_dir: str = "logs/aloha_data_graphs/graphs",
        limit_start_nodes: Optional[int] = None,
    ) -> None:
        self.scene_items_path = Path(scene_items_path)
        self.layout_rules_path = Path(layout_rules_path)
        self.ratio = int(ratio)
        self.robot_radius = float(robot_radius)
        self.collision_margin = float(collision_margin)
        self.tracking_margin = float(tracking_margin)
        self.passage_center = passage_center
        self.passage_width = float(passage_width)
        self.limit_start_nodes = limit_start_nodes

        if self.ratio <= 0:
            raise ValueError("ratio must be positive")
        if self.robot_radius < 0 or self.collision_margin < 0 or self.tracking_margin < 0:
            raise ValueError(
                "robot_radius, collision_margin and tracking_margin must be non-negative"
            )
        if self.passage_width <= 0:
            raise ValueError("passage_width must be positive")

        with self.scene_items_path.open("r", encoding="utf-8") as f:
            self.scene_cfg = json.load(f)
        with self.layout_rules_path.open("r", encoding="utf-8") as f:
            self.rules = json.load(f)

        room_cfg = self.rules.get("room_layout", {})
        self.outer_size = float(room_cfg.get("outer_size", 20.0))
        self.half_extent = self.outer_size * 0.5
        self.shift = (self.half_extent, self.half_extent)
        self.room_bounds = {
            "x_min": -self.half_extent,
            "x_max": self.half_extent,
            "y_min": -self.half_extent,
            "y_max": self.half_extent,
        }
        self.room_centers: List[Point3] = [
            self._to_point3(p)
            for p in room_cfg.get(
                "room_centers",
                [(-5.0, 5.0, 0.0), (5.0, 5.0, 0.0), (-5.0, -5.0, 0.0), (5.0, -5.0, 0.0)],
            )
        ]
        if len(self.room_centers) != 4:
            raise ValueError(f"Expected four room centers, got {len(self.room_centers)}")

        self.subroom_size = float(room_cfg.get("subroom_size", 10.0))
        self.inner_wall_object_clearance = float(room_cfg.get("inner_wall_clearance", 1.0))
        self.allow_inner_corner_cells = bool(room_cfg.get("allow_inner_corner_cells", True))

        active_room_numbers = room_cfg.get("active_rooms", [1, 2, 3, 4])
        if not isinstance(active_room_numbers, list) or not active_room_numbers:
            raise ValueError("room_layout.active_rooms must be a non-empty list")
        if len(set(active_room_numbers)) != len(active_room_numbers):
            raise ValueError("room_layout.active_rooms contains duplicates")
        if any(int(room) < 1 or int(room) > len(self.room_centers) for room in active_room_numbers):
            raise ValueError(
                f"room_layout.active_rooms must contain room numbers 1..{len(self.room_centers)}"
            )
        self.active_room_indices = tuple(int(room) - 1 for room in active_room_numbers)
        self.active_room_centers = [self.room_centers[i] for i in self.active_room_indices]

        if self.passage_center is None:
            self.passage_center = float(room_cfg.get("passage_center", 3.0))
        else:
            self.passage_center = float(self.passage_center)
        self.passage_width = float(room_cfg.get("passage_width", self.passage_width))

        self.grids = self.rules.get("grids", {})
        self.semantic_blocks = self.rules.get("semantic_blocks", {})
        if not self.grids or not self.semantic_blocks:
            raise RuntimeError(
                "layout_rules.json must contain non-empty 'grids' and 'semantic_blocks'"
            )

        (
            self.local_obstacle_grid,
            obstacle_object_names,
        ) = self._collect_obstacle_configuration()
        self.local_goal_grid = self._collect_goal_grid()

        self.obstacle_radius = self._max_radius_for_objects(obstacle_object_names)
        self.obstacles = self._expand_fixed_obstacles()
        self.targets_real = self._expand_targets()
        if not self.targets_real:
            raise RuntimeError("No valid goal positions remain after four-room expansion")

        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        self.paths_file = save_path / paths_filename
        self.graphs_dir = Path(graphs_dir)
        self.graphs_dir.mkdir(parents=True, exist_ok=True)

        self.all_paths: PathMap = {}

    @staticmethod
    def _to_point3(value) -> Point3:
        if len(value) == 2:
            return float(value[0]), float(value[1]), 0.0
        if len(value) == 3:
            return float(value[0]), float(value[1]), float(value[2])
        raise ValueError(f"Expected 2D or 3D point, got {value!r}")

    def _grid_coordinates(self, grid_name: str) -> List[Point3]:
        if grid_name not in self.grids:
            raise KeyError(f"Missing grid {grid_name!r} in layout_rules.json")
        coordinates = self.grids[grid_name].get("coordinates", [])
        if not coordinates:
            raise RuntimeError(f"Grid {grid_name!r} has no coordinates")
        return [self._to_point3(point) for point in coordinates]

    def _collect_obstacle_configuration(self) -> tuple[List[Point3], set[str]]:
        coordinates: set[Point3] = set()
        object_names: set[str] = set()

        for block_name, block_cfg in self.semantic_blocks.items():
            for obstacle_cfg in block_cfg.get("obstacles", []):
                grid_name = obstacle_cfg.get("grid")
                object_name = obstacle_cfg.get("object")
                if not grid_name or not object_name:
                    raise RuntimeError(
                        f"Semantic block {block_name!r} contains an incomplete obstacle config"
                    )
                coordinates.update(self._grid_coordinates(grid_name))
                object_names.add(str(object_name))

        if not coordinates:
            return [], set()
        return sorted(coordinates), object_names

    def _collect_goal_grid(self) -> List[Point3]:
        coordinates: set[Point3] = set()
        for block_name, block_cfg in self.semantic_blocks.items():
            goal_cfg = block_cfg.get("goal")
            if not goal_cfg:
                raise RuntimeError(f"Semantic block {block_name!r} has no goal config")
            grid_name = goal_cfg.get("grid")
            if not grid_name:
                raise RuntimeError(
                    f"Semantic block {block_name!r} goal has no grid"
                )
            coordinates.update(self._grid_coordinates(grid_name))

        if not coordinates:
            raise RuntimeError("No goal grid coordinates found in semantic_blocks")
        return sorted(coordinates)

    def _max_radius_for_objects(self, object_names: set[str]) -> float:
        if not object_names:
            return 0.0

        radii: List[float] = []
        found_names: set[str] = set()
        for obj in self.scene_cfg.get("objects", []):
            name = obj.get("name")
            if name not in object_names:
                continue
            size = obj.get("size")
            if not size or len(size) < 2:
                raise RuntimeError(f"Object {name!r} has no valid XY size")
            radii.append(math.hypot(float(size[0]) * 0.5, float(size[1]) * 0.5))
            found_names.add(str(name))

        missing = sorted(object_names.difference(found_names))
        if missing:
            raise RuntimeError(
                f"Obstacle objects referenced by semantic blocks are missing "
                f"from scene_items.json: {missing}"
            )
        return max(radii)

    def _point_in_active_room(self, x: float, y: float) -> bool:
        half = 0.5 * self.subroom_size
        eps = 1e-9
        for cx, cy, _ in self.active_room_centers:
            if (
                cx - half - eps <= x <= cx + half + eps
                and cy - half - eps <= y <= cy + half + eps
            ):
                return True
        return False

    def _local_to_full(self, local: Point3, center: Point3) -> Point3:
        return local[0] + center[0], local[1] + center[1], local[2] + center[2]

    def _forbidden_near_inner_wall(self, point: Point3) -> bool:
        near_x = abs(point[0]) <= self.inner_wall_object_clearance
        near_y = abs(point[1]) <= self.inner_wall_object_clearance
        if self.allow_inner_corner_cells and near_x and near_y:
            return False
        return near_x or near_y

    def _expand_fixed_obstacles(self) -> List[Point3]:
        out: List[Point3] = []
        for center in self.active_room_centers:
            for local in self.local_obstacle_grid:
                point = self._local_to_full(local, center)
                if not self._forbidden_near_inner_wall(point):
                    out.append(point)
        return sorted(set(out))

    def _expand_targets(self) -> List[Point3]:
        out: List[Point3] = []
        for center in self.active_room_centers:
            for local in self.local_goal_grid:
                point = self._local_to_full(local, center)
                if not self._forbidden_near_inner_wall(point):
                    out.append(point)
        return sorted(set(out))

    def grid_to_real(self, grid_point: GridNode) -> Tuple[float, float]:
        return (
            grid_point[0] / self.ratio - self.shift[0],
            grid_point[1] / self.ratio - self.shift[1],
        )

    def real_to_grid(self, real_point: Point3 | Tuple[float, float]) -> GridNode:
        return (
            int(round((float(real_point[0]) + self.shift[0]) * self.ratio)),
            int(round((float(real_point[1]) + self.shift[1]) * self.ratio)),
        )

    @property
    def footprint_clearance(self) -> float:
        # Extra margin is needed because a discrete controller does not execute
        # the graph polyline exactly. Without it, valid graph nodes can be only
        # a few centimetres away from the inflated wall/obstacle boundary.
        return self.robot_radius + self.collision_margin + self.tracking_margin

    @property
    def usable_passage_half_width(self) -> float:
        return self.passage_width * 0.5 - self.footprint_clearance

    def _in_vertical_opening(self, y: float) -> bool:
        half = self.usable_passage_half_width
        if half < 0.0:
            return False
        return (
            abs(y - self.passage_center) <= half
            or abs(y + self.passage_center) <= half
        )

    def _in_horizontal_opening(self, x: float) -> bool:
        half = self.usable_passage_half_width
        if half < 0.0:
            return False
        return (
            abs(x - self.passage_center) <= half
            or abs(x + self.passage_center) <= half
        )

    def _point_collides_with_wall(self, x: float, y: float) -> bool:
        c = self.footprint_clearance
        vertical_blocked = abs(x) <= c and not self._in_vertical_opening(y)
        horizontal_blocked = abs(y) <= c and not self._in_horizontal_opening(x)
        return vertical_blocked or horizontal_blocked

    def _point_collides_with_obstacle(self, x: float, y: float) -> bool:
        inflated = self.obstacle_radius + self.footprint_clearance
        inflated2 = inflated * inflated
        for ox, oy, _ in self.obstacles:
            if (x - ox) ** 2 + (y - oy) ** 2 < inflated2:
                return True
        return False

    def _point_is_valid(self, node: GridNode) -> bool:
        x, y = self.grid_to_real(node)
        c = self.footprint_clearance
        b = self.room_bounds
        if not (
            b["x_min"] + c <= x <= b["x_max"] - c
            and b["y_min"] + c <= y <= b["y_max"] - c
        ):
            return False
        if not self._point_in_active_room(x, y):
            return False
        if self._point_collides_with_wall(x, y):
            return False
        if self._point_collides_with_obstacle(x, y):
            return False
        return True

    @staticmethod
    def _point_segment_distance_sq(
        px: float,
        py: float,
        ax: float,
        ay: float,
        bx: float,
        by: float,
    ) -> float:
        dx = bx - ax
        dy = by - ay
        denom = dx * dx + dy * dy
        if denom <= 1e-12:
            return (px - ax) ** 2 + (py - ay) ** 2
        t = ((px - ax) * dx + (py - ay) * dy) / denom
        t = max(0.0, min(1.0, t))
        qx = ax + t * dx
        qy = ay + t * dy
        return (px - qx) ** 2 + (py - qy) ** 2

    def _segment_hits_obstacle(self, a: Tuple[float, float], b: Tuple[float, float]) -> bool:
        inflated = self.obstacle_radius + self.footprint_clearance
        inflated2 = inflated * inflated
        for ox, oy, _ in self.obstacles:
            if self._point_segment_distance_sq(ox, oy, a[0], a[1], b[0], b[1]) < inflated2:
                return True
        return False

    @staticmethod
    def _coordinate_at_axis_crossing(
        a_axis: float,
        b_axis: float,
        a_other: float,
        b_other: float,
    ) -> float:
        delta = b_axis - a_axis
        if abs(delta) <= 1e-12:
            return a_other
        t = max(0.0, min(1.0, -a_axis / delta))
        return a_other + t * (b_other - a_other)

    def _segment_hits_inner_wall(self, a: Tuple[float, float], b: Tuple[float, float]) -> bool:
        c = self.footprint_clearance
        ax, ay = a
        bx, by = b

        touches_vertical = min(ax, bx) <= c and max(ax, bx) >= -c
        if touches_vertical:
            y_at_wall = self._coordinate_at_axis_crossing(ax, bx, ay, by)
            if not self._in_vertical_opening(y_at_wall):
                return True

        touches_horizontal = min(ay, by) <= c and max(ay, by) >= -c
        if touches_horizontal:
            x_at_wall = self._coordinate_at_axis_crossing(ay, by, ax, bx)
            if not self._in_horizontal_opening(x_at_wall):
                return True

        return False

    def _segment_is_valid(self, u: GridNode, v: GridNode) -> bool:
        a = self.grid_to_real(u)
        b = self.grid_to_real(v)
        midpoint = ((a[0] + b[0]) * 0.5, (a[1] + b[1]) * 0.5)
        return not (
            not self._point_in_active_room(midpoint[0], midpoint[1])
            or self._segment_hits_inner_wall(a, b)
            or self._segment_hits_obstacle(a, b)
        )

    def _build_graph(self) -> nx.Graph:
        side = int(round(self.outer_size * self.ratio)) + 1
        graph = nx.Graph()

        valid_nodes: List[GridNode] = []
        for gx in range(side):
            for gy in range(side):
                node = (gx, gy)
                if self._point_is_valid(node):
                    valid_nodes.append(node)
        graph.add_nodes_from(valid_nodes)
        valid_set = set(valid_nodes)

        directions = ((1, 0, 1.0), (0, 1, 1.0), (1, 1, math.sqrt(2.0)), (1, -1, math.sqrt(2.0)))
        for node in valid_nodes:
            x, y = node
            for dx, dy, weight in directions:
                other = (x + dx, y + dy)
                if other in valid_set and self._segment_is_valid(node, other):
                    graph.add_edge(node, other, weight=weight)

        if graph.number_of_nodes() == 0:
            raise RuntimeError("Generated graph is empty")
        return graph

    @staticmethod
    def _nearest_reachable(graph: nx.Graph, target: GridNode) -> Optional[GridNode]:
        if target in graph and graph.degree(target) > 0:
            return target
        tx, ty = target
        candidates = (n for n in graph.nodes if graph.degree(n) > 0)
        return min(candidates, key=lambda n: abs(tx - n[0]) + abs(ty - n[1]), default=None)

    @staticmethod
    def _simplify_direction_changes(path: List[GridNode]) -> List[GridNode]:
        """Remove only exactly collinear grid points; never create unsafe shortcuts."""
        if len(path) <= 2:
            return path
        out = [path[0]]
        previous_direction = (
            path[1][0] - path[0][0],
            path[1][1] - path[0][1],
        )
        for i in range(1, len(path) - 1):
            next_direction = (
                path[i + 1][0] - path[i][0],
                path[i + 1][1] - path[i][1],
            )
            if next_direction != previous_direction:
                out.append(path[i])
            previous_direction = next_direction
        out.append(path[-1])
        return out

    def _save_debug_graph(
        self,
        graph: nx.Graph,
        target: GridNode,
        sample_path: Optional[List[GridNode]],
        index: int,
    ) -> None:
        import matplotlib.pyplot as plt
        from matplotlib.patches import Circle

        fig, ax = plt.subplots(figsize=(9, 9), dpi=160)
        ax.set_aspect("equal")

        nodes_real = [self.grid_to_real(n) for n in graph.nodes]
        ax.scatter([p[0] for p in nodes_real], [p[1] for p in nodes_real], s=2)

        for ox, oy, _ in self.obstacles:
            ax.add_patch(
                Circle(
                    (ox, oy),
                    self.obstacle_radius + self.footprint_clearance,
                    fill=False,
                    linewidth=1.0,
                )
            )

        target_real = self.grid_to_real(target)
        ax.scatter([target_real[0]], [target_real[1]], s=55, marker="*")

        if sample_path:
            path_real = [self.grid_to_real(n) for n in sample_path]
            ax.plot([p[0] for p in path_real], [p[1] for p in path_real], linewidth=1.5)

        ax.axvline(0.0, linewidth=0.8)
        ax.axhline(0.0, linewidth=0.8)
        ax.set_xlim(self.room_bounds["x_min"] - 0.5, self.room_bounds["x_max"] + 0.5)
        ax.set_ylim(self.room_bounds["y_min"] - 0.5, self.room_bounds["y_max"] + 0.5)
        ax.set_title(f"Fixed four-room graph, target={target_real}")
        fig.tight_layout()
        fig.savefig(self.graphs_dir / f"fixed_graph_target_{index}.png")
        plt.close(fig)

    def generate(self, *, save_graph_images: bool = True) -> str:
        started = time.perf_counter()
        print(f"Generate start: {datetime.now().strftime('%H:%M:%S')}")
        print(f"Fixed obstacles: {len(self.obstacles)}")
        print(f"Target positions: {self.targets_real}")

        graph = self._build_graph()
        print(f"Graph: {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges")

        result: PathMap = {}
        target_nodes = [self.real_to_grid(p) for p in self.targets_real]

        for target_index, requested_target in enumerate(target_nodes):
            target = self._nearest_reachable(graph, requested_target)
            if target is None:
                raise RuntimeError(f"No reachable node for target {requested_target}")

            paths_from_target = nx.single_source_dijkstra_path(graph, target, weight="weight")
            starts = list(paths_from_target.keys())
            if self.limit_start_nodes is not None:
                starts = starts[: self.limit_start_nodes]

            target_key = f"{target[0]},{target[1]}"
            start_map: Dict[str, List[GridNode]] = {}
            farthest_sample: Optional[List[GridNode]] = None
            farthest_length = -1

            for start in starts:
                raw_path = list(reversed(paths_from_target[start]))
                # Keep every 0.25 m graph node. The discrete controller needs
                # dense waypoints; direction-only simplification makes it cut
                # corners near walls and obstacles.
                start_map[f"{start[0]},{start[1]}"] = raw_path
                if len(raw_path) > farthest_length:
                    farthest_length = len(raw_path)
                    farthest_sample = raw_path

            result[target_key] = start_map
            print(
                f"Target {target_index + 1}/{len(target_nodes)} {target_key}: "
                f"{len(start_map)} start nodes"
            )

            if save_graph_images:
                self._save_debug_graph(graph, target, farthest_sample, target_index)

        self.all_paths = result
        with self.paths_file.open("w", encoding="utf-8") as f:
            json.dump(result, f, separators=(",", ":"))

        elapsed = time.perf_counter() - started
        print(f"Saved: {self.paths_file}")
        print(f"Elapsed: {elapsed:.2f} s")
        return str(self.paths_file)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate fixed four-room expert paths")
    parser.add_argument(
        "--scene-items",
        default="source/isaaclab_tasks/isaaclab_tasks/direct/aloha_nav/configs/scene_items.json",
    )
    parser.add_argument(
        "--layout-rules",
        default="source/isaaclab_tasks/isaaclab_tasks/direct/aloha_nav/configs/layout_rules.json",
    )
    parser.add_argument("--save-dir", default="data")
    parser.add_argument("--ratio", type=int, default=4)
    parser.add_argument("--robot-radius", type=float, default=0.18)
    parser.add_argument("--collision-margin", type=float, default=0.03)
    parser.add_argument(
        "--tracking-margin",
        type=float,
        default=0.12,
        help="Extra planning clearance for discrete path tracking",
    )
    parser.add_argument(
        "--passage-center",
        type=float,
        default=None,
        help="Override room_layout.passage_center",
    )
    parser.add_argument("--passage-width", type=float, default=1.0)
    parser.add_argument("--no-images", action="store_true")
    args = parser.parse_args()

    generator = FixedFourRoomPathGenerator(
        scene_items_path=args.scene_items,
        layout_rules_path=args.layout_rules,
        ratio=args.ratio,
        robot_radius=args.robot_radius,
        collision_margin=args.collision_margin,
        tracking_margin=args.tracking_margin,
        passage_center=args.passage_center,
        passage_width=args.passage_width,
        save_dir=args.save_dir,
    )
    generator.generate(save_graph_images=not args.no_images)


if __name__ == "__main__":
    main()