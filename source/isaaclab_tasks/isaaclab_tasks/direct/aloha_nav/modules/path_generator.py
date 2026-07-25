from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import networkx as nx

GridNode = Tuple[int, int]
Point3 = Tuple[float, float, float]
PathMap = Dict[str, Dict[str, List[GridNode]]]


@dataclass(frozen=True)
class TargetApproach:
    """Target-specific pre-goal point and its graph connections."""

    goal_real: Point3
    room_center: Point3
    approach_node: GridNode
    approach_distance_used: float
    connector_nodes: Tuple[GridNode, ...]


class FixedFourRoomPathGenerator:
    """Generate one path database for a fixed four-room geometry.

    The generated JSON has no scene-configuration key. Its format is:

        {target_node: {start_node: path}}

    All local obstacle cells are treated as occupied in every room, so the
    database is conservative when the runtime scene contains fewer obstacles.
    A configurable wall-side strip is excluded from the base graph. For every
    physical goal, one pre-goal point is placed ``approach_distance`` metres from
    the goal toward the room center. The graph leads to that point, after which
    one direct segment is appended from the pre-goal point to the physical goal.
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
        obstacle_center_clearance: float = 0.8,
        wall_obstacle_depth: float = 1.5,
        approach_distance: float = 1.0,
        approach_connection_radius: float = 2.0,
        passage_center: float | None = None,
        passage_width: float | None = None,
        navigation_outer_limit: float = 9.5,
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
        self.obstacle_center_clearance = float(obstacle_center_clearance)
        self.wall_obstacle_depth = float(wall_obstacle_depth)
        self.approach_distance = float(approach_distance)
        self.approach_connection_radius = float(approach_connection_radius)
        self.passage_center = passage_center
        self.passage_width = passage_width
        self.navigation_outer_limit = float(navigation_outer_limit)
        self.limit_start_nodes = limit_start_nodes

        if self.ratio <= 0:
            raise ValueError("ratio must be positive")
        if self.robot_radius < 0 or self.collision_margin < 0 or self.tracking_margin < 0:
            raise ValueError(
                "robot_radius, collision_margin and tracking_margin must be non-negative"
            )
        if self.obstacle_center_clearance <= 0:
            raise ValueError("obstacle_center_clearance must be positive")
        if self.wall_obstacle_depth < 0:
            raise ValueError("wall_obstacle_depth must be non-negative")
        if self.approach_distance <= 0:
            raise ValueError("approach_distance must be positive")
        if self.approach_connection_radius <= 0:
            raise ValueError("approach_connection_radius must be positive")
        if self.navigation_outer_limit <= 0:
            raise ValueError("navigation_outer_limit must be positive")

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

        # Keep the offline planner synchronized with RoomCoordinateMapper's
        # centralized navigation geometry. Do not inherit stale 3.0/1.0 values
        # from older layout_rules.json files.
        self.passage_center = (
            5.0 if self.passage_center is None else float(self.passage_center)
        )
        self.passage_width = (
            1.2 if self.passage_width is None else float(self.passage_width)
        )
        if self.passage_width <= 0:
            raise ValueError("passage_width must be positive")
        if self.navigation_outer_limit > self.half_extent:
            raise ValueError(
                "navigation_outer_limit cannot exceed the map half extent"
            )

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

        # Minimum allowed center-to-center distance between the robot and an obstacle.
        # This is a complete clearance value, so robot radius and other margins are
        # not added to it in obstacle collision checks.
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

    def _room_center_for_point(self, x: float, y: float) -> Optional[Point3]:
        """Return the active room containing a point, choosing the nearest on boundaries."""
        half = 0.5 * self.subroom_size
        eps = 1e-9
        candidates: List[Point3] = []
        for center in self.active_room_centers:
            cx, cy, _ = center
            if (
                cx - half - eps <= x <= cx + half + eps
                and cy - half - eps <= y <= cy + half + eps
            ):
                candidates.append(center)
        if not candidates:
            return None
        return min(
            candidates,
            key=lambda c: (x - c[0]) ** 2 + (y - c[1]) ** 2,
        )

    def _point_in_active_room(self, x: float, y: float) -> bool:
        return self._room_center_for_point(x, y) is not None

    def _local_to_full(self, local: Point3, center: Point3) -> Point3:
        return local[0] + center[0], local[1] + center[1], local[2] + center[2]

    def _forbidden_near_inner_wall(self, point: Point3) -> bool:
        near_x = abs(point[0]) <= self.inner_wall_object_clearance
        near_y = abs(point[1]) <= self.inner_wall_object_clearance
        if self.allow_inner_corner_cells and near_x and near_y:
            return False
        return near_x or near_y

    def _local_point_is_in_wall_obstacle_band(self, local: Point3) -> bool:
        """Return whether an obstacle belongs to the wall-side 1.5 m band."""
        half = 0.5 * self.subroom_size
        return (
            half - abs(local[0]) <= self.wall_obstacle_depth + 1e-9
            or half - abs(local[1]) <= self.wall_obstacle_depth + 1e-9
        )

    def _expand_fixed_obstacles(self) -> List[Point3]:
        out: List[Point3] = []
        for center in self.active_room_centers:
            for local in self.local_obstacle_grid:
                # Wall-side objects are represented by the continuous wall band.
                # Keeping their center circles as well would double-inflate them
                # and can close door lanes that runtime placement leaves clear.
                if self._local_point_is_in_wall_obstacle_band(local):
                    continue
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
    def inner_wall_center_clearance(self) -> float:
        # Runtime navigation excludes the fixed +/-0.5 m inner-wall strip.
        # Inflate it for robot footprint and path-tracking error.
        return 0.5 + self.footprint_clearance

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
        c = self.inner_wall_center_clearance
        vertical_blocked = abs(x) <= c and not self._in_vertical_opening(y)
        horizontal_blocked = abs(y) <= c and not self._in_horizontal_opening(x)
        return vertical_blocked or horizontal_blocked

    def _point_collides_with_obstacle(self, x: float, y: float) -> bool:
        clearance2 = self.obstacle_center_clearance * self.obstacle_center_clearance
        for ox, oy, _ in self.obstacles:
            if (x - ox) ** 2 + (y - oy) ** 2 < clearance2:
                return True
        return False

    def _point_collides_with_wall_obstacle_strip(self, x: float, y: float) -> bool:
        """Model furniture occupying 1.5 m inward from room walls.

        The robot-center exclusion depth additionally includes footprint and
        tracking clearance. Door access lanes through the two internal walls
        remain open. Target-specific final corridors may explicitly bypass this
        rule, but never physical walls or explicit obstacle circles.
        """
        center = self._room_center_for_point(x, y)
        if center is None:
            return False

        cx, cy, _ = center
        half = 0.5 * self.subroom_size
        local_x = x - cx
        local_y = y - cy
        strip = self.wall_obstacle_depth + self.footprint_clearance
        eps = 1e-9

        x_blocked = half - abs(local_x) <= strip + eps
        if x_blocked:
            wall_x = cx + (half if local_x >= 0.0 else -half)
            is_inner_x_wall = abs(wall_x) <= eps
            if is_inner_x_wall and self._in_vertical_opening(y):
                x_blocked = False

        y_blocked = half - abs(local_y) <= strip + eps
        if y_blocked:
            wall_y = cy + (half if local_y >= 0.0 else -half)
            is_inner_y_wall = abs(wall_y) <= eps
            if is_inner_y_wall and self._in_horizontal_opening(x):
                y_blocked = False

        return x_blocked or y_blocked

    def _point_is_valid_xy(
        self,
        x: float,
        y: float,
        *,
        allow_wall_obstacle_strip: bool = False,
    ) -> bool:
        if not (
            -self.navigation_outer_limit <= x <= self.navigation_outer_limit
            and -self.navigation_outer_limit <= y <= self.navigation_outer_limit
        ):
            return False
        if not self._point_in_active_room(x, y):
            return False
        if self._point_collides_with_wall(x, y):
            return False
        if self._point_collides_with_obstacle(x, y):
            return False
        if (
            not allow_wall_obstacle_strip
            and self._point_collides_with_wall_obstacle_strip(x, y)
        ):
            return False
        return True

    def _point_is_valid(self, node: GridNode) -> bool:
        x, y = self.grid_to_real(node)
        return self._point_is_valid_xy(x, y)

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
        clearance2 = self.obstacle_center_clearance * self.obstacle_center_clearance
        for ox, oy, _ in self.obstacles:
            if self._point_segment_distance_sq(ox, oy, a[0], a[1], b[0], b[1]) < clearance2:
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
        c = self.inner_wall_center_clearance
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

    def _segment_real_is_valid(
        self,
        a: Tuple[float, float],
        b: Tuple[float, float],
        *,
        allow_wall_obstacle_strip: bool = False,
    ) -> bool:
        if self._segment_hits_inner_wall(a, b):
            return False
        if self._segment_hits_obstacle(a, b):
            return False

        distance = math.hypot(b[0] - a[0], b[1] - a[1])
        sample_spacing = 0.5 / self.ratio
        steps = max(1, int(math.ceil(distance / sample_spacing)))
        for i in range(steps + 1):
            t = i / steps
            x = a[0] + t * (b[0] - a[0])
            y = a[1] + t * (b[1] - a[1])
            if not self._point_is_valid_xy(
                x,
                y,
                allow_wall_obstacle_strip=allow_wall_obstacle_strip,
            ):
                return False
        return True

    def _segment_is_valid(self, u: GridNode, v: GridNode) -> bool:
        return self._segment_real_is_valid(
            self.grid_to_real(u),
            self.grid_to_real(v),
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

    @staticmethod
    def _grid_line(a: GridNode, b: GridNode) -> List[GridNode]:
        """Return a dense 8-connected integer line including both endpoints."""
        dx = b[0] - a[0]
        dy = b[1] - a[1]
        steps = max(abs(dx), abs(dy))
        if steps == 0:
            return [a]

        out: List[GridNode] = []
        for i in range(steps + 1):
            t = i / steps
            node = (
                int(round(a[0] + t * dx)),
                int(round(a[1] + t * dy)),
            )
            if not out or node != out[-1]:
                out.append(node)
        return out

    def _build_target_approach_graph(
        self,
        base_graph: nx.Graph,
        goal_real: Point3,
    ) -> tuple[nx.Graph, TargetApproach]:
        """Add one pre-goal node exactly on the room-center-to-goal line.

        The base graph is not extended toward the room center. Instead, the
        pre-goal node is placed ``approach_distance`` metres from the physical
        goal toward the room center and connected only to nearby base-graph
        nodes with direct collision-free visibility. The final path segment is
        later appended directly from this pre-goal node to the goal node.
        """
        room_center = self._room_center_for_point(goal_real[0], goal_real[1])
        if room_center is None:
            raise RuntimeError(f"Goal {goal_real} is outside all active rooms")

        vx = room_center[0] - goal_real[0]
        vy = room_center[1] - goal_real[1]
        norm = math.hypot(vx, vy)
        if norm <= 1e-9:
            raise RuntimeError(
                f"Goal {goal_real} coincides with room center; "
                "a unique center-side approach direction does not exist"
            )

        inward_x = vx / norm
        inward_y = vy / norm
        approach_real = (
            goal_real[0] + inward_x * self.approach_distance,
            goal_real[1] + inward_y * self.approach_distance,
        )
        approach_node = self.real_to_grid(approach_real)
        approach_xy = self.grid_to_real(approach_node)
        goal_node = self.real_to_grid(goal_real)
        goal_xy = self.grid_to_real(goal_node)

        if approach_node == goal_node:
            raise RuntimeError(
                f"Approach point for goal {goal_real} quantizes to the goal node. "
                f"Increase approach_distance or ratio; current values are "
                f"{self.approach_distance:.3f} m and {self.ratio}."
            )

        if not self._point_is_valid_xy(
            approach_xy[0],
            approach_xy[1],
            allow_wall_obstacle_strip=True,
        ):
            raise RuntimeError(
                f"The requested approach point {approach_xy} for goal {goal_real} "
                f"is invalid. approach_distance={self.approach_distance:.2f} m."
            )

        if not self._segment_real_is_valid(
            approach_xy,
            goal_xy,
            allow_wall_obstacle_strip=True,
        ):
            raise RuntimeError(
                f"Direct final segment {approach_xy} -> {goal_xy} is not "
                f"collision-free for goal {goal_real}."
            )

        target_graph = base_graph.copy()
        target_graph.add_node(approach_node)

        radius_grid = self.approach_connection_radius * self.ratio
        radius_grid_sq = radius_grid * radius_grid
        connector_nodes: List[GridNode] = []

        for node in base_graph.nodes:
            dx = node[0] - approach_node[0]
            dy = node[1] - approach_node[1]
            distance_grid_sq = dx * dx + dy * dy
            if distance_grid_sq <= 1e-12 or distance_grid_sq > radius_grid_sq:
                continue

            node_xy = self.grid_to_real(node)
            if not self._segment_real_is_valid(
                node_xy,
                approach_xy,
                allow_wall_obstacle_strip=True,
            ):
                continue

            distance_grid = math.sqrt(distance_grid_sq)
            target_graph.add_edge(node, approach_node, weight=distance_grid)
            connector_nodes.append(node)

        # If the approach point already belongs to the base graph, retain its
        # normal graph neighbours in the debug information as well.
        if approach_node in base_graph:
            for node in base_graph.neighbors(approach_node):
                if node not in connector_nodes:
                    connector_nodes.append(node)

        if target_graph.degree(approach_node) == 0:
            raise RuntimeError(
                f"No base-graph node can connect to approach point {approach_xy} "
                f"within {self.approach_connection_radius:.2f} m for goal "
                f"{goal_real}. Increase --approach-connection-radius or inspect "
                "the obstacle geometry."
            )

        actual_distance = math.hypot(
            approach_xy[0] - goal_xy[0],
            approach_xy[1] - goal_xy[1],
        )
        return target_graph, TargetApproach(
            goal_real=goal_real,
            room_center=room_center,
            approach_node=approach_node,
            approach_distance_used=actual_distance,
            connector_nodes=tuple(sorted(set(connector_nodes))),
        )

    def _save_debug_graph(
        self,
        base_graph: nx.Graph,
        approach: TargetApproach,
        sample_path: Optional[List[GridNode]],
        index: int,
    ) -> None:
        import matplotlib.pyplot as plt
        from matplotlib.patches import Circle

        fig, ax = plt.subplots(figsize=(9, 9), dpi=160)
        ax.set_aspect("equal")

        nodes_real = [self.grid_to_real(n) for n in base_graph.nodes]
        ax.scatter([p[0] for p in nodes_real], [p[1] for p in nodes_real], s=2)

        for ox, oy, _ in self.obstacles:
            ax.add_patch(
                Circle(
                    (ox, oy),
                    self.obstacle_center_clearance,
                    fill=False,
                    linewidth=1.0,
                )
            )

        goal_node = self.real_to_grid(approach.goal_real)
        goal_xy = self.grid_to_real(goal_node)
        approach_xy = self.grid_to_real(approach.approach_node)
        ax.scatter([goal_xy[0]], [goal_xy[1]], s=65, marker="*")
        ax.scatter([approach_xy[0]], [approach_xy[1]], s=42, marker="o")

        if approach.connector_nodes:
            connector_real = [self.grid_to_real(n) for n in approach.connector_nodes]
            ax.scatter(
                [p[0] for p in connector_real],
                [p[1] for p in connector_real],
                s=18,
                marker="s",
            )
            for point in connector_real:
                ax.plot(
                    [point[0], approach_xy[0]],
                    [point[1], approach_xy[1]],
                    linewidth=0.6,
                    alpha=0.35,
                )

        # The required final segment: exactly approach -> physical goal.
        ax.plot(
            [approach_xy[0], goal_xy[0]],
            [approach_xy[1], goal_xy[1]],
            linewidth=2.8,
        )

        if sample_path:
            path_real = [self.grid_to_real(n) for n in sample_path]
            ax.plot(
                [p[0] for p in path_real],
                [p[1] for p in path_real],
                linewidth=1.5,
            )

        ax.axvline(0.0, linewidth=0.8)
        ax.axhline(0.0, linewidth=0.8)
        ax.set_xlim(
            self.room_bounds["x_min"] - 0.5,
            self.room_bounds["x_max"] + 0.5,
        )
        ax.set_ylim(
            self.room_bounds["y_min"] - 0.5,
            self.room_bounds["y_max"] + 0.5,
        )
        ax.set_title(
            "Fixed graph: goal star, pre-goal circle, connector nodes squares"
        )
        fig.tight_layout()
        fig.savefig(self.graphs_dir / f"fixed_graph_target_{index}.png")
        plt.close(fig)

    def generate(self, *, save_graph_images: bool = True) -> str:
        started = time.perf_counter()
        print(f"Generate start: {datetime.now().strftime('%H:%M:%S')}")
        print(f"Fixed obstacle points: {len(self.obstacles)}")
        print(
            "Wall obstacle strip: "
            f"depth={self.wall_obstacle_depth:.2f} m, "
            f"robot-center exclusion="
            f"{self.wall_obstacle_depth + self.footprint_clearance:.2f} m"
        )
        print(
            "Final approach: "
            f"pre-goal distance={self.approach_distance:.2f} m, "
            f"connection radius={self.approach_connection_radius:.2f} m"
        )
        print(f"Target positions: {self.targets_real}")

        base_graph = self._build_graph()
        print(
            f"Graph: {base_graph.number_of_nodes()} nodes, "
            f"{base_graph.number_of_edges()} edges"
        )

        result: PathMap = {}
        for target_index, goal_real in enumerate(self.targets_real):
            requested_target = self.real_to_grid(goal_real)
            target_graph, approach = self._build_target_approach_graph(
                base_graph,
                goal_real,
            )

            paths_from_approach = nx.single_source_dijkstra_path(
                target_graph,
                approach.approach_node,
                weight="weight",
            )
            # Runtime starts are ordinary navigation-graph nodes. The temporary
            # pre-goal node itself is not emitted as a start key.
            starts = [node for node in paths_from_approach if node in base_graph]
            if self.limit_start_nodes is not None:
                starts = starts[: self.limit_start_nodes]

            target_key = f"{requested_target[0]},{requested_target[1]}"
            if target_key in result:
                raise RuntimeError(
                    f"Multiple goals quantize to the same target key {target_key}"
                )

            start_map: Dict[str, List[GridNode]] = {}
            farthest_sample: Optional[List[GridNode]] = None
            farthest_length = -1

            for start in starts:
                # Dijkstra paths are approach -> start, so reverse them. The
                # resulting base path ends exactly at approach_node. Append only
                # the physical goal node: this creates one straight final segment.
                base_path = list(reversed(paths_from_approach[start]))
                raw_path = base_path
                if raw_path[-1] != requested_target:
                    raw_path = raw_path + [requested_target]

                start_map[f"{start[0]},{start[1]}"] = raw_path
                if len(raw_path) > farthest_length:
                    farthest_length = len(raw_path)
                    farthest_sample = raw_path

            result[target_key] = start_map
            approach_xy = self.grid_to_real(approach.approach_node)
            goal_xy = self.grid_to_real(requested_target)
            print(
                f"Target {target_index + 1}/{len(self.targets_real)} "
                f"goal={goal_xy}, approach={approach_xy}, "
                f"distance={approach.approach_distance_used:.2f} m, "
                f"connectors={len(approach.connector_nodes)}: "
                f"{len(start_map)} start nodes"
            )

            if save_graph_images:
                self._save_debug_graph(
                    base_graph,
                    approach,
                    farthest_sample,
                    target_index,
                )

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
        "--obstacle-center-clearance",
        type=float,
        default=0.8,
        help="Minimum center-to-center distance between robot and fixed obstacle",
    )
    parser.add_argument(
        "--wall-obstacle-depth",
        type=float,
        default=1.5,
        help="Depth occupied by wall-side obstacles, measured inward from each wall",
    )
    parser.add_argument(
        "--approach-distance",
        type=float,
        default=1.0,
        help=(
            "Distance from the physical goal to the mandatory pre-goal point, "
            "measured toward the room center"
        ),
    )
    parser.add_argument(
        "--approach-connection-radius",
        type=float,
        default=2.0,
        help=(
            "Maximum radius for connecting the pre-goal point to visible "
            "base-graph nodes"
        ),
    )
    parser.add_argument(
        "--passage-center",
        type=float,
        default=None,
        help="Override room_layout.passage_center",
    )
    parser.add_argument(
        "--passage-width",
        type=float,
        default=None,
        help="Override fixed 1.2 m passage width",
    )
    parser.add_argument(
        "--navigation-outer-limit",
        type=float,
        default=9.5,
        help="Maximum absolute x/y coordinate allowed by runtime navigation geometry",
    )
    parser.add_argument("--no-images", action="store_true")
    args = parser.parse_args()

    generator = FixedFourRoomPathGenerator(
        scene_items_path=args.scene_items,
        layout_rules_path=args.layout_rules,
        ratio=args.ratio,
        robot_radius=args.robot_radius,
        collision_margin=args.collision_margin,
        tracking_margin=args.tracking_margin,
        obstacle_center_clearance=args.obstacle_center_clearance,
        wall_obstacle_depth=args.wall_obstacle_depth,
        approach_distance=args.approach_distance,
        approach_connection_radius=args.approach_connection_radius,
        passage_center=args.passage_center,
        passage_width=args.passage_width,
        navigation_outer_limit=args.navigation_outer_limit,
        save_dir=args.save_dir,
    )
    generator.generate(save_graph_images=not args.no_images)


if __name__ == "__main__":
    main()