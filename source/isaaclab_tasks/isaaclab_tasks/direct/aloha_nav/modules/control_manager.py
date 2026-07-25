from __future__ import annotations

import math
from typing import Iterable

import torch


class VectorizedPurePursuit:
    """Vectorized Pure Pursuit over the complete path polyline.

    Algorithmic behavior is intentionally the same as the previously working
    controller:
      * the closest segment is searched over the complete valid path;
      * progress is monotonic in global path arclength;
      * lookahead may cross a vertex and lie on a later segment;
      * there are no start-turn or per-segment-turn states;
      * reaching the last path point sets ``finished=True``;
      * a finished environment only turns toward ``target_positions``.

    Set ``debug=True`` to print a complete controller log automatically from
    every ``compute_controls`` call. ``debug_every_n_steps`` controls cadence.
    """

    PATH_FOLLOWING = 0
    FINAL_ALIGNMENT = 1
    IDLE = 2

    STAGE_NAMES = {
        PATH_FOLLOWING: "PATH_FOLLOWING",
        FINAL_ALIGNMENT: "FINAL_ALIGNMENT",
        IDLE: "IDLE",
    }

    def __init__(
        self,
        num_envs: int,
        device: str = "cuda",
        max_path_length: int = 150,
        lookahead_distance: float = 0.35,
        base_linear_velocity: float = 1.0,
        max_angular_velocity: float = 1.8,
        arrival_threshold: float = 0.2,
        low_linear_velocity_threshold: float = 0.2,
        sharp_turn_angular_velocity: float = 2.8,
        final_alignment_threshold: float = 0.1,
        final_alignment_angular_velocity: float = 2.0,
        segment_length_epsilon: float = 1.0e-6,
        numeric_epsilon: float = 1.0e-8,
        invalid_position_limit: float = 100.0,
        max_next_point_distance: float = 100.0,
        debug: bool = False,
        debug_every_n_steps: int = 1,
        debug_env_indices: Iterable[int] | None = None,
        debug_precision: int = 3,
    ) -> None:
        self.num_envs = int(num_envs)
        self.device = torch.device(device)
        self.max_path_length = int(max_path_length)

        self.lookahead_distance = float(lookahead_distance)
        self.base_linear_velocity = float(base_linear_velocity)
        self.max_angular_velocity = float(max_angular_velocity)
        self.arrival_threshold = float(arrival_threshold)
        self.low_linear_velocity_threshold = float(low_linear_velocity_threshold)
        self.sharp_turn_angular_velocity = float(sharp_turn_angular_velocity)
        self.final_alignment_threshold = float(final_alignment_threshold)
        self.final_alignment_angular_velocity = float(
            final_alignment_angular_velocity
        )
        self.segment_length_epsilon = float(segment_length_epsilon)
        self.numeric_epsilon = float(numeric_epsilon)
        self.invalid_position_limit = float(invalid_position_limit)
        self.max_next_point_distance = float(max_next_point_distance)

        self.debug = bool(debug)
        self.debug_every_n_steps = int(debug_every_n_steps)
        self.debug_env_indices = (
            None
            if debug_env_indices is None
            else tuple(int(i) for i in debug_env_indices)
        )
        self.debug_precision = int(debug_precision)
        self._control_step = 0

        self._validate_config()

        self.paths = torch.full(
            (self.num_envs, self.max_path_length, 2),
            float("nan"),
            dtype=torch.float32,
            device=self.device,
        )
        self.path_lengths = torch.zeros(
            self.num_envs, dtype=torch.int64, device=self.device
        )
        self.finished = torch.ones(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        self.target_positions = torch.full(
            (self.num_envs, 2),
            float("nan"),
            dtype=torch.float32,
            device=self.device,
        )
        self.progress_arclen = torch.zeros(
            self.num_envs, dtype=torch.float32, device=self.device
        )

        # Diagnostic state from the last compute_controls call.
        self.last_stage = torch.full(
            (self.num_envs,),
            self.IDLE,
            dtype=torch.int8,
            device=self.device,
        )
        self.last_closest_segment = torch.full(
            (self.num_envs,), -1, dtype=torch.int64, device=self.device
        )
        self.last_closest_projection = torch.full(
            (self.num_envs,), float("nan"), dtype=torch.float32, device=self.device
        )
        self.last_closest_distance = torch.full(
            (self.num_envs,), float("nan"), dtype=torch.float32, device=self.device
        )
        self.last_closest_arclen = torch.full(
            (self.num_envs,), float("nan"), dtype=torch.float32, device=self.device
        )
        self.last_target_arclen = torch.full(
            (self.num_envs,), float("nan"), dtype=torch.float32, device=self.device
        )
        self.last_total_arclen = torch.full(
            (self.num_envs,), float("nan"), dtype=torch.float32, device=self.device
        )
        self.last_lookahead_segment = torch.full(
            (self.num_envs,), -1, dtype=torch.int64, device=self.device
        )
        self.last_lookahead_points = torch.full(
            (self.num_envs, 2),
            float("nan"),
            dtype=torch.float32,
            device=self.device,
        )
        self.last_alpha = torch.full(
            (self.num_envs,), float("nan"), dtype=torch.float32, device=self.device
        )
        self.last_distance_to_path_end = torch.full(
            (self.num_envs,), float("nan"), dtype=torch.float32, device=self.device
        )
        self.last_distance_to_target = torch.full(
            (self.num_envs,), float("nan"), dtype=torch.float32, device=self.device
        )
        self.last_linear_vels = torch.zeros(
            self.num_envs, dtype=torch.float32, device=self.device
        )
        self.last_angular_vels = torch.zeros_like(self.last_linear_vels)
        self.last_valid_robot = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update_paths(self, env_indices, new_paths, target_positions) -> None:
        env_indices = self._as_env_indices(env_indices)
        targets = torch.as_tensor(
            target_positions, dtype=torch.float32, device=self.device
        )
        if targets.shape != (env_indices.numel(), 2):
            raise ValueError(
                "target_positions must have shape "
                f"({env_indices.numel()}, 2), got {tuple(targets.shape)}"
            )
        if len(new_paths) != env_indices.numel():
            raise ValueError(
                f"new_paths must contain {env_indices.numel()} paths, "
                f"got {len(new_paths)}"
            )

        self.target_positions[env_indices] = targets

        for row, env_id_tensor in enumerate(env_indices):
            env_id = int(env_id_tensor.item())
            path = torch.as_tensor(
                new_paths[row], dtype=torch.float32, device=self.device
            )
            if path.ndim != 2 or path.shape[1] != 2:
                raise ValueError(
                    f"Path for env {env_id} must have shape (N, 2), "
                    f"got {tuple(path.shape)}"
                )

            finite_rows = torch.isfinite(path).all(dim=1)
            first_invalid = torch.where(~finite_rows)[0]
            length = (
                int(first_invalid[0].item())
                if first_invalid.numel() > 0
                else int(path.shape[0])
            )
            if length > self.max_path_length:
                raise ValueError(
                    f"Path length {length} exceeds max_path_length "
                    f"{self.max_path_length}"
                )

            self.paths[env_id].fill_(float("nan"))
            if length > 0:
                self.paths[env_id, :length] = path[:length]
            self.path_lengths[env_id] = length
            self.progress_arclen[env_id] = 0.0
            self.finished[env_id] = False
            self._reset_debug_state(torch.tensor([env_id], device=self.device))

    def compute_controls(self, positions, orientations):
        positions = torch.as_tensor(
            positions, dtype=torch.float32, device=self.device
        )
        orientations = torch.as_tensor(
            orientations, dtype=torch.float32, device=self.device
        ).flatten()
        self._validate_input_shapes(positions, orientations)

        linear_vels = torch.zeros(
            self.num_envs, dtype=torch.float32, device=self.device
        )
        angular_vels = torch.zeros_like(linear_vels)

        self._reset_step_debug_state()

        valid_robot = self._valid_robot_mask(positions, orientations)
        self.last_valid_robot.copy_(valid_robot)
        self._report_invalid_robot_state(
            positions, orientations, ~valid_robot
        )

        path_following = (
            (self.path_lengths >= 2)
            & (~self.finished)
            & valid_robot
        )
        final_alignment = (
            self.finished
            & torch.isfinite(self.target_positions).all(dim=1)
            & valid_robot
        )

        self.last_stage[path_following] = self.PATH_FOLLOWING
        self.last_stage[final_alignment] = self.FINAL_ALIGNMENT

        self._run_path_following(
            positions,
            orientations,
            path_following,
            linear_vels,
            angular_vels,
        )
        self._run_final_alignment(
            positions,
            orientations,
            valid_robot,
            linear_vels,
            angular_vels,
        )

        self._sanitize_outputs(linear_vels, angular_vels)
        self.last_linear_vels.copy_(linear_vels)
        self.last_angular_vels.copy_(angular_vels)

        self._control_step += 1
        if self.debug and self._control_step % self.debug_every_n_steps == 0:
            self.debug_print(
                positions=positions,
                orientations=orientations,
                env_indices=self.debug_env_indices,
                precision=self.debug_precision,
            )

        return linear_vels, angular_vels

    def debug_print(
        self,
        positions=None,
        orientations=None,
        env_indices: Iterable[int] | torch.Tensor | None = None,
        precision: int | None = None,
    ) -> None:
        """Print complete controller state.

        By default all environments are printed. The path is printed through
        the first NaN row, including that row.
        """
        precision = self.debug_precision if precision is None else int(precision)
        ids = (
            torch.arange(self.num_envs, device=self.device)
            if env_indices is None
            else self._as_env_indices(env_indices)
        )

        pos = (
            None
            if positions is None
            else torch.as_tensor(positions).detach().cpu()
        )
        yaw = (
            None
            if orientations is None
            else torch.as_tensor(orientations).flatten().detach().cpu()
        )

        paths = self.paths.detach().cpu()
        lengths = self.path_lengths.detach().cpu()
        targets = self.target_positions.detach().cpu()
        finished = self.finished.detach().cpu()
        progress = self.progress_arclen.detach().cpu()
        stage = self.last_stage.detach().cpu()
        closest_seg = self.last_closest_segment.detach().cpu()
        closest_proj = self.last_closest_projection.detach().cpu()
        closest_dist = self.last_closest_distance.detach().cpu()
        closest_s = self.last_closest_arclen.detach().cpu()
        target_s = self.last_target_arclen.detach().cpu()
        total_s = self.last_total_arclen.detach().cpu()
        look_seg = self.last_lookahead_segment.detach().cpu()
        look = self.last_lookahead_points.detach().cpu()
        alpha = self.last_alpha.detach().cpu()
        end_dist = self.last_distance_to_path_end.detach().cpu()
        target_dist = self.last_distance_to_target.detach().cpu()
        linear = self.last_linear_vels.detach().cpu()
        angular = self.last_angular_vels.detach().cpu()
        valid_robot = self.last_valid_robot.detach().cpu()

        def scalar(value: torch.Tensor) -> str:
            number = float(value.item())
            return (
                f"{number:.{precision}f}"
                if math.isfinite(number)
                else "nan"
            )

        def point(value: torch.Tensor) -> str:
            if value.numel() < 2 or not torch.isfinite(value[:2]).all():
                return "(nan, nan)"
            return (
                f"({value[0]:.{precision}f}, "
                f"{value[1]:.{precision}f})"
            )

        print(
            f"[PP DEBUG step={self._control_step}] "
            f"lookahead={self.lookahead_distance:.{precision}f} "
            f"base_v={self.base_linear_velocity:.{precision}f}"
        )

        for env_id_tensor in ids.detach().cpu():
            env_id = int(env_id_tensor.item())
            length = int(lengths[env_id].item())
            display_end = min(length + 1, self.max_path_length)
            shown_path = paths[env_id, :display_end]
            path_text = " -> ".join(point(p) for p in shown_path)
            if not path_text:
                path_text = "(nan, nan)"

            stage_name = self.STAGE_NAMES[int(stage[env_id].item())]
            pos_text = "-" if pos is None else point(pos[env_id, :2])
            yaw_text = "-" if yaw is None else scalar(yaw[env_id])

            closest_idx = int(closest_seg[env_id].item())
            closest_segment_text = self._debug_segment_text(
                paths, length, env_id, closest_idx, point
            )
            look_idx = int(look_seg[env_id].item())
            look_segment_text = self._debug_segment_text(
                paths, length, env_id, look_idx, point
            )

            print(
                f"[PP env={env_id:03d}] "
                f"stage={stage_name:<15} valid_robot={bool(valid_robot[env_id])} "
                f"finished={bool(finished[env_id])}\n"
                f"  robot: pos={pos_text} yaw={yaw_text}\n"
                f"  closest: seg={closest_segment_text} "
                f"proj={scalar(closest_proj[env_id])} "
                f"dist={scalar(closest_dist[env_id])} "
                f"closest_s={scalar(closest_s[env_id])}\n"
                f"  progress: s={scalar(progress[env_id])} "
                f"target_s={scalar(target_s[env_id])} "
                f"total_s={scalar(total_s[env_id])}\n"
                f"  lookahead: point={point(look[env_id])} "
                f"seg={look_segment_text} alpha={scalar(alpha[env_id])}\n"
                f"  finish: path_end_dist={scalar(end_dist[env_id])} "
                f"target={point(targets[env_id])} "
                f"target_dist={scalar(target_dist[env_id])}\n"
                f"  command: linear={scalar(linear[env_id])} "
                f"angular={scalar(angular[env_id])}\n"
                f"  path[{length}]: {path_text}"
            )

    # ------------------------------------------------------------------
    # Pure Pursuit pipeline
    # ------------------------------------------------------------------

    def _run_path_following(
        self,
        positions: torch.Tensor,
        orientations: torch.Tensor,
        active_mask: torch.Tensor,
        linear_vels: torch.Tensor,
        angular_vels: torch.Tensor,
    ) -> None:
        if not active_mask.any():
            return

        env_ids = torch.where(active_mask)[0]
        pos = positions[env_ids, :2]
        yaw = orientations[env_ids]
        paths = self.paths[env_ids]
        path_lengths = self.path_lengths[env_ids]

        geometry = self._build_path_geometry(paths, path_lengths)
        projection = self._project_onto_path(pos, geometry)

        valid_geometry = projection["valid"]
        if (~valid_geometry).any():
            bad_ids = env_ids[~valid_geometry]
            print(
                "[PURE PURSUIT ERROR] Cannot project robot onto path: "
                f"envs={bad_ids.tolist()}. Commands remain zero."
            )

        self.last_closest_segment[env_ids] = projection["segment"]
        self.last_closest_projection[env_ids] = projection["fraction"]
        self.last_closest_distance[env_ids] = projection["distance"]
        self.last_closest_arclen[env_ids] = projection["arclen"]

        progress = self._update_monotonic_progress(
            env_ids,
            projection["arclen"],
            valid_geometry,
        )

        target_arclen = progress + self.lookahead_distance
        total_arclen = geometry["total_arclen"]
        self.last_target_arclen[env_ids] = target_arclen
        self.last_total_arclen[env_ids] = total_arclen

        lookahead = self._find_lookahead_points(
            paths,
            path_lengths,
            geometry,
            target_arclen,
            valid_geometry,
        )
        self.last_lookahead_points[env_ids] = lookahead["points"]
        self.last_lookahead_segment[env_ids] = lookahead["segment"]

        commands = self._compute_pure_pursuit_commands(
            pos,
            yaw,
            lookahead["points"],
            lookahead["valid"],
        )
        self.last_alpha[env_ids] = commands["alpha"]

        linear_vels[env_ids] = commands["linear"]
        angular_vels[env_ids] = commands["angular"]

        self._mark_path_completion(
            env_ids,
            pos,
            paths,
            path_lengths,
            linear_vels,
            angular_vels,
        )

    def _build_path_geometry(
        self,
        paths: torch.Tensor,
        path_lengths: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        batch_size = paths.shape[0]
        max_segments = self.max_path_length - 1

        starts = paths[:, :-1, :]
        ends = paths[:, 1:, :]
        vectors = ends - starts
        lengths = torch.linalg.norm(vectors, dim=-1)

        segment_indices = torch.arange(
            max_segments, device=self.device
        ).unsqueeze(0).expand(batch_size, max_segments)
        valid = segment_indices < (path_lengths - 1).unsqueeze(1)
        valid &= torch.isfinite(starts).all(dim=-1)
        valid &= torch.isfinite(ends).all(dim=-1)
        valid &= torch.isfinite(lengths)
        valid &= lengths > self.segment_length_epsilon

        safe_starts = torch.where(
            valid.unsqueeze(-1), starts, torch.zeros_like(starts)
        )
        safe_vectors = torch.where(
            valid.unsqueeze(-1), vectors, torch.zeros_like(vectors)
        )
        safe_lengths = torch.where(valid, lengths, torch.zeros_like(lengths))

        cumulative = torch.cat(
            [
                torch.zeros(batch_size, 1, device=self.device),
                safe_lengths,
            ],
            dim=1,
        )
        cumulative = torch.cumsum(cumulative, dim=1)
        rows = torch.arange(batch_size, device=self.device)
        total_arclen = cumulative[
            rows,
            (path_lengths - 1).clamp(min=0, max=max_segments),
        ]

        return {
            "starts": safe_starts,
            "vectors": safe_vectors,
            "lengths": safe_lengths,
            "valid": valid,
            "cumulative": cumulative,
            "total_arclen": total_arclen,
        }

    def _project_onto_path(
        self,
        positions: torch.Tensor,
        geometry: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        starts = geometry["starts"]
        vectors = geometry["vectors"]
        lengths = geometry["lengths"]
        segment_valid = geometry["valid"]
        cumulative = geometry["cumulative"]

        to_starts = positions.unsqueeze(1) - starts
        raw_fraction = torch.sum(to_starts * vectors, dim=-1) / (
            lengths.square() + self.numeric_epsilon
        )
        fraction = raw_fraction.clamp(0.0, 1.0)
        closest_points = starts + vectors * fraction.unsqueeze(-1)
        distances = torch.linalg.norm(
            positions.unsqueeze(1) - closest_points,
            dim=-1,
        )
        distances = distances.masked_fill(~segment_valid, float("inf"))

        min_distance, segment = torch.min(distances, dim=1)
        rows = torch.arange(positions.shape[0], device=self.device)
        min_fraction = fraction[rows, segment]
        min_length = lengths[rows, segment]
        arclen = cumulative[rows, segment] + min_fraction * min_length

        valid = (
            segment_valid.any(dim=1)
            & torch.isfinite(min_distance)
            & torch.isfinite(arclen)
        )

        return {
            "segment": segment,
            "fraction": min_fraction,
            "distance": min_distance,
            "arclen": arclen,
            "valid": valid,
        }

    def _update_monotonic_progress(
        self,
        env_ids: torch.Tensor,
        closest_arclen: torch.Tensor,
        valid: torch.Tensor,
    ) -> torch.Tensor:
        previous = self.progress_arclen[env_ids].clone()
        previous_finite = torch.isfinite(previous)
        if (~previous_finite).any():
            bad_ids = env_ids[~previous_finite]
            print(
                "[PURE PURSUIT ERROR] Invalid progress_arclen: "
                f"envs={bad_ids.tolist()}. Progress reset to zero."
            )
            previous[~previous_finite] = 0.0
            self.progress_arclen[bad_ids] = 0.0

        usable = valid & previous_finite
        updated = previous.clone()
        updated[usable] = torch.maximum(
            previous[usable], closest_arclen[usable]
        )
        self.progress_arclen[env_ids[usable]] = updated[usable]
        return updated

    def _find_lookahead_points(
        self,
        paths: torch.Tensor,
        path_lengths: torch.Tensor,
        geometry: dict[str, torch.Tensor],
        target_arclen: torch.Tensor,
        valid_geometry: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        batch_size = paths.shape[0]
        max_segments = self.max_path_length - 1
        rows = torch.arange(batch_size, device=self.device)
        total_arclen = geometry["total_arclen"]
        cumulative = geometry["cumulative"]
        lengths = geometry["lengths"]
        starts = geometry["starts"]
        vectors = geometry["vectors"]

        last_point_indices = (path_lengths - 1).clamp(
            min=0, max=self.max_path_length - 1
        )
        last_points = paths[rows, last_point_indices]

        points = torch.full(
            (batch_size, 2),
            float("nan"),
            dtype=torch.float32,
            device=self.device,
        )
        segments = torch.full(
            (batch_size,), -1, dtype=torch.int64, device=self.device
        )

        beyond = valid_geometry & (target_arclen >= total_arclen)
        points[beyond] = last_points[beyond]
        segments[beyond] = (path_lengths[beyond] - 2).clamp(
            min=0, max=max_segments - 1
        )

        inside = valid_geometry & (~beyond)
        if inside.any():
            inside_target = target_arclen[inside]
            inside_cumulative = cumulative[inside]
            target_segments = torch.searchsorted(
                inside_cumulative,
                inside_target.unsqueeze(1),
                right=False,
            ).squeeze(1) - 1
            target_segments = target_segments.clamp(
                min=0, max=max_segments - 1
            )

            inside_rows = torch.arange(
                target_segments.shape[0], device=self.device
            )
            segment_start_s = inside_cumulative[
                inside_rows, target_segments
            ]
            segment_lengths = lengths[inside, target_segments]
            fractions = (
                inside_target - segment_start_s
            ) / (segment_lengths + self.numeric_epsilon)
            fractions = fractions.clamp(0.0, 1.0)

            points[inside] = (
                starts[inside, target_segments]
                + vectors[inside, target_segments]
                * fractions.unsqueeze(1)
            )
            segments[inside] = target_segments

        distances = torch.linalg.norm(points - paths[:, 0, :], dim=1)
        # The actual robot-to-lookahead distance is checked in the command
        # method. Here only finite geometry is asserted.
        valid = (
            valid_geometry
            & torch.isfinite(points).all(dim=1)
            & (segments >= 0)
            & torch.isfinite(distances)
        )

        return {
            "points": points,
            "segment": segments,
            "valid": valid,
        }

    def _compute_pure_pursuit_commands(
        self,
        positions: torch.Tensor,
        orientations: torch.Tensor,
        lookahead_points: torch.Tensor,
        valid_lookahead: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        batch_size = positions.shape[0]
        linear = torch.zeros(
            batch_size, dtype=torch.float32, device=self.device
        )
        angular = torch.zeros_like(linear)
        alpha_all = torch.full_like(linear, float("nan"))

        robot_to_lookahead = lookahead_points - positions
        point_distance = torch.linalg.norm(robot_to_lookahead, dim=1)
        valid = (
            valid_lookahead
            & torch.isfinite(point_distance)
            & (point_distance < self.max_next_point_distance)
        )

        if valid.any():
            rows = torch.where(valid)[0]
            vectors = robot_to_lookahead[rows]
            target_angles = torch.atan2(vectors[:, 1], vectors[:, 0])
            alpha = self._wrap_angle(target_angles - orientations[rows])
            alpha_all[rows] = alpha

            curvature = 2.0 * torch.sin(alpha) / (
                self.lookahead_distance + self.numeric_epsilon
            )
            angular_valid = curvature * self.base_linear_velocity
            angular_valid = angular_valid.clamp(
                -self.max_angular_velocity,
                self.max_angular_velocity,
            )
            linear_valid = self.base_linear_velocity * (
                1.0
                - angular_valid.abs()
                / (self.max_angular_velocity + self.numeric_epsilon)
            )
            linear_valid = linear_valid.clamp(min=0.0)

            low_linear = (
                linear_valid < self.low_linear_velocity_threshold
            )
            if low_linear.any():
                signs = torch.sign(angular_valid[low_linear])
                signs[signs == 0] = 1.0
                angular_valid[low_linear] = (
                    signs * self.sharp_turn_angular_velocity
                )

            linear[rows] = linear_valid
            angular[rows] = angular_valid

        return {
            "linear": linear,
            "angular": angular,
            "alpha": alpha_all,
        }

    def _mark_path_completion(
        self,
        env_ids: torch.Tensor,
        positions: torch.Tensor,
        paths: torch.Tensor,
        path_lengths: torch.Tensor,
        linear_vels: torch.Tensor,
        angular_vels: torch.Tensor,
    ) -> None:
        rows = torch.arange(env_ids.numel(), device=self.device)
        last_indices = (path_lengths - 1).clamp(
            min=0, max=self.max_path_length - 1
        )
        last_points = paths[rows, last_indices]
        distance_to_end = torch.linalg.norm(
            positions - last_points, dim=1
        )
        self.last_distance_to_path_end[env_ids] = distance_to_end

        reached = (
            torch.isfinite(last_points).all(dim=1)
            & torch.isfinite(distance_to_end)
            & (distance_to_end < self.arrival_threshold)
        )
        if reached.any():
            reached_ids = env_ids[reached]
            self.finished[reached_ids] = True
            self.last_stage[reached_ids] = self.FINAL_ALIGNMENT
            linear_vels[reached_ids] = 0.0
            angular_vels[reached_ids] = 0.0

    def _run_final_alignment(
        self,
        positions: torch.Tensor,
        orientations: torch.Tensor,
        valid_robot: torch.Tensor,
        linear_vels: torch.Tensor,
        angular_vels: torch.Tensor,
    ) -> None:
        mask = (
            self.finished
            & torch.isfinite(self.target_positions).all(dim=1)
            & valid_robot
        )
        ids = torch.where(mask)[0]
        if ids.numel() == 0:
            return

        vectors = self.target_positions[ids] - positions[ids, :2]
        distances = torch.linalg.norm(vectors, dim=1)
        desired = torch.atan2(vectors[:, 1], vectors[:, 0])
        alpha = self._wrap_angle(desired - orientations[ids])

        valid = (
            torch.isfinite(distances)
            & (distances < self.max_next_point_distance)
            & torch.isfinite(alpha)
        )
        self.last_distance_to_target[ids] = distances
        self.last_alpha[ids] = alpha
        self.last_stage[ids] = self.FINAL_ALIGNMENT

        commands = torch.zeros_like(alpha)
        signs = torch.sign(alpha[valid])
        signs[signs == 0] = 1.0
        commands[valid] = (
            signs * self.final_alignment_angular_velocity
        ).clamp(
            -self.max_angular_velocity,
            self.max_angular_velocity,
        )

        aligned = valid & (alpha.abs() < self.final_alignment_threshold)
        commands[aligned] = 0.0

        linear_vels[ids] = 0.0
        angular_vels[ids] = commands

    # ------------------------------------------------------------------
    # Validation and diagnostics
    # ------------------------------------------------------------------

    def _reset_step_debug_state(self) -> None:
        self.last_stage.fill_(self.IDLE)
        self.last_closest_segment.fill_(-1)
        self.last_closest_projection.fill_(float("nan"))
        self.last_closest_distance.fill_(float("nan"))
        self.last_closest_arclen.fill_(float("nan"))
        self.last_target_arclen.fill_(float("nan"))
        self.last_total_arclen.fill_(float("nan"))
        self.last_lookahead_segment.fill_(-1)
        self.last_lookahead_points.fill_(float("nan"))
        self.last_alpha.fill_(float("nan"))
        self.last_distance_to_path_end.fill_(float("nan"))
        self.last_distance_to_target.fill_(float("nan"))

    def _reset_debug_state(self, env_ids: torch.Tensor) -> None:
        self.last_stage[env_ids] = self.IDLE
        self.last_closest_segment[env_ids] = -1
        self.last_closest_projection[env_ids] = float("nan")
        self.last_closest_distance[env_ids] = float("nan")
        self.last_closest_arclen[env_ids] = float("nan")
        self.last_target_arclen[env_ids] = float("nan")
        self.last_total_arclen[env_ids] = float("nan")
        self.last_lookahead_segment[env_ids] = -1
        self.last_lookahead_points[env_ids] = float("nan")
        self.last_alpha[env_ids] = float("nan")
        self.last_distance_to_path_end[env_ids] = float("nan")
        self.last_distance_to_target[env_ids] = float("nan")
        self.last_linear_vels[env_ids] = 0.0
        self.last_angular_vels[env_ids] = 0.0

    @staticmethod
    def _debug_segment_text(
        paths: torch.Tensor,
        path_length: int,
        env_id: int,
        segment_idx: int,
        point_formatter,
    ) -> str:
        if segment_idx < 0 or segment_idx >= path_length - 1:
            return "-"
        return (
            f"{segment_idx}:"
            f"{point_formatter(paths[env_id, segment_idx])}→"
            f"{point_formatter(paths[env_id, segment_idx + 1])}"
        )

    def _valid_robot_mask(
        self,
        positions: torch.Tensor,
        orientations: torch.Tensor,
    ) -> torch.Tensor:
        return (
            torch.isfinite(positions[:, :2]).all(dim=1)
            & (
                positions[:, :2].abs() <= self.invalid_position_limit
            ).all(dim=1)
            & torch.isfinite(orientations)
        )

    def _report_invalid_robot_state(
        self,
        positions: torch.Tensor,
        orientations: torch.Tensor,
        invalid_mask: torch.Tensor,
    ) -> None:
        if not invalid_mask.any():
            return
        ids = torch.where(invalid_mask)[0]
        print(
            "[PURE PURSUIT ERROR] Invalid robot state: "
            f"envs={ids.tolist()}, "
            f"positions={positions[ids, :2].detach().cpu().tolist()}, "
            f"yaws={orientations[ids].detach().cpu().tolist()}. "
            "Commands are zero."
        )

    def _sanitize_outputs(
        self,
        linear_vels: torch.Tensor,
        angular_vels: torch.Tensor,
    ) -> None:
        invalid = (
            ~torch.isfinite(linear_vels)
            | ~torch.isfinite(angular_vels)
        )
        if invalid.any():
            ids = torch.where(invalid)[0]
            print(
                "[PURE PURSUIT ERROR] NaN/Inf command: "
                f"envs={ids.tolist()}. Commands replaced with zero."
            )
            linear_vels[invalid] = 0.0
            angular_vels[invalid] = 0.0
            self.progress_arclen[ids] = 0.0

    def _validate_input_shapes(
        self,
        positions: torch.Tensor,
        orientations: torch.Tensor,
    ) -> None:
        if (
            positions.ndim != 2
            or positions.shape[0] != self.num_envs
            or positions.shape[1] < 2
        ):
            raise ValueError(
                "positions must have shape (num_envs, >=2), got "
                f"{tuple(positions.shape)}"
            )
        if orientations.shape != (self.num_envs,):
            raise ValueError(
                "orientations must contain one yaw per environment, got "
                f"{tuple(orientations.shape)}"
            )

    def _validate_config(self) -> None:
        positive = {
            "num_envs": self.num_envs,
            "max_path_length": self.max_path_length,
            "lookahead_distance": self.lookahead_distance,
            "base_linear_velocity": self.base_linear_velocity,
            "max_angular_velocity": self.max_angular_velocity,
            "arrival_threshold": self.arrival_threshold,
            "sharp_turn_angular_velocity": self.sharp_turn_angular_velocity,
            "final_alignment_threshold": self.final_alignment_threshold,
            "final_alignment_angular_velocity": (
                self.final_alignment_angular_velocity
            ),
            "segment_length_epsilon": self.segment_length_epsilon,
            "numeric_epsilon": self.numeric_epsilon,
            "invalid_position_limit": self.invalid_position_limit,
            "max_next_point_distance": self.max_next_point_distance,
            "debug_every_n_steps": self.debug_every_n_steps,
        }
        for name, value in positive.items():
            if value <= 0:
                raise ValueError(f"{name} must be > 0, got {value}")

        if self.max_path_length < 2:
            raise ValueError("max_path_length must be >= 2")
        if self.low_linear_velocity_threshold < 0:
            raise ValueError(
                "low_linear_velocity_threshold must be >= 0"
            )
        if self.debug_precision < 0:
            raise ValueError("debug_precision must be >= 0")
        if self.debug_env_indices is not None:
            invalid = [
                i
                for i in self.debug_env_indices
                if i < 0 or i >= self.num_envs
            ]
            if invalid:
                raise ValueError(
                    f"debug_env_indices out of range: {invalid}"
                )

    def _as_env_indices(self, env_indices) -> torch.Tensor:
        if isinstance(env_indices, torch.Tensor):
            return env_indices.to(
                device=self.device, dtype=torch.int64
            ).flatten()
        return torch.as_tensor(
            env_indices, dtype=torch.int64, device=self.device
        ).flatten()

    @staticmethod
    def _wrap_angle(angle: torch.Tensor) -> torch.Tensor:
        return (angle + math.pi) % (2.0 * math.pi) - math.pi