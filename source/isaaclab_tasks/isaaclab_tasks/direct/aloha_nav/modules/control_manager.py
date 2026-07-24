import torch
import math

class VectorizedPurePursuit:
    PATH_FOLLOWING = 0
    FINAL_TURN = 1
    FINAL_DRIVE = 2
    COMPLETE = 3

    def __init__(self, num_envs, device='cuda', max_path_length=150, lookahead_distance=0.35,
                 base_linear_velocity=1.0, max_angular_velocity=2.8, arrival_threshold=0.2):
        self.num_envs = num_envs
        self.device = torch.device(device)
        self.max_path_length = max_path_length
        self.lookahead_distance = lookahead_distance
        self.base_linear_velocity = float(base_linear_velocity)
        self.max_angular_velocity = float(max_angular_velocity)
        self.arrival_threshold = float(arrival_threshold)

        # Final approach consists of two explicit phases:
        # 1) rotate toward target_positions; 2) drive directly to the target.
        self.final_turn_threshold = 0.10
        self.final_drive_realign_threshold = 0.35
        self.final_turn_gain = 2.0
        self.final_drive_turn_gain = 2.0

        # A robot position outside this range is considered corrupted.
        self.invalid_position_limit = 100.0
        # A computed steering point farther than this is considered invalid.
        self.max_next_point_distance = 100.0

        # paths: (num_envs, max_path_length, 2) padded with NaN
        self.paths = torch.full(
            (num_envs, max_path_length, 2),
            float('nan'),
            dtype=torch.float32,
            device=self.device,
        )
        self.path_lengths = torch.zeros(
            num_envs, dtype=torch.int64, device=self.device
        )
        self.finished = torch.ones(
            num_envs, dtype=torch.bool, device=self.device
        )
        self.target_positions = torch.full(
            (num_envs, 2),
            float('nan'),
            dtype=torch.float32,
            device=self.device,
        )

        # Progress along the path. It is never allowed to move backwards.
        self.progress_arclen = torch.zeros(
            num_envs, dtype=torch.float32, device=self.device
        )
        self.final_phase = torch.full(
            (num_envs,),
            self.COMPLETE,
            dtype=torch.int8,
            device=self.device,
        )

    def update_paths(self, env_indices, new_paths, target_positions):
        if not isinstance(env_indices, torch.Tensor):
            env_indices = torch.tensor(
                env_indices, dtype=torch.int64, device=self.device
            )
        else:
            env_indices = env_indices.to(
                device=self.device, dtype=torch.int64
            )

        self.target_positions[env_indices] = torch.as_tensor(
            target_positions,
            dtype=torch.float32,
            device=self.device,
        )

        for i, env_id in enumerate(env_indices):
            path = torch.as_tensor(
                new_paths[i],
                dtype=torch.float32,
                device=self.device,
            )
            length = int(path.shape[0])
            if length > self.max_path_length:
                raise ValueError(
                    f"Path length {length} exceeds max_path_length "
                    f"{self.max_path_length}"
                )

            self.paths[env_id].fill_(float('nan'))
            if length > 0:
                self.paths[env_id, :length] = path
            self.path_lengths[env_id] = length
            self.progress_arclen[env_id] = 0.0
            self.final_phase[env_id] = (
                self.PATH_FOLLOWING if length >= 2 else self.FINAL_TURN
            )

        self.finished[env_indices] = False

    def _report_invalid_robot_state(
        self,
        positions: torch.Tensor,
        orientations: torch.Tensor,
        invalid_mask: torch.Tensor,
    ) -> None:
        if not invalid_mask.any():
            return

        ids = torch.where(invalid_mask)[0]
        pos_values = positions[ids, :2].detach().cpu().tolist()
        yaw_values = orientations[ids].detach().cpu().tolist()
        has_nan = (
            torch.isnan(positions[ids, :2]).any()
            or torch.isnan(orientations[ids]).any()
        )

        print(
            "[PURE PURSUIT ERROR] Invalid robot state: "
            f"envs={ids.tolist()}, contains_nan={bool(has_nan)}, "
            f"positions={pos_values}, yaws={yaw_values}. "
            "Linear and angular speeds are set to 0."
        )

    def _apply_final_approach(
        self,
        positions: torch.Tensor,
        orientations: torch.Tensor,
        valid_robot: torch.Tensor,
        linear_vels: torch.Tensor,
        angular_vels: torch.Tensor,
    ) -> None:
        """Run the two-stage final approach: turn first, then drive."""
        in_final = (
            (self.final_phase == self.FINAL_TURN)
            | (self.final_phase == self.FINAL_DRIVE)
        )
        candidates = in_final & (~self.finished) & valid_robot
        if not candidates.any():
            return

        candidate_ids = torch.where(candidates)[0]
        targets = self.target_positions[candidate_ids]
        pos = positions[candidate_ids, :2]

        target_finite = torch.isfinite(targets).all(dim=1)
        target_distances = torch.linalg.norm(targets - pos, dim=1)
        target_distance_valid = (
            torch.isfinite(target_distances)
            & (target_distances < self.max_next_point_distance)
        )
        valid_target = target_finite & target_distance_valid

        invalid_target = ~valid_target
        if invalid_target.any():
            bad_ids = candidate_ids[invalid_target]
            bad_targets = targets[invalid_target].detach().cpu().tolist()
            bad_distances = target_distances[invalid_target].detach().cpu().tolist()
            has_nan = torch.isnan(targets[invalid_target]).any()
            print(
                "[PURE PURSUIT ERROR] Invalid final target: "
                f"envs={bad_ids.tolist()}, contains_nan={bool(has_nan)}, "
                f"points={bad_targets}, distances={bad_distances}. "
                "Linear and angular speeds are set to 0."
            )

        valid_ids = candidate_ids[valid_target]
        if valid_ids.numel() == 0:
            return

        pos_valid = positions[valid_ids, :2]
        ori_valid = orientations[valid_ids]
        target_valid = self.target_positions[valid_ids]
        to_targets = target_valid - pos_valid
        distances = torch.linalg.norm(to_targets, dim=1)
        target_angles = torch.atan2(to_targets[:, 1], to_targets[:, 0])
        alphas = (
            (target_angles - ori_valid + math.pi) % (2 * math.pi)
            - math.pi
        )

        # Reaching the physical target completes the controller.
        reached = distances <= self.arrival_threshold
        if reached.any():
            reached_ids = valid_ids[reached]
            self.final_phase[reached_ids] = self.COMPLETE
            self.finished[reached_ids] = True
            linear_vels[reached_ids] = 0.0
            angular_vels[reached_ids] = 0.0

        remaining = ~reached
        if not remaining.any():
            return

        ids = valid_ids[remaining]
        errors = alphas[remaining]
        distances_remaining = distances[remaining]
        phases = self.final_phase[ids]

        # Phase 1: rotate in place until the target is in front of the robot.
        turn_rows = phases == self.FINAL_TURN
        if turn_rows.any():
            turn_ids = ids[turn_rows]
            turn_errors = errors[turn_rows]
            turn_commands = torch.clamp(
                self.final_turn_gain * turn_errors,
                -self.max_angular_velocity,
                self.max_angular_velocity,
            )
            linear_vels[turn_ids] = 0.0
            angular_vels[turn_ids] = turn_commands

            aligned = turn_errors.abs() <= self.final_turn_threshold
            if aligned.any():
                aligned_ids = turn_ids[aligned]
                self.final_phase[aligned_ids] = self.FINAL_DRIVE
                angular_vels[aligned_ids] = 0.0

        # Phase 2: drive directly to the target with heading correction.
        # Re-read phases because some environments may have just transitioned.
        phases = self.final_phase[ids]
        drive_rows = phases == self.FINAL_DRIVE
        if drive_rows.any():
            drive_ids = ids[drive_rows]
            drive_errors = errors[drive_rows]
            drive_distances = distances_remaining[drive_rows]

            # If the robot deviates too far, return to the turning phase.
            needs_realign = (
                drive_errors.abs() > self.final_drive_realign_threshold
            )
            if needs_realign.any():
                realign_ids = drive_ids[needs_realign]
                realign_errors = drive_errors[needs_realign]
                self.final_phase[realign_ids] = self.FINAL_TURN
                linear_vels[realign_ids] = 0.0
                angular_vels[realign_ids] = torch.clamp(
                    self.final_turn_gain * realign_errors,
                    -self.max_angular_velocity,
                    self.max_angular_velocity,
                )

            drive_ok = ~needs_realign
            if drive_ok.any():
                move_ids = drive_ids[drive_ok]
                move_errors = drive_errors[drive_ok]
                move_distances = drive_distances[drive_ok]

                angular = torch.clamp(
                    self.final_drive_turn_gain * move_errors,
                    -self.max_angular_velocity,
                    self.max_angular_velocity,
                )
                heading_scale = torch.clamp(
                    1.0
                    - move_errors.abs()
                    / self.final_drive_realign_threshold,
                    min=0.0,
                    max=1.0,
                )
                distance_scale = torch.clamp(
                    move_distances
                    / max(self.arrival_threshold * 2.0, 1.0e-6),
                    min=0.15,
                    max=1.0,
                )
                linear = (
                    self.base_linear_velocity
                    * heading_scale
                    * distance_scale
                )

                linear_vels[move_ids] = linear
                angular_vels[move_ids] = angular

    def compute_controls(self, positions, orientations):
        positions = torch.as_tensor(
            positions, dtype=torch.float32, device=self.device
        )
        orientations = torch.as_tensor(
            orientations, dtype=torch.float32, device=self.device
        ).flatten()

        linear_vels = torch.zeros(
            self.num_envs, dtype=torch.float32, device=self.device
        )
        angular_vels = torch.zeros_like(linear_vels)

        # Robot input validation requested for the continuous controller.
        robot_position_finite = torch.isfinite(positions[:, :2]).all(dim=1)
        robot_position_in_range = (
            positions[:, :2].abs() <= self.invalid_position_limit
        ).all(dim=1)
        robot_yaw_finite = torch.isfinite(orientations)
        valid_robot = (
            robot_position_finite
            & robot_position_in_range
            & robot_yaw_finite
        )

        invalid_robot = ~valid_robot
        self._report_invalid_robot_state(
            positions,
            orientations,
            invalid_robot,
        )
        if invalid_robot.any():
            # Do not preserve a previously corrupted NaN progress value.
            self.progress_arclen[invalid_robot] = 0.0

        active = (
            (self.path_lengths >= 2)
            & (self.final_phase == self.PATH_FOLLOWING)
            & (~self.finished)
            & valid_robot
        )

        if not active.any():
            self._apply_final_approach(
                positions,
                orientations,
                valid_robot,
                linear_vels,
                angular_vels,
            )
            return linear_vels, angular_vels

        active_indices = torch.where(active)[0]
        num_active = active_indices.shape[0]
        pos = positions[active_indices, :2]
        ori = orientations[active_indices]
        paths_active = self.paths[active_indices]
        path_lens = self.path_lengths[active_indices]

        max_segments = self.max_path_length - 1
        segment_starts = paths_active[:, :-1, :]
        segment_ends = paths_active[:, 1:, :]
        segment_vecs = segment_ends - segment_starts
        segment_lengths = torch.linalg.norm(segment_vecs, dim=-1)

        seg_index = torch.arange(
            max_segments, device=self.device
        ).unsqueeze(0).expand(num_active, max_segments)
        segment_mask = seg_index < (path_lens - 1).unsqueeze(1)
        segment_mask &= torch.isfinite(segment_starts).all(dim=-1)
        segment_mask &= torch.isfinite(segment_ends).all(dim=-1)
        segment_mask &= torch.isfinite(segment_lengths)
        segment_mask &= segment_lengths > 1.0e-6

        # Replace invalid/padded values before arithmetic so NaN padding cannot
        # contaminate otherwise valid environments.
        safe_starts = torch.where(
            segment_mask.unsqueeze(-1),
            segment_starts,
            torch.zeros_like(segment_starts),
        )
        safe_vecs = torch.where(
            segment_mask.unsqueeze(-1),
            segment_vecs,
            torch.zeros_like(segment_vecs),
        )
        safe_lengths = torch.where(
            segment_mask,
            segment_lengths,
            torch.zeros_like(segment_lengths),
        )

        pos_exp = pos.unsqueeze(1)
        to_starts = pos_exp - safe_starts
        denom = safe_lengths.square() + 1.0e-8
        projs = torch.sum(to_starts * safe_vecs, dim=-1) / denom
        projs_clamped = torch.clamp(projs, min=0.0, max=1.0)
        closest_points = (
            safe_starts
            + safe_vecs * projs_clamped.unsqueeze(-1)
        )
        dists = torch.linalg.norm(pos_exp - closest_points, dim=-1)
        dists[~segment_mask] = float('inf')

        min_dists, min_segments = torch.min(dists, dim=1)
        row_ids = torch.arange(num_active, device=self.device)
        min_projs = projs_clamped[row_ids, min_segments]
        min_seg_lengths = safe_lengths[row_ids, min_segments]

        padded_segment_lengths = safe_lengths
        cum_lengths = torch.cat(
            [
                torch.zeros(num_active, 1, device=self.device),
                padded_segment_lengths,
            ],
            dim=1,
        )
        cum_lengths = torch.cumsum(cum_lengths, dim=1)

        cum_at_min_seg = cum_lengths[row_ids, min_segments]
        closest_arclen = (
            cum_at_min_seg + min_projs * min_seg_lengths
        )

        valid_path_geometry = (
            segment_mask.any(dim=1)
            & torch.isfinite(min_dists)
            & torch.isfinite(closest_arclen)
        )

        invalid_path_geometry = ~valid_path_geometry
        if invalid_path_geometry.any():
            bad_ids = active_indices[invalid_path_geometry]
            has_nan = torch.isnan(
                paths_active[invalid_path_geometry]
            ).any()
            print(
                "[PURE PURSUIT ERROR] Cannot calculate next point from path: "
                f"envs={bad_ids.tolist()}, contains_nan={bool(has_nan)}, "
                f"path_lengths={path_lens[invalid_path_geometry].tolist()}. "
                "Linear and angular speeds are set to 0."
            )

        prev_progress = self.progress_arclen[active_indices].clone()
        invalid_previous_progress = ~torch.isfinite(prev_progress)
        if invalid_previous_progress.any():
            bad_ids = active_indices[invalid_previous_progress]
            print(
                "[PURE PURSUIT ERROR] NaN/Inf in progress_arclen: "
                f"envs={bad_ids.tolist()}. Progress is reset and speeds "
                "are set to 0 for this step."
            )
            prev_progress[invalid_previous_progress] = 0.0
            self.progress_arclen[bad_ids] = 0.0
            valid_path_geometry &= ~invalid_previous_progress

        new_progress = prev_progress.clone()
        new_progress[valid_path_geometry] = torch.maximum(
            prev_progress[valid_path_geometry],
            closest_arclen[valid_path_geometry],
        )
        self.progress_arclen[active_indices[valid_path_geometry]] = (
            new_progress[valid_path_geometry]
        )

        target_arclen = new_progress + self.lookahead_distance
        total_lengths = cum_lengths[
            row_ids,
            (path_lens - 1).clamp(max=max_segments),
        ]

        lookahead_points = pos.clone()
        is_beyond = (
            valid_path_geometry
            & (target_arclen >= total_lengths)
        )

        last_point_indices = (
            path_lens - 1
        ).clamp(max=self.max_path_length - 1)
        last_points = paths_active[row_ids, last_point_indices]
        lookahead_points[is_beyond] = last_points[is_beyond]

        not_beyond = valid_path_geometry & (~is_beyond)
        if not_beyond.any():
            target_arclen_nb = target_arclen[not_beyond]
            cum_lengths_nb = cum_lengths[not_beyond]
            segs_nb = torch.searchsorted(
                cum_lengths_nb,
                target_arclen_nb.unsqueeze(1),
                right=False,
            ).squeeze(1) - 1
            segs_nb = torch.clamp(
                segs_nb,
                min=0,
                max=max_segments - 1,
            )

            nb_rows = torch.arange(
                segs_nb.shape[0], device=self.device
            )
            cum_at_seg_nb = cum_lengths_nb[nb_rows, segs_nb]
            seg_lengths_nb = padded_segment_lengths[
                not_beyond, segs_nb
            ]
            fracs_nb = (
                target_arclen_nb - cum_at_seg_nb
            ) / (seg_lengths_nb + 1.0e-8)
            fracs_nb = torch.clamp(fracs_nb, 0.0, 1.0)
            starts_nb = safe_starts[not_beyond, segs_nb]
            vecs_nb = safe_vecs[not_beyond, segs_nb]
            lookahead_points[not_beyond] = (
                starts_nb + vecs_nb * fracs_nb.unsqueeze(-1)
            )

        # Validate the actual point the controller is about to follow.
        next_point_finite = torch.isfinite(
            lookahead_points
        ).all(dim=1)
        next_point_distance = torch.linalg.norm(
            lookahead_points - pos,
            dim=1,
        )
        next_point_distance_valid = (
            torch.isfinite(next_point_distance)
            & (next_point_distance < self.max_next_point_distance)
        )
        valid_next_point = (
            valid_path_geometry
            & next_point_finite
            & next_point_distance_valid
        )

        invalid_next_point = valid_path_geometry & (~valid_next_point)
        if invalid_next_point.any():
            bad_ids = active_indices[invalid_next_point]
            bad_points = lookahead_points[
                invalid_next_point
            ].detach().cpu().tolist()
            bad_distances = next_point_distance[
                invalid_next_point
            ].detach().cpu().tolist()

            # A non-finite lookahead point normally means that the path has
            # reached NaN padding or contains corrupted coordinates. Stop path
            # following and reuse the existing final-alignment stage. The
            # _apply_final_approach below immediately starts the two-stage
            # final maneuver: rotate toward target_positions, then drive.
            nonfinite_next_point = (
                invalid_next_point & (~next_point_finite)
            )
            if nonfinite_next_point.any():
                alignment_ids = active_indices[nonfinite_next_point]
                self.final_phase[alignment_ids] = self.FINAL_TURN
                self.finished[alignment_ids] = False
                self.progress_arclen[alignment_ids] = 0.0

                print(
                    "[PURE PURSUIT WARNING] Non-finite next point: "
                    f"envs={alignment_ids.tolist()}. "
                    "Path following is stopped; switching to final turn "
                    "and direct drive toward target_positions."
                )

            # A finite point farther than the configured limit is not treated
            # as path completion. Keep zero speed for this step and report it.
            finite_but_far = invalid_next_point & next_point_finite
            if finite_but_far.any():
                far_ids = active_indices[finite_but_far]
                far_points = lookahead_points[
                    finite_but_far
                ].detach().cpu().tolist()
                far_distances = next_point_distance[
                    finite_but_far
                ].detach().cpu().tolist()

                print(
                    "[PURE PURSUIT ERROR] Next point is too far: "
                    f"envs={far_ids.tolist()}, points={far_points}, "
                    f"distances={far_distances}. Expected distance less "
                    "than 100 m. Linear and angular speeds are set to 0."
                )

        lin_vels_active = torch.zeros(
            num_active, dtype=torch.float32, device=self.device
        )
        ang_vels_active = torch.zeros_like(lin_vels_active)

        if valid_next_point.any():
            valid_rows = torch.where(valid_next_point)[0]
            to_targets = (
                lookahead_points[valid_rows] - pos[valid_rows]
            )
            target_angles = torch.atan2(
                to_targets[:, 1], to_targets[:, 0]
            )
            alphas = target_angles - ori[valid_rows]
            alphas = (
                (alphas + math.pi) % (2 * math.pi)
                - math.pi
            )
            curvatures = (
                2.0 * alphas
                / (self.lookahead_distance + 1.0e-8)
            )

            angular = curvatures * self.base_linear_velocity
            angular = torch.clamp(
                angular,
                -self.max_angular_velocity,
                self.max_angular_velocity,
            )
            linear = self.base_linear_velocity * (
                1.0
                - torch.abs(angular)
                / (self.max_angular_velocity + 1.0e-8)
            )
            linear = torch.clamp(linear, min=0.0)

            low_linear = linear < 0.2
            if low_linear.any():
                signs = torch.sign(angular[low_linear])
                signs[signs == 0] = 1
                angular[low_linear] = signs * 2.8

            lin_vels_active[valid_rows] = linear
            ang_vels_active[valid_rows] = angular

        # Mark valid active environments that reached the final path point.
        last_point_valid = torch.isfinite(last_points).all(dim=1)
        dists_to_end = torch.full(
            (num_active,),
            float('inf'),
            dtype=torch.float32,
            device=self.device,
        )
        end_check = valid_next_point & last_point_valid
        if end_check.any():
            dists_to_end[end_check] = torch.linalg.norm(
                pos[end_check] - last_points[end_check],
                dim=1,
            )

        finished_active = (
            end_check
            & torch.isfinite(dists_to_end)
            & (dists_to_end < self.arrival_threshold)
        )
        if finished_active.any():
            final_ids = active_indices[finished_active]
            self.final_phase[final_ids] = self.FINAL_TURN
            self.finished[final_ids] = False
            self.progress_arclen[final_ids] = 0.0
            lin_vels_active[finished_active] = 0.0
            ang_vels_active[finished_active] = 0.0

        linear_vels[active_indices] = lin_vels_active
        angular_vels[active_indices] = ang_vels_active

        self._apply_final_approach(
            positions,
            orientations,
            valid_robot,
            linear_vels,
            angular_vels,
        )

        # Final containment: a NaN produced anywhere in the controller never
        # reaches the wheel commands.
        invalid_output = (
            ~torch.isfinite(linear_vels)
            | ~torch.isfinite(angular_vels)
        )
        if invalid_output.any():
            bad_ids = torch.where(invalid_output)[0]
            print(
                "[PURE PURSUIT ERROR] Controller produced NaN/Inf speed: "
                f"envs={bad_ids.tolist()}, "
                f"linear={linear_vels[bad_ids].detach().cpu().tolist()}, "
                f"angular={angular_vels[bad_ids].detach().cpu().tolist()}. "
                "Speeds are replaced with 0."
            )
            linear_vels[invalid_output] = 0.0
            angular_vels[invalid_output] = 0.0
            self.progress_arclen[bad_ids] = 0.0

        return linear_vels, angular_vels


class VectorizedDiscretePathController:
    """Vectorized path follower for Habitat-style discrete navigation.

    Actions:
        0: turn left
        1: turn right
        2: move forward

    The controller advances through path waypoints. It moves forward when the
    heading error is within ``heading_threshold_deg``; otherwise it selects the
    fixed-angle turn that reduces the error.

    By default, the heading threshold is half of the discrete turn angle. This
    is the correct quantization boundary: above half a turn step, one turn is
    closer to the desired heading than moving without turning.
    """

    TURN_LEFT = 0
    TURN_RIGHT = 1
    MOVE_FORWARD = 2

    def __init__(
        self,
        num_envs: int,
        device: str = "cuda",
        max_path_length: int = 128,
        turn_angle_deg: float = 35.0,
        heading_threshold_deg: float | None = None,
        waypoint_threshold: float = 0.15,
        final_waypoint_threshold: float = 0.20,
        invalid_coordinate_limit: float = 100.0,
    ):
        self.num_envs = int(num_envs)
        self.device = torch.device(device)
        self.max_path_length = int(max_path_length)

        self.turn_angle_rad = math.radians(float(turn_angle_deg))
        if heading_threshold_deg is None:
            heading_threshold_deg = 0.5 * float(turn_angle_deg)
        self.heading_threshold_rad = math.radians(float(heading_threshold_deg))

        self.waypoint_threshold = float(waypoint_threshold)
        self.final_waypoint_threshold = float(final_waypoint_threshold)
        self.invalid_coordinate_limit = float(invalid_coordinate_limit)

        if self.max_path_length < 1:
            raise ValueError("max_path_length must be positive")
        if self.waypoint_threshold <= 0:
            raise ValueError("waypoint_threshold must be positive")
        if self.final_waypoint_threshold <= 0:
            raise ValueError("final_waypoint_threshold must be positive")

        self.paths = torch.full(
            (self.num_envs, self.max_path_length, 2),
            float("nan"),
            dtype=torch.float32,
            device=self.device,
        )
        self.path_lengths = torch.zeros(
            self.num_envs, dtype=torch.long, device=self.device
        )
        self.waypoint_indices = torch.zeros(
            self.num_envs, dtype=torch.long, device=self.device
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
        self.last_heading_error = torch.zeros(
            self.num_envs, dtype=torch.float32, device=self.device
        )

    @staticmethod
    def _wrap_to_pi(angle: torch.Tensor) -> torch.Tensor:
        return torch.atan2(torch.sin(angle), torch.cos(angle))

    def _sanitize_path(self, path: torch.Tensor) -> torch.Tensor:
        """Remove padding, invalid coordinates, and consecutive duplicates."""
        path = path.to(device=self.device, dtype=torch.float32).reshape(-1, 2)

        valid = torch.isfinite(path).all(dim=-1)
        valid &= (path.abs() <= self.invalid_coordinate_limit).all(dim=-1)
        path = path[valid]

        if path.shape[0] <= 1:
            return path

        delta = torch.linalg.norm(path[1:] - path[:-1], dim=-1)
        keep = torch.ones(path.shape[0], dtype=torch.bool, device=self.device)
        keep[1:] = delta > 1e-6
        return path[keep]

    @staticmethod
    def _validate_path_length(path: torch.Tensor, max_length: int) -> torch.Tensor:
        if path.shape[0] > max_length:
            raise ValueError(
                f"Dense path contains {path.shape[0]} nodes, "
                f"but max_path_length={max_length}. Increase max_path_length; "
                "resampling would create unsafe shortcuts."
            )
        return path

    @torch.no_grad()
    def update_paths(
        self,
        env_indices: torch.Tensor,
        new_paths,
        target_positions,
    ) -> None:
        env_indices = torch.as_tensor(
            env_indices, dtype=torch.long, device=self.device
        ).flatten()
        target_positions = torch.as_tensor(
            target_positions, dtype=torch.float32, device=self.device
        )

        if target_positions.shape != (env_indices.numel(), 2):
            raise ValueError(
                "target_positions must have shape "
                f"[{env_indices.numel()}, 2], got {tuple(target_positions.shape)}"
            )

        self.target_positions[env_indices] = target_positions

        if isinstance(new_paths, torch.Tensor):
            if new_paths.shape[0] != env_indices.numel():
                raise ValueError(
                    "new_paths first dimension must match env_indices: "
                    f"{new_paths.shape[0]} != {env_indices.numel()}"
                )
            path_rows = [new_paths[i] for i in range(new_paths.shape[0])]
        else:
            path_rows = list(new_paths)
            if len(path_rows) != env_indices.numel():
                raise ValueError(
                    "new_paths length must match env_indices: "
                    f"{len(path_rows)} != {env_indices.numel()}"
                )

        for row, env_id_tensor in enumerate(env_indices):
            env_id = int(env_id_tensor.item())
            path = torch.as_tensor(
                path_rows[row], dtype=torch.float32, device=self.device
            )
            path = self._sanitize_path(path)
            path = self._validate_path_length(path, self.max_path_length)

            self.paths[env_id].fill_(float("nan"))
            length = int(path.shape[0])
            if length > 0:
                self.paths[env_id, :length] = path

            self.path_lengths[env_id] = length
            self.waypoint_indices[env_id] = 0
            self.finished[env_id] = length == 0
            self.last_heading_error[env_id] = 0.0

    @torch.no_grad()
    def compute_actions(
        self,
        positions: torch.Tensor,
        orientations: torch.Tensor,
    ) -> torch.Tensor:
        """Return one discrete action for every environment.

        ``positions`` must be env-local XY coordinates with shape [num_envs, 2].
        ``orientations`` must be yaw angles in radians with shape [num_envs].
        """
        positions = torch.as_tensor(
            positions, dtype=torch.float32, device=self.device
        )
        orientations = torch.as_tensor(
            orientations, dtype=torch.float32, device=self.device
        ).flatten()

        if positions.shape[0] != self.num_envs or positions.shape[-1] < 2:
            raise ValueError(
                f"positions must have shape [{self.num_envs}, 2+], "
                f"got {tuple(positions.shape)}"
            )
        if orientations.shape[0] != self.num_envs:
            raise ValueError(
                f"orientations must have shape [{self.num_envs}], "
                f"got {tuple(orientations.shape)}"
            )

        positions = positions[:, :2]

        # There is no no-op in Discrete(3). TURN_LEFT is only a safe fallback;
        # normally actions from inactive envs are not copied into the environment.
        actions = torch.full(
            (self.num_envs,),
            self.TURN_LEFT,
            dtype=torch.long,
            device=self.device,
        )

        active = (self.path_lengths > 0) & (~self.finished)

        # Advance only after the robot has actually reached the dense waypoint.
        # The threshold must stay below the 0.25 m forward action. A threshold
        # of 0.30 m skipped the next graph node before the first movement.
        for _ in range(self.max_path_length):
            active_ids = torch.where(active)[0]
            if active_ids.numel() == 0:
                break

            indices = self.waypoint_indices[active_ids]
            points = self.paths[active_ids, indices]
            distances = torch.linalg.norm(
                points - positions[active_ids], dim=-1
            )
            last_indices = self.path_lengths[active_ids] - 1

            advance = (
                (distances <= self.waypoint_threshold)
                & (indices < last_indices)
            )
            if not advance.any():
                break

            self.waypoint_indices[active_ids[advance]] += 1

        active_ids = torch.where(active)[0]
        if active_ids.numel() > 0:
            indices = self.waypoint_indices[active_ids]
            points = self.paths[active_ids, indices]
            distances = torch.linalg.norm(
                points - positions[active_ids], dim=-1
            )
            last_indices = self.path_lengths[active_ids] - 1
            at_last = indices >= last_indices

            newly_finished = (
                at_last & (distances <= self.final_waypoint_threshold)
            )
            if newly_finished.any():
                self.finished[active_ids[newly_finished]] = True

        # Non-finished environments steer toward the current path waypoint.
        path_follow_ids = torch.where(
            (self.path_lengths > 0) & (~self.finished)
        )[0]
        if path_follow_ids.numel() > 0:
            waypoint_ids = self.waypoint_indices[path_follow_ids]
            steering_points = self.paths[path_follow_ids, waypoint_ids]
            self._write_steering_actions(
                actions,
                path_follow_ids,
                positions,
                orientations,
                steering_points,
            )

        # Once the last path waypoint is reached, rotate toward the actual goal.
        # The environment's goal_reached() terminates the episode as soon as the
        # robot is close enough and aligned, so no explicit stop action is needed.
        final_align_mask = (
            self.finished
            & torch.isfinite(self.target_positions).all(dim=-1)
        )
        final_align_ids = torch.where(final_align_mask)[0]
        if final_align_ids.numel() > 0:
            self._write_steering_actions(
                actions,
                final_align_ids,
                positions,
                orientations,
                self.target_positions[final_align_ids],
            )

        return actions

    def _write_steering_actions(
        self,
        actions: torch.Tensor,
        env_ids: torch.Tensor,
        positions: torch.Tensor,
        orientations: torch.Tensor,
        steering_points: torch.Tensor,
    ) -> None:
        delta = steering_points - positions[env_ids]
        desired_yaw = torch.atan2(delta[:, 1], delta[:, 0])
        error = self._wrap_to_pi(desired_yaw - orientations[env_ids])
        self.last_heading_error[env_ids] = error

        aligned = error.abs() <= self.heading_threshold_rad
        turn_left = error > self.heading_threshold_rad
        turn_right = error < -self.heading_threshold_rad

        actions[env_ids[aligned]] = self.MOVE_FORWARD
        actions[env_ids[turn_left]] = self.TURN_LEFT
        actions[env_ids[turn_right]] = self.TURN_RIGHT
