import torch
import math

class VectorizedPurePursuit:
    def __init__(self, num_envs, device='cuda', max_path_length=150, lookahead_distance=0.35,
                 base_linear_velocity=1.0, max_angular_velocity=1.8, arrival_threshold=0.2):
        self.num_envs = num_envs
        self.device = torch.device(device)
        self.max_path_length = max_path_length
        self.lookahead_distance = lookahead_distance
        self.base_linear_velocity = float(base_linear_velocity)
        self.max_angular_velocity = float(max_angular_velocity)
        self.arrival_threshold = float(arrival_threshold)

        # paths: (num_envs, max_path_length, 2) padded with NaN
        self.paths = torch.full((num_envs, max_path_length, 2),
                                float('nan'), dtype=torch.float32, device=self.device)
        self.path_lengths = torch.zeros(num_envs, dtype=torch.int64, device=self.device)
        self.finished = torch.ones(num_envs, dtype=torch.bool, device=self.device)
        self.target_positions = torch.full((num_envs, 2), float('nan'),
                                           dtype=torch.float32, device=self.device)

        # new: прогресс по арк-длине (не падает назад)
        self.progress_arclen = torch.zeros(num_envs, dtype=torch.float32, device=self.device)

    def update_paths(self, env_indices, new_paths, target_positions):
        if not isinstance(env_indices, torch.Tensor):
            env_indices = torch.tensor(env_indices, dtype=torch.int64, device=self.device)

        # set target positions (assume target_positions aligns with env_indices)
        self.target_positions[env_indices] = torch.tensor(target_positions, dtype=torch.float32, device=self.device)

        for i, env_id in enumerate(env_indices):
            path = new_paths[i]
            # accept numpy or torch
            if not isinstance(path, torch.Tensor):
                path = torch.tensor(path, dtype=torch.float32, device=self.device)
            length = int(path.shape[0])
            if length > self.max_path_length:
                raise ValueError(f"Path length {length} exceeds max_path_length {self.max_path_length}")
            self.paths[env_id, :length] = path
            # pad remainder with NaN
            if length < self.max_path_length:
                self.paths[env_id, length:] = float('nan')
            self.path_lengths[env_id] = length

            # reset progress counter for this env
            self.progress_arclen[env_id] = 0.0

        # mark these envs as not finished (start following)
        self.finished[env_indices] = False

    def compute_controls(self, positions, orientations):
        linear_vels = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
        angular_vels = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
        # print("pos rob: ", positions)
        # active: есть путь >=2 и не finished
        active = (self.path_lengths >= 2) & (~self.finished)
        if not active.any():
            # Нет активных движений — обрабатываем только выравнивание для пришедших (finished)
            # Но только для тех, у кого есть валидная target_positions
            # print("chachacha")
            valid_target_mask = self.finished & torch.isfinite(self.target_positions).all(dim=1)
            jf_indices = torch.where(valid_target_mask)[0]
            if jf_indices.numel() == 0:
                return linear_vels, angular_vels

            pos_jf = positions[jf_indices]
            ori_jf = orientations[jf_indices]
            target_pos_jf = self.target_positions[jf_indices]

            to_targets_jf = target_pos_jf - pos_jf
            target_angles_jf = torch.atan2(to_targets_jf[:, 1], to_targets_jf[:, 0]) # α_norm​ = (α+π) mod 2π − π
            alphas_jf = target_angles_jf - ori_jf 
            alphas_jf = (alphas_jf + math.pi) % (2 * math.pi) - math.pi # α=atan2(ytarget​−yrobot​, xtarget​−xrobot​)−θrobot
            signs = torch.sign(alphas_jf)
            signs[signs == 0] = 1
            
            ang_vels_jf = torch.clamp(signs * 2.0, -self.max_angular_velocity, self.max_angular_velocity)
            # P-controller for orientation
            # kp = 2.0
            # ang_vels_jf = torch.clamp(kp * alphas_jf, -self.max_angular_velocity, self.max_angular_velocity)

            linear_vels[jf_indices] = 0.0
            angular_vels[jf_indices] = ang_vels_jf

            return linear_vels, angular_vels

        # subset active
        active_indices = torch.where(active)[0]
        num_active = active_indices.shape[0]
        pos = positions[active_indices]  # (num_active, 2)
        ori = orientations[active_indices]  # (num_active,)
        paths_active = self.paths[active_indices]  # (num_active, max_path_length, 2)
        path_lens = self.path_lengths[active_indices]  # (num_active,)

        max_segments = self.max_path_length - 1
        segment_starts = paths_active[:, :-1, :]  # (num_active, max_segments, 2)
        segment_ends = paths_active[:, 1:, :]     # (num_active, max_segments, 2)
        segment_vecs = segment_ends - segment_starts
        segment_lengths = torch.norm(segment_vecs, dim=-1)  # (num_active, max_segments)

        # mask valid segments by path length and by non-zero length
        seg_index = torch.arange(max_segments, device=self.device).unsqueeze(0).expand(num_active, max_segments)
        segment_mask = seg_index < (path_lens - 1).unsqueeze(1)
        # also ignore extremely short segments (to avoid div by ~0)
        segment_mask = segment_mask & (segment_lengths > 1e-6)

        # compute projections safely
        pos_exp = pos.unsqueeze(1)  # (num_active, 1, 2)
        to_starts = pos_exp - segment_starts
        denom = (segment_lengths ** 2) + 1e-8
        projs = torch.sum(to_starts * segment_vecs, dim=-1) / denom # t=2(R−A)⋅(B−A)/∣∣B−A∣∣​, tclamp​=clamp(t,0,1)
        projs_clamped = torch.clamp(projs, min=0.0, max=1.0)
        closest_points = segment_starts + segment_vecs * projs_clamped.unsqueeze(-1) # C=A+tclamp​⋅(B−A)(ближайшая точка на сегменте)
        dists = torch.norm(pos_exp - closest_points, dim=-1) # d=∣∣R−C∣∣

        # invalidate non-valid segments
        dists[~segment_mask] = float('inf')

        # choose closest segment per env
        min_dists, min_segments = torch.min(dists, dim=1)  # (num_active,)
        min_projs = projs_clamped[torch.arange(num_active, device=self.device), min_segments]
        min_seg_lengths = segment_lengths[torch.arange(num_active, device=self.device), min_segments]

        # cumulative arclengths along path (start-of-segment positions)
        padded_segment_lengths = segment_lengths.clone()
        padded_segment_lengths[~segment_mask] = 0.0
        cum_lengths = torch.cat([torch.zeros(num_active, 1, device=self.device), padded_segment_lengths], dim=1)  # (num_active, max_segments+1)
        cum_lengths = torch.cumsum(cum_lengths, dim=1)

        cum_at_min_seg = cum_lengths[torch.arange(num_active, device=self.device), min_segments]
        closest_arclen = cum_at_min_seg + min_projs * min_seg_lengths  # arclength to the closest point on path

        # **Главная правка**: не позволяем прогрессу уменьшаться (чтобы не идти назад)
        prev_progress = self.progress_arclen[active_indices]
        new_progress = torch.maximum(prev_progress, closest_arclen)
        self.progress_arclen[active_indices] = new_progress
        curr_arclen = new_progress

        # цель по арк-длине
        target_arclen = curr_arclen + self.lookahead_distance

        # полная длина пути:
        total_lengths = cum_lengths[torch.arange(num_active, device=self.device), (path_lens - 1).clamp(max=max_segments)]

        is_beyond = target_arclen >= total_lengths

        lookahead_points = torch.zeros(num_active, 2, dtype=torch.float32, device=self.device)
        last_point_indices = (path_lens - 1).clamp(max=self.max_path_length - 1)
        last_points = paths_active[torch.arange(num_active, device=self.device), last_point_indices]
        lookahead_points[is_beyond] = last_points[is_beyond]
        

        not_beyond = ~is_beyond
        if not_beyond.any():
            nb_idx = torch.where(not_beyond)[0]
            target_arclen_nb = target_arclen[not_beyond]
            cum_lengths_nb = cum_lengths[not_beyond]  # (num_nb, max_segments+1)
            # find segment that contains target_arclen
            segs_nb = torch.searchsorted(cum_lengths_nb, target_arclen_nb.unsqueeze(1), right=False).squeeze(1) - 1
            segs_nb = torch.clamp(segs_nb, min=0, max=max_segments - 1)
            cum_at_seg_nb = cum_lengths_nb[torch.arange(segs_nb.shape[0], device=self.device), segs_nb]
            seg_lengths_nb = padded_segment_lengths[not_beyond, segs_nb]
            fracs_nb = (target_arclen_nb - cum_at_seg_nb) / (seg_lengths_nb + 1e-8)
            fracs_nb = torch.clamp(fracs_nb, min=0.0, max=1.0)
            starts_nb = segment_starts[not_beyond, segs_nb]
            vecs_nb = segment_vecs[not_beyond, segs_nb]
            lookahead_points[not_beyond] = starts_nb + vecs_nb * fracs_nb.unsqueeze(-1)
        # print("lookahead_points ", lookahead_points)
        # управление
        to_targets = lookahead_points - pos
        target_angles = torch.atan2(to_targets[:, 1], to_targets[:, 0])
        alphas = target_angles - ori
        alphas = (alphas + math.pi) % (2 * math.pi) - math.pi
        curvatures = 2 * alphas / (self.lookahead_distance + 1e-8)

        
        ang_vels_active = curvatures * self.base_linear_velocity
        # FIXED: Boost ang_vel на sharp turns (lin < 0.2 → full ang=2.8 с сохранением знака)
        ang_vels_active = torch.clamp(ang_vels_active, -self.max_angular_velocity, self.max_angular_velocity)
        lin_vels_active = self.base_linear_velocity * (1 - torch.abs(ang_vels_active) / (self.max_angular_velocity + 1e-8))
        lin_vels_active = torch.clamp(lin_vels_active, min=0.0)

        # NEW: Если lin < 0.2 — boost ang to ±2.8 (sign from original ang)
        low_lin_mask = lin_vels_active < 0.2
        if low_lin_mask.any():
            signs = torch.sign(ang_vels_active[low_lin_mask])  # Сохраняем знак (±1 или 0→1)
            signs[signs == 0] = 1  # Default positive если 0
            ang_vels_active[low_lin_mask] = signs * 2.8  # Boost to 2.8 rad/s
            # Optional: Recalc lin после boost (если нужно update slowdown)
            # ang_vels_active[low_lin_mask] = torch.clamp(ang_vels_active[low_lin_mask], -self.max_angular_velocity, self.max_angular_velocity)
            # lin_vels_active[low_lin_mask] = self.base_linear_velocity * (1 - torch.abs(ang_vels_active[low_lin_mask]) / (self.max_angular_velocity + 1e-8))

        
        # пометим те активные среды, которые достигли конца пути
        dists_to_end = torch.norm(pos - last_points, dim=1)
        finished_active = dists_to_end < self.arrival_threshold
        if finished_active.any():
            # объявляем их finished; дальше они будут обрабатываться в блоке finished
            self.finished[active_indices[finished_active]] = True
            lin_vels_active[finished_active] = 0.0
            ang_vels_active[finished_active] = 0.0

        # записываем результаты в глобальные векторы
        
        linear_vels[active_indices] = lin_vels_active
        angular_vels[active_indices] = ang_vels_active

        # блок выравнивания ориентации для всех finished (и имеющих валидные target_positions)
        valid_target_mask = self.finished & torch.isfinite(self.target_positions).all(dim=1)
        jf_indices = torch.where(valid_target_mask)[0]
        if jf_indices.numel() > 0:
            pos_jf = positions[jf_indices]
            ori_jf = orientations[jf_indices]
            target_pos_jf = self.target_positions[jf_indices]
            to_targets_jf = target_pos_jf - pos_jf
            target_angles_jf = torch.atan2(to_targets_jf[:, 1], to_targets_jf[:, 0])
            alphas_jf = target_angles_jf - ori_jf
            alphas_jf = (alphas_jf + math.pi) % (2 * math.pi) - math.pi
            kp = 2.0
            # ang_vels_jf = torch.clamp(kp * alphas_jf, -self.max_angular_velocity, self.max_angular_velocity)
            signs = torch.sign(alphas_jf)
            signs[signs == 0] = 1
            
            ang_vels_jf = torch.clamp(signs * 2.0, -self.max_angular_velocity, self.max_angular_velocity)
            linear_vels[jf_indices] = 0.0
            angular_vels[jf_indices] = ang_vels_jf

            # если уже выровнялись — обнулим угловую скорость
            aligned_mask = torch.abs(alphas_jf) < 0.1
            if aligned_mask.any():
                aligned_idxs = jf_indices[aligned_mask]
                angular_vels[aligned_idxs] = 0.0
                # (оставляем self.finished=True — внешняя логика может считать эту среду "done")


        to_targets_dbg = lookahead_points - pos                          # вектор до lookahead
        dist_to_lp = torch.norm(to_targets_dbg, dim=1)                  # расстояние до него

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
