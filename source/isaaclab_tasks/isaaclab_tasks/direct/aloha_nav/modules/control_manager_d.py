import torch
import math


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