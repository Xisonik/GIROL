from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Literal

import torch


CollisionMode = Literal["gt", "contact"]
ActionMode = Literal["velocity", "discrete"]


@dataclass(frozen=True)
class CollisionCheckResult:
    """Result of evaluating one vectorized action batch."""

    candidate_pos_w: torch.Tensor
    candidate_yaw: torch.Tensor
    object_collision: torch.Tensor
    inner_wall_collision: torch.Tensor
    out_of_bounds: torch.Tensor
    invalid_action: torch.Tensor
    valid_action: torch.Tensor
    translational_action: torch.Tensor
    done_action: torch.Tensor


class CollisionManager:
    """Unified contact-based and geometry-based collision handling.

    ``mode="gt"`` predicts the candidate state before the action is applied and
    rejects actions whose candidate state intersects an active object, an inner
    wall, or the allowed navigation boundary.

    ``mode="contact"`` does not reject actions in advance. The collision event
    is read from the Isaac Lab contact sensor after the physics step. Boundary
    violations remain a geometric check because they are not necessarily
    represented by a physical collider.
    """

    _MODE_ALIASES = {
        "gt": "gt",
        "ground_truth": "gt",
        "ground-truth": "gt",
        "geometry": "gt",
        "contact": "contact",
        "sensor": "contact",
    }

    def __init__(
        self,
        *,
        num_envs: int,
        device: str | torch.device,
        scene_manager,
        to_local: Callable[[torch.Tensor], torch.Tensor],
        mode: str = "gt",
        action_mode: ActionMode = "velocity",
        prediction_horizon_s: float = 1.0,
        agent_radius: float = 0.4,
        collision_margin: float = 0.03,
        contact_force_threshold: float = 0.05,
        passage_center: float = 3.0,
        passage_width: float = 1.0,
        exclude_goal: bool = True,
        contact_forces_getter: Callable[[], torch.Tensor | None] | None = None,
    ) -> None:
        normalized_mode = self._MODE_ALIASES.get(str(mode).strip().lower())
        if normalized_mode is None:
            raise ValueError(
                f"Unknown collision mode {mode!r}. Expected 'gt' or 'contact'."
            )
        if action_mode not in ("velocity", "discrete"):
            raise ValueError(
                f"Unknown action mode {action_mode!r}. "
                "Expected 'velocity' or 'discrete'."
            )
        if prediction_horizon_s <= 0.0:
            raise ValueError("prediction_horizon_s must be positive")
        if agent_radius <= 0.0:
            raise ValueError("agent_radius must be positive")
        if collision_margin < 0.0:
            raise ValueError("collision_margin must be non-negative")
        if contact_force_threshold < 0.0:
            raise ValueError("contact_force_threshold must be non-negative")
        if passage_width <= 0.0:
            raise ValueError("passage_width must be positive")

        self.num_envs = int(num_envs)
        self.device = torch.device(device)
        self.scene_manager = scene_manager
        self.to_local = to_local
        self.mode: CollisionMode = normalized_mode  # type: ignore[assignment]
        self.action_mode = action_mode
        self.prediction_horizon_s = float(prediction_horizon_s)
        self.agent_radius = float(agent_radius)
        self.collision_margin = float(collision_margin)
        self.contact_force_threshold = float(contact_force_threshold)
        self.passage_center = float(passage_center)
        self.passage_width = float(passage_width)
        self.exclude_goal = bool(exclude_goal)
        self.contact_forces_getter = contact_forces_getter

        shape = (self.num_envs,)
        self.object_collision_buf = torch.zeros(
            shape, dtype=torch.bool, device=self.device
        )
        self.inner_wall_collision_buf = torch.zeros_like(
            self.object_collision_buf
        )
        self.collision_buf = torch.zeros_like(self.object_collision_buf)
        self.out_of_bounds_buf = torch.zeros_like(self.object_collision_buf)
        self.invalid_action_buf = torch.zeros_like(self.object_collision_buf)
        self.contact_collision_buf = torch.zeros_like(self.object_collision_buf)
        self.action_evaluated_buf = torch.zeros_like(self.object_collision_buf)

    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        if env_ids is None:
            env_ids = torch.arange(
                self.num_envs, device=self.device, dtype=torch.long
            )
        else:
            env_ids = torch.as_tensor(
                env_ids, device=self.device, dtype=torch.long
            )
        self.object_collision_buf[env_ids] = False
        self.inner_wall_collision_buf[env_ids] = False
        self.collision_buf[env_ids] = False
        self.out_of_bounds_buf[env_ids] = False
        self.invalid_action_buf[env_ids] = False
        self.contact_collision_buf[env_ids] = False
        self.action_evaluated_buf[env_ids] = False

    @staticmethod
    def wrap_to_pi(angle: torch.Tensor) -> torch.Tensor:
        return torch.atan2(torch.sin(angle), torch.cos(angle))

    def evaluate_discrete(
        self,
        *,
        current_pos_w: torch.Tensor,
        current_yaw: torch.Tensor,
        actions: torch.Tensor,
        turn_angle_rad: float,
        forward_distance: float,
        turn_task: bool = False,
        left_action: int = 0,
        right_action: int = 1,
        forward_action: int = 2,
        done_action: int | None = None,
    ) -> CollisionCheckResult:
        """Build and evaluate Habitat-style discrete candidate poses."""
        actions = actions.to(device=self.device, dtype=torch.long).flatten()
        if actions.numel() != self.num_envs:
            raise ValueError(
                f"Expected {self.num_envs} discrete actions, got {actions.numel()}"
            )

        is_left = actions == int(left_action)
        is_right = actions == int(right_action)
        is_forward = actions == int(forward_action)
        is_done = (
            actions == int(done_action)
            if done_action is not None
            else torch.zeros_like(actions, dtype=torch.bool)
        )

        candidate_pos_w = current_pos_w.clone()
        candidate_yaw = current_yaw.clone()
        candidate_yaw[is_left] = self.wrap_to_pi(
            candidate_yaw[is_left] + float(turn_angle_rad)
        )
        candidate_yaw[is_right] = self.wrap_to_pi(
            candidate_yaw[is_right] - float(turn_angle_rad)
        )

        translational = is_forward & (~is_done) & (not turn_task)
        if translational.any():
            candidate_pos_w[translational, 0] += (
                float(forward_distance)
                * torch.cos(current_yaw[translational])
            )
            candidate_pos_w[translational, 1] += (
                float(forward_distance)
                * torch.sin(current_yaw[translational])
            )

        return self._evaluate_candidate(
            current_pos_w=current_pos_w,
            candidate_pos_w=candidate_pos_w,
            candidate_yaw=candidate_yaw,
            translational_action=translational,
            done_action=is_done,
        )

    def evaluate_velocity(
        self,
        *,
        current_pos_w: torch.Tensor,
        current_yaw: torch.Tensor,
        linear_speed: torch.Tensor,
        angular_speed: torch.Tensor,
        horizon_s: float | None = None,
        turn_task: bool = False,
    ) -> CollisionCheckResult:
        """Predict a differential-drive/unicycle state over a time horizon.

        Straight motion uses ``distance = linear_speed * horizon``. For non-zero
        angular speed, the exact constant-velocity circular-arc integration is
        used instead of moving along the initial heading only.
        """
        horizon = (
            self.prediction_horizon_s
            if horizon_s is None
            else float(horizon_s)
        )
        if horizon <= 0.0:
            raise ValueError("horizon_s must be positive")

        linear_speed = linear_speed.to(self.device).flatten()
        angular_speed = angular_speed.to(self.device).flatten()
        if linear_speed.numel() != self.num_envs:
            raise ValueError(
                f"Expected {self.num_envs} linear speeds, "
                f"got {linear_speed.numel()}"
            )
        if angular_speed.numel() != self.num_envs:
            raise ValueError(
                f"Expected {self.num_envs} angular speeds, "
                f"got {angular_speed.numel()}"
            )

        candidate_pos_w = current_pos_w.clone()
        delta_yaw = angular_speed * horizon
        candidate_yaw = self.wrap_to_pi(current_yaw + delta_yaw)

        translational = linear_speed.abs() > 1.0e-8
        if turn_task:
            translational[:] = False

        if translational.any():
            v = linear_speed[translational]
            w = angular_speed[translational]
            yaw0 = current_yaw[translational]
            yaw1 = yaw0 + w * horizon
            straight = w.abs() <= 1.0e-6

            dx = torch.empty_like(v)
            dy = torch.empty_like(v)
            if straight.any():
                distance = v[straight] * horizon
                dx[straight] = distance * torch.cos(yaw0[straight])
                dy[straight] = distance * torch.sin(yaw0[straight])
            if (~straight).any():
                radius = v[~straight] / w[~straight]
                dx[~straight] = radius * (
                    torch.sin(yaw1[~straight]) - torch.sin(yaw0[~straight])
                )
                dy[~straight] = -radius * (
                    torch.cos(yaw1[~straight]) - torch.cos(yaw0[~straight])
                )

            candidate_pos_w[translational, 0] += dx
            candidate_pos_w[translational, 1] += dy

        no_done = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        return self._evaluate_candidate(
            current_pos_w=current_pos_w,
            candidate_pos_w=candidate_pos_w,
            candidate_yaw=candidate_yaw,
            translational_action=translational,
            done_action=no_done,
        )

    def _evaluate_candidate(
        self,
        *,
        current_pos_w: torch.Tensor,
        candidate_pos_w: torch.Tensor,
        candidate_yaw: torch.Tensor,
        translational_action: torch.Tensor,
        done_action: torch.Tensor,
    ) -> CollisionCheckResult:
        self.action_evaluated_buf[:] = True

        if self.mode == "contact":
            # Contact mode deliberately performs no predictive rejection.
            self.object_collision_buf[:] = False
            self.inner_wall_collision_buf[:] = False
            self.collision_buf[:] = False
            self.out_of_bounds_buf[:] = False
            self.invalid_action_buf[:] = False
        else:
            object_collision = self.check_object_collision(candidate_pos_w)
            inner_wall_collision = self.check_inner_wall_collision(
                current_pos_w=current_pos_w,
                candidate_pos_w=candidate_pos_w,
            ) & translational_action
            out_of_bounds = self.check_out_of_bounds(candidate_pos_w)
            invalid_action = (
                object_collision | inner_wall_collision | out_of_bounds
            ) & (~done_action)

            self.object_collision_buf[:] = object_collision
            self.inner_wall_collision_buf[:] = inner_wall_collision
            self.collision_buf[:] = object_collision | inner_wall_collision
            self.out_of_bounds_buf[:] = out_of_bounds
            self.invalid_action_buf[:] = invalid_action

        return CollisionCheckResult(
            candidate_pos_w=candidate_pos_w,
            candidate_yaw=candidate_yaw,
            object_collision=self.object_collision_buf.clone(),
            inner_wall_collision=self.inner_wall_collision_buf.clone(),
            out_of_bounds=self.out_of_bounds_buf.clone(),
            invalid_action=self.invalid_action_buf.clone(),
            valid_action=(~self.invalid_action_buf).clone(),
            translational_action=translational_action.clone(),
            done_action=done_action.clone(),
        )

    def check_object_collision(
        self, candidate_pos_w: torch.Tensor
    ) -> torch.Tensor:
        """2D circle-footprint check against active scene objects."""
        candidate_pos_l = self.to_local(candidate_pos_w)
        agent_xy = candidate_pos_l[:, None, :2]

        obj_xy = self.scene_manager.positions[:, :, :2]
        active = self.scene_manager.active.bool()
        obj_radii = self.scene_manager.radii.expand(self.num_envs, -1)

        if self.exclude_goal:
            env_ids = torch.arange(self.num_envs, device=self.device)
            goal_idxs = self.scene_manager.active_goal_indices.long().clamp(
                0, self.scene_manager.num_total_objects - 1
            )
            active = active.clone()
            active[env_ids, goal_idxs] = False

        distances = torch.linalg.norm(obj_xy - agent_xy, dim=-1)
        threshold = self.agent_radius + obj_radii + self.collision_margin
        return (active & (distances < threshold)).any(dim=1)

    def check_out_of_bounds(
        self, candidate_pos_w: torch.Tensor
    ) -> torch.Tensor:
        candidate_pos_l = self.to_local(candidate_pos_w)
        x = candidate_pos_l[:, 0]
        y = candidate_pos_l[:, 1]

        bounds = self.scene_manager.room_bounds
        clearance = self.agent_radius + self.collision_margin
        inside_outer = (
            (x >= bounds["x_min"] + clearance)
            & (x <= bounds["x_max"] - clearance)
            & (y >= bounds["y_min"] + clearance)
            & (y <= bounds["y_max"] - clearance)
        )
        inside_active = self.scene_manager.positions_in_active_navigation_area(
            candidate_pos_l
        )
        return ~(inside_outer & inside_active)

    def check_inner_wall_collision(
        self,
        *,
        current_pos_w: torch.Tensor,
        candidate_pos_w: torch.Tensor,
    ) -> torch.Tensor:
        """Swept-segment check against the two internal cross walls."""
        current_l = self.to_local(current_pos_w)[:, :2]
        candidate_l = self.to_local(candidate_pos_w)[:, :2]

        x0, y0 = current_l[:, 0], current_l[:, 1]
        x1, y1 = candidate_l[:, 0], candidate_l[:, 1]
        dx, dy = x1 - x0, y1 - y0

        clearance = self.agent_radius + self.collision_margin
        usable_half_passage = 0.5 * self.passage_width - clearance

        if usable_half_passage <= 0.0:
            vertical_open = torch.zeros_like(x0, dtype=torch.bool)
            horizontal_open = torch.zeros_like(x0, dtype=torch.bool)
        else:
            eps = torch.finfo(current_l.dtype).eps

            safe_dx = torch.where(dx.abs() > eps, dx, torch.ones_like(dx))
            tx = torch.clamp(-x0 / safe_dx, 0.0, 1.0)
            tx = torch.where(dx.abs() > eps, tx, torch.zeros_like(tx))
            y_at_vertical_wall = y0 + tx * dy

            upper_vertical_open = (
                torch.abs(y_at_vertical_wall - self.passage_center)
                <= usable_half_passage
            )
            lower_vertical_open = (
                torch.abs(y_at_vertical_wall + self.passage_center)
                <= usable_half_passage
            )
            if not self.scene_manager.room_mapper.vertical_passage_open(True):
                upper_vertical_open &= False
            if not self.scene_manager.room_mapper.vertical_passage_open(False):
                lower_vertical_open &= False
            vertical_open = upper_vertical_open | lower_vertical_open

            safe_dy = torch.where(dy.abs() > eps, dy, torch.ones_like(dy))
            ty = torch.clamp(-y0 / safe_dy, 0.0, 1.0)
            ty = torch.where(dy.abs() > eps, ty, torch.zeros_like(ty))
            x_at_horizontal_wall = x0 + ty * dx

            right_horizontal_open = (
                torch.abs(x_at_horizontal_wall - self.passage_center)
                <= usable_half_passage
            )
            left_horizontal_open = (
                torch.abs(x_at_horizontal_wall + self.passage_center)
                <= usable_half_passage
            )
            if not self.scene_manager.room_mapper.horizontal_passage_open(True):
                right_horizontal_open &= False
            if not self.scene_manager.room_mapper.horizontal_passage_open(False):
                left_horizontal_open &= False
            horizontal_open = right_horizontal_open | left_horizontal_open

        touches_vertical_wall = (
            torch.minimum(x0, x1) <= clearance
        ) & (torch.maximum(x0, x1) >= -clearance)
        touches_horizontal_wall = (
            torch.minimum(y0, y1) <= clearance
        ) & (torch.maximum(y0, y1) >= -clearance)

        vertical_collision = touches_vertical_wall & (~vertical_open)
        horizontal_collision = touches_horizontal_wall & (~horizontal_open)
        return vertical_collision | horizontal_collision

    def read_contact_collision(self) -> torch.Tensor:
        if self.contact_forces_getter is None:
            raise RuntimeError(
                "collision_mode='contact' requires a ContactSensor getter"
            )
        force_matrix = self.contact_forces_getter()
        if force_matrix is None or force_matrix.numel() == 0:
            self.contact_collision_buf[:] = False
            return self.contact_collision_buf.clone()

        force_matrix = force_matrix.to(self.device)
        horizontal_force = force_matrix[..., :2]
        force_norm = torch.linalg.norm(horizontal_force, dim=-1)
        if force_norm.ndim == 1:
            collision = force_norm > self.contact_force_threshold
        else:
            collision = (
                force_norm.reshape(self.num_envs, -1)
                > self.contact_force_threshold
            ).any(dim=1)
        self.contact_collision_buf[:] = collision
        return collision.clone()

    def get_collision_component(self) -> torch.Tensor:
        """Collision excluding navigation-boundary violations."""
        if self.mode == "contact":
            return self.read_contact_collision()
        return self.collision_buf.clone()

    def get_out_of_bounds(self, current_pos_w: torch.Tensor) -> torch.Tensor:
        if self.mode == "gt" and self.action_evaluated_buf.all():
            return self.out_of_bounds_buf.clone()
        return self.check_out_of_bounds(current_pos_w)

    def get_collision_event(self, current_pos_w: torch.Tensor) -> torch.Tensor:
        """Return the unified event used for penalty and termination."""
        if self.mode == "gt":
            return self.invalid_action_buf.clone()
        return self.read_contact_collision() | self.check_out_of_bounds(
            current_pos_w
        )
