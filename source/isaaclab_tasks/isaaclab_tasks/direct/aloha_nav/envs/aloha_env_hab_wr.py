from __future__ import annotations

import math
import gymnasium as gym
import torch
import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObject, RigidObjectCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import TiledCamera, TiledCameraCfg
from ..modules.asset_manager import AssetManager
from isaaclab.sim import SimulationCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from ..modules.env_config import load_env_config
from ..modules.control_manager import VectorizedDiscretePathController

from .aloha_env_base import (
    BaseWheeledRobotEnv,
    BaseWheeledRobotEnvCfg,
    WheeledRobotEnvWindow,
    num_total_objects,
)

import os
import json

_ENV_CFG = json.loads(os.environ.get("ALOHA_NAV_ENV_CFG", "{}"))

def env_get(name: str, default=None):
    return _ENV_CFG.get(name, default)

ACTION_ANGLE_DEG = int(env_get("action_angle_deg", 35))
ANGULAR_VALUES = ACTION_ANGLE_DEG

def yaw_to_quat_wxyz(yaw: torch.Tensor) -> torch.Tensor:
    """yaw [N] -> quaternion [N, 4] in Isaac/Isaac Lab convention (w, x, y, z)."""
    half = 0.5 * yaw
    quat = torch.zeros(yaw.shape[0], 4, device=yaw.device, dtype=torch.float32)
    quat[:, 0] = torch.cos(half)
    quat[:, 3] = torch.sin(half)
    return quat

def quat_wxyz_to_yaw(quat: torch.Tensor) -> torch.Tensor:
    """quaternion [N, 4] in (w, x, y, z) -> yaw [N] in radians."""
    w, x, y, z = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]
    return torch.atan2(
        2.0 * (w * z + x * y),
        1.0 - 2.0 * (y * y + z * z),
    )

def wrap_to_pi(angle: torch.Tensor) -> torch.Tensor:
    return torch.atan2(torch.sin(angle), torch.cos(angle))


@configclass
class WheeledRobotEnvCfg(BaseWheeledRobotEnvCfg):
    decimation = 1
    action_space = gym.spaces.Discrete(3)

    sim: SimulationCfg = SimulationCfg(
        dt=1/60,
        render_interval=decimation,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="min",
            restitution_combine_mode="min",
            static_friction=0.2,
            dynamic_friction=0.15,
            restitution=0.0,
        ),
    )
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        collision_group=-1,
        debug_vis=False,
    )
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=32, env_spacing=24, replicate_physics=True)

    agent: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/Agent",
        spawn=sim_utils.CuboidCfg(
            size=(0.25, 0.25, 0.25),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                rigid_body_enabled=True,
                kinematic_enabled=True,
                disable_gravity=True,
                max_linear_velocity=1000.0,
                max_angular_velocity=1000.0,
                max_depenetration_velocity=0.0,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.1, 0.4, 1.0),
                metallic=0.0,
                roughness=0.5,
            ),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.25),
            rot=(1.0, 0.0, 0.0, 0.0),
        ),
    )

    tiled_camera: TiledCameraCfg = TiledCameraCfg(
        prim_path="/World/envs/env_.*/Agent/Camera",
        offset=TiledCameraCfg.OffsetCfg(
            pos=(0.0, 0.0, 0.25),
            rot=(1.0, 0.0, 0.0, 0.0),
            convention="world",
        ),
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=35.0, focus_distance=2.0, horizontal_aperture=36, clipping_range=(0.2, 10.0)
        ),
        width=224,
        height=224,
    )


class WheeledRobotEnv(BaseWheeledRobotEnv):
    cfg: WheeledRobotEnvCfg

    def _init_actuator_handles(self) -> None:
        # Kinematic cube-agent has no wheel joints. Keep the rest of base __init__ reusable.
        self.agent_collision_radius = 0.18
        self.collision_margin = 0.03
        self.future_collision_buf = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.future_inner_wall_collision_buf = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        self.future_out_of_bounds_buf = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.future_invalid_action_buf = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        # Action 3 is a Habitat/AKGVP-style Done/Stop action.
        # It terminates the episode; success is still computed by goal_reached().
        self.done_action_buf = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.last_macro_actions = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)

    def _init_variant_after_actuators(self) -> None:
        """Replace the continuous expert with a discrete path follower."""
        if not self.use_controller:
            return

        max_path_length = int(
            getattr(self.path_manager, "max_path_length", 64)
        )
        self.control_module = VectorizedDiscretePathController(
            num_envs=self.num_envs,
            device=self.device,
            max_path_length=max_path_length,
            turn_angle_deg=float(ACTION_ANGLE_DEG),
            # Half a discrete turn is the natural quantization threshold.
            heading_threshold_deg=0.5 * float(ACTION_ANGLE_DEG),
            waypoint_threshold=0.15,
            final_waypoint_threshold=0.20,
        )

    def _reset_variant_state(self, env_ids: torch.Tensor) -> None:
        # RigidObject variant has no wheel joints/contact sensor. Clear only its own buffers.
        self.future_collision_buf[env_ids] = False
        self.future_inner_wall_collision_buf[env_ids] = False
        self.future_out_of_bounds_buf[env_ids] = False
        self.future_invalid_action_buf[env_ids] = False
        self.done_action_buf[env_ids] = False
        self.last_macro_actions[env_ids] = 0

    def _write_actor_state_to_sim(
        self,
        env_ids: torch.Tensor,
        robot_pos: torch.Tensor,
        robot_quats: torch.Tensor,
    ) -> None:
        # RigidObject cube-agent has no joints. Reset only root pose and root velocity.
        default_root_state = self._robot.data.default_root_state[env_ids].clone()
        default_root_state[:, :2] = self.to_global(robot_pos, env_ids)
        default_root_state[:, 2] = 0.25
        default_root_state[:, 3:7] = robot_quats
        default_root_state[:, 7:] = 0.0
        self._robot.write_root_pose_to_sim(default_root_state[:, :7], env_ids)
        self._robot.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids)

    def _write_actor_state_to_sim(
        self,
        env_ids: torch.Tensor,
        actor_pos_local: torch.Tensor,
        actor_quats: torch.Tensor,
    ) -> None:
        default_root_state = self._robot.data.default_root_state[env_ids].clone()
        default_root_state[:, :2] = self.to_global(actor_pos_local, env_ids)
        default_root_state[:, 2] = 0.25
        default_root_state[:, 3:7] = actor_quats
        default_root_state[:, 7:] = 0.0

        self._robot.write_root_pose_to_sim(default_root_state[:, :7], env_ids)
        self._robot.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids)

    def _setup_scene(self):
            # TOTHEHOLE: scene setup rewritten for kinematic RigidObject cube-agent.
            from omni.usd import get_context
            from pxr import UsdGeom
            from isaaclab.sim.spawners.from_files import spawn_from_usd

            self._robot = RigidObject(self.cfg.agent)  # TOTHEHOLE: keep name _robot for compatibility with old code.
            self.cfg.terrain.num_envs = self.scene.cfg.num_envs
            self.cfg.terrain.env_spacing = self.scene.cfg.env_spacing
            self._terrain = self.cfg.terrain.class_type(self.cfg.terrain)

            if self.CAMERA:
                self._tiled_camera = TiledCamera(self.cfg.tiled_camera)
                self.scene.sensors["tiled_camera"] = self._tiled_camera

            stage = get_context().get_stage()
            UsdGeom.Xform.Define(stage, "/World/envs/env_0/obstacles")

            spawn_from_usd(
                prim_path="/World/envs/env_0/obstacles/room",
                cfg=self.cfg.room,
                translation=(0.0, 0.0, 0.0),
                orientation=(0.0, 0.0, 0.0, 1.0),
            )

            self.asset_manager = AssetManager(config_path=self.config_path)
            prim_paths_env0, counts = self.asset_manager.spawn_assets_in_env0()

            self.scene.clone_environments(copy_from_source=False)

            # TOTHEHOLE: the agent is a rigid object. Keep key "robot" to minimize downstream changes.
            self.scene.rigid_objects["robot"] = self._robot

            self.scene_objects = {}
            for name, count in counts.items():
                for i in range(count):
                    if "/obstacles/" in prim_paths_env0[name][i]:
                        prim_path_view = f"/World/envs/env_.*/obstacles/{name}_{i}"
                    else:
                        prim_path_view = f"/World/envs/env_.*/{name}_{i}"

                    ro_view = RigidObject(RigidObjectCfg(prim_path=prim_path_view, spawn=None))
                    self.scene.rigid_objects[f"{name}_{i}"] = ro_view
                    self.scene_objects.setdefault(name, []).append(ro_view)

            # TOTHEHOLE: contact sensor is intentionally not created. Collision is handled by check_future_collision().
            self._contact_sensor = None

            light_cfg = sim_utils.DomeLightCfg(intensity=300.0, color=(0.75, 0.75, 0.75))
            light_cfg.func("/World/Light", light_cfg)

    def _pre_physics_step(self, actions: torch.Tensor):
            """
            TOTHEHOLE: Kinematic Habitat-style actions with pre-action collision/bounds check.

            actions:
                0 -> turn_left_30
                1 -> turn_right_30
                2 -> move_forward_25cm
                3 -> done / stop

            The candidate pose is checked BEFORE writing it to simulation:
                - future collision with active scene objects
                - future out-of-bounds
            If invalid, the pose is not moved for that env and future_invalid_action_buf
            terminates the episode in _get_dones().
            """
            input_actions = actions
            actions = actions.to(self.device)
            if actions.dim() > 1:
                actions = actions.squeeze(-1)
            actions = actions.long()

            # TOTHEHOLE: reset pre-action flags every env step.
            self.future_collision_buf[:] = False
            self.future_inner_wall_collision_buf[:] = False
            self.future_out_of_bounds_buf[:] = False
            self.future_invalid_action_buf[:] = False
            self.done_action_buf[:] = False

            root_pos_w = self._robot.data.root_pos_w.clone()
            root_quat_w = self._robot.data.root_quat_w.clone()

            base_quat = torch.tensor(
                [1.0, 0.0, 0.0, 0.0],
                device=self.device,
                dtype=root_quat_w.dtype,
            )
            quat_bad = torch.isnan(root_quat_w).any(dim=-1) | torch.isinf(root_quat_w).any(dim=-1)
            if quat_bad.any():
                root_quat_w[quat_bad] = base_quat

            yaw = quat_wxyz_to_yaw(root_quat_w)
            turn_angle = ANGULAR_VALUES * math.pi / 180.0
            forward_distance = 0.25

            # Replace policy actions only in environments assigned to the expert.
            controlled_mask = torch.zeros(
                self.num_envs, dtype=torch.bool, device=self.device
            )
            if self.imitation:
                controlled_mask[:] = True
            elif self.use_controller and self.controlled_env_ids:
                controlled_ids = torch.tensor(
                    sorted(self.controlled_env_ids),
                    dtype=torch.long,
                    device=self.device,
                )
                controlled_mask[controlled_ids] = True

            if self.use_controller and controlled_mask.any():
                robot_pos_local = self.to_local(root_pos_w)
                expert_actions = self.control_module.compute_actions(
                    positions=robot_pos_local,
                    orientations=yaw,
                )
                actions = actions.clone()
                actions[controlled_mask] = expert_actions[controlled_mask]

                # Match the continuous environment: replay/imitation must store
                # the action that was actually executed, not the policy proposal.
                executed_for_input = actions.to(
                    device=input_actions.device,
                    dtype=input_actions.dtype,
                )
                if input_actions.dim() > 1:
                    executed_for_input = executed_for_input.view_as(input_actions)
                input_actions.copy_(executed_for_input)

            self.last_macro_actions[:] = actions

            is_left = actions == 0
            is_right = actions == 1
            is_forward = actions == 2 # TODO: delete zeroslike
            is_done = torch.zeros_like(actions, dtype=torch.bool)
            self.done_action_buf[:] = is_done

            candidate_pos_w = root_pos_w.clone()
            candidate_yaw = yaw.clone()

            # 0 -> turn_left_30
            candidate_yaw[is_left] = wrap_to_pi(candidate_yaw[is_left] + turn_angle)

            # 1 -> turn_right_30
            candidate_yaw[is_right] = wrap_to_pi(candidate_yaw[is_right] - turn_angle)

            # 2 -> move_forward_25cm
            if is_forward.any() and not self.TURN_TASK:
                candidate_pos_w[is_forward, 0] += forward_distance * torch.cos(yaw[is_forward])
                candidate_pos_w[is_forward, 1] += forward_distance * torch.sin(yaw[is_forward])

            candidate_quat_w = yaw_to_quat_wxyz(candidate_yaw)

            # TOTHEHOLE: future checks happen before writing the candidate pose to sim.
            future_object_collision = self.check_future_collision(
                candidate_pos_w, exclude_goal=True
            )
            future_inner_wall_collision = self.check_future_inner_wall_collision(
                current_pos_w=root_pos_w,
                candidate_pos_w=candidate_pos_w,
            )
            # Turning in place does not cross a wall. Apply the swept-wall test only
            # to translational actions.
            if self.TURN_TASK:
                future_inner_wall_collision = torch.zeros_like(
                    future_inner_wall_collision,
                    dtype=torch.bool,
                )
            else:
                future_inner_wall_collision = (
                    future_inner_wall_collision & is_forward
                )

            future_oob = self.check_future_out_of_bounds(candidate_pos_w)
            future_collision = future_object_collision | future_inner_wall_collision

            # Done is a semantic termination action, not a motion candidate.
            # Do not mark it invalid even though the candidate pose equals the current pose.
            invalid_action = (future_collision | future_oob) & (~is_done)

            self.future_collision_buf[:] = future_collision
            self.future_inner_wall_collision_buf[:] = future_inner_wall_collision
            self.future_out_of_bounds_buf[:] = future_oob
            self.future_invalid_action_buf[:] = invalid_action

            valid_action = ~invalid_action

            # TOTHEHOLE: apply candidate pose only for valid envs. Invalid envs stay at their old pose.
            if valid_action.any():
                valid_env_ids = torch.where(valid_action)[0].to(dtype=torch.long)
                root_pose = torch.cat(
                    [candidate_pos_w[valid_action], candidate_quat_w[valid_action]],
                    dim=-1,
                )
                root_vel = torch.zeros(
                    valid_env_ids.numel(),
                    6,
                    device=self.device,
                    dtype=torch.float32,
                )
                self._robot.write_root_pose_to_sim(root_pose, env_ids=valid_env_ids)
                self._robot.write_root_velocity_to_sim(root_vel, env_ids=valid_env_ids)

            # TOTHEHOLE: memory still expects action_size=2; store semantic delta [forward_m, yaw_rad].
            linear_delta = torch.zeros(self.num_envs, device=self.device, dtype=torch.float32)
            yaw_delta = torch.zeros(self.num_envs, device=self.device, dtype=torch.float32)
            linear_delta[is_forward] = forward_distance
            yaw_delta[is_left] = turn_angle
            yaw_delta[is_right] = -turn_angle

            # TOTHEHOLE: invalid actions did not actually move the agent.
            linear_delta[invalid_action] = 0.0
            if self.TURN_TASK:
                linear_delta[:] = 0.0
            yaw_delta[invalid_action] = 0.0

            self.velocities = torch.stack([linear_delta, yaw_delta], dim=1)
            self._actions = self.velocities
            self.angular_speed = yaw_delta
            self.last_actions = self.velocities

    def _apply_action(self):
            # TOTHEHOLE: no wheel/joint command. Pose was already written in _pre_physics_step().
            pass

    def check_future_out_of_bounds(
            self, candidate_pos_w: torch.Tensor
        ) -> torch.Tensor:
            candidate_pos_l = self.to_local(candidate_pos_w)
            x = candidate_pos_l[:, 0]
            y = candidate_pos_l[:, 1]

            bounds = self.scene_manager.room_bounds
            margin = self.agent_collision_radius + self.collision_margin
            inside_outer = (
                (x >= bounds['x_min'] + margin)
                & (x <= bounds['x_max'] - margin)
                & (y >= bounds['y_min'] + margin)
                & (y <= bounds['y_max'] - margin)
            )
            inside_active = (
                self.scene_manager.positions_in_active_navigation_area(
                    candidate_pos_l
                )
            )
            return ~(inside_outer & inside_active)

    def check_future_inner_wall_collision(
            self,
            current_pos_w: torch.Tensor,
            candidate_pos_w: torch.Tensor,
            *,
            passage_center: float = 3.0,
            passage_width: float = 1.0,
        ) -> torch.Tensor:
            """
            Check a swept movement segment against the two internal cross walls.

            Geometry in env-local coordinates:
                vertical wall:   x = 0
                horizontal wall: y = 0

            Openings:
                x = 0 wall: openings centered at y = -3 and y = +3
                y = 0 wall: openings centered at x = -3 and x = +3

            Each opening has full width ``passage_width``. The usable opening is
            reduced by the agent collision radius and collision margin, so the
            whole circular agent footprint must fit through the passage.

            The full segment from the current position to the candidate position
            is checked. Therefore, increasing the forward step cannot make the
            agent jump through a wall.

            Returns:
                Bool tensor [num_envs]. True means the candidate movement touches
                an internal wall outside a valid opening.
            """
            current_l = self.to_local(current_pos_w)[:, :2]
            candidate_l = self.to_local(candidate_pos_w)[:, :2]

            x0 = current_l[:, 0]
            y0 = current_l[:, 1]
            x1 = candidate_l[:, 0]
            y1 = candidate_l[:, 1]

            dx = x1 - x0
            dy = y1 - y0

            clearance = float(self.agent_collision_radius + self.collision_margin)
            half_passage = 0.5 * float(passage_width)
            usable_half_passage = half_passage - clearance

            # If the footprint is wider than the opening, no passage is usable.
            if usable_half_passage <= 0.0:
                vertical_open = torch.zeros_like(x0, dtype=torch.bool)
                horizontal_open = torch.zeros_like(x0, dtype=torch.bool)
            else:
                eps = torch.finfo(current_l.dtype).eps

                # Point on the movement segment closest to x = 0.
                safe_dx = torch.where(
                    dx.abs() > eps,
                    dx,
                    torch.ones_like(dx),
                )
                tx = torch.clamp(-x0 / safe_dx, 0.0, 1.0)
                tx = torch.where(dx.abs() > eps, tx, torch.zeros_like(tx))
                y_at_vertical_wall = y0 + tx * dy

                upper_vertical_open = (
                    torch.abs(y_at_vertical_wall - passage_center)
                    <= usable_half_passage
                )
                lower_vertical_open = (
                    torch.abs(y_at_vertical_wall + passage_center)
                    <= usable_half_passage
                )
                if not self.scene_manager.room_mapper.vertical_passage_open(True):
                    upper_vertical_open &= False
                if not self.scene_manager.room_mapper.vertical_passage_open(False):
                    lower_vertical_open &= False
                vertical_open = upper_vertical_open | lower_vertical_open

                # Point on the movement segment closest to y = 0.
                safe_dy = torch.where(
                    dy.abs() > eps,
                    dy,
                    torch.ones_like(dy),
                )
                ty = torch.clamp(-y0 / safe_dy, 0.0, 1.0)
                ty = torch.where(dy.abs() > eps, ty, torch.zeros_like(ty))
                x_at_horizontal_wall = x0 + ty * dx

                right_horizontal_open = (
                    torch.abs(x_at_horizontal_wall - passage_center)
                    <= usable_half_passage
                )
                left_horizontal_open = (
                    torch.abs(x_at_horizontal_wall + passage_center)
                    <= usable_half_passage
                )
                if not self.scene_manager.room_mapper.horizontal_passage_open(True):
                    right_horizontal_open &= False
                if not self.scene_manager.room_mapper.horizontal_passage_open(False):
                    left_horizontal_open &= False
                horizontal_open = right_horizontal_open | left_horizontal_open

            # The swept segment intersects the wall-clearance strip.
            touches_vertical_wall = (
                torch.minimum(x0, x1) <= clearance
            ) & (
                torch.maximum(x0, x1) >= -clearance
            )

            touches_horizontal_wall = (
                torch.minimum(y0, y1) <= clearance
            ) & (
                torch.maximum(y0, y1) >= -clearance
            )

            vertical_collision = touches_vertical_wall & (~vertical_open)
            horizontal_collision = touches_horizontal_wall & (~horizontal_open)

            return vertical_collision | horizontal_collision

    def check_future_collision(
            self,
            candidate_pos_w: torch.Tensor,
            exclude_goal: bool = True,
        ) -> torch.Tensor:
            """
            TOTHEHOLE: Vectorized future collision check against all active scene objects.

            Fast 2D circle-footprint approximation:
                dist(agent_xy, object_xy) < agent_radius + object_radius + margin

            candidate_pos_w: [num_envs, 3] future world position.
            exclude_goal:
                True keeps the current active goal from being treated as an obstacle.

            return: [num_envs] bool, True means candidate action would collide.
            """
            device = self.device

            # Candidate agent position in each env-local frame.
            candidate_pos_l = self.to_local(candidate_pos_w)  # [N, 2]
            agent_xy = candidate_pos_l[:, None, :2]           # [N, 1, 2]

            # SceneManager stores object positions in env-local coordinates.
            obj_xy = self.scene_manager.positions[:, :, :2]   # [N, M, 2]
            active = self.scene_manager.active.bool()         # [N, M]
            obj_radii = self.scene_manager.radii.expand(self.num_envs, -1)  # [N, M]

            # TOTHEHOLE: do not consider the active goal as an obstacle by default.
            if exclude_goal:
                env_ids = torch.arange(self.num_envs, device=device)
                goal_idxs = self.scene_manager.active_goal_indices.long().clamp(
                    0,
                    self.scene_manager.num_total_objects - 1,
                )
                active = active.clone()
                active[env_ids, goal_idxs] = False

            dists = torch.linalg.norm(obj_xy - agent_xy, dim=-1)  # [N, M]
            collision_threshold = self.agent_collision_radius + obj_radii + self.collision_margin
            colliding_objects = active & (dists < collision_threshold)
            return colliding_objects.any(dim=1)

    def _get_rewards(self) -> torch.Tensor:
            goal_reached, num_subs, r_error, a_error = self.goal_reached(get_num_subs=True)
            gamma = 0.5
            if self.TURN_TASK:
                F_s = -self.previous_angle_error 
                F_s_next = -a_error
                turnes = (F_s_next - F_s)/ANGULAR_VALUES*0.5
                self.previous_angle_error = a_error
            else:
                progress = self.previous_distance_error - r_error  # >0 если ближе к цели
                turnes = gamma * progress

            # TOTHEHOLE: no contact sensor; penalize the pre-action invalid flag.
            invalid_action = self.future_invalid_action_buf | self.out_of_bounds()

            collision_penalty = -3.0 * invalid_action.float()
            goal_bonus = 5.0 * (goal_reached).float() #TODO: self.done_action_buf & ADD this!
            # A wrong Done must not become an attractive cheap reset.
            wrong_done_penalty = -1.0 * (self.done_action_buf & (~goal_reached)).float()
            reward = -0.05 + turnes + collision_penalty + wrong_done_penalty + goal_bonus

            died, time_out = self._get_dones(inner=True)
            if torch.any(died | time_out):
                sr = self.update_success_rate(goal_reached)

            self.previous_distance_error = r_error
            self._current_episode_reward += reward.detach()
            return reward

    def get_contact(self):
            # TOTHEHOLE: ContactSensor is removed. This compatibility method exposes predicted future collisions.
            return self.future_collision_buf.clone()

    def _get_dones(self, inner=False) -> tuple[torch.Tensor, torch.Tensor]:
            """
            inner flag - not changes in buffers
            """
            time_out = self.is_time_out(self.my_episode_lenght - 1)
            # TOTHEHOLE: episode terminates on success, Done, predicted invalid action, or current out-of-bounds.
            # For this test task, auto-success is preserved; later AKGVP mode can require Done for success.
            died = self.future_invalid_action_buf | self.out_of_bounds() | self.goal_reached(get_num_subs=False)#TODO:|  time_out self.done_action_buf | | self.goal_reached(get_num_subs=False)

            if not inner:
                self.episode_length_buf[died] = 0
            return died, time_out
