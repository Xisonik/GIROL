from __future__ import annotations

import math
import gymnasium as gym
import torch
import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObject, RigidObjectCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg, TiledCamera, TiledCameraCfg
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
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=32, env_spacing=30, replicate_physics=True)

    agent: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/Agent",
        spawn=sim_utils.CuboidCfg(
            size=(0.25, 0.25, 0.25),
            activate_contact_sensors=True,
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

    contact_sensor = ContactSensorCfg(
        prim_path="/World/envs/env_.*/Agent",
        update_period=0.0,
        history_length=3,
        debug_vis=False,
        filter_prim_paths_expr=["/World/envs/env_.*"],
    )


class WheeledRobotEnv(BaseWheeledRobotEnv):
    cfg: WheeledRobotEnvCfg

    def _collision_action_mode(self) -> str:
        return "discrete"

    def _default_collision_radius(self) -> float:
        return 0.18

    def _init_actuator_handles(self) -> None:
        # Kinematic cube-agent has no wheel joints.
        self.done_action_buf = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        self.last_macro_actions = torch.zeros(
            self.num_envs, dtype=torch.long, device=self.device
        )

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
        super()._reset_variant_state(env_ids)
        self.done_action_buf[env_ids] = False
        self.last_macro_actions[env_ids] = 0

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

            # Contact mode remains available, while GT is the default.
            from isaaclab.sensors import ContactSensor
            self._contact_sensor = ContactSensor(self.cfg.contact_sensor)
            self.scene.sensors["contact_sensor"] = self._contact_sensor

            light_cfg = sim_utils.DomeLightCfg(intensity=300.0, color=(0.75, 0.75, 0.75))
            light_cfg.func("/World/Light", light_cfg)

    def _pre_physics_step(self, actions: torch.Tensor):
        input_actions = actions
        actions = actions.to(self.device)
        if actions.dim() > 1:
            actions = actions.squeeze(-1)
        actions = actions.long()

        root_pos_w = self._robot.data.root_pos_w.clone()
        root_quat_w = self._robot.data.root_quat_w.clone()
        base_quat = torch.tensor(
            [1.0, 0.0, 0.0, 0.0],
            device=self.device,
            dtype=root_quat_w.dtype,
        )
        quat_bad = (
            torch.isnan(root_quat_w).any(dim=-1)
            | torch.isinf(root_quat_w).any(dim=-1)
        )
        if quat_bad.any():
            root_quat_w[quat_bad] = base_quat

        yaw = quat_wxyz_to_yaw(root_quat_w)
        turn_angle = ANGULAR_VALUES * math.pi / 180.0
        forward_distance = 0.25

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
            expert_actions = self.control_module.compute_actions(
                positions=self.to_local(root_pos_w),
                orientations=yaw,
            )
            actions = actions.clone()
            actions[controlled_mask] = expert_actions[controlled_mask]

            executed_for_input = actions.to(
                device=input_actions.device,
                dtype=input_actions.dtype,
            )
            if input_actions.dim() > 1:
                executed_for_input = executed_for_input.view_as(input_actions)
            input_actions.copy_(executed_for_input)

        self.last_macro_actions[:] = actions

        # action_space is Discrete(3), therefore no Done action is active.
        self.done_action_buf[:] = False
        collision_result = self.collision_manager.evaluate_discrete(
            current_pos_w=root_pos_w,
            current_yaw=yaw,
            actions=actions,
            turn_angle_rad=turn_angle,
            forward_distance=forward_distance,
            turn_task=self.TURN_TASK,
            left_action=0,
            right_action=1,
            forward_action=2,
            done_action=None,
        )

        candidate_quat_w = yaw_to_quat_wxyz(
            collision_result.candidate_yaw
        )
        valid_action = collision_result.valid_action

        if valid_action.any():
            valid_env_ids = torch.where(valid_action)[0].to(
                dtype=torch.long
            )
            root_pose = torch.cat(
                [
                    collision_result.candidate_pos_w[valid_action],
                    candidate_quat_w[valid_action],
                ],
                dim=-1,
            )
            root_vel = torch.zeros(
                valid_env_ids.numel(),
                6,
                device=self.device,
                dtype=torch.float32,
            )
            self._robot.write_root_pose_to_sim(
                root_pose, env_ids=valid_env_ids
            )
            self._robot.write_root_velocity_to_sim(
                root_vel, env_ids=valid_env_ids
            )

        is_left = actions == 0
        is_right = actions == 1
        is_forward = actions == 2
        linear_delta = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.float32
        )
        yaw_delta = torch.zeros_like(linear_delta)
        linear_delta[is_forward] = forward_distance
        yaw_delta[is_left] = turn_angle
        yaw_delta[is_right] = -turn_angle

        invalid_action = collision_result.invalid_action
        linear_delta[invalid_action] = 0.0
        yaw_delta[invalid_action] = 0.0
        if self.TURN_TASK:
            linear_delta[:] = 0.0

        self.velocities = torch.stack([linear_delta, yaw_delta], dim=1)
        self._actions = self.velocities
        self.angular_speed = yaw_delta
        self.last_actions = self.velocities

    def _apply_action(self):
        # The kinematic pose is written in _pre_physics_step().
        pass

    def check_future_out_of_bounds(
        self, candidate_pos_w: torch.Tensor
    ) -> torch.Tensor:
        return self.collision_manager.check_out_of_bounds(candidate_pos_w)

    def check_future_inner_wall_collision(
        self,
        current_pos_w: torch.Tensor,
        candidate_pos_w: torch.Tensor,
        **_unused,
    ) -> torch.Tensor:
        return self.collision_manager.check_inner_wall_collision(
            current_pos_w=current_pos_w,
            candidate_pos_w=candidate_pos_w,
        )

    def check_future_collision(
        self,
        candidate_pos_w: torch.Tensor,
        exclude_goal: bool = True,
    ) -> torch.Tensor:
        old_value = self.collision_manager.exclude_goal
        self.collision_manager.exclude_goal = bool(exclude_goal)
        try:
            return self.collision_manager.check_object_collision(
                candidate_pos_w
            )
        finally:
            self.collision_manager.exclude_goal = old_value

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

            collision_event = self.collision_manager.get_collision_event(
                self._robot.data.root_pos_w
            )
            if self.TURN_TASK:
                collision_event = collision_event | self.out_of_bounds()

            collision_penalty = -3.0 * collision_event.float()
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
        return self.collision_manager.get_collision_component()

    def _get_dones(self, inner=False) -> tuple[torch.Tensor, torch.Tensor]:
        time_out = self.is_time_out(self.my_episode_lenght - 1)
        collision_event = self.collision_manager.get_collision_event(
            self._robot.data.root_pos_w
        )
        if self.TURN_TASK:
            collision_event = collision_event | self.out_of_bounds()
        died = collision_event | self.goal_reached(get_num_subs=False)

        if not inner:
            self.episode_length_buf[died] = 0
        return died, time_out

