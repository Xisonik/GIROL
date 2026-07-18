from __future__ import annotations

import gymnasium as gym
import torch
import isaaclab.sim as sim_utils
from isaaclab.sim import SimulationCfg
from isaaclab.utils import configclass

from ..modules.HabitatActionController import HabitatActionController
from .aloha_env_base import BaseWheeledRobotEnv, BaseWheeledRobotEnvCfg


@configclass
class WheeledRobotEnvCfg(BaseWheeledRobotEnvCfg):
    decimation = 24
    action_space = gym.spaces.Discrete(3)
    sim: SimulationCfg = SimulationCfg(
        dt=1/60,
        render_interval=decimation,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="min",
            static_friction=1.0,
            dynamic_friction=0.8,
            restitution=0.0,
        ),
    )


class WheeledRobotEnv(BaseWheeledRobotEnv):
    cfg: WheeledRobotEnvCfg

    def _default_use_staff(self) -> bool:
        return False

    def _default_use_obstacles(self) -> bool:
        return False

    def _init_variant_after_actuators(self) -> None:
        self.habitat_action_controller = HabitatActionController(
            env=self,
            robot=self._robot,
            num_envs=self.num_envs,
            device=self.device,
            decimation=self.cfg.decimation,
            turn_angle_deg=30.0,
            forward_distance=0.5,
            yaw_tolerance_deg=2.5,
            distance_tolerance=0.02,
            kp_yaw=5.0,
            kp_distance=6.0,
            kp_heading=4.0,
            max_linear_speed=2,
            max_angular_speed=6,
            print_stats=True,
        )
        self._macro_substep = 0

    def _reset_variant_state(self, env_ids: torch.Tensor) -> None:
        if hasattr(self, "habitat_action_controller"):
            self.habitat_action_controller.reset(env_ids)

    def _pre_physics_step(self, actions: torch.Tensor):
        self._macro_substep = 0
        actions = actions.to(self.device)
        if actions.dim() > 1:
            actions = actions.squeeze(-1)
        actions = actions.long().clamp(0, 2)
        self.habitat_action_controller.start(actions)

    def _apply_action(self):
        r = self.cfg.wheel_radius
        L = self.cfg.wheel_distance

        if self.DEF_TURN:
            linear_speed = torch.zeros(self.num_envs, device=self.device, dtype=torch.float32)
            angular_speed = torch.full((self.num_envs,), -2.0, device=self.device, dtype=torch.float32)
        else:
            linear_speed, angular_speed = self.habitat_action_controller.step()

        self.angular_speed = angular_speed
        self.velocities = torch.stack([linear_speed, angular_speed], dim=1)
        self._actions = self.velocities
        self._left_wheel_vel = (linear_speed - (angular_speed * L / 2)) / r
        self._right_wheel_vel = (linear_speed + (angular_speed * L / 2)) / r

        wheel_velocities = torch.stack([self._left_wheel_vel, self._right_wheel_vel], dim=1).unsqueeze(-1).to(dtype=torch.float32)
        self.last_actions = wheel_velocities
        self._robot.set_joint_velocity_target(wheel_velocities, joint_ids=[self._left_wheel_id, self._right_wheel_id])
