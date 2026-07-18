from __future__ import annotations

import os
import gymnasium as gym
import numpy as np
import torch
from isaaclab.utils import configclass

from .aloha_env_base import BaseWheeledRobotEnv, BaseWheeledRobotEnvCfg


TCONF_ID = int(os.environ.get("ALOHA_TCONF_ID", "0"))

ANGULAR_CONFIGS = {
    0: [-3.0, 0.0, 3.0],
    1: [-4.0, 0.0, 4.0],
    2: [-4.0, -3.0, 0.0, 3.0, 4.0],
    3: [-4.0, -3.0, -2.0, 0.0, 2.0, 3.0, 4.0],
}

ANGULAR_VALUES = ANGULAR_CONFIGS[TCONF_ID]


@configclass
class WheeledRobotEnvCfg(BaseWheeledRobotEnvCfg):
    tconfig = TCONF_ID
    angular_values = tuple(ANGULAR_VALUES)

    action_space = gym.spaces.MultiDiscrete(
        np.array([2, len(ANGULAR_VALUES)], dtype=np.int64)
    )


class WheeledRobotEnv(BaseWheeledRobotEnv):
    cfg: WheeledRobotEnvCfg

    def _pre_physics_step(self, actions: torch.Tensor):
        r = self.cfg.wheel_radius
        L = self.cfg.wheel_distance

        actions = actions.to(self.device)
        if actions.dim() == 1:
            actions = actions.unsqueeze(-1)

        angular_values = torch.tensor(
            self.cfg.angular_values,
            device=self.device,
            dtype=torch.float32,
        )

        action_idx = actions.long()
        action_idx[:, 0] = action_idx[:, 0]
        action_idx[:, 1] = action_idx[:, 1]

        linear_values = torch.tensor(
            [0.0, 1.0],
            device=self.device,
            dtype=torch.float32,
        )

        linear_speed = linear_values[action_idx[:, 0]]
        angular_speed = angular_values[action_idx[:, 1]]

        if self.TURN_TASK:
            linear_speed = torch.zeros_like(linear_speed)

        self._actions = torch.stack([linear_speed, angular_speed], dim=1)
        self.angular_speed = angular_speed
        self.velocities = self._actions
        self._left_wheel_vel = (linear_speed - (angular_speed * L / 2)) / r
        self._right_wheel_vel = (linear_speed + (angular_speed * L / 2)) / r
