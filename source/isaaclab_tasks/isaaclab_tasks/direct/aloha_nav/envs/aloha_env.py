from __future__ import annotations

from isaaclab.utils import configclass

from .aloha_env_base import BaseWheeledRobotEnv, BaseWheeledRobotEnvCfg


@configclass
class WheeledRobotEnvCfg(BaseWheeledRobotEnvCfg):
    pass


class WheeledRobotEnv(BaseWheeledRobotEnv):
    cfg: WheeledRobotEnvCfg
