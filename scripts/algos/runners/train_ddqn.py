# -*- coding: utf-8 -*-
"""DDQN launcher for discrete navigation."""

from __future__ import annotations

from copy import deepcopy

import gymnasium as gym
from skrl_train_base import BaseSkrlTrain, recurrent_enabled

from skrl.agents.torch.dqn.ddqn import DDQN, DDQN_DEFAULT_CONFIG


def _default_ddqn_cfg():
    return deepcopy(DDQN_DEFAULT_CONFIG)



class DDQNTrain(BaseSkrlTrain):
    algo_name = "ddqn"
    supports_recurrent = False

    def agent_class(self, cfg: dict):
        return DDQN

    def default_skrl_cfg(self) -> dict:
        return _default_ddqn_cfg()

    def memory_size(self, cfg: dict) -> int:
        return int(cfg["agent"]["memory_size"])

    def validate_algorithm_contract(self, cfg: dict, env=None) -> None:
        agent = cfg["agent"]
        for key in ["memory_size", "gradient_steps", "batch_size", "gamma", "polyak", "learning_rate"]:
            if key not in agent:
                raise ValueError(f"Missing agent.{key}")

        if recurrent_enabled(cfg):
            raise ValueError("DDQN runner does not support recurrent models")

        if env is not None and not isinstance(env.action_space, gym.spaces.Discrete):
            raise ValueError(
                f"DDQN requires gym.spaces.Discrete action space, got {type(env.action_space)}. "
                "If the task uses MultiDiscrete, flatten it to a single Discrete action id before using DDQN."
            )

    def build_models(self, env, cfg: dict, modules: dict, device):
        q_spec = cfg["model"].get("q_network")
        if q_spec is None:
            raise ValueError("DDQN config requires model.q_network")

        return {
            "q_network": self.build_one_model(env, cfg, modules, device, q_spec),
            "target_q_network": self.build_one_model(env, cfg, modules, device, q_spec),
        }

    def build_skrl_cfg(self, cfg: dict, env, device, exp_dir):
        agent = cfg["agent"]
        skrl_cfg = self.base_agent_cfg()

        skrl_cfg["gradient_steps"] = int(agent["gradient_steps"])
        skrl_cfg["batch_size"] = int(agent["batch_size"])
        skrl_cfg["discount_factor"] = float(agent["gamma"])
        skrl_cfg["polyak"] = float(agent["polyak"])
        skrl_cfg["learning_rate"] = float(agent["learning_rate"])
        skrl_cfg["random_timesteps"] = int(agent.get("random_timesteps", 0))
        skrl_cfg["learning_starts"] = int(agent.get("learning_starts", 1000))
        skrl_cfg["update_interval"] = int(agent.get("update_interval", 1))
        skrl_cfg["target_update_interval"] = int(agent.get("target_update_interval", 10))

        return self.apply_common_skrl_cfg(skrl_cfg, cfg, env, device, exp_dir)


if __name__ == "__main__":
    DDQNTrain().run()
