# -*- coding: utf-8 -*-
"""SAC launcher.

SAC requires continuous Box actions and the following model roles:
    policy
    critic_1
    critic_2
    target_critic_1
    target_critic_2
"""

from __future__ import annotations

import gymnasium as gym
from skrl.agents.torch.sac import SAC, SAC_DEFAULT_CONFIG

from skrl_train_base import BaseSkrlTrain, EvalActionSource


class SACTrain(BaseSkrlTrain):
    algo_name = "sac"
    supports_recurrent = False
    eval_action_source = EvalActionSource.MEAN_ACTION

    def agent_class(self, cfg: dict):
        return SAC

    def default_skrl_cfg(self) -> dict:
        return SAC_DEFAULT_CONFIG.copy()

    def memory_size(self, cfg: dict) -> int:
        return int(cfg["agent"]["memory_size"])

    def validate_algorithm_contract(self, cfg: dict, env=None) -> None:
        agent = cfg["agent"]
        for key in ["memory_size", "gradient_steps", "batch_size", "gamma", "polyak", "actor_learning_rate", "critic_learning_rate"]:
            if key not in agent:
                raise ValueError(f"Missing agent.{key}")

        if env is not None and not isinstance(env.action_space, gym.spaces.Box):
            raise ValueError("SAC requires continuous gym.spaces.Box action space. Use task_name='Aloha_nav'.")

    def build_models(self, env, cfg: dict, modules: dict, device):
        model_cfg = cfg["model"]
        policy_spec = model_cfg.get("policy", model_cfg.get("actor"))
        critic_spec = model_cfg.get("q_critic", model_cfg.get("critic"))

        if policy_spec is None:
            raise ValueError("SAC config requires model.policy or model.actor")
        if critic_spec is None:
            raise ValueError("SAC config requires model.q_critic or model.critic")

        return {
            "policy": self.build_one_model(env, cfg, modules, device, policy_spec),
            "critic_1": self.build_one_model(env, cfg, modules, device, critic_spec),
            "critic_2": self.build_one_model(env, cfg, modules, device, critic_spec),
            "target_critic_1": self.build_one_model(env, cfg, modules, device, critic_spec),
            "target_critic_2": self.build_one_model(env, cfg, modules, device, critic_spec),
        }

    def build_skrl_cfg(self, cfg: dict, env, device, exp_dir):
        agent = cfg["agent"]
        skrl_cfg = self.base_agent_cfg()

        skrl_cfg["gradient_steps"] = int(agent["gradient_steps"])
        skrl_cfg["batch_size"] = int(agent["batch_size"])
        skrl_cfg["discount_factor"] = float(agent["gamma"])
        skrl_cfg["polyak"] = float(agent["polyak"])
        skrl_cfg["actor_learning_rate"] = float(agent["actor_learning_rate"])
        skrl_cfg["critic_learning_rate"] = float(agent["critic_learning_rate"])
        skrl_cfg["random_timesteps"] = int(agent.get("random_timesteps", 0))
        skrl_cfg["learning_starts"] = int(agent.get("learning_starts", 100))
        skrl_cfg["grad_norm_clip"] = float(agent.get("grad_norm_clip", 0.0))
        skrl_cfg["learn_entropy"] = bool(agent.get("learn_entropy", True))
        skrl_cfg["entropy_learning_rate"] = float(agent.get("entropy_learning_rate", 5e-3))
        skrl_cfg["initial_entropy_value"] = float(agent.get("initial_entropy_value", 1.0))

        # Compatibility with newer skrl configs that use one learning_rate tuple.
        if "learning_rate" in skrl_cfg:
            skrl_cfg["learning_rate"] = (
                float(agent["actor_learning_rate"]),
                float(agent["critic_learning_rate"]),
                float(agent.get("entropy_learning_rate", 5e-3)),
            )

        return self.apply_common_skrl_cfg(skrl_cfg, cfg, env, device, exp_dir)


if __name__ == "__main__":
    SACTrain().run()
