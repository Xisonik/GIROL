# -*- coding: utf-8 -*-
"""PPO launcher."""

from __future__ import annotations

from skrl.agents.torch.ppo import PPO, PPO_RNN, PPO_DEFAULT_CONFIG

from skrl_train_base import BaseSkrlTrain, recurrent_enabled


class PPOTrain(BaseSkrlTrain):
    algo_name = "ppo"
    supports_recurrent = True

    def agent_class(self, cfg: dict):
        return PPO_RNN if recurrent_enabled(cfg) else PPO

    def default_skrl_cfg(self) -> dict:
        return PPO_DEFAULT_CONFIG.copy()

    def memory_size(self, cfg: dict) -> int:
        return int(cfg["agent"]["rollouts"])

    def validate_algorithm_contract(self, cfg: dict, env=None) -> None:
        agent = cfg["agent"]
        for key in ["rollouts", "learning_epochs", "mini_batches", "learning_rate", "gamma", "gae_lambda"]:
            if key not in agent:
                raise ValueError(f"Missing agent.{key}")

        if recurrent_enabled(cfg):
            rollouts = int(agent["rollouts"])
            sequence_length = int(cfg["model"].get("recurrent", {}).get("sequence_length", 1))
            if sequence_length <= 0:
                raise ValueError("model.recurrent.sequence_length must be positive")
            if rollouts % sequence_length != 0:
                raise ValueError(f"agent.rollouts={rollouts} must be divisible by sequence_length={sequence_length}")

    def build_models(self, env, cfg: dict, modules: dict, device):
        return {
            "policy": self.build_one_model(env, cfg, modules, device, cfg["model"]["actor"]),
            "value": self.build_one_model(env, cfg, modules, device, cfg["model"]["critic"]),
        }

    def build_skrl_cfg(self, cfg: dict, env, device, exp_dir):
        agent = cfg["agent"]
        skrl_cfg = self.base_agent_cfg()

        skrl_cfg["rollouts"] = int(agent["rollouts"])
        skrl_cfg["learning_epochs"] = int(agent["learning_epochs"])
        skrl_cfg["mini_batches"] = int(agent["mini_batches"])
        skrl_cfg["discount_factor"] = float(agent["gamma"])
        skrl_cfg["gae_lambda"] = float(agent["gae_lambda"])
        skrl_cfg["learning_rate"] = float(agent["learning_rate"])
        skrl_cfg["random_timesteps"] = int(agent.get("random_timesteps", 0))
        skrl_cfg["learning_starts"] = int(agent.get("learning_starts", 0))
        skrl_cfg["grad_norm_clip"] = float(agent.get("grad_norm_clip", 0.5))
        skrl_cfg["ratio_clip"] = float(agent.get("ratio_clip", 0.2))
        skrl_cfg["value_clip"] = float(agent.get("value_clip", 0.2))
        skrl_cfg["entropy_loss_scale"] = float(agent.get("entropy_loss_scale", 0.0))
        skrl_cfg["value_loss_scale"] = float(agent.get("value_loss_scale", 2.5))
        skrl_cfg["kl_threshold"] = float(agent.get("kl_threshold", 0.0))

        return self.apply_common_skrl_cfg(skrl_cfg, cfg, env, device, exp_dir)


if __name__ == "__main__":
    PPOTrain().run()
