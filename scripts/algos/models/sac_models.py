"""SAC concat navigation models.

SAC requires:
    policy
    critic_1
    critic_2
    target_critic_1
    target_critic_2

This file provides the non-branch/early-concat SAC models:
    states -> NavFeatureExtractor -> concat feature vector
    Q critic additionally concatenates taken_actions.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from skrl.models.torch import DeterministicMixin, GaussianMixin, Model

from models.nav_features import NavFeatureExtractor


class MLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dims: list[int] | tuple[int, ...],
        activation=nn.ReLU,
        output_activation=None,
    ):
        super().__init__()
        if hidden_dims is None:
            raise ValueError("hidden_dims must be explicit")

        layers: list[nn.Module] = []
        prev = int(input_dim)
        for hidden in hidden_dims:
            layers.append(nn.Linear(prev, int(hidden)))
            layers.append(activation())
            prev = int(hidden)

        layers.append(nn.Linear(prev, int(output_dim)))
        if output_activation is not None:
            layers.append(output_activation())

        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class SACNavPolicy(GaussianMixin, Model):
    def __init__(
        self,
        observation_space,
        action_space,
        device,
        modules: dict[str, nn.Module],
        model_cfg: dict,
        num_envs: int = 1,
        *,
        hidden_dims: list[int] | tuple[int, ...],
    ):
        Model.__init__(self, observation_space, action_space, device)
        GaussianMixin.__init__(
            self,
            clip_actions=False,
            clip_log_std=True,
            min_log_std=-5,
            max_log_std=2,
        )
        self.encoder = NavFeatureExtractor(
            observation_space=observation_space,
            modules=modules,
            features_cfg=model_cfg["features"],
        )
        self.net = MLP(self.encoder.output_dim, self.num_actions, hidden_dims, nn.ReLU, nn.Tanh)
        self.log_std_parameter = nn.Parameter(torch.zeros(self.num_actions))

    def compute(self, inputs, role):
        x = self.encoder(inputs["states"])
        return self.net(x), self.log_std_parameter, {}


class SACNavQCritic(DeterministicMixin, Model):
    def __init__(
        self,
        observation_space,
        action_space,
        device,
        modules: dict[str, nn.Module],
        model_cfg: dict,
        num_envs: int = 1,
        *,
        hidden_dims: list[int] | tuple[int, ...],
    ):
        Model.__init__(self, observation_space, action_space, device)
        DeterministicMixin.__init__(self, clip_actions=False)
        self.encoder = NavFeatureExtractor(
            observation_space=observation_space,
            modules=modules,
            features_cfg=model_cfg["features"],
        )
        self.net = MLP(self.encoder.output_dim + self.num_actions, 1, hidden_dims, nn.ReLU)

    def compute(self, inputs, role):
        z = self.encoder(inputs["states"])
        actions = inputs["taken_actions"]
        x = torch.cat([z, actions], dim=-1)
        return self.net(x), {}
