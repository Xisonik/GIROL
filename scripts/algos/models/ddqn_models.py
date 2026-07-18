"""DDQN navigation models.

DDQN requires:
    q_network
    target_q_network

Pipeline:
    states -> NavFeatureExtractor -> MLP -> Q-values for all discrete actions

Important difference from SAC:
    The DDQN Q-network does not receive taken_actions as input. It returns one
    Q-value per discrete action, and the agent selects/updates Q(s)[a].
"""

from __future__ import annotations

import gymnasium as gym
import torch
import torch.nn as nn
from skrl.models.torch import DeterministicMixin, Model

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


class DDQNNavQNetwork(DeterministicMixin, Model):
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
        if not isinstance(action_space, gym.spaces.Discrete):
            raise ValueError(f"DDQNNavQNetwork requires gym.spaces.Discrete, got {type(action_space)}")

        Model.__init__(self, observation_space, action_space, device)
        DeterministicMixin.__init__(self, clip_actions=False)

        self.encoder = NavFeatureExtractor(
            observation_space=observation_space,
            modules=modules,
            features_cfg=model_cfg["features"],
        )
        self.net = MLP(
            input_dim=self.encoder.output_dim,
            output_dim=self.num_actions,
            hidden_dims=hidden_dims,
            activation=nn.ReLU,
        )

    def compute(self, inputs, role):
        x = self.encoder(inputs["states"])
        return self.net(x), {}
