"""Non-recurrent A2C navigation models.

Use these classes with skrl A2C, not A2C_RNN.

Pipeline:
    states -> NavFeatureExtractor -> MLP head -> action/value
"""

from __future__ import annotations

import gymnasium as gym
import torch
import torch.nn as nn
from skrl.models.torch import (
    CategoricalMixin,
    DeterministicMixin,
    GaussianMixin,
    Model,
    MultiCategoricalMixin,
)

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
        for h in hidden_dims:
            layers.append(nn.Linear(prev, int(h)))
            layers.append(activation())
            prev = int(h)
        layers.append(nn.Linear(prev, int(output_dim)))
        if output_activation is not None:
            layers.append(output_activation())
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ContinuousA2CNavActor(GaussianMixin, Model):
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
            clip_actions=True,
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
        return self.net(x), {"log_std": self.log_std_parameter}


class DiscreteA2CNavActor(CategoricalMixin, Model):
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
        CategoricalMixin.__init__(self, unnormalized_log_prob=True)
        self.encoder = NavFeatureExtractor(
            observation_space=observation_space,
            modules=modules,
            features_cfg=model_cfg["features"],
        )
        self.net = MLP(self.encoder.output_dim, self.num_actions, hidden_dims, nn.ReLU)

    def compute(self, inputs, role):
        x = self.encoder(inputs["states"])
        return self.net(x), {}


class MultiDiscreteA2CNavActor(MultiCategoricalMixin, Model):
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
        MultiCategoricalMixin.__init__(self, unnormalized_log_prob=True, reduction="sum")
        self.encoder = NavFeatureExtractor(
            observation_space=observation_space,
            modules=modules,
            features_cfg=model_cfg["features"],
        )
        self.net = MLP(self.encoder.output_dim, self.num_actions, hidden_dims, nn.ReLU)

    def compute(self, inputs, role):
        x = self.encoder(inputs["states"])
        return self.net(x), {}


def make_a2c_nav_actor(
    observation_space,
    action_space,
    device,
    modules: dict[str, nn.Module],
    model_cfg: dict,
    num_envs: int = 1,
    **kwargs,
):
    if isinstance(action_space, gym.spaces.Discrete):
        return DiscreteA2CNavActor(observation_space, action_space, device, modules, model_cfg, num_envs=num_envs, **kwargs)
    if isinstance(action_space, gym.spaces.MultiDiscrete):
        return MultiDiscreteA2CNavActor(observation_space, action_space, device, modules, model_cfg, num_envs=num_envs, **kwargs)
    if isinstance(action_space, gym.spaces.Box):
        return ContinuousA2CNavActor(observation_space, action_space, device, modules, model_cfg, num_envs=num_envs, **kwargs)
    raise ValueError(f"Unsupported action space: {type(action_space)}")


make_a2c_actor = make_a2c_nav_actor


class A2CNavValue(DeterministicMixin, Model):
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
        self.net = MLP(self.encoder.output_dim, 1, hidden_dims, nn.ReLU)

    def compute(self, inputs, role):
        x = self.encoder(inputs["states"])
        return self.net(x), {}
