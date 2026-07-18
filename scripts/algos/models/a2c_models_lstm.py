"""LSTM A2C navigation models.

Use these classes with skrl A2C_RNN, not plain A2C.

Pipeline:
    states -> NavFeatureExtractor -> LSTM -> MLP head -> action/value

The LSTM parameters are explicit constructor kwargs. They are not read from
`model.recurrent` and there is no non-recurrent fallback in this file.
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


class LSTMBlock(nn.Module):
    """Small wrapper matching skrl recurrent-model expectations."""

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int,
        sequence_length: int,
        num_envs: int,
    ):
        super().__init__()
        self.input_size = int(input_size)
        self.hidden_size = int(hidden_size)
        self.num_layers = int(num_layers)
        self.sequence_length = int(sequence_length)
        self.num_envs = int(num_envs)

        if self.sequence_length <= 0:
            raise ValueError("sequence_length must be positive")
        if self.num_layers <= 0:
            raise ValueError("lstm_num_layers must be positive")
        if self.hidden_size <= 0:
            raise ValueError("lstm_hidden_size must be positive")

        self.lstm = nn.LSTM(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            batch_first=True,
        )

    def get_specification(self) -> dict:
        return {
            "rnn": {
                "sequence_length": self.sequence_length,
                "sizes": [
                    (self.num_layers, self.num_envs, self.hidden_size),
                    (self.num_layers, self.num_envs, self.hidden_size),
                ],
            }
        }

    def forward(self, x: torch.Tensor, inputs: dict) -> tuple[torch.Tensor, list[torch.Tensor]]:
        if "rnn" not in inputs:
            raise RuntimeError("LSTM model expected inputs['rnn']; use skrl A2C_RNN")

        hidden_states, cell_states = inputs["rnn"][0], inputs["rnn"][1]
        terminated = inputs.get("terminated", None)

        if self.training:
            rnn_input = x.view(-1, self.sequence_length, x.shape[-1])

            hidden_states = hidden_states.view(
                self.num_layers,
                -1,
                self.sequence_length,
                hidden_states.shape[-1],
            )
            cell_states = cell_states.view(
                self.num_layers,
                -1,
                self.sequence_length,
                cell_states.shape[-1],
            )

            hidden_states = hidden_states[:, :, 0, :].contiguous()
            cell_states = cell_states[:, :, 0, :].contiguous()

            if terminated is not None and torch.any(terminated):
                rnn_outputs = []
                terminated = terminated.bool().view(-1, self.sequence_length)
                indexes = (
                    [0]
                    + (terminated[:, :-1].any(dim=0).nonzero(as_tuple=True)[0] + 1).tolist()
                    + [self.sequence_length]
                )

                for i in range(len(indexes) - 1):
                    i0, i1 = indexes[i], indexes[i + 1]
                    rnn_output, (hidden_states, cell_states) = self.lstm(
                        rnn_input[:, i0:i1, :],
                        (hidden_states, cell_states),
                    )
                    hidden_states[:, terminated[:, i1 - 1], :] = 0
                    cell_states[:, terminated[:, i1 - 1], :] = 0
                    rnn_outputs.append(rnn_output)

                rnn_output = torch.cat(rnn_outputs, dim=1)
                rnn_states = [hidden_states, cell_states]
            else:
                rnn_output, rnn_states = self.lstm(rnn_input, (hidden_states, cell_states))
                rnn_states = [rnn_states[0], rnn_states[1]]

            return torch.flatten(rnn_output, start_dim=0, end_dim=1), rnn_states

        # Rollout / evaluation: one step per environment.
        rnn_input = x.view(-1, 1, x.shape[-1])
        rnn_output, rnn_states = self.lstm(rnn_input, (hidden_states, cell_states))
        return torch.flatten(rnn_output, start_dim=0, end_dim=1), [rnn_states[0], rnn_states[1]]


class ContinuousA2CNavLSTMActor(GaussianMixin, Model):
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
        lstm_hidden_size: int,
        sequence_length: int,
        lstm_num_layers: int = 1,
    ):
        Model.__init__(self, observation_space, action_space, device)
        GaussianMixin.__init__(
            self,
            clip_actions=True,
            clip_log_std=True,
            min_log_std=-5,
            max_log_std=2,
        )
        self.encoder = NavFeatureExtractor(observation_space, modules, model_cfg["features"])
        self.rnn = LSTMBlock(
            input_size=self.encoder.output_dim,
            hidden_size=lstm_hidden_size,
            num_layers=lstm_num_layers,
            sequence_length=sequence_length,
            num_envs=num_envs,
        )
        self.net = MLP(self.rnn.hidden_size, self.num_actions, hidden_dims, nn.ReLU, nn.Tanh)
        self.log_std_parameter = nn.Parameter(torch.zeros(self.num_actions))

    def get_specification(self):
        return self.rnn.get_specification()

    def compute(self, inputs, role):
        x = self.encoder(inputs["states"])
        x, rnn_states = self.rnn(x, inputs)
        mean_actions = self.net(x)
        return mean_actions, self.log_std_parameter, {"rnn": rnn_states}


class DiscreteA2CNavLSTMActor(CategoricalMixin, Model):
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
        lstm_hidden_size: int,
        sequence_length: int,
        lstm_num_layers: int = 1,
    ):
        Model.__init__(self, observation_space, action_space, device)
        CategoricalMixin.__init__(self, unnormalized_log_prob=True)
        self.encoder = NavFeatureExtractor(observation_space, modules, model_cfg["features"])
        self.rnn = LSTMBlock(
            input_size=self.encoder.output_dim,
            hidden_size=lstm_hidden_size,
            num_layers=lstm_num_layers,
            sequence_length=sequence_length,
            num_envs=num_envs,
        )
        self.net = MLP(self.rnn.hidden_size, self.num_actions, hidden_dims, nn.ReLU)

    def get_specification(self):
        return self.rnn.get_specification()

    def compute(self, inputs, role):
        x = self.encoder(inputs["states"])
        x, rnn_states = self.rnn(x, inputs)
        return self.net(x), {"rnn": rnn_states}


class MultiDiscreteA2CNavLSTMActor(MultiCategoricalMixin, Model):
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
        lstm_hidden_size: int,
        sequence_length: int,
        lstm_num_layers: int = 1,
    ):
        Model.__init__(self, observation_space, action_space, device)
        MultiCategoricalMixin.__init__(self, unnormalized_log_prob=True, reduction="sum")
        self.encoder = NavFeatureExtractor(observation_space, modules, model_cfg["features"])
        self.rnn = LSTMBlock(
            input_size=self.encoder.output_dim,
            hidden_size=lstm_hidden_size,
            num_layers=lstm_num_layers,
            sequence_length=sequence_length,
            num_envs=num_envs,
        )
        self.net = MLP(self.rnn.hidden_size, self.num_actions, hidden_dims, nn.ReLU)

    def get_specification(self):
        return self.rnn.get_specification()

    def compute(self, inputs, role):
        x = self.encoder(inputs["states"])
        x, rnn_states = self.rnn(x, inputs)
        return self.net(x), {"rnn": rnn_states}


def make_a2c_nav_lstm_actor(
    observation_space,
    action_space,
    device,
    modules: dict[str, nn.Module],
    model_cfg: dict,
    num_envs: int = 1,
    **kwargs,
):
    if isinstance(action_space, gym.spaces.Discrete):
        return DiscreteA2CNavLSTMActor(observation_space, action_space, device, modules, model_cfg, num_envs=num_envs, **kwargs)
    if isinstance(action_space, gym.spaces.MultiDiscrete):
        return MultiDiscreteA2CNavLSTMActor(observation_space, action_space, device, modules, model_cfg, num_envs=num_envs, **kwargs)
    if isinstance(action_space, gym.spaces.Box):
        return ContinuousA2CNavLSTMActor(observation_space, action_space, device, modules, model_cfg, num_envs=num_envs, **kwargs)
    raise ValueError(f"Unsupported action space: {type(action_space)}")


make_a2c_lstm_actor = make_a2c_nav_lstm_actor


class A2CNavLSTMValue(DeterministicMixin, Model):
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
        lstm_hidden_size: int,
        sequence_length: int,
        lstm_num_layers: int = 1,
    ):
        Model.__init__(self, observation_space, action_space, device)
        DeterministicMixin.__init__(self, clip_actions=False)
        self.encoder = NavFeatureExtractor(observation_space, modules, model_cfg["features"])
        self.rnn = LSTMBlock(
            input_size=self.encoder.output_dim,
            hidden_size=lstm_hidden_size,
            num_layers=lstm_num_layers,
            sequence_length=sequence_length,
            num_envs=num_envs,
        )
        self.net = MLP(self.rnn.hidden_size, 1, hidden_dims, nn.ReLU)

    def get_specification(self):
        return self.rnn.get_specification()

    def compute(self, inputs, role):
        x = self.encoder(inputs["states"])
        x, rnn_states = self.rnn(x, inputs)
        return self.net(x), {"rnn": rnn_states}
