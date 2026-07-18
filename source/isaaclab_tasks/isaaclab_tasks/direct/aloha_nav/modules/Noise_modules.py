from __future__ import annotations

import math
import torch
import torch.nn as nn


def wrap_angle_rad(x: torch.Tensor) -> torch.Tensor:
    """Wrap radians to [-pi, pi]."""
    return torch.atan2(torch.sin(x), torch.cos(x))


class RelativeYawObservationNoise(nn.Module):
    """Mixture noise for relative yaw observation.

    Input/output:
        relative_yaw: [num_envs, 1], radians

    accuracy:
        Probability of using Gaussian-corrupted yaw around true yaw.
        Example:
            accuracy=60 -> 60% Gaussian, 40% Uniform(-pi, pi)

    variance:
        Gaussian variance in rad^2.
    """

    def __init__(
        self,
        accuracy: float = 100.0,
        variance: float = 0.0,
        enabled: bool = True,
    ):
        super().__init__()

        accuracy = float(accuracy)
        if accuracy > 1.0:
            accuracy /= 100.0

        if not 0.0 <= accuracy <= 1.0:
            raise ValueError(f"accuracy must be in [0, 1] or [0, 100], got {accuracy}")

        variance = float(variance)
        if variance < 0.0:
            raise ValueError(f"variance must be non-negative, got {variance}")

        self.accuracy = accuracy
        self.variance = variance
        self.enabled = bool(enabled)

    @property
    def std(self) -> float:
        return math.sqrt(self.variance)

    def forward(self, true_yaw: torch.Tensor) -> torch.Tensor:
        if true_yaw.dim() != 2 or true_yaw.shape[-1] != 1:
            raise ValueError(f"true_yaw must have shape [num_envs, 1], got {tuple(true_yaw.shape)}")

        if not self.enabled:
            return true_yaw

        gaussian_mask = torch.rand_like(true_yaw) < self.accuracy

        gaussian_yaw = true_yaw + torch.randn_like(true_yaw) * self.std
        gaussian_yaw = wrap_angle_rad(gaussian_yaw)

        uniform_yaw = torch.rand_like(true_yaw) * (2.0 * math.pi) - math.pi

        return torch.where(gaussian_mask, gaussian_yaw, uniform_yaw)