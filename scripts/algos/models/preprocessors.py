from __future__ import annotations

import torch.nn as nn
from skrl.resources.preprocessors.torch import RunningStandardScaler
from skrl.utils.spaces.torch import flatten_tensorized_space, unflatten_tensorized_space


class DictRunningStandardScaler(nn.Module):
    """Normalize image features only; leave task/graph scalars untouched."""

    def __init__(self, size, img_space, device=None, epsilon=1e-8, clip_threshold=5.0):
        super().__init__()
        self.full_space = size
        self.img_scaler = RunningStandardScaler(
            size=img_space,
            epsilon=epsilon,
            clip_threshold=clip_threshold,
            device=device,
        )

    def forward(self, x, train=False, inverse=False, no_grad=True):
        s = unflatten_tensorized_space(self.full_space, x)
        if "img" in s:
            s["img"] = self.img_scaler(s["img"], train=train, inverse=inverse, no_grad=no_grad)
        return flatten_tensorized_space(s)
