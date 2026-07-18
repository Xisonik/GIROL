"""SAC branch-fusion models.

SAC needs:
    policy: stochastic Gaussian actor
    critic_1 / critic_2 / target_critic_1 / target_critic_2:
        deterministic Q(s, a) critics

These models are non-recurrent and intended for continuous Box action spaces.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from skrl.models.torch import DeterministicMixin, GaussianMixin, Model

from models.a2c_branch_models import BranchFeatureBackbone, MLP


class SACBranchPolicy(GaussianMixin, Model):
    def __init__(
        self,
        observation_space,
        action_space,
        device,
        modules: dict[str, nn.Module],
        model_cfg: dict,
        num_envs: int = 1,
        *,
        head_hidden_dims: list[int] | tuple[int, ...],
        orientation_latent_dim: int = 16,
        orientation_hidden_dims: list[int] | tuple[int, ...] = (16,),
        img_latent_dim: int = 16,
        img_hidden_dims: list[int] | tuple[int, ...] = (32,),
        graph_latent_dim: int = 16,
        graph_hidden_dims: list[int] | tuple[int, ...] = (32,),
        goal_latent_dim: int = 16,
        goal_hidden_dims: list[int] | tuple[int, ...] = (16,),
        memory_latent_dim: int = 16,
        memory_hidden_dims: list[int] | tuple[int, ...] = (32,),
    ):
        Model.__init__(self, observation_space, action_space, device)
        GaussianMixin.__init__(
            self,
            clip_actions=False,
            clip_log_std=True,
            min_log_std=-5,
            max_log_std=2,
        )
        self.backbone = BranchFeatureBackbone(
            observation_space,
            modules,
            model_cfg,
            orientation_latent_dim=orientation_latent_dim,
            orientation_hidden_dims=orientation_hidden_dims,
            img_latent_dim=img_latent_dim,
            img_hidden_dims=img_hidden_dims,
            graph_latent_dim=graph_latent_dim,
            graph_hidden_dims=graph_hidden_dims,
            goal_latent_dim=goal_latent_dim,
            goal_hidden_dims=goal_hidden_dims,
            memory_latent_dim=memory_latent_dim,
            memory_hidden_dims=memory_hidden_dims,
        )
        self.net = MLP(self.backbone.output_dim, self.num_actions, head_hidden_dims, nn.ReLU, nn.Tanh)
        self.log_std_parameter = nn.Parameter(torch.zeros(self.num_actions))

    def compute(self, inputs, role):
        x = self.backbone(inputs["states"])
        return self.net(x), self.log_std_parameter, {}


class SACBranchQCritic(DeterministicMixin, Model):
    def __init__(
        self,
        observation_space,
        action_space,
        device,
        modules: dict[str, nn.Module],
        model_cfg: dict,
        num_envs: int = 1,
        *,
        head_hidden_dims: list[int] | tuple[int, ...],
        orientation_latent_dim: int = 16,
        orientation_hidden_dims: list[int] | tuple[int, ...] = (16,),
        img_latent_dim: int = 16,
        img_hidden_dims: list[int] | tuple[int, ...] = (32,),
        graph_latent_dim: int = 16,
        graph_hidden_dims: list[int] | tuple[int, ...] = (32,),
        goal_latent_dim: int = 16,
        goal_hidden_dims: list[int] | tuple[int, ...] = (16,),
        memory_latent_dim: int = 16,
        memory_hidden_dims: list[int] | tuple[int, ...] = (32,),
    ):
        Model.__init__(self, observation_space, action_space, device)
        DeterministicMixin.__init__(self, clip_actions=False)
        self.backbone = BranchFeatureBackbone(
            observation_space,
            modules,
            model_cfg,
            orientation_latent_dim=orientation_latent_dim,
            orientation_hidden_dims=orientation_hidden_dims,
            img_latent_dim=img_latent_dim,
            img_hidden_dims=img_hidden_dims,
            graph_latent_dim=graph_latent_dim,
            graph_hidden_dims=graph_hidden_dims,
            goal_latent_dim=goal_latent_dim,
            goal_hidden_dims=goal_hidden_dims,
            memory_latent_dim=memory_latent_dim,
            memory_hidden_dims=memory_hidden_dims,
        )
        self.net = MLP(self.backbone.output_dim + self.num_actions, 1, head_hidden_dims, nn.ReLU)

    def compute(self, inputs, role):
        z = self.backbone(inputs["states"])
        actions = inputs["taken_actions"]
        x = torch.cat([z, actions], dim=-1)
        return self.net(x), {}
