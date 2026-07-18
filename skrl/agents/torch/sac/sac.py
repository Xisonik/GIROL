from typing import Any, Mapping, Optional, Tuple, Union

import copy
import itertools
import gymnasium
from packaging import version

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from skrl import config, logger
from skrl.agents.torch import Agent
from skrl.memories.torch import Memory
from skrl.models.torch import Model


# fmt: off
# [start-config-dict-torch]
SAC_DEFAULT_CONFIG = {
    "gradient_steps": 1,            # gradient steps
    "batch_size": 64,               # training batch size

    "discount_factor": 0.99,        # discount factor (gamma)
    "polyak": 0.005,                # soft update hyperparameter (tau)

    "actor_learning_rate": 1e-3,    # actor learning rate
    "critic_learning_rate": 1e-3,   # critic learning rate
    "learning_rate_scheduler": None,        # learning rate scheduler class (see torch.optim.lr_scheduler)
    "learning_rate_scheduler_kwargs": {},   # learning rate scheduler's kwargs (e.g. {"step_size": 1e-3})

    "state_preprocessor": None,             # state preprocessor class (see skrl.resources.preprocessors)
    "state_preprocessor_kwargs": {},        # state preprocessor's kwargs (e.g. {"size": env.observation_space})

    "random_timesteps": 0,          # random exploration steps
    "learning_starts": 0,           # learning starts after this many steps

    "grad_norm_clip": 0,            # clipping coefficient for the norm of the gradients

    "learn_entropy": True,          # learn entropy
    "entropy_learning_rate": 1e-3,  # entropy learning rate
    "initial_entropy_value": 0.2,   # initial entropy value
    "target_entropy": None,         # target entropy

    "rewards_shaper": None,         # rewards shaping function: Callable(reward, timestep, timesteps) -> reward

    "mixed_precision": False,       # enable automatic mixed precision for higher performance

    "experiment": {
        "directory": "",            # experiment's parent directory
        "experiment_name": "",      # experiment name
        "write_interval": "auto",   # TensorBoard writing interval (timesteps)

        "checkpoint_interval": "auto",      # interval for checkpoints (timesteps)
        "store_separately": False,          # whether to store checkpoints separately

        "wandb": False,             # whether to use Weights & Biases
        "wandb_kwargs": {}          # wandb kwargs (see https://docs.wandb.ai/ref/python/init)
    }
}
# [end-config-dict-torch]
# fmt: on


class SAC(Agent):
    def __init__(
        self,
        models: Mapping[str, Model],
        memory: Optional[Union[Memory, Tuple[Memory]]] = None,
        observation_space: Optional[Union[int, Tuple[int], gymnasium.Space]] = None,
        action_space: Optional[Union[int, Tuple[int], gymnasium.Space]] = None,
        device: Optional[Union[str, torch.device]] = None,
        cfg: Optional[dict] = None,
    ) -> None:
        """Soft Actor-Critic (SAC)

        https://arxiv.org/abs/1801.01290

        :param models: Models used by the agent
        :type models: dictionary of skrl.models.torch.Model
        :param memory: Memory to storage the transitions.
                       If it is a tuple, the first element will be used for training and
                       for the rest only the environment transitions will be added
        :type memory: skrl.memory.torch.Memory, list of skrl.memory.torch.Memory or None
        :param observation_space: Observation/state space or shape (default: ``None``)
        :type observation_space: int, tuple or list of int, gymnasium.Space or None, optional
        :param action_space: Action space or shape (default: ``None``)
        :type action_space: int, tuple or list of int, gymnasium.Space or None, optional
        :param device: Device on which a tensor/array is or will be allocated (default: ``None``).
                       If None, the device will be either ``"cuda"`` if available or ``"cpu"``
        :type device: str or torch.device, optional
        :param cfg: Configuration dictionary
        :type cfg: dict

        :raises KeyError: If the models dictionary is missing a required key
        """
        _cfg = copy.deepcopy(SAC_DEFAULT_CONFIG)
        _cfg.update(cfg if cfg is not None else {})
        super().__init__(
            models=models,
            memory=memory,
            observation_space=observation_space,
            action_space=action_space,
            device=device,
            cfg=_cfg,
        )

        # models
        self.policy = self.models.get("policy", None)
        self.critic_1 = self.models.get("critic_1", None)
        self.critic_2 = self.models.get("critic_2", None)
        self.target_critic_1 = self.models.get("target_critic_1", None)
        self.target_critic_2 = self.models.get("target_critic_2", None)

        # checkpoint models
        self.checkpoint_modules["policy"] = self.policy
        self.checkpoint_modules["critic_1"] = self.critic_1
        self.checkpoint_modules["critic_2"] = self.critic_2
        self.checkpoint_modules["target_critic_1"] = self.target_critic_1
        self.checkpoint_modules["target_critic_2"] = self.target_critic_2

        # broadcast models' parameters in distributed runs
        if config.torch.is_distributed:
            logger.info(f"Broadcasting models' parameters")
            if self.policy is not None:
                self.policy.broadcast_parameters()
            if self.critic_1 is not None:
                self.critic_1.broadcast_parameters()
            if self.critic_2 is not None:
                self.critic_2.broadcast_parameters()

        if self.target_critic_1 is not None and self.target_critic_2 is not None:
            # freeze target networks with respect to optimizers (update via .update_parameters())
            self.target_critic_1.freeze_parameters(True)
            self.target_critic_2.freeze_parameters(True)

            # update target networks (hard update)
            # NOTE: We manually copy only .net parameters because critic_1 may have additional
            # registered modules (shared_graph, orientation_module) that target critics don't have
            # in their .parameters() list. This prevents dimension mismatch errors.
            with torch.no_grad():
                # Copy critic_1.net → target_critic_1.net
                if hasattr(self.critic_1, 'net') and hasattr(self.target_critic_1, 'net'):
                    for target_p, source_p in zip(
                        self.target_critic_1.net.parameters(),
                        self.critic_1.net.parameters()
                    ):
                        target_p.data.copy_(source_p.data)
                else:
                    # Fallback to standard update if no .net attribute
                    self.target_critic_1.update_parameters(self.critic_1, polyak=1)
                
                # Copy critic_2.net → target_critic_2.net
                if hasattr(self.critic_2, 'net') and hasattr(self.target_critic_2, 'net'):
                    for target_p, source_p in zip(
                        self.target_critic_2.net.parameters(),
                        self.critic_2.net.parameters()
                    ):
                        target_p.data.copy_(source_p.data)
                else:
                    # Fallback to standard update if no .net attribute
                    self.target_critic_2.update_parameters(self.critic_2, polyak=1)

        # configuration
        self._gradient_steps = self.cfg["gradient_steps"]
        self._batch_size = self.cfg["batch_size"]

        self._discount_factor = self.cfg["discount_factor"]
        self._polyak = self.cfg["polyak"]

        self._actor_learning_rate = self.cfg["actor_learning_rate"]
        self._critic_learning_rate = self.cfg["critic_learning_rate"]
        self._learning_rate_scheduler = self.cfg["learning_rate_scheduler"]

        self._state_preprocessor = self.cfg["state_preprocessor"]

        self._random_timesteps = self.cfg["random_timesteps"]
        self._learning_starts = self.cfg["learning_starts"]

        self._grad_norm_clip = self.cfg["grad_norm_clip"]

        self._entropy_learning_rate = self.cfg["entropy_learning_rate"]
        self._learn_entropy = self.cfg["learn_entropy"]
        self._entropy_coefficient = self.cfg["initial_entropy_value"]

        self._rewards_shaper = self.cfg["rewards_shaper"]

        self._mixed_precision = self.cfg["mixed_precision"]

        # set up automatic mixed precision
        self._device_type = torch.device(device).type
        if version.parse(torch.__version__) >= version.parse("2.4"):
            self.scaler = torch.amp.GradScaler(device=self._device_type, enabled=self._mixed_precision)
        else:
            self.scaler = torch.cuda.amp.GradScaler(enabled=self._mixed_precision)

        # entropy
        if self._learn_entropy:
            self._target_entropy = self.cfg["target_entropy"]
            if self._target_entropy is None:
                if issubclass(type(self.action_space), gymnasium.spaces.Box):
                    self._target_entropy = -np.prod(self.action_space.shape).astype(np.float32)
                elif issubclass(type(self.action_space), gymnasium.spaces.Discrete):
                    self._target_entropy = -self.action_space.n
                else:
                    self._target_entropy = 0

            self.log_entropy_coefficient = torch.log(
                torch.ones(1, device=self.device) * self._entropy_coefficient
            ).requires_grad_(True)
            self.entropy_optimizer = torch.optim.Adam([self.log_entropy_coefficient], lr=self._entropy_learning_rate)

            self.checkpoint_modules["entropy_optimizer"] = self.entropy_optimizer

        # set up optimizers and learning rate schedulers
        if self.policy is not None and self.critic_1 is not None and self.critic_2 is not None:
            self.policy_optimizer = torch.optim.Adam(self.policy.parameters(), lr=self._actor_learning_rate)
            self.critic_optimizer = torch.optim.Adam(
                itertools.chain(self.critic_1.parameters(), self.critic_2.parameters()), lr=self._critic_learning_rate
            )
            if self._learning_rate_scheduler is not None:
                self.policy_scheduler = self._learning_rate_scheduler(
                    self.policy_optimizer, **self.cfg["learning_rate_scheduler_kwargs"]
                )
                self.critic_scheduler = self._learning_rate_scheduler(
                    self.critic_optimizer, **self.cfg["learning_rate_scheduler_kwargs"]
                )

            self.checkpoint_modules["policy_optimizer"] = self.policy_optimizer
            self.checkpoint_modules["critic_optimizer"] = self.critic_optimizer

        # set up preprocessors
        if self._state_preprocessor:
            self._state_preprocessor = self._state_preprocessor(**self.cfg["state_preprocessor_kwargs"])
            self.checkpoint_modules["state_preprocessor"] = self._state_preprocessor
        else:
            self._state_preprocessor = self._empty_preprocessor

    def init(self, trainer_cfg: Optional[Mapping[str, Any]] = None) -> None:
        """Initialize the agent"""
        super().init(trainer_cfg=trainer_cfg)
        self.set_mode("eval")

        # create tensors in memory
        if self.memory is not None:
            self.memory.create_tensor(name="states", size=self.observation_space, dtype=torch.float32)
            self.memory.create_tensor(name="next_states", size=self.observation_space, dtype=torch.float32)
            self.memory.create_tensor(name="actions", size=self.action_space, dtype=torch.float32)
            self.memory.create_tensor(name="rewards", size=1, dtype=torch.float32)
            self.memory.create_tensor(name="terminated", size=1, dtype=torch.bool)
            self.memory.create_tensor(name="truncated", size=1, dtype=torch.bool)

            self._tensors_names = ["states", "actions", "rewards", "next_states", "terminated", "truncated"]

    def act(self, states: torch.Tensor, timestep: int, timesteps: int) -> torch.Tensor:
        """Process the environment's states to make a decision (actions) using the main policy

        :param states: Environment's states
        :type states: torch.Tensor
        :param timestep: Current timestep
        :type timestep: int
        :param timesteps: Number of timesteps
        :type timesteps: int

        :return: Actions
        :rtype: torch.Tensor
        """
        # sample random actions
        # TODO, check for stochasticity
        if timestep < self._random_timesteps:
            return self.policy.random_act({"states": self._state_preprocessor(states)}, role="policy")

        # sample stochastic actions
        with torch.autocast(device_type=self._device_type, enabled=self._mixed_precision):
            actions, _, outputs = self.policy.act({"states": self._state_preprocessor(states)}, role="policy")

        return actions, None, outputs

    def record_transition(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        next_states: torch.Tensor,
        terminated: torch.Tensor,
        truncated: torch.Tensor,
        infos: Any,
        timestep: int,
        timesteps: int,
    ) -> None:
        """Record an environment transition in memory

        :param states: Observations/states of the environment used to make the decision
        :type states: torch.Tensor
        :param actions: Actions taken by the agent
        :type actions: torch.Tensor
        :param rewards: Instant rewards achieved by the current actions
        :type rewards: torch.Tensor
        :param next_states: Next observations/states of the environment
        :type next_states: torch.Tensor
        :param terminated: Signals to indicate that episodes have terminated
        :type terminated: torch.Tensor
        :param truncated: Signals to indicate that episodes have been truncated
        :type truncated: torch.Tensor
        :param infos: Additional information about the environment
        :type infos: Any type supported by the environment
        :param timestep: Current timestep
        :type timestep: int
        :param timesteps: Number of timesteps
        :type timesteps: int
        """
        super().record_transition(
            states, actions, rewards, next_states, terminated, truncated, infos, timestep, timesteps
        )

        if self.memory is not None:
            # reward shaping
            if self._rewards_shaper is not None:
                rewards = self._rewards_shaper(rewards, timestep, timesteps)
            # storage transition in memory
            self.memory.add_samples(
                states=states,
                actions=actions,
                rewards=rewards,
                next_states=next_states,
                terminated=terminated,
                truncated=truncated,
            )
            for memory in self.secondary_memories:
                memory.add_samples(
                    states=states,
                    actions=actions,
                    rewards=rewards,
                    next_states=next_states,
                    terminated=terminated,
                    truncated=truncated,
                )

    def pre_interaction(self, timestep: int, timesteps: int) -> None:
        """Callback called before the interaction with the environment

        :param timestep: Current timestep
        :type timestep: int
        :param timesteps: Number of timesteps
        :type timesteps: int
        """
        pass

    def post_interaction(self, timestep: int, timesteps: int) -> None:
        """Callback called after the interaction with the environment

        :param timestep: Current timestep
        :type timestep: int
        :param timesteps: Number of timesteps
        :type timesteps: int
        """
        if timestep >= self._learning_starts:
            self.set_mode("train")
            self._update(timestep, timesteps)
            self.set_mode("eval")

        # write tracking data and checkpoints
        super().post_interaction(timestep, timesteps)

    def _update(self, timestep: int, timesteps: int) -> None:
        """Algorithm's main update step

        :param timestep: Current timestep
        :type timestep: int
        :param timesteps: Number of timesteps
        :type timesteps: int
        """

        # gradient steps
        DEBUG = False
        for gradient_step in range(self._gradient_steps):

            # sample a batch from memory
            (
                sampled_states,
                sampled_actions,
                sampled_rewards,
                sampled_next_states,
                sampled_terminated,
                sampled_truncated,
            ) = self.memory.sample(names=self._tensors_names, batch_size=self._batch_size)[0]

            with torch.autocast(device_type=self._device_type, enabled=self._mixed_precision):

                sampled_states = self._state_preprocessor(sampled_states, train=True)
                sampled_next_states = self._state_preprocessor(sampled_next_states, train=True)

                # compute target values
                with torch.no_grad():
                    next_actions, next_log_prob, _ = self.policy.act({"states": sampled_next_states}, role="policy")

                    target_q1_values, _, _ = self.target_critic_1.act(
                        {"states": sampled_next_states, "taken_actions": next_actions}, role="target_critic_1"
                    )
                    target_q2_values, _, _ = self.target_critic_2.act(
                        {"states": sampled_next_states, "taken_actions": next_actions}, role="target_critic_2"
                    )
                    target_q_values = (
                        torch.min(target_q1_values, target_q2_values) - self._entropy_coefficient * next_log_prob
                    )
                    target_values = (
                        sampled_rewards
                        + self._discount_factor
                        * (sampled_terminated | sampled_truncated).logical_not()
                        * target_q_values
                    )

                # compute critic loss
                critic_1_values, _, critic_1_outputs = self.critic_1.act(
                    {"states": sampled_states, "taken_actions": sampled_actions}, role="critic_1"
                )
                critic_2_values, _, _ = self.critic_2.act(
                    {"states": sampled_states, "taken_actions": sampled_actions}, role="critic_2"
                )

                critic_loss = (
                    F.mse_loss(critic_1_values, target_values) + F.mse_loss(critic_2_values, target_values)
                ) / 2

            # ============ ORIENTATION LOSS INTEGRATION ============
            if 'orientation_loss' in critic_1_outputs:
                orient_loss = critic_1_outputs['orientation_loss']
                # orient_accuracy = critic_1_outputs.get('orientation_accuracy', None)
                localization_weight = 1.2 #1.2  # НАЧНИ С МАЛОГО ВЕСА
                # if orient_accuracy.item() > 0.90:
                #     localization_weight = 0.6
                # if orient_accuracy.item() > 0.97:
                #     localization_weight = 0.1
                # КРИТИЧНО: ДОБАВЛЯЕМ К CRITIC LOSS
                # print("CHECK MY: ", critic_loss, orient_loss, localization_weight * orient_loss)
                critic_loss = critic_loss + localization_weight * orient_loss
                
                # Логирование (только первый gradient step каждого update)
                if self.write_interval > 0 and gradient_step == 0:
                    orient_accuracy = critic_1_outputs.get('orientation_accuracy', None)
                    self.track_data("Localization / Orientation Loss", orient_loss.item())
                    if orient_accuracy is not None:
                        self.track_data("Localization / Orientation Accuracy", orient_accuracy.item())
                    # DEBUG: детальная информация
                    if DEBUG and timestep % 1000 == 0:
                        print(f"\n[SAC._update] Timestep {timestep}:")
                        print(f"  Critic loss (base): {((F.mse_loss(critic_1_values, target_values) + F.mse_loss(critic_2_values, target_values)) / 2).item():.4f}")
                        print(f"  Orientation loss: {orient_loss.item():.4f}")
                        print(f"  Weighted orient: {(localization_weight * orient_loss).item():.4f}")
                        print(f"  Total critic loss: {critic_loss.item():.4f}")
                        if orient_accuracy is not None:
                            print(f"  Orientation accuracy: {orient_accuracy.item():.4f}")
                        
                        # Проверяем распределение labels
                        if 'orientation_label' in critic_1_outputs:
                            labels = critic_1_outputs['orientation_label']
                            label_counts = torch.bincount(labels, minlength=36)
                            label_std = label_counts.float().std().item()
                            print(f"  Label distribution std: {label_std:.2f} (higher is better, ~10 is good)")
                            
                            # Топ-3 наиболее частых bins
                            top3_bins = torch.topk(label_counts, k=3)
                            print(f"  Top 3 bins: {top3_bins.indices.tolist()} with counts {top3_bins.values.tolist()}")
            
            elif DEBUG and timestep % 1000 == 0 and gradient_step == 0:
                print(f"\n[SAC._update] ⚠️  WARNING: 'orientation_loss' not in critic_1_outputs!")
                print(f"  Available keys: {list(critic_1_outputs.keys())}")
            
            # ============ END ORIENTATION LOSS ============

            # optimization step (critic)
            self.critic_optimizer.zero_grad()
            self.scaler.scale(critic_loss).backward()

            if config.torch.is_distributed:
                self.critic_1.reduce_parameters()
                self.critic_2.reduce_parameters()

            if self._grad_norm_clip > 0:
                self.scaler.unscale_(self.critic_optimizer)
                nn.utils.clip_grad_norm_(
                    itertools.chain(self.critic_1.parameters(), self.critic_2.parameters()), self._grad_norm_clip
                )

            self.scaler.step(self.critic_optimizer)

            with torch.autocast(device_type=self._device_type, enabled=self._mixed_precision):
                # compute policy (actor) loss
                actions, log_prob, _ = self.policy.act({"states": sampled_states}, role="policy")
                critic_1_values, _, outputs = self.critic_1.act(
                    {"states": sampled_states, "taken_actions": actions}, role="critic_1"
                )
                critic_2_values, _, _ = self.critic_2.act(
                    {"states": sampled_states, "taken_actions": actions}, role="critic_2"
                )

                policy_loss = (
                    self._entropy_coefficient * log_prob - torch.min(critic_1_values, critic_2_values)
                ).mean()

            # optimization step (policy)
            self.policy_optimizer.zero_grad()
            self.scaler.scale(policy_loss).backward()

            if config.torch.is_distributed:
                self.policy.reduce_parameters()

            if self._grad_norm_clip > 0:
                self.scaler.unscale_(self.policy_optimizer)
                nn.utils.clip_grad_norm_(self.policy.parameters(), self._grad_norm_clip)

            self.scaler.step(self.policy_optimizer)

            # entropy learning
            if self._learn_entropy:
                with torch.autocast(device_type=self._device_type, enabled=self._mixed_precision):
                    # compute entropy loss
                    entropy_loss = -(self.log_entropy_coefficient * (log_prob + self._target_entropy).detach()).mean()

                # optimization step (entropy)
                self.entropy_optimizer.zero_grad()
                self.scaler.scale(entropy_loss).backward()
                self.scaler.step(self.entropy_optimizer)

                # compute entropy coefficient
                self._entropy_coefficient = torch.exp(self.log_entropy_coefficient.detach())

            self.scaler.update()  # called once, after optimizers have been stepped

            # update target networks (polyak averaging)
            # NOTE: We manually update only .net parameters for the same reason as in __init__
            with torch.no_grad():
                # Polyak averaging for critic_1 → target_critic_1
                if hasattr(self.critic_1, 'net') and hasattr(self.target_critic_1, 'net'):
                    for target_p, source_p in zip(
                        self.target_critic_1.net.parameters(),
                        self.critic_1.net.parameters()
                    ):
                        target_p.data.mul_(1.0 - self._polyak)
                        target_p.data.add_(source_p.data, alpha=self._polyak)
                else:
                    self.target_critic_1.update_parameters(self.critic_1, polyak=self._polyak)
                
                # Polyak averaging for critic_2 → target_critic_2
                if hasattr(self.critic_2, 'net') and hasattr(self.target_critic_2, 'net'):
                    for target_p, source_p in zip(
                        self.target_critic_2.net.parameters(),
                        self.critic_2.net.parameters()
                    ):
                        target_p.data.mul_(1.0 - self._polyak)
                        target_p.data.add_(source_p.data, alpha=self._polyak)
                else:
                    self.target_critic_2.update_parameters(self.critic_2, polyak=self._polyak)

            # update learning rate
            if self._learning_rate_scheduler:
                self.policy_scheduler.step()
                self.critic_scheduler.step()

            # record data
            if self.write_interval > 0:
                self.track_data("Loss / Policy loss", policy_loss.item())
                self.track_data("Loss / Critic loss", critic_loss.item())

                self.track_data("Q-network / Q1 (max)", torch.max(critic_1_values).item())
                self.track_data("Q-network / Q1 (min)", torch.min(critic_1_values).item())
                self.track_data("Q-network / Q1 (mean)", torch.mean(critic_1_values).item())

                self.track_data("Q-network / Q2 (max)", torch.max(critic_2_values).item())
                self.track_data("Q-network / Q2 (min)", torch.min(critic_2_values).item())
                self.track_data("Q-network / Q2 (mean)", torch.mean(critic_2_values).item())

                self.track_data("Target / Target (max)", torch.max(target_values).item())
                self.track_data("Target / Target (min)", torch.min(target_values).item())
                self.track_data("Target / Target (mean)", torch.mean(target_values).item())
                if 'orientation_accuracy' in critic_1_outputs:
                    self.track_data("Localization / Accuracy (±10°)", critic_1_outputs['orientation_accuracy'].item())
                if 'orientation_accuracy_strict' in critic_1_outputs:
                    self.track_data("Localization / Accuracy (strict)", critic_1_outputs['orientation_accuracy_strict'].item())
                if 'orientation_confidence' in critic_1_outputs:
                    self.track_data("Localization / Confidence", critic_1_outputs['orientation_confidence'].item())
                if 'orientation_entropy' in critic_1_outputs:
                    self.track_data("Localization / Entropy", critic_1_outputs['orientation_entropy'].item())
                if 'orientation_mean_error_deg' in critic_1_outputs:
                    self.track_data("Localization / Mean Error (deg)", critic_1_outputs['orientation_mean_error_deg'].item())

                if self._learn_entropy:
                    self.track_data("Loss / Entropy loss", entropy_loss.item())
                    self.track_data("Coefficient / Entropy coefficient", self._entropy_coefficient.item())

                if self._learning_rate_scheduler:
                    self.track_data("Learning / Policy learning rate", self.policy_scheduler.get_last_lr()[0])
                    self.track_data("Learning / Critic learning rate", self.critic_scheduler.get_last_lr()[0])

    # def _update(self, timestep: int, timesteps: int) -> None:
    #     """Algorithm's main update step

    #     :param timestep: Current timestep
    #     :type timestep: int
    #     :param timesteps: Number of timesteps
    #     :type timesteps: int
    #     """

    #     # gradient steps
    #     for gradient_step in range(self._gradient_steps):

    #         # sample a batch from memory
    #         (
    #             sampled_states,
    #             sampled_actions,
    #             sampled_rewards,
    #             sampled_next_states,
    #             sampled_terminated,
    #             sampled_truncated,
    #         ) = self.memory.sample(names=self._tensors_names, batch_size=self._batch_size)[0]

    #         with torch.autocast(device_type=self._device_type, enabled=self._mixed_precision):

    #             sampled_states = self._state_preprocessor(sampled_states, train=True)
    #             sampled_next_states = self._state_preprocessor(sampled_next_states, train=True)

    #             # compute target values
    #             with torch.no_grad():
    #                 next_actions, next_log_prob, _ = self.policy.act({"states": sampled_next_states}, role="policy")

    #                 target_q1_values, _, critic_1_outputs = self.target_critic_1.act(
    #                     {"states": sampled_next_states, "taken_actions": next_actions}, role="target_critic_1"
    #                 )
    #                 target_q2_values, _, _ = self.target_critic_2.act(
    #                     {"states": sampled_next_states, "taken_actions": next_actions}, role="target_critic_2"
    #                 )
    #                 target_q_values = (
    #                     torch.min(target_q1_values, target_q2_values) - self._entropy_coefficient * next_log_prob
    #                 )
    #                 target_values = (
    #                     sampled_rewards
    #                     + self._discount_factor
    #                     * (sampled_terminated | sampled_truncated).logical_not()
    #                     * target_q_values
    #                 )

    #             # compute critic loss
    #             critic_1_values, _, critic_1_outputs = self.critic_1.act(
    #                 {"states": sampled_states, "taken_actions": sampled_actions}, role="critic_1"
    #             )
    #             critic_2_values, _, _ = self.critic_2.act(
    #                 {"states": sampled_states, "taken_actions": sampled_actions}, role="critic_2"
    #             )

    #             critic_loss = (
    #                 F.mse_loss(critic_1_values, target_values) + F.mse_loss(critic_2_values, target_values)
    #             ) / 2

    #         localization_weight = 0.3
    #         if 'orientation_logits' in critic_1_outputs:
    #             loc_loss = F.cross_entropy(
    #                 critic_1_outputs['orientation_logits'],
    #                 critic_1_outputs['orientation_label'].squeeze(-1)
    #             )
    #             # critic_loss = critic_loss + localization_weight * loc_loss
                
    #             # # Логирование
    #             if self.write_interval > 0 and gradient_step == 0:
    #                 pred_bins = torch.argmax(critic_1_outputs['orientation_logits'], dim=-1)
    #                 accuracy = (pred_bins == critic_1_outputs['orientation_label']).float().mean()
    #                 self.track_data("Localization / Orientation Accuracy", accuracy.item())
    #                 self.track_data("Localization / Orientation Loss", loc_loss.item())
    #         # optimization step (critic)
    #         self.critic_optimizer.zero_grad()
    #         self.scaler.scale(critic_loss).backward()

    #         if config.torch.is_distributed:
    #             self.critic_1.reduce_parameters()
    #             self.critic_2.reduce_parameters()

    #         if self._grad_norm_clip > 0:
    #             self.scaler.unscale_(self.critic_optimizer)
    #             nn.utils.clip_grad_norm_(
    #                 itertools.chain(self.critic_1.parameters(), self.critic_2.parameters()), self._grad_norm_clip
    #             )

    #         self.scaler.step(self.critic_optimizer)

    #         with torch.autocast(device_type=self._device_type, enabled=self._mixed_precision):
    #             # compute policy (actor) loss
    #             actions, log_prob, _ = self.policy.act({"states": sampled_states}, role="policy")
    #             critic_1_values, _, outputs = self.critic_1.act(
    #                 {"states": sampled_states, "taken_actions": actions}, role="critic_1"
    #             )
    #             critic_2_values, _, _ = self.critic_2.act(
    #                 {"states": sampled_states, "taken_actions": actions}, role="critic_2"
    #             )

    #             policy_loss = (
    #                 self._entropy_coefficient * log_prob - torch.min(critic_1_values, critic_2_values)
    #             ).mean()

    #         # optimization step (policy)
    #         self.policy_optimizer.zero_grad()
    #         self.scaler.scale(policy_loss).backward()

    #         if config.torch.is_distributed:
    #             self.policy.reduce_parameters()

    #         if self._grad_norm_clip > 0:
    #             self.scaler.unscale_(self.policy_optimizer)
    #             nn.utils.clip_grad_norm_(self.policy.parameters(), self._grad_norm_clip)

    #         self.scaler.step(self.policy_optimizer)

    #         # entropy learning
    #         if self._learn_entropy:
    #             with torch.autocast(device_type=self._device_type, enabled=self._mixed_precision):
    #                 # compute entropy loss
    #                 entropy_loss = -(self.log_entropy_coefficient * (log_prob + self._target_entropy).detach()).mean()

    #             # optimization step (entropy)
    #             self.entropy_optimizer.zero_grad()
    #             self.scaler.scale(entropy_loss).backward()
    #             self.scaler.step(self.entropy_optimizer)

    #             # compute entropy coefficient
    #             self._entropy_coefficient = torch.exp(self.log_entropy_coefficient.detach())

    #         self.scaler.update()  # called once, after optimizers have been stepped

    #         # update target networks
    #         self.target_critic_1.update_parameters(self.critic_1, polyak=self._polyak)
    #         self.target_critic_2.update_parameters(self.critic_2, polyak=self._polyak)

    #         # update learning rate
    #         if self._learning_rate_scheduler:
    #             self.policy_scheduler.step()
    #             self.critic_scheduler.step()

    #         # record data
    #         if self.write_interval > 0:
    #             self.track_data("Loss / Policy loss", policy_loss.item())
    #             self.track_data("Loss / Critic loss", critic_loss.item())

    #             self.track_data("Q-network / Q1 (max)", torch.max(critic_1_values).item())
    #             self.track_data("Q-network / Q1 (min)", torch.min(critic_1_values).item())
    #             self.track_data("Q-network / Q1 (mean)", torch.mean(critic_1_values).item())

    #             self.track_data("Q-network / Q2 (max)", torch.max(critic_2_values).item())
    #             self.track_data("Q-network / Q2 (min)", torch.min(critic_2_values).item())
    #             self.track_data("Q-network / Q2 (mean)", torch.mean(critic_2_values).item())

    #             self.track_data("Target / Target (max)", torch.max(target_values).item())
    #             self.track_data("Target / Target (min)", torch.min(target_values).item())
    #             self.track_data("Target / Target (mean)", torch.mean(target_values).item())

    #             if self._learn_entropy:
    #                 self.track_data("Loss / Entropy loss", entropy_loss.item())
    #                 self.track_data("Coefficient / Entropy coefficient", self._entropy_coefficient.item())

    #             if self._learning_rate_scheduler:
    #                 self.track_data("Learning / Policy learning rate", self.policy_scheduler.get_last_lr()[0])
    #                 self.track_data("Learning / Critic learning rate", self.critic_scheduler.get_last_lr()[0])
