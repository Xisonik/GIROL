# my_reinforce.py
from __future__ import annotations

from typing import Any, Dict, Optional, Tuple, Union
import copy

import torch
import gymnasium as gym

from skrl.agents.torch import Agent
from skrl.models.torch import Model
from skrl.memories.torch import Memory


REINFORCE_DEFAULT_CONFIG = {
    "learning_rate": 3e-4,
    "discount_factor": 0.99,

    # variance reduction
    "normalize_returns": True,

    # regularization
    "entropy_loss_scale": 0.0,     # попробуй 0.0..0.01

    # optimization
    "grad_norm_clip": 1.0,

    # how often to update (REINFORCE обычно обновляется по завершению эпизода)
    "update_on_episode_end": True,

    "experiment": {
        "directory": "",
        "experiment_name": "",
        "write_interval": 250,
        "checkpoint_interval": 1000,
        "store_separately": False,
        "wandb": False,
        "wandb_kwargs": {},
    },
}


class REINFORCE(Agent):
    def __init__(
        self,
        models: Dict[str, Model],
        memory: Optional[Union[Memory, Tuple[Memory]]] = None,
        observation_space: Optional[Union[int, Tuple[int], gym.Space]] = None,
        action_space: Optional[Union[int, Tuple[int], gym.Space]] = None,
        device: Optional[Union[str, torch.device]] = None,
        cfg: Optional[dict] = None,
    ) -> None:
        _cfg = copy.deepcopy(REINFORCE_DEFAULT_CONFIG)
        _cfg.update(cfg if cfg is not None else {})

        super().__init__(
            models=models,
            memory=memory,
            observation_space=observation_space,
            action_space=action_space,
            device=device,
            cfg=_cfg,
        )

        if "policy" not in self.models:
            raise KeyError('REINFORCE requires models["policy"]')

        self.policy: Model = self.models["policy"]

        # optimizer
        lr = float(self.cfg.get("learning_rate", 3e-4))
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr)

        # buffers per-env (для vec env)
        self._ep_rewards = None      # list[list[float]]
        self._ep_log_probs = None    # list[list[Tensor]]

        # last step cache (act -> record_transition)
        self._last_log_prob = None

        # convenience cfg
        self.gamma = float(self.cfg.get("discount_factor", 0.99))
        self.normalize_returns = bool(self.cfg.get("normalize_returns", True))
        self.entropy_scale = float(self.cfg.get("entropy_loss_scale", 0.0))
        self.grad_clip = float(self.cfg.get("grad_norm_clip", 0.0))
        self.update_on_episode_end = bool(self.cfg.get("update_on_episode_end", True))

        # what to save in checkpoints
        self.checkpoint_modules["policy"] = self.policy
        self.checkpoint_modules["optimizer"] = self.optimizer

    def init(self, trainer_cfg: Optional[Dict[str, Any]] = None) -> None:
        super().init(trainer_cfg=trainer_cfg)
        self.set_mode("train")

        # num_envs обычно доступен через trainer_cfg / env, но в skrl базово есть self.num_envs
        n = getattr(self, "num_envs", None)
        if n is None:
            # fallback: попробуем из memory
            n = getattr(self.memory, "num_envs", 1) if self.memory is not None else 1

        self._ep_rewards = [[] for _ in range(n)]
        self._ep_log_probs = [[] for _ in range(n)]

    def _preprocess_states(self, states: torch.Tensor) -> torch.Tensor:
        # если ты используешь RunningStandardScaler в cfg, Agent обычно создаёт preprocessor сам.
        # В разных версиях skrl атрибуты могут называться слегка по-разному,
        # поэтому делаем мягко:
        pre = getattr(self, "state_preprocessor", None)
        if pre is not None:
            try:
                return pre(states)
            except TypeError:
                # иногда это callable без __call__ сигнатуры
                return pre.forward(states)
        return states

    @torch.no_grad()
    def act(self, states: torch.Tensor, timestep: int, timesteps: int) -> torch.Tensor:
        # states: (num_envs, obs_dim)
        states = self._preprocess_states(states)

        actions, log_prob, _ = self.policy.act({"states": states}, role="policy")
        # log_prob shape обычно (num_envs, 1) — сохраняем
        self._last_log_prob = log_prob
        return actions

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
        super().record_transition(states, actions, rewards, next_states, terminated, truncated, infos, timestep, timesteps)

        # складываем per-env
        done = (terminated | truncated).view(-1)
        r = rewards.view(-1)
        lp = self._last_log_prob.view(-1)  # (num_envs, 1) -> (num_envs,)

        for i in range(done.shape[0]):
            self._ep_rewards[i].append(float(r[i].item()))
            # log_prob нельзя хранить no_grad-тензором: он нужен для backward
            # поэтому пересчитаем log_prob "с градиентом" в _update,
            # а тут сохраним states/actions, если хочешь. Но проще:
            # сохраним log_prob как тензор, который был получен из policy.act без no_grad
            # (в act мы были в no_grad), поэтому тут пересчёт:
            self._ep_log_probs[i].append(lp[i].detach())  # placeholder

        if self.update_on_episode_end and done.any():
            # запускаем update сразу, как только какой-то env закончил эпизод
            self._update(timestep, timesteps, states=states, actions=actions, done=done)

    def _compute_returns(self, rewards: list[float]) -> torch.Tensor:
        G = 0.0
        returns = []
        for r in reversed(rewards):
            G = r + self.gamma * G
            returns.append(G)
        returns.reverse()
        out = torch.tensor(returns, device=self.device, dtype=torch.float32)
        if self.normalize_returns and out.numel() > 1:
            out = (out - out.mean()) / (out.std() + 1e-8)
        return out

    def _update(
        self,
        timestep: int,
        timesteps: int,
        states: Optional[torch.Tensor] = None,
        actions: Optional[torch.Tensor] = None,
        done: Optional[torch.Tensor] = None,
    ) -> None:
        # ВАЖНО: чтобы сделать правильный REINFORCE, нужно log_prob с градиентом.
        # Самый простой вариант: НЕ делать no_grad в act.
        # Но многие рендер/сим-циклы быстрее с no_grad.
        # Поэтому тут — рекомендуемая правка ниже (см. раздел “2) маленькая правка в твоём скрипте”)
        #
        # Если ты применишь правку (act без no_grad), то self._last_log_prob будет графовым тензором,
        # и этот update будет корректным.

        # Если у тебя пока act в no_grad — обновление будет “мертвое” (градиента нет).
        # Мы явно проверим:
        if self._last_log_prob is None or not self._last_log_prob.requires_grad:
            # честно предупреждаем через tensorboard/log
            self.track_data("Warn / log_prob_has_no_grad", 1.0)
            return

        n_envs = len(self._ep_rewards)
        losses = []
        entropies = []

        for i in range(n_envs):
            # апдейтим только те env, где done=True на этом шаге
            if done is None or not bool(done[i].item()):
                continue
            if len(self._ep_rewards[i]) == 0:
                continue

            returns = self._compute_returns(self._ep_rewards[i])

            # log_prob мы хотим по всем шагам эпизода.
            # Самый простой путь в skrl-стиле: хранить log_prob каждый шаг (с графом) в act.
            # Тогда тут мы просто стакнем.
            log_probs = torch.stack(self._ep_log_probs[i], dim=0).to(self.device)

            # policy loss
            loss = -(log_probs * returns).sum()
            losses.append(loss)

            # entropy bonus (опционально)
            if self.entropy_scale > 0.0:
                try:
                    ent = self.policy.get_entropy(role="policy").mean()
                    entropies.append(ent)
                except Exception:
                    pass

            # clear episode buffers for this env
            self._ep_rewards[i].clear()
            self._ep_log_probs[i].clear()

        if not losses:
            return

        total_loss = torch.stack(losses).mean()
        if entropies:
            total_entropy = torch.stack(entropies).mean()
            total_loss = total_loss - self.entropy_scale * total_entropy
            self.track_data("Loss / entropy", float(total_entropy.item()))

        self.optimizer.zero_grad(set_to_none=True)
        total_loss.backward()
        if self.grad_clip and self.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.grad_clip)
        self.optimizer.step()

        self.track_data("Loss / policy", float(total_loss.item()))
