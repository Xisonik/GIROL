# -*- coding: utf-8 -*-
"""Common skrl launcher for IsaacLab experiments.

Subclass responsibilities:
    - choose skrl Agent class
    - choose default skrl config
    - map cfg["agent"] fields into skrl cfg
    - build model dict expected by the agent

The environment/module/config/aux/logging path is shared.
"""

from __future__ import annotations

import argparse
import gc
import importlib
import os
import sys
from copy import deepcopy
from enum import Enum
from pathlib import Path
from typing import Any

import gymnasium as gym
import torch
import torch.nn as nn
from skrl.envs.loaders.torch import load_isaaclab_env
from skrl.envs.wrappers.torch import wrap_env
from skrl.memories.torch import RandomMemory
from skrl.trainers.torch import SequentialTrainer
from skrl.utils import set_seed

ALGOS_DIR = Path(__file__).resolve().parents[1]
ISAACLAB_ROOT = ALGOS_DIR.parent.parent
CONFIG_DIR = ALGOS_DIR / "configs"
sys.path.insert(0, str(ALGOS_DIR))

from configs.config_utils import (  # noqa: E402
    experiment_dir,
    export_env_config,
    get_by_path,
    make_exp_name,
    resolve_config,
    save_json,
    validate_config,
)
from configs.clock import reset_global_step  # noqa: E402
from models.logging_utils import save_experiment_logs  # noqa: E402
from models.preprocessors import DictRunningStandardScaler  # noqa: E402
from perception.orientation_module import print_orientation_accuracy  # noqa: E402

DEFAULT_CONFIG = CONFIG_DIR / "base.json"


class EvalActionSource(str, Enum):
    """Source used by skrl's evaluation loop to select an action.

    RETURNED_ACTION:
        Use the primary action returned by agent.act(...). This is required by
        discrete value-based agents such as DQN/DDQN whose auxiliary output may
        be None.

    MEAN_ACTION:
        Use the deterministic mean action exposed by stochastic policies such
        as A2C/PPO/SAC.
    """

    RETURNED_ACTION = "returned_action"
    MEAN_ACTION = "mean_action"


def recurrent_enabled(cfg: dict) -> bool:
    return bool(cfg["model"].get("recurrent", {}).get("enabled", False))


def parse_common_args():
    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--folder", default=None)
    parser.add_argument("--headless", type=int, choices=[0, 1], default=None)
    parser.add_argument("--video", type=int, choices=[0, 1], default=None)
    parser.add_argument("--eval", type=int, choices=[0, 1], default=None)

    # Optional runtime overrides. If omitted, values are read from cfg["paths"].
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--state-preprocessor-checkpoint", default=None)

    args, unknown = parser.parse_known_args()
    sys.argv = [sys.argv[0], *unknown]
    return args


def import_class(class_path: str):
    module_name, attr = class_path.rsplit(".", 1)
    return getattr(importlib.import_module(module_name), attr)


def obs_dim(observation_space, key: str) -> int:
    return int(observation_space.spaces[key].shape[0])


def resolve_runtime_value(value, cfg: dict, env):
    if isinstance(value, dict):
        if set(value.keys()) == {"cfg"}:
            return get_by_path(cfg, value["cfg"])
        if set(value.keys()) == {"obs_dim"}:
            return obs_dim(env.observation_space, value["obs_dim"])
        if set(value.keys()) == {"runtime"}:
            if value["runtime"] == "env":
                return env
            raise ValueError(f"Unsupported runtime ref: {value['runtime']!r}")
        return {k: resolve_runtime_value(v, cfg, env) for k, v in value.items()}
    if isinstance(value, list):
        return [resolve_runtime_value(v, cfg, env) for v in value]
    return value


def _as_optional_path(value: Any) -> Path | None:
    """Normalize optional checkpoint path values from config/CLI.

    Accepted disabled values: None, "", "null", "none", "false".
    Relative paths are resolved from ISAACLAB_ROOT because run() switches cwd there.
    """
    if value is None:
        return None

    if isinstance(value, str):
        text = value.strip()
        if text == "" or text.lower() in {"null", "none", "false"}:
            return None
        path = Path(text).expanduser()
    else:
        path = Path(value).expanduser()

    if not path.is_absolute():
        path = ISAACLAB_ROOT / path
    return path.resolve()


class BaseSkrlTrain:
    algo_name = "base"
    supports_recurrent = False

    # Most actor-policy agents expose mean_actions during evaluation.
    # Value-based discrete runners must override this explicitly.
    eval_action_source = EvalActionSource.MEAN_ACTION

    def __init__(self):
        self.args = parse_common_args()
        self.cfg: dict | None = None
        self.env = None
        self.modules: dict[str, nn.Module] = {}
        self.models: dict[str, nn.Module] = {}
        self.agent = None
        self.memory = None
        self.trainer = None

    def agent_class(self, cfg: dict):
        raise NotImplementedError

    def default_skrl_cfg(self) -> dict:
        raise NotImplementedError

    def memory_size(self, cfg: dict) -> int:
        raise NotImplementedError

    def build_models(self, env, cfg: dict, modules: dict[str, nn.Module], device) -> dict[str, nn.Module]:
        raise NotImplementedError

    def build_skrl_cfg(self, cfg: dict, env, device, exp_dir: Path) -> dict:
        raise NotImplementedError

    def validate_algorithm_contract(self, cfg: dict, env=None) -> None:
        pass

    def run(self) -> None:
        os.chdir(ISAACLAB_ROOT)

        cfg = resolve_config(self.args.config)
        self.apply_cli_overrides(cfg)
        validate_config(cfg, self.algo_name)
        self.validate_common_config(cfg)
        self.validate_algorithm_contract(cfg)
        self.cfg = cfg

        set_seed(int(cfg["run"]["seed"]))
        reset_global_step(int(cfg["run"].get("start_step", 0)))
        export_env_config(cfg)

        num_envs = self.resolve_num_envs(cfg)
        cli_args = self.build_isaac_cli_args(cfg)

        env = self.load_env(cfg, num_envs, cli_args)
        env = self.wrap_video_if_needed(env, cfg)
        env = wrap_env(env)
        self.env = env
        device = env.device

        self.validate_algorithm_contract(cfg, env)

        self.modules = self.build_model_modules(env, cfg, device)
        self.models = self.build_models(env, cfg, self.modules, device)

        exp_name = make_exp_name(cfg)
        exp_dir = experiment_dir(cfg, exp_name)

        self.memory = RandomMemory(
            memory_size=self.memory_size(cfg),
            num_envs=env.num_envs,
            device=device,
        )

        skrl_cfg = self.build_skrl_cfg(cfg, env, device, exp_dir)
        skrl_cfg = self.adapt_agent_cfg_for_run_mode(skrl_cfg, cfg)
        AgentClass = self.agent_class(cfg)

        print(f"[{self.algo_name.upper()}] task={cfg['run']['task_name']} exp={exp_name}", flush=True)
        print(f"[{self.algo_name.upper()}] logs={exp_dir}", flush=True)
        print(f"[{self.algo_name.upper()}] modules={sorted(self.modules.keys())}", flush=True)
        print(f"[{self.algo_name.upper()}] agent={AgentClass.__name__}", flush=True)
        if cfg["run"].get("eval", False):
            print(
                f"[{self.algo_name.upper()}] eval_action_source={self.eval_action_source.value}",
                flush=True,
            )

        save_json(cfg, exp_dir / "config.json")
        save_experiment_logs(
            exp_dir=exp_dir,
            cfg=cfg,
            env=env,
            models=self.models,
            pipeline_modules=self.modules,
        )

        self.agent = AgentClass(
            models=self.models,
            memory=self.memory,
            cfg=skrl_cfg,
            observation_space=env.observation_space,
            action_space=env.action_space,
            device=device,
        )

        if recurrent_enabled(cfg) and not hasattr(self.agent, "scheduler"):
            # Some skrl recurrent agents expect this attribute even when no scheduler is configured.
            self.agent.scheduler = None

        self.maybe_load_pretrained_weights(cfg, device)
        self.attach_aux_trainer(self.agent, env, cfg, self.modules, device, exp_dir)

        self.trainer = SequentialTrainer(
            cfg=self.build_trainer_cfg(cfg),
            env=env,
            agents=self.agent,
        )

        try:
            if cfg["run"].get("eval", False):
                self.trainer.eval()
            else:
                self.trainer.train()

            self.print_final_env_metrics(env, cfg)
        finally:
            self.close()

    def adapt_agent_cfg_for_run_mode(self, skrl_cfg: dict, cfg: dict) -> dict:
        """Apply mode-wide agent settings without knowing the algorithm details.

        Evaluation must not execute a generic random warm-up. Algorithm-specific
        exploration (for example DDQN epsilon-greedy) remains the responsibility
        of the corresponding runner override.
        """
        if cfg["run"].get("eval", False) and "random_timesteps" in skrl_cfg:
            skrl_cfg["random_timesteps"] = 0
        return skrl_cfg

    def build_trainer_cfg(self, cfg: dict) -> dict:
        """Translate the runner's semantic action contract to skrl trainer cfg."""
        source = self.eval_action_source
        if not isinstance(source, EvalActionSource):
            raise TypeError(
                f"{type(self).__name__}.eval_action_source must be an "
                f"EvalActionSource, got {source!r}"
            )

        return {
            "timesteps": int(cfg["run"]["timesteps"]),
            # In skrl this flag controls whether eval consumes outputs[0]
            # or outputs[-1]["mean_actions"]. It does not by itself control
            # DDQN epsilon-greedy exploration.
            "stochastic_evaluation": (
                source is EvalActionSource.RETURNED_ACTION
            ),
        }

    def apply_cli_overrides(self, cfg: dict) -> None:
        if self.args.folder is not None:
            cfg["run"]["folder"] = self.args.folder
        if self.args.headless is not None:
            cfg["run"]["headless"] = bool(self.args.headless)
        if self.args.video is not None:
            cfg["run"]["video"] = bool(self.args.video)
        if self.args.eval is not None:
            cfg["run"]["eval"] = bool(self.args.eval)
        if self.args.checkpoint is not None:
            cfg.setdefault("paths", {})["agent_checkpoint"] = self.args.checkpoint
        if self.args.state_preprocessor_checkpoint is not None:
            cfg.setdefault("paths", {})["state_preprocessor_checkpoint"] = self.args.state_preprocessor_checkpoint

    def validate_common_config(self, cfg: dict) -> None:
        for section in ["run", "agent", "model", "env", "paths"]:
            if section not in cfg:
                raise ValueError(f"Missing config section: {section}")

        if cfg["run"].get("algo") != self.algo_name:
            raise ValueError(f"Runner algo={self.algo_name}, config run.algo={cfg['run'].get('algo')}")

        if not isinstance(self.eval_action_source, EvalActionSource):
            raise TypeError(
                f"{type(self).__name__}.eval_action_source must be an "
                f"EvalActionSource, got {self.eval_action_source!r}"
            )

        if int(cfg["run"].get("num_envs", 0)) <= 0:
            raise ValueError("run.num_envs must be positive")

        model = cfg["model"]
        if recurrent_enabled(cfg) and not self.supports_recurrent:
            raise ValueError(f"{self.algo_name} runner does not support recurrent models")

        if not isinstance(model.get("modules", {}), dict):
            raise ValueError("model.modules must be a dict")

    def resolve_num_envs(self, cfg: dict) -> int:
        num_envs = int(cfg["run"]["num_envs"])
        if cfg["run"].get("video", False) or cfg["run"].get("eval", False) or not cfg["run"].get("headless", True):
            num_envs = 1
        return num_envs

    def build_isaac_cli_args(self, cfg: dict) -> list[str]:
        cli_args = ["--enable_cameras"] if cfg["env"].get("camera", True) else []
        if cfg["run"].get("video", False):
            cli_args += ["--video", "--livestream", "2"]
        return cli_args

    def load_env(self, cfg: dict, num_envs: int, cli_args: list[str]):
        if bool(cfg["run"].get("headless", True)):
            return load_isaaclab_env(
                task_name=cfg["run"]["task_name"],
                headless=True,
                num_envs=num_envs,
                cli_args=cli_args,
            )
        return load_isaaclab_env(
            task_name=cfg["run"]["task_name"],
            num_envs=num_envs,
            cli_args=cli_args,
        )

    def wrap_video_if_needed(self, env, cfg: dict):
        if not cfg["run"].get("video", False):
            return env
        from gymnasium.wrappers import RecordVideo
        return RecordVideo(
            env,
            video_folder="logs/skrl/videos",
            name_prefix="aloha_eval",
            episode_trigger=lambda ep: True,
        )

    def build_model_modules(self, env, cfg: dict, device) -> dict[str, nn.Module]:
        modules: dict[str, nn.Module] = {}
        for ref, spec in sorted(cfg["model"].get("modules", {}).items()):
            cls = import_class(spec["class_path"])
            kwargs = resolve_runtime_value(spec.get("kwargs", {}), cfg, env)
            module = cls(**kwargs)
            if isinstance(module, nn.Module):
                module = module.to(device)
                if bool(spec.get("eval", False)):
                    module.eval()
            modules[ref] = module
        return modules

    def build_one_model(self, env, cfg: dict, modules: dict[str, nn.Module], device, spec: dict):
        cls = import_class(spec["class_path"])
        kwargs = resolve_runtime_value(spec.get("kwargs", {}), cfg, env)
        return cls(
            env.observation_space,
            env.action_space,
            device,
            modules=modules,
            model_cfg=cfg["model"],
            num_envs=env.num_envs,
            **kwargs,
        )

    def base_agent_cfg(self) -> dict:
        return deepcopy(self.default_skrl_cfg())

    def apply_common_skrl_cfg(self, skrl_cfg: dict, cfg: dict, env, device, exp_dir: Path) -> dict:
        agent_cfg = cfg["agent"]
        if agent_cfg.get("normalize_img", True):
            skrl_cfg["state_preprocessor"] = DictRunningStandardScaler
            skrl_cfg["state_preprocessor_kwargs"] = {
                "size": env.observation_space,
                "img_space": env.observation_space["img"],
                "device": device,
            }

        skrl_cfg["experiment"]["directory"] = str(exp_dir.parent)
        skrl_cfg["experiment"]["experiment_name"] = exp_dir.name
        skrl_cfg["experiment"]["write_interval"] = int(cfg["run"].get("write_interval", 10))
        skrl_cfg["experiment"]["checkpoint_interval"] = int(cfg["run"].get("checkpoint_interval", 1000))
        return skrl_cfg

    def maybe_load_pretrained_weights(self, cfg: dict, device) -> None:
        """Load agent and optional state preprocessor checkpoints from cfg["paths"].

        Contract:
            paths.agent_checkpoint = null      -> start from scratch
            paths.agent_checkpoint = "...pt"   -> self.agent.load("...pt") before train/eval

        The model architecture and observation/action spaces must match the checkpoint.
        """
        paths_cfg = cfg.get("paths", {}) or {}

        agent_checkpoint = _as_optional_path(paths_cfg.get("agent_checkpoint", None))
        if agent_checkpoint is not None:
            if not agent_checkpoint.exists():
                raise FileNotFoundError(f"agent checkpoint not found: {agent_checkpoint}")
            if self.agent is None:
                raise RuntimeError("agent is not initialized before checkpoint loading")

            print(f"[{self.algo_name.upper()}] loading agent checkpoint: {agent_checkpoint}", flush=True)
            self.agent.load(str(agent_checkpoint))

        preprocessor_checkpoint = _as_optional_path(paths_cfg.get("state_preprocessor_checkpoint", None))
        if preprocessor_checkpoint is not None:
            if not preprocessor_checkpoint.exists():
                raise FileNotFoundError(f"state preprocessor checkpoint not found: {preprocessor_checkpoint}")
            if self.agent is None:
                raise RuntimeError("agent is not initialized before state preprocessor loading")
            if not hasattr(self.agent, "_state_preprocessor") or self.agent._state_preprocessor is None:
                raise RuntimeError(
                    "paths.state_preprocessor_checkpoint was set, "
                    "but this agent has no _state_preprocessor. "
                    "Check agent.normalize_img / state_preprocessor config."
                )

            print(f"[{self.algo_name.upper()}] loading state preprocessor: {preprocessor_checkpoint}", flush=True)
            payload = torch.load(str(preprocessor_checkpoint), map_location=device)
            self.agent._state_preprocessor.load_state_dict(payload)

    def attach_aux_trainer(self, agent, env, cfg: dict, modules: dict[str, nn.Module], device, exp_dir: Path) -> None:
        aux_cfg = cfg.get("aux", {})
        if not aux_cfg.get("enabled", False):
            return

        cls = import_class(aux_cfg["class_path"])
        kwargs = resolve_runtime_value(aux_cfg.get("kwargs", {}), cfg, env)

        aux_dir = exp_dir / "aux_checkpoints"
        tb_dir = exp_dir / "tensorboard"

        kwargs.setdefault("checkpoint_dir", str(aux_dir))
        kwargs.setdefault("tensorboard_dir", str(tb_dir))
        kwargs.setdefault("save_interval", int(aux_cfg.get("save_interval", cfg["run"].get("checkpoint_interval", 1000))))
        kwargs.setdefault("save_optimizer", bool(aux_cfg.get("save_optimizer", True)))

        aux_checkpoint = cfg.get("paths", {}).get("aux_checkpoint", None)
        if aux_checkpoint:
            kwargs.setdefault("resume_from", aux_checkpoint)

        aux_trainer = cls(
            modules=modules,
            agent=agent,
            obs_space=env.observation_space,
            device=device,
            **kwargs,
        )

        writer = None
        try:
            from torch.utils.tensorboard import SummaryWriter

            writer = SummaryWriter(str(tb_dir))
        except Exception as exc:
            print(f"[TensorBoard] writer disabled: {exc}", flush=True)

        original_post = agent.post_interaction

        def post_with_aux(timestep, timesteps):
            original_post(timestep, timesteps)
            aux_trainer.step(timestep)

            log_interval = int(kwargs.get("log_interval", 5000))
            if timestep % log_interval == 0:
                if hasattr(env.unwrapped, "get_metrics"):
                    metrics = env.unwrapped.get_metrics()
                    for key, value in metrics.items():
                        scalar = self._to_scalar(value)
                        if scalar is not None and writer is not None:
                            writer.add_scalar(f"env/{key}", scalar, timestep)

                acc = print_orientation_accuracy(peep=True)
                if acc is not None:
                    print(f"[orient/eval] acc10={acc[0]:.3f} acc20={acc[1]:.3f} acc30={acc[2]:.3f}")
                    if writer is not None:
                        writer.add_scalar("orientation/acc10", float(acc[0]), timestep)
                        writer.add_scalar("orientation/acc20", float(acc[1]), timestep)
                        writer.add_scalar("orientation/acc30", float(acc[2]), timestep)

                if writer is not None:
                    writer.flush()

            checkpoint_interval = int(cfg["run"].get("checkpoint_interval", 1000))
            if checkpoint_interval > 0 and timestep % checkpoint_interval == 0:
                if hasattr(agent, "_state_preprocessor") and agent._state_preprocessor is not None:
                    ckpt_dir = exp_dir / "checkpoints"
                    ckpt_dir.mkdir(parents=True, exist_ok=True)
                    payload = agent._state_preprocessor.state_dict()
                    torch.save(payload, ckpt_dir / f"state_preprocessor_{int(timestep)}.pt")
                    torch.save(payload, ckpt_dir / "state_preprocessor_latest.pt")

        agent.post_interaction = post_with_aux

    @staticmethod
    def _to_scalar(value) -> float | None:
        if torch.is_tensor(value):
            if value.numel() == 0:
                return None
            return float(value.detach().float().mean().cpu().item())

        if hasattr(value, "item"):
            try:
                value = value.item()
            except Exception:
                return None

        if isinstance(value, bool):
            return float(value)
        if isinstance(value, (int, float)):
            return float(value)
        return None

    def print_final_env_metrics(self, env, cfg: dict) -> None:
        """Print final environment metrics, including SR if the env exposes it.

        The method is deliberately tolerant: it prints any scalar metric returned by
        env.unwrapped.get_metrics(). If the env exposes success/episode counters
        instead of SR directly, it derives SR for common key names.
        """
        raw_env = env.unwrapped
        mode = "EVAL" if cfg["run"].get("eval", False) else "TRAIN"

        if not hasattr(raw_env, "get_metrics"):
            print(f"\n[{self.algo_name.upper()}][{mode}] final metrics unavailable: env has no get_metrics()", flush=True)
            return

        metrics = raw_env.get_metrics()
        if not isinstance(metrics, dict) or not metrics:
            print(f"\n[{self.algo_name.upper()}][{mode}] final metrics unavailable: empty metrics", flush=True)
            return

        cleaned: dict[str, float] = {}
        for key, value in metrics.items():
            scalar = self._to_scalar(value)
            if scalar is not None:
                cleaned[str(key)] = scalar

        if not cleaned:
            print(f"\n[{self.algo_name.upper()}][{mode}] final metrics unavailable: no scalar metrics", flush=True)
            return

        self._derive_success_rate(cleaned)

        print(f"\n[{self.algo_name.upper()}][{mode}] final env metrics", flush=True)

        preferred = [
            "SR",
            "sr",
            "success_rate",
            "success",
            "success_mean",
            "SPL",
            "spl",
            "episode_length",
            "mean_episode_length",
            "reward",
            "mean_reward",
        ]

        printed: set[str] = set()
        for key in preferred:
            if key in cleaned:
                print(f"  {key}: {cleaned[key]:.4f}", flush=True)
                printed.add(key)

        for key in sorted(cleaned):
            if key not in printed:
                print(f"  {key}: {cleaned[key]:.4f}", flush=True)

    @staticmethod
    def _derive_success_rate(metrics: dict[str, float]) -> None:
        if any(key in metrics for key in ("SR", "sr", "success_rate")):
            return

        pairs = [
            ("successes", "episodes"),
            ("success_count", "episode_count"),
            ("num_successes", "num_episodes"),
            ("success_total", "episode_total"),
            ("success", "episodes"),
        ]
        for success_key, episode_key in pairs:
            if success_key in metrics and episode_key in metrics and metrics[episode_key] > 0:
                metrics["SR"] = metrics[success_key] / metrics[episode_key]
                return

    def close(self) -> None:
        if self.env is not None:
            self.env.close()
        self.env = None
        self.agent = None
        self.memory = None
        self.trainer = None
        torch.cuda.empty_cache()
        gc.collect()