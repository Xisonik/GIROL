# -*- coding: utf-8 -*-
"""Algorithm-dispatching launcher.

Reads run.algo from config and delegates to the matching trainer.
"""

from __future__ import annotations

import argparse
import json
from importlib import import_module
from pathlib import Path


RUNNERS: dict[str, tuple[str, str]] = {
    "a2c": ("runners.train_a2c", "A2CTrain"),
    "ppo": ("runners.train_ppo", "PPOTrain"),
    "sac": ("runners.train_sac", "SACTrain"),
    "ddqn": ("runners.train_ddqn", "DDQNTrain"),
}


def _read_algo(config_path: str | Path) -> str:
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    try:
        algo = cfg["run"]["algo"]
    except (KeyError, TypeError) as exc:
        raise ValueError("Config must define run.algo") from exc

    return str(algo).strip().lower()


def _load_runner_class(algo: str):
    try:
        module_name, class_name = RUNNERS[algo]
    except KeyError as exc:
        supported = ", ".join(sorted(RUNNERS))
        raise ValueError(
            f"Unsupported run.algo={algo!r}. Supported: {supported}"
        ) from exc

    module = import_module(module_name)
    try:
        return getattr(module, class_name)
    except AttributeError as exc:
        raise ImportError(
            f"Runner class {class_name!r} was not found in {module_name!r}"
        ) from exc


def main() -> None:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--config", required=True)
    args, _ = parser.parse_known_args()

    algo = _read_algo(args.config)
    RunnerClass = _load_runner_class(algo)
    RunnerClass().run()


if __name__ == "__main__":
    main()
