# -*- coding: utf-8 -*-
"""Algorithm-dispatching launcher.

Reads run.algo from config and delegates to the matching trainer.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _read_algo(config_path: str | Path) -> str:
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    return str(cfg["run"]["algo"]).lower()


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--config", required=True)
    args, _ = parser.parse_known_args()

    algo = _read_algo(args.config)

    if algo == "a2c":
        from runners.train_a2c import A2CTrain
        A2CTrain().run()
    elif algo == "ppo":
        from runners.train_ppo import PPOTrain
        PPOTrain().run()
    elif algo == "sac":
        from runners.train_sac import SACTrain
        SACTrain().run()
    else:
        raise ValueError(f"Unsupported run.algo: {algo}")


if __name__ == "__main__":
    main()
