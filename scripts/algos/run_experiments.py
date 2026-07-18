# -*- coding: utf-8 -*-
"""Run config-folder experiments as separate IsaacLab processes.

Directory contract:
    configs/
      base.json
      experiment_a.json
      experiment_b.json

Every JSON list is interpreted as a grid. For list-valued hyperparameters use
one list per choice, e.g. "hidden_dims": [[32, 32]], or {"$value": [32, 32]}.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from datetime import datetime
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
ISAACLAB_ROOT = SCRIPT_DIR.parent.parent
sys.path.insert(0, str(SCRIPT_DIR))

from configs.config_utils import expand_config_dir, save_json  # noqa: E402

CONFIGS_DIR = SCRIPT_DIR / "configs" / "cur_dqn"
RUNNER_SCRIPT = SCRIPT_DIR / "runners" / "train_ddqn.py"
PYTHON = sys.executable


def parse_args():
    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument("--configs-dir", default=str(CONFIGS_DIR))
    parser.add_argument("--runner-script", default=str(RUNNER_SCRIPT))
    parser.add_argument("--start"
                        , type=int, default=0)
    parser.add_argument("--end", type=int, default=None)
    parser.add_argument("--dry-run", type=int, choices=[0, 1], default=0)
    parser.add_argument("--continue-on-error", type=int, choices=[0, 1], default=0)
    parser.add_argument("--folder-prefix", default=None)
    args, _ = parser.parse_known_args()
    return args


def _resolved_config_path(out_dir: Path, index: int, name: str) -> Path:
    safe = "".join(c if c.isalnum() or c in "._-" else "-" for c in name).strip("-") or "exp"
    return out_dir / f"{index:04d}_{safe}.json"


def main():
    args = parse_args()
    configs_dir = Path(args.configs_dir).resolve()
    runner_path = Path(args.runner_script).resolve()

    records = expand_config_dir(configs_dir)
    total = len(records)
    start = max(0, int(args.start))
    end = total if args.end is None else min(int(args.end), total)
    dry_run = bool(args.dry_run)
    continue_on_error = bool(args.continue_on_error)

    folder_prefix = args.folder_prefix or configs_dir.name
    folder = f"{datetime.now().strftime('%m.%d_%H-%M-%S')}_{folder_prefix}"
    resolved_dir = configs_dir / ".resolved" / folder

    print(f"[GRID] configs_dir={configs_dir}", flush=True)
    print(f"[GRID] runner={runner_path}", flush=True)
    print(f"[GRID] range={start}:{end} of {total}", flush=True)
    print(f"[GRID] folder={folder}", flush=True)
    print(f"[GRID] resolved_configs={resolved_dir}", flush=True)

    for record in records[start:end]:
        idx = int(record["index"])
        cfg = record["config"]
        cfg["run"]["folder"] = folder

        resolved_path = _resolved_config_path(resolved_dir, idx, record["name"])
        save_json(cfg, resolved_path)

        cmd = [
            PYTHON,
            str(runner_path),
            "--config",
            str(resolved_path),
        ]

        print(f"\n[GRID {idx + 1}/{total}] {record['name']}", flush=True)
        print(" ".join(cmd), flush=True)
        if dry_run:
            continue

        result = subprocess.run(cmd, cwd=ISAACLAB_ROOT)
        if result.returncode != 0:
            message = f"config_index={idx} name={record['name']} failed with return code {result.returncode}"
            if continue_on_error:
                print(f"[WARNING] {message}", flush=True)
            else:
                raise SystemExit(message)

    print("\n[GRID] done", flush=True)


if __name__ == "__main__":
    main()
