#!/usr/bin/env python3
"""Update checkpoint paths in an IsaacLab experiment config.

The repository directory may have any name. The script locates the nearest
parent containing both:

    scripts/algos/configs
    logs/skrl

Usage
-----
Derived experiment path from the config:

    python set_checkpoint_paths.py CONFIG RUN_FOLDER [STEP]

The source directory is constructed as:

    logs/skrl/<run.task_name>/<RUN_FOLDER>/<run.name>

For example, if the config contains:

    run.task_name = "Aloha_nav_hab_wr"
    run.name      = "ddqn_discrete"

then:

    python set_checkpoint_paths.py \
        cur_dqn/ddqn_discrete.json \
        07.21_16-26-19_cur_dqn

uses:

    logs/skrl/Aloha_nav_hab_wr/07.21_16-26-19_cur_dqn/ddqn_discrete

Full source path override:

    python set_checkpoint_paths.py CONFIG \
        --p Aloha_nav_hab_wr/07.21_16-26-19_cur_dqn/ddqn_discrete

Exact checkpoint step:

    python set_checkpoint_paths.py CONFIG RUN_FOLDER 1000
    python set_checkpoint_paths.py CONFIG --p FULL_RUN_PATH --step 1000

Write null to all checkpoint fields:

    python set_checkpoint_paths.py CONFIG

Without STEP, the newest matching file by modification time is selected
independently for agent, state preprocessor, and aux. The config is replaced
atomically only after the config and all three requested checkpoint files have
been validated.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Iterable

CONFIGS_REL = Path("scripts/algos/configs")
LOGS_REL = Path("logs/skrl")

PATH_KEYS = (
    "agent_checkpoint",
    "state_preprocessor_checkpoint",
    "aux_checkpoint",
)


class UpdateError(RuntimeError):
    """Expected validation error that must leave the config unchanged."""


def _is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
        return True
    except ValueError:
        return False


def _ancestor_chain(path: Path) -> Iterable[Path]:
    path = path.resolve()
    if path.is_file():
        path = path.parent
    yield path
    yield from path.parents


def find_repo_root(explicit_root: str | None) -> Path:
    """Find the repository root without depending on its directory name."""
    if explicit_root is not None:
        root = Path(explicit_root).expanduser().resolve()
        if not (root / CONFIGS_REL).is_dir():
            raise UpdateError(f"Missing directory: {root / CONFIGS_REL}")
        if not (root / LOGS_REL).is_dir():
            raise UpdateError(f"Missing directory: {root / LOGS_REL}")
        return root

    starts = [Path.cwd(), Path(__file__).resolve().parent]
    checked: set[Path] = set()

    for start in starts:
        for candidate in _ancestor_chain(start):
            if candidate in checked:
                continue
            checked.add(candidate)
            if (candidate / CONFIGS_REL).is_dir() and (candidate / LOGS_REL).is_dir():
                return candidate

    raise UpdateError(
        "Repository root was not found. Run the script from inside the repository "
        "or pass --repo-root PATH. Expected both directories: "
        f"{CONFIGS_REL} and {LOGS_REL}."
    )


def resolve_inside_repo_base(
    value: str,
    *,
    repo_root: Path,
    base_rel: Path,
    description: str,
) -> Path:
    """Resolve an absolute, repository-relative, or base-relative path safely."""
    raw = Path(value).expanduser()
    base = (repo_root / base_rel).resolve()

    if raw.is_absolute():
        resolved = raw.resolve()
    else:
        repo_relative = (repo_root / raw).resolve()
        base_relative = (base / raw).resolve()

        # Accept values already prefixed with scripts/algos/configs or logs/skrl.
        if _is_relative_to(repo_relative, base):
            resolved = repo_relative
        else:
            resolved = base_relative

    if not _is_relative_to(resolved, base):
        raise UpdateError(
            f"{description} must be located inside {base}, got: {resolved}"
        )
    return resolved


def load_config(config_path: Path) -> dict:
    if not config_path.is_file():
        raise UpdateError(f"Config file does not exist: {config_path}")

    try:
        with config_path.open("r", encoding="utf-8") as file:
            config = json.load(file)
    except json.JSONDecodeError as error:
        raise UpdateError(
            f"Invalid JSON in {config_path}: line {error.lineno}, "
            f"column {error.colno}: {error.msg}"
        ) from error

    if not isinstance(config, dict):
        raise UpdateError("Top-level JSON value must be an object")

    paths = config.get("paths")
    if not isinstance(paths, dict):
        raise UpdateError("Config field 'paths' must be an object")

    return config


def _required_run_string(run: dict, key: str) -> str:
    value = run.get(key)
    if not isinstance(value, str) or not value.strip():
        raise UpdateError(f"Config field 'run.{key}' must be a non-empty string")

    value = value.strip()
    if value in {".", ".."} or "/" in value or "\\" in value:
        raise UpdateError(
            f"Config field 'run.{key}' must be one directory name, got: {value!r}"
        )
    return value


def read_run_layout(config: dict) -> tuple[str, str]:
    """Read task and algorithm/run directory names from the target config."""
    run = config.get("run")
    if not isinstance(run, dict):
        raise UpdateError("Config field 'run' must be an object")

    task_name = _required_run_string(run, "task_name")
    run_name = _required_run_string(run, "name")
    return task_name, run_name


def validate_run_folder_name(run_folder: str) -> str:
    value = run_folder.strip()
    if not value:
        raise UpdateError("RUN_FOLDER must not be empty")
    if value in {".", ".."} or "/" in value or "\\" in value:
        raise UpdateError(
            "RUN_FOLDER must be only the experiment folder name, for example "
            "'07.21_16-26-19_cur_dqn'. Use --p to pass a complete path."
        )
    return value


def derive_run_dir(
    *,
    repo_root: Path,
    task_name: str,
    run_folder: str,
    run_name: str,
) -> Path:
    """Build logs/skrl/<task>/<run-folder>/<run-name> from config metadata."""
    folder = validate_run_folder_name(run_folder)
    logs_root = (repo_root / LOGS_REL).resolve()
    run_dir = (logs_root / task_name / folder / run_name).resolve()

    if not _is_relative_to(run_dir, logs_root):
        raise UpdateError(f"Derived experiment directory escapes {logs_root}")
    return run_dir


def newest_file(files: Iterable[Path], label: str) -> Path:
    candidates = [path for path in files if path.is_file()]
    if not candidates:
        raise UpdateError(f"No matching {label} checkpoint found")

    # Modification time is the primary criterion; name breaks exact ties.
    return max(candidates, key=lambda path: (path.stat().st_mtime_ns, path.name))


def resolve_checkpoints(run_dir: Path, step: int | None) -> dict[str, Path]:
    checkpoints_dir = run_dir / "checkpoints"
    aux_dir = run_dir / "aux_checkpoints"

    if not run_dir.is_dir():
        raise UpdateError(f"Experiment directory does not exist: {run_dir}")
    if not checkpoints_dir.is_dir():
        raise UpdateError(f"Missing checkpoint directory: {checkpoints_dir}")
    if not aux_dir.is_dir():
        raise UpdateError(f"Missing aux checkpoint directory: {aux_dir}")

    if step is not None:
        selected = {
            "agent_checkpoint": checkpoints_dir / f"agent_{step}.pt",
            "state_preprocessor_checkpoint": (
                checkpoints_dir / f"state_preprocessor_{step}.pt"
            ),
            "aux_checkpoint": aux_dir / f"aux_{step}.pt",
        }
        missing = [path for path in selected.values() if not path.is_file()]
        if missing:
            formatted = "\n".join(f"  - {path}" for path in missing)
            raise UpdateError(
                f"Checkpoint step {step} is incomplete. Missing files:\n{formatted}"
            )
        return selected

    agent_files = [
        path
        for path in checkpoints_dir.glob("*.pt")
        if path.name == "best_agent.pt" or path.name.startswith("agent_")
    ]
    preprocessor_files = list(checkpoints_dir.glob("state_preprocessor_*.pt"))
    aux_files = list(aux_dir.glob("aux_*.pt"))

    return {
        "agent_checkpoint": newest_file(agent_files, "agent"),
        "state_preprocessor_checkpoint": newest_file(
            preprocessor_files, "state preprocessor"
        ),
        "aux_checkpoint": newest_file(aux_files, "aux"),
    }


def atomic_write_json(config_path: Path, config: dict) -> None:
    """Write valid JSON atomically in the same filesystem as the target."""
    config_path.parent.mkdir(parents=True, exist_ok=True)

    temp_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=config_path.parent,
            prefix=f".{config_path.name}.",
            suffix=".tmp",
            delete=False,
        ) as temp_file:
            temp_name = temp_file.name
            json.dump(config, temp_file, indent=2, ensure_ascii=False)
            temp_file.write("\n")
            temp_file.flush()
            os.fsync(temp_file.fileno())

        # Validate the exact temporary file before replacing the original.
        with open(temp_name, "r", encoding="utf-8") as temp_file:
            json.load(temp_file)

        os.replace(temp_name, config_path)
        temp_name = None
    finally:
        if temp_name is not None:
            try:
                os.unlink(temp_name)
            except FileNotFoundError:
                pass


def repo_relative_string(path: Path, repo_root: Path) -> str:
    try:
        return path.resolve().relative_to(repo_root.resolve()).as_posix()
    except ValueError as error:
        raise UpdateError(
            f"Selected checkpoint is outside repository root: {path}"
        ) from error


def print_summary(
    *,
    repo_root: Path,
    config_path: Path,
    run_dir: Path | None,
    source_mode: str,
    values: dict[str, str | None],
    step: int | None,
) -> None:
    width = 96
    print("=" * width)
    print("[ CHECKPOINT CONFIG UPDATE ]")
    print("=" * width)
    print(f"Repository root : {repo_root}")
    print(f"Target config   : {config_path}")
    print(f"Source mode     : {source_mode}")
    print(f"Experiment dir  : {run_dir if run_dir is not None else 'none (write null)'}")
    if run_dir is not None:
        selection = f"exact step {step}" if step is not None else "latest by mtime"
        print(f"Selection mode  : {selection}")
    print("-" * width)
    for key in PATH_KEYS:
        value = values[key]
        shown = "null" if value is None else value
        print(f"{key:<31}: {shown}")
    print("=" * width)
    print("Config updated successfully.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Set agent, state-preprocessor, and aux checkpoint paths in an "
            "IsaacLab JSON config."
        )
    )
    parser.add_argument(
        "target_config",
        help=(
            "Destination JSON config, normally relative to "
            "scripts/algos/configs"
        ),
    )
    parser.add_argument(
        "run_folder",
        nargs="?",
        help=(
            "Experiment folder name only, e.g. 07.21_16-26-19_cur_dqn. "
            "run.task_name and run.name are read from the target config. "
            "Omit it to write null to all checkpoint fields."
        ),
    )
    parser.add_argument(
        "step_positional",
        nargs="?",
        type=int,
        metavar="STEP",
        help="Optional exact checkpoint step when RUN_FOLDER is positional",
    )
    parser.add_argument(
        "-p",
        "--p",
        "--path",
        dest="full_run_path",
        help=(
            "Override the complete experiment path. Normally relative to "
            "logs/skrl, e.g. Aloha_nav_hab_wr/RUN_FOLDER/ddqn_discrete"
        ),
    )
    parser.add_argument(
        "--step",
        dest="step_option",
        type=int,
        help="Optional exact checkpoint step; works with RUN_FOLDER or --p",
    )
    parser.add_argument(
        "--repo-root",
        help=(
            "Repository root override. Normally unnecessary when running from "
            "inside the repository."
        ),
    )
    args = parser.parse_args()

    # Permit: CONFIG --p FULL_PATH 1000
    # Argparse places 1000 into run_folder because --p consumes its own value.
    if (
        args.full_run_path is not None
        and args.run_folder is not None
        and args.step_positional is None
        and args.run_folder.isdecimal()
    ):
        args.step_positional = int(args.run_folder)
        args.run_folder = None

    if args.full_run_path is not None and args.run_folder is not None:
        parser.error("RUN_FOLDER and --p/--path are mutually exclusive")

    if args.step_positional is not None and args.step_option is not None:
        parser.error("Specify checkpoint step only once")

    args.step = (
        args.step_option
        if args.step_option is not None
        else args.step_positional
    )

    has_source = args.run_folder is not None or args.full_run_path is not None
    if args.step is not None and not has_source:
        parser.error("STEP cannot be provided without RUN_FOLDER or --p")
    if args.step is not None and args.step < 0:
        parser.error("STEP must be non-negative")

    return args


def main() -> int:
    args = parse_args()

    try:
        repo_root = find_repo_root(args.repo_root)
        config_path = resolve_inside_repo_base(
            args.target_config,
            repo_root=repo_root,
            base_rel=CONFIGS_REL,
            description="Target config",
        )

        # Load and validate the target before resolving or writing anything.
        config = load_config(config_path)

        run_dir: Path | None
        source_mode: str

        if args.run_folder is None and args.full_run_path is None:
            run_dir = None
            source_mode = "none"
            values: dict[str, str | None] = {key: None for key in PATH_KEYS}
        else:
            if args.full_run_path is not None:
                run_dir = resolve_inside_repo_base(
                    args.full_run_path,
                    repo_root=repo_root,
                    base_rel=LOGS_REL,
                    description="Experiment directory",
                )
                source_mode = "full path override"
            else:
                task_name, run_name = read_run_layout(config)
                run_dir = derive_run_dir(
                    repo_root=repo_root,
                    task_name=task_name,
                    run_folder=args.run_folder,
                    run_name=run_name,
                )
                source_mode = (
                    "derived from config: "
                    f"run.task_name={task_name}, run.name={run_name}"
                )

            selected = resolve_checkpoints(run_dir, args.step)

            # Convert values only after all three files passed validation.
            values = {
                key: repo_relative_string(selected[key], repo_root)
                for key in PATH_KEYS
            }

        for key in PATH_KEYS:
            config["paths"][key] = values[key]

        atomic_write_json(config_path, config)
        print_summary(
            repo_root=repo_root,
            config_path=config_path,
            run_dir=run_dir,
            source_mode=source_mode,
            values=values,
            step=args.step,
        )
        return 0

    except UpdateError as error:
        print(f"ERROR: {error}", file=sys.stderr)
        print("Config was not modified.", file=sys.stderr)
        return 1
    except OSError as error:
        print(f"ERROR: filesystem operation failed: {error}", file=sys.stderr)
        print("Config was not modified or was replaced atomically.", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())