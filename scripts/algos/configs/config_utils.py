from __future__ import annotations

import hashlib
import json
import os
import re
from copy import deepcopy
from itertools import product
from pathlib import Path
from typing import Any


PROTECTED_VALUE_KEY = "$value"
BASE_CONFIG_NAME = "base.json"

# First item = outer loop, last item = inner loop.
# "other" means all remaining grid fields in natural JSON traversal order.
GRID_ORDER = ["other"] #, "run.seed", "model.actor.kwargs.hidden_dims"

GRID_ORDER_ALIASES = {
    "actor.kwargs.hidden_dims": "model.actor.kwargs.hidden_dims",
    "critic.kwargs.hidden_dims": "model.critic.kwargs.hidden_dims",
}


def load_json(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(data: dict, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def get_by_path(data: dict, dotted_path: str) -> Any:
    cur = data
    for part in dotted_path.split("."):
        cur = cur[part]
    return cur


def set_by_path(data: dict, dotted_path: str, value: Any) -> None:
    cur = data
    parts = dotted_path.split(".")
    for part in parts[:-1]:
        if not isinstance(cur.get(part), dict):
            cur[part] = {}
        cur = cur[part]
    cur[parts[-1]] = value


def deep_merge(base: dict, override: dict) -> dict:
    out = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict) and not _is_protected_value(value):
            out[key] = deep_merge(out[key], value)
        else:
            out[key] = deepcopy(value)
    return out


def _is_protected_value(value) -> bool:
    return isinstance(value, dict) and set(value.keys()) == {PROTECTED_VALUE_KEY}


def _unwrap_protected_values(value):
    if _is_protected_value(value):
        return deepcopy(value[PROTECTED_VALUE_KEY])
    if isinstance(value, dict):
        return {k: _unwrap_protected_values(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_unwrap_protected_values(v) for v in value]
    return value


def _collect_grid_leaves(value, prefix: str = "") -> list[tuple[str, list]]:
    if _is_protected_value(value):
        return []

    if isinstance(value, dict):
        items: list[tuple[str, list]] = []
        for key, child in value.items():
            path = key if not prefix else f"{prefix}.{key}"
            items.extend(_collect_grid_leaves(child, path))
        return items

    if isinstance(value, list):
        if not prefix:
            raise ValueError("Top-level JSON list is not a valid experiment config")
        if len(value) == 0:
            raise ValueError(f"Grid field {prefix!r} contains an empty list")
        return [(prefix, value)]

    return []


def _normalize_grid_path(path: str) -> str:
    return GRID_ORDER_ALIASES.get(path, path)


def _ordered_grid_items(grid_items: list[tuple[str, list]]) -> list[tuple[str, list]]:
    """Order grid fields according to GRID_ORDER.

    Convention:
        - first item in GRID_ORDER is the outer loop
        - last item in GRID_ORDER is the inner loop
        - "other" expands to all non-listed grid fields in natural order
    """
    if not GRID_ORDER:
        return grid_items

    item_by_path = {path: values for path, values in grid_items}

    explicit_paths = [
        _normalize_grid_path(path)
        for path in GRID_ORDER
        if path != "other"
    ]

    duplicates = [
        path for path in explicit_paths
        if explicit_paths.count(path) > 1
    ]
    if duplicates:
        raise ValueError(f"GRID_ORDER contains duplicate paths: {sorted(set(duplicates))}")

    unknown = [path for path in explicit_paths if path not in item_by_path]
    if unknown:
        raise ValueError(f"GRID_ORDER references non-grid fields: {unknown}")

    explicit_set = set(explicit_paths)
    other_items = [
        (path, values)
        for path, values in grid_items
        if path not in explicit_set
    ]

    ordered: list[tuple[str, list]] = []
    for item in GRID_ORDER:
        if item == "other":
            ordered.extend(other_items)
        else:
            path = _normalize_grid_path(item)
            ordered.append((path, item_by_path[path]))

    if "other" not in GRID_ORDER:
        ordered.extend(other_items)

    return ordered


def materialize_grid_config(raw_cfg: dict) -> list[tuple[dict, dict]]:
    """Expand all JSON-list leaves into grid combinations.

    A plain list means sweep choices. For a list-valued hyperparameter, wrap the
    list as one choice, e.g. "hidden_dims": [[32, 32]], or use {"$value": [32, 32]}.
    """
    grid_items = _ordered_grid_items(_collect_grid_leaves(raw_cfg))
    if not grid_items:
        return [(_unwrap_protected_values(deepcopy(raw_cfg)), {})]

    paths = [path for path, _ in grid_items]
    values = [choices for _, choices in grid_items]
    out: list[tuple[dict, dict]] = []

    for combo in product(*values):
        cfg = deepcopy(raw_cfg)
        selected: dict[str, Any] = {}

        for path, value in zip(paths, combo):
            selected[path] = _unwrap_protected_values(value)
            set_by_path(cfg, path, selected[path])

        out.append((_unwrap_protected_values(cfg), selected))

    return out


def experiment_config_paths(configs_dir: str | Path) -> list[Path]:
    configs_dir = Path(configs_dir)
    return sorted(
        path for path in configs_dir.glob("*.json")
        if path.name != BASE_CONFIG_NAME and not path.name.startswith("_")
    )


def expand_config_dir(configs_dir: str | Path) -> list[dict]:
    """Return resolved experiment records from a config directory.

    Directory contract:
        configs/
          base.json
          experiment_a.json
          experiment_b.json

    Grid expansion happens after base+experiment merge, so GRID_ORDER can
    control fields from both files.
    """
    configs_dir = Path(configs_dir).resolve()
    base_path = configs_dir / BASE_CONFIG_NAME
    if not base_path.exists():
        raise FileNotFoundError(f"Config directory must contain {BASE_CONFIG_NAME}: {base_path}")

    exp_paths = experiment_config_paths(configs_dir)
    if not exp_paths:
        raise FileNotFoundError(f"No experiment JSON files found in {configs_dir}; add files besides {BASE_CONFIG_NAME}")

    base_raw = load_json(base_path)

    records: list[dict] = []
    for exp_path in exp_paths:
        exp_raw = load_json(exp_path)
        merged_raw = deep_merge(base_raw, exp_raw)
        variants = materialize_grid_config(merged_raw)

        for grid_idx, (cfg, sweep) in enumerate(variants):
            cfg.setdefault("meta", {})
            cfg["meta"].update({
                "base_config": str(base_path),
                "experiment_config": str(exp_path),
                "experiment": exp_path.stem,
                "experiment_grid_id": grid_idx,
                "sweep": sweep,
            })

            apply_model_conventions(cfg)

            records.append({
                "name": make_exp_name(cfg),
                "config": cfg,
                "source": str(exp_path),
            })

    total = len(records)
    for i, record in enumerate(records):
        record["index"] = i
        record["total"] = total
        record["config"]["meta"]["config_index"] = i
        record["config"]["meta"]["config_total"] = total

    return records


def resolve_config(path: str | Path) -> dict:
    path = Path(path).resolve()
    cfg = _unwrap_protected_values(load_json(path))
    cfg.setdefault("meta", {})["resolved_config"] = str(path)
    apply_model_conventions(cfg)
    return cfg


def _validate_model_spec(spec: dict, name: str) -> None:
    if not isinstance(spec, dict):
        raise ValueError(f"{name} must be a dict with class_path and optional kwargs")
    if not spec.get("class_path"):
        raise ValueError(f"{name}.class_path is required")
    kwargs = spec.get("kwargs", {})
    if not isinstance(kwargs, dict):
        raise ValueError(f"{name}.kwargs must be a dict")


def _get_required_model_specs(model: dict, algo: str) -> list[tuple[str, dict]]:
    """Return algorithm-specific model specs using user-facing role names.

    A2C/PPO configs use actor/critic.
    SAC configs may use either policy/q_critic or actor/critic aliases.
    DDQN configs use q_network. The target network is built from the same spec
    by the DDQN runner, so it is not required as a separate config entry.
    """
    if algo in {"a2c", "ppo"}:
        return [
            ("model.actor", model.get("actor")),
            ("model.critic", model.get("critic")),
        ]

    if algo == "sac":
        policy_spec = model.get("policy", model.get("actor"))
        critic_spec = model.get("q_critic", model.get("critic"))
        return [
            ("model.policy or model.actor", policy_spec),
            ("model.q_critic or model.critic", critic_spec),
        ]

    if algo == "ddqn":
        return [
            ("model.q_network", model.get("q_network")),
        ]

    raise ValueError(f"Unsupported algo: {algo}")


def validate_config(cfg: dict, algo: str) -> None:
    """Validate only the runner-level contract.

    Important boundary:
        - The runner validates sections, class paths, grid-materialized values,
          and skrl-required hyperparameters.
        - Actor/critic/policy classes validate their own feature requirements.
        - The config is not a behavior DSL: arbitrary kwargs are allowed.
    """
    for section in ["run", "agent", "model", "env", "paths"]:
        if section not in cfg:
            raise ValueError(f"Missing config section: {section}")

    run = cfg["run"]
    agent = cfg["agent"]
    model = cfg["model"]
    algo = str(algo).lower()
    run_algo = str(run.get("algo", "")).lower()

    if algo != run_algo:
        raise ValueError(f"Runner algo={algo}, but config run.algo={run.get('algo')}")

    model_specs = _get_required_model_specs(model, algo)
    for spec_name, spec in model_specs:
        _validate_model_spec(spec, spec_name)

    features = model.get("features", {})
    if features is not None and not isinstance(features, dict):
        raise ValueError("model.features must be a dict when provided")

    modules = model.get("modules", {})
    if not isinstance(modules, dict):
        raise ValueError("model.modules must be a dict")
    for ref, spec in modules.items():
        if not isinstance(spec, dict):
            raise ValueError(f"model.modules.{ref} must be a dict")
        if not spec.get("class_path"):
            raise ValueError(f"model.modules.{ref}.class_path is required")
        kwargs = spec.get("kwargs", {})
        if not isinstance(kwargs, dict):
            raise ValueError(f"model.modules.{ref}.kwargs must be a dict")

    aux = cfg.get("aux", {}) or {}
    if not isinstance(aux, dict):
        raise ValueError("aux must be a dict when provided")
    if aux.get("enabled", False):
        if not aux.get("class_path"):
            raise ValueError("aux.class_path is required when aux.enabled=true")
        kwargs = aux.get("kwargs", {})
        if not isinstance(kwargs, dict):
            raise ValueError("aux.kwargs must be a dict")

    rnn_cfg = model.get("recurrent", {}) or {}
    if not isinstance(rnn_cfg, dict):
        raise ValueError("model.recurrent must be a dict when provided")

    recurrent = bool(rnn_cfg.get("enabled", False))
    if recurrent:
        if algo in {"sac", "ddqn"}:
            raise ValueError(f"{algo.upper()} recurrent models are not supported by the current {algo.upper()} runner")
        if str(rnn_cfg.get("type", "lstm")).lower() != "lstm":
            raise ValueError("Only model.recurrent.type='lstm' is supported by the current recurrent models")
        if int(rnn_cfg.get("sequence_length", 1)) <= 0:
            raise ValueError("model.recurrent.sequence_length must be positive")

    if algo in {"a2c", "ppo"}:
        actor_spec = model.get("actor", {})
        critic_spec = model.get("critic", {})
        actor_path = str(actor_spec.get("class_path", ""))
        critic_path = str(critic_spec.get("class_path", ""))
        uses_lstm_classes = "lstm" in actor_path.lower() or "lstm" in critic_path.lower()

        if uses_lstm_classes and not recurrent:
            raise ValueError("LSTM actor/critic classes require model.recurrent.enabled=true")
        if recurrent and not uses_lstm_classes:
            raise ValueError("model.recurrent.enabled=true requires LSTM actor/critic classes")

    if algo == "a2c":
        required = ["rollouts", "mini_batches", "learning_rate", "gamma", "gae_lambda"]
    elif algo == "ppo":
        required = ["rollouts", "learning_epochs", "mini_batches", "learning_rate", "gamma", "gae_lambda"]
    elif algo == "sac":
        required = ["memory_size", "batch_size", "actor_learning_rate", "critic_learning_rate", "gamma"]
    elif algo == "ddqn":
        required = ["memory_size", "gradient_steps", "batch_size", "learning_rate", "gamma", "polyak"]
    else:
        raise ValueError(f"Unsupported algo: {algo}")

    missing = [key for key in required if key not in agent]
    if missing:
        raise ValueError(f"Missing agent fields for {algo}: {missing}")

    if int(run.get("num_envs", 0)) <= 0:
        raise ValueError("run.num_envs must be positive")

    if algo in {"a2c", "ppo"} and recurrent:
        rollouts = int(agent["rollouts"])
        sequence_length = int(rnn_cfg.get("sequence_length", 1))
        if rollouts % sequence_length != 0:
            raise ValueError(f"agent.rollouts={rollouts} must be divisible by sequence_length={sequence_length}")


def _env_key(name: str) -> str:
    return "ALOHA_ENV_" + name.upper().replace(".", "_").replace("-", "_")


def _env_value(value) -> str:
    if isinstance(value, bool):
        return "1" if value else "0"
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, ensure_ascii=False)
    return str(value)


def export_env_config(cfg: dict) -> None:
    env_cfg = cfg.get("env", {})
    os.environ["ALOHA_NAV_ENV_CFG"] = json.dumps(env_cfg, ensure_ascii=False)
    for key, value in env_cfg.items():
        os.environ[_env_key(key)] = _env_value(value)


def camel_case_last(path: str) -> str:
    name = path.split(".")[-1]
    parts = name.split("_")
    return parts[0] + "".join(p.capitalize() for p in parts[1:])


def _slug(value: str, max_len: int = 48) -> str:
    value = re.sub(r"[^A-Za-z0-9_.-]+", "-", str(value)).strip("-")
    return value[:max_len] or "x"


NAME_KEY_ALIASES = {
    "run.seed": "s",
    "run.num_envs": "env",
    "model.actor.kwargs.hidden_dims": "net",
    "model.actor.kwargs.head_hidden_dims": "head",
}


def short_value(value) -> str:
    if isinstance(value, bool):
        return "1" if value else "0"

    if isinstance(value, float):
        return str(value).replace(".", "p")

    if isinstance(value, int):
        return str(value)

    if isinstance(value, list):
        if all(isinstance(x, int) for x in value):
            return "x".join(str(x) for x in value)

    if isinstance(value, (dict, list, tuple)):
        raw = json.dumps(value, sort_keys=True, ensure_ascii=False)
        digest = hashlib.sha1(raw.encode("utf-8")).hexdigest()[:8]
        return f"cfg{digest}"

    return _slug(value)


def apply_model_conventions(cfg: dict) -> None:
    model = cfg.get("model", {})

    if model.get("critic_hidden_dims_from_actor", False):
        actor_kwargs = model.get("actor", {}).get("kwargs", {})
        critic_kwargs = model.get("critic", {}).setdefault("kwargs", {})

        if "hidden_dims" not in actor_kwargs:
            raise ValueError(
                "model.critic_hidden_dims_from_actor=true requires "
                "model.actor.kwargs.hidden_dims"
            )

        if "hidden_dims" in critic_kwargs:
            if critic_kwargs["hidden_dims"] != actor_kwargs["hidden_dims"]:
                raise ValueError(
                    "model.critic_hidden_dims_from_actor=true but "
                    "model.critic.kwargs.hidden_dims is set to a different value. "
                    "Remove critic hidden_dims or make it equal to actor hidden_dims."
                )
            return

        critic_kwargs["hidden_dims"] = deepcopy(actor_kwargs["hidden_dims"])

    if model.get("critic_branch_kwargs_from_actor", False):
        actor_kwargs = model.get("actor", {}).get("kwargs", {})
        critic_kwargs = model.get("critic", {}).setdefault("kwargs", {})

        for key, value in actor_kwargs.items():
            if key in critic_kwargs:
                if critic_kwargs[key] != value:
                    raise ValueError(
                        f"model.critic_branch_kwargs_from_actor=true but "
                        f"model.critic.kwargs.{key} differs from actor kwargs"
                    )
                continue
            critic_kwargs[key] = deepcopy(value)


def make_exp_name(cfg: dict) -> str:
    meta = cfg.get("meta", {})
    name = meta.get("experiment", cfg.get("run", {}).get("name", "debug"))
    sweep = meta.get("sweep", {})
    if not sweep:
        return _slug(name)

    parts = [_slug(name)]
    for path, value in sorted(sweep.items()):
        key = NAME_KEY_ALIASES.get(path, camel_case_last(path))
        parts.append(f"{key}{short_value(value)}")
    return "_".join(parts)


def experiment_dir(cfg: dict, exp_name: str | None = None) -> Path:
    exp_name = exp_name or make_exp_name(cfg)
    return (
        Path(cfg["paths"]["log_root"])
        / cfg["run"]["task_name"]
        / cfg["run"].get("folder", "debug")
        / exp_name
    )

