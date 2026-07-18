from __future__ import annotations

import json
from pathlib import Path

import torch.nn as nn


def space_to_dict(space) -> dict:
    cls = space.__class__.__name__
    if cls == "Dict":
        return {"type": "Dict", "spaces": {k: space_to_dict(v) for k, v in space.spaces.items()}}

    out = {"type": cls}
    if hasattr(space, "shape"):
        out["shape"] = tuple(space.shape) if space.shape is not None else None
    if hasattr(space, "dtype"):
        out["dtype"] = str(space.dtype)
    if hasattr(space, "n"):
        out["n"] = int(space.n)
    if hasattr(space, "nvec"):
        out["nvec"] = [int(x) for x in space.nvec]
    if hasattr(space, "low"):
        try:
            out["low_shape"] = tuple(space.low.shape)
        except Exception:
            pass
    if hasattr(space, "high"):
        try:
            out["high_shape"] = tuple(space.high.shape)
        except Exception:
            pass
    return out


def count_parameters(module: nn.Module) -> dict:
    total = sum(p.numel() for p in module.parameters())
    trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
    return {"total": int(total), "trainable": int(trainable)}


def module_layers(module: nn.Module) -> list[dict]:
    layers = []
    for name, child in module.named_modules():
        if name == "":
            continue

        item = {"name": name, "type": child.__class__.__name__}
        if isinstance(child, nn.Linear):
            item["in_features"] = child.in_features
            item["out_features"] = child.out_features
        elif isinstance(child, nn.Conv2d):
            item["in_channels"] = child.in_channels
            item["out_channels"] = child.out_channels
            item["kernel_size"] = child.kernel_size
            item["stride"] = child.stride
        elif isinstance(child, nn.LayerNorm):
            item["normalized_shape"] = tuple(child.normalized_shape)
        elif isinstance(child, nn.LSTM):
            item["input_size"] = child.input_size
            item["hidden_size"] = child.hidden_size
            item["num_layers"] = child.num_layers
            item["batch_first"] = child.batch_first
        layers.append(item)
    return layers


def model_to_dict(model: nn.Module) -> dict:
    out = {
        "type": model.__class__.__name__,
        "parameters": count_parameters(model),
        "layers": module_layers(model),
        "repr": str(model),
    }
    if hasattr(model, "get_specification"):
        try:
            out["specification"] = model.get_specification()
        except Exception:
            pass
    return out


def save_experiment_logs(
    exp_dir: str | Path,
    cfg: dict,
    env,
    models: dict[str, nn.Module],
    pipeline_modules: dict[str, nn.Module] | None = None,
) -> None:
    exp_dir = Path(exp_dir)
    exp_dir.mkdir(parents=True, exist_ok=True)

    description = cfg.get("run", {}).get("description", "test")
    pipeline_modules = pipeline_modules or {}

    raw_observation_space = space_to_dict(env.observation_space)
    action_space = space_to_dict(env.action_space)

    active_networks = {}
    for name, module in pipeline_modules.items():
        if module is not None:
            active_networks[name] = model_to_dict(module)
    for name, model in models.items():
        active_networks[name] = model_to_dict(model)

    architecture = {
        "description": description,
        "flow": "raw_observation_space -> active_networks -> action_space",
        "raw_observation_space": raw_observation_space,
        "active_networks": active_networks,
        "action_space": action_space,
    }

    with open(exp_dir / "description.txt", "w", encoding="utf-8") as f:
        f.write(str(description).strip() + "\n")

    with open(exp_dir / "architecture.json", "w", encoding="utf-8") as f:
        json.dump(architecture, f, indent=2, ensure_ascii=False)

    with open(exp_dir / "architecture.txt", "w", encoding="utf-8") as f:
        f.write(f"Description: {description}\n\n")
        f.write("FLOW\n")
        f.write(architecture["flow"] + "\n\n")

        f.write("RAW ENV OBSERVATION SPACE\n")
        f.write(json.dumps(raw_observation_space, indent=2, ensure_ascii=False))
        f.write("\n\n")

        f.write("ACTIVE NETWORKS\n")
        for name, module in active_networks.items():
            f.write(f"\n--- {name} ---\n")
            f.write(f"type: {module['type']}\n")
            f.write(f"parameters: {module['parameters']}\n")
            if "specification" in module:
                f.write("specification:\n")
                f.write(json.dumps(module["specification"], indent=2, ensure_ascii=False))
                f.write("\n")
            f.write("layers:\n")
            f.write(json.dumps(module["layers"], indent=2, ensure_ascii=False))
            f.write("\n")

        f.write("\nACTION SPACE\n")
        f.write(json.dumps(action_space, indent=2, ensure_ascii=False))
        f.write("\n")
