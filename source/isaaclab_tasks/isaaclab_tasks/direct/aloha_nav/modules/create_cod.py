"""Build an object-id CLIP semantic cache for GraphEncoder.

Place this file in:
    source/isaaclab_tasks/isaaclab_tasks/direct/aloha_nav/create_cod.py

Input and output paths are resolved relative to this file:
    configs/scene_items.json -> configs/cdecode_dict.json
                             -> text_embeddings.pt

Each object must define a stable positive integer ``id``. The cache contains
only semantic-name embeddings; color is deliberately excluded. Each semantic
class may define multiple natural-language ``info.clip_prompts``. Their
normalized CLIP embeddings are averaged and normalized again to obtain one
stable embedding per semantic class.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch
import torch.nn.functional as F

if TYPE_CHECKING:
    from transformers import CLIPModel, CLIPProcessor

BASE_DIR = Path(__file__).resolve().parent.parent
INPUT_PATH = BASE_DIR / "configs" / "scene_items.json"
OUTPUT_PATH = BASE_DIR / "configs" / "cdecode_dict.json"
CLIP_EMB_PATH = BASE_DIR / "text_embeddings.pt"
MODEL_ID = "openai/clip-vit-base-patch32"
CACHE_FORMAT_VERSION = 3


def normalize_raw_name(name: str) -> str:
    value = str(name or "").strip().lower()
    if not value:
        raise ValueError("Object name must not be empty")
    return value


def normalize_semantic_name(name: str) -> str:
    """Normalize a semantic name without destroying meaningful compounds."""
    value = str(name or "").strip()
    if not value:
        raise ValueError("Semantic object name must not be empty")

    value = re.sub(r"[_\-]\d+$", "", value)
    value = re.sub(r"[_\-]+", " ", value)
    value = re.sub(r"\s+", " ", value)
    return value.strip().lower()


def semantic_name_for_object(obj: dict[str, Any]) -> str:
    info = obj.get("info") or {}
    source_name = info.get("semantic_name") or obj.get("name", "")
    return normalize_semantic_name(source_name)


def clip_prompts_for_object(
    obj: dict[str, Any],
    semantic_name: str,
) -> tuple[str, ...]:
    """Return validated, de-duplicated natural-language prompts."""
    info = obj.get("info") or {}
    raw_prompts = info.get("clip_prompts")

    if raw_prompts is None:
        raw_prompts = [
            f"a photo of a {semantic_name}",
            f"an indoor {semantic_name}",
        ]

    if not isinstance(raw_prompts, list) or not raw_prompts:
        raise ValueError(
            f"Object {obj.get('name')!r} must define a non-empty "
            "info.clip_prompts list"
        )

    prompts: list[str] = []
    seen: set[str] = set()
    for prompt in raw_prompts:
        if not isinstance(prompt, str):
            raise ValueError(
                f"Object {obj.get('name')!r} has a non-string CLIP prompt: "
                f"{prompt!r}"
            )
        normalized = re.sub(r"\s+", " ", prompt).strip()
        if not normalized:
            raise ValueError(
                f"Object {obj.get('name')!r} has an empty CLIP prompt"
            )
        key = normalized.lower()
        if key not in seen:
            prompts.append(normalized)
            seen.add(key)

    return tuple(prompts)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _clip_encode(
    texts: list[str],
    model: "CLIPModel",
    processor: "CLIPProcessor",
    device: torch.device,
) -> torch.Tensor:
    projection_dim = int(model.config.projection_dim)
    if not texts:
        return torch.zeros((0, projection_dim), dtype=torch.float32)

    inputs = processor(
        text=texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
    ).to(device)

    with torch.inference_mode():
        embeddings = model.get_text_features(**inputs).float()
        embeddings = F.normalize(embeddings, dim=-1)

    return embeddings.cpu()


def _strict_object_id(obj: dict[str, Any]) -> int:
    if "id" not in obj:
        raise ValueError(f"Object {obj.get('name')!r} has no required integer id")

    raw_id = obj["id"]
    if isinstance(raw_id, bool) or not isinstance(raw_id, int):
        raise ValueError(
            f"Object {obj.get('name')!r} must have an integer id, "
            f"got {raw_id!r}"
        )
    if raw_id <= 0:
        raise ValueError(
            f"Object id must be positive, got {raw_id} "
            f"for {obj.get('name')!r}"
        )
    return raw_id


def prepare_object_specs(data: dict[str, Any]) -> list[dict[str, Any]]:
    objects = data.get("objects", [])
    if not isinstance(objects, list) or not objects:
        raise ValueError("scene_items.json contains no objects")

    specs: list[dict[str, Any]] = []
    used_ids: set[int] = set()
    used_raw_names: dict[str, int] = {}

    for obj in objects:
        if not isinstance(obj, dict):
            raise ValueError(
                f"Every object entry must be a JSON object, got {obj!r}"
            )

        object_id = _strict_object_id(obj)
        if object_id in used_ids:
            raise ValueError(f"Duplicate object id: {object_id}")
        used_ids.add(object_id)

        raw_name = str(obj.get("name") or "").strip()
        raw_key = normalize_raw_name(raw_name)
        if raw_key in used_raw_names:
            raise ValueError(
                f"Duplicate raw object name {raw_name!r}: object IDs "
                f"{used_raw_names[raw_key]} and {object_id}"
            )
        used_raw_names[raw_key] = object_id

        semantic_name = semantic_name_for_object(obj)
        prompts = clip_prompts_for_object(obj, semantic_name)
        specs.append(
            {
                "id": object_id,
                "name": semantic_name,
                "raw_name": raw_name,
                "raw_key": raw_key,
                "clip_prompts": prompts,
            }
        )

    return sorted(specs, key=lambda item: int(item["id"]))


def collect_semantic_prompts(
    object_specs: list[dict[str, Any]],
) -> dict[str, tuple[str, ...]]:
    """Require one deterministic prompt ensemble per semantic class."""
    semantic_to_prompts: dict[str, tuple[str, ...]] = {}
    semantic_to_first_id: dict[str, int] = {}

    for spec in object_specs:
        name = str(spec["name"])
        prompts = tuple(str(prompt) for prompt in spec["clip_prompts"])
        if name in semantic_to_prompts and semantic_to_prompts[name] != prompts:
            raise ValueError(
                f"Semantic class {name!r} has different prompt ensembles for "
                f"object IDs {semantic_to_first_id[name]} and {spec['id']}. "
                "Objects sharing a semantic_name must share clip_prompts."
            )
        semantic_to_prompts.setdefault(name, prompts)
        semantic_to_first_id.setdefault(name, int(spec["id"]))

    return dict(sorted(semantic_to_prompts.items()))


def encode_semantic_names(
    semantic_to_prompts: dict[str, tuple[str, ...]],
    model: "CLIPModel",
    processor: "CLIPProcessor",
    device: torch.device,
) -> tuple[list[str], torch.Tensor]:
    """Encode all prompts in one batch, then mean-pool per semantic class."""
    names = sorted(semantic_to_prompts)
    flat_prompts: list[str] = []
    ranges: dict[str, tuple[int, int]] = {}

    for name in names:
        start = len(flat_prompts)
        flat_prompts.extend(semantic_to_prompts[name])
        ranges[name] = (start, len(flat_prompts))

    prompt_embeddings = _clip_encode(
        flat_prompts,
        model=model,
        processor=processor,
        device=device,
    )

    class_embeddings: list[torch.Tensor] = []
    for name in names:
        start, stop = ranges[name]
        embedding = prompt_embeddings[start:stop].mean(dim=0)
        embedding = F.normalize(embedding, dim=0)
        class_embeddings.append(embedding)

    return names, torch.stack(class_embeddings, dim=0)


def build_codebook(
    object_specs: list[dict[str, Any]],
    semantic_to_prompts: dict[str, tuple[str, ...]],
) -> dict[str, Any]:
    names = sorted(semantic_to_prompts)
    name_to_idx = {name: index for index, name in enumerate(names)}

    raw_name_to_id: dict[str, int] = {}
    name_to_ids: dict[str, list[int]] = {}
    for spec in object_specs:
        object_id = int(spec["id"])
        name = str(spec["name"])
        raw_name_to_id[str(spec["raw_key"])] = object_id
        name_to_ids.setdefault(name, []).append(object_id)

    name_to_default_id = {
        name: min(object_ids) for name, object_ids in name_to_ids.items()
    }

    return {
        "names": name_to_idx,
        "objects": {
            str(spec["id"]): {
                "name": spec["name"],
                "raw_name": spec["raw_name"],
                "clip_prompts": list(spec["clip_prompts"]),
            }
            for spec in object_specs
        },
        "semantic_prompt_ensembles": {
            name: list(prompts)
            for name, prompts in semantic_to_prompts.items()
        },
        "raw_name_to_id": raw_name_to_id,
        "name_to_ids": name_to_ids,
        "name_to_default_id": name_to_default_id,
    }


def build_cache(
    data: dict[str, Any],
    model: "CLIPModel",
    processor: "CLIPProcessor",
    device: torch.device,
) -> tuple[dict[str, Any], dict[str, Any]]:
    object_specs = prepare_object_specs(data)
    semantic_to_prompts = collect_semantic_prompts(object_specs)
    names, name_embeddings = encode_semantic_names(
        semantic_to_prompts,
        model=model,
        processor=processor,
        device=device,
    )
    codebook = build_codebook(object_specs, semantic_to_prompts)

    name_to_idx = {name: index for index, name in enumerate(names)}
    max_id = max(int(spec["id"]) for spec in object_specs)
    clip_dim = int(name_embeddings.shape[-1])
    id_to_name_emb = torch.zeros(
        (max_id + 1, clip_dim),
        dtype=torch.float32,
    )
    object_id_to_name: dict[int, str] = {}

    for spec in object_specs:
        object_id = int(spec["id"])
        name = str(spec["name"])
        id_to_name_emb[object_id] = name_embeddings[name_to_idx[name]]
        object_id_to_name[object_id] = name

    payload: dict[str, Any] = {
        "metadata": {
            "format_version": CACHE_FORMAT_VERSION,
            "model_id": MODEL_ID,
            "projection_dim": clip_dim,
            "normalized": True,
            "prompt_pooling": "mean_of_l2_normalized_prompts_then_l2_normalize",
            "contains_color_embeddings": False,
            "source_file": str(INPUT_PATH),
            "source_sha256": file_sha256(INPUT_PATH),
            "created_utc": datetime.now(timezone.utc).isoformat(),
        },
        **codebook,
        "name_embs": name_embeddings,
        "id_to_name_emb": id_to_name_emb,
        "object_id_to_name": object_id_to_name,
    }
    return codebook, payload


def _write_json_temp(path: Path, data: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=2)
        file.write("\n")
        file.flush()
        os.fsync(file.fileno())
    return temporary


def _write_torch_temp(path: Path, payload: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    torch.save(payload, temporary)
    return temporary


def save_outputs(codebook: dict[str, Any], payload: dict[str, Any]) -> None:
    """Build both temporary files before replacing either destination."""
    json_tmp = OUTPUT_PATH.with_name(f".{OUTPUT_PATH.name}.tmp")
    pt_tmp = CLIP_EMB_PATH.with_name(f".{CLIP_EMB_PATH.name}.tmp")
    json_tmp.unlink(missing_ok=True)
    pt_tmp.unlink(missing_ok=True)

    try:
        json_tmp = _write_json_temp(OUTPUT_PATH, codebook)
        pt_tmp = _write_torch_temp(CLIP_EMB_PATH, payload)
        os.replace(json_tmp, OUTPUT_PATH)
        os.replace(pt_tmp, CLIP_EMB_PATH)
    finally:
        json_tmp.unlink(missing_ok=True)
        pt_tmp.unlink(missing_ok=True)


def main() -> None:
    with INPUT_PATH.open("r", encoding="utf-8") as file:
        data = json.load(file)

    try:
        from transformers import CLIPModel, CLIPProcessor
    except ImportError as exc:
        raise RuntimeError(
            "The 'transformers' package is required. Install project "
            "requirements before generating text_embeddings.pt."
        ) from exc

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CLIPModel.from_pretrained(MODEL_ID).to(device)
    processor = CLIPProcessor.from_pretrained(MODEL_ID)
    model.eval()

    codebook, payload = build_cache(data, model, processor, device)
    save_outputs(codebook, payload)

    object_ids = [int(key) for key in codebook["objects"]]
    prompt_count = sum(
        len(prompts)
        for prompts in codebook["semantic_prompt_ensembles"].values()
    )
    print(f"[INFO] Saved semantic codebook JSON to {OUTPUT_PATH}")
    print(f"[INFO] Saved object-id CLIP cache to {CLIP_EMB_PATH}")
    print("[INFO] Color embeddings are disabled")
    print(
        f"[INFO] object IDs: {len(object_ids)}, "
        f"semantic classes: {len(codebook['names'])}, "
        f"CLIP prompts: {prompt_count}, max_id={max(object_ids)}"
    )


if __name__ == "__main__":
    main()