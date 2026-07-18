"""Build compact object-id semantic cache for GraphEncoder.

Each object entry in scene_items.json must define a stable integer `id`.
The cache stores CLIP name/color embeddings indexed directly by object_id.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch
from transformers import CLIPModel, CLIPProcessor

INPUT_PATH = Path("/home/xiso/IsaacLab/source/isaaclab_tasks/isaaclab_tasks/direct/aloha_nav/configs/scene_items.json")
OUTPUT_PATH = Path("/home/xiso/IsaacLab/source/isaaclab_tasks/isaaclab_tasks/direct/aloha_nav/configs/cdecode_dict.json")
CLIP_EMB_PATH = Path("/home/xiso/IsaacLab/source/isaaclab_tasks/isaaclab_tasks/direct/aloha_nav/text_embeddings.pt")


def normalize_name(name: str) -> str:
    return str(name or "").split("_", 1)[0].lower()


def normalize_color(color: str | None) -> str:
    return str(color or "gray").strip().lower() or "gray"


def _clip_encode(texts: list[str], model, processor, device: str) -> torch.Tensor:
    if not texts:
        return torch.empty(0, 512)
    inputs = processor(text=texts, return_tensors="pt", padding=True, truncation=True).to(device)
    with torch.no_grad():
        embs = model.get_text_features(**inputs).float()
        embs = embs / (embs.norm(dim=-1, keepdim=True) + 1e-9)
    return embs.cpu()


def build_cache(data: dict):
    objects = data.get("objects", [])
    if not objects:
        raise ValueError("scene_items.json contains no objects")

    object_specs = []
    used_ids = set()
    for obj in objects:
        if "id" not in obj:
            raise ValueError(f"Object {obj.get('name')!r} has no required integer id")
        object_id = int(obj["id"])
        if object_id <= 0:
            raise ValueError(f"Object id must be positive, got {object_id} for {obj.get('name')!r}")
        if object_id in used_ids:
            raise ValueError(f"Duplicate object id: {object_id}")
        used_ids.add(object_id)

        name = normalize_name(obj.get("name", ""))
        color = normalize_color((obj.get("info") or {}).get("color", "gray"))
        object_specs.append({"id": object_id, "name": name, "color": color, "raw_name": obj.get("name", "")})

    names = sorted({o["name"] for o in object_specs})
    colors = sorted({o["color"] for o in object_specs} | {"gray"})
    name_to_idx = {name: i for i, name in enumerate(names)}
    color_to_idx = {color: i for i, color in enumerate(colors)}

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    model.eval()

    name_embs = _clip_encode(names, model, processor, device)
    color_embs = _clip_encode(colors, model, processor, device)

    max_id = max(o["id"] for o in object_specs)
    clip_dim = int(name_embs.shape[-1])
    id_to_name_emb = torch.zeros(max_id + 1, clip_dim)
    id_to_color_emb = torch.zeros(max_id + 1, clip_dim)
    object_id_to_name: dict[int, str] = {}
    object_id_to_color: dict[int, str] = {}
    class_color_to_id: dict[str, int] = {}
    name_to_default_id: dict[str, int] = {}

    for spec in object_specs:
        oid = int(spec["id"])
        name = spec["name"]
        color = spec["color"]
        id_to_name_emb[oid] = name_embs[name_to_idx[name]]
        id_to_color_emb[oid] = color_embs[color_to_idx[color]]
        object_id_to_name[oid] = name
        object_id_to_color[oid] = color
        class_color_to_id[f"{name}|{color}"] = oid
        name_to_default_id.setdefault(name, oid)

    codebook = {
        "names": name_to_idx,
        "colors": color_to_idx,
        "objects": {str(o["id"]): {"name": o["name"], "color": o["color"], "raw_name": o["raw_name"]} for o in object_specs},
        "class_color_to_id": class_color_to_id,
        "name_to_default_id": name_to_default_id,
    }
    payload = {
        **codebook,
        "name_embs": name_embs,
        "color_embs": color_embs,
        "id_to_name_emb": id_to_name_emb,
        "id_to_color_emb": id_to_color_emb,
        "object_id_to_name": object_id_to_name,
        "object_id_to_color": object_id_to_color,
    }
    return codebook, payload


def main():
    with INPUT_PATH.open("r", encoding="utf-8") as f:
        data = json.load(f)

    codebook, payload = build_cache(data)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_PATH.open("w", encoding="utf-8") as f:
        json.dump(codebook, f, ensure_ascii=False, indent=2)

    CLIP_EMB_PATH.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, CLIP_EMB_PATH)

    print(f"[INFO] Saved codebook JSON to {OUTPUT_PATH}")
    print(f"[INFO] Saved object-id CLIP cache to {CLIP_EMB_PATH}")
    print(f"[INFO] object ids: {len(codebook['objects'])}, max_id={max(map(int, codebook['objects'].keys()))}")


if __name__ == "__main__":
    main()
