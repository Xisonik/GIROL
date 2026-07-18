import json
import os

SCENE_ITEMS_FILE = "/home/xiso/IsaacLab/source/isaaclab_tasks/isaaclab_tasks/direct/aloha_nav/scene_items.json"
SCENES_FILE = "/home/xiso/IsaacLab/source/isaaclab_tasks/isaaclab_tasks/direct/aloha_nav/scene_items_maps.json"   # положи сюда файл из вопроса
OUTPUT_DIR = "/home/xiso/IsaacLab/eval_scenes_gt"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# -----------------------------
# 1. Load item sizes
# -----------------------------
with open(SCENE_ITEMS_FILE, "r") as f:
    scene_items = json.load(f)

sizes = {obj["name"]: obj["size"] for obj in scene_items["objects"]}


# -----------------------------
# 2. Load scenes
# -----------------------------
with open(SCENES_FILE, "r") as f:
    scenes_data = json.load(f)["scenes"]


def compute_bboxes(center, size):
    # center: [x, y, z]
    # size: [sx, sy, sz]

    sx, sy, sz = size
    cx, cy, cz = center

    # AABB
    aabb_min = [cx - sx/2, cy - sy/2, cz - sz/2]
    aabb_max = [cx + sx/2, cy + sy/2, cz + sz/2]

    # OBB
    obb_center = [cx, cy, cz + sz/2]  # как в примере
    obb_extent = [sx/2, sy/2, sz/2]

    return {
        "aabb": {
            "min": aabb_min,
            "max": aabb_max
        },
        "obb": {
            "center": obb_center,
            "extent": obb_extent
        }
    }


# -----------------------------
# 3. Generate JSON per scene
# -----------------------------
for scene in scenes_data:
    scene_id = scene["id"]
    objects = scene["objects"]

    graph = {
        "dataset": "isaacsim",
        "num_objects": sum(len(v) for v in objects.values()),
        "nodes": {}
    }

    track_id = 0

    for class_name, positions in objects.items():
        if class_name not in sizes:
            print(f"WARNING: no size for {class_name}")
            continue

        size = sizes[class_name]

        for pos in positions:
            bbox = compute_bboxes(pos, size)

            graph["nodes"][str(track_id)] = {
                "track_id": track_id,
                "class_name": class_name,
                "description": class_name,   # можешь поменять как надо
                "bbox_3d": bbox,
                "edges": []
            }

            track_id += 1

    # -----------------------------
    # 4. Save file
    # -----------------------------
    out_path = f"{OUTPUT_DIR}/scene_{scene_id}_graph.json"
    with open(out_path, "w") as f:
        json.dump(graph, f, indent=2)

    print(f"Saved {out_path}")