# akgvp_prior_builder_annotated.py
# Standalone offline AKGVP-style prior graph builder for Isaac Lab generated rooms.
# No Isaac Sim / Isaac Lab imports.

"""
Annotated AKGVP-style offline prior graph builder.
===================================================

This file is the commented version of the current prior builder. The executable
logic is the same as in the density+XY version: repeated-object density is
preserved through ``count_norm`` and normalized XY can be appended to the KMeans
input.

Purpose
-------
Build an offline *zone-level semantic prior graph* for a procedurally generated
Isaac Lab room family. This file does not implement the runtime RL graph encoder,
SceneManager object graph, high-level controller, low-level controller, or online
graph adaptation. It produces ``prior_graph.pt``.

AKGVP mapping
-------------
This file corresponds mainly to AKGVP Section II-C, Knowledge Graph Modeling:

1. collect object observations at candidate agent positions;
2. average detections over viewing directions;
3. cluster spatial locations into zones;
4. construct a graph with zones as nodes;
5. estimate adjacent-zone probabilities;
6. align scene-specific graphs through Kuhn-Munkres / Hungarian matching;
7. average aligned graphs into a scene-type prior graph.

Parts of AKGVP *not* implemented here:

- Section II-B multimodal/aligned encoder for the RL policy;
- online graph adaptation during an episode;
- high-level controller / sub-goal selection;
- low-level controller / action policy;
- GCN policy integration.

Current engineering deviations from the paper
---------------------------------------------

1. The paper uses visual detections. This file uses a GT oracle/FOV detector.
2. The paper uses real training scenes. This file samples procedural layouts.
3. The paper treats object detections essentially as object presence/features.
   This file can preserve repeated-instance density with ``count_norm``.
4. This file can append normalized XY to the clustering input. Exported node
   features remain semantic-only; XY only stabilizes zone assignment.

Notation
--------

S: number of sampled scenes.
M: number of object instances.
V: number of object classes in ``prior_object_vocab``.
G: number of accessible grid cells in one scene.
Y: number of yaw directions.
K: number of zones / graph nodes.
D: object/text embedding dimension.

Core tensor formulas
--------------------

Instance-to-class projection:

    A[m, v] = 1 iff object instance m belongs to vocab class v

Visibility from grid cell g and yaw y:

    visible[g, y, m] = 1{active[m], in_FOV, in_range, not_occluded}

Class-level visible counts:

    counts[g, y, v] = sum_m visible[g, y, m] * A[m, v]

Yaw-averaged class probabilities:

    P[g, v] = (1 / Y) * sum_y aggregate(counts[g, y, v])

Semantic cell features:

    F_sem = P @ E

where E is either one-hot identity or CLIP text embeddings.

KMeans zone assignment:

    z_g = argmin_k ||F_cluster[g] - c_k||_2^2

Zone features:

    node_features[k]             = mean_{g: z_g = k} F_sem[g]
    cluster_feature_centers[k]   = semantic-space KMeans center for zone k
    cluster_input_centers[k]     = raw KMeans input center; includes XY if enabled
    zone_object_probs[k]         = mean_{g: z_g = k} P[g]
    zone_centers_xy[k]           = mean_{g: z_g = k} grid_xy[g]

Adjacent probability:

    C[i, j] = number of neighboring grid-cell pairs crossing zone i -> j
    adjacency[i, j] = C[i, j] / sum_j C[i, j]

Graph alignment cost:

    cost(i, j) = w_f * (1 - cosine(node_i, ref_node_j))
               + w_s * ||center_i - ref_center_j||_2 / room_diagonal
"""
from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass
from itertools import permutations
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F


# =============================================================================
# EDIT THIS CONFIG FIRST
# =============================================================================
# Research switches are intentionally centralized here.
#
# Paper-facing parameters:
# - num_scenes: number of sampled layouts used to estimate one prior graph.
# - num_zones: number of spatial clusters / graph nodes.
# - num_yaws: number of viewing directions whose detections are averaged.
# - embedding_backend: symbolic debug one-hot or CLIP text embeddings.
#
# Our diagnostic/stabilization parameters:
# - class_visibility_aggregation="count_norm" preserves repeated-object density.
# - cluster_include_xy=True appends normalized XY to KMeans input only.
# - occlusion_enabled=True adds crude top-down disk occlusion.
#
# Closest paper-like ablation:
#   class_visibility_aggregation="binary"
#   cluster_include_xy=False
#   embedding_backend="clip"
#   detector replaced by real Faster R-CNN/RGB detections.

CONFIG: dict[str, Any] = {
    # Paths.
    "scene_items_path": "source/isaaclab_tasks/isaaclab_tasks/direct/aloha_nav/configs/scene_items.json",
    "layout_rules_path": "source/isaaclab_tasks/isaaclab_tasks/direct/aloha_nav/configs/layout_rules.json",
    "output_path": "source/isaaclab_tasks/isaaclab_tasks/direct/aloha_nav/off_train_modules/prior_graph.pt",

    # Reproducibility and scale.
    "seed": 42,
    "num_scenes": 500,
    "device": "cuda" if torch.cuda.is_available() else "cpu",

    # Scene/grid geometry.
    "room_bounds": {"x_min": -5.0, "x_max": 5.0, "y_min": -5.0, "y_max": 5.0},
    "grid_step": 0.5,
    "wall_margin": 0.5,

    # Free-space pruning. Per current assumption: all object blocking radii are 1 m.
    "object_blocking_radius": 1.0,
    "blocking_types": [
        "surface_provider",
        "static_obstacle",
        "staff_obstacle",
        "movable_obstacle",
    ],

    # Oracle/FOV detector. This is not a real visual detector.
    "num_yaws": 8,
    "fov_deg": 90.0,
    "max_visible_distance": 3.0,   # detector radius, top-down crude visibility cutoff

    # Crude object-level occlusion.
    # A closer active object blocks farther objects inside its angular interval.
    "occlusion_enabled": True,
    "occlusion_object_radius": 0.45,

    # Graph construction.
    "num_zones": 5,
    "kmeans_iters": 60,
    "kmeans_restarts": 4,
    "adjacency": "4_neighbour",

    # Hungarian graph alignment.
    "matching_feature_weight": 0.7,
    "matching_spatial_weight": 0.3,

    # Object vocabularies. Change freely; builder fails if a name is absent.
    "prior_object_vocab": [
        "table_2",
        "box",
        "oven",
        "desk",
        "chair_2",
        "chair_3",
        "cabinet",
        "TrashCan",
        "vase",
        "clock",
        "Crestwood_Chair",
        "ladder",
        "lamp",
        "Standard_HalfUnit",
        "Yucca_Cane",
        "teddy",
    ],
    "train_goal_vocab": ["teddy", "table_2"],

    # Text features.
    # "one_hot" is dependency-free and good for debugging.
    # "clip" requires transformers + openai/clip-vit-base-patch32 availability.
    "embedding_backend": "one_hot",  # "one_hot" | "clip"
    "clip_model_name": "openai/clip-vit-base-patch32",

    # Placement overflow policy.
    # "error": fail if a block has more floor instances than grid cells.
    # "truncate": place only as many instances as fit; the rest stay inactive.
    "placement_overflow": "truncate",

    # Visibility aggregation.
    # "binary": old behavior, one visible instance == many visible instances.
    # "count": raw count of visible instances per class.
    # "count_norm": count divided by total class instances, preserves repeated-object density in [0, 1].
    "class_visibility_aggregation": "count_norm",

    # KMeans clustering input.
    # Semantic features alone can collapse empty cells into identical zero vectors.
    # XY is used only for clustering; exported node_features stay semantic-only.
    "cluster_include_xy": True,
    "cluster_xy_weight": 0.75,
    "cluster_semantic_weight": 1.0,

    # Debug artifacts.
    "save_debug_plots": True,
    "debug_plot_dir": "source/isaaclab_tasks/isaaclab_tasks/direct/aloha_nav/prior_debug",
}


# =============================================================================
# Data structures
# =============================================================================
# These dataclasses define the offline prior-builder state.
#
# Do not confuse SceneZoneGraph with SceneManager.SceneGraph:
# - SceneManager.SceneGraph is runtime object-level state/relations/prompts.
# - SceneZoneGraph is AKGVP-style zone-level prior graph.

@dataclass(frozen=True)
class InstanceCatalog:
    class_names: list[str]          # length M, e.g. table_2, chair_3
    instance_names: list[str]       # length M, e.g. table_2_0, chair_3_2
    object_ids: torch.Tensor        # [M]
    sizes: torch.Tensor             # [M, 3]
    types_by_class: dict[str, set[str]]
    class_to_indices: dict[str, torch.Tensor]
    type_to_indices: dict[str, torch.Tensor]

    @property
    def num_instances(self) -> int:
        return len(self.class_names)


@dataclass
class SceneBatch:
    positions: torch.Tensor         # [S, M, 3]
    active: torch.Tensor            # [S, M]
    on_surface_idx: torch.Tensor    # [S, M]
    surface_level: torch.Tensor     # [S, M]


@dataclass
class SceneZoneGraph:
    node_features: torch.Tensor              # [K, D]
    # AKGVP_RUNTIME_ALIGNMENT:
    # Explicit centers for runtime current-zone matching. The graph module will
    # compare observed_feature = detector_probs @ text_embeddings against these
    # semantic centers instead of approximating centers with node_features by name.
    cluster_feature_centers: torch.Tensor    # [K, D]
    # Raw KMeans input centers. If cluster_include_xy=True, this tensor is [K, D+2]
    # and is saved for diagnostics only; runtime semantic matching should use
    # cluster_feature_centers.
    cluster_input_centers: torch.Tensor      # [K, C]
    zone_object_probs: torch.Tensor          # [K, V]
    zone_centers_xy: torch.Tensor    # [K, 2]
    adjacency: torch.Tensor          # [K, K]
    adjacency_counts: torch.Tensor   # [K, K]
    grid_xy: torch.Tensor            # [G, 2]
    grid_to_zone: torch.Tensor       # [G]
    cell_features: torch.Tensor      # [G, D]
    cell_object_probs: torch.Tensor  # [G, V]


# =============================================================================
# Loading and validation
# =============================================================================
# Converts JSON configs into tensor-friendly metadata.
#
# Important object hierarchy:
#   object class:    "chair_3"
#   object instance: "chair_3_0", "chair_3_1", ...
#   vocab slot:      index of "chair_3" in prior_object_vocab
#
# Later the detector works at instance level and projects to class/vocab level.

def seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_json(path: str | Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_instance_catalog(scene_items: dict[str, Any], device: str) -> InstanceCatalog:
    # -------------------------------------------------------------------------
    # Expand scene_items.json from class-level descriptions into instance-level
    # tensors and lookup tables.
    #
    # Example:
    #   {"name": "chair_3", "count": 4}
    # becomes four instances:
    #   chair_3_0, chair_3_1, chair_3_2, chair_3_3
    #
    # This is needed because visibility/occlusion is computed per instance, while
    # the final prior object distribution is stored per class/vocab entry.
    # -------------------------------------------------------------------------
    objects = scene_items["objects"]

    class_names: list[str] = []
    instance_names: list[str] = []
    object_ids: list[int] = []
    sizes: list[list[float]] = []
    types_by_class: dict[str, set[str]] = {}
    class_to_indices: dict[str, list[int]] = {}
    type_to_indices: dict[str, list[int]] = {}

    idx = 0
    for obj in objects:
        name = obj["name"]
        count = int(obj["count"])
        obj_types = set(obj["type"])
        obj_id = int(obj["id"])
        size = [float(x) for x in obj["size"]]

        if name in types_by_class:
            raise RuntimeError(f"Duplicate object class in scene_items: {name!r}")
        if count <= 0:
            raise RuntimeError(f"Object {name!r} must have positive count")
        if obj_id <= 0:
            raise RuntimeError(f"Object {name!r} must have positive id")

        types_by_class[name] = obj_types
        class_to_indices[name] = []

        for i in range(count):
            class_names.append(name)
            instance_names.append(f"{name}_{i}")
            object_ids.append(obj_id)
            sizes.append(size)
            class_to_indices[name].append(idx)
            for t in obj_types:
                type_to_indices.setdefault(t, []).append(idx)
            idx += 1

    class_to_indices_t = {
        k: torch.tensor(v, dtype=torch.long, device=device)
        for k, v in class_to_indices.items()
    }
    type_to_indices_t = {
        k: torch.tensor(sorted(v), dtype=torch.long, device=device)
        for k, v in type_to_indices.items()
    }

    return InstanceCatalog(
        class_names=class_names,
        instance_names=instance_names,
        object_ids=torch.tensor(object_ids, dtype=torch.long, device=device),
        sizes=torch.tensor(sizes, dtype=torch.float32, device=device),
        types_by_class=types_by_class,
        class_to_indices=class_to_indices_t,
        type_to_indices=type_to_indices_t,
    )


def validate_vocab(catalog: InstanceCatalog, cfg: dict[str, Any]) -> None:
    known = set(catalog.class_to_indices.keys())
    for key in ["prior_object_vocab", "train_goal_vocab"]:
        missing = [name for name in cfg[key] if name not in known]
        if missing:
            raise RuntimeError(f"{key} contains unknown object classes: {missing}")


def validate_layout_rules(catalog: InstanceCatalog, rules: dict[str, Any]) -> None:
    if not rules.get("grids"):
        raise RuntimeError("layout_rules must contain non-empty 'grids'")
    if not rules.get("blocks"):
        raise RuntimeError("layout_rules must contain non-empty 'blocks'")

    used: dict[str, str] = {}
    for block_name, block_cfg in rules["blocks"].items():
        grid_name = block_cfg["grid"]
        if grid_name not in rules["grids"]:
            raise RuntimeError(f"Block {block_name!r} requests missing grid {grid_name!r}")

        for obj_name in block_cfg["objects"]:
            if obj_name not in catalog.class_to_indices:
                raise RuntimeError(f"Block {block_name!r} references unknown object {obj_name!r}")
            if obj_name in used:
                raise RuntimeError(
                    f"Object {obj_name!r} is assigned to two blocks: "
                    f"{used[obj_name]!r}, {block_name!r}"
                )
            used[obj_name] = block_name


# =============================================================================
# Offline layout sampler
# =============================================================================
# Generates sampled GT layouts without Isaac Sim.
#
# AKGVP uses a set of training scenes for a scene type. Here we approximate that
# by sampling many layouts from layout_rules.json. Each sampled layout plays the
# role of one training scene for prior estimation.
#
# Block placement rules:
# - each semantic block owns a grid;
# - class names expand to all configured instances;
# - root is placed first;
# - remaining objects are shuffled;
# - overflow can truncate, leaving unplaced objects inactive;
# - surface_only objects are placed on active surface_provider instances.

def make_default_positions(num_scenes: int, num_instances: int, device: str) -> torch.Tensor:
    graveyard_start_x = -8.0
    graveyard_start_y = 6.0
    spacing = 1.1
    max_per_row = 14

    pos = torch.zeros(num_instances, 3, dtype=torch.float32, device=device)
    ids = torch.arange(num_instances, device=device)
    row = torch.div(ids, max_per_row, rounding_mode="floor")
    col = ids % max_per_row
    pos[:, 0] = graveyard_start_x + col.float() * spacing
    pos[:, 1] = graveyard_start_y + row.float() * spacing
    return pos.unsqueeze(0).repeat(num_scenes, 1, 1)


def is_surface_only(catalog: InstanceCatalog, class_name: str) -> bool:
    return "surface_only" in catalog.types_by_class[class_name]


def floor_indices_for_block(catalog: InstanceCatalog, block_cfg: dict[str, Any]) -> list[int]:
    out: list[int] = []
    for name in block_cfg["objects"]:
        if not is_surface_only(catalog, name):
            out.extend(int(i) for i in catalog.class_to_indices[name].tolist())
    return out


def surface_indices_for_block(catalog: InstanceCatalog, block_cfg: dict[str, Any]) -> list[int]:
    out: list[int] = []
    for name in block_cfg["objects"]:
        if is_surface_only(catalog, name):
            out.extend(int(i) for i in catalog.class_to_indices[name].tolist())
    return out


def choose_root_index(
    catalog: InstanceCatalog,
    block_name: str,
    block_cfg: dict[str, Any],
    floor_indices: list[int],
    device: str,
) -> int | None:
    if not floor_indices:
        return None

    explicit_root = block_cfg.get("root")
    if explicit_root is not None:
        if explicit_root not in catalog.class_to_indices:
            raise RuntimeError(f"Block {block_name!r} root {explicit_root!r} is unknown")
        if is_surface_only(catalog, explicit_root):
            raise RuntimeError(f"Block {block_name!r} root {explicit_root!r} cannot be surface_only")
        candidates = catalog.class_to_indices[explicit_root]
    else:
        provider_set = set(int(i) for i in catalog.type_to_indices.get(
            "surface_provider",
            torch.empty(0, dtype=torch.long, device=device),
        ).tolist())

        provider_indices = [i for i in floor_indices if i in provider_set]
        candidates = torch.tensor(
            provider_indices if provider_indices else floor_indices,
            dtype=torch.long,
            device=device,
        )

    rel = int(torch.randint(0, int(candidates.numel()), (1,), device=device).item())
    return int(candidates[rel].item())


def grid_tensor(grid_cfg: dict[str, Any], device: str) -> torch.Tensor:
    coords = grid_cfg["coordinates"]
    if not coords:
        raise RuntimeError("Grid must define non-empty 'coordinates'")
    return torch.tensor(coords, dtype=torch.float32, device=device)


def neighbour_cells(
    grid: torch.Tensor,
    occupied_cells: set[int],
    block_cells: set[int],
) -> list[int]:
    num_cells = int(grid.shape[0])
    free = [i for i in range(num_cells) if i not in occupied_cells]
    if not free:
        return []
    if not block_cells:
        return free

    free_t = torch.tensor(free, dtype=torch.long, device=grid.device)
    block_t = torch.tensor(list(block_cells), dtype=torch.long, device=grid.device)

    d = torch.cdist(grid[free_t, :2], grid[block_t, :2], p=1)
    min_d = d.min(dim=1).values
    threshold = float(min_d.min().item()) + 0.3

    return [free[i] for i in torch.nonzero(min_d <= threshold, as_tuple=False).flatten().tolist()]


def sample_cell(candidates: list[int], device: str) -> int:
    if not candidates:
        raise RuntimeError("No free grid cells left for layout sampling")
    rel = int(torch.randint(0, len(candidates), (1,), device=device).item())
    return int(candidates[rel])


def sample_scene_batch(
    catalog: InstanceCatalog,
    rules: dict[str, Any],
    cfg: dict[str, Any],
) -> SceneBatch:
    # -------------------------------------------------------------------------
    # Sample S procedural layouts.
    #
    # Outputs:
    #   positions:      [S, M, 3]
    #   active:         [S, M]
    #   on_surface_idx: [S, M]
    #   surface_level:  [S, M]
    #
    # Paper correspondence:
    #   AKGVP estimates priors over multiple training scenes of the same type.
    #   Here each sampled procedural layout acts as one training scene.
    # -------------------------------------------------------------------------
    device = cfg["device"]
    num_scenes = int(cfg["num_scenes"])
    M = catalog.num_instances

    positions = make_default_positions(num_scenes, M, device)
    active = torch.zeros(num_scenes, M, dtype=torch.bool, device=device)
    on_surface_idx = torch.full((num_scenes, M), -1, dtype=torch.long, device=device)
    surface_level = torch.zeros(num_scenes, M, dtype=torch.long, device=device)

    for scene_id in range(num_scenes):
        occupied_by_grid: dict[str, set[int]] = {}
        placed_by_block: dict[str, list[int]] = {}

        for block_name, block_cfg in rules["blocks"].items():
            grid_name = block_cfg["grid"]
            grid_cfg = rules["grids"][grid_name]
            grid = grid_tensor(grid_cfg, device)

            occupied_cells = occupied_by_grid.setdefault(grid_name, set())
            block_cells: set[int] = set()
            placed: list[int] = []

            floor_indices = floor_indices_for_block(catalog, block_cfg)
            root_idx = choose_root_index(catalog, block_name, block_cfg, floor_indices, device)

            # Keep root first, but shuffle the remaining instances so overflow truncation
            # does not always drop the same object class / instance.
            if root_idx is None:
                order = []
            else:
                rest = [i for i in floor_indices if i != root_idx]
                if rest:
                    perm = torch.randperm(len(rest), device=device).tolist()
                    rest = [rest[i] for i in perm]
                order = [root_idx] + rest

            if len(order) > int(grid.shape[0]):
                policy = cfg.get("placement_overflow", "error")

                if policy == "error":
                    raise RuntimeError(
                        f"Block {block_name!r} has {len(order)} floor instances, "
                        f"but grid {grid_name!r} has only {int(grid.shape[0])} cells"
                    )

                if policy == "truncate":
                    order = order[: int(grid.shape[0])]
                else:
                    raise RuntimeError(
                        f"Unknown placement_overflow={policy!r}. "
                        f"Expected 'error' or 'truncate'."
                    )

            for obj_idx in order:
                candidates = neighbour_cells(grid, occupied_cells, block_cells)
                cell_idx = sample_cell(candidates, device)

                positions[scene_id, obj_idx] = grid[cell_idx]
                active[scene_id, obj_idx] = True
                on_surface_idx[scene_id, obj_idx] = -1
                surface_level[scene_id, obj_idx] = 0

                occupied_cells.add(cell_idx)
                block_cells.add(cell_idx)
                placed.append(obj_idx)

            placed_by_block[block_name] = placed

        for block_name, block_cfg in rules["blocks"].items():
            surface_indices = surface_indices_for_block(catalog, block_cfg)
            if not surface_indices:
                continue

            provider_set = set(int(i) for i in catalog.type_to_indices.get(
                "surface_provider",
                torch.empty(0, dtype=torch.long, device=device),
            ).tolist())

            active_providers = [i for i in placed_by_block[block_name] if i in provider_set]
            if not active_providers:
                raise RuntimeError(
                    f"Block {block_name!r} has surface_only objects but no active surface_provider"
                )

            for obj_idx in surface_indices:
                parent_rel = int(torch.randint(0, len(active_providers), (1,), device=device).item())
                parent_idx = active_providers[parent_rel]

                parent_pos = positions[scene_id, parent_idx]
                parent_size = catalog.sizes[parent_idx]
                obj_size = catalog.sizes[obj_idx]

                new_pos = parent_pos.clone()
                new_pos[2] = parent_pos[2] + parent_size[2] + obj_size[2] * 0.5

                positions[scene_id, obj_idx] = new_pos
                active[scene_id, obj_idx] = True
                on_surface_idx[scene_id, obj_idx] = parent_idx
                surface_level[scene_id, obj_idx] = surface_level[scene_id, parent_idx] + 1

    return SceneBatch(
        positions=positions,
        active=active,
        on_surface_idx=on_surface_idx,
        surface_level=surface_level,
    )


# =============================================================================
# Grid, oracle/FOV detector, text embeddings
# =============================================================================
# This is the offline observation-collection stage.
#
# AKGVP uses visual observations and an object detector. This file uses GT object
# positions and a geometric FOV detector. The output has the same conceptual role:
# object evidence per spatial location and viewing direction.
#
# Main formula:
#   visible[g, y, m] = 1{active[m] and in_range and in_FOV and not_occluded}
#   counts[g, y, v] = sum_m visible[g, y, m] * A[m, v]
#   P[g, v] = mean_y aggregate(counts[g, y, v])
#
# ``P`` is later used as the interpretable object distribution per grid cell.

def make_room_grid(cfg: dict[str, Any]) -> torch.Tensor:
    device = cfg["device"]
    b = cfg["room_bounds"]
    step = float(cfg["grid_step"])
    margin = float(cfg["wall_margin"])

    xs = torch.arange(b["x_min"] + margin, b["x_max"] - margin + 1e-6, step, device=device)
    ys = torch.arange(b["y_min"] + margin, b["y_max"] - margin + 1e-6, step, device=device)
    xx, yy = torch.meshgrid(xs, ys, indexing="xy")
    return torch.stack([xx.reshape(-1), yy.reshape(-1)], dim=-1)


def blocking_instance_mask(catalog: InstanceCatalog, cfg: dict[str, Any]) -> torch.Tensor:
    device = cfg["device"]
    mask = torch.zeros(catalog.num_instances, dtype=torch.bool, device=device)
    for t in cfg["blocking_types"]:
        if t in catalog.type_to_indices:
            mask[catalog.type_to_indices[t]] = True
    return mask


def accessible_grid_for_scene(
    full_grid_xy: torch.Tensor,
    catalog: InstanceCatalog,
    batch: SceneBatch,
    scene_id: int,
    cfg: dict[str, Any],
) -> torch.Tensor:
    # -------------------------------------------------------------------------
    # Prune grid cells that collide with active blocking objects.
    #
    # Formula:
    #   free(g) = all_m ||x_g - p_m||_2 >= object_blocking_radius
    #
    # This approximates the set of positions from which AKGVP would collect
    # observations in an environment.
    # -------------------------------------------------------------------------
    block_mask = blocking_instance_mask(catalog, cfg) & batch.active[scene_id]
    obj_xy = batch.positions[scene_id, block_mask, :2]

    if int(obj_xy.shape[0]) == 0:
        return full_grid_xy

    d = torch.cdist(full_grid_xy, obj_xy, p=2)
    free = (d >= float(cfg["object_blocking_radius"])).all(dim=1)

    grid_xy = full_grid_xy[free]
    if int(grid_xy.shape[0]) < int(cfg["num_zones"]):
        raise RuntimeError(
            f"Scene {scene_id} has only {int(grid_xy.shape[0])} accessible cells, "
            f"but num_zones={cfg['num_zones']}"
        )

    return grid_xy


def instance_to_vocab_matrix(
    catalog: InstanceCatalog,
    prior_object_vocab: list[str],
    device: str,
) -> torch.Tensor:
    M = catalog.num_instances
    V = len(prior_object_vocab)
    mat = torch.zeros(M, V, dtype=torch.float32, device=device)
    vocab_to_idx = {name: i for i, name in enumerate(prior_object_vocab)}

    for inst_idx, class_name in enumerate(catalog.class_names):
        if class_name in vocab_to_idx:
            mat[inst_idx, vocab_to_idx[class_name]] = 1.0

    return mat


def vocab_instance_counts(
    catalog: InstanceCatalog,
    prior_object_vocab: list[str],
    device: str,
) -> torch.Tensor:
    counts = torch.zeros(len(prior_object_vocab), dtype=torch.float32, device=device)
    vocab_to_idx = {name: i for i, name in enumerate(prior_object_vocab)}

    for class_name in catalog.class_names:
        if class_name in vocab_to_idx:
            counts[vocab_to_idx[class_name]] += 1.0

    return counts.clamp_min(1.0)


def yaw_values(num_yaws: int, device: str) -> torch.Tensor:
    return torch.arange(num_yaws, dtype=torch.float32, device=device) * (2.0 * math.pi / num_yaws)


def oracle_fov_class_probs(
    grid_xy: torch.Tensor,
    yaw: torch.Tensor,
    catalog: InstanceCatalog,
    batch: SceneBatch,
    scene_id: int,
    inst_to_vocab: torch.Tensor,
    vocab_counts: torch.Tensor,
    cfg: dict[str, Any],
) -> torch.Tensor:
    # -------------------------------------------------------------------------
    # GT oracle/FOV detector.
    #
    # AKGVP stage:
    #   This replaces the paper's visual object detector during offline prior
    #   construction. The shape of the evidence is still object evidence per
    #   spatial coordinate and viewing direction.
    #
    # Per-instance geometry:
    #   delta[g,m] = p_m - x_g
    #   dist[g,m]  = ||delta[g,m]||_2
    #   angle[g,m] = atan2(delta_y, delta_x)
    #   diff[g,y,m] = wrap(angle[g,m] - yaw[y])
    #
    # Visibility:
    #   visible[g,y,m] = active[m]
    #                    and dist in (0, max_visible_distance]
    #                    and |diff| <= fov/2
    #                    and not occluded
    #
    # Occlusion:
    #   A closer object with radius r_occ casts angular shadow
    #   asin(r_occ / dist_to_occluder). Farther objects inside that sector are
    #   hidden. This is a top-down approximation, not ray tracing.
    #
    # Class projection:
    #   counts[g,y,v] = sum_m visible[g,y,m] * A[m,v]
    #
    # Aggregation:
    #   binary:     min(counts, 1)
    #   count:      counts
    #   count_norm: counts / vocab_counts[v]
    #
    # Yaw averaging:
    #   P[g,v] = mean_y visible_class[g,y,v]
    # -------------------------------------------------------------------------
    """Return class-level visibility probabilities per grid cell: [G, V].

    This uses GT object positions, active flags, FOV angle, max distance,
    and an optional crude angular occlusion model.

    It is still an oracle/FOV detector, not a real visual detector.
    """
    obj_xy = batch.positions[scene_id, :, :2].float()        # [M, 2]
    active = batch.active[scene_id].bool()                   # [M]

    G = int(grid_xy.shape[0])
    Y = int(yaw.shape[0])
    M = int(obj_xy.shape[0])

    delta = obj_xy.unsqueeze(0) - grid_xy.unsqueeze(1)       # [G, M, 2]
    dist = torch.linalg.norm(delta, dim=-1)                  # [G, M]

    # Angle from detector position to object center.
    obj_angle = torch.atan2(delta[..., 1], delta[..., 0])     # [G, M]

    # Signed wrapped angle object - yaw in [-pi, pi].
    angle_diff = obj_angle[:, None, :] - yaw[None, :, None]  # [G, Y, M]
    angle_diff = torch.atan2(torch.sin(angle_diff), torch.cos(angle_diff))

    half_fov = math.radians(float(cfg["fov_deg"])) * 0.5

    visible = angle_diff.abs() <= half_fov                   # [G, Y, M]
    visible = visible & (dist[:, None, :] > 1e-6)
    visible = visible & active[None, None, :]

    max_distance = cfg.get("max_visible_distance", None)
    if max_distance is not None:
        visible = visible & (dist[:, None, :] <= float(max_distance))

    if bool(cfg.get("occlusion_enabled", False)):
        r = float(cfg.get("occlusion_object_radius", 0.45))

        # Angular half-width of each object disk from each grid point.
        # asin(r / d), clamped for numerical safety.
        angular_radius = torch.asin((r / dist.clamp_min(r + 1e-6)).clamp(max=0.999))  # [G, M]

        occluded = torch.zeros((G, Y, M), dtype=torch.bool, device=grid_xy.device)

        # Sort objects by distance for each grid point.
        # A nearer object can occlude farther objects.
        sorted_dist, sorted_idx = torch.sort(dist, dim=1)  # [G, M]

        # Loop over potential occluders. M is small; this is acceptable and clear.
        for rank in range(M):
            occ_idx = sorted_idx[:, rank]                  # [G]
            occ_dist = sorted_dist[:, rank]                # [G]

            occ_active = active[occ_idx]                   # [G]
            occ_valid = occ_active & (occ_dist > 1e-6)

            if max_distance is not None:
                occ_valid = occ_valid & (occ_dist <= float(max_distance))

            if not bool(occ_valid.any().item()):
                continue

            batch_idx = torch.arange(G, device=grid_xy.device)

            occ_angle = obj_angle[batch_idx, occ_idx]          # [G]
            occ_ang_r = angular_radius[batch_idx, occ_idx]     # [G]
            occ_visible = visible[batch_idx, :, occ_idx]       # [G, Y]

            # For every target object, check whether it lies behind this occluder.
            target_angle_diff = obj_angle - occ_angle[:, None] # [G, M]
            target_angle_diff = torch.atan2(
                torch.sin(target_angle_diff),
                torch.cos(target_angle_diff),
            )

            behind = dist > (occ_dist[:, None] + 1e-6)         # [G, M]
            inside_shadow = target_angle_diff.abs() <= occ_ang_r[:, None]
            blocked_by_occ = behind & inside_shadow & occ_valid[:, None]  # [G, M]

            # Occluder blocks only for yaws where occluder itself is visible.
            occluded = occluded | (blocked_by_occ[:, None, :] & occ_visible[:, :, None])

        visible = visible & ~occluded

    # Instance-to-class projection: counts[g,y,v] = sum_m visible[g,y,m] * A[m,v].
    visible_class_counts = torch.einsum("gym,mv->gyv", visible.float(), inst_to_vocab)

    agg = cfg.get("class_visibility_aggregation", "binary")
    if agg == "binary":
        visible_class = visible_class_counts.clamp(max=1.0)
    elif agg == "count":
        visible_class = visible_class_counts
    elif agg == "count_norm":
        visible_class = visible_class_counts / vocab_counts.view(1, 1, -1)
    else:
        raise RuntimeError(
            f"Unknown class_visibility_aggregation={agg!r}. "
            f"Expected 'binary', 'count', or 'count_norm'."
        )

    return visible_class.mean(dim=1)  # [G, V]


def build_text_embeddings(vocab: list[str], cfg: dict[str, Any]) -> torch.Tensor:
    # -------------------------------------------------------------------------
    # Build E in R^{V x D}: an embedding vector for every object class name.
    #
    # one_hot:
    #   E = identity(V). Useful for debugging because every object class is
    #   directly interpretable.
    #
    # clip:
    #   E[v] = CLIP_text_encoder(class_name_v). This is closer to AKGVP's
    #   continuous visual-language feature motivation.
    #
    # Note:
    #   CLIP image embeddings are not used in this offline builder. They belong
    #   to the later RL aligned encoder / graph adaptation path.
    # -------------------------------------------------------------------------
    backend = cfg["embedding_backend"]
    device = cfg["device"]

    if backend == "one_hot":
        return torch.eye(len(vocab), dtype=torch.float32, device=device)

    if backend == "clip":
        from transformers import CLIPModel, CLIPTokenizer

        tokenizer = CLIPTokenizer.from_pretrained(cfg["clip_model_name"])
        model = CLIPModel.from_pretrained(cfg["clip_model_name"]).to(device)
        model.eval()

        prompts = [name.replace("_", " ") for name in vocab]
        tokens = tokenizer(prompts, padding=True, return_tensors="pt").to(device)

        with torch.no_grad():
            emb = model.get_text_features(**tokens).float()

        return F.normalize(emb, dim=-1)

    raise RuntimeError(f"Unknown embedding_backend={backend!r}")


# =============================================================================
# KMeans and zone graph
# =============================================================================
# This is the core AKGVP-style Knowledge Graph Modeling stage.
#
# Given object class probabilities P[g, v] and text embeddings E[v, d]:
#   F_sem[g] = P[g] @ E
#
# KMeans clusters grid cells into K zones. By default clustering uses:
#   F_cluster = concat(w_sem * F_sem, w_xy * xy_norm)
#
# Exported node features remain semantic-only:
#   node_features[k] = mean F_sem over cells in zone k
#
# Edges are built by counting neighboring grid cells that cross zone boundaries,
# then row-normalizing the counts into adjacent probabilities.

def normalized_room_xy(grid_xy: torch.Tensor, cfg: dict[str, Any]) -> torch.Tensor:
    b = cfg["room_bounds"]

    x = grid_xy[:, 0]
    y = grid_xy[:, 1]

    x_norm = 2.0 * (x - float(b["x_min"])) / (float(b["x_max"]) - float(b["x_min"])) - 1.0
    y_norm = 2.0 * (y - float(b["y_min"])) / (float(b["y_max"]) - float(b["y_min"])) - 1.0

    return torch.stack([x_norm, y_norm], dim=-1)


def kmeans_once(x: torch.Tensor, k: int, iters: int) -> tuple[torch.Tensor, torch.Tensor, float]:
    N, _D = x.shape
    if N < k:
        raise RuntimeError(f"KMeans requires N >= K, got N={N}, K={k}")

    init_idx = torch.randperm(N, device=x.device)[:k]
    centers = x[init_idx].clone()
    labels = torch.zeros(N, dtype=torch.long, device=x.device)

    for _ in range(iters):
        dist = torch.cdist(x, centers, p=2)
        labels = torch.argmin(dist, dim=1)

        one_hot = F.one_hot(labels, num_classes=k).float()
        counts = one_hot.sum(dim=0).clamp_min(1.0)
        new_centers = (one_hot.T @ x) / counts[:, None]

        empty = torch.nonzero(one_hot.sum(dim=0) == 0, as_tuple=False).flatten()
        if int(empty.numel()) > 0:
            min_dist = dist.min(dim=1).values
            farthest = torch.topk(min_dist, k=int(empty.numel())).indices
            new_centers[empty] = x[farthest]

        if torch.allclose(new_centers, centers, atol=1e-5, rtol=1e-5):
            centers = new_centers
            break

        centers = new_centers

    # BUGFIX / AKGVP_RUNTIME_ALIGNMENT:
    # The last loop iteration may update centers after labels were computed.
    # Recompute final labels once so exported grid_to_zone and cluster centers
    # correspond to the returned centers.
    dist = torch.cdist(x, centers, p=2)
    labels = torch.argmin(dist, dim=1)
    inertia = torch.sum((x - centers[labels]) ** 2).item()
    return labels, centers, float(inertia)


def kmeans(x: torch.Tensor, k: int, iters: int, restarts: int) -> tuple[torch.Tensor, torch.Tensor]:
    best_labels = None
    best_centers = None
    best_inertia = float("inf")

    for _ in range(restarts):
        labels, centers, inertia = kmeans_once(x, k, iters)
        if inertia < best_inertia:
            best_labels = labels
            best_centers = centers
            best_inertia = inertia

    assert best_labels is not None
    assert best_centers is not None

    return best_labels, best_centers


def average_by_label(x: torch.Tensor, labels: torch.Tensor, k: int) -> torch.Tensor:
    one_hot = F.one_hot(labels, num_classes=k).float()
    counts = one_hot.sum(dim=0).clamp_min(1.0)
    return (one_hot.T @ x) / counts[:, None]


def zone_adjacency_4_neighbour(
    grid_xy: torch.Tensor,
    labels: torch.Tensor,
    k: int,
    step: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    q = torch.round(grid_xy / step).to(torch.long).cpu()
    labels_cpu = labels.to(torch.long).cpu()

    coord_to_idx = {(int(q[i, 0]), int(q[i, 1])): i for i in range(q.shape[0])}
    counts = torch.zeros(k, k, dtype=torch.float32, device=grid_xy.device)

    for i in range(q.shape[0]):
        x, y = int(q[i, 0]), int(q[i, 1])
        zi = int(labels_cpu[i])

        for nb in [(x + 1, y), (x, y + 1)]:
            j = coord_to_idx.get(nb, None)
            if j is None:
                continue

            zj = int(labels_cpu[j])
            if zi == zj:
                continue

            counts[zi, zj] += 1.0
            counts[zj, zi] += 1.0

    # Row-normalized adjacent probability. Rows with no outgoing boundary keep denom=1.
    denom = counts.sum(dim=1, keepdim=True).clamp_min(1.0)
    adj = counts / denom
    return adj, counts


def build_scene_zone_graph(
    grid_xy: torch.Tensor,
    class_probs: torch.Tensor,
    text_embeddings: torch.Tensor,
    cfg: dict[str, Any],
) -> SceneZoneGraph:
    # -------------------------------------------------------------------------
    # Build a scene-specific zone graph from grid-cell object probabilities.
    #
    # Semantic feature:
    #   F_sem = class_probs @ text_embeddings
    #
    # Clustering feature:
    #   F_cluster = F_sem                              if cluster_include_xy=False
    #   F_cluster = concat(w_sem*F_sem, w_xy*xy_norm)  if cluster_include_xy=True
    #
    # KMeans:
    #   z_g = argmin_k ||F_cluster[g] - center[k]||^2
    #
    # Node construction:
    #   node_features[k]             = mean F_sem[g] over z_g=k
    #   cluster_feature_centers[k]   = semantic center used by runtime graph adaptation
    #   cluster_input_centers[k]     = raw KMeans center, including XY if enabled
    #   zone_object_probs[k]         = mean class_probs[g] over z_g=k
    #   zone_centers_xy[k]           = mean grid_xy[g] over z_g=k
    #
    # This is the concrete code block most directly corresponding to AKGVP
    # spatial clustering / zone graph modeling.
    # -------------------------------------------------------------------------
    k = int(cfg["num_zones"])

    # Semantic cell embeddings: F_sem = P @ E, where P=class_probs and E=text_embeddings.
    semantic_features = class_probs @ text_embeddings  # [G, D]

    cluster_features = semantic_features * float(cfg.get("cluster_semantic_weight", 1.0))
    if bool(cfg.get("cluster_include_xy", True)):
        xy_features = normalized_room_xy(grid_xy, cfg)
        xy_features = xy_features * float(cfg.get("cluster_xy_weight", 0.75))
        cluster_features = torch.cat([cluster_features, xy_features], dim=-1)

    labels, cluster_input_centers = kmeans(
        cluster_features,
        k=k,
        iters=int(cfg["kmeans_iters"]),
        restarts=int(cfg["kmeans_restarts"]),
    )

    # Keep exported cell/node features semantic-only. XY affected only the labels.
    cell_features = semantic_features

    # AKGVP_RUNTIME_ALIGNMENT:
    # Save explicit semantic cluster centers for the runtime graph module.
    # Runtime graph adaptation observes detector_probs @ text_embeddings with shape [D],
    # so it must compare against [K, D] semantic centers, not [K, D+2] raw
    # KMeans centers when normalized XY is appended to clustering input.
    cluster_feature_centers = average_by_label(cell_features, labels, k)

    # node_features intentionally remain the same semantic zone centroids. Keeping
    # both names makes the later graph module contract explicit and avoids relying
    # on node_features as an implicit proxy for KMeans centers.
    node_features = cluster_feature_centers
    zone_object_probs = average_by_label(class_probs, labels, k)
    zone_centers_xy = average_by_label(grid_xy, labels, k)

    if cfg["adjacency"] != "4_neighbour":
        raise RuntimeError(f"Unsupported adjacency={cfg['adjacency']!r}")

    adj, counts = zone_adjacency_4_neighbour(
        grid_xy=grid_xy,
        labels=labels,
        k=k,
        step=float(cfg["grid_step"]),
    )

    return SceneZoneGraph(
        node_features=node_features,
        cluster_feature_centers=cluster_feature_centers,
        cluster_input_centers=cluster_input_centers,
        zone_object_probs=zone_object_probs,
        zone_centers_xy=zone_centers_xy,
        adjacency=adj,
        adjacency_counts=counts,
        grid_xy=grid_xy,
        grid_to_zone=labels,
        cell_features=cell_features,
        cell_object_probs=class_probs,
    )


# =============================================================================
# Hungarian alignment and averaging
# =============================================================================
# KMeans labels are arbitrary per scene. This block aligns all scene-specific
# graphs to the first graph before averaging, following AKGVP's use of
# Kuhn-Munkres / Hungarian matching.
#
# Matching cost:
#   cost(i,j) = w_f * (1 - cosine(feature_i, ref_feature_j))
#             + w_s * normalized_distance(center_i, ref_center_j)
#
# After matching, node features, semantic cluster centers, raw cluster-input
# centers, object probabilities, spatial centers, and adjacency matrices are
# reordered and averaged.

def brute_force_hungarian(cost: torch.Tensor) -> torch.Tensor:
    # -------------------------------------------------------------------------
    # Exact small-K assignment solver.
    #
    # AKGVP uses Kuhn-Munkres / Hungarian matching. For K=5, brute-force over all
    # K! permutations is simpler and dependency-free:
    #
    #   best_perm = argmin_perm sum_j cost[perm[j], j]
    #
    # If K grows, replace this with scipy.optimize.linear_sum_assignment.
    # -------------------------------------------------------------------------
    """Return current indices ordered by reference index.

    cost[current_idx, ref_idx].
    Exact and dependency-free for small K. For K=5 it checks 120 permutations.
    """
    k = int(cost.shape[0])
    if cost.shape != (k, k):
        raise RuntimeError(f"Cost must be square, got {tuple(cost.shape)}")
    if k > 9:
        raise RuntimeError("brute_force_hungarian is intended for K <= 9")

    cost_cpu = cost.detach().cpu()

    best_perm = None
    best = float("inf")

    for cur_for_ref in permutations(range(k)):
        v = 0.0
        for ref_idx, cur_idx in enumerate(cur_for_ref):
            v += float(cost_cpu[cur_idx, ref_idx])

        if v < best:
            best = v
            best_perm = cur_for_ref

    assert best_perm is not None

    return torch.tensor(best_perm, dtype=torch.long, device=cost.device)


def graph_matching_cost(
    current: SceneZoneGraph,
    reference: SceneZoneGraph,
    cfg: dict[str, Any],
) -> torch.Tensor:
    # -------------------------------------------------------------------------
    # Node-to-node matching cost for graph alignment.
    #
    # cost[i,j] = w_f * (1 - cosine(feature_i, ref_feature_j))
    #           + w_s * ||center_i - ref_center_j||_2 / room_diagonal
    #
    # The paper specifies matching/alignment at the graph level. This explicit
    # feature+spatial cost is our implementation choice.
    # -------------------------------------------------------------------------
    # AKGVP_RUNTIME_ALIGNMENT:
    # Align graphs by explicit semantic cluster centers. In this builder
    # node_features currently equal these centers, but using the named center
    # tensor keeps the matching contract clear for the runtime graph module.
    cur_feat = F.normalize(current.cluster_feature_centers, dim=-1)
    ref_feat = F.normalize(reference.cluster_feature_centers, dim=-1)
    # Cosine distance term: 1 - cos(current_node, reference_node).
    feature_cost = 1.0 - cur_feat @ ref_feat.T

    room = cfg["room_bounds"]
    room_diag = math.hypot(room["x_max"] - room["x_min"], room["y_max"] - room["y_min"])
    # Normalized spatial distance between zone centers.
    spatial_cost = torch.cdist(current.zone_centers_xy, reference.zone_centers_xy, p=2) / room_diag

    return (
        float(cfg["matching_feature_weight"]) * feature_cost
        + float(cfg["matching_spatial_weight"]) * spatial_cost
    )


def invert_grid_labels(labels: torch.Tensor, cur_for_ref: torch.Tensor) -> torch.Tensor:
    # old current label -> new aligned/reference label
    inv = torch.empty_like(cur_for_ref)
    inv[cur_for_ref] = torch.arange(cur_for_ref.numel(), device=cur_for_ref.device)
    return inv[labels]


def reorder_graph(graph: SceneZoneGraph, cur_for_ref: torch.Tensor) -> SceneZoneGraph:
    p = cur_for_ref

    return SceneZoneGraph(
        node_features=graph.node_features[p],
        cluster_feature_centers=graph.cluster_feature_centers[p],
        cluster_input_centers=graph.cluster_input_centers[p],
        zone_object_probs=graph.zone_object_probs[p],
        zone_centers_xy=graph.zone_centers_xy[p],
        adjacency=graph.adjacency[p][:, p],
        adjacency_counts=graph.adjacency_counts[p][:, p],
        grid_xy=graph.grid_xy,
        grid_to_zone=invert_grid_labels(graph.grid_to_zone, p),
        cell_features=graph.cell_features,
        cell_object_probs=graph.cell_object_probs,
    )


def tensor_mean_std(xs: list[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
    stacked = torch.stack(xs, dim=0)
    mean = stacked.mean(dim=0)
    std = stacked.std(dim=0, unbiased=False)
    return mean, std


def align_and_average_graphs(graphs: list[SceneZoneGraph], cfg: dict[str, Any]) -> dict[str, torch.Tensor]:
    # -------------------------------------------------------------------------
    # Align all sampled scene graphs and average them into one prior graph.
    #
    # For each graph s:
    #   assignment_s = Hungarian(cost(G_s, G_reference))
    #   G_s_aligned  = reorder(G_s, assignment_s)
    #
    # Then:
    #   prior_node_features             = mean_s G_s_aligned.node_features
    #   prior_cluster_feature_centers   = mean_s G_s_aligned.cluster_feature_centers
    #   prior_cluster_input_centers     = mean_s G_s_aligned.cluster_input_centers
    #   prior_adjacency                 = mean_s G_s_aligned.adjacency
    #
    # Std tensors are exported for debugging. Large std means unstable zones,
    # unstable layout distribution, or poor matching.
    # -------------------------------------------------------------------------
    if not graphs:
        raise RuntimeError("No graphs to average")

    reference = graphs[0]
    aligned = [reference]
    assignments = [torch.arange(int(cfg["num_zones"]), device=reference.node_features.device)]

    for graph in graphs[1:]:
        cost = graph_matching_cost(graph, reference, cfg)
        cur_for_ref = brute_force_hungarian(cost)
        aligned.append(reorder_graph(graph, cur_for_ref))
        assignments.append(cur_for_ref)

    node_features_mean, node_features_std = tensor_mean_std([g.node_features for g in aligned])
    cluster_feature_centers_mean, cluster_feature_centers_std = tensor_mean_std(
        [g.cluster_feature_centers for g in aligned]
    )
    cluster_input_centers_mean, cluster_input_centers_std = tensor_mean_std(
        [g.cluster_input_centers for g in aligned]
    )
    zone_object_probs_mean, zone_object_probs_std = tensor_mean_std([g.zone_object_probs for g in aligned])
    zone_centers_xy_mean, zone_centers_xy_std = tensor_mean_std([g.zone_centers_xy for g in aligned])
    adjacency_mean, adjacency_std = tensor_mean_std([g.adjacency for g in aligned])
    adjacency_counts_mean, adjacency_counts_std = tensor_mean_std([g.adjacency_counts for g in aligned])

    return {
        "aligned_graphs": aligned,
        "reference_graph": reference,

        "node_features": node_features_mean,
        "node_features_std": node_features_std,

        # AKGVP_RUNTIME_ALIGNMENT:
        # cluster_feature_centers is the feature-space reference used by runtime
        # current-zone matching. cluster_input_centers preserves the actual KMeans
        # input centers for diagnostics / visualization.
        "cluster_feature_centers": cluster_feature_centers_mean,
        "cluster_feature_centers_std": cluster_feature_centers_std,
        "cluster_input_centers": cluster_input_centers_mean,
        "cluster_input_centers_std": cluster_input_centers_std,

        "zone_object_probs": zone_object_probs_mean,
        "zone_object_probs_std": zone_object_probs_std,

        "zone_centers_xy": zone_centers_xy_mean,
        "zone_centers_xy_std": zone_centers_xy_std,

        "adjacency": adjacency_mean,
        "adjacency_std": adjacency_std,

        "adjacency_counts_mean": adjacency_counts_mean,
        "adjacency_counts_std": adjacency_counts_std,

        "assignments_current_for_ref": torch.stack(assignments, dim=0),
    }


# =============================================================================
# Debug plot
# =============================================================================
# Simple per-scene plot for early inspection. This is not part of AKGVP.
# Use the separate visualizer for full diagnostics.

def save_debug_plot(graph: SceneZoneGraph, path: str | Path, title: str) -> None:
    import matplotlib.pyplot as plt

    grid = graph.grid_xy.detach().cpu()
    labels = graph.grid_to_zone.detach().cpu()
    centers = graph.zone_centers_xy.detach().cpu()

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(7, 7))
    plt.scatter(grid[:, 0], grid[:, 1], c=labels, s=16)
    plt.scatter(centers[:, 0], centers[:, 1], marker="x", s=120)
    plt.gca().set_aspect("equal", adjustable="box")
    plt.title(title)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


# =============================================================================
# Main pipeline
# =============================================================================
# End-to-end execution:
#   1. load JSON configs;
#   2. build instance catalog;
#   3. sample S layouts;
#   4. build accessible grid;
#   5. compute oracle/FOV object probabilities;
#   6. build one zone graph per sampled layout;
#   7. align all zone graphs;
#   8. average them;
#   9. save prior_graph.pt.
#
# The saved prior is consumed later by the RL graph module. This file itself does
# not consume observation_space and does not output actions.

def build_prior_graph(cfg: dict[str, Any]) -> dict[str, Any]:
    # -------------------------------------------------------------------------
    # Main offline construction function.
    #
    # This writes a torch dictionary to cfg["output_path"]. The saved prior is
    # intended for a later RL graph module. That later module should combine this
    # prior with observation_space information and produce graph embeddings or
    # zone beliefs.
    # -------------------------------------------------------------------------
    seed_everything(int(cfg["seed"]))

    device = cfg["device"]

    scene_items = load_json(cfg["scene_items_path"])
    layout_rules = load_json(cfg["layout_rules_path"])

    catalog = build_instance_catalog(scene_items, device=device)
    validate_vocab(catalog, cfg)
    validate_layout_rules(catalog, layout_rules)

    batch = sample_scene_batch(catalog, layout_rules, cfg)

    full_grid_xy = make_room_grid(cfg)
    inst_to_vocab = instance_to_vocab_matrix(catalog, cfg["prior_object_vocab"], device=device)
    vocab_counts = vocab_instance_counts(catalog, cfg["prior_object_vocab"], device=device)
    text_embeddings = build_text_embeddings(cfg["prior_object_vocab"], cfg)
    yaws = yaw_values(int(cfg["num_yaws"]), device=device)

    graphs: list[SceneZoneGraph] = []

    for scene_id in range(int(cfg["num_scenes"])):
        grid_xy = accessible_grid_for_scene(
            full_grid_xy=full_grid_xy,
            catalog=catalog,
            batch=batch,
            scene_id=scene_id,
            cfg=cfg,
        )

        class_probs = oracle_fov_class_probs(
            grid_xy=grid_xy,
            yaw=yaws,
            catalog=catalog,
            batch=batch,
            scene_id=scene_id,
            inst_to_vocab=inst_to_vocab,
            vocab_counts=vocab_counts,
            cfg=cfg,
        )

        graph = build_scene_zone_graph(
            grid_xy=grid_xy,
            class_probs=class_probs,
            text_embeddings=text_embeddings,
            cfg=cfg,
        )

        graphs.append(graph)

        if bool(cfg["save_debug_plots"]) and scene_id < 5:
            save_debug_plot(
                graph,
                Path(cfg["debug_plot_dir"]) / f"scene_{scene_id:03d}_zones.png",
                title=f"Scene {scene_id:03d} zones",
            )

    averaged = align_and_average_graphs(graphs, cfg)
    reference_graph: SceneZoneGraph = averaged["reference_graph"]

    prior_object_vocab = list(cfg["prior_object_vocab"])

    output = {
        # New explicit schema marker.
        "schema_version": 3,

        # Basic metadata.
        "scene_family": "default_room",
        "K": int(cfg["num_zones"]),
        "num_scenes": int(cfg["num_scenes"]),
        "object_vocab": prior_object_vocab,
        "prior_object_vocab": prior_object_vocab,
        "train_goal_vocab": list(cfg["train_goal_vocab"]),

        # Text/object embedding basis.
        "text_embeddings": text_embeddings.detach().cpu(),
        "vocab_instance_counts": vocab_counts.detach().cpu(),

        # Averaged zone-level AKGVP-style prior.
        "node_features": averaged["node_features"].detach().cpu(),
        "node_features_std": averaged["node_features_std"].detach().cpu(),

        # AKGVP_RUNTIME_ALIGNMENT:
        # These centers are required by the runtime AKGVP graph module to assign
        # the current observation feature to a prior zone during graph adaptation.
        "cluster_feature_centers": averaged["cluster_feature_centers"].detach().cpu(),
        "cluster_feature_centers_std": averaged["cluster_feature_centers_std"].detach().cpu(),
        "cluster_input_centers": averaged["cluster_input_centers"].detach().cpu(),
        "cluster_input_centers_std": averaged["cluster_input_centers_std"].detach().cpu(),

        # Both names are saved intentionally.
        # node_object_probs is easier for the visualizer/debugger.
        # zone_object_probs is more semantically explicit.
        "node_object_probs": averaged["zone_object_probs"].detach().cpu(),
        "node_object_probs_std": averaged["zone_object_probs_std"].detach().cpu(),
        "zone_object_probs": averaged["zone_object_probs"].detach().cpu(),
        "zone_object_probs_std": averaged["zone_object_probs_std"].detach().cpu(),

        "zone_centers_xy": averaged["zone_centers_xy"].detach().cpu(),
        "zone_centers_xy_std": averaged["zone_centers_xy_std"].detach().cpu(),

        "adjacency": averaged["adjacency"].detach().cpu(),
        "adjacency_std": averaged["adjacency_std"].detach().cpu(),
        "adjacency_counts_mean": averaged["adjacency_counts_mean"].detach().cpu(),
        "adjacency_counts_std": averaged["adjacency_counts_std"].detach().cpu(),

        "assignments_current_for_ref": averaged["assignments_current_for_ref"].detach().cpu(),

        # Reference aligned grid map for visualization/debug.
        # This is not the averaged prior itself; it is a readable reference map.
        "grid_xy": reference_graph.grid_xy.detach().cpu(),
        "grid_to_zone": reference_graph.grid_to_zone.detach().cpu(),
        "cell_features": reference_graph.cell_features.detach().cpu(),
        "cell_object_probs": reference_graph.cell_object_probs.detach().cpu(),

        # Raw sampled scene data for reproducibility/debug.
        "instance_names": list(catalog.instance_names),
        "instance_class_names": list(catalog.class_names),
        "instance_object_ids": catalog.object_ids.detach().cpu(),
        "instance_sizes": catalog.sizes.detach().cpu(),

        "sampled_positions": batch.positions.detach().cpu(),
        "sampled_active": batch.active.detach().cpu(),
        "sampled_on_surface_idx": batch.on_surface_idx.detach().cpu(),
        "sampled_surface_level": batch.surface_level.detach().cpu(),

        # Config copy.
        "builder_config": dict(cfg),
    }

    output_path = Path(cfg["output_path"])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(output, output_path)

    return output


def main() -> None:
    result = build_prior_graph(CONFIG)

    print(f"Saved prior graph to: {CONFIG['output_path']}")
    print(f"schema_version={result['schema_version']}")
    print(f"K={result['K']}, num_scenes={result['num_scenes']}")
    print("node_features:", tuple(result["node_features"].shape))
    print("cluster_feature_centers:", tuple(result["cluster_feature_centers"].shape))
    print("cluster_input_centers:", tuple(result["cluster_input_centers"].shape))
    print("node_object_probs:", tuple(result["node_object_probs"].shape))
    print("zone_centers_xy:", tuple(result["zone_centers_xy"].shape))
    print("adjacency:", tuple(result["adjacency"].shape))
    print("grid_xy:", tuple(result["grid_xy"].shape))
    print("grid_to_zone:", tuple(result["grid_to_zone"].shape))


if __name__ == "__main__":
    main()