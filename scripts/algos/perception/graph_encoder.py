"""Fast configurable scene-graph encoder.

Raw transport contract (fixed):
    graph_flat [B, num_nodes * 6]
    per object = [object_id, active, is_goal, x_env, y_env, z_env]

The raw contract is intentionally fixed. ``active`` and ``is_goal`` are topology
metadata, while coordinates are required to derive edge relations. They are not
necessarily used as learned node features.

Learned node interface (configurable):
    always: object name embedding, object color embedding
    optional: metric XYZ embedding            (include_node_metric=True)
    optional: is_goal embedding                (include_node_is_goal=True)

Goal-star relation topology:
    for every active non-goal object, create exactly two parallel directed edges
    from goal to object:
        1) Y-axis edge: in_front_of / behind / same
        2) X-axis edge: left_of / right_of / same
    each edge also carries an independent axis-distance category. Direction
    and distance IDs are converted to fixed one-hot features before GATv2.

Coordinate convention for an object relative to the goal:
    delta = object_position - goal_position
    +Y -> in_front_of
    -Y -> behind
    -X -> left_of
    +X -> right_of

Self-loops are optional. Empty synthetic fallback graphs always receive one
internal self-loop so that batching remains well-defined.
"""

from __future__ import annotations

import glob
import json
import os
from collections import OrderedDict
from dataclasses import dataclass
from typing import Hashable, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import (
    GATv2Conv,
    global_add_pool,
    global_max_pool,
    global_mean_pool,
)
from torch_geometric.utils import softmax


NUM_GRAPH_NODES = 21
PER_OBJECT_DIM = 6
GRAPH_EMB_DIM = 128

# -----------------------------------------------------------------------------
# Raw node transport interface
# -----------------------------------------------------------------------------

RAW_NODE_FIELDS = (
    "object_id",
    "active",
    "is_goal",
    "x_env",
    "y_env",
    "z_env",
)
RAW_NODE_FIELD_TO_INDEX = {name: idx for idx, name in enumerate(RAW_NODE_FIELDS)}

OBJECT_ID_IDX = RAW_NODE_FIELD_TO_INDEX["object_id"]
ACTIVE_IDX = RAW_NODE_FIELD_TO_INDEX["active"]
IS_GOAL_IDX = RAW_NODE_FIELD_TO_INDEX["is_goal"]
POSITION_SLICE = slice(
    RAW_NODE_FIELD_TO_INDEX["x_env"],
    RAW_NODE_FIELD_TO_INDEX["z_env"] + 1,
)

# -----------------------------------------------------------------------------
# Edge relation dictionaries -- intentionally visible and stable
# -----------------------------------------------------------------------------

DIRECTION_LABELS = (
    "same",
    "in_front_of",
    "behind",
    "left_of",
    "right_of",
    "self",
)
DIRECTION_TO_ID = {label: idx for idx, label in enumerate(DIRECTION_LABELS)}
ID_TO_DIRECTION = {idx: label for label, idx in DIRECTION_TO_ID.items()}

DISTANCE_LABELS = (
    "very_close",
    "close",
    "far",
    "very_far",
)
DISTANCE_TO_ID = {label: idx for idx, label in enumerate(DISTANCE_LABELS)}
ID_TO_DISTANCE = {idx: label for label, idx in DISTANCE_TO_ID.items()}

# Fixed edge feature interface passed to GATv2:
#   direction one-hot: len(DIRECTION_LABELS) = 6
#   distance one-hot:  len(DISTANCE_LABELS) = 4
#   total edge_dim:    10
EDGE_DIRECTION_DIM = len(DIRECTION_LABELS)
EDGE_DISTANCE_DIM = len(DISTANCE_LABELS)
EDGE_DIM = EDGE_DIRECTION_DIM + EDGE_DISTANCE_DIM

# Axis-wise distance intervals in metres:
#   very_close: 0 <= d < 5
#   close:      5 <= d < 10
#   far:       10 <= d < 15
#   very_far:  15 <= d <= 20
# Values above 20 are classified as very_far defensively.
DISTANCE_THRESHOLDS_M = (5.0, 10.0, 15.0)
LAYOUT_AXIS_EXTENT_M = 20.0
RELATION_EPS = 1e-6

# Edge kind is topology metadata used to choose which axis relation to compute.
EDGE_KIND_Y = 0
EDGE_KIND_X = 1
EDGE_KIND_SELF = 2


def _norm_name(name: str) -> str:
    return str(name or "").split("_", 1)[0].lower()


def _norm_color(color: str | None) -> str:
    return str(color or "gray").strip().lower() or "gray"


def _distance_category_ids(axis_distance: torch.Tensor) -> torch.Tensor:
    """Return axis-wise distance IDs for non-negative distances."""
    d = axis_distance.abs()
    ids = torch.zeros_like(d, dtype=torch.long)
    ids = ids + (d >= DISTANCE_THRESHOLDS_M[0]).long()
    ids = ids + (d >= DISTANCE_THRESHOLDS_M[1]).long()
    ids = ids + (d >= DISTANCE_THRESHOLDS_M[2]).long()
    return ids.clamp_max(DISTANCE_TO_ID["very_far"])


def build_edge_type_ids(
    delta: torch.Tensor,
    edge_kind: torch.Tensor,
    eps: float = RELATION_EPS,
) -> torch.Tensor:
    """Build categorical edge type IDs ``[direction_id, distance_id]``.

    Args:
        delta:
            ``pos[dst] - pos[src]`` with shape ``[E, 3]``. For goal-star
            relation edges, ``src`` is the goal and ``dst`` is the object, so
            the labels describe the object relative to the goal.
        edge_kind:
            Shape ``[E]``. ``EDGE_KIND_Y`` selects front/behind, ``EDGE_KIND_X``
            selects left/right, and ``EDGE_KIND_SELF`` creates a self relation.
    """
    if delta.dim() != 2 or delta.shape[-1] != 3:
        raise ValueError(f"delta must have shape [E, 3], got {tuple(delta.shape)}")
    if edge_kind.dim() != 1 or edge_kind.shape[0] != delta.shape[0]:
        raise ValueError(
            "edge_kind must have shape [E] aligned with delta, got "
            f"{tuple(edge_kind.shape)} for delta {tuple(delta.shape)}"
        )

    direction_ids = torch.full(
        (delta.shape[0],),
        DIRECTION_TO_ID["same"],
        dtype=torch.long,
        device=delta.device,
    )
    distance_ids = torch.full(
        (delta.shape[0],),
        DISTANCE_TO_ID["very_close"],
        dtype=torch.long,
        device=delta.device,
    )

    y_mask = edge_kind == EDGE_KIND_Y
    if y_mask.any():
        dy = delta[y_mask, 1]
        y_direction = torch.full_like(dy, DIRECTION_TO_ID["same"], dtype=torch.long)
        y_direction[dy > eps] = DIRECTION_TO_ID["in_front_of"]
        y_direction[dy < -eps] = DIRECTION_TO_ID["behind"]
        direction_ids[y_mask] = y_direction
        distance_ids[y_mask] = _distance_category_ids(dy)

    x_mask = edge_kind == EDGE_KIND_X
    if x_mask.any():
        dx = delta[x_mask, 0]
        x_direction = torch.full_like(dx, DIRECTION_TO_ID["same"], dtype=torch.long)
        # Coordinate convention requested for object relative to goal:
        #   -X -> left_of, +X -> right_of.
        x_direction[dx < -eps] = DIRECTION_TO_ID["left_of"]
        x_direction[dx > eps] = DIRECTION_TO_ID["right_of"]
        direction_ids[x_mask] = x_direction
        distance_ids[x_mask] = _distance_category_ids(dx)

    self_mask = edge_kind == EDGE_KIND_SELF
    if self_mask.any():
        direction_ids[self_mask] = DIRECTION_TO_ID["self"]
        distance_ids[self_mask] = DISTANCE_TO_ID["very_close"]

    unknown = ~(y_mask | x_mask | self_mask)
    if unknown.any():
        bad = torch.unique(edge_kind[unknown]).detach().cpu().tolist()
        raise ValueError(f"Unknown edge kinds: {bad}")

    return torch.stack([direction_ids, distance_ids], dim=-1)


def _parallel_axis_edges(
    src: torch.Tensor,
    dst: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Duplicate each directed pair into one Y edge and one X edge."""
    if src.shape != dst.shape:
        raise ValueError("src and dst must have equal shape")
    edge_index = torch.stack(
        [torch.cat([src, src], dim=0), torch.cat([dst, dst], dim=0)],
        dim=0,
    )
    edge_kind = torch.cat(
        [
            torch.full_like(src, EDGE_KIND_Y),
            torch.full_like(src, EDGE_KIND_X),
        ],
        dim=0,
    )
    return edge_index, edge_kind


def _append_self_edges(
    edge_index: torch.Tensor,
    edge_kind: torch.Tensor,
    nodes: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    if nodes.numel() == 0:
        return edge_index, edge_kind
    loops = torch.stack([nodes, nodes], dim=0)
    return (
        torch.cat([edge_index, loops], dim=1),
        torch.cat(
            [edge_kind, torch.full_like(nodes, EDGE_KIND_SELF)],
            dim=0,
        ),
    )


# -----------------------------------------------------------------------------
# Legacy single-graph builders kept for ablations and backwards imports
# -----------------------------------------------------------------------------


def build_complete_graph_edges(
    pos: torch.Tensor,
    offset: int,
    device: torch.device,
    *,
    self_edges: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build a directed complete graph with two parallel axis edges per pair.

    Returns:
        edge_index, edge_type_ids where edge_type_ids is [E, 2].
    """
    n = int(pos.shape[0])
    if n <= 0:
        raise ValueError("build_complete_graph_edges expects at least one node")

    nodes = torch.arange(n, device=device, dtype=torch.long)
    src = nodes.repeat_interleave(n)
    dst = nodes.repeat(n)
    non_self = src != dst
    src = src[non_self] + offset
    dst = dst[non_self] + offset
    edge_index, edge_kind = _parallel_axis_edges(src, dst)

    if self_edges:
        edge_index, edge_kind = _append_self_edges(
            edge_index, edge_kind, nodes + offset
        )

    local_src = edge_index[0] - offset
    local_dst = edge_index[1] - offset
    delta = pos[local_dst] - pos[local_src]
    edge_type_ids = build_edge_type_ids(delta, edge_kind)
    return edge_index, edge_type_ids


def build_goal_star_edges(
    pos: torch.Tensor,
    goal_mask: torch.Tensor,
    offset: int,
    device: torch.device,
    *,
    self_edges: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build goal -> object edges, two parallel axis relations per object."""
    n = int(pos.shape[0])
    if n <= 0:
        raise ValueError("build_goal_star_edges expects at least one node")
    if goal_mask.shape[0] != n:
        raise ValueError(
            f"goal_mask length {goal_mask.shape[0]} does not match node count {n}"
        )

    candidates = torch.nonzero(goal_mask, as_tuple=False).view(-1)
    if candidates.numel() != 1:
        raise ValueError(
            f"Expected exactly one active goal node, got {candidates.numel()}"
        )
    goal = candidates[0].long()
    nodes = torch.arange(n, device=device, dtype=torch.long)
    objects = nodes[nodes != goal]

    src = goal.expand(objects.numel()) + offset
    dst = objects + offset
    edge_index, edge_kind = _parallel_axis_edges(src, dst)

    if self_edges:
        edge_index, edge_kind = _append_self_edges(
            edge_index, edge_kind, nodes + offset
        )

    local_src = edge_index[0] - offset
    local_dst = edge_index[1] - offset
    delta = pos[local_dst] - pos[local_src]
    edge_type_ids = build_edge_type_ids(delta, edge_kind)
    return edge_index, edge_type_ids


def build_goal_star_random_edges(
    pos: torch.Tensor,
    goal_mask: torch.Tensor,
    offset: int,
    device: torch.device,
    *,
    num_random_edges: int = 16,
    self_edges: bool = True,
    random_bidirectional: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Goal-star plus random object-object pairs, with two axis edges per pair."""
    n = int(pos.shape[0])
    if n <= 0:
        raise ValueError("build_goal_star_random_edges expects at least one node")
    if goal_mask.shape[0] != n:
        raise ValueError(
            f"goal_mask length {goal_mask.shape[0]} does not match node count {n}"
        )

    candidates = torch.nonzero(goal_mask, as_tuple=False).view(-1)
    if candidates.numel() != 1:
        raise ValueError(
            f"Expected exactly one active goal node, got {candidates.numel()}"
        )
    goal = candidates[0].long()
    nodes = torch.arange(n, device=device, dtype=torch.long)
    objects = nodes[nodes != goal]

    pair_src = [goal.expand(objects.numel())]
    pair_dst = [objects]

    m = int(objects.numel())
    if m >= 2 and num_random_edges > 0:
        src_grid = objects.repeat_interleave(m)
        dst_grid = objects.repeat(m)
        valid = src_grid != dst_grid
        src_grid = src_grid[valid]
        dst_grid = dst_grid[valid]

        sample_count = min(int(num_random_edges), int(src_grid.numel()))
        perm = torch.randperm(src_grid.numel(), device=device)[:sample_count]
        rand_src = src_grid[perm]
        rand_dst = dst_grid[perm]
        pair_src.append(rand_src)
        pair_dst.append(rand_dst)
        if random_bidirectional:
            pair_src.append(rand_dst)
            pair_dst.append(rand_src)

    src = torch.cat(pair_src, dim=0) + offset
    dst = torch.cat(pair_dst, dim=0) + offset
    edge_index, edge_kind = _parallel_axis_edges(src, dst)

    if self_edges:
        edge_index, edge_kind = _append_self_edges(
            edge_index, edge_kind, nodes + offset
        )

    local_src = edge_index[0] - offset
    local_dst = edge_index[1] - offset
    delta = pos[local_dst] - pos[local_src]
    edge_type_ids = build_edge_type_ids(delta, edge_kind)
    return edge_index, edge_type_ids


@dataclass(frozen=True)
class PackedTopology:
    """Topology-only data for a packed PyG batch."""

    B: int
    N: int
    active_flat_idx: torch.Tensor
    empty_batches: torch.Tensor
    batch_vec: torch.Tensor
    edge_index: torch.Tensor
    edge_kind: torch.Tensor


class GraphEncoder(nn.Module):
    def __init__(
        self,
        embeddings_path: str,
        graphs_dir: Optional[str] = None,
        num_nodes: int = NUM_GRAPH_NODES,
        per_object_dim: int = PER_OBJECT_DIM,
        text_dim: int = 32,
        hidden_dim: int = 128,
        out_dim: int = GRAPH_EMB_DIM,
        num_layers: int = 2,
        heads: int = 2,
        dropout: float = 0.1,
        edge_mode: str = "goal_star",
        random_edges: int = 16,
        validate_graph: bool = False,
        topology_cache_size: int = 32,
        include_node_metric: bool = True,
        include_node_is_goal: bool = True,
        self_loops: bool = True,
        print_graph_config: bool = True,
    ):
        super().__init__()
        self.num_nodes = int(num_nodes)
        self.per_object_dim = int(per_object_dim)
        self.text_dim = int(text_dim)
        self.hidden_dim = int(hidden_dim)
        self.out_dim = int(out_dim)
        self.dropout = float(dropout)
        self.edge_mode = str(edge_mode)
        self.random_edges = int(random_edges)
        self.validate_graph = bool(validate_graph)
        self.topology_cache_size = int(topology_cache_size)

        # Public configuration flags requested by the model config.
        self.include_node_metric = bool(include_node_metric)
        self.include_node_is_goal = bool(include_node_is_goal)
        self.self_loops = bool(self_loops)
        self.edge_dim = EDGE_DIM

        if self.per_object_dim != len(RAW_NODE_FIELDS):
            raise ValueError(
                f"per_object_dim must remain {len(RAW_NODE_FIELDS)} because the "
                f"raw transport interface is {RAW_NODE_FIELDS}, got {self.per_object_dim}"
            )
        if self.edge_mode not in {"goal_star", "goal_star_random", "complete"}:
            raise ValueError(f"Unsupported edge_mode={self.edge_mode!r}")
        payload = torch.load(embeddings_path, map_location="cpu")
        if "id_to_name_emb" not in payload or "id_to_color_emb" not in payload:
            raise KeyError(
                "GraphEncoder expects a compact object-id cache with keys "
                "'id_to_name_emb' and 'id_to_color_emb'. Regenerate it with create_cod.py."
            )

        name_emb = payload["id_to_name_emb"].float()
        color_emb = payload["id_to_color_emb"].float()
        if name_emb.shape != color_emb.shape:
            raise ValueError(
                "name/color embedding tables must have same shape, got "
                f"{name_emb.shape} and {color_emb.shape}"
            )

        self.register_buffer("id_to_name_emb", name_emb, persistent=False)
        self.register_buffer("id_to_color_emb", color_emb, persistent=False)
        clip_dim = int(name_emb.shape[-1])

        self.name_proj = nn.Sequential(
            nn.Linear(clip_dim, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, text_dim),
        )
        self.color_proj = nn.Sequential(
            nn.Linear(clip_dim, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, text_dim),
        )

        if self.include_node_metric:
            self.pos_proj: Optional[nn.Module] = nn.Sequential(
                nn.Linear(3, text_dim),
                nn.ReLU(inplace=True),
                nn.Linear(text_dim, text_dim),
            )
        else:
            self.pos_proj = None

        if self.include_node_is_goal:
            self.goal_proj: Optional[nn.Module] = nn.Sequential(
                nn.Linear(1, text_dim),
                nn.ReLU(inplace=True),
                nn.Linear(text_dim, text_dim),
            )
        else:
            self.goal_proj = None

        node_blocks = 2
        if self.include_node_metric:
            node_blocks += 1
        if self.include_node_is_goal:
            node_blocks += 1
        node_in = text_dim * node_blocks

        self.node_mlp = nn.Sequential(
            nn.Linear(node_in, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
        )

        if hidden_dim % heads != 0:
            raise ValueError(
                f"hidden_dim={hidden_dim} must be divisible by heads={heads}"
            )
        self.convs = nn.ModuleList(
            [
                GATv2Conv(
                    hidden_dim,
                    hidden_dim // heads,
                    heads=heads,
                    edge_dim=self.edge_dim,
                    dropout=dropout,
                    concat=True,
                    add_self_loops=False,
                )
                for _ in range(int(num_layers))
            ]
        )
        self.norms = nn.ModuleList(
            [nn.LayerNorm(hidden_dim) for _ in range(int(num_layers))]
        )
        self.attn_score = nn.Linear(hidden_dim, 1)
        self.head = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, out_dim),
        )

        self.class_color_to_id: dict[str, int] = dict(
            payload.get("class_color_to_id", {})
        )
        self.name_to_default_id: dict[str, int] = dict(
            payload.get("name_to_default_id", {})
        )
        self.scene_graph_cache: dict[int, torch.Tensor] = {}
        self._topology_cache: OrderedDict[Hashable, PackedTopology] = OrderedDict()

        if graphs_dir is not None:
            self._load_scene_graphs(graphs_dir)
        if print_graph_config:
            self._print_graph_configuration()

    @property
    def learned_node_fields(self) -> tuple[str, ...]:
        fields = ["name_embedding", "color_embedding"]
        if self.include_node_metric:
            fields.append("xyz_metric_embedding")
        if self.include_node_is_goal:
            fields.append("is_goal_embedding")
        return tuple(fields)

    def _print_graph_configuration(self) -> None:
        line = "=" * 78
        print(f"\n{line}")
        print("[ GRAPH ENCODER CONFIGURATION ]")
        print(line)
        print("Raw transport fields (not all are learned node features):")
        for idx, field in enumerate(RAW_NODE_FIELDS):
            role = {
                "object_id": "semantic lookup",
                "active": "topology/filter only; NEVER a learned node feature",
                "is_goal": "goal selection + optional learned node feature",
                "x_env": "edge construction + optional metric node feature",
                "y_env": "edge construction + optional metric node feature",
                "z_env": "optional metric node feature",
            }[field]
            print(f"  [{idx}] {field:<10} : {role}")

        print("\nLearned node interface:")
        print(f"  include_node_metric  : {'ON' if self.include_node_metric else 'OFF'}")
        print(
            "  include_node_is_goal : "
            f"{'ON' if self.include_node_is_goal else 'OFF'}"
        )
        for field in self.learned_node_fields:
            print(f"    - {field}")

        print("\nEdge topology:")
        print(f"  edge_mode            : {self.edge_mode}")
        print("  goal relation edges  : goal -> object")
        print("  parallel edges/pair  : 2 (Y relation + X relation)")
        print(f"  self_loops           : {'ON' if self.self_loops else 'OFF'}")
        print("  relation payload     : [direction_id, distance_id] -> fixed one-hot")
        print(
            "  one-hot dimensions   : "
            f"direction={EDGE_DIRECTION_DIM}, distance={EDGE_DISTANCE_DIM}, "
            f"total={EDGE_DIM}"
        )

        print("\nDirection IDs:")
        for label, idx in DIRECTION_TO_ID.items():
            print(f"  {idx:>2} : {label}")

        print("\nDistance IDs (axis-wise metres):")
        distance_desc = {
            "very_close": "0 <= d < 5",
            "close": "5 <= d < 10",
            "far": "10 <= d < 15",
            "very_far": "15 <= d <= 20 (and defensive overflow)",
        }
        for label, idx in DISTANCE_TO_ID.items():
            print(f"  {idx:>2} : {label:<11} [{distance_desc[label]}]")

        print("\nCoordinate convention: delta = object - goal")
        print("  +Y -> in_front_of, -Y -> behind")
        print("  -X -> left_of,     +X -> right_of")
        print(line + "\n")

    @staticmethod
    def _cc_key(name: str, color: str) -> str:
        return f"{_norm_name(name)}|{_norm_color(color)}"

    def _lookup_object_id(self, name: str, color: str | None = None) -> int:
        key = self._cc_key(name, color or "gray")
        if key in self.class_color_to_id:
            return int(self.class_color_to_id[key])
        base = _norm_name(name)
        return int(self.name_to_default_id.get(base, 0))

    def clear_topology_cache(self) -> None:
        self._topology_cache.clear()

    def _load_scene_graphs(self, graphs_dir: str) -> None:
        pattern = os.path.join(graphs_dir, "scene_*_graph.json")
        files = sorted(glob.glob(pattern))
        if not files:
            print(f"[GraphEncoder] WARNING: no scene graph files found at {pattern}")
            return
        for fpath in files:
            base = os.path.basename(fpath)
            try:
                scene_id = int(base.split("_")[1])
            except (IndexError, ValueError):
                print(f"[GraphEncoder] Skipping {base}: cannot parse scene_id")
                continue
            graph = self._parse_scene_graph_json(fpath)
            if graph is not None:
                self.scene_graph_cache[scene_id] = graph
        print(
            f"[GraphEncoder] Cached {len(self.scene_graph_cache)} compact scene graphs"
        )

    def _parse_scene_graph_json(self, fpath: str) -> Optional[torch.Tensor]:
        with open(fpath, "r", encoding="utf-8") as f:
            data = json.load(f)
        nodes = data.get("nodes", {})
        if not nodes:
            return None

        rows = []
        for node in nodes.values():
            raw_name = node.get("class_name", "")
            color = node.get("color", node.get("base_color", "gray"))
            obj_id = self._lookup_object_id(raw_name, color)
            obb = node.get("bbox_3d", {}).get("obb", {})
            center = obb.get("center", [0.0, 0.0, 0.0])
            rows.append(
                [
                    float(obj_id),
                    1.0,
                    0.0,
                    float(center[0]),
                    float(center[1]),
                    float(center[2]),
                ]
            )
        graph = torch.tensor(rows, dtype=torch.float32)
        graph[:, IS_GOAL_IDX] = 0.0
        graph[0, IS_GOAL_IDX] = 1.0
        return graph

    def _cache_get(self, key: Optional[Hashable]) -> Optional[PackedTopology]:
        if key is None or self.topology_cache_size <= 0:
            return None
        topo = self._topology_cache.get(key)
        if topo is not None:
            self._topology_cache.move_to_end(key)
        return topo

    def _cache_put(self, key: Optional[Hashable], topo: PackedTopology) -> None:
        if key is None or self.topology_cache_size <= 0:
            return
        self._topology_cache[key] = topo
        self._topology_cache.move_to_end(key)
        while len(self._topology_cache) > self.topology_cache_size:
            self._topology_cache.popitem(last=False)

    def _validate_goal_counts(
        self, active: torch.Tensor, is_goal: torch.Tensor
    ) -> None:
        if not self.validate_graph:
            return
        non_empty = active.any(dim=1)
        counts = (active & is_goal).sum(dim=1)
        bad = non_empty & (counts != 1)
        if bool(bad.any().item()):
            bad_ids = (
                torch.nonzero(bad, as_tuple=False)
                .view(-1)
                .detach()
                .cpu()
                .tolist()
            )
            bad_counts = counts[bad].detach().cpu().tolist()
            pairs = list(zip(bad_ids, bad_counts))[:16]
            raise ValueError(
                "Expected exactly one active goal per non-empty graph. "
                f"Bad batch entries: {pairs}"
            )

    def _active_pack_metadata(self, node_raw: torch.Tensor):
        device = node_raw.device
        B, N, _ = node_raw.shape
        active = node_raw[:, :, ACTIVE_IDX] > 0.5
        is_goal = node_raw[:, :, IS_GOAL_IDX] > 0.5
        self._validate_goal_counts(active, is_goal)

        active_flat_idx = torch.nonzero(
            active.reshape(-1), as_tuple=False
        ).view(-1)
        active_count = int(active_flat_idx.numel())
        if active_count > 0:
            batch_active = torch.div(active_flat_idx, N, rounding_mode="floor")
            local_active = active_flat_idx - batch_active * N
        else:
            batch_active = torch.empty(0, dtype=torch.long, device=device)
            local_active = torch.empty(0, dtype=torch.long, device=device)

        empty_batches = torch.nonzero(
            ~active.any(dim=1), as_tuple=False
        ).view(-1)

        active_goal = active & is_goal
        has_goal = active_goal.any(dim=1)
        first_goal = active_goal.to(torch.int64).argmax(dim=1)
        first_active = active.to(torch.int64).argmax(dim=1)
        goal_local = torch.where(has_goal, first_goal, first_active)

        packed_by_local = torch.full(
            (B, N), -1, dtype=torch.long, device=device
        )
        if active_count > 0:
            packed_by_local.reshape(-1)[active_flat_idx] = torch.arange(
                active_count, dtype=torch.long, device=device
            )

        return (
            active,
            active_flat_idx,
            active_count,
            batch_active,
            local_active,
            empty_batches,
            goal_local,
            packed_by_local,
        )

    def _build_goal_star_topology(self, node_raw: torch.Tensor) -> PackedTopology:
        device = node_raw.device
        B, N, _ = node_raw.shape
        (
            active,
            active_flat_idx,
            active_count,
            batch_active,
            local_active,
            empty_batches,
            goal_local,
            packed_by_local,
        ) = self._active_pack_metadata(node_raw)

        if active_count > 0:
            active_packed = torch.arange(
                active_count, dtype=torch.long, device=device
            )
            goal_packed_per_batch = packed_by_local[
                torch.arange(B, device=device), goal_local
            ]
            goal_for_active = goal_packed_per_batch[batch_active]
            is_chosen_goal = local_active == goal_local[batch_active]
            object_nodes = active_packed[~is_chosen_goal]
            object_goals = goal_for_active[~is_chosen_goal]

            edge_index, edge_kind = _parallel_axis_edges(
                object_goals, object_nodes
            )
            if self.self_loops:
                edge_index, edge_kind = _append_self_edges(
                    edge_index, edge_kind, active_packed
                )
        else:
            edge_index = torch.empty((2, 0), dtype=torch.long, device=device)
            edge_kind = torch.empty(0, dtype=torch.long, device=device)

        empty_count = int(empty_batches.numel())
        if empty_count > 0:
            # Internal fallback loop is independent of user self-loop semantics.
            fallback_nodes = active_count + torch.arange(
                empty_count, dtype=torch.long, device=device
            )
            edge_index, edge_kind = _append_self_edges(
                edge_index, edge_kind, fallback_nodes
            )

        batch_vec = torch.cat([batch_active, empty_batches], dim=0)
        return PackedTopology(
            B=B,
            N=N,
            active_flat_idx=active_flat_idx,
            empty_batches=empty_batches,
            batch_vec=batch_vec,
            edge_index=edge_index,
            edge_kind=edge_kind,
        )

    def _build_complete_topology(self, node_raw: torch.Tensor) -> PackedTopology:
        device = node_raw.device
        B, N, _ = node_raw.shape
        (
            active,
            active_flat_idx,
            active_count,
            batch_active,
            _local_active,
            empty_batches,
            _goal_local,
            packed_by_local,
        ) = self._active_pack_metadata(node_raw)

        pair_mask = active[:, :, None] & active[:, None, :]
        eye = torch.eye(N, dtype=torch.bool, device=device).unsqueeze(0)
        pair_mask &= ~eye
        pair_idx = torch.nonzero(pair_mask, as_tuple=False)

        if pair_idx.numel() > 0:
            b = pair_idx[:, 0]
            src = packed_by_local[b, pair_idx[:, 1]]
            dst = packed_by_local[b, pair_idx[:, 2]]
            edge_index, edge_kind = _parallel_axis_edges(src, dst)
        else:
            edge_index = torch.empty((2, 0), dtype=torch.long, device=device)
            edge_kind = torch.empty(0, dtype=torch.long, device=device)

        if self.self_loops and active_count > 0:
            nodes = torch.arange(active_count, dtype=torch.long, device=device)
            edge_index, edge_kind = _append_self_edges(
                edge_index, edge_kind, nodes
            )

        empty_count = int(empty_batches.numel())
        if empty_count > 0:
            fallback_nodes = active_count + torch.arange(
                empty_count, dtype=torch.long, device=device
            )
            edge_index, edge_kind = _append_self_edges(
                edge_index, edge_kind, fallback_nodes
            )

        batch_vec = torch.cat([batch_active, empty_batches], dim=0)
        return PackedTopology(
            B=B,
            N=N,
            active_flat_idx=active_flat_idx,
            empty_batches=empty_batches,
            batch_vec=batch_vec,
            edge_index=edge_index,
            edge_kind=edge_kind,
        )

    def _build_topology(self, node_raw: torch.Tensor) -> PackedTopology:
        if self.edge_mode == "goal_star":
            return self._build_goal_star_topology(node_raw)
        if self.edge_mode == "complete":
            return self._build_complete_topology(node_raw)
        raise RuntimeError(
            "goal_star_random uses _pack_node_tensor_random_legacy, "
            "not topology cache"
        )

    def _pack_node_tensor_random_legacy(
        self,
        node_raw: torch.Tensor,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        device = node_raw.device
        B, _N, D = node_raw.shape
        if D != self.per_object_dim:
            raise ValueError(
                f"Expected per-object dim {self.per_object_dim}, got {D}"
            )

        all_ids = []
        all_pos = []
        all_goal = []
        all_batch = []
        edge_indices = []
        edge_type_ids = []
        offset = 0

        for b in range(B):
            ids = node_raw[b, :, OBJECT_ID_IDX].round().long()
            active = node_raw[b, :, ACTIVE_IDX] > 0.5
            is_goal = node_raw[b, :, IS_GOAL_IDX] > 0.5
            pos = node_raw[b, :, POSITION_SLICE].float()

            has_active = bool(active.any().item())
            if has_active:
                ids_b = ids[active]
                pos_b = pos[active]
                goal_b = is_goal[active]
            else:
                ids_b = torch.zeros(1, dtype=torch.long, device=device)
                pos_b = torch.zeros(1, 3, dtype=torch.float32, device=device)
                goal_b = torch.ones(1, dtype=torch.bool, device=device)

            n = int(ids_b.shape[0])
            all_ids.append(ids_b)
            all_pos.append(pos_b)
            all_goal.append(goal_b.float())
            all_batch.append(
                torch.full((n,), b, dtype=torch.long, device=device)
            )

            ei, et = build_goal_star_random_edges(
                pos_b,
                goal_b,
                offset,
                device,
                num_random_edges=self.random_edges,
                # A synthetic fallback node always gets one internal loop.
                self_edges=self.self_loops or not has_active,
            )
            edge_indices.append(ei)
            edge_type_ids.append(et)
            offset += n

        return (
            torch.cat(all_ids, dim=0),
            torch.cat(all_pos, dim=0),
            torch.cat(all_goal, dim=0),
            torch.cat(all_batch, dim=0),
            torch.cat(edge_indices, dim=1),
            torch.cat(edge_type_ids, dim=0),
        )

    def _topology_cache_key(
        self,
        topology_cache_key: Optional[Hashable],
        B: int,
        N: int,
        device: torch.device,
    ) -> Optional[Hashable]:
        if topology_cache_key is None:
            return None
        return (
            self.edge_mode,
            self.self_loops,
            B,
            N,
            str(device),
            topology_cache_key,
        )

    def _pack_node_tensor(
        self,
        node_raw: torch.Tensor,
        topology_cache_key: Optional[Hashable] = None,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        device = node_raw.device
        B, N, D = node_raw.shape
        if D != self.per_object_dim:
            raise ValueError(
                f"Expected per-object dim {self.per_object_dim}, got {D}"
            )
        if self.edge_mode == "goal_star_random":
            return self._pack_node_tensor_random_legacy(node_raw)

        full_key = self._topology_cache_key(
            topology_cache_key, B, N, device
        )
        topo = self._cache_get(full_key)
        if topo is None:
            topo = self._build_topology(node_raw)
            self._cache_put(full_key, topo)

        flat = node_raw.reshape(B * N, D)
        if topo.active_flat_idx.numel() > 0:
            active_rows = flat[topo.active_flat_idx]
            node_ids = active_rows[:, OBJECT_ID_IDX].round().long()
            node_is_goal = active_rows[:, IS_GOAL_IDX].float()
            node_pos = active_rows[:, POSITION_SLICE].float()
        else:
            node_ids = torch.empty(0, dtype=torch.long, device=device)
            node_is_goal = torch.empty(0, dtype=torch.float32, device=device)
            node_pos = torch.empty(0, 3, dtype=torch.float32, device=device)

        empty_count = int(topo.empty_batches.numel())
        if empty_count > 0:
            node_ids = torch.cat(
                [
                    node_ids,
                    torch.zeros(empty_count, dtype=torch.long, device=device),
                ],
                dim=0,
            )
            node_is_goal = torch.cat(
                [
                    node_is_goal,
                    torch.ones(empty_count, dtype=torch.float32, device=device),
                ],
                dim=0,
            )
            node_pos = torch.cat(
                [
                    node_pos,
                    torch.zeros(
                        empty_count, 3, dtype=torch.float32, device=device
                    ),
                ],
                dim=0,
            )

        edge_index = topo.edge_index
        delta = node_pos[edge_index[1]] - node_pos[edge_index[0]]
        edge_type_ids = build_edge_type_ids(delta, topo.edge_kind)
        return (
            node_ids,
            node_pos,
            node_is_goal,
            topo.batch_vec,
            edge_index,
            edge_type_ids,
        )

    def _pack_flat_graph(
        self,
        graph_flat: torch.Tensor,
        topology_cache_key: Optional[Hashable] = None,
    ) -> tuple[
        tuple[
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
        ],
        int,
    ]:
        if graph_flat.dim() != 2:
            graph_flat = torch.flatten(graph_flat, start_dim=1)
        B = int(graph_flat.shape[0])
        if graph_flat.shape[1] % self.per_object_dim != 0:
            raise ValueError(
                f"graph_flat dim {graph_flat.shape[1]} is not divisible by "
                f"per_object_dim={self.per_object_dim}"
            )
        N = graph_flat.shape[1] // self.per_object_dim
        node_raw = graph_flat.reshape(B, N, self.per_object_dim)
        return (
            self._pack_node_tensor(
                node_raw, topology_cache_key=topology_cache_key
            ),
            B,
        )

    @staticmethod
    def _encode_edge_types(edge_type_ids: torch.Tensor) -> torch.Tensor:
        """Convert categorical edge IDs to fixed one-hot edge features.

        The IDs themselves are stable labels, not ordinal numeric features and
        not trainable parameters. GATv2 learns how to process the resulting
        one-hot vectors through its own edge-feature weights.
        """
        if edge_type_ids.dim() != 2 or edge_type_ids.shape[-1] != 2:
            raise ValueError(
                "edge_type_ids must have shape [E, 2] as "
                "[direction_id, distance_id]"
            )

        direction_ids = edge_type_ids[:, 0].long()
        distance_ids = edge_type_ids[:, 1].long()

        if direction_ids.numel() > 0:
            if direction_ids.min() < 0 or direction_ids.max() >= EDGE_DIRECTION_DIM:
                raise ValueError("direction_id is outside DIRECTION_TO_ID")
            if distance_ids.min() < 0 or distance_ids.max() >= EDGE_DISTANCE_DIM:
                raise ValueError("distance_id is outside DISTANCE_TO_ID")

        direction_one_hot = F.one_hot(
            direction_ids, num_classes=EDGE_DIRECTION_DIM
        )
        distance_one_hot = F.one_hot(
            distance_ids, num_classes=EDGE_DISTANCE_DIM
        )
        return torch.cat(
            [direction_one_hot, distance_one_hot], dim=-1
        ).to(dtype=torch.float32)

    def _encode_packed(
        self,
        node_ids: torch.Tensor,
        node_pos: torch.Tensor,
        node_is_goal: torch.Tensor,
        batch_vec: torch.Tensor,
        edge_index: torch.Tensor,
        edge_type_ids: torch.Tensor,
        B: int,
    ) -> torch.Tensor:
        max_id = self.id_to_name_emb.shape[0] - 1
        node_ids = node_ids.clamp(0, max_id)

        features = [
            self.name_proj(self.id_to_name_emb[node_ids]),
            self.color_proj(self.id_to_color_emb[node_ids]),
        ]
        if self.include_node_metric:
            assert self.pos_proj is not None
            features.append(self.pos_proj(node_pos))
        if self.include_node_is_goal:
            assert self.goal_proj is not None
            features.append(self.goal_proj(node_is_goal.unsqueeze(-1)))

        x = self.node_mlp(torch.cat(features, dim=-1))
        edge_attr = self._encode_edge_types(edge_type_ids)

        for conv, norm in zip(self.convs, self.norms):
            residual = x
            x = conv(x, edge_index, edge_attr=edge_attr)
            x = norm(x + residual)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        mean_pool = global_mean_pool(x, batch_vec, size=B)
        max_pool = global_max_pool(x, batch_vec, size=B)
        weights = softmax(self.attn_score(x).squeeze(-1), batch_vec)
        attn_pool = global_add_pool(
            x * weights.unsqueeze(-1), batch_vec, size=B
        )
        return self.head(torch.cat([mean_pool, max_pool, attn_pool], dim=-1))

    def _forward_from_flat(
        self,
        graph_flat: torch.Tensor,
        topology_cache_key: Optional[Hashable] = None,
    ) -> torch.Tensor:
        packed, B = self._pack_flat_graph(
            graph_flat, topology_cache_key=topology_cache_key
        )
        return self._encode_packed(*packed, B=B)

    def _forward_from_json_scenes(
        self, scene_ids: torch.Tensor
    ) -> torch.Tensor:
        if not self.scene_graph_cache:
            raise RuntimeError(
                "scene_ids were provided, but no scene graph cache is loaded"
            )

        device = next(self.parameters()).device
        scene_list = [
            int(sid)
            for sid in scene_ids.detach().cpu().view(-1).tolist()
        ]
        fallback = next(iter(self.scene_graph_cache.values()))
        rows = [self.scene_graph_cache.get(sid, fallback) for sid in scene_list]

        B = len(rows)
        max_nodes = max(int(row.shape[0]) for row in rows)
        node_raw = torch.zeros(
            B,
            max_nodes,
            self.per_object_dim,
            dtype=torch.float32,
            device=device,
        )
        for b, row in enumerate(rows):
            row_dev = row.to(device=device, dtype=torch.float32)
            n = int(row_dev.shape[0])
            node_raw[b, :n, :] = row_dev
            node_raw[b, :n, ACTIVE_IDX] = 1.0

        topo_key = ("json_scenes", tuple(scene_list))
        packed = self._pack_node_tensor(
            node_raw, topology_cache_key=topo_key
        )
        return self._encode_packed(*packed, B=B)

    def encode_graph(
        self,
        graph_flat: torch.Tensor,
        scene_ids: Optional[torch.Tensor] = None,
        topology_cache_key: Optional[Hashable] = None,
    ) -> torch.Tensor:
        """Return one graph embedding per batch item: ``[B, out_dim]``."""
        if scene_ids is not None:
            return self._forward_from_json_scenes(scene_ids)
        return self._forward_from_flat(
            graph_flat, topology_cache_key=topology_cache_key
        )

    def forward(
        self,
        graph_flat: torch.Tensor,
        scene_ids: Optional[torch.Tensor] = None,
        topology_cache_key: Optional[Hashable] = None,
    ) -> torch.Tensor:
        # One-shot scene dump for inspect_graph.py: GIROL_DUMP_GRAPH=1
        import os as _os
        if _os.environ.get("GIROL_DUMP_GRAPH") == "1" and not getattr(self, "_dump_done", False):
            self._dump_done = True
            _os.makedirs("logs", exist_ok=True)
            torch.save(graph_flat.detach().cpu(), "logs/scene_dump.pt")
            print(f"[dump] saved graph_flat {tuple(graph_flat.shape)} -> logs/scene_dump.pt "
                  "(inspect with scripts/algos/inspect_graph.py)")
        return self.encode_graph(
            graph_flat,
            scene_ids=scene_ids,
            topology_cache_key=topology_cache_key,
        )
