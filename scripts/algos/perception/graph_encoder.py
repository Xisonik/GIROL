"""Fast compact scene graph encoder.

Input contract:
    graph_flat [B, num_nodes * 6]
    per object = [object_id, active, is_goal, x_room, y_room, z_room]

The hot path is `edge_mode="goal_star"`. It is fully vectorized over the batch:
no `for b in range(B)`, no `torch.unique`, no per-sample `randperm`.

SAC note:
    This module exposes `encode_graph(...)` as an explicit one-shot graph
    embedding call. To actually reuse the same graph embedding across actor,
    critic_1, critic_2 and target critics, the SAC model/runner must call this
    once and pass the returned embedding into the heads. A module cannot remove
    repeated upstream calls by itself.
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
from torch_geometric.nn import GATv2Conv, global_add_pool, global_max_pool, global_mean_pool
from torch_geometric.utils import softmax

NUM_GRAPH_NODES = 21
PER_OBJECT_DIM = 6
GRAPH_EMB_DIM = 128

# edge_attr = [left_of, right_of]
# Relations are evaluated in the room top-down frame. No metric geometry is
# duplicated in edge_attr: positions already live in node features.
RELATION_LABELS = ("left_of", "right_of")
RELATION_DIM = len(RELATION_LABELS)
EDGE_DIM = RELATION_DIM


def _norm_name(name: str) -> str:
    return str(name or "").split("_", 1)[0].lower()


def _norm_color(color: str | None) -> str:
    return str(color or "gray").strip().lower() or "gray"


def build_topdown_relation_attr(delta: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Classify dst relative to src as left/right only.

    delta is pos[dst] - pos[src]. Relations are evaluated in absolute room
    coordinates, as if looking at the XY layout from the room center with a
    camera perpendicular to the floor plane. They are not robot-centric.

    Convention used here:
        +Y -> left_of
        -Y -> right_of

    If |dy| <= eps, both attributes are 0. There is intentionally no distance,
    front/back, same-position, or continuous geometry in edge_attr.
    """
    if delta.dim() != 2 or delta.shape[-1] != 3:
        raise ValueError(f"delta must have shape [E, 3], got {tuple(delta.shape)}")

    dy = delta[:, 1]
    rel = torch.zeros(delta.shape[0], RELATION_DIM, device=delta.device, dtype=delta.dtype)
    rel[:, 0] = (dy > eps).to(delta.dtype)   # left_of
    rel[:, 1] = (dy < -eps).to(delta.dtype)  # right_of
    return rel


def build_edge_attr_from_delta(delta: torch.Tensor) -> torch.Tensor:
    """Build relation-only edge attributes: [left_of, right_of]."""
    return build_topdown_relation_attr(delta)


# -----------------------------------------------------------------------------
# Legacy single-graph builders kept for ablations and backwards imports.
# The GraphEncoder hot path does not use them for edge_mode="goal_star".
# -----------------------------------------------------------------------------


def build_complete_graph_edges(pos: torch.Tensor, offset: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    """Build directed complete graph for one active-node set."""
    n = int(pos.shape[0])
    if n <= 0:
        raise ValueError("build_complete_graph_edges expects at least one node")

    local = torch.arange(n, device=device)
    src = local.repeat_interleave(n)
    dst = local.repeat(n)
    delta = pos[dst] - pos[src]
    edge_attr = build_edge_attr_from_delta(delta)
    edge_index = torch.stack([src + offset, dst + offset], dim=0)
    return edge_index, edge_attr


def build_goal_star_edges(
    pos: torch.Tensor,
    goal_mask: torch.Tensor,
    offset: int,
    device: torch.device,
    *,
    bidirectional: bool = True,
    self_edges: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build a goal-centered sparse graph for one active-node set.

    No `torch.unique` is used. Duplicates are removed analytically:
        goal -> every node
        non-goal -> goal          if bidirectional=True
        non-goal -> itself        if self_edges=True

    The goal self-loop is already present in `goal -> every node`.
    """
    n = int(pos.shape[0])
    if n <= 0:
        raise ValueError("build_goal_star_edges expects at least one node")
    if goal_mask.shape[0] != n:
        raise ValueError(f"goal_mask length {goal_mask.shape[0]} does not match node count {n}")

    goal_candidates = torch.nonzero(goal_mask, as_tuple=False).view(-1)
    if goal_candidates.numel() != 1:
        raise ValueError(f"Expected exactly one active goal node, got {goal_candidates.numel()}")
    goal = goal_candidates[0].long()

    nodes = torch.arange(n, device=device, dtype=torch.long)
    non_goal = nodes != goal

    src_parts = [goal.expand(n)]
    dst_parts = [nodes]

    if bidirectional and bool(non_goal.any()):
        src_parts.append(nodes[non_goal])
        dst_parts.append(goal.expand(int(non_goal.sum().item())))

    if self_edges and bool(non_goal.any()):
        src_parts.append(nodes[non_goal])
        dst_parts.append(nodes[non_goal])

    src = torch.cat(src_parts, dim=0)
    dst = torch.cat(dst_parts, dim=0)
    delta = pos[dst] - pos[src]
    edge_attr = build_edge_attr_from_delta(delta)
    edge_index = torch.stack([src + offset, dst + offset], dim=0)
    return edge_index, edge_attr


def build_goal_star_random_edges(
    pos: torch.Tensor,
    goal_mask: torch.Tensor,
    offset: int,
    device: torch.device,
    *,
    num_random_edges: int = 16,
    bidirectional_star: bool = True,
    self_edges: bool = True,
    random_bidirectional: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build bidirectional goal-star plus random object-object edges.

    This remains a legacy ablation path. It is intentionally not the fast path:
    per-graph random topology is expensive and does not cache safely.
    """
    n = int(pos.shape[0])
    if n <= 0:
        raise ValueError("build_goal_star_random_edges expects at least one node")
    if goal_mask.shape[0] != n:
        raise ValueError(f"goal_mask length {goal_mask.shape[0]} does not match node count {n}")

    goal_candidates = torch.nonzero(goal_mask, as_tuple=False).view(-1)
    if goal_candidates.numel() != 1:
        raise ValueError(f"Expected exactly one active goal node, got {goal_candidates.numel()}")
    goal = goal_candidates[0].long()

    nodes = torch.arange(n, device=device, dtype=torch.long)
    non_goal = nodes != goal
    src_parts = [goal.expand(n)]
    dst_parts = [nodes]

    if bidirectional_star and bool(non_goal.any()):
        src_parts.append(nodes[non_goal])
        dst_parts.append(goal.expand(int(non_goal.sum().item())))

    if self_edges and bool(non_goal.any()):
        src_parts.append(nodes[non_goal])
        dst_parts.append(nodes[non_goal])

    obj_nodes = nodes[non_goal]
    m = int(obj_nodes.numel())
    if m >= 2 and num_random_edges > 0:
        src_grid = obj_nodes.repeat_interleave(m)
        dst_grid = obj_nodes.repeat(m)
        valid = src_grid != dst_grid
        src_grid = src_grid[valid]
        dst_grid = dst_grid[valid]

        sample_count = min(int(num_random_edges), int(src_grid.numel()))
        perm = torch.randperm(src_grid.numel(), device=device)[:sample_count]
        rand_src = src_grid[perm]
        rand_dst = dst_grid[perm]

        src_parts.append(rand_src)
        dst_parts.append(rand_dst)

        if random_bidirectional:
            src_parts.append(rand_dst)
            dst_parts.append(rand_src)

    src = torch.cat(src_parts, dim=0)
    dst = torch.cat(dst_parts, dim=0)
    delta = pos[dst] - pos[src]
    edge_attr = build_edge_attr_from_delta(delta)
    edge_index = torch.stack([src + offset, dst + offset], dim=0)
    return edge_index, edge_attr


@dataclass(frozen=True)
class PackedTopology:
    """Topology-only data for a packed PyG batch.

    `active_flat_idx` gathers active rows from node_raw.reshape(B * N, D).
    Empty graphs get one synthetic fallback node appended after all active nodes.
    """

    B: int
    N: int
    active_flat_idx: torch.Tensor
    empty_batches: torch.Tensor
    batch_vec: torch.Tensor
    edge_index: torch.Tensor


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
            raise ValueError(f"name/color embedding tables must have same shape, got {name_emb.shape} and {color_emb.shape}")

        self.register_buffer("id_to_name_emb", name_emb, persistent=False)
        self.register_buffer("id_to_color_emb", color_emb, persistent=False)
        clip_dim = int(name_emb.shape[-1])

        self.name_proj = nn.Sequential(nn.Linear(clip_dim, 128), nn.ReLU(inplace=True), nn.Linear(128, text_dim))
        self.color_proj = nn.Sequential(nn.Linear(clip_dim, 128), nn.ReLU(inplace=True), nn.Linear(128, text_dim))
        self.pos_proj = nn.Sequential(nn.Linear(3, text_dim), nn.ReLU(inplace=True), nn.Linear(text_dim, text_dim))

        node_in = text_dim * 3
        self.node_mlp = nn.Sequential(nn.Linear(node_in, hidden_dim), nn.ReLU(inplace=True), nn.Linear(hidden_dim, hidden_dim))

        if hidden_dim % heads != 0:
            raise ValueError(f"hidden_dim={hidden_dim} must be divisible by heads={heads}")
        self.convs = nn.ModuleList([
            GATv2Conv(
                hidden_dim,
                hidden_dim // heads,
                heads=heads,
                edge_dim=EDGE_DIM,
                dropout=dropout,
                concat=True,
                add_self_loops=False,
            )
            for _ in range(int(num_layers))
        ])
        self.norms = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(int(num_layers))])
        self.attn_score = nn.Linear(hidden_dim, 1)
        self.head = nn.Sequential(nn.Linear(hidden_dim * 3, hidden_dim), nn.ReLU(inplace=True), nn.Linear(hidden_dim, out_dim))

        self.class_color_to_id: dict[str, int] = dict(payload.get("class_color_to_id", {}))
        self.name_to_default_id: dict[str, int] = dict(payload.get("name_to_default_id", {}))
        self.scene_graph_cache: dict[int, torch.Tensor] = {}
        self._topology_cache: OrderedDict[Hashable, PackedTopology] = OrderedDict()

        if graphs_dir is not None:
            self._load_scene_graphs(graphs_dir)

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
        """Clear explicit topology cache.

        Use this if object active masks or goal assignment changed but the caller
        reused the same `topology_cache_key`.
        """
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
        print(f"[GraphEncoder] Cached {len(self.scene_graph_cache)} compact scene graphs")

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
            rows.append([float(obj_id), 1.0, 0.0, float(center[0]), float(center[1]), float(center[2])])
        graph = torch.tensor(rows, dtype=torch.float32)
        # JSON scene cache has no task target metadata. Use the first node as a
        # deterministic placeholder target for scene_id-based ablations.
        graph[:, 2] = 0.0
        graph[0, 2] = 1.0
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

    def _validate_goal_counts(self, active: torch.Tensor, is_goal: torch.Tensor) -> None:
        """Optional correctness check. Disabled by default to avoid GPU sync."""
        if not self.validate_graph:
            return
        non_empty = active.any(dim=1)
        counts = (active & is_goal).sum(dim=1)
        bad = non_empty & (counts != 1)
        if bool(bad.any().item()):
            bad_ids = torch.nonzero(bad, as_tuple=False).view(-1).detach().cpu().tolist()
            bad_counts = counts[bad].detach().cpu().tolist()
            pairs = list(zip(bad_ids, bad_counts))[:16]
            raise ValueError(f"Expected exactly one active goal per non-empty graph. Bad batch entries: {pairs}")

    def _build_goal_star_topology(self, node_raw: torch.Tensor) -> PackedTopology:
        device = node_raw.device
        B, N, _ = node_raw.shape

        active = node_raw[:, :, 1] > 0.5
        is_goal = node_raw[:, :, 2] > 0.5
        self._validate_goal_counts(active, is_goal)

        active_flat = active.reshape(-1)
        active_flat_idx = torch.nonzero(active_flat, as_tuple=False).view(-1)
        active_count = int(active_flat_idx.numel())

        if active_count > 0:
            batch_active = torch.div(active_flat_idx, N, rounding_mode="floor")
            local_active = active_flat_idx - batch_active * N
        else:
            batch_active = torch.empty(0, dtype=torch.long, device=device)
            local_active = torch.empty(0, dtype=torch.long, device=device)

        empty_batches = torch.nonzero(~active.any(dim=1), as_tuple=False).view(-1)
        empty_count = int(empty_batches.numel())

        # Choose the active goal. If validation is disabled and a graph is bad,
        # fall back to the first active node. This avoids host synchronization in
        # the normal training path.
        active_goal = active & is_goal
        has_goal = active_goal.any(dim=1)
        first_goal = active_goal.to(torch.int64).argmax(dim=1)
        first_active = active.to(torch.int64).argmax(dim=1)
        goal_local = torch.where(has_goal, first_goal, first_active)

        packed_by_local = torch.full((B, N), -1, dtype=torch.long, device=device)
        if active_count > 0:
            packed_by_local.reshape(-1)[active_flat_idx] = torch.arange(active_count, dtype=torch.long, device=device)
            active_packed = torch.arange(active_count, dtype=torch.long, device=device)
            goal_packed_per_batch = packed_by_local[torch.arange(B, device=device), goal_local]
            goal_for_active = goal_packed_per_batch[batch_active]
            is_chosen_goal_node = local_active == goal_local[batch_active]
            non_goal = ~is_chosen_goal_node

            # Unique goal-star topology, analytically deduplicated:
            #   goal -> all active nodes
            #   non-goal -> goal
            #   non-goal -> itself
            edge_src = torch.cat([goal_for_active, active_packed[non_goal], active_packed[non_goal]], dim=0)
            edge_dst = torch.cat([active_packed, goal_for_active[non_goal], active_packed[non_goal]], dim=0)
        else:
            edge_src = torch.empty(0, dtype=torch.long, device=device)
            edge_dst = torch.empty(0, dtype=torch.long, device=device)

        if empty_count > 0:
            fallback_nodes = active_count + torch.arange(empty_count, dtype=torch.long, device=device)
            edge_src = torch.cat([edge_src, fallback_nodes], dim=0)
            edge_dst = torch.cat([edge_dst, fallback_nodes], dim=0)

        batch_vec = torch.cat([batch_active, empty_batches], dim=0)
        edge_index = torch.stack([edge_src, edge_dst], dim=0)
        return PackedTopology(B=B, N=N, active_flat_idx=active_flat_idx, empty_batches=empty_batches, batch_vec=batch_vec, edge_index=edge_index)

    def _build_complete_topology(self, node_raw: torch.Tensor) -> PackedTopology:
        device = node_raw.device
        B, N, _ = node_raw.shape

        active = node_raw[:, :, 1] > 0.5
        active_flat = active.reshape(-1)
        active_flat_idx = torch.nonzero(active_flat, as_tuple=False).view(-1)
        active_count = int(active_flat_idx.numel())

        if active_count > 0:
            batch_active = torch.div(active_flat_idx, N, rounding_mode="floor")
        else:
            batch_active = torch.empty(0, dtype=torch.long, device=device)

        empty_batches = torch.nonzero(~active.any(dim=1), as_tuple=False).view(-1)
        empty_count = int(empty_batches.numel())

        packed_by_local = torch.full((B, N), -1, dtype=torch.long, device=device)
        if active_count > 0:
            packed_by_local.reshape(-1)[active_flat_idx] = torch.arange(active_count, dtype=torch.long, device=device)

        pair_mask = active[:, :, None] & active[:, None, :]
        pair_idx = torch.nonzero(pair_mask, as_tuple=False)
        if int(pair_idx.shape[0]) > 0:
            b = pair_idx[:, 0]
            src_local = pair_idx[:, 1]
            dst_local = pair_idx[:, 2]
            edge_src = packed_by_local[b, src_local]
            edge_dst = packed_by_local[b, dst_local]
        else:
            edge_src = torch.empty(0, dtype=torch.long, device=device)
            edge_dst = torch.empty(0, dtype=torch.long, device=device)

        if empty_count > 0:
            fallback_nodes = active_count + torch.arange(empty_count, dtype=torch.long, device=device)
            edge_src = torch.cat([edge_src, fallback_nodes], dim=0)
            edge_dst = torch.cat([edge_dst, fallback_nodes], dim=0)

        batch_vec = torch.cat([batch_active, empty_batches], dim=0)
        edge_index = torch.stack([edge_src, edge_dst], dim=0)
        return PackedTopology(B=B, N=N, active_flat_idx=active_flat_idx, empty_batches=empty_batches, batch_vec=batch_vec, edge_index=edge_index)

    def _build_topology(self, node_raw: torch.Tensor) -> PackedTopology:
        if self.edge_mode == "goal_star":
            return self._build_goal_star_topology(node_raw)
        if self.edge_mode == "complete":
            return self._build_complete_topology(node_raw)
        raise RuntimeError("goal_star_random uses _pack_node_tensor_random_legacy, not topology cache")

    def _pack_node_tensor_random_legacy(
        self,
        node_raw: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Original-style stochastic packing for the random ablation mode."""
        device = node_raw.device
        B, N, D = node_raw.shape
        if D != self.per_object_dim:
            raise ValueError(f"Expected per-object dim {self.per_object_dim}, got {D}")

        all_ids = []
        all_pos = []
        all_batch = []
        edge_indices = []
        edge_attrs = []
        offset = 0

        for b in range(B):
            ids = node_raw[b, :, 0].round().long()
            active = node_raw[b, :, 1] > 0.5
            is_goal = node_raw[b, :, 2] > 0.5
            pos = node_raw[b, :, 3:6].float()

            if bool(active.any().item()):
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
            all_batch.append(torch.full((n,), b, dtype=torch.long, device=device))

            ei, ea = build_goal_star_random_edges(
                pos_b,
                goal_b,
                offset,
                device,
                num_random_edges=self.random_edges,
            )
            edge_indices.append(ei)
            edge_attrs.append(ea)
            offset += n

        return (
            torch.cat(all_ids, dim=0),
            torch.cat(all_pos, dim=0),
            torch.cat(all_batch, dim=0),
            torch.cat(edge_indices, dim=1),
            torch.cat(edge_attrs, dim=0),
        )

    def _topology_cache_key(self, topology_cache_key: Optional[Hashable], B: int, N: int, device: torch.device) -> Optional[Hashable]:
        if topology_cache_key is None:
            return None
        return (self.edge_mode, B, N, str(device), topology_cache_key)

    def _pack_node_tensor(
        self,
        node_raw: torch.Tensor,
        topology_cache_key: Optional[Hashable] = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        device = node_raw.device
        B, N, D = node_raw.shape
        if D != self.per_object_dim:
            raise ValueError(f"Expected per-object dim {self.per_object_dim}, got {D}")
        if self.edge_mode == "goal_star_random":
            return self._pack_node_tensor_random_legacy(node_raw)

        full_key = self._topology_cache_key(topology_cache_key, B, N, device)
        topo = self._cache_get(full_key)
        if topo is None:
            topo = self._build_topology(node_raw)
            self._cache_put(full_key, topo)

        flat = node_raw.reshape(B * N, D)
        if int(topo.active_flat_idx.numel()) > 0:
            active_rows = flat[topo.active_flat_idx]
            node_ids = active_rows[:, 0].round().long()
            node_pos = active_rows[:, 3:6].float()
        else:
            node_ids = torch.empty(0, dtype=torch.long, device=device)
            node_pos = torch.empty(0, 3, dtype=torch.float32, device=device)

        empty_count = int(topo.empty_batches.numel())
        if empty_count > 0:
            node_ids = torch.cat([node_ids, torch.zeros(empty_count, dtype=torch.long, device=device)], dim=0)
            node_pos = torch.cat([node_pos, torch.zeros(empty_count, 3, dtype=torch.float32, device=device)], dim=0)

        edge_index = topo.edge_index
        edge_attr = build_edge_attr_from_delta(node_pos[edge_index[1]] - node_pos[edge_index[0]])
        return node_ids, node_pos, topo.batch_vec, edge_index, edge_attr

    def _pack_flat_graph(
        self,
        graph_flat: torch.Tensor,
        topology_cache_key: Optional[Hashable] = None,
    ) -> tuple[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor], int]:
        if graph_flat.dim() != 2:
            graph_flat = torch.flatten(graph_flat, start_dim=1)
        B = int(graph_flat.shape[0])
        if graph_flat.shape[1] % self.per_object_dim != 0:
            raise ValueError(f"graph_flat dim {graph_flat.shape[1]} is not divisible by per_object_dim={self.per_object_dim}")
        N = graph_flat.shape[1] // self.per_object_dim
        node_raw = graph_flat.reshape(B, N, self.per_object_dim)
        return self._pack_node_tensor(node_raw, topology_cache_key=topology_cache_key), B

    def _encode_packed(self, node_ids, node_pos, batch_vec, edge_index, edge_attr, B: int) -> torch.Tensor:
        max_id = self.id_to_name_emb.shape[0] - 1
        node_ids = node_ids.clamp(0, max_id)
        name = self.name_proj(self.id_to_name_emb[node_ids])
        color = self.color_proj(self.id_to_color_emb[node_ids])
        pos = self.pos_proj(node_pos)
        x = self.node_mlp(torch.cat([name, color, pos], dim=-1))

        for conv, norm in zip(self.convs, self.norms):
            residual = x
            x = conv(x, edge_index, edge_attr=edge_attr)
            x = norm(x + residual)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        mean_pool = global_mean_pool(x, batch_vec, size=B)
        max_pool = global_max_pool(x, batch_vec, size=B)
        weights = softmax(self.attn_score(x).squeeze(-1), batch_vec)
        attn_pool = global_add_pool(x * weights.unsqueeze(-1), batch_vec, size=B)
        return self.head(torch.cat([mean_pool, max_pool, attn_pool], dim=-1))

    def _forward_from_flat(
        self,
        graph_flat: torch.Tensor,
        topology_cache_key: Optional[Hashable] = None,
    ) -> torch.Tensor:
        packed, B = self._pack_flat_graph(graph_flat, topology_cache_key=topology_cache_key)
        return self._encode_packed(*packed, B=B)

    def _forward_from_json_scenes(self, scene_ids: torch.Tensor) -> torch.Tensor:
        if not self.scene_graph_cache:
            raise RuntimeError("scene_ids were provided, but no scene graph cache is loaded")

        device = next(self.parameters()).device
        scene_list = [int(sid) for sid in scene_ids.detach().cpu().view(-1).tolist()]
        fallback = next(iter(self.scene_graph_cache.values()))
        rows = [self.scene_graph_cache.get(sid, fallback) for sid in scene_list]

        B = len(rows)
        max_nodes = max(int(row.shape[0]) for row in rows)
        node_raw = torch.zeros(B, max_nodes, self.per_object_dim, dtype=torch.float32, device=device)
        for b, row in enumerate(rows):
            row_dev = row.to(device=device, dtype=torch.float32)
            n = int(row_dev.shape[0])
            node_raw[b, :n, :] = row_dev
            node_raw[b, :n, 1] = 1.0

        # scene_ids are naturally a stable topology key for JSON-scene ablations.
        topo_key = ("json_scenes", tuple(scene_list))
        packed = self._pack_node_tensor(node_raw, topology_cache_key=topo_key)
        return self._encode_packed(*packed, B=B)

    def encode_graph(
        self,
        graph_flat: torch.Tensor,
        scene_ids: Optional[torch.Tensor] = None,
        topology_cache_key: Optional[Hashable] = None,
    ) -> torch.Tensor:
        """Return graph embedding [B, out_dim].

        `topology_cache_key` is explicit on purpose. A naive automatic cache key
        would require hashing GPU tensors and would usually make SAC slower.
        Pass a key only when the caller can guarantee unchanged active masks and
        unchanged goal assignment for this batch/order.
        """
        if scene_ids is not None:
            return self._forward_from_json_scenes(scene_ids)
        return self._forward_from_flat(graph_flat, topology_cache_key=topology_cache_key)

    def forward(
        self,
        graph_flat: torch.Tensor,
        scene_ids: Optional[torch.Tensor] = None,
        topology_cache_key: Optional[Hashable] = None,
    ) -> torch.Tensor:
        return self.encode_graph(graph_flat, scene_ids=scene_ids, topology_cache_key=topology_cache_key)
