# -*- coding: utf-8 -*-
"""
Hierarchical room-aware (non-metric) graph encoder for the DDQN pipeline.

Drop-in for perception.graph_encoder.GraphEncoder in the config system:
    "graph_encoder": {
      "class_path": "perception.hierarchical_graph_encoder.HierarchicalGraphEncoder",
      "kwargs": { "embeddings_path": {...}, "out_dim": 128, "dropout": 0.1,
                  "include_node_metric": false },
      "eval": true
    }

Interface (matches the pipeline):
    enc = HierarchicalGraphEncoder(embeddings_path=..., out_dim=128, include_node_metric=False)
    graph_emb = enc(graph_flat)          # graph_flat: [B, 6*M] -> [B, out_dim]

Graph (built inside forward from the flat obs [object_id, active, is_goal, x, y, z]):
  Nodes:  M object nodes + R=4 quadrant-room nodes (hierarchy; no scene node, no robot).
  Edges (bidirectional, typed): goal-star (goal<->object), object<->room containment,
         room<->room, self-loops.
  Edge features (DIRECTION-ONLY, non-metric): [relation, x_dir, y_dir, room_relation]
         3-way per axis at a threshold; NO distance / coordinates in nodes or edges
         (unless include_node_metric=True, which re-adds xyz to object nodes).
  Readout (goal-centric): [h_goal, h_goal_room, h_global] -> MLP -> out_dim.

Only x,y are read to DERIVE room membership + qualitative directions at build time;
raw coordinates are not fed to the GNN (when include_node_metric=False).
"""
import os
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv

REL_SELF, REL_GOAL_STAR, REL_OBJ_IN_ROOM, REL_ROOM_HAS_OBJ, REL_ROOM_ROOM = 0, 1, 2, 3, 4
NUM_REL = 5
DIR_NONE, DIR_NEG, DIR_ALIGNED, DIR_POS = 0, 1, 2, 3          # neg=left/behind, pos=right/front
NUM_DIR = 4
ROOMREL_NONE, ROOMREL_SAME, ROOMREL_DIFF = 0, 1, 2
NUM_ROOMREL = 3


class HierarchicalGraphEncoder(nn.Module):
    def __init__(
        self,
        embeddings_path: str,
        out_dim: int = 128,
        dropout: float = 0.1,
        include_node_metric: bool = False,
        num_rooms: int = 4,
        text_dim: int = 16,
        hidden_dim: int = 128,
        num_layers: int = 3,
        heads: int = 4,
        align_threshold: float = 0.4,
        print_graph_config: bool = False,
        **kwargs,                    # tolerate/ignore perception-specific kwargs (graphs_dir, edge_mode, ...)
    ):
        super().__init__()
        self.use_metric = bool(include_node_metric)
        self.R = int(num_rooms)
        self.text_dim = int(text_dim)
        self.hidden_dim = int(hidden_dim)
        self.dropout = float(dropout)
        self.align_threshold = float(align_threshold)
        self._dump_done = False
        if hidden_dim % heads != 0:
            raise ValueError(f"hidden_dim={hidden_dim} must be divisible by heads={heads}")

        payload = torch.load(embeddings_path, map_location="cpu")
        id_name = payload["id_to_name_emb"].float()
        self.register_buffer("id_to_name_emb", id_name, persistent=False)
        self.max_object_id = id_name.shape[0] - 1
        clip_dim = id_name.shape[-1]

        self.name_proj = nn.Sequential(
            nn.Linear(clip_dim, 128), nn.ReLU(inplace=True), nn.Linear(128, text_dim))
        self.pos_proj = None
        if self.use_metric:
            self.pos_proj = nn.Sequential(
                nn.Linear(3, text_dim), nn.ReLU(inplace=True), nn.Linear(text_dim, text_dim))

        obj_in = text_dim + 2 + (text_dim if self.use_metric else 0)
        self.object_mlp = nn.Sequential(
            nn.Linear(obj_in, hidden_dim), nn.ReLU(inplace=True), nn.Linear(hidden_dim, hidden_dim))
        self.room_mlp = nn.Sequential(
            nn.Linear(1 + 2 + 2, hidden_dim), nn.ReLU(inplace=True), nn.Linear(hidden_dim, hidden_dim))

        self.rel_emb = nn.Embedding(NUM_REL, 8)
        self.xdir_emb = nn.Embedding(NUM_DIR, 4)
        self.ydir_emb = nn.Embedding(NUM_DIR, 4)
        self.roomrel_emb = nn.Embedding(NUM_ROOMREL, 4)
        edge_dim = 8 + 4 + 4 + 4

        self.convs = nn.ModuleList([
            GATv2Conv(hidden_dim, hidden_dim // heads, heads=heads, edge_dim=edge_dim,
                      dropout=dropout, concat=True, add_self_loops=False)
            for _ in range(num_layers)])
        self.norms = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(num_layers)])
        self.readout = nn.Sequential(
            nn.Linear(hidden_dim * 3, 256), nn.ReLU(inplace=True),
            nn.Dropout(dropout), nn.Linear(256, out_dim))

        R = self.R
        x_right = torch.tensor([1 if (r % 2 == 0) else 0 for r in range(R)])
        y_front = torch.tensor([1 if (r // 2 == 0) else 0 for r in range(R)])
        self.register_buffer("x_zone_onehot", torch.stack([x_right, 1 - x_right], -1).float(), persistent=False)
        self.register_buffer("y_zone_onehot", torch.stack([y_front, 1 - y_front], -1).float(), persistent=False)

        ri, rj, xdir, ydir = [], [], [], []
        xpos = lambda r: 1.0 if (r % 2 == 0) else -1.0
        ypos = lambda r: 1.0 if (r // 2 == 0) else -1.0
        for i in range(R):
            for j in range(R):
                if i == j:
                    continue
                ri.append(i); rj.append(j)
                dx, dy = xpos(j) - xpos(i), ypos(j) - ypos(i)
                xdir.append(DIR_POS if dx > 0 else (DIR_NEG if dx < 0 else DIR_ALIGNED))
                ydir.append(DIR_POS if dy > 0 else (DIR_NEG if dy < 0 else DIR_ALIGNED))
        self.register_buffer("rr_ri", torch.tensor(ri, dtype=torch.long), persistent=False)
        self.register_buffer("rr_rj", torch.tensor(rj, dtype=torch.long), persistent=False)
        self.register_buffer("rr_xdir", torch.tensor(xdir, dtype=torch.long), persistent=False)
        self.register_buffer("rr_ydir", torch.tensor(ydir, dtype=torch.long), persistent=False)

        if print_graph_config:
            print(f"[HierarchicalGraphEncoder] rooms={R} metric={self.use_metric} "
                  f"layers={num_layers} hidden={hidden_dim} out={out_dim} "
                  f"edges=goal_star+containment+room_room+self  edge_feats=direction_only(no distance)")

    def _dir(self, delta):
        ids = torch.full(delta.shape, DIR_ALIGNED, dtype=torch.long, device=delta.device)
        ids = torch.where(delta < -self.align_threshold, torch.full_like(ids, DIR_NEG), ids)
        ids = torch.where(delta > self.align_threshold, torch.full_like(ids, DIR_POS), ids)
        return ids

    def _edge_encoder(self, fields):
        return torch.cat([self.rel_emb(fields[:, 0]), self.xdir_emb(fields[:, 1]),
                          self.ydir_emb(fields[:, 2]), self.roomrel_emb(fields[:, 3])], dim=-1)

    def forward(self, graph_flat: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        if os.environ.get("GIROL_DUMP_GRAPH") == "1" and not self._dump_done:
            self._dump_done = True
            os.makedirs("logs", exist_ok=True)
            torch.save(graph_flat.detach().cpu(), "logs/scene_dump.pt")
            print(f"[dump] saved graph_flat {tuple(graph_flat.shape)} -> logs/scene_dump.pt")

        B, D = graph_flat.shape[0], 6
        M = graph_flat.shape[1] // D
        R, N = self.R, M + self.R
        device = graph_flat.device
        ar = torch.arange(B, device=device)

        g = graph_flat.view(B, M, D)
        object_id = g[..., 0].long().clamp(0, self.max_object_id)
        active, is_goal = g[..., 1], g[..., 2]
        pos, xy = g[..., 3:6], g[..., 3:5]

        goal_idx = is_goal.argmax(dim=1)
        room_id = (xy[..., 0] < 0).long() + 2 * (xy[..., 1] < 0).long()
        goal_room = room_id[ar, goal_idx]

        name_feat = self.name_proj(self.id_to_name_emb[object_id])
        obj_in = torch.cat([name_feat, active.unsqueeze(-1), is_goal.unsqueeze(-1)], dim=-1)
        if self.use_metric:
            obj_in = torch.cat([obj_in, self.pos_proj(pos)], dim=-1)
        h_obj = self.object_mlp(obj_in)

        is_goal_room = (torch.arange(R, device=device)[None, :] == goal_room[:, None]).float()
        xz = self.x_zone_onehot.unsqueeze(0).expand(B, R, 2)
        yz = self.y_zone_onehot.unsqueeze(0).expand(B, R, 2)
        h_room = self.room_mlp(torch.cat([is_goal_room.unsqueeze(-1), xz, yz], dim=-1))

        h = torch.cat([h_obj, h_room], dim=1).reshape(B * N, -1)

        off = (ar * N).view(B, 1)
        objL = torch.arange(M, device=device).view(1, M)
        src_l, dst_l, fld_l = [], [], []

        selfL = torch.arange(N, device=device).view(1, N)
        s = (off + selfL).reshape(-1)
        src_l.append(s); dst_l.append(s)
        f = torch.zeros(B * N, 4, dtype=torch.long, device=device); f[:, 0] = REL_SELF
        fld_l.append(f)

        pos_goal = xy[ar, goal_idx]
        dx = xy[..., 0] - pos_goal[:, 0:1]
        dy = xy[..., 1] - pos_goal[:, 1:2]
        xdir, ydir = self._dir(dx), self._dir(dy)
        roomrel = torch.where(room_id == goal_room[:, None],
                              torch.full_like(room_id, ROOMREL_SAME),
                              torch.full_like(room_id, ROOMREL_DIFF))
        goal_glob = (off + goal_idx.view(B, 1)).expand(B, M).reshape(-1)
        obj_glob = (off + objL).reshape(-1)
        rel_gs = torch.full_like(xdir, REL_GOAL_STAR)
        src_l.append(goal_glob); dst_l.append(obj_glob)
        fld_l.append(torch.stack([rel_gs, xdir, ydir, roomrel], -1).reshape(-1, 4))
        src_l.append(obj_glob); dst_l.append(goal_glob)
        fld_l.append(torch.stack([rel_gs, self._dir(-dx), self._dir(-dy), roomrel], -1).reshape(-1, 4))

        room_node = (off + M + room_id).reshape(-1)
        src_l.append(obj_glob); dst_l.append(room_node)
        f = torch.zeros(B * M, 4, dtype=torch.long, device=device); f[:, 0] = REL_OBJ_IN_ROOM; f[:, 3] = ROOMREL_SAME
        fld_l.append(f)
        src_l.append(room_node); dst_l.append(obj_glob)
        f = torch.zeros(B * M, 4, dtype=torch.long, device=device); f[:, 0] = REL_ROOM_HAS_OBJ; f[:, 3] = ROOMREL_SAME
        fld_l.append(f)

        P = self.rr_ri.shape[0]
        src_l.append((off + M + self.rr_ri.view(1, P)).reshape(-1))
        dst_l.append((off + M + self.rr_rj.view(1, P)).reshape(-1))
        xrr = self.rr_xdir.view(1, P).expand(B, P).reshape(-1)
        yrr = self.rr_ydir.view(1, P).expand(B, P).reshape(-1)
        fld_l.append(torch.stack([torch.full_like(xrr, REL_ROOM_ROOM), xrr, yrr,
                                  torch.full_like(xrr, ROOMREL_DIFF)], -1))

        edge_index = torch.stack([torch.cat(src_l), torch.cat(dst_l)], dim=0)
        edge_attr = self._edge_encoder(torch.cat(fld_l, dim=0))

        for conv, norm in zip(self.convs, self.norms):
            m = conv(h, edge_index, edge_attr)
            h = norm(h + F.dropout(F.relu(m), p=self.dropout, training=self.training))

        hv = h.view(B, N, -1)
        return self.readout(torch.cat(
            [hv[ar, goal_idx], hv[ar, M + goal_room], hv.mean(dim=1)], dim=-1))
