# -*- coding: utf-8 -*-
"""
Room-Aware Qualitative (Non-Metric) Graph Encoder for GIROL orientation.

Drop-in replacement for networks_orm.GraphEncoder:
    enc = NonMetricGraphEncoder(embeddings_path=..., env=env, use_metric=False)
    graph_emb = enc(graph_flat)          # graph_flat: [B, 6*M] -> [B, out_dim]

The env observation is unchanged: per object [object_id, active, is_goal, x, y, z].
Raw x,y are read ONLY to derive qualitative categories at build time, then dropped
from the node features (unless use_metric=True, for a metric variant on this encoder).

Graph (built inside forward from the flat obs):
  Nodes: M object nodes + R=4 room nodes (scene quadrants). No scene node, no robot.
  Edges (bidirectional, typed):
    - goal-star:  goal object  <-> every object
    - containment: object <-> its room
    - room-room:  every ordered room pair
    - self-loops
  Edge features (DIRECTION ONLY, non-metric): categorical
    [relation_type, x_direction, y_direction, room_relation]  (learned embeddings)
    directions are 3-way per axis (neg / aligned / pos) at a 0.4 m threshold; NO
    distance/gap magnitude is stored.
  Readout (goal-centric, static / robot-independent):
    [h_goal, h_goal_room, h_global] -> MLP -> out_dim
"""
import os
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv

# ---- categorical vocabularies (kept small and explicit) ---------------------
REL_SELF, REL_GOAL_STAR, REL_OBJ_IN_ROOM, REL_ROOM_HAS_OBJ, REL_ROOM_ROOM = 0, 1, 2, 3, 4
NUM_REL = 5

DIR_NONE, DIR_NEG, DIR_ALIGNED, DIR_POS = 0, 1, 2, 3   # neg=left/behind, pos=right/front
NUM_DIR = 4

ROOMREL_NONE, ROOMREL_SAME, ROOMREL_DIFF = 0, 1, 2
NUM_ROOMREL = 3


class NonMetricGraphEncoder(nn.Module):
    def __init__(
        self,
        embeddings_path: str,
        env=None,                      # accepted for interface compat; unused
        graphs_dir: Optional[str] = None,   # unused
        use_metric: bool = False,
        num_rooms: int = 4,
        text_dim: int = 16,
        hidden_dim: int = 128,
        out_dim: int = 128,
        num_layers: int = 3,
        heads: int = 4,
        dropout: float = 0.1,
        align_threshold: float = 0.4,
    ):
        super().__init__()
        self.use_metric = bool(use_metric)
        self.R = int(num_rooms)
        self.text_dim = int(text_dim)
        self.hidden_dim = int(hidden_dim)
        self.dropout = float(dropout)
        self.align_threshold = float(align_threshold)

        if hidden_dim % heads != 0:
            raise ValueError(f"hidden_dim={hidden_dim} must be divisible by heads={heads}")

        # --- CLIP name table, indexed directly by object_id (0..16) ---
        payload = torch.load(embeddings_path, map_location="cpu")
        id_name = payload["id_to_name_emb"].float()          # [num_ids+1, clip_dim]
        self.register_buffer("id_to_name_emb", id_name, persistent=False)
        self.max_object_id = id_name.shape[0] - 1
        clip_dim = id_name.shape[-1]

        self.name_proj = nn.Sequential(
            nn.Linear(clip_dim, 128), nn.ReLU(inplace=True), nn.Linear(128, text_dim),
        )
        # metric variant: project raw xyz and concat into object features
        self.pos_proj = None
        if self.use_metric:
            self.pos_proj = nn.Sequential(
                nn.Linear(3, text_dim), nn.ReLU(inplace=True), nn.Linear(text_dim, text_dim),
            )

        obj_in = text_dim + 2 + (text_dim if self.use_metric else 0)  # name + [active,is_goal] (+pos)
        self.object_mlp = nn.Sequential(
            nn.Linear(obj_in, hidden_dim), nn.ReLU(inplace=True), nn.Linear(hidden_dim, hidden_dim),
        )
        room_in = 1 + 2 + 2  # is_goal_room + x_zone(onehot) + y_zone(onehot)
        self.room_mlp = nn.Sequential(
            nn.Linear(room_in, hidden_dim), nn.ReLU(inplace=True), nn.Linear(hidden_dim, hidden_dim),
        )

        # --- edge feature encoder: per-field embeddings ---
        self.rel_emb = nn.Embedding(NUM_REL, 8)
        self.xdir_emb = nn.Embedding(NUM_DIR, 4)
        self.ydir_emb = nn.Embedding(NUM_DIR, 4)
        self.roomrel_emb = nn.Embedding(NUM_ROOMREL, 4)
        edge_dim = 8 + 4 + 4 + 4  # 20

        self.convs = nn.ModuleList([
            GATv2Conv(hidden_dim, hidden_dim // heads, heads=heads, edge_dim=edge_dim,
                      dropout=dropout, concat=True, add_self_loops=False)
            for _ in range(num_layers)
        ])
        self.norms = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(num_layers)])

        self.readout = nn.Sequential(
            nn.Linear(hidden_dim * 3, 256), nn.ReLU(inplace=True),
            nn.Dropout(dropout), nn.Linear(256, out_dim),
        )

        # --- static per-room geometry (quadrant zones) ---
        # room_id = (x<0) + 2*(y<0):  0:R/F  1:L/F  2:R/B  3:L/B
        R = self.R
        x_right = torch.tensor([1 if (r % 2 == 0) else 0 for r in range(R)])  # right=1
        y_front = torch.tensor([1 if (r // 2 == 0) else 0 for r in range(R)])  # front=1
        # onehot [right, left] / [front, back]
        x_zone = torch.stack([x_right, 1 - x_right], dim=-1).float()
        y_zone = torch.stack([y_front, 1 - y_front], dim=-1).float()
        self.register_buffer("x_zone_onehot", x_zone, persistent=False)  # [R,2]
        self.register_buffer("y_zone_onehot", y_zone, persistent=False)  # [R,2]

        # static room-room edges (all ordered pairs) + their qualitative directions
        ri, rj, xdir, ydir = [], [], [], []
        xpos = lambda r: 1.0 if (r % 2 == 0) else -1.0   # right>0
        ypos = lambda r: 1.0 if (r // 2 == 0) else -1.0  # front>0
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

        self._dump_done = False   # one-shot GIROL_DUMP_GRAPH capture (see inspect_graph.py)

    def _dir(self, delta: torch.Tensor) -> torch.Tensor:
        """3-way axis direction id from a signed offset (non-metric: sign only)."""
        ids = torch.full(delta.shape, DIR_ALIGNED, dtype=torch.long, device=delta.device)
        ids = torch.where(delta < -self.align_threshold, torch.full_like(ids, DIR_NEG), ids)
        ids = torch.where(delta > self.align_threshold, torch.full_like(ids, DIR_POS), ids)
        return ids

    def _edge_encoder(self, fields: torch.Tensor) -> torch.Tensor:
        # fields: [E, 4] -> [relation, x_dir, y_dir, room_rel]
        return torch.cat([
            self.rel_emb(fields[:, 0]),
            self.xdir_emb(fields[:, 1]),
            self.ydir_emb(fields[:, 2]),
            self.roomrel_emb(fields[:, 3]),
        ], dim=-1)

    def forward(self, graph_flat: torch.Tensor) -> torch.Tensor:
        if (not self._dump_done) and os.environ.get("GIROL_DUMP_GRAPH") == "1":
            self._dump_done = True
            os.makedirs("logs", exist_ok=True)
            torch.save(graph_flat.detach().cpu(), "logs/scene_dump.pt")
            print(f"[dump] saved graph_flat {tuple(graph_flat.shape)} -> logs/scene_dump.pt "
                  "(inspect with scripts/algos/inspect_graph.py)")
        B = graph_flat.shape[0]
        D = 6
        M = graph_flat.shape[1] // D
        R = self.R
        N = M + R
        device = graph_flat.device
        ar = torch.arange(B, device=device)

        g = graph_flat.view(B, M, D)
        object_id = g[..., 0].long().clamp(0, self.max_object_id)  # [B,M]
        active = g[..., 1]                                         # [B,M]
        is_goal = g[..., 2]                                        # [B,M]
        pos = g[..., 3:6]                                          # [B,M,3]
        xy = pos[..., :2]                                          # [B,M,2]

        goal_idx = is_goal.argmax(dim=1)                           # [B]
        room_id = (xy[..., 0] < 0).long() + 2 * (xy[..., 1] < 0).long()  # [B,M] in 0..3
        goal_room = room_id[ar, goal_idx]                          # [B]

        # ---------- node features ----------
        name_feat = self.name_proj(self.id_to_name_emb[object_id])       # [B,M,td]
        obj_in = torch.cat([name_feat, active.unsqueeze(-1), is_goal.unsqueeze(-1)], dim=-1)
        if self.use_metric:
            obj_in = torch.cat([obj_in, self.pos_proj(pos)], dim=-1)
        h_obj = self.object_mlp(obj_in)                                  # [B,M,H]

        is_goal_room = (torch.arange(R, device=device)[None, :] == goal_room[:, None]).float()  # [B,R]
        xz = self.x_zone_onehot.unsqueeze(0).expand(B, R, 2)
        yz = self.y_zone_onehot.unsqueeze(0).expand(B, R, 2)
        room_in = torch.cat([is_goal_room.unsqueeze(-1), xz, yz], dim=-1)  # [B,R,5]
        h_room = self.room_mlp(room_in)                                  # [B,R,H]

        h = torch.cat([h_obj, h_room], dim=1).reshape(B * N, -1)         # [B*N, H]

        # ---------- edges (batched, env-major with offset e*N) ----------
        off = (ar * N).view(B, 1)                                        # [B,1]
        objL = torch.arange(M, device=device).view(1, M)                # [1,M]
        src_l, dst_l, fld_l = [], [], []

        # (A) self-loops
        selfL = torch.arange(N, device=device).view(1, N)
        s = (off + selfL).reshape(-1)
        src_l.append(s); dst_l.append(s)
        f = torch.zeros(B * N, 4, dtype=torch.long, device=device); f[:, 0] = REL_SELF
        fld_l.append(f)

        # (B) goal-star  goal<->object
        pos_goal = xy[ar, goal_idx]                                     # [B,2]
        dx = xy[..., 0] - pos_goal[:, 0:1]                              # [B,M]
        dy = xy[..., 1] - pos_goal[:, 1:2]
        xdir, ydir = self._dir(dx), self._dir(dy)
        roomrel = torch.where(room_id == goal_room[:, None],
                              torch.full_like(room_id, ROOMREL_SAME),
                              torch.full_like(room_id, ROOMREL_DIFF))
        goal_glob = (off + goal_idx.view(B, 1)).expand(B, M).reshape(-1)
        obj_glob = (off + objL).reshape(-1)
        rel_gs = torch.full_like(xdir, REL_GOAL_STAR)
        # goal -> object
        src_l.append(goal_glob); dst_l.append(obj_glob)
        fld_l.append(torch.stack([rel_gs, xdir, ydir, roomrel], -1).reshape(-1, 4))
        # object -> goal (reverse: negate offsets)
        src_l.append(obj_glob); dst_l.append(goal_glob)
        fld_l.append(torch.stack([rel_gs, self._dir(-dx), self._dir(-dy), roomrel], -1).reshape(-1, 4))

        # (C) containment  object<->room
        room_node = (off + M + room_id).reshape(-1)                    # [B*M]
        src_l.append(obj_glob); dst_l.append(room_node)
        f = torch.zeros(B * M, 4, dtype=torch.long, device=device)
        f[:, 0] = REL_OBJ_IN_ROOM; f[:, 3] = ROOMREL_SAME
        fld_l.append(f)
        src_l.append(room_node); dst_l.append(obj_glob)
        f = torch.zeros(B * M, 4, dtype=torch.long, device=device)
        f[:, 0] = REL_ROOM_HAS_OBJ; f[:, 3] = ROOMREL_SAME
        fld_l.append(f)

        # (D) room-room (all ordered pairs)
        P = self.rr_ri.shape[0]
        src_l.append((off + M + self.rr_ri.view(1, P)).reshape(-1))
        dst_l.append((off + M + self.rr_rj.view(1, P)).reshape(-1))
        xrr = self.rr_xdir.view(1, P).expand(B, P).reshape(-1)
        yrr = self.rr_ydir.view(1, P).expand(B, P).reshape(-1)
        fld_l.append(torch.stack([
            torch.full_like(xrr, REL_ROOM_ROOM), xrr, yrr, torch.full_like(xrr, ROOMREL_DIFF)
        ], -1))

        edge_index = torch.stack([torch.cat(src_l), torch.cat(dst_l)], dim=0)
        edge_attr = self._edge_encoder(torch.cat(fld_l, dim=0))

        # ---------- message passing ----------
        for conv, norm in zip(self.convs, self.norms):
            m = conv(h, edge_index, edge_attr)
            h = norm(h + F.dropout(F.relu(m), p=self.dropout, training=self.training))

        # ---------- goal-centric readout ----------
        hv = h.view(B, N, -1)
        h_goal = hv[ar, goal_idx]                 # [B,H]
        h_goal_room = hv[ar, M + goal_room]       # [B,H]
        h_global = hv.mean(dim=1)                 # [B,H]
        return self.readout(torch.cat([h_goal, h_goal_room, h_global], dim=-1))
