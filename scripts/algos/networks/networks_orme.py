"""
SAC + отдельно обучаемые GraphEncoder и OrientationModule
---------------------------------------------------------
Архитектура:
  - GraphEncoder: CLIP text lookup + GATv2 → graph_emb (128)
  - OrientationModule: img + graph_emb → orientation angle
  - Actor/Critic: используют graph_emb и orientation через no_grad
  - GraphEncoder и OrientationModule имеют СВОИ оптимизаторы,
    обучаются из replay buffer по orientation loss
"""
import json
import os
import glob
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from skrl.utils.spaces.torch import unflatten_tensorized_space, flatten_tensorized_space
from skrl.models.torch import DeterministicMixin, GaussianMixin, Model
from skrl.resources.preprocessors.torch import RunningStandardScaler
from torch_geometric.nn import GATv2Conv, global_mean_pool
from transformers import CLIPProcessor, CLIPModel

# =====================================================================
# Константы
# =====================================================================
NUM_GRAPH_NODES = 21
PER_OBJECT_DIM  = 24
TEXT_EMB_DIM    = 16
GRAPH_EMB_DIM   = 128
NUM_ORIENT_BINS = 36
GOAL_NODE_INDEX = 0

# Пространственные отношения на рёбрах
RELATION_TO_IDX: Dict[str, int] = {
    "in front of":     0,
    "behind":          1,
    "to the left of":  2,
    "left of":         2,  # алиас
    "to the right of": 3,
    "right of":        3,  # алиас
    "above":           4,
    "below":           5,
}
NUM_RELATIONS = 6   # 0-5 — реальные отношения
EDGE_EMB_DIM  = 16  # размерность эмбеддинга ребра
# NUM_RELATIONS используется как padding_idx (self-loops и неизвестные → нулевой вектор)

# =====================================================================
# Accuracy helpers
# =====================================================================
eval_gt_angles    = []
eval_pred_angles  = []
eval_step_counter = 0
step = 0

def collect_orientation_data(gt, pred):
    global eval_gt_angles, eval_pred_angles, eval_step_counter
    eval_gt_angles.append(gt.detach().cpu())
    eval_pred_angles.append(pred.detach().cpu())
    eval_step_counter += 1

def print_orientation_accuracy(peep=False):
    global eval_gt_angles, eval_pred_angles, eval_step_counter, step
    step += 1
    if step > 3000 or peep:
        if not peep:
            step = 0
        if len(eval_gt_angles) == 0:
            print("No orientation data collected")
            return
        gt   = torch.cat(eval_gt_angles,   dim=0)
        pred = torch.cat(eval_pred_angles, dim=0)
        error = torch.abs(gt - pred)
        error = torch.minimum(error, 2 * torch.pi - error)
        acc_10 = (error < 10.0 * torch.pi / 180.0).float().mean().item()
        acc_20 = (error < 20.0 * torch.pi / 180.0).float().mean().item()
        acc_30 = (error < 30.0 * torch.pi / 180.0).float().mean().item()
        if not peep:
            eval_gt_angles.clear()
            eval_pred_angles.clear()
            eval_step_counter = 0
        return acc_10, acc_20, acc_30

# =====================================================================
# Edge builder (star + chain, без типов)
# =====================================================================
def build_star_chain_edge_index(num_nodes, batch_size, device, add_self_loops=True):
    N = num_nodes
    src, dst = [], []
    for i in range(1, N):
        src += [0, i]; dst += [i, 0]
    for i in range(N - 1):
        src += [i, i + 1]; dst += [i + 1, i]
    if add_self_loops:
        for i in range(N):
            src.append(i); dst.append(i)
    ei = torch.tensor([src, dst], device=device, dtype=torch.long)
    return torch.cat([ei + b * N for b in range(batch_size)], dim=1)


# =====================================================================
# GraphEncoder
# =====================================================================
class GraphEncoder(nn.Module):
    """
    graph_flat [B, N*24] → graph_emb [B, 128]

    Два режима:
      1. JSON-граф (основной): graphs_dir задан → сцены из scene_{id}_graph.json.
         Топология: star+chain, центр звезды = нода ближайшая к цели из env.
         Отношения: вычисляются геометрически по позициям нод.
      2. graph_flat (fallback): graphs_dir=None → старый путь.

    Пайплайн:
      CLIP text lookup → text_proj → Node MLP → GATv2(edge_dim) × N
      → global_mean_pool → head → graph_emb
    """

    def __init__(
        self,
        embeddings_path: str,
        env,
        graphs_dir: Optional[str] = None,
        num_nodes:      int   = NUM_GRAPH_NODES,
        per_object_dim: int   = PER_OBJECT_DIM,
        text_dim:       int   = TEXT_EMB_DIM,
        hidden_dim:     int   = 128,
        out_dim:        int   = GRAPH_EMB_DIM,
        num_layers:     int   = 2,
        heads:          int   = 2,
        dropout:        float = 0.1,
    ):
        super().__init__()
        self.num_nodes      = num_nodes
        self.per_object_dim = per_object_dim
        self.text_dim       = text_dim
        self.out_dim        = out_dim
        self.dropout        = dropout

        # --- Frozen CLIP text lookup ---
        payload = torch.load(embeddings_path, map_location="cpu")
        self._name_embs: torch.Tensor = payload["name_embs"].float()  # [K, 512], CPU
        self.register_buffer("color_embs", payload["color_embs"].float(), persistent=False)
        self.register_buffer("name_embs",  self._name_embs,              persistent=False)
        clip_dim = self._name_embs.shape[-1]  # 512

        # name → idx в name_embs. Мутабелен: новые имена добавляются через CLIP.
        self.name_to_idx: Dict[str, int] = {
            "air":       0,
            "box":       1,
            "cabinet":   2,
            "chair":     3,
            "clock":     4,
            "crestwood": 5,
            "desk":      6,
            "ladder":    7,
            "lamp":      8,
            "standard":  9,
            "table":     10,
            "teddy":     11,
            "trashcan":  12,
            "vase":      13,
            "yucca":     14,
            "bowl":      15,
        }

        # CLIP-модель — plain-атрибут, не попадает в state_dict
        self.__dict__['_clip_model']     = None
        self.__dict__['_clip_processor'] = None

        # --- text projection ---
        self.text_proj = nn.Sequential(
            nn.Linear(clip_dim, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, text_dim),
        )

        # --- Edge embedding ---
        # padding_idx=NUM_RELATIONS: self-loops и неизвестные отношения → нулевой вектор
        self.edge_emb = nn.Embedding(NUM_RELATIONS + 1, EDGE_EMB_DIM,
                                     padding_idx=NUM_RELATIONS)

        # --- Node MLP + GATv2 ---
        node_in = per_object_dim + text_dim
        self.node_mlp = nn.Sequential(
            nn.Linear(node_in, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
        )

        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        for _ in range(num_layers):
            self.convs.append(GATv2Conv(
                hidden_dim, hidden_dim // heads,
                heads=heads, edge_dim=EDGE_EMB_DIM, dropout=dropout, concat=True,
            ))
            self.norms.append(nn.LayerNorm(hidden_dim))

        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, out_dim),
        )

        self.env = env
        self._edge_cache:      Dict[Tuple, torch.Tensor]         = {}
        self.scene_graph_cache: Dict[int, dict]                  = {}
        self._json_edge_cache: Dict[Tuple[int, int], torch.Tensor] = {}

        if graphs_dir is not None:
            self._load_scene_graphs(graphs_dir)

    # ------------------------------------------------------------------
    # Загрузка JSON-графов
    # ------------------------------------------------------------------
    def _load_scene_graphs(self, graphs_dir: str) -> None:
        pattern = os.path.join(graphs_dir, "scene_*_graph.json")
        files   = sorted(glob.glob(pattern))
        if not files:
            print(f"[GraphEncoder] WARNING: no scene graph files found at {pattern}")
            return

        GRAY_COLOR_IDX = 7  # позиция 'gray' в color_embs

        for fpath in files:
            basename = os.path.basename(fpath)
            try:
                scene_id = int(basename.split("_")[1])
            except (IndexError, ValueError):
                print(f"[GraphEncoder] Skipping {basename}: cannot parse scene_id")
                continue
            entry = self._parse_scene_graph_json(fpath, scene_id, GRAY_COLOR_IDX)
            if entry is not None:
                self.scene_graph_cache[scene_id] = entry

        print(f"[GraphEncoder] Cached {len(self.scene_graph_cache)} scene graphs: "
              f"{sorted(self.scene_graph_cache.keys())}")
        print(f"[GraphEncoder] name_embs size: {self._name_embs.shape[0]}")

    # ------------------------------------------------------------------
    # CLIP lazy load
    # ------------------------------------------------------------------
    def _ensure_clip_loaded(self) -> None:
        if self.__dict__.get('_clip_model') is not None:
            return
        clip_model     = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to("cpu")
        clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        clip_model.eval()
        self.__dict__['_clip_model']     = clip_model
        self.__dict__['_clip_processor'] = clip_processor
        print("[GraphEncoder] CLIP model loaded for unknown name encoding.")

    def _get_or_encode_name(self, raw_name: str) -> int:
        base_name  = raw_name.lower().split("_")[0]
        full_lower = raw_name.lower()

        if base_name  in self.name_to_idx: return self.name_to_idx[base_name]
        if full_lower in self.name_to_idx: return self.name_to_idx[full_lower]

        self._ensure_clip_loaded()
        print(f"[GraphEncoder] Unknown name '{raw_name}' (base='{base_name}'), encoding via CLIP...")

        clip_model     = self.__dict__['_clip_model']
        clip_processor = self.__dict__['_clip_processor']
        with torch.no_grad():
            inp_base = clip_processor(text=[base_name],  return_tensors="pt", padding=True)
            inp_full = clip_processor(text=[raw_name],   return_tensors="pt", padding=True)
            emb_base = clip_model.get_text_features(**inp_base).float()
            emb_full = clip_model.get_text_features(**inp_full).float()
            new_emb  = 0.5 * (emb_base + emb_full)

        new_idx = self._name_embs.shape[0]
        self._name_embs = torch.cat([self._name_embs, new_emb], dim=0)
        self.register_buffer("name_embs", self._name_embs, persistent=False)
        self.name_to_idx[base_name]  = new_idx
        self.name_to_idx[full_lower] = new_idx
        print(f"[GraphEncoder]   → idx={new_idx}, name_embs now [{self._name_embs.shape[0]}, 512]")
        return new_idx

    # ------------------------------------------------------------------
    # Парсинг JSON
    # ------------------------------------------------------------------
    def _parse_scene_graph_json(
        self,
        fpath: str,
        scene_id: int,
        gray_color_idx: int,
    ) -> Optional[dict]:
        with open(fpath, "r", encoding="utf-8") as f:
            data = json.load(f)

        nodes_data: dict = data.get("nodes", {})
        if not nodes_data:
            print(f"[GraphEncoder] Scene {scene_id}: empty nodes, skipping")
            return None

        track_to_local: Dict[int, int] = {}
        for local_i, node in enumerate(nodes_data.values()):
            track_to_local[node["track_id"]] = local_i

        num_nodes      = len(nodes_data)
        node_feats_list = []
        name_idx_list   = []
        color_idx_list  = []
        src_edges, dst_edges, edge_attrs = [], [], []

        for local_i, node in enumerate(nodes_data.values()):
            obb    = node["bbox_3d"]["obb"]
            center = obb["center"]
            extent = obb["extent"]
            radius = (sum(e ** 2 for e in extent) ** 0.5) / 2.0

            feat = (
                list(center)        # 0-2:  pos
                + list(extent)      # 3-5:  size
                + [radius]          # 6:    radius
                + [0.5, 0.5, 0.5]   # 7-9:  gray color
                + [float(local_i)]  # 10:   object_id
                + [1.0]             # 11:   active
                + [-1.0]            # 12:   parent (нет)
                + [0.0]             # 13:   level
                + [0.0] * 10       # 14-23: нули
            )
            node_feats_list.append(feat)
            name_idx_list.append(self._get_or_encode_name(node.get("class_name", "")))
            color_idx_list.append(gray_color_idx)

            for edge in node.get("edges", []):
                tgt_track = edge.get("target_id")
                tgt_local = track_to_local.get(tgt_track)
                if tgt_local is not None:
                    src_edges.append(local_i)
                    dst_edges.append(tgt_local)
                    rel_str = edge.get("relation_type", "").lower()
                    edge_attrs.append(RELATION_TO_IDX.get(rel_str, NUM_RELATIONS))

        # Self-loops → padding_idx (нулевой эмбеддинг)
        for i in range(num_nodes):
            src_edges.append(i)
            dst_edges.append(i)
            edge_attrs.append(NUM_RELATIONS)

        return {
            "node_feats": torch.tensor(node_feats_list, dtype=torch.float32),
            "name_idx":   torch.tensor(name_idx_list,   dtype=torch.long),
            "color_idx":  torch.tensor(color_idx_list,  dtype=torch.long),
            "edge_index": torch.tensor([src_edges, dst_edges], dtype=torch.long),
            "edge_attr":  torch.tensor(edge_attrs,      dtype=torch.long),
            "num_nodes":  num_nodes,
        }

    # ------------------------------------------------------------------
    # Геометрические отношения по позициям нод
    # ------------------------------------------------------------------
    @staticmethod
    def _compute_geometric_edge_attr(
        edge_index: torch.Tensor,  # [2, E]
        node_feats: torch.Tensor,  # [N, 24], позиции в dims 0-2
    ) -> torch.Tensor:
        """
        Вычисляет relation_id для каждого ребра по доминирующей оси delta = dst - src.
          X+ → in front of (0),  X- → behind (1)
          Y+ → left of (2),      Y- → right of (3)
          Z+ → above (4),        Z- → below (5)
        Self-loops → NUM_RELATIONS (padding → нулевой эмбеддинг).
        """
        src       = edge_index[0]
        dst       = edge_index[1]
        delta     = node_feats[dst, 0:3] - node_feats[src, 0:3]   # [E, 3]
        self_loop = (src == dst)
        dom_axis  = delta.abs().argmax(dim=-1)                     # 0 / 1 / 2

        rel = torch.full((edge_index.shape[1],), NUM_RELATIONS,
                         dtype=torch.long, device=edge_index.device)

        rel[(dom_axis == 0) & (delta[:, 0] >= 0) & ~self_loop] = 0  # in front of
        rel[(dom_axis == 0) & (delta[:, 0] <  0) & ~self_loop] = 1  # behind
        rel[(dom_axis == 1) & (delta[:, 1] >= 0) & ~self_loop] = 2  # left of
        rel[(dom_axis == 1) & (delta[:, 1] <  0) & ~self_loop] = 3  # right of
        rel[(dom_axis == 2) & (delta[:, 2] >= 0) & ~self_loop] = 4  # above
        rel[(dom_axis == 2) & (delta[:, 2] <  0) & ~self_loop] = 5  # below
        return rel

    # ------------------------------------------------------------------
    # Edge index helpers
    # ------------------------------------------------------------------
    def _get_edge_index(self, B: int, device: torch.device) -> torch.Tensor:
        """Star+chain для fallback (num_nodes фиксирован)."""
        key = (B, device.index if device.type == "cuda" else -1)
        ei  = self._edge_cache.get(key)
        if ei is None or ei.device != device:
            ei = build_star_chain_edge_index(self.num_nodes, B, device)
            self._edge_cache[key] = ei
        return ei

    def _get_star_chain_edge_index(self, N: int, B: int, device: torch.device) -> torch.Tensor:
        """Star+chain для произвольного N."""
        key = (N, B, device.index if device.type == "cuda" else -1)
        ei  = self._edge_cache.get(key)
        if ei is None or ei.device != device:
            ei = build_star_chain_edge_index(N, B, device)
            self._edge_cache[key] = ei
        return ei

    # ------------------------------------------------------------------
    # GATv2 forward (общий)
    # ------------------------------------------------------------------
    def _run_gat(
        self,
        x:          torch.Tensor,           # [total_nodes, node_in]
        edge_index: torch.Tensor,           # [2, total_edges]
        batch_vec:  torch.Tensor,           # [total_nodes]
        edge_attr:  Optional[torch.Tensor], # [total_edges], relation ids
    ) -> torch.Tensor:
        x = self.node_mlp(x)
        assert edge_index.max() < x.shape[0], (
            f"edge_index out of bounds: max={edge_index.max().item()} >= {x.shape[0]}"
        )
        ea = self.edge_emb(edge_attr) if edge_attr is not None else None  # [E, EDGE_EMB_DIM]
        for conv, norm in zip(self.convs, self.norms):
            x = conv(x, edge_index, edge_attr=ea)
            x = norm(x)
            x = torch.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        g = global_mean_pool(x, batch_vec)
        return self.head(g)

    # ------------------------------------------------------------------
    # Fallback: graph_flat
    # ------------------------------------------------------------------
    def _encode_text(self, name_idx, color_bits_or_idx):
        """Только для fallback (_forward_from_flat)."""
        if name_idx.dim() == 3:
            name_idx = name_idx.argmax(-1)
        name_idx = name_idx.long().clamp(0, self.name_embs.shape[0] - 1)

        if color_bits_or_idx.dim() == 3 and color_bits_or_idx.size(-1) == 3:
            bits      = color_bits_or_idx.round().long().clamp(0, 1)
            color_idx = (bits[..., 0] * 4 + bits[..., 1] * 2 + bits[..., 2]) - 1
        else:
            color_idx = color_bits_or_idx.round().long()
        color_idx = color_idx.clamp(0, self.color_embs.shape[0] - 1)

        emb = 0.5 * (self.name_embs[name_idx] + self.color_embs[color_idx])
        return self.text_proj(emb)

    def _forward_from_flat(self, graph_flat: torch.Tensor) -> torch.Tensor:
        B        = graph_flat.shape[0]
        N        = self.num_nodes
        node_raw = graph_flat.view(B, N, self.per_object_dim)
        text_emb = self._encode_text(node_raw[..., 20], node_raw[..., 21:24])
        x        = torch.cat([node_raw, text_emb], dim=-1).view(B * N, -1)

        edge_index = self._get_edge_index(B, x.device)
        batch_vec  = torch.repeat_interleave(torch.arange(B, device=x.device), N)

        # Геометрические отношения по позициям первого env
        ei_single = build_star_chain_edge_index(N, 1, x.device)
        ea_single = self._compute_geometric_edge_attr(ei_single, node_raw[0])
        edge_attr = ea_single.repeat(B)

        return self._run_gat(x, edge_index, batch_vec, edge_attr)

    # ------------------------------------------------------------------
    # JSON-режим
    # ------------------------------------------------------------------
    def _forward_from_json_scenes(
        self, scene_ids: torch.Tensor, B: int
    ) -> torch.Tensor:
        device      = next(self.parameters()).device
        out         = torch.zeros(B, self.out_dim, device=device)
        fallback_id = next(iter(self.scene_graph_cache))
        unique_ids  = torch.unique(scene_ids[:B])

        for sid_t in unique_ids:
            sid         = int(sid_t.item())
            env_indices = (scene_ids[:B] == sid_t).nonzero(as_tuple=True)[0]
            n_envs      = int(env_indices.shape[0])

            cache = self.scene_graph_cache.get(sid)
            if cache is None:
                print(f"[GraphEncoder] WARNING: scene_id={sid} not in cache, "
                      f"using fallback id={fallback_id}")
                cache = self.scene_graph_cache[fallback_id]

            N_i        = cache["num_nodes"]
            node_feats = cache["node_feats"].to(device)
            name_idx   = cache["name_idx"].to(device)

            # --- Reorder: goal-нода на позицию 0 (центр звезды) ---
            env_idx_sample = int(env_indices[0].item())
            sm        = self.env.unwrapped.scene_manager
            goal_pos  = sm.positions[
                env_idx_sample,
                sm.active_goal_indices[env_idx_sample]
            ].to(device)  # [3]

            dists          = torch.norm(node_feats[:, 0:2] - goal_pos[:2], dim=-1)
            goal_node_idx  = int(dists.argmin().item())
            order          = list(range(N_i))
            order.insert(0, order.pop(goal_node_idx))
            order_t        = torch.tensor(order, device=device)

            node_feats = node_feats[order_t]
            name_idx   = name_idx[order_t]
            # --- конец reorder ---

            # Text embedding
            name_raw = self.name_embs[name_idx]
            text_emb = self.text_proj(name_raw)

            x_single = torch.cat([node_feats, text_emb], dim=-1)  # [N_i, 24+text_dim]
            x        = x_single.repeat(n_envs, 1)                 # [n_envs*N_i, ...]

            # Star+chain рёбра
            edge_index = self._get_star_chain_edge_index(N_i, n_envs, device)

            # Геометрические отношения (по уже reordered node_feats)
            ei_single = self._get_star_chain_edge_index(N_i, 1, device)
            ea_single = self._compute_geometric_edge_attr(ei_single, node_feats)
            edge_attr = ea_single.repeat(n_envs)

            batch_vec = torch.repeat_interleave(torch.arange(n_envs, device=device), N_i)
            emb       = self._run_gat(x, edge_index, batch_vec, edge_attr)
            out[env_indices] = emb

        return out

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------
    def forward(self, graph_flat: torch.Tensor) -> torch.Tensor:
        B         = graph_flat.shape[0]
        scene_ids = self.env.unwrapped.get_current_scene_ids()

        if self.scene_graph_cache and scene_ids is not None:
            scene_ids_b = scene_ids[:B].to(next(self.parameters()).device)
            return self._forward_from_json_scenes(scene_ids_b, B)
        else:
            return self._forward_from_flat(graph_flat)

    # ------------------------------------------------------------------
    # Debug print
    # ------------------------------------------------------------------
    def print_scene_graph(self, scene_id: int) -> None:
        cache = self.scene_graph_cache.get(scene_id)
        if cache is None:
            print(f"[GraphEncoder] scene_id={scene_id} not in cache")
            return

        node_feats = cache["node_feats"]
        name_idx   = cache["name_idx"]
        edge_index = cache["edge_index"]
        edge_attr  = cache["edge_attr"]
        N          = cache["num_nodes"]
        idx_to_name = {v: k for k, v in self.name_to_idx.items()}
        idx_to_rel  = {v: k for k, v in RELATION_TO_IDX.items() if v < NUM_RELATIONS}

        print(f"\n{'='*80}")
        print(f"  SCENE GRAPH  id={scene_id}   nodes={N}   edges={edge_index.shape[1]}")
        print(f"{'='*80}")
        print(f"  {'#':>3}  {'name':<20}  {'pos_x':>7}  {'pos_y':>7}  {'pos_z':>7}  "
              f"{'ext_x':>7}  {'ext_y':>7}  {'ext_z':>7}  {'radius':>7}")
        print(f"  {'-'*3}  {'-'*20}  {'-'*7}  {'-'*7}  {'-'*7}  "
              f"{'-'*7}  {'-'*7}  {'-'*7}  {'-'*7}")
        for i in range(N):
            name = idx_to_name.get(int(name_idx[i].item()), f"unk_{name_idx[i].item()}")
            pos  = node_feats[i, 0:3]
            ext  = node_feats[i, 3:6]
            rad  = node_feats[i, 6].item()
            print(f"  {i:>3}  {name:<20}  "
                  f"{pos[0].item():>7.2f}  {pos[1].item():>7.2f}  {pos[2].item():>7.2f}  "
                  f"{ext[0].item():>7.2f}  {ext[1].item():>7.2f}  {ext[2].item():>7.2f}  "
                  f"{rad:>7.2f}")

        print(f"\n  EDGES (src → dst : relation)")
        print(f"  {'-'*60}")
        src_list  = edge_index[0].tolist()
        dst_list  = edge_index[1].tolist()
        attr_list = edge_attr.tolist()
        adj: Dict[int, list] = {i: [] for i in range(N)}
        for s, d, a in zip(src_list, dst_list, attr_list):
            if s != d:
                adj[s].append((d, a))

        for i in range(N):
            if not adj[i]:
                continue
            src_name = idx_to_name.get(int(name_idx[i].item()), f"unk_{i}")
            parts = []
            for j, a in adj[i]:
                dst_name = idx_to_name.get(int(name_idx[j].item()), f"unk_{j}")
                rel_name = idx_to_rel.get(a, "unknown")
                parts.append(f"{j}:{dst_name} [{rel_name}]")
            print(f"  {i:>3} {src_name:<20} → {', '.join(parts)}")
        print(f"{'='*80}\n")


# =====================================================================
# OrientationModule
# =====================================================================
class OrientationModule(nn.Module):
    def __init__(self, img_dim: int, graph_emb_dim: int = GRAPH_EMB_DIM,
                 num_bins: int = NUM_ORIENT_BINS):
        super().__init__()
        self.num_bins = num_bins
        self.net = nn.Sequential(
            nn.Linear(img_dim + graph_emb_dim, 256), nn.ReLU(),
            nn.Linear(256, 128),                     nn.ReLU(),
            nn.Linear(128, num_bins),
        )
        bin_size = 2 * torch.pi / num_bins
        centers  = torch.linspace(-torch.pi, torch.pi, num_bins + 1)[:-1] + bin_size / 2
        self.register_buffer("bin_centers", centers)

    def forward(self, img, graph_emb):
        logits     = self.net(torch.cat([img, graph_emb], dim=-1))
        probs      = F.softmax(logits, dim=-1)
        pred_angle = self.bin_centers[probs.argmax(-1)].unsqueeze(-1)
        return pred_angle, probs, logits

    def compute_loss(self, logits, probs, gt_yaw):
        if gt_yaw.dim() == 2:
            gt_yaw = gt_yaw.squeeze(-1)

        global step
        if step > 2999:
            with torch.no_grad():
                pred_bins = probs.argmax(-1)
                print(f"\n=== OrientationModule Debug ===")
                print(f"logits mean: {logits.mean().item():.4f}, std: {logits.std().item():.4f}")
                print(f"probs max: {probs.max().item():.4f}")
                print(f"predicted bins unique: {torch.unique(pred_bins)}")
                print(f"===============================\n")

        gt_norm  = torch.atan2(torch.sin(gt_yaw), torch.cos(gt_yaw))
        bin_size = 2 * torch.pi / self.num_bins
        labels   = ((gt_norm + torch.pi) / bin_size).long().clamp(0, self.num_bins - 1)
        loss     = F.cross_entropy(logits, labels, label_smoothing=0.05)

        with torch.no_grad():
            pred_bins  = logits.argmax(-1)
            bd         = torch.abs(pred_bins - labels)
            bd         = torch.minimum(bd, self.num_bins - bd)
            pred_angles = self.bin_centers[pred_bins]
            ang_err    = torch.atan2(
                torch.sin(gt_norm - pred_angles), torch.cos(gt_norm - pred_angles)
            )
            metrics = {
                "orient/loss":          loss.item(),
                "orient/acc_relaxed":   (bd <= 1).float().mean().item(),
                "orient/acc_strict":    (pred_bins == labels).float().mean().item(),
                "orient/mean_error_deg": (ang_err.abs().mean() * 180 / torch.pi).item(),
                "orient/confidence":    probs.max(-1)[0].mean().item(),
            }
        return loss, metrics


# =====================================================================
# Preprocessor
# =====================================================================
class DictRunningStandardScaler(nn.Module):
    def __init__(self, size, img_space, device=None, epsilon=1e-8, clip_threshold=5.0):
        super().__init__()
        self.full_space = size
        self.img_scaler = RunningStandardScaler(
            size=img_space, epsilon=epsilon,
            clip_threshold=clip_threshold, device=device,
        )

    def forward(self, x, train=False, inverse=False, no_grad=True):
        s        = unflatten_tensorized_space(self.full_space, x)
        s["img"] = self.img_scaler(s["img"], train=train, inverse=inverse, no_grad=no_grad)
        return flatten_tensorized_space(s)


# =====================================================================
# Actor & Critic
# =====================================================================
class StochasticActor(GaussianMixin, Model):
    def __init__(self, observation_space, action_space, device,
                 graph_encoder, orient_module,
                 clip_actions=False, clip_log_std=True, min_log_std=-5, max_log_std=2):
        Model.__init__(self, observation_space, action_space, device)
        GaussianMixin.__init__(self, clip_actions, clip_log_std, min_log_std, max_log_std)
        self.__dict__["graph_encoder"] = graph_encoder
        self.__dict__["orient_module"] = orient_module

        img_dim  = int(observation_space["img"].shape[0])
        goal_dim = int(observation_space["goal"].shape[0])
        mlp_in   = img_dim + goal_dim + GRAPH_EMB_DIM + 1

        self.net = nn.Sequential(
            nn.Linear(mlp_in, 512), nn.ReLU(),
            nn.Linear(512, 256),    nn.ReLU(),
            nn.Linear(256, self.num_actions), nn.Tanh(),
        )
        self.log_std_parameter = nn.Parameter(torch.zeros(self.num_actions))

    def compute(self, inputs, role):
        states         = unflatten_tensorized_space(self.observation_space, inputs["states"])
        img            = states["img"]
        goal           = states["goal"]
        graph_flat     = states["graph"]
        gt_orientation = states["orientation"]

        with torch.no_grad():
            graph_emb  = self.graph_encoder(graph_flat)
            pred_angle, _, _ = self.orient_module(img, graph_emb)
            collect_orientation_data(gt_orientation, pred_angle)
            print_orientation_accuracy()

        x = torch.cat([img, goal, graph_emb, gt_orientation], dim=-1)
        return self.net(x), self.log_std_parameter, {}


class Critic(DeterministicMixin, Model):
    def __init__(self, observation_space, action_space, device,
                 graph_encoder, orient_module, clip_actions=False):
        Model.__init__(self, observation_space, action_space, device)
        DeterministicMixin.__init__(self, clip_actions)
        self.__dict__["graph_encoder"] = graph_encoder
        self.__dict__["orient_module"] = orient_module

        img_dim  = int(observation_space["img"].shape[0])
        goal_dim = int(observation_space["goal"].shape[0])
        mlp_in   = img_dim + goal_dim + GRAPH_EMB_DIM + self.num_actions + 1

        self.net = nn.Sequential(
            nn.Linear(mlp_in, 512), nn.ReLU(),
            nn.Linear(512, 256),    nn.ReLU(),
            nn.Linear(256, 1),
        )

    def compute(self, inputs, role):
        states         = unflatten_tensorized_space(self.observation_space, inputs["states"])
        img            = states["img"]
        goal           = states["goal"]
        graph_flat     = states["graph"]
        actions        = inputs["taken_actions"]
        gt_orientation = states["orientation"]

        with torch.no_grad():
            graph_emb = self.graph_encoder(graph_flat)
            _, _, _   = self.orient_module(img, graph_emb)

        x = torch.cat([img, goal, graph_emb, gt_orientation, actions], dim=-1)
        return self.net(x), {}


# =====================================================================
# Auxiliary trainer
# =====================================================================
class AuxModuleTrainer:
    def __init__(self, graph_encoder, orient_module, agent,
                 obs_space, device,
                 lr_graph=3e-4, lr_orient=1e-3,
                 batch_size=512, train_steps_per_call=2,
                 log_interval=1000):
        self.graph_encoder = graph_encoder
        self.orient_module = orient_module
        self.agent         = agent
        self.obs_space     = obs_space
        self.device        = device
        self.batch_size    = batch_size
        self.train_steps   = train_steps_per_call
        self.log_interval  = log_interval

        self.graph_optimizer = torch.optim.AdamW(
            graph_encoder.parameters(), lr=lr_graph, weight_decay=1e-4
        )
        self.orient_optimizer = torch.optim.AdamW(
            orient_module.parameters(), lr=lr_orient, weight_decay=1e-4
        )
        self._metrics      = {}
        self._metric_count = 0

    def step(self, timestep):
        mem = self.agent.memory
        if not mem.filled and mem.memory_index < self.batch_size:
            return

        self.graph_encoder.train()
        self.orient_module.train()

        for _ in range(self.train_steps):
            sample     = mem.sample(names=["states"], batch_size=self.batch_size)[0]
            raw_states = sample[0]

            with torch.no_grad():
                processed = self.agent._state_preprocessor(raw_states, train=False)

            s          = unflatten_tensorized_space(self.obs_space, processed)
            img        = s["img"]
            graph_flat = s["graph"]
            gt_yaw     = s["orientation"]

            graph_emb            = self.graph_encoder(graph_flat)
            pred_angle, probs, logits = self.orient_module(img, graph_emb)
            loss, metrics        = self.orient_module.compute_loss(logits, probs, gt_yaw)

            self.graph_optimizer.zero_grad()
            self.orient_optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.graph_encoder.parameters(), 1.0)
            torch.nn.utils.clip_grad_norm_(self.orient_module.parameters(), 1.0)
            self.graph_optimizer.step()
            self.orient_optimizer.step()

            for k, v in metrics.items():
                self._metrics[k] = self._metrics.get(k, 0.0) + v
            self._metric_count += 1

        self.graph_encoder.eval()
        self.orient_module.eval()

        if timestep % self.log_interval == 0 and self._metric_count > 0:
            n    = self._metric_count
            line = " | ".join(f"{k}: {v/n:.4f}" for k, v in sorted(self._metrics.items()))
            print(f"🧭 [{timestep}] AuxTrain ({n} steps): {line}")
            self._metrics.clear()
            self._metric_count = 0