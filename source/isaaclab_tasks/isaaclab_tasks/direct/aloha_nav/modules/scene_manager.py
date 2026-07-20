# -*- coding: utf-8 -*-
"""
Scene Manager V3
----------------
Полностью переразложенный модуль с чётким разделением ответственности:

- SceneManager: отвечает за векторизованное состояние сцены, раскладки, выбор цели,
  размещение робота и предоставление данных.
- SceneGraph: инкапсулирует *всю* графовую логику: построение граф-наблюдения,
  расчёт пространственных отношений и сборку текстовых промптов для навигации.

⚙️ Совместимость сохранена:
- Формы тензоров и имена публичных методов не менялись, но теперь `get_graph_obs` делегирует в SceneGraph.
- Добавлены новые методы: `compute_relations` и `get_navigation_prompts` (делегируют в SceneGraph).

📌 Обновление графа:
- Граф не копирует тензоры, а работает с менеджером по ссылке.
- После любых изменений сцены менеджер вызывает `self.graph.refresh()`.

Как использовать в aloha_env.py:
  from scene_manager_v3 import SceneManager
  sm = SceneManager(num_envs, config_path, device)
  ...
  prompts = sm.get_navigation_prompts(env_ids, radius=5.0)
  text_embeds = clip.encode_text(prompts)

"""
from __future__ import annotations
import os
from typing import Dict, List, Optional

import torch
import math
import random
import json
from collections import defaultdict
from tabulate import tabulate
import importlib.util

from .room_geometry import RoomCoordinateMapper

# =====================
# Placement strategies
# =====================
from .placement_strategies import (
    PlacementStrategy,
    GridPlacement,
    GridPlacementWithOrientation,
    OnSurfacePlacement,
)


# =====================
# Vocab & color helpers
# =====================
class RelationVocab:
    LABELS = [
        'in_front_of',  # +x
        'behind',       # -x
        'left_of',      # +y
        'right_of',     # -y
        'above',        # +z
        'below',        # -z
        'inside',       # AABB containment
        'overlapping',  # AABB intersects but not containing
    ]
    TO_ID = {k: i for i, k in enumerate(LABELS)}


class ColorQuantizer:
    """Квантует RGB в 7 базовых цветов (L2 в RGB)."""
    """Квантует RGB в 10 базовых цветов (L2 в RGB)."""
    BASE = torch.tensor([
        [1.0, 0.0, 0.0],   # red
        [0.0, 1.0, 0.0],   # green
        [0.0, 0.0, 1.0],   # blue
        [1.0, 1.0, 0.0],   # yellow
        [1.0, 0.65, 0.0],  # orange
        [0.5, 0.0, 0.5],   # purple
        [1.0, 1.0, 1.0],   # white
        [0.5, 0.5, 0.5],   # gray
        [0.0, 0.0, 0.0],   # black
        [0.6, 0.3, 0.0],   # brown
    ], dtype=torch.float32)

    NAMES = [
        'red', 'green', 'blue',
        'yellow', 'orange', 'purple',
        'white', 'gray', 'black', 'brown'
    ]

    @classmethod
    def rgb_to_name(cls, rgb: torch.Tensor) -> str:
        rgb = rgb.to(dtype=torch.float32).view(1, 3)
        base = cls.BASE.to(device=rgb.device, dtype=rgb.dtype)
        d = torch.cdist(rgb, base)  # [1,10]  ✅
        idx = int(torch.argmin(d, dim=1).item())
        return cls.NAMES[idx]

    # ----------- Кодбук для имён и цветов (encoder) -----------

# ==============
# SceneGraph
# ==============
class SceneGraph:
    """Вся графовая логика: наблюдения, отношения, промпты."""
    def __init__(self, manager: 'SceneManager'):
        self.m = manager
        self._dirty = True  # если нужна инвалидация кэшей в будущем

    def refresh(self):
        """Вызывается менеджером после изменений сцены. Сейчас ничего не кэшируем, но оставляем хук."""
        self._dirty = True

    # ---------- Graph observation ----------
    @torch.no_grad()
    def get_observation(self, env_ids: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        m = self.m
        device = m.device
        if env_ids is None:
            env_ids = torch.arange(m.num_envs, device=device)
        E = len(env_ids)

        # --- node features ---
        positions   = m.positions[env_ids]                            # (E, M, 3)
        # print("positions", positions)
        sizes       = m.sizes.expand(E, -1, -1)                             # (E, M, 3)
        radii       = m.radii.expand(E, -1).unsqueeze(-1)                   # (E, M, 1)
        colors      = m.colors.expand(E, -1, -1)                           # (E, M, 3)
        object_ids  = m.object_ids.expand(E, -1).unsqueeze(-1).float()    # (E, M, 1)
        active      = m.active[env_ids].unsqueeze(-1).float()             # (E, M, 1)

        raw_parents = m.on_surface_idx[env_ids]                                    # (E, M)  int
        parents_feat= raw_parents.unsqueeze(-1).float()                   # (E, M, 1) — ТОЛЬКО для node_features

        levels      = m.surface_level[env_ids].unsqueeze(-1).float()        # (E, M, 1)

        node_features = torch.cat(
            [positions, sizes, radii, colors, object_ids, active, parents_feat, levels],
            dim=-1
        )  # (E, M, 14)

        # --- edge features (используем СЫРЫЕ индексы, без деления) ---
        edge_exists = (raw_parents >= 0).float().unsqueeze(-1)                     # (E, M, 1)
        valid_mask  = (raw_parents >= 0)                                           # (E, M)

        z_diff = torch.zeros(E, m.num_total_objects, 1, device=device)
        level_diff = torch.zeros_like(z_diff)
        dist = torch.zeros_like(z_diff)
        color_diff_norm = torch.zeros_like(z_diff)
        id_diff = torch.zeros_like(z_diff)

        if valid_mask.any():
            batch_idx = torch.arange(E, device=device)[:, None].expand(-1, m.num_total_objects)[valid_mask]
            obj_idx   = torch.arange(m.num_total_objects, device=device)[None, :].expand(E, -1)[valid_mask]
            parent_idx= raw_parents[valid_mask].long()

            # z diff
            z_diff[valid_mask] = positions[batch_idx, obj_idx, 2:3] - positions[batch_idx, parent_idx, 2:3]
            # level diff
            level_diff[valid_mask] = levels[batch_idx, obj_idx] - levels[batch_idx, parent_idx]
            # xy distance
            child_xy  = positions[batch_idx, obj_idx, :2]
            parent_xy = positions[batch_idx, parent_idx, :2]
            dist[valid_mask] = torch.norm(child_xy - parent_xy, dim=-1, keepdim=True)
            # color diff
            child_color  = colors[batch_idx, obj_idx]
            parent_color = colors[batch_idx, parent_idx]
            color_diff_norm[valid_mask] = torch.norm(child_color - parent_color, dim=-1, keepdim=True)
            # id diff
            child_id  = object_ids[batch_idx, obj_idx]
            parent_id = object_ids[batch_idx, parent_idx]
            id_diff[valid_mask] = child_id - parent_id

        edge_features = torch.cat([edge_exists, z_diff, level_diff, dist, color_diff_norm, id_diff], dim=-1)  # (E, M, 6)
        return {"node_features": node_features, "edge_features": edge_features}


    # ---------- Spatial relations ----------
    @torch.no_grad()
    def compute_relations(
        self,
        env_ids: torch.Tensor,
        reference: str | int = 'goal',
        *,
        use_local_frame: bool = True,
        reference_yaws: Optional[torch.Tensor] = None,
        radius: Optional[float] = 5.0,
        include_inactive: bool = False,
    ) -> List[Dict[str, int]]:
        m = self.m
        pos = m.positions[env_ids]                 # [E, M, 3]
        sizes = m.sizes[0]                         # [M, 3]
        active = m.active[env_ids].bool()          # [E, M]
        names = m.names

        E, M = pos.shape[:2]

        # Resolve reference index per env
        if isinstance(reference, int):
            ref_idx = torch.full((E,), int(reference), device=pos.device, dtype=torch.long)
        else:
            if reference in ('goal', 'robot'):
                if reference == 'goal':
                    ref_idx = m.active_goal_indices[env_ids]
                else:
                    ref_idx = m.robot_global_index_tensor.expand(E)
            else:
                if reference not in m.object_map:
                    raise KeyError(f"Unknown reference name: {reference}")
                ref_idx = m.object_map[reference]['indices'][0].expand(E)

        batch_idx = torch.arange(E, device=pos.device).view(E, 1).expand(E, M)
        ref_pos = pos[batch_idx, ref_idx.view(E, 1).expand(E, M)]
        deltas = ref_pos - pos   # [E, M, 3]

        # Rotate to local frame if needed
        if use_local_frame and reference_yaws is not None:
            cy = torch.cos(-reference_yaws).view(E, 1)
            sy = torch.sin(-reference_yaws).view(E, 1)
            x = deltas[..., 0]
            y = deltas[..., 1]
            x_r = x * cy + y * -sy
            y_r = x * sy + y *  cy
            deltas = torch.stack([x_r, y_r, deltas[..., 2]], dim=-1)

        dist = torch.linalg.norm(deltas, dim=-1)  # [E, M]
        mask = torch.ones_like(dist, dtype=torch.bool)
        if radius is not None:
            mask &= (dist <= radius)
        if not include_inactive:
            mask &= active

        # Exclude the reference itself
        ref_mask = torch.zeros_like(mask)
        ref_mask[torch.arange(E, device=pos.device), ref_idx] = True
        mask &= ~ref_mask

        abs_d = deltas.abs()
        dom_axis = torch.argmax(abs_d, dim=-1)  # 0/1/2
        signs = torch.sign(deltas).to(torch.int8)

        rel_id = torch.full((E, M), -1, device=pos.device, dtype=torch.long)
        # X axis
        xpos = (dom_axis == 0) & (signs[..., 0] >= 0)
        xneg = (dom_axis == 0) & (signs[..., 0] <  0)
        rel_id[xpos] = RelationVocab.TO_ID['in_front_of']
        rel_id[xneg] = RelationVocab.TO_ID['behind']
        # Y axis
        ypos = (dom_axis == 1) & (signs[..., 1] >= 0)
        yneg = (dom_axis == 1) & (signs[..., 1] <  0)
        rel_id[ypos] = RelationVocab.TO_ID['left_of']
        rel_id[yneg] = RelationVocab.TO_ID['right_of']
        # Z axis
        zpos = (dom_axis == 2) & (signs[..., 2] >= 0)
        zneg = (dom_axis == 2) & (signs[..., 2] <  0)
        rel_id[zpos] = RelationVocab.TO_ID['above']
        rel_id[zneg] = RelationVocab.TO_ID['below']

        # Inside / overlap
        half = sizes / 2.0
        half_b = half.unsqueeze(0).expand(E, M, 3)
        other_to_ref = -deltas
        inside = (other_to_ref.abs() <= half_b + 1e-6).all(dim=-1)
        rel_id[inside] = RelationVocab.TO_ID['inside']

        ref_sizes = sizes[ref_idx]              # [E,3]
        ref_half_b = (ref_sizes / 2.0).view(E, 1, 3).expand(E, M, 3)
        overlap = (deltas.abs() <= (ref_half_b + half_b) + 1e-6).all(dim=-1) & ~inside
        rel_id[overlap] = RelationVocab.TO_ID['overlapping']

        rel_id = torch.where(mask, rel_id, torch.full_like(rel_id, -1))

        out: List[Dict[str, int]] = []
        for e in range(E):
            dct: Dict[str, int] = {}
            valid = rel_id[e] >= 0
            idxs = torch.nonzero(valid, as_tuple=False).view(-1)
            for j in idxs.tolist():
                dct[names[j]] = int(rel_id[e, j].item())
            out.append(dct)
        return out

    # ---------- Prompt builder ----------
    @torch.no_grad()
    def build_navigation_prompt(
        self,
        env_ids: torch.Tensor,
        goal_name: Optional[str] = None,
        radius: float = 5.0,
        use_local_frame: bool = True,
        reference_yaws: Optional[torch.Tensor] = None,
    ) -> List[str]:
        m = self.m
        E = len(env_ids)
        goal_idxs = m.active_goal_indices[env_ids]   # [E]
        names = m.names
        colors = m.colors[0]                         # [M,3] on (cpu/cuda)

        # словарь для быстрого поиска индекса по имени (O(1) вместо names.index(...))
        name_to_idx = {n: i for i, n in enumerate(names)}

        rel_dicts = self.compute_relations(
            env_ids=env_ids,
            reference='goal',
            use_local_frame=use_local_frame,
            reference_yaws=reference_yaws,
            radius=radius,
            include_inactive=False,
        )

        prompts: List[str] = []
        for k in range(E):
            g_idx = int(goal_idxs[k].item())
            # имя цели без суффикса "_i"
            g_name_raw = goal_name if goal_name is not None else names[g_idx]
            g_name = g_name_raw.split('_', 1)[0]
            # цвет цели -> базовое название
            g_color = ColorQuantizer.rgb_to_name(colors[g_idx])

            # Собираем фразы отношений с ЦВЕТОМ каждого объекта
            rels = rel_dicts[k]              # {obj_name: relation_id}
            phrases: List[str] = []
            for obj_name, rid in rels.items():
                obj_idx = name_to_idx.get(obj_name, None)
                if obj_idx is None:
                    continue
                obj_simple = obj_name.split('_', 1)[0]
                obj_color = ColorQuantizer.rgb_to_name(colors[obj_idx])

                label = RelationVocab.LABELS[rid]
                if label == 'in_front_of':
                    phrases.append(f"in front of {obj_color} {obj_simple}")
                elif label == 'behind':
                    phrases.append(f"behind {obj_color} {obj_simple}")
                elif label == 'left_of':
                    phrases.append(f"left from {obj_color} {obj_simple}")
                elif label == 'right_of':
                    phrases.append(f"right from {obj_color} {obj_simple}")
                elif label == 'above':
                    phrases.append(f"above {obj_color} {obj_simple}")
                elif label == 'below':
                    phrases.append(f"below {obj_color} {obj_simple}")
                elif label == 'inside':
                    phrases.append(f"inside {obj_color} {obj_simple}")
                elif label == 'overlapping':
                    phrases.append(f"overlapping {obj_color} {obj_simple}")

            rel_part = ''
            if phrases:
                # перечисление через запятую: "... that is behind yellow table, left from green vase"
                rel_part = ' that is ' + ', '.join(phrases)

            # финальный промпт: "Move to yellow bowl that is behind green table, left from blue vase"
            prompt = f"Move to {g_color} {g_name}{rel_part}"
            prompts.append(prompt)

        return prompts



# =====================
# Block layout sampler
# =====================
class BlockLayoutSampler:
    """Assign complete semantic blocks to active physical rooms.

    Object instances are global pools declared by scene_items.json. They are not
    copied per room. For every reset, exactly ``len(active_rooms)`` semantic
    blocks are sampled without replacement and assigned one-to-one to active
    rooms.
    """

    VALID_GOAL_MODES = {"selected_only", "all_candidates"}

    def __init__(self, manager: 'SceneManager', rules_path: str):
        self.m = manager
        self.rules_path = rules_path
        if not os.path.exists(rules_path):
            raise FileNotFoundError(f"layout_rules.json not found: {rules_path}")
        with open(rules_path, 'r') as f:
            self.rules = json.load(f)

        self.grids = self.rules.get('grids', {})
        self.blocks = self.rules.get('semantic_blocks', {})
        self.goal_mode = self.rules.get('goal_mode', 'selected_only')

        # Placement constraints use the common 20 x 20 env-local coordinate frame.
        placement_cfg = self.rules.get('placement', {})
        self.placement_collision_margin = float(
            placement_cfg.get('collision_margin', 0.05)
        )
        if self.placement_collision_margin < 0.0:
            raise RuntimeError('placement.collision_margin must be non-negative')

        self.passage_exclusion_zones: list[dict] = []
        for zone_idx, raw_zone in enumerate(
            placement_cfg.get('passage_exclusion_zones', [])
        ):
            if not isinstance(raw_zone, dict):
                raise RuntimeError(
                    'placement.passage_exclusion_zones entries must be objects'
                )
            if not bool(raw_zone.get('enabled', True)):
                continue
            center = raw_zone.get('center')
            if not isinstance(center, list) or len(center) not in (2, 3):
                raise RuntimeError(
                    'Each passage exclusion zone must define center=[x,y] '
                    f'or [x,y,z], got {center!r}'
                )
            radius = float(raw_zone.get('radius', 0.0))
            if radius < 0.0:
                raise RuntimeError(
                    f'Passage exclusion radius must be non-negative, got {radius}'
                )
            self.passage_exclusion_zones.append({
                'name': str(raw_zone.get('name', f'passage_{zone_idx}')),
                'center': torch.tensor(
                    center[:2], device=self.m.device, dtype=torch.float32
                ),
                'radius': radius,
            })

        if self.goal_mode not in self.VALID_GOAL_MODES:
            raise RuntimeError(
                f"goal_mode must be one of {sorted(self.VALID_GOAL_MODES)}, "
                f"got {self.goal_mode!r}"
            )
        if not self.grids:
            raise RuntimeError("layout_rules.json must define non-empty 'grids'")
        if not self.blocks:
            raise RuntimeError(
                "layout_rules.json must define non-empty 'semantic_blocks'"
            )
        if len(self.blocks) < self.m.num_active_rooms:
            raise RuntimeError(
                f"There are {len(self.blocks)} semantic blocks but "
                f"{self.m.num_active_rooms} active rooms. At least one block per "
                "active room is required."
            )

        self.block_names = list(self.blocks.keys())
        self.block_to_id = {name: i for i, name in enumerate(self.block_names)}
        self.m.semantic_block_names = self.block_names
        self.m.semantic_block_to_id = self.block_to_id
        self.m.room_block_ids = torch.full(
            (self.m.num_envs, self.m.num_rooms),
            -1,
            dtype=torch.long,
            device=self.m.device,
        )
        self._validate_rules()

    def _require_grid(self, grid_name: str, context: str) -> dict:
        if grid_name not in self.grids:
            raise RuntimeError(
                f"{context} references missing grid {grid_name!r}"
            )
        grid_cfg = self.grids[grid_name]
        coordinates = grid_cfg.get('coordinates', [])
        if not coordinates:
            raise RuntimeError(f"Grid {grid_name!r} has no coordinates")
        return grid_cfg

    def _require_object(self, object_name: str, context: str) -> dict:
        if object_name not in self.m.object_map:
            raise RuntimeError(
                f"{context} references unknown object {object_name!r}"
            )
        return self.m.object_map[object_name]

    @staticmethod
    def _provider_specs(goal_cfg: dict) -> list[dict]:
        raw = goal_cfg.get('surface_providers', [])
        specs = []
        for entry in raw:
            if isinstance(entry, str):
                specs.append({'object': entry, 'count': 1})
            elif isinstance(entry, dict):
                specs.append({
                    'object': entry['object'],
                    'count': int(entry.get('count', 1)),
                })
            else:
                raise RuntimeError(
                    "surface_providers entries must be strings or objects"
                )
        return specs

    def _validate_rules(self):
        for block_name, block_cfg in self.blocks.items():
            context = f"semantic block {block_name!r}"

            staff_cfg = block_cfg.get('staff', {})
            if staff_cfg:
                self._require_grid(staff_cfg.get('grid'), f"{context}.staff")
                staff_objects = staff_cfg.get('objects', [])
                if not isinstance(staff_objects, list):
                    raise RuntimeError(f"{context}.staff.objects must be a list")
                for object_name in staff_objects:
                    self._require_object(object_name, f"{context}.staff")

            goal_cfg = block_cfg.get('goal')
            if not isinstance(goal_cfg, dict):
                raise RuntimeError(f"{context} must define a goal object")
            self._require_grid(goal_cfg.get('grid'), f"{context}.goal")
            goal_name = goal_cfg.get('object')
            goal_meta = self._require_object(goal_name, f"{context}.goal")
            if 'possible_goal' not in goal_meta['types']:
                raise RuntimeError(
                    f"Goal object {goal_name!r} in {context} lacks type 'possible_goal'"
                )
            providers = self._provider_specs(goal_cfg)
            for provider in providers:
                if provider['count'] <= 0:
                    raise RuntimeError(
                        f"Provider count must be positive in {context}.goal"
                    )
                provider_meta = self._require_object(
                    provider['object'], f"{context}.goal.surface_providers"
                )
                if 'surface_provider' not in provider_meta['types']:
                    raise RuntimeError(
                        f"Provider {provider['object']!r} in {context} lacks "
                        "type 'surface_provider'"
                    )
            if 'surface_only' in goal_meta['types'] and not providers:
                raise RuntimeError(
                    f"Surface-only goal {goal_name!r} in {context} needs a provider"
                )

            obstacles = block_cfg.get('obstacles', [])
            if not isinstance(obstacles, list):
                raise RuntimeError(f"{context}.obstacles must be a list")
            for obstacle_cfg in obstacles:
                object_name = obstacle_cfg.get('object')
                self._require_object(object_name, f"{context}.obstacles")
                self._require_grid(
                    obstacle_cfg.get('grid'), f"{context}.obstacles"
                )
                min_count = int(obstacle_cfg.get('min_count', 0))
                max_count = int(obstacle_cfg.get('max_count', min_count))
                if min_count < 0 or max_count < min_count:
                    raise RuntimeError(
                        f"Invalid obstacle count range [{min_count}, {max_count}] "
                        f"for {object_name!r} in {context}"
                    )

    def _grid_tensor(self, grid_name: str) -> torch.Tensor:
        coordinates = self.grids[grid_name]['coordinates']
        return torch.tensor(
            coordinates, device=self.m.device, dtype=torch.float32
        )

    def _new_object_pool(self) -> dict[str, list[int]]:
        pool = {}
        for name, meta in self.m.object_map.items():
            indices = [int(i) for i in meta['indices'].tolist()]
            if len(indices) > 1:
                permutation = torch.randperm(
                    len(indices), device=self.m.device
                ).tolist()
                indices = [indices[i] for i in permutation]
            pool[name] = indices
        return pool

    def _take(
        self,
        pool: dict[str, list[int]],
        object_name: str,
        count: int,
        context: str,
    ) -> list[int]:
        available = pool.get(object_name, [])
        if len(available) < count:
            raise RuntimeError(
                f"Not enough physical instances of {object_name!r} for {context}: "
                f"need {count}, have {len(available)}. Increase count in "
                "scene_items.json or avoid selecting these blocks together."
            )
        selected = available[:count]
        del available[:count]
        return selected

    def _take_all(
        self,
        pool: dict[str, list[int]],
        object_name: str,
        context: str,
    ) -> list[int]:
        count = len(pool.get(object_name, []))
        if count == 0:
            raise RuntimeError(
                f"No unused physical instances of {object_name!r} for {context}"
            )
        return self._take(pool, object_name, count, context)

    def _candidate_rejection_reason(
        self,
        env_id: int,
        object_idx: int,
        candidate_position: torch.Tensor,
        placed_floor_indices: list[int],
        *,
        check_existing_objects: bool,
    ) -> str | None:
        """Return why a floor candidate is invalid, or ``None`` if valid.

        Collision checks use circular XY footprints and work across every grid
        used by the semantic block.
        """
        object_radius = float(self.m.radii[0, object_idx].item())
        candidate_xy = candidate_position[:2]

        for zone in self.passage_exclusion_zones:
            distance = float(torch.linalg.norm(
                candidate_xy - zone['center']
            ).item())
            required = float(zone['radius']) + object_radius
            if distance < required:
                return (
                    f"passage zone {zone['name']!r}: distance={distance:.3f}, "
                    f"required>={required:.3f}"
                )

        if not check_existing_objects:
            return None

        for placed_idx in placed_floor_indices:
            placed_xy = self.m.positions[env_id, placed_idx, :2]
            placed_radius = float(self.m.radii[0, placed_idx].item())
            distance = float(torch.linalg.norm(
                candidate_xy - placed_xy
            ).item())
            required = (
                object_radius
                + placed_radius
                + self.placement_collision_margin
            )
            if distance < required:
                return (
                    f"object {self.m.names[placed_idx]!r}: "
                    f"distance={distance:.3f}, required>={required:.3f}"
                )

        return None

    def _place_floor_indices(
        self,
        env_id: int,
        room_id: int,
        object_indices: list[int],
        grid_name: str,
        used_cells_by_grid: dict[str, set[int]],
        placed_floor_indices: list[int],
        context: str,
        *,
        check_existing_objects: bool = True,
    ) -> list[int]:
        """Place objects using shared room-level occupancy across all grids."""
        if not object_indices:
            return []

        local_grid = self._grid_tensor(grid_name)
        allowed = set(
            self.m.room_mapper.allowed_cell_indices(local_grid, room_id)
        )
        used_cells = used_cells_by_grid.setdefault(grid_name, set())
        placed_now: list[int] = []

        for object_idx in object_indices:
            candidate_cells = sorted(allowed.difference(used_cells))
            if candidate_cells:
                permutation = torch.randperm(
                    len(candidate_cells), device=self.m.device
                ).tolist()
                candidate_cells = [candidate_cells[i] for i in permutation]

            selected_cell: int | None = None
            selected_position: torch.Tensor | None = None
            rejection_log: list[str] = []

            for cell_idx in candidate_cells:
                position = self.m.room_mapper.local_to_global(
                    local_grid[cell_idx], room_id
                )
                reason = self._candidate_rejection_reason(
                    env_id=env_id,
                    object_idx=object_idx,
                    candidate_position=position,
                    placed_floor_indices=placed_floor_indices,
                    check_existing_objects=check_existing_objects,
                )
                if reason is None:
                    selected_cell = cell_idx
                    selected_position = position
                    break
                rejection_log.append(f"cell={cell_idx}: {reason}")

            if selected_cell is None or selected_position is None:
                details = '; '.join(rejection_log[-8:])
                raise RuntimeError(
                    f"No valid cell in grid {grid_name!r}, room {room_id + 1} "
                    f"for {self.m.names[object_idx]!r} ({context}). "
                    f"Allowed cells={sorted(allowed)}, used={sorted(used_cells)}. "
                    f"Recent rejections: {details}"
                )

            self.m.positions[env_id, object_idx] = selected_position
            self.m.active[env_id, object_idx] = True
            self.m.object_room_ids[env_id, object_idx] = room_id
            self.m.on_surface_idx[env_id, object_idx] = -1
            self.m.surface_level[env_id, object_idx] = 0
            used_cells.add(selected_cell)
            placed_floor_indices.append(object_idx)
            placed_now.append(object_idx)

        return placed_now

    def _place_goal_on_provider(
        self,
        env_id: int,
        room_id: int,
        goal_idx: int,
        provider_indices: list[int],
    ) -> None:
        if not provider_indices:
            raise RuntimeError(
                "A surface goal was selected but no provider was placed"
            )
        parent_rel = int(torch.randint(
            0, len(provider_indices), (1,), device=self.m.device
        ).item())
        parent_idx = provider_indices[parent_rel]
        parent_pos = self.m.positions[env_id, parent_idx]
        parent_size = self.m.sizes[0, parent_idx]
        goal_size = self.m.sizes[0, goal_idx]

        goal_pos = parent_pos.clone()
        # Preserve the existing project's USD-origin convention.
        goal_pos[2] = parent_pos[2] + parent_size[2] + goal_size[2] * 0.5
        self.m.positions[env_id, goal_idx] = goal_pos
        self.m.active[env_id, goal_idx] = True
        self.m.object_room_ids[env_id, goal_idx] = room_id
        self.m.on_surface_idx[env_id, goal_idx] = parent_idx
        self.m.surface_level[env_id, goal_idx] = (
            self.m.surface_level[env_id, parent_idx] + 1
        )

    def _place_block(
        self,
        env_id: int,
        room_id: int,
        block_name: str,
        block_cfg: dict,
        object_pool: dict[str, list[int]],
        place_goal: bool,
    ) -> int | None:
        # Priority: goal/provider -> obstacles -> staff.
        used_cells_by_grid: dict[str, set[int]] = {}
        placed_floor_indices: list[int] = []
        context = f"block {block_name!r} in room {room_id + 1}"

        goal_cfg = block_cfg['goal']
        provider_indices: list[int] = []
        for provider_cfg in self._provider_specs(goal_cfg):
            provider_indices.extend(
                self._take(
                    object_pool,
                    provider_cfg['object'],
                    provider_cfg['count'],
                    context + '.goal.surface_providers',
                )
            )

        # Providers/tables have highest priority. They still obey passage and
        # wall exclusions, then reserve their footprint for lower priorities.
        self._place_floor_indices(
            env_id,
            room_id,
            provider_indices,
            goal_cfg['grid'],
            used_cells_by_grid,
            placed_floor_indices,
            context + '.goal.surface_providers',
            check_existing_objects=False,
        )

        selected_goal_idx = None
        if place_goal:
            selected_goal_idx = self._take(
                object_pool,
                goal_cfg['object'],
                1,
                context + '.goal',
            )[0]
            if provider_indices:
                self._place_goal_on_provider(
                    env_id, room_id, selected_goal_idx, provider_indices
                )
            else:
                self._place_floor_indices(
                    env_id,
                    room_id,
                    [selected_goal_idx],
                    goal_cfg['grid'],
                    used_cells_by_grid,
                    placed_floor_indices,
                    context + '.goal',
                    check_existing_objects=False,
                )

        for obstacle_cfg in block_cfg.get('obstacles', []):
            min_count = int(obstacle_cfg.get('min_count', 0))
            max_count = int(obstacle_cfg.get('max_count', min_count))
            count = int(torch.randint(
                min_count,
                max_count + 1,
                (1,),
                device=self.m.device,
            ).item())
            obstacle_indices = self._take(
                object_pool,
                obstacle_cfg['object'],
                count,
                context + '.obstacles',
            )
            self._place_floor_indices(
                env_id,
                room_id,
                obstacle_indices,
                obstacle_cfg['grid'],
                used_cells_by_grid,
                placed_floor_indices,
                context + '.obstacles',
                check_existing_objects=True,
            )

        staff_cfg = block_cfg.get('staff', {})
        if staff_cfg:
            staff_indices: list[int] = []
            for object_name in staff_cfg.get('objects', []):
                staff_indices.extend(
                    self._take_all(object_pool, object_name, context + '.staff')
                )
            self._place_floor_indices(
                env_id,
                room_id,
                staff_indices,
                staff_cfg['grid'],
                used_cells_by_grid,
                placed_floor_indices,
                context + '.staff',
                check_existing_objects=True,
            )

        return selected_goal_idx

    def sample_and_apply(self, env_ids: torch.Tensor):
        env_ids = torch.as_tensor(
            env_ids, device=self.m.device, dtype=torch.long
        )
        self.m.active[env_ids] = False
        self.m.positions[env_ids] = self.m.default_positions[env_ids]
        self.m.object_room_ids[env_ids] = -1
        self.m.on_surface_idx[env_ids] = -1
        self.m.surface_level[env_ids] = 0
        self.m.room_block_ids[env_ids] = -1

        selected_goal_indices = torch.full(
            (env_ids.numel(),),
            -1,
            dtype=torch.long,
            device=self.m.device,
        )

        active_room_ids = list(self.m.room_mapper.active_room_ids)
        block_count = len(self.block_names)
        rooms_to_fill = len(active_room_ids)

        for env_offset, env_id in enumerate(env_ids.tolist()):
            block_perm = torch.randperm(
                block_count, device=self.m.device
            ).tolist()
            selected_blocks = [
                self.block_names[i] for i in block_perm[:rooms_to_fill]
            ]
            room_perm = torch.randperm(
                rooms_to_fill, device=self.m.device
            ).tolist()
            assigned_rooms = [active_room_ids[i] for i in room_perm]
            assignments = list(zip(assigned_rooms, selected_blocks))

            target_assignment = int(torch.randint(
                0, len(assignments), (1,), device=self.m.device
            ).item())
            object_pool = self._new_object_pool()
            candidate_goal_indices: list[int] = []

            for assignment_idx, (room_id, block_name) in enumerate(assignments):
                self.m.room_block_ids[env_id, room_id] = self.block_to_id[block_name]
                place_goal = (
                    self.goal_mode == 'all_candidates'
                    or assignment_idx == target_assignment
                )
                goal_idx = self._place_block(
                    env_id=env_id,
                    room_id=room_id,
                    block_name=block_name,
                    block_cfg=self.blocks[block_name],
                    object_pool=object_pool,
                    place_goal=place_goal,
                )
                if goal_idx is not None:
                    candidate_goal_indices.append(goal_idx)
                    if assignment_idx == target_assignment:
                        selected_goal_indices[env_offset] = goal_idx

            if selected_goal_indices[env_offset] < 0:
                raise RuntimeError(
                    f"No navigation goal was placed in env {env_id}"
                )

        self.m.chose_active_goal_state(
            env_ids,
            selected_goal_indices=selected_goal_indices,
        )
        self.m.graph.refresh()


# ==============
# SceneManager
# ==============
class SceneManager:
    def __init__(self, num_envs: int, config_path: str, device: str):
        self.num_envs = num_envs
        self.device = device
        with open(config_path, 'r') as f:
            raw = json.load(f)
        self.raw_config = raw
        self.config = raw['objects']
        self.type_placements_cfg = raw.get('type_placements', {})

        layout_rules_path = os.path.join(
            os.path.dirname(config_path), 'layout_rules.json'
        )
        if not os.path.exists(layout_rules_path):
            raise FileNotFoundError(
                f"layout_rules.json not found: {layout_rules_path}"
            )
        with open(layout_rules_path, 'r') as f:
            layout_rules = json.load(f)

        self.room_mapper = RoomCoordinateMapper(
            device=self.device,
            config=layout_rules.get('room_layout', {}),
        )
        self.num_rooms = self.room_mapper.num_rooms
        self.num_active_rooms = self.room_mapper.num_active_rooms
        self.active_room_ids = torch.tensor(
            self.room_mapper.active_room_ids,
            dtype=torch.long,
            device=self.device,
        )

        self.colors_dict = {
            'red':[1,0,0], 'green':[0,1,0], 'blue':[0,0,1],
            'yellow':[1,1,0], 'orange':[1,0.65,0],
            'purple':[0.5,0,0.5], 'white':[1,1,1],
            'gray':[0.5,0.5,0.5], 'black':[0,0,0],
            'brown':[0.6,0.3,0],
        }

        # Physical object count is exactly the sum from scene_items.json.
        self.num_total_objects = sum(
            int(obj['count']) for obj in self.config
        )
        self.object_ids = torch.zeros(
            1, self.num_total_objects, device=self.device
        )
        # Room ownership is dynamic because blocks move between rooms per reset.
        self.object_room_ids = torch.full(
            (self.num_envs, self.num_total_objects),
            -1,
            dtype=torch.long,
            device=self.device,
        )
        self.object_map: Dict[str, Dict] = {}
        self.type_map = defaultdict(list)

        self.positions = torch.zeros(
            self.num_envs, self.num_total_objects, 3, device=self.device
        )
        self.sizes = torch.zeros(
            1, self.num_total_objects, 3, device=self.device
        )
        self.radii = torch.zeros(
            1, self.num_total_objects, device=self.device
        )
        self.colors = torch.ones(
            1, self.num_total_objects, 3, device=self.device
        )
        self.names: List[str] = []
        self.active = torch.zeros(
            self.num_envs,
            self.num_total_objects,
            dtype=torch.bool,
            device=self.device,
        )
        self.on_surface_idx = torch.full(
            (self.num_envs, self.num_total_objects),
            -1,
            dtype=torch.long,
            device=self.device,
        )
        self.surface_level = torch.zeros(
            self.num_envs,
            self.num_total_objects,
            dtype=torch.long,
            device=self.device,
        )

        self._initialize_object_data()
        self.default_positions = self.positions.clone()
        self.placement_strategies = self._initialize_strategies()

        self.robot_radius = 0.4
        self.room_bounds = self.room_mapper.room_bounds
        self.goal_positions = torch.zeros((num_envs, 3), device=self.device)
        self.active_goal_indices = torch.zeros(
            self.num_envs, dtype=torch.long, device=self.device
        )
        self.active_goal_room_ids = torch.full(
            (self.num_envs,), -1, dtype=torch.long, device=self.device
        )
        if 'robot' in self.object_map:
            self.robot_global_index = int(
                self.object_map['robot']['indices'][0].item()
            )
        else:
            self.robot_global_index = 0
        self.robot_global_index_tensor = torch.tensor(
            self.robot_global_index,
            device=self.device,
            dtype=torch.long,
        )

        n_angles = 36
        angle_step = 2 * math.pi / n_angles
        self.discrete_angles = torch.arange(
            0, 2 * math.pi, angle_step, device=self.device
        )
        self.candidate_vectors = torch.stack(
            [torch.cos(self.discrete_angles), torch.sin(self.discrete_angles)],
            dim=1,
        )

        self.graph = SceneGraph(self)
        self.layout_sampler = BlockLayoutSampler(self, layout_rules_path)

    def update_prims(self):
        pass

    # ----------- Векторные данные сцены -----------
    def get_scene_data_dict(self):
        return {
            "positions": self.positions,
            "sizes": self.sizes.expand(self.num_envs, -1, -1),
            "radii": self.radii.expand(self.num_envs, -1),
            "active": self.active,
            "on_surface_idx": self.on_surface_idx,
            "surface_level": self.surface_level,
            "colors": self.colors.expand(self.num_envs, -1, -1),   # 👈 добавить
            "object_ids": self.object_ids.expand(self.num_envs, -1),
            "room_ids": self.object_room_ids,
            "names": self.names,
        }


    # ----------- Фиксированная раскладка -----------
    def apply_fixed_positions(self, env_ids: torch.Tensor, positions_config: List[dict]):
        self.active[env_ids] = False
        self.positions[env_ids] = self.default_positions[env_ids]
        self.object_room_ids[env_ids] = -1
        self.on_surface_idx[env_ids] = -1
        self.surface_level[env_ids] = 0
        scene_data = self.get_scene_data_dict()
        for env_id in env_ids:
            env_dict = positions_config[env_id.item()]
            for obj_name, pos_list in env_dict.items():
                if obj_name not in self.object_map:
                    continue
                indices = self.object_map[obj_name]["indices"]
                for i, pos in enumerate(pos_list):
                    if i >= len(indices):
                        print("[WARN] Too many instances for", obj_name)
                        break
                    scene_data["positions"][env_id.item(), indices[i]] = torch.tensor(pos, device=self.device)
                    scene_data["active"][env_id.item(), indices[i]] = True
                    room_id = self.room_mapper.room_ids_from_positions(
                        torch.tensor(pos[:2], device=self.device).view(1, 2)
                    )[0]
                    self.object_room_ids[env_id.item(), indices[i]] = room_id
                    scene_data["on_surface_idx"][env_id.item(), indices[i]] = -1
                    scene_data["surface_level"][env_id.item(), indices[i]] = 0
        self.chose_active_goal_state(env_ids)
        self.graph.refresh()

    # ----------- Инициализация объектов -----------
    def _initialize_object_data(self):
        start_idx = 0
        default_pos_tensor = torch.zeros(
            1, self.num_total_objects, 3, device=self.device
        )

        spacing = 0.5
        max_per_row = 16
        for i in range(self.num_total_objects):
            row = i // max_per_row
            col = i % max_per_row
            default_pos_tensor[0, i, 0] = (col - max_per_row / 2) * spacing
            default_pos_tensor[0, i, 1] = (row - 2.0) * spacing
            default_pos_tensor[0, i, 2] = -20.0

        seen_ids = set()
        for obj_cfg in self.config:
            name = obj_cfg['name']
            count = int(obj_cfg['count'])
            if count < 0:
                raise ValueError(f"Negative count for object {name!r}: {count}")
            indices = torch.arange(
                start_idx,
                start_idx + count,
                device=self.device,
                dtype=torch.long,
            )
            types = set(obj_cfg['type'])

            info = obj_cfg.get('info', {}) or {}
            info_color = info.get('color')
            if isinstance(info_color, str):
                color_name = info_color.strip().lower()
                color_rgb = self.colors_dict.get(
                    color_name, self.colors_dict['gray']
                )
                if color_name not in self.colors_dict:
                    print(
                        f"[WARN] Unknown color {info_color!r} for {name!r}; "
                        "using gray"
                    )
                self.colors[0, indices] = torch.tensor(
                    color_rgb,
                    device=self.device,
                    dtype=torch.float32,
                )

            object_id = int(obj_cfg.get('id', 0))
            if object_id <= 0:
                raise ValueError(
                    f"Object {name!r} must define a positive integer id"
                )
            if object_id in seen_ids:
                raise ValueError(
                    f"Duplicate semantic object id {object_id} for {name!r}"
                )
            seen_ids.add(object_id)
            self.object_ids[0, indices] = float(object_id)

            self.object_map[name] = {
                'indices': indices,
                'types': types,
                'count': count,
                'id': object_id,
            }
            for type_str in types:
                self.type_map[type_str].extend(indices.tolist())

            for instance_id in range(count):
                self.names.append(f"{name}_{instance_id}")

            size_tensor = torch.tensor(
                obj_cfg['size'], device=self.device, dtype=torch.float32
            )
            self.sizes[0, indices] = size_tensor
            self.radii[0, indices] = torch.norm(size_tensor[:2] / 2)
            start_idx += count

        if start_idx != self.num_total_objects:
            raise RuntimeError(
                f"Object indexing mismatch: initialized {start_idx}, "
                f"expected {self.num_total_objects}"
            )

        for type_str, indices in self.type_map.items():
            self.type_map[type_str] = torch.tensor(
                sorted(indices), device=self.device, dtype=torch.long
            )

        self.type_vocab = sorted(self.type_map.keys())
        self.num_types = len(self.type_vocab)
        self.default_positions = default_pos_tensor.expand(
            self.num_envs, -1, -1
        )
        self.positions = self.default_positions.clone()

    def _initialize_strategies(self):
        strategies_by_type = {}

        def _indices_for_types(type_names):
            if isinstance(type_names, str):
                type_names = [type_names]
            acc = []
            for t in type_names:
                inds = self.type_map.get(t, torch.tensor([], dtype=torch.long, device=self.device))
                if len(inds):
                    acc.extend(inds.tolist())
            return sorted(set(acc))

        if self.type_placements_cfg:
            for t, t_cfg in self.type_placements_cfg.items():
                stype = t_cfg["strategy"]
                if stype == "grid":
                    strategies_by_type[t] = GridPlacement(self.device, t_cfg["grid_coordinates"])
                if stype == "grid_with_orient":
                    strategies_by_type[t] = GridPlacementWithOrientation(self.device, t_cfg["grid_coordinates"])
                elif stype == "on_surface":
                    surf_types = t_cfg.get("surface_types", ["surface_provider"])
                    surf_inds = _indices_for_types(surf_types)
                    strategies_by_type[t] = OnSurfacePlacement(self.device, surf_inds, t_cfg["margin"])
        if not strategies_by_type:
            print("[ ERR ] WE HAVE AN ERROR IN _initialize_strategies")
        return strategies_by_type
    
    # ----------- Рандомизация сцены -----------
    def randomize_scene(self, env_ids: torch.Tensor):
        """Build a semantic block layout for selected environments.

        The old type-level random placement is intentionally replaced by
        layout_rules.json driven sequential anchored sampling. All blocks from
        the rules file are active and are processed in JSON order.
        """
        self.layout_sampler.sample_and_apply(env_ids)

    # ----------- Помощники для плана пути -----------
    def get_active_obstacle_positions_for_path_planning(self, env_ids: torch.Tensor) -> list:
        obs_indices = self.type_map.get("movable_obstacle", torch.tensor([], dtype=torch.long))
        if len(obs_indices) == 0:
            return [[] for _ in env_ids]
        active_mask = self.active[env_ids][:, obs_indices]
        positions = self.positions[env_ids][:, obs_indices].cpu().numpy()
        output_list = []
        for i in range(len(env_ids)):
            active_positions = positions[i, active_mask[i].cpu().numpy()]
            rounded_pos = [(round(p[0], 1), round(p[1], 1), round(p[2], 1)) for p in active_positions]
            output_list.append(sorted(rounded_pos))
        return output_list

    def get_graph_embedding(
        self, robot_position, env_ids: torch.Tensor
    ) -> torch.Tensor:
        """Return the legacy 7x2 compact embedding using active objects only."""
        device = self.device
        env_ids = env_ids.to(device=device, dtype=torch.long)
        E = len(env_ids)
        env_pos = self.positions[env_ids, :, :2]
        active = self.active[env_ids]

        if robot_position.shape[0] == self.num_envs:
            robot_xy = robot_position[env_ids, :2]
        else:
            robot_xy = robot_position[:, :2]

        goal_idx = self.active_goal_indices[env_ids]
        movable_idxs = self.type_map.get(
            'movable_obstacle',
            torch.empty(0, device=device, dtype=torch.long),
        )
        poi_idxs = self.type_map.get(
            'possible_goal',
            torch.empty(0, device=device, dtype=torch.long),
        )
        surface_idxs = self.type_map.get(
            'surface_provider',
            torch.empty(0, device=device, dtype=torch.long),
        )

        embed = torch.zeros(E, 7, 2, device=device)
        max_dist = 4.0

        for e in range(E):
            g_idx = int(goal_idx[e].item())
            goal_xy = env_pos[e, g_idx]
            r_xy = robot_xy[e]
            embed[e, 0] = goal_xy - r_xy

            if surface_idxs.numel() > 0:
                mask = active[e, surface_idxs]
                candidates = surface_idxs[mask]
                if candidates.numel() > 0:
                    d = torch.norm(env_pos[e, candidates] - goal_xy, dim=-1)
                    chosen = candidates[torch.argmin(d)]
                    embed[e, 1] = env_pos[e, chosen] - r_xy

            if movable_idxs.numel() > 0:
                mask = active[e, movable_idxs]
                candidates = movable_idxs[mask]
                if candidates.numel() > 0:
                    d = torch.norm(env_pos[e, candidates] - goal_xy, dim=-1)
                    candidates = candidates[d <= max_dist]
                    if candidates.numel() > 0:
                        d_robot = torch.norm(
                            env_pos[e, candidates] - r_xy, dim=-1
                        )
                        order = torch.argsort(d_robot)
                        for slot, idx in enumerate(candidates[order[:3]], start=2):
                            embed[e, slot] = env_pos[e, idx] - r_xy

            if poi_idxs.numel() > 0:
                mask = active[e, poi_idxs]
                candidates = poi_idxs[mask]
                candidates = candidates[candidates != g_idx]
                if candidates.numel() > 0:
                    d = torch.norm(env_pos[e, candidates] - goal_xy, dim=-1)
                    candidates = candidates[d <= max_dist]
                    if candidates.numel() > 0:
                        d = torch.norm(env_pos[e, candidates] - goal_xy, dim=-1)
                        order = torch.argsort(d)
                        for slot, idx in enumerate(candidates[order[:2]], start=5):
                            embed[e, slot] = env_pos[e, idx] - r_xy

        return embed.reshape(E, -1)

    def print_graph_info(self, env_id: int):
        print(f"\n=== Scene Information (Env ID: {env_id}) ===")
        positions = self.positions[env_id]
        active_states = self.active[env_id]
        surface_indices = self.on_surface_idx[env_id]
        surface_levels = self.surface_level[env_id]

        # возьмём RGB с устройства менеджера; ColorQuantizer сам приведёт device/dtype
        colors = self.colors[0]   # [M,3]

        table_data = []
        for i in range(self.num_total_objects):
            name = self.names[i]
            pos = positions[i]
            types = ", ".join([t for t, inds in self.type_map.items() if i in inds])

            rgb = colors[i]
            color_name = ColorQuantizer.rgb_to_name(rgb)
            rgb_str = f"({float(rgb[0]):.2f}, {float(rgb[1]):.2f}, {float(rgb[2]):.2f})"

            row = [
                i, name, types,
                f"({pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f})",
                f"{self.radii[0, i]:.2f}",
                str(active_states[i].item()),
                surface_indices[i].item(),
                surface_levels[i].item(),
                color_name,
                rgb_str,
            ]
            table_data.append(row)

        headers = [
            "ID", "Name", "Types", "Position", "Radius",
            "Active", "On Surface", "Surface Level",
            "Color", "RGB"  # 👈 новые колонки
        ]
        print(tabulate(table_data, headers=headers, tablefmt="grid"))


    # ----------- Выбор активной цели -----------
    def chose_active_goal_state(
        self,
        env_ids: torch.Tensor,
        selected_goal_indices: Optional[torch.Tensor] = None,
    ):
        env_ids = torch.as_tensor(
            env_ids, device=self.device, dtype=torch.long
        )
        goal_indices = self.type_map.get(
            'possible_goal',
            torch.empty(0, dtype=torch.long, device=self.device),
        )
        if goal_indices.numel() == 0:
            raise RuntimeError("No possible_goal objects found in scene_items.json")

        if selected_goal_indices is None:
            active_goal_mask = self.active[env_ids][:, goal_indices]
            active_counts = active_goal_mask.sum(dim=1)
            if not torch.all(active_counts == 1):
                raise RuntimeError(
                    "Automatic goal selection requires exactly one active "
                    "possible_goal per env. goal_mode='all_candidates' must pass "
                    "selected_goal_indices explicitly. Got counts "
                    f"{active_counts.tolist()}"
                )
            chosen_rel = torch.argmax(
                active_goal_mask.to(torch.int64), dim=1
            )
            chosen = goal_indices[chosen_rel]
        else:
            chosen = torch.as_tensor(
                selected_goal_indices,
                device=self.device,
                dtype=torch.long,
            ).flatten()
            if chosen.numel() != env_ids.numel():
                raise ValueError(
                    "selected_goal_indices must align with env_ids"
                )
            possible_mask = torch.zeros(
                self.num_total_objects,
                dtype=torch.bool,
                device=self.device,
            )
            possible_mask[goal_indices] = True
            if not possible_mask[chosen].all():
                raise RuntimeError(
                    "A selected goal index is not of type possible_goal"
                )
            if not self.active[env_ids, chosen].all():
                raise RuntimeError("A selected goal is not physically active")

        self.goal_positions[env_ids] = self.positions[env_ids, chosen]
        self.active_goal_indices[env_ids] = chosen
        self.active_goal_room_ids[env_ids] = self.object_room_ids[
            env_ids, chosen
        ]

    def get_active_goal_state(self, env_ids: torch.Tensor):
        return self.goal_positions[env_ids]
    
    def positions_in_active_navigation_area(
        self, positions: torch.Tensor
    ) -> torch.Tensor:
        return self.room_mapper.positions_in_active_navigation_area(positions)

    def _sample_robot_positions_in_active_rooms(
        self,
        num_positions: int,
        wall_margin: float = 2.0,
    ) -> torch.Tensor:
        active_ids = self.active_room_ids
        choices = torch.randint(
            0,
            active_ids.numel(),
            (num_positions,),
            device=self.device,
        )
        room_ids = active_ids[choices]
        centers = self.room_mapper.centers[room_ids, :2]
        half = self.room_mapper.subroom_half_extent - float(wall_margin)
        if half <= 0:
            raise RuntimeError(
                f"wall_margin={wall_margin} leaves no robot spawn area"
            )
        offsets = (torch.rand(
            num_positions, 2, device=self.device
        ) * 2.0 - 1.0) * half
        return centers + offsets

    def place_robot_for_goal_stage_4_old(
        self, config, env_ids: torch.Tensor, mean_dist: float,
        min_dist: float, max_dist: float, angle_error: float
    ):
        num_envs = len(env_ids)
        final_robot_positions = self._sample_robot_positions_in_active_rooms(
            num_envs, wall_margin=1.6 + self.robot_radius
        )
        # choices = torch.tensor(
        #     [-torch.pi / 2, torch.pi / 2], device=self.device
        # )
        # final_yaw = choices[torch.randint(
        #     0, 2, (num_envs,), device=self.device
        # )]

        direction_to_goal = goal_pos[:, :2] - final_robot_positions
        goal_yaw = torch.atan2(direction_to_goal[:, 1], direction_to_goal[:, 0])

        side = torch.where(
            torch.rand(num_envs, device=self.device) < 0.5,
            -torch.pi / 2,
            torch.pi / 2,
        )
        final_yaw = goal_yaw + side

        robot_quats = torch.zeros(num_envs, 4, device=self.device)
        robot_quats[:, 0] = torch.cos(final_yaw / 2.0)
        robot_quats[:, 3] = torch.sin(final_yaw / 2.0)
        self.remove_colliding_obstacles(env_ids, final_robot_positions)
        return final_robot_positions, robot_quats

    def place_robot_for_goal_stage_4(
        self,
        config,
        env_ids: torch.Tensor,
        mean_dist: float,
        min_dist: float,
        max_dist: float,
        angle_error: float,
    ):
        """Спавнит робота в случайной активной комнате.

        Начальная ориентация перпендикулярна направлению на цель:
        либо -90°, либо +90° относительно вектора robot -> goal.
        """
        env_ids = torch.as_tensor(
            env_ids,
            device=self.device,
            dtype=torch.long,
        )
        num_envs = env_ids.numel()

        # Случайная позиция внутри одной из активных комнат.
        spawn_margin = 1.6 + self.robot_radius
        final_robot_positions = self._sample_robot_positions_in_active_rooms(
            num_positions=num_envs,
            wall_margin=spawn_margin,
        )

        # Цели соответствующих окружений.
        goal_positions = self.goal_positions[env_ids, :2]

        # Направление от робота к цели.
        direction_to_goal = goal_positions - final_robot_positions

        # Ориентация, при которой робот смотрел бы прямо на цель.
        goal_yaw = torch.atan2(
            direction_to_goal[:, 1],
            direction_to_goal[:, 0],
        )

        # Для каждого env случайно выбираем поворот -90° или +90°.
        side_sign = torch.where(
            torch.randint(
                low=0,
                high=2,
                size=(num_envs,),
                device=self.device,
            ) == 0,
            torch.full(
                (num_envs,),
                -1.0,
                device=self.device,
                dtype=goal_yaw.dtype,
            ),
            torch.full(
                (num_envs,),
                1.0,
                device=self.device,
                dtype=goal_yaw.dtype,
            ),
        )

        final_yaw = goal_yaw + side_sign * (torch.pi / 2.0)

        # Нормализуем yaw в диапазон [-pi, pi].
        final_yaw = torch.atan2(
            torch.sin(final_yaw),
            torch.cos(final_yaw),
        )

        # Кватернион Isaac Lab в формате (w, x, y, z).
        robot_quats = torch.zeros(
            num_envs,
            4,
            device=self.device,
            dtype=final_robot_positions.dtype,
        )
        robot_quats[:, 0] = torch.cos(final_yaw / 2.0)
        robot_quats[:, 3] = torch.sin(final_yaw / 2.0)

        self.remove_colliding_obstacles(
            env_ids,
            final_robot_positions,
        )

        return final_robot_positions, robot_quats

    def place_robot_for_goal_stage_3(
        self, config, env_ids: torch.Tensor, mean_dist: float,
        min_dist: float, max_dist: float, angle_error: float
    ):
        num_envs = len(env_ids)
        final_robot_positions = self._sample_robot_positions_in_active_rooms(
            num_envs, wall_margin=1.6 + self.robot_radius
        )
        room_ids = self.room_mapper.room_ids_from_positions(
            final_robot_positions
        )
        centers = self.room_mapper.centers[room_ids, :2]
        direction_from_center = final_robot_positions - centers
        final_yaw = torch.atan2(
            direction_from_center[:, 1], direction_from_center[:, 0]
        )
        robot_quats = torch.zeros(num_envs, 4, device=self.device)
        robot_quats[:, 0] = torch.cos(final_yaw / 2.0)
        robot_quats[:, 3] = torch.sin(final_yaw / 2.0)
        self.remove_colliding_obstacles(env_ids, final_robot_positions)
        return final_robot_positions, robot_quats

    def place_robot_for_goal_stage_2(
        self, config, env_ids: torch.Tensor, mean_dist: float,
        min_dist: float, max_dist: float, angle_error: float
    ):
        num_envs = len(env_ids)
        final_robot_positions = self._sample_robot_positions_in_active_rooms(
            num_envs, wall_margin=1.6 + self.robot_radius
        )
        final_yaw = (
            torch.rand(num_envs, device=self.device) * 2 * torch.pi
        ) - torch.pi
        robot_quats = torch.zeros(num_envs, 4, device=self.device)
        robot_quats[:, 0] = torch.cos(final_yaw / 2.0)
        robot_quats[:, 3] = torch.sin(final_yaw / 2.0)
        self.remove_colliding_obstacles(env_ids, final_robot_positions)
        return final_robot_positions, robot_quats

    def place_robot_for_goal_stage_1(
        self, config, env_ids: torch.Tensor, mean_dist: float,
        min_dist: float, max_dist: float, angle_error: float
    ):
        num_envs = len(env_ids)
        goal_pos = self.goal_positions[env_ids]
        mean_dist_with_shift = mean_dist + 1.31
        radii = torch.normal(
            mean=mean_dist_with_shift,
            std=0.1,
            size=(num_envs, 1),
            device=self.device,
        ).clamp_(min_dist, max_dist)
        candidates = (
            goal_pos[:, None, :2]
            + radii.unsqueeze(1) * self.candidate_vectors
        )

        spawn_margin = 1.6 + self.robot_radius
        valid = self.room_mapper.positions_in_active_room_interiors(
            candidates,
            margin=spawn_margin,
        )

        has_valid = valid.any(dim=1)
        weights = valid.float()
        weights[~has_valid, 0] = 1.0
        chosen_angle_idx = torch.multinomial(weights, 1).squeeze(-1)
        batch_indices = torch.arange(num_envs, device=self.device)
        final_robot_positions = candidates[
            batch_indices, chosen_angle_idx
        ]
        if (~has_valid).any():
            final_robot_positions[~has_valid] = (
                self._sample_robot_positions_in_active_rooms(
                    int((~has_valid).sum().item()),
                    wall_margin=spawn_margin,
                )
            )

        direction_to_goal = goal_pos[:, :2] - final_robot_positions
        base_yaw = torch.atan2(
            direction_to_goal[:, 1], direction_to_goal[:, 0]
        )
        error = (
            torch.rand(num_envs, device=self.device) - 0.5
        ) * 2 * angle_error
        final_yaw = base_yaw + error
        robot_quats = torch.zeros(num_envs, 4, device=self.device)
        robot_quats[:, 0] = torch.cos(final_yaw / 2.0)
        robot_quats[:, 3] = torch.sin(final_yaw / 2.0)
        self.remove_colliding_obstacles(env_ids, final_robot_positions)
        return final_robot_positions, robot_quats

    def place_robot_for_goal_stage_0(
        self, config, env_ids: torch.Tensor, mean_dist: float,
        min_dist: float, max_dist: float, angle_error: float
    ):
        num_envs = len(env_ids)
        final_robot_positions = self._sample_robot_positions_in_active_rooms(
            num_envs, wall_margin=1.6 + self.robot_radius
        )
        final_yaw = torch.rand(
            num_envs, device=self.device
        ) * 2 * math.pi
        robot_quats = torch.zeros(num_envs, 4, device=self.device)
        robot_quats[:, 0] = torch.cos(final_yaw / 2.0)
        robot_quats[:, 3] = torch.sin(final_yaw / 2.0)
        self.remove_colliding_obstacles(env_ids, final_robot_positions)
        return final_robot_positions, robot_quats

    def remove_colliding_obstacles(self, env_ids: torch.Tensor, robot_positions: torch.Tensor):
        """
        Удаляет colliding active объекты (все типы) с роботом.
        - Проверяет dist < (robot_radius + obj_r + 0.2) для всех active obj.
        - Деактивирует и перемещает в default_pos.
        """
        obs_indices = torch.arange(self.num_total_objects, device=self.device)  # ВСЕ индексы (не только movable)
        if len(obs_indices) == 0:
            return
        E = len(env_ids)
        obs_pos = self.positions[env_ids][:, obs_indices, :2]  # [E, M, 2]
        obs_r = self.radii.expand(E, -1)[:, obs_indices]       # [E, M]
        active_obs_mask = self.active[env_ids][:, obs_indices] # [E, M] — только active

        # Маскируем неактивные: их pos/r игнорируем (inf dist)
        obs_pos = torch.where(active_obs_mask.unsqueeze(-1), obs_pos, 
                            torch.full_like(obs_pos, 999.0))
        obs_r = torch.where(active_obs_mask, obs_r, 
                            torch.full_like(obs_r, 999.0))

        dists = torch.norm(obs_pos - robot_positions[:, None, :2], dim=2)  # [E, M]
        coll_mask = dists < (self.robot_radius + 0.4 + 0.2)  # [E, M]
            
        if coll_mask.any():
            # print("robot pos: ", robot_positions[:, None, :2])
            # print("collisiona mask: ", coll_mask)
            # Деактивируем colliding
            batch_idx, obs_idx = torch.where(coll_mask)
            # print("obs_idx :", obs_idx)
            env_batch_idx = env_ids[batch_idx]
            obs_indices_sel = obs_indices[obs_idx]
            
            self.active[env_batch_idx, obs_indices_sel] = False
            # Перемещаем в default
            default_pos = self.default_positions[env_batch_idx, obs_indices_sel]
            self.positions[env_batch_idx, obs_indices_sel] = default_pos
            
            # Лог (опционально, для дебага)
            # print(f"[COLL DEBUG] Deactivated {coll_mask.sum().item()} obstacles in envs {env_ids.tolist()}")

    # ----------- Делегаты в SceneGraph -----------
    def get_graph_obs(self, env_ids: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        return self.graph.get_observation(env_ids)

    @torch.no_grad()
    def compute_relations(self, env_ids: torch.Tensor, reference: str | int = 'goal', *,
                          use_local_frame: bool = True, reference_yaws: Optional[torch.Tensor] = None,
                          radius: Optional[float] = 5.0, include_inactive: bool = False) -> List[Dict[str, int]]:
        return self.graph.compute_relations(env_ids, reference, use_local_frame=use_local_frame,
                                            reference_yaws=reference_yaws, radius=radius,
                                            include_inactive=include_inactive)

    @torch.no_grad()
    def get_navigation_prompts(self, env_ids: torch.Tensor, goal_name: Optional[str] = None, radius: float = 5.0,
                               use_local_frame: bool = True, reference_yaws: Optional[torch.Tensor] = None) -> List[str]:
        return self.graph.build_navigation_prompt(env_ids, goal_name=goal_name, radius=radius,
                                                  use_local_frame=use_local_frame, reference_yaws=reference_yaws)

    @torch.no_grad()
    def encode_scene_graph(
        self,
        env_ids: Optional[torch.Tensor] = None,
        flatten: bool = True,
    ) -> torch.Tensor:
        """Compact graph state for GraphEncoder.

        Per object: [object_id, active, is_goal, x_room, y_room, z_room].
        Static semantics are not duplicated in the state; GraphEncoder resolves
        name/color embeddings by object_id from the precomputed cache.
        is_goal marks the current navigation target for goal-centered sparse
        edge construction.
        """
        device = self.device
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=device)
        else:
            env_ids = env_ids.to(device=device, dtype=torch.long)

        E = len(env_ids)
        M = self.num_total_objects
        object_ids = self.object_ids.expand(E, -1).unsqueeze(-1).float()
        active = self.active[env_ids].unsqueeze(-1).float()
        positions = self.positions[env_ids].float()

        is_goal = torch.zeros(E, M, 1, device=device, dtype=torch.float32)
        goal_idx = self.active_goal_indices[env_ids].long().clamp(0, M - 1)
        is_goal[torch.arange(E, device=device), goal_idx, 0] = 1.0

        graph = torch.cat([object_ids, active, is_goal, positions], dim=-1)  # [E, M, 6]

        if flatten:
            return graph.reshape(E, M * 6)
        return graph

    @torch.no_grad()
    def detect_visible_objects_2d(
        self,
        robot_pos_local: torch.Tensor,
        robot_yaw: torch.Tensor,
        env_ids: Optional[torch.Tensor] = None,
        *,
        fov_deg: float = 90.0,
        max_distance: Optional[float] = None,
        eps: float = 1e-6,
    ) -> torch.Tensor:
        """
        Геометрический 2D-detector видимых объектов.

        Args:
            robot_pos_local:
                Позиция робота в локальной системе комнаты.
                Shape:
                    [num_envs, 2/3] или [E, 2/3].
            robot_yaw:
                Ориентация робота вокруг Z в радианах.
                Shape:
                    [num_envs], [num_envs, 1], [E] или [E, 1].
            env_ids:
                Индексы env. Если None — считаем для всех env.
            fov_deg:
                Горизонтальный угол обзора detector'а в градусах.
            max_distance:
                Максимальная дистанция видимости. Если None — фильтра по дальности нет.
            eps:
                Защита от деления на ноль.

        Returns:
            out:
                Tensor shape [E, M, 2],
                где:
                    out[..., 0] = object_id
                    out[..., 1] = visible_flag {0.0, 1.0}

                E = len(env_ids)
                M = self.num_total_objects
        """
        device = self.device

        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=device)
        else:
            env_ids = env_ids.to(device=device, dtype=torch.long)

        E = int(env_ids.numel())
        M = self.num_total_objects

        robot_pos_local = robot_pos_local.to(device=device, dtype=torch.float32)
        robot_yaw = robot_yaw.to(device=device, dtype=torch.float32)

        if robot_yaw.dim() == 2 and robot_yaw.shape[-1] == 1:
            robot_yaw = robot_yaw.squeeze(-1)

        # Разрешаем два режима:
        # 1) передали позы всех env: [num_envs, ...]
        # 2) передали позы только выбранных env_ids: [E, ...]
        if robot_pos_local.shape[0] == self.num_envs:
            robot_xy = robot_pos_local[env_ids, :2]
        elif robot_pos_local.shape[0] == E:
            robot_xy = robot_pos_local[:, :2]
        else:
            raise ValueError(
                f"robot_pos_local first dim must be num_envs={self.num_envs} or E={E}, "
                f"got {robot_pos_local.shape[0]}"
            )

        if robot_yaw.shape[0] == self.num_envs:
            yaw = robot_yaw[env_ids]
        elif robot_yaw.shape[0] == E:
            yaw = robot_yaw
        else:
            raise ValueError(
                f"robot_yaw first dim must be num_envs={self.num_envs} or E={E}, "
                f"got {robot_yaw.shape[0]}"
            )

        obj_xy = self.positions[env_ids, :, :2].float()      # [E, M, 2]
        active = self.active[env_ids]                        # [E, M]

        delta = obj_xy - robot_xy[:, None, :]                # [E, M, 2]
        dist = torch.linalg.norm(delta, dim=-1)              # [E, M]

        delta_unit = delta / dist.clamp_min(eps).unsqueeze(-1)

        forward = torch.stack(
            [torch.cos(yaw), torch.sin(yaw)],
            dim=-1,
        )                                                    # [E, 2]

        cos_angle = torch.sum(delta_unit * forward[:, None, :], dim=-1)  # [E, M]
        cos_threshold = math.cos(math.radians(fov_deg) * 0.5)

        visible = active & (dist > eps) & (cos_angle >= cos_threshold)

        if max_distance is not None:
            visible = visible & (dist <= float(max_distance))

        object_ids = self.object_ids.expand(E, -1).float()   # [E, M]
        visible_f = visible.float()                          # [E, M]

        return torch.stack([object_ids, visible_f], dim=-1)  # [E, M, 2]