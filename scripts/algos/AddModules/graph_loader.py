# -*- coding: utf-8 -*-
"""
graph_loader.py
---------------
Загрузчик внешних сцен-графов (scene_{id}_graph.json) с конвертацией
в формат GraphEncoder.

Топология GraphEncoder — звезда + цепь:
  - Узел 0 (goal/teddy) соединён со всеми остальными (центр звезды)
  - Все узлы соединены последовательно (цепь 0→1→2→...→M-1)
  - Self-loops на каждом узле

Порядок узлов в тензоре:
  [0]     goal (teddy bear)
  [1..K-1] активные объекты, отсортированные по дистанции XY до goal
  [K..M-1] нулевые слоты (inactive)

Формат каждого узла (24 float):
  [0:3]   position xyz
  [3:6]   size whd (extent)
  [6]     radius  = norm(extent[:2]/2)
  [7:10]  color rgb
  [10]    object_id  (из OBJECT_ID_MAP)
  [11]    active  (1.0 для всех узлов из внешнего графа)
  [12]    on_surface_idx  (локальный индекс родителя или -1.0)
  [13]    surface_level
  [14]    edge_exists  (1.0 если есть родитель)
  [15]    z_diff       (z_child - z_parent)
  [16]    level_diff   (level_child - level_parent)
  [17]    xy_dist      (norm(xy_child - xy_parent))
  [18]    color_diff_norm
  [19]    id_diff
  [20]    name_idx     (из codebook["names"])
  [21]    color_idx    (из codebook["colors"])
  [22:24] zeros        (pad)
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch


# ─────────────────────────────────────────────────────────────────────────────
# Константы (должны совпадать с scene_manager и models.py)
# ─────────────────────────────────────────────────────────────────────────────

NUM_GRAPH_NODES = 21
NODE_DIM = 24

OBJECT_ID_MAP: Dict[str, int] = {
    "air": 0, "box": 1, "cabinet": 2, "chair": 3, "clock": 4,
    "crestwood": 5, "desk": 6, "ladder": 7, "lamp": 8, "standard": 9,
    "table": 10, "teddy": 11, "trashcan": 12, "vase": 13, "yucca": 14, "bowl": 15,
}

# Нормализация class_name из внешних графов → имена в OBJECT_ID_MAP
CLASS_ALIASES: Dict[str, str] = {
    "teddy bear": "teddy",
    "bear":       "teddy",
    "trash can":  "trashcan",
    "trash_can":  "trashcan",
    "potted plant": "yucca",
    "plant":      "yucca",
}

GOAL_CLASS = "teddy bear"   # всегда


# ─────────────────────────────────────────────────────────────────────────────
# Вспомогательные функции
# ─────────────────────────────────────────────────────────────────────────────

def _normalize_class(raw: str) -> str:
    """Приводит произвольное имя класса к ключу OBJECT_ID_MAP."""
    key = raw.strip().lower()
    return CLASS_ALIASES.get(key, key)


def _compute_radius(extent: List[float]) -> float:
    """norm(extent[:2] / 2) — аналогично SceneManager."""
    return math.sqrt((extent[0] / 2) ** 2 + (extent[1] / 2) ** 2)


def _build_surface_relations(
    nodes: Dict[str, dict],
) -> Tuple[Dict[str, Optional[str]], Dict[str, int]]:
    """
    Восстанавливает родительские отношения из edges_bs.

    Соглашение edges_bs:
        node_A["edges_bs"][node_B_id] = "above"
        → A находится ВЫШЕ B → A лежит НА B → parent(A) = B

    Возвращает:
        on_surface_parent : {node_id → parent_node_id | None}
        surface_level     : {node_id → int}
    """
    on_surface_parent: Dict[str, Optional[str]] = {}

    for node_id, node_data in nodes.items():
        # edges_bs — список {"target_id": int, "relation_type": str}
        edges = node_data.get("edges_bs", [])
        parent: Optional[str] = None
        for edge in edges:
            if edge.get("relation_type") == "above":
                # target_id — int, но ключи nodes — строки
                parent = str(edge["target_id"])
                break
        on_surface_parent[node_id] = parent

    # Итеративный подсчёт глубины цепочки
    surface_level: Dict[str, int] = {}
    for node_id in nodes:
        level = 0
        visited: set = set()
        cur = node_id
        while on_surface_parent.get(cur) is not None and cur not in visited:
            visited.add(cur)
            cur = on_surface_parent[cur]   # type: ignore[assignment]
            level += 1
        surface_level[node_id] = level

    return on_surface_parent, surface_level


# ─────────────────────────────────────────────────────────────────────────────
# Звезда + цепь: перегруппировка узлов
# ─────────────────────────────────────────────────────────────────────────────

def reorder_nodes_star_chain(
    node_ids: List[str],
    nodes:    Dict[str, dict],
) -> List[str]:
    """
    Возвращает node_ids, переупорядоченные точно так же как reorder_by_goal
    в SceneManager — циклический сдвиг, чтобы goal оказался на позиции 0:

      original:  [0, 1, 2, 3, 4, 5, ...]   (goal на позиции goal_idx)
      result:    [goal_idx, goal_idx+1, ..., M-1, 0, 1, ..., goal_idx-1]

    Это важно: GraphEncoder уже обучен на данных encode_scene_graph,
    которая использует именно такой циклический сдвиг через (base + goal_idx) % M.
    Менять порядок нельзя — веса модели ожидают именно его.
    """
    # Находим позицию goal в исходном списке
    goal_idx: int = 0
    found = False
    for i, nid in enumerate(node_ids):
        raw = nodes[nid]["class_name"]
        normalized = _normalize_class(raw)
        # Проверяем точное совпадение после нормализации ИЛИ вхождение "teddy"/"bear"
        if normalized == GOAL_CLASS or "teddy" in raw.lower() or "bear" in raw.lower() or "teddy bear" in raw.lower():
            print("FIND!")
            goal_idx = i
            found = True
            break

    if not found:
        print(f"[graph_loader] WARNING: goal class '{GOAL_CLASS}' not found, "
              f"using node '{node_ids[0]}' as center")

    # Циклический сдвиг: [goal_idx, goal_idx+1, ..., M-1, 0, 1, ..., goal_idx-1]
    return node_ids[goal_idx:] + node_ids[:goal_idx]


# ─────────────────────────────────────────────────────────────────────────────
# Основная функция конвертации
# ─────────────────────────────────────────────────────────────────────────────

def convert_external_graph(
    graph_path:    str,
    scene_manager,          # SceneManager instance
    device:        str = "cpu",
) -> torch.Tensor:
    """
    Читает scene_{id}_graph.json и возвращает тензор [NUM_GRAPH_NODES * NODE_DIM].

    Шаги:
      1. Парсим JSON, нормализуем class_name.
      2. Восстанавливаем parent/level из edges_bs.
      3. Переупорядочиваем узлы: goal=0, далее по дистанции (star+chain).
      4. Заполняем тензор [M, 24] числовыми признаками.
      5. Вычисляем edge features (второй проход по родителям).
      6. Добавляем нулевые (inactive) слоты до M=21.
      7. Возвращаем reshape(-1) → [M*24].
    """
    # ── 1. Загрузка ──────────────────────────────────────────────────────────
    with open(graph_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    nodes: Dict[str, dict] = data["nodes"]

    # ── 2. Родительские отношения из edges_bs ────────────────────────────────
    on_surface_parent, surface_level = _build_surface_relations(nodes)

    # ── 3. Переупорядочиваем узлы (star + chain) ─────────────────────────────
    original_ids = list(nodes.keys())
    ordered_ids  = reorder_nodes_star_chain(original_ids, nodes)

    # Маппинг: node_id → позиция в новом порядке (нужен для on_surface_idx)
    node_to_slot: Dict[str, int] = {nid: slot for slot, nid in enumerate(ordered_ids)}

    # ── Вспомогательные словари из SceneManager ───────────────────────────────
    sm = scene_manager
    codebook_names  = sm.codebook.get("names",  {})
    codebook_colors = sm.codebook.get("colors", {})

    # class_name → RGB tensor (берём цвет первого инстанса из SceneManager)
    class_to_color: Dict[str, torch.Tensor] = {}
    for obj_name, obj_info in sm.object_map.items():
        idx = int(obj_info["indices"][0].item())
        class_to_color[obj_name] = sm.colors[0, idx].cpu()

    # Импортируем ColorQuantizer из scene_manager
    # (предполагается что он в том же пакете)
    from scene_manager_v3 import ColorQuantizer

    # ── 4. Заполняем тензор [M, D] ───────────────────────────────────────────
    M = NUM_GRAPH_NODES
    D = NODE_DIM
    result = torch.zeros(M, D, dtype=torch.float32)

    K = min(len(ordered_ids), M)   # сколько активных узлов поместится

    for slot, node_id in enumerate(ordered_ids[:K]):
        node       = nodes[node_id]
        class_name = _normalize_class(node["class_name"])
        obb        = node["bbox_3d"]["obb"]

        center = torch.tensor(obb["center"], dtype=torch.float32)
        extent = torch.tensor(obb["extent"], dtype=torch.float32)
        radius = _compute_radius(obb["extent"])

        # Цвет: из SceneManager по нормализованному class_name, fallback → gray
        color_rgb = class_to_color.get(
            class_name,
            torch.tensor([0.5, 0.5, 0.5], dtype=torch.float32),
        )

        obj_id = float(OBJECT_ID_MAP.get(class_name, 0))

        # on_surface_idx → переводим node_id родителя в новый слот
        parent_node_id = on_surface_parent.get(node_id)
        if parent_node_id is not None and parent_node_id in node_to_slot:
            parent_slot = float(node_to_slot[parent_node_id])
        else:
            parent_slot = -1.0

        s_level = float(surface_level.get(node_id, 0))

        # codebook lookups
        name_idx_val  = float(int(codebook_names.get(class_name, 0)))
        color_name    = ColorQuantizer.rgb_to_name(color_rgb)
        color_idx_val = float(int(codebook_colors.get(color_name, 0)))

        result[slot, 0:3]  = center
        result[slot, 3:6]  = extent
        result[slot, 6]    = radius
        result[slot, 7:10] = color_rgb
        result[slot, 10]   = obj_id
        result[slot, 11]   = 1.0           # active
        result[slot, 12]   = parent_slot   # on_surface_idx
        result[slot, 13]   = s_level
        # [14:20] — edge features: второй проход
        result[slot, 20]   = name_idx_val
        result[slot, 21]   = color_idx_val
        # [22:24] = 0 (pad)

    # ── 5. Edge features (второй проход) ─────────────────────────────────────
    for slot in range(K):
        parent_slot = int(result[slot, 12].item())
        if parent_slot < 0 or parent_slot >= K:
            # edge_exists = 0, остальное уже 0
            continue

        p = parent_slot
        result[slot, 14] = 1.0                                                        # edge_exists
        result[slot, 15] = float(result[slot, 2]    - result[p, 2])                  # z_diff
        result[slot, 16] = float(result[slot, 13]   - result[p, 13])                 # level_diff
        result[slot, 17] = float(torch.norm(result[slot, 0:2] - result[p, 0:2]))     # xy_dist
        result[slot, 18] = float(torch.norm(result[slot, 7:10] - result[p, 7:10]))   # color_diff
        result[slot, 19] = float(result[slot, 10]   - result[p, 10])                 # id_diff

    # ── 6. Слоты [K..M-1] уже нули (inactive) ───────────────────────────────
    # active=0 уже стоит по умолчанию, on_surface_idx=-1 нужно поставить явно
    for slot in range(K, M):
        result[slot, 12] = -1.0

    return result.reshape(-1).to(device)   # [M * D]


# ─────────────────────────────────────────────────────────────────────────────
# Кэш всех графов
# ─────────────────────────────────────────────────────────────────────────────

class ExternalGraphCache:
    """
    Загружает и кэширует все внешние графы из папки graphs_dir.
    Имена файлов: scene_{id}_graph.json (id совпадает с id в scene_items_maps.json).

    Использование
    -------------
    # При инициализации env:
        cache = ExternalGraphCache(graphs_dir, scene_manager)
        cache.load_all()

    # В apply_fixed_scene (после выбора scene_id):
        scene_manager.env_scene_ids[env_id] = scene_id

    # При сборке obs (вместо encode_scene_graph):
        scene_ids = scene_manager.env_scene_ids[env_ids]  # [E]
        self.scene_embeddings[env_ids] = cache.get_batch(scene_ids, device)
    """

    def __init__(
        self,
        graphs_dir:    str,
        scene_manager,
        device:        str = "cpu",
    ):
        self.graphs_dir    = graphs_dir
        self.sm            = scene_manager
        self.device        = device
        self._cache: Dict[int, torch.Tensor] = {}  # scene_id → [M*D]

    # ── Загрузка ──────────────────────────────────────────────────────────────

    def load_all(self) -> None:
        """Парсит все scene_*_graph.json из папки и кэширует тензоры."""
        path  = Path(self.graphs_dir)
        files = sorted(path.glob("scene_*_graph.json"))

        if not files:
            print(f"[ExternalGraphCache] WARNING: no graph files in '{self.graphs_dir}'")
            return

        for fpath in files:
            scene_id = self._parse_scene_id(fpath.name)
            if scene_id is None:
                continue

            tensor = convert_external_graph(
                graph_path    = str(fpath),
                scene_manager = self.sm,
                device        = self.device,
            )
            self._cache[scene_id] = tensor

        print(f"[ExternalGraphCache] Loaded {len(self._cache)} graphs, "
              f"ids: {sorted(self._cache.keys())}")

    @staticmethod
    def _parse_scene_id(filename: str) -> Optional[int]:
        """scene_42_graph.json → 42,  иначе None."""
        # filename: "scene_42_graph.json"  →  parts: ["scene","42","graph"]
        stem  = filename.replace(".json", "")
        parts = stem.split("_")
        try:
            return int(parts[1])
        except (IndexError, ValueError):
            print(f"[ExternalGraphCache] Cannot parse scene_id from '{filename}', skipping")
            return None

    # ── Доступ ────────────────────────────────────────────────────────────────

    def get(self, scene_id: int, device: Optional[str] = None) -> torch.Tensor:
        """Возвращает тензор [M*D]. При отсутствии — нули + warning."""
        tensor = self._cache.get(scene_id)
        if tensor is None:
            print(f"[ExternalGraphCache] WARNING: scene_id={scene_id} not found, "
                  f"returning zeros")
            tensor = torch.zeros(
                NUM_GRAPH_NODES * NODE_DIM, dtype=torch.float32
            )
        return tensor.to(device or self.device)

    def get_batch(
        self,
        scene_ids: torch.Tensor,   # LongTensor [E]
        device:    Optional[str] = None,
    ) -> torch.Tensor:
        """Возвращает батч [E, M*D]."""
        tensors = [self.get(int(sid.item()), device) for sid in scene_ids]
        return torch.stack(tensors, dim=0)

    def __len__(self)                  -> int:  return len(self._cache)
    def __contains__(self, sid: int)   -> bool: return sid in self._cache