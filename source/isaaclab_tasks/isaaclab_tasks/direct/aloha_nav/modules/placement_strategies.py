import torch
import random
from abc import ABC, abstractmethod
import math
# ПРИМЕЧАНИЕ: ObstacleGraph больше не используется. Вместо него передаются тензоры
# из нового VectorizedSceneManager.

class PlacementStrategy(ABC):
    def __init__(self, device: str, **kwargs):
        self.device = device

    @abstractmethod
    def apply(self, env_ids: torch.Tensor,
        all_env_ids: torch.Tensor, object_indices: torch.Tensor, scene_data: dict, mess: bool):
        # scene_data будет словарем с тензорами: 'positions', 'active' и т.д.
        pass

class FixedPlacement(PlacementStrategy):
    def __init__(self, device: str, positions_dict: dict):
        super().__init__(device)
        self.positions_dict = positions_dict  # { "chair": [[x,y,z], ...], "table": [...], ... }

    def apply(self, env_ids: torch.Tensor,
        all_env_ids: torch.Tensor, obj_indices: torch.Tensor, scene_data: dict, config: dict[str, torch.Tensor], mess: bool,):
        # obj_indices: (num_envs, num_to_place) — индексы объектов, которые нужно разместить
        # Для фиксированного размещения мы просто берём позиции из словаря
        for env_i, env_id in enumerate(env_ids.tolist()):
            for j, obj_idx in enumerate(obj_indices[env_i].tolist()):
                # Определяем имя по индексу (scene_manager.names[obj_idx])
                obj_name = scene_data["names"][obj_idx] if "names" in scene_data else f"obj_{obj_idx}"
                if obj_name.split("_")[0] in self.positions_dict:
                    pos_list = self.positions_dict[obj_name.split("_")[0]]
                    if j < len(pos_list):
                        scene_data["positions"][env_id, obj_idx] = torch.tensor(pos_list[j], device=self.device)
                        scene_data["active"][env_id, obj_idx] = True
                        scene_data["on_surface_idx"][env_id, obj_idx] = -1
                        scene_data["surface_level"][env_id, obj_idx] = 0


class GridPlacement(PlacementStrategy):
    def __init__(self, device: str, grid_coordinates: list[list[float]]):
        super().__init__(device)
        self.grid = torch.tensor(grid_coordinates, device=self.device, dtype=torch.float32)

    def apply(
        self,
        env_ids: torch.Tensor,
        all_env_ids: torch.Tensor,
        obj_indices_to_place: torch.Tensor,  # [E, K]
        scene_data: dict,
        config: dict[str, torch.Tensor],
        mess: bool, 
    ):
        E, K = obj_indices_to_place.shape
        if K == 0 or E == 0:
            return

        G = self.grid.size(0)
        if G == 0:
            return

        # сколько реально можем поставить (не больше точек сетки)
        KK = min(K, G)

        # сэмплим перестановки сетки и берём первые KK
        pos_indices = torch.rand(E, G, device=self.device).argsort(dim=1)[:, :KK]  # [E, KK]
        selected_positions = self.grid[pos_indices]  # [E, KK, 3]

        # индексация env × obj, попарно
        env_idx_tensor = env_ids.view(-1, 1).expand(E, KK)

        # берём только первые KK объектов
        obj_idx_slice = obj_indices_to_place[:, :KK]

        scene_data['positions'][env_idx_tensor, obj_idx_slice] = selected_positions
        scene_data['active'][env_idx_tensor, obj_idx_slice] = True
        scene_data['on_surface_idx'][env_idx_tensor, obj_idx_slice] = -1
        scene_data['surface_level'][env_idx_tensor, obj_idx_slice] = 0


class OnSurfacePlacement(PlacementStrategy):
    def __init__(self, device: str, surface_indices: list[int], margin: float):
        super().__init__(device)
        self.surface_indices = torch.tensor(surface_indices, device=self.device, dtype=torch.long)
        self.margin = margin

    def apply(
        self,
        env_ids: torch.Tensor,
        all_env_ids: torch.Tensor,
        obj_indices_to_place: torch.Tensor,  # [E, K]
        scene_data: dict,
        config: dict[str, torch.Tensor],
        mess: bool, 
    ):
        E, K = obj_indices_to_place.shape
        if K == 0 or E == 0:
            return

        # активные поверхности в этих env
        active_surf_mask = scene_data['active'][env_ids][:, self.surface_indices]  # [E, S]
        S = active_surf_mask.size(1)
        if S == 0:
            return

        # Сколько реально можем положить: не больше числа активных поверхностей (если без replacement)
        # Если хотите разрешить несколько объектов на один стол — поставьте replacement=True и уберите clamp ниже.
        num_active_per_env = active_surf_mask.sum(dim=1)  # [E]
        KK_per_env = torch.clamp(num_active_per_env, max=K)  # [E]

        # Готовим выходные контейнеры
        chosen_surface_idx = torch.full((E, K), fill_value=-1, dtype=torch.long, device=self.device)

        # По каждой среде отдельно (чтобы корректно учесть разное число доступных столов)
        for e in range(E):
            kk = int(KK_per_env[e].item())
            if kk == 0:
                continue
            probs = active_surf_mask[e].float()
            # защита от всех нулей
            if probs.sum() == 0:
                continue
            # равномерные вероятности по активным
            probs = probs / probs.sum()

            # выбираем kk активных поверхностей без повтора
            rel_idx = torch.multinomial(probs, kk, replacement=False)    # [kk] относительные индексы в 0..S-1
            chosen_surface_idx[e, :kk] = self.surface_indices[rel_idx]   # глобальные индексы поверхностей

        # Маска реально выбранных пар (E,K') где K' <= K
        valid_mask = chosen_surface_idx >= 0
        if not valid_mask.any():
            return

        # Собираем индексы env и объектов только там, где valid
        env_idx_tensor = env_ids.view(-1, 1).expand_as(chosen_surface_idx)[valid_mask]     # [N_valid]
        surf_idx_flat  = chosen_surface_idx[valid_mask]                                     # [N_valid]
        obj_idx_flat   = obj_indices_to_place[valid_mask]                                   # [N_valid]

        # Позиции/размеры
        surface_pos  = scene_data['positions'][env_idx_tensor, surf_idx_flat]              # [N_valid, 3]
        surface_size = scene_data['sizes'][env_idx_tensor, surf_idx_flat]                  # [N_valid, 3]
        obj_size     = scene_data['sizes'][env_idx_tensor, obj_idx_flat]                   # [N_valid, 3]

        # XY-джиттер внутри габаритов поверхности (с учётом margin)
        if mess:
            max_offsets = torch.clamp(surface_size[:, :2] - 2 * self.margin, min=0.0)
            rand_xy = (torch.rand_like(max_offsets) - 0.5) * max_offsets
        else:
            rand_xy = torch.zeros_like(surface_pos[:, :2])

        new_xy = surface_pos[:, :2] + rand_xy
        new_z  = surface_pos[:, 2] + surface_size[:, 2] + obj_size[:, 2] * 0.5

        new_xyz = torch.zeros_like(surface_pos)
        new_xyz[:, :2] = new_xy
        new_xyz[:, 2]  = new_z

        # Применяем
        scene_data['positions'][env_idx_tensor, obj_idx_flat] = new_xyz
        scene_data['active'][env_idx_tensor, obj_idx_flat] = True
        scene_data['on_surface_idx'][env_idx_tensor, obj_idx_flat] = surf_idx_flat

        # уровни: уровень поверхности + 1 (если у вас surface_level — тензор [N, M], как в коде)
        surface_levels = scene_data['surface_level'][env_idx_tensor, surf_idx_flat]
        scene_data['surface_level'][env_idx_tensor, obj_idx_flat] = surface_levels + 1

class GridPlacementWithOrientation:
    """Grid placement strategy that supports rotation by multiples of π/2."""
    
    def __init__(self, device: torch.device, grid_coordinates: list[list[float]]):
        self.device = device
        self.grid = torch.tensor(grid_coordinates, device=self.device, dtype=torch.float32)
        self.base_coordinates = torch.tensor(grid_coordinates, device=device, dtype=torch.float32)  # [N, 3]
        
    def apply(
        self,
        env_ids: torch.Tensor,
        all_env_ids: torch.Tensor,
        obj_indices_to_place: torch.Tensor,  # [E, K]
        scene_data: dict,
        config: dict[str, torch.Tensor],
        mess: bool, 
    ):
        E, K = obj_indices_to_place.shape
        if K == 0 or E == 0:
           # print(f"[DEBUG] E={E}, K={K} - returning early")
            return

        G = self.grid.size(0)
        if G == 0:
           # print(f"[DEBUG] G={G} - grid is empty, returning")
            return

       # print(f"[DEBUG] ====== ENTER apply ======")
       # print(f"[DEBUG] env_ids: {env_ids}, shape: {env_ids.shape}")
       # print(f"[DEBUG] all_env_ids: {all_env_ids}, shape: {all_env_ids.shape}")
       # print(f"[DEBUG] E={E}, K={K}, G={G}")

        # сколько реально можем поставить (не больше точек сетки)
        KK = min(K, G)
       # print(f"[DEBUG] KK = min(K={K}, G={G}) = {KK}")

        # сэмплим перестановки сетки и берём первые KK
        pos_indices = torch.rand(E, G, device=self.device).argsort(dim=1)[:, :KK]  # [E, KK]
        selected_positions = self.grid[pos_indices]  # [E, KK, 3]
       # print(f"[DEBUG] selected_positions shape: {selected_positions.shape}")
       # print(f"[DEBUG] selected_positions sample: {selected_positions[0] if E > 0 else 'empty'}")

        # === ДОБАВЛЕННЫЙ КОД: ПОВОРОТ ===
        orientations = config["orientation"]  # [num_total_envs] - ориентации для all_env_ids
       # print(f"[DEBUG] orientations from config: shape={orientations.shape}, values={orientations}")
        
        # Вычитаем π/2
        orientations = orientations - math.pi / 2
       # print(f"[DEBUG] orientations after -π/2: {orientations}")
        
        # Выбираем ориентации для нужных окружений
        # Нужно найти, каким индексам в orientations соответствуют env_ids
        if orientations.numel() == E:
            # Уже правильный размер - редкий случай
            selected_orientations = orientations
           # print(f"[DEBUG] Case 1: orientations already correct size")
        else:
            # Основной случай: orientations содержит ориентации для all_env_ids
            # Нужно найти позиции env_ids в all_env_ids
           # print(f"[DEBUG] Case 2: mapping env_ids to positions in all_env_ids")
            
            # Создаем словарь для быстрого поиска позиций
            # all_env_ids = [0, 2] (например)
            # env_ids = [2] (ищем позицию 2 в [0, 2] → позиция 1)
            
            all_env_ids_list = all_env_ids.tolist()
            env_ids_list = env_ids.tolist()
            
            # Находим индексы в all_env_ids для каждого env_id
            position_indices = []
            for env_id in env_ids_list:
                if env_id in all_env_ids_list:
                    pos_idx = all_env_ids_list.index(env_id)
                    position_indices.append(pos_idx)
                else:
                    print(f"[WARN] env_id {env_id} not found in all_env_ids {all_env_ids_list}")
                    position_indices.append(0)  # fallback на первую позицию
            
            position_indices_tensor = torch.tensor(position_indices, device=self.device, dtype=torch.long)
           # print(f"[DEBUG]   env_ids: {env_ids_list}, position_indices: {position_indices}")
            
            try:
                selected_orientations = orientations[position_indices_tensor]
               # print(f"[DEBUG]   Success: selected_orientations shape: {selected_orientations.shape}")
            except Exception as e:
                print(f"[ERROR] Failed to index orientations: {e}")
                print(f"[ERROR]   orientations shape: {orientations.shape}")
                print(f"[ERROR]   position_indices: {position_indices}")
                print(f"[ERROR]   position_indices_tensor: {position_indices_tensor}")
                # Fallback
                selected_orientations = orientations[0].unsqueeze(0).expand(E)
        
       # print(f"[DEBUG] Final selected_orientations: shape={selected_orientations.shape}, values={selected_orientations}")
        
        # Проверяем размерность
        if selected_orientations.shape[0] != E:
            print(f"[ERROR] Size mismatch: selected_orientations {selected_orientations.shape} != E={E}")
            print(f"[ERROR] Fixing: using first orientation for all")
            selected_orientations = selected_orientations[0].unsqueeze(0).expand(E)
        
        # Остальной код без изменений...
        # Правильный способ с сохранением размерностей
        cos_vals = torch.cos(selected_orientations).view(E, 1)  # [E, 1]
        sin_vals = torch.sin(selected_orientations).view(E, 1)  # [E, 1]
       # print(f"[DEBUG] cos_vals: shape={cos_vals.shape}, values={cos_vals}")
       # print(f"[DEBUG] sin_vals: shape={sin_vals.shape}, values={sin_vals}")

        # Поворачиваем координаты XY, Z оставляем без изменений
        x = selected_positions[:, :, 0]  # [E, KK]
        y = selected_positions[:, :, 1]  # [E, KK]
        z = selected_positions[:, :, 2]  # [E, KK]
       # print(f"[DEBUG] x shape: {x.shape}, values:\n{x}")
       # print(f"[DEBUG] y shape: {y.shape}, values:\n{y}")
       # print(f"[DEBUG] z shape: {z.shape}, values:\n{z}")
        
        # Broadcasting: [E, 1] * [E, KK] = [E, KK]
        x_rot = x * cos_vals - y * sin_vals
        y_rot = x * sin_vals + y * cos_vals
       # print(f"[DEBUG] x_rot shape: {x_rot.shape}, values:\n{x_rot}")
       # print(f"[DEBUG] y_rot shape: {y_rot.shape}, values:\n{y_rot}")
        
        # Собираем обратно
        selected_positions = torch.stack([x_rot, y_rot, z], dim=2)  # [E, KK, 3]
       # print(f"[DEBUG] Final selected_positions shape: {selected_positions.shape}")
       # print(f"[DEBUG] Final selected_positions sample: {selected_positions[0] if E > 0 else 'empty'}")
        # === КОНЕЦ ДОБАВЛЕННОГО КОДА ===

        # индексация env × obj, попарно
        env_idx_tensor = env_ids.view(-1, 1).expand(E, KK)
       # print(f"[DEBUG] env_idx_tensor shape: {env_idx_tensor.shape}, values:\n{env_idx_tensor}")

        # берём только первые KK объектов
        obj_idx_slice = obj_indices_to_place[:, :KK]
       # print(f"[DEBUG] obj_idx_slice shape: {obj_idx_slice.shape}, values:\n{obj_idx_slice}")

        # Проверяем индексацию перед записью
       # print(f"[DEBUG] Writing to positions at indices:")
       # print(f"[DEBUG]   env indices: {env_idx_tensor.flatten().tolist()}")
       # print(f"[DEBUG]   obj indices: {obj_idx_slice.flatten().tolist()}")
        
        try:
            scene_data['positions'][env_idx_tensor, obj_idx_slice] = selected_positions
            scene_data['active'][env_idx_tensor, obj_idx_slice] = True
            scene_data['on_surface_idx'][env_idx_tensor, obj_idx_slice] = -1
            scene_data['surface_level'][env_idx_tensor, obj_idx_slice] = 0
           # print(f"[DEBUG] Successfully wrote to scene_data")
        except Exception as e:
            print(f"[ERROR] Failed to write to scene_data: {e}")
            print(f"[ERROR]   positions shape: {scene_data['positions'].shape}")
            print(f"[ERROR]   active shape: {scene_data['active'].shape}")
            print(f"[ERROR]   env_idx_tensor max: {env_idx_tensor.max().item()}")
            print(f"[ERROR]   obj_idx_slice max: {obj_idx_slice.max().item()}")
            raise
        
       # print(f"[DEBUG] ====== EXIT apply ======")