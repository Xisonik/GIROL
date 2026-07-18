import torch
from collections import defaultdict
from typing import Dict, List, Optional
import json


class SceneHistoryTracker:
    """
    Отслеживает историю позиций объектов за последние N шагов.
    При обнаружении NaN остановит выполнение и выведет историю.
    """
    
    def __init__(self, num_envs: int, num_objects: int, max_history: int = 100, device: str = "cpu"):
        self.num_envs = num_envs
        self.num_objects = num_objects
        self.max_history = max_history
        self.device = device
        
        # История: {object_name: [[env_0_pos_3d], ...]}
        self.history: Dict[str, List[torch.Tensor]] = defaultdict(list)
        
        self.step_numbers: List[int] = []
        self.object_names: List[str] = []
        self.current_step = 0
        self.active_history: Dict[int, torch.Tensor] = {}
    
    
    def record_step(self, positions: torch.Tensor, names: List[str], 
                    active: Optional[torch.Tensor] = None, step_number: Optional[int] = None):
        """
        Записывает позиции объектов на текущем шаге.
        
        Args:
            positions: [num_envs, num_objects, 3]
            names: Имена объектов
            active: [num_envs, num_objects] маска активности
            step_number: Номер шага
        """
        if step_number is None:
            step_number = self.current_step
        
        self.step_numbers.append(step_number)
        self.object_names = names
        
        for obj_idx, obj_name in enumerate(names):
            obj_pos = positions[:, obj_idx, :].detach().cpu()
            self.history[obj_name].append(obj_pos)
            
            if len(self.history[obj_name]) > self.max_history:
                self.history[obj_name].pop(0)
        
        if active is not None:
            self.active_history[step_number] = active.detach().cpu()
        
        self.current_step += 1
    
    
    def get_object_trajectory(self, object_name: str, env_id: int, last_n: Optional[int] = None) -> List[dict]:
        """Получить траекторию объекта за последние N шагов."""
        if object_name not in self.history:
            return [{"error": f"Object '{object_name}' not found"}]
        
        trajectory = []
        positions_list = self.history[object_name]
        
        if last_n is not None:
            positions_list = positions_list[-last_n:]
        
        start_idx = max(0, len(self.history[object_name]) - len(positions_list))
        
        for i, pos in enumerate(positions_list):
            step_idx = start_idx + i
            if step_idx < len(self.step_numbers):
                step_num = self.step_numbers[step_idx]
            else:
                step_num = -1
            
            if env_id < len(pos):
                x, y, z = pos[env_id].tolist()
                has_nan = (x != x) or (y != y) or (z != z)
                
                trajectory.append({
                    "step": int(step_num),
                    "x": float(x),
                    "y": float(y),
                    "z": float(z),
                    "has_nan": has_nan
                })
        
        return trajectory
    
    
    def find_first_nan_step(self) -> Optional[tuple]:
        """Найти первый шаг с NaN."""
        for obj_name, positions_list in self.history.items():
            for step_idx, pos in enumerate(positions_list):
                if torch.isnan(pos).any():
                    step_num = self.step_numbers[step_idx] if step_idx < len(self.step_numbers) else -1
                    env_ids_with_nan = torch.where(torch.isnan(pos).any(dim=1))[0]
                    for env_id in env_ids_with_nan.tolist():
                        return (step_num, obj_name, env_id)
        
        return None
    
    
    def print_environment_report(self, env_id: int, last_steps: int = 20):
        """Печать отчета для среды."""
        print(f"\n{'='*80}")
        print(f"SCENE HISTORY - Environment {env_id}")
        print(f"{'='*80}\n")
        
        print(f"Last {last_steps} steps:\n")
        print(f"{'Object':<20} | {'Step':>6} | {'X':>8} | {'Y':>8} | {'Z':>8} | {'Status':>6}")
        print("-" * 80)
        
        for obj_name in self.object_names:
            trajectory = self.get_object_trajectory(obj_name, env_id, last_n=last_steps)
            
            for entry in trajectory:
                if "error" in entry:
                    print(f"{obj_name:<20} | {entry['error']}")
                else:
                    status = "NaN!" if entry["has_nan"] else "OK"
                    print(f"{obj_name:<20} | {entry['step']:>6} | {entry['x']:>8.3f} | "
                          f"{entry['y']:>8.3f} | {entry['z']:>8.3f} | {status:>6}")
    
    
    def print_critical_analysis(self):
        """Анализ критических событий."""
        print(f"\n{'='*80}")
        print(f"CRITICAL EVENTS ANALYSIS")
        print(f"{'='*80}\n")
        
        nan_event = self.find_first_nan_step()
        if nan_event:
            step_num, obj_name, env_id = nan_event
            print(f"FIRST NaN:")
            print(f"  Step: {step_num}")
            print(f"  Object: {obj_name}")
            print(f"  Environment: {env_id}\n")
            
            print(f"Trajectory of '{obj_name}' in env {env_id}:")
            trajectory = self.get_object_trajectory(obj_name, env_id, last_n=15)
            for entry in trajectory:
                if "error" not in entry:
                    status = "NaN!" if entry["has_nan"] else "OK"
                    print(f"  Step {entry['step']:>5}: ({entry['x']:>7.3f}, {entry['y']:>7.3f}, "
                          f"{entry['z']:>7.3f}) {status}")
        else:
            print("No NaN detected")
    
    
    def emergency_dump(self, output_dir: str = "."):
        """Аварийный дамп истории в JSON."""
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        dump_data = {
            "metadata": {
                "num_envs": self.num_envs,
                "num_objects": self.num_objects,
                "current_step": self.current_step,
            },
            "critical_event": None,
            "trajectories": {}
        }
        
        nan_event = self.find_first_nan_step()
        if nan_event:
            step_num, obj_name, env_id = nan_event
            dump_data["critical_event"] = {
                "step": int(step_num),
                "object": obj_name,
                "environment": int(env_id),
            }
        
        for obj_name in self.object_names:
            dump_data["trajectories"][obj_name] = {}
            for env_id in range(min(self.num_envs, 4)):
                dump_data["trajectories"][obj_name][f"env_{env_id}"] = \
                    self.get_object_trajectory(obj_name, env_id)
        
        filepath = os.path.join(output_dir, "scene_history_dump.json")
        with open(filepath, "w") as f:
            json.dump(dump_data, f, indent=2)
        
        print(f"Emergency dump: {filepath}")
        
        for env_id in range(min(self.num_envs, 2)):
            self.print_environment_report(env_id, last_steps=10)
        
        self.print_critical_analysis()


class NaNDetector:
    """Детектор NaN с логированием."""
    
    def __init__(self, tracker: SceneHistoryTracker, 
                 output_dir: str = "./nan_debug", 
                 raise_on_nan: bool = True):
        self.tracker = tracker
        self.output_dir = output_dir
        self.raise_on_nan = raise_on_nan
    
    def check(self, tensor: torch.Tensor, context: str = "") -> bool:
        """Проверить тензор на NaN/Inf."""
        has_nan = torch.isnan(tensor).any()
        has_inf = torch.isinf(tensor).any()
        
        if has_nan or has_inf:
            print(f"\n{'='*80}")
            print(f"NaN/Inf DETECTED in {context}")
            print(f"{'='*80}")
            print(f"Shape: {tensor.shape}")
            print(f"Has NaN: {has_nan}, Has Inf: {has_inf}")
            print(f"{'='*80}\n")
            
            self.tracker.emergency_dump(self.output_dir)
            
            if self.raise_on_nan:
                raise RuntimeError(f"NaN/Inf in {context}!")
            
            return True
        
        return False