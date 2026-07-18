"""
Конфигурация для hyperparameter tuning Aloha Orientation PPO с помощью Optuna.
Этот файл определяет пространство поиска гиперпараметров и настройки оптимизации.
"""

from ray import tune
from ray.tune.search.optuna import OptunaSearch
from ray.tune.schedulers import ASHAScheduler

class AlohaOrientationJobCfg:
    """
    Конфигурация для оптимизации гиперпараметров Aloha Orientation PPO.
    
    Использует:
    - Optuna для интеллектуального поиска (Bayesian optimization)
    - ASHA scheduler для early stopping плохих trials
    """
    
    def __init__(self):
        # ============= БАЗОВАЯ ИНФОРМАЦИЯ =============
        self.name = "AlohaOrientation"  # Имя эксперимента для MLFlow
        
        # ============= ПАРАМЕТРЫ ОКРУЖЕНИЯ =============
        self.task_name = "Isaac-Aloha-Direct-v0"
        self.num_envs = 32  # Количество параллельных окружений
        self.headless = True  # Без GUI для скорости
        self.enable_cameras = True  # Камеры нужны для img observation
        
        # ============= ПРОСТРАНСТВО ПОИСКА ГИПЕРПАРАМЕТРОВ =============
        self.params_to_tune = {
            # --- PPO параметры обучения ---
            "learning_rate": tune.loguniform(1e-4, 5e-4),  # Логарифмическое распределение
            "rollouts": tune.choice([24, 32, 48]),  # Размер буфера памяти
            "mini_batches": tune.choice([4, 8]),  # Количество мини-батчей
            "learning_epochs": tune.choice([3, 5, 8]),  # Эпох обучения на rollout
            
            # --- Веса loss-функций ---
            "entropy_loss_scale": tune.uniform(0.01, 0.08),  # Exploration
            "value_loss_scale": tune.uniform(0.3, 0.7),  # Вес value loss
            "orientation_loss_weight": tune.loguniform(0.005, 0.05),  # КЛЮЧЕВОЙ параметр
            
            # --- Архитектурные параметры (Orientation Module) ---
            "orientation_num_bins": tune.choice([18, 36, 72]),  # Количество бинов угла
            "orientation_emb_dim": tune.choice([16, 32, 64]),  # Размерность embedding
            
            # --- Архитектурные параметры (Graph Encoder) ---
            "graph_hidden_dim": tune.choice([64, 128, 256]),  # Скрытая размерность GNN
            "graph_num_layers": tune.choice([2, 3]),  # Количество слоёв GAT
            "graph_heads": tune.choice([2, 4]),  # Attention heads
        }
        
        # ============= ФИКСИРОВАННЫЕ ПАРАМЕТРЫ =============
        self.fixed_params = {
            # PPO fundamentals
            "discount_factor": 0.99,
            "lambda_gae": 0.95,
            "ratio_clip": 0.2,
            "value_clip": 0.2,
            "grad_norm_clip": 0.5,
            "clip_predicted_values": True,
            
            # Пути и настройки
            "task_name": self.task_name,
            "num_envs": self.num_envs,
            "headless": self.headless,
            "enable_cameras": self.enable_cameras,
            "embeddings_path": "/home/xiso/IsaacLab/source/isaaclab_tasks/isaaclab_tasks/direct/aloha/text_embeddings.pt",
        }
        
        # ============= OPTUNA SEARCH ALGORITHM =============
        self.search_algorithm = OptunaSearch(
            metric="episode_reward_mean",  # Метрика для оптимизации
            mode="max",  # Максимизируем reward
        )
        
        # ============= ASHA SCHEDULER (EARLY STOPPING) =============
        self.scheduler = ASHAScheduler(
            time_attr="training_iteration",  # По итерациям
            metric="episode_reward_mean",  # Метрика для stopping
            mode="max",  # Максимизируем
            max_t=20,  # ТЕСТ: Максимум 20 итераций
            grace_period=5,  # Минимум 5 итераций до остановки
            reduction_factor=2,  # Убираем худшие 50%
        )
        
        # ============= ПАРАМЕТРЫ TUNING RUN =============
        self.num_samples = 5  # ТЕСТ: Всего 5 trials
        self.max_concurrent_trials = 1  # 1 trial на 1 GPU
        
        # ============= ПАРАМЕТРЫ ТРЕНИРОВКИ =============
        self.timesteps = 1000  # ТЕСТ: 1000 timesteps
        self.checkpoint_interval = 500
        self.write_interval = 50
        
        # ============= MLFLOW НАСТРОЙКИ =============
        self.mlflow_experiment_name = f"IsaacRay-{self.name}-tune"
        
        # ============= ВАЖНО: runner_args ДОЛЖЕН БЫТЬ DICT! =============
        # Формат: {флаг_командной_строки: ключ_в_config}
        self.runner_args = {
            # Тюнимые параметры
            "--learning_rate": "learning_rate",
            "--rollouts": "rollouts",
            "--mini_batches": "mini_batches",
            "--learning_epochs": "learning_epochs",
            "--entropy_loss_scale": "entropy_loss_scale",
            "--value_loss_scale": "value_loss_scale",
            "--orientation_loss_weight": "orientation_loss_weight",
            "--orientation_num_bins": "orientation_num_bins",
            "--orientation_emb_dim": "orientation_emb_dim",
            "--graph_hidden_dim": "graph_hidden_dim",
            "--graph_num_layers": "graph_num_layers",
            "--graph_heads": "graph_heads",
            
            # Фиксированные параметры
            "--timesteps": "timesteps",
        }
        
        # ============= СОЗДАЁМ self.cfg (требуется tuner.py) =============
        self.cfg = {
            "name": self.name,
            "task_name": self.task_name,
            "num_envs": self.num_envs,
            "headless": self.headless,
            "enable_cameras": self.enable_cameras,
            "params_to_tune": self.params_to_tune,
            "fixed_params": self.fixed_params,
            "search_algorithm": self.search_algorithm,
            "scheduler": self.scheduler,
            "num_samples": self.num_samples,
            "max_concurrent_trials": self.max_concurrent_trials,
            "timesteps": self.timesteps,
            "checkpoint_interval": self.checkpoint_interval,
            "write_interval": self.write_interval,
            "mlflow_experiment_name": self.mlflow_experiment_name,
            "runner_args": self.runner_args,  # <- DICT, не list!
        }
    
    def get_config_dict(self, trial_params):
        """
        Объединяет параметры trial с фиксированными параметрами.
        
        Args:
            trial_params: dict с гиперпараметрами от Optuna
            
        Returns:
            dict: полная конфигурация для тренировки
        """
        config = self.fixed_params.copy()
        config.update(trial_params)
        config["timesteps"] = self.timesteps
        config["checkpoint_interval"] = self.checkpoint_interval
        config["write_interval"] = self.write_interval
        config["runner_args"] = self.runner_args
        return config