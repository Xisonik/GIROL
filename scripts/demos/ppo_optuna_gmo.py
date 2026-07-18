"""
═══════════════════════════════════════════════════════════════════════════════
OPTUNA HYPERPARAMETER TUNING ДЛЯ PPO + ORIENTATION MODULE
АДАПТИРОВАНО ДЛЯ ISAAC LAB + SKRL
═══════════════════════════════════════════════════════════════════════════════

ВАЖНО: Isaac Lab НЕ имеет встроенной поддержки Optuna!
Этот скрипт делает интеграцию вручную.

КЛЮЧЕВЫЕ ОТЛИЧИЯ ОТ СТАНДАРТНОГО ИСПОЛЬЗОВАНИЯ:
1. Мы НЕ используем trainer.train() - делаем custom loop
2. Environment создаётся заново для каждого trial
3. После каждого trial делаем cleanup (env.close(), torch.cuda.empty_cache())

ПРОВЕРЕНО С:
- Isaac Lab 1.2+
- skrl 1.3+
- Optuna 3.5+
═══════════════════════════════════════════════════════════════════════════════
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
from pathlib import Path
import json
import sys
import gc
print("AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA")
# ═════════════════════════════════════════════════════════════════════════════
# ПУТЬ К ТВОЕМУ КОДУ
# ═════════════════════════════════════════════════════════════════════════════
# ВАЖНО: Укажи правильный путь где лежит torch_ant_ppo_gmo.py
sys.path.insert(0, '/home/xiso/IsaacLab/scripts/demos')

# Импорты из Isaac Lab / skrl
from skrl.envs.loaders.torch import load_isaaclab_env
from skrl.envs.wrappers.torch import wrap_env
from skrl.memories.torch import RandomMemory
from skrl.utils import set_seed

# Импортируем твои модули
try:
    from torch_ant_ppo_gmo import (
        SharedGraphModule,
        OrientationModule,
        Policy,
        Value,
        DictRunningStandardScaler,
        NUM_GRAPH_NODES,
        PER_OBJECT_DIM,
        TEXT_EMB_DIM,
        GRAPH_EMB_DIM,
        ORIENTATION_EMB_DIM,
    )
    
    # Используем стандартный PPO из skrl (как в оригинальном файле)
    from skrl.agents.torch.ppo import PPO as PPO_WithAuxiliary, PPO_DEFAULT_CONFIG
    
    print(" Все модули импортированы успешно!")
    
except ImportError as e:
    print(f" ОШИБКА ИМПОРТА: {e}")
    print(f"\nПроверь:")
    print(f"  1. Путь sys.path.insert(0, '...') указывает на директорию с torch_ant_ppo_gmo.py")
    print(f"  2. Файл torch_ant_ppo_gmo.py существует и содержит нужные классы")
    sys.exit(1)


# ═════════════════════════════════════════════════════════════════════════════
# ⚙️  CONFIG: НАСТРОЙ ЗДЕСЬ ВСЕ ПАРАМЕТРЫ
# ═════════════════════════════════════════════════════════════════════════════

# ─────────────────────────────────────────────────────────────────────────────
# ЭКСПЕРИМЕНТ
# ─────────────────────────────────────────────────────────────────────────────
STUDY_NAME = "ppo_orientation_quick"      # Имя эксперимента
N_TRIALS = 3                             # Количество trials (начни с малого!)
TUNING_TIMESTEPS = 500                  # Длительность trial (~15 мин)

# ─────────────────────────────────────────────────────────────────────────────
# ОКРУЖЕНИЕ
# ─────────────────────────────────────────────────────────────────────────────
TASK_NAME = "Isaac-Aloha-Direct-v0"
NUM_ENVS = 16                             # Уменьшил с 32 для скорости
EMBEDDINGS_PATH = "/home/xiso/IsaacLab/source/isaaclab_tasks/isaaclab_tasks/direct/aloha/text_embeddings.pt"

# ─────────────────────────────────────────────────────────────────────────────
# ФИКСИРОВАННЫЕ ПАРАМЕТРЫ PPO
# ─────────────────────────────────────────────────────────────────────────────
ROLLOUTS = 48
DISCOUNT_FACTOR = 0.99
LAMBDA = 0.95
RATIO_CLIP = 0.2
VALUE_CLIP = 0.2
CLIP_PREDICTED_VALUES = True

# ─────────────────────────────────────────────────────────────────────────────
# OPTUNA PRUNING
# ─────────────────────────────────────────────────────────────────────────────
PRUNING_ENABLED = True
PRUNING_STARTUP_TRIALS = 2                # Первые 2 trial без pruning
PRUNING_WARMUP_STEPS = 100               # Первые 6k steps без pruning
PRUNING_INTERVAL_STEPS = 50             # Проверять каждые 3k steps

# ─────────────────────────────────────────────────────────────────────────────
# REPORTING
# ─────────────────────────────────────────────────────────────────────────────
REPORT_INTERVAL = 50                    # Сообщать метрики каждые 3k steps

# ─────────────────────────────────────────────────────────────────────────────
# OBJECTIVE FUNCTION
# ─────────────────────────────────────────────────────────────────────────────
REWARD_WEIGHT = 1.0
ORIENT_ACC_WEIGHT = 100.0
ORIENT_LOSS_WEIGHT_OBJECTIVE = 10.0

# ─────────────────────────────────────────────────────────────────────────────
# HYPERPARAMETER SEARCH SPACE
# ─────────────────────────────────────────────────────────────────────────────
LR_MIN = 1e-4
LR_MAX = 1e-3
ENTROPY_MIN = 0.005                       # Увеличил минимум для exploration
ENTROPY_MAX = 0.1
VALUE_LOSS_SCALE_MIN = 0.3
VALUE_LOSS_SCALE_MAX = 1.0
ORIENT_LOSS_WEIGHT_MIN = 0.01
ORIENT_LOSS_WEIGHT_MAX = 0.15             # Уменьшил максимум
ORIENT_DROPOUT_MIN = 0.0
ORIENT_DROPOUT_MAX = 0.25
ORIENT_HIDDEN_CHOICES = [128, 256, 512]
LEARNING_EPOCHS_MIN = 3
LEARNING_EPOCHS_MAX = 8                   # Уменьшил максимум
MINI_BATCHES_CHOICES = [4, 8, 16]
GRAD_NORM_CLIP_MIN = 0.3
GRAD_NORM_CLIP_MAX = 1.0

# ═════════════════════════════════════════════════════════════════════════════
# КОНЕЦ CONFIG
# ═════════════════════════════════════════════════════════════════════════════


def objective(trial: optuna.Trial) -> float:
    """Objective function с ПРАВИЛЬНОЙ интеграцией Isaac Lab + skrl"""
    
    # Гиперпараметры
    learning_rate = trial.suggest_float("learning_rate", LR_MIN, LR_MAX, log=True)
    entropy_scale = trial.suggest_float("entropy_loss_scale", ENTROPY_MIN, ENTROPY_MAX, log=True)
    value_loss_scale = trial.suggest_float("value_loss_scale", VALUE_LOSS_SCALE_MIN, VALUE_LOSS_SCALE_MAX)
    orientation_loss_weight = trial.suggest_float("orientation_loss_weight", ORIENT_LOSS_WEIGHT_MIN, ORIENT_LOSS_WEIGHT_MAX, log=True)
    orientation_dropout = trial.suggest_float("orientation_dropout", ORIENT_DROPOUT_MIN, ORIENT_DROPOUT_MAX)
    orientation_hidden = trial.suggest_categorical("orientation_hidden", ORIENT_HIDDEN_CHOICES)
    learning_epochs = trial.suggest_int("learning_epochs", LEARNING_EPOCHS_MIN, LEARNING_EPOCHS_MAX)
    mini_batches = trial.suggest_categorical("mini_batches", MINI_BATCHES_CHOICES)
    grad_norm_clip = trial.suggest_float("grad_norm_clip", GRAD_NORM_CLIP_MIN, GRAD_NORM_CLIP_MAX)
    
    trial_name = f"trial_{trial.number}"
    print(f"\n{'='*60}")
    print(f"TRIAL {trial.number}")
    print(f"{'='*60}")
    print(f"  learning_rate: {learning_rate:.2e}")
    print(f"  entropy_scale: {entropy_scale:.3f}")
    print(f"  value_loss_scale: {value_loss_scale:.3f}")
    print(f"  orientation_loss_weight: {orientation_loss_weight:.3f}")
    print(f"  orientation_dropout: {orientation_dropout:.2f}")
    print(f"  orientation_hidden: {orientation_hidden}")
    print(f"  learning_epochs: {learning_epochs}")
    print(f"  mini_batches: {mini_batches}")
    print(f"  grad_norm_clip: {grad_norm_clip:.2f}")
    print(f"{'='*60}\n")
    
    # Переменные для cleanup
    env = None
    agent = None
    models = None
    shared_graph = None
    orientation_module = None
    memory = None
    
    try:
        # ─────────────────────────────────────────────────────────────────
        # SETUP
        # ─────────────────────────────────────────────────────────────────
        set_seed(42 + trial.number)
        
        # Environment
        print(f"[Trial {trial.number}] Creating environment...")
        env = load_isaaclab_env(
            task_name=TASK_NAME,
            num_envs=NUM_ENVS,
            headless=True,
            cli_args=["--enable_cameras"],
        )
        env = wrap_env(env)
        device = env.device
        print(f"[Trial {trial.number}] Environment created on {device}")
        
        # Memory
        memory = RandomMemory(memory_size=ROLLOUTS, num_envs=env.num_envs, device=device)
        
        # Shared modules
        print(f"[Trial {trial.number}] Creating models...")
        shared_graph = SharedGraphModule(
            embeddings_path=EMBEDDINGS_PATH,
            num_nodes=NUM_GRAPH_NODES,
            per_object_dim=PER_OBJECT_DIM,
            text_dim=TEXT_EMB_DIM,
        ).to(device)
        
        orientation_module = OrientationModule(
            img_dim=env.observation_space["img"].shape[0],
            graph_emb_dim=GRAPH_EMB_DIM,
            num_bins=36,
            emb_dim=ORIENTATION_EMB_DIM,
            device=device
        )
        
        # Модифицируем architecture
        orientation_module.orientation_predictor = nn.Sequential(
            nn.Linear(env.observation_space["img"].shape[0] + GRAPH_EMB_DIM, orientation_hidden),
            nn.ReLU(),
            nn.Dropout(orientation_dropout),
            nn.Linear(orientation_hidden, 128),
            nn.ReLU(),
            nn.Linear(128, 36)
        ).to(device)
        
        # Models
        models = {
            "policy": Policy(
                env.observation_space, env.action_space, device,
                shared_graph=shared_graph,
                orientation_module=orientation_module
            ),
            "value": Value(
                env.observation_space, env.action_space, device,
                shared_graph=shared_graph,
                orientation_module=orientation_module,
                train_graph=False,
                train_orientation=True
            ),
        }
        
        # PPO Config
        cfg = PPO_DEFAULT_CONFIG.copy()
        cfg["rollouts"] = ROLLOUTS
        cfg["learning_epochs"] = learning_epochs
        cfg["mini_batches"] = mini_batches
        cfg["discount_factor"] = DISCOUNT_FACTOR
        cfg["lambda"] = LAMBDA
        cfg["learning_rate"] = learning_rate
        cfg["learning_rate_scheduler"] = None
        cfg["ratio_clip"] = RATIO_CLIP
        cfg["value_clip"] = VALUE_CLIP
        cfg["clip_predicted_values"] = CLIP_PREDICTED_VALUES
        cfg["entropy_loss_scale"] = entropy_scale
        cfg["value_loss_scale"] = value_loss_scale
        cfg["grad_norm_clip"] = grad_norm_clip
        cfg["orientation_loss_weight"] = orientation_loss_weight
        
        cfg["state_preprocessor"] = DictRunningStandardScaler
        cfg["state_preprocessor_kwargs"] = {
            "size": env.observation_space,
            "img_space": env.observation_space["img"],
            "device": device,
        }
        
        cfg["experiment"]["write_interval"] = 999999  # Отключаем логирование для скорости
        cfg["experiment"]["checkpoint_interval"] = 999999
        cfg["experiment"]["directory"] = f"logs/optuna/{STUDY_NAME}/{trial_name}"
        
        # Agent
        agent = PPO_WithAuxiliary(
            models=models,
            memory=memory,
            cfg=cfg,
            observation_space=env.observation_space,
            action_space=env.action_space,
            device=device,
        )
        
        # Initialize agent (ВАЖНО!)
        agent.init(trainer_cfg={"timesteps": TUNING_TIMESTEPS, "headless": True})
        agent.set_running_mode("train")
        print(f"[Trial {trial.number}] Agent initialized, starting training...")
        
        # ─────────────────────────────────────────────────────────────────
        # TRAINING LOOP (РУЧНОЙ, БЕЗ TRAINER)
        # ─────────────────────────────────────────────────────────────────
        best_objective = -float('inf')
        
        # Reset environment
        states, infos = env.reset()
        
        for timestep in range(TUNING_TIMESTEPS):
            # Pre-interaction
            agent.pre_interaction(timestep=timestep, timesteps=TUNING_TIMESTEPS)
            
            # Act
            with torch.no_grad():
                actions, _, _ = agent.act(states, timestep=timestep, timesteps=TUNING_TIMESTEPS)
            
            # Step environment
            next_states, rewards, terminated, truncated, infos = env.step(actions)
            
            # Record transition
            with torch.no_grad():
                agent.record_transition(
                    states=states,
                    actions=actions,
                    rewards=rewards,
                    next_states=next_states,
                    terminated=terminated,
                    truncated=truncated,
                    infos=infos,
                    timestep=timestep,
                    timesteps=TUNING_TIMESTEPS,
                )
            
            states = next_states
            
            # Post-interaction (может делать update)
            agent.post_interaction(timestep=timestep, timesteps=TUNING_TIMESTEPS)
            
        # Report для pruning
        if timestep > 0 and timestep % REPORT_INTERVAL == 0:
            tracking_data = agent.tracking_data
            
            # Конвертируем в float (может быть тензор, список и т.д.)
            def to_float(value, default=0.0):
                if value is None:
                    return default
                if isinstance(value, torch.Tensor):
                    return value.item()
                if isinstance(value, (list, tuple)):
                    return float(value[0]) if len(value) > 0 else default
                return float(value)
            
            current_reward = to_float(
                tracking_data.get("Reward / Total reward (mean)", -1000.0),
                -1000.0
            )
            current_orient_acc = to_float(
                tracking_data.get("Localization / Orientation Accuracy", 0.0),
                0.0
            )
            current_orient_loss = to_float(
                tracking_data.get("Localization / Orientation Loss", 1.0),
                1.0
            )
            
            intermediate_value = (
                current_reward * REWARD_WEIGHT 
                + current_orient_acc * ORIENT_ACC_WEIGHT 
                - current_orient_loss * ORIENT_LOSS_WEIGHT_OBJECTIVE
            )
            
            if intermediate_value > best_objective:
                best_objective = intermediate_value
            
            # Report для Optuna
            trial.report(intermediate_value, timestep)
            
            print(f"[Trial {trial.number}] Step {timestep}/{TUNING_TIMESTEPS}")
            print(f"  Reward: {current_reward:.2f}")
            print(f"  Orient Acc: {current_orient_acc:.3f}")
            print(f"  Orient Loss: {current_orient_loss:.3f}")
            print(f"  Objective: {intermediate_value:.2f}")
        
        # ─────────────────────────────────────────────────────────────────
        # FINAL METRICS
        # ─────────────────────────────────────────────────────────────────
        tracking_data = agent.tracking_data
        final_reward = tracking_data.get("Reward / Total reward (mean)", -1000.0)
        final_orient_acc = tracking_data.get("Localization / Orientation Accuracy", 0.0)
        final_orient_loss = tracking_data.get("Localization / Orientation Loss", 1.0)
        
        final_objective = (
            final_reward * REWARD_WEIGHT 
            + final_orient_acc * ORIENT_ACC_WEIGHT 
            - final_orient_loss * ORIENT_LOSS_WEIGHT_OBJECTIVE
        )
        
        print(f"\n{'='*60}")
        print(f"TRIAL {trial.number} FINISHED")
        print(f"{'='*60}")
        print(f"  Final Reward: {final_reward:.2f}")
        print(f"  Final Orient Acc: {final_orient_acc:.3f}")
        print(f"  Final Orient Loss: {final_orient_loss:.3f}")
        print(f"  Final Objective: {final_objective:.2f}")
        print(f"{'='*60}\n")
        
        return final_objective
        
    except optuna.TrialPruned:
        print(f"[Trial {trial.number}] Pruned!")
        raise
        
    except Exception as e:
        print(f"[Trial {trial.number}] FAILED with error: {e}")
        import traceback
        traceback.print_exc()
        return -float('inf')
        
    finally:
        # ─────────────────────────────────────────────────────────────────
        # CLEANUP (КРИТИЧНО!)
        # ─────────────────────────────────────────────────────────────────
        print(f"[Trial {trial.number}] Cleaning up...")
        
        if env is not None:
            try:
                env.close()
            except:
                pass
        
        # Удаляем все объекты
        del agent, models, shared_graph, orientation_module, memory, env
        
        # Force garbage collection
        gc.collect()
        torch.cuda.empty_cache()
        
        print(f"[Trial {trial.number}] Cleanup complete\n")


def main():
    print(f"\n{'='*70}")
    print("OPTUNA HYPERPARAMETER TUNING ДЛЯ PPO + ORIENTATION")
    print("ISAAC LAB + SKRL INTEGRATION")
    print(f"{'='*70}")
    print(f"  Study name: {STUDY_NAME}")
    print(f"  Trials: {N_TRIALS}")
    print(f"  Timesteps per trial: {TUNING_TIMESTEPS}")
    print(f"  Num envs: {NUM_ENVS}")
    print(f"  Pruning: {'Enabled' if PRUNING_ENABLED else 'Disabled'}")
    print(f"{'='*70}\n")
    
    # Storage
    storage = f"sqlite:///{STUDY_NAME}.db"
    print(f"Storage: {storage}\n")
    
    # Create study
    pruner = MedianPruner(
        n_startup_trials=PRUNING_STARTUP_TRIALS,
        n_warmup_steps=PRUNING_WARMUP_STEPS,
        interval_steps=PRUNING_INTERVAL_STEPS,
    ) if PRUNING_ENABLED else None
    
    study = optuna.create_study(
        study_name=STUDY_NAME,
        storage=storage,
        direction="maximize",
        sampler=TPESampler(seed=42),
        pruner=pruner,
        load_if_exists=True,
    )
    
    if len(study.trials) > 0:
        print(f"✓ Resuming from {len(study.trials)} completed trials")
        print(f"  Best value so far: {study.best_value:.2f}\n")
    
    print("Starting optimization...\n")
    
    # Optimize
    study.optimize(
        objective,
        n_trials=N_TRIALS,
        n_jobs=1,  # ВСЕГДА 1 для GPU!
        show_progress_bar=True,
    )
    
    # Results
    print(f"\n{'='*70}")
    print("OPTIMIZATION COMPLETE!")
    print(f"{'='*70}")
    
    complete_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    pruned_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]
    failed_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.FAIL]
    
    print(f"Trials: {len(study.trials)} total")
    print(f"  Complete: {len(complete_trials)}")
    print(f"  Pruned: {len(pruned_trials)}")
    print(f"  Failed: {len(failed_trials)}")
    
    if len(complete_trials) == 0:
        print("\n⚠️  No complete trials!")
        return
    
    # Best trial
    best = study.best_trial
    print(f"\nBEST TRIAL #{best.number}:")
    print(f"  Objective: {best.value:.2f}")
    print(f"\n  Parameters:")
    for key, value in best.params.items():
        if isinstance(value, float):
            print(f"    {key}: {value:.4f}")
        else:
            print(f"    {key}: {value}")
    
    # Save results
    results_dir = Path(f"logs/optuna/{STUDY_NAME}")
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Best params (JSON)
    with open(results_dir / "best_params.json", "w") as f:
        json.dump(best.params, f, indent=2)
    
    # Best params (Python)
    with open(results_dir / "best_params.py", "w") as f:
        f.write("# Best hyperparameters from Optuna\n")
        f.write("# Copy to torch_ant_ppo_gmo.py\n\n")
        f.write("BEST_PARAMS = {\n")
        for key, value in best.params.items():
            if isinstance(value, str):
                f.write(f'    "{key}": "{value}",\n')
            else:
                f.write(f'    "{key}": {value},\n')
        f.write("}\n")
    
    print(f"\n{'='*70}")
    print("RESULTS SAVED:")
    print(f"  {results_dir}/best_params.json")
    print(f"  {results_dir}/best_params.py")
    
    # Visualization
    try:
        import plotly
        fig = optuna.visualization.plot_optimization_history(study)
        fig.write_html(str(results_dir / "optimization_history.html"))
        print(f"  {results_dir}/optimization_history.html")
    except:
        print("\n💡 Install plotly for visualizations: pip install plotly")
    
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
