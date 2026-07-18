import torch
import torch.nn as nn

# skrl / Isaac Lab imports
from skrl.agents.torch.ppo import PPO, PPO_DEFAULT_CONFIG
from skrl.envs.loaders.torch import load_isaaclab_env
from skrl.envs.wrappers.torch import wrap_env
from skrl.memories.torch import RandomMemory
from skrl.models.torch import DeterministicMixin, GaussianMixin, Model
from skrl.resources.preprocessors.torch import RunningStandardScaler
from skrl.trainers.torch import SequentialTrainer
from skrl.utils import set_seed

# ---------------------------------------------------------------------
# Режим: train / eval
# ---------------------------------------------------------------------
EVAL = False      # ← тут переключаешь на True, когда хочешь инференс + видео
# EVAL = True
# seed for reproducibility
set_seed(42)

# ---------------------------------------------------------------------
# Модели
# ---------------------------------------------------------------------
class StochasticActor(GaussianMixin, Model):
    def __init__(self, observation_space, action_space, device, clip_actions=False,
                 clip_log_std=True, min_log_std=-5, max_log_std=2):
        Model.__init__(self, observation_space, action_space, device)
        GaussianMixin.__init__(self, clip_actions, clip_log_std, min_log_std, max_log_std)

        self.net = nn.Sequential(
            nn.Linear(self.num_observations, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, self.num_actions),
            nn.Tanh()
        )
        self.log_std_parameter = nn.Parameter(torch.zeros(self.num_actions))

    def compute(self, inputs, role):
        # PPO ожидает: действия + лог-std
        return self.net(inputs["states"]), self.log_std_parameter, {}


class Value(DeterministicMixin, Model):
    def __init__(self, observation_space, action_space, device, clip_actions=False):
        Model.__init__(self, observation_space, action_space, device)
        DeterministicMixin.__init__(self, clip_actions)

        self.net = nn.Sequential(
            nn.Linear(self.num_observations, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )

    def compute(self, inputs, role):
        # PPO ожидает скаляр value V(s)
        return self.net(inputs["states"]), {}


# ---------------------------------------------------------------------
# Environment: train vs eval
# ---------------------------------------------------------------------
if EVAL:
    from gymnasium.wrappers import RecordVideo

    print("[INFO] Running evaluation (PPO)...")
    env = load_isaaclab_env(
        task_name="Isaac-Aloha-Direct-v0",
        num_envs=1,
        headless=True,          # offscreen + видео
        cli_args=[
            "--enable_cameras",
            "--video",          # IsaacLab включит offscreen-рендер
        ],
    )

    # пишем видео вручную через gymnasium
    env = RecordVideo(
        env,
        video_folder="logs/skrl/aloha/videos",  # куда писать mp4
        name_prefix="aloha_eval_ppo",
        episode_trigger=lambda ep: True,        # писать каждую попытку
    )
else:
    print("[INFO] Running training (PPO)...")
    env = load_isaaclab_env(
        task_name="Isaac-Aloha-Direct-v0",
        num_envs=32,
        headless=True,
        cli_args=[
            "--enable_cameras" # камеры и видео не включаем — для скорости
        ],
    )

# skrl-обёртка
env = wrap_env(env)
device = env.device

# ---------------------------------------------------------------------
# Конфиг PPO
# ---------------------------------------------------------------------
cfg = PPO_DEFAULT_CONFIG.copy()
# базовые гиперы (можешь потом подкрутить под задачу)
cfg["rollouts"] = 32              # столько шагов *до* апдейта
cfg["learning_epochs"] = 5        # эпох на один апдейт
cfg["mini_batches"] = 4           # минибатчей на эпоху

cfg["discount_factor"] = 0.99
cfg["lambda"] = 0.95
cfg["learning_rate"] = 3e-4       # и для policy, и для value
cfg["random_timesteps"] = 0
cfg["learning_starts"] = 0
cfg["grad_norm_clip"] = 1.0
cfg["ratio_clip"] = 0.2
cfg["value_clip"] = 0.2
cfg["clip_predicted_values"] = True
cfg["entropy_loss_scale"] = 0.0   # можно поднять до 0.01–0.02
cfg["value_loss_scale"] = 1.0
cfg["kl_threshold"] = 0.0

# нормализация состояний (как у тебя было)
cfg["state_preprocessor"] = RunningStandardScaler
cfg["state_preprocessor_kwargs"] = {"size": env.observation_space, "device": device}
# можно также нормализовать value, но не обязательно
# cfg["value_preprocessor"] = RunningStandardScaler
# cfg["value_preprocessor_kwargs"] = {"size": 1, "device": device}

# логи и чекпоинты
cfg["experiment"]["directory"] = "logs/skrl/aloha_ppo"
cfg["experiment"]["write_interval"] = 200
cfg["experiment"]["checkpoint_interval"] = 1000

# ---------------------------------------------------------------------
# Память (rollout buffer)
# Важно: memory_size == cfg["rollouts"], см. обсуждения skrl + Isaac
# ---------------------------------------------------------------------
memory = RandomMemory(
    memory_size=cfg["rollouts"],
    num_envs=env.num_envs,
    device=device,
)

# ---------------------------------------------------------------------
# Модели агента
# ---------------------------------------------------------------------
models = {
    "policy": StochasticActor(env.observation_space, env.action_space, device),
    "value": Value(env.observation_space, env.action_space, device),
}

agent = PPO(
    models=models,
    memory=memory if not EVAL else None,   # при eval память не нужна
    cfg=cfg,
    observation_space=env.observation_space,
    action_space=env.action_space,
    device=device,
)

# ---------------------------------------------------------------------
# Trainer: train / eval
# ---------------------------------------------------------------------
if not EVAL:
    # -------- TRAINING --------
    cfg_trainer = {"timesteps": 100000}
    trainer = SequentialTrainer(cfg=cfg_trainer, env=env, agents=agent)
    trainer.train()
else:
    # -------- EVALUATION --------
    cfg_trainer = {"timesteps": 1000}
    trainer = SequentialTrainer(cfg=cfg_trainer, env=env, agents=agent)

    # сюда поставь свой реальный чекпоинт PPO
    checkpoint_path = "/home/xiso/IsaacLab/logs/skrl/aloha_ppo/25-12-11_20-25-57-458499_PPO/checkpoints/agent_34000.pt"
    agent.load(checkpoint_path)

    trainer.eval()
