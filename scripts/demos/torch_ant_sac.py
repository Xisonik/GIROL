import argparse
import torch
import torch.nn as nn

# skrl / Isaac Lab imports
from skrl.agents.torch.sac import SAC, SAC_DEFAULT_CONFIG
from skrl.envs.loaders.torch import load_isaaclab_env
from skrl.envs.wrappers.torch import wrap_env
from skrl.memories.torch import RandomMemory
from skrl.models.torch import DeterministicMixin, GaussianMixin, Model
from skrl.resources.preprocessors.torch import RunningStandardScaler
from skrl.trainers.torch import SequentialTrainer
from skrl.utils import set_seed

# ---------------------------------------------------------------------
# CLI аргументы
# ---------------------------------------------------------------------
# EVAL = True
EVAL = False
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
        return self.net(inputs["states"]), self.log_std_parameter, {}


class Critic(DeterministicMixin, Model):
    def __init__(self, observation_space, action_space, device, clip_actions=False):
        Model.__init__(self, observation_space, action_space, device)
        DeterministicMixin.__init__(self, clip_actions)

        self.net = nn.Sequential(
            nn.Linear(self.num_observations + self.num_actions, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )

    def compute(self, inputs, role):
        x = torch.cat([inputs["states"], inputs["taken_actions"]], dim=1)
        return self.net(x), {}


# ---------------------------------------------------------------------
# Environment: train vs eval
# ---------------------------------------------------------------------
if EVAL:
    from gymnasium.wrappers import RecordVideo

    print("[INFO] Running evaluation...")
    # 1 env, камеры включены, запись видео через IsaacLab (--video)
    env = load_isaaclab_env(
        task_name="Isaac-Aloha-Direct-v0",
        num_envs=1,
        headless=True,          # offscreen + видео
        cli_args=[
            "--enable_cameras",
            "--video",
        ],
    )
    env = RecordVideo(
        env,
        video_folder="logs/skrl/aloha/videos",  # куда писать mp4
        name_prefix="aloha_eval",
        episode_trigger=lambda ep: True,       # писать каждую попытку
    )
else:
    print("[INFO] Running training...")
    # Много окружений, без камер для скорости
    env = load_isaaclab_env(
        task_name="Isaac-Aloha-Direct-v0",
        num_envs=32,
        headless=True,
        cli_args=[
            "--enable_cameras", #и --video для максимальной скорости
        ],
    )

env = wrap_env(env)
device = env.device

# ---------------------------------------------------------------------
# Память
# ---------------------------------------------------------------------
memory = RandomMemory(memory_size=8000, num_envs=env.num_envs, device=device)

# ---------------------------------------------------------------------
# Модели агента
# ---------------------------------------------------------------------
models = {
    "policy": StochasticActor(env.observation_space, env.action_space, device),
    "critic_1": Critic(env.observation_space, env.action_space, device),
    "critic_2": Critic(env.observation_space, env.action_space, device),
    "target_critic_1": Critic(env.observation_space, env.action_space, device),
    "target_critic_2": Critic(env.observation_space, env.action_space, device),
}

# ---------------------------------------------------------------------
# Конфиг SAC
# ---------------------------------------------------------------------
cfg = SAC_DEFAULT_CONFIG.copy()
cfg["gradient_steps"] = 4
cfg["batch_size"] = 512
cfg["discount_factor"] = 0.99
cfg["polyak"] = 0.005
cfg["actor_learning_rate"] = 3e-4
cfg["critic_learning_rate"] = 3e-4
cfg["random_timesteps"] = 0
cfg["learning_starts"] = 1000
cfg["grad_norm_clip"] = 0
cfg["learn_entropy"] = True
cfg["entropy_learning_rate"] = 5e-3
cfg["initial_entropy_value"] = 1.0

cfg["state_preprocessor"] = RunningStandardScaler
cfg["state_preprocessor_kwargs"] = {"size": env.observation_space, "device": device}

# логи и чекпоинты
cfg["experiment"]["write_interval"] = 100
cfg["experiment"]["checkpoint_interval"] = 1000
cfg["experiment"]["directory"] = "logs/skrl/aloha_ppo"

agent = SAC(
    models=models,
    memory=memory,
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
    cfg_trainer = {"timesteps": 33000}
    trainer = SequentialTrainer(cfg=cfg_trainer, env=env, agents=agent)
    trainer.train()
else:
    # -------- EVALUATION --------
    cfg_trainer = {"timesteps": 1000}
    trainer = SequentialTrainer(cfg=cfg_trainer, env=env, agents=agent)

    # путь к чекпоинту — подставь свой
    checkpoint_path = "/home/xiso/IsaacLab/logs/skrl/aloha_ppo/25-12-22_16-45-26-425903_SAC/checkpoints/agent_11000.pt"
    agent.load(checkpoint_path)

    trainer.eval()
