# -*- coding: utf-8 -*-
import os
import torch
import torch.nn as nn
import torch.nn.functional as F

from skrl.agents.torch.sac import SAC, SAC_DEFAULT_CONFIG
from skrl.envs.loaders.torch import load_isaaclab_env
from skrl.envs.wrappers.torch import wrap_env
from skrl.memories.torch import RandomMemory
from skrl.resources.preprocessors.torch import RunningStandardScaler
from skrl.trainers.torch import SequentialTrainer
from skrl.utils import set_seed

from networks.networks_orm import *
# 56 42 10 16 38

# ---------------------------------------------------------------------------
# Configuration (all overridable via GIROL_* environment variables so the
# launch scripts can drive a run without editing this file).
# See launch_orientation_baseline.sh and BASELINE.md.
# ---------------------------------------------------------------------------
def _env_flag(name: str, default: str = "0") -> bool:
    return os.environ.get(name, default) == "1"

NAME = os.environ.get("GIROL_RUN_NAME", "bbq")
seed = int(os.environ.get("GIROL_SEED", "42"))
set_seed(seed)

"""
- Пайплайны:
    1. навигации - Aloha_nav

- Для лайвстрима:
    headless=True,
    cli_args=["--enable_cameras", "--video", "--livestream", "2",],
"""
TASK_NAME = os.environ.get("GIROL_TASK", "Aloha_nav")
EVAL = _env_flag("GIROL_EVAL")
VIDEO = _env_flag("GIROL_VIDEO")
USE_PRETRAINED = _env_flag("GIROL_USE_PRETRAINED")
USE_GRAPH = _env_flag("GIROL_USE_GRAPH")            # policy conditions on graph_emb
USE_METRIC = _env_flag("GIROL_USE_METRIC", "1")    # baseline=1 (metric graph); ablation=0
USE_NONMETRIC = _env_flag("GIROL_NONMETRIC")       # 1 = room-aware qualitative encoder (non-metric study)
USE_COMET = _env_flag("GIROL_USE_COMET")           # off by default (no external logging)
num_envs = int(os.environ.get("GIROL_NUM_ENVS", "64"))
timestepslen = int(os.environ.get("GIROL_TIMESTEPS", "100000"))
headless = _env_flag("GIROL_HEADLESS", "1")
LOG_ROOT = os.environ.get("GIROL_LOG_ROOT", "logs/skrl")

if EVAL or VIDEO:
    num_envs = 1
    timestepslen = 500

if VIDEO:
    cli_args = ["--enable_cameras", "--video", "--livestream", "2"]
    from gymnasium.wrappers import RecordVideo
    num_envs = 1
    headless = True
else:
    cli_args = ["--enable_cameras"]

if headless:
    # from gymnasium.wrappers import RecordVideo
    env = load_isaaclab_env(
        task_name=TASK_NAME, 
        headless=headless, 
        num_envs=num_envs,
        cli_args=cli_args
    )
else:
    env = load_isaaclab_env(
        task_name=TASK_NAME, 
        num_envs=num_envs,
        cli_args=cli_args
    )

if VIDEO:
    env = RecordVideo(
        env,
        video_folder="logs/skrl/videos",
        name_prefix="aloha_eval",
        episode_trigger=lambda ep: True,
    )

env = wrap_env(env)
device = env.device

_EMB_PATH = "source/isaaclab_tasks/isaaclab_tasks/direct/aloha_nav/text_embeddings.pt"
if USE_NONMETRIC:
    # Room-aware qualitative (direction-only) encoder for the non-metric study.
    from networks.nonmetric_graph_encoder import NonMetricGraphEncoder
    print("[graph] using NonMetricGraphEncoder (room-aware, direction-only)")
    graph_encoder = NonMetricGraphEncoder(
        embeddings_path=_EMB_PATH, env=env, use_metric=USE_METRIC,
    ).to(device)
else:
    graph_encoder = GraphEncoder(
        embeddings_path=_EMB_PATH,
        graphs_dir=None,          # compact 6-dim path is active; no JSON scene cache needed
        env=env,
        use_metric=USE_METRIC,    # baseline keeps object x,y,z; ablation zeros them
    ).to(device)

orient_module = OrientationModule(
    img_dim=env.observation_space["img"].shape[0],
).to(device)

graph_encoder.eval()
orient_module.eval() # custom trainer turn it to train in train steps

memory_size = 2000
if num_envs == 128:
    memory_size = 1300
elif num_envs == 64:
    memory_size = 2000
elif num_envs == 32:
    memory_size = 5000
print("memory size: ", memory_size)
memory = RandomMemory(memory_size=memory_size, num_envs=env.num_envs, device=device)

models = {
    "policy": StochasticActor(
        env.observation_space, env.action_space, device,
        graph_encoder=graph_encoder, orient_module=orient_module,
        use_graph = USE_GRAPH,
    ),
    "critic_1": Critic(
        env.observation_space, env.action_space, device,
        graph_encoder=graph_encoder, orient_module=orient_module,
        use_graph = USE_GRAPH,
    ),
    "critic_2": Critic(
        env.observation_space, env.action_space, device,
        graph_encoder=graph_encoder, orient_module=orient_module,
        use_graph = USE_GRAPH,
    ),
    "target_critic_1": Critic(
        env.observation_space, env.action_space, device,
        graph_encoder=graph_encoder, orient_module=orient_module,
        use_graph = USE_GRAPH,
    ),
    "target_critic_2": Critic(
        env.observation_space, env.action_space, device,
        graph_encoder=graph_encoder, orient_module=orient_module,
        use_graph = USE_GRAPH,
    ),
}

cfg = SAC_DEFAULT_CONFIG.copy()
cfg["gradient_steps"] = 1
cfg["batch_size"] = 512
cfg["discount_factor"] = 0.99
cfg["polyak"] = 0.005
cfg["actor_learning_rate"] = 3e-4
cfg["critic_learning_rate"] = 3e-4
cfg["random_timesteps"] = 0
cfg["learning_starts"] = 100
cfg["grad_norm_clip"] = 0
cfg["learn_entropy"] = True
cfg["entropy_learning_rate"] = 5e-3
cfg["initial_entropy_value"] = 1.0

cfg["state_preprocessor"] = DictRunningStandardScaler
cfg["state_preprocessor_kwargs"] = {
    "size": env.observation_space,
    "img_space": env.observation_space["img"],
    "device": device,
}
# cfg["state_preprocessor"] = None  
# cfg["state_preprocessor_kwargs"] = {}
# TODO: delete state_preprocessor and make image net image statistics

cfg["experiment"]["write_interval"] = 10
cfg["experiment"]["checkpoint_interval"] = 1000
cfg["experiment"]["directory"] = f"{LOG_ROOT}/aloha_sac"

agent = SAC(
    models=models, memory=memory, cfg=cfg,
    observation_space=env.observation_space,
    action_space=env.action_space, device=device,
)

# Auxiliary trainer + callback
aux_trainer = AuxModuleTrainer(
    graph_encoder=graph_encoder,
    orient_module=orient_module,
    agent=agent,
    obs_space=env.observation_space,
    device=device,
    lr_graph=3e-5,
    lr_orient=3e-5,
    batch_size=1024,
    train_steps_per_call=1,
    log_interval=50,
)

# Saving of the aux modules (GraphEncoder + OrientationModule). skrl only
# checkpoints the SAC agent (actor/critic/preprocessor); the graph encoder and
# orientation module live outside the agent, so we save them ourselves.
# Note: id_to_name_emb is a non-persistent buffer (rebuilt from text_embeddings.pt
# on load), so these files stay small — only the learned weights + optimizers.
AUX_SAVE_DIR = f"{LOG_ROOT}/aloha_sac/aux_checkpoints/{NAME}"   # per-run: no cross-run overwrite
AUX_SAVE_INTERVAL = int(os.environ.get("GIROL_AUX_SAVE_INTERVAL", "2000"))

def save_aux_modules(tag):
    os.makedirs(AUX_SAVE_DIR, exist_ok=True)
    payload = {
        "graph_encoder": graph_encoder.state_dict(),
        "orient_module": orient_module.state_dict(),
        "graph_optimizer": aux_trainer.graph_optimizer.state_dict(),
        "orient_optimizer": aux_trainer.orient_optimizer.state_dict(),
        "timestep": tag,
        "use_metric": USE_METRIC,
    }
    torch.save(payload, f"{AUX_SAVE_DIR}/aux_{tag}.pt")
    torch.save(payload, f"{AUX_SAVE_DIR}/aux_latest.pt")  # convenience pointer
    print(f"[save] aux modules -> {AUX_SAVE_DIR}/aux_{tag}.pt")

# Optional Comet ML logging. Disabled by default: set GIROL_USE_COMET=1 and
# provide your OWN COMET_API_KEY / COMET_WORKSPACE env vars. skrl already writes
# TensorBoard logs to cfg["experiment"]["directory"] regardless.
experiment = None
if USE_COMET:
    from comet_ml import start
    experiment = start(
        api_key=os.environ.get("COMET_API_KEY", ""),
        project_name=os.environ.get("COMET_PROJECT", "general"),
        workspace=os.environ.get("COMET_WORKSPACE", ""),
    )
_original_post = agent.post_interaction
mode_1 = False
def _post_with_aux(timestep, timesteps):
    _original_post(timestep, timesteps)
    if not mode_1:
        if timestep > cfg["learning_starts"]:
            aux_trainer.step(timestep)

    # Periodically save the aux modules (this is the orientation-module baseline
    # we care about — skrl does NOT checkpoint these).
    if timestep > 0 and timestep % AUX_SAVE_INTERVAL == 0:
        save_aux_modules(timestep)
    if timestep % 50 == 0:
        metrics = env.unwrapped.get_metrics()
        if timestep % 3000 == 0:
            print(metrics)
        acc = print_orientation_accuracy(True)
        acc_10, acc_20, acc_30 = acc if acc is not None else (0.0, 0.0, 0.0)

        # TensorBoard: surface orientation + curriculum metrics through skrl's
        # writer (they end up in logs/skrl/aloha_sac/<run>). SAC losses/rewards
        # are already logged by skrl itself.
        agent.track_data("Orient / acc_10deg", acc_10)
        agent.track_data("Orient / acc_20deg", acc_20)
        agent.track_data("Orient / acc_30deg", acc_30)
        agent.track_data("Nav / success_rate", float(metrics["success_rate"]))
        agent.track_data("Nav / angle_error", float(metrics["cur_angle_error"]))
        agent.track_data("Curriculum / stage", float(metrics["stage"]))
        agent.track_data("Curriculum / mean_radius", float(metrics["mean_radius"]))

        if timestep % 500 == 0:
            print(f"[{timestep}] stage={metrics['stage']} "
                  f"success_rate={metrics['success_rate']:.3f} "
                  f"angle_err={metrics['cur_angle_error']:.3f} "
                  f"orient_acc(10/20/30)={acc_10:.2f}/{acc_20:.2f}/{acc_30:.2f}")
        if experiment is not None:
            experiment.log_metric("success_rate", metrics["success_rate"], step=timestep)
            experiment.log_metric("mean_radius", metrics["mean_radius"], step=timestep)
            experiment.log_metric("angle_error", metrics["cur_angle_error"], step=timestep)
            experiment.log_metric("accuracy orientation module 10 grad", acc_10, step=timestep)
            experiment.log_metric("accuracy orientation module 20 grad", acc_20, step=timestep)
            experiment.log_metric("accuracy orientation module 30 grad", acc_30, step=timestep)

agent.post_interaction = _post_with_aux

if mode_1:
    USE_PRETRAINED = True

if not EVAL:
    if USE_PRETRAINED:
        checkpoint_path = os.environ.get("GIROL_CHECKPOINT_DIR", f"{LOG_ROOT}/aloha_sac")
        # graph_encoder.load_state_dict(
        #     torch.load(f"{checkpoint_path}/added/graph_encoder_80000.pt")
        # )
        # orient_module.load_state_dict(
        #     torch.load(f"{checkpoint_path}/added/orient_module_80000.pt")
        # # )
        agent_path = f"{checkpoint_path}/26-04-21_11-59-59-958306_SAC/checkpoints/agent_20000.pt"
        agent.load(agent_path)
        # graph_encoder.load_state_dict(
        #     torch.load(f"logs/skrl/aloha_sac/added/graph_encoder_42000.pt")
        # )
        # orient_module.load_state_dict(
        #     torch.load(f"logs/skrl/aloha_sac/added/orient_module_42000.pt")
        # )
        if mode_1:
            graph_encoder.eval()
            orient_module.eval()

            for param in graph_encoder.parameters():
                param.requires_grad = False

            for param in orient_module.parameters():
                param.requires_grad = False
    trainer = SequentialTrainer(cfg={"timesteps": timestepslen}, env=env, agents=agent)
    trainer.train()
    save_aux_modules("final")   # final GraphEncoder + OrientationModule weights
    mem_dir = f"{LOG_ROOT}/memory/4img_128"
    os.makedirs(mem_dir, exist_ok=True)
    memory.save(directory=mem_dir)
    torch.save(agent._state_preprocessor.state_dict(), f"{mem_dir}/preprocessor.pt")
else:
    # memory.load(directory="logs/skrl/aloha_sac/memory")
    # agent._state_preprocessor.load_state_dict(
    #     torch.load("logs/skrl/aloha_sac/memory/preprocessor.pt")
    # )
    env.unwrapped.start_eval_metrics(seed=seed, name=NAME)
    checkpoint_path = os.environ.get("GIROL_CHECKPOINT_DIR", f"{LOG_ROOT}/aloha_sac")
    agent_path = f"{checkpoint_path}/26-04-20_11-29-05-192862_SAC/checkpoints/agent_1000.pt"
    agent.load(agent_path)
    # graph_encoder.load_state_dict(
    #     torch.load(f"{checkpoint_path}/added/archive/postrain_nav_1_img/graph_encoder_98000.pt")
    # )
    # orient_module.load_state_dict(
    #     torch.load(f"{checkpoint_path}/added/archive/postrain_nav_1_img/orient_module_98000.pt")
    # )
    trainer = SequentialTrainer(cfg={"timesteps": timestepslen}, env=env, agents=agent)
    trainer.eval()

    acc_10, acc_20, acc_30 = print_orientation_accuracy(True)
    metrics = env.unwrapped.get_metrics()
    print(metrics)
    metrics = env.unwrapped.get_eval_metrics()
    print(metrics)

    # Сохраняем accuracy рядом с eval_log
    orient_tensor = torch.tensor([acc_10, acc_20, acc_30], dtype=torch.float32)
    save_dir = f"{LOG_ROOT}/logs/old/{NAME}"
    os.makedirs(save_dir, exist_ok=True)
    torch.save(orient_tensor, f"{save_dir}/orient_acc_{seed}.pt")
    print(f"[Eval] Saved orient accuracy: acc10={acc_10:.3f}, acc20={acc_20:.3f}, acc30={acc_30:.3f}")