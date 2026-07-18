# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Script to play a checkpoint of an RL agent from skrl.

Visit the skrl documentation (https://skrl.readthedocs.io) to see the examples structured in
a more user-friendly way.
"""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys

from isaaclab.app import AppLauncher
import torch.nn as nn
import torch.nn.utils.prune as prune

# >>> ADDED: нужен torch/numpy для метрик/распределений/снип-прунинга
import torch
import numpy as np

# >>> ADDED: ВЫБОР ИЗНУТРИ КОДА (просто меняешь эти константы)
# 0 = no pruning
# 1 = global unstructured L1
# 2 = N:M (semi-structured) pruning (например 2:4)
# 3 = SNIP-like (градиентная важность на батче наблюдений)
PRUNE_METHOD = 0
PRUNE_AMOUNT = 0.0      # 0..1
NM_N = 1
NM_M = 4
SNIP_COLLECT_STEPS = 512
EPS_ZERO = 0.0          # порог "нулевого" веса; 0.0 честнее, можно поставить 1e-8 если надо

# add argparse arguments
parser = argparse.ArgumentParser(description="Play a checkpoint of an RL agent from skrl.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--checkpoint", type=str, default=None, help="Path to model checkpoint.")
parser.add_argument(
    "--use_pretrained_checkpoint",
    action="store_true",
    help="Use the pre-trained checkpoint from Nucleus.",
)
parser.add_argument(
    "--ml_framework",
    type=str,
    default="torch",
    choices=["torch", "jax", "jax-numpy"],
    help="The ML framework used for training the skrl agent.",
)
parser.add_argument(
    "--algorithm",
    type=str,
    default="PPO",
    choices=["AMP", "PPO", "IPPO", "MAPPO", "SAC"],
    help="The RL algorithm used for training the skrl agent.",
)
parser.add_argument("--real-time", action="store_true", default=False, help="Run in real-time, if possible.")

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli, hydra_args = parser.parse_known_args()
# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args
# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import os
import time
import torch

import skrl
from packaging import version

# check for minimum supported skrl version
SKRL_VERSION = "1.4.2"
if version.parse(skrl.__version__) < version.parse(SKRL_VERSION):
    skrl.logger.error(
        f"Unsupported skrl version: {skrl.__version__}. "
        f"Install supported version using 'pip install skrl>={SKRL_VERSION}'"
    )
    exit()

if args_cli.ml_framework.startswith("torch"):
    from skrl.utils.runner.torch import Runner
elif args_cli.ml_framework.startswith("jax"):
    from skrl.utils.runner.jax import Runner

from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.dict import print_dict
from isaaclab.utils.pretrained_checkpoint import get_published_pretrained_checkpoint

from isaaclab_rl.skrl import SkrlVecEnvWrapper

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

# PLACEHOLDER: Extension template (do not remove this comment)

# config shortcuts
algorithm = args_cli.algorithm.lower()
agent_cfg_entry_point = "skrl_cfg_entry_point" if algorithm in ["ppo", "sac"] else f"skrl_{algorithm}_cfg_entry_point"

def get_size(model):
    torch.save(model.state_dict(), "temp.pth")
    size_mb = os.path.getsize("temp.pth") / (1024*1024)
    os.remove("temp.pth")

    return size_mb

# >>> ADDED: общие хелперы (актер, метрики, распределение)
def get_actor_mlp(agent):
    policy = getattr(agent, "policy", None)
    if policy is None and hasattr(agent, "models"):
        policy = agent.models.get("policy", None)
    if policy is None:
        print("[WARN] Не удалось найти policy в агенте")
        return None
    if not hasattr(policy, "net"):
        print("[WARN] У policy нет .net")
        return None
    return policy.net

def iter_linear_layers(mlp):
    for m in mlp.modules():
        if isinstance(m, nn.Linear):
            yield m

def count_weight_stats(mlp, eps_zero: float = 0.0):
    total = 0
    nonzero = 0
    abs_all = []
    for m in iter_linear_layers(mlp):
        w = m.weight.detach()
        total += w.numel()
        nonzero += (w.abs() > eps_zero).sum().item()
        abs_all.append(w.abs().flatten().cpu())
    abs_all = torch.cat(abs_all) if abs_all else torch.tensor([])
    return total, nonzero, abs_all

def count_dead_neurons(mlp, eps_zero: float = 0.0):
    """
    "Нейроны" считаем для Linear:
      - dead rows: выходные нейроны (строки), у которых все веса ~ 0
      - dead cols: входные фичи (столбцы), у которых все веса ~ 0
    """
    total_rows = dead_rows = 0
    total_cols = dead_cols = 0
    for m in iter_linear_layers(mlp):
        w = m.weight.detach()
        row_dead = (w.abs() <= eps_zero).all(dim=1)
        col_dead = (w.abs() <= eps_zero).all(dim=0)
        total_rows += w.size(0)
        dead_rows += row_dead.sum().item()
        total_cols += w.size(1)
        dead_cols += col_dead.sum().item()
    return (total_rows, dead_rows, total_cols, dead_cols)

def print_weight_distribution(abs_weights: torch.Tensor):
    if abs_weights.numel() == 0:
        print("[INFO] No weights to analyze")
        return
    aw = abs_weights.float()
    q = torch.quantile(aw, torch.tensor([0.0, 0.25, 0.5, 0.75, 0.9, 0.99, 1.0], device=aw.device)).cpu().numpy()
    print("[INFO] |w| stats: min={:.3e} p25={:.3e} p50={:.3e} p75={:.3e} p90={:.3e} p99={:.3e} max={:.3e}".format(*q))
    print("[INFO] |w| mean={:.3e} std={:.3e}".format(aw.mean().item(), aw.std().item()))

    # "корзины" по |w| (лог-шкала) — удобно видеть сколько реально "участвуют"
    bins = np.array([0.0, 1e-12, 1e-10, 1e-8, 1e-6, 1e-4, 1e-3, 1e-2, 1e-1, 1.0, np.inf], dtype=np.float64)
    hist, edges = np.histogram(aw.cpu().numpy(), bins=bins)
    print("[INFO] |w| histogram (counts):")
    for i in range(len(hist)):
        lo, hi = edges[i], edges[i+1]
        print(f"  [{lo:.0e}, {hi:.0e}): {hist[i]}")

def print_model_report(tag: str, mlp, eps_zero: float = 0.0):
    size_mb = get_size(mlp)
    total, nonzero, abs_all = count_weight_stats(mlp, eps_zero=eps_zero)
    sparsity = 1.0 - (nonzero / max(total, 1))
    tr, dr, tc, dc = count_dead_neurons(mlp, eps_zero=eps_zero)

    print(f"\n========== REPORT: {tag} ==========")
    print(f"[INFO] size(MB): {size_mb:.4f}")
    print(f"[INFO] weights: total={total} nonzero={nonzero} sparsity={sparsity*100:.2f}% (eps_zero={eps_zero})")
    print(f"[INFO] dead neurons (rows): {dr}/{tr} = {100.0*dr/max(tr,1):.2f}%")
    print(f"[INFO] dead inputs  (cols): {dc}/{tc} = {100.0*dc/max(tc,1):.2f}%")
    print_weight_distribution(abs_all)
    print("===================================\n")


def prune_actor_mlp(agent, amount: float = 0.3):
    """Unstructured L1-pruning линейных слоёв MLP актёра.

    amount=0.3 => обнуляем 30% самых маленьких по модулю весов.
    """
    # пробуем достать policy из агента
    policy = getattr(agent, "policy", None)
    if policy is None and hasattr(agent, "models"):
        # запасной вариант через словарь моделей
        policy = agent.models.get("policy", None)

    if policy is None:
        print("[WARN] Не удалось найти policy в агенте, прунинг пропущен")
        return

    if not hasattr(policy, "net"):
        print("[WARN] У policy нет .net, прунинг пропущен")
        return

    print(f"[INFO] Applying unstructured pruning to actor MLP with amount={amount}")
    mlp = policy.net

    # 1) навешиваем маски
    for module in mlp.modules():
        if isinstance(module, nn.Linear):
            print("[INFO] Pruning layer:", module)
            prune.l1_unstructured(module, name="weight", amount=amount)

    # 2) зашиваем маски в сами веса, чтобы не было weight_orig/weight_mask
    for module in mlp.modules():
        if isinstance(module, nn.Linear) and hasattr(module, "weight_mask"):
            prune.remove(module, "weight")
    size_mb = get_size(mlp)
    print("size: ", size_mb)
    print("[INFO] Pruning done")

# >>> ADDED: 3 альтернативы прунинга + функция выбора

def prune_actor_global_unstructured(agent, amount: float = 0.5):
    mlp = get_actor_mlp(agent)
    if mlp is None:
        return
    params = []
    for m in iter_linear_layers(mlp):
        params.append((m, "weight"))
    print(f"[INFO] Global unstructured L1 pruning amount={amount}, layers={len(params)}")
    prune.global_unstructured(
        params,
        pruning_method=prune.L1Unstructured,
        amount=amount,
    )
    for (m, name) in params:
        prune.remove(m, name)

def nm_prune_linear_weight_(w: torch.Tensor, n: int = 2, m: int = 4):
    assert w.dim() == 2, "Ожидался weight матрицы Linear (out_features x in_features)"
    in_features = w.size(1)
    trimmed = (in_features // m) * m
    if trimmed == 0:
        return
    if trimmed != in_features:
        print(f"[WARN] in_features={in_features} не кратно m={m}. Хвост {in_features-trimmed} не трогаем.")
    w_main = w[:, :trimmed].view(w.size(0), -1, m)
    tail = w[:, trimmed:] if trimmed != in_features else None

    # 2:4 => оставляем 2 НЕнулевых => topk=2
    topk = n
    abs_w = w_main.abs()
    idx = torch.topk(abs_w, k=topk, dim=-1).indices
    mask = torch.zeros_like(w_main, dtype=torch.bool)
    mask.scatter_(-1, idx, True)

    pruned = torch.where(mask, w_main, torch.zeros_like(w_main)).view(w.size(0), -1)
    if tail is not None:
        pruned = torch.cat([pruned, tail], dim=1)
    w.copy_(pruned)

def prune_actor_nm(agent, n: int = 2, m: int = 4):
    mlp = get_actor_mlp(agent)
    if mlp is None:
        return
    print(f"[INFO] N:M pruning n={n}, m={m}")
    for layer in iter_linear_layers(mlp):
        nm_prune_linear_weight_(layer.weight.data, n=n, m=m)

def collect_obs_batch(env, runner, steps: int = 512):
    obs_list = []
    obs, _ = env.reset()
    for _ in range(steps):
        obs_list.append(obs)
        with torch.inference_mode():
            outputs = runner.agent.act(obs, timestep=0, timesteps=0)
            if hasattr(env, "possible_agents"):
                actions = {a: outputs[-1][a].get("mean_actions", outputs[0][a]) for a in env.possible_agents}
            else:
                actions = outputs[-1].get("mean_actions", outputs[0])
        obs, _, _, _, _ = env.step(actions)

    # obs может быть dict или tensor
    if isinstance(obs_list[0], dict):
        key = "policy" if "policy" in obs_list[0] else list(obs_list[0].keys())[0]
        batch = torch.cat([o[key] for o in obs_list], dim=0)
    else:
        batch = torch.cat(obs_list, dim=0)
    return batch

def prune_actor_snip_like(agent, mlp_obs_batch: torch.Tensor, amount: float = 0.5):
    mlp = get_actor_mlp(agent)
    if mlp is None:
        return
    linears = list(iter_linear_layers(mlp))
    if not linears:
        return

    # считаем градиенты по прокси-лоссу на выходе MLP
    mlp.train()
    for l in linears:
        if l.weight.grad is not None:
            l.weight.grad.zero_()

    out = mlp(mlp_obs_batch)
    loss = (out ** 2).mean()
    loss.backward()

    scores = []
    for l in linears:
        g = l.weight.grad
        s = (l.weight.data * g).abs()
        scores.append(s.flatten())
    all_scores = torch.cat(scores)

    keep = int(all_scores.numel() * (1.0 - amount))
    if keep < 1:
        print("[WARN] amount слишком большой, SNIP-like пропущен")
        mlp.eval()
        return

    thr = torch.topk(all_scores, k=keep, largest=True).values.min()
    with torch.no_grad():
        for l in linears:
            s = (l.weight.data * l.weight.grad).abs()
            mask = s >= thr
            l.weight.data *= mask

    for l in linears:
        l.weight.grad = None
    mlp.eval()
    print(f"[INFO] SNIP-like pruning amount={amount} (proxy loss={loss.item():.6f})")

def apply_pruning_choice(agent, env, runner):
    """
    Выбор делаем ВНУТРИ КОДА через PRUNE_METHOD / PRUNE_AMOUNT.
    """
    mlp = get_actor_mlp(agent)
    if mlp is None:
        print("[WARN] Actor MLP not found -> skipping pruning")
        return

    # report before
    print_model_report("BEFORE", mlp, eps_zero=EPS_ZERO)

    if PRUNE_METHOD == 0:
        print("[INFO] PRUNE_METHOD=0 (no pruning)")
    elif PRUNE_METHOD == 1:
        prune_actor_global_unstructured(agent, amount=PRUNE_AMOUNT)
    elif PRUNE_METHOD == 2:
        prune_actor_nm(agent, n=NM_N, m=NM_M)
    elif PRUNE_METHOD == 3:
        print(f"[INFO] Collecting obs for SNIP-like: steps={SNIP_COLLECT_STEPS}")
        obs_batch = collect_obs_batch(env, runner, steps=SNIP_COLLECT_STEPS)
        prune_actor_snip_like(agent, obs_batch, amount=PRUNE_AMOUNT)
    else:
        print("[WARN] Unknown PRUNE_METHOD, skipping")

    # report after
    print_model_report("AFTER", mlp, eps_zero=EPS_ZERO)


@hydra_task_config(args_cli.task, agent_cfg_entry_point)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, experiment_cfg: dict):
    """Play with skrl agent."""
    # override configurations with non-hydra CLI arguments
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    # configure the ML framework into the global skrl variable
    if args_cli.ml_framework.startswith("jax"):
        skrl.config.jax.backend = "jax" if args_cli.ml_framework == "jax" else "numpy"

    task_name = args_cli.task.split(":")[-1]

    # specify directory for logging experiments (load checkpoint)
    log_root_path = os.path.join("logs", "skrl", experiment_cfg["agent"]["experiment"]["directory"])
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Loading experiment from directory: {log_root_path}")
    # get checkpoint path
    if args_cli.use_pretrained_checkpoint:
        resume_path = get_published_pretrained_checkpoint("skrl", task_name)
        if not resume_path:
            print("[INFO] Unfortunately a pre-trained checkpoint is currently unavailable for this task.")
            return
    elif args_cli.checkpoint:
        resume_path = os.path.abspath(args_cli.checkpoint)
    else:
        resume_path = get_checkpoint_path(
            log_root_path, run_dir=f".*_{algorithm}_{args_cli.ml_framework}", other_dirs=["checkpoints"]
        )
    log_dir = os.path.dirname(os.path.dirname(resume_path))

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv) and algorithm in ["ppo"]:
        env = multi_agent_to_single_agent(env)

    # get environment (step) dt for real-time evaluation
    try:
        dt = env.step_dt
    except AttributeError:
        dt = env.unwrapped.step_dt

    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "play"),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # wrap around environment for skrl
    env = SkrlVecEnvWrapper(env, ml_framework=args_cli.ml_framework)  # same as: `wrap_env(env, wrapper="auto")`

    # configure and instantiate the skrl runner
    # https://skrl.readthedocs.io/en/latest/api/utils/runner.html
    experiment_cfg["trainer"]["close_environment_at_exit"] = False
    experiment_cfg["agent"]["experiment"]["write_interval"] = 0  # don't log to TensorBoard
    experiment_cfg["agent"]["experiment"]["checkpoint_interval"] = 0  # don't generate checkpoints
    runner = Runner(env, experiment_cfg)

    print(f"[INFO] Loading model checkpoint from: {resume_path}")
    runner.agent.load(resume_path)
    # set agent to evaluation mode
    runner.agent.set_running_mode("eval")

    # >>> CHANGED: вместо одного prune_actor_mlp — выбор + отчёт до/после
    apply_pruning_choice(runner.agent, env, runner)

    # reset environment
    obs, _ = env.reset()
    timestep = 0
    # simulate environment
    while simulation_app.is_running():
        start_time = time.time()

        # run everything in inference mode
        with torch.inference_mode():
            # agent stepping
            outputs = runner.agent.act(obs, timestep=0, timesteps=0)
            # - multi-agent (deterministic) actions
            if hasattr(env, "possible_agents"):
                actions = {a: outputs[-1][a].get("mean_actions", outputs[0][a]) for a in env.possible_agents}
            # - single-agent (deterministic) actions
            else:
                actions = outputs[-1].get("mean_actions", outputs[0])
            # env stepping
            obs, _, _, _, _ = env.step(actions)
        if args_cli.video:
            timestep += 1
            # exit the play loop after recording one video
            if timestep == args_cli.video_length:
                break

        # time delay for real-time evaluation
        sleep_time = dt - (time.time() - start_time)
        if args_cli.real_time and sleep_time > 0:
            time.sleep(sleep_time)

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
