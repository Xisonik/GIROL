# GIROL — Repo Notes

Companion to [BASELINE.md](BASELINE.md) (which covers the orientation-module baseline).
This file documents two things that sit *around* that baseline:

1. The **curriculum stages 0–4** in the environment and how they shape orientation training.
2. The **config-driven DDQN pipeline** (`configs/cur_dqn/ddqn_discrete.json`) — a second,
   separate training system from the SAC baseline.

---

## 0. Task mode: TURN_TASK (orientation baseline)

The orientation baseline runs in **`TURN_TASK`** mode (enabled by the launch scripts
via `ALOHA_NAV_ENV_CFG='{"TURN_TASK": true}'`). In this mode the task is **"turn in
place to face the goal"**:

- **Action:** linear speed is forced to 0 — the robot only rotates (`angular_speed = 2·action[:,1]`).
- **Reward:** reduction of the angle between the robot's heading and the goal direction.
- **Success = `facing_goal`:** heading within **20°** of the goal (no distance requirement),
  but **not** in the first `turn_min_steps` (default 3) steps — so a robot that spawns
  already aligned can't win at step 0 (`ALOHA_NAV_ENV_CFG='{"TURN_MIN_STEPS": 5}'`, 0 disables).
- **No termination for "facing away".** The stock TURN_TASK killed the episode when the
  angle-to-goal exceeded 150° (`out_of_bounds`), which instant-terminated any robot spawned
  facing away and caused a constant reset loop (and a flood of USD warnings). Removed — the
  robot may now turn toward the goal from *any* starting angle; episodes end on success or timeout.

This makes the curriculum's `success_rate` gate mean exactly **"the robot oriented to
the goal"**, which is why the stage transitions below are driven by it.

> By default `TURN_TASK=False` (full navigation). Enabling it also keeps the
> curriculum ON — the stock code disabled it (`CL_ON=False`); we changed that so the
> stages still advance on facing-goal success.
>
> ⚠️ Caveat unchanged: the policy that turns the robot is fed the **ground-truth**
> orientation; the orientation *module*'s prediction is a decoupled side-task and does
> not drive turning. So `success_rate` measures the *policy* orienting the robot (with
> GT), not the *module*'s prediction accuracy. Gate on `Orient / acc_*` instead if you
> want stage advancement to track the module itself.

## 1. Curriculum stages (`place_robot_for_goal_stage_{0..4}`)

At every episode reset the environment spawns the robot using a **stage-specific
pose strategy**. The dispatch is:

```python
# aloha_env_base.py (_reset)
method_name = f"place_robot_for_goal_stage_{self.stage}"   # -> scene_manager.py
robot_pos, robot_quat = method(env_ids, mean_dist=self.mean_radius,
                               min_dist=1.2, max_dist=8.0,
                               angle_error=self.cur_angle_error)
```

Each `place_robot_for_goal_stage_N` in
[modules/scene_manager.py](source/isaaclab_tasks/isaaclab_tasks/direct/aloha_nav/modules/scene_manager.py)
differs in **where the robot spawns** and **which way it faces** — i.e. the
distribution of `relative_yaw` (robot→goal heading) the robot starts with:

| Stage | Robot position | Initial yaw | Character |
|-------|----------------|-------------|-----------|
| **0** | random in active rooms | fully random `[0, 2π)` | Warm-up: no relation to goal |
| **1** | on a ring at radius ≈ `mean_dist` **around the goal** | `dir_to_goal ± rand(angle_error)` | **Curriculum-controlled** (the important one) |
| **2** | random in active rooms | random `[−π, π)` | Hard / final: heading unrelated to goal |
| **3** | random in active rooms | facing **away from room center** | Alternative placement variant |
| **4** | random in active rooms | `±π/2` | Alternative placement variant |

Stage 1 is the key one: the robot is placed a controllable **distance**
(`mean_dist = mean_radius + 1.31`, clamped to `[min_dist, max_dist]`) from the goal
and rotated to *roughly face the goal* with a controllable **angular error**
(`± angle_error`). Both knobs are driven by the curriculum.

### Stage progression (`curriculum_learning_module`)
Init: `stage=0, mean_radius=0, cur_angle_error=0, warm_len=2000, max_angle_error=π`,
`max_curriculum_stage=4`.

- **Stage 0 → 1**: after `warm_len` (2000) steps. Warm-up done, main training starts.
- **Stage 1 → 2**: when `mean_radius ≥ 6.0` m.
- **Stage 2 → 3 → 4**: **success-gated** — after the policy sustains a high success
  rate (`≥ sr_treshhold` for >512 accumulated success episodes) at the current stage
  (`_advance_stage`). Stages 2/3/4 use random placement with different yaw schemes,
  so there is no radius/angle knob to ramp; sustained success just switches the pose
  distribution to the next stage. Capped at `max_curriculum_stage`.
- **Difficulty adaptation within stage 1**, based on rolling success rate:
  - success high (`≥ threshold` for >512 success episodes) → `_increase_difficulty`:
    first grow `cur_angle_error` by `max_angle_error/7` each step; once angle error
    exceeds `max_angle_error (π)`, reset it and grow `mean_radius` (+1.0, up to 8.0 m).
  - success low (`≤ 70%` for a long stretch) → `_decrease_difficulty`: shrink `mean_radius`.

> The curriculum now auto-progresses **0→1→2→3→4** (success-gated for 2→3→4).
> Cap it lower via `ALOHA_NAV_ENV_CFG='{"MAX_STAGE": 2}'` to restore the old 0→1→2
> behavior. Reaching stage 2+ requires the policy to master stage 1 first, so in a
> short run you will mostly see stages 0 and 1 (watch `Curriculum / stage` in TensorBoard).

### Why stages matter for orientation-module training
The orientation module is trained (as a supervised side-task, see BASELINE.md §3)
to predict `relative_yaw` from `img ⊕ graph_emb`. The curriculum **controls the
label distribution and difficulty** that end up in the replay buffer:

- **`angle_error` (stage 1)** is the direct lever on label diversity. Small →
  labels cluster near 0° (robot already faces the goal): easy, but the module only
  sees a narrow slice of the 36 orientation bins. As it grows toward `π`, labels
  spread across the **full circle**, giving rich supervision on every heading.
- **`mean_radius`** sets goal distance → changes the camera view and geometry the
  module must map to a heading.
- So progression is effectively a **schedule from easy/low-diversity to hard/full-range**
  orientation labels. A module that looks accurate early (stage 0/low angle_error)
  may just be exploiting an easy label distribution — compare accuracy *at a given
  stage/angle_error*, and watch `Curriculum / stage` + `Nav / angle_error` alongside
  `Orient / acc_*` in TensorBoard when reading results.

---

## 2. The DDQN config pipeline (`configs/cur_dqn/ddqn_discrete.json`)

This is a **second, config-driven training system**, separate from the SAC baseline
(`run_sac_ORM.py` + `networks/networks_orm.py`). It reuses the same idea —
GraphEncoder + OrientationModule + auxiliary trainer — but with a **discrete-action
Double-DQN** policy and a cleaner `perception/` implementation.

### How it runs
```
run_experiments.py  ──spawns──▶  runners/train_ddqn.py   (one process per config)
     │
     └─ reads configs/cur_dqn/ : base.json + <experiment>.json
```
- Every config dir **must contain `base.json`**; each experiment JSON is
  **deep-merged over `base.json`** ([configs/config_utils.py](scripts/algos/configs/config_utils.py)).
  So `ddqn_discrete.json` = `base.json` + its overrides.
- **Value resolvers** used inside the JSON:
  - `{"cfg": "paths.text_embeddings"}` → pull a value from another config path.
  - `{"obs_dim": "img"}` → fill in a dimension from the env observation space.
  - `{"$value": [256]}` → a literal (also the grid-sweep axis marker).

Run it with:
```bash
conda activate isaaclab45
cd /home/rizo/work/GIROL
./isaaclab.sh -p scripts/algos/run_experiments.py --configs-dir scripts/algos/configs/cur_dqn
```

### Parameter blocks

**`run`** — process / experiment control
| key | value | role |
|-----|-------|------|
| `algo` | `ddqn` | selects the Double-DQN runner |
| `task_name` | `Aloha_nav_hab_wr` | a **different** registered env variant (habitat-style, wheeled-robot, discrete turning) |
| `num_envs` | `32` | parallel envs (VRAM: too high for 16 GB — lower it) |
| `seed`, `headless`, `video`, `eval` | | run mode |
| `folder` | `debug` | output subfolder under `log_root` |
| `write_interval` / `checkpoint_interval` | `10` / `1000` | TensorBoard write & checkpoint cadence |
| `timesteps` | `20000000` | full training length (20 M) |

**`agent`** — DDQN hyperparameters
| key | value | role |
|-----|-------|------|
| `memory_size` | `4000` | replay buffer size |
| `gradient_steps` | `1` | Q-updates per env step |
| `batch_size` | `512` | minibatch |
| `gamma` | `0.99` | discount factor |
| `polyak` | `0.005` | soft target-network update rate (τ) |
| `learning_rate` | `3e-4` | Q-network LR |
| `random_timesteps` | `1000` | initial pure-random exploration |
| `learning_starts` | `1000` | steps collected before learning begins |
| `update_interval` | `1` | env steps between Q updates |
| `target_update_interval` | `10` | env steps between target-net syncs |
| `normalize_img` | `true` | normalize image observations |

**`model`** — what feeds the Q-network and which sub-modules exist
- `recurrent.enabled=false` → no LSTM.
- `features` — inputs concatenated into the Q-network:
  - `use_img=true`, `use_goal=true`, `use_memory=false`
  - `use_gt_orientation=false` → do **not** feed the ground-truth yaw
  - `use_graph=true` (`graph_dim=128`, `graph_ref="graph_encoder"`) → feed `graph_emb`
  - `use_orientation_module=true` (`orientation_dim=1`, `orientation_ref="orientation_module"`)
    → feed the **predicted** yaw
  > Contrast with the SAC baseline, where `USE_GRAPH=0` (the policy ignores the graph).
  > Here the **policy actually consumes** graph_emb *and* the predicted orientation.
- `modules` — instantiated components:
  - `graph_encoder`: `perception.graph_encoder.GraphEncoder` (the config-pipeline's own
    encoder, **not** `networks_orm.py`), `out_dim=128`, `dropout=0.1`, `eval=true`
    (runs frozen in the policy forward; it is trained by the aux trainer).
  - `orientation_module`: `perception.orientation_module.OrientationFeature`,
    `mode="pred"` (use the learned predictor, not GT), `force_predictor=true`,
    `img_dim` from the `img` obs, `graph_emb_dim=128`, `eval=true`.
  - `q_network`: `models.ddqn_models.DDQNNavQNetwork`, `hidden_dims=[256]`.

**`aux`** — the auxiliary orientation trainer (`perception.aux_trainer.OrientationAuxTrainer`, from base.json)
- `enabled=true`; trains `graph_encoder` + `orientation_module` from the DDQN replay
  buffer with the orientation loss — the same side-task pattern as SAC's `AuxModuleTrainer`.
- `lr_graph=3e-5`, `lr_orient=3e-5`, `batch_size=256`, `train_steps_per_call=1`,
  `grad_norm_clip=1.0`, `save_interval=1000`, `save_optimizer=true`.

**`env`** — environment options
| key | value | role |
|-----|-------|------|
| `task` | `hab_nav_wr` | env variant selector |
| `action_angle_deg` | `20` | **discrete** action = turn in 20° increments (DDQN needs discrete actions) |
| `turn_task` | `false` | full navigation (not turn-only) |
| `camera` | `true` | render camera → image obs |
| `expert` | `true` | expert controller (pure-pursuit) assists during collection |
| `curriculum` | `true` | enable the **stage curriculum from §1** |

**`paths`** — files & checkpoints
- `text_embeddings` (local, fine), `log_root=logs/skrl`.
- ⚠ `graphs_dir`, `aux_checkpoint`, `state_preprocessor_checkpoint` in `base.json`
  point at **`/home/xiso/IsaacLab/...`** — same hardcoded-path problem the SAC runner
  had. If you actually run this DDQN pipeline, fix these (and lower `num_envs`) the
  way BASELINE.md §6 describes for the SAC runner. The `perception/` GraphEncoder may
  also need the same 6-dim compatibility check the baseline encoder got.

### Role summary
`ddqn_discrete.json` is an **alternative experiment**: same GraphEncoder +
OrientationModule + aux-trainer design, but a discrete-turn DDQN policy that *uses*
the graph and predicted orientation as policy inputs, and can warm-start from a
pretrained aux checkpoint. The orientation module plays the identical role — predict
heading-to-goal, trained by the aux trainer from the replay buffer — so everything
in §1 about the curriculum shaping orientation labels applies here too.
