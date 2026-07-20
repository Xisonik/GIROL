# Orientation-Module Baseline (Metric Graph)

This document describes the **baseline** for the orientation-module study in GIROL:
how the robot's environment is represented as a graph, how the orientation module
is trained, and how to launch it. The baseline uses a **metric graph** (object
positions included). Later experiments remove the metric information and rely on
semantic/topological structure only; the baseline is the reference point.

- Task id: `Aloha_nav` (registered in
  [source/isaaclab_tasks/isaaclab_tasks/direct/aloha_nav/__init__.py](source/isaaclab_tasks/isaaclab_tasks/direct/aloha_nav/__init__.py)).
  > Note: the older name `Isaac-Aloha-Direct-v0` in the top-level README is **not**
  > registered anywhere — use `Aloha_nav`.
- Task mode: **`TURN_TASK`** — "turn in place to face the goal". The robot only
  rotates (no translation); **success = facing the goal within 20°**. This makes the
  curriculum's success gate mean "the robot oriented to the goal". The launch scripts
  set it via `ALOHA_NAV_ENV_CFG='{"TURN_TASK": true, "MAX_STAGE": 4}'`. Full details:
  [GIROL.md §0–1](GIROL.md).
- Runner: [scripts/algos/run_sac_ORM.py](scripts/algos/run_sac_ORM.py)
- Encoders: [scripts/algos/networks/networks_orm.py](scripts/algos/networks/networks_orm.py)
- Launcher: [scripts/algos/launch_orientation_baseline.sh](scripts/algos/launch_orientation_baseline.sh)
- Smoke test: [scripts/algos/smoke_test_orientation.sh](scripts/algos/smoke_test_orientation.sh)
- Broader repo notes (curriculum stages, the DDQN config pipeline): [GIROL.md](GIROL.md)

---

## 1. Architecture

Three trainable groups, each with its **own optimizer**, learning from the same
data stream (the SAC replay buffer). RL and the orientation task are decoupled.

```
   obs = { img[512], goal[2], orientation[1], graph[6·M] }
                         │
     ┌───────────────────┼───────────────────────────────┐
     │                   │                                 │
 SAC actor/critic   GraphEncoder (GATv2)          OrientationModule
 (navigation)       graph[6·M] → graph_emb[128]   img ⊕ graph_emb → 36 bins
 own Adam           own AdamW                      own AdamW
     ▲                   │                                 ▲
     └── reads graph_emb & pred_angle under no_grad ───────┘
```

- The **SAC policy** collects navigation experience. In the baseline the policy
  is *not* conditioned on the graph (`USE_GRAPH=0`); it uses the ground-truth
  `orientation` + `goal`. This isolates the orientation module for study.
- The **GraphEncoder** and **OrientationModule** are trained as a supervised
  side-task by `AuxModuleTrainer` — **not** by the RL reward.
- Actor/critic call the encoders under `torch.no_grad()`, so RL gradients never
  reach the graph / orientation modules.

---

## 2. Graph representation (the "metric graph")

Produced every reset by `SceneManager.encode_scene_graph`
([modules/scene_manager.py](source/isaaclab_tasks/isaaclab_tasks/direct/aloha_nav/modules/scene_manager.py)).
For each of the `M` objects in the scene, a **6-dim** vector:

| dim | field       | meaning                                                             |
|-----|-------------|--------------------------------------------------------------------|
| 0   | `object_id` | semantic id (config `id`, 0–16) → CLIP name embedding lookup       |
| 1   | `active`    | 1.0 if the object is present in the current scene                   |
| 2   | `is_goal`   | 1.0 for the current navigation target                              |
| 3–5 | `x, y, z`   | **metric position** in room coordinates ← *the metric information* |

The full observation field is `graph[6·M]` (flattened). For the current scene
`M = 22` objects (`table_2`×3, `chair_2`×2, `chair_3`×4, + 13 singletons; see
[configs/scene_items.json](source/isaaclab_tasks/isaaclab_tasks/direct/aloha_nav/configs/scene_items.json)),
so `graph` has 132 values. `M` is inferred from the tensor width at runtime, so
changing the scene's object count does not require code changes.

**Semantics.** Identity is *not* stored per step. `object_id` indexes the
precomputed table `id_to_name_emb [17, 512]` inside
[text_embeddings.pt](source/isaaclab_tasks/isaaclab_tasks/direct/aloha_nav/text_embeddings.pt)
(a frozen CLIP text embedding per object class). No CLIP model runs at train time.

**Multiple rooms.** Objects from all rooms are pooled into one flat node set of
size `M`. Rooms are *not* separate subgraphs — they are distinguished only by the
objects' `x, y` coordinates and by which objects are `active`. Removing the metric
dims (the ablation) therefore removes the only explicit spatial/room signal in
the graph, forcing reliance on semantics + the camera image.

**Encoder pipeline** (`GraphEncoder._forward_compact`):

```
object_id ─► id_to_name_emb ─► text_proj ─────┐
                                              ├─► node_mlp ─► GATv2 ×2 ─► mean-pool ─► head ─► graph_emb[128]
[active, is_goal, x, y, z] ────────────────────┘
```

- Edges: a **star + chain** topology over the `M` nodes (node 0 = hub) plus
  self-loops, shared across the batch.
- 2 × `GATv2Conv` (2 heads) → `global_mean_pool` → MLP head → **128-dim** embedding.

---

## 3. Orientation module & how it is trained

**Target.** `relative_yaw` — the heading from the robot to the goal, in the
robot's local frame (`atan2` of the robot→goal vector,
[aloha_env_base.py](source/isaaclab_tasks/isaaclab_tasks/direct/aloha_nav/envs/aloha_env_base.py)),
delivered as `obs["orientation"]`.

**Framing — classification, not regression.**
- 36 bins over [−π, π] (**10° per bin**).
- `OrientationModule`: `MLP( img[512] ⊕ graph_emb[128] ) → 36 logits → softmax → argmax bin → angle`.
- Loss: **cross-entropy** with label smoothing 0.05.

**Training loop** (`AuxModuleTrainer.step`), fired from SAC's `post_interaction`
every step after `learning_starts`:
1. Sample a batch (1024) of **states** from the SAC replay buffer.
2. `graph_emb = GraphEncoder(graph)`, then `logits = OrientationModule(img, graph_emb)`.
3. Cross-entropy loss vs. the binned `relative_yaw`.
4. Backward flows into **both** the orientation module *and* the graph encoder
   (two AdamW optimizers, lr `3e-5`, grad-clip 1.0). The graph representation is
   thus shaped by the orientation objective.

Only `img` is normalized by the preprocessor; `graph` and `orientation` are passed
through raw, so the encoder sees true metric coordinates.

**What the orientation module sees depends on the curriculum.** Episodes reset the
robot with a pose strategy chosen by the current curriculum *stage*
(`place_robot_for_goal_stage_{0..4}`). The stage/difficulty controls the
distribution of `relative_yaw` labels the module is trained on — e.g. early on the
robot starts roughly facing the goal (labels near 0), and as difficulty rises the
angular error widens and labels span all 36 bins. See **[GIROL.md](GIROL.md) §1**
for the full stage breakdown.

**Metrics.** Orientation accuracy at `<10° / <20° / <30°`, mean angular error, plus
navigation `success_rate`, curriculum `stage`/`mean_radius`. These are:
- **printed to stdout** every 500 steps (`[t] stage=.. success_rate=.. orient_acc(10/20/30)=..`),
- **written to TensorBoard** via skrl (tags `Orient / acc_*`, `Nav / *`, `Curriculum / *`),
- optionally sent to **Comet ML** if `GIROL_USE_COMET=1`.

---

## 4. Running the baseline

```bash
conda activate isaaclab45
cd /home/rizo/work/GIROL
./scripts/algos/launch_orientation_baseline.sh          # 4 envs, 100k steps
# or:
./scripts/algos/launch_orientation_baseline.sh 8 200000 # 8 envs, 200k steps
```

### 100k smoke test (do this first)
Verify the whole pipeline end-to-end on 16 GB before a long run. Uses `num_envs=2`
(lowest VRAM) and 100k timesteps:

```bash
conda activate isaaclab45
cd /home/rizo/work/GIROL
./scripts/algos/smoke_test_orientation.sh        # num_envs=2, 100k steps
./scripts/algos/smoke_test_orientation.sh 4      # bump envs if nvidia-smi shows headroom
```

Success signals:
- Isaac Sim 4.5 starts on the RTX 5080 with no crash.
- Periodic stdout lines: `[<t>] stage=.. success_rate=.. angle_err=.. orient_acc(10/20/30)=..`.
- In TensorBoard, `Orient / acc_10deg` trends upward as training proceeds.

### Watching the training process (TensorBoard)
skrl writes TensorBoard event files during the run. In a **second terminal**:

```bash
conda activate isaaclab45
cd /home/rizo/work/GIROL
tensorboard --logdir logs/skrl/aloha_sac
# open http://localhost:6006
```

Scalars to watch:

| TensorBoard tag                 | Meaning                                              |
|---------------------------------|------------------------------------------------------|
| `Orient / acc_10deg` (20/30)    | Orientation accuracy within 10°/20°/30° — should rise |
| `Nav / success_rate`            | Navigation success (%)                               |
| `Nav / angle_error`             | Current curriculum angular error (rad)               |
| `Curriculum / stage`            | Curriculum stage 0→2 (see GIROL.md §1)               |
| `Curriculum / mean_radius`      | Goal distance the curriculum is sampling (m)         |
| `Loss / *`, `Reward / *`        | SAC losses & episode reward (logged by skrl)         |

`tensorboard` ships in the `isaaclab45` env (`requirements.txt`). If port 6006 is
busy, add `--port 6007`. Orientation accuracy is *also* printed to stdout, so you
can follow progress without TensorBoard.

### 16 GB VRAM guidance
Each env renders a camera and runs CLIP image encoding, so `num_envs` is the main
memory knob. On the 16 GB RTX 5080:
- **Start at `num_envs=4`** for the first successful run.
- Watch `nvidia-smi`; if there's headroom, raise to 8 → 16.
- If it OOMs at launch, drop to 2.

### Configuration (environment variables)
All are consumed by [run_sac_ORM.py](scripts/algos/run_sac_ORM.py); the launcher
sets sensible defaults.

| Variable              | Default            | Meaning                                             |
|-----------------------|--------------------|-----------------------------------------------------|
| `GIROL_NUM_ENVS`      | `4` (launcher)     | Parallel envs (VRAM knob)                            |
| `GIROL_TIMESTEPS`     | `100000`           | Training timesteps                                   |
| `GIROL_USE_METRIC`    | `1`                | **1 = metric baseline; 0 = metric-ablation**        |
| `GIROL_USE_GRAPH`     | `0`                | 1 = policy also conditions on `graph_emb`            |
| `GIROL_TASK`          | `Aloha_nav`        | Registered task id                                   |
| `GIROL_EVAL`          | `0`                | 1 = evaluation mode (loads a checkpoint)             |
| `GIROL_HEADLESS`      | `1`                | 1 = no GUI                                           |
| `GIROL_USE_COMET`     | `0`                | 1 = Comet logging (needs your own `COMET_API_KEY`)  |
| `GIROL_LOG_ROOT`      | `logs/skrl`        | Output root                                          |
| `GIROL_RUN_NAME`      | `baseline_metric`  | Run label                                            |
| `GIROL_SEED`          | `42`               | RNG seed                                             |

### Outputs & checkpoints
| What | Where | Cadence |
|------|-------|---------|
| **Aux modules** (GraphEncoder + OrientationModule + their optimizers) | `logs/skrl/aloha_sac/aux_checkpoints/aux_<step>.pt` + `aux_latest.pt` | every `GIROL_AUX_SAVE_INTERVAL` (default **2000**) steps, and once as `aux_final.pt` |
| SAC agent (actor/critic/target/optimizers/preprocessor) | `logs/skrl/aloha_sac/<timestamp>_SAC/checkpoints/` | every 1000 steps (skrl) |
| TensorBoard event files | `logs/skrl/aloha_sac/<timestamp>_SAC/` | live |
| Replay buffer + preprocessor | `logs/skrl/memory/4img_128/` | end of run |

The **orientation module is the deliverable**, and skrl does *not* checkpoint it
(it lives outside the SAC agent), so the runner saves it separately to
`aux_checkpoints/`. Each `aux_*.pt` holds `graph_encoder`, `orient_module`, both
aux optimizers, the `timestep`, and the `use_metric` flag. To reload: rebuild
`GraphEncoder`/`OrientationModule` (the `id_to_name_emb` buffer is rebuilt from
`text_embeddings.pt`), then `load_state_dict` from the file. Change the interval
with `GIROL_AUX_SAVE_INTERVAL=1000` (etc.).

---

## 5. The metric ablation (next step)

Re-run with the metric coordinates hidden — everything else identical:

```bash
GIROL_USE_METRIC=0 GIROL_RUN_NAME=ablation_nometric \
  ./scripts/algos/launch_orientation_baseline.sh 4 100000
```

With `use_metric=0`, `GraphEncoder` zeroes the `x, y, z` dims (dims 3–5) before
encoding, keeping `active`, `is_goal`, and semantic identity. Compare orientation
accuracy against the metric baseline to measure how much heading prediction
depends on explicit metric coordinates vs. semantics + image.

New graph representations (e.g. relational scene-graph edges instead of the
coordinate list) plug into `GraphEncoder._forward_compact` — a dormant richer
path (`_forward_from_json_scenes`, 24-dim OBB node features + relational edges)
already exists as a starting point.

---

## 6. Fixes applied to make the baseline correct

- **Encoder dimension crash (blocker).** The active path previously reshaped the
  graph to a hardcoded `21×24 = 504`, but the env emits `6×22 = 132`. Replaced
  with `_forward_compact`, which parses the true 6-dim layout and infers `M`.
- **Semantic lookup.** Now indexes `id_to_name_emb[object_id]` directly (matches
  the env's `object_id`), instead of the stale/misaligned `name_to_idx`.
- **`use_metric` switch** added to `GraphEncoder` for the ablation.
- **Path fixes.** Corrected the `text_embeddings.pt` path (`aloha` → `aloha_nav`),
  disabled the missing `graphs_dir`, and replaced all hardcoded `/home/xiso/...`
  output paths with repo-relative `logs/` paths.
- **External logging.** Removed the hardcoded third-party Comet API key; Comet is
  now opt-in via `GIROL_USE_COMET=1` + your own `COMET_API_KEY`. TensorBoard is
  always written locally.
- **Env-var configuration.** `run_sac_ORM.py` reads `GIROL_*` env vars so runs
  need no source edits.
