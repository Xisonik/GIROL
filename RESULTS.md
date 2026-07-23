# Results — Metric vs Hierarchical Non-Metric Graph (DDQN pipeline)

> ⚠️ **STALE SCENE — kept for reference.** The numbers in the **Comparison** table below were produced on
> the **old 4-room scene** (23 objects, old object set) *before* merging the teammate's `main`, and were
> **fixed at curriculum stage 4** (old curriculum). The current scene is now **3 rooms** (office/utility/
> bathroom, **28 objects**) and the live runs **progress stages 0→1→2** (teammate's curriculum). So the
> stale table and the live table differ in **both scene and stage regime** and are **not comparable** —
> see the live results directly below.

Comparison of the two graph representations trained through the **DDQN** pipeline
(`run_experiments.py`, task `Aloha_nav_hab_wr`, **4-room** scene at the time). Same everything
except the graph encoder. See [NONMETRIC_GRAPH.md](NONMETRIC_GRAPH.md).

- **metric** — `perception.GraphEncoder`, `include_node_metric: true` (coords in nodes, direction **+ distance** edges, flat)
- **non-metric (hierarchical)** — `perception.HierarchicalGraphEncoder`, `include_node_metric: false` (no coords, **direction-only** edges, object+room hierarchy)

Metric definitions (aux orientation head, 36 bins = 10°/bin):
`acc_strict` = exact bin (≈ within 10°) · `acc_relaxed` = ±1 bin (≈ within 20°) · `mean_error_deg` = mean |heading error| (lower better).

---

## Current 3-room scene — live runs  (Jul 23)

Runs on the **current 3-room / 28-object scene** through the **teammate's curriculum**, which
**progresses stages 0 → 1 → 2** (0→1 time-gated at 2000 steps, 1→2 when spawn radius ≥ 6 m, then
continuous success-gated difficulty). This is a **different regime** from the two runs in the *stale*
Comparison table below, which were **fixed at stage 4** on the old 4-room scene — so the two tables are
**not directly comparable** (different scene **and** different curriculum/stage).

| metric | **flat non-metric (teammate)** | metric | hierarchical |
|---|---|---|---|
| orientation **acc_strict** (≈≤10°) | 0.524  (best 0.609 @70k) | *pending re-run* | *pending re-run* |
| orientation **acc_relaxed** (≈≤20°) | 0.698  (best 0.784 @70k) | — | — |
| orientation **mean_error_deg** ↓ | 26.8°  (best 17.8° @70k) | — | — |
| orientation confidence | 0.447  (best 0.516) | — | — |
| nav success_rate (policy-only) | 28%  @ r≈5.5 m, stage 2 | — | — |
| nav all_success_rate (w/ controller) | 64.6% | — | — |
| avg_episode_length | 51.1 | — | — |
| **curriculum stage reached** | **0 → 1 → 2 (progressing)** | — | — |
| Q-network loss ↓ | 0.201 (min 0.083) | — | — |
| total reward (mean) | 1.47 (best 8.78 @122k) | — | — |
| num_envs / steps | 24 / 275,000 | — | — |

⚠ **Orientation peaked early, then regressed.** Best `acc_strict` **0.609 @ 70k** decayed to **0.524 @ 275k**
(mean error 17.8° → 26.8°). Most likely the curriculum ramping the spawn radius/difficulty (into stage 2,
r → 5.5–6.5 m) made heading harder as training progressed, and/or the orientation head is coupled into the
policy loop (`use_orientation_module: true`). Treat the **best** column as "what it can reach", the **final**
as "where it settled under stage-2 difficulty."

**Metric + hierarchical on this scene are not yet re-run.** Launch them with the same command
(`--configs-dir scripts/algos/configs/metric_vs_nonmetric`) so all three share this scene + curriculum,
then this becomes a clean 3-way table.

- run dir: `logs/skrl/Aloha_nav_hab_wr/07.23_05-45-11_metric_vs_nonmetric/nonmetric_flat/`
- encoder: `perception.GraphEncoder`, `include_node_metric: false`, `edge_mode: goal_star` · seed 42

---

## Comparison  (STALE — old 4-room scene, stage-4-only)  ·  values: `final (best/min)`

| metric | **metric baseline** | **non-metric (hierarchical)** | winner |
|---|---|---|---|
| orientation **acc_strict** (≈≤10°) | 0.700 (0.708) | **0.790 (0.826)** | non-metric |
| orientation **acc_relaxed** (≈≤20°) | 0.751 (0.760) | **0.816 (0.854)** | non-metric |
| orientation **mean_error_deg** ↓ | 22.1° (21.8°) | **15.9° (13.2°)** | non-metric |
| orientation confidence | 0.567 | **0.668** | non-metric |
| nav success_rate (%) | **81.7 (90.8)** | 63.0 (81.0) | metric |
| avg_episode_length | 33.4 | 52.4 | — |
| curriculum stage reached | 4 (fixed) | 4 (fixed) | tie |
| Q-network loss ↓ | 0.31 (0.17) | 0.31 (0.15) | tie |
| total reward (mean) | 3.33 (7.0) | 5.80 (7.45) | — |
| **num_envs** | **24** | **10** | ⚠ differ |
| **training steps** | **249,750** | **178,400** | ⚠ differ |

---

## ⚠ This is NOT yet a clean comparison
The two runs used **different `num_envs` (24 vs 10)** and **different step counts (249.7k vs 178.4k)** →
the metric run saw **~3× more environment data**. So:
- The **orientation** result is a *strong* signal **for** the non-metric graph: it reaches **higher accuracy
  and lower error despite ~3× less data**. Hard to explain away by training budget.
- The **nav success** gap (metric higher) is **confounded** — metric simply trained on far more data. Not a fair nav comparison.

**To make it publishable, rerun both with identical `num_envs` and to the same plateau** (e.g. both 24 envs, both to ~250k or until `acc_strict` flattens), then re-read.

Still open: the **img-only** control (zero the graph contribution). Metric ≈/< non-metric doesn't prove the
graph carries spatial info unless img-only is clearly worse than both.

---

## Reading (with the caveat above)
- **Orientation:** the hierarchical **non-metric graph beats the metric graph** — `acc_strict` 0.79 vs 0.70,
  mean error ~16° vs ~22° — and does so with fewer envs/steps. Qualitative direction + room hierarchy appears
  *at least as informative* as raw coordinates + distance for heading prediction.
- **Navigation:** metric shows higher success (81.7% vs 63%), but that's the run with 3× the data — treat as inconclusive.

---

## Run provenance
| | metric | non-metric (hierarchical) |
|---|---|---|
| run dir | `logs/skrl/Aloha_nav_hab_wr/07.21_20-50-59_metric_vs_nonmetric/metric/` | `logs/skrl/Aloha_nav_hab_wr/07.21_15-16-25_metric_vs_nonmetric/hierarchical/` |
| encoder | `perception.GraphEncoder` (metric, flat) | `perception.HierarchicalGraphEncoder` (non-metric, hierarchical) |
| num_envs / steps | 24 / 249,750 | 10 / 178,400 |
| seed | 42 | 42 |

Extract command (for re-reads): read `.../{run}/tensorboard/` scalars
(`aux/orient/acc_strict`, `aux/orient/acc_relaxed`, `aux/orient/mean_error_deg`, `env/success_rate`).

---

## Re-run (on the current 3-room / 28-object scene) — now **3-way**
The comparison now has **three** representations (see [GRAPHS.md](GRAPHS.md) for the architectures):

| run file | representation | encoder |
|---|---|---|
| `metric.json` | **metric** | `GraphEncoder`, `include_node_metric: true` |
| `hierarchical.json` | **hierarchical non-metric** (mine) | `HierarchicalGraphEncoder` |
| `nonmetric_flat.json` | **flat non-metric** (teammate) | `GraphEncoder`, `include_node_metric: false` |

`metric` and `nonmetric_flat` are the same encoder toggled by one flag → the clean control.
All share `base.json` (`num_envs=24`, `timesteps=300000`, seed 42). Run all three, matched:
```bash
PYTHONNOUSERSITE=1 ./isaaclab.sh -p scripts/algos/run_experiments.py \
  --configs-dir scripts/algos/configs/metric_vs_nonmetric
```
Or just the teammate's flat non-metric (index 2 — sorted `hierarchical=0, metric=1, nonmetric_flat=2`):
```bash
PYTHONNOUSERSITE=1 ./isaaclab.sh -p scripts/algos/run_experiments.py \
  --configs-dir scripts/algos/configs/metric_vs_nonmetric --start 2 --end 3
```
When they finish, tell me the run dirs and I'll extract the new scalars and add a fresh 3-way table above.
