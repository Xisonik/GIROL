# Results — Metric vs Hierarchical Non-Metric Graph (DDQN pipeline)

Comparison of the two graph representations trained through the **DDQN** pipeline
(`run_experiments.py`, task `Aloha_nav_hab_wr`, **4-room** scene). Same everything
except the graph encoder. See [NONMETRIC_GRAPH.md](NONMETRIC_GRAPH.md).

- **metric** — `perception.GraphEncoder`, `include_node_metric: true` (coords in nodes, direction **+ distance** edges, flat)
- **non-metric (hierarchical)** — `perception.HierarchicalGraphEncoder`, `include_node_metric: false` (no coords, **direction-only** edges, object+room hierarchy)

Metric definitions (aux orientation head, 36 bins = 10°/bin):
`acc_strict` = exact bin (≈ within 10°) · `acc_relaxed` = ±1 bin (≈ within 20°) · `mean_error_deg` = mean |heading error| (lower better).

---

## Comparison  (values: `final (best/min)`)

| metric | **metric baseline** | **non-metric (hierarchical)** | winner |
|---|---|---|---|
| orientation **acc_strict** (≈≤10°) | 0.700 (0.708) | **0.790 (0.826)** | non-metric |
| orientation **acc_relaxed** (≈≤20°) | 0.751 (0.760) | **0.816 (0.854)** | non-metric |
| orientation **mean_error_deg** ↓ | 22.1° (21.8°) | **15.9° (13.2°)** | non-metric |
| orientation confidence | 0.567 | **0.668** | non-metric |
| nav success_rate (%) | **81.7 (90.8)** | 63.0 (81.0) | metric |
| avg_episode_length | 33.4 | 52.4 | — |
| curriculum stage reached | 4 | 4 | tie |
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
