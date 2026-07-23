# GRAPHS.md — Scene-graph representations & their encoders

Three graph representations feed the DDQN pipeline, all consuming the **same raw
observation** and all producing a **128-d graph embedding** for the Q-network + orientation aux head.
They differ only in *what they keep* from the scene and *how they structure it*.

| | **1. Metric** | **2. Hierarchical (non-metric)** | **3. Flat non-metric (teammate)** |
|---|---|---|---|
| encoder class | `perception.graph_encoder.GraphEncoder` | `perception.hierarchical_graph_encoder.HierarchicalGraphEncoder` | `perception.graph_encoder.GraphEncoder` |
| key flag | `include_node_metric: true` | (own class) `include_node_metric: false` | `include_node_metric: false` |
| node structure | 28 flat object nodes | object nodes **+ 4 room nodes** | 28 flat object nodes |
| coordinates in nodes | **yes** (x,y,z) | no | no |
| edge features | direction (6) **+ distance (4)** | direction only (3-way ×2 + room-rel) | direction (6) **+ distance (4)** |
| distance kept | **yes** (quantized buckets) | **no** | **yes** (quantized buckets) |
| room hierarchy | no | **yes** (quadrant rooms + zones) | no |
| GNN | 2× GATv2, 2 heads | 3× GATv2, 4 heads | 2× GATv2, 2 heads |
| readout | mean ⊕ max ⊕ attention pool | goal ⊕ goal-room ⊕ global mean | mean ⊕ max ⊕ attention pool |
| author | Nikita | Rizo | Nikita |

> **The clean control:** #1 and #3 are the *same encoder* — they differ by exactly **one flag**
> (`include_node_metric`). That is the tightest metric-vs-non-metric comparison. #2 is a more
> aggressive redesign (adds rooms, drops distance entirely).

---

## Shared raw contract (all three read this)

```
graph_flat : [B, 28 * 6]
per object : [ object_id, active, is_goal, x_env, y_env, z_env ]
```

- `object_id` → CLIP name lookup.
- `active` → topology / padding mask; **never a learned node feature**.
- `is_goal` → picks the goal node (+ optional learned feature).
- `x,y,z` → always used to *derive edges*; only fed as node features when `include_node_metric = true`.

Every representation below starts from this identical tensor. What changes is the **information
architecture** built on top of it.

---

## 1. Metric graph + encoder

`perception.graph_encoder.GraphEncoder`, `include_node_metric: true`

### Structure
```
        goal ──dir+dist──▶ obj₁
         │  ╲──dir+dist──▶ obj₂
         │   ╲─dir+dist──▶ obj₃ ...        (goal-star: 2 parallel edges/obj — X-axis + Y-axis)
     28 flat object nodes, no rooms
```

### Node features  (each projected to 32-d, then concatenated → MLP → 128)
| block | source | on/off |
|---|---|---|
| `name_embedding` | CLIP name vector → 128 → 32 | always |
| `xyz_metric_embedding` | raw (x,y,z) → 32 → 32 | **ON** (this is what makes it "metric") |
| `is_goal_embedding` | is_goal → 32 | off in our base |

### Edge features  (`edge_dim = 10`, fixed one-hot)
- **direction (6):** `same · in_front_of · behind · left_of · right_of · self`
- **distance (4 buckets, per axis):** `very_close <5m · close <10m · far <15m · very_far ≤20m`
- topology: `goal_star` (goal → every active non-goal object, 2 edges: Y-relation + X-relation) — or `complete`.

### Encoder
`node_mlp (→128) → 2× GATv2Conv(heads=2, edge_dim=10, LayerNorm) → readout`
**Readout:** `concat[ mean_pool, max_pool, attention_pool ] (384) → 128 → 128`.

**What it "knows":** absolute coordinates (in nodes) **and** quantized distance + direction (on edges).

---

## 2. Hierarchical graph + encoder  (my non-metric)

`perception.hierarchical_graph_encoder.HierarchicalGraphEncoder`, `include_node_metric: false`

### Structure
```
                 ┌── Room R/F ──┐   ┌── Room L/F ──┐
   (containment) │  obj  obj    │   │  obj  obj    │  (room ↔ room, all pairs, direction-labelled)
                 └──────┬───────┘   └──────┬───────┘
        goal ★ ─direction─▶ every object (goal-star, both directions)
   object nodes  +  4 quadrant room nodes   (room = (x<0) + 2·(y<0))
```

### Node features
| node | features |
|---|---|
| **object** | `name_embedding` (CLIP → 128 → 16) · `active` · `is_goal` |
| **room** (×4) | `is_goal_room` · x-zone one-hot (LEFT/RIGHT) · y-zone one-hot (FRONT/BACK) |

### Edge features  (`edge_dim = 20`, embedded — **direction only, no distance**)
Four categorical fields → embeddings `8 + 4 + 4 + 4`:
- `relation (5)`: self · goal-star · obj→room · room→obj · room↔room
- `x_dir (4)`: none · LEFT · aligned · RIGHT   (3-way at ±0.4 m)
- `y_dir (4)`: none · BACK · aligned · FRONT
- `room_relation (3)`: none · SAME room · DIFFERENT room

Edge types: **goal-star** (goal↔object) + **containment** (obj↔room) + **room↔room** (all pairs) + **self-loops**.

### Encoder
`object_mlp / room_mlp (→128) → 3× GATv2Conv(heads=4, edge_dim=20, LayerNorm) → goal-centric readout`
**Readout:** `concat[ h_goal, h_goal_room, h_global(mean) ] (384) → 256 → 128`.

**What it "knows":** which *room* an object is in, and which *way* it lies (L/R, F/B, same/other room).
No coordinates, **no distance** — the most purely qualitative of the three.

*(Interactive anatomy: the published artifact "What the non-metric graph actually looks like".)*

---

## 3. Flat non-metric graph + encoder  (teammate's)

`perception.graph_encoder.GraphEncoder`, `include_node_metric: false`
— **the same class as #1 with the metric flag off.**

### Structure
Identical topology to the metric graph (28 flat nodes, goal-star or `complete`), but nodes lose
their coordinates:
```
        goal ──dir+dist──▶ obj₁ ...     (same edges as metric)
     28 flat object nodes  ·  name-only  ·  no rooms  ·  no xyz in nodes
```

### Node features
| block | source | on/off |
|---|---|---|
| `name_embedding` | CLIP name vector → 128 → 32 | always |
| `xyz_metric_embedding` | — | **OFF** |
| `is_goal_embedding` | is_goal → 32 | off in our base |

### Edge features
**Identical to metric** — direction (6) + distance buckets (4) = `edge_dim 10`. Distance is **kept**.

### Encoder
Identical architecture to #1 (`2× GATv2, heads=2`, mean⊕max⊕attention readout); only the node input
is narrower (no position block).

**What it "knows":** object identities + quantized distance + direction. It removes only the
*coordinates in the nodes* — it is therefore "coordinate-free in the nodes but still distance-aware on
the edges," i.e. **less aggressively non-metric than #2**.

> The teammate's own `configs/cur_dqn/ddqn_discrete.json` runs this encoder with `edge_mode: complete`.
> For a matched comparison against #1 and #2 we keep `edge_mode: goal_star` (inherited from `base.json`),
> so #1 and #3 differ by the single `include_node_metric` flag.

---

## How they're wired (configs)

All three live in `scripts/algos/configs/metric_vs_nonmetric/` and share `base.json`
(task `Aloha_nav_hab_wr`, `num_envs 24`, `timesteps 300000`, seed 42, orientation aux head).

| run file | representation | override on top of `base.json` |
|---|---|---|
| `metric.json` | **1. Metric** | `include_node_metric: true` |
| `hierarchical.json` | **2. Hierarchical** | `class_path → HierarchicalGraphEncoder`, `include_node_metric: false` |
| `nonmetric_flat.json` | **3. Flat non-metric** | `include_node_metric: false` |

Sorted run index (for `--start/--end`): `hierarchical = 0`, `metric = 1`, `nonmetric_flat = 2`.

### Launch commands
Teammate's flat non-metric only:
```bash
PYTHONNOUSERSITE=1 ./isaaclab.sh -p scripts/algos/run_experiments.py \
  --configs-dir scripts/algos/configs/metric_vs_nonmetric --start 2 --end 3
```
All three, matched, back-to-back (full sweep for RESULTS.md):
```bash
PYTHONNOUSERSITE=1 ./isaaclab.sh -p scripts/algos/run_experiments.py \
  --configs-dir scripts/algos/configs/metric_vs_nonmetric
```

After training, hand me the run dirs and I'll extract the scalars
(`aux/orient/acc_strict`, `aux/orient/acc_relaxed`, `aux/orient/mean_error_deg`, `env/success_rate`)
into the 3-way table in [RESULTS.md](RESULTS.md).

---

## See also
- [RESULTS.md](RESULTS.md) — trained metrics & comparison table.
- [NONMETRIC_GRAPH.md](NONMETRIC_GRAPH.md) — deep dive on the hierarchical (#2) representation.
