#!/usr/bin/env bash
#
# Launch BASELINE orientation-module training (metric graph) on GIROL.
#
# The orientation module is trained as a supervised side-task on top of SAC
# navigation rollouts (see BASELINE.md). This baseline uses the *metric* graph
# representation: each object node carries its (x, y, z) position. The later
# ablation re-runs the same command with GIROL_USE_METRIC=0.
#
# Usage:
#   ./scripts/algos/launch_orientation_baseline.sh [NUM_ENVS] [TIMESTEPS] [gui]
#
# Examples:
#   ./scripts/algos/launch_orientation_baseline.sh            # 4 envs, 100k, headless
#   ./scripts/algos/launch_orientation_baseline.sh 8 200000   # 8 envs, 200k, headless
#   ./scripts/algos/launch_orientation_baseline.sh 2 100000 gui   # open the Isaac Sim window
#
# Pass 'gui' as the 3rd arg to open the Isaac Sim viewport and watch training live.
# The GUI adds VRAM + slows training a lot, so use FEW envs (2-4) when watching.
#
# Prereqs (once):
#   conda activate isaaclab45
#   assets installed, data/all_paths.json present, custom skrl symlinked.
#
set -Eeuo pipefail

# Resolve repo root (two levels up from scripts/algos/) and run from there so
# all relative paths (source/..., data/all_paths.json, logs/...) resolve.
REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

# ---- Run configuration (env vars consumed by run_sac_ORM.py) --------------
# num_envs is the main VRAM knob. Each env renders a camera + runs CLIP image
# encoding, so on a 16 GB GPU keep it small. Start at 4; if it launches with
# headroom in `nvidia-smi`, raise to 8/16. If it OOMs, drop to 2.
export GIROL_NUM_ENVS="${1:-4}"
export GIROL_TIMESTEPS="${2:-100000}"
MODE="${3:-headless}"              # 'gui' -> open the Isaac Sim window; anything else -> headless

export GIROL_TASK="Aloha_nav"      # registered id (see aloha_nav/__init__.py)
export GIROL_USE_METRIC="1"        # BASELINE: metric graph (x,y,z kept). Ablation -> 0
export GIROL_USE_GRAPH="0"         # policy uses gt-orientation+goal; graph feeds orient head
export GIROL_EVAL="0"
export GIROL_VIDEO="0"
export GIROL_USE_PRETRAINED="0"
if [ "$MODE" = "gui" ]; then
    export GIROL_HEADLESS="0"      # open the Isaac Sim viewport (watch training live)
else
    export GIROL_HEADLESS="1"      # headless (fast training, no window)
fi
export GIROL_USE_COMET="0"         # no external logging; TensorBoard is written locally
export GIROL_LOG_ROOT="logs/skrl"
export GIROL_RUN_NAME="baseline_metric"
export GIROL_SEED="42"

# Environment-side config (read by the env at process start via ALOHA_NAV_ENV_CFG):
#   TURN_TASK=true  -> "turn in place to face the goal" task; success = facing goal
#                      within 20° (no distance). This makes the curriculum stage gate
#                      (success_rate) mean "the robot oriented to the goal".
#   MAX_STAGE=4      -> success-gated curriculum walks 0->1->2->3->4.
export ALOHA_NAV_ENV_CFG='{"TURN_TASK": true, "MAX_STAGE": 4}'

# Keep the conda env hermetic (avoid ~/.local site-packages leaking in).
export PYTHONNOUSERSITE="1"

echo "[baseline] task=$GIROL_TASK num_envs=$GIROL_NUM_ENVS timesteps=$GIROL_TIMESTEPS headless=$GIROL_HEADLESS"
echo "[baseline] use_metric=$GIROL_USE_METRIC use_graph=$GIROL_USE_GRAPH log_root=$GIROL_LOG_ROOT"
echo "[baseline] TensorBoard: tensorboard --logdir $GIROL_LOG_ROOT/aloha_sac"
[ "$MODE" = "gui" ] && echo "[baseline] GUI mode: Isaac Sim window will open (slower; use few envs)"

exec ./isaaclab.sh -p scripts/algos/run_sac_ORM.py
