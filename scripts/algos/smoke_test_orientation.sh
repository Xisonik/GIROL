#!/usr/bin/env bash
#
# 100k-timestep SMOKE TEST for the orientation-module baseline.
#
# Purpose: verify the whole pipeline works end-to-end on the 16 GB GPU before
# committing to a long run — Isaac Sim 4.5 launches on the RTX 5080, the scene
# loads, SAC collects rollouts, the aux trainer updates GraphEncoder +
# OrientationModule, and orientation accuracy climbs. Uses the *metric* graph.
#
# It runs with num_envs=2 (lowest VRAM) so it launches even with little headroom.
#
# Usage:
#   ./scripts/algos/smoke_test_orientation.sh [NUM_ENVS]
#     NUM_ENVS default 2. Bump to 4 if `nvidia-smi` shows headroom.
#
# Watch it (in a second terminal):
#   conda activate isaaclab45
#   tensorboard --logdir logs/skrl/aloha_sac      # then open http://localhost:6006
#
set -Eeuo pipefail

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

export GIROL_NUM_ENVS="${1:-2}"
export GIROL_TIMESTEPS="100000"       # 100k smoke run
export GIROL_TASK="Aloha_nav"
export GIROL_USE_METRIC="1"           # metric graph baseline
export GIROL_USE_GRAPH="0"
export GIROL_EVAL="0"
export GIROL_VIDEO="0"
export GIROL_USE_PRETRAINED="0"
export GIROL_HEADLESS="1"
export GIROL_USE_COMET="0"
export GIROL_LOG_ROOT="logs/skrl"
export GIROL_RUN_NAME="smoke_metric"
export GIROL_SEED="42"

# Env-side: turn-to-face-goal task (success = facing goal within 20°) + full
# 0->1->2->3->4 success-gated curriculum. See BASELINE.md / GIROL.md.
export ALOHA_NAV_ENV_CFG='{"TURN_TASK": true, "MAX_STAGE": 4}'

export PYTHONNOUSERSITE="1"

echo "=============================================================="
echo " SMOKE TEST — orientation baseline (metric graph)"
echo "   num_envs   = $GIROL_NUM_ENVS"
echo "   timesteps  = $GIROL_TIMESTEPS"
echo "   logs       = $GIROL_LOG_ROOT/aloha_sac/"
echo "   watch with : tensorboard --logdir $GIROL_LOG_ROOT/aloha_sac"
echo "=============================================================="
echo " Success signals:"
echo "   * sim window / 'Simulation is stopped' free startup, no crash"
echo "   * periodic lines: [<t>] stage=.. success_rate=.. orient_acc(10/20/30)=.."
echo "   * in TensorBoard: 'Orient / acc_10deg' trending up over time"
echo "=============================================================="

exec ./isaaclab.sh -p scripts/algos/run_sac_ORM.py
