#!/usr/bin/env bash
set -euo pipefail

ROOT="${1:-outputs/tree_neural_speed_overnight_$(date -u +%Y%m%d_%H%M%S)}"
QUICK_PROBE_ROOT="${QUICK_PROBE_ROOT:-outputs/teacher_first_speed_probe_1024_20260323_rapid}"
QUICK_PROBE_TIMEOUT_S="${QUICK_PROBE_TIMEOUT_S:-1200}"
LEGACY_CONFIRM_UNIT="${LEGACY_CONFIRM_UNIT:-tree-neural-teacher-first-halving-resume-20260322_174231.service}"
LEGACY_5K_UNIT="${LEGACY_5K_UNIT:-tree-neural-teacher-first-5k-followon-20260322_220350.service}"

mkdir -p "$ROOT"
cd "$(dirname "$0")/.."
source venv/bin/activate

timestamp() {
  date -u --iso-8601=seconds
}

probe_pattern="scripts/run_tree_neural_teacher_first_push.py --output-root ${QUICK_PROBE_ROOT}"
legacy_scaling_pattern="tree_neural_teacher_first_halving_overnight_20260322_084727"
legacy_followon_pattern="tree_neural_teacher_first_5k_followon_20260322_220350"

echo "[$(timestamp)] Overnight speed run root: $ROOT"
echo "[$(timestamp)] Waiting on quick probe: $QUICK_PROBE_ROOT"

probe_wait_s=0
while pgrep -af "$probe_pattern" >/dev/null 2>&1; do
  if [ "$probe_wait_s" -ge "$QUICK_PROBE_TIMEOUT_S" ]; then
    echo "[$(timestamp)] Quick probe exceeded timeout ${QUICK_PROBE_TIMEOUT_S}s; terminating it so overnight work can proceed"
    pkill -TERM -f "$probe_pattern" || true
    sleep 10
    pkill -KILL -f "$probe_pattern" || true
    break
  fi
  sleep 60
  probe_wait_s=$((probe_wait_s + 60))
  echo "[$(timestamp)] Quick probe still active after ${probe_wait_s}s"
done

echo "[$(timestamp)] Stopping legacy queued follow-on service if present: $LEGACY_5K_UNIT"
systemctl --user stop "$LEGACY_5K_UNIT" >/dev/null 2>&1 || true
echo "[$(timestamp)] Stopping legacy confirmation service if present: $LEGACY_CONFIRM_UNIT"
systemctl --user stop "$LEGACY_CONFIRM_UNIT" >/dev/null 2>&1 || true

pkill -f "$legacy_scaling_pattern" || true
pkill -f "$legacy_followon_pattern" || true
sleep 5

echo "[$(timestamp)] Starting main async scaling sweep"
python3 scripts/run_tree_neural_teacher_first_scaling_push.py \
  --output-root "$ROOT/main_scaling_1024_2048" \
  --benchmark recoverable_v4 \
  --train-doc-counts 1024 2048 \
  --phase1-seeds 0 1 \
  --phase2-seeds 0 \
  --stage1-epochs 6 \
  --stage1-rung-epochs 2 4 6 \
  --stage1-rung-promote-k 3 2 \
  --stage1-screen-metric val_root_mae \
  --stage2-epochs 6 \
  --stage2-epochs-by-count 1024:4 2048:6 \
  --stage2-survivors-by-count 1024:2 2048:2 \
  --async-promote-per-count \
  --group-stage2-conditions \
  --torch-threads 1 \
  --use-cuda

if [ ! -f "$ROOT/main_scaling_1024_2048/teacher_first_scaling_summary.json" ]; then
  echo "[$(timestamp)] Main scaling summary was not written; refusing to launch 5k follow-on"
  exit 1
fi

echo "[$(timestamp)] Starting focused 5k follow-on"
python3 scripts/run_tree_neural_teacher_first_push.py \
  --output-root "$ROOT/focused_5000" \
  --benchmark recoverable_v4 \
  --phase1-train-docs 5000 \
  --phase2-train-docs 5000 \
  --phase1-seeds 0 \
  --phase2-seeds 0 \
  --surrogate-labels \
    teacherfirst_shared_feature_adapters_phi128 \
    teacherfirst_shared_feature_phi192 \
    teacherfirst_scorefiber_s1_f15 \
    teacherfirst_scorefiber_s1_f31 \
  --root-search-labels \
    teacherfirst_shared_feature_phi192 \
    teacherfirst_scorefiber_s1_f15 \
    teacherfirst_scorefiber_s1_f31 \
  --stage1-root-weight-grid 0.5 \
  --promote-top-k 1 \
  --stage1-epochs 6 \
  --stage2-epochs 4 \
  --tree-stage1-eval-mode end_only \
  --tree-stage1-screen-doc-limit 128 \
  --tree-stage1-final-exact-doc-limit 32 \
  --group-stage2-conditions \
  --torch-threads 1 \
  --use-cuda

echo "[$(timestamp)] Overnight speed run complete"
