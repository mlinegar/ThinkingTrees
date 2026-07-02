#!/usr/bin/env bash
# One-shot: switch to overnight v2 (all GPUs per stage + level-wave batched eval).
#   1. stop the v1 orchestrator (still parked in step 0)
#   2. stop the small leaf=1 run (iter-0/1 metrics already on disk; its iter-2
#      eval was hours from finishing under the old node-serial walk)
#   3. restart the FNO full-grid job on GPU 2
#   4. launch orchestrator v2
set -uo pipefail
cd /home/mlinegar/ThinkingTrees
PY=./venv/bin/python

$PY scripts/long_job.py stop --job-root outputs/overnight_substrate_comparison_launcher || true
$PY scripts/long_job.py stop --job-root outputs/manifesto_qsentence_diffusiongemma_small_launcher || true
sleep 10

bash scripts/restart_fno_full_on_gpu2.sh || echo "WARN: FNO GPU restart failed (CPU job may still be running)"

$PY scripts/long_job.py launch \
  --name overnight_substrate_comparison \
  --job-root outputs/overnight_substrate_comparison_launcher \
  --cwd /home/mlinegar/ThinkingTrees \
  --replace-existing \
  -- bash scripts/run_overnight_substrate_comparison_v2.sh
echo "overnight v2 launched"
