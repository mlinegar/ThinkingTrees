#!/usr/bin/env bash
# Day-2 FNO leaf-size axis (CPU): full-grid leaves 16,8,4,2 with
# EmbeddingGemma, matching the DiffusionGemma leaf grid. Waits for the
# in-flight FNO leaf=1 full run to release the CPU first. Cheap cells first
# so results land incrementally (leaf=16 ~1h ... leaf=2 ~6h).
#
# Launch:
#   ./venv/bin/python scripts/long_job.py launch \
#     --name day2_fno_leafgrid \
#     --job-root outputs/day2_fno_leafgrid_launcher \
#     --cwd /home/mlinegar/ThinkingTrees --replace-existing \
#     -- bash scripts/run_day2_fno_leafgrid.sh
set -uo pipefail
cd /home/mlinegar/ThinkingTrees

log() { echo "[day2-fno $(date -u +%H:%M:%S)] $*"; }

log "waiting for FNO full leaf=1 run to finish (CPU contention)"
while pgrep -f "manifesto_qsentence_fno_embeddinggemma_full --verbose" >/dev/null 2>&1; do
  sleep 120
done
log "CPU free; running FNO full-grid leaves 16,8,4,2"

./venv/bin/python scripts/run_manifesto_qsentence_dspy_ladder.py \
  --family fno \
  --embedding-backend local-hf \
  --embedding-model /mnt/data/models/google/embeddinggemma-300m \
  --embedding-device cpu \
  --embedding-batch-size 64 \
  --fg-grid-dir outputs/manifesto_qsentence_dspy_labeled_grid \
  --leaf-qsentences "16,8,4,2" \
  --max-iterations 2 \
  --fno-epochs 8 \
  --fno-batch-size 16 \
  --fno-learning-rate 3e-3 \
  --output-dir outputs/manifesto_qsentence_fno_embeddinggemma_full_leafgrid \
  --verbose \
  || log "WARN: FNO leaf-grid run failed"

log "day-2 FNO leaf grid complete"
