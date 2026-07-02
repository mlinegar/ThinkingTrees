#!/usr/bin/env bash
# Second per-dim HPO chain: eu + decentralization.
# Waits for the immigration/environment chain to free the GPUs, then runs the
# same tune-on-val + 3-seed-test recipe.
#
# Rationale from the diagnosis (outputs/benoit_6dim_fno/FINAL_SUMMARY.md + ceiling check):
#   * eu: LLM leaf-score rollup ceiling = 0.78 but FNO only got 0.23 -> BIGGEST gap;
#     learned-g is failing to recover diffuse leaf signal. HPO is the legit lever
#     (we deliberately do NOT hard-code mean_rollup merges -- that would make g a
#      fixed mean, not a learned composition).
#   * decentralization: leaf-score ceiling is only 0.46 (LLM genuinely can't score
#     federalism per-sentence) -> SCORE-limited, report honestly; HPO unlikely to
#     beat ~0.46 and that's the finding, not a failure.
set -uo pipefail
cd /home/mlinegar/ThinkingTrees
PY=./venv/bin/python
LOG() { echo "[$(date +%H:%M:%S)] $*"; }

LOG "Waiting for the immigration/environment HPO chain to finish..."
while systemctl --user is-active --quiet 'codex-long-job-*perdim_hpo_chain.service' 2>/dev/null; do sleep 60; done
LOG "Prior chain done. Starting eu + decentralization."

DIMS=(eu decentralization)
N_TRIALS=48
SEEDS=(101 202 303)

for dim in "${DIMS[@]}"; do
  GRID=outputs/benoit_chunkgrid_forced_${dim}_llmspan
  [ -d "$GRID" ] || { LOG "SKIP $dim (no grid)"; continue; }
  OUT=outputs/hpo_${dim}_chunkfno

  LOG "=== $dim: HPO ($N_TRIALS trials, tune on VAL, 4 GPUs) ==="
  HPO_DIM=$dim HPO_GRID=$GRID $PY scripts/hpo_econ_chunkfno.py \
    --n-trials $N_TRIALS --gpus 0,1,2,3 --eval-split val \
    --out-root "$OUT" > outputs/hpo_${dim}.log 2>&1
  LOG "$dim: HPO done -> $OUT/best.json"

  best="$OUT/best.json"
  [ -f "$best" ] || { LOG "$dim: no best.json, skipping seed confirm"; continue; }
  read -r NM HC NL HH EP LR WD LW MW RW < <($PY - "$best" <<'PYEOF'
import json,sys
p=json.load(open(sys.argv[1]))["params"]
print(p["n_modes"],p["hidden"],p["layers"],p["head"],p["epochs"],
      f'{p["lr"]:.6f}',f'{p["wd"]:.7f}',f'{p["lw"]:.3f}',f'{p["mw"]:.3f}',f'{p["rw"]:.3f}')
PYEOF
)
  LOG "$dim: winner modes=$NM hid=$HC layers=$NL head=$HH ep=$EP lr=$LR wd=$WD lw=$LW mw=$MW rw=$RW"

  for idx in 0 1 2; do
    s=${SEEDS[$idx]}; g=$idx
    out=outputs/benoit_6dim_fno_hpo/${dim}/seed_${s}
    mkdir -p "$(dirname "$out")"
    CUDA_VISIBLE_DEVICES=$g TT_EXPORT_FULL_TREE_TRACES=0 \
      $PY scripts/run_manifesto_qsentence_dspy_ladder.py \
      --family fno --embedding-backend local-hf \
      --embedding-model /mnt/data/models/google/embeddinggemma-300m \
      --embedding-device cuda --embedding-batch-size 64 --fno-device cuda \
      --fg-grid-dir "$GRID" --leaf-qsentences 16 --max-iterations 2 \
      --fno-target-dimension "$dim" --eval-split test \
      --fno-n-modes "$NM" --fno-hidden-channels "$HC" --fno-n-layers "$NL" \
      --fno-head-hidden-dim "$HH" --fno-epochs "$EP" --fno-learning-rate "$LR" \
      --fno-weight-decay "$WD" --fno-leaf-weight "$LW" --fno-merge-weight "$MW" \
      --fno-root-weight "$RW" --fno-seed "$s" \
      --output-dir "$out" > "${out}.log" 2>&1 &
  done
  wait
  LOG "$dim: 3-seed TEST confirm done -> outputs/benoit_6dim_fno_hpo/${dim}/"
done

LOG "PERDIM_HPO_EU_DEC_COMPLETE"
