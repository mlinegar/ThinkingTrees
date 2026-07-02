#!/usr/bin/env bash
# Per-dim HPO chain for the mid-tier Benoit dims that transfer left below ceiling.
# Grids + chunk scores already exist (built by the overnight 6-dim chain), so this
# is pure FNO HPO on the idle GPUs — no LLM scoring, no server fleet needed.
#
#   For each dim: 48-trial Optuna (TPE) tuned on VAL across 4 GPUs, then re-run the
#   winning config on 3 seeds on the TEST split for an honest mean ± std.
#
# eu/decentralization are deliberately EXCLUDED — they floor at ~0.23 from signal,
# not from tuning (decentralization has full 217-doc coverage and still floors).
#
# Self-driving; launch under long_job.py so it survives session close.
set -uo pipefail
cd /home/mlinegar/ThinkingTrees
PY=./venv/bin/python
LOG() { echo "[$(date +%H:%M:%S)] $*"; }

DIMS=(immigration environment)
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

  # extract winning params and confirm on 3 seeds / TEST
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

LOG "PERDIM_HPO_COMPLETE"
