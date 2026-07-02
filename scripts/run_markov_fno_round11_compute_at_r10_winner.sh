#!/usr/bin/env bash
# Round 11: compute scaling at the R10 winner architecture.
#
# R9 saturated at n_iter=1500 at the R7 architecture (m32/l3/sum).
# R10 winner is wider+deeper (m64/l5/mean, count_mae=0.0069 at iter=1500).
# Open question: does the new architecture saturate at the same iter
# count, or does more compute help at higher capacity?
#
# Single cell (~3.2h on a clean GPU; ~2.9 sec/iter from R10 timings).
set -euo pipefail
source venv/bin/activate

STAMP="${STAMP:-$(date -u +%Y%m%d_%H%M%S)}"
OUT_ROOT="${OUT_ROOT:-outputs/markov_fno_round11_compute_r10_winner_${STAMP}}"
GPU="${GPU:-1}"
N_ITER="${N_ITER:-4000}"
TRAIN_DOCS=102400
LEAF=64
LABEL="m64__l5__pmean__iter${N_ITER}"

BUNDLE="outputs/_bundles/markov_hazard_panels_train102400/paper_hazard_panel_v1_t128/seed_0/base_bundle.json"
if [ ! -f "$BUNDLE" ]; then echo "ERROR: bundle missing: $BUNDLE"; exit 1; fi
mkdir -p "$OUT_ROOT/lanes"

OUT="$OUT_ROOT/$LABEL"
echo "[gpu$GPU] start $LABEL (n_iter=$N_ITER)"

CUDA_VISIBLE_DEVICES="$GPU" \
XLA_PYTHON_CLIENT_PREALLOCATE=false \
XLA_PYTHON_CLIENT_MEM_FRACTION=0.30 \
./venv/bin/ctreepo sim run contextual-sbijax \
  --data-source markov --load-data-bundle "$BUNDLE" \
  --sbijax-trainer learned_local_laws --sbijax-method nasss --sbijax-package-theta markov_exact_sketch \
  --sbijax-input-encoding regime_one_hot --local-law-summary-family jax_fno \
  --local-law-summary-fno-n-modes 64 --local-law-summary-fno-n-layers 5 --local-law-summary-fno-pooling-mode mean \
  --law-architecture fully_learned --c2-merge-target theta \
  --merge-family mlp --merge-fno-n-modes 16 --merge-fno-n-layers 2 --merge-fno-hidden-channels 32 \
  --decoder-head linear \
  --local-law-merge-loss nass_jsd --merge-nasss-n-slices 16 \
  --learned-merge-hidden-dim 128 --learned-decoder-hidden-dim 128 \
  --train-docs $TRAIN_DOCS --val-docs 256 --test-docs 256 \
  --fragment-len $LEAF --context-samples-per-doc 1 \
  --response-signature-contexts 16 --response-signature-slices 8 \
  --embedding-dim 64 --state-dim 25 --hidden-dim 128 \
  --learning-rate 0.0003 --lr-schedule cosine \
  --n-iter "$N_ITER" --batch-size 128 \
  --local-law-weight 1.0 --local-law-leaf-weight 1.0 --local-law-merge-weight 1.0 \
  --local-law-idempotence-weight 0.0 --local-law-contextual-weight 1.0 --local-law-package-weight 0.0 \
  --local-law-count-only --local-law-rep-dim 0 \
  --seed 0 --output-root "$OUT" \
  >"$OUT_ROOT/lanes/${LABEL}.log" 2>&1
echo "[gpu$GPU] done $LABEL"

if [ -f "$OUT/summary.json" ]; then
  ./venv/bin/python -c "import json; d=json.load(open('$OUT/summary.json')); t=d['diagnostics']['test']; print(f\"R11 result: count_mae={t.get('theta_count_raw_mae')}\")"
fi
echo ">>> Round 11 complete: $OUT_ROOT"
