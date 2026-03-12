#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

CV_OUT="outputs/manifesto_cv_overnight_smooth_$(date +%Y%m%d_%H%M)"

./venv/bin/python scripts/run_kfold_cv.py \
  --task manifesto_rile --dataset manifesto \
  --cv-output-dir "$CV_OUT" \
  --n-samples 400 --k 5 --stratify dist --bins 10 --seed 42 \
  --phase1-cache --max-parallel-folds 1 \
  --make-fold-reports --make-cv-report \
  -- \
  --dynamic-gpu --dynamic-gpu-soft-quiesce \
  --data-seed 42 --max-chunk-chars 8192 \
  --concurrent-docs 24 --concurrent-requests 256 --num-threads 64 \
  --no-phase1-score-requests \
  --optimizer gepa --optimizer-budget heavy --max-metric-calls 1200 --n-iterations 1 \
  --summarizer-max-leaf-examples 192 --summarizer-max-merge-examples 80 --summarizer-metric-eval-samples 80 \
  --enable-genrm --genrm-port 8001 --genrm-max-concurrent 8 --genrm-tree-concurrency 8 --genrm-request-timeout-sec 600 \
  --gepa-reflection-minibatch-size 10 \
  --honest-chunking --honest-split-seed 17 --three-layer-honesty --three-layer-seed 23 \
  --summarizer-leaf-max-ratio 0.25 --summarizer-merge-max-ratio 0.6 --summarizer-ratio-min-input-chars 200 \
  --phase1-batch-size 100 --no-adaptive-chunking --no-adaptive-embedding-proxy \
  --initial-scorer-instruction-file prompts/manifesto_rile/initial_scorer_instruction.txt \
  --sanitize-optimized-instructions \
  --scorer-tail-weighting power --scorer-tail-weight-alpha 3.0 --scorer-tail-weight-gamma 2.0 --scorer-tail-neutral 0.5 \
  --eval-scorer-ensemble-samples 5 --eval-scorer-temperature 0.7 --eval-scorer-ensemble-aggregator mean \
  --eval-scorer-calibration mean_shift --eval-scorer-calibration-split val

echo "DONE: $CV_OUT"

