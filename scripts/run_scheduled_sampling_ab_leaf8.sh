#!/usr/bin/env bash
# A/B: scheduled sampling (rate 1.0) vs control (rate 0.0) on the Path A
# alternating ladder at leaf=2 (deepest tree, where exposure-bias collapse is
# worst). Same dgemma server, sequential. Compares per-dim Pearson at iter_02.
set -euo pipefail
cd /home/mlinegar/ThinkingTrees
PY=./venv/bin/python
TS="$(date +%Y%m%d_%H%M%S)"
GRID=outputs/manifesto_qsentence_dspy_labeled_grid
MODEL="openai/RedHatAI/diffusiongemma-26B-A4B-it-NVFP4"
API=http://localhost:8004/v1,http://localhost:8005/v1,http://localhost:8006/v1,http://localhost:8007/v1
ROOT=outputs/sched_sampling_ab_leaf8_$TS
mkdir -p "$ROOT"

run_arm() {
  local name="$1" rate="$2"
  echo "[ab] === arm=$name rate=$rate ==="
  TT_DSPY_DROP_RESPONSE_FORMAT=1 TT_DSPY_WRAP_BARE_FIELD_OUTPUT=1 TT_SKIP_FULL_TREE_TRACES=1 \
  "$PY" scripts/run_manifesto_qsentence_dspy_ladder.py \
    --fg-grid-dir "$GRID" \
    --output-dir "$ROOT/$name" \
    --leaf-qsentences 8 \
    --max-iterations 2 \
    --initial-f-degree 2 --initial-g-degree 1 \
    --stage-naming powers \
    --target-dimensions all \
    --max-eval-trees 16 \
    --eval-sample-seed 20260623 \
    --dspy-optimizer gepa --dspy-budget light \
    --dspy-max-train-records 256 \
    --dspy-g-scheduled-sampling-rate "$rate" \
    --dspy-model "$MODEL" --dspy-api-base "$API" \
    --dspy-num-threads 48 --dspy-batch-max-concurrent 48 --dspy-batch-size 16 --dspy-batch-routing-policy round_robin \
    --dspy-lm-context-tokens 32768 --dspy-max-tokens 1024 \
    --verbose \
    > "$ROOT/${name}.log" 2>&1 || echo "[ab] arm $name exited nonzero (see log)"
}

run_arm control_rate0 0.0
run_arm sched_rate1   1.0

echo "[ab] ===== per-dim Pearson comparison (iter_02 learned g) ====="
# Single source of truth for per-dim Pearson + deltas (see
# scripts/compare_qsentence_per_dim_pearson.py). Control = this run's rate0 arm.
"$PY" scripts/compare_qsentence_per_dim_pearson.py \
  --control "$ROOT/control_rate0" \
  --test "$ROOT/sched_rate1" \
  --leaf 8 --iter 2 --labels control,sched \
  --json-out "$ROOT/per_dim_pearson_comparison.json"
echo "[ab] done: $ROOT"
