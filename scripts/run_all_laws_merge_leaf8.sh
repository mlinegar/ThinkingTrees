#!/usr/bin/env bash
# All-laws + lopsidedness-weighted MERGE experiment at leaf=8 on the dgemma fleet.
#
# Tests whether training g on ALL FOUR paper local laws (C2 base + C1 sufficiency
# + C3a joint-faithfulness reward terms) WITH lopsidedness-weighted node loss makes
# g learn the MASS-WEIGHTED ratio merge -- i.e. BEAT equal-averaging, toward the
# mass-weighted=0 ceiling -- instead of collapsing to a balanced-leaf averager.
#
# Two arms over the SAME bundle/seeds, sequential, all 4 GPUs (round_robin):
#   baseline : no laws, no lopsidedness weighting (current default g)
#   alllaws  : C1=C3a=1.0, lopsidedness strength=4.0 (C3b via scheduled sampling)
# Then dumps each arm's per-node g states and scores them per-level vs the
# equal-average bar and mass-weighted ceiling (eval_qsentence_merge_by_level.py).
set -euo pipefail
cd /home/mlinegar/ThinkingTrees
PY=./venv/bin/python
TS="$(date +%Y%m%d_%H%M%S)"
GRID=outputs/manifesto_qsentence_dspy_labeled_grid
MODEL="openai/RedHatAI/diffusiongemma-26B-A4B-it-NVFP4"
API=http://localhost:8004/v1,http://localhost:8005/v1,http://localhost:8006/v1,http://localhost:8007/v1
LEAF=8
ROOT="${ALL_LAWS_ROOT:-outputs/all_laws_merge_leaf8_$TS}"
mkdir -p "$ROOT"
echo "[all-laws] root=$ROOT"

run_arm() {
  local name="$1"; shift
  echo "[all-laws] === arm=$name $* ==="
  TT_DSPY_DROP_RESPONSE_FORMAT=1 TT_DSPY_WRAP_BARE_FIELD_OUTPUT=1 TT_SKIP_FULL_TREE_TRACES=1 \
  "$PY" scripts/run_manifesto_qsentence_dspy_ladder.py \
    --fg-grid-dir "$GRID" \
    --output-dir "$ROOT/$name" \
    --leaf-qsentences "$LEAF" \
    --max-iterations 2 \
    --initial-f-degree 2 --initial-g-degree 1 \
    --stage-naming powers \
    --target-dimensions all \
    --max-eval-trees 16 \
    --eval-sample-seed 20260623 \
    --dspy-optimizer gepa --dspy-budget light \
    --dspy-max-train-records 256 \
    --dspy-model "$MODEL" --dspy-api-base "$API" \
    --dspy-num-threads 48 --dspy-batch-max-concurrent 48 --dspy-batch-size 2 \
    --dspy-batch-routing-policy round_robin \
    --dspy-batch-request-timeout 120 --dspy-batch-await-response-timeout 180 \
    --dspy-lm-context-tokens 32768 --dspy-max-tokens 1024 \
    "$@" \
    --verbose \
    > "$ROOT/${name}.log" 2>&1 || echo "[all-laws] arm $name exited nonzero (see log)"
}

# Arm 1: current default g (the averager baseline).
# Arm matrix isolates the CHEAP, high-value levers. Scheduled sampling (C3b) is
# DELIBERATELY OFF: at leaf8 its per-tree bottom-up g-state generation is
# single-GPU-bound and (in the first run) stalled in a 262-thread futex wait for
# 30+min with no progress. C1 + C3a + lopsidedness need NO per-tree generation,
# so every arm here trains at the ~5min/arm baseline pace. C3b is best handled by
# a separate, smaller-tree scheduled-sampling experiment once it's de-stalled.
run_arm baseline                              # current default g (the averager)
run_arm lopsided   --dspy-g-lopsidedness-weight-strength 4.0
run_arm laws_lop   --dspy-g-lopsidedness-weight-strength 4.0 \
                   --dspy-g-law-c1-reward-weight 1.0 \
                   --dspy-g-law-c3a-reward-weight 1.0

score_arm() {
  local name="$1"
  local g_art="$ROOT/$name/dspy/leafq00${LEAF}/iter_02_train_g/g_qsentence_dspy_iter_02.json"
  if [[ ! -f "$g_art" ]]; then
    echo "[all-laws] no g artifact for $name ($g_art) -- skipping score"; return
  fi
  echo "[all-laws] === dumping per-node g states: $name ==="
  TT_DSPY_DROP_RESPONSE_FORMAT=1 TT_DSPY_WRAP_BARE_FIELD_OUTPUT=1 \
  "$PY" scripts/dump_qsentence_g_node_states.py \
    --g-artifact "$g_art" \
    --fg-grid-dir "$GRID" --leaf-qsentences "$LEAF" \
    --eval-split test --max-trees 16 \
    --dspy-model "$MODEL" --dspy-api-base "$API" \
    --out-jsonl "$ROOT/$name/g_node_states_leaf${LEAF}.jsonl" \
    > "$ROOT/${name}_dump.log" 2>&1 || echo "[all-laws] dump $name nonzero (see log)"
  echo "[all-laws] === per-level merge score: $name ==="
  "$PY" scripts/eval_qsentence_merge_by_level.py \
    --labeled-trees "$GRID/leafq00${LEAF}/labeled_trees.jsonl" \
    --lopsidedness-strength 4.0 \
    --g-states-jsonl "$ROOT/$name/g_node_states_leaf${LEAF}.jsonl" \
    --split test \
    --out-json "$ROOT/$name/merge_by_level.json" \
    | tee "$ROOT/${name}_merge_score.txt"
}

ARMS="baseline lopsided laws_lop"
for a in $ARMS; do score_arm "$a"; done

echo "[all-laws] ===== DONE: $ROOT ====="
echo "[all-laws] learned_g wmae per arm (lower=better; bar=equal_avg, ceiling=mass_wtd=0):"
for a in $ARMS; do
  f="$ROOT/$a/merge_by_level.json"
  [[ -f "$f" ]] && "$PY" -c "import json;d=json.load(open('$f'));p=d['pooled_weighted'];print(f'  $a: learned_g={p.get(\"learned_g_wmae\")}  equal_avg(bar)={p.get(\"equal_avg_wmae\")}  mass_wtd(ceiling)={p.get(\"mass_wtd_wmae\")}')"
done
