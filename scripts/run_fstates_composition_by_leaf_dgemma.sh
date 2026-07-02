#!/usr/bin/env bash
# f_states composition (eval-only, skip GEPA) at leaf 2/4/8 on dgemma, to show
# composition-vs-depth. leaf16 is already done (EVALONLY_DGEMMA run); this fills
# the deeper leaves. Reuses the per-leaf FULL218 f-artifacts (no f retraining).
set -euo pipefail
cd /home/mlinegar/ThinkingTrees
PY=./venv/bin/python
BASE=outputs/manifesto_qsentence_diffusiongemma_FULL218_leafgrid/dspy
OUT=outputs/fstates_composition_by_leaf_dgemma_$(date +%Y%m%d_%H%M%S)
mkdir -p "$OUT"
MODEL="openai/RedHatAI/diffusiongemma-26B-A4B-it-NVFP4"
API=http://localhost:8004/v1,http://localhost:8005/v1,http://localhost:8006/v1,http://localhost:8007/v1

run_leaf() {
  local L="$1" LN="$2"
  local F="$BASE/leafq$L/iter_01_train_f/f_qsentence_dspy_iter_01"
  echo "[fstates] leaf=$LN"
  TT_DSPY_DROP_RESPONSE_FORMAT=1 TT_DSPY_WRAP_BARE_FIELD_OUTPUT=1 TT_SKIP_FULL_TREE_TRACES=1 \
  "$PY" scripts/run_manifesto_qsentence_sampled_supervision.py \
    --output-dir "$OUT/leaf$LN" \
    --f-artifact "$F" \
    --leaf-qsentences "$LN" --leaf-size-tokens 512 \
    --sample-leaf-count "$LN" --samples-per-doc 1 \
    --train-docs 0 --eval-docs 0 --max-train-examples 64 \
    --sample-state-source f_states \
    --skip-gepa-if-base-score-at-least 0.5 \
    --dspy-api-base "$API" --dspy-model "$MODEL" \
    --dspy-num-threads 48 --f-prewarm-threads 48 \
    --dspy-batch-max-concurrent 48 --dspy-batch-size 16 --dspy-batch-routing-policy round_robin \
    --dspy-budget light --dspy-max-tokens 1024 \
    --verbose > "$OUT/leaf$LN.log" 2>&1 || echo "[fstates] leaf=$LN nonzero (see log)"
}
run_leaf 008 8
run_leaf 004 4
run_leaf 002 2

echo "[fstates] === composition headline by leaf (g vs sample_baseline) ==="
"$PY" - "$OUT" <<'PYEOF'
import sys, json, glob, os
root=sys.argv[1]
print(f'{"leaf":>4s} {"g_beats":>8s} {"g_mae":>7s} {"base_mae":>8s}  per-dim g_direct pearson')
for d in sorted(glob.glob(os.path.join(root,'leaf*','metrics.json'))):
    m=json.load(open(d))
    ch=m.get('composition_headline',{})
    leaf=os.path.basename(os.path.dirname(d)).replace('leaf','')
    g=m.get('methods',{}).get('g_direct',{})
    rils=' '.join(f'{k}={g[k]["pearson"]:+.2f}' for k in ['rile','domain_4','domain_6'] if k in g and g[k].get('pearson') is not None)
    print(f'{leaf:>4s} {str(ch.get("g_beats_baseline")):>8s} {ch.get("g_direct_all_dims_mae",float("nan")):>7.3f} {ch.get("sample_baseline_all_dims_mae",float("nan")):>8.3f}  {rils}')
PYEOF
echo "[fstates] done: $OUT"
