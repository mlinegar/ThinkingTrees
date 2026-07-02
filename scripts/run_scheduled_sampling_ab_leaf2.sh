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
API=http://localhost:8004/v1
ROOT=outputs/sched_sampling_ab_leaf2_$TS
mkdir -p "$ROOT"

run_arm() {
  local name="$1" rate="$2"
  echo "[ab] === arm=$name rate=$rate ==="
  TT_DSPY_DROP_RESPONSE_FORMAT=1 TT_DSPY_WRAP_BARE_FIELD_OUTPUT=1 TT_SKIP_FULL_TREE_TRACES=1 \
  "$PY" scripts/run_manifesto_qsentence_dspy_ladder.py \
    --fg-grid-dir "$GRID" \
    --output-dir "$ROOT/$name" \
    --leaf-qsentences 2 \
    --max-iterations 2 \
    --initial-f-degree 2 --initial-g-degree 1 \
    --stage-naming powers \
    --target-dimensions all \
    --max-eval-trees 24 \
    --eval-sample-seed 20260623 \
    --dspy-optimizer gepa --dspy-budget light \
    --dspy-max-train-records 256 \
    --dspy-g-scheduled-sampling-rate "$rate" \
    --dspy-model "$MODEL" --dspy-api-base "$API" \
    --dspy-num-threads 32 --dspy-batch-max-concurrent 32 --dspy-batch-size 16 \
    --dspy-lm-context-tokens 32768 --dspy-max-tokens 1024 \
    --fail-on-row-error --verbose \
    > "$ROOT/${name}.log" 2>&1 || echo "[ab] arm $name exited nonzero (see log)"
}

run_arm control_rate0 0.0
run_arm sched_rate1   1.0

echo "[ab] ===== per-dim Pearson comparison (iter_02 learned g) ====="
"$PY" - "$ROOT" <<'PYEOF'
import sys, json, os
from collections import defaultdict
root=sys.argv[1]
def pear(xs,ys):
    n=len(xs)
    if n<2: return float('nan')
    mx=sum(xs)/n; my=sum(ys)/n
    vx=sum((x-mx)**2 for x in xs); vy=sum((y-my)**2 for y in ys)
    if vx<=0 or vy<=0: return float('nan')
    return sum((x-mx)*(y-my) for x,y in zip(xs,ys))/(vx*vy)**0.5
def perdim(arm):
    f=os.path.join(root,arm,'dspy','leafq002','prediction_records','iter_02_post_eval.jsonl')
    if not os.path.exists(f): return None
    P=defaultdict(list); T=defaultdict(list)
    for l in open(f):
        r=json.loads(l)
        try: p=float(r['prediction']); t=float(r['teacher_score'])
        except: continue
        d=r.get('dimension')
        if d: P[d].append(p); T[d].append(t)
    return {d:pear(P[d],T[d]) for d in sorted(P)}
c=perdim('control_rate0'); s=perdim('sched_rate1')
dims=['rile','domain_1','domain_2','domain_3','domain_4','domain_5','domain_6','domain_7']
print(f'{"dim":10s} {"control":>9s} {"sched":>9s} {"delta":>9s}')
for d in dims:
    cv=(c or {}).get(d,float('nan')); sv=(s or {}).get(d,float('nan'))
    print(f'{d:10s} {cv:>9.3f} {sv:>9.3f} {(sv-cv):>9.3f}')
PYEOF
echo "[ab] done: $ROOT"
