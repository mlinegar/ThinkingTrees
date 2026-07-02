#!/usr/bin/env bash
# Per-leaf local-law violation audit (C1/C2/C3a/C3b) on FULL218 trained f+g,
# gold + learned-g, leaf 2/4/8. Tests: do merge-law violations (C3a/C3b) grow
# with tree depth for learned-g (the small-leaf collapse hypothesis)?
set -euo pipefail
cd /home/mlinegar/ThinkingTrees
PY=./venv/bin/python
BASE=outputs/manifesto_qsentence_diffusiongemma_FULL218_leafgrid/dspy
OUT=outputs/law_audit_full218_$(date +%Y%m%d_%H%M%S)
mkdir -p "$OUT"
for L in 002 004 008; do
  LN=$((10#$L))
  F="$BASE/leafq$L/iter_01_train_f/f_qsentence_dspy_iter_01"
  G="$BASE/leafq$L/iter_02_train_g/g_qsentence_dspy_iter_02.json"
  echo "[audit] leaf=$LN  f=$F  g=$G"
  TT_DSPY_DROP_RESPONSE_FORMAT=1 TT_DSPY_WRAP_BARE_FIELD_OUTPUT=1 \
  "$PY" scripts/audit_qsentence_local_law_violations.py \
    --fg-grid-dir outputs/manifesto_qsentence_dspy_labeled_grid \
    --f-artifact "$F" --g-artifact "$G" \
    --leaf-qsentences "$LN" --n-docs 10 \
    --dspy-model openai/RedHatAI/diffusiongemma-26B-A4B-it-NVFP4 \
    --dspy-api-base http://localhost:8004/v1 \
    --dspy-num-threads 16 \
    --output-dir "$OUT/leaf$LN" > "$OUT/leaf$LN.log" 2>&1 || echo "[audit] leaf=$LN nonzero (see log)"
done
echo "[audit] === combined law-violation-by-depth (learned_g) ==="
"$PY" - "$OUT" <<'PYEOF'
import sys, json, glob, os
root=sys.argv[1]
print(f'{"src":9s} {"leaf":>4s} {"law":24s} {"level":>5s} {"viol":>8s} {"n":>5s}')
for d in sorted(glob.glob(os.path.join(root,'leaf*','law_violation_summary.json'))):
    j=json.load(open(d))
    for r in j['by_level']:
        if r['state_source']=='learned_g':
            mv=r['mean_violation']
            print(f'{r["state_source"]:9s} {r["leaf"]:>4d} {r["law"]:24s} {r["level"]:>5d} {(mv if mv is not None else float("nan")):>8.4f} {r["n"]:>5d}')
print()
print("=== law share (learned_g) ===")
for d in sorted(glob.glob(os.path.join(root,'leaf*','law_violation_summary.json'))):
    j=json.load(open(d))
    for r in j['law_share']:
        if r['state_source']=='learned_g':
            sh=r['share_of_total']
            print(f'  leaf={r["leaf"]:<3d} {r["law"]:24s} viol={r["mean_violation"]:.4f} share={(f"{sh:.1%}" if sh is not None else "n/a")}')
PYEOF
echo "[audit] done: $OUT"
