#!/usr/bin/env bash
# Scoped local-law violation audit: per-doc cost explodes at small leaves
# (leaf2 ~208 g-calls/doc via C3b's 3x multiplier), so use FEWER docs at deep
# leaves. Still gets the leaf x depth gradient for the hypothesis.
set -euo pipefail
cd /home/mlinegar/ThinkingTrees
PY=./venv/bin/python
BASE=outputs/manifesto_qsentence_diffusiongemma_FULL218_leafgrid/dspy
OUT=outputs/law_audit_scoped_$(date +%Y%m%d_%H%M%S)
mkdir -p "$OUT"
# leaf -> n_docs (cheaper leaves get more docs)
run_leaf() {
  local L="$1" LN="$2" NDOCS="$3"
  local F="$BASE/leafq$L/iter_01_train_f/f_qsentence_dspy_iter_01"
  local G="$BASE/leafq$L/iter_02_train_g/g_qsentence_dspy_iter_02.json"
  echo "[audit] leaf=$LN n_docs=$NDOCS"
  TT_DSPY_DROP_RESPONSE_FORMAT=1 TT_DSPY_WRAP_BARE_FIELD_OUTPUT=1 \
  "$PY" scripts/audit_qsentence_local_law_violations.py \
    --fg-grid-dir outputs/manifesto_qsentence_dspy_labeled_grid \
    --f-artifact "$F" --g-artifact "$G" \
    --leaf-qsentences "$LN" --n-docs "$NDOCS" \
    --dspy-model openai/RedHatAI/diffusiongemma-26B-A4B-it-NVFP4 \
    --dspy-api-base http://localhost:8004/v1,http://localhost:8005/v1,http://localhost:8006/v1,http://localhost:8007/v1 --dspy-num-threads 64 --doc-concurrency 12 --doc-timeout 600 --dspy-batch-size 2 --dspy-batch-request-timeout 90 --dspy-batch-await-response-timeout 120 \
    --output-dir "$OUT/leaf$LN" > "$OUT/leaf$LN.log" 2>&1 || echo "[audit] leaf=$LN nonzero (see log)"
}
run_leaf 016 16 12
run_leaf 008 8 12
run_leaf 004 4 8
run_leaf 002 2 6
echo "[audit] === learned_g law violation by depth ==="
"$PY" - "$OUT" <<'PYEOF'
import sys, json, glob, os
root=sys.argv[1]
print(f'{"src":9s} {"leaf":>4s} {"law":24s} {"level":>5s} {"viol":>8s} {"n":>5s}')
for d in sorted(glob.glob(os.path.join(root,'leaf*','law_violation_summary.json'))):
    for r in json.load(open(d))['by_level']:
        mv=r['mean_violation']
        print(f'{r["state_source"]:9s} {r["leaf"]:>4d} {r["law"]:24s} {r["level"]:>5d} {(mv if mv is not None else float("nan")):>8.4f} {r["n"]:>5d}')
print("\n=== law share (learned_g) ===")
for d in sorted(glob.glob(os.path.join(root,'leaf*','law_violation_summary.json'))):
    for r in json.load(open(d))['law_share']:
        if r['state_source']=='learned_g':
            sh=r['share_of_total']
            print(f'  leaf={r["leaf"]:<3d} {r["law"]:24s} viol={r["mean_violation"]:.4f} share={(f"{sh:.1%}" if sh is not None else "n/a")}')
PYEOF
echo "[audit] done: $OUT"
