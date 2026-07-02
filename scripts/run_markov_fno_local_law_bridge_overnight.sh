#!/usr/bin/env bash
# Overnight CleanUnifiedNO bridge grid for Markov local-law transfer.
set -euo pipefail

source venv/bin/activate

STAMP="${STAMP:-$(date +%Y%m%d_%H%M%S)}"
OUT_ROOT="${OUT_ROOT:-outputs/markov_fno_local_law_bridge_${STAMP}}"
GPU="${GPU:-0}"
BUNDLE="${BUNDLE:-outputs/_bundles/markov_hazard_panels/paper_hazard_panel_v1_t128/seed_0/base_bundle.json}"
PY="${PY:-./venv/bin/python}"

mkdir -p "$OUT_ROOT"

common_probe_args=(
  --train-docs 10240
  --eval-docs 1024
  --epochs 60
  --channels-grid 128
  --g-n-layers 2
  --scorer-n-modes 16
  --scorer-n-layers 2
  --lr 0.0001
  --optimizer adamw
  --weight-decay 0.01
  --lr-schedule cosine
  --grad-clip 1.0
  --leaf-pool sum
  --diagnostic-baselines none
  --seeds 0
  --device cuda
  --gpu "$GPU"
  --keep-going
)

echo "==> primary t128 laws bridge grid"
"$PY" scripts/run_clean_unified_no_grid.py \
  --load-data-bundle "$BUNDLE" \
  --leaf-tokens-grid 128,64,32,16,4,2 \
  --objectives root,contextual_none,markov_local_laws_fno \
  --g-n-modes-grid 16 \
  --batch-size 64 \
  --batch-size-map "128=512;64=256;32=128;16=64;4=32;2=16" \
  --output-root "$OUT_ROOT/fno_t128_laws_primary" \
  "${common_probe_args[@]}"

echo "==> t128 direct witness capacity side grid"
"$PY" scripts/run_clean_unified_no_grid.py \
  --load-data-bundle "$BUNDLE" \
  --leaf-tokens-grid 128,16,2 \
  --objectives markov_node_witness \
  --g-n-modes-grid 16 \
  --batch-size 64 \
  --batch-size-map "128=512;16=64;2=16" \
  --output-root "$OUT_ROOT/fno_t128_witness_capacity" \
  "${common_probe_args[@]}"

echo "==> t128 local-law mode side grid"
"$PY" scripts/run_clean_unified_no_grid.py \
  --load-data-bundle "$BUNDLE" \
  --leaf-tokens-grid 128,16,2 \
  --objectives markov_local_laws_fno \
  --g-n-modes-grid 8,32 \
  --batch-size 64 \
  --batch-size-map "128=512;16=64;2=16" \
  --output-root "$OUT_ROOT/fno_t128_law_modes" \
  "${common_probe_args[@]}"

RUN_T2048="${RUN_T2048:-auto}"
if [[ "$RUN_T2048" == "auto" ]]; then
  RUN_T2048="$("$PY" - <<PY
import csv, math
from pathlib import Path
path = Path("$OUT_ROOT/fno_t128_laws_primary/grid_summary.csv")
rows = list(csv.DictReader(path.open())) if path.exists() else []
def val(row):
    try:
        return float(row.get("test_root_mae", "nan"))
    except Exception:
        return math.nan
root = [val(r) for r in rows if r.get("objective") == "root"]
law = [val(r) for r in rows if r.get("objective") == "markov_local_laws_fno"]
root_best = min([x for x in root if math.isfinite(x)], default=math.inf)
law_best = min([x for x in law if math.isfinite(x)], default=math.inf)
print("1" if law_best < root_best else "0")
PY
)"
fi

if [[ "$RUN_T2048" == "1" ]]; then
  echo "==> t2048 composition-stress law grid"
  "$PY" scripts/run_clean_unified_no_grid.py \
    --benchmark recoverable_v5_t2048 \
    --leaf-tokens-grid 2048,256 \
    --channels-grid 128 \
    --g-n-modes-grid 8,16 \
    --objectives root,markov_local_laws_fno \
    --train-docs 10240 \
    --eval-docs 1024 \
    --epochs 40 \
    --batch-size 8 \
    --batch-size-map "2048=16;256=8" \
    --g-n-layers 2 \
    --scorer-n-modes 16 \
    --scorer-n-layers 2 \
    --lr 0.0001 \
    --optimizer adamw \
    --weight-decay 0.01 \
    --lr-schedule cosine \
    --grad-clip 1.0 \
    --leaf-pool sum \
    --diagnostic-baselines none \
    --seeds 0 \
    --device cuda \
    --gpu "$GPU" \
    --keep-going \
    --output-root "$OUT_ROOT/fno_t2048_composition_stress"
else
  echo "skipping t2048 composition-stress grid (RUN_T2048=$RUN_T2048)"
  cat > "$OUT_ROOT/fno_t2048_composition_stress_skipped.json" <<JSON
{"status": "skipped", "reason": "primary t128 markov_local_laws_fno did not beat root, or RUN_T2048=0"}
JSON
fi

"$PY" - <<PY
import json
from pathlib import Path
root = Path("$OUT_ROOT")
payload = {
    "status": "completed",
    "output_root": str(root),
    "bundle": "$BUNDLE",
    "gpu": "$GPU",
    "grids": {
        "fno_t128_laws_primary": str(root / "fno_t128_laws_primary"),
        "fno_t128_witness_capacity": str(root / "fno_t128_witness_capacity"),
        "fno_t128_law_modes": str(root / "fno_t128_law_modes"),
        "fno_t2048_composition_stress": str(root / "fno_t2048_composition_stress"),
    },
}
(root / "summary.json").write_text(json.dumps(payload, indent=2) + "\n")
PY

echo "wrote $OUT_ROOT/summary.json"
