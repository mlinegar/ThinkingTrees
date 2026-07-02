#!/usr/bin/env bash
# Round 2 Stage 1: single-leaf encoder-capacity diagnostic.
#
# Three parallel runs, each at n_leaves=1 (doc_tokens == leaf_tokens), so root
# state IS the leaf state. This isolates "can FNO learn the right per-leaf
# representation under direct, increasingly easy supervision?" from any merge
# composition confound.
#
#   GPU 0: boundary BCE — per-token "regime[i] != regime[i+1]" labels via the
#          existing _run_boundary_supervision_ablation. Easiest signal: tells
#          the encoder where the boundaries are.
#
#   GPU 2: markov_node_witness — direct (count, first, last) labels on the
#          single leaf state.
#
#   GPU 3: markov_local_laws_fno with merge=0 idemp=0 — C1 leaf calibration
#          only (no merge, no idempotence). The "trivial local laws" cell.
#
# Same width grid across all three: doc=leaf in {32, 64, 128} × channels in
# {128, 256} × g_n_modes in {16, 32}. 12 cells per GPU.
set -euo pipefail
source venv/bin/activate

STAMP="${STAMP:-$(date -u +%Y%m%d_%H%M%S)}"
OUT_ROOT="${OUT_ROOT:-outputs/markov_fno_round2_stage1_${STAMP}}"
TRAIN_DOCS="${TRAIN_DOCS:-4096}"
EVAL_DOCS="${EVAL_DOCS:-512}"
EPOCHS="${EPOCHS:-48}"
PY="${PY:-./venv/bin/python}"

mkdir -p "$OUT_ROOT"

run_single_leaf_cell() {
  local out="$1"
  local gpu="$2"
  local objective="$3"
  local doc_tokens="$4"
  local channels="$5"
  local g_n_modes="$6"
  local seed="${7:-0}"

  if [ -f "$out/summary.json" ]; then
    echo "skip $out"
    return 0
  fi
  mkdir -p "$out"

  local extra_args=()
  local main_epochs="$EPOCHS"
  case "$objective" in
    boundary)
      # The boundary ablation trains a fresh model; the prior main run
      # (--root-only) only emits a baseline. Keep main_epochs short.
      main_epochs=2
      extra_args+=(
        --run-boundary-supervision-ablation
        --boundary-supervision-epochs "$EPOCHS"
        --boundary-supervision-weight 1.0
        --root-only
      )
      ;;
    witness)
      extra_args+=(
        --training-objective markov_node_witness
        --markov-witness-readout flatten
      )
      ;;
    trivial_laws)
      extra_args+=(
        --training-objective markov_local_laws_fno
        --markov-law-leaf-weight 1.0
        --markov-law-merge-weight 0.0
        --markov-law-idempotence-weight 0.0
        --markov-law-readout flatten
      )
      ;;
    *)
      echo "unknown objective: $objective" >&2
      return 1
      ;;
  esac

  echo "==> [$objective] doc=$doc_tokens ch=$channels gm=$g_n_modes -> $out"
  CUDA_VISIBLE_DEVICES="$gpu" \
  "$PY" scripts/probe_clean_unified_no.py \
    --doc-tokens "$doc_tokens" \
    --leaf-tokens "$doc_tokens" \
    --train-docs "$TRAIN_DOCS" \
    --eval-docs "$EVAL_DOCS" \
    --epochs "$main_epochs" \
    --batch-size 32 \
    --channels "$channels" \
    --g-n-modes "$g_n_modes" \
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
    --seed "$seed" \
    --device cuda \
    --output-root "$out" \
    "${extra_args[@]}" 2>&1 | tail -3
}

run_lane() {
  local gpu="$1"
  local objective="$2"

  echo ">>> Lane: GPU=$gpu objective=$objective"
  for doc in 32 64 128; do
    for channels in 128 256; do
      for modes in 16 32; do
        out="$OUT_ROOT/${objective}/doc${doc}_ch${channels}_gm${modes}_seed0"
        run_single_leaf_cell "$out" "$gpu" "$objective" "$doc" "$channels" "$modes" 0
      done
    done
  done
}

# Launch all three lanes in parallel, log to separate files.
mkdir -p "$OUT_ROOT/logs"

GPU=0 OBJECTIVE=boundary
LOG_BOUNDARY="$OUT_ROOT/logs/lane_boundary_gpu0.log"
LOG_WITNESS="$OUT_ROOT/logs/lane_witness_gpu2.log"
LOG_TRIV="$OUT_ROOT/logs/lane_trivial_laws_gpu3.log"

bash -c "$(declare -f run_single_leaf_cell run_lane); export OUT_ROOT='$OUT_ROOT' EPOCHS='$EPOCHS' TRAIN_DOCS='$TRAIN_DOCS' EVAL_DOCS='$EVAL_DOCS' PY='$PY'; run_lane 0 boundary" > "$LOG_BOUNDARY" 2>&1 &
PID_B=$!
bash -c "$(declare -f run_single_leaf_cell run_lane); export OUT_ROOT='$OUT_ROOT' EPOCHS='$EPOCHS' TRAIN_DOCS='$TRAIN_DOCS' EVAL_DOCS='$EVAL_DOCS' PY='$PY'; run_lane 2 witness" > "$LOG_WITNESS" 2>&1 &
PID_W=$!
bash -c "$(declare -f run_single_leaf_cell run_lane); export OUT_ROOT='$OUT_ROOT' EPOCHS='$EPOCHS' TRAIN_DOCS='$TRAIN_DOCS' EVAL_DOCS='$EVAL_DOCS' PY='$PY'; run_lane 3 trivial_laws" > "$LOG_TRIV" 2>&1 &
PID_T=$!

echo "boundary lane    pid=$PID_B log=$LOG_BOUNDARY"
echo "witness lane     pid=$PID_W log=$LOG_WITNESS"
echo "trivial laws ln  pid=$PID_T log=$LOG_TRIV"

wait $PID_B $PID_W $PID_T

echo ">>> Aggregating Stage 1 results"
"$PY" - <<PY
import json
import os
import csv

root = "$OUT_ROOT"
rows = []
for objective in ["boundary", "witness", "trivial_laws"]:
    obj_root = os.path.join(root, objective)
    if not os.path.isdir(obj_root):
        continue
    for cell_dir in sorted(os.listdir(obj_root)):
        path = os.path.join(obj_root, cell_dir, "summary.json")
        if not os.path.exists(path):
            continue
        try:
            with open(path) as fh:
                d = json.load(fh)
        except Exception as exc:
            print(f"  skip {path}: {exc}")
            continue
        row = {
            "objective": objective,
            "cell": cell_dir,
            "best_val_root_mae": d.get("best_val_root_mae"),
            "test_root_mae": d.get("test_root_mae"),
            "n_leaves_per_doc": d.get("n_leaves_per_doc"),
            "target_scale": d.get("target_scale"),
        }
        # boundary diagnostic specifics
        bd = d.get("boundary_supervision_ablation") or d.get("stronger_supervision_ablation") or {}
        bd_test = (bd.get("boundary_diagnostics") or {}).get("test") or {}
        if bd_test:
            row["boundary_test_loss"] = bd_test.get("loss") or bd_test.get("bce")
            row["boundary_test_acc"] = bd_test.get("accuracy") or bd_test.get("acc")
            row["boundary_test_f1"] = bd_test.get("f1")
        # markov_local_laws_fno diagnostics
        law = d.get("markov_local_law_fno_diagnostics") or {}
        law_test = (law.get("splits") or {}).get("test") or {}
        for slot in ["leaf", "merge", "root"]:
            block = law_test.get(slot) or {}
            row[f"law_{slot}_theta_mae"] = block.get("theta_mae")
            row[f"law_{slot}_first_acc"] = block.get("theta_first_regime_accuracy")
            row[f"law_{slot}_last_acc"] = block.get("theta_last_regime_accuracy")
            row[f"law_{slot}_full_exact"] = block.get("full_witness_exact_rate")
        # node_witness diagnostics
        wit = d.get("markov_node_witness_diagnostics") or {}
        wit_test = (wit.get("splits") or {}).get("test") or {}
        for slot in ["leaf", "merge", "root"]:
            block = wit_test.get(slot) or {}
            row[f"witness_{slot}_theta_mae"] = block.get("theta_mae")
            row[f"witness_{slot}_first_acc"] = block.get("theta_first_regime_accuracy")
            row[f"witness_{slot}_last_acc"] = block.get("theta_last_regime_accuracy")
            row[f"witness_{slot}_full_exact"] = block.get("full_witness_exact_rate")
        rows.append(row)

if rows:
    fieldnames = list(dict.fromkeys(k for r in rows for k in r.keys()))
    csv_path = os.path.join(root, "stage1_summary.csv")
    with open(csv_path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {csv_path} ({len(rows)} rows)")
else:
    print("no results found")
PY

echo ">>> Stage 1 complete: $OUT_ROOT"
