#!/usr/bin/env bash
# Fan-out driver: runs rescore_variants.py over every existing cell with
# cached per_manifesto.jsonl, at multiple (T, N) configs.
#
# Outputs land under outputs/rescore/T{T}_N{N}/<same_subdir_as_source>/
# so the comparison-table resolvers can find them by path swap.
#
# Usage:
#   bash scripts/launch_rescore_sweep.sh              # full sweep
#   CONFIGS="0.2:3" bash scripts/launch_rescore_sweep.sh    # only T=0.2 N=3

set -u
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"
. venv/bin/activate
export TT_DSPY_ENABLE_DISK_CACHE=false
export TT_DSPY_ENABLE_MEMORY_CACHE=false
export MANIFESTO_MAX_TOKENS=8192

PORTS="8010 8011 8012 8013"
CONFIGS="${CONFIGS:-0.2:3 0.3:1 0.7:1}"   # T:N pairs; T=0.0 is the baseline (already have)
CHUNKS="64000 32000 16000 8000"
DIMS="economic social immigration eu environment decentralization"

for cfg in $CONFIGS; do
  T="${cfg%%:*}"
  N="${cfg##*:}"
  label="T${T}_N${N}"
  echo "=== rescore config $label ==="

  # --- per-dim chunk_sweep cells ---
  for d in $DIMS; do
    for c in $CHUNKS; do
      src="outputs/chunk_sweep/${d}_c${c}"
      dst="outputs/rescore/${label}/chunk_sweep/${d}_c${c}"
      [[ -f "$src/per_manifesto.jsonl" ]] || continue
      [[ -f "$dst/report.json" ]] && continue  # skip if already done
      mkdir -p "$dst"
      nohup python scripts/rescore_variants.py \
          --mode per-dim --dimension "$d" \
          --input-dir "$src" --output-dir "$dst" \
          --temperature "$T" --n-samples "$N" \
          --ports $PORTS \
          > "$dst/run.log" 2>&1 &
      sleep 1
    done
  done

  # --- per-dim full_pipeline (chunk=24K, full test n≈215) ---
  for d in $DIMS; do
    src="outputs/overnight_benoit/full_pipeline/${d}"
    dst="outputs/rescore/${label}/overnight_benoit/full_pipeline/${d}"
    [[ -f "$src/per_manifesto.jsonl" ]] || continue
    [[ -f "$dst/report.json" ]] && continue
    mkdir -p "$dst"
    nohup python scripts/rescore_variants.py \
        --mode per-dim --dimension "$d" \
        --input-dir "$src" --output-dir "$dst" \
        --temperature "$T" --n-samples "$N" \
        --ports $PORTS \
        > "$dst/run.log" 2>&1 &
    sleep 1
  done

  # --- combined cells (phase3/combined_c* + phase2/combined_pipeline) ---
  for c in $CHUNKS; do
    src="outputs/phase3/combined_c${c}"
    dst="outputs/rescore/${label}/phase3/combined_c${c}"
    [[ -f "$src/per_manifesto.jsonl" ]] || continue
    [[ -f "$dst/report.json" ]] && continue
    mkdir -p "$dst"
    nohup python scripts/rescore_variants.py \
        --mode combined \
        --input-dir "$src" --output-dir "$dst" \
        --temperature "$T" --n-samples "$N" \
        --ports $PORTS \
        > "$dst/run.log" 2>&1 &
    sleep 1
  done
  # combined_pipeline (chunk=24K, full test n=229)
  src="outputs/phase2/combined_pipeline"
  dst="outputs/rescore/${label}/phase2/combined_pipeline"
  if [[ -f "$src/per_manifesto.jsonl" ]] && [[ ! -f "$dst/report.json" ]]; then
    mkdir -p "$dst"
    nohup python scripts/rescore_variants.py \
        --mode combined \
        --input-dir "$src" --output-dir "$dst" \
        --temperature "$T" --n-samples "$N" \
        --ports $PORTS \
        > "$dst/run.log" 2>&1 &
    sleep 1
  fi
done

echo "=== rescore sweep launched ==="
pgrep -c -f 'python scripts/rescore_variants' | xargs -I{} echo "{} rescore jobs running"
