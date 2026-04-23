#!/usr/bin/env bash
# Overnight: full Benoit comparison across all 6 dimensions.
# Three flights: scorer-only on Benoit summaries, full-pipeline on MP text,
# DSPy-optimized scorer trained on the held-out (Benoit-disjoint) pool.
#
# All flights write to outputs/overnight_benoit/{flight}/{dim}/ with report.json.
# vllm port 8010 (Gemma-4-31B-IT-NVFP4) is shared; vllm dynamic batching
# multiplexes the parallel streams. Disk cache disabled to avoid SQLite races.

set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

ROOT_OUT="outputs/overnight_benoit"
mkdir -p "$ROOT_OUT"

PORT=${PORT:-8010}
DIMS=(economic social immigration eu environment decentralization)

# Disable DSPy disk cache to avoid contention across parallel processes.
export TT_DSPY_ENABLE_DISK_CACHE=false
export TT_DSPY_ENABLE_MEMORY_CACHE=false

source venv/bin/activate

echo "=== launching overnight Benoit comparison @ $(date -u +%FT%TZ) ==="
echo "  port=$PORT  dims=${DIMS[*]}"

# --- Flight 1: scorer-only on Benoit GPT-4o summaries (~15 min/dim) -------
for d in "${DIMS[@]}"; do
  out="$ROOT_OUT/scorer_only/$d"
  mkdir -p "$out"
  echo "scorer_only/$d  →  $out/run.log"
  nohup python scripts/phase0_score_benoit_summaries.py \
      --port "$PORT" --dimension "$d" \
      --output-dir "$out" \
      > "$out/run.log" 2>&1 &
  sleep 5
done

# --- Flight 2: full-pipeline on MP text, all dims except Economic ---------
# (Economic full-pipeline is already running in outputs/phase0_full_pipeline_economic_229)
COUNTRIES=(11 12 13 14 21 22 23 31 32 33 34 35 41 42 51 53 54 56 61 62 64 81 82 83 86 87 88 92 93 94 95 96 97)
FP_DIMS=(social immigration eu environment decentralization)
for d in "${FP_DIMS[@]}"; do
  out="$ROOT_OUT/full_pipeline/$d"
  mkdir -p "$out"
  echo "full_pipeline/$d  →  $out/run.log"
  nohup python scripts/phase0_economic_pilot.py \
      --port "$PORT" --dimension "$d" \
      --mp-data-dir data/raw/manifesto_corpus_benoit \
      --countries "${COUNTRIES[@]}" \
      --min-year 1989 --max-year 2019 \
      --max-manifestos 1000 \
      --chunk-chars 24000 \
      --output-dir "$out" \
      > "$out/run.log" 2>&1 &
  sleep 30
done

# --- Flight 3: DSPy-optimized scorer (BootstrapFewShot, train OFF the test set) ---
for d in "${DIMS[@]}"; do
  out="$ROOT_OUT/optimizer_bootstrap/$d"
  mkdir -p "$out"
  echo "optimizer_bootstrap/$d  →  $out/run.log"
  nohup python scripts/phase1_optimize_scorer.py \
      --port "$PORT" --dimension "$d" \
      --optimizer bootstrap \
      --train-pool openweight \
      --output-dir "$out" \
      > "$out/run.log" 2>&1 &
  sleep 10
done

echo ""
echo "=== all flights launched at $(date -u +%FT%TZ) ==="
echo "Tail logs in $ROOT_OUT/{scorer_only,full_pipeline,optimizer_bootstrap}/{dim}/run.log"
echo "Result reports land in same dirs as report.json when each finishes."
echo ""
echo "Active python jobs:"
pgrep -af 'python scripts/' | head -30
