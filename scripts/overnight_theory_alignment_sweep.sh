#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────────────────────
# Overnight theory-alignment sweep: tree vs FNO parity + C2 learned resummary
#
# Core hypothesis: with all local laws enforced and enough data, the additive
# tree model should exactly recover FNO-level root error — because the local
# laws ARE the information-preservation guarantee (Lean: one_pass + nodewise_
# preservation).
#
# All experiments share identical DGP, partition, and seed settings so the
# ONLY variable is model family and local-law configuration.
#
# Layout:
#   Group 1 — FNO baselines (the ceiling to match)
#   Group 2 — Additive tree, root-only (the floor / no local laws)
#   Group 3 — Additive tree + all laws at varying weights
#   Group 4 — Additive tree + all laws + learned C2 resummary
#   Group 5 — C2 weight ablation (does C2 pressure → idempotence?)
# ──────────────────────────────────────────────────────────────────────────────
set -euo pipefail

PYTHON="/home/mlinegar/ThinkingTrees/venv/bin/python"
RUN="$PYTHON -m src.ctreepo.sim.cli.run_markov_changepoint_ops_count"
OUTDIR="/home/mlinegar/ThinkingTrees/outputs/overnight_theory_alignment_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$OUTDIR"

# ── Shared DGP + partition settings (identical across ALL runs) ──────────────
#    These pin the data distribution so any performance difference is purely
#    due to model family / local-law configuration.
SHARED_DGP="\
  --n-regimes 4 \
  --vocab-size 96 \
  --min-tokens 384 --max-tokens 384 \
  --min-segments 12 --max-segments 24 \
  --min-seg-len 8 --max-seg-len 32 \
  --fixed-leaf-tokens 16 \
  --data-seed 42 --model-seed 0"

# ── Shared training settings ─────────────────────────────────────────────────
SHARED_TRAIN="\
  --lr 3e-4 --weight-decay 1e-5 --batch-size 16 \
  --grad-clip-norm 1.0"

# ── Shared eval settings ─────────────────────────────────────────────────────
SHARED_EVAL="\
  --test-docs 2048 --violation-tau 0.5"

COMMON="$SHARED_DGP $SHARED_TRAIN $SHARED_EVAL"

echo "Output directory: $OUTDIR"
echo "Start time: $(date)"
echo "Shared DGP:   $SHARED_DGP"
echo "Shared train: $SHARED_TRAIN"
echo "Shared eval:  $SHARED_EVAL"
echo ""

run_experiment() {
    local name="$1"
    shift
    echo "[$(date +%H:%M:%S)] Starting: $name"
    $RUN $COMMON "$@" \
        --json-summary "$OUTDIR/${name}.json" \
        --csv-summary "$OUTDIR/${name}.csv" \
        > "$OUTDIR/${name}.log" 2>&1
    local rc=$?
    if [ $rc -eq 0 ]; then
        echo "[$(date +%H:%M:%S)] Done:     $name"
    else
        echo "[$(date +%H:%M:%S)] FAILED:   $name (exit $rc)"
    fi
    return $rc
}

# ═══════════════════════════════════════════════════════════════════════════════
# Group 1: FNO baselines (the ceiling)
#   FNO sees the full token sequence — this is the best we can hope for.
# ═══════════════════════════════════════════════════════════════════════════════
echo "──── Group 1: FNO baselines ────"
for N in 1024 4096 10240; do
    run_experiment "g1_fno_root_only_n${N}" \
        --model-family neural --train-docs $N --n-epochs 20 \
        --law-package root_only &
done
wait
echo ""

# ═══════════════════════════════════════════════════════════════════════════════
# Group 2: Additive tree, root-only (the floor)
#   No local laws — the tree model only optimizes root MSE.
# ═══════════════════════════════════════════════════════════════════════════════
echo "──── Group 2: Additive tree, root-only ────"
for N in 1024 4096 10240; do
    run_experiment "g2_tree_root_only_n${N}" \
        --model-family additive --train-docs $N --n-epochs 20 \
        --law-package root_only &
done
wait
echo ""

# ═══════════════════════════════════════════════════════════════════════════════
# Group 3: Additive tree + all laws (closing the gap)
#   Theory: local laws preserve information → tree should match FNO.
#   Sweep local-law weight to find the sweet spot.
# ═══════════════════════════════════════════════════════════════════════════════
echo "──── Group 3: Additive tree + all laws ────"
for N in 1024 4096 10240; do
    for LW in 0.1 0.3 0.5 0.7; do
        run_experiment "g3_tree_all_laws_lw${LW}_n${N}" \
            --model-family additive --train-docs $N --n-epochs 30 \
            --law-package all_laws --local-law-weight $LW &
    done
done
wait
echo ""

# ═══════════════════════════════════════════════════════════════════════════════
# Group 4: Additive tree + all laws + learned C2 resummary
#   Same as Group 3 but with C2 as a genuine functional test.
# ═══════════════════════════════════════════════════════════════════════════════
echo "──── Group 4: Tree + all laws + learned C2 ────"
for N in 1024 4096 10240; do
    for LW in 0.1 0.3 0.5 0.7; do
        run_experiment "g4_tree_c2learned_lw${LW}_n${N}" \
            --model-family additive --train-docs $N --n-epochs 30 \
            --law-package all_laws --local-law-weight $LW \
            --c2-learned-resummary &
    done
done
wait
echo ""

# ═══════════════════════════════════════════════════════════════════════════════
# Group 5: C2 weight ablation (fixed N=4096)
#   Does increasing C2 pressure drive the learned cycle toward idempotence?
# ═══════════════════════════════════════════════════════════════════════════════
echo "──── Group 5: C2 weight ablation ────"
for C2W in 0.0 0.5 1.0 3.0 10.0; do
    run_experiment "g5_c2_ablation_c2w${C2W}_n4096" \
        --model-family additive --train-docs 4096 --n-epochs 30 \
        --law-package all_laws --local-law-weight 0.3 \
        --c2-relative-weight $C2W \
        --c2-learned-resummary &
done
wait
echo ""

# ═══════════════════════════════════════════════════════════════════════════════
# Summary extraction
# ═══════════════════════════════════════════════════════════════════════════════
echo "══════════════════════════════════════════════════════════════════════"
echo "ALL EXPERIMENTS COMPLETE — $(date)"
echo "══════════════════════════════════════════════════════════════════════"
echo ""

$PYTHON -c "
import json, glob, os, sys

outdir = '$OUTDIR'
results = []
for path in sorted(glob.glob(os.path.join(outdir, '*.json'))):
    name = os.path.basename(path).replace('.json', '')
    try:
        with open(path) as f:
            d = json.load(f)
        le = d['metrics']['learned_test']
        cfg = d.get('config', {})
        cert = d['metrics'].get('certificate_envelope', {})
        results.append({
            'name': name,
            'root_mae': le.get('test_root_mae', float('nan')),
            'c2_r1': le.get('test_c2_r1_mae', float('nan')),
            'c2_recon': le.get('test_c2_bottleneck_reconstruction_mse', float('nan')),
            'leaf_mae': le.get('test_leaf_mae', float('nan')),
            'merge_mae': le.get('test_merge_mae', float('nan')),
            'spread': le.get('test_schedule_spread_mean', float('nan')),
            'b_est': cert.get('b_est', float('nan')),
            'N': cfg.get('train_docs', '?'),
            'family': cfg.get('model_family', '?'),
        })
    except Exception as e:
        print(f'  WARN: {name}: {e}', file=sys.stderr)

# ── Print table ──
hdr = f'{\"Name\":<50} {\"root_mae\":>10} {\"c2_r1\":>8} {\"leaf\":>10} {\"merge\":>10} {\"spread\":>8} {\"N\":>6}'
sep = '-' * len(hdr)
print(hdr)
print(sep)

# Group by prefix for readability
prev_group = ''
for r in results:
    group = r['name'].split('_')[0]
    if group != prev_group:
        if prev_group:
            print()
        prev_group = group
    print(f'{r[\"name\"]:<50} {r[\"root_mae\"]:>10.6f} {r[\"c2_r1\"]:>8.4f} {r[\"leaf_mae\"]:>10.2e} {r[\"merge_mae\"]:>10.2e} {r[\"spread\"]:>8.4f} {r[\"N\"]:>6}')

# ── Parity summary: tree gap to FNO at each scale ──
print()
print('=== PARITY GAP: tree root_mae - FNO root_mae ===')
fno_by_n = {r['N']: r['root_mae'] for r in results if 'fno' in r['name']}
for r in results:
    if 'fno' in r['name']:
        continue
    n = r['N']
    if n in fno_by_n:
        gap = r['root_mae'] - fno_by_n[n]
        pct = 100.0 * gap / fno_by_n[n] if fno_by_n[n] > 0 else float('nan')
        print(f'  {r[\"name\"]:<50} gap={gap:+.6f}  ({pct:+.1f}%)')
" | tee "$OUTDIR/summary.txt"

echo ""
echo "Results saved to: $OUTDIR/summary.txt"
echo "JSON files in:    $OUTDIR/"
