# Learned Sketch Simulation (HLL Baseline + Oracle-Learned Model)

This simulation tests whether a learned tree sketch can approach a classical mergeable sketch baseline under oracle supervision.

The canonical package surface for this work is now the standalone `treepo` package inside this repo. The legacy root-level scripts still work, but they are compatibility entrypoints over the `treepo` implementation.

## Preferred `treepo` workflow

```bash
cd treepo
pip install -e ".[torch]"
treepo-bench suite cardinality-paper --out-root ../outputs/cardinality --jobs 4
treepo-bench report cardinality --output-root ../outputs/cardinality
```

This report emits the paper-facing cardinality/HLL artifacts, including:

- exact HLL streaming invariance checks
- TreePO distance-to-HLL-floor curves
- exact-set and wrong-baseline comparisons

## What it runs

1. **Classical baseline (HLL)**
   - Exact HyperLogLog update + merge over tree schedules.
   - Matched by memory budget to the learned state size.
   - The paper/report path now also includes exact-set truth and a wrong baseline using summed leaf uniques.

2. **Learned sketch (PyTorch)**
   - Learns leaf encoding, merge operator, and readout from oracle cardinalities.
   - In `latent_proxy_baseline` mode, uses a proxy latent merge-state loss: merged state vs jointly encoded union state.
   - In `law_backed_learned_sketch` mode, also reports decoded approximate local-law budgets `eps_leaf`, `eps_merge`, and `eps_idemp`.

3. **Ablation sweeps**
   - State budget sweep (`state_dim`).
   - Oracle-query budget sweep (`train_size`).
   - Reports excess error vs HLL and merge-consistency distortion.

## Main script

`scripts/run_learned_sketch_simulation.py`

Outputs:

- JSON summary (`--json-summary`)
- CSV summary (`--csv-summary`)

Key columns:

- `learned_rmse`, `learned_mean_abs_rel_error`
- `learned_relative_rmse`
- `hll_rmse`, `hll_mean_abs_rel_error`
- `hll_relative_rmse`, `hll_rse_theory`
- `distance_to_hll_floor_rel_rmse` (primary metric)
- `distance_to_hll_empirical_rel_rmse`
- `rmse_gap_vs_hll`, `abs_rel_error_gap_vs_hll`
- `latent_merge_state_mse`
- `eps_leaf`, `eps_merge`, `eps_idemp`
- `evidence_status`, `simulation_mode`
- `train_mean_internal_nodes`, `train_audit_nodes_mean`, `train_audit_coverage_mean`
- `train_total_queries_estimate`

## Aggressive multi-seed script

`scripts/run_learned_sketch_sampling_sweep.py`

This script runs a larger sweep across multiple seeds and aggregates results, so you can see how much oracle sample budget (`train_size`) changes both mean error and run-to-run variance.

Additional outputs:

- Raw per-seed CSV (`--raw-csv`)
- Aggregated CSV with mean/std/p10/p90 by `(state_dim, train_size)` (`--agg-csv`)
- JSON summary with both raw and aggregated rows (`--json-summary`)

Device/parallel controls:

- `--device auto|cpu|cuda`
- `--cuda-device N` (single GPU)
- `--cuda-devices i,j,...` (round-robin GPU assignment across seeds)
- `--parallel-workers K` (run up to `K` seed jobs concurrently)
- `--torch-threads T` (threads per worker process)
- `--audit-policy all|fixed|fraction|sqrt|log2`
- `--audit-fixed-nodes N`
- `--audit-fraction F`
- `--audit-scale S`
- `--no-root-query` (optional, disables root-label supervision in training)

## Plot script

`scripts/plot_learned_sketch_simulation.py`

Panel summary:

1. Learning curves (relative RMSE vs train size) with HLL empirical and theoretical floor references.
2. Primary metric: distance to theoretical HLL floor (0 line is optimal floor match).
3. Audit geometry: mean internal nodes/doc vs audited nodes/doc and audit coverage.

## Example runs

### Quick smoke run

```bash
source venv/bin/activate
python3 scripts/run_learned_sketch_simulation.py \
  --state-dims 16 \
  --train-sizes 64 \
  --n-val 32 \
  --n-test 64 \
  --n-epochs 3 \
  --hidden-dim 64 \
  --cpu \
  --json-summary outputs/learned_sketch_simulation_smoke.json \
  --csv-summary outputs/learned_sketch_simulation_smoke.csv
```

### Full default sweep

```bash
source venv/bin/activate
python3 scripts/run_learned_sketch_simulation.py \
  --cpu \
  --json-summary outputs/learned_sketch_simulation_summary.json \
  --csv-summary outputs/learned_sketch_simulation_summary.csv
```

### Aggressive sampling sweep (recommended for paper figures)

```bash
source venv/bin/activate
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
python3 -u scripts/run_learned_sketch_sampling_sweep.py \
  --state-dims 32,64,96,128 \
  --train-sizes 128,256,512,1024 \
  --seeds 0,1,2 \
  --device cpu \
  --parallel-workers 3 \
  --n-val 128 \
  --n-test 256 \
  --n-epochs 12 \
  --hidden-dim 160 \
  --batch-size 24 \
  --torch-threads 1 \
  --json-summary outputs/learned_sketch_sampling_sweep_summary.json \
  --raw-csv outputs/learned_sketch_sampling_sweep_raw.csv \
  --agg-csv outputs/learned_sketch_sampling_sweep_agg.csv
```

### Larger aggressive run

```bash
source venv/bin/activate
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
python3 -u scripts/run_learned_sketch_sampling_sweep.py \
  --state-dims 32,64,96,128 \
  --train-sizes 256,512,1024,2048 \
  --seeds 0,1,2 \
  --device cuda \
  --cuda-devices 2,3 \
  --parallel-workers 2 \
  --n-val 128 \
  --n-test 256 \
  --n-epochs 12 \
  --hidden-dim 160 \
  --batch-size 24 \
  --torch-threads 1 \
  --json-summary outputs/learned_sketch_sampling_sweep_large_summary.json \
  --raw-csv outputs/learned_sketch_sampling_sweep_large_raw.csv \
  --agg-csv outputs/learned_sketch_sampling_sweep_large_agg.csv
```

### Stronger learning preset

```bash
source venv/bin/activate
python3 scripts/run_learned_sketch_simulation.py \
  --state-dims 32,64 \
  --train-sizes 256,512,1024 \
  --n-epochs 18 \
  --hidden-dim 192 \
  --cpu \
  --json-summary outputs/learned_sketch_simulation_strong_summary.json \
  --csv-summary outputs/learned_sketch_simulation_strong_summary.csv
```

### Generate figure

```bash
source venv/bin/activate
python3 scripts/plot_learned_sketch_simulation.py \
  --json-summary outputs/learned_sketch_simulation_summary.json \
  --output outputs/learned_sketch_simulation.png
```

### Plot aggressive sampling sweep

```bash
source venv/bin/activate
python3 scripts/plot_learned_sketch_sampling_sweep.py \
  --json-summary outputs/learned_sketch_sampling_sweep_summary.json \
  --output outputs/learned_sketch_sampling_sweep.png
```

## Interpretation guide

- If `train_size` increases and learned errors decrease, the model is learning from oracle queries.
- If `distance_to_hll_floor_rel_rmse` moves toward 0, learned performance approaches the matched-memory theoretical floor.
- If `state_dim` is too small, error plateaus (structural under-capacity).
- If `latent_merge_state_mse` decreases alongside `abs_rel_error_gap_vs_hll`, the proxy latent merge objective is acting as a useful control signal.
- Treat rows with `evidence_status="proxy_only"` as empirical baselines, not Lean-backed local-law evidence.
- HLL schedule spread should be near zero (exact mergeability sanity check).
- In the multi-seed sweep, if `train_size` increases and both mean error and std decrease, sampling is improving both average performance and stability.
