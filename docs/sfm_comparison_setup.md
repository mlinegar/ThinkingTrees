# SFM-Constrained Comparison Setup

This benchmark enforces SFM-style constraints directly:

1. Fixed memory: PCSA sketch size is `B x P` bits.
2. Fixed privacy budget: each local sketch uses `epsilon`; merged sketches are evaluated at effective `epsilon*`.

## What is implemented

Script: `scripts/run_sfm_comparison.py`  
Core module: `src/tree/private_sfm_comparison.py`

Methods:

- `pcsa_non_private_mle`: non-private PCSA + composite-likelihood MLE
- `sfm_sym_randmerge_mle`: local `Msym` + randomized merge `g` (Theorem 4.8) + MLE
- `sfm_xor_detxor_mle`: local `Mxor` + deterministic xor merge (Theorem 4.4 path) + MLE
- `sym_local_detor_mle`: local `Msym` + deterministic or merge (theorem counterfactual)
- `sym_local_detxor_mle`: local `Msym` + deterministic xor merge (theorem counterfactual)
- `hll_non_private`: non-private HLL baseline (memory matched to `B x P` bits)
- `ours_ridge_sym_local_detor`: learned decoder on local-`Msym` deterministic-or merged sketches

Primary outputs:

- `rrmse`
- `mean_abs_rel_error`
- `mse`
- `rel_eff_vs_sfm_sym` (MSE ratio against `sfm_sym_randmerge_mle`)
- `channel_calibration_l1` (bit-channel mismatch vs target `M_{p,q}` channel; lower is better)

## Privacy/merge details

- `Msym` uses symmetric RR with `p = e^epsilon / (e^epsilon + 1)`, `q = 1 - p`.
- `Mxor` uses asymmetric RR with `p = 1/2`, `q = 1 / (2 e^epsilon)`.
- Effective epsilon after merging `k` sketches with equal `epsilon`:
  - `epsilon* = -log(1 - (1 - exp(-epsilon))^k)`

## Quick run

```bash
source venv/bin/activate
python3 scripts/run_sfm_comparison.py \
  --n-values 100,1000,10000 \
  --n-trials 50 \
  --merge-counts 1,2,8 \
  --epsilons 0.5,1,2 \
  --buckets 1024 \
  --levels 24 \
  --json-summary outputs/sfm_comparison_summary.json \
  --csv-summary outputs/sfm_comparison_summary.csv
```

## Notes

- To mirror SFM defaults more closely, use `--buckets 4096 --levels 24` and larger `--n-trials`.
- `ours_ridge_sym_local_detor` is included to test whether learned decoding can recover utility under the same memory/privacy constraints even when deterministic merge breaks exact commutation.
