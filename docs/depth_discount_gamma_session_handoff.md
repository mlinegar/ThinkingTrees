# Depth Discount Gamma — Session Handoff (2026-04-09)

## Contract Status

This handoff now reflects the repaired parity contract rather than the older standalone gamma sweep.

The authoritative depth-discount / parity path is:

- `scripts/run_markov_optimization_tradeoff_pipeline.py`
- `tree_supervision_source="manifest"`
- `local_estimand_mode="span_mass_ipw_sum"`
- `c2_pair_weighting_mode="pair_ipw_geomean"`

See [docs/fno_parity_session_handoff.md](/home/mlinegar/ThinkingTrees/docs/fno_parity_session_handoff.md) for the matching parity-contract summary.

## What Changed

### 1. Pipeline is now the only authoritative gamma path

- `scripts/sweep_depth_discount_gamma.py` is now a thin wrapper over the supervision-recovery pipeline.
- The wrapper launches the requested parity-authoritative grid:
  - scopes `recoverable_v4` and `r12_seg10to12`
  - train docs `10240`
  - packages `full100` and `r100_mass_local_eq_15p0`
  - leaf tokens `32,16,8`
  - seeds `0,1`
  - gammas `1.0,0.9,0.75`
- The supervision-recovery pipeline now carries `depth_discount_gamma` through task ids, configs, reduced summaries, runtime payloads, and report rows.
- Tree-reference overrides no longer clobber the gamma axis back to `1.0`.

### 2. Legacy low-level runner is explicitly legacy

- `run_markov_changepoint_ops_count_experiment()` now rejects parity-only settings.
- It fails fast unless:
  - `tree_supervision_source == "rate"`
  - `tree_local_weighting_mode == "fixed_k_hajek"`
- This prevents the old low-level path from silently running non-authoritative “parity” experiments under the wrong estimand.

### 3. C2 is now part of the repaired parity contract

The earlier gap was real: C1/C3 had a repaired manifest-side node-IPW path, but C2 was not yet aligned across fused and non-fused execution.

That gap is now closed for authoritative manifest runs:

- Node scales are:
  - root: `1.0`
  - leaf/internal: `span_mass * gamma^depth`
- Pair scale is:
  - `sqrt(node_scale_i * node_scale_j)`
- Pair inclusion probabilities follow the uniform-subset design implied by the realized deterministic subset sizes:
  - leaf-leaf: `k_leaf (k_leaf - 1) / (n_leaf (n_leaf - 1))`
  - merge-merge: `k_merge (k_merge - 1) / (n_merge (n_merge - 1))`
  - leaf-merge: `(k_leaf / n_leaf) * (k_merge / n_merge)`
  - root-leaf: `k_leaf / n_leaf`
  - root-merge: `k_merge / n_merge`
- Pair weights are Horvitz-Thompson:
  - `pair_weight(i,j) = pair_scale(i,j) / pair_inclusion_prob(i,j)`

The same weighted C2 semantics now apply in:

- fused fast-mask C2
- mask-based pairwise helper
- list-based fallback path

Legacy `rate` + `fixed_k_hajek` training keeps the old unweighted C2 behavior and is labeled legacy.

### 4. Reporting and diagnostics are now explicit

Supervision-recovery payloads and summaries now carry:

- `tree_supervision_source`
- `local_estimand_mode`
- `depth_discount_gamma`
- `c2_pair_weighting_mode`
- `c2_same_pair_count`
- `c2_different_pair_count`
- `c2_pair_weight_ess`
- `c2_pair_weight_max`

Report rows also mark whether a row is authoritative for gamma/parity interpretation:

- `is_authoritative_parity_row`
- `is_authoritative_gamma_row`

These are only true when the repaired manifest/IPW contract is present.

## Lean Alignment

The Lean motivation still matters, but the claims need to be narrower than the older handoff implied.

- C1/C3 node weighting is now aligned with the repaired sampled objective:
  - manifest supervision
  - `span_mass_ipw_sum`
  - optional `gamma^depth`
- C2 alignment is now achieved through the sampled-pair objective above, not by the earlier legacy Hajek-style checks.

The old statement that “depth discounting exists in Lean but not Python” is no longer true for the parity path.

## What Is Now Legacy-Only

These should not be used for repaired parity claims:

- the old standalone depth-discount sweep logic that ran directly through the low-level ops-count path
- any result with `tree_supervision_source != "manifest"`
- any result with `local_estimand_mode != "span_mass_ipw_sum"`
- any gamma/C2 study with `c2_pair_weighting_mode != "pair_ipw_geomean"`
- legacy Hajek/IPW tests that do not exercise the production manifest + sampled-objective path

## Test Coverage Added

Production-path tests now cover:

- explicit manifest supervision disabling fallback rate-based local sampling
- `span_mass_ipw_sum` matching hand computation
- C2 pair inclusion probabilities for:
  - leaf-leaf
  - merge-merge
  - leaf-merge
  - root-leaf
  - root-merge
- weighted C2 helper agreement across direct, mask-based, and batched implementations
- `gamma=0` causing authoritative sampled C2 to vanish
- low-level runner rejection of parity-only supervision modes
- supervision-recovery pipeline gamma-axis task expansion and gamma-distinct aggregation rows
- manifest planner geometry / contract checks on actual prepared-bundle geometry

## Important Corrections to the Older Handoff

The older depth-discount handoff is stale on three points:

1. The standalone gamma sweep is no longer authoritative.
2. “C2 unresolved” is no longer accurate for the manifest parity path.
3. Legacy Hajek tests do not certify the repaired production estimator by themselves.

## Remaining Work

The implementation and tests are in place. The full requested `10240`-doc diagnostic reruns have not been launched from this handoff update.

Those reruns should be interpreted only if the emitted rows show:

- `tree_supervision_source="manifest"`
- `local_estimand_mode="span_mass_ipw_sum"`
- `c2_pair_weighting_mode="pair_ipw_geomean"`
