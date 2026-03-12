# Non-Separable Preference Suite Spec

## Scope
- Priority: simple, explicit non-separable DGPs with optimization-separation gates.
- This pass includes no heavy neural-operator training.
- Outputs:
  - `scripts/run_nonseparable_preference_suite.py`
  - `scripts/plot_nonseparable_preference_suite.py`
  - JSON/CSV summaries + figure/report artifacts.

## DGPs
1. `dgp1_complementarity_and`
- Utility: `u = 1{c_left >= k1 and c_right >= k2}`.
- Bounded in `[0, 1]`.
- Separation mechanism: missing one side breaks complementarity.

2. `dgp2_boundary_interaction`
- Utility: `u = sigmoid(2.5 * (theta·unigram + lambda·bigram))`.
- Includes cross-boundary bigram contribution.
- Bounded in `[0, 1]`.
- Separation mechanism: methods without endpoint-aware statistics miss boundary term.

## Method Arms
- `oracle`
- `supported_merge_safe`
- `undersupported_sketch`
- `right_rule_wrong_chunker`
- `naive_non_merge_safe`

## Metrics
- Primary: mean gap-to-oracle pairwise loss (DPO-style surrogate gap).
- Secondary:
  - mean utility regret,
  - mean conditional-event bias,
  - empirical coverage (Wilson CI hit rate).
- Bound diagnostics:
  - per-arm mean bound envelope,
  - consistency check: `observed_regret <= bound_envelope + tolerance`.

## Separation Gates
- For each DGP and misspecified arm:
  - `delta = mean_gap(arm) - mean_gap(supported)`.
  - Gate pass if `delta >= 0.05` and 95% CI lower bound `> 0`.
- Strong separation target:
  - at least one misspecified arm with `delta >= 0.10` and CI lower bound `> 0`.

## DGP-Implied Bound Path
- Emitted per-candidate components:
  - support insufficiency indicators,
  - missed required events (DGP-1),
  - missing boundary term and dropped mass (DGP-2).
- Envelope is aggregated per pair and per replicate.
- Flag stress cells where `observed_regret - bound_envelope > 0.02`.

## Interface Hooks for Neural Operators
- Keep estimator interface separable from DGP/estimand logic:
  - `estimate(candidate, arm, config, rng) -> (score, bound_components)`.
- Required capability map:
  - DGP-1: dual-sided threshold support (left/right sufficient stats).
  - DGP-2: endpoint-aware merge state for cross-boundary terms.
  - Both: calibrated bounded score output in `[0,1]`, plus bound component emit.
- Follow-on checklist:
  1. Add operator-backed `supported_merge_safe` estimator preserving current stat contracts.
  2. Add operator ablations that remove endpoint state and/or reduce sketch order.
  3. Re-run separation gates with operator arms and verify bound consistency.
  4. Track compute/memory tradeoffs alongside effect-size gate results.
