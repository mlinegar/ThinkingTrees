# Identifiable-Zero: Neural-Operator Visualization Gaps (Paper-Oriented)

This note is a checklist of **what we still cannot “see”** clearly from the current appendix report, and what **CPU sweeps** we run to generate publishable visuals.

## Markov neural merger (LearnedCountSketch)

### Gaps in current visuals

1. **Information density / capacity:** we do not show how `state_dim`, `hidden_dim`, or `feature_mode` changes the curve `root_mae(q_infer)` and `schedule_spread(q_infer)`.
2. **Guidance semantics interaction:** we do not show when `guidance_override_mode=reset` vs `adjust` matters (or *doesn’t*) as model capacity changes.
3. **Associativity pressure:** we only have one regularization value; we need a small `schedule_consistency_weight` sweep to show when schedule sensitivity collapses.

### Visuals to add (paper-friendly)

- **Root/merge error vs `q_infer`**, faceted by:
  - `state_dim` (lines) and `feature_mode` (panels), with fixed `q_train` and fixed regularization.
- **Schedule spread vs `q_infer`** (same faceting).
- Optional: **effective guidance coverage** (`effective_q_mean`) vs `q_infer` to show the stochastic selection is behaving as expected.

## C-TreePO topic operator (phi estimators + neural refiner)

### Gaps in current visuals

1. **Information density:** we need `phi_error(topic_phi_docs)` scaling curves for multiple estimators.
2. **Neural refiner regime map:** we need to see whether `neural_ctreepo` only helps when seed fraction is large, or when base estimator is weak.

### Visuals to add (paper-friendly)

- **Phi L2 error vs `topic_phi_docs`** for:
  - `spectral_numpy`, `tensor_lda`, `online_tensor_lda`, and `neural_ctreepo` (with multiple seed fractions).
- **Downstream root error at `q_infer=0`** vs `topic_phi_docs` (same overlay).
- Optional: scatter **phi error vs root error** across the whole sweep to show correlation and failure regimes.

## Overnight CPU sweeps

The default overnight sweep script builds and runs the two “density” sweeps above:

- `venv/bin/python -m src.ctreepo.cli sim suite identifiable-zero-neural-operator ...`

It writes outputs under an `OUT_ROOT` like:

- `outputs/identifiable_zero_suite_20260303_longrun_equiv_v1_neural_operator_overnight_<timestamp>/`

and logs under:

- `logs/<timestamp>_neural_operator_overnight_run.log`
