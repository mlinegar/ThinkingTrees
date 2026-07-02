# Markov Contextual-Sufficiency Ablation Handoff (2026-05-05)

This is the follow-up handoff after the optimize-to-zero resolution. It records
what was run after `learned_local_laws` became the exact-zero path, what changed
in code, what the ablations found, and how the results connect to the Lean
formalization. It is intended for another LLM or engineer to resume without
reconstructing the session.

For the latest top-level snapshot, including the follow-up regime-one-hot
recovery grid and the current FNO bridge state, start with
[`markov_sim_status.md`](markov_sim_status.md).

## Short Status

The Markov contextual-sufficiency thread is empirically resolved for the
theorem-sketch lane:

- `learned_local_laws + markov_exact_sketch` remains the exact-zero path.
- Adding NASS/NASSS as an auxiliary can help slightly, but it does not replace
  the local-law / sketch-supervision signal.
- Learned merge and learned decoder variants work inside the local-law lane
  because the state is still supervised toward the Markov sufficient sketch.
- A standalone general learned `f/g` path (`CleanUnifiedNO`) improves with
  smaller leaves and contextual loss, but it still does not discover the exact
  Markov law on its own.

Canonical report artifact:

- [`outputs/markov_contextual_ablation_grid_report_20260505.md`](../outputs/markov_contextual_ablation_grid_report_20260505.md)

Aggregate JSON artifacts:

- `outputs/optimize_to_zero_fg_architecture_ablation_t128/summary.json`
- `outputs/optimize_to_zero_laws_hybrid_grid_t128/summary.json`
- `outputs/clean_unified_fg_contextual_ablation_t128/summary.json`

## What We Ran

Three detached grids completed successfully through `scripts/long_job.py`.

| grid | runner | rows | output root | purpose |
|---|---|---:|---|---|
| JAX f/g architecture | `scripts/run_optimize_to_zero_fg_architecture_ablation.sh` | 36 | `outputs/optimize_to_zero_fg_architecture_ablation_t128/` | Remove exact Markov merge/readout one piece at a time inside `learned_local_laws`. |
| Hybrid NASS/NASSS + laws | `scripts/run_optimize_to_zero_laws_hybrid_grid.sh` | 42 | `outputs/optimize_to_zero_laws_hybrid_grid_t128/` | Test package-style NASS/NASSS auxiliary losses while local laws stay active. |
| CleanUnifiedNO general f/g | `scripts/run_clean_unified_fg_contextual_ablation.sh` | 15 | `outputs/clean_unified_fg_contextual_ablation_t128/` | Test the general learned leaf adapter + shared learned `g` + learned `f` surface without exact Markov merge/decoder installed. |

All three launchers finished with `systemd result=success` and wrote aggregate
`summary.json` files at their output roots.

## Main Results

| grid | best row | best metric | interpretation |
|---|---|---:|---|
| JAX f/g architecture | `nasss/w_0/learned_merge/c2_self_consistency/leaf_1` | contextual MAE `1.53e-5` | Learned merge works when the local-law lane still supervises the sufficient sketch; all architecture groups kept first/last accuracy at `1.0 / 1.0`. |
| Hybrid NASS/NASSS + laws | `nasss / regime_one_hot / leaf=1 / w=0.1` | contextual MAE `1.47e-5` | NASSS auxiliary at low weight is the best hybrid setting; NASS is weaker and harder encodings still degrade. |
| CleanUnifiedNO general f/g | `contextual_sufficiency/dep_none/leaf_tokens_16` | root MAE `1.1451`, contextual MAE `1.1187` | Smaller leaves help, but generic learned `f/g` is still far from exact-zero. |

The most important diagnostic remains sufficient-state recovery, not contextual
fit alone. In these ablations that means tracking:

- `theta_mae`
- `theta_first_regime_accuracy`
- `theta_last_regime_accuracy`
- `eps_leaf`, `eps_merge`, `eps_idemp`
- contextual/raw MAE as a secondary fit metric

## What Changed In Code

Primary code surface:

- [`src/ctreepo/sim/core/contextual_sbijax.py`](../src/ctreepo/sim/core/contextual_sbijax.py)
- [`scripts/probe_contextual_sbijax.py`](../scripts/probe_contextual_sbijax.py)
- [`src/ctreepo/sim/core/clean_unified_fg.py`](../src/ctreepo/sim/core/clean_unified_fg.py)
- [`scripts/probe_clean_unified_no.py`](../scripts/probe_clean_unified_no.py)

New or newly load-bearing contextual-sbijax config/provenance fields:

| field / flag | purpose |
|---|---|
| `local_law_package_weight` / `--local-law-package-weight` | Adds an opt-in NASS/NASSS-style auxiliary loss inside `learned_local_laws`. Default is `0.0`, so the exact-zero baseline is unchanged. |
| `law_architecture` / `--law-architecture` | Chooses `analytic`, `learned_merge`, `learned_decoder`, or `fully_learned` inside the local-law lane. |
| `c2_merge_target` / `--c2-merge-target` | Chooses `theta` or `self_consistency` for C2 merge supervision. |
| `learned_merge_hidden_dim` / `--learned-merge-hidden-dim` | Hidden width for learned `g(s_left, s_right)`. |
| `learned_decoder_hidden_dim` / `--learned-decoder-hidden-dim` | Hidden width for learned `f(state)` decoder. |

Important implementation detail: `local_law_package_weight=0.0` is the default.
Do not compare new hybrid runs to historical exact-zero artifacts unless this
weight and the architecture flags are explicitly recorded.

The `learned_local_laws` trainer now reports package auxiliary state in
provenance:

- `local_law_package_weight`
- `local_law_package_objective`
- `local_law_package_aux_active`
- `law_architecture`
- `c2_merge_target`
- `merge_network`
- `decoder_kind`

Regression coverage:

- `tests/ctreepo/test_contextual_sbijax.py` includes a tiny
  `learned_local_laws` package-auxiliary run and the broader contextual-sbijax
  suite passed (`29 passed`, with only dependency deprecation warnings).

## How To Reproduce The Three Grids

These scripts are executable and resume by skipping cells that already have a
`summary.json`.

```bash
source venv/bin/activate

scripts/run_optimize_to_zero_fg_architecture_ablation.sh 0
scripts/run_optimize_to_zero_laws_hybrid_grid.sh 3
scripts/run_clean_unified_fg_contextual_ablation.sh 2
```

To detach through the repo launcher:

```bash
./venv/bin/python scripts/long_job.py launch \
  --name optimize_to_zero_fg_architecture_ablation \
  --job-root outputs/optimize_to_zero_fg_architecture_ablation_t128/launcher \
  --cwd /home/mlinegar/ThinkingTrees \
  --no-replace-existing \
  -- ./scripts/run_optimize_to_zero_fg_architecture_ablation.sh 0
```

Use the same pattern for the other two scripts. Status commands:

```bash
./venv/bin/python scripts/long_job.py status --job-root outputs/optimize_to_zero_fg_architecture_ablation_t128/launcher
./venv/bin/python scripts/long_job.py status --job-root outputs/optimize_to_zero_laws_hybrid_grid_t128/launcher
./venv/bin/python scripts/long_job.py status --job-root outputs/clean_unified_fg_contextual_ablation_t128/launcher
```

## Current Interpretation

1. **Local laws are the sufficiency selector.** Package NASS/NASSS objectives
   can fit response signatures while leaving the learned state non-canonical.
   The laws force recovery of the `(count, first, last)` state.
2. **NASSS auxiliary is useful only as an auxiliary.** Low-weight NASSS helps
   in some rows, especially `regime_one_hot` leaf=1. NASS is weaker here.
   Neither should be treated as the primary exact-zero objective.
3. **Learned `g` and learned `f` are viable after state supervision.** In the
   JAX local-law lane, learned merge, learned decoder, and fully learned
   variants all preserve first/last accuracy at `1.0 / 1.0`. That is not the
   same as saying generic `f/g` discovers the law unaided; the local-law state
   target is still active.
4. **CleanUnifiedNO is the honest general f/g test.** It installs no exact
   Markov merge or decoder in the model. Best root MAE is `1.1451` at
   `leaf_tokens=16`, so this path has not solved exact recovery yet.

## Lean Crosswalk

The empirical result is not itself a Lean theorem about SGD. The Lean stack
formalizes the condition the learned state must satisfy and the consequences
once that condition holds.

| Python / artifact concept | Lean anchor | relation |
|---|---|---|
| Exact Markov sketch `(count, first, last)` | `FormalProofs/OPT/MarkovCountSketchExample.lean` (`MarkovCountSketch`, `L1_gExact`, `L2_gExact`, `exactSketch_root_distortion_zero`) | Formal exact-sketch witness for zero root distortion. |
| Markov path / changepoint count semantics | `FormalProofs/OPT/MarkovPathDGP.lean` | Connects path encoding and changepoint count to the sketch. |
| Two-sided contextual sufficiency | `FormalProofs/OPT/ContextualQuerySufficiency.lean` and `FormalProofs/OPT/MarkovSufficiency.lean` | Formalizes that exact `(count, first, last)` suffices for all two-sided Markov contexts; count-only summaries do not. |
| SSS/NASSS slice bridge | `FormalProofs/OPT/SlicedContextualSufficiency.lean`; `FormalProofs/OPT/RandomSlicedContextualSufficiency.lean` | If selected slices cover response fibers, preserving slices implies contextual sufficiency. This is a bridge condition, not a proof that random NASSS training finds the right state. |
| NASS / MI-style auxiliary objectives | `FormalProofs/OPT/DependenceObjectiveProxies.lean` | Symbolic proxy/loss equivalences. Does not formalize Shannon MI estimator correctness or guarantee sufficiency from low loss alone. |
| Hybrid exact + neural summaries | `FormalProofs/OPT/HybridSummarySufficiency.lean`; `FormalProofs/OPT/HybridInformationObjectives.lean` | Product-summary refinement and symbolic hybrid objective algebra. |
| Neural operator / learned f/g bridge | `FormalProofs/OPT/NeuralOperatorTheoremBridge.lean`; `FormalProofs/OPT/NeuralOperatorPreferenceBridge.lean`; `FormalProofs/ML/FNOFormalization.lean` | Interface layer: if the learned operator realizes the required local laws/readout within slack, theorem-side transport applies. It does not prove the current CleanUnifiedNO training loop will discover the exact operator. |
| Public crosswalk | `lean3/docs/PAPER_TO_LEAN_MAP.md`, `lean3/docs/CONTEXTUAL_SUFFICIENCY_AND_CHEN_2020.md`, `docs/literature/contextual_sufficiency/README.md` | Human-readable map from code and literature to Lean theorem surfaces. |

Practical rule for future writeups: say "Lean proves the exact sketch and the
contextual/sliced/local-law implication surface"; do not say "Lean proves the
neural trainer converges." The ablation evidence says the local-law objective
finds the sufficient state in this Markov setup.

## Recommended Next Experiment

The next research step is not more package NASSS iterations. It is a stronger
general f/g learner with local-law-compatible supervision:

1. Keep `CleanUnifiedNO` as the honest general surface.
2. Add direct state/witness supervision analogous to the Markov sketch
   `(count, first, last)` or an auditable proxy for it.
3. Compare against the current `leaf_tokens=16` baseline:
   root MAE `1.1451`, contextual MAE `1.1187`.
4. Report first/last-style sufficient-state diagnostics whenever the target
   admits them; otherwise report law eps metrics and contextual collision
   diagnostics, not only root MAE.

This would bridge the current gap: the local-law JAX lane proves the objective
can select the sufficient state; the CleanUnifiedNO lane tests whether the
production-shaped `f/g` surface can learn it without hard-coded Markov merge or
decoder assumptions.

**Bridge experiment design now lives at**
[`docs/markov_fno_local_law_bridge.md`](markov_fno_local_law_bridge.md).
Audit finding: the PyTorch probe already implements `markov_node_witness`
(every leaf and merge state regressed onto exact `(count, first, last)`) and
`markov_local_laws_fno` (C1 leaf calibration + C2 relational merge + C3
idempotence/range, no exact merge targets) — see
[`scripts/probe_clean_unified_no.py:2728-2742`](../scripts/probe_clean_unified_no.py#L2728-L2742)
and the loss body at
[`:1805`](../scripts/probe_clean_unified_no.py#L1805). The earlier
`clean_unified_fg_contextual_ablation` only exercised
`contextual_sufficiency`; the bridge experiment runs the existing witness
and laws cells on the same `paper_hazard_panel_v1_t128` bundle as the JAX
control, with matched seeds and a unified metric schema.

## Cross-Architecture Parity Matrix

| Concept | JAX `learned_local_laws` | PyTorch `CleanUnifiedNO` | PyTorch `fno_family.py` |
|---|---|---|---|
| Backend | JAX + Haiku + sbijax | PyTorch + neuraloperator FNO | PyTorch + FNO |
| Local laws as training objective | C1+C2+C3 active; four arch variants | `markov_local_laws_fno` (C1 + C2 relational + C3 idempotence) | absent — `local_law_trace_metadata` is diagnostic only |
| Direct sketch witness supervision | `c2_merge_target=theta` | `markov_node_witness` | absent |
| `analytic` exact-Markov merge / decoder | yes | no — exact-Markov f/g lives in `recoverable_v5_t2048` pipeline, not CleanUnifiedNO | n/a |
| `learned_merge` vs `learned_decoder` separability | yes (`--law-architecture`) | no — encoder/g/decoder jointly learned | n/a |
| Reported metrics including `theta_mae`, `eps_*` | yes | needs root-node `theta_mae`, `root_first/last_regime_acc`, `eps_leaf`, `eps_merge` columns added (see bridge doc) | absent |

The parity alignment for the bridge experiment:

- PyTorch `markov_node_witness` ↔ JAX `*/c2_theta`
- PyTorch `markov_local_laws_fno` ↔ JAX `*/c2_self_consistency`
- JAX `analytic` is the gold-reference control; no PyTorch CleanUnifiedNO
  analog (the exact-Markov f/g surface lives elsewhere).
