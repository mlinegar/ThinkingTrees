# Adaptive Chunking With Honest Splits

This note formalizes the adaptive chunking intuition and ties it to the Python API.

## Core Setup

For document `x`, let `P_theta(x)` be a chunk partition policy (boundaries) and let `u_i(x)` be a bounded per-leaf diagnostic (for example a local oracle error proxy, law violation score, or normalized prediction error).

Adaptive chunking wants to choose `theta` so that high-information regions get finer leaves and low-information/noisy regions get coarser leaves.

## Why Honesty Is Required

If the same data is used both to:
1. choose partition boundaries (`theta`), and
2. estimate oracle error under those boundaries,

then the estimate is optimistically biased by selection on noise.

Simple argument:
- Suppose two candidate partitions `A, B` have equal true risk `R`.
- Empirical estimates are `R_hat(A) = R + eps_A`, `R_hat(B) = R + eps_B`.
- If you choose `theta = argmin{R_hat(A), R_hat(B)}`, then
  `E[R_hat(theta)] = R + E[min(eps_A, eps_B)] < R`.
- You "improve" by fitting noise, not signal.

This is the same failure mode that motivates honest trees in random forests and causal forests.

## Comparison To RF / GRF

Random forests (RF) and generalized random forests (GRF) use an "honest" split:
- one subsample chooses tree structure,
- a disjoint subsample estimates leaf values/effects.

Our chunking setting is structurally analogous:
- boundary split chooses chunk boundaries (partition policy),
- evaluation split estimates oracle/prediction error under those boundaries.

Mapping:
- RF/GRF split criterion ↔ chunk boundary adaptation criterion.
- RF/GRF leaf estimation ↔ held-out oracle error estimation per document/chunk regime.
- Out-of-bag (OOB) estimates ↔ held-out `evaluation` split metrics in pipeline reports.

Key difference:
- RF/GRF partitions feature space; we partition document text into leaves.
- But the same selection-bias mechanism applies, so the same honesty principle is needed.

## Honest Policy

Use a deterministic split per sample/document:
- boundary split: used only to build/update chunk boundaries.
- evaluation split: used only to estimate oracle performance and report metrics.

Conditioning on the boundary split, evaluation remains out-of-sample for boundary decisions, breaking chunk-boundary/oracle-noise feedback loops.

## What Goes Wrong Without Honesty

- Inflated gains: apparent oracle improvement disappears on fresh data.
- Boundary collapse: chunker overreacts to transient proxy spikes.
- Error correlation leakage: chunk placement and oracle residuals become mechanically coupled.
- Invalid uncertainty: concentration/IPW intervals calibrated for honest evaluation become anti-conservative.

## Python Mapping

Implemented interfaces in `src/preprocessing/chunker.py`:

- `HonestChunkingPolicy`: boundary/evaluation split configuration.
- `assign_honest_split(sample_id, policy)`: deterministic split assignment.
- `AdaptiveChunkMemory.update_signals(..., honest_role=...)`: role-tagged feedback storage.
- `AdaptiveChunkMemory.get_signals_for_chunking(..., honest_policy=...)`: boundary-only retrieval.
- `AdaptiveChunkMemory.get_signals_for_evaluation(..., honest_policy=...)`: held-out retrieval.
- `feedback_from_prediction_errors(..., honest_role=...)`: builds role-tagged feedback.
- `chunk_for_ops(..., adaptive_config=..., feedback_signals=...)`: adaptive chunking execution.

Tree-level wiring:
- `src/tree/builder.py` now passes adaptive config/signals through `BuildConfig`.
- Tree metadata includes `chunk_boundaries` and `chunking` provenance for auditability.
- `src/training/run_pipeline.py` now:
  - resolves adaptive/honest policy from CLI + `config/settings.yaml`,
  - assigns deterministic honest splits by `doc_id`,
  - stores doc-level feedback in `AdaptiveChunkMemory`,
  - applies boundary-only signals during tree building.
  - applies three-layer training filters when enabled:
    - chunker feedback updates use `chunk=train` docs,
    - tree/demos/preferences use `summarizer=train` docs,
    - scorer optimization objective uses `oracle=train` trainset and `oracle=eval` valset.

## Recommended Workflow

1. Assign each sample/document a split with `assign_honest_split`.
2. For boundary split only: compute feedback signals and update `AdaptiveChunkMemory`.
3. Build chunks with `get_signals_for_chunking`.
4. Evaluate oracle metrics on evaluation split only with `get_signals_for_evaluation` (or raw held-out labels), without updating boundaries from those metrics.

This preserves adaptivity while maintaining honest error measurement.

## Three-Layer Honesty (Thinking-Trees)

For this project, honesty is needed at three layers:

1. Chunker (`C`): adaptive boundary policy.
2. Summarizer (`S`): summary generator over chunked text.
3. Oracle/scorer (`O`): evaluator used for training signals and metrics.

If any layer reuses in-fold noise to both *adapt* and *evaluate*, reported gains are biased.

One clean decomposition is:

- `C` trains on `T_C`, evaluates on `E_C`.
- `S` trains on `T_S`, evaluates on `E_S`.
- `O` trains/calibrates on `T_O`, produces out-of-fold signals on `E_O`.
- Final report is on `E = E_C ∩ E_S ∩ E_O` (or outer-fold held-out docs that play that role).

This is the direct analogue of GRF "split selection vs estimation", but lifted to three adaptive modules instead of one tree learner.

### Parallelization Guidance

Yes, a lot can run in parallel, with a DAG constraint:

1. Across outer folds: fully parallel.
2. Within a fold:
   - Oracle OOF prediction jobs are parallelizable by shard/batch.
   - Chunker and summarizer training can run in parallel **only if** summarizer training inputs do not depend on newly updated chunk boundaries from that same job.
   - If summarizer consumes the updated chunker output for that fold, run `O → C → S` sequentially for that fold.

Practical compromise:
- Parallelize by fold and by model-serving batch.
- Preserve per-fold dependency order when one stage consumes another stage's freshly updated artifacts.

## Additional Ideas To Import From RF/GRF Literature

1. K-fold cross-fitting for chunk policies.
   Use rotating boundary/evaluation folds to reduce variance versus a single split.
2. Subsampled chunk-policy fitting.
   Fit boundary policies on document subsamples and aggregate; this stabilizes boundary placement.
3. Minimum leaf-mass style constraints.
   Enforce minimum chunk length / token mass and boundary smoothness penalties to avoid brittle micro-chunks.
4. Orthogonalized proxy signals.
   Residualize low-info/noise proxies against baseline predictors before adaptation (GRF-style nuisance robustness).
5. Split-seed variance reporting.
   Repeat with multiple honest seeds and report variability of held-out MAE (IJ/bootstrap-style uncertainty analogue).
6. OOB-like diagnostics.
   Track `boundary` vs `evaluation` MAE gaps as an overfitting alarm for chunk-policy updates.

### Minimal Python skeleton

```python
from src.preprocessing.chunker import (
    AdaptiveChunkMemory,
    AdaptiveChunkingConfig,
    HonestChunkingPolicy,
    assign_honest_split,
)
from src.tree.builder import BuildConfig, TreeBuilder

honesty = HonestChunkingPolicy(enabled=True, boundary_fraction=0.5, split_seed=42)
memory = AdaptiveChunkMemory()

split = assign_honest_split(doc_id, honesty)
if split == honesty.boundary_role:
    # derive feedback from boundary data only, then store:
    # memory.update_signals(doc_id, signals, honest_role=honesty.boundary_role)
    pass

build_cfg = BuildConfig(
    max_chunk_chars=2000,
    adaptive_chunking=AdaptiveChunkingConfig(enabled=True),
    chunk_feedback_signals=memory.get_signals_for_chunking(doc_id, honest_policy=honesty),
)
builder = TreeBuilder(strategy=strategy, config=build_cfg)
result = await builder.build(text, rubric)
```

## Scope Clarification: What Honesty Is For

Honesty is not a claim that every training stage must be disjoint in every workflow.
It is a requirement for **valid evaluation and uncertainty claims under adaptivity**.

For this project, the goal is not only downstream optimization but also:
- minimizing human labeling effort,
- reporting reliable per-stage diagnostics,
- and supporting IPW/concentration-style certificates.

That goal requires honest evaluation splits.

The GRF analogy should be interpreted narrowly:
- same selection-on-noise failure mode,
- different structural role for trees (GRF trees are estimators; ThinkingTrees chunkers are adaptive preprocessing policies).

## Interference Model With a Trained Oracle Approximator

At round `t`, define:
- `C_t`: chunk boundary policy.
- `S_t`: summarizer/prompt construction policy over chunks.
- `O_t`: learned oracle approximator for target truth labels.
- `O_t^online`: online/update view of `O_t` used for adaptation.
- `O_t^eval`: frozen or OOF view of `O_t` used for reporting/estimation.
- `Q_t`: query policy deciding which units receive expensive human labels.
- `Y*`: target truth label function (human annotation or trusted dataset labels).

Interference paths that create bias if not isolated:
- `O -> C`: oracle residuals/predictions drive boundary updates.
- `C -> S`: boundary choices change summarizer input distribution.
- `S -> O`: summary style shifts oracle training/eval distribution.
- `Q -> {C,S,O}`: adaptive labeling changes observed data distribution for all modules.

Because `O` is trained, these paths form a closed loop; this is the motivation for three-layer honesty.

## Optional Proxy Model For Chunking

Core formalization only needs one oracle model class `O`.

An additional cheap proxy model `P` (small LM or embedding+head) is optional:
- `P` can be used for high-throughput chunking/query prioritization.
- Final evaluation claims must remain anchored to truth labels and honest `O^eval` evaluation.
- If no proxy is used, set `P = O_t^online` operationally and keep the same honesty contract.

Cold start rule of thumb:
- Do **not** turn on embedding-based span feedback until `P` is trained on
  trusted labels / oracle scores; otherwise it’s just noise. The current
  training pipeline enforces this by only enabling span feedback when a trained
  proxy artifact is available (see `docs/pipeline_ordering.md`).

This keeps theorem statements simple while preserving practical flexibility.

## Truth Labels: Human vs Dataset

Truth labels can come from:
- human annotation (`human` source), or
- trusted existing data labels (`dataset` source).

In Lean, this is represented by `TruthLabelSource` and `LabelObservation`
(`lean3/FormalProofs/DSL/CoreDefinitions.lean`), so downstream estimators can
audit mixed-source label provenance explicitly.

## Minimal Honesty Contract (Single Oracle, Triple-Layer)

Assign deterministic document roles:
- `r_C(x) in {train, eval}`
- `r_S(x) in {train, eval}`
- `r_O(x) in {train, eval}`

Rules:
1. Update `C` only on docs with `r_C=train`.
2. Update `S` only on docs with `r_S=train`.
3. Update `O^online` only from truth-labeled docs with `r_O=train`.
4. Use `O^eval` as frozen/OOF for evaluation and DR nuisance prediction on `r_O=eval`.
5. Any oracle predictions used to update `C`/`S` should be out-of-fold with respect to `r_O`.
6. Final reporting set is `E = E_C ∩ E_S ∩ E_O`; no gradient/state updates from `E`.
7. Log joint propensities for queried units and enforce positivity floor `pi >= epsilon > 0`.
8. If iterative rounds are used, evaluate each frozen snapshot prequentially before further updates.

This is the minimum contract that blocks the main leakage channels while preserving adaptivity.

## Estimation With Sparse Human Labels

For queried units `i` with indicator `Z_i`, propensity `pi_i`, and human-derived target `R_i`:

- HT/IPW estimator:
\[
\hat{J}_{\mathrm{HT}} = \frac{1}{n}\sum_i \frac{Z_i}{\pi_i} R_i
\]

- Cross-fitted DR estimator (preferred in practice):
\[
\hat{J}_{\mathrm{DR}} =
\frac{1}{n}\sum_i
\left[\hat{m}_{-k(i)}(v_i) + \frac{Z_i}{\pi_i}\left(R_i - \hat{m}_{-k(i)}(v_i)\right)\right]
\]

where `hat{m}` is an out-of-fold outcome model from `O^eval`.

## Worst-Case Error Envelope (Formal)

Let:
- `G*` = true oracle gap we want to certify.
- `G^J` = oracle/judge-view gap on the honest eval view.
- `G^E` = sampled/IPW estimate of that gap.
- `G^C` = clipped estimate used in computation.

Assume envelopes:
- calibration: `|G* - G^J| ≤ B_cal`,
- estimation: `|G^J - G^E| ≤ B_est`,
- clipping: `|G^E - G^C| ≤ B_clip`.

Then the worst-case envelopes are:
\[
|G^*| \le |G^E| + B_{\mathrm{cal}} + B_{\mathrm{est}}
\]
\[
|G^*| \le |G^C| + B_{\mathrm{cal}} + B_{\mathrm{est}} + B_{\mathrm{clip}}.
\]

One-shot high-probability form:
\[
\Pr\!\left(|G^*| \ge |G^E| + B_{\mathrm{cal}} + B_{\mathrm{est}}\right)
\le \delta_{\mathrm{cal}} + \delta_{\mathrm{est}}
\]
\[
\Pr\!\left(|G^*| \ge |G^C| + B_{\mathrm{cal}} + B_{\mathrm{est}} + B_{\mathrm{clip}}\right)
\le \delta_{\mathrm{cal}} + \delta_{\mathrm{est}} + \delta_{\mathrm{clip}}.
\]

Lean theorem hooks:
- `treepo_gap_with_calibration_and_estimation`
- `treepo_gap_with_calibration_estimation_clipping`
- `dsl_abs_gap_bound_from_estimate`
- `dsl_abs_gap_bound_from_clipped_estimate`
- `dsl_abs_gap_bound_from_estimate_high_prob`
- `dsl_abs_gap_bound_from_clipped_estimate_high_prob`
- `dsl_abs_gap_bound_from_estimate_high_prob_total`
- `dsl_abs_gap_bound_from_clipped_estimate_high_prob_total`
- `dsl_bound_valid_from_events`
- `dsl_bound_valid_from_events_total`
- `computeDSLBound_valid_from_events`
in `lean3/FormalProofs/DSL/TreeIPW.lean`.

For the three adaptive components (chunker/summarizer/oracle), we also have:
- deterministic envelope: `threeLayer_abs_envelope`
- failure-probability union bound: `threeLayer_error_union_bound`
in `lean3/FormalProofs/DSL/Honesty.lean`.

### Bernstein vs Serfling/BM

For applied one-shot certificates, use whichever bound matches the sampling mechanism at that layer:

- Empirical Bernstein: for non-uniform/IPW-weighted estimators and heteroskedastic leaf/document contributions.
- Serfling (finite-population WOR): for within-document chunk audits sampled without replacement.
- BM/Azuma/Freedman-style martingale bounds: for adaptive/sequential chunk querying where WOR exchangeability is broken.

In most TreePO runs, the best stack is additive:

- within-doc chunk audit radius from Serfling (or BM if adaptive-sequential),
- across-doc/IPW estimation radius from empirical Bernstein,
- then compose with the one-shot DSL envelope + union bound.

### Assumption Ledger (Current)

- No tautological `True := by trivial` validity theorem remains in the TreePO DSL path.
- Final bound validity is now event-based: `computeDSLBound_valid_from_events`.
- Empirical-Bernstein concentration interfaces in TreePO are now event-based.
- Calibration integration in TreePO is now event-based (`h_rmse_upper`), with
  `*_from_axioms` compatibility wrappers retained.
- Adaptive sampling honesty contracts (`AdaptiveSamplingAxioms`) now use concrete
  constraints (positivity/boundedness/measurability), not tautological placeholders.

## Final Applied Protocol (Per Round)

1. Freeze `(C_t, S_t, O_t^online, O_t^eval, Q_t)`.
2. Build candidate units `(x, u, p)` from current chunking/summarization.
3. Query truth labels (human or dataset) using `Q_t`; log `(Z, pi, versions, roles, truth_source)`.
4. Train/update `O_{t+1}^online` on `r_O=train` truth labels (weighted/corrected by propensity when needed).
5. Produce `O_{t+1}^eval` as frozen snapshot or strict OOF predictions from oracle-train folds.
6. Optional: update proxy `P_{t+1}` for chunk/query speed (not required for guarantees).
7. Update `C_{t+1}` using allowed train-role feedback (prefer OOF oracle signals).
8. Update `S_{t+1}` on `r_S=train` docs (respecting dependency order when needed).
9. Evaluate frozen round-`t` performance on `E` with IPW/DR and split-aware diagnostics.
10. Repeat until the marginal value of new truth labels is below threshold.

## Practical Plan To Finalize Documentation

1. Keep GRF language as a limited analogy (selection bias only), not structural identity.
2. Add explicit distinction `Y*` (truth target) vs `O` (trained approximator) and optional proxy `P`.
3. Document the eight-rule honesty contract above as normative.
4. Define required logging schema: `doc_id`, roles, policy versions, propensities, oracle model id, round id.
5. Standardize reported diagnostics: train/eval gaps per layer, ESS, calibration by split, seed variance.
6. Add ablations under fixed truth-label budget: `single-oracle`, `single-oracle+proxy`.
