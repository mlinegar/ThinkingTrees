# Markov Tree Parity Handoff (`_cdx`)

This is the conservative Codex handoff for the Markov tree-parity line. If it conflicts with chat logs, scratch notes, or optimistic interim writeups, prefer this document **when it cites actual repo artifacts, current code, or exported Lean aliases**.

## Executive Summary

The research goal is to show that the randomly sampled tree path recovers the same answer as the internal FNO when the local laws hold, at least in the clean Markov changepoint setting where the theorem-facing sufficient statistic is the exact sketch `(count, first, last)`.

The current blocker is not the exact Markov algebra itself. That part is in good shape. The unresolved issue is whether the learned runtime state actually preserves the theorem-facing Markov sketch well enough for the Lean guarantees to become relevant in practice. The math says root distortion is zero if the right sketch is preserved; the experiments are still trying to make the learned state realize that sketch reliably.

Current working belief:

- exact algebra is verified
- strict local-law training is the primary research line
- oracle parent supervision is diagnostic only, as a feasibility ceiling
- topology claims remain blocked until the one-leaf / exact-leaf theorem-facing path is healthy

## Current Research Contract

These choices are locked unless explicitly changed:

- Primary theorem-facing lane: `opaque carrier / exact theorem-facing sketch`
- Main scientific target: strict `C1/C2/C3` first
- `teacher_parent_full_sketch` is allowed only as diagnostic supervision
- `teacher_parent_count` is diagnostic and known insufficient in Lean
- No hard-coded slotwise semantics for the theorem-facing latent state

### Do Not Claim

- Do not treat oracle parent supervision as the main paper claim.
- Do not treat topology reruns as valid until exact-leaf / one-leaf issues are resolved.
- Do not treat count-only sufficiency as enough for topology-sensitive Markov claims.
- Do not treat active `unified_g` logs as evidence of success unless a finished artifact clearly shows it.

## Lean Map

Lean proves what is true **if the right sketch is preserved**. The current Python experiments are about whether the learned runtime state actually realizes that theorem-facing surface.

| Concept | Lean file | Exported alias / theorem | Runtime meaning |
|---|---|---|---|
| Local-law naming | `lean3/FormalProofs/OPT/LocalLaws.lean` | `C1 = L1`, `C2 = L3`, `C3 = L2` | Paper naming vs theorem naming must stay aligned in reports and code comments. |
| Local-to-global preservation | `lean3/FormalProofs/OPT/MainTheorems.lean` | `multi_round_preservation` | If local laws hold, repeated reduction preserves oracle information globally. |
| Zero distortion corollary | `lean3/FormalProofs/OPT/MainTheorems.lean` | `delta_r_zr_zero_of_local_laws` | Document-level distortion is exactly zero under local laws. |
| Regularized-objective bridge | `lean3/FormalProofs/OPT/MainTheorems.lean` | `certified_regularized_objective` | The optimization-facing objective can include audited / approximate local-law penalties. |
| Markov sufficiency collision result | `lean3/FormalProofs/OPT/MainTheorems.lean` | `markov_sufficiency_collision_implies_exact_sketch_eq` | If a summary is Markov-query sufficient, collisions force equality of the full exact sketch. |
| Markov sufficiency decoder | `lean3/FormalProofs/OPT/MainTheorems.lean` | `markov_sufficiency_has_exact_sketch_decoder` | Successful decoded `(count, first, last)` recovery is a sufficiency witness. |
| Count-only insufficiency | `lean3/FormalProofs/OPT/MainTheorems.lean` | `markov_countOnly_not_sufficient` | Count alone does not support arbitrary topology / context-sensitive changepoint queries. |
| Opaque-carrier exact merge | `lean3/FormalProofs/OPT/MainTheorems.lean` | `markov_opaque_carrier_exact_sketch_merge_exact` | Exact projected merge preserves the theorem-facing Markov sketch. |
| Opaque-carrier zero root distortion | `lean3/FormalProofs/OPT/MainTheorems.lean` | `markov_opaque_carrier_exact_sketch_root_distortion_zero` | If the projected sketch is exact, the root count is exact by Theorem 1. |
| Opaque-carrier C2 route | `lean3/FormalProofs/OPT/MainTheorems.lean` | `markov_opaque_carrier_exact_sketch_L3_of_projection_preserving_reencode` | Projection-preserving reencode is enough for theorem-facing C2/L3. |
| Worked opaque-carrier example | `lean3/FormalProofs/OPT/MainTheorems.lean` | `markov_opaque_carrier_exact_sketch_example_oracle_correct` | The 4-leaf worked example still produces the exact `2`-changepoint answer. |
| Parent full-sketch implies C3 | `lean3/FormalProofs/OPT/MainTheorems.lean` | `markov_exact_parent_fullSketch_implies_L2` | Direct exact parent sketch supervision recovers the Markov `L2/C3` route. |
| Exact leaves + parents give zero root distortion | `lean3/FormalProofs/OPT/MainTheorems.lean` | `markov_exact_leaf_and_parent_fullSketch_zero_root_distortion` | Diagnostic ceiling: exact leaf and parent sketches are enough for exact root answers. |
| Parent count-only is insufficient | `lean3/FormalProofs/OPT/MainTheorems.lean` | `markov_parent_countOnly_not_sufficient` | Count-only teacher guidance does not determine Markov merge correctness in general. |
| Positive weighting preserves zero-loss optimum | `lean3/FormalProofs/OPT/MainTheorems.lean` | `markov_positive_weighted_nodewise_zero_iff` | Weighting changes optimization geometry, not the exact zero-loss optimum. |
| Disjoint-palette recoverability | `lean3/FormalProofs/OPT/MainTheorems.lean` | `piecewise_disjoint_palette_observed_tokens_recover_exact_sketch` | On the clean Markov DGP, observed tokens identify the exact theorem-domain sketch. |
| Zero Bayes error on clean DGP | `lean3/FormalProofs/OPT/MainTheorems.lean` | `piecewise_disjoint_palette_zero_bayes_error` | The clean disjoint-palette setting is recoverable in principle. |

### Intended Interpretation

- Lean does **not** say the current neural runtime already has the right state.
- Lean says that if the runtime preserves the right theorem-facing sketch, then local-to-global preservation follows.
- The sufficiency theorems say that exact decoded recovery of `(count, first, last)` is not cosmetic. It is the right empirical witness for the Markov theorem route.
- The merge-supervision theorems say teacher-guided full parent sketches are a legitimate **diagnostic ceiling** for feasibility, while count-only parent supervision is formally too weak.

## Python / Runtime Map

### `src/ctreepo/sim/core/markov_neural_operator_baselines.py`

This is the main runtime implementation of the Markov tree model family.

Important concepts here:

- merge objective modes:
  - `strict_c3`
  - `teacher_parent_count`
  - `teacher_parent_full_sketch`
- theorem-facing surface modes, including the opaque-carrier exact-sketch lane
- exact projected merge vs learned merger paths
- direct metric emission for theorem-facing diagnostics

Interpretation:

- `strict_c3` is the main local-law training lane
- `teacher_parent_count` and `teacher_parent_full_sketch` are diagnostic comparison lanes
- the opaque-carrier exact-sketch path is the current theorem-facing runtime anchor because it keeps the latent opaque while routing theorem-facing correctness through the decoded Markov sketch

### `scripts/test_markov_exact_progression.py`

This is the current exact-leaf merger-feasibility lab.

It now serves two purposes:

- `Step 0`: verify exact sketch algebra on the worked-example baseline
- `Phase 1`: train exact leaves plus a learned merge path under different merge-objective modes

If older notes refer to `Step 1`, `Step 2`, `Step 3`, use this interpretation:

- `Step 0`: exact sketch + exact merge baseline
- `Step 1`: exact leaves + learned merge feasibility
- `Step 2` / `Step 3`: older shorthand for learned-encoder / full-system progression, but these should now be read through the current harness and current output artifacts rather than chat summaries alone

### `scripts/run_tree_neural_full_doc_mig.py`

This is the broader experiment launcher for:

- representation sufficiency studies
- representation learnability sweeps
- promotion / comparison summaries across learned families and controls

Use this when the question is broader than the exact-leaf merge lab, for example:

- whether the opaque learned state can recover the Markov sketch as data scale grows
- whether exact projected merge removes the learned-merger bottleneck
- whether controls are healthy enough to justify a claim

### `src/ctreepo/sim/core/full_doc_anchor_diagnostics.py`

This is the main theorem-facing diagnostics bridge.

It is responsible for:

- normalizing theorem-facing summary fields
- emitting sufficiency / exact-projection metrics
- separating theorem-facing exact-sketch diagnostics from auxiliary runtime diagnostics

### Metric Vocabulary

Use these names consistently:

- `step1_root_mae`
- `step1_merge_exact_summary_match_rate`
- `step1_count_only_root_mae`
- `step1_endpoint_only_root_mae`
- `merger_grad_norm_root`
- `merger_grad_norm_local`
- `merger_grad_ratio_root_to_local`
- `exact_sketch_markov_sufficiency_gap_score`
- `exact_projected_root_mae`
- `learned_merger_gap`

Recommended interpretation:

- `step1_root_mae`: overall exact-leaf merge quality
- `step1_count_only_root_mae`: residual error if endpoints are made oracle; isolates count-side weakness
- `step1_endpoint_only_root_mae`: residual error if counts are made oracle; isolates endpoint / join weakness
- `merger_grad_ratio_root_to_local`: whether the merger is mainly being taught by root loss or by local merge signal
- `exact_sketch_markov_sufficiency_gap_score`: theorem-facing decoded-sketch failure signal
- `exact_projected_root_mae`: root error under exact projected merge
- `learned_merger_gap`: difference between learned-root path and exact-projected path; helpful for separating merger vs sketch issues

## Established Results vs Hypotheses

### Established

These points are supported by current code and on-disk artifacts.

1. Exact Markov sketch algebra is correct by construction.

- `scripts/test_markov_exact_progression.py` checks the 4-leaf worked example.
- The exact `Step 0` baseline reduces the exact sketches with zero root error.

2. The clean Markov DGP is recoverable in principle.

- Lean exports `piecewise_disjoint_palette_observed_tokens_recover_exact_sketch`.
- Lean exports `piecewise_disjoint_palette_zero_bayes_error`.
- So failure on the clean benchmark is not a theorem-level impossibility result.

3. The current small exact-leaf merge-feasibility study shows a specific pattern.

Primary artifact:

- `outputs/markov_merge_signal_feasibility_small_20260407_171619/merge_signal_feasibility_summary.md`

Concrete pattern from that artifact:

- `strict_c3 + root_loss` currently gives the best tiny-scale `step1_root_mae`
- `teacher_parent_full_sketch` materially improves local merger signal and merge exactness
- `teacher_parent_full_sketch` is stronger than `teacher_parent_count`
- `strict_c3` remains strongly root-loss dominated on gradient ratio

Representative rows from that artifact:

- `strict_c3__root1__depth_balanced__head_scalar_mse__n32`
  - `step1_root_mae = 0.7405`
  - `step1_merge_exact_summary_match_rate = 0.0102`
  - `merger_grad_ratio_root_to_local = 5.3780`
- `teacher_parent_full_sketch__root1__depth_balanced__head_scalar_mse__n32`
  - `step1_root_mae = 0.7524`
  - `step1_merge_exact_summary_match_rate = 0.0289`
  - `merger_grad_ratio_root_to_local = 1.6110`
- `teacher_parent_count__root1__flat_mean__head_scalar_mse__n32`
  - `step1_root_mae = 0.7581`
  - `step1_merge_exact_summary_match_rate = 0.0211`
  - `merger_grad_ratio_root_to_local = 1.6462`

Careful conclusion:

- stronger local teacher signal helps
- at this tiny scale, it has **not yet** overtaken `strict_c3 + root_loss` on actual `Step 1` root MAE

### Diagnostic-Only Evidence

Oracle / teacher-guided findings are useful, but only as feasibility evidence.

Lean licenses this diagnostic interpretation because:

- exact parent full-sketch supervision implies the Markov `L2/C3` route
- exact leaves plus exact parent full sketches imply zero root distortion
- count-only parent supervision is formally insufficient

Runtime meaning:

- `teacher_parent_full_sketch` is a ceiling on what the merger / carrier interface can represent
- it answers “is this interface capable of learning the right merge target if we give it the target directly?”
- it does **not** answer the main paper question, which is whether strict local laws alone are enough in practice

### Open Hypotheses

These are active research hypotheses, not established facts.

#### 1. Leaf-summary bottleneck hypothesis

The leaf path may be compressing or scrambling count information before it reaches the theorem-facing route. Under this view, the merger is not the only issue; the leaf representation may already be discarding or entangling the theorem-facing sketch.

#### 2. Unified-g alignment hypothesis

Leaves and merges may need to pass through the same `encode_summary`-style map for stronger C3 alignment. The current design note for this idea is:

- `/home/mlinegar/.claude/plans/moonlit-squishing-puddle.md`

The core idea there is not “introduce slots,” but “use the same `g`-style route for leaves and merges.”

#### 3. Width / no-bottleneck hypothesis

The issue may be generic compression rather than lack of a special slot layout. Under this view, wider summary surfaces or a less compressive leaf-summary path may preserve theorem-facing information better while keeping the latent opaque.

Rejected default:

- hard-coded slot semantics as the theorem-facing default

## Unified-G and Wider-Summary Discussion

This area needs to be described carefully because it is easy to overfit the conclusion to one optimistic thread.

### Structured Slots

Structured slots would make the sufficient statistic explicit, for example by dedicating coordinates to count, first, and last. That can be useful as a control, but it is too restrictive as the default theorem-facing story. The research target is not “prove a hand-designed slot layout works.” The target is to show an opaque learned state can still realize the theorem-facing sketch.

### Unified `g`

The `unified_g` idea is attractive because it aligns the leaf and merge routes without forcing slot semantics inside the latent state. The design note at `/home/mlinegar/.claude/plans/moonlit-squishing-puddle.md` argues that leaves should go through the same `encode_summary`-style map that merges already use, so the same summarizer surface is exercised at every tree level.

This should be read as an **alignment hypothesis**, not an established fix.

### Wider Opaque Summary / No-Bottleneck Summary

Widening the summary surface is the general non-slot version of “do not squeeze out the count signal.” If the current summary path is a bottleneck, a wider or less compressive summary route may preserve theorem-facing information without baking in a task-specific slot layout.

Active artifacts in this line of investigation:

- `outputs/unified_g_no_bottleneck_4096.log`
- `outputs/unified_g_wide_4096.log`
- `outputs/unified_g_8192_100ep.log`

What can be said safely:

- these logs represent active investigations into unified-`g` and reduced-bottleneck summary routes
- they support the existence of this line of work
- they do **not** yet establish a successful fix unless a finished result artifact clearly demonstrates it

## Recommended Next Experiments

The order below is intentional. Do not skip ahead to topology.

### 1. Exact-leaf merge-signal matrix at honest scales

Question:

- Is strict local-law merge training failing because the signal is weak, or because the representation / merger interface is fundamentally incapable?

Compare:

- `strict_c3`
- `teacher_parent_count`
- `teacher_parent_full_sketch`
- with and without root loss where relevant
- flat vs depth-balanced weighting where relevant

Interpretation:

- if `teacher_parent_full_sketch` succeeds and strict `C3` fails, the problem is optimization / signal, not representation feasibility
- if `teacher_parent_count` fails while `teacher_parent_full_sketch` succeeds, endpoints / join information are the missing signal
- if both teacher-guided lanes fail, the interface itself is the bottleneck

### 2. Unified-`g` / widened-summary ablation, if the strict-vs-diagnostic gap remains

Question:

- Is the main failure now in the leaf summary path rather than the merge law?

Compare:

- current opaque-carrier theorem-facing lane
- unified-`g` variant
- wider-summary / no-bottleneck leaf path variant

Interpretation:

- if unified-`g` or widened-summary helps while keeping the latent opaque, the bottleneck is leaf summary alignment / compression rather than theorem mismatch
- if neither helps, the problem is deeper representation / optimization, not just merge supervision

### 3. Only then return to learned-encoder + merger full-system studies

Question:

- Once the exact-leaf and leaf-summary issues are better understood, does the learned encoder close the remaining gap?

This stage should only happen after the previous two questions are cleaner. Otherwise the full-system result is too confounded to interpret.

### 4. Topology remains blocked

Do not use topology reruns for claims until the one-leaf / exact-leaf theorem-facing path is healthy. If the theorem-facing anchor is still unstable, topology results are not informative about the actual theorem question.

## Reproduction Appendix

### Exact-leaf merge-feasibility study

Current harness:

```bash
source venv/bin/activate
python scripts/test_markov_exact_progression.py \
  --use-cuda \
  --output-root outputs/markov_merge_signal_feasibility_$(date +%Y%m%d_%H%M%S)
```

Current small artifact:

```bash
sed -n '1,220p' \
  outputs/markov_merge_signal_feasibility_small_20260407_171619/merge_signal_feasibility_summary.md
```

### Active unified-g / wider-summary investigation

Inspect these logs without assuming they prove success:

```bash
sed -n '1,120p' outputs/unified_g_no_bottleneck_4096.log
sed -n '1,120p' outputs/unified_g_wide_4096.log
sed -n '1,120p' outputs/unified_g_8192_100ep.log
```

### Supporting design note

Agent-local note for the unified-`g` hypothesis:

```bash
sed -n '1,220p' /home/mlinegar/.claude/plans/moonlit-squishing-puddle.md
```

## What Another LLM Should Be Able To Answer From This Document

- What is the current theorem-facing claim?
- What is the main current bottleneck?
- Which experiments are valid evidence versus diagnostic-only evidence?
- What should be run next, and in what order?

If this document stops making those answers obvious, it should be updated before new large runs are launched.
