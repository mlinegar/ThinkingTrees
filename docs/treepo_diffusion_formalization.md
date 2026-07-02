# TreePO as Verified LLM Diffusion on Oracle Fibers

This note formalizes a diffusion-model interpretation of the tree repository.
The right object is not a Gaussian token diffusion or an SDE. The current repo
already formalizes something more directly useful for trustworthy intermediate
states:

- a stochastic multiscale reduction process over tree nodes;
- an iterative re-summary/refinement process on realized summaries; and
- exact or audited guarantees that the entire process stays on the same
  oracle/feature fiber as the source document.

That is the sense in which the repository can be viewed as a verified LLM
diffusion system.

## 1. Core Objects Already in the Repo

Fix:

- `Strings`: theorem-domain document/span space with monoid structure.
- `Y`: oracle space with pseudo-metric.
- `f* : Strings -> Y`: the target oracle.
- `g : Strings -> PMF Strings`: stochastic summarizer / refinement kernel.
- `T : BinTree Strings`: reduction tree for a document `x` with `S T = x`.

Existing theorem-facing definitions:

- local laws: `L1`, `L2`, `L3` in `lean3/FormalProofs/OPT/LocalLaws.lean`
- multi-round reduction: `ZR g x R T`
- exact theorem-backed interface: `ExactTheoremBacked`
- approximate theorem-backed interface: `ApproxTheoremBacked`
- audited approximate bundles: `ApproxLocalLawsBundle`,
  `AuditedApproxUpperBounds`, `NodewiseEmpiricalAuditCertificate`

Python mirrors:

- tree structure: `src/core/data_models.py`
- builder/runtime surface: `src/tree/builder.py`
- node verification: `src/tree/verification.py`
- audit sampling and guarantees: `src/tree/auditor.py`, `src/tree/ipw.py`
- theorem-backing capability surface: `src/core/ops_checks.py`,
  `src/tree/theorem_backing.py`
- public certificate API: `src/harness.py`

## 2. Tree Diffusion as a Spatio-Temporal Markov Process

The clean formalization has two time axes.

### 2.1 Structural time

For each realized node `u` in `T`, let `S(u)` be its theorem-domain span:

- for a leaf `u`, `S(u)` is the raw leaf text;
- for an internal node, `S(u)` is the concatenated theorem-domain span of its
  subtree.

Define the node state `Z_(u,0)` by:

- leaf case: `Z_(u,0) ~ g(S(u))`
- internal case: `Z_(u,0) ~ reduce g u`

Interpretation:

- leaf transitions are local denoising/compression moves;
- internal transitions are multiscale denoising/merging moves.

### 2.2 Refinement time

For any realized node state, define iterative refinement

- `Z_(u,r+1) ~ g(Z_(u,r))`

for `r >= 0`.

At the root `rho`, this is exactly the existing multi-round object:

- `Law(Z_(rho,R)) = ZR g x R T`

up to the repo's indexing convention for the initial tree reduction and later
resummary rounds.

## 3. Exact Intermediate Correctness

### Definition 3.1: Exact fiber-preserving tree diffusion

The tree diffusion induced by `(g, T, f*)` is exact if for every realized node
`u`, every refinement round `r`, and every support point `z` of `Z_(u,r)`,

- `D f* z (S(u)) = 0`

equivalently, `z` lies on the same oracle fiber as the clean theorem-domain
span for that node.

At the root this becomes:

- every intermediate root state is oracle-equivalent to the original document.

### Existing exact theorem route

The repo already packages the exact sufficient assumptions:

- `LocalLawsBundle` in `LocalLaws.lean`
- `ExactTheoremBacked.ofLocalLaws` in
  `lean3/FormalProofs/OPT/TheoremBackingAssumptions.lean`

The strongest support-level characterization already proved is:

- `exactTheoremBacked_nonempty_iff_supportExactTheoremBacked` in
  `lean3/FormalProofs/OPT/TheoremBackingStructure.lean`

This theorem says exact theorem-backedness is equivalent to support-level zero
distortion on:

- every realized leaf summary;
- every realized internal-node reduction; and
- every in-range resummary.

That is already a formal statement that intermediate outputs are correct.

### Existing root/global exact consequences

The repo already proves:

- `multi_round_preservation` in `lean3/FormalProofs/OPT/MainTheorems.lean`
- `delta_r_zr_zero_of_local_laws` in `lean3/FormalProofs/OPT/MainTheorems.lean`
- `zr_support_same_oracle_fiber` in `lean3/FormalProofs/OPT/MainTheorems.lean`

Interpretation:

- local correctness of leaf, merge, and resummary transitions implies that the
  entire root diffusion chain stays on the same oracle fiber as the source
  document.

## 4. Feature-Level Diffusion, Not Just Oracle-Level Diffusion

Often the intended "clean state" is not a scalar oracle value but a theorem
feature `phi : Strings -> Feature`.

Assume:

- `OracleRecoversFeature f* phi`

Then exact theorem-backedness upgrades from oracle-fiber preservation to
feature-fiber preservation.

Existing exported results:

- `leaf_support_same_feature_fiber`
- `merge_support_same_feature_fiber`
- `idempotent_support_same_feature_fiber`
- `zr_support_same_feature_fiber`

all exported in `lean3/FormalProofs/OPT/MainTheorems.lean`.

This is the key bridge for an LLM diffusion interpretation:

- the stochastic intermediate states may vary textually or latently,
- but Lean can already certify that they remain in the same theorem-bearing
  feature class.

## 5. Theory-Aligned Diffusion Heads

Let `readout_t : Strings -> R_t` be a time-indexed denoising head, score head,
or intermediate supervision head.

### Definition 5.1: Theory-aligned head

`readout_t` is theory-aligned if it factors through the theorem feature:

- `exists recover_t, readout_t = recover_t o phi`

This is exactly the repo's `ReadoutFactorsThroughFeature` notion in
`lean3/FormalProofs/OPT/ReadoutAlignment.lean`.

Existing exact transport results:

- `factored_readout_expected_loss_transport`
- `factored_readout_supervised_transport`
- `same_surface_supervised_transport`

Existing structural diagnostics:

- `same_surface_implies_factored_readout`
- `factored_readout_respects_theorem_feature`
- `separated_auxiliary_head_not_theory_aligned`

Interpretation:

- if an intermediate diffusion head only depends on the theorem-bearing feature,
  then exact transport through the tree diffusion is already formalized;
- if a head separates points that the theorem feature identifies, it is outside
  the theorem route.

## 6. A Diffusion-Loss Formalization

This is the clean objective-level formulation for a tree-backed LLM diffusion
model.

Let:

- `mu` be a document distribution;
- `alpha_r >= 0` be per-round weights;
- `h_(theta,r)` be a time-indexed denoising head;
- `ell_r` be a bounded supervised loss against the clean theorem target;
- `phi` be the theorem-bearing clean feature.

Define the tree diffusion loss

`L_diff(theta) = sum_r alpha_r * E_[x ~ mu, z_r ~ ZR g x r T] [ ell_r(h_(theta,r)(z_r), phi(x)) ]`

and the raw clean-state loss

`L_clean(theta) = sum_r alpha_r * E_[x ~ mu] [ ell_r(h_(theta,r)(x), phi(x)) ]`.

### Exact theorem-backed transport corollary

If:

- `ExactTheoremBacked g T f*`
- `OracleRecoversFeature f* phi`
- for each `r`, `h_(theta,r)` is same-surface or factors through `phi`
- each `ell_r` is feature-indexed / supervised-state compatible

then `L_diff(theta) = L_clean(theta)`.

This corollary is not currently exported under one name, but it is a direct
finite-sum consequence of the already-exported exact transport theorems in:

- `ReadoutAlignment.lean`
- `ExactUtilityTransport.lean`
- `MainTheorems.lean`

So the repo already supports a formal interpretation of diffusion training in
which every intermediate denoising state is a theorem-valid carrier of the same
clean feature.

## 7. Approximate and Audited Intermediate Correctness

Exact local laws are the strongest route, but the repo also formalizes the
approximate route needed in practice.

### Approximate theorem-backed interfaces

Existing objects:

- `ApproxLocalLawsBundle`
- `ApproxTheoremBacked.ofApproxLocalLaws`
- `AuditedApproxUpperBounds`
- `NodewiseEmpiricalAuditCertificate`

all in `lean3/FormalProofs/OPT/ApproximateLocalLaws.lean` and
`lean3/FormalProofs/OPT/TheoremBackingAssumptions.lean`.

These give objective-gap lifts already formalized for:

- DPO: `dpo_gap_via_audited_confidence_event`
- GRPO-PL: `grpo_pl_gap_via_audited_confidence_event`
- GRPO-RL: `grpo_rl_gap_via_audited_confidence_event`

### Stochastic/adaptive diffusion schedules

If the "diffusion schedule" is itself stochastic or adaptive, the repo already
has the right bridge:

- `StochasticAdaptiveLocalLaws`
- `StochasticAdaptiveApproxLocalLaws`
- `Exp_Δ_R_ZR_eq_zero_of_stochastic_adaptive_local_laws`
- `Exp_Δ_R_ZR_le_of_stochastic_adaptive_approx_local_laws`
- `Exp_dpo_gap_le_of_stochastic_adaptive_approx_local_laws`
- analogous GRPO theorems

in `lean3/FormalProofs/OPT/AdaptiveChunkingBridge.lean`.

Interpretation:

- even when the tree schedule is learned or randomized, the expected diffusion
  error and downstream objective gap are still formally bounded.

## 8. Audit Certificates for Intermediate Diffusion States

The repo does not stop at asymptotic existence theorems. It already has a
design-based audit layer for certifying the approximate route from sampled
intermediate states.

Lean/IPW layer:

- `TreePropensity`, `TreeSample`, `DSLBound`, `computeDSLBound`
- empirical Bernstein wrappers
- honesty and K-fold evaluation helpers
- `dsl_bound_valid`
- `dsl_bound_valid_with_oracleMeasurement`

in `lean3/FormalProofs/DSL/TreeIPW.lean`, `lean3/FormalProofs/DSL/IPWTheory.lean`,
and `lean3/FormalProofs/DSL/Honesty.lean`.

Python layer:

- `src/tree/auditor.py`
- `src/tree/ipw.py`
- `src/harness.py`

Interpretation in diffusion language:

- sample a subset of intermediate states `Z_(u,r)`;
- estimate leaf/merge/resummary failure rates with inverse-probability weights;
- lift those estimates to a certificate on global diffusion error or downstream
  training gap.

So the tree repository does not just give a diffusion process. It gives an
auditable diffusion process.

## 9. Literal Score-Based Diffusion Connection

If the intended diffusion model is literally score-based, the repo already has
part of the needed factorization story.

Current status summary:

- `docs/treepo_score_transport_iff_summary.md`

Relevant Lean results summarized there:

- `local_laws_imply_score_factorization_of_bridge`
- `not_score_factorization_implies_one_local_law_failed_of_bridge`

Interpretation:

- under the score-transport bridge hypotheses, valid local tree laws imply
  score factorization through the oracle sigma-algebra;
- failure of score factorization implies failure of at least one local law.

So even a score-network reading of diffusion fits the same theorem pattern:

- intermediate score heads must factor through the theorem-bearing object, or
  they are outside the verified route.

## 10. Latent-State Diffusion, Not Just Text Diffusion

The repo already supports theorem-backed latent operators, not only direct text
summaries.

Python surfaces:

- `src/tree/compositional_operator.py`
- `src/tree/theorem_backing.py`
- `src/core/ops_checks.py`

Lean surfaces:

- `SketchCodecExactAssumptions`
- `SketchCodecApproxAssumptions`
- `mergeableStateUtility_exact_on_tree`
- `markovStateUtility_exact_on_tree`
- `markovCountOnlyUtility_exact_on_tree`
- `markovCountEndpointsUtility_exact_on_tree`
- `topicSketchUtility_exact_on_tree`

Interpretation:

- a diffusion state can be a latent sketch/state rather than text;
- if there is an encode/merge/decode theorem route, intermediate latent states
  can be formally correct in exactly the same sense.

This is the right formal home for "LLM diffusion over tree latents with
verified intermediate states."

## 11. What Is Already Formalized vs. What Would Still Be New

Already formalized:

- local leaf/merge/resummary correctness
- exact and approximate theorem-backed reduction
- feature/readout factorization
- objective transport for factored heads
- adaptive/stochastic schedule bounds
- IPW audit certificates for sampled intermediate states

Not yet formalized in the current Lean stack:

- Gaussian forward kernels
- continuous-time diffusion or reverse-time SDEs
- score-matching Fisher objectives as first-class objects
- a one-line exported theorem named exactly "tree diffusion loss transport"

But the mathematically important part for trustworthy intermediate states is
already present:

- stochastic transitions;
- support-level exactness or audited approximate exactness; and
- downstream objective transport for theory-aligned heads.

## 12. Formal Thesis

The repository already supports the following formal thesis.

> A tree-backed LLM diffusion model is a stochastic process over node states
> and refinement rounds whose randomness is confined to oracle/feature fibers
> when local laws hold exactly, and whose deviation from those fibers is
> auditable and quantitatively bounded when only approximate local laws hold.
> Any intermediate denoising or score head that factors through the same
> theorem-bearing feature inherits exact or bounded-loss transport through the
> whole process.

That is the precise sense in which the tree repository gives a path to
trustworthy intermediate diffusion states, and the reason Lean is directly
useful here.
