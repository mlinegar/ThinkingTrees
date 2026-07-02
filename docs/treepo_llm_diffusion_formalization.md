# TreePO as Tree-Indexed LLM Diffusion

## Scope

This note gives a Lean-aligned formalization of the "ThinkingTrees as an LLM
diffusion system" viewpoint.

The key claim is precise:

- the repository already supports a **tree-indexed reverse denoising process**;
- the tree structure turns global generation into **local transitions** at
  leaves, merges, and re-summarization steps; and
- the existing Lean development already proves that if those local transitions
  are exact, or approximately audited, then the resulting **intermediate
  reverse-diffusion states** are globally correct in the oracle sense.

This is not the claim that the repo already formalizes a continuous Gaussian
diffusion model in the image-model sense. The Lean-backed core is a **discrete,
tree-structured denoising process** over strings or theorem-domain latent
states. That is the mathematically honest bridge to "LLM diffusion" in the
current codebase.

Relevant repo surfaces:

- Python runtime:
  [src/tree/builder.py](/home/mlinegar/ThinkingTrees/src/tree/builder.py),
  [src/tree/auditor.py](/home/mlinegar/ThinkingTrees/src/tree/auditor.py),
  [src/tree/verification.py](/home/mlinegar/ThinkingTrees/src/tree/verification.py),
  [src/core/data_models.py](/home/mlinegar/ThinkingTrees/src/core/data_models.py),
  [src/core/ops_checks.py](/home/mlinegar/ThinkingTrees/src/core/ops_checks.py),
  [src/harness.py](/home/mlinegar/ThinkingTrees/src/harness.py)
- Lean theorem surface:
  [lean3/FormalProofs/OPT/LocalLaws.lean](/home/mlinegar/ThinkingTrees/lean3/FormalProofs/OPT/LocalLaws.lean),
  [lean3/FormalProofs/OPT/ApproximateLocalLaws.lean](/home/mlinegar/ThinkingTrees/lean3/FormalProofs/OPT/ApproximateLocalLaws.lean),
  [lean3/FormalProofs/OPT/AdaptiveChunkingBridge.lean](/home/mlinegar/ThinkingTrees/lean3/FormalProofs/OPT/AdaptiveChunkingBridge.lean),
  [lean3/FormalProofs/OPT/ExactUtilityTransport.lean](/home/mlinegar/ThinkingTrees/lean3/FormalProofs/OPT/ExactUtilityTransport.lean),
  [lean3/FormalProofs/DSL/RuntimeCertificates.lean](/home/mlinegar/ThinkingTrees/lean3/FormalProofs/DSL/RuntimeCertificates.lean),
  [lean3/FormalProofs/OPT/MainTheorems.lean](/home/mlinegar/ThinkingTrees/lean3/FormalProofs/OPT/MainTheorems.lean)

## 1. Core Formal Objects

We work with the same core objects already used throughout TreePO.

- `Strings`: document/span space with monoid operation `*`
  (concatenation / merge-input composition)
- `Y`: oracle space with pseudo-metric `D`
- `f* : Strings -> Y`: target oracle or theorem-facing task feature
- `T : BinTree Strings`: a binary tree whose leaves are document spans and whose
  reconstruction is `S T`
- `g : Strings -> PMF Strings`: one-step stochastic denoiser / summarizer

The Python correspondence is direct:

- `Node.raw_text_span`, `Node.ops_span`, and `Node.summary` in
  [src/core/data_models.py](/home/mlinegar/ThinkingTrees/src/core/data_models.py)
  are the runtime carriers of span-level state.
- `TreeBuilder` in
  [src/tree/builder.py](/home/mlinegar/ThinkingTrees/src/tree/builder.py)
  instantiates the tree `T`.
- `format_merge_input` is the implementation-side realization of the monoid
  merge surface used by Lean's local laws.

## 2. Tree-Indexed Reverse Diffusion

### 2.1 Reverse process

For a fixed document `x` and a sound tree `T` with `S T = x`, define the
reverse-diffusion checkpoints by the already formalized TreePO reduction:

`Z_r(x, T) ~ ZR g x r T`, for `r >= 1`.

Interpretation:

- `r = 1` is one round of local denoising/merging through the tree.
- larger `r` repeatedly re-apply the denoiser to already-denoised states.
- the root sample after round `r` is an **intermediate reverse-diffusion
  checkpoint**.

This is the exact Lean object already used in the preservation theorems. The
"diffusion time" is the round index `r`.

### 2.2 Oracle distortion at a checkpoint

Define the checkpoint distortion

`Delta_r(x, T) := E_{z ~ ZR g x r T}[ D(f*(z), f*(x)) ]`.

In the Lean files this is the document-level distortion surface
`Δ_R_ZR g x r T f*`.

Meaning:

- `Delta_r(x, T) = 0` means the reverse-diffusion checkpoint at time `r`
  preserves the oracle exactly.
- `Delta_r(x, T) <= eps` means the checkpoint is `eps`-accurate in oracle space.

This is the right quantity if the goal is to say "some intermediate outputs of
the diffusion are actually correct."

## 3. Local Transition Laws = Diffusion-Step Correctness

The repo already formalizes three local laws in
[lean3/FormalProofs/OPT/LocalLaws.lean](/home/mlinegar/ThinkingTrees/lean3/FormalProofs/OPT/LocalLaws.lean).
Using the current theorem-facing naming:

- `L1` = leaf preservation
- `L2` = merge preservation
- `L3` = on-range idempotence

The paper aliases are:

- `C1 = L1`
- `C3 = L2`
- `C2 = L3`

These are exactly the local transition laws needed for a diffusion-style view.

### 3.1 Leaf law

`L1` says that denoising a raw leaf span preserves the oracle:

`E[D(f*(g(b)), f*(b))] = 0`.

Diffusion interpretation:

- the first reverse update from a noisy/local span to its cleaned state is
  oracle-correct.

### 3.2 Merge law

`L2` says that denoising the merged children preserves the oracle of the full
parent span.

Diffusion interpretation:

- the reverse transition that fuses two local denoised states into a coarser
  denoised parent state is oracle-correct.

### 3.3 Idempotence law

`L3` says that once a state is already on the denoiser's range, re-denoising it
does not move it in oracle space.

Diffusion interpretation:

- after a checkpoint has landed on the reverse manifold, further denoising
  steps do not corrupt it.
- this is exactly the condition needed to talk about correctness of
  **intermediate** reverse-diffusion states rather than only the final one.

## 4. Exact Theorem: Intermediate Diffusion Checkpoints Are Correct

The exact local-to-global theorem is already exported from
[lean3/FormalProofs/OPT/MainTheorems.lean](/home/mlinegar/ThinkingTrees/lean3/FormalProofs/OPT/MainTheorems.lean)
as `multi_round_preservation` and `delta_r_zr_zero_of_local_laws`.

### Theorem A: Exact checkpoint correctness

Assume:

- `S T = x`
- `LocalLawsBundle g T f*`
- `r >= 1`
- the standard boundedness side conditions already required by
  `multi_round_preservation` / `delta_r_zr_zero_of_local_laws`

Then

`Delta_r(x, T) = 0`.

Interpretation:

- every reverse-diffusion checkpoint produced by the tree reduction is oracle
  exact;
- correctness is not only about the terminal root sample;
- the tree factorization is what makes this possible, because correctness is
  discharged locally at leaves, merges, and re-summarization steps.

This is the cleanest answer to the intuition that "the tree repository gives us
a way to make sure intermediate diffusion outputs are correct."

## 5. Approximate Theorem: Audited Intermediate Correctness

Exact laws are too strong for practical LLMs, so the right practical object is
the approximate local-law bundle already formalized in
[lean3/FormalProofs/OPT/ApproximateLocalLaws.lean](/home/mlinegar/ThinkingTrees/lean3/FormalProofs/OPT/ApproximateLocalLaws.lean):

`ApproxLocalLawsBundle g T f*`

with components:

- `epsLeaf`
- `epsMerge`
- `epsIdemp`

### Theorem B: Approximate checkpoint correctness

If `ApproxLocalLawsBundle g T f*` holds, `r >= 1`, and the standard side
conditions used by `Δ_R_ZR_le_of_approx_bundle` hold
(tree soundness, bounded distortion, and the monotonicity condition on
idempotence mass), then the current theorem surface yields the bound

`Delta_r(x, T) <= epsLeaf + epsMerge + (r - 1) * epsIdemp`.

This is the exact formal meaning of:

- leaf errors accumulate through the initial local denoising steps;
- merge errors accumulate through tree fusion;
- repeated reverse-diffusion refinement pays an additional idempotence tax.

For adaptive tree policies `tau(x)`, the already formalized bridge in
[lean3/FormalProofs/OPT/AdaptiveChunkingBridge.lean](/home/mlinegar/ThinkingTrees/lean3/FormalProofs/OPT/AdaptiveChunkingBridge.lean)
gives:

- deterministic adaptive control via `Δ_R_ZR_le_of_adaptive_approx_bundle`
- stochastic adaptive control via
  `Exp_Δ_R_ZR_le_of_stochastic_adaptive_approx_local_laws`

So the current repo already supports:

- document-specific diffusion trees
- randomized diffusion-tree policies
- high-level checkpoint error budgets

## 6. Audit Certificates for Intermediate Diffusion States

The key implementation point is that the repo already has runtime artifacts that
transport into Lean without inventing a parallel proof language.

### 6.1 Nodewise empirical audit certificate

[lean3/FormalProofs/OPT/ApproximateLocalLaws.lean](/home/mlinegar/ThinkingTrees/lean3/FormalProofs/OPT/ApproximateLocalLaws.lean)
defines:

- `NodewiseEmpiricalAuditCertificate`
- `audited_upper_bounds_of_nodewise_empirical_certificate`
- `approx_bundle_of_nodewise_empirical_certificate`

This means a sampled audit over local diffusion transitions can be lifted into
the theorem-side approximate bundle that controls checkpoint correctness.

### 6.2 Runtime checkability

[lean3/FormalProofs/DSL/RuntimeCertificates.lean](/home/mlinegar/ThinkingTrees/lean3/FormalProofs/DSL/RuntimeCertificates.lean)
defines:

- `RuntimeNodewiseAuditArtifact`
- `RuntimeNodewiseAuditArtifact.check`
- `RuntimeNodewiseAuditArtifact.approx_bundle_eq_of_check`

So, if the stored runtime artifact checks, Lean certifies that the runtime audit
object is exactly the approximate local-law object used in the theorem chain.

That is the formal bridge from:

- empirical node-level LLM evaluations

to

- theorem-level correctness claims for intermediate diffusion checkpoints.

### 6.3 Objective-level gap certificates

If we also care about training or scoring gaps, the runtime surface in
[lean3/FormalProofs/DSL/RuntimeCertificates.lean](/home/mlinegar/ThinkingTrees/lean3/FormalProofs/DSL/RuntimeCertificates.lean)
provides:

- `RuntimeDSLArtifact.check`
- `RuntimeDSLArtifact.valid_from_events_of_check`
- `RuntimeDSLArtifact.valid_from_events_with_oracleMeasurement_of_check`

This upgrades "the checkpoint is locally accurate" into
"the downstream objective gap is bounded with high probability."

## 7. Stronger Variant: Diffusion in Theorem-Domain Latent State

The strongest formalization in the repo is not free-form text diffusion. It is
diffusion over an **exact mergeable latent state**.

Let:

- `Sketch` be a theorem-domain latent state space
- `encode : Strings -> Sketch`
- `merge : Sketch -> Sketch -> Sketch`
- `feature : Strings -> Sketch`

Assume:

- `encode x = feature x`
- `merge (feature x) (feature y) = feature (x * y)`

Then
[lean3/FormalProofs/OPT/ExactUtilityTransport.lean](/home/mlinegar/ThinkingTrees/lean3/FormalProofs/OPT/ExactUtilityTransport.lean)
proves:

- `mergeFold_eq_feature`
- `mergeableStateUtility_exact_on_tree`

### Theorem C: Exact latent-state checkpoint correctness

For any downstream readout or utility `u : Sketch -> beta`,

`u(mergeFold encode merge T) = u(feature(S T))`.

Interpretation:

- each internal tree state can be treated as a diffusion hidden state;
- those hidden states are not just heuristically good;
- they are **exact sufficient statistics** for any downstream utility that
  factors through the latent state.

This is the most rigorous way to get Lean-backed correctness of intermediate
diffusion states.

The Markov example in
[lean3/FormalProofs/OPT/MarkovPathDGP.lean](/home/mlinegar/ThinkingTrees/lean3/FormalProofs/OPT/MarkovPathDGP.lean)
shows exactly this pattern:

- the full mergeable Markov sketch is theorem-backed;
- a count-only under-supported feature is formally shown to be insufficient.

That distinction is exactly what we should want from a diffusion formalization:
not every hidden state representation is certifiable, but some are.

## 8. Formal Definition: Tree-Diffusion Certificate

The repo does not currently define this name, but the concept can be introduced
without changing the underlying theorem surface.

### Definition

A **Tree-Diffusion Certificate** for document `x`, tree `T`, checkpoint `r`, and
oracle `f*` is a tuple

`C(x, T, r) = (A_local, A_runtime, A_gap)`

where:

- `A_local` is either:
  exact `LocalLawsBundle g T f*`, or approximate
  `ApproxLocalLawsBundle g T f*`
- `A_runtime` is an optional checked runtime artifact witnessing that the stored
  empirical audit object equals the theorem-side bundle
- `A_gap` is an optional checked DSL artifact carrying a high-probability bound
  on downstream objective gap

with semantics:

- exact mode:
  `Delta_r(x, T) = 0` under the exact theorem's standard side conditions
- approximate mode:
  `Delta_r(x, T) <= epsLeaf + epsMerge + (r - 1) * epsIdemp` under the same
  side conditions used by `Δ_R_ZR_le_of_approx_bundle`
- objective mode:
  downstream loss/gap bounds follow from the already exported TreePO/DSL
  theorem chain on the corresponding good events

This is the repository-native notion of "certified intermediate diffusion
output."

## 9. Mapping Back to Runtime Code

The current Python code already matches this view:

- [src/tree/builder.py](/home/mlinegar/ThinkingTrees/src/tree/builder.py)
  constructs the reverse-diffusion tree and intermediate states.
- [src/tree/verification.py](/home/mlinegar/ThinkingTrees/src/tree/verification.py)
  checks local laws on node transitions.
- [src/tree/auditor.py](/home/mlinegar/ThinkingTrees/src/tree/auditor.py)
  turns sampled local checks into statistical guarantees.
- [src/harness.py](/home/mlinegar/ThinkingTrees/src/harness.py)
  packages them into end-to-end certificates.
- [src/core/ops_checks.py](/home/mlinegar/ThinkingTrees/src/core/ops_checks.py)
  already exposes the theorem-facing law vocabulary and approximate-law bundle.

So, operationally:

1. Build a tree.
2. Treat each node transition as a reverse-diffusion step.
3. Audit local transitions.
4. Lift the local audit to a theorem-side approximate bundle.
5. Conclude correctness of intermediate checkpoints and, when needed,
   downstream objective control.

## 10. What Is Fully Lean-Backed vs. What Is Research Extension

### Already Lean-backed

- tree-indexed reverse process `ZR`
- exact local laws `L1/L2/L3`
- exact checkpoint correctness `Delta_r = 0`
- approximate checkpoint bounds from audited local laws
- runtime artifact checking for nodewise audits and DSL gap artifacts
- exact latent-state correctness for mergeable theorem-domain states

### Natural extension, not yet formalized as such

- time-inhomogeneous denoisers `g_1, ..., g_R`
- continuous latent diffusion with Gaussian or score-based noise
- classifier-free guidance analogues for tree-conditioned denoising
- learned text decoders from exact theorem-domain hidden states

These are reasonable next steps, but they should be stated as extensions, not
as current theorem coverage.

## 11. Bottom Line

The right formal statement is:

ThinkingTrees already supports a rigorous view of LLM diffusion as a
tree-indexed reverse denoising process. The tree makes the reverse process
locally checkable. Lean already proves that exact or approximately audited local
transition correctness implies global correctness of every intermediate
reverse-diffusion checkpoint in oracle space. When we can lift the state into an
exact mergeable theorem-domain latent representation, those intermediate states
become fully certifiable sufficient statistics rather than merely plausible text
summaries.
