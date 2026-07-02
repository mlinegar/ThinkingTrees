# C-TreePO Appendix Proof Audit

Date: 2026-04-23; updated 2026-04-24

Authoritative manuscript: `paper/ctreepo/main_new.tex`.

This audit ignores older duplicate labels in `paper/ctreepo/main.tex`,
`paper/ctreepo/main_v2.tex`, and `paper/ctreepo/sections/` unless they are
included by `main_new.tex`.

## Scope And Commands

Included manuscript files:

- `paper/ctreepo/sections/v2/01_introduction.tex` through
  `paper/ctreepo/sections/v2/14_conclusion.tex`
- `paper/ctreepo/appendix/A_notation.tex` through
  `paper/ctreepo/appendix/I_operator_overlap.tex`

Commands used during the audit:

```bash
rg -n "input\\{" paper/ctreepo/main_new.tex
rg -n "begin\\{definition\\}|begin\\{assumption\\}|begin\\{theorem\\}|begin\\{proposition\\}|begin\\{lemma\\}|begin\\{corollary\\}|begin\\{example\\}" paper/ctreepo/sections/v2 paper/ctreepo/appendix
rg -n "begin\\{proof\\}|\\\\qed" paper/ctreepo/appendix paper/ctreepo/sections/v2
rg -n "<paper labels>" paper/ctreepo/sections/v2 paper/ctreepo/appendix
rg -n "<Lean anchors>" lean3/FormalProofs -g "*.lean"
rg -n "thm:dpo-equiv|thm:grpo-pl|thm:grpo-rl|thm:l3-necessary|thm:expected-tree-opt|paper/sections/03_main_theorems" lean3/docs paper/ctreepo
```

Final build commands and results are recorded in "Verification".

Status legend:

- `Aligned`: paper statement, proof, and Lean anchor match directly.
- `Aligned with caveat`: the result matches, but packaging or scope differs.
- `Patched`: a manuscript or proof-map edit was made in this pass.
- `Paper-only`: intentionally not backed by a dedicated Lean theorem.
- `Narrative/external`: expository or external-theorem interface, not a local Lean proof.
- `Residual`: a non-blocking gap or future Lean alias/export would improve the crosswalk.

## Inventory

| Item | Text location | Proof location | Lean anchor | Status | Notes / action |
|---|---|---|---|---|---|
| `def:oracle` | `sections/v2/03_framework.tex:39` | statement only | `CoreDefinitions.lean`, `LocalLaws.lean` concepts | Aligned | Minimal reader-facing definition of task oracle. |
| `def:summarizer` | `sections/v2/03_framework.tex:47` | statement only | `Summarizer` in `CoreDefinitions.lean`; `deterministic_summarizer_embedding` | Patched | Paper now states the Lean hierarchy directly: summarizers are PMF-valued; deterministic neural operators are point-mass specializations. |
| `def:readout` | `sections/v2/03_framework.tex:47` | statement only | readout/judge objects in DSL and OPT files | Aligned with caveat | Not a theorem; used to separate oracle from empirical judge. |
| `ass:context` | `sections/v2/03_framework.tex:150` | used in preservation and mergeable proofs | Lean L2/internal-node form absorbs part of this propagation | Aligned with caveat | Appendix crosswalk already explains C3/L2 packaging. |
| `prop:mergeable-reduction` | `sections/v2/02_mergeable_sketches.tex:143` | `appendix/B_proofs.tex:67` | `prop3_mergeable_classical`, `ops_reduction_to_classical_mergeable` | Patched | Added proof note that A3 is the Lean/classical congruence interface while the paper derivation writes the congruence through context compatibility. |
| `prop:neural-operator-bridge` | `sections/v2/08_theory.tex:21` | Appendix B neural-operator notes | `approxLocalLawsBundle_of_uniformApproxExactTheoremBacked`, `approxTheoremBacked_of_uniformApproxExactTheoremBacked`; stochastic/direct route exports in `MainTheorems.lean` | Patched | Public statement now says the bridge certifies deterministic realizers; randomized summarizers use the broader PMF local-law theorem stack unless a randomized operator theorem is supplied. |
| Neural-operator-to-gap bridge | `sections/v2/08_theory.tex:207`, `appendix/B_proofs.tex:318` | Appendix B neural-operator and gap proof notes | `neural_operator_transfer_local_law_budget`, `neural_operator_transfer_method_gap_budget`, `neural_operator_delta_r_transfer_moduli_bound`, `expectedObjectiveGap_via_neuralOperatorTransferModuli`, `expectedObjectiveGap_via_neuralOperatorFDTransferModuli`, method-specific DPO/GRPO neural-operator gap wrappers | Patched | Public text and Lean now share the transfer-moduli formula: upstream operator tolerance -> local-law budget -> method transport. |
| `prop:law-constrained-no` | `sections/v2/08_theory.tex:39` | Appendix I / proof map | `mergeableSketchSummaryClass_subset_exactLocalLawSubspace`, `mergeableSketch_overlap_subset_exactLocalLawNeuralOperators`; transformer architecture interfaces | Aligned | Transformer claim is architectural inclusion, not a guarantee about a trained checkpoint. |
| `thm:one-pass` | `sections/v2/08_theory.tex:131` | `appendix/B_proofs.tex:27` | `one_pass` | Patched | Preservation proof now states explicitly that paper C3 decomposes what Lean L2 packages as internal-node zero distortion. |
| `cor:schedule` | `sections/v2/08_theory.tex:145` | follows from one-pass | `schedule_invariance` | Aligned | Minimal statement: compares schedules only when each satisfies local laws on its realized edges. |
| `cor:folds` | `sections/v2/08_theory.tex:149` | follows from one-pass + C2 for intermediate summaries | `fold_of_folds` | Aligned | C2 assumption is visible in the statement. |
| `thm:multi-round` | `sections/v2/08_theory.tex:155` | `appendix/B_proofs.tex:55` | `multi_round_proper`, `multi_round_bounded`, `multi_round_typeclass` | Aligned | Needs C2/L3 and boundedness side conditions in Lean. |
| `ass:CF` | `sections/v2/08_theory.tex:167` | used in preference proof | `ScoreTransport.lean`, preference interfaces | Aligned with caveat | Paper-facing factorization assumption; Lean uses oracle-measurability/indexing predicates for the loss/generator layer. |
| `thm:pref-equiv` | `sections/v2/08_theory.tex:178`; minimal `sections/minimal/04_markov_state.tex:27` | `appendix/B_proofs.tex:292`; minimal `appendix/minimal/C_proofs_formalization.tex:24` | `PaperPreferenceStack`, `paper_preference_stack_same_argmin`, `paper_preference_stack_summary_argmin_full_epsilon`; exact method anchors `preference_learning_equivalence`, `same_oracle_measurable_argmin_general_of_loss_eq`, `dpo_exact_metric`, `grpo_pl_exact_metric`, `grpo_rl_exact_metric`, `dpo_equivalence`, `grpo_equivalence`, `grpo_rl_equivalence` | Patched | Public wording now cites the generic residual stack: residual `0` gives same argmins; residual `epsilon` gives `2 * epsilon` full-objective optimality from exact summary minimizers. |
| `thm:unified-gap` | `sections/v2/08_theory.tex:241` | `appendix/B_proofs.tex:312` | `unified_preference_gap_bounded`, `coupling_delta_eq_delta_r_zr` | Patched | Crosswalk now points both to the generic bounded product-coupling theorem and the ZR coupling wrapper used downstream. |
| `thm:e2e` | `sections/v2/08_theory.tex:282`; minimal `sections/minimal/04_markov_state.tex:46` | `appendix/B_proofs.tex:346`; minimal `appendix/minimal/C_proofs_formalization.tex:24` | `treepo_objective_unbiased`, `treepo_distortion_unbiased`, `PaperLocalLawErrorBudget`, `PaperErrorCertificate`, `PaperErrorStack`, `paper_error_certificate_formula`, `paper_error_certificate_high_prob`, `paper_error_stack_high_prob`, method-specific end-to-end certificates and oracle-measurement variants | Patched | Paper proof is the high-level composition of HT unbiasedness, deterministic/high-probability gap transport, clipping envelopes, and method transport. Minimal wording now uses `C_meth * hatDelta_R + B_cal + B_est + B_clip`. |
| `ass:pref` | `sections/v2/09_manifesto_llm.tex:398` | assumption only | `DPOApplicationAssumptionBundle`, `dpo_application_bundle_supplies_premises`, `GRPOPLApplicationAssumptionBundle`, `grpo_pl_application_bundle_supplies_premises`, `GRPORLApplicationAssumptionBundle`, `grpo_rl_application_bundle_supplies_premises`, predicate exports | Patched | Manifesto/RILE prose now states this is an application assumption package instantiating theorem premises, not a theorem about manifesto data; noisy/approximate preference residuals go through oracle-measurement terms. |
| `prop:m_lt_k` | `appendix/B_proofs.tex:7` | `appendix/B_proofs.tex:11` | none | Paper-only / Patched | Crosswalk now marks it as an appendix-only finite mechanism diagnostic, not a Lean-backed theorem. |
| `lem:sigmoid-lip` | `appendix/B_proofs.tex:239` | proof sentence in statement | `sigmoid_lipschitz` | Aligned with caveat | Constant 1 is intentionally loose; derivative gives 1/4. |
| `lem:neglogsig-lip` | `appendix/B_proofs.tex:243` | proof sentence in statement | `neg_log_sigmoid_lipschitz` | Aligned | Minimal calculus fact. |
| `lem:dpo-lip` | `appendix/B_proofs.tex:247` | `appendix/B_proofs.tex:258` | `dpo_loss_pointwise_lipschitz` | Aligned | Paper constant `2|beta| L_pol` matches Lean. |
| `lem:dpo-oracle-meas` | `appendix/B_proofs.tex:266` | `appendix/B_proofs.tex:270` | `dpo_loss_oracle_measurable` | Aligned | Direct factor-through-oracle argument. |
| `lem:zero-dist-support` | `appendix/B_proofs.tex:274` | `appendix/B_proofs.tex:278` | nonnegative expectation/support lemmas used by preference proofs | Aligned with caveat | Paper uses finite/discrete support notation; Lean carries PMF support and summability details. |
| `thm:fixed-partition` | `appendix/C_fixed_partition.tex:3` | `appendix/C_fixed_partition.tex:17` | `fixed_partition_extension_instantiation` | Aligned | Quantifier-order paragraph correctly prevents the "any partition automatically works" misread. |
| `ex:c2-independent` | `appendix/D_counterexample.tex:5` | verification paragraphs in Appendix D | `ex_c2_independent_formalized`, `thm10_1_L3_not_derivable` | Patched | Lean now has a public theorem matching the paper shape: C1 on fresh inputs, fresh-input C3, POS in range, and C2 failure under explicit first-token concat assumptions. |
| `thm:projection-iff` | `appendix/I_operator_overlap.tex:221` | `appendix/I_operator_overlap.tex:240` | `localLawWeightsAreProjection_iff_approximationErrorStructuredByLocalLaws`, `localLawWeightsAreProjectionOn_iff_approximationErrorStructuredByLocalLawsOn` | Aligned | Minimal set-extensionality theorem. |
| Kovachki finite-dimensionalization notes | `appendix/B_proofs.tex:107` | explanatory derivation | `KovachkiFiniteDimensionalization.lean`, neural-operator bridge exports | Narrative/external | Lean records the interface layer and route into local-law budgets; the external approximation theorem is not reproved here. |
| Equation 6 / transformer operator notes | `appendix/E_proof_artifacts.tex`, Appendix I | proof-map rows | `Equation6NeuralOperator*`, `transformerEncoder_mem_equation6Class` | Narrative/external | Crosswalk is clear that this is architecture/interface inclusion. |
| Audit robustness note | `appendix/B_proofs.tex:459`, `sections/v2/10_estimation.tex:144` | variance-bound note | `ht_unbiased_of_logged_marginals`, `ht_uniform_mean_unbiased_of_logged_marginals`, `ht_uniform_mean_covariance_controlled_independent_bernoulli`, `ht_uniform_mean_variance_bound_constrained`, `ht_uniform_mean_variance_bound_independent_bernoulli`, TreePO distortion wrappers | Patched | Fully formalized: arbitrary logged marginal propensities give HT unbiasedness; covariance-controlled and independent Bernoulli designs give the stated variance bound. |

## Running Notes

### Structural Tier

`def:oracle`, `def:summarizer`, and `def:readout` are minimal and readable.
They do not overclaim Lean theorem content. `def:summarizer` now states the
formal hierarchy directly: the core Lean interface is PMF-valued and
stochastic, while deterministic operators enter through
`deterministic_summarizer_embedding`.

`ass:context` is necessary in the paper proof because equivalence must be
transported through concatenation. Lean's L2 packages the realized internal
merge conclusion more directly. The proof-map note on C3/L2 packaging is
therefore important and should remain.

`prop:mergeable-reduction` was mostly aligned, but the proof did not explain
where A3 visibly enters relative to the Lean theorem. I patched the proof to
state that A3 supplies the classical congruence/merge interface used in Lean,
while the displayed paper derivation writes equivalent congruence steps through
context compatibility.

`thm:one-pass`, `cor:schedule`, `cor:folds`, and `thm:multi-round` are aligned.
The one-pass proof was patched to avoid suggesting that only one link of C3 is
the Lean statement. The paper now says C3 gives the child-summary equivalence
and Lean L2 packages the composed internal-node zero-distortion conclusion.

`thm:fixed-partition` is correctly conditional: a deterministic partition rule
does not automatically satisfy local laws; preservation follows once the laws
hold on the realized hierarchy.

`ex:c2-independent` now has a public-shape Lean theorem,
`ex_c2_independent_formalized`. The theorem keeps the extra first-token concat
assumptions explicit: concat preserves the left oracle value, fresh strings
concatenate to fresh strings, token summaries concatenate to fresh strings, and
a fresh positive witness puts POS in the range. Under those assumptions Lean
proves C1 on fresh inputs, the fresh-input C3 chain, POS-in-range, and C2
failure.

### Neural / Operator Tier

`prop:neural-operator-bridge` is aligned with Lean as an interface theorem:
uniform approximation on compact realized call sets plus explicit transfer
moduli yields approximate local-law budgets. The public text now matches the
Lean split: the bridge certifies deterministic realizers, embeds them through
the point-mass summarizer specialization, and leaves randomized summarizers to
the broader PMF-valued local-law theorem stack unless a separate randomized
operator-approximation theorem is supplied.

The follow-up neural-operator pass made the quantitative route more explicit.
The main text and Appendix B now state the composition
``external equation-(6) approximation -> transfer moduli -> local-law budget
-> method transport'', with the combined design-target bound written before
the sampled-audit certificate. This matches the Lean bridge layer in
`NeuralOperatorPreferenceBridge.lean`. A second Lean pass added paper-facing
budget names for the first quantitative step:
`ApproxNeuralOperatorPreferenceBridge.localLawBudget`,
`ApproxNeuralOperatorPreferenceBridge.delta_R_ZR_le_localLawBudget`, and the
finite-dimensionalization analogues, plus `MainTheorems.lean` exports
`neural_operator_realization_local_law_budget` and
`neural_operator_delta_r_bound`.

The direct transfer-moduli pass made the exact paper formula first-class in
Lean. `NeuralOperatorTransferModuli.localLawBudget` names
`ω_leaf(ε)+ω_merge(ε)+(R-1)ω_idemp(ε)`, and
`NeuralOperatorTransferModuli.methodGapBudget` names the same quantity
multiplied by `C_meth`. The uniform and finite-dimensionalization bridges now
have `matchesTransferModuli`, `localLawBudget_eq_transferModuliBudget`,
`delta_R_ZR_le_transferModuliBudget`, and generic
`expectedObjectiveGap_via_neuralOperatorTransferModuli` wrappers.

The Kovachki finite-dimensionalization discussion is narrative/external. The
Lean repo carries the reusable interface layer and bridge into local-law
budgets, while the external Kovachki approximation theorems remain cited
inputs.

`prop:law-constrained-no` and `thm:projection-iff` are aligned. The former is
a subspace inclusion statement; the latter is a set-extensional equivalence.
Neither needs a stronger proof statement.

### Optimization Tier

`ass:CF` and `ass:pref` are understandable assumptions. Lean formalizes the
downstream obligations through oracle-measurable losses/policies and
oracle-indexed generators. The manuscript now says the manifesto/RILE workflow
is an application assumption package: DPO supplies oracle-measurable policies
and an oracle-indexed pair generator; GRPO-PL/RL supply the analogous policy,
reward, ranker, and group-generator predicates.

The DPO lemmas in Appendix B align with Lean: sigmoid and negative-log-sigmoid
Lipschitz facts, the pointwise DPO Lipschitz constant, and oracle measurability
all have direct anchors. The sigmoid constant is loose but harmless.

`thm:pref-equiv` had stale crosswalk labels (`thm:dpo-equiv`,
`thm:grpo-pl`, `thm:grpo-rl`) in repository docs. The Lean side now includes
`same_oracle_measurable_argmin_general_of_loss_eq`, `grpo_pl_exact_metric`, and
`grpo_rl_exact_metric`, so the public argmin phrasing has named anchors for
DPO, GRPO-PL, and GRPO-RL. The latest wrapper pass adds
`PaperPreferenceStack` as the public surface: exact theorem-backed objectives
are the residual-zero case, and noisy/approximate preference objects are handled
as uniform residuals with a `2 * residual` argmin-transfer theorem.

`thm:unified-gap` is aligned with a packaging caveat. The paper proof is the
coupled-summary argument. The core Lean theorem is stated using an explicit
bounded product-coupling double sum, and downstream ZR wrappers identify the
document-summary distortion used in tree applications. The proof map now lists
both layers.

### Certificate Tier

`thm:e2e` is aligned at the right abstraction level. The appendix proof shows
the HT unbiasedness calculation and then composes the neural-operator/local-law
budget or sampled audit estimate with calibration, estimation, clipping, and
method transport. Lean now packages the paper formula as
`PaperErrorCertificate`, proves the formula expansion, and gives both the
event-level high-probability wrapper and the bundled `PaperErrorStack`
high-probability surface from calibration/estimation/clipping events plus a
local-law transport envelope. Stress-grid reporting over
`(pi_min, w_max)` remains certificate design rather than a separate theorem.

The adversarial-sampling robustness note now has direct Lean anchors. The
generic IPW layer proves arbitrary-design HT unbiasedness from logged marginal
propensities and strict positivity. It also proves the constrained-design
variance bound from a covariance-control predicate, and proves independent
Bernoulli product sampling satisfies that predicate. TreePO wrappers instantiate
the same statements for uniform finite-population distortion audit units.

### Crosswalk Tier

`paper/ctreepo/appendix/E_proof_artifacts.tex`,
`lean3/docs/PAPER_TO_LEAN_MAP.md`, and `lean3/docs/CORE_PROOFS.md` now use the
active `main_new.tex` labels for the public claims. The stale method-specific
paper labels and stale C2-independence label were removed from the docs.

The proof map now explicitly marks `prop:m_lt_k` as paper-only and clarifies
the Lean scope of `ex:c2-independent`.

The proof map now also lists the stochastic/direct theorem-backed routes, the
deterministic neural-operator specialization, application-facing DPO/GRPO
assumption packages, and the audit-robustness HT/variance anchors.

The latest error/preference pass adds the paper-facing error certificate object,
clipped-IPW/Hajek envelope exports, explicit DPO/GRPO application bundles, and
oracle-measurement method-certificate anchors for noisy or approximate
preferences.

The minimal-manuscript wrapper pass adds `PaperErrorStack` and
`PaperPreferenceStack` as the public API names for finite-sample certificates
and exact/residual-bounded preference alignment. `main_minimal.tex` now uses the
same `C_meth * hatDelta_R + B_cal + B_est + B_clip` certificate language as the
Lean stack.

## Patch Log

- Patched `paper/ctreepo/appendix/B_proofs.tex`:
  - preservation proof now aligns paper C3 with Lean L2 packaging;
  - mergeable-reduction proof now explains A3/congruence packaging.
- Patched `paper/ctreepo/appendix/G_mechanism_checks.tex`:
  - corrected the single-leaf regime from "C2 (merge consistency)" to C3/L2
    vacuity and separated C2/idempotence.
- Patched `paper/ctreepo/sections/07_empirical.tex`:
  - made the same C2/C3 wording correction in the older non-authoritative
    duplicate section so repository-wide text searches do not rediscover it.
- Patched `paper/ctreepo/sections/v2/10_estimation.tex`:
  - simplified the adversarial-sampling appendix reference;
  - added the explicit four-piece certificate phrasing
    `C_meth * Delta_R + B_cal + B_est + B_clip`;
  - noted deterministic clipped-vs-unclipped Hajek envelopes for `B_clip`.
- Patched `paper/ctreepo/appendix/E_proof_artifacts.tex`:
  - added paper-only status for `prop:m_lt_k`;
  - added ZR coupling wrapper for `thm:unified-gap`;
  - updated `ex:c2-independent` to the public-shape Lean theorem;
  - added DPO, GRPO-PL, and GRPO-RL same-argmin anchors.
- Patched `lean3/FormalProofs/OPT/CounterexampleExistence.lean`:
  - added `ex_c2_independent_formalized`, which proves the paper-shaped
    C1+C3-not-C2 counterexample under explicit first-token concat assumptions.
- Patched `lean3/FormalProofs/OPT/PreferenceLearning.lean`:
  - added a generic same-argmin lemma from loss equality;
  - added `grpo_pl_exact_metric` and `grpo_rl_exact_metric`.
- Patched `lean3/FormalProofs/OPT/MainTheorems.lean`:
  - added doc-facing exports for the generic same-argmin lemma and the GRPO
    same-argmin theorems.
- Patched `lean3/FormalProofs/OPT/ClassicalSketchLocalLaws.lean` and
  `lean3/FormalProofs/Assumptions.lean`:
  - updated public comments from the older L3-only counterexample anchor to the
    paper-shaped C2 independence theorem.
- Patched `lean3/docs/PAPER_TO_LEAN_MAP.md`:
  - replaced stale labels with active `main_new.tex` labels;
  - added active rows for mergeable reduction, neural/operator results,
    fixed partition, end-to-end certificate, and projection iff.
- Patched `lean3/docs/CORE_PROOFS.md`:
  - updated the active paper path and method-equivalence labels.
- Follow-up neural-operator pass:
  - patched `paper/ctreepo/sections/v2/05_fno_primer.tex`,
    `paper/ctreepo/sections/v2/08_theory.tex`,
    `paper/ctreepo/sections/v2/09_manifesto_llm.tex`,
    `paper/ctreepo/sections/v2/11_verification.tex`,
    `paper/ctreepo/sections/v2/12_related.tex`, and
    `paper/ctreepo/appendix/B_proofs.tex` so neural-operator approximation
    and transfer budgets appear before Lipschitz/method transport;
  - added neural-operator-to-preference gap bridge anchors to
    `paper/ctreepo/appendix/E_proof_artifacts.tex`,
    `lean3/docs/PAPER_TO_LEAN_MAP.md`, and `lean3/docs/CORE_PROOFS.md`.
- Patched `lean3/FormalProofs/OPT/NeuralOperatorPreferenceBridge.lean` and
  `lean3/FormalProofs/OPT/MainTheorems.lean`:
  - added paper-facing budget definitions and `Δ_R` budget theorems for the
    uniform and finite-dimensionalization neural-operator bridges.
- Direct transfer-moduli Lean pass:
  - added `NeuralOperatorTransferModuli`, transfer-moduli budget equalities,
    paper-form `Δ_R` bounds, and generic paper-form expected-objective gap
    theorems to `lean3/FormalProofs/OPT/NeuralOperatorPreferenceBridge.lean`;
  - exported those names from `lean3/FormalProofs/OPT/MainTheorems.lean`;
  - updated `paper/ctreepo/appendix/E_proof_artifacts.tex`,
    `lean3/docs/PAPER_TO_LEAN_MAP.md`, `lean3/docs/CORE_PROOFS.md`, and
    `lean3/FormalProofs/OPT/README.lean`.
- Stochasticity/application/audit-robustness pass:
  - patched `paper/ctreepo/sections/v2/03_framework.tex`,
    `paper/ctreepo/sections/v2/05_fno_primer.tex`,
    `paper/ctreepo/sections/v2/08_theory.tex`,
    `paper/ctreepo/sections/v2/09_manifesto_llm.tex`, and
    `paper/ctreepo/appendix/B_proofs.tex` to state the stochastic PMF
    hierarchy, deterministic neural-operator specialization, application
    assumption-package status, and formal audit-robustness conditions;
  - patched `lean3/FormalProofs/DSL/IPWTheory.lean` with logged-marginal HT
    unbiasedness, uniform finite-population HT unbiasedness, constrained-design
    variance bounds, and an independent-Bernoulli proof of the variance proxy;
  - patched `lean3/FormalProofs/DSL/TreeIPW.lean` with TreePO distortion wrappers
    for the same audit-robustness results;
  - patched `lean3/FormalProofs/OPT/MainTheorems.lean` with public stochastic,
    deterministic, application-package, and audit-robustness exports;
  - updated `paper/ctreepo/appendix/E_proof_artifacts.tex` and
    `lean3/docs/PAPER_TO_LEAN_MAP.md` with the new public anchors.
- Error/preference alignment pass:
  - patched `lean3/FormalProofs/DSL/TreePOEndToEnd.lean` with
    `PaperLocalLawErrorBudget`, `PaperErrorCertificate`, the displayed formula
    expansion, and a high-probability certificate wrapper;
  - patched `lean3/FormalProofs/OPT/MainTheorems.lean` with paper-facing error
    certificate exports, clipped-IPW/Hajek exports, DPO/GRPO application bundles,
    method end-to-end oracle-measurement aliases, and the fixed-ranker
    Plackett--Luce GRPO-PL sufficient-condition export;
  - patched `paper/ctreepo/sections/v2/09_manifesto_llm.tex` and
    `paper/ctreepo/sections/v2/11_verification.tex` to make application
    assumptions conditional and to route noisy/approximate preferences through
    oracle-measurement residuals;
  - updated `paper/ctreepo/appendix/E_proof_artifacts.tex` and
    `lean3/docs/PAPER_TO_LEAN_MAP.md` with the new public anchors.
- Minimal wrapper/API pass:
  - patched `lean3/FormalProofs/DSL/TreePOEndToEnd.lean` with
    `PaperErrorStack` and `PaperErrorStack.high_prob_total`;
  - patched `lean3/FormalProofs/OPT/MainTheorems.lean` with
    `PaperPreferenceStack`, exact same-argmin and `2 * residual`
    epsilon-optimality exports, plus `paper_error_stack_high_prob`;
  - patched `paper/ctreepo/sections/minimal/04_markov_state.tex`,
    `paper/ctreepo/sections/minimal/08_audit_label_budget.tex`,
    `paper/ctreepo/appendix/minimal/C_proofs_formalization.tex`,
    `paper/ctreepo/appendix/minimal/H_audit_details.tex`, and
    `paper/ctreepo/sections/minimal/BLUEPRINT.md` to preserve the public stack
    language;
  - updated `paper/ctreepo/appendix/E_proof_artifacts.tex` and
    `lean3/docs/PAPER_TO_LEAN_MAP.md` with the new stack anchors.
- Root-observed Manifesto clarification pass:
  - patched the minimal Manifesto, audit, and appendix wording to separate the
    valid root-observed corpus evaluation from the stronger node-level C1/C2/C3
    local-law certificate;
  - updated `paper/ctreepo/sections/minimal/BLUEPRINT.md` so future minimal edits
    preserve the same two-tier claim boundary.

## Residual Risks

- `prop:m_lt_k`: intentionally paper-only. Formalize only if this diagnostic
  becomes a central theorem rather than a mechanism note.
- Kovachki and Equation 6 neural-operator statements depend on cited external
  approximation theorems and local interface wrappers, not a full local reproving
  of those external results.
- No separate randomized neural-operator approximation theorem is supplied.
  Randomized summarizers are covered by the PMF-valued local-law theorem stack;
  the neural-operator approximation bridge remains deterministic unless such an
  external randomized-operator result is added.

## Verification

Reference/stale-label searches:

```bash
rg -n "thm:dpo-equiv|thm:grpo-pl|thm:grpo-rl|thm:l3-necessary|thm:expected-tree-opt|paper/sections/03_main_theorems|C2 \\(merge consistency\\)" lean3/docs paper/ctreepo docs/ctreepo_appendix_proof_audit.md
rg -n "Appendix~\\\\ref\\{app:proofs\\} \\(Appendix|C2 \\(merge consistency\\)" paper/ctreepo/sections/v2 paper/ctreepo/appendix paper/ctreepo/sections
```

Result: no active source hits. The first command only hits this audit file,
where those old labels are recorded as patched drift.

Manuscript build:

```bash
cd paper/ctreepo && latexmk -pdf -interaction=nonstopmode main_new.tex
rg -n "Undefined|undefined|Citation.*undefined|Reference.*undefined|Label\\(s\\) may have changed|Rerun" main_new.log
```

Result: `main_new.pdf` built successfully. The log search found no undefined
references/citations and no final rerun-required warnings. Existing layout
warnings remain, mostly overfull boxes and oversized floats in tables/figures.

Initial Lean build before the follow-up neural-operator wording pass:

```bash
cd lean3 && lake build FormalProofs
```

Result: build completed successfully (`7921` jobs). Existing linter warnings
remain in unrelated Lean files, mostly `simpa`/`simp` suggestions and sequence
focus suggestions.

Follow-up neural-operator pass verification:

```bash
cd paper/ctreepo && latexmk -pdf -interaction=nonstopmode main_new.tex
rg -n "Undefined|undefined|Citation.*undefined|Reference.*undefined|Label\\(s\\) may have changed|Rerun" main_new.log
rg -n "NeuralOperatorTransferModuli|delta_R_ZR_le_transferModuliBudget|expectedObjectiveGap_via_neuralOperatorTransferModuli|expectedObjectiveGap_via_neuralOperatorFDTransferModuli|expectedObjectiveGap_via_neuralOperatorUniformBridge|expectedObjectiveGap_via_neuralOperatorFDBridge|dpo_gap_via_neuralOperatorUniformBridge|dpo_gap_via_neuralOperatorFDBridge|grpo_pl_gap_via_neuralOperatorUniformBridge|grpo_pl_gap_via_neuralOperatorFDBridge|grpo_rl_gap_via_neuralOperatorUniformBridge|grpo_rl_gap_via_neuralOperatorFDBridge|transformerEncoder_mem_equation6Class" lean3/FormalProofs/OPT/NeuralOperatorPreferenceBridge.lean lean3/FormalProofs/OPT/MainTheorems.lean lean3/FormalProofs/ML/TransformerAsNeuralOperator.lean paper/ctreepo/appendix/E_proof_artifacts.tex lean3/docs/PAPER_TO_LEAN_MAP.md lean3/docs/CORE_PROOFS.md
cd lean3 && lake env lean FormalProofs/OPT/NeuralOperatorPreferenceBridge.lean
cd lean3 && lake env lean FormalProofs/OPT/NeuralOperatorPreferenceBridge.lean -o .lake/build/lib/lean/FormalProofs/OPT/NeuralOperatorPreferenceBridge.olean -i .lake/build/lib/lean/FormalProofs/OPT/NeuralOperatorPreferenceBridge.ilean -c .lake/build/ir/FormalProofs/OPT/NeuralOperatorPreferenceBridge.c
cd lean3 && lake env lean FormalProofs/OPT/MainTheorems.lean
cd lean3 && lake env lean FormalProofs/OPT/MainTheorems.lean -o .lake/build/lib/lean/FormalProofs/OPT/MainTheorems.olean -i .lake/build/lib/lean/FormalProofs/OPT/MainTheorems.ilean -c .lake/build/ir/FormalProofs/OPT/MainTheorems.c
cd lean3 && lake env lean /dev/stdin <<'EOF'
import FormalProofs.OPT.MainTheorems
#check MainTheorems.neural_operator_transfer_local_law_budget
#check MainTheorems.neural_operator_delta_r_transfer_moduli_bound
#check MainTheorems.expected_objective_gap_via_neural_operator_transfer_moduli
#check MainTheorems.expected_objective_gap_via_neural_operator_fd_transfer_moduli
EOF
cd lean3 && lake build FormalProofs.OPT.NeuralOperatorPreferenceBridge
```

Result: the manuscript build passed and the log search again found no undefined
references/citations or final rerun-required warnings. The anchor search found
all neural-operator preference/gap and transformer-inclusion names in Lean and
the public crosswalks. Direct Lean checks for
`FormalProofs/OPT/NeuralOperatorPreferenceBridge.lean` and
`FormalProofs/OPT/MainTheorems.lean` passed. The bridge file still emits
pre-existing unused-section-variable linter warnings on exact bridge theorems.
Because `MainTheorems.lean` imports compiled `.olean` files, the edited bridge
module and then `MainTheorems.lean` were also compiled directly into the local
build cache; after that, downstream imports saw the new paper-form exports. The
`#check` command confirmed the public transfer-moduli budget, paper-form
`Δ_R` bound, and uniform/FD generic objective-gap theorem exports.

Stochasticity/application/audit-robustness pass verification:

```bash
cd lean3 && lake env lean FormalProofs/DSL/IPWTheory.lean
cd lean3 && lake build FormalProofs.DSL.IPWTheory
cd lean3 && lake env lean FormalProofs/DSL/TreeIPW.lean
cd lean3 && lake build FormalProofs.DSL.TreeIPW
cd lean3 && lake env lean FormalProofs/OPT/MainTheorems.lean
cd lean3 && lake env lean FormalProofs/DSL/TreePOEndToEnd.lean
cd lean3 && lake env lean FormalProofs/OPT/NeuralOperatorTheoremBridge.lean
cd lean3 && lake build FormalProofs
cd paper/ctreepo && latexmk -pdf -interaction=nonstopmode main_new.tex
rg -n "Undefined|undefined|Citation.*undefined|Reference.*undefined|Label\\(s\\) may have changed|Rerun" paper/ctreepo/main_new.log
rg -n "thm:dpo-equiv|thm:grpo-pl|thm:grpo-rl|thm:l3-necessary|thm:expected-tree-opt|paper/sections/03_main_theorems|C2 \\(merge consistency\\)|no dedicated public theorem row|deterministic neural-operator bridge specializes|paper allows randomized" lean3/docs paper/ctreepo docs/ctreepo_appendix_proof_audit.md
rg -n "stochastic_direct_exact_theorem_backed|stochastic_direct_approx_theorem_backed|deterministic_summarizer_embedding|dpo_application_oracle_measurable_policies|grpo_pl_application_oracle_measurable_bundle|ht_unbiased_of_logged_marginals|ht_uniform_mean_covariance_controlled_independent_bernoulli|tree_audit_uniform_distortion_variance_bound_independent_bernoulli" lean3/FormalProofs lean3/docs paper/ctreepo docs/ctreepo_appendix_proof_audit.md
```

Result: all targeted Lean checks passed. The full Lean target completed
successfully (`7923` jobs) with existing linter warnings. `main_new.pdf` built
successfully. The log search found no undefined references/citations and no
final rerun-required warnings; the only `Rerun` hit is the `rerunfilecheck`
package banner. The stale-label search has no active source hits beyond this
audit's historical patch notes. The anchor search finds the new stochastic,
deterministic, application-package, and audit-robustness exports in Lean and the
public crosswalks.

Error/preference alignment pass verification:

```bash
cd lean3 && lake env lean FormalProofs/DSL/TreeIPW.lean
cd lean3 && lake env lean FormalProofs/DSL/TreePOEndToEnd.lean
cd lean3 && lake env lean FormalProofs/OPT/MainTheorems.lean
cd lean3 && lake build FormalProofs.OPT.MainTheorems
cd lean3 && lake build FormalProofs
cd lean3 && lake env lean /dev/stdin <<'EOF'
import FormalProofs.OPT.MainTheorems
#check PaperErrorCertificate
#check MainTheorems.paper_error_certificate_high_prob
#check MainTheorems.clipped_hajek_gap_bound
#check MainTheorems.DPOApplicationAssumptionBundle
#check MainTheorems.GRPOPLApplicationAssumptionBundle
#check MainTheorems.GRPORLApplicationAssumptionBundle
#check MainTheorems.grpo_pl_expected_lipschitz_from_plackett_luce_fixed_ranker
#check MainTheorems.dpo_treepo_end_to_end_with_oracleMeasurement
#check MainTheorems.grpo_pl_treepo_end_to_end_with_oracleMeasurement
#check MainTheorems.grpo_rl_treepo_end_to_end_with_oracleMeasurement
EOF
cd paper/ctreepo && latexmk -pdf -interaction=nonstopmode main_new.tex
rg -n "Undefined|undefined|Citation.*undefined|Reference.*undefined|Label\\(s\\) may have changed|Rerun" paper/ctreepo/main_new.log
rg -n "PaperErrorCertificate|paper_error_certificate_high_prob|DPOApplicationAssumptionBundle|GRPOPLApplicationAssumptionBundle|GRPORLApplicationAssumptionBundle|clipped_hajek_gap_bound|grpo_pl_expected_lipschitz_from_plackett_luce_fixed_ranker|dpo_treepo_end_to_end_with_oracleMeasurement|grpo_rl_treepo_end_to_end_with_oracleMeasurement" lean3/FormalProofs lean3/docs paper/ctreepo docs/ctreepo_appendix_proof_audit.md
```

Result: targeted Lean checks passed, `FormalProofs.OPT.MainTheorems` built
successfully (`7855` jobs), and the full `FormalProofs` target built
successfully (`7923` jobs). The `#check` command confirmed the new public error
certificate, clipping, DPO/GRPO application-bundle, GRPO-PL sufficient-condition,
and oracle-measurement certificate anchors. `main_new.pdf` rebuilt successfully.
The log search found no undefined references/citations and no final
rerun-required warnings; the only `Rerun` hit remains the `rerunfilecheck`
package banner. The stale-label search has no active source hits beyond this
audit's historical patch notes.

Minimal wrapper/API pass verification:

```bash
cd lean3 && lake env lean FormalProofs/DSL/TreePOEndToEnd.lean
cd lean3 && lake env lean FormalProofs/OPT/MainTheorems.lean
cd lean3 && lake env lean /dev/stdin <<'EOF'
import FormalProofs.OPT.MainTheorems
#check MainTheorems.paper_error_stack_high_prob
#check MainTheorems.paper_preference_stack_same_argmin
#check MainTheorems.paper_preference_stack_summary_argmin_full_epsilon
EOF
cd lean3 && lake build FormalProofs.OPT.MainTheorems
cd lean3 && lake build FormalProofs
cd paper/ctreepo && latexmk -pdf -interaction=nonstopmode main_minimal.tex
cd paper/ctreepo && latexmk -pdf -interaction=nonstopmode -g main_new.tex
cd paper/ctreepo && rg -n "Undefined|undefined|Citation.*undefined|Reference.*undefined|Label\\(s\\) may have changed|Rerun" main_minimal.log main_new.log
rg -n -F "L\\hat{\\Delta}_R" paper/ctreepo/sections/minimal paper/ctreepo/appendix/minimal paper/ctreepo/main_minimal.tex lean3/docs/PAPER_TO_LEAN_MAP.md paper/ctreepo/appendix/E_proof_artifacts.tex docs/ctreepo_appendix_proof_audit.md
rg -n "PaperErrorStack|paper_error_stack_high_prob|PaperPreferenceStack|paper_preference_stack_same_argmin|paper_preference_stack_summary_argmin_full_epsilon" lean3/FormalProofs lean3/docs paper/ctreepo docs/ctreepo_appendix_proof_audit.md
```

Result: targeted Lean checks passed. The `#check` command confirmed
`paper_error_stack_high_prob`, `paper_preference_stack_same_argmin`, and
`paper_preference_stack_summary_argmin_full_epsilon`. `FormalProofs.OPT.MainTheorems`
built successfully (`7856` jobs), and the full `FormalProofs` target built
successfully (`7924` jobs), with existing linter warnings in unrelated files.
`main_minimal.pdf` and `main_new.pdf` built successfully. The log search found
no undefined references/citations and no final rerun-required warnings; the only
`Rerun` hits are the `rerunfilecheck` package banners. The stale
`L\hat{\Delta}_R` search returned no hits in the minimal paper or public maps,
and the anchor search finds the new stack names in Lean, docs, and paper
crosswalks.
