# Lean ↔ Paper Alignment Audit (v12_external) — 2026-07-02

> **RESOLUTION STATUS (same day, second pass).** Most findings below are now
> RESOLVED; the paper draft advanced to `main_v13_triangle.tex` and the Lean
> tree gained `OPT/MergeTriangle.lean`. Resolved: F1 (merge triangle +
> `one_pass_of_local` + `ContextCompatible` + `L2_of_local` bridge; legacy
> theorems carry honest docstrings), F2 (`fixed_partition_population` tower
> step), F3/F4 (`fold_of_folds_of_local` with explicit `graft` structure,
> `schedule_invariance_of_local` with the hypothesis used), F5
> (`unified_preference_gap_bounded_coupled` + `documentSummaryCoupling` +
> `population_gap_zero_of_local`), F6-population (`population_loss_transport`;
> CF-vs-pointwise disclosed in the v13 crosswalk), F8 (support-form laws need
> no boundedness; disclosed), eq:error_budget (`paper_error_budget_union_bound`),
> D1 (Gibbons reworded to stipulated cost model), D2 (KLL/GK/MG added to the
> external list), D3 (Feldman Thm-3 schema parameterized in FormalProbability +
> both re-export sites), D4 ("axiom" → assumption-with-verified-sufficient-
> conditions in B_proofs), D5/D6 (crosswalk rows for `ass:CF`, `ass:context`,
> `lem:merge-triangle`, App-I calibrated objective; `\ref`s added for
> projection-iff/ass:pref/discussion-bound; `_statement` names annotated),
> M2 (`Axioms.lean` imported), M3 (`DoublyRobustMinimizationObjective`
> imported), M4 (quarantine exception documented), M5 (ghost `DPO.lean` refs
> fixed). Law labels aligned: paper-named law objects (`LeafSufficiency`,
> `RangeIdempotence`, `MergeSufficiency`, `MergeTriangle`, `ContextCompatible`)
> are canonical; L1/L2/L3 retained as legacy. `lake build` green (7,975 jobs);
> v13 compiles with zero undefined references; all 281 crosswalk names verify.
>
> **RESOLUTION STATUS (third pass, same day — "built and used" sweep).**
> Newly resolved:
> - **M6 (orphans) DONE.** Every on-disk module now either builds or is
>   explicitly archived. Re-imported + repaired: `OPT/AnalysisPartitionMismatch`
>   (broken `Decidable` elaboration was silently generating a `sorry`),
>   `OPT/AnalysisSummaryLocalLaws` (`abs_add` → `abs_add_le`, dead `ring`),
>   `OPT/WorkedExampleCMSTree` (claims written for an additive merge were
>   FALSE under the monoid `(· * ·)` on `Nat` — corrected 10 → 24, comments
>   fixed; plus a dead `decide`), `OPT/ContextualStateRecovery` (clean).
>   READMEs (CLT/ML/OPT/DSL) now imported ⇒ type-checked. Econometrics subtree
>   fully built via two new lake targets split along the namespace fault line:
>   `FormalProofsEconometrics` (local `Econometrics.Core` route) and
>   `FormalProofsEconometricsSemiparametric` (`AsymptoticOLS`+dependents and
>   `Overidentification`, which reach `FormalProbability.Econometrics.*` via
>   `DSL/AsymptoticTheory`). Archived per `OLD_` convention:
>   `DSL/OLD_MEstimationCore`, `DSL/OLD_SectionIPWTwoStage`;
>   `Deprecated/PointwiseLipschitz` stays folder-quarantined.
>   Full build green: 8,021 jobs (was 7,972 with 32 unbuilt orphans).
> - **F9-partial DONE.** `OPT/PaperSupportingLemmas.lean`: the five B-appendix
>   lemmas turned out to be mostly re-exports of results the repo already had
>   (`sigmoid_lipschitz`, `neg_log_sigmoid_lipschitz`,
>   `dpo_loss_pointwise_lipschitz`, `dpo_loss_oracle_measurable`,
>   `dist_zero_on_support_of_Exp_zero`) — the gap was crosswalk visibility, not
>   missing math. Genuinely new: `paper_m_lt_k_sketch_state_collision` /
>   `paper_m_lt_k_no_estimator` (prop:m_lt_k, general-n, no `decide`).
>   Crosswalk rows added; the "No dedicated Lean theorem" row is gone.
> - **F7 DONE.** `DSL/TreePOEndToEndGlue.lean` (897 lines):
>   `dpo_treepo_realized_estimator_certificate` — the paper's actual display,
>   gap ≤ C_meth·(realized μ̂_HT + t) w.p. ≥ 1 − 2·exp(−t²/(8N(D_max/π_min)²))
>   via the in-repo Hoeffding bound (sharper than the planned Chebyshev);
>   `dpoTreePOErrorStack` + `dpo_treepo_certificate_instantiates_error_stack`
>   genuinely instantiate the abstract `PaperErrorStack` with the concrete
>   Bernoulli-design HT estimator; GRPO-PL analogues included. Disclosed
>   caveats (in-file + new B_proofs "Formalization" paragraph): judge leg
>   degenerate (f = f*), unclipped estimator (B_clip = 0 case), stack legs in
>   estimator units; GRPO-RL analogue mechanical but not yet added.
>
> - **M1 DONE.** `OPT/MainTheorems.lean` (5,547 lines, 91 imports, 73/1,051
>   declarations paper-cited) split into `OPT/PaperTheorems.lean` (595 lines,
>   **14 imports**, the 73 crosswalk names + the `PaperPreferenceStack` helper
>   block; fully-qualified names unchanged via the same inner namespace) and
>   `OPT/ExtendedExports.lean` (5,139 lines, everything else verbatim);
>   MainTheorems is now a 21-line back-compat shim. Paper-relevant import
>   closure: **140 → 69 modules**. Line-multiset check confirms a lossless
>   partition. Crosswalk file columns updated (10 rows → PaperTheorems; the
>   calibrated-objective row corrected to `OPT/NeuralOperatorPreferenceBridge.lean`
>   where that name actually lives); REPRODUCIBILITY.md gained
>   `lake build FormalProofs.OPT.PaperTheorems` as the paper-surface build.
>   Note: the crosswalk's `paper_error_budget_union_bound` in this surface is
>   the primed alias `paper_error_budget_union_bound'`; the unprimed original
>   is in `OPT/MergeTriangle.lean`.
>
> **RESOLUTION STATUS (fourth pass, same day — M7 consolidation executed).**
> Six parallel agents merged the accreted clusters via a shim protocol (old
> module paths kept as one-line import shims until a central retarget pass),
> then a single integration pass repointed 24 importer files and deleted 39
> shims. Result: **218 → 184 modules on disk**, everything builds, paper
> closure 69 → **66**. New consolidated modules:
> - `OPT/TheoremBacking.lean` (MeasurementError + ApproxMeasurementError) atop
>   `OPT/TheoremBackingConsequences.lean` (which absorbed Assumptions +
>   Structure — kept at its crosswalk-cited path; 5 files → 2)
> - `OPT/UnifiedOracleRoute.lean` (+ TwoStageDecomposition,
>   TwoStageLabelScoreObjectives; 1,263 lines)
> - `OPT/OracleFibers.lean` (laws: ReadoutAlignment, FeatureFiberLaws,
>   LabelScoreObjectives) + `OPT/OracleFiberObjectives.lean` (objectives:
>   OracleFiberRelations, FiberPreservingObjective, FeatureClassObjectives,
>   SharedFeatureMultihead, ApproxFiberTransport)
> - `OPT/LocalLawObjectives.lean` (the ten-file objective ladder, 2,430 lines,
>   terminal section = the ACTIVE `drMinimizationValue` surface matching
>   Python ObjectiveSpec v1)
> - `OPT/AuditBounds.lean` (+ AuditCore, AuditSizes),
>   `OPT/ExactUtilityTransport.lean` (+ Instances),
>   `OPT/BayesianPersuasion.lean` (+ Economics, Direct; 1,256 lines),
>   `FormalProofs/CLT.lean` (absorbed all 9 re-export stubs)
> DAG-forced live survivors (each sits between or above its cluster's target;
> merging would create import cycles): `OPT/TwoStageOracleSurrogate.lean`,
> `OPT/Audit.lean`, `OPT/NodeIndexedLatentState.lean`,
> `OPT/ProductScoreFiber.lean`. One content deviation across the whole pass:
> a byte-identical duplicated `private theorem discountedTreeMetaLoss_congr_all`
> was elided once in `LocalLawObjectives` (module-scoped `private` collision).
> All chunks otherwise machine-verified byte-identical to their sources.
> Post-integration: build green 7,989 jobs across 3 targets; 313 crosswalk
> names verify; all 32 crosswalk-cited files exist; unbuilt = the 3 archives.
>
> **RESOLUTION STATUS (fifth pass, same day — mathlib alignment + obligation
> discharge).** Four parallel agents closed the remaining roadmap:
> - **Mathlib alignment** (plan + implementation log in
>   `docs/lean_mathlib_alignment_plan.md`): `expit` is now
>   `abbrev expit := Real.sigmoid` in both repos (mathlib v4.27 gained a full
>   sigmoid API); the proved lemma misnamed `hoeffding_iid_bounded_axiom` is
>   deleted (only a wrapper — the real `hoeffding_iid_bounded` already
>   existed); ~230 lines of conditional-Hoeffding/Azuma proofs deduped out of
>   TT (`SerflingAudit` now delegates to FormalProbability's
>   `SamplingConcentration`); PMF tsum expectations gained mathlib Bochner
>   bridges (`Exp/Eg/Egu_eq_integral` + Fintype corollaries, promoted into
>   `ExpectationTheory.lean`); `innerProduct` → `dotProduct` abbrev;
>   `IsPosDef' ↔ Matrix.PosDef` bridge. Verdicts kept: PMF stack = BRIDGE not
>   replace (34-file blast radius); custom CLT stays (mathlib v4.27 has no
>   CLT/Lévy continuity); `BoundedPseudoMetricSpace` full replace not worth it.
> - **GRPO-RL glue leg** (F7 completion): `grpo_rl_treepo_realized_estimator_certificate`,
>   `grpoRLTreePOErrorStack` + instantiation + high-prob corollary in
>   `DSL/TreePOEndToEndGlue.lean`; all three methods (DPO/GRPO-PL/GRPO-RL) now
>   have realized-μ̂ certificates; crosswalk + B_proofs updated.
> - **Misra–Gries Theorem 1 FULLY DISCHARGED** (was an undisclosed external
>   obligation at audit time): new undercount bound
>   `estimateCount_ge_frequency_sub` (n/(k+1) debt accounting), size bounds,
>   and — the real content — an executable merge with a checked Lemma-2.1
>   envelope invariant (`mergeClosed_mgValid`), packaged as
>   `executableMGAlgorithm` / `theorem1_executable` with ZERO caller-supplied
>   fields. SpaceSaving remains the disclosed-external bundle (min-counter
>   overcount + Corollary-1 transfer unformalized). Appendix E boundary text
>   updated in both directions; `Literature.externalObligationSurface` synced.
> - **Feldman Theorem 3 schema now non-vacuous**: concrete per-node private-seed
>   MUD model (`PrivateRandomGeneralMUDFamily`, seeding tree, uniform
>   seed-counting success ≥ 2/3) + reuse of the existing public-random
>   streaming success layer; both predicate classes proved inhabited and the
>   deterministic classes proved to embed; still an external obligation, but
>   no longer trivially satisfiable.
> End state: FP build green (8,212 jobs), TT build green (7,989 jobs, all
> three targets), `main_v13_triangle.pdf` compiles with zero undefined
> references, 334 crosswalk names verify (2 dotted-name scripted false
> positives only).
>
> **Remaining roadmap (research-grade, unscheduled):** SpaceSaving
> frequency-error/merge discharge + Corollary-1 MG↔SS transfer; GK/KLL
> quantitative bounds (disclosed obligations); proving the Feldman Thm-3
> separation itself; deferred mathlib items (shared two-sided tail lemma,
> Bayes posterior equivalence theorem, BoundedSpace instances); upstream-PR
> candidates: `sigmoid_lipschitz`, `neg_log_sigmoid_lipschitz`.
>
> **Historical (superseded by the passes above):**
> 2. **Phase F (feasibility confirmed for MG):** `HeavyHitters.lean` already
>    proves `size_le_capacity` (structural half of `MGAlgorithm.size_bound`);
>    the overcount direction of the frequency invariant is also proved. A
>    concrete `MGAlgorithm` instance discharging the bundle is a realistic
>    next FormalProbability task; GK/KLL quantitative bounds stay disclosed
>    obligations; Feldman Thm 3 needs a concrete private-coin MUD model
>    (research-grade, unscheduled).
> 3. GRPO-RL realized-estimator analogue in TreePOEndToEndGlue (mechanical).
>
> **End-of-day state:** all three lake targets green, 8,023 jobs; the only
> unbuilt modules on disk are the three explicit archives
> (`DSL/OLD_MEstimationCore`, `DSL/OLD_SectionIPWTwoStage`,
> `Deprecated/PointwiseLipschitz`); 313 crosswalk names verify (the single
> scripted "miss", `MarkovCountSketch.mul`, is the known dotted-name false
> positive — it is `def mul` inside the namespace).

Scope: `lean3/FormalProofs` (+ `~/FormalProbability` MergeableSummaries layer) audited
against `paper/ctreepo/main_v12_external.tex` for (a) fidelity in both directions,
(b) accuracy of Appendix E (proof map / Lean crosswalk), (c) minimality and
organization. Every load-bearing finding below was verified directly in source,
not just reported by a sub-audit.

## 0. What is solid

- `lake build` green: 7,972 jobs, zero errors (only `simpa` style lints in
  legacy Econometrics files).
- **Zero `sorry`, `axiom`, `opaque`, `unsafe`, `partial def`** across both repos.
  External facts are carried as typed hypotheses/structure fields — the right
  architecture.
- All **250 Lean names** cited in Appendix E's crosswalk table exist in source;
  all `\ref`s in Appendix E resolve.
- The C1/C2/C3 ↔ L1/L3/L2 naming crosswalk is literally correct
  (`LocalLaws.lean:155-161` has `abbrev C1 := @L1`, `C2 := @L3`, `C3 := @L2`).
- The genuinely strong formalizations: `multi_round_proper` (the real L3
  round-induction), the Azuma–Hoeffding same-weight-halving analysis
  (Agarwal), VC/Sauer–Shelah range-space layer via mathlib, the Kovachki
  Lemma 21/22 finite-dimensionalization (honestly proved), the Bayes/mathlib
  bridge, and the HT/audit-robustness lemmas.

---

## 1. Fidelity findings (paper ↔ Lean), ranked

### F1. `one_pass` is circular at the root; `ass:context` has no Lean counterpart — the flagship gap

`L2` (`OPT/LocalLaws.lean:117-120`) asserts, per internal node,
`Egu g (node T_L T_R) (D f* · vs S(node)) = 0` — zero distortion of the **full
recursive reduction of that subtree**. Instantiated at the root pair, this *is*
the `one_pass` conclusion; `nodewise_preservation`'s root case is literally
`exact h2 (T_R, hT_L)` (`PreservationTheorems.lean:78`). So the certified
Theorem 1 assumes subtree-level preservation at every node and reads it off,
while the paper bills Theorem 1 as *one-call local checks composing to the
root* (C1 at leaves + two-link C3 per merge + context compatibility,
Appendix B proof by induction).

- `ass:context` (context-compatible oracle) appears **nowhere** in the Lean
  tree. Appendix E's note that it is "absorbed into the local hypothesis form"
  of L2 and that the difference is "purely cosmetic / a presentation choice"
  is not accurate: the hypotheses are not equivalent packagings, and the
  compositional content of the paper's induction is not certified.
- The only derivation of L2 from more primitive laws is
  `A1_A2_A3_implies_L2` (`GlobalAssumptions.lean:524`), which needs
  **global** sufficiency (`∀ z, D f* (g z) z = 0`, all strings, not realized
  leaves) and a **deterministic** summarizer.

**Recommendation (new theorem, highest value):** define a genuinely local
per-merge law, e.g. `L2local g T f* : ∀ (T_L,T_R) ∈ internal_nodes T,
E_{z_L~reduce g T_L, z_R~reduce g T_R}[D f* (g(z_L·z_R)) (S T_L · S T_R)] = 0`
conditioned only on the one merge call — or the paper-faithful pair
(two-link C3 + a `ContextCompatible` predicate) — and prove
`one_pass_of_local : L1 → L2local → [context-compat] → Egu … = 0` by real
induction (nonneg distortion + zero expectation ⇒ a.s. correctness of children,
then the merge law). This single theorem makes Theorem 1's story true in Lean
and retires the circularity. Until it lands, Appendix E's packaging note should
say outright that L2 is subtree-level and strictly stronger per node than the
paper's C3.

### F2. `thm:fixed-partition` is a rename, not a formalization

`fixed_partition_extension_instantiation := @multi_round_proper`
(`MainTheorems.lean:426`) — an `abbrev`. The content of Appendix C (partition
rule Π, non-binary finite rooted trees, measurable document-indexed tree map,
outer expectation over X via tower property) is unformalized. Either
(a) formalize: document distribution `μ_X : PMF Strings`, tree policy
`TΠ : Strings → BinTree Strings` with `S (TΠ x) = x`, and
`E_{x~μ_X} E_{ZR}[D] = 0` by tower over per-document `multi_round_proper`
(this is a short, real proof); or (b) soften the crosswalk entry from
"(corollary of multi_round_proper)" to "per-tree kernel only; the Π/tower
extension is paper-only".

### F3. `fold_of_folds` does not formalize Corollary 2

Body is `exact one_pass …`; the L3 hypothesis is `_h3` (unused); `T_comp` is an
arbitrary tree with no two-level fold structure
(`PreservationTheorems.lean:167-170`). Either formalize the fold composition
(folds as a partition of leaves; C2 on intermediate summaries) or delete the
theorem and the crosswalk row and let Corollary 2 cite `one_pass` +
`multi_round` honestly.

### F4. `schedule_invariance` has an unused hypothesis and no oracle-value content

Same-partition hypothesis `_h_l` is unused; the theorem equates two zeros
(both distortions vanish separately) rather than the corollary's "same expected
oracle value" claim (`PreservationTheorems.lean:145-147`). A faithful version:
under C1+C3 on both trees, `E[f*(Z_T)] = E[f*(Z_T')]` (follows from a.s.
correctness of each against the shared span — needs `leaves T = leaves T'` to
make `S T = S T'`, which is exactly the hypothesis currently ignored).

### F5. `thm:unified-gap`: coupling mismatch + hidden diameter bound

`unified_preference_gap_bounded` (`PreferenceBounds.lean:642-652`) hard-codes
the **independent product coupling** `Δ_R = Σ_z Σ_x μ_Z(z)μ_X(x)·dist`, while
the paper says "any coupled pair (X, Z^(R)(X))". These agree only for
point-mass `μ_X` (which is what `coupling_Δ_eq_Δ_R_ZR` proves). It also needs a
global `D_max` bound on oracle distances that the paper never states.
Fix options: state the Lean theorem over an arbitrary joint coupling
`μ : PMF (Strings × Strings)` with marginals (mechanical rewrite of the same
proof), or add a paper/crosswalk sentence that the formalized version is the
per-document (point-mass) case plus the product-coupling generalization.

### F6. `thm:pref-equiv`: single-document scope; CF replaced by a stronger premise; hidden boundedness

- The via-ZR equivalences (`dpo_equivalence`, `PreferenceBounds.lean:1829`;
  `grpo_equivalence`, `PreferenceLearning.lean:663`) fix `μ_X = PMF.pure x`.
  The general-μ forms require the whole support of μ_X to be one
  oracle-equivalence class. The population statement over heterogeneous X
  (per-document equality + tower) is never taken — same missing tower step as F2.
- Paper `ass:CF` (conditional-expectation factorization) is not formalized;
  Lean uses pointwise oracle-measurability of the loss, strictly stronger.
  (The paper's own Appendix B proof also quietly upgrades to pointwise — so
  either weaken Lean to CF or strengthen the paper's stated assumption; today
  the paper *statement* and the Lean *theorem* disagree while the paper *proof*
  and Lean agree.)
- `[BoundedMetricSpace Y]` is required (inherited from multi-round);
  argmin statements are restricted to the oracle-measurable policy class
  (consistent with Table tab:measurability, but bare "argmin_π" in the theorem
  statement hides it).

### F7. `thm:e2e`: two proved halves, never glued

- Concrete half (`DSL/TreePOEndToEnd.lean:259-288` + grpo variants): requires
  `[Fintype Strings/Node/A]`, independent-Bernoulli sampling with propensities,
  a constant pair generator for DPO (GRPO-PL has a generalized version; DPO
  doesn't), and bounds the gap by the **expected** HT estimator (an integral),
  not the realized `μ̂_dist` the paper displays.
- Abstract half (`PaperErrorCertificate/Stack`): faithful triangle+union-bound
  shape, but the transported term is a deterministic envelope dominating
  `|gap_clip ω|` for all ω, again not the realized estimator; and
  `paper_error_certificate_formula` is a `rfl`.
- **No Lean theorem instantiates the abstract stack with the concrete HT
  estimator.** Recommendation: one glue theorem
  (`treepo_certificate_instantiates_error_stack`) constructing a
  `PaperErrorStack` from the certificate theorem's objects; and a
  realized-estimator variant using the existing variance/Chebyshev or
  Serfling/Bernstein pieces so the bound holds with `μ̂_dist` at confidence
  1−δ_est, matching the paper's display.

### F8. Multi-round's hidden hypothesis

`multi_round_proper/bounded` need a global distortion bound `M` (or
`[BoundedPseudoMetricSpace Y]`); the paper's Recompression Stack states no
boundedness. This replaced a previously **unsound summability axiom** — good —
but the paper should now say "bounded oracle metric" in the Recompression
Stack (one clause), since the tsum-based law definitions are vacuously
satisfiable in unbounded spaces (non-summable ⇒ `tsum = 0` by convention).
Worth a one-line remark in Appendix E as well.

### F9. Reverse-coverage gaps (paper claims with no Lean artifact)

| Item | Status | Suggested action |
|---|---|---|
| `ass:CF` | never mentioned in Appendix E | add a crosswalk row mapping it to the `OracleMeasurableLoss`/bundle premises, noting the pointwise-vs-conditional strengthening (F6) |
| `ass:context` | absent from Lean | resolve via F1; until then, disclose |
| `eq:error_budget` (§10 union bound) | no Lean | trivial finite union bound — formalize (~30 lines) or mark paper-only |
| `prop:m_lt_k` | acknowledged paper-only | finite pigeonhole counterexample; formalizable with `decide` in an afternoon — the paper's only impossibility result deserves Lean backing |
| 5 lemmas in B (`sigmoid-lip`, `neglogsig-lip`, `dpo-lip`, `dpo-oracle-meas`, `zero-dist-support`) | LaTeX-only | sigmoid/neglogsig Lipschitz are one-liners over mathlib; `dpo-lip` likely already implicit inside `unified_preference_gap` machinery — either port or add "subsumed by X" notes to E |
| App I `trueOracle_delta_R_ZR_le_of_calibrated_neuralOperatorBridge` | cited inline in I, missing from E's table | add row |
| `thm:projection-iff`, `ass:pref`, `eq:discussion-bound` | matched by row description only, no `\ref` | add `\ref`s so label-search auditing works |

---

## 2. Disclosure accuracy (Appendix E boundary paragraphs)

Direction of error is uniform: **more is assumed than disclosed**. Nothing
declared external turned out to be secretly proved.

- **D1 (worst): Gibbons §5 "reference runtime package" is vacuous.**
  `referenceSection5RuntimeClaims` (`Gibbons1996.lean:1159-1166`) instantiates
  each `SizedCostModel` with `worstCase :=` the target growth function and
  closes the Big-O obligations by `isBigO_refl`. It checks a *stipulated cost
  model*, not operation counts. Appendix E says the coverage pass "checks a
  Gibbons Section 5 reference runtime package" — reword to "stipulated
  reference cost model" (or drop the claim).
- **D2: KLL, GK, and Misra–Gries quantitative bounds are undisclosed external
  obligations.** `KLL.Algorithm.rank_error/theorem4_space_mergeable`,
  `GK.Algorithm.theorem1_space/corollary2_one_way`, `MGAlgorithm.size_bound/
  frequency_error` are caller-supplied structure fields with **no concrete
  instance in either repo**, yet E's external list ((i) HLL Mellin,
  (ii) Agarwal discrepancy/geometry, (iii) Feldman communication bounds,
  (iv) Kovachki 11/13) omits them — even though
  `Literature.externalObligationSurface` (FP `Literature.lean:104`) names them.
  Fix: cite the repo's own obligation surface verbatim in E.
- **D3: Feldman Theorem 3 schema is malformed.**
  `theorem3_private_randomness_separation_statement` (`Feldman2008.lean:2419`)
  existentially quantifies **its own computability predicates**, so it is
  trivially satisfiable (`RandomStreamingComputable := fun _ => True`,
  `PrivateRandomMUDComputable := fun _ => False`). Theorems 4/5 use concrete
  predicates; repair Theorem 3 to match. Also: two `_statement` `def`s are
  listed in E's table under "Theorem / proof names" without distinguishing
  them from proved entries — annotate.
- **D4: "axiom" wording.** `B_proofs.tex:639` says the GRPO transport constant
  "uses the `ExpectedGroupLossLipschitz` axiom". It is a Prop-valued
  *assumption interface* (`PreferenceBounds.lean:1969`), with sufficient
  conditions **proved** in `OPT/RUMSufficientConditions.lean` (fixed-ranker PL;
  finite pointwise GRPO-RL). Current wording both overstates the trust cost
  and undersells the formalization. Say "assumption, with formally verified
  sufficient conditions".
- Minor: Agarwal disclosure should also name the random-sample ε-approximation
  and randomized-quantile (Cor 3.12) constructions; Feldman disclosure should
  say the assumed set includes a deterministic Symmetric-Index bound and the
  full Thm 4/5 separation schemas.

---

## 3. Minimality and organization

Repo: 211 files, ~90.5k lines; umbrella reaches 179 modules; paper crosswalk
cites 27 FormalProofs files (+2 in Appendix J: `OPT/BagOfWordsLDARecovery`,
`OPT/LDAAggregateStatistics`).

- **M1 (dominant): `OPT/MainTheorems.lean` is a 5,485-line hub** (91 imports,
  1,024 `abbrev`s) that inflates the paper closure from ~58 modules to 137.
  Recommendation: split into a paper-facing `OPT/PaperTheorems.lean`
  restricted to Appendix E's rows (with `#check`s), and `OPT/ExtendedExports.lean`
  for everything else; point the crosswalk and REPRODUCIBILITY.md at a
  dedicated `lake build FormalProofs.Paper` target. This one change is most of
  "minimal companion".
- **M2: three overlapping assumption registries**, one unbuilt.
  `Axioms.lean` (642 lines) is advertised by the umbrella docstring
  (FormalProofs.lean:197-198) but imported by nothing ⇒ never built, already
  duplicating definitions that live in `PreferenceBounds.lean` (drift hazard).
  Merge `Axioms.lean` + `Assumptions.lean` + `TechnicalAxioms.lean` into one
  imported registry.
- **M3: the file naming the *active* objective is orphaned.**
  `OPT/DoublyRobustMinimizationObjective.lean` (394 lines; the
  (1−Λ)·root + Λ·DR-adjusted surface matching ObjectiveSpec v1 on the Python
  side) is unbuilt, while its five precursor layers
  (`NodeLocalLawAggregate → NodeAIPW → Unified → DoublyRobust → DiscountedIPW`)
  are all imported. Consolidate the ladder into one `LocalLawObjective.lean`
  whose final section is the active surface; re-import.
- **M4: Econometrics quarantine breached.** Umbrella says the legacy
  `Econometrics.*` subtree is direct-import-only (namespace clash), but
  `DSL/MainTheorems.lean:49` imports `Econometrics.OLS.AsymptoticOLS`,
  dragging 1,894 lines in silently. Cut the import (the paper needs only
  `DSL/{TreePOEndToEnd, TreeIPW, IPWTheory, LabelRateBounds}`), then archive
  the whole legacy Econometrics subtree (~5.4k lines, uncited) or migrate it
  to FormalProbability.
- **M5: ghost references.** 12 doc references cite a nonexistent `DPO.lean`
  (content lives in `PreferenceLearning`/`PreferenceBounds`); `OPT/README.lean`
  and `DSL/README.lean` are never type-checked and have drifted. Fix refs;
  import the READMEs (doc modules are cheap) or delete them.
- **M6: 32 orphaned modules.** Re-import: `Axioms.lean` (M2),
  `DoublyRobustMinimizationObjective` (M3); decide:
  `ContextualStateRecovery`, `AnalysisPartitionMismatch`/`AnalysisSummaryLocalLaws`
  (companions of live `LeafLocalMixtureUtilityGap`), `WorkedExampleCMSTree`.
  Archive: `Deprecated/PointwiseLipschitz`, `DSL/SectionIPWTwoStage` (renaming
  shim), `DSL/MEstimationCore` (superseded), legacy Econometrics subtree.
  Per repo convention, archive = `OLD_` prefix + header note, never delete.
- **M7: consolidation clusters** (beyond M1, ranked by lines/clarity):
  DSL textbook cluster (~6.3k lines uncited by the paper — archive or move);
  sufficiency/SBI-literature cluster (18 files ~6.1k lines; keep the 3
  paper-cited Bayes files, fold or archive the rest — the Bayesian-persuasion
  trio is a clean archive); `TheoremBacking*` 5 files → 1–2;
  two-stage cluster → fold into `UnifiedOracleRoute` (its self-declared
  purpose); fiber/feature-class 9 files → 1–2; `Audit`+`AuditSizes` →
  `AuditBounds`; 10 CLT stub files → `CLT.lean`.

---

## 4. Suggested new Lean work, in priority order

1. **`one_pass_of_local`** — local per-merge law (+ context-compat) ⇒ root
   preservation by real induction (F1). Changes the formalization's story from
   "assumed composition" to "proved composition".
2. **Tower-step population theorems** — document distribution μ_X + measurable
   tree policy: fixed-partition (F2) and population `thm:pref-equiv` (F6).
   Small proofs, large fidelity gain.
3. **Glue theorem for thm:e2e** (F7) + realized-estimator (μ̂_dist) variant.
4. **Coupled-Δ_R unified gap** (F5).
5. **Honest `fold_of_folds` and `schedule_invariance`** or deletion (F3/F4).
6. **`prop:m_lt_k`** finite counterexample; `eq:error_budget` union bound;
   the five B lemmas (F9) — cheap completeness wins.
7. **Repair Feldman Theorem 3 schema; instantiate or clearly flag KLL/GK/MG
   obligation bundles** (D2/D3).

## 5. Paper edits implied (for the next prose pass)

- Appendix E: rewrite the "C3 vs L2 packaging" note honestly (subtree-level,
  strictly stronger, context-compat unformalized) until F1 lands; add rows for
  `ass:CF`, App-I inline theorem; add missing `\ref`s; annotate `_statement`
  entries; extend the external-obligation list per D2; reword Gibbons per D1.
- B_proofs: "axiom" → "assumption with verified sufficient conditions" (D4).
- Theory section: state the bounded-oracle-metric hypothesis in the
  Recompression Stack (F8); flag the argmin class restriction (F6).
- Crosswalk row for fixed-partition: "per-tree kernel formalized; Π/tower
  extension paper-only" until item 2 lands.
