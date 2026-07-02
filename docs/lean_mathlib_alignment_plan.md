# Lean / Mathlib Alignment Plan

**Date:** 2026-07-02
**Repos surveyed:**
- `~/ThinkingTrees/lean3` (package `FormalProofs`, mathlib `v4.27.0-rc1`, rev `32d24245c7a1`, toolchain `leanprover/lean4:v4.27.0-rc1`)
- `~/FormalProbability` (same mathlib pin; TT depends on it via `lakefile.toml` path dep `../../FormalProbability`)

**Goal:** lean on original mathlib wherever possible instead of custom re-implementations, for trust (mathlib-reviewed proofs) and external-audience credibility. Every mathlib declaration below was verified by `rg` against `.lake/packages/mathlib` at the pinned rev — not from memory.

**Verdict legend:** REPLACE = mathlib is a drop-in; BRIDGE = keep custom def, add proved equivalence lemma to the mathlib object; KEEP = mathlib has nothing / custom is genuinely different. DONE = already built on mathlib, only hygiene remains.

## Executive Summary (ranked by credibility gain / effort)

| # | Area | Repo has | Mathlib v4.27 has | Verdict | Effort | Blast radius |
|---|------|----------|-------------------|---------|--------|--------------|
| 1 | Sigmoid / expit | duplicate `expit` def (TT `FormalProofs/DSL/CoreDefinitions.lean:229`, FP `DSL/CoreDefinitions.lean:172`) | `Real.sigmoid` + full API (`Mathlib/Analysis/SpecialFunctions/Sigmoid.lean`) | REPLACE | S | 3 files TT + 3 files FP |
| 2 | Concentration naming/dedup | proved-on-mathlib Hoeffding still named `..._axiom`; conditional-Hoeffding bridge duplicated in 2 repos | `HasSubgaussianMGF` Hoeffding/Azuma stack (`Mathlib/Probability/Moments/SubGaussian.lean`) | DONE + hygiene | S | 2 files TT + 1 file FP |
| 3 | Misc one-liners (`tsum_eq_zero_of_nonneg`, `innerProduct`, `IsPosDef'`) | local re-proofs/dupes | `hasSum_zero_iff_of_nonneg`, `dotProduct`, `Matrix.PosDef` | REPLACE | S | 1–4 files each |
| 4 | PMF expectation (`Exp`/`Eg`/`Egu`) | tsum-based defs (`OPT/ExpectationTheory.lean:38`, `OPT/CoreDefinitions.lean:39`); bridge exists but buried | `PMF.toMeasure`, `PMF.integral_eq_tsum` (`ProbabilityMassFunction/Integrals.lean:32`) | BRIDGE (promote + extend existing bridge) | M | 34 files use `Exp`, 15 use `Eg`/`Egu` |
| 5 | Bayes layer | finite-algebraic Bayes (`OPT/FiniteBayesOnState.lean:43-84`); alias bridge exists | `ProbabilityTheory.posterior` kernel API (`Probability/Kernel/Posterior.lean:69`) | BRIDGE (one real equivalence theorem missing) | M | 3 files |
| 6 | `BoundedPseudoMetricSpace` | custom class (`Shared/BoundedMetricSpace.lean:33`) | `BoundedSpace` + `Metric.diam` + `dist_le_diam_of_mem` | BRIDGE (instance + diam lemmas); full replace not worth it | M (bridge) / L (replace) | 30–39 files |
| 7 | Sub-Gaussian API for FP/TT IPW Hoeffding | `htExpEstimator_hoeffding_bound` (TT `DSL/IPWTheory.lean:848`) | already stated ON `HasSubgaussianMGF` | DONE | — | — |
| 8 | CLT | full custom CLT, no axioms/sorries (FP `CLT/CLT.lean:1951`) | **no CLT, no Lévy continuity** in mathlib v4.27 | KEEP (uses mathlib `charFun`/`gaussianReal` already); upstream candidate | — | 17 FP files |

---

## 1. Sigmoid / logistic — REPLACE (best ratio)

**(a) Repo has**

- TT `FormalProofs/DSL/CoreDefinitions.lean:229`: `def expit (x : ℝ) : ℝ := 1 / (1 + Real.exp (-x))` (+ `logit` at :231, `expit_range` at :238).
- FP `FormalProbability/DSL/CoreDefinitions.lean:172`: identical duplicate `expit`.
- `expitDerivative` re-derivations: TT `FormalProofs/DSL/LogisticRegression.lean:114`, FP `DSL/LogisticRegression.lean:116`.
- The OPT side already uses mathlib: `FormalProofs/OPT/PreferenceLearning.lean` (BTL/DPO, ~30 uses of `Real.sigmoid`), `OPT/PreferenceBounds.lean:111` (`sigmoid_lipschitz`), `:134` (`neg_log_sigmoid_lipschitz`), `OPT/PaperSupportingLemmas.lean:67-91` (paper-facing wrappers).

**(b) Mathlib v4.27 actually provides** (`Mathlib/Analysis/SpecialFunctions/Sigmoid.lean`; there is NO `Real.logistic` and NO `logit` anywhere in mathlib)

- `Real.sigmoid` (defined as `(1 + exp (-x))⁻¹` — defeq-close to `expit`, differs only `1/x` vs `x⁻¹`, i.e. `one_div`).
- `Real.sigmoid_pos:71`, `sigmoid_lt_one:79`, `sigmoid_le_one:83`, `sigmoid_strictMono:86`, `sigmoid_neg:108` (`sigmoid (-x) = 1 - sigmoid x`), `range_sigmoid:118`, `tendsto_sigmoid_atTop/atBot:128/132`, `hasDerivAt_sigmoid:136`, `deriv_sigmoid:142`, `analyticAt_sigmoid:152`, `differentiable_sigmoid:191`, `differentiableAt_sigmoid:199`; plus `unitInterval.sigmoid`, `OrderEmbedding.sigmoid`, `measurableEmbedding_sigmoid`.
- NOT in mathlib: `sigmoid_lipschitz`, `neg_log_sigmoid_lipschitz` (the repo's `OPT/PreferenceBounds.lean` proofs are genuine extensions — keep them; both are clean upstream-PR candidates).

**(c) Verdict:** REPLACE. Add `lemma expit_eq_sigmoid : expit = Real.sigmoid := by funext x; simp [expit, Real.sigmoid_def, one_div]` and then either rewrite the 6 files to use `Real.sigmoid` directly, or keep `expit` as an `abbrev`-level alias with the equation lemma. `logit` stays custom (mathlib has none) but should be stated against `Real.sigmoid`.

**(d) Effort:** S. Blast radius: TT 3 files (`DSL/CoreDefinitions.lean`, `DSL/LogisticRegression.lean`, `DSL/MomentFunctions.lean`), FP 3 files (`DSL/CoreDefinitions.lean`, `DSL/LogisticRegression.lean`, +1). High credibility gain: "our logistic regression layer's link function IS mathlib's" is a one-line audit story.

## 2. Concentration hygiene — DONE on mathlib; rename + dedupe

**(a) Repo has**

- TT `FormalProofs/OPT/MeasureTheoreticAudit.lean`: `hoeffding_iid_bounded` — a **proved lemma** built on mathlib's Hoeffding lemma (`hasSubgaussianMGF_of_mem_Icc_of_integral_eq_zero` appears in the proof at ~:407) — plus a backward-compat alias at `:533` named `hoeffding_iid_bounded_axiom`. The name says "axiom"; the term is a theorem. Pure credibility own-goal.
- TT `FormalProofs/OPT/SerflingAudit.lean:68`: `hasCondSubgaussianMGF_of_mem_Icc_of_condExp_eq_zero` (conditional Hoeffding bridge), `:138`/`:192` one/two-sided Azuma, plus a without-replacement permutation model. Mathlib has NO Serfling inequality, so the file itself is a KEEP.
- FP `FormalProbability/DSL/SamplingConcentration.lean:50-137`: **the same conditional-Hoeffding bridge lemma and Azuma wrappers, duplicated near-verbatim** across the two repos.
- TT `FormalProofs/DSL/IPWTheory.lean:848` (`htExpEstimator_hoeffding_bound`): see §7 — already fully on mathlib.
- Both repos hand-roll the same two-sided `|·| ≥ ε` union-bound wrapper (MeasureTheoreticAudit ~:500-530, IPWTheory ~:950-984, SamplingConcentration :140-200, SerflingAudit :192).

**(b) Mathlib v4.27 actually provides** (all verified in `Mathlib/Probability/Moments/`)

- `ProbabilityTheory.HasSubgaussianMGF` (SubGaussian.lean, measure version doc :66, kernel structure :142) and `Kernel.HasSubgaussianMGF`, `HasCondSubgaussianMGF`.
- Hoeffding's lemma: `hasSubgaussianMGF_of_mem_Icc_of_integral_eq_zero` (:843), corollary `hasSubgaussianMGF_of_mem_Icc` (:859).
- Hoeffding inequality: `HasSubgaussianMGF.measure_sum_ge_le_of_iIndepFun` (:780), `measure_sum_range_ge_le_of_iIndepFun` (:786).
- Azuma–Hoeffding: `measure_sum_ge_le_of_HasCondSubgaussianMGF` (:923) via `HasSubgaussianMGF_sum_of_HasCondSubgaussianMGF` (:905).
- Chernoff: `ProbabilityTheory.measure_ge_le_exp_mul_mgf` (Moments/Basic.lean:411); `mgf`/`cgf` (:121/:125).
- Chebyshev: `ProbabilityTheory.meas_ge_le_variance_div_sq` (Moments/Variance.lean:378); `variance_le_sq_of_bounded`.
- **Absent from mathlib:** Bernstein inequality (only `Analysis/SpecialFunctions/Bernstein.lean` = polynomial approximation), McDiarmid, Serfling, and any two-sided `|·|` sub-Gaussian tail.

**(c) Verdict:** DONE + hygiene. Actions: (i) rename/deprecate `hoeffding_iid_bounded_axiom` → keep only `hoeffding_iid_bounded` (S; 2 call sites: `TechnicalAxioms.lean`, `MeasureTheoreticAudit.lean`); (ii) de-duplicate the conditional-Hoeffding bridge: FP `SamplingConcentration.lean` is the natural home (TT already depends on FP), TT `SerflingAudit.lean` re-exports (M — cross-repo import shuffle); (iii) factor one shared two-sided lemma `measure_abs_ge_le_of_iIndepFun` and use it in all 4 duplication sites (S–M). The conditional-Hoeffding bridge itself is a strong mathlib upstream-PR candidate — that is the max-credibility endgame.

**(d) Effort:** S for (i), S–M for (ii)/(iii). Blast radius: TT `TechnicalAxioms.lean`, `OPT/MeasureTheoreticAudit.lean`, `OPT/SerflingAudit.lean`; FP `DSL/SamplingConcentration.lean`.

## 3. Misc one-liners — REPLACE

| Repo item | Location | Mathlib replacement (verified) | Effort |
|---|---|---|---|
| `tsum_eq_zero_of_nonneg` | TT `OPT/ExpectationTheory.lean:168` | `hasSum_zero_iff_of_nonneg` (to_additive of `hasProd_one_iff_of_one_le`, `Topology/Algebra/InfiniteSum/Order.lean:186`) + `Summable.hasSum` | S |
| `innerProduct` | TT `DSL/CoreDefinitions.lean:225`, FP `DSL/CoreDefinitions.lean:168` | `dotProduct` (`Mathlib/Data/Matrix/Mul.lean`); FP `Econometrics/Matrix.lean:12` already aliases it — unify on the alias | S |
| `IsPosDef'` | FP `Media/ElectionCoverageTV/Core/Lemmas/CholeskyUniqueness.lean:55` | `Matrix.PosDef` (`LinearAlgebra/Matrix/PosDef.lean:160`) — restate `IsPosDef'` as `Matrix.PosDef` or add iff lemma. Cholesky itself: mathlib has **nothing** (rg `cholesky` = 0 hits) → the uniqueness development is KEEP | S |
| `Exp_pure`, `ExpENN_bind`, `PMF.toReal_tsum_coe` | TT `OPT/ExpectationTheory.lean:44-124` | partially: `PMF.tsum_coe` covers the mass identity; keep the rest (they're about the custom `Exp`) | S |

Also checked, already clean (no action): FP `CLT/WeakLaw.lean:28` derives WLLN from mathlib `strong_law_ae` (`Probability/StrongLaw.lean:790`); `ConvergesInProbability`/`ConvergesInDistribution` (FP `CLT/Core.lean:101-116`) are thin defs over mathlib `TendstoInMeasure` / `MeasureTheory.TendstoInDistribution` (`MeasureTheory/Function/ConvergenceInDistribution.lean`); FP Econometrics OLS/GMM/IV files use mathlib `Matrix` API directly with only domain-level defs.

## 4. PMF expectation stack (`Exp`, `Eg`, `Egu`) — BRIDGE, not replace

**(a) Repo has**

- `def Exp (p : PMF α) (f : α → ℝ) : ℝ := ∑' z, (p z).toReal * f z` — TT `OPT/ExpectationTheory.lean:38` (+ `ExpENN` :41).
- `def Eg (g : Summarizer α) (f : α → ℝ) (x : α) : ℝ := ∑' z, (g x z).toReal * f z` — TT `OPT/CoreDefinitions.lean:39`.
- `Egu` — TT `OPT/LocalLaws.lean:84`.
- Supporting summability lemmas (`PMF.summable_coe_real`, `PMF.summable_coe_real_mul_of_bounded`, `summable_D_of_bounded`) — `OPT/ExpectationTheory.lean:69-152`. The old unsound `PMF.summable_coe_real_mul` axiom is already removed (docstrings at :88/:120 record the history).
- **A bridge already exists**: `Exp_eq_integral` (`Exp p f = ∫ z, f z ∂p.toMeasure` given `Integrable`) at TT `OPT/MeasureTheoreticAudit.lean:56-63`, proved from `PMF.integral_eq_tsum`. It is buried in an audit file rather than in the expectation home.

**(b) Mathlib v4.27 actually provides** (`Mathlib/Probability/ProbabilityMassFunction/Integrals.lean`)

- `PMF.integral_eq_tsum (p : PMF α) (f : α → E) (hf : Integrable f p.toMeasure) : ∫ a, f a ∂p.toMeasure = ∑' a, (p a).toReal • f a` (:32).
- `PMF.integral_eq_sum` for `[Fintype α]` (:47), `PMF.bernoulli_expectation` (:57).
- No `PMF.expect`-style tsum definition exists in mathlib; the mathlib-native notion of PMF expectation IS the Bochner integral against `p.toMeasure`.

**(c) Verdict:** BRIDGE. Full replacement (`Exp p f` ↦ `∫ z, f z ∂p.toMeasure`) would push `Integrable` side-conditions into ~34 files whose current proofs run on plain `tsum` algebra (`tsum_nonneg`, bind/pure unfolding in `reduce`/`ZR`) — bad trade. Instead: (i) move `Exp_eq_integral` from `MeasureTheoreticAudit.lean` into `ExpectationTheory.lean` next to the definition; (ii) add the two missing corollaries `Eg_eq_integral` (`Eg g f x = ∫ z, f z ∂(g x).toMeasure`) and `Egu_eq_integral`; (iii) add a Fintype special case via `PMF.integral_eq_sum` with no integrability hypothesis; (iv) state in the module docstring that `Exp` is definitionally mathlib's `PMF.toMeasure` expectation modulo `PMF.integral_eq_tsum`. That gives auditors a one-hop proof that the whole `Exp`-world is standard.

**(d) Effort:** M (the lemmas are S; placing them without import cycles and sweeping docstrings is the M part). Blast radius if replacing instead: 34 files (`\bExp\b`), 15 files (`Eg`/`Egu`) — which is exactly why not to replace.

## 5. Bayes layer — BRIDGE (one real theorem missing)

**(a) Repo has**

- TT `OPT/FiniteBayesOnState.lean` (995 lines): finite-algebraic `BayesNumerator:43`, `BayesEvidence:51`, `BayesPosterior:60`, state versions :69-84, MAP :142-179, posterior expectation/predictive/risk :273-476. Deliberately finite and real-valued (docstring says so).
- TT `OPT/PosteriorConsistency.lean` (634 lines): consistency predicates.
- TT `OPT/MathlibBayesBridge.lean` (607 lines): ~50 `abbrev` aliases to mathlib (`ProbabilityTheory.cond*`, `condExp*`, `posterior*`, `PMF.*`, `pdf`) **plus proved content**: `posteriorConsistent_iff_mathlib_tendstoInMeasure:268`, `bayesPosteriorPMF_apply:370`, `bayesPosteriorPMF_toMeasure_singleton/_set:384/404`, state versions :476-522, and `bayesPosteriorPMF_likelihoodOnState_eq_stateBayesPosteriorPMF:573`.

**(b) Mathlib v4.27 actually provides**

- Kernel posterior API, all verified in `Mathlib/Probability/Kernel/Posterior.lean`: `ProbabilityTheory.posterior` (:69, notation `κ†μ`), `compProd_posterior_eq_map_swap:80`, `posterior_eq_withDensity`, `posterior_eq_withDensity_of_countable`, `rnDeriv_posterior`, `ae_eq_posterior_of_compProd_eq`, `posterior_comp_self`, `posterior_id`, `posterior_posterior`, `posterior_comp`.
- Event Bayes: `ProbabilityTheory.cond_eq_inv_mul_cond_mul`, `cond_apply`, `cond_cond_eq_cond_inter`, etc. (`Probability/ConditionalProbability.lean`), `MeasureTheory.condExp` (note: current mathlib name is `condExp`, aliased in the bridge already).

**(c) Verdict:** BRIDGE — mostly done; the alias wall is fine but aliases prove nothing. The one missing high-credibility theorem: **`bayesPosteriorPMF` = mathlib's `posterior` for the induced finite kernel**, i.e. build `κ : Kernel State Obs` from the likelihood on a Fintype, and prove `(bayesPosteriorPMF …).toMeasure = (κ†(priorPMF.toMeasure)) obs` via `posterior_eq_withDensity_of_countable`. That single lemma certifies the whole finite Bayes stack against mathlib's disintegration semantics. Everything else in `FiniteBayesOnState` (MAP, risk, predictive) is domain content on top and should stay.

**(d) Effort:** M (kernel construction on a Fintype + one `withDensity` computation). Blast radius: additive only — no existing file changes; 3 files involved.

## 6. `BoundedPseudoMetricSpace` — BRIDGE now; full replace is L and not worth it

**(a) Repo has** — TT `FormalProofs/Shared/BoundedMetricSpace.lean` (154 lines):

- `class BoundedPseudoMetricSpace (α) extends PseudoMetricSpace α` with **data** `diameterBound : ℝ`, `diameterBound_pos : 0 < diameterBound`, `dist_le_diameterBound` (:33-40).
- `UnitBoundedPseudoMetricSpace` (diam = 1, :70), `BoundedMetricSpace` (proper-metric version, :117), constructors `ofBound`/`ofBoundOne` (:91-104).
- Usage pattern: `diameterBound` is consumed as an explicit constant `M` for summability (`OPT/ExpectationTheory.lean:775`, `multi_round_typeclass` :769-780) and instance transport (`OPT/ProductScoreFiber.lean:122`).

**(b) Mathlib v4.27 actually provides**

- `class BoundedSpace (α) [Bornology α] : Prop` with `bounded_univ` (`Topology/Bornology/Basic.lean:273`); `Bornology.isBounded_univ : IsBounded univ ↔ BoundedSpace α` (:286).
- `Metric.diam : Set α → ℝ` (`Topology/MetricSpace/Bounded.lean:387`), `Metric.dist_le_diam_of_mem : IsBounded s → x ∈ s → y ∈ s → dist x y ≤ diam s` (:474), `Metric.diam_le_of_forall_dist_le` (:429), `Metric.ediam_le_of_forall_dist_le` (:423), `Metric.isBounded_iff` (`∃ C, ∀ …, dist x y ≤ C`, :452 region).

**What breaks under naive replacement** `[PseudoMetricSpace Y] [BoundedSpace Y]` + `Metric.diam Set.univ`:

1. `diameterBound_pos` has no analogue — `Metric.diam univ = 0` for subsingletons (and `diam_univ_of_noncompact` even forces 0 for unbounded proper spaces). Proofs that use strict positivity (e.g. `ProductScoreFiber.lean:122-124`) need `Nonempty`+nontriviality or must weaken to `0 ≤`.
2. The custom class carries a *chosen* bound (often exactly `1` via `UnitBoundedPseudoMetricSpace`), not the tight diameter; statements like `dist_le_one` become `diam univ ≤ 1` side-conditions instead of definitional facts.
3. `BoundedSpace` is a `Prop` mixin over `Bornology`; extracting a real constant always goes through `Metric.diam` + `IsBounded`, adding a hypothesis to every summability call site.

**(c) Verdict:** BRIDGE. Add to `Shared/BoundedMetricSpace.lean`: (i) `instance : BoundedSpace α` for `[BoundedPseudoMetricSpace α]` (via `Metric.isBounded_iff` / `isBounded_univ`); (ii) `lemma diam_univ_le_diameterBound : Metric.diam (Set.univ : Set α) ≤ diameterBound` (via `Metric.diam_le_of_forall_dist_le`); (iii) a reverse constructor `BoundedPseudoMetricSpace.ofBoundedSpace [Nonempty α]` taking `M := max (Metric.diam univ) 1`. This makes the class *conservative over mathlib* — an auditor sees it is mathlib's `BoundedSpace` plus a chosen constant. Full REPLACE = L across 30–39 files for near-zero extra credibility once the bridge instances exist.

**(d) Effort:** M for the bridge (S code, M to check no instance-diamond with the existing `extends PseudoMetricSpace`). Blast radius of the class today: 30 files reference the identifiers (39 counting `BoundedMetricSpace` too); only 2 files import the module directly (rest transitive).

## 7. Sub-Gaussian API for the IPW Hoeffding — already DONE

**(a) Repo has** — TT `FormalProofs/DSL/IPWTheory.lean:848` `htExpEstimator_hoeffding_bound` (+ `_unit:985`, `_indicator:999`): tail bound `2·exp(−ε²/(8·|ι|·(M/π_min)²))` for the Horvitz–Thompson estimator under `bernoulliProductMeasure`.

**(b) What it builds on** — inspected the proof body (:855-984): it constructs centered terms, then uses **mathlib's** `hasSubgaussianMGF_of_mem_Icc_of_integral_eq_zero` (:900-903), `HasSubgaussianMGF.measure_sum_ge_le_of_iIndepFun` (:912, :938), `HasSubgaussianMGF.neg`, `iIndepFun.comp`. FP's `DSL/SamplingConcentration.lean` Azuma route similarly terminates in mathlib's `measure_sum_ge_le_of_HasCondSubgaussianMGF`.

**(c) Verdict:** DONE — the question "could the MGF-route Hoeffding be restated on top of `ProbabilityTheory.HasSubgaussianMGF`?" is already answered yes in the code. Optional S improvement: expose an intermediate lemma `hasSubgaussianMGF_htExpCenteredTerm` in the public API so the sub-Gaussian parameter is citable, and reuse the shared two-sided wrapper from §2(iii).

**(d) Effort:** — (S if the optional refactor is taken). Blast radius: 1 file.

## 8. CLT — KEEP (mathlib has none); already maximally mathlib-based

**(a) Repo has**

- TT `FormalProofs/CLT/*` are thin shims: `CLT/CLT.lean` re-exports `FormalProbability.CLT.CLT`; the other 9 files are consolidation shims (2026-07-02).
- FP `FormalProbability/CLT/` = 5,256 lines, **zero `axiom`/`sorry`** (verified by rg): `central_limit_theorem_iid_bounded` (`CLT.lean:1951`), `_cdf:1977`, `_of_charFunScale:1996`, `_abs_pow3:2020`, `_finite_variance:2043`, `_stdNormal:2099`; custom `LevyContinuity.lean` (676 lines), `HellySelection.lean` (235), `Inversion.lean` (72), `Tightness.lean` (285), `TriangularArray.lean`, `MultivariateCLT.lean`.

**(b) Mathlib v4.27 actually provides**

- **No CLT**: `rg -i 'central.?limit'` over mathlib returns zero probability hits. **No Lévy continuity theorem**: `rg 'tendsto_of_tendsto_charFun|continuity theorem'` = zero hits.
- It DOES provide the substrate the custom CLT already consumes (verified in FP imports): `MeasureTheory.charFun`, `Measure.ext_of_charFun` (`MeasureTheory/Measure/CharacteristicFunction.lean:237`), `IntegralCharFun.lean`, `ProbabilityTheory.gaussianReal` (+ `Distributions/Gaussian/CharFun.lean`), `ProbabilityMeasure` weak topology, `Portmanteau.lean`, `LevyProkhorovMetric.lean`, `Tight.lean`, `IdentDistrib`, `TendstoInDistribution`, `ComplexMGF.lean`, `strong_law_ae`.

**(c) Verdict:** KEEP. The custom parts (Lévy continuity, Helly selection, inversion, triangular array) are exactly the parts mathlib is missing, and they sit on mathlib's own `charFun`/Gaussian/portmanteau API — this is the correct architecture. Two actions: (i) add a doc header to `FormalProbability/CLT/README.lean` stating precisely which mathlib decls are consumed and which theorems are original (auditors' map); (ii) track mathlib upstream — Lévy continuity is under active development in the community; when it lands, `LevyContinuity.lean` (676 lines) becomes a REPLACE. `HellySelection`/`LevyContinuity` are the repo's strongest mathlib upstream-PR candidates alongside §2's conditional Hoeffding.

**(d) Effort:** — now; L when upstream lands. Blast radius: 17 FP files + TT shims.

---

## Recommended execution order

1. **Sigmoid/expit REPLACE** (§1) — S effort, both repos, immediate "we use mathlib's sigmoid" story. Includes `expit_eq_sigmoid` lemma + call-site sweep (6 files).
2. **Concentration hygiene** (§2) — retire the `hoeffding_iid_bounded_axiom` name (it is a proved theorem; the name actively damages credibility), dedupe the conditional-Hoeffding bridge into FP, factor the shared two-sided tail lemma.
3. **Misc one-liners** (§3) — `tsum_eq_zero_of_nonneg` → `hasSum_zero_iff_of_nonneg`, `innerProduct` → `dotProduct`, `IsPosDef'` → `Matrix.PosDef`.
4. **PMF expectation bridge promotion** (§4) — move `Exp_eq_integral` to `ExpectationTheory.lean`, add `Eg_eq_integral`/`Egu_eq_integral` + Fintype corollaries.
5. **Bayes kernel-posterior equivalence theorem** (§5) — one real theorem tying `bayesPosteriorPMF` to `ProbabilityTheory.posterior` via `posterior_eq_withDensity_of_countable`.
6. **BoundedSpace bridge instances** (§6) — instance + `diam_univ_le_diameterBound`; do NOT attempt the 30-file replacement.
7. **CLT doc map + upstream watch** (§8); assemble the upstream-PR shortlist: conditional Hoeffding bridge, `sigmoid_lipschitz`, `neg_log_sigmoid_lipschitz`, Lévy continuity, Helly selection.

## Notes / non-findings

- No live `axiom` declarations remain in TT `FormalProofs` (all `axiom` mentions are docstring history; `TechnicalAxioms.lean` documents 0 remaining unsound items).
- FP Econometrics (Wooldridge chapters, OLS/GMM/IV/Panel) and DSL regression files define only domain objects over mathlib `Matrix`/`Integrable`/`TendstoInMeasure` — no re-implemented linear algebra found beyond §3's `innerProduct`/`IsPosDef'`.
- Mathlib's `PMF.integral_eq_tsum` uses `•`; for `ℝ` this is `smul_eq_mul` — the existing bridge already handles it.
- Mathlib has no Bernstein/McDiarmid/Serfling concentration and no `logit`; anything in the repos depending on those is KEEP by necessity.

---

## Implemented 2026-07-02

Top actions (§1–§4 + §3 misc one-liners) implemented; both repos build green (`lake build` in FP: 8212 jobs OK; TT: all three targets — `FormalProofs`, `FormalProofsEconometrics`, `FormalProofsEconometricsSemiparametric` — OK). Zero errors, zero `sorry`, no new axioms. Crosswalk check: no name in `paper/ctreepo/appendix/v13_triangle/*.tex` references any changed/removed Lean name (rg clean).

1. **expit → `Real.sigmoid` (DONE, both repos).** TT `FormalProofs/DSL/CoreDefinitions.lean` and FP `FormalProbability/DSL/CoreDefinitions.lean`: `def expit` replaced by `abbrev expit := Real.sigmoid` + `theorem expit_eq_sigmoid : expit = Real.sigmoid := rfl`. `expit_range` reproved from `Real.sigmoid_pos`/`Real.sigmoid_lt_one`. All call sites (`LogisticRegression.lean`, `MomentFunctions.lean` in both repos) compile unchanged. `logit` kept custom (mathlib has none), docstring notes this.

2. **`hoeffding_iid_bounded_axiom` retired (DONE).** The `_axiom`-named backward-compat wrapper in TT `FormalProofs/OPT/MeasureTheoreticAudit.lean` was deleted outright (clean rename — the proved lemma `hoeffding_iid_bounded` already existed; rg found no reference in FP, in the paper appendix, or anywhere outside TT, so no deprecated alias was needed). `FormalProofs/TechnicalAxioms.lean` `hoeffding_inequality` abbrev now points at `@hoeffding_iid_bounded` and its docstring records the removal.

3. **Conditional-Hoeffding bridge deduped into FP (DONE).** TT `FormalProofs/OPT/SerflingAudit.lean` now imports `FormalProbability.DSL.SamplingConcentration`; the TT proof bodies of `OPT.hasCondSubgaussianMGF_of_mem_Icc_of_condExp_eq_zero` **and** the two duplicated Azuma wrappers (`OPT.azuma_hoeffding_of_mem_Icc_of_condExp_eq_zero`, `OPT.azuma_hoeffding_abs_of_mem_Icc_of_condExp_eq_zero`) were replaced by one-line re-exports of the `DSL.*` versions (TT names kept because `ExtendedExports.lean`/`NamespaceCompat.lean` re-export them). FP is the single home of the proofs.

4. **PMF integral bridge promoted (DONE).** `Exp_eq_integral` moved from `FormalProofs/OPT/MeasureTheoreticAudit.lean` into `FormalProofs/OPT/ExpectationTheory.lean` (new "Mathlib Integral Bridges" section next to the `Exp` definition; move is upward along the DAG — MeasureTheoreticAudit imports AuditBounds imports ExpectationTheory). Added `Eg_eq_integral`, `Egu_eq_integral` (both in ExpectationTheory.lean, which transitively imports `OPT/CoreDefinitions.lean` (`Eg`) and `OPT/LocalLaws.lean` (`Egu`)), plus `Fintype` corollaries with no integrability hypothesis via `PMF.integral_eq_sum`: `Exp_eq_integral_of_fintype`, `Eg_eq_integral_of_fintype`, `Egu_eq_integral_of_fintype`. Each carries the "mathlib bridge" docstring.

5. **Misc one-liners (DONE with one bridge instead of replace):**
   - `tsum_eq_zero_of_nonneg` (TT `OPT/ExpectationTheory.lean`): kept the repo name/statement (8 call-site files use the pointwise form), proof replaced by a two-line derivation from mathlib's `hasSum_zero_iff_of_nonneg`.
   - `innerProduct` (TT + FP `DSL/CoreDefinitions.lean`): now `abbrev innerProduct x β := dotProduct x β` + `innerProduct_eq_dotProduct := rfl`; `innerProduct_comm` reproved as `dotProduct_comm`. Drop-in — all downstream defs compile unchanged.
   - `IsPosDef'` (FP `Media/ElectionCoverageTV/Core/Lemmas/CholeskyUniqueness.lean`): NOT a drop-in replace — mathlib's `Matrix.PosDef` (Finsupp-quadratic-form definition in v4.27) additionally requires `IsHermitian`, which `IsPosDef'` lacks. Added bridge `isPosDef'_iff_posDef (hsym : Sigma.IsSymm) : IsPosDef' Sigma ↔ Sigma.PosDef` (via `Matrix.posDef_iff_dotProduct_mulVec` + `Matrix.conjTranspose_eq_transpose_of_trivial`) and corollary `CorrelationMatrix'.posDef`. `IsPosDef'` kept as the file-local primitive.

Not implemented in this pass (per plan ranking, deliberately deferred): §2(iii) shared two-sided `measure_abs_ge_le_of_iIndepFun` factoring, §5 Bayes kernel-posterior equivalence theorem, §6 `BoundedSpace` bridge instances, §8 CLT doc map / upstream-PR shortlist.
