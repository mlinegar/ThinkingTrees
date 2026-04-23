import FormalProofs.OPT.ScoreTransport
import FormalProofs.OPT.ExpectationTheory
import FormalProofs.OPT.AdaptiveChunkingBridge
import FormalProofs.OPT.BagOfWordsLDARecovery
import Mathlib.Probability.ProbabilityMassFunction.Constructions
import Mathlib.Probability.ProbabilityMassFunction.Integrals

/-!
# FormalProofs/OPT/InformationSufficiency.lean

This file adds a measure-theoretic information-sufficiency layer for C-TreePO.

The emphasis is deliberately narrow:

- oracle sufficiency / Doob-Dynkin factorization,
- zero task-relevant KLIC for oracle-indexed conditional densities,
- a tree-policy joint-law bridge from local laws to a.e. oracle
  factorization and score transport, for both deterministic and stochastic
  document-indexed tree schedulers.

It does NOT formalize full Shannon or mutual-information machinery; those remain
optional future extensions.
-/

set_option linter.mathlibStandardSet false

open scoped BigOperators
open scoped Real
open scoped Nat
open scoped Classical
open scoped Pointwise
open scoped MeasureTheory
open scoped ProbabilityTheory

set_option maxHeartbeats 0
set_option maxRecDepth 4000
set_option relaxedAutoImplicit false
set_option autoImplicit false

noncomputable section

namespace FormalProofs.OPT

open MeasureTheory ProbabilityTheory Set

section KLIC

variable {Strings Obs Y : Type*}
variable [PseudoMetricSpace Y]
variable [MeasurableSpace Obs]

/-- A conditional density is oracle-indexed when it factors through the oracle
value of the input. -/
def OracleIndexedConditionalDensity
    (p : Strings → Obs → ℝ) (fstar : Strings → Y) : Prop :=
  ∃ pbar : Y → Obs → ℝ, ∀ x y, p x y = pbar (fstar x) y

/-- Narrow task-relevant KLIC used by the oracle-sufficiency bridge. -/
def taskRelevantKLIC (μ : Measure Obs) (f g : Obs → ℝ) : ℝ :=
  ∫ y, Real.log (f y / g y) * f y ∂μ

/-- Pointwise equality of densities collapses the KLIC to zero. -/
theorem kullbackLeibler_zero_of_pointwise_eq
    (μ : Measure Obs) (f g : Obs → ℝ)
    (hfg : ∀ y, f y = g y) :
    taskRelevantKLIC μ f g = 0 := by
  unfold taskRelevantKLIC
  calc
    ∫ y, Real.log (f y / g y) * f y ∂μ = ∫ y, (0 : ℝ) ∂μ := by
      apply integral_congr_ae
      refine Filter.Eventually.of_forall ?_
      intro y
      change Real.log (f y / g y) * f y = 0
      by_cases hy : f y = 0
      · rw [hy]
        simp
      · have hdiv : f y / f y = (1 : ℝ) := by
          field_simp [hy]
        have hgy : g y = f y := (hfg y).symm
        rw [hgy, hdiv]
        simp
    _ = 0 := by simp

/-- If two inputs share the same oracle value, every oracle-indexed conditional
density induces the same task-relevant KLIC. -/
theorem taskRelevantKLIC_zero_of_oracleIndexedConditionalDensity_of_eq
    (μ : Measure Obs) (p : Strings → Obs → ℝ) (fstar : Strings → Y)
    (hidx : OracleIndexedConditionalDensity p fstar)
    {x x' : Strings}
    (hEq : fstar x = fstar x') :
    taskRelevantKLIC μ (p x) (p x') = 0 := by
  rcases hidx with ⟨pbar, hpbar⟩
  apply kullbackLeibler_zero_of_pointwise_eq
  intro y
  rw [hpbar x y, hpbar x' y, hEq]

/-- A.e. oracle equality implies zero task-relevant KLIC almost surely. -/
theorem taskRelevantKLIC_zero_ae_of_ae_oracle_eq
    {Ω : Type*} [MeasurableSpace Ω]
    (μΩ : Measure Ω)
    (μObs : Measure Obs)
    (X Z : Ω → Strings)
    (p : Strings → Obs → ℝ) (fstar : Strings → Y)
    (hidx : OracleIndexedConditionalDensity p fstar)
    (hEqAE : ∀ᵐ ω ∂μΩ, fstar (X ω) = fstar (Z ω)) :
    ∀ᵐ ω ∂μΩ, taskRelevantKLIC μObs (p (X ω)) (p (Z ω)) = 0 := by
  filter_upwards [hEqAE] with ω hω
  exact taskRelevantKLIC_zero_of_oracleIndexedConditionalDensity_of_eq
    (μ := μObs) (p := p) (fstar := fstar) hidx hω

/-- If a deterministic summary collides two oracle-distinct inputs, then no
decoder can recover the oracle from that summary. -/
theorem no_oracle_decoder_of_summary_collision
    {Summary : Type*}
    {summary : Strings → Summary} {fstar : Strings → Y}
    {x x' : Strings}
    (hSummary : summary x = summary x')
    (hOracle : fstar x ≠ fstar x') :
    ¬ ∃ recover : Summary → Y, fstar = recover ∘ summary := by
  intro hRecover
  rcases hRecover with ⟨recover, hRecover⟩
  apply hOracle
  calc
    fstar x = recover (summary x) := by simpa [Function.comp] using congrArg (fun h => h x) hRecover
    _ = recover (summary x') := by rw [hSummary]
    _ = fstar x' := by simpa [Function.comp] using (congrArg (fun h => h x') hRecover).symm

section Examples

variable {α : Type*}

/-- Positive example: the bag-of-words count sketch admits an explicit oracle
decoder, so the sufficiency layer is non-vacuous on an exact statistic lane. -/
theorem bagOfWords_countSketch_has_oracle_decoder
    (T : BinTree (List α)) :
    ∃ decode : List α → Multiset α,
      decode (sketchSummary (countSketchOperator (α := α)) T) = bagOfWords (S T) := by
  refine ⟨bagOfWords, ?_⟩
  exact bagOfWords_sketchSummary_countSketch (α := α) (T := T)

/-- Negative example: a constant summary cannot recover the identity oracle on
`ℕ`, so the impossibility surface is non-vacuous as well. -/
theorem constant_summary_cannot_recover_identity :
    ¬ ∃ recover : PUnit → ℕ, (fun n : ℕ => n) = recover ∘ fun _ => PUnit.unit := by
  intro hRecover
  rcases hRecover with ⟨recover, hRecover⟩
  have h0 : (0 : ℕ) = recover PUnit.unit := by
    simpa [Function.comp] using congrArg (fun h : ℕ → ℕ => h 0) hRecover
  have h1 : (1 : ℕ) = recover PUnit.unit := by
    simpa [Function.comp] using congrArg (fun h : ℕ → ℕ => h 1) hRecover
  have : (0 : ℕ) = 1 := by
    calc
      0 = recover PUnit.unit := h0
      _ = 1 := h1.symm
  exact Nat.zero_ne_one this

end Examples

end KLIC

section JointLaw

variable {Strings Y A Obs : Type*}
variable [Monoid Strings]
variable [MeasurableSpace Strings] [MeasurableSingletonClass Strings]
variable [BoundedMetricSpace Y] [MeasurableSpace Y] [BorelSpace Y]
variable [MeasurableSpace Obs]

/-- If the expected oracle distortion under a PMF is zero, then distortion is
zero at every support point of that PMF. Copied locally to avoid pulling the
heavier theorem-backing consequence stack into this module. -/
lemma dist_zero_on_support_of_Exp_zero_info
    (p : PMF Strings) (fstar : Strings → Y) (x : Strings)
    (h_exp_zero : Exp p (fun z => D fstar z x) = 0) :
    ∀ z ∈ p.support, dist (fstar z) (fstar x) = 0 := by
  let M : ℝ := BoundedMetricSpace.diameterBound (α := Y)
  have hM : 0 ≤ M := BoundedMetricSpace.diameterBound_nonneg (α := Y)
  have hbound : ∀ z, D fstar z x ≤ M := by
    intro z
    simpa [M, D] using (BoundedMetricSpace.dist_le (fstar z) (fstar x))
  have h_summable : Summable (fun z => (p z).toReal * D fstar z x) :=
    summable_D_of_bounded p fstar x M hM hbound
  have h_term_zero : ∀ z, (p z).toReal * D fstar z x = 0 :=
    tsum_eq_zero_of_nonneg
      (fun z => (p z).toReal * D fstar z x)
      (fun z => mul_nonneg ENNReal.toReal_nonneg dist_nonneg)
      h_summable
      (by simpa [Exp] using h_exp_zero)
  intro z hz
  have hz_ne0 : p z ≠ 0 := by
    simpa [PMF.mem_support_iff] using hz
  have hz_toReal_pos : 0 < (p z).toReal :=
    ENNReal.toReal_pos hz_ne0 (PMF.apply_ne_top p z)
  have hz_mul : (p z).toReal * D fstar z x = 0 := h_term_zero z
  rcases mul_eq_zero.mp hz_mul with hz_toReal | hz_dist
  · exfalso
    exact (ne_of_gt hz_toReal_pos) hz_toReal
  · simpa [D] using hz_dist

/-- Joint law of a raw document and one realized tree summary produced from it
under a deterministic document-indexed tree policy. -/
def jointTreeSummaryLaw
    (μDoc : PMF Strings)
    (treeOf : Strings → BinTree Strings)
    (g : Summarizer Strings) (R : ℕ) : PMF (Strings × Strings) :=
  μDoc.bind fun x => (ZR g x R (treeOf x)).map fun z => (x, z)

/-- Pair-level view of a raw-document score. -/
def rawScoreOnJoint
    (Sstar : Strings → A → ℝ) : (Strings × Strings) → A → ℝ :=
  fun ω a => Sstar ω.1 a

/-- Pair-level view of the raw-document oracle. -/
def rawOracleOnJoint
    (fstar : Strings → Y) : (Strings × Strings) → Y :=
  fun ω => fstar ω.1

/-- Support characterization for `jointTreeSummaryLaw`. -/
theorem mem_support_jointTreeSummaryLaw_iff
    (μDoc : PMF Strings)
    (treeOf : Strings → BinTree Strings)
    (g : Summarizer Strings) (R : ℕ)
    (ω : Strings × Strings) :
    ω ∈ (jointTreeSummaryLaw μDoc treeOf g R).support ↔
      ω.1 ∈ μDoc.support ∧ ω.2 ∈ (ZR g ω.1 R (treeOf ω.1)).support := by
  rw [jointTreeSummaryLaw, PMF.mem_support_bind_iff]
  constructor
  · rintro ⟨x, hx, hmap⟩
    rw [PMF.mem_support_map_iff] at hmap
    rcases hmap with ⟨z, hz, hzEq⟩
    have hxEq : x = ω.1 := by
      simpa using congrArg Prod.fst hzEq
    have hzEq' : z = ω.2 := by
      simpa using congrArg Prod.snd hzEq
    subst hxEq
    subst hzEq'
    exact ⟨hx, hz⟩
  · rintro ⟨hx, hz⟩
    refine ⟨ω.1, hx, ?_⟩
    rw [PMF.mem_support_map_iff]
    exact ⟨ω.2, hz, rfl⟩

/-- A supportwise property for a PMF holds almost surely under `toMeasure`. -/
lemma ae_of_forall_support_pmf
    (p : PMF (Strings × Strings)) {P : (Strings × Strings) → Prop}
    (hP : ∀ ω ∈ p.support, P ω) :
    ∀ᵐ ω ∂p.toMeasure, P ω := by
  rw [MeasureTheory.ae_iff]
  rw [PMF.toMeasure_apply_eq_toOuterMeasure, PMF.toOuterMeasure_apply_eq_zero_iff]
  refine Set.disjoint_left.2 ?_
  intro ω hω hbad
  exact hbad (hP ω hω)

/-- Under local laws on every realized deterministic-policy tree, the joint law only
places mass on raw/summary pairs with zero oracle distortion. -/
theorem jointTreeSummaryLaw_oracle_dist_zero_on_support_of_localLaws
    (μDoc : PMF Strings)
    (treeOf : Strings → BinTree Strings)
    (g : Summarizer Strings) (R : ℕ)
    (fstar : Strings → Y)
    (hTree : ∀ x, S (treeOf x) = x)
    (h1 : ∀ x, L1 g (treeOf x) fstar)
    (h2 : ∀ x, L2 g (treeOf x) fstar)
    (h3 : L3 g fstar)
    (hR : R ≥ 1) :
    ∀ ω ∈ (jointTreeSummaryLaw μDoc treeOf g R).support,
      dist (fstar ω.2) (fstar ω.1) = 0 := by
  intro ω hω
  rcases (mem_support_jointTreeSummaryLaw_iff μDoc treeOf g R ω).1 hω with ⟨_, hz⟩
  have hExpZero :
      Exp (ZR g ω.1 R (treeOf ω.1)) (fun z => D fstar z ω.1) = 0 :=
    multi_round_typeclass g (treeOf ω.1) ω.1 R fstar
      (hTree ω.1) (h1 ω.1) (h2 ω.1) h3 hR
  exact dist_zero_on_support_of_Exp_zero_info
    (p := ZR g ω.1 R (treeOf ω.1)) (fstar := fstar) (x := ω.1) hExpZero ω.2 hz

/-- Deterministic tree-policy local laws imply almost-sure oracle equality under the
joint raw/summary law. -/
theorem jointTreeSummaryLaw_oracle_eq_ae_of_localLaws
    (μDoc : PMF Strings)
    (treeOf : Strings → BinTree Strings)
    (g : Summarizer Strings) (R : ℕ)
    (fstar : Strings → Y)
    (hTree : ∀ x, S (treeOf x) = x)
    (h1 : ∀ x, L1 g (treeOf x) fstar)
    (h2 : ∀ x, L2 g (treeOf x) fstar)
    (h3 : L3 g fstar)
    (hR : R ≥ 1) :
    ∀ᵐ ω ∂(jointTreeSummaryLaw μDoc treeOf g R).toMeasure, fstar ω.1 = fstar ω.2 := by
  apply ae_of_forall_support_pmf
  intro ω hω
  exact (dist_eq_zero.mp <|
    jointTreeSummaryLaw_oracle_dist_zero_on_support_of_localLaws
      μDoc treeOf g R fstar hTree h1 h2 h3 hR ω hω).symm

/-- The raw-document oracle factors almost surely through the realized summary
under the joint raw/summary law. -/
theorem jointTreeSummaryLaw_oracle_factorizationAE_of_localLaws
    (μDoc : PMF Strings)
    (treeOf : Strings → BinTree Strings)
    (g : Summarizer Strings) (R : ℕ)
    (fstar : Strings → Y)
    (hfstar : Measurable fstar)
    (hTree : ∀ x, S (treeOf x) = x)
    (h1 : ∀ x, L1 g (treeOf x) fstar)
    (h2 : ∀ x, L2 g (treeOf x) fstar)
    (h3 : L3 g fstar)
    (hR : R ≥ 1) :
    OracleFactorizationAE' fstar Prod.fst Prod.snd
      (jointTreeSummaryLaw μDoc treeOf g R).toMeasure := by
  refine ⟨fstar, hfstar, ?_⟩
  filter_upwards
    [jointTreeSummaryLaw_oracle_eq_ae_of_localLaws
      μDoc treeOf g R fstar hTree h1 h2 h3 hR] with ω hω
  simpa [Function.comp] using hω

/-- Conditional score on the raw/summary joint law, conditioned on the realized
summary string. -/
def jointSummaryScore
    (μJoint : Measure (Strings × Strings))
    (Sstar : Strings → A → ℝ) (a : A) : (Strings × Strings) → ℝ :=
  μJoint[fun ω => Sstar ω.1 a | MeasurableSpace.comap Prod.snd ‹_›]

/-- If the raw-document score factors through the oracle after conditioning on
the raw document itself, then the raw score agrees almost surely with the
oracle-factored score on the joint law. -/
theorem rawScoreOnJoint_eq_factored_ae_of_conditionalFactorization
    (μJoint : Measure (Strings × Strings))
    (Sstar : Strings → A → ℝ) (fstar : Strings → Y)
    (hSstar_meas : ∀ a, Measurable (fun x => Sstar x a))
    (hCFraw : ConditionalFactorization' Sstar fstar Prod.fst μJoint measurable_fst)
    [_hμFst : SigmaFinite (μJoint.trim measurable_fst.comap_le)]
    (a : A)
    (hint : Integrable (fun ω : Strings × Strings => Sstar ω.1 a) μJoint) :
    (fun ω : Strings × Strings => Sstar ω.1 a) =ᵐ[μJoint]
      fun ω : Strings × Strings => hCFraw.choose (fstar ω.1) a := by
  have hRawMeasFst :
      StronglyMeasurable[MeasurableSpace.comap Prod.fst ‹_›]
        (fun ω : Strings × Strings => Sstar ω.1 a) := by
    apply Measurable.stronglyMeasurable
    have hfst_meas :
        Measurable[MeasurableSpace.comap Prod.fst ‹_›] (Prod.fst : Strings × Strings → Strings) := by
      simpa using (comap_measurable (f := Prod.fst))
    exact (hSstar_meas a).comp hfst_meas
  have hRawSelfFst :
      μJoint[fun ω : Strings × Strings => Sstar ω.1 a | MeasurableSpace.comap Prod.fst ‹_›]
        = fun ω : Strings × Strings => Sstar ω.1 a :=
    condExp_of_stronglyMeasurable measurable_fst.comap_le hRawMeasFst hint
  calc
    (fun ω : Strings × Strings => Sstar ω.1 a)
        =ᵐ[μJoint]
          μJoint[fun ω : Strings × Strings => Sstar ω.1 a | MeasurableSpace.comap Prod.fst ‹_›] := by
            exact Filter.EventuallyEq.of_eq hRawSelfFst.symm
    _ =ᵐ[μJoint] fun ω : Strings × Strings => hCFraw.choose (fstar ω.1) a :=
      hCFraw.choose_spec.2 a

/-- A.e. oracle equality on the raw/summary joint law lets us replace the
oracle-factored raw score with the oracle-factored summary score. -/
theorem factoredRaw_eq_factoredSummary_ae_of_oracle_eq
    (μJoint : Measure (Strings × Strings))
    (fstar : Strings → Y)
    (Qbar : Y → A → ℝ)
    (a : A)
    (hEqAE : ∀ᵐ ω ∂μJoint, fstar ω.1 = fstar ω.2) :
    (fun ω : Strings × Strings => Qbar (fstar ω.1) a) =ᵐ[μJoint]
      fun ω : Strings × Strings => Qbar (fstar ω.2) a := by
  filter_upwards [hEqAE] with ω hω
  exact congrArg (fun y => Qbar y a) hω

/-- The oracle-factored summary score is measurable with respect to the
summary σ-algebra on the joint raw/summary law. -/
theorem factoredSummary_stronglyMeasurable
    (fstar : Strings → Y)
    (hfstar : Measurable fstar)
    (Qbar : Y → A → ℝ)
    (hQbar_meas : ∀ a, Measurable (fun y => Qbar y a))
    (a : A) :
    StronglyMeasurable[MeasurableSpace.comap Prod.snd ‹_›]
      (fun ω : Strings × Strings => Qbar (fstar ω.2) a) := by
  apply Measurable.stronglyMeasurable
  have hsnd_meas :
      Measurable[MeasurableSpace.comap Prod.snd ‹_›] (Prod.snd : Strings × Strings → Strings) := by
    simpa using (comap_measurable (f := Prod.snd))
  exact (hQbar_meas a).comp (hfstar.comp hsnd_meas)

/-- If `f = g` almost surely and `g` is measurable with respect to the
conditioning σ-algebra, then conditioning `f` on that σ-algebra recovers `g`. -/
theorem condExp_eq_of_ae_eq_stronglyMeasurable
    {Ω : Type*} {m₀ m : MeasurableSpace Ω}
    (μ : @Measure Ω m₀)
    (hm : m ≤ m₀)
    [_hμm : SigmaFinite (μ.trim hm)]
    (f g : Ω → ℝ)
    (hfg : f =ᵐ[μ] g)
    (hg_meas : StronglyMeasurable[m] g)
    (hg_int : Integrable g μ) :
    μ[f | m] =ᵐ[μ] g := by
  calc
    μ[f | m] =ᵐ[μ] μ[g | m] := condExp_congr_ae hfg
    _ =ᵐ[μ] g := Filter.EventuallyEq.of_eq <|
      condExp_of_stronglyMeasurable hm hg_meas hg_int

/-- Under a raw-document conditional factorization assumption, deterministic
tree-policy
local laws imply that conditioning on the realized summary recovers the same
oracle-factored score almost surely on the joint raw/summary law. -/
theorem jointTreeSummaryLaw_score_factorization_of_localLaws
    (μDoc : PMF Strings)
    (treeOf : Strings → BinTree Strings)
    (g : Summarizer Strings) (R : ℕ)
    (Sstar : Strings → A → ℝ) (fstar : Strings → Y)
    (hfstar : Measurable fstar)
    (hSstar_meas : ∀ a, Measurable (fun x => Sstar x a))
    (hTree : ∀ x, S (treeOf x) = x)
    (h1 : ∀ x, L1 g (treeOf x) fstar)
    (h2 : ∀ x, L2 g (treeOf x) fstar)
    (h3 : L3 g fstar)
    (hR : R ≥ 1)
    (hCFraw :
      ConditionalFactorization' Sstar fstar Prod.fst
        (jointTreeSummaryLaw μDoc treeOf g R).toMeasure measurable_fst)
    (a : A)
    (hint :
      Integrable (fun ω : Strings × Strings => Sstar ω.1 a)
        (jointTreeSummaryLaw μDoc treeOf g R).toMeasure) :
    jointSummaryScore
      (jointTreeSummaryLaw μDoc treeOf g R).toMeasure
      Sstar a
      =ᵐ[(jointTreeSummaryLaw μDoc treeOf g R).toMeasure]
      fun ω => hCFraw.choose (fstar ω.1) a := by
  let μJoint := (jointTreeSummaryLaw μDoc treeOf g R).toMeasure
  letI : SigmaFinite (μJoint.trim measurable_fst.comap_le) := by
    infer_instance
  letI : SigmaFinite (μJoint.trim measurable_snd.comap_le) := by
    infer_instance
  let qraw : (Strings × Strings) → ℝ := fun ω => hCFraw.choose (fstar ω.1) a
  let qsum : (Strings × Strings) → ℝ := fun ω => hCFraw.choose (fstar (Prod.snd ω)) a
  have hRawEqQraw :
      (fun ω : Strings × Strings => Sstar ω.1 a) =ᵐ[μJoint] qraw := by
    simpa [qraw] using rawScoreOnJoint_eq_factored_ae_of_conditionalFactorization
      (μJoint := μJoint) (Sstar := Sstar) (fstar := fstar)
      hSstar_meas hCFraw a hint
  have hQrawEqQsum :
      qraw =ᵐ[μJoint] qsum := by
    simpa [qraw, qsum] using factoredRaw_eq_factoredSummary_ae_of_oracle_eq
      (μJoint := μJoint) (fstar := fstar) (Qbar := hCFraw.choose) (a := a)
      (jointTreeSummaryLaw_oracle_eq_ae_of_localLaws
        μDoc treeOf g R fstar hTree h1 h2 h3 hR)
  have hRawEqQsum :
      (fun ω : Strings × Strings => Sstar ω.1 a) =ᵐ[μJoint] qsum :=
    hRawEqQraw.trans hQrawEqQsum
  have hQsumMeas :
      StronglyMeasurable[MeasurableSpace.comap Prod.snd ‹_›] qsum := by
    simpa [qsum] using factoredSummary_stronglyMeasurable
      (fstar := fstar) (hfstar := hfstar) (Qbar := hCFraw.choose)
      (hQbar_meas := hCFraw.choose_spec.1) (a := a)
  have hQsumInt : Integrable qsum μJoint := by
    exact (integrable_congr hRawEqQsum).mp hint
  have hCondEqQsum :
      μJoint[fun ω : Strings × Strings => Sstar ω.1 a | MeasurableSpace.comap Prod.snd ‹_›]
        =ᵐ[μJoint] qsum := by
    exact condExp_eq_of_ae_eq_stronglyMeasurable
      μJoint measurable_snd.comap_le
      (fun ω : Strings × Strings => Sstar ω.1 a) qsum
      hRawEqQsum hQsumMeas hQsumInt
  unfold jointSummaryScore
  exact hCondEqQsum.trans hQrawEqQsum.symm

/-- Integral version of `jointTreeSummaryLaw_score_factorization_of_localLaws`. -/
theorem jointTreeSummaryLaw_score_transport_of_localLaws
    (μDoc : PMF Strings)
    (treeOf : Strings → BinTree Strings)
    (g : Summarizer Strings) (R : ℕ)
    (Sstar : Strings → A → ℝ) (fstar : Strings → Y)
    (hfstar : Measurable fstar)
    (hSstar_meas : ∀ a, Measurable (fun x => Sstar x a))
    (hTree : ∀ x, S (treeOf x) = x)
    (h1 : ∀ x, L1 g (treeOf x) fstar)
    (h2 : ∀ x, L2 g (treeOf x) fstar)
    (h3 : L3 g fstar)
    (hR : R ≥ 1)
    (hCFraw :
      ConditionalFactorization' Sstar fstar Prod.fst
        (jointTreeSummaryLaw μDoc treeOf g R).toMeasure measurable_fst)
    (a : A)
    (hint :
      Integrable (fun ω : Strings × Strings => Sstar ω.1 a)
        (jointTreeSummaryLaw μDoc treeOf g R).toMeasure) :
    ∫ ω, Sstar ω.1 a ∂(jointTreeSummaryLaw μDoc treeOf g R).toMeasure =
      ∫ ω, jointSummaryScore
        (jointTreeSummaryLaw μDoc treeOf g R).toMeasure
        Sstar a ω ∂(jointTreeSummaryLaw μDoc treeOf g R).toMeasure ∧
    ∫ ω, Sstar ω.1 a ∂(jointTreeSummaryLaw μDoc treeOf g R).toMeasure =
      ∫ ω, hCFraw.choose (fstar ω.1) a ∂(jointTreeSummaryLaw μDoc treeOf g R).toMeasure := by
  constructor
  · unfold jointSummaryScore
    exact (MeasureTheory.integral_condExp measurable_snd.comap_le).symm
  · calc
      ∫ ω, Sstar ω.1 a ∂(jointTreeSummaryLaw μDoc treeOf g R).toMeasure =
        ∫ ω, jointSummaryScore
          (jointTreeSummaryLaw μDoc treeOf g R).toMeasure
          Sstar a ω ∂(jointTreeSummaryLaw μDoc treeOf g R).toMeasure := by
            unfold jointSummaryScore
            exact (MeasureTheory.integral_condExp measurable_snd.comap_le).symm
      _ = ∫ ω, hCFraw.choose (fstar ω.1) a ∂(jointTreeSummaryLaw μDoc treeOf g R).toMeasure := by
          apply integral_congr_ae
          exact jointTreeSummaryLaw_score_factorization_of_localLaws
            μDoc treeOf g R Sstar fstar hfstar hSstar_meas
            hTree h1 h2 h3 hR hCFraw a hint

/-- Deterministic tree-policy local laws imply zero task-relevant KLIC almost surely under
the joint raw/summary law. -/
theorem jointTreeSummaryLaw_taskRelevantKLIC_zero_ae_of_localLaws
    (μDoc : PMF Strings)
    (treeOf : Strings → BinTree Strings)
    (g : Summarizer Strings) (R : ℕ)
    (fstar : Strings → Y)
    (hTree : ∀ x, S (treeOf x) = x)
    (h1 : ∀ x, L1 g (treeOf x) fstar)
    (h2 : ∀ x, L2 g (treeOf x) fstar)
    (h3 : L3 g fstar)
    (hR : R ≥ 1)
    (μObs : Measure Obs)
    (p : Strings → Obs → ℝ)
    (hidx : OracleIndexedConditionalDensity p fstar) :
    ∀ᵐ ω ∂(jointTreeSummaryLaw μDoc treeOf g R).toMeasure,
      taskRelevantKLIC μObs (p ω.1) (p ω.2) = 0 := by
  exact taskRelevantKLIC_zero_ae_of_ae_oracle_eq
    (μΩ := (jointTreeSummaryLaw μDoc treeOf g R).toMeasure)
    (μObs := μObs)
    (X := Prod.fst) (Z := Prod.snd)
    (p := p) (fstar := fstar) hidx
    (jointTreeSummaryLaw_oracle_eq_ae_of_localLaws
      μDoc treeOf g R fstar hTree h1 h2 h3 hR)

/-- Joint law of a raw document and one realized tree summary produced from it
under a stochastic document-indexed tree policy. The tree itself is sampled
first, then the multi-round summary is sampled from that realized tree. -/
def stochasticJointTreeSummaryLaw
    (μDoc : PMF Strings)
    (τ : StochasticAdaptiveTreeMap Strings)
    (g : Summarizer Strings) (R : ℕ) : PMF (Strings × Strings) :=
  μDoc.bind fun x =>
    (τ x).bind fun T =>
      (ZR g x R T).map fun z => (x, z)

/-- Support characterization for `stochasticJointTreeSummaryLaw`. -/
theorem mem_support_stochasticJointTreeSummaryLaw_iff
    (μDoc : PMF Strings)
    (τ : StochasticAdaptiveTreeMap Strings)
    (g : Summarizer Strings) (R : ℕ)
    (ω : Strings × Strings) :
    ω ∈ (stochasticJointTreeSummaryLaw μDoc τ g R).support ↔
      ω.1 ∈ μDoc.support ∧
        ∃ T, T ∈ (τ ω.1).support ∧ ω.2 ∈ (ZR g ω.1 R T).support := by
  rw [stochasticJointTreeSummaryLaw, PMF.mem_support_bind_iff]
  constructor
  · rintro ⟨x, hx, hbind⟩
    rw [PMF.mem_support_bind_iff] at hbind
    rcases hbind with ⟨T, hT, hmap⟩
    rw [PMF.mem_support_map_iff] at hmap
    rcases hmap with ⟨z, hz, hzEq⟩
    have hxEq : x = ω.1 := by
      simpa using congrArg Prod.fst hzEq
    have hzEq' : z = ω.2 := by
      simpa using congrArg Prod.snd hzEq
    subst hxEq
    subst hzEq'
    exact ⟨hx, T, hT, hz⟩
  · rintro ⟨hx, T, hT, hz⟩
    refine ⟨ω.1, hx, ?_⟩
    rw [PMF.mem_support_bind_iff]
    refine ⟨T, hT, ?_⟩
    rw [PMF.mem_support_map_iff]
    exact ⟨ω.2, hz, rfl⟩

/-- Under supportwise local laws on a stochastic tree policy, the joint law only
places mass on raw/summary pairs with zero oracle distortion. -/
theorem stochasticJointTreeSummaryLaw_oracle_dist_zero_on_support_of_localLaws
    (μDoc : PMF Strings)
    (τ : StochasticAdaptiveTreeMap Strings)
    (g : Summarizer Strings) (R : ℕ)
    (fstar : Strings → Y)
    (hTree : StochasticAdaptiveChunkingSound τ)
    (hLaws : StochasticAdaptiveLocalLaws (g := g) (fstar := fstar) τ)
    (hR : R ≥ 1) :
    ∀ ω ∈ (stochasticJointTreeSummaryLaw μDoc τ g R).support,
      dist (fstar ω.2) (fstar ω.1) = 0 := by
  intro ω hω
  rcases (mem_support_stochasticJointTreeSummaryLaw_iff μDoc τ g R ω).1 hω with
    ⟨_, T, hT, hz⟩
  have hExpZero :
      Exp (ZR g ω.1 R T) (fun z => D fstar z ω.1) = 0 :=
    multi_round_typeclass_of_stochastic_adaptive g fstar τ hTree hLaws ω.1 R hR T hT
  exact dist_zero_on_support_of_Exp_zero_info
    (p := ZR g ω.1 R T) (fstar := fstar) (x := ω.1) hExpZero ω.2 hz

/-- Supportwise local laws on a stochastic tree policy imply almost-sure oracle
equality under the induced raw/summary joint law. -/
theorem stochasticJointTreeSummaryLaw_oracle_eq_ae_of_localLaws
    (μDoc : PMF Strings)
    (τ : StochasticAdaptiveTreeMap Strings)
    (g : Summarizer Strings) (R : ℕ)
    (fstar : Strings → Y)
    (hTree : StochasticAdaptiveChunkingSound τ)
    (hLaws : StochasticAdaptiveLocalLaws (g := g) (fstar := fstar) τ)
    (hR : R ≥ 1) :
    ∀ᵐ ω ∂(stochasticJointTreeSummaryLaw μDoc τ g R).toMeasure, fstar ω.1 = fstar ω.2 := by
  apply ae_of_forall_support_pmf
  intro ω hω
  exact (dist_eq_zero.mp <|
    stochasticJointTreeSummaryLaw_oracle_dist_zero_on_support_of_localLaws
      μDoc τ g R fstar hTree hLaws hR ω hω).symm

/-- Under supportwise local laws on a stochastic tree policy, the raw-document
oracle factors almost surely through the realized summary under the induced
joint law. -/
theorem stochasticJointTreeSummaryLaw_oracle_factorizationAE_of_localLaws
    (μDoc : PMF Strings)
    (τ : StochasticAdaptiveTreeMap Strings)
    (g : Summarizer Strings) (R : ℕ)
    (fstar : Strings → Y)
    (hfstar : Measurable fstar)
    (hTree : StochasticAdaptiveChunkingSound τ)
    (hLaws : StochasticAdaptiveLocalLaws (g := g) (fstar := fstar) τ)
    (hR : R ≥ 1) :
    OracleFactorizationAE' fstar Prod.fst Prod.snd
      (stochasticJointTreeSummaryLaw μDoc τ g R).toMeasure := by
  refine ⟨fstar, hfstar, ?_⟩
  filter_upwards
    [stochasticJointTreeSummaryLaw_oracle_eq_ae_of_localLaws
      μDoc τ g R fstar hTree hLaws hR] with ω hω
  simpa [Function.comp] using hω

/-- Under a raw-document conditional factorization assumption, supportwise
local laws on a stochastic tree policy imply that conditioning on the realized
summary recovers the same oracle-factored score almost surely on the induced
joint law. -/
theorem stochasticJointTreeSummaryLaw_score_factorization_of_localLaws
    (μDoc : PMF Strings)
    (τ : StochasticAdaptiveTreeMap Strings)
    (g : Summarizer Strings) (R : ℕ)
    (Sstar : Strings → A → ℝ) (fstar : Strings → Y)
    (hfstar : Measurable fstar)
    (hSstar_meas : ∀ a, Measurable (fun x => Sstar x a))
    (hTree : StochasticAdaptiveChunkingSound τ)
    (hLaws : StochasticAdaptiveLocalLaws (g := g) (fstar := fstar) τ)
    (hR : R ≥ 1)
    (hCFraw :
      ConditionalFactorization' Sstar fstar Prod.fst
        (stochasticJointTreeSummaryLaw μDoc τ g R).toMeasure measurable_fst)
    (a : A)
    (hint :
      Integrable (fun ω : Strings × Strings => Sstar ω.1 a)
        (stochasticJointTreeSummaryLaw μDoc τ g R).toMeasure) :
    jointSummaryScore
      (stochasticJointTreeSummaryLaw μDoc τ g R).toMeasure
      Sstar a
      =ᵐ[(stochasticJointTreeSummaryLaw μDoc τ g R).toMeasure]
      fun ω => hCFraw.choose (fstar ω.1) a := by
  let μJoint := (stochasticJointTreeSummaryLaw μDoc τ g R).toMeasure
  letI : SigmaFinite (μJoint.trim measurable_fst.comap_le) := by
    infer_instance
  letI : SigmaFinite (μJoint.trim measurable_snd.comap_le) := by
    infer_instance
  let qraw : (Strings × Strings) → ℝ := fun ω => hCFraw.choose (fstar ω.1) a
  let qsum : (Strings × Strings) → ℝ := fun ω => hCFraw.choose (fstar (Prod.snd ω)) a
  have hRawEqQraw :
      (fun ω : Strings × Strings => Sstar ω.1 a) =ᵐ[μJoint] qraw := by
    simpa [qraw] using rawScoreOnJoint_eq_factored_ae_of_conditionalFactorization
      (μJoint := μJoint) (Sstar := Sstar) (fstar := fstar)
      hSstar_meas hCFraw a hint
  have hQrawEqQsum :
      qraw =ᵐ[μJoint] qsum := by
    simpa [qraw, qsum] using factoredRaw_eq_factoredSummary_ae_of_oracle_eq
      (μJoint := μJoint) (fstar := fstar) (Qbar := hCFraw.choose) (a := a)
      (stochasticJointTreeSummaryLaw_oracle_eq_ae_of_localLaws
        μDoc τ g R fstar hTree hLaws hR)
  have hRawEqQsum :
      (fun ω : Strings × Strings => Sstar ω.1 a) =ᵐ[μJoint] qsum :=
    hRawEqQraw.trans hQrawEqQsum
  have hQsumMeas :
      StronglyMeasurable[MeasurableSpace.comap Prod.snd ‹_›] qsum := by
    simpa [qsum] using factoredSummary_stronglyMeasurable
      (fstar := fstar) (hfstar := hfstar) (Qbar := hCFraw.choose)
      (hQbar_meas := hCFraw.choose_spec.1) (a := a)
  have hQsumInt : Integrable qsum μJoint := by
    exact (integrable_congr hRawEqQsum).mp hint
  have hCondEqQsum :
      μJoint[fun ω : Strings × Strings => Sstar ω.1 a | MeasurableSpace.comap Prod.snd ‹_›]
        =ᵐ[μJoint] qsum := by
    exact condExp_eq_of_ae_eq_stronglyMeasurable
      μJoint measurable_snd.comap_le
      (fun ω : Strings × Strings => Sstar ω.1 a) qsum
      hRawEqQsum hQsumMeas hQsumInt
  unfold jointSummaryScore
  exact hCondEqQsum.trans hQrawEqQsum.symm

/-- Integral version of
`stochasticJointTreeSummaryLaw_score_factorization_of_localLaws`. -/
theorem stochasticJointTreeSummaryLaw_score_transport_of_localLaws
    (μDoc : PMF Strings)
    (τ : StochasticAdaptiveTreeMap Strings)
    (g : Summarizer Strings) (R : ℕ)
    (Sstar : Strings → A → ℝ) (fstar : Strings → Y)
    (hfstar : Measurable fstar)
    (hSstar_meas : ∀ a, Measurable (fun x => Sstar x a))
    (hTree : StochasticAdaptiveChunkingSound τ)
    (hLaws : StochasticAdaptiveLocalLaws (g := g) (fstar := fstar) τ)
    (hR : R ≥ 1)
    (hCFraw :
      ConditionalFactorization' Sstar fstar Prod.fst
        (stochasticJointTreeSummaryLaw μDoc τ g R).toMeasure measurable_fst)
    (a : A)
    (hint :
      Integrable (fun ω : Strings × Strings => Sstar ω.1 a)
        (stochasticJointTreeSummaryLaw μDoc τ g R).toMeasure) :
    ∫ ω, Sstar ω.1 a ∂(stochasticJointTreeSummaryLaw μDoc τ g R).toMeasure =
      ∫ ω, jointSummaryScore
        (stochasticJointTreeSummaryLaw μDoc τ g R).toMeasure
        Sstar a ω ∂(stochasticJointTreeSummaryLaw μDoc τ g R).toMeasure ∧
    ∫ ω, Sstar ω.1 a ∂(stochasticJointTreeSummaryLaw μDoc τ g R).toMeasure =
      ∫ ω, hCFraw.choose (fstar ω.1) a
        ∂(stochasticJointTreeSummaryLaw μDoc τ g R).toMeasure := by
  constructor
  · unfold jointSummaryScore
    exact (MeasureTheory.integral_condExp measurable_snd.comap_le).symm
  · calc
      ∫ ω, Sstar ω.1 a ∂(stochasticJointTreeSummaryLaw μDoc τ g R).toMeasure =
        ∫ ω, jointSummaryScore
          (stochasticJointTreeSummaryLaw μDoc τ g R).toMeasure
          Sstar a ω ∂(stochasticJointTreeSummaryLaw μDoc τ g R).toMeasure := by
            unfold jointSummaryScore
            exact (MeasureTheory.integral_condExp measurable_snd.comap_le).symm
      _ = ∫ ω, hCFraw.choose (fstar ω.1) a
          ∂(stochasticJointTreeSummaryLaw μDoc τ g R).toMeasure := by
            apply integral_congr_ae
            exact stochasticJointTreeSummaryLaw_score_factorization_of_localLaws
              μDoc τ g R Sstar fstar hfstar hSstar_meas
              hTree hLaws hR hCFraw a hint

/-- Supportwise local laws on a stochastic tree policy imply zero task-relevant
KLIC almost surely under the induced raw/summary joint law. -/
theorem stochasticJointTreeSummaryLaw_taskRelevantKLIC_zero_ae_of_localLaws
    (μDoc : PMF Strings)
    (τ : StochasticAdaptiveTreeMap Strings)
    (g : Summarizer Strings) (R : ℕ)
    (fstar : Strings → Y)
    (hTree : StochasticAdaptiveChunkingSound τ)
    (hLaws : StochasticAdaptiveLocalLaws (g := g) (fstar := fstar) τ)
    (hR : R ≥ 1)
    (μObs : Measure Obs)
    (p : Strings → Obs → ℝ)
    (hidx : OracleIndexedConditionalDensity p fstar) :
    ∀ᵐ ω ∂(stochasticJointTreeSummaryLaw μDoc τ g R).toMeasure,
      taskRelevantKLIC μObs (p ω.1) (p ω.2) = 0 := by
  exact taskRelevantKLIC_zero_ae_of_ae_oracle_eq
    (μΩ := (stochasticJointTreeSummaryLaw μDoc τ g R).toMeasure)
    (μObs := μObs)
    (X := Prod.fst) (Z := Prod.snd)
    (p := p) (fstar := fstar) hidx
    (stochasticJointTreeSummaryLaw_oracle_eq_ae_of_localLaws
      μDoc τ g R fstar hTree hLaws hR)

end JointLaw

end FormalProofs.OPT
