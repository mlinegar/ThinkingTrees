import FormalProofs.DSL.TreePOEndToEnd

/-!
# FormalProofs/DSL/TreePOEndToEndGlue.lean

Glue between the two halves of the paper's audited preference-gap bound
(Paper: thm:e2e, proof in app:proof-e2e):

1. The concrete method certificates in `DSL/TreePOEndToEnd.lean`
   (`dpo_treepo_end_to_end_certificate` and GRPO variants) prove HT
   unbiasedness over `bernoulliProductMeasure` and bound the full-document vs
   tree objective gap by `C_meth` times the **expected** Bernoulli
   Horvitz-Thompson distortion estimator.
2. The abstract stack (`PaperErrorCertificate` / `PaperErrorStack`) performs
   the triangle-inequality + union-bound bookkeeping (Paper: thm:e2e parts
   (c)/(d)) over abstract random variables.

Previously these halves were never composed (audit finding F7). This file
composes them on the concrete Bernoulli product sampling space:

* `dpo_treepo_realized_estimator_certificate` /
  `grpo_pl_treepo_realized_estimator_certificate` /
  `grpo_rl_treepo_realized_estimator_certificate`: the paper's displayed
  realized-estimator bound. With probability at least `1 - δ_est(t)` under
  the sampling design, the objective gap is bounded by
  `C_meth * (realized HT distortion estimate + t)`, with the explicit
  failure probability
  `δ_est(t) = 2 * exp(-t² / (8 * N * (D_max / pi_min)²))`
  coming from the in-repo Hoeffding concentration bound
  `htExpEstimator_hoeffding_bound` (sub-Gaussian MGF route; strictly sharper
  than the Chebyshev/variance route also available via
  `treeAuditUniformDistortion_variance_bound_of_independent_bernoulli`).
* `dpoTreePOErrorStack` / `grpoPLTreePOErrorStack` / `grpoRLTreePOErrorStack`
  and the theorems `dpo_treepo_certificate_instantiates_error_stack` /
  `grpo_pl_treepo_certificate_instantiates_error_stack` /
  `grpo_rl_treepo_certificate_instantiates_error_stack`: an actual
  `PaperErrorStack` instance whose probability space is
  `bernoulliProductMeasure`, whose estimation deviation
  `gap_judge - gap_est` is (minus) the realized HT estimator minus its mean,
  and whose `delta_est`/`delta_clip` are the explicit Hoeffding failure
  probabilities. The transport envelope field is discharged by the concrete
  method certificate.

Unit conventions in the stack instances: the transport leg
(`C_meth * delta_R`) is in objective units, while the estimation/clipping
legs are carried in distortion-estimator units (threshold `t`); this keeps
the instantiation valid for the degenerate constants `C_meth = 0`. The fully
method-transported display `C_meth * (μ̂_dist + t)` is the realized-estimator
certificate above. The judge-calibration leg is degenerate in the concrete
Lean model (the audited distortion is evaluated by `fstar` itself), so any
strictly positive calibration envelope `B_cal` is accepted with zero failure
probability, matching the paper's `B_cal` hypothesis slot.
-/

set_option linter.mathlibStandardSet false

open MeasureTheory
open scoped BigOperators Real Nat Classical Pointwise NNReal ENNReal

set_option maxHeartbeats 400000
set_option maxRecDepth 4000
set_option relaxedAutoImplicit false
set_option autoImplicit false

noncomputable section

/-! ## Generic Bernoulli-Design Concentration Wrappers -/

/-- The Bernoulli product sampling design is a probability measure.
Extracted from the inline `letI` derivations used throughout the IPW layer so
that `PaperErrorStack.is_probability` can be discharged directly. -/
lemma bernoulliProductMeasure_isProbabilityMeasure
    {ι : Type*} [Fintype ι] [DecidableEq ι]
    (pi : ι → ℝ) (hpi_pos : ∀ i, 0 < pi i) (hpi_le : ∀ i, pi i ≤ 1) :
    IsProbabilityMeasure (bernoulliProductMeasure pi hpi_pos hpi_le) := by
  let μi : ι → Measure Bool := fun i => bernoulliMeasure pi hpi_pos hpi_le i
  letI : ∀ i, IsProbabilityMeasure (μi i) := by
    intro i
    dsimp [μi, bernoulliMeasure]
    infer_instance
  simpa [bernoulliProductMeasure, μi] using
    (Measure.pi.instIsProbabilityMeasure (μ := μi))

/-- ENNReal-valued form of the Hoeffding concentration bound for the HT
estimator of `Exp p f` under independent Bernoulli sampling. This is the
event-bound shape consumed by `PaperErrorStack.estimation_event` /
`PaperErrorStack.clipping_event`. Paper: thm:e2e part (d) (concentration
event `E_est`). -/
lemma htExpEstimator_hoeffding_bound_ennreal
    {ι : Type*} [Fintype ι] [DecidableEq ι]
    (p : PMF ι)
    (pi : ι → ℝ) (hpi_pos : ∀ i, 0 < pi i) (hpi_le : ∀ i, pi i ≤ 1)
    (f : ι → ℝ) (M : ℝ) (hM : 0 ≤ M) (hbound : ∀ i, |f i| ≤ M)
    (pi_min : ℝ) (hpi_min_pos : 0 < pi_min) (hpi_min_le : ∀ i, pi_min ≤ pi i)
    (ε : ℝ) (hε : 0 < ε) :
    bernoulliProductMeasure pi hpi_pos hpi_le
      {ω | |htExpEstimator p pi f ω - Exp p f| ≥ ε} ≤
      ENNReal.ofReal
        (2 * Real.exp (- ε ^ 2 /
          (8 * (Fintype.card ι) * (M / pi_min) ^ 2))) := by
  letI : IsProbabilityMeasure (bernoulliProductMeasure pi hpi_pos hpi_le) :=
    bernoulliProductMeasure_isProbabilityMeasure pi hpi_pos hpi_le
  have h_real :
      (bernoulliProductMeasure pi hpi_pos hpi_le).real
        {ω | |htExpEstimator p pi f ω - Exp p f| ≥ ε} ≤
        2 * Real.exp (- ε ^ 2 / (8 * (Fintype.card ι) * (M / pi_min) ^ 2)) :=
    htExpEstimator_hoeffding_bound (p := p) (pi := pi) (hpi_pos := hpi_pos)
      (hpi_le := hpi_le) (f := f) (M := M) (hM := hM) (hbound := hbound)
      (pi_min := pi_min) (hpi_min_pos := hpi_min_pos) (hpi_min_le := hpi_min_le)
      (ε := ε) (hε := hε)
  have hne :
      bernoulliProductMeasure pi hpi_pos hpi_le
        {ω | |htExpEstimator p pi f ω - Exp p f| ≥ ε} ≠ ∞ :=
    measure_ne_top _ _
  calc
    bernoulliProductMeasure pi hpi_pos hpi_le
        {ω | |htExpEstimator p pi f ω - Exp p f| ≥ ε}
        = ENNReal.ofReal
            ((bernoulliProductMeasure pi hpi_pos hpi_le
              {ω | |htExpEstimator p pi f ω - Exp p f| ≥ ε}).toReal) :=
          (ENNReal.ofReal_toReal hne).symm
    _ ≤ ENNReal.ofReal
          (2 * Real.exp (- ε ^ 2 /
            (8 * (Fintype.card ι) * (M / pi_min) ^ 2))) :=
        ENNReal.ofReal_le_ofReal (by simpa [measureReal_def] using h_real)

/-- Realized-estimator upgrade of an expected-estimator gap bound.

If a (deterministic) gap satisfies `|gap| ≤ C * Exp p f` — the expected-HT
form produced by the end-to-end method certificates — then, with probability
at least `1 - 2 exp(-t² / (8 N (M/pi_min)²))` under the Bernoulli sampling
design, the gap is bounded by `C` times the **realized** HT estimate plus
slack `t`. Paper: thm:e2e parts (c)/(d) with `B_est = C·t` and
`B_cal = B_clip = 0`. -/
lemma abs_gap_le_scaled_realized_htExpEstimator_high_prob
    {ι : Type*} [Fintype ι] [DecidableEq ι]
    (p : PMF ι)
    (pi : ι → ℝ) (hpi_pos : ∀ i, 0 < pi i) (hpi_le : ∀ i, pi i ≤ 1)
    (f : ι → ℝ) (M : ℝ) (hM : 0 ≤ M) (hbound : ∀ i, |f i| ≤ M)
    (pi_min : ℝ) (hpi_min_pos : 0 < pi_min) (hpi_min_le : ∀ i, pi_min ≤ pi i)
    (gap C : ℝ) (hC : 0 ≤ C)
    (h_gap : |gap| ≤ C * Exp p f)
    (t : ℝ) (ht : 0 < t) :
    (bernoulliProductMeasure pi hpi_pos hpi_le).real
      {ω | |gap| ≤ C * (htExpEstimator p pi f ω + t)} ≥
      1 - 2 * Real.exp (- t ^ 2 /
        (8 * (Fintype.card ι) * (M / pi_min) ^ 2)) := by
  classical
  letI : IsProbabilityMeasure (bernoulliProductMeasure pi hpi_pos hpi_le) :=
    bernoulliProductMeasure_isProbabilityMeasure pi hpi_pos hpi_le
  have h_bad :
      (bernoulliProductMeasure pi hpi_pos hpi_le).real
        {ω | |htExpEstimator p pi f ω - Exp p f| ≥ t} ≤
        2 * Real.exp (- t ^ 2 / (8 * (Fintype.card ι) * (M / pi_min) ^ 2)) :=
    htExpEstimator_hoeffding_bound (p := p) (pi := pi) (hpi_pos := hpi_pos)
      (hpi_le := hpi_le) (f := f) (M := M) (hM := hM) (hbound := hbound)
      (pi_min := pi_min) (hpi_min_pos := hpi_min_pos) (hpi_min_le := hpi_min_le)
      (ε := t) (hε := ht)
  have h_subset :
      {ω : ι → Bool | |htExpEstimator p pi f ω - Exp p f| ≥ t}ᶜ ⊆
        {ω : ι → Bool | |gap| ≤ C * (htExpEstimator p pi f ω + t)} := by
    intro ω hω
    simp only [Set.mem_compl_iff, Set.mem_setOf_eq, ge_iff_le, not_le] at hω
    -- hω : |htExpEstimator p pi f ω - Exp p f| < t
    have h_mean_le : Exp p f ≤ htExpEstimator p pi f ω + t := by
      have h1 : Exp p f - htExpEstimator p pi f ω ≤
          |htExpEstimator p pi f ω - Exp p f| := by
        rw [abs_sub_comm]
        exact le_abs_self _
      linarith
    have h_scaled : C * Exp p f ≤ C * (htExpEstimator p pi f ω + t) :=
      mul_le_mul_of_nonneg_left h_mean_le hC
    simp only [Set.mem_setOf_eq]
    exact le_trans h_gap h_scaled
  have h_cover :
      ({ω : ι → Bool | |htExpEstimator p pi f ω - Exp p f| ≥ t} ∪
        {ω : ι → Bool | |htExpEstimator p pi f ω - Exp p f| ≥ t}ᶜ) =
        Set.univ :=
    Set.union_compl_self _
  have h_one :
      (1 : ℝ) ≤
        (bernoulliProductMeasure pi hpi_pos hpi_le).real
          {ω | |htExpEstimator p pi f ω - Exp p f| ≥ t} +
        (bernoulliProductMeasure pi hpi_pos hpi_le).real
          {ω : ι → Bool | |htExpEstimator p pi f ω - Exp p f| ≥ t}ᶜ := by
    calc
      (1 : ℝ) =
          (bernoulliProductMeasure pi hpi_pos hpi_le).real Set.univ :=
        probReal_univ.symm
      _ = (bernoulliProductMeasure pi hpi_pos hpi_le).real
            ({ω : ι → Bool | |htExpEstimator p pi f ω - Exp p f| ≥ t} ∪
              {ω : ι → Bool | |htExpEstimator p pi f ω - Exp p f| ≥ t}ᶜ) := by
        rw [h_cover]
      _ ≤ (bernoulliProductMeasure pi hpi_pos hpi_le).real
            {ω | |htExpEstimator p pi f ω - Exp p f| ≥ t} +
          (bernoulliProductMeasure pi hpi_pos hpi_le).real
            {ω : ι → Bool | |htExpEstimator p pi f ω - Exp p f| ≥ t}ᶜ :=
        measureReal_union_le _ _
  have h_mono :
      (bernoulliProductMeasure pi hpi_pos hpi_le).real
        {ω : ι → Bool | |htExpEstimator p pi f ω - Exp p f| ≥ t}ᶜ ≤
        (bernoulliProductMeasure pi hpi_pos hpi_le).real
          {ω | |gap| ≤ C * (htExpEstimator p pi f ω + t)} :=
    measureReal_mono h_subset
  linarith

/-! ## Method-Level Objective Gaps -/

/-- The DPO full-document vs tree objective gap (Paper: thm:e2e, `G_meth`
for DPO). This is exactly the quantity bounded in
`dpo_treepo_end_to_end_certificate`. -/
def dpoTreeObjectiveGap
    {Strings Node A : Type*}
    (model : OPT.TreePreferenceSamplingModel Strings Node (A × A) 1)
    (pol pol_ref : Policy Strings A) (β : ℝ) (gpair : PMF (A × A)) : ℝ :=
  ExpectedDPOLoss pol pol_ref β model.docDist (fun _ => gpair) -
    OPT.ExpectedTreePreferenceLoss model
      (fun x group => DPOLossPointwise pol pol_ref β x (group 0).1 (group 0).2)

/-- The GRPO-PL full-document vs tree objective gap (Paper: thm:e2e, `G_meth`
for GRPO-PL with constant group generator). This is exactly the quantity
bounded in `grpo_pl_treepo_end_to_end_certificate`. -/
def grpoPLTreeObjectiveGap
    {Strings Node A : Type*} [Monoid Strings] {k : ℕ}
    (model : OPT.TreePreferenceSamplingModel Strings Node A k)
    (pol : Policy' Strings A) (ranker : Strings → GroupRanker A k)
    (g : PMF (Fin k → A)) : ℝ :=
  ExpectedGRPOLoss pol ranker model.docDist (fun _ => g) -
    OPT.ExpectedTreePreferenceLoss model
      (fun x group => GRPOLossPointwise pol x group (ranker x group))

/-- The GRPO-RL full-document vs tree objective gap (Paper: thm:e2e, `G_meth`
for GRPO-RL with constant group generator). This is exactly the quantity
bounded in `grpo_rl_treepo_end_to_end_certificate`. -/
def grpoRLTreeObjectiveGap
    {Strings Node A : Type*} [Monoid Strings] {k : ℕ}
    (model : OPT.TreePreferenceSamplingModel Strings Node A k)
    (pol pol_old pol_ref : Policy' Strings A)
    (reward : Strings → A → ℝ) (eps beta : ℝ)
    (g : PMF (Fin k → A)) : ℝ :=
  ExpectedGRPORLLoss pol pol_old pol_ref reward eps beta model.docDist
      (fun _ => g) -
    OPT.ExpectedTreePreferenceLoss model
      (fun x group => GRPORLLossPointwise pol pol_old pol_ref reward eps beta x group)

/-! ## DPO: Realized-Estimator Certificate (Paper Display) -/

/-- Realized-estimator end-to-end TreePO certificate for DPO
(Paper: thm:e2e parts (c)/(d), the displayed audited bound).

Under the hypotheses of `dpo_treepo_end_to_end_certificate` plus a bounded
distortion envelope `D_max` and strict propensity positivity `pi_min`, for
every slack `t > 0` the DPO objective gap is bounded by
`C_DPO * (μ̂_dist(ω) + t)` — the method transport constant times the
**realized** HT distortion estimate plus slack — with probability at least
`1 - δ_est(t)` under the Bernoulli sampling design, where
`δ_est(t) = 2 exp(-t² / (8 N (D_max/pi_min)²))` comes from the in-repo
Hoeffding bound `htExpEstimator_hoeffding_bound`. -/
theorem dpo_treepo_realized_estimator_certificate
    {Strings Node A Y : Type*}
    [Monoid Strings] [PseudoMetricSpace Y]
    [Fintype Strings] [Fintype Node] [Fintype A]
    [DecidableEq Strings] [DecidableEq Node] [DecidableEq A]
    (model : OPT.TreePreferenceSamplingModel Strings Node (A × A) 1)
    (fstar : Strings → Y)
    (pol pol_ref : Policy Strings A)
    (β : ℝ) (L_pol : ℝ≥0)
    (gpair : PMF (A × A))
    (h_group : ∀ u,
      model.groupGen u = PMF.map (fun p : A × A => (fun _ : Fin 1 => p)) gpair)
    (h_lip : PolicyLipschitz pol pol_ref fstar L_pol)
    (M_loss : ℝ) (hM_loss : 0 ≤ M_loss)
    (h_loss_bound : ∀ (x : Strings) (p : A × A),
      |DPOLossPointwise pol pol_ref β x p.1 p.2| ≤ M_loss)
    (pi : TreeUnit Strings Node (A × A) 1 → ℝ)
    (hpi_pos : ∀ i, 0 < pi i)
    (hpi_le : ∀ i, pi i ≤ 1)
    (pi_min : ℝ) (hpi_min_pos : 0 < pi_min) (hpi_min_le : ∀ i, pi_min ≤ pi i)
    (D_max : ℝ) (hD_max : 0 ≤ D_max)
    (h_dist_bound : ∀ u, |treeDistortion model fstar u| ≤ D_max)
    (t : ℝ) (ht : 0 < t) :
    let loss : Strings → (Fin 1 → (A × A)) → ℝ :=
      fun x group => DPOLossPointwise pol pol_ref β x (group 0).1 (group 0).2
    (bernoulliProductMeasure pi hpi_pos hpi_le).real
      {ω | |ExpectedDPOLoss pol pol_ref β model.docDist (fun _ => gpair) -
            OPT.ExpectedTreePreferenceLoss model loss| ≤
          (2 * |β| * (L_pol : ℝ)) *
            (htExpEstimator (p := treeUnitPMF model) (pi := pi)
                (treeDistortion model fstar) ω + t)} ≥
      1 - 2 * Real.exp (- t ^ 2 /
        (8 * (Fintype.card (TreeUnit Strings Node (A × A) 1)) *
          (D_max / pi_min) ^ 2)) := by
  classical
  dsimp
  have hC : (0 : ℝ) ≤ 2 * |β| * (L_pol : ℝ) := by positivity
  obtain ⟨-, h_gap⟩ :=
    dpo_treepo_end_to_end_certificate
      (model := model) (fstar := fstar)
      (pol := pol) (pol_ref := pol_ref) (β := β) (L_pol := L_pol)
      (gpair := gpair) (h_group := h_group) (h_lip := h_lip)
      (M_loss := M_loss) (hM_loss := hM_loss) (h_loss_bound := h_loss_bound)
      (pi := pi) (hpi_pos := hpi_pos) (hpi_le := hpi_le)
  rw [ipw_tree_distortion_unbiased model fstar pi hpi_pos hpi_le] at h_gap
  exact abs_gap_le_scaled_realized_htExpEstimator_high_prob
    (p := treeUnitPMF model) (pi := pi) (hpi_pos := hpi_pos) (hpi_le := hpi_le)
    (f := treeDistortion model fstar) (M := D_max) (hM := hD_max)
    (hbound := h_dist_bound) (pi_min := pi_min) (hpi_min_pos := hpi_min_pos)
    (hpi_min_le := hpi_min_le)
    (gap := ExpectedDPOLoss pol pol_ref β model.docDist (fun _ => gpair) -
      OPT.ExpectedTreePreferenceLoss model
        (fun x group => DPOLossPointwise pol pol_ref β x (group 0).1 (group 0).2))
    (C := 2 * |β| * (L_pol : ℝ)) (hC := hC) (h_gap := h_gap) (t := t) (ht := ht)

/-! ## DPO: Concrete `PaperErrorStack` Instance -/

/-- Concrete `PaperErrorStack` instance for DPO, built from the objects of
`dpo_treepo_end_to_end_certificate` (audit finding F7 glue).

* probability space: `bernoulliProductMeasure pi hpi_pos hpi_le`;
* `gap_oracle = gap_judge = gap_clip`: the (deterministic) DPO objective gap;
* `gap_est`: the gap shifted by the realized HT distortion estimator minus
  its mean, so the estimation and clipping deviations are exactly the
  realized HT fluctuation `μ̂_dist(ω) - E[μ̂_dist]`;
* transport envelope: discharged by the concrete method certificate
  (`|G_DPO| ≤ C_DPO * Δ_R` with `Δ_R = ExpectedTreeDistortion`);
* `delta_est`/`delta_clip`: explicit Hoeffding failure probabilities from
  `htExpEstimator_hoeffding_bound`;
* calibration leg: degenerate (judge = `fstar` in this model); any `B_cal > 0`
  is certified with zero failure probability.

Paper: thm:e2e (all four parts composed). -/
def dpoTreePOErrorStack
    {Strings Node A Y : Type*}
    [Monoid Strings] [PseudoMetricSpace Y]
    [Fintype Strings] [Fintype Node] [Fintype A]
    [DecidableEq Strings] [DecidableEq Node] [DecidableEq A]
    (model : OPT.TreePreferenceSamplingModel Strings Node (A × A) 1)
    (fstar : Strings → Y)
    (pol pol_ref : Policy Strings A)
    (β : ℝ) (L_pol : ℝ≥0)
    (gpair : PMF (A × A))
    (h_group : ∀ u,
      model.groupGen u = PMF.map (fun p : A × A => (fun _ : Fin 1 => p)) gpair)
    (h_lip : PolicyLipschitz pol pol_ref fstar L_pol)
    (M_loss : ℝ) (hM_loss : 0 ≤ M_loss)
    (h_loss_bound : ∀ (x : Strings) (p : A × A),
      |DPOLossPointwise pol pol_ref β x p.1 p.2| ≤ M_loss)
    (pi : TreeUnit Strings Node (A × A) 1 → ℝ)
    (hpi_pos : ∀ i, 0 < pi i)
    (hpi_le : ∀ i, pi i ≤ 1)
    (pi_min : ℝ) (hpi_min_pos : 0 < pi_min) (hpi_min_le : ∀ i, pi_min ≤ pi i)
    (D_max : ℝ) (hD_max : 0 ≤ D_max)
    (h_dist_bound : ∀ u, |treeDistortion model fstar u| ≤ D_max)
    (B_cal : ℝ) (hB_cal : 0 < B_cal)
    (t : ℝ) (ht : 0 < t) :
    PaperErrorStack (TreeUnit Strings Node (A × A) 1 → Bool) where
  μ := bernoulliProductMeasure pi hpi_pos hpi_le
  is_probability :=
    bernoulliProductMeasure_isProbabilityMeasure pi hpi_pos hpi_le
  gap_oracle := fun _ => dpoTreeObjectiveGap model pol pol_ref β gpair
  gap_judge := fun _ => dpoTreeObjectiveGap model pol pol_ref β gpair
  gap_est := fun ω =>
    dpoTreeObjectiveGap model pol pol_ref β gpair +
      (htExpEstimator (p := treeUnitPMF model) (pi := pi)
          (treeDistortion model fstar) ω -
        ExpectedTreeDistortion model fstar)
  gap_clip := fun _ => dpoTreeObjectiveGap model pol pol_ref β gpair
  certificate :=
    { localLaw :=
        { delta_R := ExpectedTreeDistortion model fstar
          methodTransport := 2 * |β| * (L_pol : ℝ) }
      calibration := B_cal
      estimation := t
      clipping := t }
  delta_cal := 0
  delta_est :=
    ENNReal.ofReal (2 * Real.exp (- t ^ 2 /
      (8 * (Fintype.card (TreeUnit Strings Node (A × A) 1)) *
        (D_max / pi_min) ^ 2)))
  delta_clip :=
    ENNReal.ofReal (2 * Real.exp (- t ^ 2 /
      (8 * (Fintype.card (TreeUnit Strings Node (A × A) 1)) *
        (D_max / pi_min) ^ 2)))
  delta_total :=
    ENNReal.ofReal (4 * Real.exp (- t ^ 2 /
      (8 * (Fintype.card (TreeUnit Strings Node (A × A) 1)) *
        (D_max / pi_min) ^ 2)))
  transport_envelope := by
    intro ω
    obtain ⟨-, h_gap⟩ :=
      dpo_treepo_end_to_end_certificate
        (model := model) (fstar := fstar)
        (pol := pol) (pol_ref := pol_ref) (β := β) (L_pol := L_pol)
        (gpair := gpair) (h_group := h_group) (h_lip := h_lip)
        (M_loss := M_loss) (hM_loss := hM_loss) (h_loss_bound := h_loss_bound)
        (pi := pi) (hpi_pos := hpi_pos) (hpi_le := hpi_le)
    rw [ipw_tree_distortion_unbiased model fstar pi hpi_pos hpi_le] at h_gap
    exact h_gap
  calibration_event := by
    show bernoulliProductMeasure pi hpi_pos hpi_le
        {ω : TreeUnit Strings Node (A × A) 1 → Bool |
          |dpoTreeObjectiveGap model pol pol_ref β gpair -
            dpoTreeObjectiveGap model pol pol_ref β gpair| ≥ B_cal} ≤ 0
    have hset :
        {ω : TreeUnit Strings Node (A × A) 1 → Bool |
          |dpoTreeObjectiveGap model pol pol_ref β gpair -
            dpoTreeObjectiveGap model pol pol_ref β gpair| ≥ B_cal} =
          (∅ : Set (TreeUnit Strings Node (A × A) 1 → Bool)) := by
      apply Set.eq_empty_iff_forall_notMem.mpr
      intro _ω hω
      simp only [Set.mem_setOf_eq, sub_self, abs_zero, ge_iff_le] at hω
      exact absurd hω (not_le.mpr hB_cal)
    rw [hset]
    simp
  estimation_event := by
    show bernoulliProductMeasure pi hpi_pos hpi_le
        {ω : TreeUnit Strings Node (A × A) 1 → Bool |
          |dpoTreeObjectiveGap model pol pol_ref β gpair -
            (dpoTreeObjectiveGap model pol pol_ref β gpair +
              (htExpEstimator (p := treeUnitPMF model) (pi := pi)
                  (treeDistortion model fstar) ω -
                ExpectedTreeDistortion model fstar))| ≥ t} ≤
      ENNReal.ofReal (2 * Real.exp (- t ^ 2 /
        (8 * (Fintype.card (TreeUnit Strings Node (A × A) 1)) *
          (D_max / pi_min) ^ 2)))
    have hset :
        {ω : TreeUnit Strings Node (A × A) 1 → Bool |
          |dpoTreeObjectiveGap model pol pol_ref β gpair -
            (dpoTreeObjectiveGap model pol pol_ref β gpair +
              (htExpEstimator (p := treeUnitPMF model) (pi := pi)
                  (treeDistortion model fstar) ω -
                ExpectedTreeDistortion model fstar))| ≥ t} =
          {ω : TreeUnit Strings Node (A × A) 1 → Bool |
            |htExpEstimator (p := treeUnitPMF model) (pi := pi)
                (treeDistortion model fstar) ω -
              ExpectedTreeDistortion model fstar| ≥ t} := by
      ext ω
      have harg :
          dpoTreeObjectiveGap model pol pol_ref β gpair -
            (dpoTreeObjectiveGap model pol pol_ref β gpair +
              (htExpEstimator (p := treeUnitPMF model) (pi := pi)
                  (treeDistortion model fstar) ω -
                ExpectedTreeDistortion model fstar)) =
          -(htExpEstimator (p := treeUnitPMF model) (pi := pi)
              (treeDistortion model fstar) ω -
            ExpectedTreeDistortion model fstar) := by
        ring
      simp only [Set.mem_setOf_eq, harg, abs_neg]
    rw [hset]
    exact htExpEstimator_hoeffding_bound_ennreal
      (p := treeUnitPMF model) (pi := pi) (hpi_pos := hpi_pos)
      (hpi_le := hpi_le) (f := treeDistortion model fstar)
      (M := D_max) (hM := hD_max) (hbound := h_dist_bound)
      (pi_min := pi_min) (hpi_min_pos := hpi_min_pos)
      (hpi_min_le := hpi_min_le) (ε := t) (hε := ht)
  clipping_event := by
    show bernoulliProductMeasure pi hpi_pos hpi_le
        {ω : TreeUnit Strings Node (A × A) 1 → Bool |
          |(dpoTreeObjectiveGap model pol pol_ref β gpair +
              (htExpEstimator (p := treeUnitPMF model) (pi := pi)
                  (treeDistortion model fstar) ω -
                ExpectedTreeDistortion model fstar)) -
            dpoTreeObjectiveGap model pol pol_ref β gpair| ≥ t} ≤
      ENNReal.ofReal (2 * Real.exp (- t ^ 2 /
        (8 * (Fintype.card (TreeUnit Strings Node (A × A) 1)) *
          (D_max / pi_min) ^ 2)))
    have hset :
        {ω : TreeUnit Strings Node (A × A) 1 → Bool |
          |(dpoTreeObjectiveGap model pol pol_ref β gpair +
              (htExpEstimator (p := treeUnitPMF model) (pi := pi)
                  (treeDistortion model fstar) ω -
                ExpectedTreeDistortion model fstar)) -
            dpoTreeObjectiveGap model pol pol_ref β gpair| ≥ t} =
          {ω : TreeUnit Strings Node (A × A) 1 → Bool |
            |htExpEstimator (p := treeUnitPMF model) (pi := pi)
                (treeDistortion model fstar) ω -
              ExpectedTreeDistortion model fstar| ≥ t} := by
      ext ω
      have harg :
          (dpoTreeObjectiveGap model pol pol_ref β gpair +
              (htExpEstimator (p := treeUnitPMF model) (pi := pi)
                  (treeDistortion model fstar) ω -
                ExpectedTreeDistortion model fstar)) -
            dpoTreeObjectiveGap model pol pol_ref β gpair =
          htExpEstimator (p := treeUnitPMF model) (pi := pi)
              (treeDistortion model fstar) ω -
            ExpectedTreeDistortion model fstar := by
        ring
      simp only [Set.mem_setOf_eq, harg]
    rw [hset]
    exact htExpEstimator_hoeffding_bound_ennreal
      (p := treeUnitPMF model) (pi := pi) (hpi_pos := hpi_pos)
      (hpi_le := hpi_le) (f := treeDistortion model fstar)
      (M := D_max) (hM := hD_max) (hbound := h_dist_bound)
      (pi_min := pi_min) (hpi_min_pos := hpi_min_pos)
      (hpi_min_le := hpi_min_le) (ε := t) (hε := ht)
  failure_budget := by
    have hnn : (0 : ℝ) ≤ 2 * Real.exp (- t ^ 2 /
        (8 * (Fintype.card (TreeUnit Strings Node (A × A) 1)) *
          (D_max / pi_min) ^ 2)) := by positivity
    rw [zero_add, ← ENNReal.ofReal_add hnn hnn]
    exact ENNReal.ofReal_le_ofReal (le_of_eq (by ring))

/-- Glue theorem for audit finding F7: the hypotheses of
`dpo_treepo_end_to_end_certificate` (plus the Hoeffding concentration data)
instantiate the abstract `PaperErrorStack` with the concrete Bernoulli
product probability space and the realized HT distortion estimator. The
witness is `dpoTreePOErrorStack`; all listed identifications hold
definitionally. Paper: thm:e2e. -/
theorem dpo_treepo_certificate_instantiates_error_stack
    {Strings Node A Y : Type*}
    [Monoid Strings] [PseudoMetricSpace Y]
    [Fintype Strings] [Fintype Node] [Fintype A]
    [DecidableEq Strings] [DecidableEq Node] [DecidableEq A]
    (model : OPT.TreePreferenceSamplingModel Strings Node (A × A) 1)
    (fstar : Strings → Y)
    (pol pol_ref : Policy Strings A)
    (β : ℝ) (L_pol : ℝ≥0)
    (gpair : PMF (A × A))
    (h_group : ∀ u,
      model.groupGen u = PMF.map (fun p : A × A => (fun _ : Fin 1 => p)) gpair)
    (h_lip : PolicyLipschitz pol pol_ref fstar L_pol)
    (M_loss : ℝ) (hM_loss : 0 ≤ M_loss)
    (h_loss_bound : ∀ (x : Strings) (p : A × A),
      |DPOLossPointwise pol pol_ref β x p.1 p.2| ≤ M_loss)
    (pi : TreeUnit Strings Node (A × A) 1 → ℝ)
    (hpi_pos : ∀ i, 0 < pi i)
    (hpi_le : ∀ i, pi i ≤ 1)
    (pi_min : ℝ) (hpi_min_pos : 0 < pi_min) (hpi_min_le : ∀ i, pi_min ≤ pi i)
    (D_max : ℝ) (hD_max : 0 ≤ D_max)
    (h_dist_bound : ∀ u, |treeDistortion model fstar u| ≤ D_max)
    (B_cal : ℝ) (hB_cal : 0 < B_cal)
    (t : ℝ) (ht : 0 < t) :
    ∃ s : PaperErrorStack (TreeUnit Strings Node (A × A) 1 → Bool),
      s.μ = bernoulliProductMeasure pi hpi_pos hpi_le ∧
      s.gap_oracle = (fun _ => dpoTreeObjectiveGap model pol pol_ref β gpair) ∧
      s.gap_est = (fun ω =>
        dpoTreeObjectiveGap model pol pol_ref β gpair +
          (htExpEstimator (p := treeUnitPMF model) (pi := pi)
              (treeDistortion model fstar) ω -
            ExpectedTreeDistortion model fstar)) ∧
      s.certificate.localLaw.delta_R = ExpectedTreeDistortion model fstar ∧
      s.certificate.localLaw.methodTransport = 2 * |β| * (L_pol : ℝ) ∧
      s.certificate.totalObjectiveBound =
        2 * |β| * (L_pol : ℝ) * ExpectedTreeDistortion model fstar +
          B_cal + t + t ∧
      s.delta_est =
        ENNReal.ofReal (2 * Real.exp (- t ^ 2 /
          (8 * (Fintype.card (TreeUnit Strings Node (A × A) 1)) *
            (D_max / pi_min) ^ 2))) ∧
      s.delta_total =
        ENNReal.ofReal (4 * Real.exp (- t ^ 2 /
          (8 * (Fintype.card (TreeUnit Strings Node (A × A) 1)) *
            (D_max / pi_min) ^ 2))) :=
  ⟨dpoTreePOErrorStack model fstar pol pol_ref β L_pol gpair h_group h_lip
      M_loss hM_loss h_loss_bound pi hpi_pos hpi_le pi_min hpi_min_pos
      hpi_min_le D_max hD_max h_dist_bound B_cal hB_cal t ht,
    rfl, rfl, rfl, rfl, rfl, rfl, rfl, rfl⟩

/-- Final unfolded certificate of the DPO `PaperErrorStack` instance
(`PaperErrorStack.high_prob_total` applied to `dpoTreePOErrorStack`):
the DPO objective gap exceeds the paper's total certificate bound
`C_DPO * Δ_R + B_cal + 2t` with probability at most
`4 exp(-t² / (8 N (D_max/pi_min)²))`. Paper: thm:e2e part (d). -/
theorem dpo_treepo_error_stack_high_prob
    {Strings Node A Y : Type*}
    [Monoid Strings] [PseudoMetricSpace Y]
    [Fintype Strings] [Fintype Node] [Fintype A]
    [DecidableEq Strings] [DecidableEq Node] [DecidableEq A]
    (model : OPT.TreePreferenceSamplingModel Strings Node (A × A) 1)
    (fstar : Strings → Y)
    (pol pol_ref : Policy Strings A)
    (β : ℝ) (L_pol : ℝ≥0)
    (gpair : PMF (A × A))
    (h_group : ∀ u,
      model.groupGen u = PMF.map (fun p : A × A => (fun _ : Fin 1 => p)) gpair)
    (h_lip : PolicyLipschitz pol pol_ref fstar L_pol)
    (M_loss : ℝ) (hM_loss : 0 ≤ M_loss)
    (h_loss_bound : ∀ (x : Strings) (p : A × A),
      |DPOLossPointwise pol pol_ref β x p.1 p.2| ≤ M_loss)
    (pi : TreeUnit Strings Node (A × A) 1 → ℝ)
    (hpi_pos : ∀ i, 0 < pi i)
    (hpi_le : ∀ i, pi i ≤ 1)
    (pi_min : ℝ) (hpi_min_pos : 0 < pi_min) (hpi_min_le : ∀ i, pi_min ≤ pi i)
    (D_max : ℝ) (hD_max : 0 ≤ D_max)
    (h_dist_bound : ∀ u, |treeDistortion model fstar u| ≤ D_max)
    (B_cal : ℝ) (hB_cal : 0 < B_cal)
    (t : ℝ) (ht : 0 < t) :
    bernoulliProductMeasure pi hpi_pos hpi_le
      {_ω : TreeUnit Strings Node (A × A) 1 → Bool |
        |dpoTreeObjectiveGap model pol pol_ref β gpair| ≥
          2 * |β| * (L_pol : ℝ) * ExpectedTreeDistortion model fstar +
            B_cal + t + t} ≤
      ENNReal.ofReal (4 * Real.exp (- t ^ 2 /
        (8 * (Fintype.card (TreeUnit Strings Node (A × A) 1)) *
          (D_max / pi_min) ^ 2))) :=
  (dpoTreePOErrorStack model fstar pol pol_ref β L_pol gpair h_group h_lip
      M_loss hM_loss h_loss_bound pi hpi_pos hpi_le pi_min hpi_min_pos
      hpi_min_le D_max hD_max h_dist_bound B_cal hB_cal t ht).high_prob_total

/-! ## GRPO-PL: Realized-Estimator Certificate and Stack Instance -/

/-- Realized-estimator end-to-end TreePO certificate for GRPO-PL with
constant group generator (Paper: thm:e2e parts (c)/(d)). GRPO-PL analogue of
`dpo_treepo_realized_estimator_certificate` with transport constant `L`. -/
theorem grpo_pl_treepo_realized_estimator_certificate
    {Strings Node A Y : Type*} [Monoid Strings] [PseudoMetricSpace Y]
    {k : ℕ} [Fintype Strings] [Fintype Node] [Fintype A]
    [DecidableEq Strings] [DecidableEq Node] [DecidableEq A]
    (model : OPT.TreePreferenceSamplingModel Strings Node A k)
    (fstar : Strings → Y)
    (pol : Policy' Strings A) (ranker : Strings → GroupRanker A k)
    (g : PMF (Fin k → A))
    (L : ℝ≥0)
    (h_group : ∀ u, model.groupGen u = g)
    (h_pol_lip : GRPOPolicyLipschitz pol fstar L)
    (h_ranker : OracleIndexedRanker ranker fstar)
    (h_rum : ∀ x z,
      ExpectedGRPOLossLipschitz pol ranker fstar g L h_pol_lip h_ranker x z)
    (pi : TreeUnit Strings Node A k → ℝ)
    (hpi_pos : ∀ i, 0 < pi i)
    (hpi_le : ∀ i, pi i ≤ 1)
    (pi_min : ℝ) (hpi_min_pos : 0 < pi_min) (hpi_min_le : ∀ i, pi_min ≤ pi i)
    (D_max : ℝ) (hD_max : 0 ≤ D_max)
    (h_dist_bound : ∀ u, |treeDistortion model fstar u| ≤ D_max)
    (t : ℝ) (ht : 0 < t) :
    let loss : Strings → (Fin k → A) → ℝ :=
      fun x group => GRPOLossPointwise pol x group (ranker x group)
    (bernoulliProductMeasure pi hpi_pos hpi_le).real
      {ω | |ExpectedGRPOLoss pol ranker model.docDist (fun _ => g) -
            OPT.ExpectedTreePreferenceLoss model loss| ≤
          (L : ℝ) *
            (htExpEstimator (p := treeUnitPMF model) (pi := pi)
                (treeDistortion model fstar) ω + t)} ≥
      1 - 2 * Real.exp (- t ^ 2 /
        (8 * (Fintype.card (TreeUnit Strings Node A k)) *
          (D_max / pi_min) ^ 2)) := by
  classical
  dsimp
  have hC : (0 : ℝ) ≤ (L : ℝ) := L.coe_nonneg
  obtain ⟨-, h_gap⟩ :=
    grpo_pl_treepo_end_to_end_certificate
      (model := model) (fstar := fstar)
      (pol := pol) (ranker := ranker) (g := g) (L := L)
      (h_group := h_group) (h_pol_lip := h_pol_lip) (h_ranker := h_ranker)
      (h_rum := h_rum) (pi := pi) (hpi_pos := hpi_pos) (hpi_le := hpi_le)
  rw [ipw_tree_distortion_unbiased model fstar pi hpi_pos hpi_le] at h_gap
  exact abs_gap_le_scaled_realized_htExpEstimator_high_prob
    (p := treeUnitPMF model) (pi := pi) (hpi_pos := hpi_pos) (hpi_le := hpi_le)
    (f := treeDistortion model fstar) (M := D_max) (hM := hD_max)
    (hbound := h_dist_bound) (pi_min := pi_min) (hpi_min_pos := hpi_min_pos)
    (hpi_min_le := hpi_min_le)
    (gap := ExpectedGRPOLoss pol ranker model.docDist (fun _ => g) -
      OPT.ExpectedTreePreferenceLoss model
        (fun x group => GRPOLossPointwise pol x group (ranker x group)))
    (C := (L : ℝ)) (hC := hC) (h_gap := h_gap) (t := t) (ht := ht)

/-- Concrete `PaperErrorStack` instance for GRPO-PL with constant group
generator; GRPO-PL analogue of `dpoTreePOErrorStack` (audit finding F7 glue).
Paper: thm:e2e. -/
def grpoPLTreePOErrorStack
    {Strings Node A Y : Type*} [Monoid Strings] [PseudoMetricSpace Y]
    {k : ℕ} [Fintype Strings] [Fintype Node] [Fintype A]
    [DecidableEq Strings] [DecidableEq Node] [DecidableEq A]
    (model : OPT.TreePreferenceSamplingModel Strings Node A k)
    (fstar : Strings → Y)
    (pol : Policy' Strings A) (ranker : Strings → GroupRanker A k)
    (g : PMF (Fin k → A))
    (L : ℝ≥0)
    (h_group : ∀ u, model.groupGen u = g)
    (h_pol_lip : GRPOPolicyLipschitz pol fstar L)
    (h_ranker : OracleIndexedRanker ranker fstar)
    (h_rum : ∀ x z,
      ExpectedGRPOLossLipschitz pol ranker fstar g L h_pol_lip h_ranker x z)
    (pi : TreeUnit Strings Node A k → ℝ)
    (hpi_pos : ∀ i, 0 < pi i)
    (hpi_le : ∀ i, pi i ≤ 1)
    (pi_min : ℝ) (hpi_min_pos : 0 < pi_min) (hpi_min_le : ∀ i, pi_min ≤ pi i)
    (D_max : ℝ) (hD_max : 0 ≤ D_max)
    (h_dist_bound : ∀ u, |treeDistortion model fstar u| ≤ D_max)
    (B_cal : ℝ) (hB_cal : 0 < B_cal)
    (t : ℝ) (ht : 0 < t) :
    PaperErrorStack (TreeUnit Strings Node A k → Bool) where
  μ := bernoulliProductMeasure pi hpi_pos hpi_le
  is_probability :=
    bernoulliProductMeasure_isProbabilityMeasure pi hpi_pos hpi_le
  gap_oracle := fun _ => grpoPLTreeObjectiveGap model pol ranker g
  gap_judge := fun _ => grpoPLTreeObjectiveGap model pol ranker g
  gap_est := fun ω =>
    grpoPLTreeObjectiveGap model pol ranker g +
      (htExpEstimator (p := treeUnitPMF model) (pi := pi)
          (treeDistortion model fstar) ω -
        ExpectedTreeDistortion model fstar)
  gap_clip := fun _ => grpoPLTreeObjectiveGap model pol ranker g
  certificate :=
    { localLaw :=
        { delta_R := ExpectedTreeDistortion model fstar
          methodTransport := (L : ℝ) }
      calibration := B_cal
      estimation := t
      clipping := t }
  delta_cal := 0
  delta_est :=
    ENNReal.ofReal (2 * Real.exp (- t ^ 2 /
      (8 * (Fintype.card (TreeUnit Strings Node A k)) *
        (D_max / pi_min) ^ 2)))
  delta_clip :=
    ENNReal.ofReal (2 * Real.exp (- t ^ 2 /
      (8 * (Fintype.card (TreeUnit Strings Node A k)) *
        (D_max / pi_min) ^ 2)))
  delta_total :=
    ENNReal.ofReal (4 * Real.exp (- t ^ 2 /
      (8 * (Fintype.card (TreeUnit Strings Node A k)) *
        (D_max / pi_min) ^ 2)))
  transport_envelope := by
    intro ω
    obtain ⟨-, h_gap⟩ :=
      grpo_pl_treepo_end_to_end_certificate
        (model := model) (fstar := fstar)
        (pol := pol) (ranker := ranker) (g := g) (L := L)
        (h_group := h_group) (h_pol_lip := h_pol_lip) (h_ranker := h_ranker)
        (h_rum := h_rum) (pi := pi) (hpi_pos := hpi_pos) (hpi_le := hpi_le)
    rw [ipw_tree_distortion_unbiased model fstar pi hpi_pos hpi_le] at h_gap
    exact h_gap
  calibration_event := by
    show bernoulliProductMeasure pi hpi_pos hpi_le
        {ω : TreeUnit Strings Node A k → Bool |
          |grpoPLTreeObjectiveGap model pol ranker g -
            grpoPLTreeObjectiveGap model pol ranker g| ≥ B_cal} ≤ 0
    have hset :
        {ω : TreeUnit Strings Node A k → Bool |
          |grpoPLTreeObjectiveGap model pol ranker g -
            grpoPLTreeObjectiveGap model pol ranker g| ≥ B_cal} =
          (∅ : Set (TreeUnit Strings Node A k → Bool)) := by
      apply Set.eq_empty_iff_forall_notMem.mpr
      intro _ω hω
      simp only [Set.mem_setOf_eq, sub_self, abs_zero, ge_iff_le] at hω
      exact absurd hω (not_le.mpr hB_cal)
    rw [hset]
    simp
  estimation_event := by
    show bernoulliProductMeasure pi hpi_pos hpi_le
        {ω : TreeUnit Strings Node A k → Bool |
          |grpoPLTreeObjectiveGap model pol ranker g -
            (grpoPLTreeObjectiveGap model pol ranker g +
              (htExpEstimator (p := treeUnitPMF model) (pi := pi)
                  (treeDistortion model fstar) ω -
                ExpectedTreeDistortion model fstar))| ≥ t} ≤
      ENNReal.ofReal (2 * Real.exp (- t ^ 2 /
        (8 * (Fintype.card (TreeUnit Strings Node A k)) *
          (D_max / pi_min) ^ 2)))
    have hset :
        {ω : TreeUnit Strings Node A k → Bool |
          |grpoPLTreeObjectiveGap model pol ranker g -
            (grpoPLTreeObjectiveGap model pol ranker g +
              (htExpEstimator (p := treeUnitPMF model) (pi := pi)
                  (treeDistortion model fstar) ω -
                ExpectedTreeDistortion model fstar))| ≥ t} =
          {ω : TreeUnit Strings Node A k → Bool |
            |htExpEstimator (p := treeUnitPMF model) (pi := pi)
                (treeDistortion model fstar) ω -
              ExpectedTreeDistortion model fstar| ≥ t} := by
      ext ω
      have harg :
          grpoPLTreeObjectiveGap model pol ranker g -
            (grpoPLTreeObjectiveGap model pol ranker g +
              (htExpEstimator (p := treeUnitPMF model) (pi := pi)
                  (treeDistortion model fstar) ω -
                ExpectedTreeDistortion model fstar)) =
          -(htExpEstimator (p := treeUnitPMF model) (pi := pi)
              (treeDistortion model fstar) ω -
            ExpectedTreeDistortion model fstar) := by
        ring
      simp only [Set.mem_setOf_eq, harg, abs_neg]
    rw [hset]
    exact htExpEstimator_hoeffding_bound_ennreal
      (p := treeUnitPMF model) (pi := pi) (hpi_pos := hpi_pos)
      (hpi_le := hpi_le) (f := treeDistortion model fstar)
      (M := D_max) (hM := hD_max) (hbound := h_dist_bound)
      (pi_min := pi_min) (hpi_min_pos := hpi_min_pos)
      (hpi_min_le := hpi_min_le) (ε := t) (hε := ht)
  clipping_event := by
    show bernoulliProductMeasure pi hpi_pos hpi_le
        {ω : TreeUnit Strings Node A k → Bool |
          |(grpoPLTreeObjectiveGap model pol ranker g +
              (htExpEstimator (p := treeUnitPMF model) (pi := pi)
                  (treeDistortion model fstar) ω -
                ExpectedTreeDistortion model fstar)) -
            grpoPLTreeObjectiveGap model pol ranker g| ≥ t} ≤
      ENNReal.ofReal (2 * Real.exp (- t ^ 2 /
        (8 * (Fintype.card (TreeUnit Strings Node A k)) *
          (D_max / pi_min) ^ 2)))
    have hset :
        {ω : TreeUnit Strings Node A k → Bool |
          |(grpoPLTreeObjectiveGap model pol ranker g +
              (htExpEstimator (p := treeUnitPMF model) (pi := pi)
                  (treeDistortion model fstar) ω -
                ExpectedTreeDistortion model fstar)) -
            grpoPLTreeObjectiveGap model pol ranker g| ≥ t} =
          {ω : TreeUnit Strings Node A k → Bool |
            |htExpEstimator (p := treeUnitPMF model) (pi := pi)
                (treeDistortion model fstar) ω -
              ExpectedTreeDistortion model fstar| ≥ t} := by
      ext ω
      have harg :
          (grpoPLTreeObjectiveGap model pol ranker g +
              (htExpEstimator (p := treeUnitPMF model) (pi := pi)
                  (treeDistortion model fstar) ω -
                ExpectedTreeDistortion model fstar)) -
            grpoPLTreeObjectiveGap model pol ranker g =
          htExpEstimator (p := treeUnitPMF model) (pi := pi)
              (treeDistortion model fstar) ω -
            ExpectedTreeDistortion model fstar := by
        ring
      simp only [Set.mem_setOf_eq, harg]
    rw [hset]
    exact htExpEstimator_hoeffding_bound_ennreal
      (p := treeUnitPMF model) (pi := pi) (hpi_pos := hpi_pos)
      (hpi_le := hpi_le) (f := treeDistortion model fstar)
      (M := D_max) (hM := hD_max) (hbound := h_dist_bound)
      (pi_min := pi_min) (hpi_min_pos := hpi_min_pos)
      (hpi_min_le := hpi_min_le) (ε := t) (hε := ht)
  failure_budget := by
    have hnn : (0 : ℝ) ≤ 2 * Real.exp (- t ^ 2 /
        (8 * (Fintype.card (TreeUnit Strings Node A k)) *
          (D_max / pi_min) ^ 2)) := by positivity
    rw [zero_add, ← ENNReal.ofReal_add hnn hnn]
    exact ENNReal.ofReal_le_ofReal (le_of_eq (by ring))

/-- GRPO-PL analogue of `dpo_treepo_certificate_instantiates_error_stack`
(audit finding F7 glue). Paper: thm:e2e. -/
theorem grpo_pl_treepo_certificate_instantiates_error_stack
    {Strings Node A Y : Type*} [Monoid Strings] [PseudoMetricSpace Y]
    {k : ℕ} [Fintype Strings] [Fintype Node] [Fintype A]
    [DecidableEq Strings] [DecidableEq Node] [DecidableEq A]
    (model : OPT.TreePreferenceSamplingModel Strings Node A k)
    (fstar : Strings → Y)
    (pol : Policy' Strings A) (ranker : Strings → GroupRanker A k)
    (g : PMF (Fin k → A))
    (L : ℝ≥0)
    (h_group : ∀ u, model.groupGen u = g)
    (h_pol_lip : GRPOPolicyLipschitz pol fstar L)
    (h_ranker : OracleIndexedRanker ranker fstar)
    (h_rum : ∀ x z,
      ExpectedGRPOLossLipschitz pol ranker fstar g L h_pol_lip h_ranker x z)
    (pi : TreeUnit Strings Node A k → ℝ)
    (hpi_pos : ∀ i, 0 < pi i)
    (hpi_le : ∀ i, pi i ≤ 1)
    (pi_min : ℝ) (hpi_min_pos : 0 < pi_min) (hpi_min_le : ∀ i, pi_min ≤ pi i)
    (D_max : ℝ) (hD_max : 0 ≤ D_max)
    (h_dist_bound : ∀ u, |treeDistortion model fstar u| ≤ D_max)
    (B_cal : ℝ) (hB_cal : 0 < B_cal)
    (t : ℝ) (ht : 0 < t) :
    ∃ s : PaperErrorStack (TreeUnit Strings Node A k → Bool),
      s.μ = bernoulliProductMeasure pi hpi_pos hpi_le ∧
      s.gap_oracle = (fun _ => grpoPLTreeObjectiveGap model pol ranker g) ∧
      s.gap_est = (fun ω =>
        grpoPLTreeObjectiveGap model pol ranker g +
          (htExpEstimator (p := treeUnitPMF model) (pi := pi)
              (treeDistortion model fstar) ω -
            ExpectedTreeDistortion model fstar)) ∧
      s.certificate.localLaw.delta_R = ExpectedTreeDistortion model fstar ∧
      s.certificate.localLaw.methodTransport = (L : ℝ) ∧
      s.certificate.totalObjectiveBound =
        (L : ℝ) * ExpectedTreeDistortion model fstar + B_cal + t + t ∧
      s.delta_est =
        ENNReal.ofReal (2 * Real.exp (- t ^ 2 /
          (8 * (Fintype.card (TreeUnit Strings Node A k)) *
            (D_max / pi_min) ^ 2))) ∧
      s.delta_total =
        ENNReal.ofReal (4 * Real.exp (- t ^ 2 /
          (8 * (Fintype.card (TreeUnit Strings Node A k)) *
            (D_max / pi_min) ^ 2))) :=
  ⟨grpoPLTreePOErrorStack model fstar pol ranker g L h_group h_pol_lip
      h_ranker h_rum pi hpi_pos hpi_le pi_min hpi_min_pos hpi_min_le
      D_max hD_max h_dist_bound B_cal hB_cal t ht,
    rfl, rfl, rfl, rfl, rfl, rfl, rfl, rfl⟩

/-- Final unfolded certificate of the GRPO-PL `PaperErrorStack` instance;
GRPO-PL analogue of `dpo_treepo_error_stack_high_prob`.
Paper: thm:e2e part (d). -/
theorem grpo_pl_treepo_error_stack_high_prob
    {Strings Node A Y : Type*} [Monoid Strings] [PseudoMetricSpace Y]
    {k : ℕ} [Fintype Strings] [Fintype Node] [Fintype A]
    [DecidableEq Strings] [DecidableEq Node] [DecidableEq A]
    (model : OPT.TreePreferenceSamplingModel Strings Node A k)
    (fstar : Strings → Y)
    (pol : Policy' Strings A) (ranker : Strings → GroupRanker A k)
    (g : PMF (Fin k → A))
    (L : ℝ≥0)
    (h_group : ∀ u, model.groupGen u = g)
    (h_pol_lip : GRPOPolicyLipschitz pol fstar L)
    (h_ranker : OracleIndexedRanker ranker fstar)
    (h_rum : ∀ x z,
      ExpectedGRPOLossLipschitz pol ranker fstar g L h_pol_lip h_ranker x z)
    (pi : TreeUnit Strings Node A k → ℝ)
    (hpi_pos : ∀ i, 0 < pi i)
    (hpi_le : ∀ i, pi i ≤ 1)
    (pi_min : ℝ) (hpi_min_pos : 0 < pi_min) (hpi_min_le : ∀ i, pi_min ≤ pi i)
    (D_max : ℝ) (hD_max : 0 ≤ D_max)
    (h_dist_bound : ∀ u, |treeDistortion model fstar u| ≤ D_max)
    (B_cal : ℝ) (hB_cal : 0 < B_cal)
    (t : ℝ) (ht : 0 < t) :
    bernoulliProductMeasure pi hpi_pos hpi_le
      {_ω : TreeUnit Strings Node A k → Bool |
        |grpoPLTreeObjectiveGap model pol ranker g| ≥
          (L : ℝ) * ExpectedTreeDistortion model fstar + B_cal + t + t} ≤
      ENNReal.ofReal (4 * Real.exp (- t ^ 2 /
        (8 * (Fintype.card (TreeUnit Strings Node A k)) *
          (D_max / pi_min) ^ 2))) :=
  (grpoPLTreePOErrorStack model fstar pol ranker g L h_group h_pol_lip
      h_ranker h_rum pi hpi_pos hpi_le pi_min hpi_min_pos hpi_min_le
      D_max hD_max h_dist_bound B_cal hB_cal t ht).high_prob_total

/-! ## GRPO-RL: Realized-Estimator Certificate and Stack Instance -/

/-- Realized-estimator end-to-end TreePO certificate for GRPO-RL with
constant group generator (Paper: thm:e2e (GRPO-RL leg), parts (c)/(d)).
GRPO-RL analogue of `dpo_treepo_realized_estimator_certificate`; the
transport constant is the pointwise-Lipschitz constant `L` of the clipped
surrogate + KL loss (shared by the policy, old-policy, reference-policy and
reward Lipschitz hypotheses), exactly as in
`grpo_rl_treepo_end_to_end_certificate`. Under the hypotheses of that
certificate plus a bounded distortion envelope `D_max` and strict propensity
positivity `pi_min`, for every slack `t > 0` the GRPO-RL objective gap is
bounded by `L * (μ̂_dist(ω) + t)` — the method transport constant times the
**realized** HT distortion estimate plus slack — with probability at least
`1 - δ_est(t)` under the Bernoulli sampling design, where
`δ_est(t) = 2 exp(-t² / (8 N (D_max/pi_min)²))` comes from the in-repo
Hoeffding bound `htExpEstimator_hoeffding_bound`. -/
theorem grpo_rl_treepo_realized_estimator_certificate
    {Strings Node A Y : Type*} [Monoid Strings] [PseudoMetricSpace Y]
    {k : ℕ} [Fintype Strings] [Fintype Node] [Fintype A]
    [DecidableEq Strings] [DecidableEq Node] [DecidableEq A]
    (model : OPT.TreePreferenceSamplingModel Strings Node A k)
    (fstar : Strings → Y)
    (pol pol_old pol_ref : Policy' Strings A)
    (reward : Strings → A → ℝ)
    (eps beta : ℝ)
    (g : PMF (Fin k → A))
    (L : ℝ≥0)
    (h_group : ∀ u, model.groupGen u = g)
    (h_pol_lip : GRPOPolicyLipschitz pol fstar L)
    (h_old_lip : GRPOPolicyLipschitz pol_old fstar L)
    (h_ref_lip : GRPOPolicyLipschitz pol_ref fstar L)
    (h_reward_lip : RewardLipschitzGRPO reward fstar L)
    (h_rum : ∀ x z,
      ExpectedGRPORLLossLipschitz k pol pol_old pol_ref reward eps beta fstar g L
        h_pol_lip h_old_lip h_ref_lip h_reward_lip x z)
    (pi : TreeUnit Strings Node A k → ℝ)
    (hpi_pos : ∀ i, 0 < pi i)
    (hpi_le : ∀ i, pi i ≤ 1)
    (pi_min : ℝ) (hpi_min_pos : 0 < pi_min) (hpi_min_le : ∀ i, pi_min ≤ pi i)
    (D_max : ℝ) (hD_max : 0 ≤ D_max)
    (h_dist_bound : ∀ u, |treeDistortion model fstar u| ≤ D_max)
    (t : ℝ) (ht : 0 < t) :
    let loss : Strings → (Fin k → A) → ℝ :=
      fun x group => GRPORLLossPointwise pol pol_old pol_ref reward eps beta x group
    (bernoulliProductMeasure pi hpi_pos hpi_le).real
      {ω | |ExpectedGRPORLLoss pol pol_old pol_ref reward eps beta model.docDist
              (fun _ => g) -
            OPT.ExpectedTreePreferenceLoss model loss| ≤
          (L : ℝ) *
            (htExpEstimator (p := treeUnitPMF model) (pi := pi)
                (treeDistortion model fstar) ω + t)} ≥
      1 - 2 * Real.exp (- t ^ 2 /
        (8 * (Fintype.card (TreeUnit Strings Node A k)) *
          (D_max / pi_min) ^ 2)) := by
  classical
  dsimp
  have hC : (0 : ℝ) ≤ (L : ℝ) := L.coe_nonneg
  obtain ⟨-, h_gap⟩ :=
    grpo_rl_treepo_end_to_end_certificate
      (model := model) (fstar := fstar)
      (pol := pol) (pol_old := pol_old) (pol_ref := pol_ref)
      (reward := reward) (eps := eps) (beta := beta)
      (g := g) (L := L) (h_group := h_group)
      (h_pol_lip := h_pol_lip) (h_old_lip := h_old_lip)
      (h_ref_lip := h_ref_lip) (h_reward_lip := h_reward_lip)
      (h_rum := h_rum) (pi := pi) (hpi_pos := hpi_pos) (hpi_le := hpi_le)
  rw [ipw_tree_distortion_unbiased model fstar pi hpi_pos hpi_le] at h_gap
  exact abs_gap_le_scaled_realized_htExpEstimator_high_prob
    (p := treeUnitPMF model) (pi := pi) (hpi_pos := hpi_pos) (hpi_le := hpi_le)
    (f := treeDistortion model fstar) (M := D_max) (hM := hD_max)
    (hbound := h_dist_bound) (pi_min := pi_min) (hpi_min_pos := hpi_min_pos)
    (hpi_min_le := hpi_min_le)
    (gap := ExpectedGRPORLLoss pol pol_old pol_ref reward eps beta model.docDist
        (fun _ => g) -
      OPT.ExpectedTreePreferenceLoss model
        (fun x group => GRPORLLossPointwise pol pol_old pol_ref reward eps beta x group))
    (C := (L : ℝ)) (hC := hC) (h_gap := h_gap) (t := t) (ht := ht)

/-- Concrete `PaperErrorStack` instance for GRPO-RL with constant group
generator; GRPO-RL analogue of `dpoTreePOErrorStack` (audit finding F7 glue).
As in the DPO instance, the transport leg is in objective units while the
estimation/clipping legs are carried in distortion-estimator units
(threshold `t`), and the judge-calibration leg is degenerate in the concrete
Lean model (the audited distortion is evaluated by `fstar` itself), so any
`B_cal > 0` is certified with zero failure probability.
Paper: thm:e2e (GRPO-RL leg). -/
def grpoRLTreePOErrorStack
    {Strings Node A Y : Type*} [Monoid Strings] [PseudoMetricSpace Y]
    {k : ℕ} [Fintype Strings] [Fintype Node] [Fintype A]
    [DecidableEq Strings] [DecidableEq Node] [DecidableEq A]
    (model : OPT.TreePreferenceSamplingModel Strings Node A k)
    (fstar : Strings → Y)
    (pol pol_old pol_ref : Policy' Strings A)
    (reward : Strings → A → ℝ)
    (eps beta : ℝ)
    (g : PMF (Fin k → A))
    (L : ℝ≥0)
    (h_group : ∀ u, model.groupGen u = g)
    (h_pol_lip : GRPOPolicyLipschitz pol fstar L)
    (h_old_lip : GRPOPolicyLipschitz pol_old fstar L)
    (h_ref_lip : GRPOPolicyLipschitz pol_ref fstar L)
    (h_reward_lip : RewardLipschitzGRPO reward fstar L)
    (h_rum : ∀ x z,
      ExpectedGRPORLLossLipschitz k pol pol_old pol_ref reward eps beta fstar g L
        h_pol_lip h_old_lip h_ref_lip h_reward_lip x z)
    (pi : TreeUnit Strings Node A k → ℝ)
    (hpi_pos : ∀ i, 0 < pi i)
    (hpi_le : ∀ i, pi i ≤ 1)
    (pi_min : ℝ) (hpi_min_pos : 0 < pi_min) (hpi_min_le : ∀ i, pi_min ≤ pi i)
    (D_max : ℝ) (hD_max : 0 ≤ D_max)
    (h_dist_bound : ∀ u, |treeDistortion model fstar u| ≤ D_max)
    (B_cal : ℝ) (hB_cal : 0 < B_cal)
    (t : ℝ) (ht : 0 < t) :
    PaperErrorStack (TreeUnit Strings Node A k → Bool) where
  μ := bernoulliProductMeasure pi hpi_pos hpi_le
  is_probability :=
    bernoulliProductMeasure_isProbabilityMeasure pi hpi_pos hpi_le
  gap_oracle := fun _ =>
    grpoRLTreeObjectiveGap model pol pol_old pol_ref reward eps beta g
  gap_judge := fun _ =>
    grpoRLTreeObjectiveGap model pol pol_old pol_ref reward eps beta g
  gap_est := fun ω =>
    grpoRLTreeObjectiveGap model pol pol_old pol_ref reward eps beta g +
      (htExpEstimator (p := treeUnitPMF model) (pi := pi)
          (treeDistortion model fstar) ω -
        ExpectedTreeDistortion model fstar)
  gap_clip := fun _ =>
    grpoRLTreeObjectiveGap model pol pol_old pol_ref reward eps beta g
  certificate :=
    { localLaw :=
        { delta_R := ExpectedTreeDistortion model fstar
          methodTransport := (L : ℝ) }
      calibration := B_cal
      estimation := t
      clipping := t }
  delta_cal := 0
  delta_est :=
    ENNReal.ofReal (2 * Real.exp (- t ^ 2 /
      (8 * (Fintype.card (TreeUnit Strings Node A k)) *
        (D_max / pi_min) ^ 2)))
  delta_clip :=
    ENNReal.ofReal (2 * Real.exp (- t ^ 2 /
      (8 * (Fintype.card (TreeUnit Strings Node A k)) *
        (D_max / pi_min) ^ 2)))
  delta_total :=
    ENNReal.ofReal (4 * Real.exp (- t ^ 2 /
      (8 * (Fintype.card (TreeUnit Strings Node A k)) *
        (D_max / pi_min) ^ 2)))
  transport_envelope := by
    intro ω
    obtain ⟨-, h_gap⟩ :=
      grpo_rl_treepo_end_to_end_certificate
        (model := model) (fstar := fstar)
        (pol := pol) (pol_old := pol_old) (pol_ref := pol_ref)
        (reward := reward) (eps := eps) (beta := beta)
        (g := g) (L := L) (h_group := h_group)
        (h_pol_lip := h_pol_lip) (h_old_lip := h_old_lip)
        (h_ref_lip := h_ref_lip) (h_reward_lip := h_reward_lip)
        (h_rum := h_rum) (pi := pi) (hpi_pos := hpi_pos) (hpi_le := hpi_le)
    rw [ipw_tree_distortion_unbiased model fstar pi hpi_pos hpi_le] at h_gap
    exact h_gap
  calibration_event := by
    show bernoulliProductMeasure pi hpi_pos hpi_le
        {ω : TreeUnit Strings Node A k → Bool |
          |grpoRLTreeObjectiveGap model pol pol_old pol_ref reward eps beta g -
            grpoRLTreeObjectiveGap model pol pol_old pol_ref reward eps beta g| ≥
          B_cal} ≤ 0
    have hset :
        {ω : TreeUnit Strings Node A k → Bool |
          |grpoRLTreeObjectiveGap model pol pol_old pol_ref reward eps beta g -
            grpoRLTreeObjectiveGap model pol pol_old pol_ref reward eps beta g| ≥
          B_cal} =
          (∅ : Set (TreeUnit Strings Node A k → Bool)) := by
      apply Set.eq_empty_iff_forall_notMem.mpr
      intro _ω hω
      simp only [Set.mem_setOf_eq, sub_self, abs_zero, ge_iff_le] at hω
      exact absurd hω (not_le.mpr hB_cal)
    rw [hset]
    simp
  estimation_event := by
    show bernoulliProductMeasure pi hpi_pos hpi_le
        {ω : TreeUnit Strings Node A k → Bool |
          |grpoRLTreeObjectiveGap model pol pol_old pol_ref reward eps beta g -
            (grpoRLTreeObjectiveGap model pol pol_old pol_ref reward eps beta g +
              (htExpEstimator (p := treeUnitPMF model) (pi := pi)
                  (treeDistortion model fstar) ω -
                ExpectedTreeDistortion model fstar))| ≥ t} ≤
      ENNReal.ofReal (2 * Real.exp (- t ^ 2 /
        (8 * (Fintype.card (TreeUnit Strings Node A k)) *
          (D_max / pi_min) ^ 2)))
    have hset :
        {ω : TreeUnit Strings Node A k → Bool |
          |grpoRLTreeObjectiveGap model pol pol_old pol_ref reward eps beta g -
            (grpoRLTreeObjectiveGap model pol pol_old pol_ref reward eps beta g +
              (htExpEstimator (p := treeUnitPMF model) (pi := pi)
                  (treeDistortion model fstar) ω -
                ExpectedTreeDistortion model fstar))| ≥ t} =
          {ω : TreeUnit Strings Node A k → Bool |
            |htExpEstimator (p := treeUnitPMF model) (pi := pi)
                (treeDistortion model fstar) ω -
              ExpectedTreeDistortion model fstar| ≥ t} := by
      ext ω
      have harg :
          grpoRLTreeObjectiveGap model pol pol_old pol_ref reward eps beta g -
            (grpoRLTreeObjectiveGap model pol pol_old pol_ref reward eps beta g +
              (htExpEstimator (p := treeUnitPMF model) (pi := pi)
                  (treeDistortion model fstar) ω -
                ExpectedTreeDistortion model fstar)) =
          -(htExpEstimator (p := treeUnitPMF model) (pi := pi)
              (treeDistortion model fstar) ω -
            ExpectedTreeDistortion model fstar) := by
        ring
      simp only [Set.mem_setOf_eq, harg, abs_neg]
    rw [hset]
    exact htExpEstimator_hoeffding_bound_ennreal
      (p := treeUnitPMF model) (pi := pi) (hpi_pos := hpi_pos)
      (hpi_le := hpi_le) (f := treeDistortion model fstar)
      (M := D_max) (hM := hD_max) (hbound := h_dist_bound)
      (pi_min := pi_min) (hpi_min_pos := hpi_min_pos)
      (hpi_min_le := hpi_min_le) (ε := t) (hε := ht)
  clipping_event := by
    show bernoulliProductMeasure pi hpi_pos hpi_le
        {ω : TreeUnit Strings Node A k → Bool |
          |(grpoRLTreeObjectiveGap model pol pol_old pol_ref reward eps beta g +
              (htExpEstimator (p := treeUnitPMF model) (pi := pi)
                  (treeDistortion model fstar) ω -
                ExpectedTreeDistortion model fstar)) -
            grpoRLTreeObjectiveGap model pol pol_old pol_ref reward eps beta g| ≥
          t} ≤
      ENNReal.ofReal (2 * Real.exp (- t ^ 2 /
        (8 * (Fintype.card (TreeUnit Strings Node A k)) *
          (D_max / pi_min) ^ 2)))
    have hset :
        {ω : TreeUnit Strings Node A k → Bool |
          |(grpoRLTreeObjectiveGap model pol pol_old pol_ref reward eps beta g +
              (htExpEstimator (p := treeUnitPMF model) (pi := pi)
                  (treeDistortion model fstar) ω -
                ExpectedTreeDistortion model fstar)) -
            grpoRLTreeObjectiveGap model pol pol_old pol_ref reward eps beta g| ≥
          t} =
          {ω : TreeUnit Strings Node A k → Bool |
            |htExpEstimator (p := treeUnitPMF model) (pi := pi)
                (treeDistortion model fstar) ω -
              ExpectedTreeDistortion model fstar| ≥ t} := by
      ext ω
      have harg :
          (grpoRLTreeObjectiveGap model pol pol_old pol_ref reward eps beta g +
              (htExpEstimator (p := treeUnitPMF model) (pi := pi)
                  (treeDistortion model fstar) ω -
                ExpectedTreeDistortion model fstar)) -
            grpoRLTreeObjectiveGap model pol pol_old pol_ref reward eps beta g =
          htExpEstimator (p := treeUnitPMF model) (pi := pi)
              (treeDistortion model fstar) ω -
            ExpectedTreeDistortion model fstar := by
        ring
      simp only [Set.mem_setOf_eq, harg]
    rw [hset]
    exact htExpEstimator_hoeffding_bound_ennreal
      (p := treeUnitPMF model) (pi := pi) (hpi_pos := hpi_pos)
      (hpi_le := hpi_le) (f := treeDistortion model fstar)
      (M := D_max) (hM := hD_max) (hbound := h_dist_bound)
      (pi_min := pi_min) (hpi_min_pos := hpi_min_pos)
      (hpi_min_le := hpi_min_le) (ε := t) (hε := ht)
  failure_budget := by
    have hnn : (0 : ℝ) ≤ 2 * Real.exp (- t ^ 2 /
        (8 * (Fintype.card (TreeUnit Strings Node A k)) *
          (D_max / pi_min) ^ 2)) := by positivity
    rw [zero_add, ← ENNReal.ofReal_add hnn hnn]
    exact ENNReal.ofReal_le_ofReal (le_of_eq (by ring))

/-- GRPO-RL analogue of `dpo_treepo_certificate_instantiates_error_stack`
(audit finding F7 glue). The witness is `grpoRLTreePOErrorStack`; all listed
identifications hold definitionally. Paper: thm:e2e (GRPO-RL leg). -/
theorem grpo_rl_treepo_certificate_instantiates_error_stack
    {Strings Node A Y : Type*} [Monoid Strings] [PseudoMetricSpace Y]
    {k : ℕ} [Fintype Strings] [Fintype Node] [Fintype A]
    [DecidableEq Strings] [DecidableEq Node] [DecidableEq A]
    (model : OPT.TreePreferenceSamplingModel Strings Node A k)
    (fstar : Strings → Y)
    (pol pol_old pol_ref : Policy' Strings A)
    (reward : Strings → A → ℝ)
    (eps beta : ℝ)
    (g : PMF (Fin k → A))
    (L : ℝ≥0)
    (h_group : ∀ u, model.groupGen u = g)
    (h_pol_lip : GRPOPolicyLipschitz pol fstar L)
    (h_old_lip : GRPOPolicyLipschitz pol_old fstar L)
    (h_ref_lip : GRPOPolicyLipschitz pol_ref fstar L)
    (h_reward_lip : RewardLipschitzGRPO reward fstar L)
    (h_rum : ∀ x z,
      ExpectedGRPORLLossLipschitz k pol pol_old pol_ref reward eps beta fstar g L
        h_pol_lip h_old_lip h_ref_lip h_reward_lip x z)
    (pi : TreeUnit Strings Node A k → ℝ)
    (hpi_pos : ∀ i, 0 < pi i)
    (hpi_le : ∀ i, pi i ≤ 1)
    (pi_min : ℝ) (hpi_min_pos : 0 < pi_min) (hpi_min_le : ∀ i, pi_min ≤ pi i)
    (D_max : ℝ) (hD_max : 0 ≤ D_max)
    (h_dist_bound : ∀ u, |treeDistortion model fstar u| ≤ D_max)
    (B_cal : ℝ) (hB_cal : 0 < B_cal)
    (t : ℝ) (ht : 0 < t) :
    ∃ s : PaperErrorStack (TreeUnit Strings Node A k → Bool),
      s.μ = bernoulliProductMeasure pi hpi_pos hpi_le ∧
      s.gap_oracle = (fun _ =>
        grpoRLTreeObjectiveGap model pol pol_old pol_ref reward eps beta g) ∧
      s.gap_est = (fun ω =>
        grpoRLTreeObjectiveGap model pol pol_old pol_ref reward eps beta g +
          (htExpEstimator (p := treeUnitPMF model) (pi := pi)
              (treeDistortion model fstar) ω -
            ExpectedTreeDistortion model fstar)) ∧
      s.certificate.localLaw.delta_R = ExpectedTreeDistortion model fstar ∧
      s.certificate.localLaw.methodTransport = (L : ℝ) ∧
      s.certificate.totalObjectiveBound =
        (L : ℝ) * ExpectedTreeDistortion model fstar + B_cal + t + t ∧
      s.delta_est =
        ENNReal.ofReal (2 * Real.exp (- t ^ 2 /
          (8 * (Fintype.card (TreeUnit Strings Node A k)) *
            (D_max / pi_min) ^ 2))) ∧
      s.delta_total =
        ENNReal.ofReal (4 * Real.exp (- t ^ 2 /
          (8 * (Fintype.card (TreeUnit Strings Node A k)) *
            (D_max / pi_min) ^ 2))) :=
  ⟨grpoRLTreePOErrorStack model fstar pol pol_old pol_ref reward eps beta g L
      h_group h_pol_lip h_old_lip h_ref_lip h_reward_lip h_rum pi hpi_pos
      hpi_le pi_min hpi_min_pos hpi_min_le D_max hD_max h_dist_bound
      B_cal hB_cal t ht,
    rfl, rfl, rfl, rfl, rfl, rfl, rfl, rfl⟩

/-- Final unfolded certificate of the GRPO-RL `PaperErrorStack` instance;
GRPO-RL analogue of `dpo_treepo_error_stack_high_prob`.
Paper: thm:e2e (GRPO-RL leg), part (d). -/
theorem grpo_rl_treepo_error_stack_high_prob
    {Strings Node A Y : Type*} [Monoid Strings] [PseudoMetricSpace Y]
    {k : ℕ} [Fintype Strings] [Fintype Node] [Fintype A]
    [DecidableEq Strings] [DecidableEq Node] [DecidableEq A]
    (model : OPT.TreePreferenceSamplingModel Strings Node A k)
    (fstar : Strings → Y)
    (pol pol_old pol_ref : Policy' Strings A)
    (reward : Strings → A → ℝ)
    (eps beta : ℝ)
    (g : PMF (Fin k → A))
    (L : ℝ≥0)
    (h_group : ∀ u, model.groupGen u = g)
    (h_pol_lip : GRPOPolicyLipschitz pol fstar L)
    (h_old_lip : GRPOPolicyLipschitz pol_old fstar L)
    (h_ref_lip : GRPOPolicyLipschitz pol_ref fstar L)
    (h_reward_lip : RewardLipschitzGRPO reward fstar L)
    (h_rum : ∀ x z,
      ExpectedGRPORLLossLipschitz k pol pol_old pol_ref reward eps beta fstar g L
        h_pol_lip h_old_lip h_ref_lip h_reward_lip x z)
    (pi : TreeUnit Strings Node A k → ℝ)
    (hpi_pos : ∀ i, 0 < pi i)
    (hpi_le : ∀ i, pi i ≤ 1)
    (pi_min : ℝ) (hpi_min_pos : 0 < pi_min) (hpi_min_le : ∀ i, pi_min ≤ pi i)
    (D_max : ℝ) (hD_max : 0 ≤ D_max)
    (h_dist_bound : ∀ u, |treeDistortion model fstar u| ≤ D_max)
    (B_cal : ℝ) (hB_cal : 0 < B_cal)
    (t : ℝ) (ht : 0 < t) :
    bernoulliProductMeasure pi hpi_pos hpi_le
      {_ω : TreeUnit Strings Node A k → Bool |
        |grpoRLTreeObjectiveGap model pol pol_old pol_ref reward eps beta g| ≥
          (L : ℝ) * ExpectedTreeDistortion model fstar + B_cal + t + t} ≤
      ENNReal.ofReal (4 * Real.exp (- t ^ 2 /
        (8 * (Fintype.card (TreeUnit Strings Node A k)) *
          (D_max / pi_min) ^ 2))) :=
  (grpoRLTreePOErrorStack model fstar pol pol_old pol_ref reward eps beta g L
      h_group h_pol_lip h_old_lip h_ref_lip h_reward_lip h_rum pi hpi_pos
      hpi_le pi_min hpi_min_pos hpi_min_le D_max hD_max h_dist_bound
      B_cal hB_cal t ht).high_prob_total
