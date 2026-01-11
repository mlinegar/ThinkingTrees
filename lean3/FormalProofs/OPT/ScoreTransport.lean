import FormalProofs.OPT.GlobalAssumptions

/-!
# FormalProofs/ScoreTransport.lean

## Paper Reference: Section 8 (Score Transport)

This file formalizes the score transport results from Section 8 of the paper:

- **Proposition 4** (`prop4_cf_implies_oracle_measurable`): Conditional factorization
  implies oracle-measurability. When P(A|X) factors through f*(X), the preference score
  depends only on the oracle value.

- **Proposition 5** (`prop5_score_transport`): Score transport via oracle σ-algebra.
  When σ(f*(X)) ⊆ σ(Z), the score can be transported from X to Z while preserving
  its conditional expectation structure.

Key definitions:
- `ConditionalFactorization'`: CF condition from the paper
- `OracleSigmaSubset'`: σ(f*(X)) ⊆ σ(Z) containment
- `SummaryScore'`: E[S*(X,a) | Z]
-/

set_option linter.mathlibStandardSet false

open scoped BigOperators
open scoped Real
open scoped Nat
open scoped Classical
open scoped Pointwise

set_option maxHeartbeats 0
set_option maxRecDepth 4000
set_option relaxedAutoImplicit false
set_option autoImplicit false

noncomputable section

section ScoreTransport

open MeasureTheory ProbabilityTheory Set Filter TopologicalSpace
open scoped ENNReal MeasureTheory ProbabilityTheory

-- Probability space with base sigma-algebra m₀
variable {Ω : Type*} {m₀ : MeasurableSpace Ω}
variable {μ : @Measure Ω m₀} [IsProbabilityMeasure μ]

-- Document space (needs measurable structure for measure theory)
variable {Strings' : Type*} [MeasurableSpace Strings'] [Monoid Strings']

-- Oracle space (with both metric and measurable structure)
variable {Y' : Type*} [PseudoMetricSpace Y'] [MeasurableSpace Y'] [BorelSpace Y']

-- Action space
variable {A : Type*}

/-!
## Conditional Factorization
-/

/-- CF: Supervision depends on document only through oracle value -/
def ConditionalFactorization'
    (Sstar : Strings' → A → ℝ) (fstar : Strings' → Y')
    (X : Ω → Strings') (μ : @Measure Ω m₀) (_hX : @Measurable Ω Strings' m₀ _ X) : Prop :=
  ∃ Qbar : Y' → A → ℝ,
    (∀ a, Measurable (fun y => Qbar y a)) ∧
    ∀ a : A,
      μ[fun ω => Sstar (X ω) a | MeasurableSpace.comap X ‹_›]
        =ᵐ[μ] fun ω => Qbar (fstar (X ω)) a

/-- Summary-indexed score: S_Z(Z, a) := E[S*(X, a) | Z] -/
def SummaryScore' (Sstar : Strings' → A → ℝ) (X Z : Ω → Strings')
    (μ : @Measure Ω m₀) (a : A) : Ω → ℝ :=
  μ[fun ω => Sstar (X ω) a | MeasurableSpace.comap Z ‹_›]

/-- σ(f*(X)) ⊆ σ(Z): Oracle information is preserved -/
def OracleSigmaSubset' (fstar : Strings' → Y') (X Z : Ω → Strings') : Prop :=
  MeasurableSpace.comap (fstar ∘ X) ‹_› ≤ MeasurableSpace.comap Z ‹_›

/-!
## Deriving σ-algebra containment from zero distortion

When the distortion D(Z, X) = 0 almost surely, we have f*(Z) = f*(X) a.s.,
which implies that f*(X) is σ(Z)-measurable, hence σ(f*(X)) ⊆ σ(Z).
-/

/-- Zero distortion implies oracle values are equal a.e.

    This lemma requires Y' to be a proper metric space (where dist=0 implies equality),
    not just a pseudo-metric space. In the paper, oracle spaces are assumed to be metric spaces.

    Proof: In a MetricSpace, dist x y = 0 ↔ x = y (by eq_of_dist_eq_zero).
    The a.e. version follows by filtering.

    Note: The hypothesis uses the section's PseudoMetricSpace instance. We require
    that this coincides with a MetricSpace structure (T0Space separates points). -/
lemma zero_distortion_implies_fstar_eq
    [T0Space Y'] (fstar : Strings' → Y') (X Z : Ω → Strings')
    (h_zero : ∀ᵐ ω ∂μ, dist (fstar (Z ω)) (fstar (X ω)) = 0) :
    ∀ᵐ ω ∂μ, fstar (Z ω) = fstar (X ω) := by
  filter_upwards [h_zero] with ω hω
  -- In a PseudoMetricSpace with T0Space, dist=0 implies Inseparable, which implies equality
  rw [← inseparable_iff_eq]
  exact Metric.inseparable_iff.mpr hω

/-- When f*(Z) = f*(X) a.e. and Z is measurable, f*(X) is σ(Z)-measurable.
    This is because f*(X)(ω) = f*(Z)(ω) a.e., and f* ∘ Z is σ(Z)-measurable.

    Note: Requires SecondCountableTopology Y' for aestronglyMeasurable. -/
lemma fstar_X_meas_of_fstar_eq
    [SecondCountableTopology Y'] (fstar : Strings' → Y') (X Z : Ω → Strings')
    (hZ : @Measurable Ω Strings' m₀ _ Z)
    (hfstar : Measurable fstar)
    (h_eq : ∀ᵐ ω ∂μ, fstar (Z ω) = fstar (X ω)) :
    AEStronglyMeasurable (fstar ∘ X) μ := by
  -- f* ∘ Z is measurable, hence AEStronglyMeasurable
  have hfZ : AEStronglyMeasurable (fstar ∘ Z) μ := (hfstar.comp hZ).aestronglyMeasurable
  -- f* ∘ X =ᵐ f* ∘ Z by h_eq, so f* ∘ X is AEStronglyMeasurable
  refine AEStronglyMeasurable.congr hfZ ?_
  filter_upwards [h_eq] with ω hω
  simp only [Function.comp_apply]
  exact hω

/-- When f*(Z) = f*(X) holds everywhere (not just a.e.), σ(f*(X)) ⊆ σ(Z).
    This is the key lemma connecting zero distortion to σ-algebra containment.

    Proof idea: Since fstar ∘ X = fstar ∘ Z pointwise, comap (fstar ∘ X) = comap (fstar ∘ Z).
    And comap (fstar ∘ Z) ≤ comap Z because:
    - A set s is in comap (fstar ∘ Z) iff s = (fstar ∘ Z)⁻¹(A) for some measurable A in Y'
    - (fstar ∘ Z)⁻¹(A) = Z⁻¹(fstar⁻¹(A))
    - Since fstar is measurable, fstar⁻¹(A) is measurable in Strings'
    - Therefore s is in comap Z

    Note: The main theorems (blackwell_transport', condexp_oracle_factored') take
    OracleSigmaSubset' as a hypothesis, so this derivation lemma is auxiliary. -/
lemma sigma_subset_of_fstar_eq_pointwise
    (fstar : Strings' → Y') (X Z : Ω → Strings')
    (hfstar : Measurable fstar)
    (h_eq : ∀ ω, fstar (Z ω) = fstar (X ω)) :
    OracleSigmaSubset' fstar X Z := by
  unfold OracleSigmaSubset'
  -- Since fstar ∘ X = fstar ∘ Z pointwise, comap (fstar ∘ X) = comap (fstar ∘ Z)
  have h_comp_eq : fstar ∘ X = fstar ∘ Z := funext (fun ω => (h_eq ω).symm)
  rw [h_comp_eq]
  -- comap (fstar ∘ Z) ≤ comap Z by:
  -- 1. comap (fstar ∘ Z) m_Y' = comap Z (comap fstar m_Y') [by comap_comp]
  -- 2. comap fstar m_Y' ≤ m_Strings' [by Measurable.comap_le since fstar is measurable]
  -- 3. comap Z (comap fstar m_Y') ≤ comap Z m_Strings' [by comap_mono]
  calc MeasurableSpace.comap (fstar ∘ Z) _
      = MeasurableSpace.comap Z (MeasurableSpace.comap fstar _) := MeasurableSpace.comap_comp.symm
    _ ≤ MeasurableSpace.comap Z _ := MeasurableSpace.comap_mono hfstar.comap_le

/-!
## Blackwell/Doob-Dynkin Score Transport
-/

/-- Blackwell/Doob-Dynkin Score Transport Lemma -/
theorem blackwell_transport'
    (Sstar : Strings' → A → ℝ) (fstar : Strings' → Y') (X Z : Ω → Strings')
    (hX : @Measurable Ω Strings' m₀ _ X) (hZ : @Measurable Ω Strings' m₀ _ Z)
    (_hfstar : Measurable fstar)
    [_hμZ : SigmaFinite (μ.trim hZ.comap_le)]
    [_hμX : SigmaFinite (μ.trim hX.comap_le)]
    (hCF : ConditionalFactorization' Sstar fstar X μ hX)
    (_hσ : OracleSigmaSubset' fstar X Z)
    (a : A) (hint : Integrable (fun ω => Sstar (X ω) a) μ) :
    -- Part 1: Tower property - E[S*] = E[E[S*|Z]]
    ∫ ω, Sstar (X ω) a ∂μ = ∫ ω, SummaryScore' Sstar X Z μ a ω ∂μ ∧
    -- Part 2: Factorization via CF - E[S*] = E[Qbar(f*(X))]
    ∫ ω, Sstar (X ω) a ∂μ = ∫ ω, hCF.choose (fstar (X ω)) a ∂μ := by
  have _hQbar_meas := hCF.choose_spec.1
  have hQbar_eq := hCF.choose_spec.2
  constructor
  -- Part 1: E[S*] = E[E[S*|Z]] by tower property
  · unfold SummaryScore'
    exact (MeasureTheory.integral_condExp hZ.comap_le).symm
  -- Part 2: E[S*] = E[Qbar(f*(X))] via CF
  · rw [← MeasureTheory.integral_condExp hX.comap_le]
    exact integral_congr_ae (hQbar_eq a)

/-- When σ(f*(X)) ⊆ σ(Z) ⊆ σ(X), conditioning on Z recovers the oracle-factored form -/
lemma condexp_oracle_factored'
    (Sstar : Strings' → A → ℝ) (fstar : Strings' → Y') (X Z : Ω → Strings')
    (_hX : @Measurable Ω Strings' m₀ _ X) (hZ : @Measurable Ω Strings' m₀ _ Z)
    (_hfstar : Measurable fstar)
    [_hμZ : SigmaFinite (μ.trim hZ.comap_le)]
    [_hμX : SigmaFinite (μ.trim _hX.comap_le)]
    (hσ : OracleSigmaSubset' fstar X Z)
    (hσ_ZX : MeasurableSpace.comap Z ‹_› ≤ MeasurableSpace.comap X ‹_›)
    (hCF : ConditionalFactorization' Sstar fstar X μ _hX)
    (a : A) (hint : Integrable (fun ω => Sstar (X ω) a) μ) :
    SummaryScore' Sstar X Z μ a =ᵐ[μ] fun ω => hCF.choose (fstar (X ω)) a := by
  -- Setup: Extract Qbar and its properties from CF
  let Qbar := hCF.choose
  have hQbar_meas : ∀ a', Measurable (fun y => Qbar y a') := hCF.choose_spec.1
  have hQbar_ae : μ[fun ω => Sstar (X ω) a | MeasurableSpace.comap X ‹_›] =ᵐ[μ]
      fun ω => Qbar (fstar (X ω)) a := hCF.choose_spec.2 a
  -- Setup sigma-algebra inclusions
  let mZ := MeasurableSpace.comap Z ‹_›
  let mX := MeasurableSpace.comap X ‹_›
  have hmX_le_m0 : mX ≤ m₀ := _hX.comap_le
  have hmZ_le_m0 : mZ ≤ m₀ := hZ.comap_le
  -- Unfold definition
  unfold SummaryScore'
  -- Step 1: Tower property gives us E[E[f|mX]|mZ] =ᵐ[μ] E[f|mZ]
  have tower : μ[μ[fun ω => Sstar (X ω) a | mX] | mZ] =ᵐ[μ] μ[fun ω => Sstar (X ω) a | mZ] :=
    condExp_condExp_of_le hσ_ZX hmX_le_m0
  -- Step 2: Apply CF - replace inner condexp with Qbar(f*(X), a)
  have step2 : μ[μ[fun ω => Sstar (X ω) a | mX] | mZ] =ᵐ[μ] μ[fun ω => Qbar (fstar (X ω)) a | mZ] :=
    condExp_congr_ae hQbar_ae
  -- Step 3: Show Qbar(f*(X), a) is mZ-strongly measurable
  -- From hσ : σ(f*(X)) ≤ σ(Z), we get fstar ∘ X is mZ-measurable
  -- Combined with measurability of Qbar(·,a), the composition is mZ-measurable
  have hQbar_fstarX_meas : StronglyMeasurable[mZ] (fun ω => Qbar (fstar (X ω)) a) := by
    apply Measurable.stronglyMeasurable
    -- fstar ∘ X is mZ-measurable by hσ (OracleSigmaSubset')
    have hfX_mZ : Measurable[mZ] (fstar ∘ X) := by
      -- OracleSigmaSubset' says comap (fstar ∘ X) _ ≤ comap Z _ = mZ
      -- This means fstar ∘ X is mZ-measurable
      exact MeasurableSpace.comap_le_iff_le_map.mp hσ
    exact (hQbar_meas a).comp hfX_mZ
  -- Step 4: Show integrability of Qbar(f*(X), a) for self-conditioning
  have hQbar_int : Integrable (fun ω => Qbar (fstar (X ω)) a) μ := by
    -- This follows from hQbar_ae (conditional expectation is integrable)
    exact (integrable_congr hQbar_ae).mp integrable_condExp
  -- Step 5: Self-conditioning - E[f|mZ] = f when f is mZ-measurable and integrable
  have step5 : μ[fun ω => Qbar (fstar (X ω)) a | mZ] = fun ω => Qbar (fstar (X ω)) a :=
    condExp_of_stronglyMeasurable hmZ_le_m0 hQbar_fstarX_meas hQbar_int
  -- Combine: E[S*|mZ] =ᵐ tower E[E[S*|mX]|mZ] =ᵐ step2 E[Qbar|mZ] = step5 Qbar
  calc μ[fun ω => Sstar (X ω) a | mZ]
      =ᵐ[μ] μ[μ[fun ω => Sstar (X ω) a | mX] | mZ] := tower.symm
    _ =ᵐ[μ] μ[fun ω => Qbar (fstar (X ω)) a | mZ] := step2
    _ = fun ω => Qbar (fstar (X ω)) a := step5

/-!
## Paper Propositions

These theorems give explicit names matching the paper's proposition numbering.
-/

/-- **Proposition 4: Conditional Factorization Implies Oracle-Measurability**

**Paper Reference:** Section 8, Proposition 4

When the supervision signal S*(X, a) factors through the oracle f*(X), meaning:
  E[S*(X, a) | X] = Q̄(f*(X), a)  for some measurable Q̄

Then the effective score function is oracle-measurable: it depends on X only through f*(X).

This is the fundamental "sufficiency" property for score transport: conditional
factorization ensures that supervision information can be represented purely in
terms of oracle values, enabling lossless summarization. -/
theorem prop4_cf_implies_oracle_measurable
    (Sstar : Strings' → A → ℝ) (fstar : Strings' → Y') (X : Ω → Strings')
    (hX : @Measurable Ω Strings' m₀ _ X)
    (hCF : ConditionalFactorization' Sstar fstar X μ hX) :
    ∃ (Qbar : Y' → A → ℝ),
      (∀ a, Measurable (fun y => Qbar y a)) ∧
      ∀ a, μ[fun ω => Sstar (X ω) a | MeasurableSpace.comap X ‹_›]
           =ᵐ[μ] fun ω => Qbar (fstar (X ω)) a :=
  hCF

/-- **Proposition 5: Score Transport via Oracle σ-algebra**

**Paper Reference:** Section 8, Proposition 5

When:
1. CF holds: supervision factors through oracle (Proposition 4)
2. σ(f*(X)) ⊆ σ(Z): oracle information is preserved in Z

Then scores can be transported: the expected score under the original distribution
equals the expected score under the summarized distribution.

This theorem combines the tower property with conditional factorization to show
that summarization preserves the gradient signal for training.

**Mathematical Statement:**
  E_X[S*(X, a)] = E_Z[S_Z(Z, a)] = E_X[Q̄(f*(X), a)]

Part 1: Tower property - expectation is preserved under conditioning
Part 2: CF application - expectation factors through oracle -/
theorem prop5_score_transport
    (Sstar : Strings' → A → ℝ) (fstar : Strings' → Y') (X Z : Ω → Strings')
    (hX : @Measurable Ω Strings' m₀ _ X) (hZ : @Measurable Ω Strings' m₀ _ Z)
    (hfstar : Measurable fstar)
    [hμZ : SigmaFinite (μ.trim hZ.comap_le)]
    [hμX : SigmaFinite (μ.trim hX.comap_le)]
    (hCF : ConditionalFactorization' Sstar fstar X μ hX)
    (hσ : OracleSigmaSubset' fstar X Z)
    (a : A) (hint : Integrable (fun ω => Sstar (X ω) a) μ) :
    -- Score transport: E[S*] = E[S_Z] = E[Qbar(f*(X))]
    ∫ ω, Sstar (X ω) a ∂μ = ∫ ω, SummaryScore' Sstar X Z μ a ω ∂μ ∧
    ∫ ω, Sstar (X ω) a ∂μ = ∫ ω, hCF.choose (fstar (X ω)) a ∂μ :=
  blackwell_transport' Sstar fstar X Z hX hZ hfstar hCF hσ a hint

/-- **Proposition 5 (Corollary): Score Factorization Under Nesting**

**Paper Reference:** Section 8, Proposition 5 (Corollary)

When σ(f*(X)) ⊆ σ(Z) ⊆ σ(X) (oracle ⊆ summary ⊆ original), conditioning on Z
exactly recovers the oracle-factored score:

  E[S*(X, a) | Z] =ᵐ Q̄(f*(X), a)

This is the "lossless" property: the summary Z contains all relevant oracle
information, so conditioning on Z gives the same result as conditioning on
the full oracle. -/
theorem prop5_score_factorization_corollary
    (Sstar : Strings' → A → ℝ) (fstar : Strings' → Y') (X Z : Ω → Strings')
    (hX : @Measurable Ω Strings' m₀ _ X) (hZ : @Measurable Ω Strings' m₀ _ Z)
    (hfstar : Measurable fstar)
    [hμZ : SigmaFinite (μ.trim hZ.comap_le)]
    [hμX : SigmaFinite (μ.trim hX.comap_le)]
    (hσ : OracleSigmaSubset' fstar X Z)
    (hσ_ZX : MeasurableSpace.comap Z ‹_› ≤ MeasurableSpace.comap X ‹_›)
    (hCF : ConditionalFactorization' Sstar fstar X μ hX)
    (a : A) (hint : Integrable (fun ω => Sstar (X ω) a) μ) :
    SummaryScore' Sstar X Z μ a =ᵐ[μ] fun ω => hCF.choose (fstar (X ω)) a :=
  condexp_oracle_factored' Sstar fstar X Z hX hZ hfstar hσ hσ_ZX hCF a hint

end ScoreTransport

end
