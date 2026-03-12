import FormalProofs.OPT.SketchRecovery
import FormalProofs.OPT.MarkovCountSketchExample
import FormalProofs.OPT.TopicBigramOracle
import FormalProofs.OPT.AdaptiveChunkingBridge

/-!
# FormalProofs/OPT/SketchRecoveryInstances.lean

Concrete instance templates showing that the generic sketch-recovery stack can be
instantiated in two settings:

1. Markov changepoint sketch state (`MarkovCountSketch`)
2. Topic unigram+bigram sketch features (`uniBigramSketch`)

Both use the identity summary operator and an encoded feature oracle:
`x ↦ Nat.encode (feature x) : ℝ`.

This keeps the instantiations flexible: any downstream objective theorem from
`SketchRecovery.lean` can be reused by swapping the feature map.
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

namespace FormalProofs.OPT

section MarkovInstances

variable {n : ℕ}
variable [Encodable (MarkovCountSketch n)]

/-- Identity sketch operator on Markov sketch states. -/
abbrev markovIdentityOp :
    SketchOperator (MarkovCountSketch n) (MarkovCountSketch n) :=
  identitySketchOperator (Strings := MarkovCountSketch n)

/-- Feature map used for encoded-oracle instantiation (full state). -/
def markovFeature : MarkovCountSketch n → MarkovCountSketch n := fun s => s

/-- Markov feature map is congruent under multiplication. -/
lemma markovFeature_congruent :
    ∀ sL sR x y,
      markovFeature (n := n) sL = markovFeature (n := n) x →
      markovFeature (n := n) sR = markovFeature (n := n) y →
      markovFeature (n := n) (sL * sR) = markovFeature (n := n) (x * y) := by
  intro sL sR x y hL hR
  have hs : sL = x := by
    simpa [markovFeature] using hL
  have ht : sR = y := by
    simpa [markovFeature] using hR
  simp [markovFeature, hs, ht]

/-- Local laws for Markov identity operator under encoded full-state oracle. -/
theorem markov_local_laws_of_encoded_feature (T : BinTree (MarkovCountSketch n)) :
    LocalLawsBundle (sketchSummarizer (markovIdentityOp (n := n))) T
      (encodedOracle (Strings := MarkovCountSketch n) (markovFeature (n := n))) := by
  simpa [markovIdentityOp] using
    (local_laws_of_identity_encoded_feature
      (Strings := MarkovCountSketch n)
      (feature := markovFeature (n := n))
      markovFeature_congruent
      (T := T))

/-- Generic pairwise preference equivalence via ZR for Markov encoded-feature oracle. -/
theorem markov_preference_equivalence_via_ZR_of_encoded_feature
    {A : Type*}
    (loss : MarkovCountSketch n → A → A → ℝ)
    (gen : PairGenerator (MarkovCountSketch n) A)
    (x : MarkovCountSketch n) (R : ℕ) (T : BinTree (MarkovCountSketch n))
    (hp : S T = x) (hR : R ≥ 1)
    (h_meas :
      OracleMeasurableLoss loss
        (encodedOracle (Strings := MarkovCountSketch n) (markovFeature (n := n))))
    (h_pair :
      OracleIndexedPairGen gen
        (encodedOracle (Strings := MarkovCountSketch n) (markovFeature (n := n))))
    (M : ℝ) (hM : 0 ≤ M)
    (hbound :
      ∀ w z,
        D (encodedOracle (Strings := MarkovCountSketch n) (markovFeature (n := n))) w z ≤ M) :
    ∑' x', (PMF.pure x x').toReal * ∑' p, (gen x' p).toReal * loss x' p.1 p.2 =
    ∑' z, (ZR (sketchSummarizer (markovIdentityOp (n := n))) x R T z).toReal *
      ∑' p, (gen z p).toReal * loss z p.1 p.2 := by
  simpa [markovIdentityOp] using
    (preference_learning_equivalence_via_ZR_of_identity_encoded_feature
      (Strings := MarkovCountSketch n)
      (feature := markovFeature (n := n))
      markovFeature_congruent
      (loss := loss) (gen := gen) (x := x) (R := R) (T := T)
      hp hR h_meas h_pair M hM hbound)

end MarkovInstances

section TopicInstances

variable {α : Type*} [DecidableEq α]
variable [Encodable (UniBigramSketch α)]

/-- Identity sketch operator on token lists (using list concatenation monoid). -/
abbrev topicListIdentityOp : SketchOperator (List α) (List α) :=
  identitySketchOperator (Strings := List α)

/-- Topic feature map used for encoded-oracle instantiation. -/
def topicFeature (xs : List α) : UniBigramSketch α := uniBigramSketch xs

/-- Topic feature map is congruent under concatenation. -/
lemma topicFeature_congruent :
    ∀ sL sR x y,
      topicFeature (α := α) sL = topicFeature (α := α) x →
      topicFeature (α := α) sR = topicFeature (α := α) y →
      topicFeature (α := α) (sL * sR) = topicFeature (α := α) (x * y) := by
  intro sL sR x y hL hR
  calc
    topicFeature (α := α) (sL * sR)
        = mergeUniBigramSketch (topicFeature (α := α) sL) (topicFeature (α := α) sR) := by
            simpa [topicFeature] using (uniBigramSketch_append (xs := sL) (ys := sR))
    _ = mergeUniBigramSketch (topicFeature (α := α) x) (topicFeature (α := α) y) := by
            simp [hL, hR]
    _ = topicFeature (α := α) (x * y) := by
            simpa [topicFeature] using (uniBigramSketch_append (xs := x) (ys := y)).symm

/-- Local laws for list identity operator under encoded topic-feature oracle. -/
theorem topic_local_laws_of_encoded_feature (T : BinTree (List α)) :
    LocalLawsBundle (sketchSummarizer (topicListIdentityOp (α := α))) T
      (encodedOracle (Strings := List α) (topicFeature (α := α))) := by
  simpa [topicListIdentityOp] using
    (local_laws_of_identity_encoded_feature
      (Strings := List α)
      (feature := topicFeature (α := α))
      topicFeature_congruent
      (T := T))

/-- Generic pairwise preference equivalence via ZR for topic encoded-feature oracle. -/
theorem topic_preference_equivalence_via_ZR_of_encoded_feature
    {A : Type*}
    (loss : List α → A → A → ℝ)
    (gen : PairGenerator (List α) A)
    (x : List α) (R : ℕ) (T : BinTree (List α))
    (hp : S T = x) (hR : R ≥ 1)
    (h_meas :
      OracleMeasurableLoss loss
        (encodedOracle (Strings := List α) (topicFeature (α := α))))
    (h_pair :
      OracleIndexedPairGen gen
        (encodedOracle (Strings := List α) (topicFeature (α := α))))
    (M : ℝ) (hM : 0 ≤ M)
    (hbound :
      ∀ w z,
        D (encodedOracle (Strings := List α) (topicFeature (α := α))) w z ≤ M) :
    ∑' x', (PMF.pure x x').toReal * ∑' p, (gen x' p).toReal * loss x' p.1 p.2 =
    ∑' z, (ZR (sketchSummarizer (topicListIdentityOp (α := α))) x R T z).toReal *
      ∑' p, (gen z p).toReal * loss z p.1 p.2 := by
  simpa [topicListIdentityOp] using
    (preference_learning_equivalence_via_ZR_of_identity_encoded_feature
      (Strings := List α)
      (feature := topicFeature (α := α))
      topicFeature_congruent
      (loss := loss) (gen := gen) (x := x) (R := R) (T := T)
      hp hR h_meas h_pair M hM hbound)

end TopicInstances

section LengthInstances

variable {α : Type*}

/-- Identity sketch operator on token lists. -/
abbrev listIdentityOp : SketchOperator (List α) (List α) :=
  identitySketchOperator (Strings := List α)

/-- Length feature on lists. -/
def lengthFeature (xs : List α) : Nat := xs.length

/-- Induced summarizer for the genuinely lossy length sketch operator. -/
abbrev lossyLengthSummarizer [Inhabited α] : Summarizer (List α) :=
  sketchSummarizer (lengthSketchOperator (α := α))

/-- Length feature is congruent under list concatenation. -/
lemma lengthFeature_congruent :
    ∀ sL sR x y,
      lengthFeature (α := α) sL = lengthFeature (α := α) x →
      lengthFeature (α := α) sR = lengthFeature (α := α) y →
      lengthFeature (α := α) (sL * sR) = lengthFeature (α := α) (x * y) := by
  intro sL sR x y hL hR
  have hLL : sL.length = x.length := by simpa [lengthFeature] using hL
  have hRR : sR.length = y.length := by simpa [lengthFeature] using hR
  change (sL ++ sR).length = (x ++ y).length
  simp [List.length_append, hLL, hRR]

/-- Local laws for list identity operator under encoded length oracle. -/
theorem length_local_laws_of_encoded_feature (T : BinTree (List α)) :
    LocalLawsBundle (sketchSummarizer (listIdentityOp (α := α))) T
      (encodedOracle (Strings := List α) (lengthFeature (α := α))) := by
  simpa [listIdentityOp] using
    (local_laws_of_identity_encoded_feature
      (Strings := List α)
      (feature := lengthFeature (α := α))
      lengthFeature_congruent
      (T := T))

/-- Local laws for the paired non-identity sketch operator under encoded length oracle. -/
theorem length_local_laws_of_paired_encoded_feature (T : BinTree (List α)) :
    LocalLawsBundle (sketchSummarizer (pairedSketchOperator (Strings := List α))) T
      (encodedOracle (Strings := List α) (lengthFeature (α := α))) := by
  exact local_laws_of_paired_encoded_feature
    (Strings := List α)
    (feature := lengthFeature (α := α))
    lengthFeature_congruent
    (T := T)

/-- Local laws for the genuinely lossy length sketch operator under encoded length oracle. -/
theorem length_local_laws_of_lossy_encoded_feature [Inhabited α] (T : BinTree (List α)) :
    LocalLawsBundle (sketchSummarizer (lengthSketchOperator (α := α))) T
      (encodedOracle (Strings := List α) (lengthFeature (α := α))) := by
  exact local_laws_of_sketch
    (op := lengthSketchOperator (α := α))
    (fstar := encodedOracle (Strings := List α) (lengthFeature (α := α)))
    (T := T)
    (lengthSketch_leaf_preserving (α := α))
    (lengthSketch_merge_compatible (α := α))
    (lengthSketch_summary_compatible (α := α))

/-- Generic pairwise preference equivalence via ZR for encoded length oracle. -/
theorem length_preference_equivalence_via_ZR_of_encoded_feature
    {A : Type*}
    (loss : List α → A → A → ℝ)
    (gen : PairGenerator (List α) A)
    (x : List α) (R : ℕ) (T : BinTree (List α))
    (hp : S T = x) (hR : R ≥ 1)
    (h_meas :
      OracleMeasurableLoss loss
        (encodedOracle (Strings := List α) (lengthFeature (α := α))))
    (h_pair :
      OracleIndexedPairGen gen
        (encodedOracle (Strings := List α) (lengthFeature (α := α))))
    (M : ℝ) (hM : 0 ≤ M)
    (hbound :
      ∀ w z,
        D (encodedOracle (Strings := List α) (lengthFeature (α := α))) w z ≤ M) :
    ∑' x', (PMF.pure x x').toReal * ∑' p, (gen x' p).toReal * loss x' p.1 p.2 =
    ∑' z, (ZR (sketchSummarizer (listIdentityOp (α := α))) x R T z).toReal *
      ∑' p, (gen z p).toReal * loss z p.1 p.2 := by
  simpa [listIdentityOp] using
    (preference_learning_equivalence_via_ZR_of_identity_encoded_feature
      (Strings := List α)
      (feature := lengthFeature (α := α))
      lengthFeature_congruent
      (loss := loss) (gen := gen) (x := x) (R := R) (T := T)
      hp hR h_meas h_pair M hM hbound)

/-- Generic pairwise preference equivalence via ZR for encoded length oracle
under the genuinely lossy length sketch operator. -/
theorem length_preference_equivalence_via_ZR_of_lossy_encoded_feature
    [Inhabited α]
    {A : Type*}
    (loss : List α → A → A → ℝ)
    (gen : PairGenerator (List α) A)
    (x : List α) (R : ℕ) (T : BinTree (List α))
    (hp : S T = x) (hR : R ≥ 1)
    (h_meas :
      OracleMeasurableLoss loss
        (encodedOracle (Strings := List α) (lengthFeature (α := α))))
    (h_pair :
      OracleIndexedPairGen gen
        (encodedOracle (Strings := List α) (lengthFeature (α := α))))
    (M : ℝ) (hM : 0 ≤ M)
    (hbound :
      ∀ w z,
        D (encodedOracle (Strings := List α) (lengthFeature (α := α))) w z ≤ M) :
    ∑' x', (PMF.pure x x').toReal * ∑' p, (gen x' p).toReal * loss x' p.1 p.2 =
    ∑' z, (ZR (sketchSummarizer (lengthSketchOperator (α := α))) x R T z).toReal *
      ∑' p, (gen z p).toReal * loss z p.1 p.2 := by
  exact preference_learning_equivalence_via_ZR_of_sketch
    (op := lengthSketchOperator (α := α))
    (fstar := encodedOracle (Strings := List α) (lengthFeature (α := α)))
    (loss := loss) (gen := gen) (x := x) (R := R) (T := T)
    hp hR
    (lengthSketch_leaf_preserving (α := α))
    (lengthSketch_merge_compatible (α := α))
    (lengthSketch_summary_compatible (α := α))
    h_meas h_pair M hM hbound

/-- End-to-end DPO gap bound for the genuinely lossy length sketch under a
stochastic adaptive tree policy with approximate local laws on support trees. -/
theorem length_dpo_gap_of_stochastic_adaptive_approx
    [Inhabited α]
    {Y : Type*} [PseudoMetricSpace Y]
    {A : Type*}
    (fstar : List α → Y)
    (pol pol_ref : Policy (List α) A)
    (gen : PairGenerator (List α) A)
    (τ : StochasticAdaptiveTreeMap (List α))
    (β : ℝ) (L_pol : NNReal)
    (D_max : ℝ) (hD_max : 0 ≤ D_max)
    (h_dist_bound : ∀ w z, dist (fstar w) (fstar z) ≤ D_max)
    (hbound : ∀ x z, D fstar z x ≤ 1)
    (hbound_global : ∀ w z, D fstar w z ≤ 1)
    (Loss_max : ℝ) (hLoss_max : 0 ≤ Loss_max)
    (hLoss_bound : ∀ x' (p : A × A), |DPOLossPointwise pol pol_ref β x' p.1 p.2| ≤ Loss_max)
    (h_meas_pol : DPO.OracleMeasurable pol fstar)
    (h_meas_ref : DPO.OracleMeasurable pol_ref fstar)
    (h_lip : PolicyLipschitz pol pol_ref fstar L_pol)
    (h_gen_fixed : ∀ x' x'', gen x' = gen x'')
    (h_mono : ∀ p, pIdemp (lossyLengthSummarizer (α := α)) fstar
      (p.bind (lossyLengthSummarizer (α := α))) ≤
      pIdemp (lossyLengthSummarizer (α := α)) fstar p)
    (ε_leaf ε_merge ε_idemp : List α → BinTree (List α) → ℝ)
    (h_sound : StochasticAdaptiveChunkingSound τ)
    (h_approx : StochasticAdaptiveApproxLocalLaws
      (g := lossyLengthSummarizer (α := α)) (fstar := fstar)
      τ ε_leaf ε_merge ε_idemp)
    (x : List α) (R : ℕ) (hR : R ≥ 1)
    (T : BinTree (List α)) (hT : T ∈ (τ x).support) :
    |ExpectedDPOLoss pol pol_ref β (PMF.pure x) gen -
     ExpectedDPOLoss pol pol_ref β (ZR (lossyLengthSummarizer (α := α)) x R T) gen| ≤
    2 * |β| * (L_pol : ℝ) *
    (ε_leaf x T + ε_merge x T + ((R : ℝ) - 1) * ε_idemp x T) := by
  have hTlaw := h_approx x T hT
  exact dpo_gap_via_approx_local_laws fstar pol pol_ref gen
    (lossyLengthSummarizer (α := α)) x R T β L_pol
    (h_sound x T hT) hR
    D_max hD_max h_dist_bound
    (hbound x) hbound_global
    Loss_max hLoss_max hLoss_bound
    h_meas_pol h_meas_ref h_lip h_gen_fixed h_mono
    (ε_leaf x T) (ε_merge x T) (ε_idemp x T)
    hTlaw.1 hTlaw.2.1 hTlaw.2.2

/-- Concrete DPO support-tree gap for the lossy length sketch with an added
oracle-measurement term separating the true target from the oracle-indexed one. -/
theorem length_dpo_gap_of_stochastic_adaptive_approx_with_oracleMeasurement
    [Inhabited α]
    {Y : Type*} [PseudoMetricSpace Y]
    {A : Type*}
    (fstar : List α → Y)
    (pol pol_ref : Policy (List α) A)
    (gen : PairGenerator (List α) A)
    (τ : StochasticAdaptiveTreeMap (List α))
    (β : ℝ) (L_pol : NNReal)
    (D_max : ℝ) (hD_max : 0 ≤ D_max)
    (h_dist_bound : ∀ w z, dist (fstar w) (fstar z) ≤ D_max)
    (hbound : ∀ x z, D fstar z x ≤ 1)
    (hbound_global : ∀ w z, D fstar w z ≤ 1)
    (Loss_max : ℝ) (hLoss_max : 0 ≤ Loss_max)
    (hLoss_bound : ∀ x' (p : A × A), |DPOLossPointwise pol pol_ref β x' p.1 p.2| ≤ Loss_max)
    (h_meas_pol : DPO.OracleMeasurable pol fstar)
    (h_meas_ref : DPO.OracleMeasurable pol_ref fstar)
    (h_lip : PolicyLipschitz pol pol_ref fstar L_pol)
    (h_gen_fixed : ∀ x' x'', gen x' = gen x'')
    (h_mono : ∀ p, pIdemp (lossyLengthSummarizer (α := α)) fstar
      (p.bind (lossyLengthSummarizer (α := α))) ≤
      pIdemp (lossyLengthSummarizer (α := α)) fstar p)
    (ε_leaf ε_merge ε_idemp : List α → BinTree (List α) → ℝ)
    (h_sound : StochasticAdaptiveChunkingSound τ)
    (h_approx : StochasticAdaptiveApproxLocalLaws
      (g := lossyLengthSummarizer (α := α)) (fstar := fstar)
      τ ε_leaf ε_merge ε_idemp)
    (x : List α) (R : ℕ) (hR : R ≥ 1)
    (T : BinTree (List α)) (hT : T ∈ (τ x).support)
    (loss_true oracle_err : ℝ)
    (h_oracle :
      |loss_true - ExpectedDPOLoss pol pol_ref β (PMF.pure x) gen| ≤ oracle_err) :
    |loss_true - ExpectedDPOLoss pol pol_ref β (ZR (lossyLengthSummarizer (α := α)) x R T) gen| ≤
      oracle_err + 2 * |β| * (L_pol : ℝ) *
        (ε_leaf x T + ε_merge x T + ((R : ℝ) - 1) * ε_idemp x T) := by
  have h_core :
      |ExpectedDPOLoss pol pol_ref β (PMF.pure x) gen -
        ExpectedDPOLoss pol pol_ref β (ZR (lossyLengthSummarizer (α := α)) x R T) gen| ≤
        2 * |β| * (L_pol : ℝ) *
          (ε_leaf x T + ε_merge x T + ((R : ℝ) - 1) * ε_idemp x T) :=
    length_dpo_gap_of_stochastic_adaptive_approx
      (fstar := fstar) (pol := pol) (pol_ref := pol_ref) (gen := gen) (τ := τ)
      (β := β) (L_pol := L_pol)
      D_max hD_max h_dist_bound hbound hbound_global
      Loss_max hLoss_max hLoss_bound
      h_meas_pol h_meas_ref h_lip h_gen_fixed h_mono
      ε_leaf ε_merge ε_idemp h_sound h_approx x R hR T hT
  have h_triangle :
      |loss_true - ExpectedDPOLoss pol pol_ref β (ZR (lossyLengthSummarizer (α := α)) x R T) gen| ≤
        |loss_true - ExpectedDPOLoss pol pol_ref β (PMF.pure x) gen| +
          |ExpectedDPOLoss pol pol_ref β (PMF.pure x) gen -
            ExpectedDPOLoss pol pol_ref β (ZR (lossyLengthSummarizer (α := α)) x R T) gen| := by
    have hdecomp :
        loss_true - ExpectedDPOLoss pol pol_ref β (ZR (lossyLengthSummarizer (α := α)) x R T) gen =
          (loss_true - ExpectedDPOLoss pol pol_ref β (PMF.pure x) gen) +
          (ExpectedDPOLoss pol pol_ref β (PMF.pure x) gen -
            ExpectedDPOLoss pol pol_ref β (ZR (lossyLengthSummarizer (α := α)) x R T) gen) := by
      ring
    rw [hdecomp]
    exact abs_add_le _ _
  linarith

/-- End-to-end GRPO-PL gap bound for the genuinely lossy length sketch under a
stochastic adaptive tree policy with approximate local laws on support trees. -/
theorem length_grpo_pl_gap_of_stochastic_adaptive_approx
    [Inhabited α]
    {Y : Type*} [PseudoMetricSpace Y]
    {A : Type*} {k : ℕ}
    (fstar : List α → Y)
    (pol : Policy' (List α) A) (ranker : List α → GroupRanker A k)
    (gen : GroupGenerator (List α) A k)
    (τ : StochasticAdaptiveTreeMap (List α))
    (L_grpo : NNReal)
    (D_max : ℝ) (hD_max : 0 ≤ D_max)
    (h_dist_bound : ∀ w z, dist (fstar w) (fstar z) ≤ D_max)
    (Loss_max : ℝ) (hLoss_max : 0 ≤ Loss_max)
    (hLoss_bound : ∀ x' (group : Fin k → A), |GRPOLossPointwise pol x' group (ranker x' group)| ≤ Loss_max)
    (h_pol_lip : GRPOPolicyLipschitz pol fstar L_grpo)
    (h_ranker : OracleIndexedRanker ranker fstar)
    (h_rum : ∀ x' z',
      ExpectedGRPOLossLipschitz pol ranker fstar (gen x') L_grpo h_pol_lip h_ranker x' z')
    (h_gen_fixed : ∀ x' x'', gen x' = gen x'')
    (hbound : ∀ x z, D fstar z x ≤ 1)
    (hbound_global : ∀ w z, D fstar w z ≤ 1)
    (h_mono : ∀ p, pIdemp (lossyLengthSummarizer (α := α)) fstar
      (p.bind (lossyLengthSummarizer (α := α))) ≤
      pIdemp (lossyLengthSummarizer (α := α)) fstar p)
    (ε_leaf ε_merge ε_idemp : List α → BinTree (List α) → ℝ)
    (h_sound : StochasticAdaptiveChunkingSound τ)
    (h_approx : StochasticAdaptiveApproxLocalLaws
      (g := lossyLengthSummarizer (α := α)) (fstar := fstar)
      τ ε_leaf ε_merge ε_idemp)
    (x : List α) (R : ℕ) (hR : R ≥ 1)
    (T : BinTree (List α)) (hT : T ∈ (τ x).support) :
    |ExpectedGRPOLoss pol ranker (PMF.pure x) gen -
     ExpectedGRPOLoss pol ranker (ZR (lossyLengthSummarizer (α := α)) x R T) gen| ≤
    (L_grpo : ℝ) * (ε_leaf x T + ε_merge x T + ((R : ℝ) - 1) * ε_idemp x T) := by
  have hTlaw := h_approx x T hT
  exact grpo_pl_gap_via_approx_local_laws (k := k) fstar pol ranker gen
    (lossyLengthSummarizer (α := α)) x R T L_grpo
    D_max hD_max h_dist_bound
    Loss_max hLoss_max hLoss_bound
    h_pol_lip h_ranker h_rum h_gen_fixed
    (h_sound x T hT) hR
    (hbound x) hbound_global h_mono
    (ε_leaf x T) (ε_merge x T) (ε_idemp x T)
    hTlaw.1 hTlaw.2.1 hTlaw.2.2

/-- Concrete GRPO-PL support-tree gap for the lossy length sketch with an
optional oracle-measurement term. -/
theorem length_grpo_pl_gap_of_stochastic_adaptive_approx_with_oracleMeasurement
    [Inhabited α]
    {Y : Type*} [PseudoMetricSpace Y]
    {A : Type*} {k : ℕ}
    (fstar : List α → Y)
    (pol : Policy' (List α) A) (ranker : List α → GroupRanker A k)
    (gen : GroupGenerator (List α) A k)
    (τ : StochasticAdaptiveTreeMap (List α))
    (L_grpo : NNReal)
    (D_max : ℝ) (hD_max : 0 ≤ D_max)
    (h_dist_bound : ∀ w z, dist (fstar w) (fstar z) ≤ D_max)
    (Loss_max : ℝ) (hLoss_max : 0 ≤ Loss_max)
    (hLoss_bound : ∀ x' (group : Fin k → A), |GRPOLossPointwise pol x' group (ranker x' group)| ≤ Loss_max)
    (h_pol_lip : GRPOPolicyLipschitz pol fstar L_grpo)
    (h_ranker : OracleIndexedRanker ranker fstar)
    (h_rum : ∀ x' z',
      ExpectedGRPOLossLipschitz pol ranker fstar (gen x') L_grpo h_pol_lip h_ranker x' z')
    (h_gen_fixed : ∀ x' x'', gen x' = gen x'')
    (hbound : ∀ x z, D fstar z x ≤ 1)
    (hbound_global : ∀ w z, D fstar w z ≤ 1)
    (h_mono : ∀ p, pIdemp (lossyLengthSummarizer (α := α)) fstar
      (p.bind (lossyLengthSummarizer (α := α))) ≤
      pIdemp (lossyLengthSummarizer (α := α)) fstar p)
    (ε_leaf ε_merge ε_idemp : List α → BinTree (List α) → ℝ)
    (h_sound : StochasticAdaptiveChunkingSound τ)
    (h_approx : StochasticAdaptiveApproxLocalLaws
      (g := lossyLengthSummarizer (α := α)) (fstar := fstar)
      τ ε_leaf ε_merge ε_idemp)
    (x : List α) (R : ℕ) (hR : R ≥ 1)
    (T : BinTree (List α)) (hT : T ∈ (τ x).support)
    (loss_true oracle_err : ℝ)
    (h_oracle :
      |loss_true - ExpectedGRPOLoss pol ranker (PMF.pure x) gen| ≤ oracle_err) :
    |loss_true - ExpectedGRPOLoss pol ranker (ZR (lossyLengthSummarizer (α := α)) x R T) gen| ≤
      oracle_err + (L_grpo : ℝ) *
        (ε_leaf x T + ε_merge x T + ((R : ℝ) - 1) * ε_idemp x T) := by
  have h_core :
      |ExpectedGRPOLoss pol ranker (PMF.pure x) gen -
        ExpectedGRPOLoss pol ranker (ZR (lossyLengthSummarizer (α := α)) x R T) gen| ≤
        (L_grpo : ℝ) * (ε_leaf x T + ε_merge x T + ((R : ℝ) - 1) * ε_idemp x T) :=
    length_grpo_pl_gap_of_stochastic_adaptive_approx
      (fstar := fstar) (pol := pol) (ranker := ranker) (gen := gen) (τ := τ)
      (L_grpo := L_grpo)
      D_max hD_max h_dist_bound
      Loss_max hLoss_max hLoss_bound
      h_pol_lip h_ranker h_rum h_gen_fixed
      hbound hbound_global h_mono
      ε_leaf ε_merge ε_idemp h_sound h_approx x R hR T hT
  have h_triangle :
      |loss_true - ExpectedGRPOLoss pol ranker (ZR (lossyLengthSummarizer (α := α)) x R T) gen| ≤
        |loss_true - ExpectedGRPOLoss pol ranker (PMF.pure x) gen| +
          |ExpectedGRPOLoss pol ranker (PMF.pure x) gen -
            ExpectedGRPOLoss pol ranker (ZR (lossyLengthSummarizer (α := α)) x R T) gen| := by
    have hdecomp :
        loss_true - ExpectedGRPOLoss pol ranker (ZR (lossyLengthSummarizer (α := α)) x R T) gen =
          (loss_true - ExpectedGRPOLoss pol ranker (PMF.pure x) gen) +
          (ExpectedGRPOLoss pol ranker (PMF.pure x) gen -
            ExpectedGRPOLoss pol ranker (ZR (lossyLengthSummarizer (α := α)) x R T) gen) := by
      ring
    rw [hdecomp]
    exact abs_add_le _ _
  linarith

/-- End-to-end GRPO-RL gap bound for the genuinely lossy length sketch under a
stochastic adaptive tree policy with approximate local laws on support trees. -/
theorem length_grpo_rl_gap_of_stochastic_adaptive_approx
    [Inhabited α]
    {Y : Type*} [PseudoMetricSpace Y]
    {A : Type*} {k : ℕ}
    (fstar : List α → Y)
    (pol pol_old pol_ref : Policy' (List α) A)
    (reward : List α → A → ℝ)
    (eps beta : ℝ)
    (gen : GroupGenerator (List α) A k)
    (τ : StochasticAdaptiveTreeMap (List α))
    (L_grpo_rl : NNReal)
    (D_max : ℝ) (hD_max : 0 ≤ D_max)
    (h_dist_bound : ∀ w z, dist (fstar w) (fstar z) ≤ D_max)
    (Loss_max : ℝ) (hLoss_max : 0 ≤ Loss_max)
    (hLoss_bound : ∀ x' (group : Fin k → A),
      |GRPORLLossPointwise pol pol_old pol_ref reward eps beta x' group| ≤ Loss_max)
    (h_pol_lip : GRPOPolicyLipschitz pol fstar L_grpo_rl)
    (h_old_lip : GRPOPolicyLipschitz pol_old fstar L_grpo_rl)
    (h_ref_lip : GRPOPolicyLipschitz pol_ref fstar L_grpo_rl)
    (h_reward_lip : RewardLipschitzGRPO reward fstar L_grpo_rl)
    (h_rum : ∀ x' z',
      ExpectedGRPORLLossLipschitz k pol pol_old pol_ref reward eps beta fstar (gen x') L_grpo_rl
        h_pol_lip h_old_lip h_ref_lip h_reward_lip x' z')
    (h_gen_fixed : ∀ x' x'', gen x' = gen x'')
    (hbound : ∀ x z, D fstar z x ≤ 1)
    (hbound_global : ∀ w z, D fstar w z ≤ 1)
    (h_mono : ∀ p, pIdemp (lossyLengthSummarizer (α := α)) fstar
      (p.bind (lossyLengthSummarizer (α := α))) ≤
      pIdemp (lossyLengthSummarizer (α := α)) fstar p)
    (ε_leaf ε_merge ε_idemp : List α → BinTree (List α) → ℝ)
    (h_sound : StochasticAdaptiveChunkingSound τ)
    (h_approx : StochasticAdaptiveApproxLocalLaws
      (g := lossyLengthSummarizer (α := α)) (fstar := fstar)
      τ ε_leaf ε_merge ε_idemp)
    (x : List α) (R : ℕ) (hR : R ≥ 1)
    (T : BinTree (List α)) (hT : T ∈ (τ x).support) :
    |ExpectedGRPORLLoss pol pol_old pol_ref reward eps beta (PMF.pure x) gen -
     ExpectedGRPORLLoss pol pol_old pol_ref reward eps beta
      (ZR (lossyLengthSummarizer (α := α)) x R T) gen| ≤
    (L_grpo_rl : ℝ) * (ε_leaf x T + ε_merge x T + ((R : ℝ) - 1) * ε_idemp x T) := by
  have hTlaw := h_approx x T hT
  exact grpo_rl_gap_via_approx_local_laws (k := k) fstar pol pol_old pol_ref reward eps beta gen
    (lossyLengthSummarizer (α := α)) x R T L_grpo_rl
    D_max hD_max h_dist_bound
    Loss_max hLoss_max hLoss_bound
    h_pol_lip h_old_lip h_ref_lip h_reward_lip
    h_rum h_gen_fixed
    (h_sound x T hT) hR
    (hbound x) hbound_global h_mono
    (ε_leaf x T) (ε_merge x T) (ε_idemp x T)
    hTlaw.1 hTlaw.2.1 hTlaw.2.2

/-- Concrete GRPO-RL support-tree gap for the lossy length sketch with an
optional oracle-measurement term. -/
theorem length_grpo_rl_gap_of_stochastic_adaptive_approx_with_oracleMeasurement
    [Inhabited α]
    {Y : Type*} [PseudoMetricSpace Y]
    {A : Type*} {k : ℕ}
    (fstar : List α → Y)
    (pol pol_old pol_ref : Policy' (List α) A)
    (reward : List α → A → ℝ)
    (eps beta : ℝ)
    (gen : GroupGenerator (List α) A k)
    (τ : StochasticAdaptiveTreeMap (List α))
    (L_grpo_rl : NNReal)
    (D_max : ℝ) (hD_max : 0 ≤ D_max)
    (h_dist_bound : ∀ w z, dist (fstar w) (fstar z) ≤ D_max)
    (Loss_max : ℝ) (hLoss_max : 0 ≤ Loss_max)
    (hLoss_bound : ∀ x' (group : Fin k → A),
      |GRPORLLossPointwise pol pol_old pol_ref reward eps beta x' group| ≤ Loss_max)
    (h_pol_lip : GRPOPolicyLipschitz pol fstar L_grpo_rl)
    (h_old_lip : GRPOPolicyLipschitz pol_old fstar L_grpo_rl)
    (h_ref_lip : GRPOPolicyLipschitz pol_ref fstar L_grpo_rl)
    (h_reward_lip : RewardLipschitzGRPO reward fstar L_grpo_rl)
    (h_rum : ∀ x' z',
      ExpectedGRPORLLossLipschitz k pol pol_old pol_ref reward eps beta fstar (gen x') L_grpo_rl
        h_pol_lip h_old_lip h_ref_lip h_reward_lip x' z')
    (h_gen_fixed : ∀ x' x'', gen x' = gen x'')
    (hbound : ∀ x z, D fstar z x ≤ 1)
    (hbound_global : ∀ w z, D fstar w z ≤ 1)
    (h_mono : ∀ p, pIdemp (lossyLengthSummarizer (α := α)) fstar
      (p.bind (lossyLengthSummarizer (α := α))) ≤
      pIdemp (lossyLengthSummarizer (α := α)) fstar p)
    (ε_leaf ε_merge ε_idemp : List α → BinTree (List α) → ℝ)
    (h_sound : StochasticAdaptiveChunkingSound τ)
    (h_approx : StochasticAdaptiveApproxLocalLaws
      (g := lossyLengthSummarizer (α := α)) (fstar := fstar)
      τ ε_leaf ε_merge ε_idemp)
    (x : List α) (R : ℕ) (hR : R ≥ 1)
    (T : BinTree (List α)) (hT : T ∈ (τ x).support)
    (loss_true oracle_err : ℝ)
    (h_oracle :
      |loss_true - ExpectedGRPORLLoss pol pol_old pol_ref reward eps beta (PMF.pure x) gen|
        ≤ oracle_err) :
    |loss_true - ExpectedGRPORLLoss pol pol_old pol_ref reward eps beta
        (ZR (lossyLengthSummarizer (α := α)) x R T) gen| ≤
      oracle_err + (L_grpo_rl : ℝ) *
        (ε_leaf x T + ε_merge x T + ((R : ℝ) - 1) * ε_idemp x T) := by
  have h_core :
      |ExpectedGRPORLLoss pol pol_old pol_ref reward eps beta (PMF.pure x) gen -
        ExpectedGRPORLLoss pol pol_old pol_ref reward eps beta
          (ZR (lossyLengthSummarizer (α := α)) x R T) gen| ≤
        (L_grpo_rl : ℝ) * (ε_leaf x T + ε_merge x T + ((R : ℝ) - 1) * ε_idemp x T) :=
    length_grpo_rl_gap_of_stochastic_adaptive_approx
      (fstar := fstar) (pol := pol) (pol_old := pol_old) (pol_ref := pol_ref)
      (reward := reward) (eps := eps) (beta := beta) (gen := gen) (τ := τ)
      (L_grpo_rl := L_grpo_rl)
      D_max hD_max h_dist_bound
      Loss_max hLoss_max hLoss_bound
      h_pol_lip h_old_lip h_ref_lip h_reward_lip
      h_rum h_gen_fixed
      hbound hbound_global h_mono
      ε_leaf ε_merge ε_idemp h_sound h_approx x R hR T hT
  have h_triangle :
      |loss_true - ExpectedGRPORLLoss pol pol_old pol_ref reward eps beta
          (ZR (lossyLengthSummarizer (α := α)) x R T) gen| ≤
        |loss_true - ExpectedGRPORLLoss pol pol_old pol_ref reward eps beta (PMF.pure x) gen| +
          |ExpectedGRPORLLoss pol pol_old pol_ref reward eps beta (PMF.pure x) gen -
            ExpectedGRPORLLoss pol pol_old pol_ref reward eps beta
              (ZR (lossyLengthSummarizer (α := α)) x R T) gen| := by
    have hdecomp :
        loss_true - ExpectedGRPORLLoss pol pol_old pol_ref reward eps beta
            (ZR (lossyLengthSummarizer (α := α)) x R T) gen =
          (loss_true - ExpectedGRPORLLoss pol pol_old pol_ref reward eps beta (PMF.pure x) gen) +
          (ExpectedGRPORLLoss pol pol_old pol_ref reward eps beta (PMF.pure x) gen -
            ExpectedGRPORLLoss pol pol_old pol_ref reward eps beta
              (ZR (lossyLengthSummarizer (α := α)) x R T) gen) := by
      ring
    rw [hdecomp]
    exact abs_add_le _ _
  linarith

end LengthInstances

end FormalProofs.OPT
