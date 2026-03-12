import FormalProofs.OPT.SketchSummaryOperators
import FormalProofs.OPT.PreferenceBounds
import FormalProofs.OPT.ApproximateLocalLaws
import FormalProofs.OPT.PreferenceLearning

/-!
# FormalProofs/OPT/SketchRecovery.lean

Generic recovery theorems from sketch-level assumptions to downstream objectives.

The key pattern is:
1. sketch assumptions ⇒ `LocalLawsBundle` via `local_laws_bundle_of_sketch`;
2. instantiate existing OPT/DSL theorems with the induced deterministic summarizer.
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

variable {Strings : Type*} [Monoid Strings]
variable {Y : Type*} [PseudoMetricSpace Y]
variable {Sketch : Type*}

/-- Induced deterministic summarizer from a sketch operator. -/
abbrev sketchSummarizer (op : SketchOperator Strings Sketch) : Summarizer Strings :=
  deterministicSummarizer (summaryFromSketch op)

/-- Local laws recovered from sketch assumptions (packaged alias). -/
theorem local_laws_of_sketch
    (op : SketchOperator Strings Sketch) (fstar : Strings → Y) (T : BinTree Strings)
    (h_leaf : SketchLeafPreserving op fstar)
    (h_merge : SketchMergeCompatible op fstar)
    (h_compat : SketchSummaryCompatible op) :
    LocalLawsBundle (sketchSummarizer op) T fstar :=
  local_laws_bundle_of_sketch (op := op) (fstar := fstar) h_leaf h_merge h_compat T

/-!
## Identity + Encoded-Feature Templates
-/

/-- Reusable local-law template for identity sketch operator with encoded feature oracle.

This is the standard "plug in feature + congruence" entrypoint for new domains. -/
theorem local_laws_of_identity_encoded_feature
    {Feature : Type*} [Encodable Feature]
    (feature : Strings → Feature)
    (h_feature_congr :
      ∀ sL sR x y,
        feature sL = feature x →
        feature sR = feature y →
        feature (sL * sR) = feature (x * y))
    (T : BinTree Strings) :
    LocalLawsBundle (sketchSummarizer (identitySketchOperator (Strings := Strings))) T
      (encodedOracle (Strings := Strings) feature) := by
  exact local_laws_of_sketch
    (op := identitySketchOperator (Strings := Strings))
    (fstar := encodedOracle (Strings := Strings) feature)
    (T := T)
    (identitySketch_leaf_preserving _)
    (identitySketch_merge_compatible_of_feature_congruent
      (feature := feature) h_feature_congr)
    (identitySketch_summary_compatible (Strings := Strings))

/-- Reusable local-law template for the paired non-identity sketch operator with
encoded feature oracle. -/
theorem local_laws_of_paired_encoded_feature
    {Feature : Type*} [Encodable Feature]
    (feature : Strings → Feature)
    (h_feature_congr :
      ∀ sL sR x y,
        feature sL = feature x →
        feature sR = feature y →
        feature (sL * sR) = feature (x * y))
    (T : BinTree Strings) :
    LocalLawsBundle (sketchSummarizer (pairedSketchOperator (Strings := Strings))) T
      (encodedOracle (Strings := Strings) feature) := by
  exact local_laws_of_sketch
    (op := pairedSketchOperator (Strings := Strings))
    (fstar := encodedOracle (Strings := Strings) feature)
    (T := T)
    (pairedSketch_leaf_preserving _)
    (pairedSketch_merge_compatible_of_feature_congruent
      (feature := feature) h_feature_congr)
    (pairedSketch_summary_compatible (Strings := Strings))

/-- Multi-round preservation recovered from sketch assumptions (explicit bound form). -/
theorem multi_round_proper_of_sketch
    (op : SketchOperator Strings Sketch) (fstar : Strings → Y)
    (x : Strings) (R : ℕ) (T : BinTree Strings)
    (hp : S T = x) (hR : R ≥ 1)
    (h_leaf : SketchLeafPreserving op fstar)
    (h_merge : SketchMergeCompatible op fstar)
    (h_compat : SketchSummaryCompatible op)
    (M : ℝ) (hM : 0 ≤ M) (hbound : ∀ w z, D fstar w z ≤ M) :
    Exp (ZR (sketchSummarizer op) x R T) (fun z => D fstar z x) = 0 := by
  let laws :=
    local_laws_of_sketch (op := op) (fstar := fstar) (T := T) h_leaf h_merge h_compat
  exact multi_round_proper (g := sketchSummarizer op) (T := T) (x := x) (R := R)
    (fstar := fstar) hp laws.law1 laws.law2 laws.law3 hR M hM hbound

/-- Multi-round preservation recovered from sketch assumptions (typeclass-bound form). -/
theorem multi_round_typeclass_of_sketch
    {Y : Type*} [BoundedPseudoMetricSpace Y]
    (op : SketchOperator Strings Sketch) (fstar : Strings → Y)
    (x : Strings) (R : ℕ) (T : BinTree Strings)
    (hp : S T = x) (hR : R ≥ 1)
    (h_leaf : SketchLeafPreserving op fstar)
    (h_merge : SketchMergeCompatible op fstar)
    (h_compat : SketchSummaryCompatible op) :
    Exp (ZR (sketchSummarizer op) x R T) (fun z => D fstar z x) = 0 := by
  let laws :=
    local_laws_of_sketch (op := op) (fstar := fstar) (T := T) h_leaf h_merge h_compat
  exact multi_round_typeclass (g := sketchSummarizer op) (T := T) (x := x) (R := R)
    (fstar := fstar) hp laws.law1 laws.law2 laws.law3 hR

/-- Distortion proxy `Δ_R_ZR` is zero for induced sketch summarizers under local sketch assumptions. -/
theorem Δ_R_eq_zero_of_sketch
    {Y : Type*} [BoundedPseudoMetricSpace Y]
    (op : SketchOperator Strings Sketch) (fstar : Strings → Y)
    (x : Strings) (R : ℕ) (T : BinTree Strings)
    (hp : S T = x) (hR : R ≥ 1)
    (h_leaf : SketchLeafPreserving op fstar)
    (h_merge : SketchMergeCompatible op fstar)
    (h_compat : SketchSummaryCompatible op) :
    Δ_R_ZR (sketchSummarizer op) x R T fstar = 0 := by
  let laws :=
    local_laws_of_sketch (op := op) (fstar := fstar) (T := T) h_leaf h_merge h_compat
  exact Δ_R_eq_zero_of_local_laws (g := sketchSummarizer op) (x := x) (R := R) (T := T)
    (fstar := fstar) hp laws.law1 laws.law2 laws.law3 hR

/-!
## Approximate Sketch Recovery
-/

/-- Approximate leaf-preservation assumption for a sketch operator, stated as
pointwise violation-probability budgets. -/
def SketchLeafApproxPreserving
    (op : SketchOperator Strings Sketch) (fstar : Strings → Y)
    (ε_leaf : Strings → ℝ) : Prop :=
  ∀ b, ViolationProb fstar ((sketchSummarizer op) b) b ≤ ε_leaf b

/-- Approximate merge-compatibility assumption for a sketch operator, stated as
pointwise internal-node violation budgets. -/
def SketchMergeApproxCompatible
    (op : SketchOperator Strings Sketch) (fstar : Strings → Y)
    (ε_merge : BinTree Strings × BinTree Strings → ℝ) : Prop :=
  ∀ p : BinTree Strings × BinTree Strings,
    ViolationProb fstar (reduce (sketchSummarizer op) (BinTree.node p.1 p.2))
      (S (BinTree.node p.1 p.2)) ≤ ε_merge p

/-- Approximate sketch assumptions induce nodewise approximate local laws. -/
theorem approx_nodewise_local_laws_of_sketch
    (op : SketchOperator Strings Sketch) (fstar : Strings → Y) (T : BinTree Strings)
    (ε_leaf : Strings → ℝ)
    (ε_merge : BinTree Strings × BinTree Strings → ℝ)
    (h_leaf : SketchLeafApproxPreserving op fstar ε_leaf)
    (h_merge : SketchMergeApproxCompatible op fstar ε_merge) :
    FormalProofs.OPT.L1εNode (sketchSummarizer op) T fstar ε_leaf ∧
      FormalProofs.OPT.L2εNode (sketchSummarizer op) T fstar ε_merge := by
  constructor
  · intro b hb
    exact h_leaf b
  · intro p hp
    simpa using h_merge p

/-- Approximate sketch assumptions plus an idempotence budget induce a full
approximate-local-law bundle. -/
def approx_bundle_of_sketch
    (op : SketchOperator Strings Sketch) (fstar : Strings → Y) (T : BinTree Strings)
    (ε_leaf : Strings → ℝ)
    (ε_merge : BinTree Strings × BinTree Strings → ℝ)
    (ε_idemp : ℝ)
    (h_leaf : SketchLeafApproxPreserving op fstar ε_leaf)
    (h_merge : SketchMergeApproxCompatible op fstar ε_merge)
    (h_idemp : FormalProofs.OPT.L3ε (sketchSummarizer op) T fstar ε_idemp) :
    FormalProofs.OPT.ApproxLocalLawsBundle (sketchSummarizer op) T fstar := by
  have h_node :
      FormalProofs.OPT.L1εNode (sketchSummarizer op) T fstar ε_leaf ∧
        FormalProofs.OPT.L2εNode (sketchSummarizer op) T fstar ε_merge :=
    approx_nodewise_local_laws_of_sketch op fstar T ε_leaf ε_merge h_leaf h_merge
  exact FormalProofs.OPT.approx_bundle_of_nodewise (g := sketchSummarizer op) (T := T) (fstar := fstar)
    (ε_leaf := ε_leaf) (ε_merge := ε_merge) (ε_idemp := ε_idemp)
    h_node.1 h_node.2 h_idemp

/-!
## Preference-Objective Recovery
-/

section PreferenceObjectives

variable {A : Type*}

/-- DPO equivalence recovered from sketch assumptions. -/
theorem dpo_equivalence_of_sketch
    {Y : Type*} [BoundedMetricSpace Y]
    (op : SketchOperator Strings Sketch) (fstar : Strings → Y)
    (pol pol_ref : Policy Strings A)
    (gen : PairGenerator Strings A)
    (x : Strings) (R : ℕ) (T : BinTree Strings)
    (β : ℝ)
    (hp : S T = x) (hR : R ≥ 1)
    (h_leaf : SketchLeafPreserving op fstar)
    (h_merge : SketchMergeCompatible op fstar)
    (h_compat : SketchSummaryCompatible op)
    (h_meas : OracleMeasurablePolicies pol pol_ref fstar)
    (h_pair : OracleIndexedPairGen gen fstar) :
    ExpectedDPOLoss pol pol_ref β (PMF.pure x) gen =
    ExpectedDPOLoss pol pol_ref β (ZR (sketchSummarizer op) x R T) gen := by
  let laws :=
    local_laws_of_sketch (op := op) (fstar := fstar) (T := T) h_leaf h_merge h_compat
  exact dpo_equivalence (fstar := fstar) (pol := pol) (pol_ref := pol_ref) (gen := gen)
    (g := sketchSummarizer op) (x := x) (R := R) (T := T) (β := β)
    hp laws hR h_meas h_pair

/-- Generic pairwise preference equivalence via ZR recovered from sketch assumptions. -/
theorem preference_learning_equivalence_via_ZR_of_sketch
    {Y : Type*} [MetricSpace Y]
    (op : SketchOperator Strings Sketch) (fstar : Strings → Y)
    (loss : Strings → A → A → ℝ)
    (gen : PairGenerator Strings A)
    (x : Strings) (R : ℕ) (T : BinTree Strings)
    (hp : S T = x) (hR : R ≥ 1)
    (h_leaf : SketchLeafPreserving op fstar)
    (h_merge : SketchMergeCompatible op fstar)
    (h_compat : SketchSummaryCompatible op)
    (h_meas : OracleMeasurableLoss loss fstar)
    (h_pair : OracleIndexedPairGen gen fstar)
    (M : ℝ) (hM : 0 ≤ M) (hbound : ∀ w z, D fstar w z ≤ M) :
    ∑' x', (PMF.pure x x').toReal * ∑' p, (gen x' p).toReal * loss x' p.1 p.2 =
    ∑' z, (ZR (sketchSummarizer op) x R T z).toReal * ∑' p, (gen z p).toReal * loss z p.1 p.2 := by
  let laws :=
    local_laws_of_sketch (op := op) (fstar := fstar) (T := T) h_leaf h_merge h_compat
  exact preference_learning_equivalence_via_ZR (fstar := fstar) (loss := loss) (gen := gen)
    (g := sketchSummarizer op) (x := x) (R := R) (T := T)
    hp laws.law1 laws.law2 laws.law3 hR h_meas h_pair M hM hbound

/-- Reusable pairwise-equivalence template for identity sketch operator with encoded
feature oracle.

For a new domain, only `feature` and `h_feature_congr` are required. -/
theorem preference_learning_equivalence_via_ZR_of_identity_encoded_feature
    {Feature : Type*} [Encodable Feature]
    (feature : Strings → Feature)
    (h_feature_congr :
      ∀ sL sR x y,
        feature sL = feature x →
        feature sR = feature y →
        feature (sL * sR) = feature (x * y))
    (loss : Strings → A → A → ℝ)
    (gen : PairGenerator Strings A)
    (x : Strings) (R : ℕ) (T : BinTree Strings)
    (hp : S T = x) (hR : R ≥ 1)
    (h_meas : OracleMeasurableLoss loss (encodedOracle (Strings := Strings) feature))
    (h_pair : OracleIndexedPairGen gen (encodedOracle (Strings := Strings) feature))
    (M : ℝ) (hM : 0 ≤ M)
    (hbound : ∀ w z, D (encodedOracle (Strings := Strings) feature) w z ≤ M) :
    ∑' x', (PMF.pure x x').toReal * ∑' p, (gen x' p).toReal * loss x' p.1 p.2 =
    ∑' z,
      (ZR (sketchSummarizer (identitySketchOperator (Strings := Strings))) x R T z).toReal *
      ∑' p, (gen z p).toReal * loss z p.1 p.2 := by
  exact preference_learning_equivalence_via_ZR_of_sketch
    (op := identitySketchOperator (Strings := Strings))
    (fstar := encodedOracle (Strings := Strings) feature)
    (loss := loss) (gen := gen) (x := x) (R := R) (T := T)
    hp hR
    (identitySketch_leaf_preserving _)
    (identitySketch_merge_compatible_of_feature_congruent
      (feature := feature) h_feature_congr)
    (identitySketch_summary_compatible (Strings := Strings))
    h_meas h_pair M hM hbound

/-- Reusable pairwise-equivalence template for the paired non-identity sketch operator
with encoded feature oracle. -/
theorem preference_learning_equivalence_via_ZR_of_paired_encoded_feature
    {Feature : Type*} [Encodable Feature]
    (feature : Strings → Feature)
    (h_feature_congr :
      ∀ sL sR x y,
        feature sL = feature x →
        feature sR = feature y →
        feature (sL * sR) = feature (x * y))
    (loss : Strings → A → A → ℝ)
    (gen : PairGenerator Strings A)
    (x : Strings) (R : ℕ) (T : BinTree Strings)
    (hp : S T = x) (hR : R ≥ 1)
    (h_meas : OracleMeasurableLoss loss (encodedOracle (Strings := Strings) feature))
    (h_pair : OracleIndexedPairGen gen (encodedOracle (Strings := Strings) feature))
    (M : ℝ) (hM : 0 ≤ M)
    (hbound : ∀ w z, D (encodedOracle (Strings := Strings) feature) w z ≤ M) :
    ∑' x', (PMF.pure x x').toReal * ∑' p, (gen x' p).toReal * loss x' p.1 p.2 =
    ∑' z,
      (ZR (sketchSummarizer (pairedSketchOperator (Strings := Strings))) x R T z).toReal *
      ∑' p, (gen z p).toReal * loss z p.1 p.2 := by
  exact preference_learning_equivalence_via_ZR_of_sketch
    (op := pairedSketchOperator (Strings := Strings))
    (fstar := encodedOracle (Strings := Strings) feature)
    (loss := loss) (gen := gen) (x := x) (R := R) (T := T)
    hp hR
    (pairedSketch_leaf_preserving _)
    (pairedSketch_merge_compatible_of_feature_congruent
      (feature := feature) h_feature_congr)
    (pairedSketch_summary_compatible (Strings := Strings))
    h_meas h_pair M hM hbound

end PreferenceObjectives

/-!
## Group-Objective Recovery (GRPO / GRPO-RL)
-/

section GroupObjectives

variable {A : Type*}

/-- GRPO listwise equivalence via ZR recovered from sketch assumptions. -/
theorem grpo_equivalence_via_ZR_of_sketch
    {Y : Type*} [MetricSpace Y] {k : ℕ}
    (op : SketchOperator Strings Sketch) (fstar : Strings → Y)
    (pol : Policy' Strings A) (ranker : Strings → GroupRanker A k)
    (gen : GroupGenerator Strings A k)
    (x : Strings) (R : ℕ) (T : BinTree Strings)
    (hp : S T = x) (hR : R ≥ 1)
    (h_leaf : SketchLeafPreserving op fstar)
    (h_merge : SketchMergeCompatible op fstar)
    (h_compat : SketchSummaryCompatible op)
    (h_pol : GRPOOracleMeasurable (Y := Y) pol fstar)
    (h_ranker : OracleIndexedRanker (Y := Y) ranker fstar)
    (h_gen : OracleIndexedGroupGen (Y := Y) gen fstar)
    (M : ℝ) (hM : 0 ≤ M) (hbound : ∀ w z, D fstar w z ≤ M) :
    ExpectedGRPOLoss pol ranker (PMF.pure x) gen =
    ExpectedGRPOLoss pol ranker (ZR (sketchSummarizer op) x R T) gen := by
  let laws :=
    local_laws_of_sketch (op := op) (fstar := fstar) (T := T) h_leaf h_merge h_compat
  exact grpo_equivalence_via_ZR (fstar := fstar) (pol := pol) (ranker := ranker) (gen := gen)
    (g := sketchSummarizer op) (x := x) (R := R) (T := T)
    hp laws.law1 laws.law2 laws.law3 hR h_pol h_ranker h_gen M hM hbound

/-- GRPO-RL equivalence via ZR recovered from sketch assumptions. -/
theorem grpo_rl_equivalence_via_ZR_of_sketch
    {Y : Type*} [MetricSpace Y] (k : ℕ)
    (op : SketchOperator Strings Sketch) (fstar : Strings → Y)
    (pol pol_old pol_ref : Policy' Strings A)
    (reward : Strings → A → ℝ) (eps beta : ℝ)
    (gen : GroupGenerator Strings A k)
    (x : Strings) (R : ℕ) (T : BinTree Strings)
    (hp : S T = x) (hR : R ≥ 1)
    (h_leaf : SketchLeafPreserving op fstar)
    (h_merge : SketchMergeCompatible op fstar)
    (h_compat : SketchSummaryCompatible op)
    (h_meas : OracleMeasurableGRPORLLoss k pol pol_old pol_ref reward eps beta fstar)
    (h_gen : OracleIndexedGroupGen (Y := Y) gen fstar)
    (M : ℝ) (hM : 0 ≤ M) (hbound : ∀ w z, D fstar w z ≤ M) :
    ExpectedGRPORLLoss pol pol_old pol_ref reward eps beta (PMF.pure x) gen =
    ExpectedGRPORLLoss pol pol_old pol_ref reward eps beta (ZR (sketchSummarizer op) x R T) gen := by
  let laws :=
    local_laws_of_sketch (op := op) (fstar := fstar) (T := T) h_leaf h_merge h_compat
  exact grpo_rl_equivalence_via_ZR (k := k) (fstar := fstar)
    (pol := pol) (pol_old := pol_old) (pol_ref := pol_ref)
    (reward := reward) (eps := eps) (beta := beta) (gen := gen)
    (g := sketchSummarizer op) (x := x) (R := R) (T := T)
    hp laws.law1 laws.law2 laws.law3 hR h_meas h_gen M hM hbound

end GroupObjectives

end
