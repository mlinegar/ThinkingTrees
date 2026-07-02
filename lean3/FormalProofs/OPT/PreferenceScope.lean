import FormalProofs.OPT.TheoremBackingMeasurementError
import FormalProofs.OPT.ReadoutAlignment
import FormalProofs.OPT.ExactUtilityTransportInstances
import FormalProofs.OPT.MergeablePreference
import FormalProofs.OPT.MergeableReduction
import FormalProofs.OPT.ClassicalSketchLocalLaws
import FormalProofs.OPT.MarkovSimulationValidation
import FormalProofs.OPT.CounterexampleExistence
import FormalProofs.OPT.OracleFiberRelations
import FormalProofs.OPT.OracleSufficientCompression
import FormalProbability.ML.MergeableSummaries

/-!
# FormalProofs/OPT/PreferenceScope.lean

Lean-facing vocabulary for the scope of preferences supported by C-TreePO.

The intended boundary is not additive separability over leaves.  The supported
class is broader and cleaner:

* the decision-relevant information factors through a task state / oracle fiber;
* local C-TreePO laws preserve that state through the text reduction; or
* an exact mergeable state recovers that state by a tree fold.

Once one of those state-preservation routes is available, downstream utilities
can be arbitrary functions of the preserved state.  Nonseparable interactions
are allowed when the state carries the interaction variables.
-/

set_option linter.mathlibStandardSet false

open scoped BigOperators
open scoped Real
open scoped Nat
open scoped Classical
open scoped Pointwise
open scoped MeasureTheory
open MeasureTheory

set_option maxHeartbeats 0
set_option maxRecDepth 4000
set_option relaxedAutoImplicit false
set_option autoImplicit false

noncomputable section

namespace FormalProofs.OPT

/-!
## Vocabulary
-/

/-- A downstream preference/readout is state-factored when it can be recovered
from the task state alone.  This is the preference-scope counterpart of
`ReadoutFactorsThroughFeature`. -/
abbrev PreferenceFactorsThroughState
    {Doc State Pref : Type*}
    (state : Doc → State) (pref : Doc → Pref) : Prop :=
  ReadoutFactorsThroughFeature state pref

/-- A loss is state-factored when its document dependence is only through the
task state. -/
def LossFactorsThroughState
    {Doc State Action : Type*}
    (state : Doc → State) (loss : Doc → Action → ℝ) : Prop :=
  ∃ objective : State → Action → ℝ,
    ∀ x a, loss x a = objective (state x) a

/-- Feature-indexed objectives are the canonical state-factored losses. -/
theorem lossFactorsThroughState_featureIndexedObjective
    {Doc State Action : Type*}
    (state : Doc → State) (objective : State → Action → ℝ) :
    LossFactorsThroughState state
      (featureIndexedObjective (Strings := Doc) state objective) := by
  refine ⟨objective, ?_⟩
  intro x a
  rfl

/-!
## Preference, Summary, and State Relations

This section names the three relations that are often conflated in prose:

* `pref = readout ∘ state`, the underlying preference shape;
* `state ∘ g = state`, the summary operator's state preservation law; and
* `state (x * y) = merge (state x) (state y)`, the global decomposability law.

The C-TreePO local laws audit the second relation on the tree surface; exact
mergeable sketches supply the third relation by construction.
-/

/-!
### Fibers

A fiber is the preimage of one value under a map.  For `sigma : X → S`, the
fiber at `s : S` is `{x | sigma x = s}`.  The induced "same fiber" relation is
`sigma x = sigma x'`.

This is the ordinary preimage/equivalence-class use of "fiber" from mathematics
and sufficient-statistic arguments.  It is distinct from the topological
"fiber surface" terminology in Futer's fiber-detection theorem.  The connection
for C-TreePO is decision theoretic: a state is sufficient for a preference when
state fibers refine preference fibers.  See Fisher's sufficient-statistic
lineage, Blackwell's comparison of experiments, Doob-Dynkin factorization, and
the mergeable-summary state interface of Agarwal--Cormode--Huang--Phillips--
Wei--Yi.

The names below are intentionally explicit so prose can say:

* `MapFiber m y x`: `x` lies in the preimage/fiber of `y` under `m`;
* `SameMapFiber m x x'`: `x` and `x'` are identified by `m`;
* `StateFiber sigma s x`: the task-state specialization;
* `PreferenceFiber pref p x`: the downstream preference specialization; and
* `StateFibersRefinePreferenceFibers sigma pref`: no preference distinction is
  made inside a state fiber.
-/

/-- The fiber/preimage of a value `y` under a map `m`: all inputs mapped exactly
to `y`. -/
def MapFiber
    {X Y : Type*}
    (m : X → Y) (y : Y) (x : X) : Prop :=
  m x = y

/-- Two inputs lie in the same fiber of `m` when `m` identifies them. -/
def SameMapFiber
    {X Y : Type*}
    (m : X → Y) (x x' : X) : Prop :=
  m x = m x'

/-- Same-fiber is an equivalence relation for every map. -/
theorem sameMapFiber_equivalence
    {X Y : Type*} (m : X → Y) :
    Equivalence (SameMapFiber m) where
  refl := by
    intro x
    rfl
  symm := by
    intro x x' h
    exact h.symm
  trans := by
    intro x y z hxy hyz
    exact hxy.trans hyz

/-- Same-fiber equality is equivalent to membership in one common value fiber. -/
theorem sameMapFiber_iff_exists_common_value
    {X Y : Type*} {m : X → Y} {x x' : X} :
    SameMapFiber m x x' ↔
      ∃ y : Y, MapFiber m y x ∧ MapFiber m y x' := by
  constructor
  · intro h
    exact ⟨m x, rfl, h.symm⟩
  · intro h
    rcases h with ⟨y, hx, hx'⟩
    exact hx.trans hx'.symm

/-- The fiber of a state value `s`: all documents whose state is exactly `s`. -/
abbrev StateFiber
    {Doc State : Type*}
    (state : Doc → State) (s : State) (x : Doc) : Prop :=
  MapFiber state s x

/-- Two documents are in the same state fiber when the state identifies them. -/
abbrev SameStateFiber
    {Doc State : Type*}
    (state : Doc → State) (x x' : Doc) : Prop :=
  SameMapFiber state x x'

/-- Same-state-fiber is an equivalence relation. -/
theorem sameStateFiber_equivalence
    {Doc State : Type*} (state : Doc → State) :
    Equivalence (SameStateFiber state) :=
  sameMapFiber_equivalence state

theorem sameStateFiber_refl
    {Doc State : Type*} (state : Doc → State) (x : Doc) :
    SameStateFiber state x x := by
  rfl

theorem sameStateFiber_symm
    {Doc State : Type*} {state : Doc → State} {x x' : Doc}
    (h : SameStateFiber state x x') :
    SameStateFiber state x' x := by
  exact h.symm

theorem sameStateFiber_trans
    {Doc State : Type*} {state : Doc → State} {x y z : Doc}
    (hxy : SameStateFiber state x y)
    (hyz : SameStateFiber state y z) :
    SameStateFiber state x z := by
  exact hxy.trans hyz

/-- Same-state-fiber equality is membership in a common state fiber. -/
theorem sameStateFiber_iff_exists_common_state
    {Doc State : Type*} {state : Doc → State} {x x' : Doc} :
    SameStateFiber state x x' ↔
      ∃ s : State, StateFiber state s x ∧ StateFiber state s x' :=
  sameMapFiber_iff_exists_common_value

/-- Two points in the same named state fiber are in the same state-fiber
equivalence class. -/
theorem sameStateFiber_of_stateFiber
    {Doc State : Type*} {state : Doc → State} {s : State} {x x' : Doc}
    (hx : StateFiber state s x)
    (hx' : StateFiber state s x') :
    SameStateFiber state x x' :=
  hx.trans hx'.symm

/-- Moving along the same-state-fiber relation preserves membership in a named
state fiber. -/
theorem stateFiber_of_sameStateFiber_left
    {Doc State : Type*} {state : Doc → State} {s : State} {x x' : Doc}
    (hx : StateFiber state s x)
    (h : SameStateFiber state x x') :
    StateFiber state s x' :=
  h.symm.trans hx

/-- Symmetric form of `stateFiber_of_sameStateFiber_left`. -/
theorem stateFiber_of_sameStateFiber_right
    {Doc State : Type*} {state : Doc → State} {s : State} {x x' : Doc}
    (hx' : StateFiber state s x')
    (h : SameStateFiber state x x') :
    StateFiber state s x :=
  h.trans hx'

/-- Fiber of a downstream preference/readout value. -/
abbrev PreferenceFiber
    {Doc Pref : Type*}
    (pref : Doc → Pref) (p : Pref) (x : Doc) : Prop :=
  MapFiber pref p x

/-- Same-preference-fiber relation: two inputs receive the same preference
value. -/
abbrev SamePreferenceFiber
    {Doc Pref : Type*}
    (pref : Doc → Pref) (x x' : Doc) : Prop :=
  SameMapFiber pref x x'

/-- Exact value fiber of an oracle/readout.  This is the equality-based version;
`SameOracleFiber` below is the metric zero-distance version used by existing
theorem-backed proofs. -/
abbrev OracleValueFiber
    {Doc Y : Type*}
    (oracle : Doc → Y) (y : Y) (x : Doc) : Prop :=
  MapFiber oracle y x

/-- Equality-based same-oracle-value fiber. -/
abbrev SameOracleValueFiber
    {Doc Y : Type*}
    (oracle : Doc → Y) (x x' : Doc) : Prop :=
  SameMapFiber oracle x x'

/-- For metric-valued theorem oracles, the existing zero-distance oracle fiber
is the same as equality of oracle values. -/
theorem sameOracleFiber_iff_sameOracleValueFiber
    {Doc Y : Type*} [Monoid Doc] [BoundedMetricSpace Y]
    {fstar : Doc → Y} {x x' : Doc} :
    SameOracleFiber fstar x x' ↔ SameOracleValueFiber fstar x x' := by
  constructor
  · intro h
    exact dist_eq_zero.mp h
  · intro h
    rw [SameOracleFiber]
    exact dist_eq_zero.mpr h

/-!
### Oracle-Compatible Readouts

The paper corollary uses the minimal condition: a downstream readout is
certified by oracle preservation exactly when it is constant on oracle fibers.
Equivalently, it respects the same zero-distance relation that the local laws
preserve.
-/

/-- A document readout is oracle-compatible when it is constant on the oracle
zero-distance equivalence classes. -/
def OracleCompatibleReadout
    {Doc Y Score : Type*} [PseudoMetricSpace Y]
    (fstar : Doc → Y) (Φ : Doc → Score) : Prop :=
  ∀ {x x' : Doc}, dist (fstar x) (fstar x') = 0 → Φ x = Φ x'

/-- A pairwise preference is oracle-compatible when changing either input within
its oracle zero-distance class does not change the comparison. -/
def PairwiseOracleCompatiblePreference
    {Doc Y Score : Type*} [PseudoMetricSpace Y]
    (fstar : Doc → Y) (P : Doc → Doc → Score) : Prop :=
  ∀ {x₁ x₁' x₂ x₂' : Doc},
    dist (fstar x₁) (fstar x₁') = 0 →
      dist (fstar x₂) (fstar x₂') = 0 →
        P x₁ x₂ = P x₁' x₂'

/-- A readout that literally factors through the oracle is oracle-compatible
when its post-processing map is constant on zero-distance oracle outputs. -/
theorem oracleCompatibleReadout_of_factorization
    {Doc Y Score : Type*} [PseudoMetricSpace Y]
    {fstar : Doc → Y} {Φ : Doc → Score}
    (ψ : Y → Score)
    (hψ : ∀ {y y' : Y}, dist y y' = 0 → ψ y = ψ y')
    (hΦ : ∀ x : Doc, Φ x = ψ (fstar x)) :
    OracleCompatibleReadout fstar Φ := by
  intro x x' hzero
  calc
    Φ x = ψ (fstar x) := hΦ x
    _ = ψ (fstar x') := hψ hzero
    _ = Φ x' := (hΦ x').symm

/-- In a separated metric oracle space, any literal post-processing of the oracle
is oracle-compatible. -/
theorem oracleCompatibleReadout_of_metric_factorization
    {Doc Y Score : Type*} [MetricSpace Y]
    {fstar : Doc → Y} {Φ : Doc → Score}
    (ψ : Y → Score)
    (hΦ : ∀ x : Doc, Φ x = ψ (fstar x)) :
    OracleCompatibleReadout fstar Φ := by
  exact oracleCompatibleReadout_of_factorization ψ
    (by
      intro y y' hzero
      rw [dist_eq_zero.mp hzero])
    hΦ

/-- Oracle zero-distance preservation transports any oracle-compatible readout. -/
theorem oracleCompatibleReadout_eq_of_zeroDist
    {Doc Y Score : Type*} [PseudoMetricSpace Y]
    {fstar : Doc → Y} {Φ : Doc → Score}
    {z x : Doc}
    (hΦ : OracleCompatibleReadout fstar Φ)
    (hzero : dist (fstar z) (fstar x) = 0) :
    Φ z = Φ x :=
  hΦ hzero

/-- Same-oracle-fiber form of `oracleCompatibleReadout_eq_of_zeroDist`. -/
theorem oracleCompatibleReadout_eq_of_sameOracleFiber
    {Doc Y Score : Type*} [Monoid Doc] [BoundedMetricSpace Y]
    {fstar : Doc → Y} {Φ : Doc → Score}
    {z x : Doc}
    (hΦ : OracleCompatibleReadout fstar Φ)
    (hFiber : SameOracleFiber fstar z x) :
    Φ z = Φ x :=
  oracleCompatibleReadout_eq_of_zeroDist hΦ hFiber

/-- Pairwise oracle zero-distance preservation transports any oracle-compatible
pairwise preference. -/
theorem pairwiseOracleCompatiblePreference_eq_of_zeroDist
    {Doc Y Score : Type*} [PseudoMetricSpace Y]
    {fstar : Doc → Y} {P : Doc → Doc → Score}
    {z₁ x₁ z₂ x₂ : Doc}
    (hP : PairwiseOracleCompatiblePreference fstar P)
    (hzero₁ : dist (fstar z₁) (fstar x₁) = 0)
    (hzero₂ : dist (fstar z₂) (fstar x₂) = 0) :
    P z₁ z₂ = P x₁ x₂ :=
  hP hzero₁ hzero₂

/-- Every realized multi-round C-Tree summary from an exact theorem-backed tree
preserves any oracle-compatible root readout. -/
theorem oracleCompatibleReadout_eq_on_ZR_support_of_exactTheoremBacked
    {Doc Y Score : Type*} [Monoid Doc] [BoundedMetricSpace Y]
    {g : Summarizer Doc} {T : BinTree Doc} {x z : Doc} {R : ℕ}
    {fstar : Doc → Y} {Φ : Doc → Score}
    (hp : S T = x)
    (hExact : ExactTheoremBacked g T fstar)
    (hR : R ≥ 1)
    (hΦ : OracleCompatibleReadout fstar Φ)
    (hz : z ∈ (ZR g x R T).support) :
    Φ z = Φ x :=
  hΦ
    (zero_distortion_on_ZR_support_of_exactTheoremBacked
      (g := g) (T := T) (x := x) (R := R) (fstar := fstar)
      hp hExact hR z hz)

/-- Pairwise version: replacing both documents by exact theorem-backed C-Tree
root summaries preserves any oracle-compatible pairwise preference. -/
theorem pairwiseOracleCompatiblePreference_eq_on_ZR_support_of_exactTheoremBacked
    {Doc Y Score : Type*} [Monoid Doc] [BoundedMetricSpace Y]
    {g₁ g₂ : Summarizer Doc} {T₁ T₂ : BinTree Doc}
    {x₁ x₂ z₁ z₂ : Doc} {R₁ R₂ : ℕ}
    {fstar : Doc → Y} {P : Doc → Doc → Score}
    (hp₁ : S T₁ = x₁)
    (hp₂ : S T₂ = x₂)
    (hExact₁ : ExactTheoremBacked g₁ T₁ fstar)
    (hExact₂ : ExactTheoremBacked g₂ T₂ fstar)
    (hR₁ : R₁ ≥ 1)
    (hR₂ : R₂ ≥ 1)
    (hP : PairwiseOracleCompatiblePreference fstar P)
    (hz₁ : z₁ ∈ (ZR g₁ x₁ R₁ T₁).support)
    (hz₂ : z₂ ∈ (ZR g₂ x₂ R₂ T₂).support) :
    P z₁ z₂ = P x₁ x₂ :=
  hP
    (zero_distortion_on_ZR_support_of_exactTheoremBacked
      (g := g₁) (T := T₁) (x := x₁) (R := R₁) (fstar := fstar)
      hp₁ hExact₁ hR₁ z₁ hz₁)
    (zero_distortion_on_ZR_support_of_exactTheoremBacked
      (g := g₂) (T := T₂) (x := x₂) (R := R₂) (fstar := fstar)
      hp₂ hExact₂ hR₂ z₂ hz₂)

/-!
### Node-Level and Readout-Error Variants
-/

/-- Every realized subtree summary from an exact theorem-backed tree remains in
the same oracle fiber as that subtree's raw span. -/
theorem node_support_sameOracleFiber_of_exactTheoremBacked
    {Doc Y : Type*} [Monoid Doc] [BoundedMetricSpace Y]
    {g : Summarizer Doc} {T u : BinTree Doc} {z : Doc}
    {fstar : Doc → Y}
    (hExact : ExactTheoremBacked g T fstar)
    (hu : u ∈ subtrees T)
    (hz : z ∈ (reduce g u).support) :
    SameOracleFiber fstar z (S u) := by
  have hEgu : Egu g u (fun y => D fstar y (S u)) = 0 :=
    nodewise_preservation g T u fstar hu
      hExact.localLaws.law1 hExact.localLaws.law2
  have hExp : Exp (reduce g u) (fun y => D fstar y (S u)) = 0 := by
    simpa [Exp, Egu] using hEgu
  exact dist_zero_on_support_of_Exp_zero
    (p := reduce g u) (fstar := fstar) (x := S u) hExp z hz

/-- Node-level version of the oracle-compatible readout corollary. -/
theorem oracleCompatibleReadout_eq_on_node_support_of_exactTheoremBacked
    {Doc Y Score : Type*} [Monoid Doc] [BoundedMetricSpace Y]
    {g : Summarizer Doc} {T u : BinTree Doc} {z : Doc}
    {fstar : Doc → Y} {Φ : Doc → Score}
    (hExact : ExactTheoremBacked g T fstar)
    (hu : u ∈ subtrees T)
    (hΦ : OracleCompatibleReadout fstar Φ)
    (hz : z ∈ (reduce g u).support) :
    Φ z = Φ (S u) :=
  hΦ (node_support_sameOracleFiber_of_exactTheoremBacked
    (g := g) (T := T) (u := u) (z := z) (fstar := fstar)
    hExact hu hz)

/-- A readout is Lipschitz against oracle distance when readout movement is
bounded by `L` times oracle movement. -/
def ReadoutLipschitzOnOracle
    {Doc Y Score : Type*} [PseudoMetricSpace Y] [PseudoMetricSpace Score]
    (fstar : Doc → Y) (Φ : Doc → Score) (L : ℝ) : Prop :=
  ∀ x x' : Doc, dist (Φ x) (Φ x') ≤ L * dist (fstar x) (fstar x')

/-- A learned or observed readout has uniform error `ε` relative to a target
readout. -/
def UniformReadoutError
    {Doc Score : Type*} [PseudoMetricSpace Score]
    (Φhat Φ : Doc → Score) (ε : ℝ) : Prop :=
  ∀ x : Doc, dist (Φhat x) (Φ x) ≤ ε

/-- Oracle-distance error transfers through a Lipschitz readout. -/
theorem readout_dist_le_of_oracle_dist_le
    {Doc Y Score : Type*} [PseudoMetricSpace Y] [PseudoMetricSpace Score]
    {fstar : Doc → Y} {Φ : Doc → Score}
    {z x : Doc} {L ε : ℝ}
    (hL_nonneg : 0 ≤ L)
    (hLip : ReadoutLipschitzOnOracle fstar Φ L)
    (hOracle : dist (fstar z) (fstar x) ≤ ε) :
    dist (Φ z) (Φ x) ≤ L * ε :=
  (hLip z x).trans (mul_le_mul_of_nonneg_left hOracle hL_nonneg)

/-- If the target readout is Lipschitz against oracle distance and `Φhat` has
uniform readout error `εReadout`, then the observed readout discrepancy is
bounded by oracle error plus two readout-error terms. -/
theorem estimatedReadout_dist_le_of_oracle_dist_le_and_readoutError
    {Doc Y Score : Type*} [PseudoMetricSpace Y] [PseudoMetricSpace Score]
    {fstar : Doc → Y} {Φhat Φ : Doc → Score}
    {z x : Doc} {L εOracle εReadout : ℝ}
    (hL_nonneg : 0 ≤ L)
    (hLip : ReadoutLipschitzOnOracle fstar Φ L)
    (hErr : UniformReadoutError Φhat Φ εReadout)
    (hOracle : dist (fstar z) (fstar x) ≤ εOracle) :
    dist (Φhat z) (Φhat x) ≤ L * εOracle + 2 * εReadout := by
  have hzx : dist (Φ z) (Φ x) ≤ L * εOracle :=
    readout_dist_le_of_oracle_dist_le hL_nonneg hLip hOracle
  have hzerr : dist (Φhat z) (Φ z) ≤ εReadout := hErr z
  have hxerr : dist (Φ x) (Φhat x) ≤ εReadout := by
    simpa [dist_comm] using hErr x
  have htri₁ :
      dist (Φhat z) (Φhat x) ≤
        dist (Φhat z) (Φ z) + dist (Φ z) (Φhat x) :=
    dist_triangle (Φhat z) (Φ z) (Φhat x)
  have htri₂ :
      dist (Φ z) (Φhat x) ≤
        dist (Φ z) (Φ x) + dist (Φ x) (Φhat x) :=
    dist_triangle (Φ z) (Φ x) (Φhat x)
  calc
    dist (Φhat z) (Φhat x)
        ≤ dist (Φhat z) (Φ z) + dist (Φ z) (Φhat x) := htri₁
    _ ≤ dist (Φhat z) (Φ z) +
          (dist (Φ z) (Φ x) + dist (Φ x) (Φhat x)) :=
        add_le_add (le_refl _) htri₂
    _ ≤ εReadout + (L * εOracle + εReadout) := by
        exact add_le_add hzerr (add_le_add hzx hxerr)
    _ = L * εOracle + 2 * εReadout := by ring

/-- Exact theorem-backed node summaries have only readout error when evaluated
through an estimated readout. -/
theorem estimatedReadout_dist_le_on_node_support_of_exactTheoremBacked
    {Doc Y Score : Type*} [Monoid Doc] [BoundedMetricSpace Y]
    [PseudoMetricSpace Score]
    {g : Summarizer Doc} {T u : BinTree Doc} {z : Doc}
    {fstar : Doc → Y} {Φhat Φ : Doc → Score} {εReadout : ℝ}
    (hExact : ExactTheoremBacked g T fstar)
    (hu : u ∈ subtrees T)
    (hΦ : OracleCompatibleReadout fstar Φ)
    (hErr : UniformReadoutError Φhat Φ εReadout)
    (hz : z ∈ (reduce g u).support) :
    dist (Φhat z) (Φhat (S u)) ≤ 2 * εReadout := by
  have hEq : Φ z = Φ (S u) :=
    oracleCompatibleReadout_eq_on_node_support_of_exactTheoremBacked
      (g := g) (T := T) (u := u) (z := z) (fstar := fstar) (Φ := Φ)
      hExact hu hΦ hz
  have hzerr : dist (Φhat z) (Φ z) ≤ εReadout := hErr z
  have hxerr : dist (Φ (S u)) (Φhat (S u)) ≤ εReadout := by
    simpa [dist_comm] using hErr (S u)
  have htri :
      dist (Φhat z) (Φhat (S u)) ≤
        dist (Φhat z) (Φ z) + dist (Φ z) (Φhat (S u)) :=
    dist_triangle (Φhat z) (Φ z) (Φhat (S u))
  calc
    dist (Φhat z) (Φhat (S u))
        ≤ dist (Φhat z) (Φ z) + dist (Φ z) (Φhat (S u)) := htri
    _ = dist (Φhat z) (Φ z) + dist (Φ (S u)) (Φhat (S u)) := by
        rw [hEq]
    _ ≤ εReadout + εReadout := add_le_add hzerr hxerr
    _ = 2 * εReadout := by ring

/-- Multi-round root-summary version of the readout-error bound. -/
theorem estimatedReadout_dist_le_on_ZR_support_of_exactTheoremBacked
    {Doc Y Score : Type*} [Monoid Doc] [BoundedMetricSpace Y]
    [PseudoMetricSpace Score]
    {g : Summarizer Doc} {T : BinTree Doc} {x z : Doc} {R : ℕ}
    {fstar : Doc → Y} {Φhat Φ : Doc → Score} {εReadout : ℝ}
    (hp : S T = x)
    (hExact : ExactTheoremBacked g T fstar)
    (hR : R ≥ 1)
    (hΦ : OracleCompatibleReadout fstar Φ)
    (hErr : UniformReadoutError Φhat Φ εReadout)
    (hz : z ∈ (ZR g x R T).support) :
    dist (Φhat z) (Φhat x) ≤ 2 * εReadout := by
  have hEq : Φ z = Φ x :=
    oracleCompatibleReadout_eq_on_ZR_support_of_exactTheoremBacked
      (g := g) (T := T) (x := x) (z := z) (R := R)
      (fstar := fstar) (Φ := Φ)
      hp hExact hR hΦ hz
  have hzerr : dist (Φhat z) (Φ z) ≤ εReadout := hErr z
  have hxerr : dist (Φ x) (Φhat x) ≤ εReadout := by
    simpa [dist_comm] using hErr x
  have htri :
      dist (Φhat z) (Φhat x) ≤
        dist (Φhat z) (Φ z) + dist (Φ z) (Φhat x) :=
    dist_triangle (Φhat z) (Φ z) (Φhat x)
  calc
    dist (Φhat z) (Φhat x)
        ≤ dist (Φhat z) (Φ z) + dist (Φ z) (Φhat x) := htri
    _ = dist (Φhat z) (Φ z) + dist (Φ x) (Φhat x) := by
        rw [hEq]
    _ ≤ εReadout + εReadout := add_le_add hzerr hxerr
    _ = 2 * εReadout := by ring

/-- A preference/readout respects state fibers when it gives the same answer to
documents with the same state. -/
def ReadoutRespectsStateFibers
    {Doc State Pref : Type*}
    (state : Doc → State) (pref : Doc → Pref) : Prop :=
  ∀ {x x' : Doc}, SameStateFiber state x x' → pref x = pref x'

/-- Explicit readout form of a state-factored preference:
`pref x = readout (state x)`. -/
def PreferenceReadoutOfState
    {Doc State Pref : Type*}
    (state : Doc → State) (readout : State → Pref) (pref : Doc → Pref) : Prop :=
  ∀ x, pref x = readout (state x)

/-- State fibers refine preference fibers when the state never identifies two
inputs that the preference separates. -/
def StateFibersRefinePreferenceFibers
    {Doc State Pref : Type*}
    (state : Doc → State) (pref : Doc → Pref) : Prop :=
  ∀ {x x' : Doc},
    SameStateFiber state x x' → SamePreferenceFiber pref x x'

/-- Refining preference fibers is the same predicate as respecting state
fibers, written in explicit partition language. -/
theorem stateFibersRefinePreferenceFibers_iff_respectsStateFibers
    {Doc State Pref : Type*}
    {state : Doc → State} {pref : Doc → Pref} :
    StateFibersRefinePreferenceFibers state pref ↔
      ReadoutRespectsStateFibers state pref := by
  rfl

/-- Explicit readout form implies the existential factorization predicate. -/
theorem preferenceReadoutOfState_factorsThroughState
    {Doc State Pref : Type*}
    {state : Doc → State} {readout : State → Pref} {pref : Doc → Pref}
    (h : PreferenceReadoutOfState state readout pref) :
    PreferenceFactorsThroughState state pref :=
  ⟨readout, h⟩

/-- State-factorization implies the preference is constant on state fibers. -/
theorem preferenceFactorsThroughState_respectsStateFibers
    {Doc State Pref : Type*}
    {state : Doc → State} {pref : Doc → Pref}
    (h : PreferenceFactorsThroughState state pref) :
    ReadoutRespectsStateFibers state pref := by
  rcases h with ⟨readout, hreadout⟩
  intro x x' hFiber
  rw [hreadout x, hreadout x', hFiber]

/-- Conversely, if a readout is constant on state fibers, then it factors
through the state.  Away from the image of `state`, the recovered readout can be
chosen arbitrarily, hence the harmless `Inhabited Pref` assumption. -/
theorem readoutRespectsStateFibers_factorsThroughState
    {Doc State Pref : Type*} [Inhabited Pref]
    {state : Doc → State} {pref : Doc → Pref}
    (h : ReadoutRespectsStateFibers state pref) :
    PreferenceFactorsThroughState state pref := by
  let readout : State → Pref := fun s =>
    if hs : ∃ x : Doc, state x = s then pref hs.choose else default
  refine ⟨readout, ?_⟩
  intro x
  dsimp [readout]
  have hs : ∃ y : Doc, state y = state x := ⟨x, rfl⟩
  simp [hs]
  have hFiber : SameStateFiber state hs.choose x := by
    exact hs.choose_spec
  exact h (sameStateFiber_symm hFiber)

/-- For inhabited preference codomains, factoring through a state is exactly
the same as being constant on that state's fibers. -/
theorem preferenceFactorsThroughState_iff_respectsStateFibers
    {Doc State Pref : Type*} [Inhabited Pref]
    {state : Doc → State} {pref : Doc → Pref} :
    PreferenceFactorsThroughState state pref ↔
      ReadoutRespectsStateFibers state pref := by
  constructor
  · exact preferenceFactorsThroughState_respectsStateFibers
  · exact readoutRespectsStateFibers_factorsThroughState

/-- State factorization is equivalently the statement that state fibers refine
preference fibers. -/
theorem preferenceFactorsThroughState_iff_stateFibersRefinePreferenceFibers
    {Doc State Pref : Type*} [Inhabited Pref]
    {state : Doc → State} {pref : Doc → Pref} :
    PreferenceFactorsThroughState state pref ↔
      StateFibersRefinePreferenceFibers state pref := by
  simpa [StateFibersRefinePreferenceFibers, ReadoutRespectsStateFibers,
    SamePreferenceFiber, SameMapFiber]
    using (preferenceFactorsThroughState_iff_respectsStateFibers
      (state := state) (pref := pref))

/-- A named state fiber maps into the corresponding named preference fiber
under an explicit state readout. -/
theorem stateFiber_subset_preferenceFiber_of_readout
    {Doc State Pref : Type*}
    {state : Doc → State} {readout : State → Pref} {pref : Doc → Pref}
    (hPref : PreferenceReadoutOfState state readout pref)
    {s : State} {x : Doc}
    (hx : StateFiber state s x) :
    PreferenceFiber pref (readout s) x := by
  rw [PreferenceFiber, MapFiber, hPref x, hx]

/-- Same-state-fiber pairs are same-preference-fiber pairs under an explicit
state readout. -/
theorem sameStateFiber_implies_samePreferenceFiber_of_stateReadout
    {Doc State Pref : Type*}
    {state : Doc → State} {readout : State → Pref} {pref : Doc → Pref}
    (hPref : PreferenceReadoutOfState state readout pref)
    {x x' : Doc}
    (hFiber : SameStateFiber state x x') :
    SamePreferenceFiber pref x x' := by
  rw [SamePreferenceFiber, SameMapFiber, hPref x, hPref x', hFiber]

/-- A deterministic summary operator preserves the underlying task state. -/
def SummaryPreservesState
    {Doc State : Type*}
    (state : Doc → State) (g : Doc → Doc) : Prop :=
  ∀ x, state (g x) = state x

/-- State preservation by `g` is exactly same-state-fiber preservation. -/
theorem summaryPreservesState_iff_sameStateFiber
    {Doc State : Type*}
    {state : Doc → State} {g : Doc → Doc} :
    SummaryPreservesState state g ↔
      ∀ x, SameStateFiber state (g x) x := by
  rfl

/-- A global task state decomposes over concatenation by a binary merge. -/
def StateDecomposesBy
    {Doc State : Type*} [Monoid Doc]
    (state : Doc → State) (merge : State → State → State) : Prop :=
  ∀ x y, state (x * y) = merge (state x) (state y)

/-- The two-route summary law stated directly on an underlying task state. -/
def SummaryMergePreservesState
    {Doc State : Type*} [Monoid Doc]
    (state : Doc → State) (g : Doc → Doc) : Prop :=
  ∀ x y, state (g (g x * g y)) = state (x * y)

/-- Two-route state preservation is exactly same-state-fiber preservation along
the merge route. -/
theorem summaryMergePreservesState_iff_sameStateFiber
    {Doc State : Type*} [Monoid Doc]
    {state : Doc → State} {g : Doc → Doc} :
    SummaryMergePreservesState state g ↔
      ∀ x y, SameStateFiber state (g (g x * g y)) (x * y) := by
  rfl

/-- If `g` preserves a decomposable state pointwise, it also preserves that
state along the two-route merge path. -/
theorem summaryMergePreservesState_of_preservesState_and_stateDecomposes
    {Doc State : Type*} [Monoid Doc]
    {state : Doc → State} {merge : State → State → State} {g : Doc → Doc}
    (hPreserve : SummaryPreservesState state g)
    (hDecompose : StateDecomposesBy state merge) :
    SummaryMergePreservesState state g := by
  intro x y
  calc
    state (g (g x * g y)) = state (g x * g y) := hPreserve (g x * g y)
    _ = merge (state (g x)) (state (g y)) := hDecompose (g x) (g y)
    _ = merge (state x) (state y) := by rw [hPreserve x, hPreserve y]
    _ = state (x * y) := (hDecompose x y).symm

/-- A state-preserving summary operator preserves any preference read out from
that state. -/
theorem summaryPreservesPreference_of_stateReadout
    {Doc State Pref : Type*}
    {state : Doc → State} {readout : State → Pref} {pref : Doc → Pref}
    {g : Doc → Doc}
    (hPref : PreferenceReadoutOfState state readout pref)
    (hState : SummaryPreservesState state g) :
    ∀ x, pref (g x) = pref x := by
  intro x
  calc
    pref (g x) = readout (state (g x)) := hPref (g x)
    _ = readout (state x) := by rw [hState x]
    _ = pref x := (hPref x).symm

/-- A two-route state-preservation law preserves any preference read out from
that state along the two-route merge path. -/
theorem summaryMergePreservesPreference_of_stateReadout
    {Doc State Pref : Type*} [Monoid Doc]
    {state : Doc → State} {readout : State → Pref} {pref : Doc → Pref}
    {g : Doc → Doc}
    (hPref : PreferenceReadoutOfState state readout pref)
    (hState : SummaryMergePreservesState state g) :
    ∀ x y, pref (g (g x * g y)) = pref (x * y) := by
  intro x y
  calc
    pref (g (g x * g y)) = readout (state (g (g x * g y))) :=
      hPref (g (g x * g y))
    _ = readout (state (x * y)) := by rw [hState x y]
    _ = pref (x * y) := (hPref (x * y)).symm

/-- State preservation implies A1 for the encoded-state oracle.  This is the
`f/g` view with `f := encodedOracle state`: summarizing by `g` leaves the
theorem-facing oracle unchanged. -/
theorem summaryPreservesState_implies_A1_encodedOracle
    {Doc State : Type*} [Monoid Doc] [Encodable State]
    {state : Doc → State} {g : Doc → Doc}
    (hState : SummaryPreservesState state g) :
    A1_global g (encodedOracle (Strings := Doc) state) := by
  intro x
  unfold D encodedOracle
  rw [hState x]
  exact dist_self _

/-- Two-route state preservation implies A2 for the encoded-state oracle. -/
theorem summaryMergePreservesState_implies_A2_encodedOracle
    {Doc State : Type*} [Monoid Doc] [Encodable State]
    {state : Doc → State} {g : Doc → Doc}
    (hState : SummaryMergePreservesState state g) :
    A2_global g (encodedOracle (Strings := Doc) state) := by
  intro x y
  unfold D encodedOracle
  rw [hState x y]
  exact dist_self _

/-- If an oracle identifies a state, then oracle-fiber equality implies
state-fiber equality. -/
theorem sameStateFiber_of_sameOracleFiber
    {Doc State Y : Type*} [Monoid Doc] [BoundedMetricSpace Y]
    {fstar : Doc → Y} {state : Doc → State} {x x' : Doc}
    (hRecover : OracleRecoversFeature fstar state)
    (hFiber : SameOracleFiber fstar x x') :
    SameStateFiber state x x' :=
  hRecover x x' hFiber

/-- A state is exactly composable when leaf encoding and binary merge recover
the same state as direct evaluation on the concatenated span. -/
structure ExactComposableState
    {Doc State : Type*} [Monoid Doc]
    (state : Doc → State)
    (encode : Doc → State)
    (merge : State → State → State) : Prop where
  encode_eq_state : ∀ x, encode x = state x
  merge_eq_state : ∀ x y, merge (state x) (state y) = state (x * y)

namespace ExactComposableState

variable {Doc State : Type*} [Monoid Doc]
variable {state : Doc → State}
variable {encode : Doc → State}
variable {merge : State → State → State}

/-- Exact composability is exactly the hypothesis needed by the generic
merge-fold theorem. -/
theorem mergeFold_eq_state
    (h : ExactComposableState state encode merge)
    (T : BinTree Doc) :
    mergeFold encode merge T = state (S T) :=
  mergeFold_eq_feature encode merge state h.encode_eq_state h.merge_eq_state T

end ExactComposableState

/-!
## Global and Local State

Sometimes the theorem-facing state `globalState` is not itself the tree-carried
object.  A local sketch state can be richer, encoded at leaves and merged
bottom-up, with a decoder back to the global/task state.  This is the formal
version of "the local state realizes and preserves the global state."
-/

/-- A local merge state realizes a global task state when leaf encoding,
local-state merge, and decoding agree with direct global-state evaluation. -/
structure LocalStateRealizesGlobalState
    {Doc GlobalState LocalState : Type*} [Monoid Doc]
    (globalState : Doc → GlobalState)
    (localState : Doc → LocalState)
    (encode : Doc → LocalState)
    (mergeLocal : LocalState → LocalState → LocalState)
    (decode : LocalState → GlobalState) : Prop where
  encode_eq_local : ∀ x, encode x = localState x
  merge_eq_local : ∀ x y, mergeLocal (localState x) (localState y) = localState (x * y)
  decode_eq_global : ∀ x, decode (localState x) = globalState x

namespace LocalStateRealizesGlobalState

variable {Doc GlobalState LocalState : Type*} [Monoid Doc]
variable {globalState : Doc → GlobalState}
variable {localState : Doc → LocalState}
variable {encode : Doc → LocalState}
variable {mergeLocal : LocalState → LocalState → LocalState}
variable {decode : LocalState → GlobalState}

/-- Folding the local merge state and decoding recovers the direct global state
of the tree span. -/
theorem decode_mergeFold_eq_global
    (h : LocalStateRealizesGlobalState globalState localState encode mergeLocal decode)
    (T : BinTree Doc) :
    decode (mergeFold encode mergeLocal T) = globalState (S T) := by
  have hFold :
      mergeFold encode mergeLocal T = localState (S T) :=
    mergeFold_eq_feature encode mergeLocal localState
      h.encode_eq_local h.merge_eq_local T
  calc
    decode (mergeFold encode mergeLocal T) = decode (localState (S T)) := by
      rw [hFold]
    _ = globalState (S T) := h.decode_eq_global (S T)

end LocalStateRealizesGlobalState

/-- A preference is captured by a local state when the local state realizes a
global task state and the preference factors through that global state. -/
structure GlobalLocalPreferenceShape
    {Doc GlobalState LocalState Pref : Type*} [Monoid Doc]
    (globalState : Doc → GlobalState)
    (localState : Doc → LocalState)
    (encode : Doc → LocalState)
    (mergeLocal : LocalState → LocalState → LocalState)
    (decode : LocalState → GlobalState)
    (pref : Doc → Pref) : Prop where
  local_realizes_global :
    LocalStateRealizesGlobalState globalState localState encode mergeLocal decode
  preference_factored : PreferenceFactorsThroughState globalState pref

namespace GlobalLocalPreferenceShape

variable {Doc GlobalState LocalState Pref : Type*} [Monoid Doc]
variable {globalState : Doc → GlobalState}
variable {localState : Doc → LocalState}
variable {encode : Doc → LocalState}
variable {mergeLocal : LocalState → LocalState → LocalState}
variable {decode : LocalState → GlobalState}
variable {pref : Doc → Pref}

/-- A global/local preference shape gives an explicit readout from the decoded
local folded state. -/
theorem readout_of_local_mergeFold
    (h : GlobalLocalPreferenceShape globalState localState encode mergeLocal decode pref)
    (T : BinTree Doc) :
    ∃ readout : GlobalState → Pref,
      pref (S T) = readout (decode (mergeFold encode mergeLocal T)) := by
  rcases h.preference_factored with ⟨readout, hreadout⟩
  refine ⟨readout, ?_⟩
  calc
    pref (S T) = readout (globalState (S T)) := hreadout (S T)
    _ = readout (decode (mergeFold encode mergeLocal T)) := by
      rw [← LocalStateRealizesGlobalState.decode_mergeFold_eq_global
        h.local_realizes_global T]

end GlobalLocalPreferenceShape

/-!
## Mergeable Preference Shape

The sketch-theoretic shape is:

`document ↦ mergeFold encode merge tree ↦ readout`.

Additive separability is the special case where `merge` is addition and the
readout is additive.  C-TreePO and mergeable sketches only need the first
displayed factorization; the readout itself can be nonlinear or thresholded.
-/

/-- A downstream preference is captured by a mergeable state when the state is
exactly composable and the preference factors through that state. -/
structure MergeablePreferenceShape
    {Doc State Pref : Type*} [Monoid Doc]
    (state : Doc → State)
    (encode : Doc → State)
    (merge : State → State → State)
    (pref : Doc → Pref) : Prop where
  state_composable : ExactComposableState state encode merge
  preference_factored : PreferenceFactorsThroughState state pref

namespace MergeablePreferenceShape

variable {Doc State Pref : Type*} [Monoid Doc]
variable {state : Doc → State}
variable {encode : Doc → State}
variable {merge : State → State → State}
variable {pref : Doc → Pref}

/-- A mergeable preference shape gives an explicit root readout from the folded
state.  This is the precise "preference captured by a mergeable sketch" form. -/
theorem readout_of_mergeFold
    (h : MergeablePreferenceShape state encode merge pref)
    (T : BinTree Doc) :
    ∃ readout : State → Pref,
      pref (S T) = readout (mergeFold encode merge T) := by
  rcases h.preference_factored with ⟨readout, hreadout⟩
  refine ⟨readout, ?_⟩
  calc
    pref (S T) = readout (state (S T)) := hreadout (S T)
    _ = readout (mergeFold encode merge T) := by
      rw [← ExactComposableState.mergeFold_eq_state h.state_composable T]

end MergeablePreferenceShape

/-!
## Relational Mergeable Preference Shape

Agarwal-style summaries need a slightly broader interface than
`MergeablePreferenceShape`: a merged state need not be definitionally equal to a
canonical state function.  It only has to be valid for the represented stream,
and the root readout must agree for every valid state.
-/

/-- A downstream preference is captured by a mergeable summary when state
validity, rather than canonical state equality, is the invariant propagated up
the tree.  The final preference is read from the merged root state. -/
structure RelationalMergeablePreferenceShape
    {α State Pref : Type*}
    (build : ML.MergeableSummary.Stream α → State)
    (merge : State → State → State)
    (valid : ML.MergeableSummary.Stream α → State → Prop)
    (readout : State → Pref)
    (pref : ML.MergeableSummary.Stream α → Pref) : Prop where
  build_valid : ∀ xs : ML.MergeableSummary.Stream α, valid xs (build xs)
  merge_valid : ML.MergeableSummary.MergeClosed valid merge
  readout_valid : ∀ xs s, valid xs s → readout s = pref xs

namespace RelationalMergeablePreferenceShape

variable {α State Pref : Type*}
variable {build : ML.MergeableSummary.Stream α → State}
variable {merge : State → State → State}
variable {valid : ML.MergeableSummary.Stream α → State → Prop}
variable {readout : State → Pref}
variable {pref : ML.MergeableSummary.Stream α → Pref}

/-- Relational mergeability is hierarchical over arbitrary binary merge trees. -/
theorem hierarchical
    (h : RelationalMergeablePreferenceShape build merge valid readout pref) :
    ML.MergeableSummary.HierarchicalMergeable build valid merge := by
  exact ML.MergeableSummary.hierarchical_of_full
    (V := { build := build, valid := valid, build_valid := h.build_valid })
    (merge := merge)
    h.merge_valid

/-- Merging summary states up a tree and reading out at the root recovers the
preference of the represented concatenated stream. -/
theorem readout_of_mergeTree
    (h : RelationalMergeablePreferenceShape build merge valid readout pref)
    (t : ML.MergeableSummary.MergeTree α) :
    readout (ML.MergeableSummary.MergeTree.eval build merge t) =
      pref (ML.MergeableSummary.MergeTree.data t) := by
  exact h.readout_valid
    (ML.MergeableSummary.MergeTree.data t)
    (ML.MergeableSummary.MergeTree.eval build merge t)
    ((hierarchical h) t)

/-- If the validity relation is backed by a canonical state equality law, the
relational Agarwal-style shape collapses to the exact C-TreePO mergeable
preference shape already used elsewhere in the formalization. -/
theorem to_mergeablePreferenceShape_of_canonical
    (h : RelationalMergeablePreferenceShape build merge valid readout pref)
    (state : ML.MergeableSummary.Stream α → State)
    (hbuild : ∀ xs : ML.MergeableSummary.Stream α, build xs = state xs)
    (hmerge : ∀ xs ys : ML.MergeableSummary.Stream α,
      merge (state xs) (state ys) = state (xs ++ ys)) :
    MergeablePreferenceShape state build merge pref := by
  refine ⟨?_, ?_⟩
  · refine ⟨hbuild, ?_⟩
    intro xs ys
    simpa using hmerge xs ys
  · refine ⟨readout, ?_⟩
    intro xs
    have hr : readout (build xs) = pref xs :=
      h.readout_valid xs (build xs) (h.build_valid xs)
    rw [hbuild xs] at hr
    exact hr.symm

end RelationalMergeablePreferenceShape

/-!
## Epsilon Relational Mergeable Preference Shape

For approximate or learned settings, the same state-level story is stated in
the task metric: validity at the root implies readout error at most `ε`.
-/

/-- Epsilon version of relational mergeable preference nesting.  Summary states
merge through a validity relation, and every valid state reads out within the
target task error `ε`. -/
structure EpsilonRelationalMergeablePreferenceShape
    {α State Pref : Type*} [PseudoMetricSpace Pref]
    (build : ML.MergeableSummary.Stream α → State)
    (merge : State → State → State)
    (valid : ML.MergeableSummary.Stream α → State → Prop)
    (readout : State → Pref)
    (pref : ML.MergeableSummary.Stream α → Pref)
    (ε : ℝ) : Prop where
  build_valid : ∀ xs : ML.MergeableSummary.Stream α, valid xs (build xs)
  merge_valid : ML.MergeableSummary.MergeClosed valid merge
  readout_valid : ∀ xs s, valid xs s → dist (readout s) (pref xs) ≤ ε

namespace EpsilonRelationalMergeablePreferenceShape

variable {α State Pref : Type*} [PseudoMetricSpace Pref]
variable {build : ML.MergeableSummary.Stream α → State}
variable {merge : State → State → State}
variable {valid : ML.MergeableSummary.Stream α → State → Prop}
variable {readout : State → Pref}
variable {pref : ML.MergeableSummary.Stream α → Pref}
variable {ε : ℝ}

/-- Epsilon relational mergeability is hierarchical over arbitrary binary merge
trees because the validity invariant is the same as in the exact relational
shape. -/
theorem hierarchical
    (h : EpsilonRelationalMergeablePreferenceShape build merge valid readout pref ε) :
    ML.MergeableSummary.HierarchicalMergeable build valid merge := by
  exact ML.MergeableSummary.hierarchical_of_full
    (V := { build := build, valid := valid, build_valid := h.build_valid })
    (merge := merge)
    h.merge_valid

/-- Merging summary states up a tree and reading out at the root gives an
`ε`-accurate task score. -/
theorem readout_error_of_mergeTree
    (h : EpsilonRelationalMergeablePreferenceShape build merge valid readout pref ε)
    (t : ML.MergeableSummary.MergeTree α) :
    dist (readout (ML.MergeableSummary.MergeTree.eval build merge t))
      (pref (ML.MergeableSummary.MergeTree.data t)) ≤ ε := by
  exact h.readout_valid
    (ML.MergeableSummary.MergeTree.data t)
    (ML.MergeableSummary.MergeTree.eval build merge t)
    ((hierarchical h) t)

end EpsilonRelationalMergeablePreferenceShape

/-!
## Randomized Relational Mergeable Preference Shape

Agarwal et al.'s randomized definition asks for validity with high probability
after an arbitrary merge tree.  The C-TreePO nesting theorem therefore lives at
the event level: on the event that the merged root state is valid for the union
stream, the root readout equals the target preference.
-/

/-- Probability that a randomized merge-tree root reads out the target
preference.  The randomness parameter `ω` represents all random choices used by
the tree evaluation. -/
def RandomizedTreeReadoutSuccess
    {Ω α State Pref : Type*} [MeasurableSpace Ω]
    (μ : Measure Ω)
    (build : Ω → ML.MergeableSummary.Stream α → State)
    (merge : State → State → State)
    (readout : State → Pref)
    (pref : ML.MergeableSummary.Stream α → Pref)
    (t : ML.MergeableSummary.MergeTree α) (p : ℝ) : Prop :=
  p ≤ (μ {ω : Ω |
    readout (ML.MergeableSummary.MergeTree.eval (build ω) merge t) =
      pref (ML.MergeableSummary.MergeTree.data t)}).toReal

/-- Probability that a randomized merge-tree root reads out within the target
task error `ε`. -/
def RandomizedTreeEpsilonReadoutSuccess
    {Ω α State Pref : Type*} [MeasurableSpace Ω] [PseudoMetricSpace Pref]
    (μ : Measure Ω)
    (build : Ω → ML.MergeableSummary.Stream α → State)
    (merge : State → State → State)
    (readout : State → Pref)
    (pref : ML.MergeableSummary.Stream α → Pref)
    (t : ML.MergeableSummary.MergeTree α) (ε p : ℝ) : Prop :=
  p ≤ (μ {ω : Ω |
    dist (readout (ML.MergeableSummary.MergeTree.eval (build ω) merge t))
      (pref (ML.MergeableSummary.MergeTree.data t)) ≤ ε}).toReal

/-- Valid-root probability implies root-readout probability whenever every
valid state reads out the target preference. -/
theorem randomizedTreeReadoutSuccess_of_randomizedTreeSuccess
    {Ω α State Pref : Type*} [MeasurableSpace Ω]
    {μ : Measure Ω} [IsProbabilityMeasure μ]
    {build : Ω → ML.MergeableSummary.Stream α → State}
    {merge : State → State → State}
    {valid : ML.MergeableSummary.Stream α → State → Prop}
    {readout : State → Pref}
    {pref : ML.MergeableSummary.Stream α → Pref}
    {t : ML.MergeableSummary.MergeTree α} {p : ℝ}
    (hreadout : ∀ xs s, valid xs s → readout s = pref xs)
    (hsuccess :
      ML.MergeableSummary.Agarwal2013Full.RandomizedTreeSuccess
        μ build valid merge t p) :
    RandomizedTreeReadoutSuccess μ build merge readout pref t p := by
  unfold RandomizedTreeReadoutSuccess
  unfold ML.MergeableSummary.Agarwal2013Full.RandomizedTreeSuccess at hsuccess
  change
    p ≤ μ.real {ω : Ω |
      readout (ML.MergeableSummary.MergeTree.eval (build ω) merge t) =
        pref (ML.MergeableSummary.MergeTree.data t)}
  change
    p ≤ μ.real {ω : Ω |
      valid (ML.MergeableSummary.MergeTree.data t)
        (ML.MergeableSummary.MergeTree.eval (build ω) merge t)} at hsuccess
  refine hsuccess.trans (measureReal_mono ?_)
  intro ω hω
  exact hreadout
    (ML.MergeableSummary.MergeTree.data t)
    (ML.MergeableSummary.MergeTree.eval (build ω) merge t)
    hω

/-- Valid-root probability implies epsilon-readout probability whenever every
valid state reads out within the target task error `ε`. -/
theorem randomizedTreeEpsilonReadoutSuccess_of_randomizedTreeSuccess
    {Ω α State Pref : Type*} [MeasurableSpace Ω] [PseudoMetricSpace Pref]
    {μ : Measure Ω} [IsProbabilityMeasure μ]
    {build : Ω → ML.MergeableSummary.Stream α → State}
    {merge : State → State → State}
    {valid : ML.MergeableSummary.Stream α → State → Prop}
    {readout : State → Pref}
    {pref : ML.MergeableSummary.Stream α → Pref}
    {t : ML.MergeableSummary.MergeTree α} {ε p : ℝ}
    (hreadout : ∀ xs s, valid xs s → dist (readout s) (pref xs) ≤ ε)
    (hsuccess :
      ML.MergeableSummary.Agarwal2013Full.RandomizedTreeSuccess
        μ build valid merge t p) :
    RandomizedTreeEpsilonReadoutSuccess μ build merge readout pref t ε p := by
  unfold RandomizedTreeEpsilonReadoutSuccess
  unfold ML.MergeableSummary.Agarwal2013Full.RandomizedTreeSuccess at hsuccess
  change
    p ≤ μ.real {ω : Ω |
      dist (readout (ML.MergeableSummary.MergeTree.eval (build ω) merge t))
        (pref (ML.MergeableSummary.MergeTree.data t)) ≤ ε}
  change
    p ≤ μ.real {ω : Ω |
      valid (ML.MergeableSummary.MergeTree.data t)
        (ML.MergeableSummary.MergeTree.eval (build ω) merge t)} at hsuccess
  refine hsuccess.trans (measureReal_mono ?_)
  intro ω hω
  exact hreadout
    (ML.MergeableSummary.MergeTree.data t)
    (ML.MergeableSummary.MergeTree.eval (build ω) merge t)
    hω

/-- Randomized Agarwal-style state-level nesting: the merged root is valid with
probability at least `p`, and any valid root state reads out the C-TreePO
preference. -/
structure RandomizedRelationalMergeablePreferenceShape
    {Ω α State Pref : Type*} [MeasurableSpace Ω]
    (μ : Measure Ω) [IsProbabilityMeasure μ]
    (build : Ω → ML.MergeableSummary.Stream α → State)
    (merge : State → State → State)
    (valid : ML.MergeableSummary.Stream α → State → Prop)
    (readout : State → Pref)
    (pref : ML.MergeableSummary.Stream α → Pref)
    (p : ℝ) : Prop where
  tree_success :
    ∀ t : ML.MergeableSummary.MergeTree α,
      ML.MergeableSummary.Agarwal2013Full.RandomizedTreeSuccess
        μ build valid merge t p
  readout_valid : ∀ xs s, valid xs s → readout s = pref xs

namespace RandomizedRelationalMergeablePreferenceShape

variable {Ω α State Pref : Type*} [MeasurableSpace Ω]
variable {μ : Measure Ω} [IsProbabilityMeasure μ]
variable {build : Ω → ML.MergeableSummary.Stream α → State}
variable {merge : State → State → State}
variable {valid : ML.MergeableSummary.Stream α → State → Prop}
variable {readout : State → Pref}
variable {pref : ML.MergeableSummary.Stream α → Pref}
variable {p : ℝ}

/-- Paper-probability nesting theorem: randomized state validity at the root
transports to randomized C-TreePO readout correctness at the same probability
level. -/
theorem readout_success_of_mergeTree
    (h :
      RandomizedRelationalMergeablePreferenceShape
        μ build merge valid readout pref p)
    (t : ML.MergeableSummary.MergeTree α) :
    RandomizedTreeReadoutSuccess μ build merge readout pref t p :=
  randomizedTreeReadoutSuccess_of_randomizedTreeSuccess
    h.readout_valid (h.tree_success t)

end RandomizedRelationalMergeablePreferenceShape

/-- Randomized epsilon Agarwal-style state-level nesting: root validity holds
with probability at least `p`, and valid roots read out within task error
`ε`. -/
structure RandomizedEpsilonRelationalMergeablePreferenceShape
    {Ω α State Pref : Type*} [MeasurableSpace Ω] [PseudoMetricSpace Pref]
    (μ : Measure Ω) [IsProbabilityMeasure μ]
    (build : Ω → ML.MergeableSummary.Stream α → State)
    (merge : State → State → State)
    (valid : ML.MergeableSummary.Stream α → State → Prop)
    (readout : State → Pref)
    (pref : ML.MergeableSummary.Stream α → Pref)
    (ε p : ℝ) : Prop where
  tree_success :
    ∀ t : ML.MergeableSummary.MergeTree α,
      ML.MergeableSummary.Agarwal2013Full.RandomizedTreeSuccess
        μ build valid merge t p
  readout_valid : ∀ xs s, valid xs s → dist (readout s) (pref xs) ≤ ε

namespace RandomizedEpsilonRelationalMergeablePreferenceShape

variable {Ω α State Pref : Type*} [MeasurableSpace Ω] [PseudoMetricSpace Pref]
variable {μ : Measure Ω} [IsProbabilityMeasure μ]
variable {build : Ω → ML.MergeableSummary.Stream α → State}
variable {merge : State → State → State}
variable {valid : ML.MergeableSummary.Stream α → State → Prop}
variable {readout : State → Pref}
variable {pref : ML.MergeableSummary.Stream α → Pref}
variable {ε p : ℝ}

/-- Paper-probability epsilon nesting theorem: randomized state validity at the
root transports to randomized C-TreePO readout accuracy at the same probability
level. -/
theorem readout_success_of_mergeTree
    (h :
      RandomizedEpsilonRelationalMergeablePreferenceShape
        μ build merge valid readout pref ε p)
    (t : ML.MergeableSummary.MergeTree α) :
    RandomizedTreeEpsilonReadoutSuccess μ build merge readout pref t ε p :=
  randomizedTreeEpsilonReadoutSuccess_of_randomizedTreeSuccess
    h.readout_valid (h.tree_success t)

end RandomizedEpsilonRelationalMergeablePreferenceShape

/-- Scalar child-query mergeability is a strictly additional assumption.  The
Agarwal/C-TreePO nesting theorem only requires state merge and root readout. -/
def ScalarQueryMergeLaw
    {α Pref : Type*}
    (pref : ML.MergeableSummary.Stream α → Pref)
    (mergePref : Pref → Pref → Pref) : Prop :=
  ∀ xs ys : ML.MergeableSummary.Stream α,
    mergePref (pref xs) (pref ys) = pref (xs ++ ys)

/-- Exact composability via additive state merge.  This is the standard
additive-statistic case used by sums, histograms, and linear sketches. -/
structure AdditiveComposableState
    {Doc State : Type*} [Monoid Doc] [Add State]
    (state : Doc → State)
    (encode : Doc → State) : Prop where
  encode_eq_state : ∀ x, encode x = state x
  state_mul_eq_add : ∀ x y, state (x * y) = state x + state y

namespace AdditiveComposableState

variable {Doc State : Type*} [Monoid Doc] [Add State]
variable {state : Doc → State}
variable {encode : Doc → State}

/-- An additive state is an exact composable state with merge `+`. -/
theorem toExactComposableState
    (h : AdditiveComposableState state encode) :
    ExactComposableState state encode (fun a b => a + b) where
  encode_eq_state := h.encode_eq_state
  merge_eq_state := by
    intro x y
    exact (h.state_mul_eq_add x y).symm

end AdditiveComposableState

/-- The readout itself is additive on state values.  This is the extra
restriction that turns a mergeable-state preference into an additively
separable utility. -/
def AdditiveStateReadout
    {State : Type*} [Add State]
    (readout : State → ℝ) : Prop :=
  ∀ a b, readout (a + b) = readout a + readout b

/-- A document utility is additively separable through a state when it factors
through that state and the state readout is additive. -/
def AdditivelySeparableThroughState
    {Doc State : Type*} [Add State]
    (state : Doc → State) (utility : Doc → ℝ) : Prop :=
  ∃ readout : State → ℝ,
    (∀ x, utility x = readout (state x)) ∧ AdditiveStateReadout readout

/-- Additive state plus additive readout gives the usual concatenation-additive
utility equation. -/
theorem additive_state_readout_yields_concat_additive
    {Doc State : Type*} [Monoid Doc] [Add State]
    {state : Doc → State} {encode : Doc → State}
    (hState : AdditiveComposableState state encode)
    {readout : State → ℝ}
    (hReadout : AdditiveStateReadout readout)
    (x y : Doc) :
    readout (state (x * y)) = readout (state x) + readout (state y) := by
  rw [hState.state_mul_eq_add x y]
  exact hReadout (state x) (state y)

/-- Additively separable utilities are state-factored preferences.  The converse
is false in general: arbitrary nonlinear readouts of a mergeable state are still
captured by the mergeable-state route. -/
theorem additivelySeparableThroughState_factorsThroughState
    {Doc State : Type*} [Add State]
    {state : Doc → State} {utility : Doc → ℝ}
    (h : AdditivelySeparableThroughState state utility) :
    PreferenceFactorsThroughState state utility := by
  rcases h with ⟨readout, hreadout, _hAdd⟩
  exact ⟨readout, hreadout⟩

/-!
## Positive Scope Theorems
-/

variable {Strings : Type*} [Monoid Strings]
variable {Y : Type*} [BoundedMetricSpace Y]

/-- C-TreePO supports any objective that factors through an oracle-identified
state, provided the text reduction is exact theorem-backed for the oracle.

This is the Lean version of the scope statement: local laws preserve the
task-relevant fiber, and any downstream state-factored preference/loss is
therefore preserved. -/
theorem ctreepo_supports_state_factored_preference
    {State Action : Type*} [Encodable State]
    (fstar : Strings → Y)
    (state : Strings → State)
    (objective : State → Action → ℝ)
    (gen : Strings → PMF Action)
    (g : Summarizer Strings) (x : Strings) (R : ℕ) (T : BinTree Strings)
    (hp : S T = x)
    (hExact : ExactTheoremBacked g T fstar)
    (hR : R ≥ 1)
    (hRecover : OracleRecoversFeature fstar state)
    (hGen : OracleIndexedGenGeneric gen (encodedOracle (Strings := Strings) state)) :
    ExpectedLossGeneric
      (featureIndexedObjective (Strings := Strings) state objective) (PMF.pure x) gen =
    ExpectedLossGeneric
      (featureIndexedObjective (Strings := Strings) state objective) (ZR g x R T) gen :=
  featureIndexedObjective_eq_via_ZR_of_exactTheoremBacked_and_oracleRecoversFeature
    (fstar := fstar)
    (feature := state)
    (objective := objective)
    (gen := gen)
    (g := g) (x := x) (R := R) (T := T)
    hp hExact hR hRecover hGen

/-- If a state has an exact leaf encoder and exact merge, every downstream
utility on that state is preserved by the tree.  The utility can be
nonseparable over leaves. -/
theorem exact_mergeable_state_supports_any_downstream_utility
    {State β : Type*}
    (state : Strings → State)
    (encode : Strings → State)
    (merge : State → State → State)
    (hState : ExactComposableState state encode merge)
    (u : State → β)
    (T : BinTree Strings) :
    u (mergeFold encode merge T) = u (state (S T)) :=
  mergeableStateUtility_exact_on_tree
    (encode := encode)
    (merge := merge)
    (feature := state)
    hState.encode_eq_state
    hState.merge_eq_state
    u
    T

/-!
## Curated Example and Counterexample Aliases
-/

/-- Supported nonseparable complementarity: threshold-AND utility over exact
left/right counts is preserved because the state carries both counts. -/
abbrev supported_nonseparable_complementarity :=
  @complementarityThresholdUtility_exact_on_tree

/-- Supported boundary interaction: topic unigram plus boundary-bigram state
preserves boundary-sensitive scores. -/
abbrev supported_boundary_interaction :=
  @topicOracleFromSketch_exact_on_tree

/-- Bag-of-words / histogram state supports any downstream histogram utility;
the readout can be nonlinear even though the state merge is additive. -/
abbrev supported_histogram_state_any_utility :=
  @histogramUtility_exact_on_tree

/-- Ordinary bag-of-words LDA likelihood is preserved by the count-sketch tree. -/
abbrev supported_lda_likelihood_histogram_utility :=
  @ldaDocumentLikelihood_exact_on_tree

/-- Generic classical state-level bridge: merge sketch states first, then query
or read out at the root. -/
abbrev classical_state_level_mergeable_preference_shape :=
  @sketch_state_level_reduction_to_classical_mergeable

/-- Agarwal-style state-level nesting: merge summary states, preserve a
validity relation, then read out the preference at the root. -/
abbrev relational_mergeable_preference_shape :=
  @FormalProofs.OPT.RelationalMergeablePreferenceShape

/-- Relational state-level summaries are hierarchical over merge trees. -/
abbrev relational_mergeable_preference_hierarchical :=
  @FormalProofs.OPT.RelationalMergeablePreferenceShape.hierarchical

/-- Relational state-level summaries recover the root preference after merging
states up the tree. -/
abbrev relational_mergeable_preference_readout_of_tree :=
  @FormalProofs.OPT.RelationalMergeablePreferenceShape.readout_of_mergeTree

/-- Epsilon relational state-level nesting: root readout is within the task
metric threshold `ε` whenever the merged root state is valid. -/
abbrev epsilon_relational_mergeable_preference_shape :=
  @FormalProofs.OPT.EpsilonRelationalMergeablePreferenceShape

/-- Epsilon relational state-level summaries are hierarchical over merge trees. -/
abbrev epsilon_relational_mergeable_preference_hierarchical :=
  @FormalProofs.OPT.EpsilonRelationalMergeablePreferenceShape.hierarchical

/-- Epsilon relational summaries give a root task-metric error bound after
state merging. -/
abbrev epsilon_relational_mergeable_preference_readout_error_of_tree :=
  @FormalProofs.OPT.EpsilonRelationalMergeablePreferenceShape.readout_error_of_mergeTree

/-- Canonical/equality-valued relational summaries collapse to the existing
exact mergeable preference shape. -/
abbrev relational_mergeable_preference_to_exact_shape_of_canonical :=
  @FormalProofs.OPT.RelationalMergeablePreferenceShape.to_mergeablePreferenceShape_of_canonical

/-- Randomized root-readout correctness event for a merge tree. -/
abbrev randomized_tree_readout_success :=
  @FormalProofs.OPT.RandomizedTreeReadoutSuccess

/-- Valid randomized root states read out correctly with the same probability
lower bound. -/
abbrev randomized_tree_readout_success_of_randomized_tree_success :=
  @FormalProofs.OPT.randomizedTreeReadoutSuccess_of_randomizedTreeSuccess

/-- Randomized root epsilon-readout accuracy event for a merge tree. -/
abbrev randomized_tree_epsilon_readout_success :=
  @FormalProofs.OPT.RandomizedTreeEpsilonReadoutSuccess

/-- Randomized root validity transfers to root epsilon-readout accuracy with
the same probability lower bound. -/
abbrev randomized_tree_epsilon_readout_success_of_randomized_tree_success :=
  @FormalProofs.OPT.randomizedTreeEpsilonReadoutSuccess_of_randomizedTreeSuccess

/-- Randomized Agarwal-style state-level nesting: root validity with high
probability plus deterministic valid-state readout. -/
abbrev randomized_relational_mergeable_preference_shape :=
  @FormalProofs.OPT.RandomizedRelationalMergeablePreferenceShape

/-- Randomized relational summaries recover the root preference in probability
after merging states up the tree. -/
abbrev randomized_relational_mergeable_preference_readout_success_of_tree :=
  @FormalProofs.OPT.RandomizedRelationalMergeablePreferenceShape.readout_success_of_mergeTree

/-- Randomized epsilon Agarwal-style state-level nesting: root validity with
high probability plus deterministic epsilon valid-state readout. -/
abbrev randomized_epsilon_relational_mergeable_preference_shape :=
  @FormalProofs.OPT.RandomizedEpsilonRelationalMergeablePreferenceShape

/-- Randomized epsilon relational summaries recover root task accuracy in
probability after merging states up the tree. -/
abbrev randomized_epsilon_relational_mergeable_preference_readout_success_of_tree :=
  @FormalProofs.OPT.RandomizedEpsilonRelationalMergeablePreferenceShape.readout_success_of_mergeTree

/-- Explicit marker for the stronger scalar child-query merge law, which is not
required by Agarwal-style state-level nesting. -/
abbrev scalar_query_merge_law :=
  @FormalProofs.OPT.ScalarQueryMergeLaw

/-- Additive linear sketches are fully mergeable in the classical sketch sense. -/
abbrev additive_linear_sketch_preference_shape :=
  @ctreepo_agarwal2013_linearSketch_fullMergeable

/-- Count-Min-style additive counter tables are state-level mergeable. -/
abbrev count_min_state_level_preference_shape :=
  @ctreepo_agarwal2013_countMin_state_level_mergeable

/-- HLL-style register states are state-level mergeable under max-register
merge. -/
abbrev hll_state_level_preference_shape :=
  @ctreepo_flajolet2007_hll_state_level_mergeable

/-!
## Scalar-Oracle Boundary

The scalar-oracle merge layer is useful for explaining additive separability:
if one insists on merging only final oracle/preference values, then the merge
operator must be well-defined on those values.  Nonlinear readouts such as
thresholds can fail there because the scalar answer has already forgotten the
interaction variables.  The state-level route above is strictly more flexible:
merge the sufficient state first, then apply the nonlinear readout.
-/

/-- Additive scalar preferences are mergeable when the final oracle values
themselves compose by addition. -/
abbrev additive_scalar_preference_is_mergeable :=
  @mergeablePreference_of_additiveSeparable

/-- Generic scalar-oracle obstruction: if equal child oracle values can lead to
different parent oracle values, no global scalar merge is well-defined. -/
abbrev scalar_oracle_concat_witness_not_expressible :=
  @not_ctreepoExpressible_of_concat_witness

/-- Threshold-AND is not scalar-oracle mergeable after collapsing each side to a
Boolean threshold value.  The supported state-level version keeps both counts
until the root readout. -/
abbrev scalar_threshold_and_not_expressible :=
  @ThresholdAND.not_ctreepoExpressible_threshold_and

/-- Boundary bigrams are not scalar-oracle mergeable if the leaf scalar omits
boundary tokens.  The supported state-level version keeps boundary state. -/
abbrev scalar_boundary_bigram_not_expressible :=
  CrossBoundaryBigram.not_ctreepoExpressible_cross_boundary_bigram

/-- Wrong state: scalar child distinct counts omit overlap information, so no
scalar merge can recover global distinct count for all inputs. -/
abbrev insufficient_scalar_distinct_count_state :=
  @scalarDistinctCount_not_child_cardinality_mergeable

/-- Wrong state: count-only Markov summaries omit endpoint information needed
for arbitrary tree topology claims. -/
abbrev insufficient_markov_count_only_state :=
  @markov_countOnly_not_exact_on_all_trees

/-- Wrong operator: C2/on-range idempotence is not derivable from the other
local requirements. -/
abbrev c2_idempotence_not_derivable :=
  @thm10_1_L3_not_derivable

/-- Public-shape C2 independence counterexample. -/
abbrev c2_independence_counterexample :=
  @ex_c2_independent_formalized

/-- Wrong preference target: if a downstream readout separates two inputs in
one theorem-state fiber, it cannot factor through that state. -/
abbrev preference_not_factored_through_state :=
  @not_readoutFactorsThroughFeature_of_distinguished_feature_fibers

/-- Everything matters: if the oracle is injective, oracle-sufficient
compression cannot gain cardinality over the raw input. -/
abbrev no_compression_when_everything_matters :=
  @OracleSufficientCompression.no_compression_gain_of_injective_oracle

end FormalProofs.OPT
