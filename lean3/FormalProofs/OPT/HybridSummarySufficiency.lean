import FormalProofs.OPT.LikelihoodOnStateSufficiency

/-!
# FormalProofs/OPT/HybridSummarySufficiency.lean

Deterministic theorem surface for hybrid summary statistics.

Hybrid summary-statistic methods augment a hand-built/domain summary with a
learned neural summary. This file formalizes the representation-level facts
that are useful for the C-TreePO Lean lane:

* the product summary refines both components;
* if either component is sufficient for a target or likelihood family, the
  hybrid product is also sufficient;
* hybrid sufficiency is exactly sufficiency inside each base-summary fiber;
* a learned neural summary supplies "extra" deterministic information when it
  separates target/likelihood/probe distinctions left unresolved by the base
  summary;
* if a target or likelihood family has a readout from the hybrid product, the
  hybrid product is sufficient;
* approximate hybrid readouts give approximate within-base sufficiency;
* likelihood-on-hybrid-state is a direct instance of likelihood-on-state.

No mutual-information objective, estimator guarantee, or posterior consistency
claim is made here.
-/

set_option linter.mathlibStandardSet false

open scoped Classical

set_option maxHeartbeats 0
set_option maxRecDepth 4000
set_option relaxedAutoImplicit false
set_option autoImplicit false

noncomputable section

namespace FormalProofs.OPT

variable {X Base Neural Target Θ Probe Y : Type*}

/-- Product summary combining a hand-built/base statistic with a neural summary. -/
def HybridSummary
    (base : X → Base)
    (neural : X → Neural)
    (x : X) :
    Base × Neural :=
  (base x, neural x)

/-- The hybrid product refines the base statistic. -/
theorem hybridSummary_sufficient_for_base
    (base : X → Base)
    (neural : X → Neural) :
    TargetSufficientRepresentation (HybridSummary base neural) base := by
  intro x y hxy
  exact congrArg Prod.fst hxy

/-- The hybrid product refines the neural statistic. -/
theorem hybridSummary_sufficient_for_neural
    (base : X → Base)
    (neural : X → Neural) :
    TargetSufficientRepresentation (HybridSummary base neural) neural := by
  intro x y hxy
  exact congrArg Prod.snd hxy

/-! ## Within-base deterministic complementarity -/

/-- Within-base target sufficiency: once the hand-built/base summaries agree,
neural-summary agreement is enough to force target agreement. This is the
deterministic fiber version of "the neural summary captures what remains beyond
the existing summary." -/
def WithinBaseTargetSufficient
    (base : X → Base)
    (neural : X → Neural)
    (target : X → Target) : Prop :=
  ∀ ⦃x y : X⦄, base x = base y → neural x = neural y → target x = target y

/-- Inside a base-summary fiber, the neural summary separates target
distinctions. -/
def NeuralSeparatesTargetWithinBase
    (base : X → Base)
    (neural : X → Neural)
    (target : X → Target) : Prop :=
  ∀ ⦃x y : X⦄, base x = base y → target x ≠ target y → neural x ≠ neural y

/-- Product-summary target sufficiency is exactly within-base target
sufficiency. -/
theorem hybridTargetSufficient_iff_withinBaseTargetSufficient
    {base : X → Base}
    {neural : X → Neural}
    {target : X → Target} :
    TargetSufficientRepresentation (HybridSummary base neural) target ↔
      WithinBaseTargetSufficient base neural target := by
  constructor
  · intro hSuff x y hb hn
    exact hSuff (Prod.ext hb hn)
  · intro hWithin x y hxy
    exact hWithin (congrArg Prod.fst hxy) (congrArg Prod.snd hxy)

/-- Within-base target sufficiency is equivalent to separating target
distinctions inside each base-summary fiber. -/
theorem withinBaseTargetSufficient_iff_neuralSeparatesTargetWithinBase
    {base : X → Base}
    {neural : X → Neural}
    {target : X → Target} :
    WithinBaseTargetSufficient base neural target ↔
      NeuralSeparatesTargetWithinBase base neural target := by
  constructor
  · intro hWithin x y hb hTargetDistinct hNeural
    exact hTargetDistinct (hWithin hb hNeural)
  · intro hSeparate x y hb hNeural
    by_contra hTargetDistinct
    exact hSeparate hb hTargetDistinct hNeural

/-- A hybrid target-sufficient summary forces the neural component to separate
all target distinctions that remain inside base-summary fibers. -/
theorem hybridTargetSufficient_neuralSeparatesTargetWithinBase
    {base : X → Base}
    {neural : X → Neural}
    {target : X → Target}
    (hHybrid : TargetSufficientRepresentation (HybridSummary base neural) target) :
    NeuralSeparatesTargetWithinBase base neural target :=
  (withinBaseTargetSufficient_iff_neuralSeparatesTargetWithinBase).mp
    ((hybridTargetSufficient_iff_withinBaseTargetSufficient).mp hHybrid)

/-- A hybrid target-sufficient summary cannot collapse a target-distinct pair in
both its base and neural components. -/
theorem hybridTargetSufficient_no_base_neural_collision_of_distinguished_target
    {base : X → Base}
    {neural : X → Neural}
    {target : X → Target}
    (hHybrid : TargetSufficientRepresentation (HybridSummary base neural) target)
    {x y : X}
    (hTargetDistinct : target x ≠ target y) :
    ¬ (base x = base y ∧ neural x = neural y) := by
  intro hCollision
  exact hTargetDistinct (hHybrid (Prod.ext hCollision.1 hCollision.2))

/-- A target readout from the hybrid product. -/
def HybridTargetReadoutRealizes
    (base : X → Base)
    (neural : X → Neural)
    (target : X → Target)
    (readout : Base → Neural → Target) : Prop :=
  ∀ x, readout (base x) (neural x) = target x

/-- If a target can be read from the hybrid product, the hybrid summary is
target-sufficient. -/
theorem hybridTargetReadout_implies_targetSufficient
    {base : X → Base}
    {neural : X → Neural}
    {target : X → Target}
    {readout : Base → Neural → Target}
    (hReadout : HybridTargetReadoutRealizes base neural target readout) :
    TargetSufficientRepresentation (HybridSummary base neural) target := by
  intro x y hxy
  have hb : base x = base y := congrArg Prod.fst hxy
  have hn : neural x = neural y := congrArg Prod.snd hxy
  calc
    target x = readout (base x) (neural x) := (hReadout x).symm
    _ = readout (base y) (neural y) := by rw [hb, hn]
    _ = target y := hReadout y

/-- If the base statistic is sufficient, the hybrid product remains sufficient.
Adding a neural summary cannot create new collisions beyond base collisions. -/
theorem hybridTargetSufficient_of_baseSufficient
    {base : X → Base}
    {neural : X → Neural}
    {target : X → Target}
    (hBase : TargetSufficientRepresentation base target) :
    TargetSufficientRepresentation (HybridSummary base neural) target := by
  intro x y hxy
  exact hBase (congrArg Prod.fst hxy)

/-- If the neural statistic is sufficient, the hybrid product remains sufficient. -/
theorem hybridTargetSufficient_of_neuralSufficient
    {base : X → Base}
    {neural : X → Neural}
    {target : X → Target}
    (hNeural : TargetSufficientRepresentation neural target) :
    TargetSufficientRepresentation (HybridSummary base neural) target := by
  intro x y hxy
  exact hNeural (congrArg Prod.snd hxy)

/-! ## Likelihood-family hybrid sufficiency -/

/-- Within-base likelihood sufficiency: inside each base-summary fiber, neural
agreement preserves every likelihood value. -/
def WithinBaseLikelihoodSufficient
    (base : X → Base)
    (neural : X → Neural)
    (likelihood : Θ → X → Y) : Prop :=
  ∀ ⦃x y : X⦄,
    base x = base y → neural x = neural y →
      ∀ θ : Θ, likelihood θ x = likelihood θ y

/-- Inside a base-summary fiber, the neural summary separates any pair that some
parameter likelihood distinguishes. -/
def NeuralSeparatesLikelihoodWithinBase
    (base : X → Base)
    (neural : X → Neural)
    (likelihood : Θ → X → Y) : Prop :=
  ∀ ⦃x y : X⦄,
    base x = base y →
      (∃ θ : Θ, likelihood θ x ≠ likelihood θ y) →
        neural x ≠ neural y

/-- Product-summary likelihood sufficiency is exactly within-base likelihood
sufficiency. -/
theorem hybridLikelihoodSufficient_iff_withinBaseLikelihoodSufficient
    {base : X → Base}
    {neural : X → Neural}
    {likelihood : Θ → X → Y} :
    LikelihoodFamilySufficient (HybridSummary base neural) likelihood ↔
      WithinBaseLikelihoodSufficient base neural likelihood := by
  constructor
  · intro hSuff x y hb hn θ
    exact hSuff (Prod.ext hb hn) θ
  · intro hWithin x y hxy θ
    exact hWithin (congrArg Prod.fst hxy) (congrArg Prod.snd hxy) θ

/-- Within-base likelihood sufficiency is equivalent to separating all
likelihood distinctions inside base-summary fibers. -/
theorem withinBaseLikelihoodSufficient_iff_neuralSeparatesLikelihoodWithinBase
    {base : X → Base}
    {neural : X → Neural}
    {likelihood : Θ → X → Y} :
    WithinBaseLikelihoodSufficient base neural likelihood ↔
      NeuralSeparatesLikelihoodWithinBase base neural likelihood := by
  constructor
  · intro hWithin x y hb hLikelihoodDistinct hNeural
    rcases hLikelihoodDistinct with ⟨θ, hθ⟩
    exact hθ (hWithin hb hNeural θ)
  · intro hSeparate x y hb hNeural θ
    by_contra hLikelihoodDistinct
    exact hSeparate hb ⟨θ, hLikelihoodDistinct⟩ hNeural

/-- A hybrid likelihood-sufficient summary forces the neural component to
separate all likelihood distinctions that remain inside base-summary fibers. -/
theorem hybridLikelihoodSufficient_neuralSeparatesLikelihoodWithinBase
    {base : X → Base}
    {neural : X → Neural}
    {likelihood : Θ → X → Y}
    (hHybrid : LikelihoodFamilySufficient (HybridSummary base neural) likelihood) :
    NeuralSeparatesLikelihoodWithinBase base neural likelihood :=
  (withinBaseLikelihoodSufficient_iff_neuralSeparatesLikelihoodWithinBase).mp
    ((hybridLikelihoodSufficient_iff_withinBaseLikelihoodSufficient).mp hHybrid)

/-- A hybrid likelihood-sufficient summary cannot collapse a pair in both
components if some likelihood parameter distinguishes that pair. -/
theorem hybridLikelihoodSufficient_no_base_neural_collision_of_distinguished_likelihood
    {base : X → Base}
    {neural : X → Neural}
    {likelihood : Θ → X → Y}
    (hHybrid : LikelihoodFamilySufficient (HybridSummary base neural) likelihood)
    {x y : X}
    (hLikelihoodDistinct : ∃ θ : Θ, likelihood θ x ≠ likelihood θ y) :
    ¬ (base x = base y ∧ neural x = neural y) := by
  intro hCollision
  rcases hLikelihoodDistinct with ⟨θ, hθ⟩
  exact hθ (hHybrid (Prod.ext hCollision.1 hCollision.2) θ)

/-- A likelihood-family readout from the hybrid product. -/
def HybridLikelihoodReadoutRealizes
    (base : X → Base)
    (neural : X → Neural)
    (likelihood : Θ → X → Y)
    (readout : Θ → Base → Neural → Y) : Prop :=
  ∀ θ x, readout θ (base x) (neural x) = likelihood θ x

/-- If every likelihood value can be read from the hybrid product, the hybrid
summary is likelihood-family sufficient. -/
theorem hybridLikelihoodReadout_implies_likelihoodSufficient
    {base : X → Base}
    {neural : X → Neural}
    {likelihood : Θ → X → Y}
    {readout : Θ → Base → Neural → Y}
    (hReadout : HybridLikelihoodReadoutRealizes base neural likelihood readout) :
    LikelihoodFamilySufficient (HybridSummary base neural) likelihood := by
  intro x y hxy θ
  have hb : base x = base y := congrArg Prod.fst hxy
  have hn : neural x = neural y := congrArg Prod.snd hxy
  calc
    likelihood θ x = readout θ (base x) (neural x) := (hReadout θ x).symm
    _ = readout θ (base y) (neural y) := by rw [hb, hn]
    _ = likelihood θ y := hReadout θ y

/-- Likelihood sufficiency of the base statistic lifts to the hybrid product. -/
theorem hybridLikelihoodSufficient_of_baseSufficient
    {base : X → Base}
    {neural : X → Neural}
    {likelihood : Θ → X → Y}
    (hBase : LikelihoodFamilySufficient base likelihood) :
    LikelihoodFamilySufficient (HybridSummary base neural) likelihood := by
  intro x y hxy θ
  exact hBase (congrArg Prod.fst hxy) θ

/-- Likelihood sufficiency of the neural statistic lifts to the hybrid product. -/
theorem hybridLikelihoodSufficient_of_neuralSufficient
    {base : X → Base}
    {neural : X → Neural}
    {likelihood : Θ → X → Y}
    (hNeural : LikelihoodFamilySufficient neural likelihood) :
    LikelihoodFamilySufficient (HybridSummary base neural) likelihood := by
  intro x y hxy θ
  exact hNeural (congrArg Prod.snd hxy) θ

/-- Likelihood-on-hybrid-state is a direct instance of likelihood-on-state. -/
theorem hybridLikelihoodOnState_family_sufficient
    (base : X → Base)
    (neural : X → Neural)
    (stateLikelihood : Θ → Base × Neural → Y) :
    LikelihoodFamilySufficient
      (HybridSummary base neural)
      (LikelihoodOnStateFamily (HybridSummary base neural) stateLikelihood) :=
  likelihoodOnState_family_sufficient (HybridSummary base neural) stateLikelihood

/-! ## Likelihood-free / probe hybrid sufficiency -/

/-- Within-base likelihood-free response sufficiency: inside each base-summary
fiber, neural agreement preserves every probe response. -/
def WithinBaseResponseSufficient
    (base : X → Base)
    (neural : X → Neural)
    (response : Probe → X → Y) : Prop :=
  ∀ ⦃x y : X⦄,
    base x = base y → neural x = neural y →
      ∀ p : Probe, response p x = response p y

/-- Inside a base-summary fiber, the neural summary separates any pair that some
probe response distinguishes. -/
def NeuralSeparatesResponseWithinBase
    (base : X → Base)
    (neural : X → Neural)
    (response : Probe → X → Y) : Prop :=
  ∀ ⦃x y : X⦄,
    base x = base y →
      (∃ p : Probe, response p x ≠ response p y) →
        neural x ≠ neural y

/-- Product-summary likelihood-free response sufficiency is exactly within-base
response sufficiency. -/
theorem hybridResponseSufficient_iff_withinBaseResponseSufficient
    {base : X → Base}
    {neural : X → Neural}
    {response : Probe → X → Y} :
    LikelihoodFreeResponseSufficient (HybridSummary base neural) response ↔
      WithinBaseResponseSufficient base neural response := by
  constructor
  · intro hSuff x y hb hn p
    exact hSuff (Prod.ext hb hn) p
  · intro hWithin x y hxy p
    exact hWithin (congrArg Prod.fst hxy) (congrArg Prod.snd hxy) p

/-- Within-base response sufficiency is equivalent to separating all probe
distinctions inside base-summary fibers. -/
theorem withinBaseResponseSufficient_iff_neuralSeparatesResponseWithinBase
    {base : X → Base}
    {neural : X → Neural}
    {response : Probe → X → Y} :
    WithinBaseResponseSufficient base neural response ↔
      NeuralSeparatesResponseWithinBase base neural response := by
  constructor
  · intro hWithin x y hb hResponseDistinct hNeural
    rcases hResponseDistinct with ⟨p, hp⟩
    exact hp (hWithin hb hNeural p)
  · intro hSeparate x y hb hNeural p
    by_contra hResponseDistinct
    exact hSeparate hb ⟨p, hResponseDistinct⟩ hNeural

/-- A hybrid response-sufficient summary forces the neural component to separate
all probe distinctions that remain inside base-summary fibers. -/
theorem hybridResponseSufficient_neuralSeparatesResponseWithinBase
    {base : X → Base}
    {neural : X → Neural}
    {response : Probe → X → Y}
    (hHybrid : LikelihoodFreeResponseSufficient (HybridSummary base neural) response) :
    NeuralSeparatesResponseWithinBase base neural response :=
  (withinBaseResponseSufficient_iff_neuralSeparatesResponseWithinBase).mp
    ((hybridResponseSufficient_iff_withinBaseResponseSufficient).mp hHybrid)

/-- A hybrid likelihood-free response-sufficient summary cannot collapse a pair
in both components if some probe response distinguishes that pair. -/
theorem hybridResponseSufficient_no_base_neural_collision_of_distinguished_response
    {base : X → Base}
    {neural : X → Neural}
    {response : Probe → X → Y}
    (hHybrid : LikelihoodFreeResponseSufficient (HybridSummary base neural) response)
    {x y : X}
    (hResponseDistinct : ∃ p : Probe, response p x ≠ response p y) :
    ¬ (base x = base y ∧ neural x = neural y) := by
  intro hCollision
  rcases hResponseDistinct with ⟨p, hp⟩
  exact hp (hHybrid (Prod.ext hCollision.1 hCollision.2) p)

/-- A likelihood-free/probe response readout from the hybrid product. -/
def HybridResponseReadoutRealizes
    (base : X → Base)
    (neural : X → Neural)
    (response : Probe → X → Y)
    (readout : Probe → Base → Neural → Y) : Prop :=
  ∀ p x, readout p (base x) (neural x) = response p x

/-- If every likelihood-free response can be read from the hybrid product, the
hybrid summary is likelihood-free response sufficient. -/
theorem hybridResponseReadout_implies_likelihoodFreeSufficient
    {base : X → Base}
    {neural : X → Neural}
    {response : Probe → X → Y}
    {readout : Probe → Base → Neural → Y}
    (hReadout : HybridResponseReadoutRealizes base neural response readout) :
    LikelihoodFreeResponseSufficient (HybridSummary base neural) response := by
  intro x y hxy p
  have hb : base x = base y := congrArg Prod.fst hxy
  have hn : neural x = neural y := congrArg Prod.snd hxy
  calc
    response p x = readout p (base x) (neural x) := (hReadout p x).symm
    _ = readout p (base y) (neural y) := by rw [hb, hn]
    _ = response p y := hReadout p y

/-! ## Approximate hybrid readout bridges -/

/-- Approximate within-base target sufficiency. -/
def WithinBaseTargetSufficientWithin
    [PseudoMetricSpace Target]
    (ε : ℝ)
    (base : X → Base)
    (neural : X → Neural)
    (target : X → Target) : Prop :=
  ∀ ⦃x y : X⦄,
    base x = base y → neural x = neural y →
      dist (target x) (target y) ≤ ε

/-- Approximate target readout from a hybrid product summary. -/
def HybridTargetReadoutRealizesWithin
    [PseudoMetricSpace Target]
    (ε : ℝ)
    (base : X → Base)
    (neural : X → Neural)
    (target : X → Target)
    (readout : Base → Neural → Target) : Prop :=
  ∀ x, dist (readout (base x) (neural x)) (target x) ≤ ε

/-- Approximate hybrid target readout implies approximate within-base target
sufficiency, paying readout error on both collapsed inputs. -/
theorem hybridTargetReadoutWithin_implies_withinBaseTargetSufficientWithin
    [PseudoMetricSpace Target]
    {ε : ℝ}
    {base : X → Base}
    {neural : X → Neural}
    {target : X → Target}
    {readout : Base → Neural → Target}
    (hReadout : HybridTargetReadoutRealizesWithin ε base neural target readout) :
    WithinBaseTargetSufficientWithin (ε + ε) base neural target := by
  intro x y hb hn
  have hLeft : dist (target x) (readout (base x) (neural x)) ≤ ε := by
    simpa [dist_comm] using hReadout x
  have hRight : dist (readout (base x) (neural x)) (target y) ≤ ε := by
    rw [hb, hn]
    exact hReadout y
  calc
    dist (target x) (target y)
        ≤ dist (target x) (readout (base x) (neural x)) +
            dist (readout (base x) (neural x)) (target y) := by
          exact dist_triangle _ _ _
    _ ≤ ε + ε := add_le_add hLeft hRight

/-- Approximate within-base likelihood sufficiency. -/
def WithinBaseLikelihoodSufficientWithin
    [PseudoMetricSpace Y]
    (ε : ℝ)
    (base : X → Base)
    (neural : X → Neural)
    (likelihood : Θ → X → Y) : Prop :=
  ∀ ⦃x y : X⦄,
    base x = base y → neural x = neural y →
      ∀ θ : Θ, dist (likelihood θ x) (likelihood θ y) ≤ ε

/-- Approximate product-summary likelihood sufficiency is exactly approximate
within-base likelihood sufficiency. -/
theorem hybridLikelihoodSufficientWithin_iff_withinBaseLikelihoodSufficientWithin
    [PseudoMetricSpace Y]
    {ε : ℝ}
    {base : X → Base}
    {neural : X → Neural}
    {likelihood : Θ → X → Y} :
    LikelihoodFamilySufficientWithin ε (HybridSummary base neural) likelihood ↔
      WithinBaseLikelihoodSufficientWithin ε base neural likelihood := by
  constructor
  · intro hSuff x y hb hn θ
    exact hSuff (Prod.ext hb hn) θ
  · intro hWithin x y hxy θ
    exact hWithin (congrArg Prod.fst hxy) (congrArg Prod.snd hxy) θ

/-- Approximate likelihood readout from a hybrid product summary. -/
def HybridLikelihoodReadoutRealizesWithin
    [PseudoMetricSpace Y]
    (ε : ℝ)
    (base : X → Base)
    (neural : X → Neural)
    (likelihood : Θ → X → Y)
    (readout : Θ → Base → Neural → Y) : Prop :=
  ∀ θ x, dist (readout θ (base x) (neural x)) (likelihood θ x) ≤ ε

/-- Approximate hybrid likelihood readout implies approximate within-base
likelihood sufficiency. -/
theorem hybridLikelihoodReadoutWithin_implies_withinBaseLikelihoodSufficientWithin
    [PseudoMetricSpace Y]
    {ε : ℝ}
    {base : X → Base}
    {neural : X → Neural}
    {likelihood : Θ → X → Y}
    {readout : Θ → Base → Neural → Y}
    (hReadout : HybridLikelihoodReadoutRealizesWithin ε base neural likelihood readout) :
    WithinBaseLikelihoodSufficientWithin (ε + ε) base neural likelihood := by
  intro x y hb hn θ
  have hLeft : dist (likelihood θ x) (readout θ (base x) (neural x)) ≤ ε := by
    simpa [dist_comm] using hReadout θ x
  have hRight : dist (readout θ (base x) (neural x)) (likelihood θ y) ≤ ε := by
    rw [hb, hn]
    exact hReadout θ y
  calc
    dist (likelihood θ x) (likelihood θ y)
        ≤ dist (likelihood θ x) (readout θ (base x) (neural x)) +
            dist (readout θ (base x) (neural x)) (likelihood θ y) := by
          exact dist_triangle _ _ _
    _ ≤ ε + ε := add_le_add hLeft hRight

/-- Approximate within-base likelihood-free response sufficiency. -/
def WithinBaseResponseSufficientWithin
    [PseudoMetricSpace Y]
    (ε : ℝ)
    (base : X → Base)
    (neural : X → Neural)
    (response : Probe → X → Y) : Prop :=
  ∀ ⦃x y : X⦄,
    base x = base y → neural x = neural y →
      ∀ p : Probe, dist (response p x) (response p y) ≤ ε

/-- Approximate product-summary likelihood-free response sufficiency is exactly
approximate within-base response sufficiency. -/
theorem hybridResponseSufficientWithin_iff_withinBaseResponseSufficientWithin
    [PseudoMetricSpace Y]
    {ε : ℝ}
    {base : X → Base}
    {neural : X → Neural}
    {response : Probe → X → Y} :
    QuerySufficientWithin ε (HybridSummary base neural) response ↔
      WithinBaseResponseSufficientWithin ε base neural response := by
  constructor
  · intro hSuff x y hb hn p
    exact hSuff (Prod.ext hb hn) p
  · intro hWithin x y hxy p
    exact hWithin (congrArg Prod.fst hxy) (congrArg Prod.snd hxy) p

/-- Approximate likelihood-free/probe response readout from a hybrid product
summary. -/
def HybridResponseReadoutRealizesWithin
    [PseudoMetricSpace Y]
    (ε : ℝ)
    (base : X → Base)
    (neural : X → Neural)
    (response : Probe → X → Y)
    (readout : Probe → Base → Neural → Y) : Prop :=
  ∀ p x, dist (readout p (base x) (neural x)) (response p x) ≤ ε

/-- Approximate hybrid response readout implies approximate within-base
likelihood-free response sufficiency. -/
theorem hybridResponseReadoutWithin_implies_withinBaseResponseSufficientWithin
    [PseudoMetricSpace Y]
    {ε : ℝ}
    {base : X → Base}
    {neural : X → Neural}
    {response : Probe → X → Y}
    {readout : Probe → Base → Neural → Y}
    (hReadout : HybridResponseReadoutRealizesWithin ε base neural response readout) :
    WithinBaseResponseSufficientWithin (ε + ε) base neural response := by
  intro x y hb hn p
  have hLeft : dist (response p x) (readout p (base x) (neural x)) ≤ ε := by
    simpa [dist_comm] using hReadout p x
  have hRight : dist (readout p (base x) (neural x)) (response p y) ≤ ε := by
    rw [hb, hn]
    exact hReadout p y
  calc
    dist (response p x) (response p y)
        ≤ dist (response p x) (readout p (base x) (neural x)) +
            dist (readout p (base x) (neural x)) (response p y) := by
          exact dist_triangle _ _ _
    _ ≤ ε + ε := add_le_add hLeft hRight

end FormalProofs.OPT
