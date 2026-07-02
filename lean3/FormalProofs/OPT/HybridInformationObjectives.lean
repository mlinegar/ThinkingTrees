import FormalProofs.OPT.HybridSummarySufficiency

/-!
# FormalProofs/OPT/HybridInformationObjectives.lean

Symbolic objective layer for Makinen et al.-style hybrid summary statistics.

The paper motivates learning a neural summary `s(d)` alongside an existing
summary `t(d)` by maximizing conditional information `I(s; theta | t)`.  This
file keeps that information layer deliberately symbolic:

* no Shannon entropy or measure theory is introduced;
* `jointMI`, `conditionalMI`, and loss/proxy functions are ordinary real-valued
  objective functions over a candidate class;
* the chain-rule identity is supplied as a named assumption/interface; and
* EPE / classifier objectives are represented by order-reversing or negated
  proxy assumptions.

The machine-checked content is the optimization algebra: constant shifts
preserve argmax sets, and losses that reverse the information order have the
same optima as information maximization.
-/

set_option linter.mathlibStandardSet false

open scoped Classical

set_option maxHeartbeats 0
set_option maxRecDepth 4000
set_option relaxedAutoImplicit false
set_option autoImplicit false

noncomputable section

namespace FormalProofs.OPT

variable {Candidate : Type*}

/-- A pointwise argmax predicate for a real-valued objective. -/
def IsArgmax (objective : Candidate → ℝ) (candidate : Candidate) : Prop :=
  ∀ other : Candidate, objective other ≤ objective candidate

/-- A pointwise argmin predicate for a real-valued objective. -/
def IsArgmin (objective : Candidate → ℝ) (candidate : Candidate) : Prop :=
  ∀ other : Candidate, objective candidate ≤ objective other

/-- Two objectives have the same order when every pairwise comparison agrees. -/
def ObjectiveOrderEquivalent
    (left right : Candidate → ℝ) : Prop :=
  ∀ a b : Candidate, left a ≤ left b ↔ right a ≤ right b

/-- A loss reverses an information objective when minimizing the loss is the
same pairwise order as maximizing the information objective. -/
def LossOrderReversesInformation
    (loss information : Candidate → ℝ) : Prop :=
  ∀ a b : Candidate, loss a ≤ loss b ↔ information b ≤ information a

/-- A symbolic EPE/proxy-loss relation: the loss is a constant minus an
information objective. -/
def NegatedInformationLoss
    (loss information : Candidate → ℝ)
    (constant : ℝ) : Prop :=
  ∀ candidate : Candidate, loss candidate = constant - information candidate

/-- Constant shifts preserve pairwise objective order. -/
theorem objectiveOrderEquivalent_add_const
    (objective : Candidate → ℝ)
    (constant : ℝ) :
    ObjectiveOrderEquivalent
      (fun candidate => objective candidate + constant)
      objective := by
  intro a b
  constructor
  · intro h
    linarith
  · intro h
    linarith

/-- If `shifted = objective + constant`, then `objective` and `shifted` have
the same argmax candidates. -/
theorem isArgmax_add_const_iff
    {objective shifted : Candidate → ℝ}
    {constant : ℝ}
    (hShift : ∀ candidate, shifted candidate = objective candidate + constant)
    {candidate : Candidate} :
    IsArgmax objective candidate ↔ IsArgmax shifted candidate := by
  constructor
  · intro hMax other
    rw [hShift other, hShift candidate]
    have h := hMax other
    linarith
  · intro hMax other
    have h := hMax other
    rw [hShift other, hShift candidate] at h
    linarith

/-- A symbolic hybrid-MI chain-rule interface.  `baseMI` is fixed because the
existing summary `t(d)` is held fixed while candidate neural summaries vary. -/
structure HybridMIChainRule (Candidate : Type*) where
  jointMI : Candidate → ℝ
  conditionalMI : Candidate → ℝ
  baseMI : ℝ
  chain_rule : ∀ candidate : Candidate,
    jointMI candidate = conditionalMI candidate + baseMI

/-- Under the symbolic chain rule
`I((t,s);theta) = I(s;theta|t) + I(t;theta)`, maximizing conditional MI is
equivalent to maximizing joint hybrid MI because the base term is constant. -/
theorem hybridCMI_argmax_iff_jointMI_argmax
    (rule : HybridMIChainRule Candidate)
    {candidate : Candidate} :
    IsArgmax rule.conditionalMI candidate ↔
      IsArgmax rule.jointMI candidate :=
  isArgmax_add_const_iff
    (objective := rule.conditionalMI)
    (shifted := rule.jointMI)
    (constant := rule.baseMI)
    rule.chain_rule

/-- Order-reversing losses have argmins exactly at information argmaxes. -/
theorem lossArgmin_iff_informationArgmax_of_orderReverses
    {loss information : Candidate → ℝ}
    (hOrder : LossOrderReversesInformation loss information)
    {candidate : Candidate} :
    IsArgmin loss candidate ↔ IsArgmax information candidate := by
  constructor
  · intro hMin other
    exact (hOrder candidate other).mp (hMin other)
  · intro hMax other
    exact (hOrder candidate other).mpr (hMax other)

/-- A loss equal to a constant minus information reverses the information
order. -/
theorem negatedInformationLoss_orderReverses
    {loss information : Candidate → ℝ}
    {constant : ℝ}
    (hLoss : NegatedInformationLoss loss information constant) :
    LossOrderReversesInformation loss information := by
  intro a b
  constructor
  · intro h
    rw [hLoss a, hLoss b] at h
    linarith
  · intro h
    rw [hLoss a, hLoss b]
    linarith

/-- EPE/posterior-style losses represented as a negated information proxy have
the same optima as maximizing that proxy. -/
theorem hybridEPELoss_argmin_iff_information_argmax
    {loss information : Candidate → ℝ}
    {constant : ℝ}
    (hLoss : NegatedInformationLoss loss information constant)
    {candidate : Candidate} :
    IsArgmin loss candidate ↔ IsArgmax information candidate :=
  lossArgmin_iff_informationArgmax_of_orderReverses
    (negatedInformationLoss_orderReverses hLoss)

/-- Classifier/JSD-style losses are handled through the weaker assumption that
the training loss reverses the chosen information-proxy order. -/
theorem hybridClassifierLoss_argmin_iff_informationProxy_argmax
    {loss informationProxy : Candidate → ℝ}
    (hOrder : LossOrderReversesInformation loss informationProxy)
    {candidate : Candidate} :
    IsArgmin loss candidate ↔ IsArgmax informationProxy candidate :=
  lossArgmin_iff_informationArgmax_of_orderReverses hOrder

/-- Combining the symbolic hybrid chain rule with a negated EPE proxy: if the
loss is a constant minus joint hybrid MI, its minimizers also maximize
conditional MI beyond the fixed base summary. -/
theorem hybridEPELoss_argmin_iff_conditionalMI_argmax
    (rule : HybridMIChainRule Candidate)
    {loss : Candidate → ℝ}
    {constant : ℝ}
    (hLoss : NegatedInformationLoss loss rule.jointMI constant)
    {candidate : Candidate} :
    IsArgmin loss candidate ↔ IsArgmax rule.conditionalMI candidate := by
  rw [hybridEPELoss_argmin_iff_information_argmax hLoss]
  exact (hybridCMI_argmax_iff_jointMI_argmax rule).symm

end FormalProofs.OPT
