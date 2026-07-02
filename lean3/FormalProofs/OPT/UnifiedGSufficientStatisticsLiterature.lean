import FormalProofs.OPT.UnifiedGEstimator
import FormalProofs.OPT.DependenceObjectiveProxies
import FormalProofs.OPT.HybridSummarySufficiency
import FormalProofs.ML.NeuralOperatorApproximation

/-!
# FormalProofs/OPT/UnifiedGSufficientStatisticsLiterature.lean

This module names the main literature routes for learning a unified `g`
sufficient statistic for an oracle/query family `f*`.

The checked statements are intentionally certificate-level:

* NASS / dependence objectives certify optimizer/proxy facts, and become
  sufficient only after a readout or exact-state certificate is supplied.
* SSS/NASSS certifies sufficiency through finite selected slices plus a
  slice-cover condition.
* SSNL/SNLE certifies likelihood-family sufficiency when likelihoods are
  evaluated through the learned state.
* Hybrid summary statistics certify a product-summary route, and collapse back
  to the unified-`g` state when the base statistic is readable from that state.
* Neural-operator approximation certifies unified-`g` sufficiency when the
  realized composed readout approximates the oracle uniformly on the relevant
  two-sided contexts.

No Shannon mutual information, estimator consistency, PAC generalization,
random-direction coverage, density/Jacobian semantics, or SGD convergence is
proved here. Those remain explicit assumptions or external theorem inputs.
-/

set_option linter.mathlibStandardSet false

open scoped Classical

set_option maxHeartbeats 0
set_option maxRecDepth 4000
set_option relaxedAutoImplicit false
set_option autoImplicit false

noncomputable section

namespace FormalProofs.OPT

variable {Candidate X Ctx Carrier Base State Y Slice SliceVal Θ : Type*}

/-! ## Literature method names -/

/-- Method families whose Lean claims are organized in this module. -/
inductive UnifiedGSufficientStatisticMethod where
  /-- Chen et al. 2021 NASS / dependence-proxy learning. -/
  | nassDependenceProxy
  /-- Chen, Gutmann, and Weller 2023 SSS/NASSS sliced-response learning. -/
  | slicedSummaryStatistics
  /-- Dirmeier, Albert, and Perez-Cruz SSNL/SNLE likelihood-on-state learning. -/
  | ssnlLikelihoodOnState
  /-- Makinen et al. hybrid hand-built plus neural summaries. -/
  | hybridSummaryStatistics
  /-- Kovachki-style neural-operator approximation of the shared state map. -/
  | neuralOperatorApproximation
  deriving DecidableEq

/-! ## NASS / dependence-proxy route -/

/-- A NASS-style dependence-proxy certificate for a selected unified-`g`
estimator. The dependence objective explains why the candidate was selected;
the readout certificate is the deterministic condition that actually proves
problem-level sufficiency. -/
structure DependenceProxyUnifiedGCertificate
    (Candidate X Ctx Carrier Y : Type*) where
  family : UnifiedGEstimatorFamily Candidate X Carrier
  selected : Candidate
  problem : UnifiedGProblem X Ctx Y
  loss : Candidate → ℝ
  proxy : Candidate → ℝ
  information : Candidate → ℝ
  order : LossOrderReversesInformation loss proxy
  selected_loss_min : IsArgmin loss selected
  readout : Carrier → Ctx → Y
  readout_realizes :
    ContextReadoutRealizes (family.realize selected).leafState problem.query readout

/-- NASS/DV/MINE/JSD/InfoNCE-style loss minimization gives symbolic proxy
maximization under the supplied order-reversal assumption. -/
theorem nass_dependenceProxy_certificate_proxyArgmax
    (cert : DependenceProxyUnifiedGCertificate Candidate X Ctx Carrier Y) :
    IsArgmax cert.proxy cert.selected :=
  (dependenceLoss_argmin_iff_proxyArgmax cert.order).mp cert.selected_loss_min

/-- If the selected dependence proxy uniformly approximates the target
information objective, proxy optimality yields deterministic near-optimality
for that information objective. This is still not a sufficiency theorem. -/
theorem nass_dependenceProxy_certificate_informationEpsilonArgmax
    {ε : ℝ}
    (cert : DependenceProxyUnifiedGCertificate Candidate X Ctx Carrier Y)
    (hProxy : UniformProxyError ε cert.proxy cert.information) :
    IsEpsilonArgmax (ε + ε) cert.information cert.selected :=
  uniformProxyError_argmax_implies_informationEpsilonArgmax
    hProxy
    (nass_dependenceProxy_certificate_proxyArgmax cert)

/-- The deterministic readout part of a NASS-style certificate is enough to
certify the selected realized unified `g` for the problem adapter. -/
theorem nass_dependenceProxy_readout_certificate_implies_unifiedG_sufficient
    (cert : DependenceProxyUnifiedGCertificate Candidate X Ctx Carrier Y) :
    UnifiedGQuerySufficient (cert.family.realize cert.selected) cert.problem :=
  unifiedG_contextReadoutRealizes_implies_querySufficient
    (E := cert.family.realize cert.selected)
    (P := cert.problem)
    cert.readout_realizes

/-! ## SSS / NASSS finite-slice route -/

/-- Exact finite selected-slice preservation plus a slice-cover condition is
the deterministic SSS/NASSS certificate for a unified `g`. -/
theorem nasss_finiteSlices_certificate_implies_unifiedG_sufficient
    {selected : Finset Slice}
    {E : UnifiedGEstimator X Carrier}
    {P : UnifiedGProblem X Ctx Y}
    {slice : Slice → (Ctx → Y) → SliceVal}
    (hCover : FiniteSlicesCoverResponseFibers selected P.query slice)
    (hZero : UnifiedGSlicedQuerySufficientOn selected E P slice) :
    UnifiedGQuerySufficient E P :=
  unifiedG_finiteSliced_zeroLoss_implies_querySufficient
    (selected := selected)
    (E := E)
    (P := P)
    (slice := slice)
    hCover
    hZero

/-- Approximate selected-slice preservation plus an approximate slice-cover
condition certifies approximate unified-`g` sufficiency. -/
theorem nasss_finiteSlices_within_certificate_implies_unifiedG_sufficientWithin
    [PseudoMetricSpace SliceVal]
    [PseudoMetricSpace Y]
    {selected : Finset Slice}
    {δ ε : ℝ}
    {E : UnifiedGEstimator X Carrier}
    {P : UnifiedGProblem X Ctx Y}
    {slice : Slice → (Ctx → Y) → SliceVal}
    (hCover : FiniteSlicesCoverResponseFibersWithin selected δ ε P.query slice)
    (hWithin : SlicedQuerySufficientWithinOn selected δ E.leafState P.query slice) :
    UnifiedGQuerySufficientWithin ε E P :=
  unifiedG_finiteSlicedWithin_implies_querySufficientWithin
    (selected := selected)
    (δ := δ)
    (ε := ε)
    (E := E)
    (P := P)
    (slice := slice)
    hCover
    hWithin

/-! ## SSNL / SNLE likelihood-on-state route -/

/-- The likelihood family induced by evaluating a likelihood head on a
unified-`g` leaf state. -/
def UnifiedGLikelihoodOnStateFamily
    (E : UnifiedGEstimator X Carrier)
    (stateLikelihood : Θ → Carrier → Y) :
    Θ → X → Y :=
  LikelihoodOnStateFamily E.leafState stateLikelihood

/-- SSNL/SNLE deterministic core: if likelihoods are evaluated only through the
unified-`g` state, then that state is likelihood-family sufficient. -/
theorem ssnl_unifiedG_likelihoodOnState_family_sufficient
    (E : UnifiedGEstimator X Carrier)
    (stateLikelihood : Θ → Carrier → Y) :
    LikelihoodFamilySufficient
      E.leafState
      (UnifiedGLikelihoodOnStateFamily E stateLikelihood) :=
  likelihoodOnState_family_sufficient E.leafState stateLikelihood

/-- If a unified-`g` state decodes to another learned state, it is sufficient
for every likelihood family evaluated through that decoded state. -/
theorem ssnl_unifiedG_stateReadout_likelihoodOnState_family_sufficient
    {E : UnifiedGEstimator X Carrier}
    {state : X → State}
    {decodeState : Carrier → State}
    (hState : TargetReadoutRealizes E.leafState state decodeState)
    (stateLikelihood : Θ → State → Y) :
    LikelihoodFamilySufficient
      E.leafState
      (LikelihoodOnStateFamily state stateLikelihood) :=
  repWithStateReadout_likelihoodOnState_family_sufficient
    (rep := E.leafState)
    (state := state)
    hState
    stateLikelihood

/-! ## Hybrid summary-statistic route -/

/-- Product of a hand-built/base statistic with the learned unified-`g` state.
This is the Makinen-style hybrid summary, specialized to the package `g`
surface. -/
def UnifiedGHybridSummary
    (base : X → Base)
    (E : UnifiedGEstimator X Carrier) :
    X → Base × Carrier :=
  HybridSummary base E.leafState

/-- The hybrid product always refines the learned unified-`g` state. -/
theorem hybrid_unifiedG_summary_sufficient_for_unifiedG_state
    (base : X → Base)
    (E : UnifiedGEstimator X Carrier) :
    TargetSufficientRepresentation
      (UnifiedGHybridSummary base E)
      E.leafState :=
  hybridSummary_sufficient_for_neural base E.leafState

/-- A hybrid readout from `(base(x), g(x))` certifies likelihood-free response
sufficiency of the hybrid product. -/
theorem hybrid_unifiedG_response_readout_implies_hybrid_sufficient
    {base : X → Base}
    {E : UnifiedGEstimator X Carrier}
    {P : UnifiedGProblem X Ctx Y}
    {readout : Ctx → Base → Carrier → Y}
    (hReadout : HybridResponseReadoutRealizes base E.leafState P.query readout) :
    QuerySufficient (UnifiedGHybridSummary base E) P.query :=
  hybridResponseReadout_implies_likelihoodFreeSufficient hReadout

/-- If the base statistic is itself readable from the unified-`g` state, a
hybrid readout collapses back to an ordinary unified-`g` sufficiency
certificate. -/
theorem hybrid_baseReadout_responseReadout_implies_unifiedG_sufficient
    {base : X → Base}
    {E : UnifiedGEstimator X Carrier}
    {P : UnifiedGProblem X Ctx Y}
    {baseReadout : Carrier → Base}
    {readout : Ctx → Base → Carrier → Y}
    (hBase : TargetReadoutRealizes E.leafState base baseReadout)
    (hReadout : HybridResponseReadoutRealizes base E.leafState P.query readout) :
    UnifiedGQuerySufficient E P := by
  apply unifiedG_contextReadoutRealizes_implies_querySufficient
    (E := E)
    (P := P)
    (readout := fun s c => readout c (baseReadout s) s)
  intro x c
  change readout c (baseReadout (E.leafState x)) (E.leafState x) = P.query c x
  rw [hBase x]
  exact hReadout c x

/-- Approximate hybrid readout also collapses back to approximate unified-`g`
sufficiency when the base statistic is readable from the unified state. The
readout error is paid on both collapsed inputs. -/
theorem hybrid_baseReadout_responseReadoutWithin_implies_unifiedG_sufficientWithin
    [PseudoMetricSpace Y]
    {ε : ℝ}
    {base : X → Base}
    {E : UnifiedGEstimator X Carrier}
    {P : UnifiedGProblem X Ctx Y}
    {baseReadout : Carrier → Base}
    {readout : Ctx → Base → Carrier → Y}
    (hBase : TargetReadoutRealizes E.leafState base baseReadout)
    (hReadout : HybridResponseReadoutRealizesWithin ε base E.leafState P.query readout) :
    UnifiedGQuerySufficientWithin (ε + ε) E P := by
  intro x y hxy c
  have hBaseEq : base x = base y := by
    calc
      base x = baseReadout (E.leafState x) := (hBase x).symm
      _ = baseReadout (E.leafState y) := by rw [hxy]
      _ = base y := hBase y
  have hLeft : dist (P.query c x) (readout c (base x) (E.leafState x)) ≤ ε := by
    simpa [dist_comm] using hReadout c x
  have hRight : dist (readout c (base x) (E.leafState x)) (P.query c y) ≤ ε := by
    rw [hBaseEq, hxy]
    exact hReadout c y
  calc
    dist (P.query c x) (P.query c y)
        ≤ dist (P.query c x) (readout c (base x) (E.leafState x)) +
            dist (readout c (base x) (E.leafState x)) (P.query c y) := by
          exact dist_triangle _ _ _
    _ ≤ ε + ε := add_le_add hLeft hRight

/-! ## Neural-operator approximation route -/

/-- Oracle value for a two-sided contextual triple. -/
def UnifiedGTwoSidedOracle
    [Monoid X]
    (fstar : X → Y) :
    X × X × X → Y :=
  fun t => fstar (t.1 * t.2.1 * t.2.2)

/-- Realized composed readout induced by one shared unified `g`. -/
def UnifiedGTwoSidedComposedReadout
    [Monoid X]
    (E : UnifiedGEstimator X Carrier)
    (readout : Carrier → Y) :
    X × X × X → Y :=
  fun t =>
    readout
      (E.mergeState
        (E.mergeState (E.leafState t.1) (E.leafState t.2.1))
        (E.leafState t.2.2))

/-- Kovachki/neural-operator route in the form needed by the unified-`g`
contract: uniform compact-set approximation over all two-sided triples implies
approximate two-sided contextual sufficiency, with the existing `2ε` slack. -/
theorem neuralOperator_uniformApproxAllTriples_implies_unifiedG_sufficientWithin
    [Monoid X]
    [PseudoMetricSpace Y]
    {ε : ℝ}
    {E : UnifiedGEstimator X Carrier}
    {readout : Carrier → Y}
    {fstar : X → Y}
    {K : ML.CompactRealizedCallSet (X × X × X)}
    (hAllTriples : ∀ t : X × X × X, t ∈ K.carrier)
    (hApprox :
      ML.UniformOperatorApproxOnCompact
        (UnifiedGTwoSidedOracle (X := X) fstar)
        (UnifiedGTwoSidedComposedReadout (X := X) E readout)
        K
        ε) :
    UnifiedGQuerySufficientWithin
      (2 * ε)
      E
      (UnifiedGProblem.twoSided fstar) := by
  apply unifiedG_composedTwoSidedReadoutWithin_implies_querySufficientWithin
    (E := E)
    (readout := readout)
    (fstar := fstar)
  intro left x right
  have h :=
    hApprox
      (left, x, right)
      (hAllTriples (left, x, right))
  simpa [UnifiedGTwoSidedOracle, UnifiedGTwoSidedComposedReadout, dist_comm] using h

/-- Direct theorem-name wrapper for the common neural-operator certificate:
if the realized composed readout approximates `f*` on every two-sided context,
the estimator is approximately sufficient. -/
theorem neuralOperator_composedReadoutWithin_implies_unifiedG_sufficientWithin
    [Monoid X]
    [PseudoMetricSpace Y]
    {ε : ℝ}
    {E : UnifiedGEstimator X Carrier}
    {readout : Carrier → Y}
    {fstar : X → Y}
    (hApprox :
      ∀ left x right,
        dist
          (readout
            (E.mergeState
              (E.mergeState (E.leafState left) (E.leafState x))
              (E.leafState right)))
          (fstar (left * x * right)) ≤ ε) :
    UnifiedGQuerySufficientWithin
      (2 * ε)
      E
      (UnifiedGProblem.twoSided fstar) :=
  unifiedG_composedTwoSidedReadoutWithin_implies_querySufficientWithin
    (E := E)
    (readout := readout)
    (fstar := fstar)
    hApprox

end FormalProofs.OPT

