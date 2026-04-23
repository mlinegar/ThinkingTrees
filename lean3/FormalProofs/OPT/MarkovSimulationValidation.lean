import FormalProofs.OPT.MarkovPathDGP
import FormalProofs.OPT.TheoremBackingAssumptions
import FormalProofs.OPT.AdaptiveChunkingBridge
import FormalProofs.DSL.RuntimeCertificates

/-!
# FormalProofs/OPT/MarkovSimulationValidation.lean

Simulation-facing validation contracts for the Markov changepoint experiments.

This file does not attempt to verify Python execution traces directly. Instead,
it makes explicit which theorem-bearing objects a simulation family must witness
before its results count as Lean-backed:

- exact lanes need a sound tree policy over realized Markov paths,
- exact topology claims then inherit supportwise exact state/count recovery,
- count-only controls are ruled out for topology claims by a concrete
  merge-fold counterexample, and
- approximate lanes need checked runtime nodewise-audit artifacts, which
  compile directly to stochastic adaptive approximate local laws.
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

open MarkovCountSketch

/-- The exact theorem-facing summarizer for realized Markov paths: keep the raw
path itself and reason through the encoded exact sketch state. -/
abbrev markovPathIdentitySummarizer (n : ℕ) : Summarizer (MarkovPath n) :=
  sketchSummarizer (identitySketchOperator (Strings := MarkovPath n))

/-- Oracle used for local-law / theorem-backing transport on realized Markov
paths: encode the exact mergeable Markov sketch state. -/
abbrev markovPathStateOracle (n : ℕ) : MarkovPath n → ℝ :=
  encodedOracle
    (Strings := MarkovPath n)
    (MarkovPath.encodePath (n := n))

/-- Stochastic one-leaf policy used by the one-leaf/full-document exact-collapse
sanity check. Deterministic policies are represented as degenerate PMFs. -/
def oneLeafMarkovPathPolicy (n : ℕ) :
    StochasticAdaptiveTreeMap (MarkovPath n) :=
  fun x => PMF.pure (BinTree.leaf x)

/-- The one-leaf Markov path policy is sound: each sampled tree reconstructs the
original document exactly. -/
theorem oneLeafMarkovPathPolicy_sound (n : ℕ) :
    StochasticAdaptiveChunkingSound (oneLeafMarkovPathPolicy n) := by
  intro x T hT
  have hEq : T = BinTree.leaf x := by
    simpa [oneLeafMarkovPathPolicy] using hT
  simp [hEq, S]

/-- Exact theorem-facing contract for Markov simulations whose tree policy is
already fixed and should preserve the exact Markov sketch state on support. -/
structure ExactMarkovPathSimulationContract
    (n : ℕ)
    (τ : StochasticAdaptiveTreeMap (MarkovPath n)) where
  sound : StochasticAdaptiveChunkingSound τ

/-- Any stochastic policy over realized Markov paths inherits exact local laws
on each support tree when the theorem-facing object is the exact encoded state. -/
theorem markovPath_stochastic_policy_local_laws
    {n : ℕ}
    (τ : StochasticAdaptiveTreeMap (MarkovPath n)) :
    StochasticAdaptiveLocalLaws
      (g := markovPathIdentitySummarizer n)
      (fstar := markovPathStateOracle n)
      τ := by
  intro x T hT
  simpa [markovPathIdentitySummarizer, markovPathStateOracle] using
    (MarkovPath.local_laws_of_encoded_state (n := n) (T := T))

/-- Every support tree of a Markov path policy is exact theorem-backed for the
encoded exact Markov sketch state. -/
theorem markovPath_exactTheoremBacked_on_support
    {n : ℕ}
    (τ : StochasticAdaptiveTreeMap (MarkovPath n))
    {x : MarkovPath n}
    {T : BinTree (MarkovPath n)}
    (hT : T ∈ (τ x).support) :
    ExactTheoremBacked
      (markovPathIdentitySummarizer n)
      T
      (markovPathStateOracle n) := by
  exact ExactTheoremBacked.ofLocalLaws
    (markovPath_stochastic_policy_local_laws (n := n) τ x T hT)

namespace ExactMarkovPathSimulationContract

/-- Support trees of an exact Markov simulation contract preserve the full exact
Markov sketch state. -/
theorem state_exact_on_support
    {n : ℕ}
    {τ : StochasticAdaptiveTreeMap (MarkovPath n)}
    (contract : ExactMarkovPathSimulationContract n τ)
    {x : MarkovPath n}
    {T : BinTree (MarkovPath n)}
    (hT : T ∈ (τ x).support) :
    mergeFold
      (encode := MarkovPath.encodePath (n := n))
      (merge := (· * ·))
      T =
    MarkovPath.encodePath (n := n) x := by
  calc
    mergeFold
        (encode := MarkovPath.encodePath (n := n))
        (merge := (· * ·))
        T =
      MarkovPath.encodePath (n := n) (S T) := by
        simpa using (MarkovPath.state_exact_on_tree (n := n) (u := fun s => s) (T := T))
    _ = MarkovPath.encodePath (n := n) x := by
        have hSound : S T = x := contract.sound x T hT
        simp [hSound]

/-- Any downstream utility on the exact Markov sketch state is preserved on
every realized support tree of an exact Markov simulation contract. -/
theorem state_utility_exact_on_support
    {n : ℕ}
    {τ : StochasticAdaptiveTreeMap (MarkovPath n)}
    (contract : ExactMarkovPathSimulationContract n τ)
    {β : Type*}
    (u : MarkovCountSketch n → β)
    {x : MarkovPath n}
    {T : BinTree (MarkovPath n)}
    (hT : T ∈ (τ x).support) :
    u
      (mergeFold
        (encode := MarkovPath.encodePath (n := n))
        (merge := (· * ·))
        T) =
      u (MarkovPath.encodePath (n := n) x) := by
  simp [state_exact_on_support (contract := contract) hT]

/-- In particular, the realized support tree preserves the changepoint-count
target exactly. This is the theorem-facing exact-collapse / topology gate for
the clean Markov benchmark. -/
theorem changepoint_count_exact_on_support
    {n : ℕ}
    {τ : StochasticAdaptiveTreeMap (MarkovPath n)}
    (contract : ExactMarkovPathSimulationContract n τ)
    {x : MarkovPath n}
    {T : BinTree (MarkovPath n)}
    (hT : T ∈ (τ x).support) :
    MarkovCountSketch.count
      (mergeFold
        (encode := MarkovPath.encodePath (n := n))
        (merge := (· * ·))
        T) =
      MarkovPath.changepointCount (n := n) x := by
  calc
    MarkovCountSketch.count
        (mergeFold
          (encode := MarkovPath.encodePath (n := n))
          (merge := (· * ·))
          T) =
      MarkovPath.changepointCount (n := n) (S T) := by
        simpa using (MarkovPath.count_exact_on_tree (n := n) (T := T))
    _ = MarkovPath.changepointCount (n := n) x := by
        have hSound : S T = x := contract.sound x T hT
        simp [hSound]

end ExactMarkovPathSimulationContract

/-- The one-leaf/full-document exact-collapse lane is theorem-backed by the
degenerate one-leaf policy. -/
def oneLeafMarkovPathExactContract (n : ℕ) :
    ExactMarkovPathSimulationContract n (oneLeafMarkovPathPolicy n) where
  sound := oneLeafMarkovPathPolicy_sound n

/-- Count-only summaries cannot certify general topology claims: there is no
single additive merge fold on the count-only statistic that recovers the true
count on every tree. -/
theorem markov_countOnly_not_exact_on_all_trees
    {n : ℕ}
    (hn : 1 < n) :
    ¬ ∀ T : BinTree (MarkovPath n),
      mergeFold
        (encode := MarkovPath.countOnlyFeature (n := n))
        (merge := Nat.add)
        T =
      MarkovPath.countOnlyFeature (n := n) (S T) := by
  intro hExact
  rcases MarkovPath.countOnly_mergeFold_counterexample (n := n) hn with ⟨T, hT⟩
  exact hT (hExact T)

/-- Runtime-audited approximate contract for Markov simulations. The runtime may
choose any theorem-facing summarizer `g`, but every realized support tree must
carry a checked nodewise audit artifact. -/
structure RuntimeAuditedMarkovPathSimulationContract
    (n : ℕ)
    (g : Summarizer (MarkovPath n))
    (τ : StochasticAdaptiveTreeMap (MarkovPath n)) where
  sound : StochasticAdaptiveChunkingSound τ
  audit :
    ∀ x T, T ∈ (τ x).support →
      DSL.RuntimeNodewiseAuditArtifact g T (markovPathStateOracle n)
  audit_checked :
    ∀ x T hT, (audit x T hT).check = true

/-- Leaf budget extracted from a checked runtime audit artifact on a support
tree, with zero off-support by convention. -/
noncomputable def runtimeAuditedMarkovPathLeafBudget
    {n : ℕ}
    {g : Summarizer (MarkovPath n)}
    {τ : StochasticAdaptiveTreeMap (MarkovPath n)}
    (contract : RuntimeAuditedMarkovPathSimulationContract n g τ)
    (x : MarkovPath n)
    (T : BinTree (MarkovPath n)) : ℝ :=
  if hT : T ∈ (τ x).support then
    (contract.audit x T hT).upper.epsLeaf
  else 0

/-- Merge budget extracted from a checked runtime audit artifact on a support
tree, with zero off-support by convention. -/
noncomputable def runtimeAuditedMarkovPathMergeBudget
    {n : ℕ}
    {g : Summarizer (MarkovPath n)}
    {τ : StochasticAdaptiveTreeMap (MarkovPath n)}
    (contract : RuntimeAuditedMarkovPathSimulationContract n g τ)
    (x : MarkovPath n)
    (T : BinTree (MarkovPath n)) : ℝ :=
  if hT : T ∈ (τ x).support then
    (contract.audit x T hT).upper.epsMerge
  else 0

/-- Idempotence budget extracted from a checked runtime audit artifact on a
support tree, with zero off-support by convention. -/
noncomputable def runtimeAuditedMarkovPathIdempBudget
    {n : ℕ}
    {g : Summarizer (MarkovPath n)}
    {τ : StochasticAdaptiveTreeMap (MarkovPath n)}
    (contract : RuntimeAuditedMarkovPathSimulationContract n g τ)
    (x : MarkovPath n)
    (T : BinTree (MarkovPath n)) : ℝ :=
  if hT : T ∈ (τ x).support then
    (contract.audit x T hT).upper.epsIdemp
  else 0

namespace RuntimeAuditedMarkovPathSimulationContract

/-- On each realized support tree, a checked runtime audit artifact agrees with
the approximate-local-law bundle reconstructed from its empirical certificate. -/
theorem approx_bundle_eq_on_support
    {n : ℕ}
    {g : Summarizer (MarkovPath n)}
    {τ : StochasticAdaptiveTreeMap (MarkovPath n)}
    (contract : RuntimeAuditedMarkovPathSimulationContract n g τ)
    {x : MarkovPath n}
    {T : BinTree (MarkovPath n)}
    (hT : T ∈ (τ x).support) :
    FormalProofs.OPT.approx_bundle_of_audited_upper_bounds
        g T (markovPathStateOracle n) (contract.audit x T hT).upper =
      FormalProofs.OPT.approx_bundle_of_nodewise_empirical_certificate
        g T (markovPathStateOracle n) (contract.audit x T hT).cert := by
  exact DSL.RuntimeNodewiseAuditArtifact.approx_bundle_eq_of_check
    (art := contract.audit x T hT)
    (h_check := contract.audit_checked x T hT)

/-- On each realized support tree, a runtime-audited Markov simulation contract
induces approximate theorem-backedness for the theorem-facing encoded state. -/
def approxTheoremBacked_on_support
    {n : ℕ}
    {g : Summarizer (MarkovPath n)}
    {τ : StochasticAdaptiveTreeMap (MarkovPath n)}
    (contract : RuntimeAuditedMarkovPathSimulationContract n g τ)
    {x : MarkovPath n}
    {T : BinTree (MarkovPath n)}
    (hT : T ∈ (τ x).support) :
    ApproxTheoremBacked g T (markovPathStateOracle n) := by
  exact ApproxTheoremBacked.ofApproxLocalLaws
    (FormalProofs.OPT.approx_bundle_of_audited_upper_bounds
      g T (markovPathStateOracle n) (contract.audit x T hT).upper)

/-- Checked runtime audit artifacts compile directly to stochastic adaptive
approximate local laws for the theorem-facing encoded Markov state. This is the
Lean validation gate for rerunning approximate topology experiments. -/
theorem stochastic_approx_local_laws
    {n : ℕ}
    {g : Summarizer (MarkovPath n)}
    {τ : StochasticAdaptiveTreeMap (MarkovPath n)}
    (contract : RuntimeAuditedMarkovPathSimulationContract n g τ) :
    StochasticAdaptiveApproxLocalLaws
      (g := g)
      (fstar := markovPathStateOracle n)
      τ
      (runtimeAuditedMarkovPathLeafBudget contract)
      (runtimeAuditedMarkovPathMergeBudget contract)
      (runtimeAuditedMarkovPathIdempBudget contract) := by
  intro x T hT
  refine ⟨?_, ?_, ?_⟩
  · simpa [runtimeAuditedMarkovPathLeafBudget, hT] using
      (contract.audit x T hT).upper.leaf_cert
  · simpa [runtimeAuditedMarkovPathMergeBudget, hT] using
      (contract.audit x T hT).upper.merge_cert
  · simpa [runtimeAuditedMarkovPathIdempBudget, hT] using
      (contract.audit x T hT).upper.idemp_cert

end RuntimeAuditedMarkovPathSimulationContract

end FormalProofs.OPT
