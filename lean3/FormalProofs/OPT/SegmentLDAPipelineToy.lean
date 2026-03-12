import FormalProofs.OPT.TopicBigramOracle
import FormalProofs.OPT.RidgeRegressionToy

/-!
# FormalProofs/OPT/SegmentLDAPipelineToy.lean

## Segment‑LDA OPS pipeline: toy convergence lemmas

This file is a small “glue layer” between two simulation-aligned formal components:

- `TopicBigramOracle.lean` proves the oracle score is **mergeable** from a unigram+bigram sketch.
- `RidgeRegressionToy.lean` proves deterministic **ridge consistency** in a toy large‑`N` regime.

The Segment‑LDA weight‑recovery simulation is designed to satisfy the assumptions of these lemmas
as label count grows, so the learned linear predictor should converge to the true oracle score.

We keep this file intentionally lightweight: it provides continuity lemmas for linear scores and a
single composition lemma with `tendsto_ridgeDiagonal_consistent`.
-/

set_option linter.mathlibStandardSet false

open scoped BigOperators
open scoped Classical
open scoped Topology

set_option maxHeartbeats 200000
set_option maxRecDepth 4000
set_option relaxedAutoImplicit false
set_option autoImplicit false

noncomputable section

namespace FormalProofs.OPT

open Filter

section LinearScore

variable {k : ℕ}

/-- A simple linear score on finite coordinate vectors: `score(x,β) = Σᵢ xᵢ βᵢ`. -/
def linearScore (x β : Fin k → ℝ) : ℝ :=
  ∑ i : Fin k, x i * β i

/-- If `β̂ₙ → β` coordinatewise, then the linear score `Σᵢ xᵢ β̂ₙᵢ` tends to `Σᵢ xᵢ βᵢ`.

This is the topological fact used to turn ridge weight consistency into oracle-score consistency
in the simulation. -/
theorem tendsto_linearScore_of_tendsto {βhat : ℕ → Fin k → ℝ} {β x : Fin k → ℝ}
    (h : Tendsto βhat atTop (𝓝 β)) :
    Tendsto (fun n : ℕ => linearScore (k := k) x (βhat n)) atTop (𝓝 (linearScore (k := k) x β)) := by
  classical
  have hi : ∀ i : Fin k, Tendsto (fun n : ℕ => x i * βhat n i) atTop (𝓝 (x i * β i)) := by
    intro i
    have hproj : Tendsto (fun n : ℕ => βhat n i) atTop (𝓝 (β i)) := (tendsto_pi_nhds).1 h i
    simpa using (hproj.const_mul (x i))

  have :
      Tendsto (fun n : ℕ => ∑ i : Fin k, x i * βhat n i) atTop (𝓝 (∑ i : Fin k, x i * β i)) := by
    refine
      (tendsto_finset_sum (s := (Finset.univ : Finset (Fin k)))
          (f := fun i n => x i * βhat n i) (a := fun i => x i * β i) (x := atTop) ?_)
    intro i _hi
    simpa using hi i

  simpa [linearScore] using this

end LinearScore

section LinearScoreRidge

/-!
### Composing toy ridge consistency with score continuity

The lemma `tendsto_ridgeDiagonal_consistent` shows `β̂ₙ → β` coordinatewise under a diagonal Gram
growth + vanishing-relative-noise assumption. Combining with `tendsto_linearScore_of_tendsto`
gives score consistency for any fixed feature vector `x`.
-/

variable {k : ℕ}

theorem tendsto_linearScore_ridgeDiagonal_consistent
    {a : ℕ → Fin k → ℝ} {η : ℕ → Fin k → ℝ}
    (lam : ℝ) (β x : Fin k → ℝ)
    (ha : ∀ i : Fin k, Tendsto (fun n : ℕ => a n i) atTop atTop)
    (hη : ∀ i : Fin k, Tendsto (fun n : ℕ => η n i / a n i) atTop (𝓝 (0 : ℝ))) :
    Tendsto
        (fun n : ℕ =>
          linearScore (k := k) x (fun i : Fin k =>
            (a n i / (a n i + lam)) * β i + η n i / (a n i + lam)))
        atTop (𝓝 (linearScore (k := k) x β)) := by
  have hβ :
      Tendsto (fun n : ℕ => fun i : Fin k =>
        (a n i / (a n i + lam)) * β i + η n i / (a n i + lam)) atTop (𝓝 β) :=
    tendsto_ridgeDiagonal_consistent (k := k) (a := a) (η := η) (lam := lam) (β := β) ha hη
  exact tendsto_linearScore_of_tendsto (k := k) (βhat := fun n => fun i =>
    (a n i / (a n i + lam)) * β i + η n i / (a n i + lam)) (β := β) (x := x) hβ

end LinearScoreRidge

end FormalProofs.OPT

