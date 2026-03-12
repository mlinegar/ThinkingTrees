import Mathlib

/-!
# FormalProofs/OPT/RidgeRegressionToy.lean

## Ridge regression identities (simulation intuition)

This file provides small deterministic lemmas that explain why the
**Segment‑LDA OPS weight‑recovery simulation** should improve with more labeled spans.

Key points (algebraic, not probabilistic):
- The ridge estimator admits an explicit **bias + noise** decomposition.
- In the noiseless case, ridge shrinkage vanishes when the effective Gram term grows
  (a toy “large‑N” regime).

These statements are intentionally minimal: we avoid formalizing the full LDA DGP, and instead
formalize the linear-algebra identities that the simulation relies on.
-/

set_option linter.mathlibStandardSet false

open scoped BigOperators
open scoped Classical
open scoped Matrix
open scoped Topology

set_option maxHeartbeats 200000
set_option maxRecDepth 4000
set_option relaxedAutoImplicit false
set_option autoImplicit false

noncomputable section

namespace FormalProofs.OPT

section RidgeAlgebra

variable {n k : ℕ}

/-- Gram matrix `XᵀX`. -/
def GramMatrix (X : Matrix (Fin n) (Fin k) ℝ) : Matrix (Fin k) (Fin k) ℝ :=
  X.transpose * X

/-- Ridge normal-equation matrix `XᵀX + λI`. -/
def RidgeMatrix (X : Matrix (Fin n) (Fin k) ℝ) (lam : ℝ) : Matrix (Fin k) (Fin k) ℝ :=
  GramMatrix X + lam • (1 : Matrix (Fin k) (Fin k) ℝ)

/-- Ridge estimator in matrix form: `β̂ = A⁻¹ Xᵀ y`, where `A⁻¹` is an inverse of `XᵀX + λI`. -/
def ridgeEstimator (X : Matrix (Fin n) (Fin k) ℝ) (Y : Fin n → ℝ)
    (Ainv : Matrix (Fin k) (Fin k) ℝ) : Fin k → ℝ :=
  (Ainv * X.transpose) *ᵥ Y

/-- Algebraic ridge decomposition:

If `y = Xβ + ε` and `Ainv` is a (left) inverse of `XᵀX + λI`, then

`β̂ = β - λ·Ainvβ + Ainv·Xᵀε`.

This makes the “large‑N intuition” explicit:
- the `-λ·Ainvβ` term is the ridge **shrinkage bias**;
- the `Ainv·Xᵀε` term is the propagated **noise**.
-/
theorem ridgeEstimator_decomposition
    (X : Matrix (Fin n) (Fin k) ℝ) (β : Fin k → ℝ) (ε : Fin n → ℝ)
    (lam : ℝ) (Ainv : Matrix (Fin k) (Fin k) ℝ)
    (h_inv : Ainv * RidgeMatrix X lam = 1) :
    ridgeEstimator X (X *ᵥ β + ε) Ainv =
      β - lam • (Ainv *ᵥ β) + Ainv *ᵥ (X.transpose *ᵥ ε) := by
  classical
  have h_design :
      (Ainv * X.transpose) *ᵥ (X *ᵥ β) = (Ainv * GramMatrix X) *ᵥ β := by
    -- `(Ainv * Xᵀ) *ᵥ (X *ᵥ β) = (Ainv * Xᵀ * X) *ᵥ β`.
    simpa [GramMatrix, Matrix.mul_assoc] using
      (Matrix.mulVec_mulVec (v := β) (M := Ainv * X.transpose) (N := X))

  have h_noise :
      (Ainv * X.transpose) *ᵥ ε = Ainv *ᵥ (X.transpose *ᵥ ε) := by
    -- `(Ainv * Xᵀ) *ᵥ ε = Ainv *ᵥ (Xᵀ *ᵥ ε)`.
    simpa [Matrix.mul_assoc] using
      (Matrix.mulVec_mulVec (v := ε) (M := Ainv) (N := X.transpose)).symm

  have h_gram : Ainv * GramMatrix X = (1 : Matrix (Fin k) (Fin k) ℝ) - lam • Ainv := by
    -- From `Ainv*(XᵀX + λI)=I`, isolate the `Ainv*XᵀX` part.
    have h_sum' :
        Ainv * GramMatrix X + Ainv * (lam • (1 : Matrix (Fin k) (Fin k) ℝ)) =
          (1 : Matrix (Fin k) (Fin k) ℝ) := by
      simpa [RidgeMatrix, GramMatrix, mul_add] using h_inv
    have h_smul : Ainv * (lam • (1 : Matrix (Fin k) (Fin k) ℝ)) = lam • Ainv := by
      -- `Ainv*(λI)=λ*(Ainv*I)=λ*Ainv`.
      simp [mul_smul]
    have h_sum : Ainv * GramMatrix X + lam • Ainv = (1 : Matrix (Fin k) (Fin k) ℝ) := by
      simpa [h_smul] using h_sum'
    exact (eq_sub_iff_add_eq).2 h_sum

  calc
    ridgeEstimator X (X *ᵥ β + ε) Ainv
        = (Ainv * X.transpose) *ᵥ (X *ᵥ β + ε) := rfl
    _ = (Ainv * X.transpose) *ᵥ (X *ᵥ β) + (Ainv * X.transpose) *ᵥ ε := by
        simp [ridgeEstimator, Matrix.mulVec_add]
    _ = (Ainv * GramMatrix X) *ᵥ β + Ainv *ᵥ (X.transpose *ᵥ ε) := by
        rw [h_design, h_noise]
    _ = ((1 : Matrix (Fin k) (Fin k) ℝ) - lam • Ainv) *ᵥ β + Ainv *ᵥ (X.transpose *ᵥ ε) := by
        simp [h_gram]
    _ = (β - lam • (Ainv *ᵥ β)) + Ainv *ᵥ (X.transpose *ᵥ ε) := by
        simp [Matrix.sub_mulVec, Matrix.smul_mulVec]
    _ = β - lam • (Ainv *ᵥ β) + Ainv *ᵥ (X.transpose *ᵥ ε) := by
        simp [add_assoc]

/-- Noiseless special case: if `y = Xβ`, ridge differs from `β` only by the shrinkage term. -/
theorem ridgeEstimator_noiseless
    (X : Matrix (Fin n) (Fin k) ℝ) (β : Fin k → ℝ) (lam : ℝ) (Ainv : Matrix (Fin k) (Fin k) ℝ)
    (h_inv : Ainv * RidgeMatrix X lam = 1) :
    ridgeEstimator X (X *ᵥ β) Ainv = β - lam • (Ainv *ᵥ β) := by
  simpa using
    (ridgeEstimator_decomposition (X := X) (β := β) (ε := (0 : Fin n → ℝ)) (lam := lam)
      (Ainv := Ainv) h_inv)

end RidgeAlgebra

section LinearPredictor

/-!
### From parameter error to oracle-score error

In the simulations, the learned span score is a **linear predictor** in learned weights.
This section records the standard Cauchy–Schwarz bound showing that score error is
Lipschitz in the parameter error (with constant given by the feature norm).
-/

theorem abs_inner_diff_le_norm_mul_norm {E : Type*} [SeminormedAddCommGroup E] [InnerProductSpace ℝ E]
    (x β βhat : E) :
    |inner ℝ x βhat - inner ℝ x β| ≤ ‖x‖ * ‖βhat - β‖ := by
  have h : inner ℝ x βhat - inner ℝ x β = inner ℝ x (βhat - β) := by
    simpa using (inner_sub_right (𝕜 := ℝ) x βhat β).symm
  simpa [h] using (abs_real_inner_le_norm x (βhat - β))

end LinearPredictor

section LargeNToy

open Filter

/-- Toy “large‑N” ridge intuition (scalar shrinkage):

If a scalar Gram term `aₙ` tends to `+∞`, then the ridge shrinkage factor
`aₙ/(aₙ + λ)` tends to `1` (for any fixed `λ`).

This is the core mechanism behind ridge consistency in the noiseless setting when
`XᵀX` scales with the number of labeled samples.
-/
theorem tendsto_shrinkageFactor_one {a : ℕ → ℝ} (lam : ℝ)
    (ha : Tendsto a atTop atTop) :
    Tendsto (fun n : ℕ => a n / (a n + lam)) atTop (𝓝 (1 : ℝ)) := by
  have h_add : Tendsto (fun n : ℕ => a n + lam) atTop atTop := by
    -- Adding a constant preserves divergence to `+∞`.
    rw [tendsto_atTop_atTop] at ha ⊢
    intro b
    rcases ha (b - lam) with ⟨i, hi⟩
    refine ⟨i, ?_⟩
    intro n hn
    have h := hi n hn
    linarith

  have h_inv : Tendsto (fun n : ℕ => (a n + lam)⁻¹) atTop (𝓝 (0 : ℝ)) :=
    (Filter.Tendsto.inv_tendsto_atTop h_add)

  have h_lam_div : Tendsto (fun n : ℕ => lam / (a n + lam)) atTop (𝓝 (0 : ℝ)) := by
    simpa [div_eq_mul_inv] using (tendsto_const_nhds.mul h_inv)

  have h_ne : ∀ᶠ n in atTop, a n + lam ≠ 0 := by
    -- Eventually the denominator is ≥ 1, hence nonzero.
    rcases (tendsto_atTop_atTop.mp h_add) (1 : ℝ) with ⟨i, hi⟩
    refine Filter.eventually_atTop.2 ⟨i, ?_⟩
    intro n hn
    have h1 : (1 : ℝ) ≤ a n + lam := hi n hn
    have hpos : (0 : ℝ) < a n + lam := lt_of_lt_of_le (by linarith) h1
    exact ne_of_gt hpos

  have h_congr :
      (fun n : ℕ => a n / (a n + lam)) =ᶠ[atTop] fun n : ℕ => (1 : ℝ) - lam / (a n + lam) := by
    refine h_ne.mono ?_
    intro n hn
    calc
      a n / (a n + lam)
          = (a n + lam - lam) / (a n + lam) := by simp
      _ = (a n + lam) / (a n + lam) - lam / (a n + lam) := by
          simpa using (sub_div (a n + lam) lam (a n + lam))
      _ = (1 : ℝ) - lam / (a n + lam) := by simp [div_self hn]

  have h_rhs : Tendsto (fun n : ℕ => (1 : ℝ) - lam / (a n + lam)) atTop (𝓝 (1 : ℝ)) := by
    simpa using (tendsto_const_nhds.sub h_lam_div)

  exact (Filter.Tendsto.congr' h_congr.symm h_rhs)

/-- Scalar ridge consistency (deterministic, noisy case).

Write the ridge estimator in 1D as

`β̂ₙ = (aₙ/(aₙ+λ))·β + bₙ/(aₙ+λ)`,

where typically `aₙ = ∑ xᵢ²` (a Gram term) and `bₙ = ∑ xᵢ εᵢ` (a noise cross-term).

If `aₙ → +∞` and the relative noise vanishes (`bₙ/aₙ → 0`), then `β̂ₙ → β`.

This is a clean “pipeline intuition” lemma: more labeled spans make `aₙ` large, and
standard LLN/CLT heuristics make `bₙ/aₙ` small, so the ridge estimator converges.
-/
theorem tendsto_ridgeScalar_consistent {a b : ℕ → ℝ} (lam β : ℝ)
    (ha : Tendsto a atTop atTop)
    (hb : Tendsto (fun n : ℕ => b n / a n) atTop (𝓝 (0 : ℝ))) :
    Tendsto (fun n : ℕ => (a n / (a n + lam)) * β + b n / (a n + lam)) atTop (𝓝 β) := by
  have hs : Tendsto (fun n : ℕ => a n / (a n + lam)) atTop (𝓝 (1 : ℝ)) :=
    tendsto_shrinkageFactor_one (a := a) lam ha

  have h_bias : Tendsto (fun n : ℕ => (a n / (a n + lam)) * β) atTop (𝓝 ((1 : ℝ) * β)) :=
    hs.mul_const β

  have ha_ne : ∀ᶠ n in atTop, a n ≠ 0 := by
    rcases (tendsto_atTop_atTop.mp ha) (1 : ℝ) with ⟨i, hi⟩
    refine Filter.eventually_atTop.2 ⟨i, ?_⟩
    intro n hn
    have h1 : (1 : ℝ) ≤ a n := hi n hn
    have hpos : (0 : ℝ) < a n := lt_of_lt_of_le (by linarith) h1
    exact ne_of_gt hpos

  have h_congr_noise :
      (fun n : ℕ => b n / (a n + lam)) =ᶠ[atTop] fun n : ℕ =>
        (b n / a n) * (a n / (a n + lam)) := by
    refine ha_ne.mono ?_
    intro n hn
    simpa using (div_mul_div_cancel₀ (a := b n) (b := a n) (c := a n + lam) hn).symm

  have h_prod :
      Tendsto (fun n : ℕ => (b n / a n) * (a n / (a n + lam))) atTop (𝓝 ((0 : ℝ) * 1)) :=
    hb.mul hs

  have h_noise : Tendsto (fun n : ℕ => b n / (a n + lam)) atTop (𝓝 (0 : ℝ)) := by
    -- Rewrite `b/(a+λ)` as `(b/a) * (a/(a+λ))` eventually, then take limits.
    have : Tendsto (fun n : ℕ => (b n / a n) * (a n / (a n + lam))) atTop (𝓝 (0 : ℝ)) := by
      simpa using h_prod
    exact (tendsto_congr' h_congr_noise).2 this

  have h_sum :
      Tendsto (fun n : ℕ => (a n / (a n + lam)) * β + b n / (a n + lam)) atTop
        (𝓝 (((1 : ℝ) * β) + 0)) :=
    h_bias.add h_noise

  simpa using h_sum

/-!
### A small vector generalization (isotropic Gram)

In the special case where the ridge normal equations reduce coordinatewise to a shared scalar
Gram term `aₙ` (e.g. `XᵀX = aₙ·I`), the consistency proof is coordinatewise and follows from
`tendsto_ridgeScalar_consistent` plus `tendsto_pi_nhds`.

This is a good “next” intuition step without pulling in spectral theory.
-/

theorem tendsto_ridgeIsotropic_consistent {k : ℕ} {a : ℕ → ℝ} {η : ℕ → Fin k → ℝ}
    (lam : ℝ) (β : Fin k → ℝ)
    (ha : Tendsto a atTop atTop)
    (hη : ∀ i : Fin k, Tendsto (fun n : ℕ => η n i / a n) atTop (𝓝 (0 : ℝ))) :
    Tendsto (fun n : ℕ => fun i : Fin k =>
        (a n / (a n + lam)) * β i + η n i / (a n + lam)) atTop (𝓝 β) := by
  -- Reduce to coordinatewise convergence on `Fin k → ℝ`.
  refine (tendsto_pi_nhds).2 ?_
  intro i
  -- Apply the scalar lemma to each coordinate.
  simpa using
    (tendsto_ridgeScalar_consistent (a := a) (b := fun n => η n i) (lam := lam) (β := β i) ha
      (hη i))

/-!
### Diagonal Gram variant (coordinatewise growth)

This generalizes the isotropic case to coordinatewise “effective sample sizes” `aₙ(i)`.
It still avoids spectral theory while capturing the common situation where features are
approximately orthogonal but not equally scaled.
-/

theorem tendsto_ridgeDiagonal_consistent {k : ℕ} {a : ℕ → Fin k → ℝ} {η : ℕ → Fin k → ℝ}
    (lam : ℝ) (β : Fin k → ℝ)
    (ha : ∀ i : Fin k, Tendsto (fun n : ℕ => a n i) atTop atTop)
    (hη : ∀ i : Fin k, Tendsto (fun n : ℕ => η n i / a n i) atTop (𝓝 (0 : ℝ))) :
    Tendsto (fun n : ℕ => fun i : Fin k =>
        (a n i / (a n i + lam)) * β i + η n i / (a n i + lam)) atTop (𝓝 β) := by
  refine (tendsto_pi_nhds).2 ?_
  intro i
  simpa using
    (tendsto_ridgeScalar_consistent (a := fun n => a n i) (b := fun n => η n i) (lam := lam)
      (β := β i) (ha i) (hη i))

end LargeNToy

end FormalProofs.OPT
