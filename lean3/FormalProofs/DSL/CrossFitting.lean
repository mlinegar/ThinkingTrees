import FormalProofs.DSL.DSLEstimator

/-!
# FormalProofs/DSL/CrossFitting.lean

## Paper Reference: Appendix B.2 (Cross-Fitting)

Cross-fitting reduces overfitting bias by training predictors on data
outside each evaluation fold and predicting only on held-out folds.

This file defines:
- `CrossFit`: fold assignment + fold-specific predictors
- `DSLMomentCF`: DSL moment using cross-fitted predictions
- `meanDSLMomentCF`: population mean of cross-fitted moments
- a basic lemma showing reduction to a single predictor
-/

set_option linter.mathlibStandardSet false

open scoped BigOperators
open scoped Classical

set_option maxHeartbeats 0
set_option maxRecDepth 4000
set_option relaxedAutoImplicit false
set_option autoImplicit false

noncomputable section

namespace DSL

/-- Cross-fitting structure: assigns each unit to a fold with a fold-specific predictor. -/
structure CrossFit (ι Obs Con Mis : Type*) [Fintype ι] where
  K : ℕ
  fold : ι → Fin K
  gk : Fin K → Obs → Con → Mis

namespace CrossFit

variable {ι Obs Con Mis : Type*} [Fintype ι]

/-- Predictor for a given unit, using its assigned fold. -/
def predict (cf : CrossFit ι Obs Con Mis) (i : ι) (d_obs : Obs) (q : Con) : Mis :=
  cf.gk (cf.fold i) d_obs q

/-- All folds share the same predictor. -/
def ConstPredictor (cf : CrossFit ι Obs Con Mis) (g0 : Obs → Con → Mis) : Prop :=
  ∀ k, cf.gk k = g0

end CrossFit

/-- DSL moment with cross-fitted predictions (Appendix B.2). -/
def DSLMomentCF {ι Obs Con Mis : Type*} [Fintype ι] {d : ℕ}
    (m : MomentFunction (Obs × Mis) d)
    (cf : CrossFit ι Obs Con Mis)
    (d_obs : ι → Obs)
    (d_mis_true : ι → Mis)
    (q : ι → Con)
    (R : ι → SamplingIndicator)
    (π : ι → ℝ)
    (β : Fin d → ℝ)
    (i : ι) : Fin d → ℝ :=
  let d_mis_pred := cf.predict i (d_obs i) (q i)
  DSLMoment m (d_obs i) d_mis_pred (d_mis_true i) (R i) (π i) β

/-- Population mean of cross-fitted DSL moments. -/
def meanDSLMomentCF {ι Obs Con Mis : Type*} [Fintype ι] [DecidableEq ι] {d : ℕ}
    (m : MomentFunction (Obs × Mis) d)
    (cf : CrossFit ι Obs Con Mis)
    (d_obs : ι → Obs)
    (d_mis_true : ι → Mis)
    (q : ι → Con)
    (R : ι → SamplingIndicator)
    (π : ι → ℝ)
    (β : Fin d → ℝ) : Fin d → ℝ :=
  fun j =>
    (∑ i, DSLMomentCF m cf d_obs d_mis_true q R π β i j) / (Fintype.card ι : ℝ)

/-- Cross-fitting reduces to a single predictor when all folds share g0. -/
lemma DSLMomentCF_eq_of_const {ι Obs Con Mis : Type*} [Fintype ι] {d : ℕ}
    (m : MomentFunction (Obs × Mis) d)
    (cf : CrossFit ι Obs Con Mis)
    (g0 : Obs → Con → Mis)
    (h_const : CrossFit.ConstPredictor cf g0)
    (d_obs : ι → Obs)
    (d_mis_true : ι → Mis)
    (q : ι → Con)
    (R : ι → SamplingIndicator)
    (π : ι → ℝ)
    (β : Fin d → ℝ)
    (i : ι) :
    DSLMomentCF m cf d_obs d_mis_true q R π β i =
      DSLMoment m (d_obs i) (g0 (d_obs i) (q i)) (d_mis_true i) (R i) (π i) β := by
  unfold DSLMomentCF CrossFit.predict
  have h := h_const (cf.fold i)
  simp [h]

end DSL

end
