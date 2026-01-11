/-!
# Econometrics Module: Foundations for IPW

## Overview

This module builds the basic econometrics objects needed before IPW is meaningful.
It is intentionally small and mirrors the level of formality in `CLT/`.

## Core Model (Potential Outcomes)

We work with:
- Treatment indicator D in {0,1}
- Potential outcomes Y0, Y1 : Omega -> R
- Observed outcome Y defined by consistency:
  Y = D * Y1 + (1 - D) * Y0

The target estimand is the average treatment effect:
  ATE = E[Y1 - Y0]

## Propensity Score

The propensity score is the conditional mean of D given covariates X:
  e(X) = E[D | X]

We track both:
- a random variable e(omega) = E[D | X](omega)
- a function e(x) with e(X) = E[D | X] a.s. (Doob-Dynkin packaged as a structure)

## Identification (IPW)

Under consistency, unconfoundedness, and positivity:
  ATE = E[ D * Y / e(X) - (1 - D) * Y / (1 - e(X)) ]

## Initial Scope

- Potential outcomes model (Y0, Y1, observed Y)
- Binary treatment and covariates
- Propensity score and positivity
- Identification of ATE via IPW

## File Structure (initial)

```
Econometrics/
├── Core.lean               # Core objects: treatment, potential outcomes, observed outcome
├── Assumptions.lean        # SUTVA, unconfoundedness, positivity
├── PropensityScore.lean    # e(X) = E[D | X] and basic properties
├── IPWIdentification.lean  # ATE identification via IPW
└── README.lean             # This file
```

## Notes

- Probability primitives come from Mathlib and `FormalProofs.CLT.Core`.
- Theorems will be stated with explicit assumptions, not axioms.
- All documentation uses ASCII-only notation for portability.
-/
