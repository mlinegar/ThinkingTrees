/-!
# CLT Module: Central Limit Theorem

## Overview

This module formalizes the **Central Limit Theorem** and supporting probability theory.
It provides the measure-theoretic foundations for asymptotic analysis.

## Main Results

| Theorem | File | Statement |
|---------|------|-----------|
| Helly Selection | HellySelection | Tight sequences have convergent subsequences |
| Lévy Continuity | LevyContinuity | Characteristic function convergence ⟹ distribution convergence |
| CLT | CLT | Lindeberg-Feller CLT for triangular arrays |

## File Structure

```
CLT/
├── Core.lean           # Basic probability definitions
├── ProbabilityLaws.lean # Laws, transfer theorem, total probability
├── Distributions.lean  # Binomial/Geometric/Poisson/Uniform/Exponential/Gaussian
├── GeneratingFunctions.lean # PGF and sums of independent RVs
├── WeakLaw.lean        # Weak law of large numbers
├── Normal.lean         # Normal distribution properties
├── HellySelection.lean # Helly's selection theorem (tightness)
├── LevyContinuity.lean # Lévy's continuity theorem
└── CLT.lean            # Central Limit Theorem
```

## Key Concepts

### Tightness

A sequence of probability measures {μ_n} is **tight** if:
```
∀ε > 0, ∃K compact: μ_n(K) > 1 - ε for all n
```

### Characteristic Functions

The characteristic function of μ is:
```
φ_μ(t) = E[exp(itX)] = ∫ exp(itx) dμ(x)
```

### Lévy Continuity Theorem

If φ_n → φ pointwise and φ is continuous at 0, then:
- φ is a characteristic function
- μ_n ⟹ μ (weak convergence)

### Central Limit Theorem

For triangular array {X_{n,k}} with:
- Row independence
- Lindeberg condition

We have:
```
(Σ_k X_{n,k} - E[...]) / σ_n ⟹ N(0,1)
```

## Axioms Used

This module uses no axioms beyond Mathlib foundations.

## References

- Billingsley, P. (1995). Probability and Measure, 3rd ed.
- Dudley, R.M. (2002). Real Analysis and Probability
-/
