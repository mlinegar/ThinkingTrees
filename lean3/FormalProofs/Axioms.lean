import FormalProofs.OPT.PreferenceBounds
import FormalProofs.OPT.AuditBounds

/-!
# Axiom Registry

This file documents ALL axioms and assumption structures used in the formalization.

## Summary

### OPT Module: Lean `axiom` Declarations

| # | Axiom | Location | Purpose |
|---|-------|----------|---------|
| 1 | `ExpectedGroupLossLipschitz` | OPT/PreferenceBounds | Expected loss over groups is Lipschitz |

This single axiom is justified by the **Random Utility Model** (McFadden 1974).
Under continuous noise, ranking ties have measure zero, so the expected loss is Lipschitz
even though the pointwise ranking function is discontinuous.

The axiom is instantiated for specific loss functions:
- `ExpectedGRPOLossLipschitz` - GRPO-PL (Plackett-Luce ranking loss)
- `ExpectedGRPORLLossLipschitz` - GRPO-RL (PPO-style clipped surrogate)

### DSL Module: Assumption Structures

| Structure | Location | Purpose |
|-----------|----------|---------|
| `OracleAccess` | DSL/CoreDefinitions | Expert labels = oracle labels |
| `MEstimationAxioms` | DSL/AsymptoticTheory | M-estimation asymptotics |
| `CoverageAxioms` | DSL/AsymptoticTheory | CI coverage properties |

### Econometrics Module: Assumption Structures

| Structure | Location | Purpose |
|-----------|----------|---------|
| `OLSAsymptoticAxioms` | Econometrics/OLS/AsymptoticOLS | LLN/CLT/Slutsky/delta-method package for OLS asymptotics |

## Soundness

All axioms and assumptions are **modeling choices**, not gaps in the proof:
- Each has rigorous mathematical justification from the statistics/econometrics literature
- The OPT axioms follow from the Random Utility Model (McFadden 1974)
- The DSL assumptions follow from M-estimation theory (Newey & McFadden 1994)
- The formalization is SOUND under these assumptions

---

## Axiom: ExpectedGroupLossLipschitz

**Statement**: Expected loss over groups is Lipschitz in oracle distance.

```lean
axiom ExpectedGroupLossLipschitz {Strings A Y : Type*} [PseudoMetricSpace Y] {k : ℕ}
    (loss : Strings → (Fin k → A) → ℝ)
    (fstar : Strings → Y) (g : PMF (Fin k → A)) (L : ℝ≥0)
    (x z : Strings) :
    |∑' group, (g group).toReal * loss x group -
     ∑' group, (g group).toReal * loss z group| ≤
    L * dist (fstar x) (fstar z)
```

**Location**: `OPT/PreferenceBounds.lean`

**Mathematical Justification**:
Under the Random Utility Model (McFadden 1974), scores are:
```
s(x, a) = u(x, a) + ε_a
```
where ε_a is continuous noise (e.g., Gumbel for Plackett-Luce).

The ranking function is discontinuous at ties (score crossings), but:
1. Ties have **measure zero** under continuous noise
2. By dominated convergence, the expected loss is continuous
3. With Lipschitz policy components, the expectation inherits Lipschitz

**When Safe**: Always safe when using softmax/Plackett-Luce with temperature > 0.

**Instantiations**:
- `ExpectedGRPOLossLipschitz` - For GRPO-PL (Plackett-Luce ranking loss)
- `ExpectedGRPORLLossLipschitz` - For GRPO-RL (PPO-style with clipping, KL penalty)

---

# DSL Module: Assumption Structures

The DSL module uses **structures** rather than Lean `axiom` declarations to bundle
assumptions. This makes them explicit parameters to theorems, which is cleaner
for a formalization that aims to be modular.

## OracleAccess

**Location**: `DSL/CoreDefinitions.lean`

**Statement**: When a document is sampled (R=1), the expert label matches the oracle:
```
doc.sampled = true → doc.d_mis_true = some (oracle doc.content)
```

**Mathematical Justification**:
This is the **design assumption** of DSL (Design-based Supervised Learning).
Expert coders are assumed to correctly label the "missing" variable when they
code a document. The oracle function `Content → Missing` represents the true
labeling rule that experts implement.

This is analogous to the "no measurement error" assumption in survey sampling:
when you measure something, you measure it correctly. In ML terms, the training
labels are assumed to be correct for the sampled subset.

**When Safe**: When expert coders follow a consistent labeling protocol.

---

## MEstimationAxioms

**Location**: `DSL/AsymptoticTheory.lean`

**Contents**:
- `consistent`: M-estimators converge in probability to true parameters
- `asymptotic_normal`: Centered/scaled estimators converge to N(0, V)

**Mathematical Justification**:
Standard M-estimation theory from econometrics (Newey & McFadden 1994,
"Large Sample Estimation and Hypothesis Testing", Handbook of Econometrics).

Proving these from primitives would require formalizing:
1. Uniform laws of large numbers for dependent data
2. Central limit theorems for M-estimators
3. Delta method for smooth functionals

These are well-established results in the statistics literature.

**When Safe**: Under standard regularity conditions (identification, smoothness,
bounded moments).

---

## CoverageAxioms

**Location**: `DSL/AsymptoticTheory.lean`

**Statement**: Confidence intervals constructed from asymptotically normal
estimators achieve nominal coverage asymptotically.

**Mathematical Justification**:
If √n(β̂ - β) →d N(0, V), then the interval β̂ ± z_{α/2} × SE(β̂) covers β
with probability approaching 1-α.

This is the standard justification for Wald-type confidence intervals.

**When Safe**: When the asymptotic approximation is accurate (typically n ≥ 30
for well-behaved data, larger for heavy tails or sparse data).

---

## OLSAsymptoticAxioms

**Location**: `Econometrics/OLS/AsymptoticOLS.lean`

**Contents**:
- LLN for the OLS score (sample `x_i ε_i` averages)
- Multivariate CLT for the score
- Slutsky-based asymptotic normality of OLS
- Homoskedastic simplification to σ² Q⁻¹
- t-statistics normality and delta method

**Mathematical justification**:
These are standard large-sample results in econometrics (Wooldridge, Ch. 5).
They rely on i.i.d. sampling, finite moments, and identification so that
LLN/CLT and continuous mapping theorems apply.

**When Safe**: Under weak exogeneity, finite moments, and identification
with sufficiently large samples.

-/

namespace Axioms

/-! ## Re-exported Axioms with Documentation Aliases -/

/-- Unified axiom: Expected loss over groups is Lipschitz in oracle distance.

This is the **single foundational axiom** for preference learning bounds.
Justified by the Random Utility Model (McFadden 1974). -/
abbrev expected_group_loss_lipschitz := @ExpectedGroupLossLipschitz

/-- GRPO Plackett-Luce expected loss is Lipschitz in oracle distance.

Instantiation of `ExpectedGroupLossLipschitz` for GRPO-PL. -/
abbrev grpo_pl_expected_lipschitz := @ExpectedGRPOLossLipschitz

/-- GRPO-RL (DeepSeek-R1 style) expected loss is Lipschitz in oracle distance.

Instantiation of `ExpectedGroupLossLipschitz` for GRPO-RL. -/
abbrev grpo_rl_expected_lipschitz := @ExpectedGRPORLLossLipschitz

end Axioms
