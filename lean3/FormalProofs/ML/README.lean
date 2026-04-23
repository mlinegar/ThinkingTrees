/-!
# ML Module: Supervised Learning Foundations

## Overview

This module provides a lightweight, textbook-style formalization of core
machine learning concepts:
- Supervised learning primitives (hypotheses, losses, risk)
- Empirical risk minimization (ERM)
- Decision trees as a hypothesis class (routing, impurity, pruning, axis-aligned splits)
- Neural-operator interfaces used by the C-TreePO paper

The intent is to supply reusable foundations for DSL and OPT without
overcommitting to a specific algorithmic stack.

## File Structure

```
ML/
├── Core.lean          # Hypotheses, loss, population risk
├── ERM.lean           # Empirical risk + ERM definitions
├── DecisionTree.lean  # Decision tree hypothesis class
├── NeuralOperatorCore.lean
│                     # Discretizations, refinements, compact realized-call sets
├── NeuralOperatorApproximation.lean
│                     # Uniform / L² approximation interfaces on compact call sets
├── NeuralOperatorArchitecture.lean
│                     # ThinkingTrees aliases to FormalProbability architecture core
├── FNOFormalization.lean
│                     # Direct-import bridge to FormalProbability FNO theorem routes
└── README.lean        # This file
```

`NeuralOperatorArchitecture.lean` reuses
`FormalProbability.ML.NeuralOperatorArchitectureCore` for the common
equation-(6), IO/NO/NOm, GNO, LNO, and MGNO architecture surfaces.  This keeps
the non-FNO neural-operator classes explicit while avoiding a forked
ThinkingTrees copy of the same definitions.

`FNOFormalization.lean` imports the low-dependency
`FormalProbability.ML.NeuralOperatorFNOCore` module.  That core contains the
Mathlib-backed Fourier/FNO routes and reuses the shared architecture core
without pulling in FormalProbability's C-TreePO or ERM modules, so it is safe to
re-export from the `FormalProofs.ML` umbrella.

## Design Principles

- Keep definitions minimal and composable.
- Prefer explicit assumptions to global axioms.
- Align with Mathlib probability infrastructure where possible.
- Use ASCII-only notation for portability.
-/
