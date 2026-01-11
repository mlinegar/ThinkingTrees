/-!
# ML Module: Supervised Learning Foundations

## Overview

This module provides a lightweight, textbook-style formalization of core
machine learning concepts:
- Supervised learning primitives (hypotheses, losses, risk)
- Empirical risk minimization (ERM)
- Decision trees as a hypothesis class (routing, impurity, pruning, axis-aligned splits)

The intent is to supply reusable foundations for DSL and OPT without
overcommitting to a specific algorithmic stack.

## File Structure

```
ML/
├── Core.lean          # Hypotheses, loss, population risk
├── ERM.lean           # Empirical risk + ERM definitions
├── DecisionTree.lean  # Decision tree hypothesis class
└── README.lean        # This file
```

## Design Principles

- Keep definitions minimal and composable.
- Prefer explicit assumptions to global axioms.
- Align with Mathlib probability infrastructure where possible.
- Use ASCII-only notation for portability.
-/
