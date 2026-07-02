# 00 Manual Gradient Descent

This is the simplest possible walkthrough: one feature, one scalar target, and
hand-written gradient descent.

Goal:
- fit the tiny rule `y = 3x + 1` with explicit numeric updates
- then fit the exact same rows through the shared `SupervisionDataset` API
- make it obvious that the “real” training surface is just a structured version
  of the same examples

Run it:

```bash
venv/bin/python scripts/tutorial_supervision_00_manual_gradient_descent.py
```

What it does:
- creates five points on a line
- runs manual gradient descent on `w` and `b`
- builds the same rows as one `SupervisionDataset`
- runs the shared SGD linear regressor on the same data

Why this matters:
- there is no hidden extra machinery in the simplest case
- the supervision layer is just the canonical way to package the same examples
- later tutorials add complexity without changing this basic pattern

Key idea:
- manual path: `examples -> gradients -> parameter updates`
- shared path: `examples -> SupervisionDataset -> optimizer backend`

Code:
- [tutorial_supervision_00_manual_gradient_descent.py](/home/mlinegar/ThinkingTrees/scripts/tutorial_supervision_00_manual_gradient_descent.py)

Next:
- continue to [00 Same Average, Local Variation](./00_same_average_local_variation.md), which adds local variation while keeping the global average fixed.
