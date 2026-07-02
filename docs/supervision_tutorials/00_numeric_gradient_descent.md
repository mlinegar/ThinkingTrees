# 00 Numeric Gradient Descent

This is the absolute floor: a one-dimensional numeric problem with an obvious
target rule.

If you want the even more stripped-down version first, start with
[00 Manual Gradient Descent](./00_manual_gradient_descent.md).
If you want the local-label version first, see
[00 Same Average, Local Variation](./00_same_average_local_variation.md).

Goal:
- use the supervision surface on a tiny dataset where the target is `y = 3x + 1`
- fit the same data a few different ways on CPU
- make it visually obvious that the abstraction is just “examples -> supervision dataset -> optimizer”

Run it:

```bash
venv/bin/python scripts/tutorial_supervision_00_numeric_gradient_descent.py
```

What it does:
- creates five rows with one feature each
- builds one `SupervisionDataset`
- fits three models on exactly the same supervision object:
  - closed-form ridge
  - SGD linear regressor
  - SGD MLP regressor

Why this matters:
- this is the clearest possible version of the training surface
- gradient descent is not special-cased; it just consumes the same scalar supervision rows
- more complex settings later are still the same pattern

Key API objects:
- `DenseSupervisionExample`
- `SupervisionDataset`
- `fit_dense_scalar_ridge_regressor`
- `fit_dense_scalar_regressor`

Code:
- [tutorial_supervision_00_numeric_gradient_descent.py](/home/mlinegar/ThinkingTrees/scripts/tutorial_supervision_00_numeric_gradient_descent.py)

Next:
- continue to [01 Dense Scalar Regression](./01_dense_scalar_regression.md), which uses the same surface but with a slightly less toy two-feature problem.
