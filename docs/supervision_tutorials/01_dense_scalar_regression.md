# 01 Dense Scalar Regression

This is the smallest useful example of the unified API.

Goal:
- create a `SupervisionDataset` with dense numeric features
- fit a tiny closed-form ridge regressor on CPU
- make predictions without touching any pairwise code

Run it:

```bash
venv/bin/python scripts/tutorial_supervision_01_dense_scalar_regression.py
```

What it does:
- builds a few `DenseSupervisionExample` rows
- converts them with `build_dense_full_document_supervision_dataset(...)`
- fits `fit_dense_scalar_ridge_regressor(...)`
- predicts on two held-out feature vectors

Why this matters:
- one attempt is already a valid supervision event
- scalar regression is first-class
- the same `SupervisionDataset` surface later supports comparative judgments and human preferences

Key API objects:
- `DenseSupervisionExample`
- `SupervisionDataset`
- `fit_dense_scalar_ridge_regressor`

Code:
- [tutorial_supervision_01_dense_scalar_regression.py](/home/mlinegar/ThinkingTrees/scripts/tutorial_supervision_01_dense_scalar_regression.py)

Next:
- move to [02 Grouped Comparative Supervision](./02_grouped_comparative_supervision.md), where several attempts for one example become one comparative record and an internal binary projection.
