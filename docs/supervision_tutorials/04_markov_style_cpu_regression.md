# 04 Markov-Style CPU Regression

This is the first example that looks like a simplified simulation setting
instead of a toy regression table.

Goal:
- generate tiny synthetic two-state Markov sequences
- featurize each document by normalized transition counts
- predict a document-level target with the shared supervision surface

Run it:

```bash
venv/bin/python scripts/tutorial_supervision_04_markov_style.py
```

What it does:
- samples small Markov sequences on CPU
- turns each sequence into a dense feature vector `[p00, p01, p10, p11]`
- sets the target to the fraction of state-1 tokens in the sequence
- builds a `SupervisionDataset`
- fits `fit_dense_scalar_ridge_regressor(...)`
- reports a tiny holdout MAE

Why this matters:
- this is the same pattern as the simple document-level simulation baselines
- the supervision/data contract is unchanged
- only the feature map and target semantics differ

Key API objects:
- `DenseSupervisionExample`
- `build_dense_full_document_supervision_dataset`
- `fit_dense_scalar_ridge_regressor`

Code:
- [tutorial_supervision_04_markov_style.py](/home/mlinegar/ThinkingTrees/scripts/tutorial_supervision_04_markov_style.py)

After this:
- the next step is usually to swap the dense feature map for a richer representation or replace ridge with another optimizer family, while keeping the same supervision objects.
