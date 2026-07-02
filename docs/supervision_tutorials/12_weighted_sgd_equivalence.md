# 12 Weighted SGD Equivalence

This makes one optimization fact explicit:
weighted SGD is just SGD on a weighted loss, and integer weights behave like row duplication.

Goal:
- fit one tiny linear problem with weighted SGD
- fit the same problem with duplicated rows and unweighted SGD
- compare both against a weighted ridge reference

Run it:

```bash
venv/bin/python scripts/tutorial_supervision_12_weighted_sgd_equivalence.py
```

Code:
- [tutorial_supervision_12_weighted_sgd_equivalence.py](/home/mlinegar/ThinkingTrees/scripts/tutorial_supervision_12_weighted_sgd_equivalence.py)
