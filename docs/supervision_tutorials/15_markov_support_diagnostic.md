# 15 Markov Support Diagnostic

This is the support/ESS story in the harder Markov setting.

Goal:
- keep the true document target fixed at `1.0`
- sample one block per document with increasingly concentrated policies
- track calibration and effective sample size as support degrades

Run it:

```bash
venv/bin/python scripts/tutorial_supervision_15_markov_support_diagnostic.py
```

Code:
- [tutorial_supervision_15_markov_support_diagnostic.py](/home/mlinegar/ThinkingTrees/scripts/tutorial_supervision_15_markov_support_diagnostic.py)
