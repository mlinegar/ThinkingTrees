# 06 IPW Regression Simulation

This moves from one estimate to a learned predictor.

Goal:
- keep the true document target fixed at `1.0`
- give each document several local labels around that target
- sample one local label per document from a biased distribution
- compare repeated-trial regression fits:
  - naive sampled regression
  - IPW-weighted sampled regression
  - full-document reference regression

Run it:

```bash
venv/bin/python scripts/tutorial_supervision_06_ipw_regression_simulation.py
```

What it shows:
- naive sampled regression drifts above the true constant target
- IPW-weighted regression moves the fitted intercept back toward `1.0`
- this is still the same canonical supervision surface; only the logged
  `sampling` changes

Code:
- [tutorial_supervision_06_ipw_regression_simulation.py](/home/mlinegar/ThinkingTrees/scripts/tutorial_supervision_06_ipw_regression_simulation.py)
