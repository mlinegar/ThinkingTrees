# 05 Easy IPW Mean Simulation

This is the easiest repeated-trials simulation for the IPW idea.

Goal:
- keep a fixed document target
- sample exactly one local label per trial from a biased distribution
- compare the empirical behavior of:
  - naive estimation
  - Horvitz-Thompson / IPW estimation

Run it:

```bash
venv/bin/python scripts/tutorial_supervision_05_ipw_mean_simulation.py
```

What it shows:
- the observed local labels are biased upward
- the naive estimator inherits that bias
- the IPW estimator is centered correctly once we divide by the logged
  propensity

Key formula:

```text
horvitz_thompson_mean = observed_target / (num_units * propensity)
```

Code:
- [tutorial_supervision_05_ipw_mean_simulation.py](/home/mlinegar/ThinkingTrees/scripts/tutorial_supervision_05_ipw_mean_simulation.py)
