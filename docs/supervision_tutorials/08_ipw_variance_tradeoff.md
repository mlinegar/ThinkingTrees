# 08 IPW Variance Tradeoff

This is the deeper follow-up: keep the true target fixed at `1.0`, then vary
how skewed the sampling design is and compare several estimators.

Goal:
- hold the true estimand fixed so the comparisons are easy to read
- increase the propensity skew gradually
- compare four estimators:
  - naive
  - Horvitz-Thompson / IPW
  - self-normalized IPW
  - clipped self-normalized IPW
- show both one-draw and four-draw regimes

Run it:

```bash
venv/bin/python scripts/tutorial_supervision_08_ipw_variance_tradeoff.py
```

What it shows:
- as propensity skew increases, naive bias gets worse
- Horvitz-Thompson stays centered but its variance grows
- self-normalized IPW changes the bias/variance mix; it is not just “strictly better HT”
- clipping can reduce the variance of self-normalized IPW, but it adds more bias

Important detail:
- with only one draw per trial, self-normalized IPW and clipped self-normalized
  IPW collapse back to the naive estimator, because the normalization cancels the
  weight

The point is not that one estimator always dominates.
The point is that logged propensities let you choose the bias/variance tradeoff
explicitly instead of being stuck with hidden selection bias.

Code:
- [tutorial_supervision_08_ipw_variance_tradeoff.py](/home/mlinegar/ThinkingTrees/scripts/tutorial_supervision_08_ipw_variance_tradeoff.py)
