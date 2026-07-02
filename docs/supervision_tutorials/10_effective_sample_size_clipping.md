# 10 Effective Sample Size And Clipping

This isolates the variance side of IPW.

Goal:
- keep the true target fixed at `1.0`
- use a very skewed sampling design
- compare raw inverse weights to clipped weights
- make ESS concrete

Run it:

```bash
venv/bin/python scripts/tutorial_supervision_10_effective_sample_size_clipping.py
```

What it shows:
- raw inverse weights can produce a low effective sample size
- clipping raises ESS
- clipping changes the bias/variance tradeoff rather than “fixing everything”

Code:
- [tutorial_supervision_10_effective_sample_size_clipping.py](/home/mlinegar/ThinkingTrees/scripts/tutorial_supervision_10_effective_sample_size_clipping.py)
