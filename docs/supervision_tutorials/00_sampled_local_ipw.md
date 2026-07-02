# 00 Sampled Local Labels With IPW

This is the smallest example of the propensity-aware case:
we only observe a biased subset of local labels, we log the inclusion
probabilities, and we correct with IPW.

Goal:
- keep the same document-level target `3x + 1`
- let local pieces differ around that target
- only observe a biased subset of the local pieces
- show the difference between:
  - naive averaging / naive regression
  - IPW-corrected averaging / IPW-weighted regression

Run it:

```bash
venv/bin/python scripts/tutorial_supervision_00_sampled_local_ipw.py
```

Setup:
- full local offsets are `[-1.5, -0.5, 0.5, 1.5]`
- we only keep two observed local pieces per document:
  - offset `-0.5` with logged propensity `0.25`
  - offset `+1.5` with logged propensity `0.75`

So the observed sample is biased upward.
The naive sample mean is therefore too high.

The correction used in the script is:

```text
normalized_ipw_mean = sum(observed_target / propensity) / sum(1 / propensity)
```

Because the logged propensities differ, the two observed pieces receive
different weights:
- propensity `0.25` -> weight `4.0`
- propensity `0.75` -> weight `1.3333...`

What the script shows:
- per document:
  - true document target
  - all local targets
  - observed sampled local targets
  - naive sample mean
  - normalized IPW mean
- across documents:
  - a reference fit from full-document labels
  - a naive fit from sampled local labels
  - an IPW-weighted fit from the same sampled local labels

Why this matters:
- this is the clearest possible bridge from local-label sampling to unbiased
  risk accounting
- it makes the role of logged propensities and `sample_weight = 1 / propensity`
  explicit
- it is the same supervision surface as the other tutorials, just with
  nontrivial `sampling`

Code:
- [tutorial_supervision_00_sampled_local_ipw.py](/home/mlinegar/ThinkingTrees/scripts/tutorial_supervision_00_sampled_local_ipw.py)

Next:
- continue to [00 Numeric Gradient Descent](./00_numeric_gradient_descent.md) for the same shared surface in a pure dense-optimization view.
