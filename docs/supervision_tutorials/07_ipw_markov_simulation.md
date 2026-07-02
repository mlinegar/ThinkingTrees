# 07 Markov IPW Simulation

This is the harder setting: the document is a short Markov sequence and the
local labels are block-level state-1 fractions.

Goal:
- keep the document-level target fixed at `1.0`
- define local block labels by centering raw block fractions around that target
- sample blocks with probabilities that favor high local state-1 fractions
- compare repeated-trial regression fits with and without IPW

Run it:

```bash
venv/bin/python scripts/tutorial_supervision_07_ipw_markov_simulation.py
```

What it shows:
- the same IPW logic still applies in a nontrivial data-generating process
- naive sampled supervision pulls predictions upward from `1.0`
- IPW-weighted supervision is better calibrated against the true target `1.0`

Code:
- [tutorial_supervision_07_ipw_markov_simulation.py](/home/mlinegar/ThinkingTrees/scripts/tutorial_supervision_07_ipw_markov_simulation.py)
