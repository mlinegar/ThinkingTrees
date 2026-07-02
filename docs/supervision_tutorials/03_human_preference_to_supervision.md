# 03 Human Preference To Supervision

This example wires human input into the core API directly.

Goal:
- enqueue requests for human review
- submit human pairwise and scalar judgments
- export the completed results as canonical supervision

Run it:

```bash
venv/bin/python scripts/tutorial_supervision_03_human_preference.py
```

What it does:
- creates a `PreferenceStore`
- enqueues one pairwise request and one scalar request
- submits a human pairwise judgment with `submit_human_pairwise_preference(...)`
- submits a human scalar judgment with `submit_human_scalar_preference(...)`
- exports:
  - `store.to_supervision_dataset()`
  - `store.to_binary_projection_dataset()`

Why this matters:
- human input is not a side channel anymore
- the same store output can feed scalar learners, grouped comparative learners, or binary optimizer projections
- truth provenance stays attached as `human`

Key API objects:
- `PreferenceStore`
- `PreferenceResponse.from_human_pairwise_preference`
- `PreferenceResponse.from_human_scalar_preference`
- `PreferenceStore.to_supervision_dataset`

Code:
- [tutorial_supervision_03_human_preference.py](/home/mlinegar/ThinkingTrees/scripts/tutorial_supervision_03_human_preference.py)

Next:
- move to [04 Markov-Style CPU Regression](./04_markov_style_cpu_regression.md), which uses the same surface for a tiny synthetic local-law style problem.
