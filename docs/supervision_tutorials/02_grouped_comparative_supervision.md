# 02 Grouped Comparative Supervision

This example shows how several scored attempts for the same input become one
comparative judgment.

Goal:
- start from three human-like scalar scores
- convert them into a canonical `SupervisionDataset`
- derive a grouped comparative record
- derive a binary projection only when an optimizer needs one

Run it:

```bash
venv/bin/python scripts/tutorial_supervision_02_grouped_comparative.py
```

What it does:
- creates three `PreferenceRequest` objects for the same source example
- creates three `PreferenceResponse.from_human_scalar_preference(...)` responses
- converts each one to a `ResponseJudgment` with a distinct `response_id`
- builds a `SupervisionDataset`
- calls `to_comparative_dataset()`
- calls `project_binary(projection="adjacent")`

Why this matters:
- 1, 2, or 10 attempts all fit the same supervision surface
- binary preference is only a derived optimizer view
- scalar human scores are enough to induce a ranking when they are comparable

Key API objects:
- `PreferenceRequest`
- `PreferenceResponse.from_human_scalar_preference`
- `PreferenceResponse.to_response_judgment`
- `SupervisionDataset.to_comparative_dataset`
- `SupervisionDataset.project_binary`

Code:
- [tutorial_supervision_02_grouped_comparative.py](/home/mlinegar/ThinkingTrees/scripts/tutorial_supervision_02_grouped_comparative.py)

Next:
- move to [03 Human Preference To Supervision](./03_human_preference_to_supervision.md), where the same objects go through the review store and come back out as canonical supervision artifacts.
