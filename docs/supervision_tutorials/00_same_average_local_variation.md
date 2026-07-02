# 00 Same Average, Local Variation

This is the next-simplest case after the hand-written line fit:
the local pieces move around, but the document-level average stays exactly the
same.

Goal:
- make local variation explicit without changing the global target
- show the same toy problem through two supervision channels:
  - `full_document_supervision`
  - `sampled_substructure_supervision`
- show that if the local deviations average to zero, both channels recover the
  same global rule

Run it:

```bash
venv/bin/python scripts/tutorial_supervision_00_same_average_local_variation.py
```

What it does:
- defines a document-level target `3x + 1`
- creates four local targets around each document target
- uses symmetric offsets whose mean is exactly zero
- fits one model from document-level labels
- fits another model from local sampled-substructure labels

Why this matters:
- this is the smallest example of “local pieces can vary while the overall
  quantity stays unchanged”
- it makes the bridge to local-law and sampled-label settings much more obvious
- it shows that the supervision surface does not require separate bespoke
  pipelines for global and local labels

Code:
- [tutorial_supervision_00_same_average_local_variation.py](/home/mlinegar/ThinkingTrees/scripts/tutorial_supervision_00_same_average_local_variation.py)

Next:
- continue to [00 Sampled Local Labels With IPW](./00_sampled_local_ipw.md), which observes only a biased subset of the local pieces and corrects with logged propensities.
