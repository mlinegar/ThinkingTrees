# 09 Support Failure

This is the clean impossibility example.

Goal:
- keep the true target fixed at `1.0`
- set one local unit’s propensity to exactly `0`
- show that no IPW correction can recover the true target once support is gone

Run it:

```bash
venv/bin/python scripts/tutorial_supervision_09_support_failure.py
```

What it shows:
- naive is biased
- Horvitz-Thompson and self-normalized IPW also fail to recover `1.0`
- the reason is not “bad estimation,” it is lack of support

Code:
- [tutorial_supervision_09_support_failure.py](/home/mlinegar/ThinkingTrees/scripts/tutorial_supervision_09_support_failure.py)
