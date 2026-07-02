# 11 Online Query Loop

This is the smallest online-learning version of the same abstraction.

Goal:
- pick a local unit with a fixed query policy
- log its propensity
- query the oracle
- update naive, Horvitz-Thompson, and self-normalized IPW estimates
- then reuse the logged queries as offline supervision

Run it:

```bash
venv/bin/python scripts/tutorial_supervision_11_online_query_loop.py
```

What it shows:
- online querying and offline logged-data reuse are the same supervision object
- the only extra ingredient is the time order of when labels arrive

Code:
- [tutorial_supervision_11_online_query_loop.py](/home/mlinegar/ThinkingTrees/scripts/tutorial_supervision_11_online_query_loop.py)
