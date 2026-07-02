# OLD_treepo_cdx

Archived 2026-06-25 during the local-law single-path migration. No live source
imports referenced `treepo_cdx`; canonical TreePO/C-TreePO local-law and
objective code now lives in `/home/mlinegar/treepo`.

Original README follows.

# treepo_cdx

Parallel implementation lane for the TreePO/C-TreePO package spine.

This package is intentionally dependency-light. It starts with the theorem-facing
contracts, audit/certificate primitives, sampling and honesty helpers, and a
single `fit()` facade. The facade is thin: when run inside the full monorepo it
dispatches to the already-working `treepo-bench`, runtime, and `src.ctreepo`
learning lanes instead of copying backend logic.

Minimal smoke:

```bash
cd treepo_cdx
../venv/bin/python -m pytest
```

Minimal package-native fits:

```python
from treepo_cdx import fit

fit({"mode": "local_law", "local_law_rows": [...]}, output_dir="outputs/local_law")
fit({"mode": "hll_sketch", "leaf_token_lists": [[1, 2, 3], [3, 4]]}, output_dir="outputs/hll")
```

The current implementation plan is in
[`docs/implementation_plan.md`](docs/implementation_plan.md).
