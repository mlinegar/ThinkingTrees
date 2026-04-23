#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.ctreepo.sim.suite.learnability_policy import (  # noqa: E402
    resolve_identifiable_zero_learnability_policy,
)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Emit the shared identifiable-zero learnability policy."
    )
    p.add_argument("--format", choices=["json", "shell"], default="json")
    p.add_argument("--train-docs-grid", type=str, default=None)
    p.add_argument("--label-rate-grid", type=str, default=None)
    p.add_argument("--heldout-docs", type=int, default=None)
    p.add_argument("--base-seeds", type=str, default=None)
    p.add_argument("--hero-seeds", type=str, default=None)
    p.add_argument("--ctree-eval-guidance-rates", type=str, default=None)
    return p.parse_args()


def main() -> int:
    args = _parse_args()
    policy = resolve_identifiable_zero_learnability_policy(
        train_docs_grid=args.train_docs_grid,
        label_rate_grid=args.label_rate_grid,
        heldout_docs=args.heldout_docs,
        base_seeds=args.base_seeds,
        hero_seeds=args.hero_seeds,
        ctree_eval_guidance_rates=args.ctree_eval_guidance_rates,
    )
    if str(args.format) == "shell":
        for key, value in policy.to_shell_exports().items():
            print(f"{key}='{value}'")
        return 0
    print(json.dumps(policy.to_dict(), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
