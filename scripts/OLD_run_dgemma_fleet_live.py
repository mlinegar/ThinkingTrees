# OLD_: archived 2026-07-02; depends on treepo._research, removed in the treepo 2026-06 minimization. Kept for reference; do not import or run.
#!/usr/bin/env python
"""Live dgemma runs across leaf sizes, fanned across a 4-GPU vLLM fleet.

dgemma is consumed through the STANDARD OpenAI `/v1/chat/completions` transport
(`treepo.llm.diffusion` `engine="openai"`), round-robin across the fleet
endpoints so every GPU is used. Writes per-leaf-size JSON + a summary.

Usage:
  run_dgemma_fleet_live.py --ports 8004 8005 8006 8007 \
      --model RedHatAI/diffusiongemma-26B-A4B-it-NVFP4 \
      --leaf-sizes 2 4 8 --out OUTDIR
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import List

import treepo.methods as M


def _make_text_trees(n_leaves: int, n_trees: int = 8):
    from treepo._research.tree.labeled import LabeledNode, LabeledTree

    trees = []
    for i in range(n_trees):
        doc_id = f"doc_{n_leaves}_{i}"
        base = 3.0 + 0.4 * i
        leaf_rows = []
        for j in range(n_leaves):
            text = f"{doc_id} leaf{j}: policy evidence on taxation welfare investment jobs {j}"
            score = base + 0.05 * (j - n_leaves / 2.0)
            leaf_rows.append((f"l0_{j}", text, score))
        root_text = " ".join(t for _, t, _ in leaf_rows)
        tree = LabeledTree(
            doc_id=doc_id,
            document_text=root_text,
            document_score=base,
            metadata={
                "split": "test",
                "teacher_score_1_7": base,
                "expert_score_1_7": base,
                "observed": True,
                "propensity": 1.0,
            },
            label_source="test",
        )
        for nid, text, score in leaf_rows:
            tree.add_node(LabeledNode(node_id=nid, doc_id=doc_id, level=0, text=text, score=score))
        current = list(leaf_rows)
        level = 1
        while len(current) > 1:
            nxt = []
            for k in range(0, len(current), 2):
                lid, lt, ls = current[k]
                rid, rt, rs = current[k + 1]
                text = f"{lt} {rt}"
                score = (ls + rs) / 2.0
                nid = "root" if len(current) == 2 else f"l{level}_{k // 2}"
                tree.add_node(
                    LabeledNode(node_id=nid, doc_id=doc_id, level=level, text=text,
                                score=score, left_child_id=lid, right_child_id=rid)
                )
                nxt.append((nid, text, score))
            current = nxt
            level += 1
        trees.append(tree)
    return trees


def _pearson(xs, ys) -> float | None:
    pairs = [(x, y) for x, y in zip(xs, ys) if x is not None and y is not None and math.isfinite(x) and math.isfinite(y)]
    if len(pairs) < 2:
        return None
    xs2 = [p[0] for p in pairs]; ys2 = [p[1] for p in pairs]
    mx = sum(xs2) / len(xs2); my = sum(ys2) / len(ys2)
    sxy = sum((x - mx) * (y - my) for x, y in pairs)
    sxx = sum((x - mx) ** 2 for x in xs2); syy = sum((y - my) ** 2 for y in ys2)
    if sxx <= 0 or syy <= 0:
        return None
    return sxy / math.sqrt(sxx * syy)


def run_diffusion(base_urls: List[str], model: str, leaf_sizes: List[int], out: Path) -> dict:
    results = {}
    for n in leaf_sizes:
        trees = _make_text_trees(n)
        cfg = {
            "diffusion_config": {
                "backend": {
                    "engine": "openai",
                    "base_urls": base_urls,
                    "model": model,
                    "max_concurrency": max(4, len(base_urls) * 2),
                    "timeout": 180.0,
                },
                "prompt_template": (
                    "Rate the economic-left content of the document on a 1-7 scale "
                    "(1=strongly right, 7=strongly left). Think briefly, then end with "
                    "'Score: N'.\n\n{text}\n\nScore:"
                ),
                "sampling_params": {"temperature": 0.0, "max_tokens": 256},
                "score_regex": r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)",
                "min_score": 1.0, "max_score": 7.0,
            }
        }
        res = M.run("fit", {
            "family": "diffusion",
            "eval_data": list(trees),
            "backend_config": cfg,
            "axis": {"max_iterations": 0, "axis_value": 0},
        })
        metrics = dict(getattr(res, "metrics", {}) or {})
        cell = {
            "leaf_size": n, "n_trees": len(trees), "status": getattr(res, "status", None),
            "metrics": {k: v for k, v in metrics.items() if isinstance(v, (int, float))},
        }
        (out / f"diffusion_leaf{n}.json").write_text(json.dumps(cell, indent=2, default=str))
        results[str(n)] = cell
        print(f"[diffusion leaf={n}] status={cell['status']} metrics_keys={sorted(cell['metrics'])[:6]}")
    return results


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ports", nargs="+", type=int, default=[8004, 8005, 8006, 8007])
    ap.add_argument("--host", default="localhost")
    ap.add_argument("--model", default="RedHatAI/diffusiongemma-26B-A4B-it-NVFP4")
    ap.add_argument("--leaf-sizes", nargs="+", type=int, default=[2, 4, 8])
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    base_urls = [f"http://{args.host}:{p}/v1" for p in args.ports]
    print(f"fleet endpoints: {base_urls}")

    summary = {"model": args.model, "fleet": base_urls, "leaf_sizes": args.leaf_sizes}
    try:
        summary["diffusion"] = run_diffusion(base_urls, args.model, args.leaf_sizes, out)
    except Exception as exc:  # noqa: BLE001
        summary["diffusion_error"] = f"{type(exc).__name__}: {exc}"
        print("diffusion run error:", summary["diffusion_error"])

    (out / "fleet_live_summary.json").write_text(json.dumps(summary, indent=2, default=str))
    print(f"DONE — summary at {out / 'fleet_live_summary.json'}")


if __name__ == "__main__":
    main()
