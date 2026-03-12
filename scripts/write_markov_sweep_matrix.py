#!/usr/bin/env python3
"""Write a machine-readable Markov sweep matrix and fairness policy."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List


def _parse_items(text: str) -> List[str]:
    out: List[str] = []
    for raw in str(text).replace(",", " ").split():
        item = raw.strip()
        if item:
            out.append(item)
    return out


def _parse_ints(text: str) -> List[int]:
    return [int(x) for x in _parse_items(text)]


def _parse_floats(text: str) -> List[float]:
    return [float(x) for x in _parse_items(text)]


def _parse_bools(text: str) -> List[bool]:
    mapping = {"1": True, "true": True, "t": True, "yes": True, "y": True, "0": False, "false": False, "f": False, "no": False, "n": False}
    out: List[bool] = []
    for item in _parse_items(text):
        key = item.lower()
        if key not in mapping:
            raise ValueError(f"bad boolean value: {item!r}")
        out.append(mapping[key])
    return out


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Write Markov sweep matrix spec.")
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-markdown", type=Path, required=True)
    parser.add_argument("--train-docs", type=str, required=True)
    parser.add_argument("--fixed-leaf-tokens", type=str, required=True)
    parser.add_argument("--model-families", type=str, required=True)
    parser.add_argument("--audit-fractions", type=str, required=True)
    parser.add_argument("--leaf-query-rates", type=str, required=True)
    parser.add_argument("--include-root-query", type=str, required=True)
    parser.add_argument("--seeds", type=str, required=True)
    parser.add_argument("--test-docs", type=int, default=256)
    parser.add_argument("--feature-mode", type=str, default="full")
    parser.add_argument("--n-epochs", type=int, default=10)
    parser.add_argument("--device", type=str, default="auto")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    swept_knobs = {
        "train_docs": _parse_ints(args.train_docs),
        "fixed_leaf_tokens": _parse_ints(args.fixed_leaf_tokens),
        "model_family": _parse_items(args.model_families),
        "audit_fraction": _parse_floats(args.audit_fractions),
        "leaf_query_rate": _parse_floats(args.leaf_query_rates),
        "include_root_query": _parse_bools(args.include_root_query),
        "seed": _parse_ints(args.seeds),
    }
    knob_roles = {
        "train_docs": {
            "group": "support_scale",
            "meaning": "Number of training documents used to fit the learned merger.",
            "comparison_note": "Changes total available repeated examples. Must be matched when asking equal-information questions.",
        },
        "fixed_leaf_tokens": {
            "group": "geometry",
            "meaning": "Approximate leaf span size used to build the balanced tree.",
            "comparison_note": "Changes the number of leaves and internal nodes per document, so nominal audit fractions are not directly comparable across leaf sizes.",
        },
        "model_family": {
            "group": "learner",
            "meaning": "Structured additive control or unstructured neural merger.",
            "comparison_note": "Controls the hypothesis class rather than the oracle budget.",
        },
        "audit_fraction": {
            "group": "oracle_access",
            "meaning": "Fraction of internal nodes labeled for merge supervision during training.",
            "comparison_note": "Maps to realized internal-label coverage after tree geometry is fixed.",
        },
        "leaf_query_rate": {
            "group": "oracle_access",
            "meaning": "Fraction of leaves labeled during training.",
            "comparison_note": "Maps to realized leaf-label coverage after tree geometry is fixed.",
        },
        "include_root_query": {
            "group": "oracle_access",
            "meaning": "Whether each training document also exposes one direct root oracle label.",
            "comparison_note": "Root-only access is not the same information type as local supervision.",
        },
        "seed": {
            "group": "randomization",
            "meaning": "Random seed for document generation, tree sampling, and training.",
            "comparison_note": "Aggregate across seeds before reading trends.",
        },
    }
    fixed_knobs = {
        "test_docs": int(args.test_docs),
        "feature_mode": str(args.feature_mode),
        "n_epochs": int(args.n_epochs),
        "device": str(args.device),
    }
    derived_metrics = {
        "leaf_label_coverage": "Realized labeled-leaf fraction, derived from mean_leaf_labels / mean_leaves.",
        "internal_label_coverage": "Realized labeled-internal-node fraction, derived from mean_internal_labels / mean_internal_nodes.",
        "root_query_rate": "Realized root labels per training document.",
        "local_oracle_coverage": "Fraction of local tree nodes (leaves + internal) that receive labels.",
        "local_undersupport": "1 - local_oracle_coverage.",
        "mean_queries_per_doc": "Average total oracle labels per document, mixing leaf, internal, and root access.",
        "info_signature": "Compact signature of realized leaf/internal/root access used for comparison grouping.",
    }
    matrix = {
        "swept_knobs": swept_knobs,
        "knob_roles": knob_roles,
        "fixed_knobs": fixed_knobs,
        "derived_metrics": derived_metrics,
        "fair_comparison_policy": {
            "equal_information_requires_same": [
                "fixed_leaf_tokens",
                "train_docs",
                "leaf_label_coverage",
                "internal_label_coverage",
                "root_query_rate",
            ],
            "same_geometry_requires_same": [
                "fixed_leaf_tokens",
                "train_docs",
            ],
            "same_local_information_requires_same": [
                "leaf_label_coverage",
                "internal_label_coverage",
            ],
            "same_total_information_proxy_requires_same": [
                "mean_queries_per_doc",
            ],
            "same_total_information_but_different_type_is_not_equal_information": True,
            "notes": [
                "Compare within the same tree geometry first: same leaf size and train-doc count.",
                "Then match realized information type: leaf coverage, internal coverage, and root access separately.",
                "A run with high root access but zero local coverage is an oracle-access ablation, not an equal-information comparator for a local-merge learner.",
            ],
        },
        "recommended_views": [
            {
                "name": "support_surface_within_geometry",
                "fix": ["fixed_leaf_tokens", "train_docs", "model_family"],
                "sweep": ["leaf_label_coverage", "internal_label_coverage"],
                "facet": ["root_query_rate"],
                "purpose": "Show how local support changes recovery when the geometry is fixed.",
            },
            {
                "name": "doc_scale_trend",
                "fix": ["fixed_leaf_tokens", "model_family", "leaf_label_coverage", "internal_label_coverage", "root_query_rate"],
                "sweep": ["train_docs"],
                "purpose": "Show whether repeated examples improve recovery once information type is held fixed.",
            },
            {
                "name": "oracle_access_pattern_ablation",
                "fix": ["fixed_leaf_tokens", "train_docs", "model_family"],
                "sweep": ["info_signature"],
                "purpose": "Contrast root-only, leaf-only, merge-only, and mixed supervision without calling them equal-information.",
            },
        ],
    }
    counts = [len(v) for v in matrix["swept_knobs"].values()]
    total = 1
    for n in counts:
        total *= n
    matrix["n_total_combos"] = int(total)
    matrix["n_unique_nonseed_combos"] = int(total // max(1, len(swept_knobs["seed"])))

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_markdown.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(matrix, indent=2), encoding="utf-8")

    lines = [
        "# Markov Sweep Matrix",
        "",
        f"- Total combos: `{matrix['n_total_combos']}`",
        "",
        "## Swept Knobs",
        "",
    ]
    for key, values in matrix["swept_knobs"].items():
        lines.append(f"- `{key}`: `{values}`")
        role = matrix["knob_roles"].get(key, {})
        lines.append(f"  role: {role.get('group', 'unknown')}")
        lines.append(f"  meaning: {role.get('meaning', '')}")
        lines.append(f"  comparison note: {role.get('comparison_note', '')}")
    lines.extend(
        [
            "",
            "## Fixed Knobs",
            "",
        ]
    )
    for key, value in matrix["fixed_knobs"].items():
        lines.append(f"- `{key}`: `{value}`")
    lines.extend(
        [
            "",
            "## Derived Information Metrics",
            "",
        ]
    )
    for key, value in matrix["derived_metrics"].items():
        lines.append(f"- `{key}`: {value}")
    lines.extend(
        [
            "",
            "## Fair Comparison Policy",
            "",
            "- Equal-information comparisons require matching:",
        ]
    )
    for key in matrix["fair_comparison_policy"]["equal_information_requires_same"]:
        lines.append(f"  - `{key}`")
    lines.extend(
        [
            "",
            "- Same-geometry comparisons require matching:",
        ]
    )
    for key in matrix["fair_comparison_policy"]["same_geometry_requires_same"]:
        lines.append(f"  - `{key}`")
    lines.extend(
        [
            "",
            "- Same-local-information comparisons require matching:",
        ]
    )
    for key in matrix["fair_comparison_policy"]["same_local_information_requires_same"]:
        lines.append(f"  - `{key}`")
    lines.extend(
        [
            "",
            "- Same total information but different information type is not treated as a fair comparator.",
            "- A root-only run is therefore not an equal-information comparator for a local-merge run with no root access.",
            "",
            "## Recommended Comparison Views",
            "",
        ]
    )
    for view in matrix["recommended_views"]:
        lines.append(f"- `{view['name']}`")
        lines.append(f"  fix: `{view.get('fix', [])}`")
        lines.append(f"  sweep: `{view.get('sweep', [])}`")
        if view.get("facet"):
            lines.append(f"  facet: `{view.get('facet', [])}`")
        lines.append(f"  purpose: {view.get('purpose', '')}")
    lines.extend(
        [
            "",
            "## Combo Counts",
            "",
            f"- Total combos including seeds: `{matrix['n_total_combos']}`",
            f"- Unique non-seed combos: `{matrix['n_unique_nonseed_combos']}`",
            "",
        ]
    )
    args.output_markdown.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"output_json": str(args.output_json), "output_markdown": str(args.output_markdown), "n_total_combos": int(total)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
