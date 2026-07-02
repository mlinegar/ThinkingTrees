#!/usr/bin/env python3
"""Build the unified local-law meta report across Markov and tree-relevant LDA."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Dict, Iterable, List, Sequence, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.ctreepo.sim.expectations import (
    ExpectationFinding,
    StructuredLocalLawAdapter,
    build_local_law_expectation_report,
)
from src.ctreepo.contracts import assert_public_contract_clean
from src.ctreepo.sim.manifest import read_manifest_jsonl


_PUBLIC_KEY_RENAMES = {
    "baseline_family": "method_id",
    "dgp": "problem_id",
    "families_scanned": "problem_ids_scanned",
    "families_with_failures": "problem_ids_with_failures",
    "family": "problem_id",
    "law_package": "law_set_id",
}


def _canonical_public_payload(value):
    if isinstance(value, dict):
        out = {}
        for raw_key, child in value.items():
            key = _PUBLIC_KEY_RENAMES.get(str(raw_key), str(raw_key))
            out[key] = _canonical_public_payload(child)
        return out
    if isinstance(value, list):
        return [_canonical_public_payload(item) for item in value]
    return value


def _run_pandoc(md_path: Path, pdf_path: Path) -> bool:
    if shutil.which("pandoc") is None or shutil.which("pdflatex") is None:
        return False
    try:
        subprocess.run(
            ["pandoc", str(md_path.name), "-o", str(pdf_path.name), "--pdf-engine=pdflatex"],
            cwd=str(md_path.parent),
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        return True
    except Exception:
        return False


def _collect_paths(input_root: Path | None, manifest_path: Path | None) -> List[Path]:
    out: List[Path] = []
    if input_root is not None and input_root.exists():
        out.extend(sorted(input_root.rglob("*.json")))
    if manifest_path is not None and manifest_path.exists():
        for run in read_manifest_jsonl(manifest_path):
            for value in dict(run.outputs).values():
                text = str(value).strip()
                if text.endswith(".json"):
                    path = Path(text)
                    if path.exists():
                        out.append(path.resolve())
    return sorted({path.resolve() for path in out})


def _finding_by_kind(findings: Sequence[ExpectationFinding], kind: str) -> List[ExpectationFinding]:
    return [finding for finding in findings if str(finding.kind) == str(kind)]


def _anchor_rows(
    loaded: Sequence[Tuple[Path, object, Dict[str, object]]],
) -> List[Dict[str, object]]:
    buckets: Dict[tuple[str, str, str], Dict[str, object]] = {}
    for _path, summary, _payload in loaded:
        key = (str(summary.dgp), str(summary.family), str(summary.suite_role))
        bucket = buckets.setdefault(
            key,
            {
                "problem_id": str(summary.dgp),
                "method_id": str(summary.family),
                "suite_role": str(summary.suite_role),
                "runs": 0,
                "train_docs": set(),
                "val_docs": set(),
                "test_docs": set(),
                "queries": [],
            },
        )
        bucket["runs"] = int(bucket["runs"]) + 1
        bucket["train_docs"].add(int(summary.support_budget.train_docs))
        bucket["val_docs"].add(int(summary.support_budget.val_docs))
        bucket["test_docs"].add(int(summary.support_budget.test_docs))
        bucket["queries"].append(float(summary.support_budget.total_queries_estimate))

    rows: List[Dict[str, object]] = []
    for bucket in buckets.values():
        rows.append(
            {
                "problem_id": bucket["problem_id"],
                "method_id": bucket["method_id"],
                "suite_role": bucket["suite_role"],
                "runs": int(bucket["runs"]),
                "train_docs": sorted(int(x) for x in bucket["train_docs"]),
                "val_docs": sorted(int(x) for x in bucket["val_docs"]),
                "test_docs": sorted(int(x) for x in bucket["test_docs"]),
                "mean_queries": (
                    sum(float(x) for x in bucket["queries"]) / float(len(bucket["queries"]))
                    if bucket["queries"]
                    else 0.0
                ),
            }
        )
    rows.sort(key=lambda row: (str(row["problem_id"]), str(row["suite_role"]), str(row["method_id"])))
    return rows


def _format_int_list(values: Iterable[int]) -> str:
    xs = sorted({int(x) for x in values})
    if not xs:
        return "-"
    if len(xs) == 1:
        return str(xs[0])
    return f"{xs[0]}..{xs[-1]}"


def _markdown_report(
    *,
    input_root: Path | None,
    manifest_path: Path | None,
    anchor_rows: Sequence[Dict[str, object]],
    findings: Sequence[ExpectationFinding],
) -> str:
    lines: List[str] = [
        "# Unified Local-Law Learnability Meta Report",
        "",
        "## Contract",
        "",
        "Every DGP is summarized under the same theorem-facing contract: define `oracle_g`, compare a DGP-specific `baseline_g` against a selected `learned_g`, serialize the `g` artifacts, and keep downstream oracle-target error test-only.",
        "",
    ]
    if input_root is not None:
        lines.append(f"- Input root: `{input_root}`")
    if manifest_path is not None:
        lines.append(f"- Manifest: `{manifest_path}`")
    lines.extend(
        [
            "",
            "## Cross-DGP Anchor Table",
            "",
            "| Problem | Method | Suite role | Runs | Train docs | Val docs | Test docs | Mean queries |",
            "| --- | --- | --- | ---: | --- | --- | --- | ---: |",
        ]
    )
    for row in anchor_rows:
        lines.append(
            "| "
            f"`{row['problem_id']}` | `{row['method_id']}` | `{row['suite_role']}` | {int(row['runs'])} | "
            f"{_format_int_list(row['train_docs'])} | {_format_int_list(row['val_docs'])} | {_format_int_list(row['test_docs'])} | "
            f"{float(row['mean_queries']):.1f} |"
        )

    lines.extend(["", "## Support Scaling", ""])
    support_findings = _finding_by_kind(findings, "support_scaling_improves_gap")
    if support_findings:
        for finding in support_findings:
            obs = dict(finding.observed_summary or {})
            lines.append(
                f"- `{finding.status.upper()}` `{finding.scenario}`: "
                f"start gap `{float(obs.get('start_gap', float('nan'))):.4f}`, "
                f"end gap `{float(obs.get('end_gap', float('nan'))):.4f}`. {obs.get('note', '')}"
            )
    else:
        lines.append("- No support-scaling findings were available.")

    lines.extend(["", "## Failure Modes", ""])
    counterexample_findings = _finding_by_kind(findings, "counterexample_breaks_target")
    if counterexample_findings:
        for finding in counterexample_findings:
            obs = dict(finding.observed_summary or {})
            lines.append(
                f"- `{finding.status.upper()}` `{finding.method}` in `{finding.scenario}`: "
                f"targeted laws `{obs.get('targeted_laws', [])}` with values `{obs.get('law_values', {})}`."
            )
    else:
        lines.append("- No standardized counterexample findings were available.")

    lines.extend(["", "## Downstream Sensitivity", ""])
    null_findings = _finding_by_kind(findings, "lambda_zero_null_control")
    if null_findings:
        for finding in null_findings:
            obs = dict(finding.observed_summary or {})
            lines.append(
                f"- `{finding.status.upper()}` `{finding.title}`: "
                f"median |primary gain| `{float(obs.get('median_abs_primary_gain_frac', float('nan'))):.4f}`, "
                f"p90 |primary gain| `{float(obs.get('p90_abs_primary_gain_frac', float('nan'))):.4f}`, "
                f"diagnostic max |Delta_vs_pooled| `{float(obs.get('max_abs_pooled_delta', float('nan'))):.4f}`, "
                f"max law-gap `{float(obs.get('max_law_gap', float('nan'))):.4f}`. {obs.get('note', '')}"
            )
    else:
        lines.append("- No lambda=0 null-control findings were available.")

    lines.extend(["", "## Selection Protocol", ""])
    selection_findings = _finding_by_kind(findings, "validation_only_selection")
    if selection_findings:
        finding = selection_findings[0]
        obs = dict(finding.observed_summary or {})
        lines.append(
            f"- `{finding.status.upper()}` validation-only selection: "
            f"{int(obs.get('n_violations', 0))} violation(s). {obs.get('note', '')}"
        )
    else:
        lines.append("- No selection-protocol finding was produced.")

    return "\n".join(lines).rstrip() + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the unified local-law meta report.")
    parser.add_argument("--input-root", type=Path, default=None)
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    report = build_local_law_expectation_report(
        output_root=args.input_root,
        manifest_path=args.manifest,
    )
    adapter = StructuredLocalLawAdapter()
    loaded = adapter.load_summaries(_collect_paths(args.input_root, args.manifest))
    anchor_rows = _anchor_rows(loaded)
    md_text = _markdown_report(
        input_root=args.input_root,
        manifest_path=args.manifest,
        anchor_rows=anchor_rows,
        findings=report.expectations,
    )

    md_path = args.output_dir / "local_law_meta_report.md"
    json_path = args.output_dir / "local_law_meta_report_summary.json"
    pdf_path = args.output_dir / "local_law_meta_report.pdf"
    summary_payload = {
        "expectations": _canonical_public_payload(report.to_dict()),
        "anchor_rows": anchor_rows,
    }
    assert_public_contract_clean(summary_payload, surface=str(json_path))
    md_path.write_text(md_text, encoding="utf-8")
    json_path.write_text(
        json.dumps(
            summary_payload,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    pdf_ok = _run_pandoc(md_path, pdf_path)

    print(f"wrote_markdown | {md_path}")
    print(f"wrote_summary | {json_path}")
    print(f"wrote_pdf | {pdf_path} | ok={pdf_ok}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
