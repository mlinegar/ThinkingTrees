#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build exact utility transport report.")
    p.add_argument("--output-root", type=Path, required=True)
    p.add_argument("--output-markdown", type=Path, required=True)
    p.add_argument("--output-pdf", type=Path, default=None)
    return p.parse_args()


def _run(cmd: list[str]) -> None:
    subprocess.run(cmd, cwd=str(REPO_ROOT), check=True)


def _run_pandoc(md_path: Path, pdf_path: Path) -> bool:
    if shutil.which("pandoc") is None or shutil.which("pdflatex") is None:
        return False
    subprocess.run(
        ["pandoc", str(md_path.name), "-o", str(pdf_path.name), "--pdf-engine=pdflatex"],
        cwd=str(md_path.parent),
        check=True,
    )
    return True


def _rel_path(base: Path, target: Path) -> str:
    return str(target.resolve().relative_to(base.resolve()))


def main() -> int:
    args = parse_args()
    summary_json = args.output_root / "utility_transport_summary.json"
    summary_csv = args.output_root / "utility_transport_summary.csv"
    figure_path = args.output_root / "figures" / "utility_transport_suite.png"
    objective_figure = args.output_root / "figures" / "utility_transport_suite_objective_comparison.png"
    preference_figure = args.output_root / "figures" / "utility_transport_suite_preference_curves.png"
    structural_figure = args.output_root / "figures" / "utility_transport_suite_structural_controls.png"
    targeted_ppo_figure = args.output_root / "figures" / "utility_transport_suite_targeted_nonseparable_ppo.png"
    targeted_objective_figure = args.output_root / "figures" / "utility_transport_suite_targeted_nonseparable_objectives.png"
    fairness_figure = args.output_root / "figures" / "utility_transport_suite_targeted_nonseparable_fairness.png"
    _run(
        [
            sys.executable,
            "scripts/summarize_treepo_preference_suite.py",
            "--output-root",
            str(args.output_root),
            "--output-json",
            str(summary_json),
            "--output-csv",
            str(summary_csv),
        ]
    )
    _run(
        [
            sys.executable,
            "scripts/plot_treepo_preference_suite.py",
            "--summary-json",
            str(summary_json),
            "--output",
            str(figure_path),
        ]
    )
    payload = json.loads(summary_json.read_text(encoding="utf-8"))
    findings = payload.get("findings", [])
    summary = payload.get("summary", {})
    hard_fails = [f for f in findings if f.get("status") == "fail"]
    unfair_warnings = [
        f
        for f in findings
        if f.get("status") == "warn" and not bool(dict(f.get("observed", {}) or {}).get("fair_comparison", True))
    ]
    structural_failures = [f for f in hard_fails if f.get("kind") == "tree_relevance"]
    exact_failures = [f for f in hard_fails if f.get("kind") == "exact_zero_error"]
    targeted_findings = [
        f
        for f in findings
        if f.get("lane") == "nonseparable" and f.get("oracle_profile") == "dgp2_boundary_interaction"
    ]
    targeted_failures = [f for f in targeted_findings if f.get("status") == "fail"]
    targeted_unfair = [
        f
        for f in targeted_findings
        if f.get("status") == "warn" and not bool(dict(f.get("observed", {}) or {}).get("fair_comparison", True))
    ]
    targeted_objective_status = {
        objective: [
            f
            for f in targeted_findings
            if f.get("objective_family") == objective and f.get("kind") == "tree_relevance"
        ]
        for objective in ("dpo", "grpo", "ppo", "hybrid_supervised_plus_ppo")
    }
    theorem_refs = sorted(
        {
            str(ref)
            for row in payload.get("rows", [])
            for ref in list(row.get("lean_theorems", []) or [])
            if str(ref).strip()
        }
    )
    figure_rel = _rel_path(args.output_root, figure_path)
    extra_figure_rels = {
        extra: _rel_path(args.output_root, extra)
        for extra in (
            objective_figure,
            preference_figure,
            structural_figure,
            targeted_ppo_figure,
            targeted_objective_figure,
            fairness_figure,
        )
    }
    lines = [
        "# Exact Utility Transport / TreePO",
        "",
        f"- Rows scanned: `{summary.get('n_rows', 0)}`",
        f"- Findings: `{summary.get('n_pass', 0)} pass / {summary.get('n_warn', 0)} warn / {summary.get('n_fail', 0)} fail / {summary.get('n_not_applicable', 0)} n/a`",
        "",
        "## Failure triage",
        "",
        f"- Exact-control failures: `{len(exact_failures)}`",
        f"- Structural/tree failures: `{len(structural_failures)}`",
        f"- Unfair-comparison warnings: `{len(unfair_warnings)}`",
        "",
        "## Core framing",
        "",
        "- This suite treats TreePO as oracle-indexed utility transport, not only as pairwise preference learning.",
        "- DPO, GRPO, PPO, and direct supervised-state learning are all objective-family instances over the same exact latent targets.",
        "- Zero utility regret is strongest only when it coincides with zero latent-state error; the suite reports both.",
        "- Local node labels, root labels, pairwise preferences, grouped preferences, and PPO rollouts are distinct support types and are not collapsed into one fairness claim.",
        "",
        "## Lean Surface",
        "",
    ]
    for ref in theorem_refs:
        lines.append(f"- `{ref}`")
    lines.extend(
        [
            "",
            "## Reading guide",
            "",
            "- The overview figure is the quickest sanity check: Markov root and merge errors versus local oracle coverage, boundary-topic utility regret versus local coverage, and nonseparable structural anchors.",
            "- The objective comparison figure holds support high and compares supervised-state, supervised-root, DPO, GRPO, PPO, and hybrid objectives on the same exact latent targets.",
            "- The preference-curve figure isolates preference-budget effects in the Markov exact-state lane.",
            "- The structural-control figure checks that exact/oracle, flat, undersupported, wrong-chunker, and one-leaf controls behave differently where theory says they should.",
            "- The targeted nonseparable PPO figures isolate the only remaining hard-fail lane and show the new `flat_span_equal_info` fairness control when local supervision is present.",
            "",
            "## Targeted Nonseparable PPO Status",
            "",
            f"- Targeted hard fails in `nonseparable / dgp2_boundary_interaction`: `{len(targeted_failures)}`",
            f"- Targeted unfair-comparison warnings in that lane: `{len(targeted_unfair)}`",
        ]
    )
    for objective, objective_findings in targeted_objective_status.items():
        if not objective_findings:
            continue
        statuses = ", ".join(sorted({str(f.get("status", "")) for f in objective_findings}))
        flat_arms = sorted(
            {
                str(dict(f.get("observed", {}) or {}).get("flat_arm", "")).strip()
                for f in objective_findings
                if str(dict(f.get("observed", {}) or {}).get("flat_arm", "")).strip()
            }
        )
        flat_arm_text = f" | flat arms={flat_arms}" if flat_arms else ""
        lines.append(f"- `{objective}` structural status: `{statuses}`{flat_arm_text}")
    lines.extend(
        [
            "- `flat_equal_info` remains the preference-only/root-only control.",
            "- `flat_span_equal_info` is now the matched local-supervision flat control and should be the comparator whenever local span labels are present.",
            "",
            "## Findings",
            "",
        ]
    )
    for finding in findings[:20]:
        lines.append(
            f"- `{finding['status']}` {finding['lane']} / {finding['oracle_profile']} / "
            f"{finding['objective_family']}: {finding['title']} | observed={finding['observed']}"
        )
    lines.extend(
        [
            "",
            "## Figures",
            "",
            f"![utility transport suite]({figure_rel})",
            "",
        ]
    )
    for extra in extra_figure_rels:
        if extra.exists():
            lines.extend([f"![{extra.stem}]({extra_figure_rels[extra]})", ""])
    args.output_markdown.parent.mkdir(parents=True, exist_ok=True)
    args.output_markdown.write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote_markdown | {args.output_markdown}")
    completion_memo = args.output_root / "completion_memo.md"
    completion_memo.write_text(
        "\n".join(
            [
                "# Exact Utility Transport Completion Memo",
                "",
                f"- Rows scanned: `{summary.get('n_rows', 0)}`",
                f"- Findings: `{summary.get('n_pass', 0)} pass / {summary.get('n_warn', 0)} warn / {summary.get('n_fail', 0)} fail / {summary.get('n_not_applicable', 0)} n/a`",
                f"- Exact-control failures: `{len(exact_failures)}`",
                f"- Structural/tree failures: `{len(structural_failures)}`",
                f"- Unfair-comparison warnings: `{len(unfair_warnings)}`",
                "",
                "## Targeted lane",
                "",
                "- Lane: `nonseparable / dgp2_boundary_interaction`",
                f"- Hard fails in targeted lane: `{len(targeted_failures)}`",
                f"- Unfair-comparison warnings in targeted lane: `{len(targeted_unfair)}`",
                "",
                "## Flat controls",
                "",
                "- `flat_equal_info` is the preference-only/root-only comparator.",
                "- `flat_span_equal_info` is the matched flat comparator for local-supervision slices.",
                "",
                "## Main artifacts",
                "",
                f"- Summary JSON: `{summary_json}`",
                f"- Summary CSV: `{summary_csv}`",
                f"- Markdown report: `{args.output_markdown}`",
                f"- Overview figure: `{figure_path}`",
            ]
        ),
        encoding="utf-8",
    )
    print(f"wrote_completion_memo | {completion_memo}")
    if args.output_pdf is not None:
        if _run_pandoc(args.output_markdown, args.output_pdf):
            print(f"wrote_pdf | {args.output_pdf}")
        else:
            print("pdf_skipped | pandoc or pdflatex not available")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
