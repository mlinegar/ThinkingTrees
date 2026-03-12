#!/usr/bin/env python3
"""Generate a draft paper-report bundle for a formal rerun root.

This script does two things:

1. Run the existing paper-facing report scripts where there is usable data.
2. Create draft placeholder reports for suites that are still missing or empty.

The intent is to keep a single paper-facing index current while long reruns are
still in flight.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Sequence


@dataclass(frozen=True)
class ReportJob:
    name: str
    title: str
    description: str
    root_rel: str
    script_rel: Optional[str]
    args: Sequence[str]
    expected_outputs: Sequence[str]
    pending_note: str = ""
    always_run: bool = False
    bundle_role: str = "paper"


@dataclass(frozen=True)
class ExcludedRoot:
    title: str
    root_rel: str
    reason: str
    note: str = ""


def _excluded_roots() -> List[ExcludedRoot]:
    return [
        ExcludedRoot(
            title="Reused LDA Local-Law Journal Root",
            root_rel="../tree_relevant_lda_local_law_20260308_210436",
            reason=(
                "Excluded from the paper-facing bundle because the reused `quadratic_utility_weight` in this root "
                "(historically serialized as `lambda_multiplier`) is a latent quadratic-utility multiplier, "
                "not a normalized local-law weight in [0,1]."
            ),
            note=(
                "Keep this root only for audit/debug use. Do not use it as the paper-facing LDA "
                "local-law lambda comparison."
            ),
        ),
    ]


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate a draft paper-report bundle for a formal rerun root.")
    p.add_argument("--formal-root", type=Path, required=True, help="Formal rerun root, e.g. outputs/formal_reruns_<stamp>.")
    p.add_argument(
        "--bundle-dir",
        type=Path,
        default=None,
        help="Output directory for the bundle manifest/index (default: <formal-root>/paper_reports).",
    )
    p.add_argument(
        "--python-bin",
        type=str,
        default=sys.executable,
        help="Python interpreter to use for subprocess report generation.",
    )
    return p.parse_args()


def _json_count(root: Path) -> int:
    if not root.exists():
        return 0
    return sum(1 for _ in root.rglob("*.json"))


def _load_json(path: Path) -> Optional[Dict[str, object]]:
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    return payload if isinstance(payload, dict) else None


def _line_count(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(1 for _ in path.open("r", encoding="utf-8"))


def _materialize(template: str, *, root: Path, formal_root: Path, bundle_dir: Path) -> str:
    return template.format(
        root=str(root),
        formal_root=str(formal_root),
        bundle_dir=str(bundle_dir),
    )


def _placeholder_outputs(job: ReportJob, *, root: Path, bundle_dir: Path) -> Dict[str, str]:
    slug = job.name
    out_dir = bundle_dir / "pending"
    out_dir.mkdir(parents=True, exist_ok=True)
    md_path = out_dir / f"{slug}.md"
    json_path = out_dir / f"{slug}.json"
    payload = {
        "status": "pending",
        "title": job.title,
        "root": str(root),
        "json_count": int(_json_count(root)),
        "script": job.script_rel,
        "bundle_role": job.bundle_role,
        "note": job.pending_note or "No usable data yet for this suite.",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    lines = [
        f"# {job.title}",
        "",
        f"- Status: `pending`",
        f"- Root: `{root}`",
        f"- JSON files seen: `{payload['json_count']}`",
        f"- Expected report script: `{job.script_rel or 'none'}`",
        f"- Bundle role: `{job.bundle_role}`",
        "",
        payload["note"],
        "",
    ]
    md_path.write_text("\n".join(lines), encoding="utf-8")
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return {"markdown": str(md_path), "json": str(json_path)}


def _run_job(job: ReportJob, *, repo_root: Path, formal_root: Path, bundle_dir: Path, python_bin: str) -> Dict[str, object]:
    root = formal_root / job.root_rel
    json_count = _json_count(root)
    status = "pending"
    stdout_path = bundle_dir / "logs" / f"{job.name}.stdout.log"
    stderr_path = bundle_dir / "logs" / f"{job.name}.stderr.log"
    stdout_path.parent.mkdir(parents=True, exist_ok=True)

    if job.script_rel is None or (json_count <= 0 and not job.always_run):
        placeholder = _placeholder_outputs(job, root=root, bundle_dir=bundle_dir)
        return {
            "name": job.name,
            "title": job.title,
            "root": str(root),
            "json_count": int(json_count),
            "status": status,
            "script": job.script_rel,
            "outputs": placeholder,
        }

    script_path = repo_root / job.script_rel
    cmd = [python_bin, str(script_path)]
    for arg in job.args:
        cmd.append(_materialize(arg, root=root, formal_root=formal_root, bundle_dir=bundle_dir))

    started = time.time()
    proc = subprocess.run(cmd, cwd=str(repo_root), capture_output=True, text=True)
    elapsed = time.time() - started
    stdout_path.write_text(proc.stdout or "", encoding="utf-8")
    stderr_path.write_text(proc.stderr or "", encoding="utf-8")

    outputs: Dict[str, str] = {}
    for rel in job.expected_outputs:
        resolved = Path(_materialize(rel, root=root, formal_root=formal_root, bundle_dir=bundle_dir))
        outputs[resolved.name] = str(resolved)

    if proc.returncode == 0:
        status = "completed"
    else:
        status = "failed"
        placeholder = _placeholder_outputs(job, root=root, bundle_dir=bundle_dir)
        outputs.update({f"pending_{k}": v for k, v in placeholder.items()})

    if status == "completed":
        if job.name == "simulation_buildout":
            meta = _load_json(formal_root / "commands" / "simulation_buildout_meta.json") or {}
            expected = int(meta.get("n_plot_commands_total", 0) or 0)
            done = len(list((root / "figures").glob("*_report.json")))
            if expected > 0 and done < expected:
                status = "partial"
        elif job.name == "publication_clean":
            diag = _load_json(root / "figures" / "identifiable_zero_publication_report_latest_diagnostics.json") or {}
            checker = diag.get("checker") or {}
            diagnostics = diag.get("diagnostics") or {}
            slice_checks = diagnostics.get("slice_consistency_checks") or {}
            if int(checker.get("returncode", 1)) != 0 or not bool(slice_checks.get("passed", False)):
                status = "partial"
        elif job.name == "publication_ctreepo_progress":
            diag = _load_json(root / "figures" / "publication_progress" / "publication_ctreepo_progress_diagnostics.json") or {}
            total = _line_count(formal_root / "commands" / "identifiable_zero_publication_ctreepo_gpu_prefetch_cmds.txt")
            done = int(diag.get("n_rows", 0) or 0)
            if total > 0 and done < total:
                status = "partial"
        elif job.name == "lda_tree_recovery_progress":
            summary_path = Path(str(outputs.get("lda_tree_recovery_progress_summary.json", "")))
            summary = _load_json(summary_path) or {}
            counts = summary.get("counts") or {}
            done_counts = summary.get("done_counts") or {}
            expected = summary.get("expected_counts") or {}
            if not done_counts:
                done_counts = {
                    "exact_cpu": int(counts.get("exact_cpu_done", 0) or 0),
                    "learned_cpu_shadow": int(counts.get("learned_cpu_shadow_done", 0) or 0),
                    "learned_gpu": int(counts.get("learned_gpu_done", 0) or 0),
                }
            if not expected:
                expected = {
                    "exact_cpu": int(counts.get("exact_cpu", 0) or 0),
                    "learned_cpu_shadow": int(counts.get("learned_cpu_shadow", 0) or 0),
                    "learned_gpu": int(counts.get("learned_gpu", 0) or 0),
                    "learned_gpu_per_bundle": int(counts.get("learned_gpu_per_bundle", 0) or 0),
                }
            lanes = sorted(set(done_counts.keys()) | set(expected.keys()))
            if any(
                int(done_counts.get(k, 0) or 0) < int(expected.get(k, 0) or 0)
                for k in lanes
                if int(expected.get(k, 0) or 0) > 0
            ):
                status = "partial"
        elif job.name == "learnability":
            diag = _load_json(root / "figures" / "learnability" / "identifiable_zero_learnability_latest_diagnostics.json") or {}
            markov = diag.get("markov") or {}
            ctree = diag.get("ctree") or {}
            setup_alignment = diag.get("setup_alignment") or {}
            if (
                int(markov.get("n_rows", 0) or 0) <= 0
                or int(ctree.get("n_rows", 0) or 0) <= 0
                or not bool(setup_alignment.get("matches", False))
            ):
                status = "partial"

    return {
        "name": job.name,
        "title": job.title,
        "description": job.description,
        "root": str(root),
        "json_count": int(json_count),
        "status": status,
        "bundle_role": job.bundle_role,
        "script": str(script_path),
        "command": cmd,
        "elapsed_seconds": float(elapsed),
        "stdout_log": str(stdout_path),
        "stderr_log": str(stderr_path),
        "returncode": int(proc.returncode),
        "outputs": outputs,
    }


def _jobs() -> List[ReportJob]:
    return [
        ReportJob(
            name="cpu_megasweep",
            title="CPU Megasweep",
            description="Baseline megasweep consolidated report.",
            root_rel="cpu_megasweep",
            script_rel="scripts/report_cpu_megasweep.py",
            args=["--output-root", "{root}", "--output-report", "{root}/figures/megasweep_consolidated_report.md"],
            expected_outputs=["{root}/figures/megasweep_consolidated_report.md"],
        ),
        ReportJob(
            name="cpu_megasweep_readable",
            title="CPU Megasweep Readable",
            description="Readable baseline megasweep report.",
            root_rel="cpu_megasweep",
            script_rel="scripts/report_cpu_megasweep_readable.py",
            args=[
                "--output-root",
                "{root}",
                "--output-markdown",
                "{root}/figures/megasweep_consolidated_readable_report.md",
                "--no-emit-pdf",
            ],
            expected_outputs=["{root}/figures/megasweep_consolidated_readable_report.md"],
        ),
        ReportJob(
            name="simulation_buildout",
            title="Simulation Buildout",
            description="Draft buildout report; may be sparse until figure-producing stages complete.",
            root_rel="simulation_buildout",
            script_rel="scripts/report_simulation_buildout.py",
            args=["--output-root", "{root}", "--no-emit-pdf"],
            expected_outputs=["{root}/figures/simulation_buildout_report.md"],
            always_run=True,
            pending_note="The buildout root exists but may still lack figure JSONs. Draft markdown is still useful.",
        ),
        ReportJob(
            name="publication_clean",
            title="Identifiable-Zero Publication Clean",
            description="Main clean cross-family publication draft.",
            root_rel="identifiable_zero_longrun_clean",
            script_rel="scripts/report_identifiable_zero_suite_publication_clean.py",
            args=["--output-root", "{root}", "--no-emit-pdf", "--allow-partial"],
            expected_outputs=[
                "{root}/figures/identifiable_zero_publication_report_latest.md",
                "{root}/figures/identifiable_zero_publication_report_latest_diagnostics.json",
            ],
        ),
        ReportJob(
            name="publication_ctreepo_progress",
            title="Publication C-TreePO Progress",
            description="Partial-run C-TreePO publication progress report.",
            root_rel="identifiable_zero_publication_ctreepo",
            script_rel="scripts/report_identifiable_zero_publication_ctreepo_progress.py",
            args=["--output-root", "{root}", "--no-emit-pdf"],
            expected_outputs=[
                "{root}/figures/publication_progress/publication_ctreepo_progress_latest.md",
                "{root}/figures/publication_progress/publication_ctreepo_progress_diagnostics.json",
            ],
        ),
        ReportJob(
            name="lda_tree_recovery_progress",
            title="Diagnostic: LDA Tree Recovery Progress",
            description="Diagnostic-only LDA tree-recovery production progress report.",
            root_rel="lda_tree_recovery_production",
            script_rel="scripts/report_lda_tree_recovery_progress.py",
            args=["--input-root", "{root}", "--output-dir", "{formal_root}/diagnostic_reports/lda_tree_recovery_production"],
            expected_outputs=[
                "{formal_root}/diagnostic_reports/lda_tree_recovery_production/lda_tree_recovery_progress_report.pdf",
                "{formal_root}/diagnostic_reports/lda_tree_recovery_production/lda_tree_recovery_progress_summary.json",
            ],
            bundle_role="diagnostic",
        ),
        ReportJob(
            name="neural_operator_overnight",
            title="Identifiable-Zero Neural Operator Overnight",
            description="Neural-operator overnight robustness report.",
            root_rel="identifiable_zero_neural_operator_v2",
            script_rel="scripts/report_identifiable_zero_neural_operator_overnight.py",
            args=["--overnight-output-root", "{root}", "--no-emit-pdf"],
            expected_outputs=[
                "{root}/figures/neural_operator_overnight/identifiable_zero_neural_operator_overnight_latest.md",
            ],
        ),
        ReportJob(
            name="learnability",
            title="Identifiable-Zero Learnability",
            description="Appendix-quality learnability report.",
            root_rel="identifiable_zero_learnability",
            script_rel="scripts/report_identifiable_zero_learnability.py",
            args=["--output-root", "{root}", "--no-emit-pdf"],
            expected_outputs=["{root}/figures/learnability/identifiable_zero_learnability_latest.md"],
            pending_note="Learnability reruns have not been generated in the current formal root yet.",
        ),
        ReportJob(
            name="lda_leafnoise",
            title="Identifiable-Zero LDA Leafnoise",
            description="Appendix-style leaf-noise progression report.",
            root_rel="identifiable_zero_lda_leafnoise",
            script_rel="scripts/report_identifiable_zero_lda_leafnoise_progression.py",
            args=["--output-root", "{root}", "--no-emit-pdf"],
            expected_outputs=["{root}/figures/lda_leafnoise/identifiable_zero_lda_leafnoise_latest.md"],
            pending_note="Leaf-noise reruns have not been generated in the current formal root yet.",
        ),
        ReportJob(
            name="dtm_lda",
            title="Identifiable-Zero DTM-LDA",
            description="DTM-LDA appendix/robustness suite.",
            root_rel="identifiable_zero_dtm_lda",
            script_rel="scripts/report_identifiable_zero_dtm_lda.py",
            args=["--output-root", "{root}", "--no-emit-pdf"],
            expected_outputs=["{root}/figures/dtm_lda/identifiable_zero_dtm_lda_latest.md"],
            pending_note="DTM-LDA reruns have not been generated in the current formal root yet.",
        ),
    ]


def _write_bundle_index(
    results: Sequence[Dict[str, object]],
    *,
    formal_root: Path,
    bundle_dir: Path,
    excluded_roots: Sequence[ExcludedRoot],
) -> Path:
    md_path = bundle_dir / "paper_report_index.md"
    lines: List[str] = []
    lines.append("# Paper Report Bundle")
    lines.append("")
    lines.append(f"- Formal root: `{formal_root}`")
    lines.append(f"- Generated: `{datetime.now(timezone.utc).isoformat()}`")
    lines.append("")
    paper_rows = [row for row in results if str(row.get("bundle_role") or "paper") == "paper"]
    diagnostic_rows = [row for row in results if str(row.get("bundle_role") or "paper") != "paper"]

    def _table_block(title: str, rows: Sequence[Dict[str, object]]) -> None:
        if not rows:
            return
        lines.append(f"## {title}")
        lines.append("")
        lines.append("| Suite | Status | JSON files | Root | Primary outputs |")
        lines.append("| --- | --- | --- | --- | --- |")
        for row in rows:
            outputs = row.get("outputs", {}) or {}
            primary = ", ".join(f"`{Path(v).name}`" for v in outputs.values()) if outputs else "—"
            lines.append(
                f"| {row.get('title')} | `{row.get('status')}` | `{row.get('json_count')}` | `{row.get('root')}` | {primary} |"
            )
        lines.append("")

    _table_block("Paper-Facing Suites", paper_rows)
    _table_block("Diagnostic Suites", diagnostic_rows)
    if excluded_roots:
        lines.append("## Excluded Roots")
        lines.append("")
        lines.append("| Root | Reason |")
        lines.append("| --- | --- |")
        for item in excluded_roots:
            root = (formal_root / item.root_rel).resolve()
            lines.append(f"| `{root}` | {item.reason} |")
        lines.append("")
    for row in results:
        lines.append(f"## {row.get('title')}")
        lines.append("")
        lines.append(f"- Status: `{row.get('status')}`")
        lines.append(f"- Root: `{row.get('root')}`")
        lines.append(f"- JSON files: `{row.get('json_count')}`")
        lines.append(f"- Bundle role: `{row.get('bundle_role', 'paper')}`")
        if row.get("script"):
            lines.append(f"- Script: `{row.get('script')}`")
        if row.get("command"):
            lines.append(f"- Command: `{json.dumps(row.get('command'))}`")
        if row.get("stdout_log"):
            lines.append(f"- Stdout log: `{row.get('stdout_log')}`")
        if row.get("stderr_log"):
            lines.append(f"- Stderr log: `{row.get('stderr_log')}`")
        outputs = row.get("outputs", {}) or {}
        if outputs:
            lines.append("- Outputs:")
            for value in outputs.values():
                lines.append(f"  - `{value}`")
        if row.get("description"):
            lines.append("")
            lines.append(str(row["description"]))
        lines.append("")
    if excluded_roots:
        for item in excluded_roots:
            root = (formal_root / item.root_rel).resolve()
            lines.append(f"## Excluded: {item.title}")
            lines.append("")
            lines.append(f"- Root: `{root}`")
            lines.append(f"- Reason: {item.reason}")
            if item.note:
                lines.append(f"- Note: {item.note}")
            lines.append("")
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return md_path


def main() -> int:
    args = _parse_args()
    repo_root = Path(__file__).resolve().parent.parent
    formal_root = args.formal_root.resolve()
    bundle_dir = args.bundle_dir.resolve() if args.bundle_dir is not None else (formal_root / "paper_reports")
    bundle_dir.mkdir(parents=True, exist_ok=True)
    pending_dir = bundle_dir / "pending"
    if pending_dir.exists():
        for child in pending_dir.iterdir():
            if child.is_file():
                child.unlink()
    excluded_roots = _excluded_roots()

    results = [
        _run_job(job, repo_root=repo_root, formal_root=formal_root, bundle_dir=bundle_dir, python_bin=args.python_bin)
        for job in _jobs()
    ]
    index_path = _write_bundle_index(
        results,
        formal_root=formal_root,
        bundle_dir=bundle_dir,
        excluded_roots=excluded_roots,
    )

    manifest = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "formal_root": str(formal_root),
        "bundle_dir": str(bundle_dir),
        "index_markdown": str(index_path),
        "excluded_roots": [
            {
                "title": item.title,
                "root": str((formal_root / item.root_rel).resolve()),
                "reason": item.reason,
                "note": item.note,
            }
            for item in excluded_roots
        ],
        "results": results,
    }
    manifest_path = bundle_dir / "paper_report_bundle_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print(
        json.dumps(
            {
                "bundle_dir": str(bundle_dir),
                "index_markdown": str(index_path),
                "manifest_json": str(manifest_path),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
