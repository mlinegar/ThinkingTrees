#!/usr/bin/env python3
"""Generate the v2 paper-report bundle from the canonical suite registry."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.ctreepo.sim.suite.registry import CanonicalSuiteTarget, iter_canonical_suite_targets


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


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate the canonical paper-report bundle for a formal rerun root.")
    parser.add_argument("--formal-root", type=Path, required=True, help="Formal rerun root, e.g. outputs/formal_reruns_<stamp>.")
    parser.add_argument(
        "--bundle-dir",
        type=Path,
        default=None,
        help="Output directory for the bundle manifest/index (default: <formal-root>/paper_reports).",
    )
    parser.add_argument(
        "--python-bin",
        type=str,
        default=sys.executable,
        help="Python interpreter to use for suite report generation.",
    )
    return parser.parse_args(list(argv) if argv is not None else None)


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


def _load_suite_meta(root: Path) -> Dict[str, object]:
    return _load_json(root / "suite_meta.json") or {}


def _materialize(template: str, *, root: Path, formal_root: Path, bundle_dir: Path) -> str:
    return template.format(
        root=str(root),
        formal_root=str(formal_root),
        bundle_dir=str(bundle_dir),
    )


def _placeholder_outputs(
    target: CanonicalSuiteTarget,
    *,
    root: Path,
    bundle_dir: Path,
) -> Dict[str, str]:
    out_dir = bundle_dir / "pending"
    out_dir.mkdir(parents=True, exist_ok=True)
    md_path = out_dir / f"{target.key}.md"
    json_path = out_dir / f"{target.key}.json"
    payload = {
        "status": "pending",
        "title": target.title,
        "root": str(root),
        "json_count": int(_json_count(root)),
        "cli_suite_name": target.cli_suite_name,
        "bundle_role": target.bundle_role,
        "note": target.pending_note or "No usable data yet for this suite.",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    lines = [
        f"# {target.title}",
        "",
        "- Status: `pending`",
        f"- Root: `{root}`",
        f"- JSON files seen: `{payload['json_count']}`",
        f"- Canonical suite: `{target.cli_suite_name}`",
        f"- Bundle role: `{target.bundle_role}`",
        "",
        str(payload["note"]),
        "",
    ]
    md_path.write_text("\n".join(lines), encoding="utf-8")
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return {"markdown": str(md_path), "json": str(json_path)}


def _suite_report_command(
    target: CanonicalSuiteTarget,
    *,
    root: Path,
    formal_root: Path,
    bundle_dir: Path,
    python_bin: str,
) -> List[str]:
    cmd = [
        str(python_bin),
        "-m",
        "src.ctreepo.cli",
        "sim",
        "suite",
        target.cli_suite_name,
        "report",
        "--output-root",
        str(root),
    ]
    for arg in target.report_args:
        cmd.append(_materialize(str(arg), root=root, formal_root=formal_root, bundle_dir=bundle_dir))
    return cmd


def _finalize_status(
    *,
    target: CanonicalSuiteTarget,
    root: Path,
    formal_root: Path,
    outputs: Mapping[str, str],
    initial_status: str,
) -> str:
    status = str(initial_status)
    if status != "completed":
        return status

    suite_meta = _load_suite_meta(root)
    if target.key == "simulation_buildout":
        legacy_meta = dict(suite_meta.get("legacy_builder_meta", {}) or {})
        fallback_meta = _load_json(formal_root / "commands" / "simulation_buildout_meta.json") or {}
        expected = int(legacy_meta.get("n_plot_commands_total", fallback_meta.get("n_plot_commands_total", 0)) or 0)
        done = len(list((root / "figures").glob("*_report.json")))
        if expected > 0 and done < expected:
            return "partial"
    elif target.key == "publication_clean":
        diag = _load_json(root / "figures" / "identifiable_zero_publication_report_latest_diagnostics.json") or {}
        checker = diag.get("checker") or {}
        diagnostics = diag.get("diagnostics") or {}
        slice_checks = diagnostics.get("slice_consistency_checks") or {}
        if int(checker.get("returncode", 1)) != 0 or not bool(slice_checks.get("passed", False)):
            return "partial"
    elif target.key == "publication_ctreepo_progress":
        diag = _load_json(root / "figures" / "publication_progress" / "publication_ctreepo_progress_diagnostics.json") or {}
        total = int(suite_meta.get("n_commands_total", 0) or 0)
        if total <= 0:
            total = _line_count(formal_root / "commands" / "identifiable_zero_publication_ctreepo_gpu_prefetch_cmds.txt")
        done = int(diag.get("n_rows", 0) or 0)
        if total > 0 and done < total:
            return "partial"
    elif target.key == "lda_tree_recovery_progress":
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
            int(done_counts.get(key, 0) or 0) < int(expected.get(key, 0) or 0)
            for key in lanes
            if int(expected.get(key, 0) or 0) > 0
        ):
            return "partial"
    elif target.key == "learnability":
        diag = _load_json(root / "figures" / "learnability" / "identifiable_zero_learnability_latest_diagnostics.json") or {}
        markov = diag.get("markov") or {}
        ctree = diag.get("ctree") or {}
        setup_alignment = diag.get("setup_alignment") or {}
        if (
            int(markov.get("n_rows", 0) or 0) <= 0
            or int(ctree.get("n_rows", 0) or 0) <= 0
            or not bool(setup_alignment.get("matches", False))
        ):
            return "partial"
    return status


def _run_target(
    target: CanonicalSuiteTarget,
    *,
    formal_root: Path,
    bundle_dir: Path,
    python_bin: str,
) -> Dict[str, object]:
    root = (formal_root / target.root_rel).resolve()
    json_count = _json_count(root)
    stdout_path = bundle_dir / "logs" / f"{target.key}.stdout.log"
    stderr_path = bundle_dir / "logs" / f"{target.key}.stderr.log"
    stdout_path.parent.mkdir(parents=True, exist_ok=True)

    if json_count <= 0 and not bool(target.always_run):
        placeholder = _placeholder_outputs(target, root=root, bundle_dir=bundle_dir)
        return {
            "name": target.key,
            "title": target.title,
            "description": target.description,
            "root": str(root),
            "json_count": int(json_count),
            "status": "pending",
            "bundle_role": target.bundle_role,
            "cli_suite_name": target.cli_suite_name,
            "outputs": placeholder,
        }

    cmd = _suite_report_command(target, root=root, formal_root=formal_root, bundle_dir=bundle_dir, python_bin=python_bin)
    started = time.time()
    proc = subprocess.run(cmd, cwd=str(Path(__file__).resolve().parent.parent), capture_output=True, text=True)
    elapsed = time.time() - started
    stdout_path.write_text(proc.stdout or "", encoding="utf-8")
    stderr_path.write_text(proc.stderr or "", encoding="utf-8")

    outputs: Dict[str, str] = {}
    for rel in target.expected_outputs:
        resolved = Path(_materialize(rel, root=root, formal_root=formal_root, bundle_dir=bundle_dir))
        outputs[resolved.name] = str(resolved)

    status = "completed" if proc.returncode == 0 else "failed"
    if status == "failed":
        placeholder = _placeholder_outputs(target, root=root, bundle_dir=bundle_dir)
        outputs.update({f"pending_{key}": value for key, value in placeholder.items()})

    status = _finalize_status(
        target=target,
        root=root,
        formal_root=formal_root,
        outputs=outputs,
        initial_status=status,
    )
    return {
        "name": target.key,
        "title": target.title,
        "description": target.description,
        "root": str(root),
        "json_count": int(json_count),
        "status": status,
        "bundle_role": target.bundle_role,
        "cli_suite_name": target.cli_suite_name,
        "command": cmd,
        "elapsed_seconds": float(elapsed),
        "stdout_log": str(stdout_path),
        "stderr_log": str(stderr_path),
        "returncode": int(proc.returncode),
        "outputs": outputs,
    }


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

    role_titles = {
        "paper": "Paper-Facing Suites",
        "appendix": "Appendix Suites",
        "diagnostic": "Diagnostic Suites",
    }
    for role, title in role_titles.items():
        rows = [row for row in results if str(row.get("bundle_role") or "paper") == role]
        if not rows:
            continue
        lines.append(f"## {title}")
        lines.append("")
        lines.append("| Suite | Status | JSON files | Root | Primary outputs |")
        lines.append("| --- | --- | --- | --- | --- |")
        for row in rows:
            outputs = row.get("outputs", {}) or {}
            primary = ", ".join(f"`{Path(value).name}`" for value in outputs.values()) if outputs else "—"
            lines.append(
                f"| {row.get('title')} | `{row.get('status')}` | `{row.get('json_count')}` | `{row.get('root')}` | {primary} |"
            )
        lines.append("")

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
        if row.get("cli_suite_name"):
            lines.append(f"- Canonical suite: `{row.get('cli_suite_name')}`")
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


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
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
        _run_target(target, formal_root=formal_root, bundle_dir=bundle_dir, python_bin=str(args.python_bin))
        for target in iter_canonical_suite_targets(bundle_roles=("paper", "appendix", "diagnostic"))
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
