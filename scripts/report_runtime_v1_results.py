#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.experiments import merge_artifacts  # noqa: E402


METHOD_LABELS = {
    "full_context": "Full context",
    "retrieval": "Retrieval",
    "summary_tree": "Summary tree",
    "state_tree": "State tree",
    "neural_operator": "Neural operator",
}
METHOD_DESCRIPTIONS = {
    "full_context": "Scorer reads the official full-context prompt.",
    "retrieval": "Embedder selects evidence; scorer answers from selected text.",
    "summary_tree": "Summarizer builds recursive text summaries; scorer answers from the tree representation.",
    "state_tree": "Summarizer/state-surrogate path renders compressed evidence; scorer answers from rendered state.",
    "neural_operator": "State model selects or renders evidence when configured; scorer produces the final answer.",
}


def _load_json(path: Path) -> Dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _iter_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    if not path.exists():
        return
    with path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            payload = json.loads(line)
            if isinstance(payload, dict):
                yield payload


def _score_for_prediction(row: Mapping[str, Any]) -> tuple[str, float]:
    metrics = dict(row.get("metrics") or {})
    primary = str(row.get("primary_metric") or "")
    if not primary and metrics:
        primary = str(next(iter(metrics.keys())))
    value = metrics.get(primary, 0.0) if primary else 0.0
    try:
        return primary or "score", float(value)
    except Exception:
        return primary or "score", 0.0


def _safe_mean(values: Sequence[float]) -> float:
    return float(sum(values) / len(values)) if values else 0.0


def _pct(value: float) -> str:
    return f"{100.0 * float(value):.1f}%"


def _method_label(method_id: str) -> str:
    return METHOD_LABELS.get(str(method_id), str(method_id))


def _group_key(row: Mapping[str, Any]) -> tuple[str, str, int, str]:
    return (
        str(row.get("phase_id", "") or ""),
        str(row.get("task_id", "") or ""),
        int(row.get("max_seq_length", 0) or 0),
        str(row.get("method", "") or ""),
    )


def build_summary(run_dir: Path) -> Dict[str, Any]:
    run_dir = Path(run_dir).expanduser().resolve()
    predictions = list(_iter_jsonl(run_dir / "predictions.jsonl"))
    calls = list(_iter_jsonl(run_dir / "calls.jsonl"))
    metrics = _load_json(run_dir / "metrics.json")
    manifest = _load_json(run_dir / "experiment_manifest.json")
    config = _load_json(run_dir / "config.json")

    method_scores: dict[str, list[float]] = defaultdict(list)
    method_failures: Counter[str] = Counter()
    method_prompt_tokens: dict[str, list[float]] = defaultdict(list)
    method_completion_tokens: dict[str, list[float]] = defaultdict(list)
    method_domains: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    method_difficulties: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    method_lengths: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    slice_scores: dict[tuple[str, str, int, str], list[float]] = defaultdict(list)
    primary_metric = ""

    for row in predictions:
        method = str(row.get("method", "") or "")
        metric_name, score = _score_for_prediction(row)
        primary_metric = primary_metric or metric_name
        method_scores[method].append(score)
        slice_scores[_group_key(row)].append(score)
        if row.get("failure"):
            method_failures[method] += 1
        cost = dict(row.get("cost") or {})
        method_prompt_tokens[method].append(float(cost.get("prompt_tokens", 0) or 0.0))
        method_completion_tokens[method].append(float(cost.get("completion_tokens", 0) or 0.0))
        problem_meta = dict((row.get("metadata") or {}).get("problem") or {})
        if problem_meta.get("domain") is not None:
            method_domains[method][str(problem_meta.get("domain"))].append(score)
        if problem_meta.get("difficulty") is not None:
            method_difficulties[method][str(problem_meta.get("difficulty"))].append(score)
        if problem_meta.get("length") is not None:
            method_lengths[method][str(problem_meta.get("length"))].append(score)

    calls_by_method: Counter[str] = Counter()
    calls_by_role: Counter[str] = Counter()
    calls_by_surface: Counter[str] = Counter()
    for call in calls:
        method = str(call.get("method_id") or call.get("method") or "")
        if method:
            calls_by_method[method] += 1
        role = str(call.get("role", "") or "")
        if role:
            calls_by_role[role] += 1
        surface = str(call.get("surface", "") or "")
        if surface:
            calls_by_surface[surface] += 1

    method_rows = []
    for method in sorted(method_scores):
        values = method_scores[method]
        method_rows.append(
            {
                "method_id": method,
                "label": _method_label(method),
                "description": METHOD_DESCRIPTIONS.get(method, ""),
                "n_predictions": len(values),
                "mean_score": _safe_mean(values),
                "failures": int(method_failures.get(method, 0)),
                "surface_calls": int(calls_by_method.get(method, 0)),
                "mean_prompt_tokens": _safe_mean(method_prompt_tokens.get(method, [])),
                "mean_completion_tokens": _safe_mean(method_completion_tokens.get(method, [])),
                "by_domain": {
                    key: _safe_mean(items)
                    for key, items in sorted(method_domains.get(method, {}).items())
                },
                "by_difficulty": {
                    key: _safe_mean(items)
                    for key, items in sorted(method_difficulties.get(method, {}).items())
                },
                "by_length": {
                    key: _safe_mean(items)
                    for key, items in sorted(method_lengths.get(method, {}).items())
                },
            }
        )

    slice_rows = []
    for (phase, task, max_seq_length, method), values in sorted(slice_scores.items()):
        slice_rows.append(
            {
                "phase_id": phase,
                "task_id": task,
                "max_seq_length": max_seq_length,
                "method_id": method,
                "n_predictions": len(values),
                "mean_score": _safe_mean(values),
            }
        )

    roles = dict((config.get("roles") or {}) if isinstance(config.get("roles"), dict) else {})
    oracle = dict((config.get("oracle") or {}) if isinstance(config.get("oracle"), dict) else {})
    return {
        "experiment_dir": str(run_dir),
        "experiment_id": str(config.get("experiment_id") or config.get("run_id") or run_dir.name),
        "benchmark": dict(config.get("benchmark") or {}),
        "primary_metric": primary_metric or str(metrics.get("primary_metric", "") or "score"),
        "n_predictions": len(predictions),
        "n_surface_calls": len(calls),
        "method_rows": method_rows,
        "slice_rows": slice_rows,
        "calls_by_role": dict(sorted(calls_by_role.items())),
        "calls_by_surface": dict(sorted(calls_by_surface.items())),
        "roles": roles,
        "oracle": oracle,
        "metrics": metrics,
        "manifest_id": str(manifest.get("experiment_id", "") or ""),
    }


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(payload), indent=2, sort_keys=False) + "\n", encoding="utf-8")


def write_markdown(path: Path, summary: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    metric_name = str(summary.get("primary_metric", "score") or "score")
    lines = [
        "# Runtime v1 Results Summary",
        "",
        f"- Experiment: `{summary.get('experiment_id')}`",
        f"- Experiment dir: `{summary.get('experiment_dir')}`",
        f"- Benchmark: `{dict(summary.get('benchmark') or {}).get('name', '')}`",
        f"- Primary metric: `{metric_name}`",
        f"- Predictions: `{summary.get('n_predictions', 0)}`",
        f"- Surface calls: `{summary.get('n_surface_calls', 0)}`",
        "",
        "## Method Matrix",
        "",
        "| Method | n | Mean score | Calls | Prompt toks | Completion toks | Failures |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in list(summary.get("method_rows") or []):
        if not isinstance(row, Mapping):
            continue
        lines.append(
            "| {label} | {n} | {score} | {calls} | {prompt:.1f} | {completion:.1f} | {failures} |".format(
                label=str(row.get("label", "") or row.get("method_id", "")),
                n=int(row.get("n_predictions", 0) or 0),
                score=_pct(float(row.get("mean_score", 0.0) or 0.0)),
                calls=int(row.get("surface_calls", 0) or 0),
                prompt=float(row.get("mean_prompt_tokens", 0.0) or 0.0),
                completion=float(row.get("mean_completion_tokens", 0.0) or 0.0),
                failures=int(row.get("failures", 0) or 0),
            )
        )
    lines.extend(
        [
            "",
            "## Slice Matrix",
            "",
            "| Phase | Task | Max length | Method | n | Mean score |",
            "|---|---|---:|---|---:|---:|",
        ]
    )
    for row in list(summary.get("slice_rows") or []):
        if not isinstance(row, Mapping):
            continue
        lines.append(
            "| {phase} | {task} | {length} | {method} | {n} | {score} |".format(
                phase=str(row.get("phase_id", "")),
                task=str(row.get("task_id", "")),
                length=int(row.get("max_seq_length", 0) or 0),
                method=_method_label(str(row.get("method_id", "") or "")),
                n=int(row.get("n_predictions", 0) or 0),
                score=_pct(float(row.get("mean_score", 0.0) or 0.0)),
            )
        )
    lines.extend(
        [
            "",
            "## Call Roles",
            "",
        ]
    )
    for role, count in dict(summary.get("calls_by_role") or {}).items():
        lines.append(f"- `{role}`: `{count}`")
    lines.extend(["", "## Method Notes", ""])
    for row in list(summary.get("method_rows") or []):
        if not isinstance(row, Mapping):
            continue
        description = str(row.get("description", "") or "")
        if description:
            lines.append(f"- **{row.get('label')}**: {description}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize runtime v1 results for paper tables.")
    parser.add_argument("--experiment-dir", type=Path, default=None, help="Runtime experiment directory.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Report output dir. Defaults to <experiment-dir>/paper_summary.",
    )
    parser.add_argument("--json-name", default="runtime_v1_summary.json")
    parser.add_argument("--md-name", default="runtime_v1_summary.md")
    parser.add_argument("--print-json", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.experiment_dir is None:
        raise SystemExit("missing --experiment-dir")
    run_dir = Path(args.experiment_dir).expanduser().resolve()
    output_dir = (
        Path(args.output_dir).expanduser().resolve()
        if args.output_dir is not None
        else run_dir / "paper_summary"
    )
    summary = build_summary(run_dir)
    json_path = output_dir / str(args.json_name)
    md_path = output_dir / str(args.md_name)
    _write_json(json_path, summary)
    write_markdown(md_path, summary)
    merge_artifacts(
        run_dir,
        {
            "runtime_v1_summary_json": str(json_path),
            "runtime_v1_summary_md": str(md_path),
        },
    )
    if args.print_json:
        print(json.dumps(summary, indent=2, default=str))
    else:
        print(f"Wrote {json_path}")
        print(f"Wrote {md_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
