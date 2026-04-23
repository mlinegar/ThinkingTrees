#!/usr/bin/env python3
"""Build a compact sanity artifact for the historical RILE GEPA control.

The default input is the strongest existing control run:

    outputs/manifesto_nested_20260417_045842/dspy_gemma31b_v3_200

It records the exact config, GEPA budget observed in the log, elapsed wall time
from log timestamps, baseline/final validation MAE, and the optimized prompt.
"""

from __future__ import annotations

import argparse
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any


DEFAULT_RUN_DIR = Path("outputs/manifesto_nested_20260417_045842/dspy_gemma31b_v3_200")
DEFAULT_OUTPUT_JSON = Path("docs/rile_gepa_control_sanity_2026-04-21.json")
DEFAULT_OUTPUT_MD = Path("docs/rile_gepa_control_sanity_2026-04-21.md")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--output-json", type=Path, default=DEFAULT_OUTPUT_JSON)
    parser.add_argument("--output-md", type=Path, default=DEFAULT_OUTPUT_MD)
    return parser.parse_args()


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _maybe_read_json(path: Path) -> dict[str, Any]:
    return _read_json(path) if path.exists() else {}


def _extract_metric(summary: dict[str, Any], name: str) -> float | None:
    metrics = summary.get("result", {}).get("metrics", {})
    if name in metrics:
        return float(metrics[name])
    if name == "zero_shot_val_mae_raw":
        root = summary.get("baseline_val", {})
        if "mae_raw" in root:
            return float(root["mae_raw"])
    if name in {"val_mae_raw", "val_mae", "mae_raw"}:
        root = summary.get("final_val", {})
        if "mae_raw" in root:
            return float(root["mae_raw"])
    root = summary.get("final_val" if name.startswith("val_") else "baseline_val", {})
    if name in root:
        return float(root[name])
    return None


def _extract_prompt(compiled: dict[str, Any]) -> str:
    candidates = [
        compiled.get("predict", {}).get("signature", {}).get("instructions"),
        compiled.get("signature", {}).get("instructions"),
    ]
    for candidate in candidates:
        if candidate:
            return str(candidate)
    return ""


def _parse_log(log_path: Path) -> dict[str, Any]:
    if not log_path.exists():
        return {}
    text = log_path.read_text(encoding="utf-8", errors="replace")
    out: dict[str, Any] = {}
    metric_calls = re.search(r"Running GEPA for approx ([0-9,]+) metric calls", text)
    if metric_calls:
        out["gepa_metric_calls_approx"] = int(metric_calls.group(1).replace(",", ""))
    pareto = re.search(r"Using ([0-9,]+) examples for tracking Pareto scores", text)
    if pareto:
        out["gepa_pareto_val_examples"] = int(pareto.group(1).replace(",", ""))
    timestamps = re.findall(r"^(\d{4}/\d{2}/\d{2} \d{2}:\d{2}:\d{2})", text, flags=re.MULTILINE)
    if timestamps:
        start = datetime.strptime(timestamps[0], "%Y/%m/%d %H:%M:%S")
        end = datetime.strptime(timestamps[-1], "%Y/%m/%d %H:%M:%S")
        out["log_start"] = timestamps[0]
        out["log_end"] = timestamps[-1]
        out["elapsed_seconds_from_log"] = max(0.0, (end - start).total_seconds())
    best = re.findall(r"Best score on valset: ([0-9.]+)", text)
    if best:
        out["best_gepa_val_score"] = float(best[-1])
    return out


def build_artifact(run_dir: Path) -> dict[str, Any]:
    run_dir = run_dir.expanduser()
    summary = _read_json(run_dir / "results" / "dspy_rile_summary.json")
    config = _maybe_read_json(run_dir.with_suffix(".config.json"))
    compiled = _maybe_read_json(run_dir / "training" / "dspy_rile" / "compiled_module.json")
    log_info = _parse_log(run_dir.with_suffix(".log"))

    params = config.get("params", {})
    baseline_mae = _extract_metric(summary, "zero_shot_val_mae_raw")
    final_mae = _extract_metric(summary, "val_mae_raw")
    improvement = None
    if baseline_mae is not None and final_mae is not None:
        improvement = baseline_mae - final_mae

    prompt = _extract_prompt(compiled)
    return {
        "run_dir": str(run_dir),
        "config_path": str(run_dir.with_suffix(".config.json")),
        "log_path": str(run_dir.with_suffix(".log")),
        "summary_path": str(run_dir / "results" / "dspy_rile_summary.json"),
        "model_name": params.get("model_name") or summary.get("model_name"),
        "api_base": params.get("api_base") or summary.get("api_base"),
        "prepared_dataset_path": params.get("prepared_dataset_path"),
        "optimizer": params.get("optimizer"),
        "gepa_auto": params.get("gepa_auto"),
        "gepa_valset_cap": params.get("gepa_valset_cap"),
        "gepa_reflection_minibatch_size": params.get("gepa_reflection_minibatch_size"),
        "max_tokens": params.get("max_tokens"),
        "n_train": summary.get("n_train") or summary.get("result", {}).get("metrics", {}).get("n_train"),
        "n_val": summary.get("n_val") or summary.get("result", {}).get("metrics", {}).get("n_val"),
        "baseline_val_mae": baseline_mae,
        "final_val_mae": final_mae,
        "mae_improvement": improvement,
        "acceptance_mae_improvement_threshold": 15.0,
        "acceptance_pass": bool(improvement is not None and improvement >= 15.0),
        "log": log_info,
        "optimized_prompt": prompt,
        "optimized_prompt_preview": prompt[:1200],
    }


def render_markdown(artifact: dict[str, Any]) -> str:
    log = artifact.get("log", {})
    prompt = str(artifact.get("optimized_prompt", "") or "")
    prompt_excerpt = prompt[:2000] + ("..." if len(prompt) > 2000 else "")
    return "\n".join(
        [
            "# RILE GEPA Control Sanity Artifact",
            "",
            "This artifact preserves the historical RILE GEPA control used to diagnose",
            "why GEPA worked in the old RILE setup but not in the later Benoit runs.",
            "",
            "## Run",
            "",
            f"- Run dir: `{artifact['run_dir']}`",
            f"- Model: `{artifact.get('model_name')}`",
            f"- API base: `{artifact.get('api_base')}`",
            f"- Dataset: `{artifact.get('prepared_dataset_path')}`",
            f"- Optimizer: `{artifact.get('optimizer')}` / auto `{artifact.get('gepa_auto')}`",
            f"- GEPA metric calls approx: `{log.get('gepa_metric_calls_approx', 'n/a')}`",
            f"- GEPA val cap: `{artifact.get('gepa_valset_cap')}`",
            f"- Pareto val examples observed: `{log.get('gepa_pareto_val_examples', 'n/a')}`",
            f"- Reflection minibatch: `{artifact.get('gepa_reflection_minibatch_size')}`",
            f"- Max tokens: `{artifact.get('max_tokens')}`",
            f"- Log elapsed seconds: `{log.get('elapsed_seconds_from_log', 'n/a')}`",
            "",
            "## Validation MAE",
            "",
            f"- Baseline MAE: `{artifact.get('baseline_val_mae')}`",
            f"- Final MAE: `{artifact.get('final_val_mae')}`",
            f"- Improvement: `{artifact.get('mae_improvement')}`",
            f"- Acceptance threshold: `{artifact.get('acceptance_mae_improvement_threshold')}`",
            f"- Acceptance pass: `{artifact.get('acceptance_pass')}`",
            "",
            "## Optimized Prompt Excerpt",
            "",
            "```text",
            prompt_excerpt,
            "```",
            "",
            "## Interpretation",
            "",
            "This control used one excerpt scorer, direct RILE labels, rich directional",
            "GEPA feedback, and a small GEPA validation surface. It should be treated as",
            "the control contract before comparing against Benoit scorer-only or",
            "full-pipeline GEPA.",
            "",
        ]
    )


def main() -> int:
    args = parse_args()
    artifact = build_artifact(args.run_dir)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(artifact, indent=2), encoding="utf-8")
    args.output_md.write_text(render_markdown(artifact), encoding="utf-8")
    print(json.dumps({"json": str(args.output_json), "markdown": str(args.output_md)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
