#!/usr/bin/env python3
"""
Train neural operator families with one command.

This orchestrator runs:
1) CTreePO operator training (`scripts/train_ctreepo.py`)
2) Mergeable embedding sketch training (`scripts/train_rile_embedding_sketch.py`)

Use `--ctreepo-args` and `--mergeable-args` for full passthrough flexibility.
"""

from __future__ import annotations

import argparse
import json
import logging
import shlex
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional


PROJECT_ROOT = Path(__file__).resolve().parent.parent
logger = logging.getLogger(__name__)


def _read_json_if_exists(path: Path) -> Dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    return payload if isinstance(payload, dict) else None


def _resolve_output_dir(raw: str | None) -> Path:
    if raw:
        out = Path(raw).expanduser()
        if not out.is_absolute():
            out = (PROJECT_ROOT / out).resolve()
        return out
    run_id = datetime.now().strftime("neural_operators_%Y%m%d_%H%M%S")
    return (PROJECT_ROOT / "outputs" / run_id).resolve()


def _run_command(label: str, cmd: List[str], log_path: Path) -> Dict[str, Any]:
    logger.info("[%s] running: %s", label, " ".join(shlex.quote(x) for x in cmd))
    started = datetime.now().isoformat()
    proc = subprocess.run(cmd, cwd=str(PROJECT_ROOT), capture_output=True, text=True)
    ended = datetime.now().isoformat()

    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(
        "\n".join(
            [
                f"label: {label}",
                f"started_at: {started}",
                f"ended_at: {ended}",
                f"returncode: {proc.returncode}",
                "",
                "=== STDOUT ===",
                proc.stdout or "",
                "",
                "=== STDERR ===",
                proc.stderr or "",
            ]
        ),
        encoding="utf-8",
    )

    if proc.returncode != 0:
        logger.error("[%s] failed (code=%d), see %s", label, proc.returncode, log_path)
    else:
        logger.info("[%s] completed successfully", label)
    return {
        "label": label,
        "returncode": int(proc.returncode),
        "log": str(log_path),
        "started_at": started,
        "ended_at": ended,
    }


def _detect_artifacts(label: str, run_dir: Path) -> Dict[str, Any]:
    """
    Detect primary model artifact paths for each operator family.
    """
    if label == "ctreepo":
        best_path = run_dir / "best.pt"
        final_path = run_dir / "final.pt"
        training_result = run_dir / "training_result.json"
        training_payload = _read_json_if_exists(training_result) or {}
        local_law_summary = (
            training_payload.get("local_law_summary")
            if isinstance(training_payload.get("local_law_summary"), dict)
            else None
        )
        compositional_learning_problem = (
            training_payload.get("compositional_learning_problem")
            if isinstance(training_payload.get("compositional_learning_problem"), dict)
            else (
                local_law_summary.get("compositional_learning_problem")
                if isinstance(local_law_summary, dict)
                and isinstance(local_law_summary.get("compositional_learning_problem"), dict)
                else None
            )
        )
        return {
            "primary_model_path": str(best_path) if best_path.exists() else (str(final_path) if final_path.exists() else None),
            "best_model_path": str(best_path) if best_path.exists() else None,
            "final_model_path": str(final_path) if final_path.exists() else None,
            "training_result_path": str(training_result) if training_result.exists() else None,
            "local_law_summary": local_law_summary,
            "compositional_learning_problem": compositional_learning_problem,
        }
    if label == "mergeable_sketch":
        best_path = run_dir / "checkpoint_best.pt"
        metrics_path = run_dir / "metrics.json"
        predictions_path = run_dir / "predictions.csv"
        return {
            "primary_model_path": str(best_path) if best_path.exists() else None,
            "best_model_path": str(best_path) if best_path.exists() else None,
            "metrics_path": str(metrics_path) if metrics_path.exists() else None,
            "predictions_path": str(predictions_path) if predictions_path.exists() else None,
        }
    return {
        "primary_model_path": None,
    }


def _build_ctreepo_local_law_config(args: argparse.Namespace) -> Dict[str, Any]:
    from src.training.local_law_oracles import normalize_local_law_oracle_spec

    oracle_spec = normalize_local_law_oracle_spec(args.ctreepo_local_law_oracle_spec)
    return {
        "root_weight": float(args.ctreepo_root_weight) if args.ctreepo_root_weight is not None else None,
        "leaf_audit_weight": (
            float(args.ctreepo_leaf_audit_weight) if args.ctreepo_leaf_audit_weight is not None else None
        ),
        "merge_audit_weight": (
            float(args.ctreepo_merge_audit_weight) if args.ctreepo_merge_audit_weight is not None else None
        ),
        "violation_threshold": (
            float(args.ctreepo_local_law_violation_threshold)
            if args.ctreepo_local_law_violation_threshold is not None
            else None
        ),
        "require_supervision": bool(args.ctreepo_require_local_law_supervision),
        "oracle_module": oracle_spec,
        "label_source_kind": (
            "task_oracle"
            if oracle_spec == "task"
            else "oracle_callback"
            if oracle_spec
            else "model_backed_teacher"
            if args.ctreepo_local_law_score_port is not None
            else "none"
        ),
        "teacher_port": int(args.ctreepo_local_law_score_port) if args.ctreepo_local_law_score_port is not None else None,
        "teacher_model": str(args.ctreepo_local_law_score_model).strip() if args.ctreepo_local_law_score_model else None,
        "score_port": int(args.ctreepo_local_law_score_port) if args.ctreepo_local_law_score_port is not None else None,
        "score_model": str(args.ctreepo_local_law_score_model).strip() if args.ctreepo_local_law_score_model else None,
        "teacher_max_tokens": (
            int(args.ctreepo_local_law_score_max_tokens)
            if args.ctreepo_local_law_score_max_tokens is not None
            else None
        ),
        "score_max_tokens": (
            int(args.ctreepo_local_law_score_max_tokens)
            if args.ctreepo_local_law_score_max_tokens is not None
            else None
        ),
        "teacher_temperature": (
            float(args.ctreepo_local_law_score_temperature)
            if args.ctreepo_local_law_score_temperature is not None
            else None
        ),
        "score_temperature": (
            float(args.ctreepo_local_law_score_temperature)
            if args.ctreepo_local_law_score_temperature is not None
            else None
        ),
        "allow_model_based_labeling": bool(args.ctreepo_allow_model_based_local_law_scoring),
        "allow_model_based_scoring": bool(args.ctreepo_allow_model_based_local_law_scoring),
    }


def _apply_ctreepo_local_law_args(cmd: List[str], config: Dict[str, Any]) -> None:
    if config.get("root_weight") is not None:
        cmd.extend(["--root-weight", str(float(config["root_weight"]))])
    if config.get("leaf_audit_weight") is not None:
        cmd.extend(["--leaf-audit-weight", str(float(config["leaf_audit_weight"]))])
    if config.get("merge_audit_weight") is not None:
        cmd.extend(["--merge-audit-weight", str(float(config["merge_audit_weight"]))])
    if config.get("violation_threshold") is not None:
        cmd.extend(["--local-law-violation-threshold", str(float(config["violation_threshold"]))])
    if config.get("oracle_module"):
        cmd.extend(["--local-law-oracle", str(config["oracle_module"])])
    if config.get("score_port") is not None:
        cmd.extend(["--local-law-teacher-port", str(int(config["score_port"]))])
    if config.get("score_model"):
        cmd.extend(["--local-law-teacher-model", str(config["score_model"])])
    if config.get("score_max_tokens") is not None:
        cmd.extend(["--local-law-teacher-max-tokens", str(int(config["score_max_tokens"]))])
    if config.get("score_temperature") is not None:
        cmd.extend(["--local-law-teacher-temperature", str(float(config["score_temperature"]))])
    if bool(config.get("require_supervision")):
        cmd.append("--require-local-law-supervision")
    if bool(config.get("allow_model_based_scoring")):
        cmd.append("--allow-model-based-local-law-labeling")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Train CTreePO and mergeable-sketch operators in one run.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--task", type=str, default="manifesto_rile")
    parser.add_argument(
        "--which",
        choices=["both", "ctreepo", "mergeable_sketch"],
        default="both",
        help="Which operator family to train.",
    )
    parser.add_argument("--embedding-url", type=str, default=None)
    parser.add_argument("--embedding-model", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--ctreepo-root-weight", type=float, default=None)
    parser.add_argument("--ctreepo-leaf-audit-weight", type=float, default=None)
    parser.add_argument("--ctreepo-merge-audit-weight", type=float, default=None)
    parser.add_argument("--ctreepo-local-law-violation-threshold", type=float, default=None)
    parser.add_argument(
        "--ctreepo-local-law-oracle",
        "--ctreepo-local-law-oracle-module",
        dest="ctreepo_local_law_oracle_spec",
        type=str,
        default=None,
        help=(
            "Node-span label source for CTreePO local-law labels. Use 'task' for the task/teacher-provided "
            "oracle, or module.path:function_name for an explicit callback."
        ),
    )
    parser.add_argument(
        "--ctreepo-local-law-teacher-port",
        "--ctreepo-local-law-score-port",
        dest="ctreepo_local_law_score_port",
        type=int,
        default=None,
        help="Optional model-backed teacher endpoint for node-span labels. Fallback only.",
    )
    parser.add_argument(
        "--ctreepo-local-law-teacher-model",
        "--ctreepo-local-law-score-model",
        dest="ctreepo_local_law_score_model",
        type=str,
        default=None,
        help="Optional model override for the model-backed teacher labeler.",
    )
    parser.add_argument(
        "--ctreepo-local-law-teacher-max-tokens",
        "--ctreepo-local-law-score-max-tokens",
        dest="ctreepo_local_law_score_max_tokens",
        type=int,
        default=None,
        help="Max tokens for model-backed teacher labeling.",
    )
    parser.add_argument(
        "--ctreepo-local-law-teacher-temperature",
        "--ctreepo-local-law-score-temperature",
        dest="ctreepo_local_law_score_temperature",
        type=float,
        default=None,
        help="Temperature for model-backed teacher labeling.",
    )
    parser.add_argument("--ctreepo-require-local-law-supervision", action="store_true")
    parser.add_argument(
        "--ctreepo-allow-model-based-local-law-labeling",
        "--ctreepo-allow-model-based-local-law-scoring",
        dest="ctreepo_allow_model_based_local_law_scoring",
        action="store_true",
        help="Explicitly allow model-backed teacher labeling for local-law supervision.",
    )
    parser.add_argument(
        "--ctreepo-args",
        type=str,
        default="--pilot",
        help="Extra args forwarded to scripts/train_ctreepo.py",
    )
    parser.add_argument(
        "--mergeable-args",
        type=str,
        default="",
        help="Extra args forwarded to scripts/train_rile_embedding_sketch.py",
    )
    parser.add_argument("--fail-fast", action="store_true", help="Stop after first failure.")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    output_dir = _resolve_output_dir(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    logs_dir = output_dir / "logs"
    logger.info("Output dir: %s", output_dir)
    if (
        args.ctreepo_local_law_oracle_spec
        and str(args.ctreepo_local_law_oracle_spec).strip().lower() != "task"
        and args.ctreepo_local_law_score_port is not None
    ):
        parser.error(
            "Choose one CTreePO local-law label source: --ctreepo-local-law-oracle "
            "or --ctreepo-local-law-teacher-port, not both."
        )
    ctreepo_local_law = _build_ctreepo_local_law_config(args)

    py = sys.executable
    common: List[str] = []
    if args.embedding_url:
        common.extend(["--embedding-url", str(args.embedding_url)])
    if args.embedding_model:
        common.extend(["--embedding-model", str(args.embedding_model)])
    if args.task:
        common.extend(["--task", str(args.task)])
    if args.seed is not None:
        common.extend(["--seed", str(int(args.seed))])

    runs: List[Dict[str, Any]] = []

    if args.which in {"both", "ctreepo"}:
        ctreepo_out = output_dir / "ctreepo"
        cmd = [py, "scripts/train_ctreepo.py", "--output-dir", str(ctreepo_out), *common]
        if args.ctreepo_args:
            cmd.extend(shlex.split(str(args.ctreepo_args)))
        _apply_ctreepo_local_law_args(cmd, ctreepo_local_law)
        result = _run_command("ctreepo", cmd, logs_dir / "ctreepo.log")
        result["run_dir"] = str(ctreepo_out)
        result["artifacts"] = _detect_artifacts("ctreepo", ctreepo_out)
        runs.append(result)
        if args.fail_fast and int(result["returncode"]) != 0:
            (output_dir / "summary.json").write_text(json.dumps({"runs": runs}, indent=2), encoding="utf-8")
            return int(result["returncode"])

    if args.which in {"both", "mergeable_sketch"}:
        merge_out = output_dir / "mergeable_sketch"
        cmd = [
            py,
            "scripts/train_rile_embedding_sketch.py",
            "--output-dir",
            str(merge_out),
            *common,
        ]
        if args.mergeable_args:
            cmd.extend(shlex.split(str(args.mergeable_args)))
        result = _run_command("mergeable_sketch", cmd, logs_dir / "mergeable_sketch.log")
        result["run_dir"] = str(merge_out)
        result["artifacts"] = _detect_artifacts("mergeable_sketch", merge_out)
        runs.append(result)
        if args.fail_fast and int(result["returncode"]) != 0:
            (output_dir / "summary.json").write_text(json.dumps({"runs": runs}, indent=2), encoding="utf-8")
            return int(result["returncode"])

    summary = {
        "created_at": datetime.now().isoformat(),
        "output_dir": str(output_dir),
        "task": str(args.task),
        "which": args.which,
        "common_args": common,
        "ctreepo_local_law": ctreepo_local_law,
        "runs": runs,
        "all_success": bool(all(int(r.get("returncode", 1)) == 0 for r in runs)),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    if summary["all_success"]:
        logger.info("All requested operator trainings completed successfully.")
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
