#!/usr/bin/env python3
"""Run a compact sbijax estimator grid on the Markov exact-sketch task."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


REPO = Path(__file__).resolve().parents[1]
PROBE = REPO / "scripts" / "probe_contextual_sbijax.py"
NO_PROBE = REPO / "scripts" / "probe_clean_unified_no.py"


DEFAULT_CANDIDATES = [
    "exact_zero_markov",
    "identity_theta",
    "learned_local_laws",
    "package_nasss",
    "package_nass",
    "posterior_npe_mdn",
    "posterior_fmpe_cnf",
    "posterior_cmpe_cm",
    "no_fno_full_doc",
    "no_clean_oneleaf",
    "no_clean_fg",
]

NO_CANDIDATES = {"no_fno_full_doc", "no_clean_oneleaf", "no_clean_fg"}


def _ts() -> str:
    return datetime.now(UTC).strftime("%Y%m%d_%H%M%S")


def _parse_int_list(raw: str) -> list[int]:
    values = [int(part.strip()) for part in str(raw).split(",") if part.strip()]
    if not values:
        raise ValueError("expected at least one integer")
    return values


def _candidate_args(candidate: str) -> list[str]:
    if candidate == "exact_zero_markov":
        return ["--sbijax-trainer", "exact_zero_markov", "--sbijax-method", "nass"]
    if candidate == "identity_theta":
        return ["--sbijax-trainer", "identity_theta", "--sbijax-method", "nass"]
    if candidate == "learned_local_laws":
        return ["--sbijax-trainer", "learned_local_laws", "--sbijax-method", "nass"]
    if candidate == "learned_local_laws_affine":
        return [
            "--sbijax-trainer",
            "learned_local_laws",
            "--sbijax-method",
            "nass",
            "--local-law-summary-family",
            "affine_probe",
        ]
    if candidate == "package_nasss":
        return ["--sbijax-trainer", "package", "--sbijax-method", "nasss"]
    if candidate == "package_nass":
        return ["--sbijax-trainer", "package", "--sbijax-method", "nass"]
    if candidate.startswith("posterior_"):
        parts = candidate.split("_")
        if len(parts) != 3:
            raise ValueError(
                "posterior candidates must be posterior_<estimator>_<density>"
            )
        return [
            "--sbijax-trainer",
            "posterior",
            "--sbijax-method",
            "nass",
            "--posterior-estimator",
            parts[1],
            "--density-family",
            parts[2],
        ]
    raise ValueError(f"unknown candidate {candidate!r}")


def _candidate_backend(candidate: str) -> str:
    return "clean_no" if str(candidate) in NO_CANDIDATES else "sbijax"


def _get_nested(payload: dict[str, Any], path: list[str], default: Any = None) -> Any:
    cur: Any = payload
    for key in path:
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


def _row_from_payload(
    *,
    candidate: str,
    train_docs: int,
    run_dir: Path,
    payload: dict[str, Any],
    returncode: int,
) -> dict[str, Any]:
    test_diag = _get_nested(payload, ["diagnostics", "test"], {}) or {}
    exact = _get_nested(payload, ["diagnostics", "exact_root_witness", "test"], {}) or {}
    oracle = _get_nested(
        payload,
        ["diagnostics", "markov_exact_sketch_oracle", "test"],
        {},
    ) or {}
    metrics = payload.get("metric_summary", {}) if isinstance(payload, dict) else {}
    provenance = payload.get("provenance", {}) if isinstance(payload, dict) else {}
    history = payload.get("history", []) if isinstance(payload, dict) else []
    last_history = history[-1] if isinstance(history, list) and history else {}
    return {
        "candidate": candidate,
        "backend": "sbijax",
        "metric_kind": "contextual_response_raw_mae",
        "test_mae": metrics.get(
            "contextual_raw_mae",
            test_diag.get("contextual_raw_mae"),
        ),
        "train_docs": int(train_docs),
        "status": str(payload.get("status", "process_error")),
        "returncode": int(returncode),
        "run_dir": str(run_dir),
        "trainer": provenance.get("trainer"),
        "sbijax_class": provenance.get("sbijax_class"),
        "posterior_estimator": provenance.get("posterior_estimator"),
        "density_family": provenance.get("density_family"),
        "downstream_readout": provenance.get("downstream_readout"),
        "decoder_kind": metrics.get("decoder_kind", provenance.get("decoder_kind")),
        "exact_zero_claim": metrics.get(
            "exact_zero_claim",
            provenance.get("exact_zero_claim"),
        ),
        "baseline_role": metrics.get("baseline_role", provenance.get("baseline_role")),
        "test_contextual_mae": metrics.get(
            "contextual_mae",
            test_diag.get("contextual_mae"),
        ),
        "test_contextual_raw_mae": metrics.get(
            "contextual_raw_mae",
            test_diag.get("contextual_raw_mae"),
        ),
        "test_theta_mae": metrics.get("theta_mae", test_diag.get("theta_mae")),
        "test_theta_mse": test_diag.get("theta_mse"),
        "test_theta_count_raw_mae": metrics.get(
            "raw_count_mae",
            test_diag.get("theta_count_raw_mae"),
        ),
        "test_first_acc": metrics.get(
            "first_accuracy",
            test_diag.get("theta_first_regime_accuracy"),
        ),
        "test_last_acc": metrics.get(
            "last_accuracy",
            test_diag.get("theta_last_regime_accuracy"),
        ),
        "test_posterior_std_mean": test_diag.get("posterior_std_mean"),
        "exact_oracle_mae": metrics.get(
            "exact_oracle_mae",
            oracle.get("contextual_mae"),
        ),
        "exact_root_mae": metrics.get("root_witness_mae", exact.get("root_mae")),
        "law_set_id": metrics.get("law_set_id", provenance.get("law_set_id")),
        "eps_leaf": metrics.get("eps_leaf", test_diag.get("eps_leaf")),
        "eps_merge": metrics.get("eps_merge", test_diag.get("eps_merge")),
        "eps_idemp": metrics.get("eps_idemp", test_diag.get("eps_idemp")),
        "last_train_loss": last_history.get("train_loss"),
        "last_val_loss": last_history.get("val_loss"),
    }


def _row_from_no_payload(
    *,
    candidate: str,
    train_docs: int,
    run_dir: Path,
    payload: dict[str, Any],
    returncode: int,
) -> dict[str, Any]:
    args = payload.get("args", {}) if isinstance(payload, dict) else {}
    learned = _get_nested(
        payload,
        ["learned_prediction_diagnostics", "test"],
        {},
    ) or {}
    baselines = payload.get("diagnostic_baselines", {}) if isinstance(payload, dict) else {}
    fno_vanilla = dict(baselines.get("fno_vanilla") or {})
    fno_vanilla_test = dict(fno_vanilla.get("test") or {})
    exact = _get_nested(
        payload,
        ["exact_palette_block_witness", "test"],
        {},
    ) or {}
    surface = _get_nested(
        payload,
        ["exact_surface_contract", "diagnostics"],
        {},
    ) or {}
    history = payload.get("history", []) if isinstance(payload, dict) else []
    last_history = history[-1] if isinstance(history, list) and history else {}
    if candidate == "no_fno_full_doc":
        selected = fno_vanilla_test
        metric_kind = "full_doc_fno_root_mae"
        baseline_role = "pure_full_doc_fno_no_tree"
        decoder_kind = "learned_fno"
    elif candidate == "no_clean_oneleaf":
        selected = learned
        metric_kind = "clean_unified_no_oneleaf_root_mae"
        baseline_role = "clean_unified_no_oneleaf_no_merge"
        decoder_kind = "learned_clean_no_g_leaf_f"
    else:
        selected = learned
        metric_kind = "clean_unified_no_fg_tree_root_mae"
        baseline_role = "clean_unified_no_fg_tree"
        decoder_kind = "learned_clean_no_fg"
    return {
        "candidate": candidate,
        "backend": "clean_no",
        "metric_kind": metric_kind,
        "test_mae": selected.get("root_mae"),
        "train_docs": int(train_docs),
        "status": str(payload.get("status", "ok" if returncode == 0 else "process_error")),
        "returncode": int(returncode),
        "run_dir": str(run_dir),
        "trainer": "clean_no",
        "sbijax_class": None,
        "posterior_estimator": None,
        "density_family": None,
        "downstream_readout": baseline_role,
        "decoder_kind": decoder_kind,
        "exact_zero_claim": False,
        "baseline_role": baseline_role,
        "n_leaves_per_doc": payload.get("n_leaves_per_doc"),
        "leaf_tokens": args.get("leaf_tokens"),
        "channels": args.get("channels"),
        "g_n_modes": args.get("g_n_modes"),
        "test_root_mae": selected.get("root_mae"),
        "test_root_mse": selected.get("root_mse"),
        "test_pred_std": selected.get("pred_std"),
        "test_pred_truth_corr": selected.get("pred_truth_corr"),
        "test_contextual_mae": None,
        "test_contextual_raw_mae": None,
        "test_theta_mae": None,
        "test_theta_mse": None,
        "test_theta_count_raw_mae": None,
        "test_first_acc": None,
        "test_last_acc": None,
        "test_posterior_std_mean": None,
        "exact_oracle_mae": None,
        "exact_root_mae": exact.get("mae"),
        "law_set_id": None,
        "eps_leaf": None,
        "eps_merge": None,
        "eps_idemp": None,
        "exact_surface_test_root_mae": surface.get("root_mae"),
        "last_train_loss": last_history.get("train_loss"),
        "last_val_loss": last_history.get("val_root_mae"),
    }


def _format_cell(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def _write_markdown(path: Path, rows: list[dict[str, Any]]) -> None:
    columns = [
        "candidate",
        "backend",
        "train_docs",
        "status",
        "metric_kind",
        "test_mae",
        "decoder_kind",
        "exact_zero_claim",
        "sbijax_class",
        "density_family",
        "n_leaves_per_doc",
        "leaf_tokens",
        "test_root_mae",
        "test_pred_std",
        "test_pred_truth_corr",
        "test_contextual_raw_mae",
        "test_theta_mae",
        "test_theta_count_raw_mae",
        "test_first_acc",
        "test_last_acc",
        "exact_oracle_mae",
        "law_set_id",
        "eps_leaf",
        "eps_merge",
        "eps_idemp",
        "test_posterior_std_mean",
        "exact_root_mae",
        "baseline_role",
    ]
    lines = [
        "# sbijax Markov Estimator Grid",
        "",
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(_format_cell(row.get(col)) for col in columns) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run sbijax estimator x data-size grid on Markov exact-sketch rows."
    )
    parser.add_argument("--train-docs", default="256,1024,4096")
    parser.add_argument("--eval-docs", type=int, default=64)
    parser.add_argument("--doc-tokens", type=int, default=24)
    parser.add_argument("--leaf-tokens", type=int, default=24)
    parser.add_argument("--fragment-len", type=int, default=6)
    parser.add_argument("--context-samples-per-doc", type=int, default=1)
    parser.add_argument("--response-signature-contexts", type=int, default=3)
    parser.add_argument("--response-signature-slices", type=int, default=2)
    parser.add_argument("--n-iter", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--embedding-dim", type=int, default=8)
    parser.add_argument("--state-dim", type=int, default=5)
    parser.add_argument("--hidden-dim", type=int, default=32)
    parser.add_argument("--posterior-samples", type=int, default=16)
    parser.add_argument("--posterior-eval-samples", type=int, default=16)
    parser.add_argument("--posterior-eval-batch-size", type=int, default=64)
    parser.add_argument("--density-components", type=int, default=5)
    parser.add_argument(
        "--no-epochs",
        type=int,
        default=25,
        help="Epochs for clean NO/FNO comparison candidates.",
    )
    parser.add_argument(
        "--no-batch-size",
        type=int,
        default=0,
        help="Batch size for clean NO/FNO candidates. 0 reuses --batch-size.",
    )
    parser.add_argument("--no-channels", type=int, default=32)
    parser.add_argument("--no-g-n-modes", type=int, default=4)
    parser.add_argument("--no-g-n-layers", type=int, default=2)
    parser.add_argument("--no-scorer-n-modes", type=int, default=4)
    parser.add_argument("--no-scorer-n-layers", type=int, default=2)
    parser.add_argument("--no-lr", type=float, default=3e-4)
    parser.add_argument(
        "--no-device",
        default="cuda",
        choices=["cuda", "cpu"],
        help="Device for clean NO/FNO candidates.",
    )
    parser.add_argument(
        "--no-fg-leaf-tokens",
        type=int,
        default=0,
        help="Leaf size for no_clean_fg. 0 uses --fragment-len.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--input-encoding",
        default="markov_exact_sketch",
        choices=[
            "normalized_token_ids",
            "one_hot_token_ids",
            "regime_ids",
            "regime_one_hot",
            "markov_exact_sketch",
        ],
    )
    parser.add_argument(
        "--candidates",
        default=",".join(DEFAULT_CANDIDATES),
        help=(
            "Comma-separated candidates. Built-ins: "
            + ", ".join(DEFAULT_CANDIDATES)
            + "; posterior candidates use posterior_<estimator>_<density>."
        ),
    )
    parser.add_argument("--output-root", default=None)
    parser.add_argument("--fail-fast", action="store_true")
    return parser


def _sbijax_command(
    args: argparse.Namespace,
    *,
    candidate: str,
    train_docs: int,
    run_dir: Path,
) -> list[str]:
    return [
        sys.executable,
        str(PROBE),
        "--data-source",
        "markov",
        "--sbijax-package-theta",
        "markov_exact_sketch",
        "--sbijax-input-encoding",
        str(args.input_encoding),
        "--doc-tokens",
        str(args.doc_tokens),
        "--leaf-tokens",
        str(args.leaf_tokens),
        "--train-docs",
        str(train_docs),
        "--eval-docs",
        str(args.eval_docs),
        "--fragment-len",
        str(args.fragment_len),
        "--context-samples-per-doc",
        str(args.context_samples_per_doc),
        "--response-signature-contexts",
        str(args.response_signature_contexts),
        "--response-signature-slices",
        str(args.response_signature_slices),
        "--n-iter",
        str(args.n_iter),
        "--batch-size",
        str(args.batch_size),
        "--embedding-dim",
        str(args.embedding_dim),
        "--state-dim",
        str(args.state_dim),
        "--hidden-dim",
        str(args.hidden_dim),
        "--posterior-samples",
        str(args.posterior_samples),
        "--posterior-eval-samples",
        str(args.posterior_eval_samples),
        "--posterior-eval-batch-size",
        str(args.posterior_eval_batch_size),
        "--density-components",
        str(args.density_components),
        "--seed",
        str(args.seed + int(train_docs)),
        "--output-root",
        str(run_dir),
        *_candidate_args(candidate),
    ]


def _no_command(
    args: argparse.Namespace,
    *,
    candidate: str,
    train_docs: int,
    run_dir: Path,
) -> list[str]:
    if candidate == "no_clean_fg":
        leaf_tokens = (
            int(args.no_fg_leaf_tokens)
            if int(args.no_fg_leaf_tokens) > 0
            else int(args.fragment_len)
        )
    else:
        leaf_tokens = int(args.doc_tokens)
    diagnostic_baselines = "fno_vanilla" if candidate == "no_fno_full_doc" else "none"
    return [
        sys.executable,
        str(NO_PROBE),
        "--doc-tokens",
        str(args.doc_tokens),
        "--leaf-tokens",
        str(leaf_tokens),
        "--train-docs",
        str(train_docs),
        "--eval-docs",
        str(args.eval_docs),
        "--epochs",
        str(args.no_epochs),
        "--batch-size",
        str(
            int(args.no_batch_size)
            if int(args.no_batch_size) > 0
            else int(args.batch_size)
        ),
        "--channels",
        str(args.no_channels),
        "--g-n-modes",
        str(args.no_g_n_modes),
        "--g-n-layers",
        str(args.no_g_n_layers),
        "--scorer-n-modes",
        str(args.no_scorer_n_modes),
        "--scorer-n-layers",
        str(args.no_scorer_n_layers),
        "--lr",
        str(args.no_lr),
        "--seed",
        str(args.seed + int(train_docs)),
        "--device",
        str(args.no_device),
        "--output-root",
        str(run_dir),
        "--diagnostic-baselines",
        diagnostic_baselines,
        "--diagnostic-baseline-epochs",
        str(args.no_epochs),
    ]


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    output_root = (
        Path(args.output_root)
        if args.output_root is not None
        else REPO / "outputs" / f"contextual_sbijax_estimator_grid_{_ts()}"
    )
    output_root.mkdir(parents=True, exist_ok=True)

    train_docs_values = _parse_int_list(args.train_docs)
    candidates = [part.strip() for part in str(args.candidates).split(",") if part.strip()]
    rows: list[dict[str, Any]] = []

    for train_docs in train_docs_values:
        for candidate in candidates:
            run_dir = output_root / f"train{train_docs}_{candidate}"
            backend = _candidate_backend(candidate)
            if backend == "clean_no":
                cmd = _no_command(
                    args,
                    candidate=candidate,
                    train_docs=int(train_docs),
                    run_dir=run_dir,
                )
            else:
                cmd = _sbijax_command(
                    args,
                    candidate=candidate,
                    train_docs=int(train_docs),
                    run_dir=run_dir,
                )
            proc = subprocess.run(
                cmd,
                cwd=str(REPO),
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
            )
            summary_path = run_dir / "summary.json"
            if summary_path.exists():
                payload = json.loads(summary_path.read_text(encoding="utf-8"))
            else:
                payload = {
                    "status": "process_error",
                    "error": proc.stderr[-4000:],
                    "stdout_tail": proc.stdout[-4000:],
                }
            if backend == "clean_no":
                row = _row_from_no_payload(
                    candidate=candidate,
                    train_docs=int(train_docs),
                    run_dir=run_dir,
                    payload=payload,
                    returncode=int(proc.returncode),
                )
            else:
                row = _row_from_payload(
                    candidate=candidate,
                    train_docs=int(train_docs),
                    run_dir=run_dir,
                    payload=payload,
                    returncode=int(proc.returncode),
                )
            rows.append(row)
            if proc.returncode != 0 and bool(args.fail_fast):
                break

    summary = {
        "status": "ok" if all(row["returncode"] == 0 for row in rows) else "partial",
        "args": vars(args),
        "rows": rows,
    }
    (output_root / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _write_markdown(output_root / "report.md", rows)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if summary["status"] == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
