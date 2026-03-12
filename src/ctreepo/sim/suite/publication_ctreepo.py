from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import shlex
from typing import Dict, List, Sequence, Tuple

from src.ctreepo.sim.manifest import RunSpec, read_manifest_jsonl, write_manifest_jsonl
from src.ctreepo.sim.runner import read_cmds_file


def _utc_run_id(default: str | None = None) -> str:
    if default and str(default).strip():
        return str(default).strip()
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def _q(x: object) -> str:
    return shlex.quote(str(x))


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _thread_env_prefix() -> str:
    # Keep per-process BLAS/threading low for sweep stability (mirrors prior bash sweep).
    return " ".join(
        [
            "OMP_NUM_THREADS=1",
            "MKL_NUM_THREADS=1",
            "OPENBLAS_NUM_THREADS=1",
            "NUMEXPR_NUM_THREADS=1",
            "VECLIB_MAXIMUM_THREADS=1",
            "BLIS_NUM_THREADS=1",
        ]
    )


@dataclass(frozen=True)
class SuitePaths:
    output_root: Path
    suite_meta: Path
    suite_cmds: Path
    suite_manifest: Path
    lane_dir: Path


def _resolve_paths(output_root: Path) -> SuitePaths:
    return SuitePaths(
        output_root=output_root,
        suite_meta=output_root / "suite_meta.json",
        suite_cmds=output_root / "suite_cmds.txt",
        suite_manifest=output_root / "suite_manifest.jsonl",
        lane_dir=output_root / "suite_lanes",
    )


def _profile_defaults(profile: str) -> Dict[str, str]:
    if profile == "smoke":
        return {
            "seeds": "0",
            "q_rates": "0",
            "q_rates_upper": "0",
            "train_docs_lda": "128",
            "train_docs_hard": "128",
            "train_docs_hard_upper": "128",
            "leaf_tokens_lda": "32",
            "leaf_tokens_hard": "16",
            "cal_rates_lda": "0.1",
            "cal_rates_hard": "0.1",
            "cal_rates_upper": "0.1",
            "n_books_test_lda": "32",
            "n_books_test_hard": "32",
            "doc_tokens_lda": "256",
        }

    # publication
    return {
        "seeds": "0 1 2 3 4 5 6 7",
        "q_rates": "0 0.25 0.5",
        "q_rates_upper": "0 0.25",
        "train_docs_lda": "128 256 512 1024 2048 4096",
        "train_docs_hard": "128 256 512 1024 2048",
        "train_docs_hard_upper": "1024 2048 4096",
        "leaf_tokens_lda": "32 16 8",
        "leaf_tokens_hard": "16 8",
        "cal_rates_lda": "0 0.05 0.1",
        "cal_rates_hard": "0.05 0.1 0.2",
        "cal_rates_upper": "0.1",
        "n_books_test_lda": "4000",
        "n_books_test_hard": "5000",
        "doc_tokens_lda": "2048",
    }


def _parse_items(text: str) -> List[str]:
    out: List[str] = []
    for raw in str(text).replace(",", " ").split():
        x = raw.strip()
        if x:
            out.append(x)
    return out


def _parse_ints(text: str) -> List[int]:
    return [int(x) for x in _parse_items(text)]


def _lane_calls(*, output_root: Path, defaults: Dict[str, str]) -> List[Tuple[str, List[str], List[int]]]:
    """
    Return list of (lane_key, sweep_segmented_lda_ctreepo argv_base, fixed_leaf_tokens_grid).
    """

    out = []
    seeds = defaults["seeds"]
    q_rates = defaults["q_rates"]
    q_rates_upper = defaults["q_rates_upper"]
    train_docs_lda = defaults["train_docs_lda"]
    train_docs_hard = defaults["train_docs_hard"]
    train_docs_hard_upper = defaults["train_docs_hard_upper"]
    leaf_tokens_lda_grid = _parse_ints(defaults["leaf_tokens_lda"])
    leaf_tokens_hard_grid = _parse_ints(defaults["leaf_tokens_hard"])
    cal_rates_lda = defaults["cal_rates_lda"]
    cal_rates_hard = defaults["cal_rates_hard"]
    cal_rates_upper = defaults["cal_rates_upper"]
    n_books_test_lda = defaults["n_books_test_lda"]
    n_books_test_hard = defaults["n_books_test_hard"]
    doc_tokens_lda = defaults["doc_tokens_lda"]

    # LDA regime (k=8,v=512,bag_of_words).
    for lane_key, lane_args in [
        (
            "lda_lane_lda_direct",
            [
                "--topic-process",
                "bag_of_words",
                "--leaf-theta-estimator",
                "sklearn_lda",
                "--topic-phi-estimators",
                "sklearn_lda",
            ],
        ),
        (
            "lda_lane_phi_base",
            [
                "--topic-process",
                "bag_of_words",
                "--leaf-theta-estimator",
                "lstsq",
                "--topic-phi-estimators",
                "tensor_lda",
            ],
        ),
        (
            "lda_lane_neural_weak",
            [
                "--topic-process",
                "bag_of_words",
                "--leaf-theta-estimator",
                "lstsq",
                "--topic-phi-estimators",
                "neural_ctreepo",
                "--neural-topic-base-estimator",
                "tensor_lda",
                "--neural-topic-seed-fraction",
                "0.125",
                "--neural-topic-operator-boost",
                "0.6",
                "--neural-topic-seed-llm-min-weight",
                "0.02",
                "--neural-topic-seed-llm-max-weight",
                "0.15",
                "--neural-topic-mix-samples",
                "64",
            ],
        ),
        (
            "lda_lane_neural_default",
            [
                "--topic-process",
                "bag_of_words",
                "--leaf-theta-estimator",
                "lstsq",
                "--topic-phi-estimators",
                "neural_ctreepo",
                "--neural-topic-base-estimator",
                "tensor_lda",
                "--neural-topic-seed-fractions",
                "0.25 0.5",
                "--neural-topic-operator-boost",
                "1.0",
                "--neural-topic-seed-llm-min-weight",
                "0.10",
                "--neural-topic-seed-llm-max-weight",
                "0.35",
                "--neural-topic-mix-samples",
                "128",
            ],
        ),
    ]:
        lane_root = (
            output_root
            / "segmented_lda_ctreepo"
            / "equivalence"
            / "lda"
            / "k8_v512"
            / f"lane_{lane_key.split('lda_lane_')[-1]}"
        )
        out.append(
            (
                lane_key,
                [
                    "--output-root",
                    str(lane_root),
                    "--train-docs",
                    str(train_docs_lda),
                    "--n-books-test",
                    str(n_books_test_lda),
                    "--calibration-rates",
                    str(cal_rates_lda),
                    "--eval-leaf-rates",
                    str(q_rates),
                    "--eval-internal-rates",
                    str(q_rates),
                    "--topic-phi-docs",
                    "0",
                    "--n-topics",
                    "8",
                    "--vocab-size",
                    "512",
                    "--min-segments",
                    "1",
                    "--max-segments",
                    "1",
                    "--min-seg-tokens",
                    str(doc_tokens_lda),
                    "--max-seg-tokens",
                    str(doc_tokens_lda),
                    "--alpha-topic",
                    "0.20",
                    "--beta-word",
                    "0.10",
                    "--segment-concentration",
                    "80.0",
                    "--segment-background",
                    "2.0",
                    "--topic-phi-permute",
                    "--eval-internal-query-design",
                    "risk",
                    "--seeds",
                    str(seeds),
                    *lane_args,
                ],
                list(leaf_tokens_lda_grid),
            )
        )

    # Hard regime (k=12,v=1024,segments).
    for lane_key, lane_args, train_docs, cal_rates, q_grid in [
        (
            "hard_lane_phi_base",
            [
                "--topic-process",
                "segments",
                "--leaf-theta-estimator",
                "lstsq",
                "--topic-phi-estimators",
                "tensor_lda",
            ],
            train_docs_hard,
            cal_rates_hard,
            q_rates,
        ),
        (
            "hard_lane_neural_weak",
            [
                "--topic-process",
                "segments",
                "--leaf-theta-estimator",
                "lstsq",
                "--topic-phi-estimators",
                "neural_ctreepo",
                "--neural-topic-base-estimator",
                "tensor_lda",
                "--neural-topic-seed-fraction",
                "0.0833333333",
                "--neural-topic-operator-boost",
                "0.6",
                "--neural-topic-seed-llm-min-weight",
                "0.02",
                "--neural-topic-seed-llm-max-weight",
                "0.15",
                "--neural-topic-mix-samples",
                "64",
            ],
            train_docs_hard,
            cal_rates_hard,
            q_rates,
        ),
        (
            "hard_lane_neural_default",
            [
                "--topic-process",
                "segments",
                "--leaf-theta-estimator",
                "lstsq",
                "--topic-phi-estimators",
                "neural_ctreepo",
                "--neural-topic-base-estimator",
                "tensor_lda",
                "--neural-topic-seed-fractions",
                "0.2 0.35",
                "--neural-topic-operator-boost",
                "1.0",
                "--neural-topic-seed-llm-min-weight",
                "0.10",
                "--neural-topic-seed-llm-max-weight",
                "0.35",
                "--neural-topic-mix-samples",
                "128",
            ],
            train_docs_hard,
            cal_rates_hard,
            q_rates,
        ),
        (
            "hard_lane_neural_upper",
            [
                "--topic-process",
                "segments",
                "--leaf-theta-estimator",
                "lstsq",
                "--topic-phi-estimators",
                "neural_ctreepo",
                "--neural-topic-base-estimator",
                "tensor_lda",
                "--neural-topic-seed-fraction",
                "1.0",
                "--neural-topic-operator-boost",
                "1.4",
                "--neural-topic-seed-llm-min-weight",
                "0.35",
                "--neural-topic-seed-llm-max-weight",
                "0.85",
                "--neural-topic-mix-samples",
                "128",
            ],
            train_docs_hard_upper,
            cal_rates_upper,
            q_rates_upper,
        ),
    ]:
        lane_root = (
            output_root
            / "segmented_lda_ctreepo"
            / "equivalence"
            / "hard"
            / "k12_v1024"
            / f"lane_{lane_key.split('hard_lane_')[-1]}"
        )
        out.append(
            (
                lane_key,
                [
                    "--output-root",
                    str(lane_root),
                    "--train-docs",
                    str(train_docs),
                    "--n-books-test",
                    str(n_books_test_hard),
                    "--calibration-rates",
                    str(cal_rates),
                    "--eval-leaf-rates",
                    str(q_grid),
                    "--eval-internal-rates",
                    str(q_grid),
                    "--topic-phi-docs",
                    "0",
                    "--n-topics",
                    "12",
                    "--vocab-size",
                    "1024",
                    "--min-segments",
                    "10",
                    "--max-segments",
                    "12",
                    "--min-seg-tokens",
                    "16",
                    "--max-seg-tokens",
                    "32",
                    "--alpha-topic",
                    "0.35",
                    "--beta-word",
                    "0.40",
                    "--segment-concentration",
                    "18.0",
                    "--segment-background",
                    "6.0",
                    "--topic-phi-permute",
                    "--eval-internal-query-design",
                    "risk",
                    "--seeds",
                    str(seeds),
                    *lane_args,
                ],
                list(leaf_tokens_hard_grid),
            )
        )

    return out


def build_suite(
    *,
    run_id: str,
    profile: str,
    python_bin: str,
    output_root: Path,
    skip_existing: bool,
    set_thread_env: bool,
    device: str,
    cuda_device: int | None,
    torch_threads: int,
) -> dict:
    output_root = output_root.resolve()
    paths = _resolve_paths(output_root)
    paths.output_root.mkdir(parents=True, exist_ok=True)
    paths.lane_dir.mkdir(parents=True, exist_ok=True)

    defaults = _profile_defaults(str(profile))
    lane_calls = _lane_calls(output_root=output_root, defaults=defaults)

    from src.ctreepo.sim.cli.sweep_segmented_lda_ctreepo import (  # noqa: WPS433
        main as _ctree_sweep,
    )

    lane_cmds: Dict[str, Path] = {}
    lane_manifests: Dict[str, Path] = {}
    counts: Dict[str, int] = {}
    all_cmds: List[str] = []
    all_runs: List[RunSpec] = []
    env_prefix = _thread_env_prefix() if bool(set_thread_env) else ""

    for lane_key, argv_lane_base, leaf_tokens_grid in lane_calls:
        out_cmds = paths.lane_dir / f"{lane_key}_cmds.txt"
        out_manifest = paths.lane_dir / f"{lane_key}_manifest.jsonl"
        lane_cmds[lane_key] = out_cmds
        lane_manifests[lane_key] = out_manifest

        lane_cmd_lines: List[str] = []
        lane_runs: List[RunSpec] = []
        for lt in leaf_tokens_grid:
            tmp_cmds = paths.lane_dir / f"{lane_key}_lt{int(lt)}_cmds.tmp.txt"
            tmp_manifest = paths.lane_dir / f"{lane_key}_lt{int(lt)}_manifest.tmp.jsonl"
            sweep_argv = [
                "--python-bin",
                str(python_bin),
                "--out-cmds",
                str(tmp_cmds),
                "--out-manifest",
                str(tmp_manifest),
                "--skip-existing" if bool(skip_existing) else "--no-skip-existing",
                "--fixed-leaf-tokens",
                str(int(lt)),
                "--device",
                str(device),
                "--torch-threads",
                str(int(torch_threads)),
                *argv_lane_base,
            ]
            if cuda_device is not None:
                sweep_argv.extend(["--cuda-device", str(int(cuda_device))])
            _ctree_sweep(sweep_argv)
            lane_cmd_lines.extend(read_cmds_file(tmp_cmds) if tmp_cmds.exists() else [])
            lane_runs.extend(read_manifest_jsonl(tmp_manifest))

        if env_prefix:
            lane_cmd_lines = [f"{env_prefix} {c}" for c in lane_cmd_lines]
            lane_runs = [
                RunSpec(
                    id=r.id,
                    family=r.family,
                    config=dict(r.config),
                    outputs=dict(r.outputs),
                    command=f"{env_prefix} {r.command}",
                    requires=list(r.requires),
                    resources=dict(r.resources),
                )
                for r in lane_runs
            ]

        _write_text(out_cmds, "\n".join(lane_cmd_lines) + ("\n" if lane_cmd_lines else ""))
        write_manifest_jsonl(out_manifest, lane_runs)

        counts[lane_key] = int(len(lane_cmd_lines))
        all_cmds.extend(lane_cmd_lines)
        all_runs.extend(lane_runs)

    _write_text(paths.suite_cmds, "\n".join(all_cmds) + ("\n" if all_cmds else ""))
    write_manifest_jsonl(paths.suite_manifest, all_runs)

    meta = {
        "run_id": str(run_id),
        "profile": str(profile),
        "python_bin": str(python_bin),
        "output_root": str(output_root),
        "skip_existing": bool(skip_existing),
        "set_thread_env": bool(set_thread_env),
        "device": str(device),
        "cuda_device": int(cuda_device) if cuda_device is not None else None,
        "torch_threads": int(torch_threads),
        "cmds_file": str(paths.suite_cmds),
        "manifest_file": str(paths.suite_manifest),
        "lane_cmd_files": {k: str(v) for k, v in lane_cmds.items()},
        "lane_manifest_files": {k: str(v) for k, v in lane_manifests.items()},
        "counts_by_lane": counts,
        "n_commands_total": int(len(all_cmds)),
    }
    _write_text(paths.suite_meta, json.dumps(meta, indent=2, sort_keys=True) + "\n")
    return meta


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Publication C-TreePO benchmark suite orchestration.")
    sub = p.add_subparsers(dest="cmd", required=True)

    b = sub.add_parser("build", help="Build command lists + manifest for the publication benchmark suite.")
    b.add_argument("--run-id", type=str, default="")
    b.add_argument("--profile", choices=["smoke", "publication"], default="publication")
    b.add_argument("--python-bin", type=str, default="")
    b.add_argument(
        "--output-root",
        type=str,
        default="",
        help="Default: outputs/identifiable_zero_publication_ctreepo_<run_id>",
    )
    b.add_argument("--skip-existing", action=argparse.BooleanOptionalAction, default=True)
    b.add_argument(
        "--set-thread-env",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Prefix commands with low-thread env vars (OMP/MKL/OpenBLAS/etc).",
    )
    b.add_argument("--device", type=str, default="auto")
    b.add_argument("--cuda-device", type=int, default=None)
    b.add_argument("--torch-threads", type=int, default=1)

    r = sub.add_parser("run", help="Execute the suite command list.")
    r.add_argument("--output-root", type=str, required=True)
    r.add_argument("--jobs", type=int, default=1)
    r.add_argument("--log-dir", type=str, default="")
    r.add_argument("--fail-fast", action=argparse.BooleanOptionalAction, default=False)

    prog = sub.add_parser("progress", help="Generate interim progress report (markdown/PDF).")
    prog.add_argument("--output-root", type=str, required=True)
    prog.add_argument("--out-dir", type=str, default="")
    prog.add_argument("--emit-pdf", action=argparse.BooleanOptionalAction, default=True)

    exp = sub.add_parser("expectations", help="Generate simulation expectation JSON/Markdown reports.")
    exp.add_argument("--output-root", type=str, required=True)
    exp.add_argument("--output-json", type=str, default="")
    exp.add_argument("--output-markdown", type=str, default="")
    exp.add_argument("--strict", action=argparse.BooleanOptionalAction, default=False)
    exp.add_argument("--seed-aggregate", choices=["median", "mean"], default="median")
    exp.add_argument("--min-effect", type=float, default=0.10)
    exp.add_argument("--adjacent-tolerance", type=float, default=0.01)

    return p


def main(argv: Sequence[str] | None = None) -> int:
    ns = _build_parser().parse_args(list(argv) if argv is not None else None)

    if ns.cmd == "build":
        run_id = _utc_run_id(ns.run_id)
        python_bin = str(ns.python_bin).strip() or __import__("sys").executable
        output_root = (
            Path(ns.output_root)
            if str(ns.output_root).strip()
            else Path(f"outputs/identifiable_zero_publication_ctreepo_{run_id}")
        )
        meta = build_suite(
            run_id=run_id,
            profile=str(ns.profile),
            python_bin=python_bin,
            output_root=output_root,
            skip_existing=bool(ns.skip_existing),
            set_thread_env=bool(ns.set_thread_env),
            device=str(ns.device),
            cuda_device=ns.cuda_device,
            torch_threads=int(ns.torch_threads),
        )
        print(json.dumps(meta, indent=2, sort_keys=True))
        return 0

    if ns.cmd == "run":
        output_root = Path(ns.output_root).resolve()
        paths = _resolve_paths(output_root)
        if not paths.suite_cmds.exists():
            raise SystemExit(f"suite cmds not found (run build first): {paths.suite_cmds}")
        from src.ctreepo.sim.cli.exec_cmds import main as _exec_main  # noqa: WPS433

        exec_argv: List[str] = ["--cmds", str(paths.suite_cmds), "--jobs", str(int(ns.jobs))]
        if str(ns.log_dir).strip():
            exec_argv.extend(["--log-dir", str(ns.log_dir)])
        if bool(ns.fail_fast):
            exec_argv.append("--fail-fast")
        return int(_exec_main(exec_argv))

    if ns.cmd == "progress":
        from src.ctreepo.sim.cli.report.publication_ctreepo_progress import (  # noqa: WPS433
            main as _progress_main,
        )

        argv_out: List[str] = ["--output-root", str(Path(ns.output_root).resolve())]
        if str(ns.out_dir).strip():
            argv_out.extend(["--out-dir", str(Path(ns.out_dir).resolve())])
        argv_out.append("--emit-pdf" if bool(ns.emit_pdf) else "--no-emit-pdf")
        return int(_progress_main(argv_out))

    if ns.cmd == "expectations":
        from src.ctreepo.sim.expectations import (  # noqa: WPS433
            ExpectationConfig,
            build_expectation_report,
            write_expectation_report,
        )

        output_root = Path(ns.output_root).resolve()
        report = build_expectation_report(
            output_root=output_root,
            config=ExpectationConfig(
                seed_aggregate=str(ns.seed_aggregate),
                min_effect_rel=float(ns.min_effect),
                adjacent_tolerance=float(ns.adjacent_tolerance),
            ),
        )
        out_json = (
            Path(ns.output_json).resolve()
            if str(ns.output_json).strip()
            else output_root / "simulation_expectations.json"
        )
        out_markdown = (
            Path(ns.output_markdown).resolve()
            if str(ns.output_markdown).strip()
            else output_root / "simulation_expectations.md"
        )
        outputs = write_expectation_report(report, output_json=out_json, output_markdown=out_markdown)
        print(
            json.dumps(
                {
                    "output_json": outputs["output_json"],
                    "output_markdown": outputs["output_markdown"],
                    "summary": report.summary,
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 1 if bool(ns.strict) and int(report.summary.get("n_fail", 0)) > 0 else 0

    raise ValueError("unreachable")


if __name__ == "__main__":
    raise SystemExit(main())
