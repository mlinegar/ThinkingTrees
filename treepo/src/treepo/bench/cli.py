from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional, Sequence

from treepo.bench.io import dump_json, load_yaml_or_json
from treepo.bench.reports import cardinality as report_cardinality
from treepo.bench.reports import classical_sketches as report_classical_sketches
from treepo.bench.reports import lda_leafnoise as report_lda_leafnoise
from treepo.bench.reports import learned_g_overnight as report_learned_g_overnight
from treepo.bench.reports import publication_progress as report_publication_progress
from treepo.bench.runner import (
    VALID_EXPERIMENTS,
    emit_commands,
    run_single,
    run_specs,
    run_sweep,
)
from treepo.bench.suites.cardinality import build_cardinality_paper_suite
from treepo.bench.suites.classical_sketches import build_classical_sketches_suite
from treepo.bench.suites.identifiable_zero import (
    build_identifiable_zero_dtm_lda,
    build_identifiable_zero_lda_leafnoise,
    build_identifiable_zero_publication_ctreepo,
)


def _parse_args(argv: Optional[Sequence[str]]) -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="treepo-bench")
    sub = p.add_subparsers(dest="cmd", required=True)

    # ------------------------------------------------------------
    # run
    # ------------------------------------------------------------
    p_run = sub.add_parser("run", help="Run a single experiment and write JSON/CSV.")
    p_run.add_argument("experiment", choices=list(VALID_EXPERIMENTS))
    p_run.add_argument("--config", type=Path, required=True)
    p_run.add_argument("--json-out", type=Path, required=True)
    p_run.add_argument("--csv-out", type=Path, required=True)
    p_run.add_argument("--print-json", action="store_true", default=False)

    # ------------------------------------------------------------
    # sweep
    # ------------------------------------------------------------
    p_sweep = sub.add_parser("sweep", help="Run a grid sweep defined by a YAML/JSON spec.")
    p_sweep.add_argument("experiment", choices=list(VALID_EXPERIMENTS))
    p_sweep.add_argument("--spec", type=Path, required=True)
    p_sweep.add_argument("--out-root", type=Path, required=True)
    p_sweep.add_argument("--jobs", type=int, required=True)
    p_sweep.add_argument("--skip-existing", action=argparse.BooleanOptionalAction, default=False)
    p_sweep.add_argument("--emit-commands", type=Path, default=None)
    p_sweep.add_argument("--commands-only", action=argparse.BooleanOptionalAction, default=False)

    # ------------------------------------------------------------
    # suite
    # ------------------------------------------------------------
    p_suite = sub.add_parser("suite", help="Run a named benchmark suite.")
    p_suite.add_argument(
        "suite",
        choices=[
            "identifiable-zero-dtm-lda",
            "identifiable-zero-lda-leafnoise",
            "identifiable-zero-publication-ctreepo",
            "cardinality-paper",
            "classical-sketches",
        ],
    )
    p_suite.add_argument("--out-root", type=Path, required=True)
    p_suite.add_argument("--jobs", type=int, required=True)
    p_suite.add_argument("--skip-existing", action=argparse.BooleanOptionalAction, default=False)
    p_suite.add_argument("--emit-commands", type=Path, default=None)
    p_suite.add_argument("--commands-only", action=argparse.BooleanOptionalAction, default=False)
    p_suite.add_argument("--seeds", type=str, default=None)
    p_suite.add_argument("--topic-phi-estimators", type=str, default=None)
    p_suite.add_argument("--leaf-counts", type=str, default=None)
    p_suite.add_argument("--capacities", type=str, default=None)
    p_suite.add_argument(
        "--execution-backend",
        choices=["unified_g", "treepo"],
        default="unified_g",
        help="Execution path for classical-sketches; unified_g routes through fit().",
    )
    p_suite.add_argument("--include-learned", action=argparse.BooleanOptionalAction, default=False)
    p_suite.add_argument("--learned-targets", type=str, default=None)
    p_suite.add_argument("--learned-variants", type=str, default=None)
    p_suite.add_argument("--learned-epochs", type=int, default=150)
    p_suite.add_argument("--learned-n-train", type=int, default=128)
    p_suite.add_argument("--learned-n-val", type=int, default=48)

    # ------------------------------------------------------------
    # report
    # ------------------------------------------------------------
    p_report = sub.add_parser("report", help="Generate a report from existing outputs.")
    rep = p_report.add_subparsers(dest="report", required=True)

    p_rep_leaf = rep.add_parser("lda-leafnoise", help="Leaf-noise progression report (LDA baseline).")
    p_rep_leaf.add_argument("--output-root", type=Path, required=True)
    p_rep_leaf.add_argument("--ctreepo-root", type=Path, default=None)
    p_rep_leaf.add_argument("--out-dir", type=Path, default=None)
    p_rep_leaf.add_argument("--emit-pdf", action=argparse.BooleanOptionalAction, default=True)

    p_rep_pub = rep.add_parser("publication-progress", help="Interim progress plots for publication suite.")
    p_rep_pub.add_argument("--output-root", type=Path, required=True)
    p_rep_pub.add_argument("--out-dir", type=Path, default=None)
    p_rep_pub.add_argument("--emit-pdf", action=argparse.BooleanOptionalAction, default=True)

    p_rep_learned_g = rep.add_parser("learned-g-overnight", help="Progress report for learned-g overnight runs.")
    p_rep_learned_g.add_argument("--output-root", type=Path, required=True)
    p_rep_learned_g.add_argument("--out-dir", type=Path, default=None)
    p_rep_learned_g.add_argument("--emit-pdf", action=argparse.BooleanOptionalAction, default=True)

    p_rep_card = rep.add_parser("cardinality", help="Cardinality/HLL report and figures.")
    p_rep_card.add_argument("--output-root", type=Path, required=True)
    p_rep_card.add_argument("--out-dir", type=Path, default=None)
    p_rep_card.add_argument("--emit-pdf", action=argparse.BooleanOptionalAction, default=True)

    p_rep_classical = rep.add_parser("classical-sketches", help="Classical sketch comparison report.")
    p_rep_classical.add_argument("--output-root", type=Path, required=True)
    p_rep_classical.add_argument("--out-dir", type=Path, default=None)
    p_rep_classical.add_argument("--tables-dir", type=Path, default=Path("paper/ctreepo/tables"))
    p_rep_classical.add_argument("--emit-pdf", action=argparse.BooleanOptionalAction, default=False)

    return p.parse_args(list(argv) if argv is not None else None)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parse_args(argv)

    if args.cmd == "run":
        payload = load_yaml_or_json(Path(args.config))
        if not isinstance(payload, dict):
            raise SystemExit("--config must contain a JSON/YAML mapping")
        res = run_single(
            experiment=str(args.experiment),
            config=payload,
            json_out=Path(args.json_out),
            csv_out=Path(args.csv_out),
            print_json=bool(args.print_json),
        )
        if not bool(args.print_json):
            print(dump_json(res))
        return 0

    if args.cmd == "sweep":
        res = run_sweep(
            experiment=str(args.experiment),
            spec_path=Path(args.spec),
            out_root=Path(args.out_root),
            jobs=int(args.jobs),
            skip_existing=bool(args.skip_existing),
            emit_commands_path=Path(args.emit_commands) if args.emit_commands is not None else None,
            commands_only=bool(args.commands_only),
        )
        print(dump_json({"results": res, "n_results": len(res)}))
        return 0

    if args.cmd == "suite":
        suite = str(args.suite)
        skip = bool(args.skip_existing)
        out_root = Path(args.out_root)

        if suite == "identifiable-zero-dtm-lda":
            specs = build_identifiable_zero_dtm_lda(
                out_root=out_root,
                skip_existing=skip,
                seeds=args.seeds,
                topic_phi_estimators=args.topic_phi_estimators,
            )
        elif suite == "identifiable-zero-lda-leafnoise":
            specs = build_identifiable_zero_lda_leafnoise(out_root=out_root, skip_existing=skip, seeds=args.seeds)
        elif suite == "identifiable-zero-publication-ctreepo":
            specs = build_identifiable_zero_publication_ctreepo(out_root=out_root, skip_existing=skip, seeds=args.seeds)
        elif suite == "cardinality-paper":
            specs = build_cardinality_paper_suite(out_root=out_root, skip_existing=skip, seeds=args.seeds)
        elif suite == "classical-sketches":
            specs = build_classical_sketches_suite(
                out_root=out_root,
                skip_existing=skip,
                seeds=args.seeds,
                leaf_counts=args.leaf_counts,
                capacities=args.capacities,
                execution_backend=args.execution_backend,
                include_learned=bool(args.include_learned),
                learned_targets=args.learned_targets,
                learned_variants=args.learned_variants,
                learned_n_epochs=int(args.learned_epochs),
                learned_n_train=int(args.learned_n_train),
                learned_n_val=int(args.learned_n_val),
            )
        else:  # pragma: no cover
            raise SystemExit(f"unknown suite: {suite}")

        if args.emit_commands is not None:
            emit_commands(specs, out_path=Path(args.emit_commands))
        if args.commands_only:
            print(dump_json({"status": "commands_only", "n_runs": len(specs)}))
            return 0

        results = run_specs(specs, jobs=int(args.jobs), skip_existing=skip)
        print(dump_json({"suite": suite, "n_runs": len(specs), "results": results}))
        return 0

    if args.cmd == "report":
        if args.report == "lda-leafnoise":
            argv2: list[str] = ["--output-root", str(Path(args.output_root))]
            if args.ctreepo_root is not None:
                argv2 += ["--ctreepo-root", str(Path(args.ctreepo_root))]
            if args.out_dir is not None:
                argv2 += ["--out-dir", str(Path(args.out_dir))]
            argv2 += ["--emit-pdf" if bool(args.emit_pdf) else "--no-emit-pdf"]
            return int(report_lda_leafnoise.main(argv2))
        if args.report == "publication-progress":
            argv2 = ["--output-root", str(Path(args.output_root))]
            if args.out_dir is not None:
                argv2 += ["--out-dir", str(Path(args.out_dir))]
            argv2 += ["--emit-pdf" if bool(args.emit_pdf) else "--no-emit-pdf"]
            return int(report_publication_progress.main(argv2))
        if args.report == "learned-g-overnight":
            argv2 = ["--output-root", str(Path(args.output_root))]
            if args.out_dir is not None:
                argv2 += ["--out-dir", str(Path(args.out_dir))]
            argv2 += ["--emit-pdf" if bool(args.emit_pdf) else "--no-emit-pdf"]
            return int(report_learned_g_overnight.main(argv2))
        if args.report == "cardinality":
            argv2 = ["--output-root", str(Path(args.output_root))]
            if args.out_dir is not None:
                argv2 += ["--out-dir", str(Path(args.out_dir))]
            argv2 += ["--emit-pdf" if bool(args.emit_pdf) else "--no-emit-pdf"]
            return int(report_cardinality.main(argv2))
        if args.report == "classical-sketches":
            argv2 = ["--output-root", str(Path(args.output_root))]
            if args.out_dir is not None:
                argv2 += ["--out-dir", str(Path(args.out_dir))]
            if args.tables_dir is not None:
                argv2 += ["--tables-dir", str(Path(args.tables_dir))]
            argv2 += ["--emit-pdf" if bool(args.emit_pdf) else "--no-emit-pdf"]
            return int(report_classical_sketches.main(argv2))
        raise SystemExit(f"unknown report: {args.report}")

    raise SystemExit(f"unknown command: {args.cmd}")


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main(sys.argv[1:]))
