#!/usr/bin/env python3
"""Run the HLL register-state precision-floor recovery sweep."""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
PY_PATHS = [
    REPO_ROOT / "treepo" / "src",
    REPO_ROOT / "parallel" / "unified_g_v1" / "src",
    REPO_ROOT,
]
for path in reversed(PY_PATHS):
    value = str(path)
    if value not in sys.path:
        sys.path.insert(0, value)

from unified_g_v1.sketch.hll_precision_floor import (  # noqa: E402
    HLLPrecisionFloorConfig,
    plot_precision_floor_recovery,
    run_precision_floor_sweep,
    stage_precision_floor_assets,
    write_precision_floor_outputs,
)


def _utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def _parse_ints(raw: str) -> tuple[int, ...]:
    values = tuple(int(item.strip()) for item in str(raw).split(",") if item.strip())
    if not values:
        raise argparse.ArgumentTypeError("expected at least one integer")
    return values


def _parse_optional_ints(raw: str | None) -> tuple[int, ...]:
    if raw is None or str(raw).strip() == "":
        return ()
    return _parse_ints(str(raw))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-root", type=Path, default=None)
    parser.add_argument("--precisions", type=_parse_ints, default=_parse_ints("6,7,8,9,10,11,12"))
    parser.add_argument("--leaf-counts", type=_parse_ints, default=_parse_ints("1,2,4,8,16"))
    parser.add_argument("--n-train", type=int, default=8192)
    parser.add_argument("--n-val", type=int, default=1024)
    parser.add_argument("--min-tokens", type=int, default=128)
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--universe-size", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n-epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--local-law-weight", type=float, default=0.9)
    parser.add_argument("--merge-state-weight", type=float, default=100.0)
    parser.add_argument("--use-cuda", action="store_true")
    parser.add_argument("--cuda-device", type=int, default=None)
    parser.add_argument("--cuda-devices", type=_parse_optional_ints, default=())
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--include-token-diagnostics", action="store_true")
    parser.add_argument("--skip-run", action="store_true", help="Only plot/report from an existing JSON summary.")
    parser.add_argument(
        "--summary-json",
        type=Path,
        default=None,
        help="Existing hll_precision_floor_recovery.json used with --skip-run.",
    )
    parser.add_argument("--stage-paper-assets", action="store_true")
    parser.add_argument(
        "--paper-figures-dir",
        type=Path,
        default=REPO_ROOT / "paper" / "ctreepo" / "assets" / "hll" / "figures",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    out_root = (
        Path(args.out_root)
        if args.out_root is not None
        else REPO_ROOT / "outputs" / f"hll_precision_floor_recovery_{_utc_stamp()}"
    ).resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    env_py = os.pathsep.join(str(path) for path in PY_PATHS)
    if os.environ.get("PYTHONPATH"):
        os.environ["PYTHONPATH"] = env_py + os.pathsep + os.environ["PYTHONPATH"]
    else:
        os.environ["PYTHONPATH"] = env_py

    config = HLLPrecisionFloorConfig(
        precisions=tuple(int(x) for x in args.precisions),
        leaf_counts=tuple(int(x) for x in args.leaf_counts),
        n_train=int(args.n_train),
        n_val=int(args.n_val),
        min_tokens=int(args.min_tokens),
        max_tokens=int(args.max_tokens),
        universe_size=int(args.universe_size),
        seed=int(args.seed),
        n_epochs=int(args.n_epochs),
        batch_size=int(args.batch_size),
        learning_rate=float(args.lr),
        hidden_dim=int(args.hidden_dim),
        local_law_weight=float(args.local_law_weight),
        merge_state_weight=float(args.merge_state_weight),
        use_cuda=bool(args.use_cuda),
        cuda_device=args.cuda_device,
        include_token_diagnostics=bool(args.include_token_diagnostics),
    )

    if bool(args.skip_run):
        if args.summary_json is None:
            raise SystemExit("--skip-run requires --summary-json")
        payload = json.loads(Path(args.summary_json).read_text(encoding="utf-8"))
        rows = list(payload.get("rows") or [])
    else:
        cuda_devices = tuple(int(x) for x in args.cuda_devices)
        if not cuda_devices and args.cuda_device is not None:
            cuda_devices = (int(args.cuda_device),)
        rows = run_precision_floor_sweep(
            config,
            output_root=out_root,
            jobs=int(args.jobs),
            cuda_devices=cuda_devices,
        )
        paths = write_precision_floor_outputs(rows, output_root=out_root, config=config)
        print(f"wrote {paths['csv']}")
        print(f"wrote {paths['json']}")

    fig_stem = out_root / "hll_precision_floor_recovery"
    written = plot_precision_floor_recovery(rows, output_stem=fig_stem)
    for path in written:
        print(f"wrote {path}")
    if bool(args.stage_paper_assets):
        for path in stage_precision_floor_assets(
            output_stem=fig_stem,
            paper_figures_dir=Path(args.paper_figures_dir),
        ):
            print(f"staged {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
