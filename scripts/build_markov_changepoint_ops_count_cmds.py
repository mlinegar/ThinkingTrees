#!/usr/bin/env python3
"""Build xargs-friendly command lists for the Markov changepoint OPS-count sweep."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, List


def _parse_items(text: str) -> List[str]:
    out: List[str] = []
    for raw in str(text).replace(",", " ").split():
        x = raw.strip()
        if x:
            out.append(x)
    return out


def _parse_ints(text: str) -> List[int]:
    return [int(x) for x in _parse_items(text)]


def _parse_floats(text: str) -> List[float]:
    return [float(x) for x in _parse_items(text)]


def _parse_bools(text: str) -> List[bool]:
    out: List[bool] = []
    for raw in _parse_items(text):
        x = raw.strip().lower()
        if x in ("1", "true", "t", "yes", "y"):
            out.append(True)
        elif x in ("0", "false", "f", "no", "n"):
            out.append(False)
        else:
            raise ValueError(f"could not parse boolean: {raw!r}")
    return out


def _fmt_float(x: float) -> str:
    s = f"{float(x):.6g}"
    return s.replace("-", "m").replace(".", "p")


def _iter_commands(
    *,
    python_bin: str,
    n_regimes: int,
    vocab_size: int,
    min_tokens: int,
    max_tokens: int,
    min_segments: int,
    max_segments: int,
    fixed_leaf_tokens: int,
    train_docs: Iterable[int],
    val_docs: int,
    test_docs: int,
    audit_fractions: Iterable[float],
    c3_audit_strategies: Iterable[str],
    c3_include_root: bool,
    leaf_query_rates: Iterable[float],
    include_root_queries: Iterable[bool],
    local_law_weights: Iterable[float],
    task_objective_weights: Iterable[float],
    c1_relative_weights: Iterable[float],
    c2_relative_weights: Iterable[float],
    c3_relative_weights: Iterable[float],
    root_weights: Iterable[float],
    schedule_consistency_weights: Iterable[float],
    guidance_override_modes: Iterable[str],
    eval_guidance_qs: Iterable[float],
    eval_guidance_trials: int,
    eval_guidance_seed_offset: int,
    eval_guidance_include_root: bool,
    include_rf_root_baseline: bool,
    rf_n_estimators: int,
    rf_max_depth: int,
    rf_min_samples_leaf: int,
    data_seeds: Iterable[int],
    seeds: Iterable[int],
    output_root: Path,
    model_families: Iterable[str],
    feature_modes: Iterable[str],
    state_dims: Iterable[int],
    hidden_dims: Iterable[int],
    hidden_dim_multiplier: float | None,
    hidden_dim_min: int,
    n_epochs: int,
    device: str,
    cuda_device: int | None,
    violation_tau: float,
    torch_threads: int,
    skip_existing: bool,
) -> List[str]:
    cmds: List[str] = []
    script = "scripts/run_markov_changepoint_ops_count_simulation.py"

    if int(n_regimes) <= 0:
        raise ValueError("n_regimes must be positive")
    if int(vocab_size) <= 0:
        raise ValueError("vocab_size must be positive")
    if int(min_tokens) <= 0 or int(max_tokens) <= 0:
        raise ValueError("min_tokens/max_tokens must be positive")
    if int(min_segments) <= 0 or int(max_segments) <= 0:
        raise ValueError("min_segments/max_segments must be positive")
    if int(fixed_leaf_tokens) <= 0:
        raise ValueError("fixed_leaf_tokens must be positive")

    include_root_values = list(include_root_queries)
    if not include_root_values:
        include_root_values = [True]
    local_law_values = [float(x) for x in local_law_weights]
    if not local_law_values:
        local_law_values = [0.0]
    task_objective_values = [float(x) for x in task_objective_weights]
    c1_relative_values = [float(x) for x in c1_relative_weights]
    if not c1_relative_values:
        c1_relative_values = [1.0]
    c2_relative_values = [float(x) for x in c2_relative_weights]
    if not c2_relative_values:
        c2_relative_values = [1.0]
    c3_relative_values = [float(x) for x in c3_relative_weights]
    if not c3_relative_values:
        c3_relative_values = [1.0]
    family_values = [str(x).strip() for x in model_families if str(x).strip()]
    if not family_values:
        family_values = ["neural"]
    guidance_override_values = [
        str(x).strip().lower() for x in guidance_override_modes if str(x).strip()
    ]
    if not guidance_override_values:
        guidance_override_values = ["reset"]
    for mode in guidance_override_values:
        if mode not in {"reset", "adjust"}:
            raise ValueError("guidance_override_modes must be a subset of {'reset','adjust'}")
    guidance_qs_list = [float(q) for q in eval_guidance_qs]
    guidance_qs_text = ",".join(f"{float(q):.6g}" for q in guidance_qs_list)
    data_seed_values = [int(x) for x in data_seeds]
    if not data_seed_values:
        data_seed_values = [None]

    feature_mode_values = [str(x).strip() for x in feature_modes if str(x).strip()]
    if not feature_mode_values:
        feature_mode_values = ["full"]
    for fm in feature_mode_values:
        if fm not in {"full", "no_endpoints"}:
            raise ValueError("feature_modes must be a subset of {'full','no_endpoints'}")

    state_dim_values = [int(x) for x in state_dims]
    if not state_dim_values:
        state_dim_values = [32]
    hidden_dim_values = [int(x) for x in hidden_dims]
    if not hidden_dim_values:
        hidden_dim_values = [128]
    hdm: float | None = None
    if hidden_dim_multiplier is not None and float(hidden_dim_multiplier) > 0.0:
        hdm = float(hidden_dim_multiplier)
    hd_min = int(hidden_dim_min)

    val_component = f"/val_{int(val_docs)}" if int(val_docs) > 0 else ""

    for fam in family_values:
        for td in train_docs:
            for frac in audit_fractions:
                for c3_strat in c3_audit_strategies:
                    for lqr in leaf_query_rates:
                        for rootq in include_root_values:
                            for llw in local_law_values:
                                for task_weight in [None] + task_objective_values:
                                    for c1_rel in c1_relative_values:
                                        for c2_rel in c2_relative_values:
                                            for c3_rel in c3_relative_values:
                                                for rw in root_weights:
                                                    for scw in schedule_consistency_weights:
                                                        for gov_mode in guidance_override_values:
                                                            gov_component = ""
                                                            if (
                                                                len(guidance_override_values) > 1
                                                                or str(gov_mode) != "reset"
                                                            ):
                                                                gov_component = (
                                                                    f"/gov_{str(gov_mode)}"
                                                                )
                                                            rf_component = ""
                                                            if bool(include_rf_root_baseline):
                                                                rf_component = "/rfroot_1"
                                                            for fm in feature_mode_values:
                                                                fm_component = ""
                                                                if (
                                                                    len(feature_mode_values) > 1
                                                                    or str(fm) != "full"
                                                                ):
                                                                    fm_component = f"/fm_{str(fm)}"
                                                                for sd in state_dim_values:
                                                                    if int(sd) <= 0:
                                                                        raise ValueError(
                                                                            "state_dims must be positive"
                                                                        )
                                                                    sd_component = ""
                                                                    if (
                                                                        len(state_dim_values) > 1
                                                                        or int(sd) != 32
                                                                    ):
                                                                        sd_component = (
                                                                            f"/sd_{int(sd)}"
                                                                        )
                                                                    derived_hidden = (
                                                                        [
                                                                            max(
                                                                                hd_min,
                                                                                int(
                                                                                    round(
                                                                                        float(hdm)
                                                                                        * float(sd)
                                                                                    )
                                                                                ),
                                                                            )
                                                                        ]
                                                                        if hdm is not None
                                                                        else []
                                                                    )
                                                                    hd_iter = (
                                                                        derived_hidden
                                                                        or hidden_dim_values
                                                                    )
                                                                    for hd in hd_iter:
                                                                        if int(hd) <= 0:
                                                                            raise ValueError(
                                                                                "hidden_dims must be positive"
                                                                            )
                                                                        hd_component = ""
                                                                        if (
                                                                            hdm is not None
                                                                            or len(
                                                                                hidden_dim_values
                                                                            )
                                                                            > 1
                                                                            or int(hd) != 128
                                                                        ):
                                                                            hd_component = (
                                                                                f"/hd_{int(hd)}"
                                                                            )
                                                                        for (
                                                                            data_seed
                                                                        ) in data_seed_values:
                                                                            for seed in seeds:
                                                                                rootq_component = ""
                                                                                if len(
                                                                                    include_root_values
                                                                                ) > 1 or not bool(
                                                                                    rootq
                                                                                ):
                                                                                    rootq_component = f"/rootq_{1 if bool(rootq) else 0}"
                                                                                fam_component = (
                                                                                    f"/family_{fam}"
                                                                                    if len(
                                                                                        family_values
                                                                                    )
                                                                                    > 1
                                                                                    else ""
                                                                                )
                                                                                law_mix_component = (
                                                                                    ""
                                                                                )
                                                                                if (
                                                                                    len(
                                                                                        c1_relative_values
                                                                                    )
                                                                                    > 1
                                                                                    or len(
                                                                                        c2_relative_values
                                                                                    )
                                                                                    > 1
                                                                                    or len(
                                                                                        c3_relative_values
                                                                                    )
                                                                                    > 1
                                                                                    or abs(
                                                                                        float(
                                                                                            c1_rel
                                                                                        )
                                                                                        - 1.0
                                                                                    )
                                                                                    > 1e-12
                                                                                    or abs(
                                                                                        float(
                                                                                            c2_rel
                                                                                        )
                                                                                        - 1.0
                                                                                    )
                                                                                    > 1e-12
                                                                                    or abs(
                                                                                        float(
                                                                                            c3_rel
                                                                                        )
                                                                                        - 1.0
                                                                                    )
                                                                                    > 1e-12
                                                                                ):
                                                                                    law_mix_component = (
                                                                                        f"/c1r_{_fmt_float(c1_rel)}"
                                                                                        f"/c2r_{_fmt_float(c2_rel)}"
                                                                                        f"/c3r_{_fmt_float(c3_rel)}"
                                                                                    )
                                                                                task_component = ""
                                                                                if (
                                                                                    task_weight
                                                                                    is not None
                                                                                ):
                                                                                    task_component = f"/taskw_{_fmt_float(task_weight)}"
                                                                                dseed_component = (
                                                                                    f"/dseed_{int(data_seed)}"
                                                                                    if data_seed
                                                                                    is not None
                                                                                    else ""
                                                                                )
                                                                                sub = (
                                                                                    f"train_{int(td)}{val_component}{fam_component}{rootq_component}{rf_component}{gov_component}{fm_component}{sd_component}{hd_component}{dseed_component}"
                                                                                    f"/budget_{_fmt_float(frac)}"
                                                                                    f"/c3_{str(c3_strat)}/c3root_{1 if c3_include_root else 0}"
                                                                                    f"/lqr_{_fmt_float(lqr)}"
                                                                                    f"/llw_{_fmt_float(llw)}{task_component}{law_mix_component}"
                                                                                    f"/rw_{_fmt_float(rw)}/scw_{_fmt_float(scw)}"
                                                                                )
                                                                                base = (
                                                                                    output_root
                                                                                    / sub
                                                                                    / f"seed_{int(seed)}"
                                                                                )
                                                                                out_json = base.with_suffix(
                                                                                    ".json"
                                                                                )
                                                                                out_csv = base.with_suffix(
                                                                                    ".csv"
                                                                                )
                                                                                if (
                                                                                    skip_existing
                                                                                    and out_json.exists()
                                                                                    and out_csv.exists()
                                                                                ):
                                                                                    continue

                                                                                parts: List[str] = [
                                                                                    f"{python_bin} -u {script}",
                                                                                    f"--n-regimes {int(n_regimes)}",
                                                                                    f"--vocab-size {int(vocab_size)}",
                                                                                    f"--min-tokens {int(min_tokens)}",
                                                                                    f"--max-tokens {int(max_tokens)}",
                                                                                    f"--min-segments {int(min_segments)}",
                                                                                    f"--max-segments {int(max_segments)}",
                                                                                    f"--fixed-leaf-tokens {int(fixed_leaf_tokens)}",
                                                                                    f"--train-docs {int(td)}",
                                                                                    f"--val-docs {int(val_docs)}",
                                                                                    f"--test-docs {int(test_docs)}",
                                                                                    f"--model-family {str(fam)}",
                                                                                    "--audit-policy fraction",
                                                                                    f"--audit-fraction {float(frac)}",
                                                                                    f"--c3-audit-strategy {str(c3_strat)}",
                                                                                    f"--leaf-query-rate {float(lqr)}",
                                                                                    f"--local-law-weight {float(llw)}",
                                                                                    f"--c1-relative-weight {float(c1_rel)}",
                                                                                    f"--c2-relative-weight {float(c2_rel)}",
                                                                                    f"--c3-relative-weight {float(c3_rel)}",
                                                                                    f"--root-weight {float(rw)}",
                                                                                    f"--schedule-consistency-weight {float(scw)}",
                                                                                    f"--feature-mode {str(fm)}",
                                                                                    f"--state-dim {int(sd)}",
                                                                                    f"--hidden-dim {int(hd)}",
                                                                                    f"--n-epochs {int(n_epochs)}",
                                                                                    f"--device {device}",
                                                                                ]
                                                                                if (
                                                                                    task_weight
                                                                                    is not None
                                                                                ):
                                                                                    parts.append(
                                                                                        f"--task-objective-weight {float(task_weight)}"
                                                                                    )
                                                                                if (
                                                                                    data_seed
                                                                                    is not None
                                                                                ):
                                                                                    parts.append(
                                                                                        f"--data-seed {int(data_seed)}"
                                                                                    )
                                                                                parts.append(
                                                                                    f"--model-seed {int(seed)}"
                                                                                )
                                                                                if bool(
                                                                                    include_rf_root_baseline
                                                                                ):
                                                                                    parts.append(
                                                                                        "--include-rf-root-baseline"
                                                                                    )
                                                                                    parts.append(
                                                                                        f"--rf-n-estimators {int(rf_n_estimators)}"
                                                                                    )
                                                                                    parts.append(
                                                                                        f"--rf-max-depth {int(rf_max_depth)}"
                                                                                    )
                                                                                    parts.append(
                                                                                        f"--rf-min-samples-leaf {int(rf_min_samples_leaf)}"
                                                                                    )
                                                                                if gov_component:
                                                                                    parts.append(
                                                                                        f"--guidance-override-mode {str(gov_mode)}"
                                                                                    )
                                                                                if not bool(rootq):
                                                                                    parts.append(
                                                                                        "--no-root-query"
                                                                                    )
                                                                                if (
                                                                                    not c3_include_root
                                                                                ):
                                                                                    parts.append(
                                                                                        "--no-c3-include-root"
                                                                                    )
                                                                                if (
                                                                                    cuda_device
                                                                                    is not None
                                                                                ):
                                                                                    parts.append(
                                                                                        f"--cuda-device {int(cuda_device)}"
                                                                                    )
                                                                                if (
                                                                                    int(
                                                                                        eval_guidance_trials
                                                                                    )
                                                                                    > 0
                                                                                    and guidance_qs_list
                                                                                ):
                                                                                    parts.append(
                                                                                        f"--eval-guidance-qs {guidance_qs_text}"
                                                                                    )
                                                                                    parts.append(
                                                                                        f"--eval-guidance-trials {int(eval_guidance_trials)}"
                                                                                    )
                                                                                    parts.append(
                                                                                        f"--eval-guidance-seed-offset {int(eval_guidance_seed_offset)}"
                                                                                    )
                                                                                    parts.append(
                                                                                        "--eval-guidance-include-root"
                                                                                        if bool(
                                                                                            eval_guidance_include_root
                                                                                        )
                                                                                        else "--no-eval-guidance-include-root"
                                                                                    )
                                                                                parts.extend(
                                                                                    [
                                                                                        f"--torch-threads {int(torch_threads)}",
                                                                                        f"--violation-tau {float(violation_tau)}",
                                                                                        f"--seed {int(seed)}",
                                                                                        f"--json-summary {out_json}",
                                                                                        f"--csv-summary {out_csv}",
                                                                                    ]
                                                                                )
                                                                                cmd = " ".join(
                                                                                    parts
                                                                                )
                                                                                cmds.append(cmd)
    return cmds


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build Markov OPS-count sweep command list.")
    p.add_argument("--python-bin", type=str, default="venv/bin/python")
    p.add_argument("--out-cmds", type=str, default="logs/markov_changepoint_ops_count_cmds.txt")
    p.add_argument("--output-root", type=str, default="outputs/markov_changepoint_ops_count")

    p.add_argument("--n-regimes", type=int, default=4)
    p.add_argument("--vocab-size", type=int, default=96)
    p.add_argument("--min-tokens", type=int, default=384)
    p.add_argument("--max-tokens", type=int, default=384)
    p.add_argument("--min-segments", type=int, default=12)
    p.add_argument("--max-segments", type=int, default=24)
    p.add_argument("--fixed-leaf-tokens", type=int, default=16)

    p.add_argument("--train-docs", type=str, default="50 100 200 500 1000 2000")
    p.add_argument("--val-docs", type=int, default=0)
    p.add_argument("--test-docs", type=int, default=1000)
    p.add_argument("--audit-fractions", type=str, default="0.05 0.1 0.2 0.5 1.0")
    p.add_argument(
        "--model-family",
        type=str,
        default="neural",
        help="Space/comma list of model families (neural, additive).",
    )
    p.add_argument("--c3-audit-strategies", type=str, default="uniform")
    p.add_argument("--c3-include-root", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--leaf-query-rates", type=str, default="1.0")
    p.add_argument(
        "--include-root-query",
        type=str,
        default="true",
        help="Space/comma list of booleans for include_root_query in learned training.",
    )
    p.add_argument(
        "--local-law-weights",
        type=str,
        default="0 0.025 0.05 0.075 0.1 0.15 0.2 0.25 0.35 0.5 0.65 0.8 0.9 1.0",
        help="Space/comma list of theorem-facing local-law tradeoff weights λ.",
    )
    p.add_argument(
        "--task-objective-weights",
        type=str,
        default="",
        help="Optional space/comma list of explicit task-objective weights. Empty keeps theorem-facing `(1-lambda)` defaults.",
    )
    p.add_argument(
        "--c1-relative-weights",
        type=str,
        default="1.0",
        help="Space/comma list of relative weights assigned to C1/L1 within λ_local_law.",
    )
    p.add_argument(
        "--c2-relative-weights",
        type=str,
        default="1.0",
        help="Space/comma list of relative weights assigned to C2/L3 within λ_local_law.",
    )
    p.add_argument(
        "--c3-relative-weights",
        type=str,
        default="1.0",
        help="Space/comma list of relative weights assigned to C3/L2 within λ_local_law.",
    )
    p.add_argument("--root-weights", type=str, default="1.0")
    p.add_argument("--schedule-consistency-weights", type=str, default="0.0")
    p.add_argument(
        "--guidance-override-modes",
        type=str,
        default="reset",
        help="Space/comma list of guidance override modes (reset, adjust).",
    )
    p.add_argument(
        "--eval-guidance-qs",
        type=str,
        default="",
        help="Optional comma/space list of inference-time oracle guidance q values.",
    )
    p.add_argument(
        "--eval-guidance-trials",
        type=int,
        default=0,
        help="Guidance trials per q for guided_eval_curve. 0 disables guidance evaluation.",
    )
    p.add_argument("--eval-guidance-seed-offset", type=int, default=100000)
    p.add_argument(
        "--eval-guidance-include-root",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Whether root node is eligible for inference-time guidance replacement.",
    )
    p.add_argument(
        "--data-seeds",
        type=str,
        default="",
        help="Optional space/comma list of fixed corpus seeds. Empty keeps legacy seed semantics.",
    )
    p.add_argument("--seeds", type=str, default="0 1 2 3 4 5 6 7")

    p.add_argument(
        "--include-rf-root-baseline", action=argparse.BooleanOptionalAction, default=False
    )
    p.add_argument("--rf-n-estimators", type=int, default=200)
    p.add_argument("--rf-max-depth", type=int, default=16)
    p.add_argument("--rf-min-samples-leaf", type=int, default=5)

    p.add_argument("--feature-mode", choices=["full", "no_endpoints"], default="full")
    p.add_argument(
        "--feature-modes",
        type=str,
        default="",
        help="Optional space/comma list of feature modes. If set, overrides --feature-mode.",
    )
    p.add_argument(
        "--state-dims",
        type=str,
        default="32",
        help="Space/comma list of learned sketch latent dimensions (state_dim).",
    )
    p.add_argument(
        "--hidden-dims",
        type=str,
        default="128",
        help="Space/comma list of MLP hidden dimensions (hidden_dim). Ignored if --hidden-dim-multiplier is set.",
    )
    p.add_argument(
        "--hidden-dim-multiplier",
        type=float,
        default=0.0,
        help="If >0, sets hidden_dim = max(hidden_dim_min, round(multiplier*state_dim)) per state_dim.",
    )
    p.add_argument("--hidden-dim-min", type=int, default=64)
    p.add_argument("--n-epochs", type=int, default=10)
    p.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda"],
        default="auto",
        help="Execution device mode. 'auto' leaves CPU/GPU placement to the launcher.",
    )
    p.add_argument(
        "--cuda-device",
        type=int,
        default=None,
        help="Optional CUDA device index for --device cuda/auto.",
    )
    p.add_argument(
        "--torch-threads",
        type=int,
        default=1,
        help="Torch thread count per process (sweep-friendly). <=0 keeps torch defaults.",
    )
    p.add_argument("--violation-tau", type=float, default=0.0)

    p.add_argument("--skip-existing", action=argparse.BooleanOptionalAction, default=True)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    out_cmds = Path(args.out_cmds)
    out_cmds.parent.mkdir(parents=True, exist_ok=True)
    Path(args.output_root).mkdir(parents=True, exist_ok=True)

    feature_modes = (
        _parse_items(args.feature_modes)
        if str(args.feature_modes).strip()
        else [str(args.feature_mode)]
    )

    cmds = _iter_commands(
        python_bin=str(args.python_bin),
        n_regimes=int(args.n_regimes),
        vocab_size=int(args.vocab_size),
        min_tokens=int(args.min_tokens),
        max_tokens=int(args.max_tokens),
        min_segments=int(args.min_segments),
        max_segments=int(args.max_segments),
        fixed_leaf_tokens=int(args.fixed_leaf_tokens),
        train_docs=_parse_ints(args.train_docs),
        val_docs=int(args.val_docs),
        test_docs=int(args.test_docs),
        audit_fractions=_parse_floats(args.audit_fractions),
        c3_audit_strategies=_parse_items(args.c3_audit_strategies),
        c3_include_root=bool(args.c3_include_root),
        leaf_query_rates=_parse_floats(args.leaf_query_rates),
        include_root_queries=_parse_bools(args.include_root_query),
        local_law_weights=_parse_floats(args.local_law_weights),
        task_objective_weights=(
            _parse_floats(args.task_objective_weights)
            if str(args.task_objective_weights).strip()
            else []
        ),
        c1_relative_weights=_parse_floats(args.c1_relative_weights),
        c2_relative_weights=_parse_floats(args.c2_relative_weights),
        c3_relative_weights=_parse_floats(args.c3_relative_weights),
        root_weights=_parse_floats(args.root_weights),
        schedule_consistency_weights=_parse_floats(args.schedule_consistency_weights),
        guidance_override_modes=_parse_items(args.guidance_override_modes),
        eval_guidance_qs=_parse_floats(args.eval_guidance_qs),
        eval_guidance_trials=int(args.eval_guidance_trials),
        eval_guidance_seed_offset=int(args.eval_guidance_seed_offset),
        eval_guidance_include_root=bool(args.eval_guidance_include_root),
        include_rf_root_baseline=bool(args.include_rf_root_baseline),
        rf_n_estimators=int(args.rf_n_estimators),
        rf_max_depth=int(args.rf_max_depth),
        rf_min_samples_leaf=int(args.rf_min_samples_leaf),
        data_seeds=_parse_ints(args.data_seeds),
        seeds=_parse_ints(args.seeds),
        output_root=Path(args.output_root),
        model_families=_parse_items(args.model_family),
        feature_modes=feature_modes,
        state_dims=_parse_ints(args.state_dims),
        hidden_dims=_parse_ints(args.hidden_dims),
        hidden_dim_multiplier=(
            float(args.hidden_dim_multiplier) if float(args.hidden_dim_multiplier) > 0.0 else None
        ),
        hidden_dim_min=int(args.hidden_dim_min),
        n_epochs=int(args.n_epochs),
        device=str(args.device),
        cuda_device=int(args.cuda_device) if args.cuda_device is not None else None,
        violation_tau=float(args.violation_tau),
        torch_threads=int(args.torch_threads),
        skip_existing=bool(args.skip_existing),
    )

    out_cmds.write_text("\n".join(cmds) + ("\n" if cmds else ""), encoding="utf-8")
    print(f"wrote_cmds | {out_cmds} | n_commands={len(cmds)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
