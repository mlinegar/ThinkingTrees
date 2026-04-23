#!/usr/bin/env python3
"""Symmetric-family alternating f/g ladder across leaf-size rows.

Runs three pure backend families (``dspy``, ``trl``, ``fno``) through the
shared alternating trampoline in :mod:`src.ctreepo.alternating`. No cross-
family (g, f) pairs by design; each family pairs its own g with its own f
and co-adapts them via alternating optimization.

Iterations use the letter-count convention in
:mod:`src.ctreepo.alternating.stage_name_for_iteration`:

- ``k=0`` -> ``fg``     (f_init + g_init, no training)
- ``k=1`` -> ``fgf``    (train f1)
- ``k=2`` -> ``fgfg``   (train g1 with current f1 as the scoring signal)

Every row's output includes the **f-vs-f\\*** gap: ``internal_f_pearson``
(how closely our f agrees with the teacher f at the root) minus
``external_expert_pearson`` (Pearson vs gold expert scores). Positive gap =
our f is drifting from expert signal while still mimicking the teacher
(reward-hacking warning).
"""

from __future__ import annotations

import argparse
import collections
import importlib.metadata as _importlib_metadata
import json
import logging
import math
import sys
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

# Work around a Python 3.12 + transformers 5.3 import crash where
# ``importlib.metadata.packages_distributions()`` dereferences ``dist.metadata``
# and hits ``None['Name']`` for at least one installed package. Transformers
# evaluates this at import time, so the patch must be applied before any
# transformers import. Applying at the grid entry point covers every family.
def _safe_packages_distributions():
    pkg_to_dist = collections.defaultdict(list)
    for dist in _importlib_metadata.distributions():
        try:
            md = dist.metadata
            if md is None:
                continue
            name = md["Name"]
        except Exception:
            continue
        for top in (
            _importlib_metadata._top_level_declared(dist)
            or _importlib_metadata._top_level_inferred(dist)
        ):
            pkg_to_dist[top].append(name)
    return dict(pkg_to_dist)


_importlib_metadata.packages_distributions = _safe_packages_distributions

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.ctreepo.alternating import (  # noqa: E402
    IterationRecord,
    run_alternating_family,
)
from src.ctreepo.distillation import load_labeled_trees  # noqa: E402
from src.ctreepo.fg_arity import auto_g_output_tokens  # noqa: E402
from src.tree.labeled import LabeledTree  # noqa: E402

LOGGER = logging.getLogger(__name__)

DEFAULT_FG_GRID_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "manifesto_teacher_fg_leaf_grid"
    / "economic_gemma4_aligned_l1_2_4_8_16"
)

KNOWN_FAMILIES = ("dspy", "trl", "fno")


def _now_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def _parse_int_grid(value: Any) -> Tuple[int, ...]:
    if isinstance(value, (list, tuple)):
        grid = tuple(int(item) for item in value)
    else:
        parts = [part.strip() for part in str(value or "").replace(";", ",").split(",")]
        grid = tuple(int(part) for part in parts if part)
    if not grid:
        raise ValueError("leaf grid must contain at least one integer")
    if any(item <= 0 for item in grid):
        raise ValueError(f"leaf grid entries must be positive: {grid!r}")
    return grid


def _parse_families(value: str) -> Tuple[str, ...]:
    if not value:
        return KNOWN_FAMILIES
    tokens = [tok.strip().lower() for tok in str(value).replace(";", ",").split(",") if tok.strip()]
    if "all" in tokens:
        return KNOWN_FAMILIES
    unknown = [tok for tok in tokens if tok not in KNOWN_FAMILIES]
    if unknown:
        raise ValueError(
            f"unknown families: {unknown!r}. Allowed: {list(KNOWN_FAMILIES)} or 'all'"
        )
    return tuple(tok for tok in KNOWN_FAMILIES if tok in tokens)


def _split_trees_for_eval(
    trees: Sequence[LabeledTree],
    *,
    eval_split: str,
    train_split: str,
) -> Tuple[List[LabeledTree], List[LabeledTree]]:
    train_trees: List[LabeledTree] = []
    eval_trees: List[LabeledTree] = []
    for tree in trees:
        split = str((tree.metadata or {}).get("split") or "").lower()
        if split == train_split.lower():
            train_trees.append(tree)
        if split == eval_split.lower():
            eval_trees.append(tree)
    return train_trees, eval_trees


def _load_leaf_trees(fg_grid_dir: Path, leaf_count: int) -> Optional[List[LabeledTree]]:
    path = Path(fg_grid_dir) / f"leaf_{int(leaf_count):03d}" / "labeled_trees.jsonl"
    if not path.exists():
        return None
    return load_labeled_trees(path)


def _record_to_dict(record: IterationRecord) -> Dict[str, Any]:
    payload = asdict(record)
    # split_metrics are SplitMetrics dataclasses; asdict handles them.
    return payload


def _resolve_torch_device(value: str) -> Any:
    import torch

    requested = str(value or "auto").strip().lower()
    if requested == "auto":
        return None
    return torch.device(requested)


def _build_fno_family(args: argparse.Namespace, *, leaf_size_tokens: int) -> Any:
    from scripts.run_manifesto_dimension_fit_existing_results import (
        HashingEmbeddingClient,
    )
    from src.ctreepo.fno_family import FNOFamily, FNOFamilyConfig

    if args.embedding_backend == "hashing":
        embedding_client = HashingEmbeddingClient(dim=int(args.hashing_embedding_dim))
    elif args.embedding_backend == "local-hf":
        from scripts.run_manifesto_dimension_fit_existing_results import (
            LocalHFEmbeddingClient,
        )

        embedding_client = LocalHFEmbeddingClient(
            model=str(args.embedding_model),
            batch_size=int(args.embedding_batch_size),
            max_length=int(args.embedding_max_length),
            device=str(args.embedding_device),
        )
    else:
        raise ValueError(
            f"embedding backend {args.embedding_backend!r} not supported for FNO family; "
            "use 'hashing' or 'local-hf'"
        )
    # When using a real embedding model, enforce no-truncation against its
    # max_length. The hashing client has no tokenizer, so disable the assertion
    # in that path.
    embedding_max_length_tokens = (
        int(args.embedding_max_length) if args.embedding_backend == "local-hf" else None
    )
    base_embedding_dim = (
        int(args.hashing_embedding_dim)
        if args.embedding_backend == "hashing"
        else int(args.embedding_dim)
    )
    chunks_per_leaf = (
        1
        if embedding_max_length_tokens is None
        else max(1, math.ceil(int(leaf_size_tokens) / float(embedding_max_length_tokens)))
    )
    effective_embedding_dim = int(base_embedding_dim) * int(chunks_per_leaf)
    return FNOFamily(
        config=FNOFamilyConfig(
            hidden_channels=int(args.fno_hidden_channels),
            n_modes=int(args.fno_n_modes),
            n_layers=int(args.fno_n_layers),
            head_hidden_dim=int(args.fno_head_hidden_dim),
            epochs_per_iteration=int(args.fno_epochs),
            batch_size=int(args.fno_batch_size),
            learning_rate=float(args.fno_learning_rate),
            weight_decay=float(args.fno_weight_decay),
            grad_clip_norm=float(args.fno_grad_clip_norm),
            target_min=float(args.target_min),
            target_max=float(args.target_max),
            identity_init=True,
            seed=int(args.seed),
            leaf_size_tokens=int(leaf_size_tokens),
            embedding_max_length_tokens=embedding_max_length_tokens,
            tokenizer_model_path=str(args.embedding_model),
            effective_embedding_dim=effective_embedding_dim,
        ),
        embedding_client=embedding_client,
        device=_resolve_torch_device(args.fno_device),
    )


def _build_dspy_family(args: argparse.Namespace, *, leaf_size_tokens: int) -> Any:
    from src.ctreepo.dspy_family import DSPyFamily, DSPyFamilyConfig

    max_tokens = auto_g_output_tokens(
        int(args.dspy_max_tokens),
        leaf_size_tokens=int(leaf_size_tokens),
    )
    lm_config: Dict[str, Any] = {}
    if args.dspy_model:
        lm_config["model"] = str(args.dspy_model)
    if args.dspy_api_base:
        lm_config["api_base"] = str(args.dspy_api_base)
    if args.dspy_api_key:
        lm_config["api_key"] = str(args.dspy_api_key)
    lm_config["max_tokens"] = int(max_tokens)
    return DSPyFamily(
        config=DSPyFamilyConfig(
            optimizer=str(args.dspy_optimizer),
            budget=str(args.dspy_budget),
            num_threads=int(args.dspy_num_threads),
            target_min=float(args.target_min),
            target_max=float(args.target_max),
            lm_config=lm_config,
            leaf_size_tokens=int(leaf_size_tokens),
            lm_context_window_tokens=int(args.dspy_lm_context_tokens),
            max_completion_tokens=int(max_tokens),
            prompt_template_overhead_tokens=int(args.dspy_prompt_overhead_tokens),
            tokenizer_model_path=str(args.embedding_model),
            dimension=str(args.dimension),
        )
    )


def _build_trl_family(args: argparse.Namespace, *, leaf_size_tokens: int) -> Any:
    from src.ctreepo.trl_family import TRLFamily, TRLFamilyConfig

    max_tokens = auto_g_output_tokens(
        int(args.trl_max_tokens),
        leaf_size_tokens=int(leaf_size_tokens),
    )
    return TRLFamily(
        config=TRLFamilyConfig(
            g_base_model=str(args.trl_g_model),
            f_base_model=str(args.trl_f_model),
            target_min=float(args.target_min),
            target_max=float(args.target_max),
            leaf_size_tokens=int(leaf_size_tokens),
            lm_context_window_tokens=int(args.trl_lm_context_tokens),
            max_completion_tokens=int(max_tokens),
            tokenizer_model_path=str(args.embedding_model),
        )
    )


def _initial_artifact(family_name: str) -> str:
    # All families accept ``"identity"`` as the sentinel for the canonical
    # initial artifact; individual families resolve the semantics.
    return "identity"


def _run_family_row(
    *,
    family_name: str,
    args: argparse.Namespace,
    trees: Sequence[LabeledTree],
    axis_kind: str,
    axis_value: int,
    leaf_size_tokens: Optional[int],
    output_dir: Path,
) -> Dict[str, Any]:
    train_trees, eval_trees = _split_trees_for_eval(
        trees, eval_split=args.eval_split, train_split=args.train_split
    )
    if not eval_trees:
        eval_trees = list(trees)
    # Resolve the per-family budget axis. For DSPy the budget check is in
    # tokens (config-time hard error). FNO and TRL accept the value but
    # only DSPy uses it today. When running a legacy --leaf-grid (count)
    # path with no --leaf-size-tokens, fall back to a conservative default
    # consistent with the existing 12k-token vLLM server.
    effective_leaf_size_tokens = (
        int(leaf_size_tokens)
        if leaf_size_tokens is not None
        else int(args.dspy_leaf_size_tokens_fallback)
    )
    if family_name == "fno":
        family = _build_fno_family(args, leaf_size_tokens=effective_leaf_size_tokens)
    elif family_name == "dspy":
        family = _build_dspy_family(args, leaf_size_tokens=effective_leaf_size_tokens)
    elif family_name == "trl":
        family = _build_trl_family(args, leaf_size_tokens=effective_leaf_size_tokens)
    else:
        raise ValueError(f"unknown family: {family_name!r}")

    row_label = (
        f"leaf{int(axis_value):04d}tok"
        if axis_kind == "leaf_size_tokens"
        else f"leaf_{int(axis_value):03d}"
    )
    row_dir = output_dir / family_name / row_label
    row_dir.mkdir(parents=True, exist_ok=True)
    records = run_alternating_family(
        family=family,
        f_init=_initial_artifact(family_name),
        g_init=_initial_artifact(family_name),
        traces=train_trees if train_trees else list(trees),
        eval_trees=eval_trees,
        max_iterations=int(args.max_iterations),
        axis_kind=axis_kind,
        axis_value=int(axis_value),
        leaf_count=None if axis_kind == "leaf_size_tokens" else int(axis_value),
        leaf_size_tokens=leaf_size_tokens,
        first_train_side=str(args.first_train_side),
        initial_f_degree=int(args.initial_f_degree),
        initial_g_degree=int(args.initial_g_degree),
        stage_naming=str(args.stage_naming),
        output_dir=row_dir,
    )
    row_manifest_path = row_dir / "iteration_history.json"
    payload = {
        "family": family_name,
        "axis_kind": axis_kind,
        "axis_value": int(axis_value),
        "leaf_count": None if axis_kind == "leaf_size_tokens" else int(axis_value),
        "leaf_size_tokens": int(leaf_size_tokens) if leaf_size_tokens is not None else None,
        "row_label": row_label,
        "max_iterations": int(args.max_iterations),
        "eval_split": args.eval_split,
        "train_split": args.train_split,
        "n_train_trees": len(train_trees),
        "n_eval_trees": len(eval_trees),
        "iterations": [_record_to_dict(r) for r in records],
    }
    row_manifest_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    LOGGER.info("Wrote %s", row_manifest_path)
    return payload


def _summarize_grid(grid_rows: List[Dict[str, Any]], *, eval_split: str) -> List[Dict[str, Any]]:
    """Flatten per-row iteration histories into a single table."""
    table: List[Dict[str, Any]] = []
    for row in grid_rows:
        family = row["family"]
        axis_kind = row.get("axis_kind", "leaf_count")
        axis_value = row.get("axis_value", row.get("leaf_count"))
        for it in row["iterations"]:
            split_metrics = it.get("split_metrics", {}) or {}
            sm = split_metrics.get(eval_split) or split_metrics.get("all") or {}
            table.append(
                {
                    "family": family,
                    "axis_kind": axis_kind,
                    "axis_value": axis_value,
                    "leaf_count": row.get("leaf_count"),
                    "leaf_size_tokens": row.get("leaf_size_tokens"),
                    "iteration": it.get("iteration"),
                    "stage_name": it.get("stage_name"),
                    "stage_label": it.get("stage_label") or it.get("stage_name"),
                    "f_degree": it.get("f_degree"),
                    "g_degree": it.get("g_degree"),
                    "trained": it.get("trained"),
                    "n_eval": sm.get("n"),
                    "internal_f_pearson": sm.get("internal_f_pearson"),
                    "external_expert_pearson": sm.get("external_expert_pearson"),
                    "f_star_gap": sm.get("f_star_gap"),
                    "internal_f_mae_1_7": sm.get("internal_f_mae_1_7"),
                    "external_expert_mae_1_7": sm.get("external_expert_mae_1_7"),
                    "mean_prediction_1_7": sm.get("mean_prediction_1_7"),
                    "mean_teacher_1_7": sm.get("mean_teacher_1_7"),
                    "mean_expert_1_7": sm.get("mean_expert_1_7"),
                }
            )
    return table


def _write_markdown_summary(rows: List[Dict[str, Any]], path: Path, *, eval_split: str) -> None:
    def fmt(value: Any, width: int = 8, digits: int = 3) -> str:
        if value is None:
            return " n/a".rjust(width)
        if isinstance(value, int):
            return f"{value:>{width}d}"
        return f"{float(value):>{width}.{digits}f}"

    lines: List[str] = []
    lines.append(f"# Alternating ladder grid summary ({eval_split} split)")
    lines.append("")
    header = (
        "| family | axis | k | stage | trained | n | int_p | ext_p | f_star_gap | "
        "int_mae | ext_mae | mean_p | mean_t | mean_e |"
    )
    sep = "|" + "|".join("-" * (len(seg) + 2) for seg in header.strip("|").split("|")) + "|"
    lines.append(header)
    lines.append(sep)
    for row in rows:
        lines.append(
                "| {family} | {axis} | {k} | {stage} | {trained} | {n} | {ip} | {ep} | {gap} | "
                "{im} | {em} | {mp} | {mt} | {me} |".format(
                family=row["family"],
                axis=(
                    f"leaf{int(row['leaf_size_tokens']):04d}tok"
                    if row.get("leaf_size_tokens") is not None
                    else f"leaf_{int(row.get('leaf_count') or row.get('axis_value') or 0):03d}"
                ),
                k=row["iteration"],
                stage=row.get("stage_label") or row["stage_name"],
                trained=row["trained"],
                n=fmt(row.get("n_eval"), width=4, digits=0),
                ip=fmt(row.get("internal_f_pearson")),
                ep=fmt(row.get("external_expert_pearson")),
                gap=fmt(row.get("f_star_gap")),
                im=fmt(row.get("internal_f_mae_1_7")),
                em=fmt(row.get("external_expert_mae_1_7")),
                mp=fmt(row.get("mean_prediction_1_7")),
                mt=fmt(row.get("mean_teacher_1_7")),
                me=fmt(row.get("mean_expert_1_7")),
            )
        )
    lines.append("")
    lines.append("Columns: `int_p` = internal Pearson (our f vs teacher f at root); "
                 "`ext_p` = external Pearson (our f vs gold expert); "
                 "`f_star_gap` = int_p - ext_p (positive = reward-hacking warning).")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Alternating f/g ladder across symmetric backend families (dspy, trl, fno)."
        )
    )
    parser.add_argument(
        "--families",
        type=str,
        default="fno,dspy",
        help=(
            "Comma-separated list of families to run, or 'all'. Default excludes "
            "TRL because TRL k>=1 is still scaffolded."
        ),
    )
    parser.add_argument("--dimension", default="economic")
    parser.add_argument(
        "--fg-grid-dir",
        "--teacher-dir",
        dest="fg_grid_dir",
        type=Path,
        default=DEFAULT_FG_GRID_DIR,
        help=(
            "Teacher-trace base directory. With --leaf-grid, expects subdirs "
            "leaf_NNN/. With --leaf-size-tokens, expects subdirs leafTTTtok/."
        ),
    )
    parser.add_argument(
        "--leaf-grid",
        type=str,
        default=None,
        help="LEGACY count-based axis (e.g. 1,2,4,8,16). Mutex with --leaf-size-tokens.",
    )
    parser.add_argument(
        "--leaf-size-tokens",
        type=str,
        default=None,
        help=(
            "Size-based leaf axis (tokens per leaf), e.g. 512,1024,2048. "
            "Each entry resolves to <fg-grid-dir>/leaf{TTT}tok/labeled_trees.jsonl. "
            "Mutex with --leaf-grid."
        ),
    )
    parser.add_argument("--max-iterations", type=int, default=2)
    parser.add_argument(
        "--first-train-side",
        choices=("f", "g"),
        default="f",
        help="Which side gets the first learned update after the k=0 baseline.",
    )
    parser.add_argument(
        "--initial-f-degree",
        type=int,
        default=1,
        help="Initial f exponent for power-notation tracking.",
    )
    parser.add_argument(
        "--initial-g-degree",
        type=int,
        default=1,
        help="Initial g exponent for power-notation tracking.",
    )
    parser.add_argument(
        "--stage-naming",
        choices=("legacy", "powers"),
        default="legacy",
        help=(
            "Stage id scheme. 'legacy' preserves fg/fgf/... for the canonical "
            "f-first (f^1,g^1) ladder; other setups fall back to power ids."
        ),
    )
    parser.add_argument("--train-split", default="train")
    parser.add_argument("--eval-split", default="test")
    parser.add_argument("--target-min", type=float, default=1.0)
    parser.add_argument("--target-max", type=float, default=7.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--verbose", action="store_true")

    # FNO family
    parser.add_argument("--embedding-backend", choices=["hashing", "local-hf"], default="local-hf")
    parser.add_argument(
        "--embedding-model",
        type=str,
        default="/mnt/data/models/google/embeddinggemma-300m",
        help=(
            "Embedding model. EmbeddingGemma-300m is the default (Gemma-family "
            "tokenizer; max_position_embeddings=2048 covers leaf sizes up to "
            "2048 tokens with no chunk-within-leaf needed; embedding dim 768)."
        ),
    )
    parser.add_argument("--hashing-embedding-dim", type=int, default=256)
    parser.add_argument(
        "--embedding-dim",
        type=int,
        default=768,
        help=(
            "Base embedding dimension for local-HF models. The FNO config checks "
            "D_eff = embedding_dim * ceil(leaf_size_tokens / embedding_max_length). "
            "EmbeddingGemma-300m defaults to D=768."
        ),
    )
    parser.add_argument("--embedding-batch-size", type=int, default=8)
    parser.add_argument("--embedding-max-length", type=int, default=2048)
    parser.add_argument("--embedding-device", default="auto")
    parser.add_argument("--fno-hidden-channels", type=int, default=32)
    parser.add_argument("--fno-n-modes", type=int, default=64)
    parser.add_argument("--fno-n-layers", type=int, default=2)
    parser.add_argument("--fno-head-hidden-dim", type=int, default=64)
    parser.add_argument("--fno-epochs", type=int, default=8)
    parser.add_argument("--fno-batch-size", type=int, default=2)
    parser.add_argument("--fno-learning-rate", type=float, default=1e-3)
    parser.add_argument("--fno-weight-decay", type=float, default=1e-4)
    parser.add_argument("--fno-grad-clip-norm", type=float, default=1.0)
    parser.add_argument(
        "--fno-device",
        default="auto",
        help=(
            "Torch device for the FNO model. Use 'cpu' when the vLLM server "
            "occupies all GPUs for DSPy/teacher LLM calls. Production runs "
            "should generally schedule FNO separately on an explicit GPU."
        ),
    )

    # DSPy family
    parser.add_argument("--dspy-optimizer", default="mipro")
    parser.add_argument("--dspy-budget", default="light")
    parser.add_argument("--dspy-num-threads", type=int, default=4)
    parser.add_argument("--dspy-model", default="openai/nvidia/Gemma-4-31B-IT-NVFP4")
    parser.add_argument("--dspy-api-base", default="http://localhost:8010/v1")
    parser.add_argument("--dspy-api-key", default="EMPTY")
    parser.add_argument(
        "--dspy-max-tokens",
        type=int,
        default=0,
        help=(
            "Max generated tokens per LM call. 0 means auto = "
            "2 * leaf_size_tokens, so g can emit a verbatim concatenation of "
            "two children."
        ),
    )
    parser.add_argument(
        "--dspy-lm-context-tokens",
        type=int,
        default=12000,
        help=(
            "Total LM context window in tokens. Default 12000 matches the vLLM "
            "server's --max-model-len 12000. The DSPy family hard-errors at "
            "construction time if 2 * leaf_size_tokens + max_completion_tokens "
            "+ prompt_template_overhead_tokens > this budget."
        ),
    )
    parser.add_argument(
        "--dspy-prompt-overhead-tokens",
        type=int,
        default=1500,
        help="Conservative estimate of DSPy signature template + demo stacking overhead.",
    )
    parser.add_argument(
        "--dspy-leaf-size-tokens-fallback",
        type=int,
        default=512,
        help=(
            "When --leaf-grid (count-based) is in use and --leaf-size-tokens is "
            "absent, this value seeds the DSPy budget check. With 12k context "
            "and auto max_tokens=2*leaf_size_tokens, 512 leaf tokens × 2 fits."
        ),
    )

    # TRL family
    parser.add_argument("--trl-g-model", default="nvidia/Gemma-4-31B-IT-NVFP4")
    parser.add_argument("--trl-f-model", default="nvidia/Gemma-4-31B-IT-NVFP4")
    parser.add_argument(
        "--trl-max-tokens",
        type=int,
        default=0,
        help="0 means auto = 2 * leaf_size_tokens.",
    )
    parser.add_argument("--trl-lm-context-tokens", type=int, default=12000)
    return parser.parse_args(argv)


def _load_leaf_size_trees(fg_grid_dir: Path, leaf_size_tokens: int) -> Optional[List[LabeledTree]]:
    """Load size-based teacher traces from <fg_grid_dir>/leaf{TTT}tok/labeled_trees.jsonl."""
    path = Path(fg_grid_dir) / f"leaf{int(leaf_size_tokens):04d}tok" / "labeled_trees.jsonl"
    if not path.exists():
        return None
    return load_labeled_trees(path)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    families = _parse_families(args.families)

    # Resolve leaf axis: count-based (legacy) XOR size-based (new).
    if args.leaf_grid and args.leaf_size_tokens:
        raise SystemExit("--leaf-grid and --leaf-size-tokens are mutually exclusive")
    if args.leaf_size_tokens:
        leaf_size_axis = _parse_int_grid(args.leaf_size_tokens)
        leaf_count_axis: Optional[Tuple[int, ...]] = None
        leaf_axis_name = "leaf_size_tokens"
    elif args.leaf_grid:
        leaf_count_axis = _parse_int_grid(args.leaf_grid)
        leaf_size_axis = None
        leaf_axis_name = "leaf_count"
    else:
        # No axis explicitly given: default to size-based sweep.
        leaf_size_axis = (512, 1024, 2048)
        leaf_count_axis = None
        leaf_axis_name = "leaf_size_tokens"

    output_dir = args.output_dir or (
        PROJECT_ROOT
        / "outputs"
        / "manifesto_fg_alternating"
        / f"{args.dimension}_{_now_stamp()}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    LOGGER.info(
        "Running families=%s %s=%s max_iterations=%d -> %s",
        families, leaf_axis_name,
        leaf_size_axis if leaf_size_axis is not None else leaf_count_axis,
        args.max_iterations, output_dir,
    )

    grid_rows: List[Dict[str, Any]] = []

    def _iter_axis():
        if leaf_size_axis is not None:
            for tok in leaf_size_axis:
                trees = _load_leaf_size_trees(args.fg_grid_dir, int(tok))
                yield ("leaf_size_tokens", int(tok), trees)
        else:
            assert leaf_count_axis is not None
            for cnt in leaf_count_axis:
                trees = _load_leaf_trees(args.fg_grid_dir, int(cnt))
                yield ("leaf_count", int(cnt), trees)

    for axis_kind, axis_value, trees in _iter_axis():
        if not trees:
            kind_label = (
                f"leaf{axis_value:04d}tok"
                if axis_kind == "leaf_size_tokens"
                else f"leaf_{axis_value:03d}"
            )
            LOGGER.warning(
                "No labeled trees for %s under %s; skipping",
                kind_label, args.fg_grid_dir,
            )
            continue
        # Pass leaf_size_tokens=None for count-based runs; the family builder
        # falls back to --dspy-leaf-size-tokens-fallback for the budget check.
        leaf_size_tokens_for_row = axis_value if axis_kind == "leaf_size_tokens" else None
        for family_name in families:
            try:
                row = _run_family_row(
                    family_name=family_name,
                    args=args,
                    trees=trees,
                    axis_kind=axis_kind,
                    axis_value=axis_value,
                    leaf_size_tokens=leaf_size_tokens_for_row,
                    output_dir=output_dir,
                )
            except Exception as exc:
                LOGGER.exception(
                    "family=%s %s=%d: unexpected error during row run -- %s",
                    family_name, axis_kind, axis_value, exc,
                )
                row = {
                    "family": family_name,
                    "axis_kind": axis_kind,
                    "axis_value": int(axis_value),
                    "leaf_count": None if axis_kind == "leaf_size_tokens" else int(axis_value),
                    "leaf_size_tokens": (
                        int(axis_value) if axis_kind == "leaf_size_tokens" else None
                    ),
                    "row_label": (
                        f"leaf{int(axis_value):04d}tok"
                        if axis_kind == "leaf_size_tokens"
                        else f"leaf_{int(axis_value):03d}"
                    ),
                    "max_iterations": int(args.max_iterations),
                    "iterations": [],
                    "error": f"{type(exc).__name__}: {exc}",
                }
            grid_rows.append(row)

    summary_table = _summarize_grid(grid_rows, eval_split=args.eval_split)
    summary_json_path = output_dir / "grid_summary.json"
    summary_json_path.write_text(
        json.dumps(
            {
                "created_at": _now_stamp(),
                "dimension": args.dimension,
                "families": list(families),
                "topology_axis": leaf_axis_name,
                "leaf_grid": list(leaf_count_axis) if leaf_count_axis is not None else None,
                "leaf_size_tokens": list(leaf_size_axis) if leaf_size_axis is not None else None,
                "max_iterations": int(args.max_iterations),
                "first_train_side": str(args.first_train_side),
                "initial_f_degree": int(args.initial_f_degree),
                "initial_g_degree": int(args.initial_g_degree),
                "stage_naming": str(args.stage_naming),
                "eval_split": args.eval_split,
                "rows": summary_table,
                "per_row_paths": [
                    f"{r['family']}/{r.get('row_label')}/iteration_history.json"
                    for r in grid_rows
                ],
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    md_path = output_dir / "grid_summary.md"
    _write_markdown_summary(summary_table, md_path, eval_split=args.eval_split)
    LOGGER.info("Wrote %s and %s", summary_json_path, md_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
