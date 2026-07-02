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
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

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

from src.ctreepo.learning import (  # noqa: E402
    run_family_runtime_ladder,
    schedule_from_max_iterations,
)
from src.ctreepo.contracts import (  # noqa: E402
    LAW_ID_LEAF_PRESERVATION,
    LAW_ID_MERGE_PRESERVATION,
    LAW_ID_ON_RANGE_IDEMPOTENCE,
    LEAF_UNIT_TEXT_TOKEN,
    LOCAL_LAW_ESTIMATOR_PROXY_ONLY,
    SOURCE_KIND_EXTERNAL_STATE,
    SOURCE_KIND_RAW_INPUT,
    TREE_BUNDLE_SCHEMA_VERSION,
    normalize_tree_bundle_manifest,
    objective_metadata,
    run_manifest_metadata,
    tree_bundle_manifest_digest,
    validate_tree_bundle_manifest,
)
from src.core.batch_transport import (  # noqa: E402
    DEFAULT_BATCH_MAX_CONCURRENT,
    DEFAULT_BATCH_REQUEST_TIMEOUT_SECONDS,
    DEFAULT_BATCH_ROUTING_POLICY,
    DEFAULT_BATCH_SIZE,
    DEFAULT_BATCH_TIMEOUT_SECONDS,
)
from src.experiments.embedding_clients import (  # noqa: E402
    HashingEmbeddingClient,
    LocalHFEmbeddingClient,
)
from src.experiments.ladder_reporting import (  # noqa: E402
    summarize_ladder_grid,
    write_alternating_markdown_summary,
)
from src.experiments.script_io import now_stamp as _now_stamp  # noqa: E402
from src.experiments.script_parse import parse_int_grid as _parse_int_grid  # noqa: E402
from src.experiments.tree_helpers import (  # noqa: E402
    load_leaf_count_trees as _load_leaf_trees,
    load_leaf_size_trees as _load_leaf_size_trees,
    split_trees_for_eval,
)
from src.ctreepo.fg_arity import (  # noqa: E402
    auto_g_output_tokens,
    two_child_lm_budget_report,
)
from src.tasks.manifesto.expert_scale import (  # noqa: E402
    EXPERT_SCALE_NORMALIZED_1_7,
    expert_scale_bounds,
)
from src.tree.labeled import LabeledTree  # noqa: E402

LOGGER = logging.getLogger(__name__)

DEFAULT_FG_GRID_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "manifesto_teacher_fg_leaf_grid"
    / "economic_gemma4_aligned_l1_2_4_8_16"
)

KNOWN_FAMILIES = ("dspy", "trl", "fno")



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


def _preflight_dspy_budget(
    args: argparse.Namespace,
    *,
    leaf_size_axis: Optional[Tuple[int, ...]],
) -> None:
    """Fail before running any rows if the requested DSPy grid cannot fit."""
    if leaf_size_axis is None:
        leaves = (int(args.dspy_leaf_size_tokens_fallback),)
    else:
        leaves = tuple(int(tok) for tok in leaf_size_axis)
    for leaf_size_tokens in leaves:
        report = two_child_lm_budget_report(
            family_name="dspy",
            leaf_size_tokens=int(leaf_size_tokens),
            lm_context_window_tokens=int(args.dspy_lm_context_tokens),
            max_completion_tokens=int(args.dspy_max_tokens),
            prompt_template_overhead_tokens=int(args.dspy_prompt_overhead_tokens),
        )
        if not report.ok:
            raise SystemExit(
                "DSPy budget preflight failed for "
                f"leaf_size_tokens={leaf_size_tokens}: "
                f"{'; '.join(report.violations)}. "
                "Use --dspy-max-tokens 0 for auto=2*leaf_size_tokens, "
                f"increase --dspy-lm-context-tokens to at least "
                f"{report.minimum_context_window_tokens}, or reduce the leaf grid."
            )


def _preflight_dspy_f_warm_start(
    args: argparse.Namespace,
    *,
    leaf_size_axis: Optional[Tuple[int, ...]],
) -> None:
    if args.dspy_f_init_path is None or not str(args.dspy_f_init_path):
        return
    if str(args.dspy_f_init_mode) not in {"pretuned_scorer", "identity"}:
        return
    if leaf_size_axis is None:
        leaf_size_tokens = int(args.dspy_leaf_size_tokens_fallback)
    else:
        leaves = tuple(int(tok) for tok in leaf_size_axis)
        leaf_size_tokens = int(leaves[0]) if leaves else int(args.dspy_leaf_size_tokens_fallback)
    family = _build_dspy_family(args, leaf_size_tokens=leaf_size_tokens)
    artifact = _initial_f_artifact("dspy", args)
    program = family._load_f_program(artifact)
    if getattr(program, "__class__", type(program)).__name__ == "DimensionScorer":
        raise RuntimeError(
            "Configured --dspy-f-init-path resolved to a bare DimensionScorer; "
            "refusing to run because the requested warm-start program was not used."
        )
    if not family._program_accepts_summary_input(program):
        raise RuntimeError(
            "Configured DSPy f warm-start does not expose the tree ladder "
            "f(summary) interface after loading/adaptation."
        )
    LOGGER.info(
        "DSPy f warm-start preflight loaded %s as %s",
        artifact,
        getattr(program, "__class__", type(program)).__name__,
    )



def _record_to_dict(record: Mapping[str, Any] | Any) -> Dict[str, Any]:
    if isinstance(record, Mapping):
        return dict(record)
    payload = asdict(record)
    # split_metrics are SplitMetrics dataclasses; asdict handles them.
    return payload


def _resolve_torch_device(value: str) -> Any:
    import torch

    requested = str(value or "auto").strip().lower()
    if requested == "auto":
        return None
    return torch.device(requested)


def _resolved_target_bounds(args: argparse.Namespace) -> Tuple[float, float]:
    if args.target_min is not None and args.target_max is not None:
        return float(args.target_min), float(args.target_max)
    if str(args.dimension).strip().lower() == "environment":
        default_min, default_max = expert_scale_bounds(
            dimension="environment",
            scale=EXPERT_SCALE_NORMALIZED_1_7,
        )
    else:
        default_min, default_max = 1.0, 7.0
    target_min = float(default_min if args.target_min is None else args.target_min)
    target_max = float(default_max if args.target_max is None else args.target_max)
    return target_min, target_max


def _effective_local_law_weight(args: argparse.Namespace) -> float:
    root_label_sources = _root_label_sources(args)
    local_raw = getattr(args, "local_law_weight", None)
    if not root_label_sources:
        if local_raw is not None and not math.isclose(float(local_raw), 1.0):
            raise ValueError("empty --root-label-sources requires --local-law-weight 1.0")
        return 1.0
    if local_raw is not None:
        local_value = float(local_raw)
        if not math.isfinite(local_value) or local_value < 0.0 or local_value > 1.0:
            raise ValueError(f"local_law_weight must be in [0, 1], got {local_raw!r}")
        return float(local_value)
    return 0.25


def _effective_root_anchor_weight(args: argparse.Namespace) -> float:
    if not _root_label_sources(args):
        _effective_local_law_weight(args)
        return 0.0
    return float(1.0 - _effective_local_law_weight(args))


def _root_label_sources(args: argparse.Namespace) -> Tuple[str, ...]:
    raw = str(getattr(args, "root_label_sources", "") or "").strip()
    if not raw:
        return tuple()
    sources: List[str] = []
    for part in raw.split(","):
        source = part.strip().lower().replace("-", "_")
        if not source:
            continue
        if source in {"stored", "summary"}:
            source = "stored_summary"
        if source == "raw":
            source = "raw_document"
        if source not in {"stored_summary", "raw_document"}:
            raise ValueError(
                f"unknown root label source {part!r}; expected stored_summary or raw_document"
            )
        if source not in sources:
            sources.append(source)
    return tuple(sources)


def _canonical_teacher_node_component_weights(local_law_weight: float) -> Dict[str, float]:
    share = float(local_law_weight) / 3.0
    return {
        LAW_ID_LEAF_PRESERVATION: share,
        LAW_ID_MERGE_PRESERVATION: share,
        LAW_ID_ON_RANGE_IDEMPOTENCE: share,
    }


def _objective_summary(args: argparse.Namespace) -> Dict[str, Any]:
    root_share = float(_effective_root_anchor_weight(args))
    local_law_weight = float(_effective_local_law_weight(args))
    target_min, target_max = _resolved_target_bounds(args)
    return {
        "root_label_sources": list(_root_label_sources(args)),
        "root_label_target": str(getattr(args, "root_label_target", "expert")),
        "root_share": root_share,
        "local_law_weight": local_law_weight,
        "local_law_component_weights": _canonical_teacher_node_component_weights(
            local_law_weight
        ),
        "teacher_trace_component": "teacher_node_trace",
        "node_weight_normalization": str(getattr(args, "node_weight_normalization", "per_tree")),
        "target_min": float(target_min),
        "target_max": float(target_max),
        "scorer_output_min": float(args.scorer_output_min),
        "scorer_output_max": float(args.scorer_output_max),
    }


def _build_fno_family(args: argparse.Namespace, *, leaf_size_tokens: int) -> Any:
    from src.ctreepo.fno_family import FNOFamily, FNOFamilyConfig

    if args.embedding_backend == "hashing":
        embedding_client = HashingEmbeddingClient(dim=int(args.hashing_embedding_dim))
    elif args.embedding_backend == "local-hf":
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
    target_min, target_max = _resolved_target_bounds(args)
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
            target_min=target_min,
            target_max=target_max,
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
    if str(args.dimension).strip().lower() in {"combined", "joint", "all", "all6"}:
        from src.ctreepo.joint_dspy_family import JointDSPyFamily, JointDSPyFamilyConfig

        target_min, target_max = _resolved_target_bounds(args)
        joint_f_init_path = (
            str(args.dspy_f_init_path)
            if args.dspy_f_init_path is not None
            else "outputs/phase2/joint_gepa/optimized_program.json"
        )

        return JointDSPyFamily(
            config=JointDSPyFamilyConfig(
                optimizer=str(args.dspy_optimizer),
                budget=str(args.dspy_budget),
                num_threads=int(args.dspy_num_threads),
                target_min=target_min,
                target_max=target_max,
                scorer_output_min=float(args.scorer_output_min),
                scorer_output_max=float(args.scorer_output_max),
                lm_config=lm_config,
                lm_transport=str(args.dspy_lm_transport),
                batch_max_concurrent=int(args.dspy_batch_max_concurrent),
                batch_size=int(args.dspy_batch_size),
                batch_timeout=float(args.dspy_batch_timeout),
                batch_request_timeout=float(args.dspy_batch_request_timeout),
                batch_await_response_timeout=args.dspy_batch_await_response_timeout,
                batch_routing_policy=str(args.dspy_batch_routing_policy),
                mipro_num_candidates=args.dspy_mipro_num_candidates,
                mipro_num_trials=args.dspy_mipro_num_trials,
                mipro_max_bootstrapped_demos=args.dspy_mipro_max_bootstrapped_demos,
                mipro_max_labeled_demos=args.dspy_mipro_max_labeled_demos,
                mipro_minibatch_size=int(args.dspy_mipro_minibatch_size),
                mipro_minibatch_full_eval_steps=int(
                    args.dspy_mipro_minibatch_full_eval_steps
                ),
                max_train_records=(
                    None
                    if int(args.dspy_max_train_records) <= 0
                    else int(args.dspy_max_train_records)
                ),
                record_sample_seed=int(args.seed),
                leaf_size_tokens=int(leaf_size_tokens),
                lm_context_window_tokens=int(args.dspy_lm_context_tokens),
                max_completion_tokens=int(max_tokens),
                prompt_template_overhead_tokens=int(args.dspy_prompt_overhead_tokens),
                tokenizer_model_path=str(args.embedding_model),
                dimension="combined",
                f_init_path=joint_f_init_path,
                f_init_mode=str(args.dspy_f_init_mode),
                root_label_sources=_root_label_sources(args),
                root_label_target=str(args.root_label_target),
                local_law_weight=args.local_law_weight,
                node_weight_normalization=str(args.node_weight_normalization),
            )
        )

    from src.ctreepo.dspy_family import DSPyFamily, DSPyFamilyConfig

    target_min, target_max = _resolved_target_bounds(args)
    return DSPyFamily(
        config=DSPyFamilyConfig(
            optimizer=str(args.dspy_optimizer),
            budget=str(args.dspy_budget),
            num_threads=int(args.dspy_num_threads),
            target_min=target_min,
            target_max=target_max,
            scorer_output_min=float(args.scorer_output_min),
            scorer_output_max=float(args.scorer_output_max),
            lm_config=lm_config,
            lm_transport=str(args.dspy_lm_transport),
            batch_max_concurrent=int(args.dspy_batch_max_concurrent),
            batch_size=int(args.dspy_batch_size),
            batch_timeout=float(args.dspy_batch_timeout),
            batch_request_timeout=float(args.dspy_batch_request_timeout),
            batch_await_response_timeout=args.dspy_batch_await_response_timeout,
            batch_routing_policy=str(args.dspy_batch_routing_policy),
            mipro_num_candidates=args.dspy_mipro_num_candidates,
            mipro_num_trials=args.dspy_mipro_num_trials,
            mipro_max_bootstrapped_demos=args.dspy_mipro_max_bootstrapped_demos,
            mipro_max_labeled_demos=args.dspy_mipro_max_labeled_demos,
            mipro_minibatch_size=int(args.dspy_mipro_minibatch_size),
            mipro_minibatch_full_eval_steps=int(
                args.dspy_mipro_minibatch_full_eval_steps
            ),
            max_train_records=(
                None
                if int(args.dspy_max_train_records) <= 0
                else int(args.dspy_max_train_records)
            ),
            record_sample_seed=int(args.seed),
            leaf_size_tokens=int(leaf_size_tokens),
            lm_context_window_tokens=int(args.dspy_lm_context_tokens),
            max_completion_tokens=int(max_tokens),
            prompt_template_overhead_tokens=int(args.dspy_prompt_overhead_tokens),
            tokenizer_model_path=str(args.embedding_model),
            dimension=str(args.dimension),
            f_init_path=str(args.dspy_f_init_path) if args.dspy_f_init_path is not None else None,
            f_init_mode=str(args.dspy_f_init_mode),
            root_label_sources=_root_label_sources(args),
            root_label_target=str(args.root_label_target),
            local_law_weight=args.local_law_weight,
            node_weight_normalization=str(args.node_weight_normalization),
        )
    )


def _build_trl_family(args: argparse.Namespace, *, leaf_size_tokens: int) -> Any:
    from src.ctreepo.trl_family import TRLFamily, TRLFamilyConfig

    max_tokens = auto_g_output_tokens(
        int(args.trl_max_tokens),
        leaf_size_tokens=int(leaf_size_tokens),
    )
    target_min, target_max = _resolved_target_bounds(args)
    return TRLFamily(
        config=TRLFamilyConfig(
            g_base_model=str(args.trl_g_model),
            f_base_model=str(args.trl_f_model),
            target_min=target_min,
            target_max=target_max,
            leaf_size_tokens=int(leaf_size_tokens),
            lm_context_window_tokens=int(args.trl_lm_context_tokens),
            max_completion_tokens=int(max_tokens),
            tokenizer_model_path=str(args.embedding_model),
        )
    )


def _initial_f_artifact(family_name: str, args: argparse.Namespace) -> str:
    if family_name == "dspy":
        mode = str(args.dspy_f_init_mode)
        if mode in {"pretuned_scorer", "identity"} and args.dspy_f_init_path is not None:
            path = str(args.dspy_f_init_path)
            if path:
                return path
        return mode
    return "identity"


def _initial_g_artifact(family_name: str, args: argparse.Namespace) -> str:
    if family_name == "dspy":
        return str(args.dspy_g_init_mode)
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
    train_trees, eval_trees = split_trees_for_eval(
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
    f_init = _initial_f_artifact(family_name, args)
    g_init = _initial_g_artifact(family_name, args)
    schedule = schedule_from_max_iterations(
        int(args.max_iterations),
        first_train_side=str(args.first_train_side),
    )
    fit_result = run_family_runtime_ladder(
        family=family,
        f_init=f_init,
        g_init=g_init,
        traces=train_trees if train_trees else list(trees),
        eval_trees=eval_trees,
        schedule=schedule,
        axis_kind=axis_kind,
        axis_value=int(axis_value),
        leaf_count=None if axis_kind == "leaf_size_tokens" else int(axis_value),
        leaf_size_tokens=leaf_size_tokens,
        first_train_side=str(args.first_train_side),
        initial_f_degree=int(args.initial_f_degree),
        initial_g_degree=int(args.initial_g_degree),
        stage_naming=str(args.stage_naming),
        output_dir=row_dir,
        metadata={
            "legacy_max_iterations": int(args.max_iterations),
            "legacy_entrypoint": "scripts/run_alternating_ladder.py",
            "tree_bundle": str(args.fg_grid_dir),
        },
    )
    records = list(fit_result.history)
    row_manifest_path = row_dir / "iteration_history.json"
    payload = {
        "family": family_name,
        "axis_kind": axis_kind,
        "axis_value": int(axis_value),
        "leaf_count": None if axis_kind == "leaf_size_tokens" else int(axis_value),
        "leaf_size_tokens": int(leaf_size_tokens) if leaf_size_tokens is not None else None,
        "row_label": row_label,
        "tree_bundle": str(args.fg_grid_dir),
        "max_iterations": int(args.max_iterations),
        "eval_split": args.eval_split,
        "train_split": args.train_split,
        "n_train_trees": len(train_trees),
        "n_eval_trees": len(eval_trees),
        "f_init": f_init,
        "g_init": g_init,
        "objective": _objective_summary(args),
        "iterations": [_record_to_dict(r) for r in records],
    }
    row_manifest_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    LOGGER.info("Wrote %s", row_manifest_path)
    return payload



def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    raw_tokens = list(sys.argv[1:] if argv is None else argv)
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
        "--tree-bundle",
        dest="fg_grid_dir",
        type=Path,
        default=DEFAULT_FG_GRID_DIR,
        help=(
            "Saved LabeledTree bundle directory. With --leaf-grid, expects "
            "leaf_NNN/labeled_trees.jsonl. With --leaf-size-tokens, expects "
            "leafTTTtok/labeled_trees.jsonl."
        ),
    )
    parser.add_argument(
        "--fg-grid-dir",
        "--teacher-dir",
        dest="fg_grid_dir",
        type=Path,
        default=argparse.SUPPRESS,
        help=(
            "Deprecated alias for --tree-bundle."
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
            "Each entry resolves to <tree-bundle>/leaf{TTT}tok/labeled_trees.jsonl. "
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
    parser.add_argument(
        "--preflight-only",
        action="store_true",
        help="Validate grid budgets and exit before loading teacher traces or running rows.",
    )
    parser.add_argument(
        "--fail-on-row-error",
        action="store_true",
        help=(
            "Exit nonzero if any grid row errors. Partial row summaries are still "
            "written before exit."
        ),
    )
    parser.add_argument(
        "--allow-legacy-tree-bundle",
        action="store_true",
        help="Permit deprecated or missing TreeBundle v1 metadata for explicit compatibility runs.",
    )
    parser.add_argument(
        "--allow-external-state-tree-bundle",
        action="store_true",
        help="Permit source_kind=external_state when paired with teacher_passthrough g.",
    )
    parser.add_argument(
        "--target-min",
        type=float,
        default=None,
        help=(
            "Objective target lower bound. Omit for the normalized 1-7 "
            "internal default."
        ),
    )
    parser.add_argument(
        "--target-max",
        type=float,
        default=None,
        help=(
            "Objective target upper bound. Omit for the normalized 1-7 "
            "internal default."
        ),
    )
    parser.add_argument("--scorer-output-min", type=float, default=1.0)
    parser.add_argument("--scorer-output-max", type=float, default=7.0)
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
    parser.add_argument("--dspy-num-threads", type=int, default=128)
    parser.add_argument(
        "--dspy-mipro-num-candidates",
        type=int,
        default=4,
        help=(
            "Manual MIPRO candidate count. Setting this or --dspy-mipro-num-trials "
            "uses auto=None, which avoids DSPy's fixed auto run plan."
        ),
    )
    parser.add_argument(
        "--dspy-mipro-num-trials",
        type=int,
        default=4,
        help="Manual MIPRO trial count. Implies auto=None.",
    )
    parser.add_argument(
        "--dspy-mipro-max-bootstrapped-demos",
        type=int,
        default=0,
        help="Override MIPRO max_bootstrapped_demos; use 0 to skip serial bootstrapping.",
    )
    parser.add_argument(
        "--dspy-mipro-max-labeled-demos",
        type=int,
        default=0,
        help="Override MIPRO max_labeled_demos.",
    )
    parser.add_argument("--dspy-mipro-minibatch-size", type=int, default=8)
    parser.add_argument("--dspy-mipro-minibatch-full-eval-steps", type=int, default=3)
    parser.add_argument(
        "--dspy-max-train-records",
        type=int,
        default=0,
        help=(
            "Optional deterministic cap on DSPy training records after filtering. "
            "0 means use all records."
        ),
    )
    parser.add_argument("--dspy-model", default="openai/nvidia/Gemma-4-31B-IT-NVFP4")
    parser.add_argument("--dspy-api-base", default="http://localhost:8010/v1")
    parser.add_argument("--dspy-api-key", default="EMPTY")
    parser.add_argument(
        "--dspy-lm-transport",
        choices=["batch", "litellm"],
        default="batch",
        help=(
            "LM transport for DSPy optimizer calls. 'batch' uses the repo's "
            "central BatchedDSPyLM/AsyncBatchLLMClient path; 'litellm' uses "
            "plain dspy.LM."
        ),
    )
    parser.add_argument("--dspy-batch-max-concurrent", type=int, default=DEFAULT_BATCH_MAX_CONCURRENT)
    parser.add_argument("--dspy-batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--dspy-batch-timeout", type=float, default=DEFAULT_BATCH_TIMEOUT_SECONDS)
    parser.add_argument(
        "--dspy-batch-request-timeout",
        type=float,
        default=DEFAULT_BATCH_REQUEST_TIMEOUT_SECONDS,
    )
    parser.add_argument("--dspy-batch-await-response-timeout", type=float, default=None)
    parser.add_argument(
        "--dspy-batch-routing-policy",
        default=DEFAULT_BATCH_ROUTING_POLICY,
        help="Routing policy for multi-endpoint DSPy batch transport.",
    )
    parser.add_argument(
        "--dspy-f-init-path",
        default=None,
        help=(
            "Optional initial f artifact. For --dimension combined, defaults "
            "inside the joint family to outputs/phase2/joint_gepa/optimized_program.json; "
            "pass an empty string to force a bare scorer."
        ),
    )
    parser.add_argument(
        "--dspy-f-init-mode",
        choices=["pretuned_scorer", "bare_scorer", "teacher_passthrough"],
        default="pretuned_scorer",
        help=(
            "Explicit DSPy f initialization. pretuned_scorer is the old behavior "
            "that was previously mislabeled as identity."
        ),
    )
    parser.add_argument(
        "--dspy-g-init-mode",
        choices=["raw_concat", "teacher_passthrough"],
        default="raw_concat",
        help=(
            "Explicit DSPy g initialization. raw_concat is the canonical "
            "TreeBundle default; teacher_passthrough is compatibility mode for "
            "cached external/root summaries."
        ),
    )
    parser.add_argument(
        "--root-label-sources",
        default="",
        help=(
            "Comma-separated root label sources for DSPy f/g training. Empty disables "
            "root-label examples; use stored_summary, raw_document, or both."
        ),
    )
    parser.add_argument(
        "--root-label-target",
        choices=["expert", "teacher"],
        default="expert",
        help="Target for root-label examples. expert uses observed Benoit labels.",
    )
    parser.add_argument(
        "--local-law-weight",
        type=float,
        default=None,
        help=(
            "Canonical local-law objective mass λ for DSPy/LLM records. "
            "Defaults to 0.25 when full-doc anchors are enabled and 1.0 when anchors are off."
        ),
    )
    parser.add_argument(
        "--node-weight-normalization",
        choices=["per_tree", "none"],
        default="per_tree",
        help="Normalize teacher/local-law weights so their per-tree total equals λ.",
    )
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
    args = parser.parse_args(argv)
    args.used_deprecated_tree_bundle_alias = any(
        token in {"--fg-grid-dir", "--teacher-dir"} for token in raw_tokens
    )
    return args



def _audit_tree_bundle_input(args: argparse.Namespace) -> Dict[str, Any]:
    bundle_dir = Path(args.fg_grid_dir)
    manifest_path = bundle_dir / "manifest.json"
    if not manifest_path.exists():
        if bool(args.allow_legacy_tree_bundle):
            return {
                "status": "legacy_allowed_missing_manifest",
                "tree_bundle": str(bundle_dir),
                "allow_legacy_tree_bundle": True,
            }
        raise SystemExit(
            f"TreeBundle audit failed: {manifest_path} is missing; rerun the "
            "teacher bundle generator or pass --allow-legacy-tree-bundle for "
            "explicit compatibility only."
        )
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise SystemExit(
            f"TreeBundle audit failed: could not parse {manifest_path}: {exc}"
        ) from exc
    payload = manifest.get("config") if isinstance(manifest, Mapping) else None
    if not isinstance(payload, Mapping):
        payload = manifest if isinstance(manifest, Mapping) else {}
    normalized = normalize_tree_bundle_manifest(payload)
    has_v1 = (
        str(payload.get("schema_version") or "") == TREE_BUNDLE_SCHEMA_VERSION
        or isinstance(payload.get("tree_bundle_manifest"), Mapping)
    )
    if not has_v1 and not bool(args.allow_legacy_tree_bundle):
        raise SystemExit(
            f"TreeBundle audit failed: {manifest_path} lacks TreeBundle v1 schema; "
            "pass --allow-legacy-tree-bundle only for explicit compatibility."
        )
    source_kind = str(normalized.get("source_kind") or "")
    if source_kind == SOURCE_KIND_EXTERNAL_STATE and not bool(
        args.allow_external_state_tree_bundle
    ):
        raise SystemExit(
            "TreeBundle audit failed: source_kind=external_state requires "
            "--allow-external-state-tree-bundle."
        )
    expected_source = (
        None if bool(args.allow_external_state_tree_bundle) else SOURCE_KIND_RAW_INPUT
    )
    try:
        validate_tree_bundle_manifest(
            normalized,
            expected_domain="manifesto_rile",
            expected_leaf_unit=LEAF_UNIT_TEXT_TOKEN,
            expected_source_kind=expected_source,
        )
    except Exception as exc:
        raise SystemExit(f"TreeBundle audit failed: {exc}") from exc
    if str(args.dspy_g_init_mode) == "teacher_passthrough" and source_kind != SOURCE_KIND_EXTERNAL_STATE:
        raise SystemExit(
            "TreeBundle audit failed: dspy_g_init_mode=teacher_passthrough "
            "requires source_kind=external_state."
        )
    if source_kind == SOURCE_KIND_EXTERNAL_STATE and str(args.dspy_g_init_mode) != "teacher_passthrough":
        raise SystemExit(
            "TreeBundle audit failed: external_state bundles require "
            "--dspy-g-init-mode teacher_passthrough."
        )
    return {
        "status": "passed",
        "tree_bundle": str(bundle_dir),
        "manifest_path": str(manifest_path),
        "allow_legacy_tree_bundle": bool(args.allow_legacy_tree_bundle),
        "allow_external_state_tree_bundle": bool(args.allow_external_state_tree_bundle),
        "tree_bundle_manifest": normalized,
        "tree_bundle_manifest_digest": tree_bundle_manifest_digest(normalized),
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    families = _parse_families(args.families)
    if args.target_min is None or args.target_max is None:
        if str(args.dimension).strip().lower() == "environment":
            default_min, default_max = expert_scale_bounds(
                dimension="environment",
                scale=EXPERT_SCALE_NORMALIZED_1_7,
            )
        else:
            default_min, default_max = 1.0, 7.0
        if args.target_min is None:
            args.target_min = float(default_min)
        if args.target_max is None:
            args.target_max = float(default_max)
    LOGGER.info(
        "Using target bounds [%.3f, %.3f] and scorer-output bounds [%.3f, %.3f]",
        float(args.target_min),
        float(args.target_max),
        float(args.scorer_output_min),
        float(args.scorer_output_max),
    )
    if getattr(args, "used_deprecated_tree_bundle_alias", False):
        LOGGER.warning(
            "--fg-grid-dir/--teacher-dir are deprecated; use --tree-bundle for "
            "saved LabeledTree bundle inputs."
        )
    LOGGER.info("Using tree_bundle=%s", args.fg_grid_dir)

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
    if "dspy" in families:
        _preflight_dspy_budget(args, leaf_size_axis=leaf_size_axis)
        _preflight_dspy_f_warm_start(args, leaf_size_axis=leaf_size_axis)
    if bool(args.preflight_only):
        LOGGER.info("Preflight checks passed; exiting before tree bundle audit and row execution.")
        return 0
    tree_bundle_audit = _audit_tree_bundle_input(args)

    grid_rows: List[Dict[str, Any]] = []
    row_errors: List[str] = []

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
                row_errors.append(f"{family_name} {axis_kind}={axis_value}: {exc}")
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

    summary_table = summarize_ladder_grid(grid_rows, eval_split=args.eval_split)
    summary_json_path = output_dir / "grid_summary.json"
    schedule = schedule_from_max_iterations(
        int(args.max_iterations),
        first_train_side=str(args.first_train_side),
    )
    summary_payload: Dict[str, Any] = {
        "created_at": _now_stamp(),
        "dimension": args.dimension,
        "tree_bundle": str(args.fg_grid_dir),
        "tree_bundle_audit": tree_bundle_audit,
        "used_deprecated_tree_bundle_alias": bool(
            getattr(args, "used_deprecated_tree_bundle_alias", False)
        ),
        "families": list(families),
        "topology_axis": leaf_axis_name,
        "leaf_grid": list(leaf_count_axis) if leaf_count_axis is not None else None,
        "leaf_size_tokens": list(leaf_size_axis) if leaf_size_axis is not None else None,
        "max_iterations": int(args.max_iterations),
        "schedule": schedule,
        "first_train_side": str(args.first_train_side),
        "initial_f_degree": int(args.initial_f_degree),
        "initial_g_degree": int(args.initial_g_degree),
        "stage_naming": str(args.stage_naming),
        "dspy_f_init_mode": str(args.dspy_f_init_mode),
        "dspy_g_init_mode": str(args.dspy_g_init_mode),
        "objective": _objective_summary(args),
        "eval_split": args.eval_split,
        "rows": summary_table,
        "per_row_paths": [
            f"{r['family']}/{r.get('row_label')}/iteration_history.json"
            for r in grid_rows
        ],
    }
    tree_bundle_contract = tree_bundle_audit.get("tree_bundle_manifest")
    tree_bundle_metadata_payload = (
        dict(tree_bundle_contract.get("metadata") or {})
        if isinstance(tree_bundle_contract, Mapping)
        else {}
    )
    split_manifest_digest = str(
        tree_bundle_metadata_payload.get("split_manifest_digest") or ""
    )
    if split_manifest_digest:
        summary_payload["split_manifest_digest"] = split_manifest_digest
    run_manifest_kwargs: Dict[str, Any] = {}
    if isinstance(tree_bundle_contract, Mapping):
        run_manifest_kwargs["tree_bundle"] = tree_bundle_contract
    summary_payload["run_manifest"] = run_manifest_metadata(
        run_id=f"manifesto.alternating_ladder.{args.dimension}",
        domain="manifesto_rile",
        role="fg_ladder_runner",
        backend="mixed" if len(families) > 1 else str(families[0]),
        status="partial" if row_errors else "completed",
        f_init="family_default",
        g_init=str(args.dspy_g_init_mode),
        f_lineage={
            "families": list(families),
            "dspy_f_init_mode": str(args.dspy_f_init_mode),
            "initial_f_degree": int(args.initial_f_degree),
        },
        g_lineage={
            "families": list(families),
            "dspy_g_init_mode": str(args.dspy_g_init_mode),
            "initial_g_degree": int(args.initial_g_degree),
        },
        schedule=schedule,
        objective=objective_metadata(
            objective_family="manifesto_alternating_ladder",
            local_law_estimator=LOCAL_LAW_ESTIMATOR_PROXY_ONLY,
            local_law_weight=float(_effective_local_law_weight(args)),
            root_share=float(_effective_root_anchor_weight(args)),
            local_law_component_weights=_canonical_teacher_node_component_weights(
                float(_effective_local_law_weight(args))
            ),
            metadata=summary_payload["objective"],
        ),
        optimizer_config={
            "max_iterations": int(args.max_iterations),
            "first_train_side": str(args.first_train_side),
            "stage_naming": str(args.stage_naming),
            "objective": summary_payload["objective"],
        },
        output_artifacts=[
            {"kind": "grid_summary", "uri": str(summary_json_path)},
            {"kind": "grid_summary_markdown", "uri": str(output_dir / "grid_summary.md")},
            {"kind": "ladder_output_directory", "uri": str(output_dir)},
        ],
        audit_results={
            "ok": not row_errors and str(tree_bundle_audit.get("status") or "") in {"passed", "legacy_allowed_missing_manifest"},
            "tree_bundle_audit": tree_bundle_audit,
            "row_errors": list(row_errors),
        },
        quarantine={
            "classification": (
                "legacy_migratable"
                if bool(args.allow_legacy_tree_bundle)
                else "valid_treebundle_v1"
            )
        },
        command=sys.argv,
        allow_legacy=bool(args.allow_legacy_tree_bundle),
        publication_ready=(
            not row_errors
            and not bool(args.allow_legacy_tree_bundle)
            and str((tree_bundle_contract or {}).get("source_kind") or "") == SOURCE_KIND_RAW_INPUT
        ),
        metadata={
            "runner": "scripts/run_alternating_ladder.py",
            "allow_external_state_tree_bundle": bool(args.allow_external_state_tree_bundle),
            "split_manifest_digest": split_manifest_digest,
        },
        **run_manifest_kwargs,
    )
    summary_json_path.write_text(
        json.dumps(summary_payload, indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )
    md_path = output_dir / "grid_summary.md"
    write_alternating_markdown_summary(summary_table, md_path, eval_split=args.eval_split)
    LOGGER.info("Wrote %s and %s", summary_json_path, md_path)
    if row_errors and bool(args.fail_on_row_error):
        LOGGER.error("Failing because %d grid row(s) errored: %s", len(row_errors), row_errors)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
