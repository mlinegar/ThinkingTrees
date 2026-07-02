#!/usr/bin/env python3
"""Run the single-quasi-sentence Manifesto f/g ladder (DSPy or FNO family).

Input is the grid emitted by
``scripts/build_manifesto_qsentence_dspy_labeled_grid.py``.  Each row uses the
same alternating trampoline as the existing f/g ladder, but the axis is
``leaf_qsentences`` and the labels are exact CMP annotation aggregates.

``--family`` selects the substrate over the SAME labeled-tree bundles and
splits, so runs are directly comparable:

- ``dspy`` (default): LLM f/g via the batched DSPy transport. Point
  ``--dspy-model``/``--dspy-api-base`` at any OpenAI-compatible server —
  Gemma-4 on :8010 or the DiffusionGemma fleet on :8004-:8007 (comma-join the
  bases; remember ``TT_DSPY_DROP_RESPONSE_FORMAT=1`` for DiffusionGemma).
  Targets the full compact dimension vector (rile + domain_1..7).
- ``fno``: embeddings + neural-operator f/g (``--embedding-backend
  hashing|local-hf|vllm``). Trains on the scalar ``node.score`` (= normalized
  RILE in this grid); per-dimension vector heads are future work.

Compare runs with ``scripts/compare_manifesto_qsentence_substrates.py``.
"""

from __future__ import annotations

import argparse
import collections
import importlib.metadata as _importlib_metadata
import json
import logging
import random
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple


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

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.core.batch_transport import (  # noqa: E402
    DEFAULT_BATCH_MAX_CONCURRENT,
    DEFAULT_BATCH_REQUEST_TIMEOUT_SECONDS,
    DEFAULT_BATCH_ROUTING_POLICY,
    DEFAULT_BATCH_SIZE,
    DEFAULT_BATCH_TIMEOUT_SECONDS,
)
from src.ctreepo.alternating import IterationRecord, run_alternating_family  # noqa: E402
from src.ctreepo.fg_arity import auto_g_output_tokens, two_child_lm_budget_report  # noqa: E402
from src.ctreepo.manifesto_qsentence_runner import (  # noqa: E402
    leafq_label,
    load_leafq_trees,
    resolve_leaf_artifact,
    retarget_trees_to_dimension,
)
from src.ctreepo.manifesto_qsentence_dspy_family import (  # noqa: E402
    ManifestoQSentenceDSPyFamily,
    ManifestoQSentenceDSPyFamilyConfig,
)
from src.ctreepo.treepo_bridge.manifesto_finetune import (  # noqa: E402
    add_manifesto_finetune_args,
    export_manifesto_finetune_bundle_from_args,
    finetune_export_config,
)
from src.experiments.embedding_clients import (  # noqa: E402
    HashingEmbeddingClient,
    LocalHFEmbeddingClient,
)
from src.experiments.ladder_reporting import (  # noqa: E402
    summarize_ladder_grid,
    write_qsentence_markdown_summary,
)
from src.experiments.script_parse import parse_int_grid  # noqa: E402
from src.experiments.tree_helpers import split_trees_for_eval  # noqa: E402
from src.tasks.manifesto.script_utils import (  # noqa: E402
    now_stamp as _now_stamp,
    parse_compact_dimensions as _parse_dimensions,
)
from src.tree.labeled import LabeledTree  # noqa: E402


LOGGER = logging.getLogger(__name__)
DEFAULT_GRID_DIR = PROJECT_ROOT / "outputs" / "manifesto_qsentence_dspy_labeled_grid"


# --------------------------------------------------------------------------- #
# Supervision axis
# --------------------------------------------------------------------------- #
# "Level of supervision" is a first-class comma-list axis, exactly parallel to
# ``--leaf-qsentences``. Each NAMED level maps to a concrete configuration of
# the EXISTING low-level knobs (the FNO f-loss ``--fno-{root,leaf,merge}-weight``
# supervision weights and the convex root/law split ``--fno-local-law-weight``).
# One command therefore sweeps the full leaf x supervision grid through the SAME
# ``run_alternating_family`` driver.
#
# The named set is derived from what the weight flags + the shell sweep
# (``scripts/supervision_mix_sweep.sh``) actually express: the sweep dials
# root-vs-node dominance and turns merge supervision on/off, so the levels are
# the canonical points on that root-vs-node / merge-content dial.
#
# ONLY the FNO substrate has these scalar f-loss weight knobs. For the DSPy
# family the analogous dial is the reward-weight mix, which is structurally
# different (reward shares, not loss weights); rather than silently reinterpret
# the levels, the DSPy path errors if a non-default ``--supervision`` set is
# requested (see ``_apply_supervision_level``). ``--supervision`` defaults to a
# single ``default`` level that applies NO overrides, so existing DSPy and FNO
# commands are unchanged.
#
# A level's ``overrides`` map to ``args`` attribute names. ``None`` means "leave
# whatever the user passed on the CLI untouched" (used by the ``default`` level
# so back-compat is exact). Only keys with non-None values overwrite ``args``.


@dataclass(frozen=True)
class SupervisionLevel:
    """One named supervision cell = a concrete set of low-level knob overrides."""

    name: str
    description: str
    # Maps ``args`` attribute -> override value. None => do not touch the flag.
    overrides: Mapping[str, Optional[float]]
    # Substrates this level is defined for. FNO-only levels error on ``dspy``.
    families: Tuple[str, ...] = ("fno",)


# Canonical supervision levels. Weight values mirror the semantics already
# documented on the ``--fno-*-weight`` help text and exercised by the shell
# sweep. ``default`` is the identity level (no overrides) that preserves the
# exact current behavior when ``--supervision`` is omitted.
SUPERVISION_LEVELS: Dict[str, SupervisionLevel] = {
    "default": SupervisionLevel(
        name="default",
        description=(
            "Identity level: applies NO overrides, so the CLI --fno-*-weight / "
            "--fno-local-law-weight values (or their defaults) pass through "
            "unchanged. This is the implicit cell when --supervision is omitted."
        ),
        overrides={
            "fno_root_weight": None,
            "fno_leaf_weight": None,
            "fno_merge_weight": None,
            "fno_local_law_weight": None,
        },
        families=("fno", "dspy"),
    ),
    "root": SupervisionLevel(
        name="root",
        description=(
            "Root-only supervision: fit the holistic doc label at the root and "
            "leave nodes unsupervised (leaf/merge weights 0). g is free to learn "
            "any aggregation driven purely by the root target."
        ),
        overrides={
            "fno_root_weight": 1.0,
            "fno_leaf_weight": 0.0,
            "fno_merge_weight": 0.0,
        },
    ),
    "leaf": SupervisionLevel(
        name="leaf",
        description=(
            "Leaf (node-level) supervision only: supervise every leaf against its "
            "local target; no root or intermediate-merge supervision."
        ),
        overrides={
            "fno_root_weight": 0.0,
            "fno_leaf_weight": 1.0,
            "fno_merge_weight": 0.0,
        },
    ),
    "node": SupervisionLevel(
        name="node",
        description=(
            "Full node-level supervision: supervise BOTH leaves and "
            "intermediate merges against their local targets, no root term. The "
            "densest local signal."
        ),
        overrides={
            "fno_root_weight": 0.0,
            "fno_leaf_weight": 1.0,
            "fno_merge_weight": 1.0,
        },
    ),
    "mix": SupervisionLevel(
        name="mix",
        description=(
            "Balanced root+node mix: root-dominant holistic supervision combined "
            "with node-level grounding (the working econ-style recipe: strong "
            "root, lighter leaf/merge)."
        ),
        overrides={
            "fno_root_weight": 3.0,
            "fno_leaf_weight": 1.0,
            "fno_merge_weight": 1.0,
        },
    ),
}

DEFAULT_SUPERVISION_LEVEL = "default"


def parse_supervision_grid(value: Any) -> Tuple[str, ...]:
    """Parse the ``--supervision`` comma-list into validated level names.

    Mirrors ``parse_int_grid`` for the leaf axis: strips/splits on commas,
    rejects unknown names, and always yields at least one cell (``default``
    when empty/None) so callers never special-case the no-supervision path.
    """
    from src.experiments.script_parse import parse_csv

    parsed = parse_csv(value) if value is not None else ()
    if not parsed:
        return (DEFAULT_SUPERVISION_LEVEL,)
    unknown = [name for name in parsed if name not in SUPERVISION_LEVELS]
    if unknown:
        raise ValueError(
            f"unknown --supervision level(s) {unknown!r}; "
            f"allowed: {sorted(SUPERVISION_LEVELS)}"
        )
    # Preserve order, drop duplicates.
    seen: Dict[str, None] = {}
    for name in parsed:
        seen.setdefault(name, None)
    return tuple(seen)


def _apply_supervision_level(
    args: argparse.Namespace, level_name: str
) -> argparse.Namespace:
    """Return a shallow copy of ``args`` with the level's knob overrides applied.

    The ``default`` level is a pure pass-through (no attributes touched), so the
    exact pre-existing behavior is preserved when ``--supervision`` is omitted.
    """
    level = SUPERVISION_LEVELS[level_name]
    family = str(args.family)
    if family not in level.families:
        raise SystemExit(
            f"--supervision level {level_name!r} is only defined for families "
            f"{list(level.families)}, not {family!r}. The FNO f-loss weight "
            "knobs it configures have no direct DSPy analog; use --supervision "
            "default (or omit it) with the --dspy-g-* reward weights instead."
        )
    import copy as _copy

    scoped = _copy.copy(args)
    applied: Dict[str, float] = {}
    for attr, value in level.overrides.items():
        if value is None:
            continue
        setattr(scoped, attr, float(value))
        applied[attr] = float(value)
    if applied:
        LOGGER.info(
            "Supervision level %r -> %s",
            level_name,
            ", ".join(f"{k}={v}" for k, v in sorted(applied.items())),
        )
    return scoped


def supervision_label(level_name: str) -> str:
    return f"sup_{level_name}"




def _record_to_dict(record: IterationRecord) -> Dict[str, Any]:
    return asdict(record)


def _preflight_dspy_budget(args: argparse.Namespace) -> None:
    max_tokens = auto_g_output_tokens(
        int(args.dspy_max_tokens),
        leaf_size_tokens=int(args.dspy_leaf_size_tokens_fallback),
    )
    report = two_child_lm_budget_report(
        family_name="manifesto_qsentence_dspy",
        leaf_size_tokens=int(args.dspy_leaf_size_tokens_fallback),
        lm_context_window_tokens=int(args.dspy_lm_context_tokens),
        max_completion_tokens=int(max_tokens),
        prompt_template_overhead_tokens=int(args.dspy_prompt_overhead_tokens),
    )
    if not report.ok:
        raise SystemExit(
            "DSPy budget preflight failed for q-sentence ladder: "
            f"{'; '.join(report.violations)}. Increase --dspy-lm-context-tokens "
            f"to at least {report.minimum_context_window_tokens}, reduce "
            "--dspy-leaf-size-tokens-fallback, or reduce --dspy-max-tokens."
        )


def _build_family(args: argparse.Namespace) -> ManifestoQSentenceDSPyFamily:
    max_tokens = auto_g_output_tokens(
        int(args.dspy_max_tokens),
        leaf_size_tokens=int(args.dspy_leaf_size_tokens_fallback),
    )
    lm_config: Dict[str, Any] = {}
    if args.dspy_model:
        lm_config["model"] = str(args.dspy_model)
    if args.dspy_api_base:
        lm_config["api_base"] = str(args.dspy_api_base)
    if args.dspy_api_key:
        lm_config["api_key"] = str(args.dspy_api_key)
    lm_config["max_tokens"] = int(max_tokens)
    return ManifestoQSentenceDSPyFamily(
        config=ManifestoQSentenceDSPyFamilyConfig(
            optimizer=str(args.dspy_optimizer),
            budget=str(args.dspy_budget),
            num_threads=int(args.dspy_num_threads),
            target_min=float(args.target_min),
            target_max=float(args.target_max),
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
            mipro_minibatch_full_eval_steps=int(args.dspy_mipro_minibatch_full_eval_steps),
            include_identity_targets=bool(args.include_identity_targets),
            record_sample_seed=int(args.dspy_record_sample_seed),
            max_train_records=(
                int(args.dspy_max_train_records)
                if int(args.dspy_max_train_records) > 0
                else None
            ),
            leaf_size_tokens=int(args.dspy_leaf_size_tokens_fallback),
            lm_context_window_tokens=int(args.dspy_lm_context_tokens),
            max_completion_tokens=int(max_tokens),
            prompt_template_overhead_tokens=int(args.dspy_prompt_overhead_tokens),
            tokenizer_model_path=str(args.tokenizer_model),
            target_dimensions=_parse_dimensions(str(args.target_dimensions)),
            f_init_path=str(args.dspy_f_init_path) if args.dspy_f_init_path is not None else "",
            g_direct_parse_reward_weight=float(args.dspy_g_direct_parse_reward_weight),
            g_f_proxy_reward_weight=float(args.dspy_g_f_proxy_reward_weight),
            fail_fast_on_invalid_g_state=bool(args.dspy_g_fail_fast_on_invalid_state),
            g_scheduled_sampling_rate=float(args.dspy_g_scheduled_sampling_rate),
            g_scheduled_sampling_rate_start=float(args.dspy_g_scheduled_sampling_rate_start),
            g_scheduled_sampling_ramp_per_iter=float(args.dspy_g_scheduled_sampling_ramp_per_iter),
            g_lopsidedness_weight_strength=float(args.dspy_g_lopsidedness_weight_strength),
            g_law_c1_reward_weight=float(args.dspy_g_law_c1_reward_weight),
            g_law_c3a_reward_weight=float(args.dspy_g_law_c3a_reward_weight),
            g_law_c3b_reward_weight=float(args.dspy_g_law_c3b_reward_weight),
        )
    )


def _build_fno_family(args: argparse.Namespace) -> Any:
    """Build an FNOFamily over the same q-sentence labeled trees.

    The FNO substrate trains on the scalar ``node.score`` (= normalized RILE
    in the q-sentence grid); the compact per-dimension vector remains a DSPy
    family feature until FNOFamily grows a vector output head.
    """
    import torch

    from src.ctreepo.fno_family import FNOFamily, FNOFamilyConfig

    backend = str(args.embedding_backend)
    if backend == "hashing":
        embedding_client: Any = HashingEmbeddingClient(dim=int(args.hashing_embedding_dim))
        embedding_max_length_tokens = None
    elif backend == "local-hf":
        embedding_client = LocalHFEmbeddingClient(
            model=str(args.embedding_model),
            batch_size=int(args.embedding_batch_size),
            max_length=int(args.embedding_max_length),
            device=str(args.embedding_device),
        )
        if getattr(args, "embedding_cache_dir", None):
            from src.ctreepo.embedding_cache import DiskCachedEmbeddingClient

            embedding_client = DiskCachedEmbeddingClient(
                embedding_client,
                cache_dir=str(args.embedding_cache_dir),
                model_id=str(args.embedding_model),
            )
            LOGGER.info("embedding disk cache: %s", args.embedding_cache_dir)
        embedding_max_length_tokens = int(args.embedding_max_length)
    elif backend == "vllm":
        from src.training.embedding_proxy import VLLMEmbeddingClient

        embedding_client = VLLMEmbeddingClient(
            api_base=str(args.embedding_api_base),
            model=str(args.embedding_model) if args.embedding_model else None,
            batch_size=int(args.embedding_batch_size),
        )
        embedding_max_length_tokens = int(args.embedding_max_length)
    else:
        raise ValueError(f"unknown embedding backend: {backend!r}")

    # Probe the actual embedding width instead of trusting a flag; this also
    # fails fast if the embedding server is down.
    probe = embedding_client.embed_texts(["embedding dimension probe"])
    effective_embedding_dim = int(len(probe[0]))
    LOGGER.info(
        "FNO family: embedding backend=%s dim=%d", backend, effective_embedding_dim
    )

    device = (
        None if str(args.fno_device) == "auto" else torch.device(str(args.fno_device))
    )
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
            seed=int(args.fno_seed),
            root_weight=float(args.fno_root_weight),
            leaf_weight=float(args.fno_leaf_weight),
            merge_weight=float(args.fno_merge_weight),
            leaf_pos_weight=float(args.fno_leaf_pos_weight),
            leaf_pos_neutral=float(args.fno_leaf_pos_neutral),
            merge_mode=str(args.fno_merge_mode),
            merge_gate_hidden_dim=int(args.fno_merge_gate_hidden_dim),
            extent_enabled=bool(args.fno_extent),
            extent_merge_init=str(args.fno_extent_merge_init),
            g_depth_lopsided_strength=float(args.fno_g_depth_lopsided_strength),
            g_null_space_weight=float(args.fno_g_null_space_weight),
            local_law_weight=float(args.fno_local_law_weight),
            g_a2_weight=float(args.fno_g_a2_weight),
            a2_mode=str(args.fno_a2_mode),
            a3_factorization_weight=float(args.fno_a3_factorization_weight),
            g_assoc_weight=float(args.fno_g_assoc_weight),
            gamma_depth=float(args.fno_gamma_depth),
            root_readout=str(args.fno_root_readout),
            root_readout_k=int(args.fno_root_readout_k),
            root_readout_attn_temp=float(args.fno_root_readout_attn_temp),
            leaf_size_tokens=int(args.dspy_leaf_size_tokens_fallback),
            embedding_max_length_tokens=embedding_max_length_tokens,
            tokenizer_model_path=str(args.tokenizer_model),
            effective_embedding_dim=effective_embedding_dim,
        ),
        embedding_client=embedding_client,
        device=device,
    )



def _run_row(
    *,
    args: argparse.Namespace,
    trees: Sequence[LabeledTree],
    leaf_qsentences: int,
    supervision: str,
    output_dir: Path,
) -> Dict[str, Any]:
    # Apply the named supervision level's low-level knob overrides to a scoped
    # copy of args; the leaf axis is untouched (both axes remain orthogonal).
    args = _apply_supervision_level(args, supervision)
    train_trees, eval_trees = split_trees_for_eval(
        trees,
        eval_split=str(args.eval_split),
        train_split=str(args.train_split),
    )
    if not eval_trees:
        eval_trees = list(trees)
    eval_cap = int(getattr(args, "max_eval_trees", 0) or 0)
    if eval_cap > 0 and len(eval_trees) > eval_cap:
        rng = random.Random(int(args.eval_sample_seed) + 1009 * int(leaf_qsentences))
        indexed_eval_trees = list(enumerate(eval_trees))
        sampled_indices = {idx for idx, _tree in rng.sample(indexed_eval_trees, eval_cap)}
        eval_trees = [tree for idx, tree in indexed_eval_trees if idx in sampled_indices]
        LOGGER.info(
            "Capped q-sentence eval trees for leafq%03d: %d -> %d (seed=%d)",
            int(leaf_qsentences),
            len(indexed_eval_trees),
            len(eval_trees),
            int(args.eval_sample_seed),
        )
    family_name = str(args.family)
    if family_name == "fno":
        family = _build_fno_family(args)
        f_init: Any = "identity"
        g_init: Any = "identity"
    else:
        family = _build_family(args)
        f_init = resolve_leaf_artifact(
            getattr(args, "dspy_initial_f_artifact", None),
            getattr(args, "dspy_initial_f_artifact_template", None),
            family.TEACHER_PASSTHROUGH,
            "f",
            int(leaf_qsentences),
        )
        g_init = resolve_leaf_artifact(
            getattr(args, "dspy_initial_g_artifact", None),
            getattr(args, "dspy_initial_g_artifact_template", None),
            family.TEACHER_PASSTHROUGH,
            "g",
            int(leaf_qsentences),
        )
    leaf_label = leafq_label(int(leaf_qsentences))
    # When the supervision axis is at its identity ``default`` level the row
    # path stays exactly ``<family>/leafqNNN`` (back-compat); any real
    # supervision level nests a ``sup_<level>`` segment so the full grid is
    # disambiguated on disk.
    if supervision == DEFAULT_SUPERVISION_LEVEL:
        row_label = leaf_label
        row_dir = Path(output_dir) / family_name / leaf_label
    else:
        row_label = f"{leaf_label}/{supervision_label(supervision)}"
        row_dir = Path(output_dir) / family_name / leaf_label / supervision_label(supervision)
    row_dir.mkdir(parents=True, exist_ok=True)
    records = run_alternating_family(
        family=family,
        f_init=f_init,
        g_init=g_init,
        traces=train_trees if train_trees else list(trees),
        eval_trees=eval_trees,
        max_iterations=int(args.max_iterations),
        axis_kind="leaf_qsentences",
        axis_value=int(leaf_qsentences),
        leaf_count=int(leaf_qsentences),
        leaf_size_tokens=None,
        first_train_side=str(args.first_train_side),
        initial_f_degree=int(args.initial_f_degree),
        initial_g_degree=int(args.initial_g_degree),
        stage_naming=str(args.stage_naming),
        output_dir=row_dir,
    )
    finetune_bundle = export_manifesto_finetune_bundle_from_args(
        args=args,
        trees=trees,
        output_dir=row_dir / "treepo_finetune",
        kind="qsentence",
        logger=LOGGER,
        log_label="qsentence",
    )
    payload = {
        "family": family_name,
        "axis_kind": "leaf_qsentences",
        "axis_value": int(leaf_qsentences),
        "leaf_count": int(leaf_qsentences),
        "leaf_size_tokens": None,
        "leaf_qsentences": int(leaf_qsentences),
        "supervision": str(supervision),
        "supervision_weights": {
            "fno_root_weight": float(args.fno_root_weight),
            "fno_leaf_weight": float(args.fno_leaf_weight),
            "fno_merge_weight": float(args.fno_merge_weight),
            "fno_local_law_weight": float(args.fno_local_law_weight),
        },
        "row_label": row_label,
        "max_iterations": int(args.max_iterations),
        "eval_split": str(args.eval_split),
        "train_split": str(args.train_split),
        "n_train_trees": len(train_trees),
        "n_eval_trees": len(eval_trees),
        "max_eval_trees": int(getattr(args, "max_eval_trees", 0) or 0),
        "eval_sample_seed": int(getattr(args, "eval_sample_seed", 0) or 0),
        "target_dimensions": (
            list(_parse_dimensions(str(args.target_dimensions)))
            if family_name == "dspy"
            else [str(args.fno_target_dimension)]
        ),
        "substrate_model": (
            str(args.dspy_model)
            if family_name == "dspy"
            else f"fno+{args.embedding_backend}"
        ),
        "dspy_initial_f_artifact": str(f_init) if family_name == "dspy" else None,
        "dspy_initial_g_artifact": str(g_init) if family_name == "dspy" else None,
        "iterations": [_record_to_dict(record) for record in records],
        "finetune": finetune_bundle,
    }
    path = row_dir / "iteration_history.json"
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    LOGGER.info("Wrote %s", path)
    return payload



def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fg-grid-dir", type=Path, default=DEFAULT_GRID_DIR)
    parser.add_argument("--leaf-qsentences", default="1,2,4,8,16")
    parser.add_argument(
        "--supervision",
        default=None,
        help=(
            "Comma-list of NAMED supervision levels swept together with "
            "--leaf-qsentences (full leaf x supervision grid through the same "
            "run_alternating_family driver). Levels: "
            + ", ".join(sorted(SUPERVISION_LEVELS))
            + ". Each maps to a concrete set of --fno-*-weight / "
            "--fno-local-law-weight overrides (FNO substrate). Omit (or pass "
            "'default') to apply NO overrides = exact current behavior with "
            "whatever --fno-*-weight flags you set."
        ),
    )
    parser.add_argument("--max-iterations", type=int, default=2)
    parser.add_argument("--first-train-side", choices=("f", "g"), default="f")
    parser.add_argument("--initial-f-degree", type=int, default=1)
    parser.add_argument("--initial-g-degree", type=int, default=1)
    parser.add_argument("--stage-naming", choices=("legacy", "powers"), default="legacy")
    parser.add_argument("--train-split", default="train")
    parser.add_argument("--eval-split", default="test")
    parser.add_argument(
        "--max-eval-trees",
        type=int,
        default=0,
        help=(
            "Cap eval documents per leaf row before recursive learned-g evaluation "
            "(<=0 = all eval docs). Useful for LLM comparison passes where full "
            "node-level eval can dominate runtime."
        ),
    )
    parser.add_argument(
        "--eval-sample-seed",
        type=int,
        default=20260621,
        help="Seed used when --max-eval-trees samples eval documents.",
    )
    parser.add_argument("--target-min", type=float, default=0.0)
    parser.add_argument("--target-max", type=float, default=1.0)
    parser.add_argument("--target-dimensions", default="all")
    parser.add_argument("--include-identity-targets", action="store_true")
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--fail-on-row-error", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=None)
    add_manifesto_finetune_args(
        parser,
        kind="qsentence",
        help_text="Write treepo PreferenceDataset/fine-tune adapter bundles per leaf row.",
    )
    parser.add_argument("--verbose", action="store_true")

    parser.add_argument(
        "--family",
        choices=("dspy", "fno"),
        default="dspy",
        help="Substrate family run over the same q-sentence labeled trees.",
    )

    parser.add_argument(
        "--embedding-backend",
        choices=("hashing", "local-hf", "vllm"),
        default="vllm",
        help="FNO-family embedding source (hashing = offline smoke).",
    )
    parser.add_argument("--embedding-model", default=None)
    parser.add_argument("--embedding-api-base", default="http://localhost:8003/v1")
    parser.add_argument("--embedding-batch-size", type=int, default=32)
    parser.add_argument("--embedding-max-length", type=int, default=2048)
    parser.add_argument("--embedding-device", default="cuda")
    parser.add_argument(
        "--embedding-cache-dir",
        default=None,
        help=(
            "Disk cache for per-text embeddings (local-hf only). Embeddings are "
            "identical across experiment arms, so caching collapses the dominant "
            "embeddinggemma pass to one run; later arms/dumps load from disk. Share "
            "ONE dir across all arms of an experiment (keyed internally by model+text)."
        ),
    )
    parser.add_argument("--hashing-embedding-dim", type=int, default=256)
    parser.add_argument("--fno-hidden-channels", type=int, default=32)
    parser.add_argument("--fno-n-modes", type=int, default=64)
    parser.add_argument("--fno-n-layers", type=int, default=2)
    parser.add_argument("--fno-head-hidden-dim", type=int, default=64)
    parser.add_argument(
        "--fno-target-dimension",
        default="rile",
        help=(
            "Compact dimension the FNO scalar head regresses (retargets "
            "node.score). Default 'rile' = current behavior. Use one of "
            "rile/domain_1..domain_7; run once per dimension for a per-dim grid."
        ),
    )
    parser.add_argument(
        "--fno-root-weight",
        type=float,
        default=1.0,
        help="Weight on ROOT (doc-level) supervision in the FNO f-loss. Raise to let "
        "g learn an aggregation driven by the holistic root target (e.g. ignore "
        "neutral chunks) while keeping node-level grounding. Mixes with leaf/merge.",
    )
    parser.add_argument(
        "--fno-leaf-weight",
        type=float,
        default=0.5,
        help="Weight on LEAF (node-level) supervision in the FNO f-loss.",
    )
    parser.add_argument(
        "--fno-merge-weight",
        type=float,
        default=0.5,
        help="Weight on intermediate-MERGE (node-level) supervision in the FNO f-loss.",
    )
    parser.add_argument(
        "--fno-leaf-pos-weight",
        type=float,
        default=1.0,
        help=(
            "Balanced-leaf-loss multiplier for informative (non-neutral) leaves "
            "in the FNO f-loss. >1.0 counteracts neutral-majority gradient "
            "dominance on sparse dims. 1.0 = unchanged."
        ),
    )
    parser.add_argument(
        "--fno-leaf-pos-neutral",
        type=float,
        default=0.0,
        help=(
            "Neutral leaf-target value for the balanced leaf loss. Leaves whose "
            "normalized target deviates from this by >threshold are upweighted by "
            "--fno-leaf-pos-weight. 0.0 = MPDS sparse-categorical (zeros are "
            "neutral); 0.5 = Benoit LLM-span leaves (0.5/4-of-7 is neutral)."
        ),
    )
    parser.add_argument(
        "--fno-merge-mode",
        type=str,
        default="mean",
        choices=["mean", "gated", "maxpool", "mlp"],
        help=(
            "g merge baseline. 'mean' (default) = 0.5*(l+r)+FNO residual; "
            "'gated' = convex per-dim router alpha*l+(1-alpha)*r+residual; "
            "'maxpool' = non-convex per-dim max(l,r)+residual; 'mlp' = free "
            "high-capacity learnable merge (concat->MLP, mean warm-start, FNO "
            "UNUSED, BREAKS channel invariant — experimental). mean/gated/maxpool "
            "keep merge(a,a)=a; mlp keeps it only at init."
        ),
    )
    parser.add_argument(
        "--fno-merge-gate-hidden-dim",
        type=int,
        default=64,
        help="Hidden width of the gated-merge gate MLP (only used with --fno-merge-mode gated).",
    )
    parser.add_argument(
        "--fno-extent",
        action="store_true",
        help=(
            "Enable the learned 'extent' latent: each node state carries an extra "
            "scalar coordinate the merge gate reads, so g can weight children by "
            "information density (mass) instead of being structurally mass-blind. "
            "REQUIRES --fno-merge-mode gated. The extent is a free latent (laws-only: "
            "nothing supervises it against the true text mass)."
        ),
    )
    parser.add_argument(
        "--fno-extent-merge-init",
        type=str,
        default="neutral",
        choices=["neutral", "additive"],
        help=(
            "Init of the extent-merge head (only with --fno-extent). 'additive' = "
            "parent extent warm-starts at m_l+m_r (mass-weighted prior basin); "
            "'neutral' = parent extent starts at 0 (equal-averaging collapse basin). "
            "The A/B uses additive for the structured arm, neutral for pure-laws."
        ),
    )
    parser.add_argument(
        "--fno-g-depth-lopsided-strength",
        type=float,
        default=0.0,
        help=(
            "g-loss reweighting strength: multiply each merge node's loss by "
            "1+strength*(depth_norm*lopsidedness), concentrating the gradient on "
            "deep lopsided merges where mass-weighting beats equal-averaging and the "
            "extent latent is identifiable. 0.0 (default) = flat weights."
        ),
    )
    parser.add_argument(
        "--fno-g-null-space-weight",
        type=float,
        default=0.0,
        help=(
            "f-null-space salience law weight (no explicit merge weight). Penalizes "
            "low-impact children (those whose leave-one-out removal barely changes "
            "f(parent)) for carrying f-visible signal, pushing negligible content "
            "into f's null space so an additive/free merge ignores it. Trained in f "
            "(reshapes leaf geometry); pair with --fno-merge-mode mlp. 0.0 = off."
        ),
    )
    parser.add_argument(
        "--fno-local-law-weight",
        type=float,
        default=0.5,
        help=(
            "Lambda = the canonical ObjectiveSpec convex split (root_share = "
            "1 - Lambda) for BOTH f and g: (1-Lambda)*rootLoss + Lambda*lawLoss. "
            "rootLoss = directly fit the doc label at the root; lawLoss = the "
            "distributed local laws (f: leaf preservation A1; g: merge preservation "
            "A2 = merge route vs independent parent-text read / gold, AIPW). "
            "Lambda=0 -> root-only (the reference baseline); Lambda=1 -> pure law. "
            "Sweep with --fno-gamma-depth."
        ),
    )
    parser.add_argument(
        "--fno-g-a2-weight",
        type=float,
        default=1.0,
        help=(
            "DEPRECATED no-op: the g law is now governed by --fno-local-law-weight "
            "(the convex root/law split), not a standalone A2 weight. Accepted so "
            "existing scripts do not crash; has no effect on the objective."
        ),
    )
    parser.add_argument(
        "--fno-a2-mode",
        type=str,
        default="state",
        choices=["state", "readout"],
        help=(
            "Retained for back-compat; A2 is always the state merge-preservation law "
            "f(merge_state) vs f*(A.B). Use --fno-a3-factorization-weight for the "
            "readout-factorization projection (formerly a2_mode=readout)."
        ),
    )
    parser.add_argument(
        "--fno-a3-factorization-weight",
        type=float,
        default=0.0,
        help=(
            "A3 readout-FACTORIZATION projection (SEPARATE law from A2): "
            "(f(merge_state) - M(f(l), f(r)))^2 with M the Aczel phi-form (assoc+comm "
            "by construction). Checks the merge factors through the scalar readout; "
            "does NOT reference the parent text. 0 = off."
        ),
    )
    parser.add_argument(
        "--fno-gamma-depth",
        type=float,
        default=1.0,
        help=(
            "Lean depth discount gamma^depth in the canonical local-law objective "
            "(root weight 1, each level x gamma). Applied to BOTH f and g. 1.0 = flat "
            "sum; <1 concentrates weight on shallow/root nodes (push the merge toward "
            "the holistic root)."
        ),
    )
    parser.add_argument(
        "--fno-g-assoc-weight",
        type=float,
        default=0.0,
        help=(
            "Associativity projection DIAGNOSTIC: |f(m(m(a,b),c)) - f(m(a,m(b,c)))| "
            "(the proven Lean merge_assoc). A separate diagnostic, never A2 evidence. "
            "0 = off."
        ),
    )
    parser.add_argument(
        "--fno-root-readout",
        type=str,
        default="mean_root",
        choices=["mean_root", "topk", "softmax"],
        help=(
            "How the DOC-level prediction is read off. 'mean_root' (default) = "
            "predict on the composed root state; 'topk' = mean of the k highest "
            "LEAF scores; 'softmax' = temperature softmax pool over leaf scores. "
            "Non-default modes target dims whose doc label tracks the MAX on-topic "
            "leaf (eu: top1-leaf r=0.79 ~= ceiling)."
        ),
    )
    parser.add_argument("--fno-root-readout-k", type=int, default=1,
                        help="k for --fno-root-readout topk (k=1 = the single max leaf).")
    parser.add_argument("--fno-root-readout-attn-temp", type=float, default=0.2,
                        help="Temperature for --fno-root-readout softmax (->0 ~= max, ->inf ~= mean).")
    parser.add_argument("--fno-epochs", type=int, default=8)
    parser.add_argument("--fno-batch-size", type=int, default=2)
    parser.add_argument("--fno-learning-rate", type=float, default=1e-3)
    parser.add_argument("--fno-weight-decay", type=float, default=1e-4)
    parser.add_argument("--fno-grad-clip-norm", type=float, default=1.0)
    parser.add_argument("--fno-seed", type=int, default=42)
    parser.add_argument(
        "--fno-device",
        default="cuda",
        help="torch device for FNO training ('auto' = cuda when available; "
        "default cpu so runs do not contend with vLLM servers on the GPUs).",
    )

    parser.add_argument("--dspy-optimizer", default="mipro")
    parser.add_argument("--dspy-budget", default="light")
    parser.add_argument("--dspy-num-threads", type=int, default=128)
    parser.add_argument("--dspy-mipro-num-candidates", type=int, default=None)
    parser.add_argument("--dspy-mipro-num-trials", type=int, default=None)
    parser.add_argument("--dspy-mipro-max-bootstrapped-demos", type=int, default=None)
    parser.add_argument("--dspy-mipro-max-labeled-demos", type=int, default=None)
    parser.add_argument("--dspy-mipro-minibatch-size", type=int, default=35)
    parser.add_argument("--dspy-mipro-minibatch-full-eval-steps", type=int, default=5)
    parser.add_argument("--dspy-model", default="openai/nvidia/Gemma-4-31B-IT-NVFP4")
    parser.add_argument("--dspy-api-base", default="http://localhost:8010/v1")
    parser.add_argument("--dspy-api-key", default="EMPTY")
    parser.add_argument("--dspy-lm-transport", choices=["batch", "litellm"], default="batch")
    parser.add_argument("--dspy-batch-max-concurrent", type=int, default=DEFAULT_BATCH_MAX_CONCURRENT)
    parser.add_argument("--dspy-batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--dspy-batch-timeout", type=float, default=DEFAULT_BATCH_TIMEOUT_SECONDS)
    parser.add_argument(
        "--dspy-batch-request-timeout",
        type=float,
        default=DEFAULT_BATCH_REQUEST_TIMEOUT_SECONDS,
    )
    parser.add_argument("--dspy-batch-await-response-timeout", type=float, default=None)
    parser.add_argument("--dspy-batch-routing-policy", default=DEFAULT_BATCH_ROUTING_POLICY)
    parser.add_argument("--dspy-f-init-path", default=None)
    parser.add_argument(
        "--dspy-initial-f-artifact",
        default=None,
        help=(
            "Initial f artifact/sentinel for DSPy rows. Use this to evaluate or "
            "continue training from a compiled f program instead of the default "
            "teacher-passthrough initializer."
        ),
    )
    parser.add_argument(
        "--dspy-initial-g-artifact",
        default=None,
        help=(
            "Initial g artifact/sentinel for DSPy rows. Use this to evaluate or "
            "continue training from a compiled g program instead of the default "
            "teacher-passthrough initializer."
        ),
    )
    parser.add_argument(
        "--dspy-initial-f-artifact-template",
        default=None,
        help=(
            "Per-leaf initial f artifact template. Supports {leafq}, {leaf}, "
            "{leaf_qsentences}, and {row_label} placeholders."
        ),
    )
    parser.add_argument(
        "--dspy-initial-g-artifact-template",
        default=None,
        help=(
            "Per-leaf initial g artifact template. Supports {leafq}, {leaf}, "
            "{leaf_qsentences}, and {row_label} placeholders."
        ),
    )
    parser.add_argument(
        "--dspy-max-train-records",
        type=int,
        default=0,
        help="Cap on filtered node records before DSPy examples are built "
        "(<=0 = use all). Bounds GEPA's auto budget on large doc splits.",
    )
    parser.add_argument(
        "--dspy-record-sample-seed",
        type=int,
        default=0,
        help=(
            "Seed for deterministic uniform sampling when --dspy-max-train-records "
            "caps q-sentence node records. The selected node keys are written "
            "next to each f/g artifact."
        ),
    )
    parser.add_argument(
        "--dspy-g-direct-parse-reward-weight",
        type=float,
        default=0.75,
        help=(
            "Weight for direct parseable compact-target preservation in DSPy g "
            "optimization. Normalized with --dspy-g-f-proxy-reward-weight."
        ),
    )
    parser.add_argument(
        "--dspy-g-f-proxy-reward-weight",
        type=float,
        default=0.25,
        help=(
            "Weight for the current f-program proxy reward in DSPy g optimization. "
            "Unparseable g outputs can earn at most this normalized share."
        ),
    )
    parser.add_argument(
        "--dspy-g-fail-fast-on-invalid-state",
        action="store_true",
        help=(
            "Abort learned-g inference on the first malformed compact state "
            "instead of retrying. Use for strict conference-quality runs."
        ),
    )
    parser.add_argument(
        "--dspy-g-scheduled-sampling-rate",
        type=float,
        default=0.0,
        help=(
            "Scheduled-sampling (DAgger) cap: max fraction of merge children fed "
            "g's OWN generated state during g-training (closes train/eval exposure "
            "bias). 0 = legacy gold-children. NOT an averaging constraint."
        ),
    )
    parser.add_argument(
        "--dspy-g-scheduled-sampling-rate-start",
        type=float,
        default=0.0,
        help="Initial scheduled-sampling rate at iteration 0 (ramps toward the cap).",
    )
    parser.add_argument(
        "--dspy-g-scheduled-sampling-ramp-per-iter",
        type=float,
        default=0.0,
        help="Linear increment of the scheduled-sampling rate per alternating iteration.",
    )
    parser.add_argument(
        "--dspy-g-lopsidedness-weight-strength",
        type=float,
        default=0.0,
        help=(
            "Weight each g node's reward by 1 + strength * sibling-mass lopsidedness "
            "so deep LOPSIDED merges (where mass-weighting beats equal-averaging) "
            "dominate the C2-calibration gradient. 0 = legacy unweighted."
        ),
    )
    parser.add_argument(
        "--dspy-g-law-c1-reward-weight",
        type=float,
        default=0.0,
        help="C1 sufficiency reward: g's leaf state reads (through f) like the raw span.",
    )
    parser.add_argument(
        "--dspy-g-law-c3a-reward-weight",
        type=float,
        default=0.0,
        help="C3a joint-faithfulness reward: g's merge reads like the child concat.",
    )
    parser.add_argument(
        "--dspy-g-law-c3b-reward-weight",
        type=float,
        default=0.0,
        help=(
            "C3b compositionality reward (extra g+f calls/record). Prefer "
            "--dspy-g-scheduled-sampling-rate for the exposure-bias law; 0 = off."
        ),
    )
    parser.add_argument("--dspy-max-tokens", type=int, default=0)
    parser.add_argument("--dspy-lm-context-tokens", type=int, default=12000)
    parser.add_argument("--dspy-prompt-overhead-tokens", type=int, default=1500)
    parser.add_argument("--dspy-leaf-size-tokens-fallback", type=int, default=512)
    parser.add_argument(
        "--tokenizer-model",
        default="/mnt/data/models/google/embeddinggemma-300m",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    leaf_grid = parse_int_grid(args.leaf_qsentences, name="leaf-qsentences grid")
    supervision_grid = parse_supervision_grid(args.supervision)
    # Validate every supervision level against the chosen family up front (before
    # any preflight/tree loading) via a no-op scoped-args application. This is
    # what makes a --family dspy + non-default --supervision request fail fast.
    for sup in supervision_grid:
        _apply_supervision_level(args, sup)
    output_dir = args.output_dir or (
        PROJECT_ROOT
        / "outputs"
        / "manifesto_fg_alternating"
        / f"manifesto_qsentence_dspy_{_now_stamp()}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    if str(args.family) == "dspy":
        _preflight_dspy_budget(args)
        if bool(args.preflight_only):
            LOGGER.info("Preflight checks passed; exiting before loading q-sentence trees.")
            return 0
    elif bool(args.preflight_only):
        _build_fno_family(args)
        LOGGER.info(
            "FNO preflight passed (embedding backend reachable); exiting before "
            "loading q-sentence trees."
        )
        return 0

    LOGGER.info(
        "Running q-sentence DSPy ladder leaf_qsentences=%s supervision=%s "
        "max_iterations=%d -> %s",
        leaf_grid,
        supervision_grid,
        int(args.max_iterations),
        output_dir,
    )
    grid_rows: List[Dict[str, Any]] = []
    row_errors: List[str] = []
    # Full grid: outer leaf axis x inner supervision axis. The leaf trees are
    # loaded/retargeted once per leaf and reused across supervision cells.
    for leaf_q in leaf_grid:
        trees = load_leafq_trees(args.fg_grid_dir, int(leaf_q))
        if not trees:
            LOGGER.warning("No q-sentence labeled trees for leafq%03d under %s", leaf_q, args.fg_grid_dir)
            continue
        if str(args.family) == "fno" and str(args.fno_target_dimension) != "rile":
            n_set = retarget_trees_to_dimension(trees, str(args.fno_target_dimension))
            LOGGER.info(
                "Retargeted %d nodes to dimension %s for leafq%03d",
                n_set,
                args.fno_target_dimension,
                int(leaf_q),
            )
        for sup in supervision_grid:
            try:
                row = _run_row(
                    args=args,
                    trees=trees,
                    leaf_qsentences=int(leaf_q),
                    supervision=sup,
                    output_dir=output_dir,
                )
            except Exception as exc:
                row_errors.append(
                    f"leaf_qsentences={leaf_q} supervision={sup}: {type(exc).__name__}: {exc}"
                )
                LOGGER.exception("leaf_qsentences=%d supervision=%s failed", leaf_q, sup)
                leaf_label = leafq_label(int(leaf_q))
                err_row_label = (
                    leaf_label
                    if sup == DEFAULT_SUPERVISION_LEVEL
                    else f"{leaf_label}/{supervision_label(sup)}"
                )
                row = {
                    "family": str(args.family),
                    "axis_kind": "leaf_qsentences",
                    "axis_value": int(leaf_q),
                    "leaf_count": int(leaf_q),
                    "leaf_size_tokens": None,
                    "leaf_qsentences": int(leaf_q),
                    "supervision": str(sup),
                    "row_label": err_row_label,
                    "max_iterations": int(args.max_iterations),
                    "iterations": [],
                    "error": f"{type(exc).__name__}: {exc}",
                }
            for iteration in row.get("iterations", []) or []:
                extra = iteration.get("extra") or {}
                if extra.get("error"):
                    row_errors.append(
                        f"leaf_qsentences={leaf_q} supervision={sup} "
                        f"k={iteration.get('iteration')}: {extra['error']}"
                    )
            grid_rows.append(row)

    summary_table = summarize_ladder_grid(
        grid_rows,
        eval_split=str(args.eval_split),
        row_fields=(
            "family",
            "substrate_model",
            "axis_kind",
            "axis_value",
            "leaf_count",
            "leaf_size_tokens",
            "leaf_qsentences",
            "supervision",
        ),
        metric_fields=(
            "internal_f_pearson",
            "external_expert_pearson",
            "f_star_gap",
            "internal_f_mae_1_7",
            "external_expert_mae_1_7",
            "mean_prediction_1_7",
            "mean_teacher_1_7",
            "mean_expert_1_7",
        ),
    )
    summary_payload = {
        "created_at": _now_stamp(),
        "family": str(args.family),
        "topology_axis": "leaf_qsentences",
        "leaf_qsentences": list(leaf_grid),
        "supervision_axis": list(supervision_grid),
        "supervision_levels": {
            name: dict(SUPERVISION_LEVELS[name].overrides) for name in supervision_grid
        },
        "target_dimensions": (
            list(_parse_dimensions(str(args.target_dimensions)))
            if str(args.family) == "dspy"
            else [str(args.fno_target_dimension)]
        ),
        "max_iterations": int(args.max_iterations),
        "first_train_side": str(args.first_train_side),
        "initial_f_degree": int(args.initial_f_degree),
        "initial_g_degree": int(args.initial_g_degree),
        "stage_naming": str(args.stage_naming),
        "eval_split": str(args.eval_split),
        "train_split": str(args.train_split),
        "fg_grid_dir": str(args.fg_grid_dir),
        "max_eval_trees": int(args.max_eval_trees),
        "eval_sample_seed": int(args.eval_sample_seed),
        "dspy_record_sample_seed": (
            int(args.dspy_record_sample_seed) if str(args.family) == "dspy" else None
        ),
        "dspy_g_direct_parse_reward_weight": (
            float(args.dspy_g_direct_parse_reward_weight)
            if str(args.family) == "dspy"
            else None
        ),
        "dspy_g_f_proxy_reward_weight": (
            float(args.dspy_g_f_proxy_reward_weight)
            if str(args.family) == "dspy"
            else None
        ),
        "dspy_g_scheduled_sampling": (
            {
                "rate_cap": float(args.dspy_g_scheduled_sampling_rate),
                "rate_start": float(args.dspy_g_scheduled_sampling_rate_start),
                "ramp_per_iter": float(args.dspy_g_scheduled_sampling_ramp_per_iter),
            }
            if str(args.family) == "dspy"
            else None
        ),
        "finetune_export": finetune_export_config(args),
        "rows": summary_table,
        "per_row_paths": [
            f"{args.family}/{row.get('row_label')}/iteration_history.json"
            for row in grid_rows
        ],
        "finetune_paths": [
            f"{args.family}/{row.get('row_label')}/treepo_finetune"
            for row in grid_rows
            if row.get("finetune")
        ],
        "row_errors": row_errors,
    }
    (output_dir / "grid_summary.json").write_text(
        json.dumps(summary_payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    write_qsentence_markdown_summary(
        summary_table,
        output_dir / "grid_summary.md",
        eval_split=str(args.eval_split),
    )
    LOGGER.info("Wrote %s", output_dir / "grid_summary.json")
    if row_errors and bool(args.fail_on_row_error):
        LOGGER.error("Failing because %d row error(s) occurred: %s", len(row_errors), row_errors)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
