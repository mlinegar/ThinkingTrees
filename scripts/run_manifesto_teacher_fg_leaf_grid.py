#!/usr/bin/env python3
"""Run a teacher f/g pass on exact manifesto leaf-size/topology rows.

This script fills the missing node-level teacher side for embedding-FNO
student runs.  It reuses the same document split and leaf-size token axis, then
queries a teacher ``g`` summarizer/merger and a teacher ``f`` scalar scorer for
every realized tree node.  The output is canonical ``LabeledTree`` JSONL per
``leafTTTTtok`` row, with legacy count-based rows still available behind
``--leaf-grid``.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import logging
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.tasks.manifesto.openai_chat import (  # noqa: E402
    DEFAULT_MAIN_MODEL,
    OpenAIChatClient,
    http_error_detail as _http_error_detail,
)
from src.experiments.script_io import (  # noqa: E402
    JsonlCallCache,
    read_jsonl as _read_jsonl,
    require_within_chars as _require_within_chars,
    stable_hash as _stable_hash,
)
from src.tasks.manifesto.result_rows import (  # noqa: E402
    DIMENSION_BY_NAME as _DIM_FROM_NAME,
    get_text_for_row as _get_text_for_row,
    load_run_metadata as _load_run_metadata,
    order_split_rows as _order_split_rows,
    phase3_split_examples as _phase3_split_examples,
    row_expert_score as _row_expert_score,
    row_manifesto_id as _row_manifesto_id,
    row_teacher_score as _row_teacher_score,
)
from src.core.prompting import parse_numeric_score  # noqa: E402
from src.ctreepo.distillation import (  # noqa: E402
    _build_binary_node_specs,
    annotate_labeled_tree_summary_coverage,
    build_labeled_tree_from_text,
    write_labeled_trees_jsonl,
)
from src.ctreepo.fg_arity import auto_g_output_tokens  # noqa: E402
from src.ctreepo.treepo_bridge.manifesto_finetune import (  # noqa: E402
    add_manifesto_finetune_args,
    export_manifesto_finetune_bundle_from_args,
    finetune_export_config,
)
from src.ctreepo.contracts import (  # noqa: E402
    LEAF_UNIT_TEXT_TOKEN,
    LOCAL_LAW_ESTIMATOR_NONE,
    REDUCER_CONTRACT_BOTTOM_UP,
    SOURCE_KIND_EXTERNAL_STATE,
    SOURCE_KIND_RAW_INPUT,
    STATE_CONTRACT_EXTERNAL_PASSTHROUGH,
    STATE_CONTRACT_RAW_CONCAT,
    default_state_contract_for_source_kind,
    legacy_tree_bundle_kind_for_source_kind,
    legacy_tree_text_source_for_source_kind,
    source_kind_for_legacy_tree_text_source,
    source_kind_for_tree_bundle_kind,
    objective_metadata,
    run_manifest_metadata,
    tree_bundle_metadata,
)
from src.tasks.manifesto import ManifestoDataset  # noqa: E402
from src.tasks.manifesto.script_utils import (  # noqa: E402
    append_jsonl,
    now_iso,
    now_stamp,
    parse_int_grid as _parse_int_grid,
    write_json,
)
from src.tasks.manifesto.benoit_scoring_contexts import get_benoit_scoring_context  # noqa: E402
from src.tasks.manifesto.dimensions import get_dimension, get_preservation_rubric  # noqa: E402
from src.tasks.manifesto.expert_scale import (  # noqa: E402
    EXPERT_SCALE_CHOICES,
    EXPERT_SCALE_NORMALIZED_1_7,
    EXPERT_SCALE_RAW,
    expert_scale_bounds,
    expert_scale_metadata,
    raw_benoit_expert_from_row,
    resolve_benoit_expert_target,
    scorer_1_7_to_expert_target,
)
from src.tasks.manifesto.scoring_contexts import get_scoring_context  # noqa: E402
from src.training.config_sections import config_to_dict  # noqa: E402
from src.tree.labeled import LabeledNode, LabeledTree  # noqa: E402


LOGGER = logging.getLogger(__name__)

TREE_BUNDLE_KIND_RAW = "raw_manifesto_token_tree"
TREE_BUNDLE_KIND_EXTERNAL_SUMMARY = "external_summary_token_tree"
TREE_BUNDLE_KIND_CHOICES = (TREE_BUNDLE_KIND_RAW, TREE_BUNDLE_KIND_EXTERNAL_SUMMARY)
SOURCE_KIND_CHOICES = (SOURCE_KIND_RAW_INPUT, SOURCE_KIND_EXTERNAL_STATE)
STATE_CONTRACT_CHOICES = (STATE_CONTRACT_RAW_CONCAT, STATE_CONTRACT_EXTERNAL_PASSTHROUGH)



_now_stamp = now_stamp
_now_iso = now_iso


def _write_json(path: Path, payload: Mapping[str, Any]) -> Path:
    return write_json(path, config_to_dict(payload))


def _append_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> Path:
    return append_jsonl(path, (config_to_dict(row) for row in rows))



def _load_alignment_split_ids(alignment_run_dir: Optional[Path]) -> Optional[Dict[str, Dict[str, str]]]:
    if alignment_run_dir is None:
        return None
    split_path = Path(alignment_run_dir) / "split_ids.json"
    if not split_path.exists():
        return None
    payload = json.loads(split_path.read_text(encoding="utf-8"))
    out: Dict[str, Dict[str, str]] = {"train": {}, "val": {}, "test": {}}
    if isinstance(payload, Mapping):
        for split, values in payload.items():
            if isinstance(values, Mapping):
                out[str(split)] = {str(key): str(value or "") for key, value in values.items()}
            elif isinstance(values, list):
                out[str(split)] = {str(item): "" for item in values}
    return out


def _load_alignment_split_metadata(alignment_run_dir: Optional[Path]) -> Dict[str, Any]:
    if alignment_run_dir is None:
        return {}
    summary_path = Path(alignment_run_dir) / "coverage_split_summary.json"
    if not summary_path.exists():
        return {}
    try:
        payload = json.loads(summary_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    return payload if isinstance(payload, dict) else {}


def _load_alignment_leaf_grid(alignment_run_dir: Optional[Path]) -> Optional[Tuple[int, ...]]:
    if alignment_run_dir is None:
        return None
    manifest_path = Path(alignment_run_dir) / "manifest.json"
    if not manifest_path.exists():
        return None
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    config = payload.get("config") if isinstance(payload, Mapping) else None
    if not isinstance(config, Mapping) or "leaf_grid" not in config:
        return None
    return _parse_int_grid(config.get("leaf_grid"))


def _load_alignment_source_results(alignment_run_dir: Optional[Path]) -> Optional[Path]:
    if alignment_run_dir is None:
        return None
    manifest_path = Path(alignment_run_dir) / "manifest.json"
    if not manifest_path.exists():
        return None
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    runs = payload.get("runs") if isinstance(payload, Mapping) else None
    if not isinstance(runs, Mapping):
        return None
    for run in runs.values():
        if not isinstance(run, Mapping):
            continue
        contract = ((run.get("contract_result") or {}).get("contract") or {})
        metadata = contract.get("metadata") if isinstance(contract, Mapping) else None
        if isinstance(metadata, Mapping) and metadata.get("source_results"):
            return Path(str(metadata["source_results"]))
    return None


def _build_dimension_score_fn(
    *,
    client: OpenAIChatClient,
    dimension_name: str,
    temperature: float,
    max_tokens: int,
    max_chars: int,
    missing_policy: str,
    scoring_context_source: str,
):
    dim = _DIM_FROM_NAME[dimension_name]
    spec = get_dimension(dim)
    scoring_context = (
        get_benoit_scoring_context(dim)
        if scoring_context_source == "benoit"
        else get_scoring_context(dim)
    )

    def _score(text: str, *, role: str, node_context: Mapping[str, Any]) -> Tuple[float, Dict[str, Any]]:
        checked = _require_within_chars(
            str(text or ""),
            max_chars=int(max_chars),
            label=f"score input for {node_context.get('doc_id')}:{node_context.get('node_id')}",
        )
        user = (
            f"{scoring_context}\n\n"
            "You are scoring one C-TreePO tree node, not necessarily a full manifesto.\n"
            f"Node role: {role}\n"
            f"Document id: {node_context.get('doc_id')}\n"
            f"Node id: {node_context.get('node_id')}\n"
            "Return strict JSON with keys: score, reasoning.\n"
            "The score must be a number on the 1-7 scale, or null if the node is truly unscorable.\n\n"
            f"NODE_TEXT:\n{checked}"
        )
        response = client.chat(
            system=(
                "You are a strict expert political-science scalar scorer. "
                "Return only strict JSON. Do not include markdown."
            ),
            user=user,
            temperature=float(temperature),
            max_tokens=int(max_tokens),
        )
        parsed = parse_numeric_score(
            response,
            min_value=float(spec.scale.min_value),
            max_value=float(spec.scale.max_value),
            allow_llm_fallback=False,
        )
        if parsed is None:
            retry = client.chat(
                system="Return only one number from 1 to 7, or NA.",
                user=(
                    f"Extract the {dimension_name} 1-7 score from your prior answer.\n"
                    "Return only the numeric score.\n\n"
                    f"PRIOR_ANSWER:\n{response}"
                ),
                temperature=0.0,
                max_tokens=16,
            )
            parsed = parse_numeric_score(
                retry,
                min_value=float(spec.scale.min_value),
                max_value=float(spec.scale.max_value),
                allow_llm_fallback=False,
            )
            response_payload = {"response": response, "retry_response": retry}
        else:
            response_payload = {"response": response}
        if parsed is None:
            if missing_policy == "neutral":
                parsed = float(spec.scale.neutral_value or 4.0)
                response_payload["missing_policy_applied"] = "neutral"
            else:
                raise ValueError(f"Could not parse 1-7 score for {node_context}: {response!r}")
        return float(spec.scale.clamp(float(parsed))), response_payload

    return _score


def _build_dimension_summary_fn(
    *,
    client: OpenAIChatClient,
    dimension_name: str,
    temperature: float,
    max_tokens: int,
    max_chars: int,
    include_parent_span_reference: bool = False,
):
    dim = _DIM_FROM_NAME[dimension_name]
    rubric = get_preservation_rubric(dim)
    spec = get_dimension(dim)

    def _summarize(text: str, context: Mapping[str, Any]) -> str:
        if bool(context.get("is_leaf")):
            checked = _require_within_chars(
                str(text or ""),
                max_chars=int(max_chars),
                label=f"leaf summary input for {context.get('doc_id')}:{context.get('node_id')}",
            )
            user = (
                f"Dimension: {dimension_name}\n"
                f"Scale: 1 = {spec.anchor_low}; 7 = {spec.anchor_high}; 4 = neutral.\n\n"
                f"{rubric}\n\n"
                "Task: Produce the teacher g state for this C-TreePO leaf span.\n"
                "Preserve all evidence needed for later scalar scoring on the dimension.\n"
                "Keep concrete policy commitments, tax/spending direction, and caveats; "
                "do not collapse the span to only a title, slogan, party name, or administrative detail.\n"
                "Do not mention any numeric score. Return only the summary text.\n\n"
                f"LEAF_SPAN:\n{checked}"
            )
        else:
            left_summary = str(context.get("left_summary") or "").strip()
            right_summary = str(context.get("right_summary") or "").strip()
            parent_ref = ""
            if include_parent_span_reference:
                parent_ref = _require_within_chars(
                    str(text or ""),
                    max_chars=int(max_chars),
                    label=f"parent span reference for {context.get('doc_id')}:{context.get('node_id')}",
                )
            _require_within_chars(
                left_summary + "\n\n" + right_summary + ("\n\n" + parent_ref if parent_ref else ""),
                max_chars=int(max_chars),
                label=f"merge summary input for {context.get('doc_id')}:{context.get('node_id')}",
            )
            parent_block = (
                f"\n\nPARENT_SPAN_REFERENCE:\n{parent_ref}"
                if parent_ref
                else ""
            )
            user = (
                f"Dimension: {dimension_name}\n"
                f"Scale: 1 = {spec.anchor_low}; 7 = {spec.anchor_high}; 4 = neutral.\n\n"
                f"{rubric}\n\n"
                "Task: Merge two child teacher g states into the parent C-TreePO state.\n"
                "Preserve all dimension-relevant commitments, caveats, and evidence. "
                "Do not add claims not supported by the children or reference span. "
                "Keep concrete policy commitments, tax/spending direction, and caveats; "
                "do not collapse the parent to only a title, slogan, party name, or administrative detail. "
                "Do not mention any numeric score. Return only the merged summary text.\n\n"
                f"LEFT_CHILD_SUMMARY:\n{left_summary}\n\n"
                f"RIGHT_CHILD_SUMMARY:\n{right_summary}"
                f"{parent_block}"
            )
        return client.chat(
            system=(
                "You are the teacher g model for tree-indexed policy summarization. "
                "Outputs must be faithful, compact, and score-preserving."
            ),
            user=user,
            temperature=float(temperature),
            max_tokens=int(max_tokens),
        ).strip()

    return _summarize


def _build_dimension_resummary_fn(
    *,
    client: OpenAIChatClient,
    dimension_name: str,
    temperature: float,
    max_tokens: int,
    max_chars: int,
):
    dim = _DIM_FROM_NAME[dimension_name]
    rubric = get_preservation_rubric(dim)

    def _resummarize(summary: str, context: Mapping[str, Any]) -> str:
        checked = _require_within_chars(
            str(summary or ""),
            max_chars=int(max_chars),
            label=f"resummary input for {context.get('doc_id')}:{context.get('node_id')}",
        )
        return client.chat(
            system=(
                "You are the teacher g idempotence model for tree-indexed policy summaries. "
                "Return only summary text."
            ),
            user=(
                f"Dimension: {dimension_name}\n\n"
                f"{rubric}\n\n"
                "Resummarize this existing node summary while preserving the same "
                "dimension-relevant information. Do not introduce new claims. "
                "Do not mention any numeric score.\n\n"
                f"Node id: {context.get('node_id')}\n"
                f"SUMMARY:\n{checked}"
            ),
            temperature=float(temperature),
            max_tokens=int(max_tokens),
        ).strip()

    return _resummarize


def _tree_root_node(tree: LabeledTree) -> Optional[LabeledNode]:
    if not tree.levels:
        return None
    for node_id in reversed(tree.levels[-1]):
        node = tree.get_node(str(node_id))
        if node is not None:
            return node
    return None


def _node_summary(node: LabeledNode) -> str:
    metadata = node.metadata if isinstance(node.metadata, Mapping) else {}
    return str(metadata.get("teacher_summary") or metadata.get("target_summary") or "").strip()


def _build_teacher_labeled_tree(
    *,
    row: Mapping[str, Any],
    text: str,
    split: str,
    dimension_name: str,
    leaf_count: int,
    source_results: Path,
    source_report: Optional[Path],
    teacher_client: OpenAIChatClient,
    scorer_client: OpenAIChatClient,
    summary_cache: JsonlCallCache,
    score_cache: JsonlCallCache,
    resummary_cache: JsonlCallCache,
    args: argparse.Namespace,
    leaf_size_tokens: Optional[int] = None,
) -> LabeledTree:
    """Build a teacher-labeled tree.

    If ``leaf_size_tokens`` is set, the document is chunked into leaves of
    exactly that many EmbeddingGemma tokens (last leaf may be shorter). Char
    spans are derived from the tokenizer's ``offset_mapping`` and passed to
    ``build_labeled_tree_from_text`` via ``explicit_char_windows``. The
    ``leaf_count`` argument is then ignored as the size axis is the
    authoritative knob.

    Otherwise (legacy path), ``leaf_count`` continues to drive
    ``target_leaves_per_doc``.

    The cache-key discriminator switches between ``count_{N}`` and
    ``tok_{T}`` so size-based and count-based runs can coexist on disk.
    """
    doc_id = _row_manifesto_id(row)
    if not doc_id:
        raise ValueError("row has no manifesto_id/doc_id")
    summary_max_tokens = (
        auto_g_output_tokens(
            int(args.summary_max_tokens),
            leaf_size_tokens=int(leaf_size_tokens),
        )
        if leaf_size_tokens is not None
        else (int(args.summary_max_tokens) if int(args.summary_max_tokens) > 0 else 700)
    )
    resummary_max_tokens = (
        auto_g_output_tokens(
            int(args.resummary_max_tokens),
            leaf_size_tokens=int(leaf_size_tokens),
        )
        if leaf_size_tokens is not None
        else (
            int(args.resummary_max_tokens)
            if int(args.resummary_max_tokens) > 0
            else 500
        )
    )
    teacher_summary_fn = (
        _build_dimension_summary_fn(
            client=teacher_client,
            dimension_name=dimension_name,
            temperature=float(args.summary_temperature),
            max_tokens=int(summary_max_tokens),
            max_chars=int(args.node_summary_max_chars),
            include_parent_span_reference=bool(args.include_parent_span_reference),
        )
        if str(args.summary_mode) == "teacher"
        else None
    )
    teacher_score_fn = _build_dimension_score_fn(
        client=scorer_client,
        dimension_name=dimension_name,
        temperature=float(args.score_temperature),
        max_tokens=int(args.score_max_tokens),
        max_chars=int(args.score_max_chars),
        missing_policy=str(args.missing_score_policy),
        scoring_context_source=str(args.scoring_context_source),
    )
    teacher_resummary_fn = _build_dimension_resummary_fn(
        client=teacher_client,
        dimension_name=dimension_name,
        temperature=float(args.resummary_temperature),
        max_tokens=int(resummary_max_tokens),
        max_chars=int(args.resummary_max_chars),
    )
    teacher_score = _row_teacher_score(row, dimension=dimension_name)
    expert_target_scale = str(getattr(args, "expert_target_scale", EXPERT_SCALE_NORMALIZED_1_7))
    expert_score = _row_expert_score(
        row,
        dimension=dimension_name,
        expert_scale=expert_target_scale,
    )
    expert_raw = raw_benoit_expert_from_row(row, dimension=dimension_name)
    expert_1_7 = resolve_benoit_expert_target(
        row,
        dimension=dimension_name,
        scale=EXPERT_SCALE_NORMALIZED_1_7,
    )
    expert_native = expert_raw
    teacher_native = scorer_1_7_to_expert_target(
        teacher_score,
        dimension=dimension_name,
        scale=expert_target_scale,
    )
    label_source = str(args.label_source)

    # Cache-key discriminator: count_{N} for legacy, tok_{T} for size-based.
    axis_tag = (
        f"tok_{int(leaf_size_tokens)}" if leaf_size_tokens is not None
        else f"count_{int(leaf_count)}"
    )
    # Pre-compute exact char windows when running in size-based mode.
    explicit_windows: Optional[List[Tuple[int, int]]] = None
    if leaf_size_tokens is not None:
        from src.preprocessing.leaf_size_utils import char_windows_from_token_budget
        explicit_windows = char_windows_from_token_budget(
            str(text), int(leaf_size_tokens)
        )

    def cached_summary(span: str, context: Mapping[str, Any]) -> str:
        if str(args.summary_mode) == "identity":
            return str(span or "").strip()
        if teacher_summary_fn is None:
            return ""
        node_id = str(context.get("node_id") or "")
        payload_hash = _stable_hash(
            json.dumps(
                {
                    "span": span,
                    "left_summary": context.get("left_summary"),
                    "right_summary": context.get("right_summary"),
                    "is_leaf": bool(context.get("is_leaf")),
                    "dimension": dimension_name,
                    "teacher_model": args.teacher_model,
                    "summary_max_tokens": int(summary_max_tokens),
                },
                sort_keys=True,
            )
        )
        key = f"summary:v2:{dimension_name}:{axis_tag}:{doc_id}:{node_id}:{payload_hash}"
        cached = summary_cache.get(key)
        if cached is not None:
            return str(cached.get("summary") or "")
        summary = teacher_summary_fn(span, {**dict(context), "doc_id": doc_id})
        summary_cache.put(
            key,
            {
                "kind": "summary",
                "dimension": dimension_name,
                "doc_id": doc_id,
                "split": split,
                "leaf_count": None if leaf_size_tokens is not None else int(leaf_count),
                "leaf_size_tokens": int(leaf_size_tokens) if leaf_size_tokens is not None else None,
                "topology_axis": "size_tokens" if leaf_size_tokens is not None else "leaf_count",
                "node_id": node_id,
                "level": int(context.get("level") or 0),
                "is_leaf": bool(context.get("is_leaf")),
                "teacher_model": str(args.teacher_model),
                "input_hash": payload_hash,
                "summary": summary,
                "summary_max_tokens": int(summary_max_tokens),
            },
        )
        return summary

    # Level-parallel prewarm of the summary cache. Within each level the nodes
    # are independent; only merges depend on the child summaries at the level
    # below. We compute the tree topology, then batch-fire ``cached_summary``
    # calls via a thread pool, level by level. The subsequent
    # ``build_labeled_tree_from_text`` call is then mostly a cache-walking
    # no-op. This turns a doc with N nodes from N sequential LM calls (with
    # per-node HTTP latency) into log(N) barrier rounds, each round firing up
    # to ``lm_concurrency`` requests at once.
    lm_concurrency = max(1, int(getattr(args, "lm_concurrency", 32)))
    if str(args.summary_mode) in {"teacher", "identity"}:
        rendered = str(text or "")
        specs, _levels, _triples = _build_binary_node_specs(
            text_len=len(rendered),
            window_size=max(1, len(rendered)),
            window_overlap=0,
            target_leaves_per_doc=(
                None if leaf_size_tokens is not None else int(leaf_count)
            ),
            explicit_char_windows=explicit_windows,
        )
        by_id = {spec.node_id: spec for spec in specs}
        # Ordered lists: one list per level, asc by level.
        by_level: Dict[int, List[Any]] = {}
        for spec in specs:
            by_level.setdefault(int(spec.level), []).append(spec)
        summary_by_node_id: Dict[str, str] = {}
        for level in sorted(by_level):
            level_specs = by_level[level]
            spans_and_contexts: List[Tuple[str, Dict[str, Any], str]] = []
            for spec in level_specs:
                span = rendered[int(spec.char_start): int(spec.char_end)]
                left_sum = (
                    summary_by_node_id.get(str(spec.left_child_id))
                    if spec.left_child_id
                    else None
                )
                right_sum = (
                    summary_by_node_id.get(str(spec.right_child_id))
                    if spec.right_child_id
                    else None
                )
                context = {
                    "node_id": str(spec.node_id),
                    "level": int(spec.level),
                    "char_start": int(spec.char_start),
                    "char_end": int(spec.char_end),
                    "is_leaf": bool(spec.is_leaf),
                    "left_child_id": spec.left_child_id,
                    "right_child_id": spec.right_child_id,
                    "left_summary": left_sum,
                    "right_summary": right_sum,
                }
                spans_and_contexts.append((span, context, str(spec.node_id)))
            # Parallel-fire all summary calls at this level.
            with ThreadPoolExecutor(max_workers=lm_concurrency) as level_pool:
                futures = {
                    level_pool.submit(cached_summary, span, ctx): node_id
                    for span, ctx, node_id in spans_and_contexts
                }
                for future in as_completed(futures):
                    nid = futures[future]
                    summary_by_node_id[nid] = str(future.result() or "")

    tree = build_labeled_tree_from_text(
        doc_id=str(doc_id),
        text=str(text),
        document_score=float(teacher_score if teacher_score is not None else 4.0),
        split=str(split),
        score_fn=lambda _span: 4.0,
        window_size=max(1, len(str(text))),
        target_leaves_per_doc=(
            None if leaf_size_tokens is not None else int(leaf_count)
        ),
        explicit_char_windows=explicit_windows,
        label_source=label_source,
        node_summary_fn=cached_summary if str(args.summary_mode) in {"teacher", "identity"} else None,
        fill_missing_summaries_from_span=False,
        summary_source=(
            "teacher_fg_node_summary"
            if str(args.summary_mode) == "teacher"
            else ("span_identity_fallback" if str(args.summary_mode) == "identity" else "score_only_no_summary")
        ),
        extra_metadata={
            "dimension": str(dimension_name),
            "teacher_score_1_7_existing_root": teacher_score,
            "teacher_score_native": teacher_native,
            "expert_score_1_7": expert_1_7,
            "expert_score_native": expert_native,
            "expert_score_for_objective": expert_score,
            "expert_score_raw_benoit": expert_raw,
            **expert_scale_metadata(
                dimension=dimension_name,
                scale=expert_target_scale,
            ),
            "source_results_path": str(source_results),
            "source_report_path": str(source_report) if source_report else None,
            **tree_bundle_metadata(
                domain="manifesto_rile",
                leaf_unit=str(args.leaf_unit),
                source_kind=str(args.source_kind),
                dimension=str(dimension_name),
                target_scale=str(expert_target_scale),
                leaf_policy={
                    "topology_axis": "size_tokens"
                    if leaf_size_tokens is not None
                    else "leaf_count",
                    "leaf_count": None
                    if leaf_size_tokens is not None
                    else int(leaf_count),
                    "leaf_size_tokens": int(leaf_size_tokens)
                    if leaf_size_tokens is not None
                    else None,
                },
                state_contract=str(args.state_contract),
                reducer_contract=str(args.reducer_contract),
                external_state_producer=(
                    str(args.external_state_producer)
                    if args.external_state_producer
                    else None
                ),
                metadata={
                    "split_manifest_digest": str(getattr(args, "split_manifest_digest", "") or ""),
                    "split_alignment_run_dir": (
                        str(args.alignment_run_dir)
                        if getattr(args, "alignment_run_dir", None)
                        else None
                    ),
                    "split_schema_version": str(getattr(args, "split_schema_version", "") or ""),
                },
            ),
            "tree_state_source": str(args.tree_state_source),
            "teacher_fg_model": {
                "g_base_url": str(args.teacher_base_url),
                "g_model": str(args.teacher_model),
                "f_base_url": str(args.scorer_base_url or args.teacher_base_url),
                "f_model": str(args.scorer_model or args.teacher_model),
                "score_input": str(args.score_input),
                "scoring_context_source": str(args.scoring_context_source),
                "summary_mode": str(args.summary_mode),
                "idempotence_mode": str(args.idempotence_mode),
            },
            "node_score_projection": None,
            "node_score_source": "teacher_f_dimension_score_1_7",
            "topology_axis": (
                "size_tokens" if leaf_size_tokens is not None else "leaf_count"
            ),
            "leaf_count": None if leaf_size_tokens is not None else int(leaf_count),
            "leaf_size_tokens": (
                int(leaf_size_tokens) if leaf_size_tokens is not None else None
            ),
            "derived_leaf_count": (
                int(len(explicit_windows)) if explicit_windows is not None else int(leaf_count)
            ),
            "tokenizer_model": (
                "/mnt/data/models/google/embeddinggemma-300m"
                if leaf_size_tokens is not None
                else None
            ),
        },
    )

    idempotence_pairs: List[Dict[str, Any]] = []
    root_node = _tree_root_node(tree)

    # Batch-compute all per-node scores concurrently. Scores are independent
    # across nodes (unlike merge summaries which depend on child summaries),
    # so we can fire all of them in one barrier round. Cache hits skip the
    # LM call entirely; misses go into a shared thread pool.
    def _resolve_score(node: LabeledNode) -> Tuple[str, Dict[str, Any]]:
        summary = _node_summary(node)
        score_input = summary if str(args.score_input) == "teacher_summary" and summary else str(node.text or "")
        score_hash = _stable_hash(
            json.dumps(
                {
                    "score_input": score_input,
                    "score_input_kind": str(args.score_input),
                    "dimension": dimension_name,
                    "scorer_model": args.scorer_model or args.teacher_model,
                    "score_max_tokens": int(args.score_max_tokens),
                    "score_max_chars": int(args.score_max_chars),
                    "scoring_context_source": str(args.scoring_context_source),
                },
                sort_keys=True,
            )
        )
        score_key = f"score:v3:{dimension_name}:{axis_tag}:{doc_id}:{node.node_id}:{score_hash}"
        cached_score = score_cache.get(score_key)
        if cached_score is not None:
            return score_key, cached_score
        score_value, score_payload = teacher_score_fn(
            score_input,
            role="teacher_summary" if str(args.score_input) == "teacher_summary" else "node_span",
            node_context={
                "doc_id": doc_id,
                "node_id": node.node_id,
                "level": int(node.level),
                "is_leaf": int(node.level) == 0,
                "leaf_count": None if leaf_size_tokens is not None else int(leaf_count),
                "leaf_size_tokens": int(leaf_size_tokens) if leaf_size_tokens is not None else None,
                "topology_axis": "size_tokens" if leaf_size_tokens is not None else "leaf_count",
            },
        )
        written = score_cache.put(
            score_key,
            {
                "kind": "score",
                "dimension": dimension_name,
                "doc_id": doc_id,
                "split": split,
                "leaf_count": None if leaf_size_tokens is not None else int(leaf_count),
                "leaf_size_tokens": int(leaf_size_tokens) if leaf_size_tokens is not None else None,
                "topology_axis": "size_tokens" if leaf_size_tokens is not None else "leaf_count",
                "node_id": str(node.node_id),
                "level": int(node.level),
                "is_leaf": int(node.level) == 0,
                "scorer_model": str(args.scorer_model or args.teacher_model),
                "score_input_kind": str(args.score_input),
                "scoring_context_source": str(args.scoring_context_source),
                "input_hash": score_hash,
                "score_1_7": float(score_value),
                **score_payload,
            },
        )
        return score_key, written

    score_results: Dict[str, Tuple[str, Dict[str, Any]]] = {}
    node_list = list(tree.nodes.values())
    with ThreadPoolExecutor(max_workers=lm_concurrency) as score_pool:
        futures = {
            score_pool.submit(_resolve_score, node): str(node.node_id)
            for node in node_list
        }
        for future in as_completed(futures):
            nid = futures[future]
            score_results[nid] = future.result()

    for node in node_list:
        summary = _node_summary(node)
        score_key, cached_score = score_results[str(node.node_id)]
        node.score = float(cached_score.get("score_1_7"))
        node.dimension_scores = {dimension_name: float(node.score)}
        node.metadata["teacher_score_1_7"] = float(node.score)
        node.metadata["teacher_score_source"] = "teacher_f_dimension_score_1_7"
        node.metadata["teacher_score_input_kind"] = str(args.score_input)
        node.metadata["teacher_scoring_context_source"] = str(args.scoring_context_source)
        node.metadata["teacher_summary_mode"] = str(args.summary_mode)
        node.metadata["teacher_score_model"] = str(args.scorer_model or args.teacher_model)
        node.metadata["teacher_score_cache_key"] = score_key

        should_resummarize = (
            str(args.idempotence_mode) == "all"
            or (str(args.idempotence_mode) == "root" and root_node is not None and str(node.node_id) == str(root_node.node_id))
        )
        if should_resummarize and summary:
            resummary_hash = _stable_hash(
                json.dumps(
                    {
                        "summary": summary,
                        "dimension": dimension_name,
                        "teacher_model": args.teacher_model,
                        "resummary_max_tokens": int(resummary_max_tokens),
                    },
                    sort_keys=True,
                )
            )
            resummary_key = f"resummary:v2:{dimension_name}:{axis_tag}:{doc_id}:{node.node_id}:{resummary_hash}"
            cached_resummary = resummary_cache.get(resummary_key)
            if cached_resummary is None:
                resummary = teacher_resummary_fn(
                    summary,
                    {
                        "doc_id": doc_id,
                        "node_id": node.node_id,
                        "level": int(node.level),
                        "is_leaf": int(node.level) == 0,
                    },
                )
                cached_resummary = resummary_cache.put(
                    resummary_key,
                    {
                        "kind": "resummary",
                        "dimension": dimension_name,
                        "doc_id": doc_id,
                        "split": split,
                        "leaf_count": None if leaf_size_tokens is not None else int(leaf_count),
                        "leaf_size_tokens": int(leaf_size_tokens) if leaf_size_tokens is not None else None,
                        "topology_axis": "size_tokens" if leaf_size_tokens is not None else "leaf_count",
                        "node_id": str(node.node_id),
                        "level": int(node.level),
                        "is_leaf": int(node.level) == 0,
                        "teacher_model": str(args.teacher_model),
                        "input_hash": resummary_hash,
                        "resummary_max_tokens": int(resummary_max_tokens),
                        "resummary": resummary,
                    },
                )
            resummary_text = str(cached_resummary.get("resummary") or "").strip()
            if resummary_text:
                node.metadata["teacher_resummary"] = resummary_text
                node.metadata["teacher_resummary_source"] = "teacher_fg_idempotence"
                idempotence_pairs.append(
                    {
                        "node_id": str(node.node_id),
                        "input_summary": summary,
                        "target_resummary": resummary_text,
                        "source": "teacher_fg_idempotence",
                    }
                )

    root_node = _tree_root_node(tree)
    if root_node is not None:
        tree.document_score = float(root_node.score)
        tree.metadata["teacher_score_1_7"] = float(root_node.score)
    tree.metadata["idempotence_pairs"] = idempotence_pairs
    annotate_labeled_tree_summary_coverage(tree)
    return tree


def _flat_node_rows(
    tree: LabeledTree,
    *,
    leaf_count: Optional[int],
    leaf_size_tokens: Optional[int],
    dimension_name: str,
) -> List[Dict[str, Any]]:
    root = _tree_root_node(tree)
    root_id = str(root.node_id) if root is not None else ""
    topology_axis = "size_tokens" if leaf_size_tokens is not None else "leaf_count"
    rows: List[Dict[str, Any]] = []
    for level_ids in tree.levels:
        for node_id in level_ids:
            node = tree.get_node(str(node_id))
            if node is None:
                continue
            metadata = node.metadata if isinstance(node.metadata, Mapping) else {}
            rows.append(
                {
                    "doc_id": tree.doc_id,
                    "split": (tree.metadata or {}).get("split"),
                    "dimension": dimension_name,
                    "topology_axis": topology_axis,
                    "leaf_count": int(leaf_count) if leaf_count is not None else None,
                    "leaf_size_tokens": int(leaf_size_tokens) if leaf_size_tokens is not None else None,
                    "derived_leaf_count": len(tree.get_leaves()),
                    "node_id": str(node.node_id),
                    "level": int(node.level),
                    "is_leaf": int(node.level) == 0,
                    "is_root": str(node.node_id) == root_id,
                    "char_start": metadata.get("char_start"),
                    "char_end": metadata.get("char_end"),
                    "score_1_7": float(node.score),
                    "teacher_summary": metadata.get("teacher_summary"),
                    "teacher_resummary": metadata.get("teacher_resummary"),
                    "teacher_score_input_kind": metadata.get("teacher_score_input_kind"),
                    "left_child_id": node.left_child_id,
                    "right_child_id": node.right_child_id,
                }
            )
    return rows


def _resolve_split_ids(args: argparse.Namespace, rows_by_id: Mapping[str, Mapping[str, Any]]) -> Dict[str, Dict[str, str]]:
    aligned = _load_alignment_split_ids(args.alignment_run_dir)
    if aligned is not None:
        return aligned
    dim = _DIM_FROM_NAME[args.dimension]
    if args.split_source == "phase3":
        return _phase3_split_examples(
            dimension=dim,
            train_n=int(args.train_n),
            val_n=int(args.val_n),
            test_n=int(args.test_n),
            seed=int(args.seed),
            split_strategy=str(args.split_strategy),
            train_pool=str(args.train_pool),
            mp_data_dir=args.mp_data_dir,
        )
    return _order_split_rows(
        rows_by_id,
        train_n=int(args.train_n),
        val_n=int(args.val_n),
        test_n=int(args.test_n),
        seed=int(args.seed),
    )


def _limit_split_ids_to_requested_sizes(
    split_ids: Mapping[str, Mapping[str, str]],
    *,
    train_n: int,
    val_n: int,
    test_n: int,
) -> Dict[str, Dict[str, str]]:
    limits = {
        "train": int(train_n),
        "val": int(val_n),
        "test": int(test_n),
    }
    limited: Dict[str, Dict[str, str]] = {}
    for split, values in split_ids.items():
        split_name = str(split)
        items = list(values.items())
        limit = limits.get(split_name)
        if limit is not None and limit >= 0:
            items = items[:limit]
        limited[split_name] = dict(items)
    for split in ("train", "val", "test"):
        limited.setdefault(split, {})
    return limited


def _build_leaf_grid(args: argparse.Namespace) -> Tuple[int, ...]:
    if args.leaf_grid:
        return _parse_int_grid(args.leaf_grid)
    aligned = _load_alignment_leaf_grid(args.alignment_run_dir)
    if aligned is not None:
        return aligned
    raise ValueError("--leaf-grid is legacy-only and must be explicit in count-based mode")


def _process_doc(
    *,
    row: Mapping[str, Any],
    text: str,
    split: str,
    dimension_name: str,
    leaf_count: int,
    source_results: Path,
    source_report: Optional[Path],
    teacher_client: OpenAIChatClient,
    scorer_client: OpenAIChatClient,
    summary_cache: JsonlCallCache,
    score_cache: JsonlCallCache,
    resummary_cache: JsonlCallCache,
    args: argparse.Namespace,
    leaf_size_tokens: Optional[int] = None,
) -> Tuple[Optional[LabeledTree], Optional[Dict[str, Any]]]:
    try:
        tree = _build_teacher_labeled_tree(
            row=row,
            text=text,
            split=split,
            dimension_name=dimension_name,
            leaf_count=leaf_count,
            source_results=source_results,
            source_report=source_report,
            teacher_client=teacher_client,
            scorer_client=scorer_client,
            summary_cache=summary_cache,
            score_cache=score_cache,
            resummary_cache=resummary_cache,
            args=args,
            leaf_size_tokens=leaf_size_tokens,
        )
        return tree, None
    except Exception as exc:
        return None, {
            "doc_id": _row_manifesto_id(row),
            "split": split,
            "dimension": dimension_name,
            "topology_axis": "size_tokens" if leaf_size_tokens is not None else "leaf_count",
            "leaf_count": None if leaf_size_tokens is not None else int(leaf_count),
            "leaf_size_tokens": (
                int(leaf_size_tokens) if leaf_size_tokens is not None else None
            ),
            "error": _http_error_detail(exc),
        }


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--dimension", choices=sorted(_DIM_FROM_NAME), default="economic")
    parser.add_argument("--alignment-run-dir", type=Path, default=None)
    parser.add_argument("--source-results", type=Path, default=None)
    parser.add_argument("--source-report", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument(
        "--leaf-grid",
        default=None,
        help=(
            "LEGACY count-based axis (e.g. 1,2,4,8,16). Mutually exclusive with "
            "--leaf-size-tokens. Count-based runs are legacy-only; omit both "
            "flags to use the size-token default 512,1024,2048."
        ),
    )
    parser.add_argument(
        "--leaf-size-tokens",
        default=None,
        help=(
            "Size-based leaf axis (tokens per leaf), e.g. 512,1024,2048. Each "
            "leaf is exactly this many EmbeddingGemma tokens (last leaf may be "
            "shorter). Output dirs are leaf{TTT}tok/. Mutually exclusive with "
            "--leaf-grid."
        ),
    )
    parser.add_argument("--split-source", choices=["phase3", "results-order"], default="phase3")
    parser.add_argument("--train-pool", choices=["expert-split", "openweight", "expert"], default="expert-split")
    parser.add_argument(
        "--expert-target-scale",
        choices=EXPERT_SCALE_CHOICES,
        default=None,
        help=(
            "Scale used for root expert targets in labeled trees. Omit for "
            "normalized_1_7; pass raw_benoit only to reproduce older raw-scale metrics."
        ),
    )
    parser.add_argument("--split-strategy", choices=["random", "label-stratified"], default="label-stratified")
    parser.add_argument(
        "--tree-text-source",
        choices=["aligned_text", "existing_summary"],
        default=None,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--tree-bundle-kind",
        choices=TREE_BUNDLE_KIND_CHOICES,
        default=None,
        help=(
            "Deprecated compatibility alias for --source-kind. "
            "raw_manifesto_token_tree maps to source_kind=raw_input; "
            "external_summary_token_tree maps to source_kind=external_state."
        ),
    )
    parser.add_argument(
        "--source-kind",
        choices=SOURCE_KIND_CHOICES,
        default=None,
        help=(
            "Generic TreeBundle v1 source provenance. raw_input is the default; "
            "external_state is explicit compatibility mode for precomputed summaries."
        ),
    )
    parser.add_argument(
        "--leaf-unit",
        default=LEAF_UNIT_TEXT_TOKEN,
        help="Generic TreeBundle v1 leaf unit for this topology.",
    )
    parser.add_argument(
        "--state-contract",
        choices=STATE_CONTRACT_CHOICES,
        default=None,
        help=(
            "Initial state contract. Defaults to raw_concat for raw_input and "
            "external_passthrough for external_state."
        ),
    )
    parser.add_argument(
        "--reducer-contract",
        choices=[REDUCER_CONTRACT_BOTTOM_UP],
        default=REDUCER_CONTRACT_BOTTOM_UP,
        help="Tree reducer contract recorded in TreeBundle v1 metadata.",
    )
    parser.add_argument(
        "--tree-state-source",
        default=None,
        help=(
            "Human-readable state source stored in bundle metadata. Defaults to "
            "raw_input for raw bundles and external_state for external bundles."
        ),
    )
    parser.add_argument(
        "--external-state-producer",
        default=None,
        help=(
            "External g/state producer for source_kind=external_state bundles, "
            "for example g_benoit."
        ),
    )
    parser.add_argument("--train-n", type=int, default=80)
    parser.add_argument("--val-n", type=int, default=20)
    parser.add_argument("--test-n", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--mp-data-dir", type=Path, default=None)

    parser.add_argument("--teacher-base-url", type=str, default="http://localhost:8001/v1")
    parser.add_argument("--teacher-model", type=str, default=DEFAULT_MAIN_MODEL)
    parser.add_argument("--teacher-api-key", type=str, default="EMPTY")
    parser.add_argument("--teacher-timeout-seconds", type=float, default=180.0)
    parser.add_argument("--scorer-base-url", type=str, default=None)
    parser.add_argument("--scorer-model", type=str, default=None)
    parser.add_argument("--scorer-api-key", type=str, default="EMPTY")
    parser.add_argument("--scorer-timeout-seconds", type=float, default=180.0)
    parser.add_argument("--enable-thinking", action=argparse.BooleanOptionalAction, default=False)

    parser.add_argument("--summary-temperature", type=float, default=0.2)
    parser.add_argument(
        "--summary-max-tokens",
        type=int,
        default=0,
        help=(
            "Teacher g output cap. In size-token mode, 0 means auto = "
            "2 * leaf_size_tokens so g can emit a verbatim concatenation of "
            "two children. Legacy count mode falls back to 700."
        ),
    )
    parser.add_argument("--node-summary-max-chars", type=int, default=12000)
    parser.add_argument(
        "--include-parent-span-reference",
        action="store_true",
        help=(
            "Include raw parent span text in merge prompts. Default is child-summary-only; "
            "if enabled, the span is hard-checked against --node-summary-max-chars."
        ),
    )
    parser.add_argument("--score-temperature", type=float, default=0.0)
    parser.add_argument("--score-max-tokens", type=int, default=180)
    parser.add_argument("--score-max-chars", type=int, default=6000)
    parser.add_argument(
        "--scoring-context-source",
        choices=["compact", "benoit"],
        default="compact",
        help=(
            "Scoring rubric used by teacher f. compact is the short local context; "
            "benoit uses the exact released Benoit rubric text before the JSON wrapper."
        ),
    )
    parser.add_argument("--score-input", choices=["teacher_summary", "node_span"], default="teacher_summary")
    parser.add_argument(
        "--summary-mode",
        choices=["teacher", "identity", "off"],
        default="teacher",
        help="How to populate teacher g summaries. Use off for true f-only node-span scoring.",
    )
    parser.add_argument("--missing-score-policy", choices=["error", "neutral"], default="error")
    parser.add_argument("--idempotence-mode", choices=["off", "root", "all"], default="root")
    parser.add_argument("--resummary-temperature", type=float, default=0.2)
    parser.add_argument(
        "--resummary-max-tokens",
        type=int,
        default=0,
        help=(
            "Teacher resummary output cap. In size-token mode, 0 means auto = "
            "2 * leaf_size_tokens; legacy count mode falls back to 500."
        ),
    )
    parser.add_argument("--resummary-max-chars", type=int, default=5000)
    parser.add_argument("--label-source", type=str, default="manifesto_dimension_teacher_fg_node_v1")
    add_manifesto_finetune_args(
        parser,
        kind="generic",
        help_text="Write treepo PreferenceDataset/fine-tune adapter bundles per leaf row.",
    )
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument(
        "--lm-concurrency",
        type=int,
        default=32,
        help=(
            "Max concurrent LM calls WITHIN each doc when batching summaries "
            "level-by-level and scores in one pass. Total concurrent LM calls "
            "= num_workers * lm_concurrency. Raise if vLLM has headroom."
        ),
    )
    parser.add_argument("--max-docs-per-split", type=int, default=None)
    parser.add_argument(
        "--min-test-docs",
        type=int,
        default=0,
        help="Fail before teacher calls if the selected split has fewer test docs.",
    )
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    legacy_tree_text_source = str(args.tree_text_source or "").strip().lower()
    if args.source_kind is None:
        if args.tree_bundle_kind is not None:
            args.source_kind = source_kind_for_tree_bundle_kind(str(args.tree_bundle_kind))
        else:
            args.source_kind = source_kind_for_legacy_tree_text_source(legacy_tree_text_source)
    if args.tree_bundle_kind is not None:
        alias_source_kind = source_kind_for_tree_bundle_kind(str(args.tree_bundle_kind))
        if alias_source_kind != str(args.source_kind):
            raise SystemExit(
                "Deprecated --tree-bundle-kind conflicts with --source-kind: "
                f"tree_bundle_kind={args.tree_bundle_kind!r} source_kind={args.source_kind!r}"
            )
    args.tree_bundle_kind = legacy_tree_bundle_kind_for_source_kind(str(args.source_kind))
    expected_tree_text_source = legacy_tree_text_source_for_source_kind(str(args.source_kind))
    if legacy_tree_text_source and legacy_tree_text_source != expected_tree_text_source:
        raise SystemExit(
            "Legacy --tree-text-source is incompatible with TreeBundle source_kind: "
            f"tree_text_source={legacy_tree_text_source!r} "
            f"source_kind={args.source_kind!r}"
        )
    args.tree_text_source = expected_tree_text_source
    args.state_contract = args.state_contract or default_state_contract_for_source_kind(
        str(args.source_kind)
    )
    if args.tree_state_source is None:
        args.tree_state_source = (
            "raw_input"
            if str(args.source_kind) == SOURCE_KIND_RAW_INPUT
            else "external_state"
        )
    if str(args.source_kind) == SOURCE_KIND_EXTERNAL_STATE:
        args.external_state_producer = args.external_state_producer or "g_benoit"
    elif args.external_state_producer:
        raise SystemExit(
            "--external-state-producer is only valid for "
            "source_kind=external_state bundles"
        )
    if args.expert_target_scale is None:
        args.expert_target_scale = EXPERT_SCALE_NORMALIZED_1_7
    target_min, target_max = expert_scale_bounds(
        dimension=str(args.dimension),
        scale=str(args.expert_target_scale),
    )
    LOGGER.info(
        "Using expert target scale=%s bounds=[%.3f, %.3f] with scorer output bounds=[1.000, 7.000]",
        args.expert_target_scale,
        target_min,
        target_max,
    )

    aligned_source_results = _load_alignment_source_results(args.alignment_run_dir)
    source_results = args.source_results or aligned_source_results
    if source_results is None:
        source_results = PROJECT_ROOT / "outputs" / "overnight_benoit" / "full_pipeline" / args.dimension / "per_manifesto.jsonl"
    source_results = Path(source_results)
    source_report = args.source_report
    if source_report is None:
        candidate = source_results.parent / "report.json"
        source_report = candidate if candidate.exists() else None

    output_dir = args.output_dir or PROJECT_ROOT / "outputs" / "manifesto_teacher_fg_leaf_grid" / _now_stamp()
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = _read_jsonl(source_results)
    rows_by_id = {_row_manifesto_id(row): row for row in rows if _row_manifesto_id(row)}
    split_ids = _resolve_split_ids(args, rows_by_id)
    split_ids = _limit_split_ids_to_requested_sizes(
        split_ids,
        train_n=int(args.train_n),
        val_n=int(args.val_n),
        test_n=int(args.test_n),
    )
    split_alignment_metadata = _load_alignment_split_metadata(args.alignment_run_dir)
    args.split_manifest_digest = str(
        split_alignment_metadata.get("split_manifest_digest") or ""
    )
    args.split_schema_version = str(
        split_alignment_metadata.get("schema_version") or ""
    )
    if args.max_docs_per_split is not None:
        limited: Dict[str, Dict[str, str]] = {}
        for split, values in split_ids.items():
            items = list(values.items())[: int(args.max_docs_per_split)]
            limited[split] = dict(items)
        split_ids = limited
    actual_split_sizes = {split: len(values) for split, values in split_ids.items()}
    LOGGER.info(
        "Selected split sizes train=%d val=%d test=%d",
        int(actual_split_sizes.get("train", 0)),
        int(actual_split_sizes.get("val", 0)),
        int(actual_split_sizes.get("test", 0)),
    )
    if int(args.min_test_docs) > 0 and actual_split_sizes.get("test", 0) < int(args.min_test_docs):
        raise SystemExit(
            "Split guard failed: "
            f"dimension={args.dimension} test_docs={actual_split_sizes.get('test', 0)} "
            f"min_test_docs={int(args.min_test_docs)} source_results={source_results}"
        )
    # Resolve the leaf axis. Either count-based (legacy --leaf-grid) or
    # size-based (new --leaf-size-tokens). Mutually exclusive.
    if args.leaf_grid and args.leaf_size_tokens:
        raise SystemExit("--leaf-grid and --leaf-size-tokens are mutually exclusive")
    leaf_size_axis: Optional[Tuple[int, ...]]
    leaf_count_axis: Optional[Tuple[int, ...]]
    if args.leaf_size_tokens:
        leaf_size_axis = _parse_int_grid(args.leaf_size_tokens)
        leaf_count_axis = None
    elif args.leaf_grid:
        leaf_count_axis = _build_leaf_grid(args)
        leaf_size_axis = None
    else:
        leaf_size_axis = (512, 1024, 2048)
        leaf_count_axis = None
    # For backward compatibility with the rest of this function (which loops
    # over ``leaf_grid``), iterate the active axis here. ``leaf_grid`` keeps
    # the legacy semantics.
    leaf_grid = leaf_count_axis if leaf_count_axis is not None else leaf_size_axis
    _write_json(output_dir / "split_ids.json", {split: sorted(values) for split, values in split_ids.items()})

    needs_dataset = any(not text for values in split_ids.values() for text in values.values())
    dataset = ManifestoDataset(data_dir=args.mp_data_dir, require_text=True) if needs_dataset else None

    teacher_client = OpenAIChatClient(
        base_url=str(args.teacher_base_url),
        model=str(args.teacher_model),
        api_key=str(args.teacher_api_key),
        timeout_seconds=float(args.teacher_timeout_seconds),
        enable_thinking=bool(args.enable_thinking),
    )
    scorer_client = OpenAIChatClient(
        base_url=str(args.scorer_base_url or args.teacher_base_url),
        model=str(args.scorer_model or args.teacher_model),
        api_key=str(args.scorer_api_key),
        timeout_seconds=float(args.scorer_timeout_seconds),
        enable_thinking=bool(args.enable_thinking),
    )

    manifest_entries: Dict[str, Any] = {}
    aggregate_rows: List[Dict[str, Any]] = []
    run_metadata = _load_run_metadata(source_report)

    use_size_axis = leaf_size_axis is not None
    for leaf_value in leaf_grid:
        # In count-mode, leaf_value is a leaf count; in size-mode, a token count.
        leaf_count = int(leaf_value) if not use_size_axis else 0
        leaf_size_tokens = int(leaf_value) if use_size_axis else None
        if use_size_axis:
            leaf_dir = output_dir / f"leaf{int(leaf_value):04d}tok"
            manifest_key = f"tok_{int(leaf_value)}"
        else:
            leaf_dir = output_dir / f"leaf_{int(leaf_value):03d}"
            manifest_key = str(int(leaf_value))
        summary_path = leaf_dir / "summary.json"
        if args.skip_existing and summary_path.exists():
            manifest_entries[manifest_key] = json.loads(summary_path.read_text(encoding="utf-8"))
            continue
        leaf_dir.mkdir(parents=True, exist_ok=True)
        summary_cache = JsonlCallCache(leaf_dir / "teacher_g_summary_cache.jsonl")
        score_cache = JsonlCallCache(leaf_dir / "teacher_f_score_cache.jsonl")
        resummary_cache = JsonlCallCache(leaf_dir / "teacher_g_resummary_cache.jsonl")

        work: List[Tuple[Mapping[str, Any], str, str]] = []
        skipped = {"missing_row": 0, "missing_text": 0}
        for split, id_to_text in split_ids.items():
            for doc_id, split_text in id_to_text.items():
                row = rows_by_id.get(str(doc_id))
                if row is None:
                    skipped["missing_row"] += 1
                    continue
                if str(args.tree_text_source) == "existing_summary":
                    text = str(row.get("summary") or row.get("root_summary") or "").strip()
                else:
                    text = _get_text_for_row(row=row, split_texts={str(doc_id): split_text}, dataset=dataset)
                if not text.strip():
                    skipped["missing_text"] += 1
                    continue
                work.append((row, text, str(split)))

        LOGGER.info(
            "Teacher f/g %s=%s docs=%d workers=%d output=%s",
            "leaf_size_tokens" if use_size_axis else "leaf_count",
            leaf_value,
            len(work),
            int(args.num_workers),
            leaf_dir,
        )
        trees: List[LabeledTree] = []
        failures: List[Dict[str, Any]] = []
        worker_count = min(max(1, int(args.num_workers)), max(1, len(work)))
        if worker_count > 1 and len(work) > 1:
            with ThreadPoolExecutor(max_workers=worker_count) as pool:
                future_to_idx = {
                    pool.submit(
                        _process_doc,
                        row=row,
                        text=text,
                        split=split,
                        dimension_name=str(args.dimension),
                        leaf_count=int(leaf_count),
                        leaf_size_tokens=leaf_size_tokens,
                        source_results=source_results,
                        source_report=source_report,
                        teacher_client=teacher_client,
                        scorer_client=scorer_client,
                        summary_cache=summary_cache,
                        score_cache=score_cache,
                        resummary_cache=resummary_cache,
                        args=args,
                    ): idx
                    for idx, (row, text, split) in enumerate(work)
                }
                for done, future in enumerate(as_completed(future_to_idx), start=1):
                    tree, failure = future.result()
                    if tree is not None:
                        trees.append(tree)
                    if failure is not None:
                        failures.append(failure)
                    if done % 10 == 0:
                        LOGGER.info(
                            "%s=%s completed=%d/%d trees=%d failures=%d",
                            "leaf_size_tokens" if use_size_axis else "leaf_count",
                            leaf_value,
                            done,
                            len(work),
                            len(trees),
                            len(failures),
                        )
        else:
            for idx, (row, text, split) in enumerate(work, start=1):
                tree, failure = _process_doc(
                    row=row,
                    text=text,
                    split=split,
                    dimension_name=str(args.dimension),
                    leaf_count=int(leaf_count),
                    leaf_size_tokens=leaf_size_tokens,
                    source_results=source_results,
                    source_report=source_report,
                    teacher_client=teacher_client,
                    scorer_client=scorer_client,
                    summary_cache=summary_cache,
                    score_cache=score_cache,
                    resummary_cache=resummary_cache,
                    args=args,
                )
                if tree is not None:
                    trees.append(tree)
                if failure is not None:
                    failures.append(failure)
                if idx % 10 == 0:
                    LOGGER.info(
                        "%s=%s completed=%d/%d trees=%d failures=%d",
                        "leaf_size_tokens" if use_size_axis else "leaf_count",
                        leaf_value,
                        idx,
                        len(work),
                        len(trees),
                        len(failures),
                    )

        trees.sort(key=lambda tree: (str((tree.metadata or {}).get("split") or ""), str(tree.doc_id)))
        labeled_tree_path = write_labeled_trees_jsonl(leaf_dir / "labeled_trees.jsonl", trees)
        finetune_bundle = export_manifesto_finetune_bundle_from_args(
            args=args,
            trees=trees,
            output_dir=leaf_dir / "treepo_finetune",
            kind="generic",
            leaf_unit_type=str(args.leaf_unit or "leaf"),
            logger=LOGGER,
            log_label="Manifesto",
        )
        node_rows: List[Dict[str, Any]] = []
        for tree in trees:
            node_rows.extend(
                _flat_node_rows(
                    tree,
                    leaf_count=None if leaf_size_tokens is not None else int(leaf_count),
                    leaf_size_tokens=leaf_size_tokens,
                    dimension_name=str(args.dimension),
                )
            )
        node_rows_path = leaf_dir / "teacher_node_rows.jsonl"
        if node_rows_path.exists():
            node_rows_path.unlink()
        _append_jsonl(node_rows_path, node_rows)
        failures_path = leaf_dir / "failures.json"
        if failures:
            _write_json(failures_path, {"failures": failures})
        elif failures_path.exists():
            failures_path.unlink()

        node_scores = [float(row["score_1_7"]) for row in node_rows if row.get("score_1_7") is not None]
        root_scores = [float(row["score_1_7"]) for row in node_rows if row.get("is_root") and row.get("score_1_7") is not None]
        summary = {
            "topology_axis": "size_tokens" if leaf_size_tokens is not None else "leaf_count",
            "leaf_count": None if leaf_size_tokens is not None else int(leaf_count),
            "leaf_size_tokens": int(leaf_size_tokens) if leaf_size_tokens is not None else None,
            "leaf_count_stats": {
                "min": min((len(tree.get_leaves()) for tree in trees), default=0),
                "max": max((len(tree.get_leaves()) for tree in trees), default=0),
                "mean": (
                    float(sum(len(tree.get_leaves()) for tree in trees) / len(trees))
                    if trees
                    else 0.0
                ),
            },
            "dimension": str(args.dimension),
            **tree_bundle_metadata(
                domain="manifesto_rile",
                leaf_unit=str(args.leaf_unit),
                source_kind=str(args.source_kind),
                dimension=str(args.dimension),
                target_scale=str(args.expert_target_scale),
                leaf_policy={
                    "topology_axis": "size_tokens"
                    if leaf_size_tokens is not None
                    else "leaf_count",
                    "leaf_count": None
                    if leaf_size_tokens is not None
                    else int(leaf_count),
                    "leaf_size_tokens": int(leaf_size_tokens)
                    if leaf_size_tokens is not None
                    else None,
                },
                state_contract=str(args.state_contract),
                reducer_contract=str(args.reducer_contract),
                external_state_producer=(
                    str(args.external_state_producer)
                    if args.external_state_producer
                    else None
                ),
                metadata={
                    "split_manifest_digest": str(args.split_manifest_digest or ""),
                    "split_alignment_run_dir": (
                        str(args.alignment_run_dir) if args.alignment_run_dir else None
                    ),
                    "split_schema_version": str(args.split_schema_version or ""),
                },
            ),
            "tree_bundle_kind": str(args.tree_bundle_kind),
            "tree_state_source": str(args.tree_state_source),
            "external_state_producer": (
                str(args.external_state_producer)
                if args.external_state_producer
                else None
            ),
            "tree_text_source": str(args.tree_text_source),
            "tree_counts": {
                "total": len(trees),
                "train": sum(1 for tree in trees if (tree.metadata or {}).get("split") == "train"),
                "val": sum(1 for tree in trees if (tree.metadata or {}).get("split") == "val"),
                "test": sum(1 for tree in trees if (tree.metadata or {}).get("split") == "test"),
                "skipped": skipped,
                "failures": len(failures),
            },
            "node_count": len(node_rows),
            "score_summary": {
                "node_mean_1_7": float(sum(node_scores) / len(node_scores)) if node_scores else None,
                "root_mean_1_7": float(sum(root_scores) / len(root_scores)) if root_scores else None,
                "node_min_1_7": float(min(node_scores)) if node_scores else None,
                "node_max_1_7": float(max(node_scores)) if node_scores else None,
            },
            "cache": {
                "summary": summary_cache.stats(),
                "score": score_cache.stats(),
                "resummary": resummary_cache.stats(),
            },
            "artifacts": {
                "labeled_trees": str(labeled_tree_path),
                "teacher_node_rows": str(node_rows_path),
                "summary_cache": str(summary_cache.path),
                "score_cache": str(score_cache.path),
                "resummary_cache": str(resummary_cache.path),
                "failures": str(leaf_dir / "failures.json") if failures else None,
                "finetune_bundle": (
                    str(leaf_dir / "treepo_finetune") if finetune_bundle else None
                ),
            },
            "finetune": finetune_bundle,
            "teacher_fg_model": {
                "g_base_url": str(args.teacher_base_url),
                "g_model": str(args.teacher_model),
                "f_base_url": str(args.scorer_base_url or args.teacher_base_url),
                "f_model": str(args.scorer_model or args.teacher_model),
                "score_input": str(args.score_input),
                "scoring_context_source": str(args.scoring_context_source),
                "summary_mode": str(args.summary_mode),
                "idempotence_mode": str(args.idempotence_mode),
            },
        }
        _write_json(summary_path, summary)
        manifest_entries[manifest_key] = summary
        aggregate_rows.append(
            {
                "topology_axis": "size_tokens" if leaf_size_tokens is not None else "leaf_count",
                "leaf_count": None if leaf_size_tokens is not None else int(leaf_count),
                "leaf_size_tokens": int(leaf_size_tokens) if leaf_size_tokens is not None else None,
                "dimension": str(args.dimension),
                **tree_bundle_metadata(
                    domain="manifesto_rile",
                    leaf_unit=str(args.leaf_unit),
                    source_kind=str(args.source_kind),
                    dimension=str(args.dimension),
                    target_scale=str(args.expert_target_scale),
                    leaf_policy={
                        "topology_axis": "size_tokens"
                        if leaf_size_tokens is not None
                        else "leaf_count",
                        "leaf_count": None
                        if leaf_size_tokens is not None
                        else int(leaf_count),
                        "leaf_size_tokens": int(leaf_size_tokens)
                        if leaf_size_tokens is not None
                        else None,
                    },
                    state_contract=str(args.state_contract),
                    reducer_contract=str(args.reducer_contract),
                    external_state_producer=(
                        str(args.external_state_producer)
                        if args.external_state_producer
                        else None
                    ),
                    metadata={
                        "split_manifest_digest": str(args.split_manifest_digest or ""),
                        "split_alignment_run_dir": (
                            str(args.alignment_run_dir) if args.alignment_run_dir else None
                        ),
                        "split_schema_version": str(args.split_schema_version or ""),
                    },
                ),
                "tree_bundle_kind": str(args.tree_bundle_kind),
                "tree_state_source": str(args.tree_state_source),
                "external_state_producer": (
                    str(args.external_state_producer)
                    if args.external_state_producer
                    else None
                ),
                "tree_text_source": str(args.tree_text_source),
                "tree_count": len(trees),
                "node_count": len(node_rows),
                "failures": len(failures),
                "node_mean_1_7": summary["score_summary"]["node_mean_1_7"],
                "root_mean_1_7": summary["score_summary"]["root_mean_1_7"],
                "labeled_trees": str(labeled_tree_path),
            }
        )

    aggregate_path = output_dir / "aggregate_teacher_metrics.jsonl"
    if aggregate_path.exists():
        aggregate_path.unlink()
    _append_jsonl(aggregate_path, aggregate_rows)
    manifest = {
        "created_at": _now_iso(),
        "dimension": str(args.dimension),
        "alignment_run_dir": str(args.alignment_run_dir) if args.alignment_run_dir else None,
        "source_results": str(source_results),
        "source_report": str(source_report) if source_report else None,
        "run_metadata": run_metadata,
        "config": {
            "leaf_grid": (
                list(leaf_count_axis) if leaf_count_axis is not None else None
            ),
            "leaf_size_tokens": (
                list(leaf_size_axis) if leaf_size_axis is not None else None
            ),
                "topology_axis": "size_tokens" if leaf_size_axis is not None else "leaf_count",
                "include_parent_span_reference": bool(args.include_parent_span_reference),
            "split_source": str(args.split_source),
            "train_n": int(args.train_n),
            "val_n": int(args.val_n),
            "test_n": int(args.test_n),
            "actual_split_sizes": actual_split_sizes,
            "min_test_docs": int(args.min_test_docs),
            "expert_target_scale": str(args.expert_target_scale),
            **tree_bundle_metadata(
                domain="manifesto_rile",
                leaf_unit=str(args.leaf_unit),
                source_kind=str(args.source_kind),
                dimension=str(args.dimension),
                target_scale=str(args.expert_target_scale),
                leaf_policy={
                    "topology_axis": "size_tokens"
                    if leaf_size_axis is not None
                    else "leaf_count",
                    "leaf_grid": (
                        list(leaf_count_axis) if leaf_count_axis is not None else None
                    ),
                    "leaf_size_tokens": (
                        list(leaf_size_axis) if leaf_size_axis is not None else None
                    ),
                },
                state_contract=str(args.state_contract),
                reducer_contract=str(args.reducer_contract),
                external_state_producer=(
                    str(args.external_state_producer)
                    if args.external_state_producer
                    else None
                ),
                metadata={
                    "split_manifest_digest": str(args.split_manifest_digest or ""),
                    "split_alignment_run_dir": (
                        str(args.alignment_run_dir) if args.alignment_run_dir else None
                    ),
                    "split_schema_version": str(args.split_schema_version or ""),
                },
            ),
            "tree_bundle_kind": str(args.tree_bundle_kind),
            "tree_state_source": str(args.tree_state_source),
            "external_state_producer": (
                str(args.external_state_producer)
                if args.external_state_producer
                else None
            ),
            "tree_text_source": str(args.tree_text_source),
            "score_input": str(args.score_input),
            "summary_mode": str(args.summary_mode),
            "idempotence_mode": str(args.idempotence_mode),
            "label_source": str(args.label_source),
            "finetune_export": finetune_export_config(args),
        },
        "teacher_fg_model": {
            "g_base_url": str(args.teacher_base_url),
            "g_model": str(args.teacher_model),
            "f_base_url": str(args.scorer_base_url or args.teacher_base_url),
            "f_model": str(args.scorer_model or args.teacher_model),
        },
        "runs": manifest_entries,
        "artifacts": {
            "split_ids": str(output_dir / "split_ids.json"),
            "aggregate_teacher_metrics": str(aggregate_path),
        },
    }
    manifest["run_manifest"] = run_manifest_metadata(
        run_id=f"manifesto.teacher_leaf_grid.{args.dimension}",
        domain="manifesto_rile",
        role="teacher_tree_bundle",
        backend="dspy",
        status="completed",
        tree_bundle=manifest["config"]["tree_bundle_manifest"],
        f_init="teacher_f",
        g_init="teacher_g",
        f_lineage={
            "init": "teacher_f",
            "base_url": str(args.scorer_base_url or args.teacher_base_url),
            "model": str(args.scorer_model or args.teacher_model),
            "score_input": str(args.score_input),
            "scoring_context_source": str(args.scoring_context_source),
        },
        g_lineage={
            "init": "teacher_g",
            "base_url": str(args.teacher_base_url),
            "model": str(args.teacher_model),
            "summary_mode": str(args.summary_mode),
            "idempotence_mode": str(args.idempotence_mode),
            "tree_text_source": str(args.tree_text_source),
        },
        reducer_contract=str(args.reducer_contract),
        schedule="teacher_trace",
        objective=objective_metadata(
            objective_family="manifesto_teacher_first_trace",
            local_law_estimator=LOCAL_LAW_ESTIMATOR_NONE,
            root_share=0.0,
            local_law_component_weights={},
            metadata={
                "dimension": str(args.dimension),
                "score_input": str(args.score_input),
                "label_source": str(args.label_source),
                "teacher_trace_component": "teacher_node_trace",
            },
        ),
        optimizer_config={
            "split_source": str(args.split_source),
            "label_source": str(args.label_source),
            "leaf_grid": list(leaf_count_axis) if leaf_count_axis is not None else None,
            "leaf_size_tokens": list(leaf_size_axis) if leaf_size_axis is not None else None,
        },
        output_artifacts=[
            {"kind": "manifest", "uri": str(output_dir / "manifest.json")},
            {"kind": "aggregate_teacher_metrics", "uri": str(aggregate_path)},
            {"kind": "tree_bundle_directory", "uri": str(output_dir)},
        ],
        audit_results={
            "ok": True,
            "source_kind": str(args.source_kind),
            "tree_text_source": str(args.tree_text_source),
        },
        quarantine={"classification": "valid_treebundle_v1"},
        command=sys.argv,
        allow_legacy=False,
        publication_ready=str(args.source_kind) == SOURCE_KIND_RAW_INPUT,
        metadata={
            "runner": "scripts/run_manifesto_teacher_fg_leaf_grid.py",
            "actual_split_sizes": actual_split_sizes,
            "split_manifest_digest": str(args.split_manifest_digest or ""),
            "split_alignment_run_dir": (
                str(args.alignment_run_dir) if args.alignment_run_dir else None
            ),
            "split_schema_version": str(args.split_schema_version or ""),
        },
    )
    _write_json(output_dir / "manifest.json", manifest)
    LOGGER.info("Wrote teacher f/g grid manifest to %s", output_dir / "manifest.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
