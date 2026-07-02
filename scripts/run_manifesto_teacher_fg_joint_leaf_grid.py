#!/usr/bin/env python3
"""Build all-six Manifesto teacher f* node labels in one scorer pass.

This is the joint counterpart to ``run_manifesto_teacher_fg_leaf_grid.py``.
It builds each exact leaf-size topology once, then asks the scorer for all six
Benoit dimension scores for each realized tree node in a single request.  The
output is the vector-labeled teacher grid consumed by ``run_alternating_ladder``
with ``--dimension combined``.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import logging
import re
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
    append_jsonl as _append_jsonl,
    read_jsonl as _read_jsonl,
    require_within_chars as _require_within_chars,
    stable_hash as _stable_hash,
    write_json as _write_json,
)
from src.experiments.script_parse import parse_int_grid as _parse_int_grid  # noqa: E402
from src.experiments.tree_helpers import root_node as _tree_root_node  # noqa: E402
from src.tasks.manifesto.result_rows import (  # noqa: E402
    DIMENSION_BY_NAME as _DIM_FROM_NAME,
    get_text_for_row as _get_text_for_row,
    load_run_metadata as _load_run_metadata,
    order_split_rows as _order_split_rows,
    phase3_split_examples as _phase3_split_examples,
    row_expert_score as _row_expert_score,
    row_manifesto_id as _row_manifesto_id,
)
from src.ctreepo.distillation import (  # noqa: E402
    annotate_labeled_tree_summary_coverage,
    build_labeled_tree_from_text,
    write_labeled_trees_jsonl,
)
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
    mean as _mean,
    now_iso as _now_iso,
    now_stamp as _now_stamp,
    safe_float as _safe_float,
)
from src.tasks.manifesto.dimensions import (  # noqa: E402
    PolicyDimension,
    get_dimension,
)
from src.training.config_sections import config_to_dict  # noqa: E402
from src.tree.labeled import LabeledNode, LabeledTree  # noqa: E402


LOGGER = logging.getLogger(__name__)

DIMS: Tuple[str, ...] = tuple(dim.value for dim in PolicyDimension)
PROMPT_VERSION = "joint_manifesto_node_scores_v1"
SOURCE_KIND_CHOICES = (SOURCE_KIND_RAW_INPUT, SOURCE_KIND_EXTERNAL_STATE)
TREE_BUNDLE_KIND_CHOICES = ("raw_manifesto_token_tree", "external_summary_token_tree")
STATE_CONTRACT_CHOICES = (STATE_CONTRACT_RAW_CONCAT, STATE_CONTRACT_EXTERNAL_PASSTHROUGH)



def _extract_json_object(text: str) -> Optional[Mapping[str, Any]]:
    rendered = str(text or "").strip()
    if not rendered:
        return None
    candidates = [rendered]
    start = rendered.find("{")
    end = rendered.rfind("}")
    if start >= 0 and end > start:
        candidates.append(rendered[start : end + 1])
    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, Mapping):
            return parsed
    return None


def _parse_joint_scores(
    response: str,
    *,
    dimensions: Sequence[str],
    missing_policy: str,
) -> Tuple[Dict[str, float], Dict[str, Any]]:
    parsed = _extract_json_object(response)
    if parsed is None:
        parsed = {}
    scores_obj = parsed.get("scores") if isinstance(parsed.get("scores"), Mapping) else parsed
    reasoning_obj = parsed.get("reasoning") if isinstance(parsed.get("reasoning"), Mapping) else {}
    scores: Dict[str, float] = {}
    missing: List[str] = []
    for dim_name in dimensions:
        raw = scores_obj.get(dim_name) if isinstance(scores_obj, Mapping) else None
        if raw is None and isinstance(scores_obj, Mapping):
            raw = scores_obj.get(dim_name.replace("_", " "))
        value = _safe_float(raw)
        if value is None:
            # Last-ditch parser for responses such as "economic: 3".
            match = re.search(rf"\b{re.escape(dim_name)}\b[^0-9-]*(-?\d+(?:\.\d+)?)", response, re.I)
            value = _safe_float(match.group(1)) if match else None
        if value is None:
            if missing_policy == "neutral":
                value = 4.0
                missing.append(dim_name)
            else:
                raise ValueError(f"Could not parse {dim_name} 1-7 score from joint response: {response!r}")
        spec = get_dimension(_DIM_FROM_NAME[dim_name])
        scores[dim_name] = float(spec.scale.clamp(float(value)))
    payload = {
        "response": response,
        "parsed_json": config_to_dict(parsed),
        "reasoning": config_to_dict(reasoning_obj),
    }
    if missing:
        payload["missing_policy_applied"] = {"neutral": missing}
    return scores, payload


def _joint_scoring_context(dimensions: Sequence[str]) -> str:
    lines = [
        "You are an expert political scientist with a PhD in political science.",
        "Score one C-TreePO tree node on all six Benoit manifesto policy dimensions.",
        "Use the 1-7 integer scale for each dimension. Use null only when the node is truly unscorable.",
        "",
        "Dimension scales:",
    ]
    for dim_name in dimensions:
        spec = get_dimension(_DIM_FROM_NAME[dim_name])
        lines.append(f"- {dim_name}: 1 = {spec.anchor_low}; 7 = {spec.anchor_high}; 4 = neutral / balanced.")
    lines.extend(
        [
            "",
            "Return strict JSON only, exactly with keys:",
            '{"scores":{"economic":number|null,"social":number|null,"immigration":number|null,"eu":number|null,"environment":number|null,"decentralization":number|null},"reasoning":{"economic":string,"social":string,"immigration":string,"eu":string,"environment":string,"decentralization":string}}',
        ]
    )
    return "\n".join(lines)


def _build_joint_score_fn(
    *,
    client: OpenAIChatClient,
    dimensions: Sequence[str],
    temperature: float,
    max_tokens: int,
    max_chars: int,
    missing_policy: str,
):
    context = _joint_scoring_context(dimensions)

    def _score(text: str, *, role: str, node_context: Mapping[str, Any]) -> Tuple[Dict[str, float], Dict[str, Any]]:
        checked = _require_within_chars(
            str(text or ""),
            max_chars=int(max_chars),
            label=f"joint score input for {node_context.get('doc_id')}:{node_context.get('node_id')}",
        )
        user = (
            f"{context}\n\n"
            "You are scoring one C-TreePO tree node, not necessarily a full manifesto.\n"
            f"Node role: {role}\n"
            f"Document id: {node_context.get('doc_id')}\n"
            f"Node id: {node_context.get('node_id')}\n\n"
            f"NODE_TEXT:\n{checked}"
        )
        response = client.chat(
            system=(
                "You are a strict expert political-science multi-dimensional scorer. "
                "Return only strict JSON. Do not include markdown."
            ),
            user=user,
            temperature=float(temperature),
            max_tokens=int(max_tokens),
        )
        return _parse_joint_scores(response, dimensions=dimensions, missing_policy=missing_policy)

    return _score


def _node_summary(node: LabeledNode) -> str:
    metadata = node.metadata if isinstance(node.metadata, Mapping) else {}
    return str(metadata.get("teacher_summary") or metadata.get("target_summary") or "").strip()


def _build_joint_labeled_tree(
    *,
    row: Mapping[str, Any],
    text: str,
    split: str,
    dimensions: Sequence[str],
    source_results: Path,
    source_report: Optional[Path],
    scorer_client: OpenAIChatClient,
    score_cache: JsonlCallCache,
    args: argparse.Namespace,
    leaf_size_tokens: int,
) -> LabeledTree:
    doc_id = _row_manifesto_id(row)
    if not doc_id:
        raise ValueError("row has no manifesto_id/doc_id")
    from src.preprocessing.leaf_size_utils import char_windows_from_token_budget

    explicit_windows = char_windows_from_token_budget(str(text), int(leaf_size_tokens))
    expert_scores = {
        dim: float(score)
        for dim in dimensions
        if (score := _row_expert_score(row, dimension=dim)) is not None
    }
    label_source = str(args.label_source)

    def node_summary_fn(span: str, context: Mapping[str, Any]) -> str:
        if str(args.summary_mode) == "off":
            return ""
        return str(span or "").strip()

    tree = build_labeled_tree_from_text(
        doc_id=str(doc_id),
        text=str(text),
        document_score=4.0,
        split=str(split),
        score_fn=lambda _span: 4.0,
        window_size=max(1, len(str(text))),
        target_leaves_per_doc=None,
        explicit_char_windows=explicit_windows,
        label_source=label_source,
        node_summary_fn=node_summary_fn if str(args.summary_mode) == "identity" else None,
        fill_missing_summaries_from_span=False,
        summary_source=(
            "span_identity_fallback"
            if str(args.summary_mode) == "identity"
            else "score_only_no_summary"
        ),
        extra_metadata={
            "dimension": "combined",
            "combined_dimensions": list(dimensions),
            "expert_dimension_scores_1_7": expert_scores,
            "expert_score_1_7": _mean(list(expert_scores.values())),
            "source_results_path": str(source_results),
            "source_report_path": str(source_report) if source_report else None,
            **tree_bundle_metadata(
                domain="manifesto_rile",
                leaf_unit=str(args.leaf_unit),
                source_kind=str(args.source_kind),
                dimension="combined",
                target_scale="normalized_1_7",
                leaf_policy={
                    "topology_axis": "size_tokens",
                    "leaf_size_tokens": int(leaf_size_tokens),
                },
                state_contract=str(args.state_contract),
                reducer_contract=str(args.reducer_contract),
                external_state_producer=(
                    str(args.external_state_producer)
                    if args.external_state_producer
                    else None
                ),
            ),
            "tree_state_source": str(args.tree_state_source),
            "teacher_fg_model": {
                "f_base_url": str(args.scorer_base_url),
                "f_model": str(args.scorer_model),
                "score_input": str(args.score_input),
                "summary_mode": str(args.summary_mode),
                "joint_score_prompt_version": PROMPT_VERSION,
            },
            "node_score_source": "teacher_f_joint_dimension_scores_1_7",
            "topology_axis": "size_tokens",
            "leaf_count": None,
            "leaf_size_tokens": int(leaf_size_tokens),
            "derived_leaf_count": int(len(explicit_windows)),
            "tokenizer_model": "/mnt/data/models/google/embeddinggemma-300m",
        },
    )

    axis_tag = f"tok_{int(leaf_size_tokens)}"
    joint_score_fn = _build_joint_score_fn(
        client=scorer_client,
        dimensions=dimensions,
        temperature=float(args.score_temperature),
        max_tokens=int(args.score_max_tokens),
        max_chars=int(args.score_max_chars),
        missing_policy=str(args.missing_score_policy),
    )

    def _resolve_score(node: LabeledNode) -> Tuple[str, Dict[str, Any]]:
        summary = _node_summary(node)
        score_input = summary if str(args.score_input) == "teacher_summary" and summary else str(node.text or "")
        input_hash = _stable_hash(
            json.dumps(
                {
                    "score_input": score_input,
                    "score_input_kind": str(args.score_input),
                    "dimensions": list(dimensions),
                    "scorer_model": str(args.scorer_model),
                    "prompt_version": PROMPT_VERSION,
                },
                sort_keys=True,
            )
        )
        key = f"joint_score:v1:{','.join(dimensions)}:{axis_tag}:{doc_id}:{node.node_id}:{input_hash}"
        cached = score_cache.get(key)
        if cached is not None:
            return key, cached
        scores, payload = joint_score_fn(
            score_input,
            role="teacher_summary" if str(args.score_input) == "teacher_summary" else "node_span",
            node_context={
                "doc_id": doc_id,
                "node_id": node.node_id,
                "level": int(node.level),
                "is_leaf": int(node.level) == 0,
                "leaf_size_tokens": int(leaf_size_tokens),
                "topology_axis": "size_tokens",
            },
        )
        written = score_cache.put(
            key,
            {
                "kind": "joint_score",
                "dimensions": list(dimensions),
                "doc_id": doc_id,
                "split": split,
                "leaf_size_tokens": int(leaf_size_tokens),
                "topology_axis": "size_tokens",
                "node_id": str(node.node_id),
                "level": int(node.level),
                "is_leaf": int(node.level) == 0,
                "scorer_model": str(args.scorer_model),
                "score_input_kind": str(args.score_input),
                "input_hash": input_hash,
                "scores_1_7": scores,
                "macro_score_1_7": float(_mean(list(scores.values())) or 4.0),
                **payload,
            },
        )
        return key, written

    node_list = list(tree.nodes.values())
    score_results: Dict[str, Tuple[str, Dict[str, Any]]] = {}
    with ThreadPoolExecutor(max_workers=max(1, int(args.lm_concurrency))) as pool:
        futures = {pool.submit(_resolve_score, node): str(node.node_id) for node in node_list}
        for future in as_completed(futures):
            node_id = futures[future]
            score_results[node_id] = future.result()

    root_node: Optional[LabeledNode] = None
    for node in node_list:
        score_key, cached = score_results[str(node.node_id)]
        raw_scores = cached.get("scores_1_7")
        scores = {
            dim: float(value)
            for dim, value in (raw_scores or {}).items()
            if dim in dimensions and _safe_float(value) is not None
        }
        if len(scores) != len(dimensions):
            if str(args.missing_score_policy) != "neutral":
                raise ValueError(f"incomplete joint score cache row for {doc_id}:{node.node_id}: {cached}")
            for dim in dimensions:
                scores.setdefault(dim, 4.0)
        macro = float(_mean(list(scores.values())) or 4.0)
        node.score = macro
        node.dimension_scores = scores
        node.metadata["dimension"] = "combined"
        node.metadata["teacher_dimension_scores_1_7"] = scores
        node.metadata["teacher_score_1_7"] = macro
        node.metadata["teacher_score_source"] = "teacher_f_joint_dimension_scores_1_7"
        node.metadata["teacher_score_input_kind"] = str(args.score_input)
        node.metadata["teacher_summary_mode"] = str(args.summary_mode)
        node.metadata["teacher_score_model"] = str(args.scorer_model)
        node.metadata["teacher_score_cache_key"] = score_key
        if root_node is None or int(node.level) >= int(root_node.level):
            root_node = node

    root_node = _tree_root_node(tree)
    if root_node is not None:
        root_scores = dict(root_node.dimension_scores or {})
        tree.document_score = float(_mean(list(root_scores.values())) or 4.0)
        tree.metadata["teacher_dimension_scores_1_7"] = root_scores
        tree.metadata["teacher_score_1_7"] = tree.document_score
    annotate_labeled_tree_summary_coverage(tree)
    return tree


def _flat_node_rows(
    tree: LabeledTree,
    *,
    dimensions: Sequence[str],
    leaf_size_tokens: int,
) -> List[Dict[str, Any]]:
    root = _tree_root_node(tree)
    root_id = str(root.node_id) if root is not None else ""
    rows: List[Dict[str, Any]] = []
    for level_ids in tree.levels:
        for node_id in level_ids:
            node = tree.get_node(str(node_id))
            if node is None:
                continue
            metadata = node.metadata if isinstance(node.metadata, Mapping) else {}
            scores = dict(node.dimension_scores or {})
            row = {
                "doc_id": tree.doc_id,
                "split": (tree.metadata or {}).get("split"),
                "dimension": "combined",
                "topology_axis": "size_tokens",
                "leaf_count": None,
                "leaf_size_tokens": int(leaf_size_tokens),
                "derived_leaf_count": len(tree.get_leaves()),
                "node_id": str(node.node_id),
                "level": int(node.level),
                "is_leaf": int(node.level) == 0,
                "is_root": str(node.node_id) == root_id,
                "char_start": metadata.get("char_start"),
                "char_end": metadata.get("char_end"),
                "score_1_7": float(node.score),
                "teacher_summary": metadata.get("teacher_summary"),
                "teacher_score_input_kind": metadata.get("teacher_score_input_kind"),
                "left_child_id": node.left_child_id,
                "right_child_id": node.right_child_id,
                "dimension_scores_1_7": scores,
            }
            for dim in dimensions:
                row[f"{dim}_score_1_7"] = scores.get(dim)
            rows.append(row)
    return rows


def _resolve_split_ids(args: argparse.Namespace, rows_by_id: Mapping[str, Mapping[str, Any]]) -> Dict[str, Dict[str, str]]:
    if args.split_source == "phase3":
        return _phase3_split_examples(
            dimension=PolicyDimension.ECONOMIC,
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


def _process_doc(
    *,
    row: Mapping[str, Any],
    text: str,
    split: str,
    dimensions: Sequence[str],
    source_results: Path,
    source_report: Optional[Path],
    scorer_client: OpenAIChatClient,
    score_cache: JsonlCallCache,
    args: argparse.Namespace,
    leaf_size_tokens: int,
) -> Tuple[Optional[LabeledTree], Optional[Dict[str, Any]]]:
    try:
        return (
            _build_joint_labeled_tree(
                row=row,
                text=text,
                split=split,
                dimensions=dimensions,
                source_results=source_results,
                source_report=source_report,
                scorer_client=scorer_client,
                score_cache=score_cache,
                args=args,
                leaf_size_tokens=int(leaf_size_tokens),
            ),
            None,
        )
    except Exception as exc:
        return None, {
            "doc_id": _row_manifesto_id(row),
            "split": split,
            "dimension": "combined",
            "dimensions": list(dimensions),
            "topology_axis": "size_tokens",
            "leaf_size_tokens": int(leaf_size_tokens),
            "error": _http_error_detail(exc),
        }


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--source-results", type=Path, required=True)
    parser.add_argument("--source-report", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--leaf-size-tokens", default="512,1024,2048")
    parser.add_argument("--dimensions", default=",".join(DIMS))
    parser.add_argument("--split-source", choices=["phase3", "results-order"], default="results-order")
    parser.add_argument("--train-pool", choices=["expert-split", "openweight", "expert"], default="expert-split")
    parser.add_argument("--split-strategy", choices=["random", "label-stratified"], default="label-stratified")
    parser.add_argument("--tree-text-source", choices=["aligned_text", "existing_summary"], default=None, help=argparse.SUPPRESS)
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
        help="Generic TreeBundle v1 source provenance. Defaults to raw_input.",
    )
    parser.add_argument("--leaf-unit", default=LEAF_UNIT_TEXT_TOKEN)
    parser.add_argument(
        "--state-contract",
        choices=STATE_CONTRACT_CHOICES,
        default=None,
        help="Defaults to raw_concat for raw_input and external_passthrough for external_state.",
    )
    parser.add_argument(
        "--reducer-contract",
        choices=[REDUCER_CONTRACT_BOTTOM_UP],
        default=REDUCER_CONTRACT_BOTTOM_UP,
    )
    parser.add_argument(
        "--tree-state-source",
        default=None,
        help="Human-readable state source stored in bundle metadata.",
    )
    parser.add_argument(
        "--external-state-producer",
        default=None,
        help="External state producer for source_kind=external_state, for example g_benoit.",
    )
    parser.add_argument("--train-n", type=int, default=140)
    parser.add_argument("--val-n", type=int, default=30)
    parser.add_argument("--test-n", type=int, default=48)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--mp-data-dir", type=Path, default=None)
    parser.add_argument("--scorer-base-url", type=str, default="http://localhost:8010/v1")
    parser.add_argument("--scorer-model", type=str, default=DEFAULT_MAIN_MODEL)
    parser.add_argument("--scorer-api-key", type=str, default="EMPTY")
    parser.add_argument("--scorer-timeout-seconds", type=float, default=180.0)
    parser.add_argument("--enable-thinking", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--score-temperature", type=float, default=0.0)
    parser.add_argument("--score-max-tokens", type=int, default=700)
    parser.add_argument("--score-max-chars", type=int, default=50000)
    parser.add_argument("--score-input", choices=["teacher_summary", "node_span"], default="teacher_summary")
    parser.add_argument("--summary-mode", choices=["identity", "off"], default="identity")
    parser.add_argument("--missing-score-policy", choices=["error", "neutral"], default="neutral")
    parser.add_argument("--label-source", type=str, default="manifesto_combined_joint_teacher_fg_node_v1")
    add_manifesto_finetune_args(
        parser,
        kind="generic",
        help_text="Write treepo PreferenceDataset/fine-tune adapter bundles per leaf row.",
    )
    parser.add_argument("--num-workers", type=int, default=32)
    parser.add_argument("--lm-concurrency", type=int, default=16)
    parser.add_argument("--max-docs-per-split", type=int, default=None)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    dimensions = [part.strip() for part in str(args.dimensions).replace(";", ",").split(",") if part.strip()]
    unknown = [dim for dim in dimensions if dim not in _DIM_FROM_NAME]
    if unknown:
        raise SystemExit(f"unknown dimensions: {unknown}")
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
            f"tree_text_source={legacy_tree_text_source!r} source_kind={args.source_kind!r}"
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
            "--external-state-producer is only valid for source_kind=external_state bundles"
        )
    leaf_sizes = _parse_int_grid(args.leaf_size_tokens)
    source_results = Path(args.source_results)
    source_report = Path(args.source_report) if args.source_report else None
    if source_report is None:
        candidate = source_results.parent / "report.json"
        source_report = candidate if candidate.exists() else None
    output_dir = args.output_dir or PROJECT_ROOT / "outputs" / "manifesto_teacher_fg_joint_leaf_grid" / _now_stamp()
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = _read_jsonl(source_results)
    rows_by_id = {_row_manifesto_id(row): row for row in rows if _row_manifesto_id(row)}
    split_ids = _resolve_split_ids(args, rows_by_id)
    if args.max_docs_per_split is not None:
        split_ids = {
            split: dict(list(values.items())[: int(args.max_docs_per_split)])
            for split, values in split_ids.items()
        }
    _write_json(output_dir / "split_ids.json", {split: sorted(values) for split, values in split_ids.items()})

    needs_dataset = (
        str(args.tree_text_source) == "aligned_text"
        and any(not text for values in split_ids.values() for text in values.values())
    )
    dataset = ManifestoDataset(data_dir=args.mp_data_dir, require_text=True) if needs_dataset else None
    scorer_client = OpenAIChatClient(
        base_url=str(args.scorer_base_url),
        model=str(args.scorer_model),
        api_key=str(args.scorer_api_key),
        timeout_seconds=float(args.scorer_timeout_seconds),
        enable_thinking=bool(args.enable_thinking),
    )

    manifest_entries: Dict[str, Any] = {}
    aggregate_rows: List[Dict[str, Any]] = []
    run_metadata = _load_run_metadata(source_report)
    for leaf_size in leaf_sizes:
        leaf_dir = output_dir / f"leaf{int(leaf_size):04d}tok"
        summary_path = leaf_dir / "summary.json"
        manifest_key = f"tok_{int(leaf_size)}"
        if args.skip_existing and summary_path.exists():
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            manifest_entries[manifest_key] = summary
            aggregate_rows.append(
                {
                    "topology_axis": "size_tokens",
                    "leaf_size_tokens": int(leaf_size),
                    "dimension": "combined",
                    "dimensions": list(dimensions),
                    "tree_count": summary.get("tree_counts", {}).get("total"),
                    "node_count": summary.get("node_count"),
                    "failures": summary.get("tree_counts", {}).get("failures"),
                    "node_mean_1_7": summary.get("score_summary", {}).get("node_macro_mean_1_7"),
                    "root_mean_1_7": summary.get("score_summary", {}).get("root_macro_mean_1_7"),
                    "labeled_trees": summary.get("artifacts", {}).get("labeled_trees"),
                }
            )
            continue
        leaf_dir.mkdir(parents=True, exist_ok=True)
        score_cache = JsonlCallCache(leaf_dir / "teacher_joint_f_score_cache.jsonl")
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
            "Joint teacher f* leaf_size_tokens=%s docs=%d workers=%d output=%s",
            leaf_size,
            len(work),
            int(args.num_workers),
            leaf_dir,
        )
        trees: List[LabeledTree] = []
        failures: List[Dict[str, Any]] = []
        worker_count = min(max(1, int(args.num_workers)), max(1, len(work)))
        with ThreadPoolExecutor(max_workers=worker_count) as pool:
            futures = {
                pool.submit(
                    _process_doc,
                    row=row,
                    text=text,
                    split=split,
                    dimensions=dimensions,
                    source_results=source_results,
                    source_report=source_report,
                    scorer_client=scorer_client,
                    score_cache=score_cache,
                    args=args,
                    leaf_size_tokens=int(leaf_size),
                ): idx
                for idx, (row, text, split) in enumerate(work)
            }
            for done, future in enumerate(as_completed(futures), start=1):
                tree, failure = future.result()
                if tree is not None:
                    trees.append(tree)
                if failure is not None:
                    failures.append(failure)
                if done % 10 == 0:
                    LOGGER.info(
                        "leaf_size_tokens=%s completed=%d/%d trees=%d failures=%d",
                        leaf_size,
                        done,
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
            log_label="joint Manifesto",
        )
        node_rows: List[Dict[str, Any]] = []
        for tree in trees:
            node_rows.extend(_flat_node_rows(tree, dimensions=dimensions, leaf_size_tokens=int(leaf_size)))
        node_rows_path = leaf_dir / "teacher_node_rows.jsonl"
        if node_rows_path.exists():
            node_rows_path.unlink()
        _append_jsonl(node_rows_path, node_rows)
        failures_path = leaf_dir / "failures.json"
        if failures:
            _write_json(failures_path, {"failures": failures})
        elif failures_path.exists():
            failures_path.unlink()

        node_macros = [float(row["score_1_7"]) for row in node_rows if row.get("score_1_7") is not None]
        root_rows = [row for row in node_rows if row.get("is_root")]
        root_macros = [float(row["score_1_7"]) for row in root_rows if row.get("score_1_7") is not None]
        per_dim_node_mean = {
            dim: _mean([float(row[f"{dim}_score_1_7"]) for row in node_rows if row.get(f"{dim}_score_1_7") is not None])
            for dim in dimensions
        }
        per_dim_root_mean = {
            dim: _mean([float(row[f"{dim}_score_1_7"]) for row in root_rows if row.get(f"{dim}_score_1_7") is not None])
            for dim in dimensions
        }
        summary = {
            "topology_axis": "size_tokens",
            "leaf_count": None,
            "leaf_size_tokens": int(leaf_size),
            "leaf_count_stats": {
                "min": min((len(tree.get_leaves()) for tree in trees), default=0),
                "max": max((len(tree.get_leaves()) for tree in trees), default=0),
                "mean": float(sum(len(tree.get_leaves()) for tree in trees) / len(trees)) if trees else 0.0,
            },
            "dimension": "combined",
            "dimensions": list(dimensions),
            **tree_bundle_metadata(
                domain="manifesto_rile",
                leaf_unit=str(args.leaf_unit),
                source_kind=str(args.source_kind),
                dimension="combined",
                target_scale="normalized_1_7",
                leaf_policy={
                    "topology_axis": "size_tokens",
                    "leaf_size_tokens": int(leaf_size),
                },
                state_contract=str(args.state_contract),
                reducer_contract=str(args.reducer_contract),
                external_state_producer=(
                    str(args.external_state_producer)
                    if args.external_state_producer
                    else None
                ),
            ),
            "tree_state_source": str(args.tree_state_source),
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
                "node_macro_mean_1_7": _mean(node_macros),
                "root_macro_mean_1_7": _mean(root_macros),
                "node_dimension_mean_1_7": per_dim_node_mean,
                "root_dimension_mean_1_7": per_dim_root_mean,
                "node_min_1_7": float(min(node_macros)) if node_macros else None,
                "node_max_1_7": float(max(node_macros)) if node_macros else None,
            },
            "cache": {"joint_score": score_cache.stats()},
            "artifacts": {
                "labeled_trees": str(labeled_tree_path),
                "teacher_node_rows": str(node_rows_path),
                "joint_score_cache": str(score_cache.path),
                "failures": str(leaf_dir / "failures.json") if failures else None,
                "finetune_bundle": (
                    str(leaf_dir / "treepo_finetune") if finetune_bundle else None
                ),
            },
            "finetune": finetune_bundle,
            "teacher_fg_model": {
                "f_base_url": str(args.scorer_base_url),
                "f_model": str(args.scorer_model),
                "score_input": str(args.score_input),
                "summary_mode": str(args.summary_mode),
                "joint_score_prompt_version": PROMPT_VERSION,
            },
        }
        _write_json(summary_path, summary)
        manifest_entries[manifest_key] = summary
        aggregate_rows.append(
            {
                "topology_axis": "size_tokens",
                "leaf_size_tokens": int(leaf_size),
                "dimension": "combined",
                "dimensions": list(dimensions),
                **tree_bundle_metadata(
                    domain="manifesto_rile",
                    leaf_unit=str(args.leaf_unit),
                    source_kind=str(args.source_kind),
                    dimension="combined",
                    target_scale="normalized_1_7",
                    leaf_policy={
                        "topology_axis": "size_tokens",
                        "leaf_size_tokens": int(leaf_size),
                    },
                    state_contract=str(args.state_contract),
                    reducer_contract=str(args.reducer_contract),
                    external_state_producer=(
                        str(args.external_state_producer)
                        if args.external_state_producer
                        else None
                    ),
                ),
                "tree_state_source": str(args.tree_state_source),
                "tree_count": len(trees),
                "node_count": len(node_rows),
                "failures": len(failures),
                "node_mean_1_7": summary["score_summary"]["node_macro_mean_1_7"],
                "root_mean_1_7": summary["score_summary"]["root_macro_mean_1_7"],
                "labeled_trees": str(labeled_tree_path),
            }
        )

    aggregate_path = output_dir / "aggregate_teacher_metrics.jsonl"
    if aggregate_path.exists():
        aggregate_path.unlink()
    _append_jsonl(aggregate_path, aggregate_rows)
    manifest = {
        "created_at": _now_iso(),
        "dimension": "combined",
        "dimensions": list(dimensions),
        "source_results": str(source_results),
        "source_report": str(source_report) if source_report else None,
        "run_metadata": run_metadata,
        "config": {
            "leaf_size_tokens": list(leaf_sizes),
            "topology_axis": "size_tokens",
            "split_source": str(args.split_source),
            "train_n": int(args.train_n),
            "val_n": int(args.val_n),
            "test_n": int(args.test_n),
            **tree_bundle_metadata(
                domain="manifesto_rile",
                leaf_unit=str(args.leaf_unit),
                source_kind=str(args.source_kind),
                dimension="combined",
                target_scale="normalized_1_7",
                leaf_policy={
                    "topology_axis": "size_tokens",
                    "leaf_size_tokens": list(leaf_sizes),
                },
                state_contract=str(args.state_contract),
                reducer_contract=str(args.reducer_contract),
                external_state_producer=(
                    str(args.external_state_producer)
                    if args.external_state_producer
                    else None
                ),
            ),
            "tree_state_source": str(args.tree_state_source),
            "tree_text_source": str(args.tree_text_source),
            "score_input": str(args.score_input),
            "summary_mode": str(args.summary_mode),
            "label_source": str(args.label_source),
            "joint_score_prompt_version": PROMPT_VERSION,
            "finetune_export": finetune_export_config(args),
        },
        "teacher_fg_model": {
            "f_base_url": str(args.scorer_base_url),
            "f_model": str(args.scorer_model),
        },
        "runs": manifest_entries,
        "artifacts": {
            "split_ids": str(output_dir / "split_ids.json"),
            "aggregate_teacher_metrics": str(aggregate_path),
        },
    }
    manifest["run_manifest"] = run_manifest_metadata(
        run_id="manifesto.teacher_joint_leaf_grid.combined",
        domain="manifesto_rile",
        role="joint_teacher_tree_bundle",
        backend="dspy",
        status="completed",
        tree_bundle=manifest["config"]["tree_bundle_manifest"],
        f_init="joint_teacher_f",
        g_init="teacher_g",
        f_lineage={
            "init": "joint_teacher_f",
            "base_url": str(args.scorer_base_url),
            "model": str(args.scorer_model),
            "dimensions": list(dimensions),
            "prompt_version": PROMPT_VERSION,
        },
        g_lineage={
            "init": "teacher_g",
            "summary_mode": str(args.summary_mode),
            "tree_text_source": str(args.tree_text_source),
        },
        reducer_contract=str(args.reducer_contract),
        schedule="joint_teacher_trace",
        objective=objective_metadata(
            objective_family="manifesto_joint_teacher_first_trace",
            local_law_estimator=LOCAL_LAW_ESTIMATOR_NONE,
            root_share=0.0,
            local_law_component_weights={},
            metadata={
                "dimensions": list(dimensions),
                "prompt_version": PROMPT_VERSION,
                "label_source": str(args.label_source),
                "teacher_trace_component": "teacher_node_trace",
            },
        ),
        optimizer_config={
            "split_source": str(args.split_source),
            "label_source": str(args.label_source),
            "leaf_size_tokens": list(leaf_sizes),
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
        metadata={"runner": "scripts/run_manifesto_teacher_fg_joint_leaf_grid.py"},
    )
    _write_json(output_dir / "manifest.json", manifest)
    LOGGER.info("Wrote joint teacher f* grid manifest to %s", output_dir / "manifest.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
