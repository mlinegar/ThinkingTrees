"""Export Manifesto qsentence labeled trees through treepo preference records."""

from __future__ import annotations

import json
from collections.abc import Sequence
from typing import Any, Mapping

from src.ctreepo.manifesto_qsentence_dspy_family import (
    ManifestoQSentenceDSPyFamily,
    ManifestoQSentenceDSPyFamilyConfig,
    _node_lopsidedness,
    _node_target_scores,
    _summary_target,
)
from src.tasks.manifesto.span_targets import COMPACT_TARGET_DIMENSIONS, parse_compact_scores_json
from src.tree.labeled import LabeledNode, LabeledTree
from treepo.state import TaskState, make_unit_id


PreferenceMode = str
DEFAULT_QSENTENCE_FINETUNE_ADAPTERS: tuple[str, ...] = (
    "embedding",
    "trl_sft",
    "trl_dpo",
    "trl_reward",
    "trl_scalar_reward",
    "trl_grpo",
    "dspy_examples",
)
DEFAULT_QSENTENCE_LEARNING_ADAPTERS: tuple[str, ...] = (
    "thinkingtrees_dspy",
)
DEFAULT_MANIFESTO_FINETUNE_ADAPTERS = DEFAULT_QSENTENCE_FINETUNE_ADAPTERS
DEFAULT_MANIFESTO_LEARNING_ADAPTERS: tuple[str, ...] = ()


def build_manifesto_qsentence_preferences(
    trees: Sequence[LabeledTree],
    *,
    mode: PreferenceMode = "scores",
    max_records: int | None = None,
) -> Any:
    """Convert qsentence labeled-tree node targets into ``PreferenceDataset``.

    The input is the existing ThinkingTrees qsentence tree format produced by
    ``build_manifesto_qsentence_dspy_labeled_grid.py``. Each node becomes a
    unit-level preference record for the unified ``g`` state: the context is
    the same prompt used by the DSPy qsentence family, and the preferred or
    high-score candidate is the node's compact CMP target state.
    """

    if mode not in {"scores", "pairwise", "ranked"}:
        raise ValueError("mode must be one of: scores, pairwise, ranked")

    from treepo.methods.preference import PreferenceDataset, PreferenceRecord

    family = _prompt_family()
    dataset = PreferenceDataset()
    limit = None if max_records is None else max(0, int(max_records))
    for tree in trees:
        root_id = _root_node_id(tree)
        parent_ids = _parent_ids_by_node(tree)
        for node in _nodes_by_tree_order(tree):
            target = _summary_target(node, include_identity_targets=False)
            scores = _node_target_scores(node)
            if not target or not scores:
                continue
            prompt = family._g_prompt_for_node(tree, node)
            node_id = str(node.node_id)
            tree_id = str(tree.doc_id)
            dataset.append(
                PreferenceRecord(
                    record_id=f"{tree_id}:{node_id}:qsentence:{mode}",
                    unit_id=make_unit_id(tree_id, node_id),
                    unit_type=_unit_type(node, root_id=root_id),
                    target="g",
                    context=prompt,
                    candidates=_candidates(
                        node=node,
                        target_state=target,
                        scores=scores,
                        prompt=prompt,
                        mode=mode,
                    ),
                    weight=_record_weight(node),
                    propensity=1.0,
                    metadata=_record_metadata(
                        tree=tree,
                        node=node,
                        root_id=root_id,
                        scores=scores,
                        prompt=prompt,
                        mode=mode,
                    ),
                    tree_id=tree_id,
                    doc_id=tree_id,
                    node_id=node_id,
                    level=int(node.level),
                    position=_node_position(tree, node),
                    parent_id=parent_ids.get(node_id),
                    left_child_id=_optional_node_id(node.left_child_id),
                    right_child_id=_optional_node_id(node.right_child_id),
                )
            )
            if limit is not None and len(dataset) >= limit:
                return dataset
    return dataset


def build_manifesto_qsentence_tree_records(
    trees: Sequence[LabeledTree],
) -> tuple[Any, ...]:
    """Convert qsentence labeled trees into treepo ``TreeRecord`` objects."""

    from treepo.tree import TreeNode, TreeRecord

    family = _prompt_family()
    records: list[Any] = []
    for tree in trees:
        root_id = _root_node_id(tree)
        parent_ids = _parent_ids_by_node(tree)
        nodes: list[Any] = []
        for node in _nodes_by_tree_order(tree):
            target = _summary_target(node, include_identity_targets=False)
            scores = _node_target_scores(node)
            prompt = family._g_prompt_for_node(tree, node) if target and scores else ""
            state = (
                _task_state_value(node=node, scores=scores, text=target, source="target_summary")
                if target and scores
                else None
            )
            node_id = str(node.node_id)
            nodes.append(
                TreeNode(
                    node_id=node_id,
                    unit_type=_unit_type(node, root_id=root_id),
                    text=str(node.text or ""),
                    level=int(node.level),
                    position=_node_position(tree, node),
                    parent_id=parent_ids.get(node_id),
                    left_child_id=_optional_node_id(node.left_child_id),
                    right_child_id=_optional_node_id(node.right_child_id),
                    label=float(node.score),
                    state=state,
                    metadata=(
                        _record_metadata(
                            tree=tree,
                            node=node,
                            root_id=root_id,
                            scores=scores,
                            prompt=prompt,
                            mode="tree_record",
                        )
                        if scores
                        else dict(node.metadata or {})
                    ),
                )
            )
        records.append(
            TreeRecord(
                tree_id=str(tree.doc_id),
                doc_id=str(tree.doc_id),
                text=str(tree.document_text or ""),
                root_label=float(tree.document_score),
                nodes=tuple(nodes),
                metadata=dict(tree.metadata or {}),
            )
        )
    return tuple(records)




def build_manifesto_labeled_tree_preferences(
    trees: Sequence[LabeledTree],
    *,
    mode: PreferenceMode = "ranked",
    max_records: int | None = None,
    leaf_unit_type: str = "leaf",
    include_root_f: bool = True,
    include_g_nodes: bool = True,
) -> Any:
    """Convert generic Manifesto labeled trees into root-``f`` and node-``g`` rows.

    This is the non-qsentence path for teacher-f/g bundles: token leaves,
    summary-token leaves, scalar dimension grids, and joint dimension grids.
    """

    if mode not in {"scores", "pairwise", "ranked"}:
        raise ValueError("mode must be one of: scores, pairwise, ranked")

    from treepo.methods.preference import PreferenceDataset, PreferenceRecord

    dataset = PreferenceDataset()
    limit = None if max_records is None else max(0, int(max_records))
    for tree in trees:
        root_id = _root_node_id(tree)
        parent_ids = _parent_ids_by_node(tree)
        for node in _nodes_by_tree_order(tree):
            unit_type = _unit_type_generic(node, root_id=root_id, leaf_unit_type=leaf_unit_type)
            scores = _manifesto_node_scores(node, tree=tree)
            target = _summary_target(node, include_identity_targets=True) or str(node.text or "")
            if not target and not scores:
                continue
            node_id = str(node.node_id)
            tree_id = str(tree.doc_id)
            if include_g_nodes:
                dataset.append(
                    PreferenceRecord(
                        record_id=f"{tree_id}:{node_id}:manifesto:g:{mode}",
                        unit_id=make_unit_id(tree_id, node_id),
                        unit_type=unit_type,
                        target="g",
                        context=_manifesto_g_context(tree, node),
                        candidates=_manifesto_candidates(
                            node=node,
                            scores=scores,
                            target_text=target,
                            mode=mode,
                        ),
                        weight=_record_weight(node),
                        propensity=1.0,
                        metadata=_manifesto_record_metadata(
                            tree=tree,
                            node=node,
                            root_id=root_id,
                            unit_type=unit_type,
                            scores=scores,
                            mode=mode,
                            target="g",
                        ),
                        tree_id=tree_id,
                        doc_id=tree_id,
                        node_id=node_id,
                        level=int(node.level),
                        position=_node_position(tree, node),
                        parent_id=parent_ids.get(node_id),
                        left_child_id=_optional_node_id(node.left_child_id),
                        right_child_id=_optional_node_id(node.right_child_id),
                    )
                )
                if limit is not None and len(dataset) >= limit:
                    return dataset
            if include_root_f and unit_type == "root":
                dataset.append(
                    PreferenceRecord(
                        record_id=f"{tree_id}:{node_id}:manifesto:f:{mode}",
                        unit_id=make_unit_id(tree_id, f"{node_id}:f"),
                        unit_type="root",
                        target="f",
                        context=_manifesto_f_context(tree, node),
                        candidates=_manifesto_candidates(
                            node=node,
                            scores=scores,
                            target_text=target,
                            mode=mode,
                        ),
                        weight=_record_weight(node),
                        propensity=1.0,
                        metadata=_manifesto_record_metadata(
                            tree=tree,
                            node=node,
                            root_id=root_id,
                            unit_type="root",
                            scores=scores,
                            mode=mode,
                            target="f",
                        ),
                        tree_id=tree_id,
                        doc_id=tree_id,
                        node_id=node_id,
                        level=int(node.level),
                        position=_node_position(tree, node),
                        parent_id=parent_ids.get(node_id),
                        left_child_id=_optional_node_id(node.left_child_id),
                        right_child_id=_optional_node_id(node.right_child_id),
                    )
                )
                if limit is not None and len(dataset) >= limit:
                    return dataset
    return dataset


def build_manifesto_labeled_tree_records(
    trees: Sequence[LabeledTree],
    *,
    leaf_unit_type: str = "leaf",
) -> tuple[Any, ...]:
    """Convert generic Manifesto labeled trees into treepo ``TreeRecord`` rows."""

    from treepo.tree import TreeNode, TreeRecord

    records: list[Any] = []
    for tree in trees:
        root_id = _root_node_id(tree)
        parent_ids = _parent_ids_by_node(tree)
        nodes: list[Any] = []
        for node in _nodes_by_tree_order(tree):
            scores = _manifesto_node_scores(node, tree=tree)
            target = _summary_target(node, include_identity_targets=True) or str(node.text or "")
            unit_type = _unit_type_generic(node, root_id=root_id, leaf_unit_type=leaf_unit_type)
            node_id = str(node.node_id)
            nodes.append(
                TreeNode(
                    node_id=node_id,
                    unit_type=unit_type,
                    text=str(node.text or ""),
                    level=int(node.level),
                    position=_node_position(tree, node),
                    parent_id=parent_ids.get(node_id),
                    left_child_id=_optional_node_id(node.left_child_id),
                    right_child_id=_optional_node_id(node.right_child_id),
                    label=float(node.score),
                    state=(
                        _manifesto_task_state_value(
                            node=node,
                            scores=scores,
                            text=target,
                            source="labeled_tree_target",
                        )
                        if target or scores
                        else None
                    ),
                    metadata=_manifesto_record_metadata(
                        tree=tree,
                        node=node,
                        root_id=root_id,
                        unit_type=unit_type,
                        scores=scores,
                        mode="tree_record",
                        target="g" if unit_type != "root" else "both",
                    ),
                )
            )
        records.append(
            TreeRecord(
                tree_id=str(tree.doc_id),
                doc_id=str(tree.doc_id),
                text=str(tree.document_text or ""),
                root_label=float(tree.document_score),
                nodes=tuple(nodes),
                metadata=dict(tree.metadata or {}),
            )
        )
    return tuple(records)


def export_manifesto_qsentence_finetune_adapters(
    preferences: Any,
    output_dir: Any,
    *,
    adapters: Sequence[str] = DEFAULT_QSENTENCE_FINETUNE_ADAPTERS,
    learning_adapters: Sequence[str] = DEFAULT_QSENTENCE_LEARNING_ADAPTERS,
    save_hf: bool = False,
) -> dict[str, Any]:
    """Export qsentence preferences through fine-tuning/trainer adapters.

    The returned artifacts are intentionally prepare-only. For DSPy learning,
    the ``thinkingtrees_dspy`` dry-run names the existing family runtime while
    writing the same qsentence rows that the runtime can consume.
    """

    from treepo.finetune import export_for_adapter
    from src.training.finetune_adapters import train_finetune_adapter

    out_dir = _path_value(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    adapter_results: dict[str, Any] = {}
    for name in _adapter_names(adapters):
        adapter_results[name] = export_for_adapter(
            name,
            preferences,
            out_dir / "exports" / name,
            save_hf=save_hf,
        )

    learning_results: dict[str, Any] = {}
    for name in _adapter_names(learning_adapters):
        learning_results[name] = train_finetune_adapter(
            name,
            preferences,
            out_dir / "learning" / name,
            dry_run=True,
            save_hf=save_hf,
        )

    return {
        "adapters": adapter_results,
        "learning_adapters": learning_results,
        "summary": {
            "n_adapters": len(adapter_results),
            "n_learning_adapters": len(learning_results),
            "adapter_names": sorted(adapter_results),
            "learning_adapter_names": sorted(learning_results),
        },
    }




def export_manifesto_qsentence_finetune_bundle(
    trees: Sequence[LabeledTree],
    output_dir: Any,
    *,
    mode: PreferenceMode = "ranked",
    max_records: int | None = None,
    adapters: Sequence[str] = DEFAULT_QSENTENCE_FINETUNE_ADAPTERS,
    learning_adapters: Sequence[str] = DEFAULT_QSENTENCE_LEARNING_ADAPTERS,
    save_hf: bool = False,
) -> dict[str, Any]:
    """Write qsentence tree records, preferences, and adapter exports together."""

    from treepo.methods.preference import export_preference_records
    from treepo.tree import write_tree_records_jsonl

    out_dir = _path_value(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tree_records_path = write_tree_records_jsonl(
        out_dir / "tree_records.jsonl",
        build_manifesto_qsentence_tree_records(trees),
    )
    preferences = build_manifesto_qsentence_preferences(
        trees,
        mode=mode,
        max_records=max_records,
    )
    preference_artifacts = export_preference_records(
        preferences,
        out_dir / "preferences",
        save_hf=save_hf,
    )
    finetune_adapters = export_manifesto_qsentence_finetune_adapters(
        preferences,
        out_dir / "finetune_adapters",
        adapters=adapters,
        learning_adapters=learning_adapters,
        save_hf=save_hf,
    )
    return _bundle_result(
        tree_records_path=tree_records_path,
        preferences=preferences,
        preference_artifacts=preference_artifacts,
        finetune_adapters=finetune_adapters,
        mode=mode,
        bundle_kind="manifesto_qsentence",
        n_trees=len(trees),
    )


def export_manifesto_labeled_tree_finetune_bundle(
    trees: Sequence[LabeledTree],
    output_dir: Any,
    *,
    mode: PreferenceMode = "ranked",
    max_records: int | None = None,
    leaf_unit_type: str = "leaf",
    adapters: Sequence[str] = DEFAULT_MANIFESTO_FINETUNE_ADAPTERS,
    learning_adapters: Sequence[str] = DEFAULT_MANIFESTO_LEARNING_ADAPTERS,
    save_hf: bool = False,
) -> dict[str, Any]:
    """Write generic Manifesto labeled-tree preference and fine-tune artifacts."""

    from treepo.methods.preference import export_preference_records
    from treepo.tree import write_tree_records_jsonl

    out_dir = _path_value(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tree_records_path = write_tree_records_jsonl(
        out_dir / "tree_records.jsonl",
        build_manifesto_labeled_tree_records(trees, leaf_unit_type=leaf_unit_type),
    )
    preferences = build_manifesto_labeled_tree_preferences(
        trees,
        mode=mode,
        max_records=max_records,
        leaf_unit_type=leaf_unit_type,
    )
    preference_artifacts = export_preference_records(
        preferences,
        out_dir / "preferences",
        save_hf=save_hf,
    )
    finetune_adapters = export_manifesto_qsentence_finetune_adapters(
        preferences,
        out_dir / "finetune_adapters",
        adapters=adapters,
        learning_adapters=learning_adapters,
        save_hf=save_hf,
    )
    return _bundle_result(
        tree_records_path=tree_records_path,
        preferences=preferences,
        preference_artifacts=preference_artifacts,
        finetune_adapters=finetune_adapters,
        mode=mode,
        bundle_kind="manifesto_labeled_tree",
        n_trees=len(trees),
    )


def _prompt_family() -> ManifestoQSentenceDSPyFamily:
    return ManifestoQSentenceDSPyFamily(
        config=ManifestoQSentenceDSPyFamilyConfig(
            leaf_size_tokens=128,
            lm_context_window_tokens=4096,
            max_completion_tokens=256,
            prompt_template_overhead_tokens=512,
        )
    )


def _adapter_names(values: Sequence[str]) -> tuple[str, ...]:
    names = tuple(str(value).strip() for value in values if str(value).strip())
    if len(set(names)) != len(names):
        raise ValueError(f"duplicate fine-tune adapter names: {names!r}")
    return names


def _path_value(value: Any):
    from pathlib import Path

    return Path(value)




def _bundle_result(
    *,
    tree_records_path: Any,
    preferences: Any,
    preference_artifacts: Mapping[str, Any],
    finetune_adapters: Mapping[str, Any],
    mode: str,
    bundle_kind: str,
    n_trees: int,
) -> dict[str, Any]:
    files = dict(preference_artifacts.get("files") or {})
    files["tree_records"] = str(tree_records_path)
    return {
        "bundle_kind": bundle_kind,
        "mode": str(mode),
        "counts": dict(preference_artifacts.get("counts") or {}),
        "summary": {
            "n_trees": int(n_trees),
            "n_units": len(preferences),
            "n_candidates": len(getattr(preferences, "candidates", ()) or ()),
            "finetune": dict(finetune_adapters.get("summary") or {}),
        },
        "files": files,
        "preferences": preference_artifacts,
        "finetune_adapters": finetune_adapters,
    }


def _nodes_by_tree_order(tree: LabeledTree) -> list[LabeledNode]:
    nodes: list[LabeledNode] = []
    seen: set[str] = set()
    for level_ids in list(getattr(tree, "levels", None) or []):
        for node_id in level_ids:
            node = tree.get_node(str(node_id))
            if node is not None and str(node.node_id) not in seen:
                nodes.append(node)
                seen.add(str(node.node_id))
    for node in sorted(
        list((getattr(tree, "nodes", None) or {}).values()),
        key=lambda item: (int(getattr(item, "level", 0)), str(getattr(item, "node_id", ""))),
    ):
        if str(node.node_id) not in seen:
            nodes.append(node)
            seen.add(str(node.node_id))
    return nodes


def _root_node_id(tree: LabeledTree) -> str | None:
    for level_ids in reversed(list(getattr(tree, "levels", None) or [])):
        for node_id in reversed(list(level_ids)):
            node = tree.get_node(str(node_id))
            if node is not None:
                return str(node.node_id)
    return None


def _parent_ids_by_node(tree: LabeledTree) -> dict[str, str]:
    parents: dict[str, str] = {}
    for node in list((getattr(tree, "nodes", None) or {}).values()):
        parent_id = str(getattr(node, "node_id", ""))
        if not parent_id:
            continue
        for child_id in (getattr(node, "left_child_id", None), getattr(node, "right_child_id", None)):
            child = _optional_node_id(child_id)
            if child is not None:
                parents.setdefault(child, parent_id)
    return parents


def _node_position(tree: LabeledTree, node: LabeledNode) -> int | None:
    level = int(node.level)
    levels = list(getattr(tree, "levels", None) or [])
    if level < 0 or level >= len(levels):
        return None
    node_id = str(node.node_id)
    for idx, candidate_id in enumerate(levels[level]):
        if str(candidate_id) == node_id:
            return int(idx)
    return None


def _unit_type(node: LabeledNode, *, root_id: str | None) -> str:
    if root_id is not None and str(node.node_id) == str(root_id):
        return "root"
    return "qsentence" if int(node.level) == 0 else "merge"


def _unit_type_generic(
    node: LabeledNode,
    *,
    root_id: str | None,
    leaf_unit_type: str,
) -> str:
    if root_id is not None and str(node.node_id) == str(root_id):
        return "root"
    return str(leaf_unit_type or "leaf") if int(node.level) == 0 else "merge"


def _optional_node_id(value: Any) -> str | None:
    return None if value is None else str(value)


def _candidates(
    *,
    node: LabeledNode,
    target_state: str,
    scores: Mapping[str, Any],
    prompt: str,
    mode: PreferenceMode,
) -> tuple[Any, ...]:
    from treepo.methods.preference import Candidate

    gold = _task_state_value(node=node, scores=scores, text=target_state, source="target_summary")
    neutral = _neutral_state(node)
    weak = _weak_candidate_state(node=node, prompt=prompt)
    if mode == "scores":
        return (
            Candidate(
                id="gold_cmp_state",
                value=gold,
                score=1.0,
                metadata={"source": "target_summary", "target_scores": dict(scores)},
            ),
            Candidate(
                id="neutral_cmp_state",
                value=neutral,
                score=0.25,
                metadata={"source": "neutral_baseline"},
            ),
        )
    if mode == "pairwise":
        return (
            Candidate(
                id="gold_cmp_state",
                value=gold,
                metadata={"source": "target_summary", "target_scores": dict(scores)},
                preferred=True,
            ),
            Candidate(
                id="weak_state",
                value=weak,
                metadata={"source": "input_copy"},
            ),
        )
    return (
        Candidate(
            id="gold_cmp_state",
            value=gold,
            score=1.0,
            rank=1,
            metadata={"source": "target_summary", "target_scores": dict(scores)},
        ),
        Candidate(
            id="neutral_cmp_state",
            value=neutral,
            score=0.25,
            rank=2,
            metadata={"source": "neutral_baseline"},
        ),
        Candidate(
            id="weak_state",
            value=weak,
            score=0.0,
            rank=3,
            metadata={"source": "input_copy"},
        ),
    )


def _task_state_value(
    *,
    node: LabeledNode,
    scores: Mapping[str, Any],
    text: str,
    source: str,
) -> TaskState:
    metadata = dict(node.metadata or {})
    counts: dict[str, float] = {
        "qsentences": float(metadata.get("total_qsentences") or metadata.get("leaf_qsentences") or 0.0),
        "non_header": float(_node_non_header_count(node) or 0.0),
    }
    for key, value in dict(metadata.get("cmp_counts") or {}).items():
        if _is_number(value):
            counts[f"code:{key}"] = float(value)
    for key, value in dict(metadata.get("domain_counts") or {}).items():
        if _is_number(value):
            counts[f"domain:{key}"] = float(value)
    return TaskState(
        kind="manifesto_policy",
        items=_state_items(node),
        counts=counts,
        measures={
            str(dim): float(value)
            for dim, value in dict(scores).items()
            if _is_number(value)
        },
        text=str(text or ""),
        metadata={
            "source": source,
            "node_id": str(node.node_id),
            "level": int(node.level),
            "target_state_text": str(text or ""),
        },
    )


def _neutral_state(node: LabeledNode) -> TaskState:
    compact = {dim: 0.0 for dim in COMPACT_TARGET_DIMENSIONS}
    compact["rile"] = 0.5
    return TaskState(
        kind="manifesto_policy",
        counts={
            "qsentences": float((node.metadata or {}).get("total_qsentences") or 0.0),
            "non_header": float(_node_non_header_count(node) or 0.0),
        },
        measures=compact,
        text=json.dumps(
            {
                "cmp_state": {"compact_targets": compact},
                "total_non_header": int(_node_non_header_count(node) or 0),
                "note": "neutral CMP state",
            },
            sort_keys=True,
        ),
        metadata={"source": "neutral_baseline", "node_id": str(node.node_id)},
    )


def _weak_candidate_state(*, node: LabeledNode, prompt: str) -> TaskState:
    raw = str(node.text or "").strip()
    text = raw or str(prompt or "No compact CMP state.").strip()
    return TaskState(
        kind="manifesto_policy",
        items=_state_items(node),
        text=text,
        metadata={"source": "input_copy", "node_id": str(node.node_id)},
    )




def _manifesto_candidates(
    *,
    node: LabeledNode,
    scores: Mapping[str, Any],
    target_text: str,
    mode: PreferenceMode,
) -> tuple[Any, ...]:
    from treepo.methods.preference import Candidate

    gold = _manifesto_task_state_value(
        node=node,
        scores=scores,
        text=target_text,
        source="labeled_tree_target",
    )
    neutral = _neutral_manifesto_state(node=node, scores=scores)
    weak = _weak_candidate_state(node=node, prompt=str(node.text or target_text or ""))
    if mode == "scores":
        return (
            Candidate(id="gold_manifesto_state", value=gold, score=1.0, preferred=True),
            Candidate(id="neutral_manifesto_state", value=neutral, score=0.25),
        )
    if mode == "pairwise":
        return (
            Candidate(id="gold_manifesto_state", value=gold, preferred=True),
            Candidate(id="weak_state", value=weak),
        )
    return (
        Candidate(id="gold_manifesto_state", value=gold, score=1.0, rank=1),
        Candidate(id="neutral_manifesto_state", value=neutral, score=0.25, rank=2),
        Candidate(id="weak_state", value=weak, score=0.0, rank=3),
    )


def _manifesto_task_state_value(
    *,
    node: LabeledNode,
    scores: Mapping[str, Any],
    text: str,
    source: str,
) -> TaskState:
    state = _task_state_value(node=node, scores=scores, text=text, source=source)
    metadata = dict(state.metadata or {})
    metadata["raw_score"] = float(node.score)
    metadata["raw_dimension_scores"] = dict(getattr(node, "dimension_scores", None) or {})
    return TaskState(
        kind=state.kind,
        items=state.items,
        counts=state.counts,
        measures=state.measures,
        text=state.text,
        metadata=metadata,
    )


def _neutral_manifesto_state(
    *,
    node: LabeledNode,
    scores: Mapping[str, Any],
) -> TaskState:
    dims = sorted(str(dim) for dim in scores) or ["rile"]
    measures = {dim: 0.5 for dim in dims}
    return TaskState(
        kind="manifesto_policy",
        counts={
            "qsentences": float((node.metadata or {}).get("total_qsentences") or 0.0),
            "non_header": float(_node_non_header_count(node) or 0.0),
        },
        measures=measures,
        text=json.dumps(
            {
                "manifesto_state": {"normalized_scores": measures},
                "note": "neutral Manifesto state",
            },
            sort_keys=True,
        ),
        metadata={"source": "neutral_baseline", "node_id": str(node.node_id)},
    )


def _manifesto_g_context(tree: LabeledTree, node: LabeledNode) -> str:
    if int(node.level) == 0:
        return (
            "Encode this Manifesto policy unit as a compact policy state.\n\n"
            f"DOC_ID: {tree.doc_id}\nNODE_ID: {node.node_id}\n\nTEXT:\n{str(node.text or '')}"
        )
    left = tree.get_node(str(node.left_child_id)) if node.left_child_id else None
    right = tree.get_node(str(node.right_child_id)) if node.right_child_id else None
    left_state = _summary_target(left, include_identity_targets=True) if left is not None else ""
    right_state = _summary_target(right, include_identity_targets=True) if right is not None else ""
    if right is None or (left is not None and right.node_id == left.node_id):
        return (
            "Promote this Manifesto child policy state as the parent state.\n\n"
            f"DOC_ID: {tree.doc_id}\nNODE_ID: {node.node_id}\n\nCHILD_STATE:\n{left_state}"
        )
    return (
        "Merge these Manifesto child policy states into one compact parent state.\n\n"
        f"DOC_ID: {tree.doc_id}\nNODE_ID: {node.node_id}\n\n"
        f"LEFT_STATE:\n{left_state}\n\nRIGHT_STATE:\n{right_state}"
    )


def _manifesto_f_context(tree: LabeledTree, node: LabeledNode) -> str:
    summary = _summary_target(node, include_identity_targets=True) or str(node.text or "")
    return (
        "Read this Manifesto root policy state and produce the document-level "
        "policy score/state.\n\n"
        f"DOC_ID: {tree.doc_id}\nNODE_ID: {node.node_id}\n\nROOT_STATE:\n{summary}"
    )


def _manifesto_node_scores(node: LabeledNode, *, tree: LabeledTree | None = None) -> dict[str, float]:
    out: dict[str, float] = {}
    raw = getattr(node, "dimension_scores", None)
    if isinstance(raw, Mapping):
        for dim, value in raw.items():
            parsed = _parse_manifesto_score_value(value, scale="auto")
            if parsed is not None:
                out[str(dim)] = parsed
    metadata = dict(node.metadata or {})
    for key, scale in (
        ("target_dimension_scores_0_1", "zero_one"),
        ("teacher_dimension_scores_1_7", "one_seven"),
        ("expert_dimension_scores_1_7", "one_seven"),
        ("dimension_scores_1_7", "one_seven"),
        ("dimension_scores", "auto"),
    ):
        raw_meta = metadata.get(key)
        if isinstance(raw_meta, Mapping):
            for dim, value in raw_meta.items():
                parsed = _parse_manifesto_score_value(value, scale=scale)
                if parsed is not None:
                    out.setdefault(str(dim), parsed)
    if not out:
        parsed_summary = parse_compact_scores_json(
            _summary_target(node, include_identity_targets=False) or ""
        )
        if parsed_summary:
            out.update({str(dim): float(value) for dim, value in parsed_summary.items()})
    if not out:
        dim = str(metadata.get("dimension") or ((tree.metadata or {}).get("dimension") if tree else "") or "rile")
        parsed = _parse_manifesto_score_value(getattr(node, "score", None), scale="auto")
        if parsed is not None:
            out[dim] = parsed
    return out


def _parse_manifesto_score_value(value: Any, *, scale: str) -> float | None:
    if not _is_number(value):
        return None
    parsed = float(value)
    if scale == "one_seven" or (scale == "auto" and parsed > 1.0):
        return max(0.0, min(1.0, (parsed - 1.0) / 6.0))
    return max(0.0, min(1.0, parsed))


def _state_items(node: LabeledNode) -> tuple[Mapping[str, Any], ...]:
    if int(node.level) != 0:
        return ()
    return (
        {
            "id": str(node.node_id),
            "text": str(node.text or ""),
            "level": int(node.level),
        },
    )


def _record_weight(node: LabeledNode) -> float:
    mass = _node_non_header_count(node)
    return float(max(1.0, mass if mass is not None else 1.0))


def _node_non_header_count(node: LabeledNode) -> float | None:
    metadata = dict(node.metadata or {})
    for key in ("total_non_header_qsentences", "total_non_header", "n_non_header"):
        value = metadata.get(key)
        if value is None:
            continue
        try:
            parsed = float(value)
        except (TypeError, ValueError):
            continue
        if parsed >= 0:
            return parsed
    return None


def _record_metadata(
    *,
    tree: LabeledTree,
    node: LabeledNode,
    root_id: str | None,
    scores: Mapping[str, Any],
    prompt: str,
    mode: PreferenceMode,
) -> dict[str, Any]:
    node_meta = dict(node.metadata or {})
    split = str((tree.metadata or {}).get("split", "") or "")
    level = int(node.level)
    return {
        "doc_id": tree.doc_id,
        "source_doc_id": tree.doc_id,
        "node_id": node.node_id,
        "split": split,
        "level": level,
        "is_leaf": level == 0,
        "is_root": root_id is not None and str(node.node_id) == str(root_id),
        "law_type": "qsentence_cmp_state",
        "law_role": "leaf_g" if level == 0 else "merge_g",
        "preference_mode": mode,
        "target_scores": {
            str(dim): float(value)
            for dim, value in dict(scores).items()
            if _is_number(value)
        },
        "total_qsentences": node_meta.get("total_qsentences"),
        "total_non_header_qsentences": node_meta.get("total_non_header_qsentences"),
        "leaf_qsentences": node_meta.get("leaf_qsentences"),
        "qsentence_start_index": node_meta.get("qsentence_start_index"),
        "qsentence_end_index": node_meta.get("qsentence_end_index"),
        "cmp_counts": dict(node_meta.get("cmp_counts") or {}),
        "domain_counts": dict(node_meta.get("domain_counts") or {}),
        "rile_raw": node_meta.get("rile_raw"),
        "rile_norm": node_meta.get("rile_norm"),
        "lopsidedness": _node_lopsidedness(tree, node),
        "c1_raw_text": str(node.text or "") if level == 0 else "",
        "c3a_concat": prompt if level > 0 else "",
        "label_source": (tree.metadata or {}).get("label_source", tree.label_source),
    }




def _manifesto_record_metadata(
    *,
    tree: LabeledTree,
    node: LabeledNode,
    root_id: str | None,
    unit_type: str,
    scores: Mapping[str, Any],
    mode: str,
    target: str,
) -> dict[str, Any]:
    metadata = _record_metadata(
        tree=tree,
        node=node,
        root_id=root_id,
        scores=scores,
        prompt="",
        mode=mode,
    )
    node_meta = dict(node.metadata or {})
    metadata.update(
        {
            "unit_type": unit_type,
            "target": target,
            "law_type": "manifesto_labeled_tree_state",
            "law_role": (
                "root_f"
                if target == "f"
                else ("leaf_g" if int(node.level) == 0 else "merge_g")
            ),
            "raw_score": float(node.score),
            "raw_dimension_scores": dict(getattr(node, "dimension_scores", None) or {}),
            "teacher_dimension_scores_1_7": dict(node_meta.get("teacher_dimension_scores_1_7") or {}),
            "expert_dimension_scores_1_7": dict(node_meta.get("expert_dimension_scores_1_7") or {}),
        }
    )
    return metadata


def _is_number(value: Any) -> bool:
    try:
        float(value)
    except (TypeError, ValueError):
        return False
    return True


__all__ = [
    "DEFAULT_MANIFESTO_FINETUNE_ADAPTERS",
    "DEFAULT_MANIFESTO_LEARNING_ADAPTERS",
    "DEFAULT_QSENTENCE_FINETUNE_ADAPTERS",
    "DEFAULT_QSENTENCE_LEARNING_ADAPTERS",
    "build_manifesto_labeled_tree_preferences",
    "build_manifesto_labeled_tree_records",
    "build_manifesto_qsentence_preferences",
    "build_manifesto_qsentence_tree_records",
    "export_manifesto_labeled_tree_finetune_bundle",
    "export_manifesto_qsentence_finetune_adapters",
    "export_manifesto_qsentence_finetune_bundle",
]
