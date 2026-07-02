"""Central local-law training-row adapters.

Family adapters should construct rows here and delegate objective arithmetic to
``treepo.training.local_law``. This module is intentionally opt-in: it does not
rewire FNO, DSPy, or TRL by itself.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
import random
from typing import Any, Iterable, Mapping, Sequence

from treepo.training.local_law import LocalLawTrainingRow

LAW_KIND_ROOT = "root"
LAW_KIND_LEAF = "leaf_preservation"
LAW_KIND_MERGE = "merge_preservation"
LAW_KIND_IDEMPOTENCE = "on_range_idempotence"

NODE_ROLE_ROOT = "root"
NODE_ROLE_LEAF = "leaf"
NODE_ROLE_INTERNAL = "internal"

SAMPLING_FULL_OBS = "full_obs"
SAMPLING_FIXED_SIZE_UNIFORM = "fixed_size_uniform"
SAMPLING_BERNOULLI = "bernoulli"
SAMPLING_PERSISTENT_MASK = "persistent_mask"


@dataclass(frozen=True)
class LocalLawRowAdapterResult:
    rows: tuple[LocalLawTrainingRow, ...]
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def by_role(self, role: str) -> tuple[LocalLawTrainingRow, ...]:
        requested = str(role)
        return tuple(row for row in self.rows if row.metadata.get("node_role") == requested)


def classify_node_role(
    node: Any,
    *,
    root_id: str | None = None,
    tree_shape: Mapping[str, Any] | None = None,
) -> str:
    metadata = _node_metadata(node)
    node_id = _node_id(node)
    if str(root_id or "") and node_id == str(root_id):
        return NODE_ROLE_ROOT
    if _truthy(metadata.get("is_root")):
        return NODE_ROLE_ROOT
    if bool(getattr(node, "is_root", False)):
        return NODE_ROLE_ROOT
    if isinstance(node, Mapping) and _truthy(node.get("is_root")):
        return NODE_ROLE_ROOT

    children = _node_children(node)
    if not children:
        if _truthy(metadata.get("is_leaf")) or bool(getattr(node, "is_leaf", False)):
            return NODE_ROLE_LEAF
        if isinstance(node, Mapping) and _truthy(node.get("is_leaf")):
            return NODE_ROLE_LEAF
        if tree_shape and node_id in set(str(x) for x in tree_shape.get("leaf_ids", ())):
            return NODE_ROLE_LEAF
        return NODE_ROLE_LEAF
    return NODE_ROLE_INTERNAL


def full_binary_tree_population_size(leaf_count: int) -> int:
    leaves = int(leaf_count)
    if leaves <= 0:
        raise ValueError("leaf_count must be positive")
    return int(2 * leaves - 1)


def build_local_law_rows(
    source: Any,
    *,
    doc_id: str = "",
    law_kind: str = LAW_KIND_MERGE,
    global_axiom: str = "",
    state_kind: str = "",
    law_channel: str = "",
    sampling_policy: str = SAMPLING_FULL_OBS,
    sample_rate: float = 1.0,
    sample_size: int | None = None,
    persistent_mask: Mapping[str, bool] | Sequence[bool] | None = None,
    seed: int = 0,
    drop_duplicate_root: bool = True,
    default_proxy_loss: float | None = None,
    default_node_weight: float = 1.0,
    metadata: Mapping[str, Any] | None = None,
) -> LocalLawRowAdapterResult:
    nodes = _ordered_nodes(source)
    if not nodes:
        return LocalLawRowAdapterResult(
            rows=tuple(),
            metadata={
                "sampling_policy": str(sampling_policy),
                "row_count": 0,
                "observed_count": 0,
            },
        )

    root_id = _root_id(source, nodes)
    deduped = _drop_duplicate_root_rows(nodes, root_id=root_id) if drop_duplicate_root else list(nodes)
    selected, propensity_by_id = _sampling_decisions(
        deduped,
        sampling_policy=sampling_policy,
        sample_rate=sample_rate,
        sample_size=sample_size,
        persistent_mask=persistent_mask,
        seed=seed,
    )

    base_metadata = dict(metadata or {})
    rows: list[LocalLawTrainingRow] = []
    for node in deduped:
        node_id = _node_id(node)
        node_meta = _node_metadata(node)
        role = classify_node_role(node, root_id=root_id)
        has_oracle = _has_oracle_payload(node, node_meta)
        observed = bool(selected.get(node_id, False) and has_oracle)
        propensity = float(propensity_by_id.get(node_id, 0.0)) if has_oracle else 0.0
        if str(sampling_policy) == SAMPLING_FULL_OBS and has_oracle:
            observed = True
            propensity = 1.0

        proxy_loss = _optional_float(_first_present(node, node_meta, "proxy_loss", "loss_proxy"))
        prediction = _optional_float(_first_present(node, node_meta, "prediction", "pred"))
        proxy_target = _optional_float(_first_present(node, node_meta, "proxy_target", "target"))
        if proxy_loss is None and default_proxy_loss is not None:
            proxy_loss = float(default_proxy_loss)
        if proxy_loss is None and (prediction is None or proxy_target is None):
            proxy_loss = 0.0

        oracle_loss = None
        oracle_target = None
        if observed:
            oracle_loss = _optional_float(_first_present(node, node_meta, "oracle_loss", "loss_oracle"))
            oracle_target = _optional_float(
                _first_present(node, node_meta, "oracle_target", "oracle_score", "oracle_target_score")
            )

        depth = _node_depth(node, root_id=root_id)
        weight = _optional_float(_first_present(node, node_meta, "node_weight", "weight"))
        if weight is None:
            weight = float(default_node_weight)

        row_metadata = {
            **base_metadata,
            **node_meta,
            "doc_id": str(doc_id or node_meta.get("doc_id", "")),
            "node_id": node_id,
            "node_role": role,
            "sampling_policy": str(sampling_policy),
            "sampling_propensity": float(propensity),
            "has_oracle": bool(has_oracle),
        }
        if role == NODE_ROLE_ROOT and str(law_kind) == LAW_KIND_MERGE:
            row_metadata.setdefault("no_double_count", True)

        rows.append(
            LocalLawTrainingRow(
                proxy_loss=proxy_loss,
                oracle_loss=oracle_loss,
                observed=observed,
                propensity=propensity,
                depth=depth,
                node_weight=float(weight),
                metadata=row_metadata,
                row_id=str(_first_present(node, node_meta, "row_id") or f"{doc_id}:{node_id}:{law_kind}"),
                law_kind=str(_first_present(node, node_meta, "law_kind") or law_kind),
                global_axiom=str(_first_present(node, node_meta, "global_axiom") or global_axiom),
                state_kind=str(_first_present(node, node_meta, "state_kind") or state_kind),
                law_channel=str(_first_present(node, node_meta, "law_channel") or law_channel),
                doc_id=str(doc_id or node_meta.get("doc_id", "")),
                node_id=node_id,
                prediction=prediction,
                proxy_target=proxy_target,
                oracle_target=oracle_target,
            )
        )

    leaf_count = sum(1 for row in rows if row.metadata.get("node_role") == NODE_ROLE_LEAF)
    result_metadata = {
        "sampling_policy": str(sampling_policy),
        "sample_rate": float(sample_rate),
        "sample_size": None if sample_size is None else int(sample_size),
        "row_count": int(len(rows)),
        "observed_count": int(sum(1 for row in rows if row.observed)),
        "leaf_count": int(leaf_count),
        "expected_full_binary_rows": (
            full_binary_tree_population_size(leaf_count) if leaf_count > 0 else 0
        ),
        "root_id": str(root_id or ""),
    }
    return LocalLawRowAdapterResult(rows=tuple(rows), metadata=result_metadata)


def _ordered_nodes(source: Any) -> list[Any]:
    if hasattr(source, "traverse_preorder"):
        return list(source.traverse_preorder())
    if isinstance(source, Mapping):
        if "nodes" in source:
            nodes = source.get("nodes")
            if isinstance(nodes, Mapping):
                return list(nodes.values())
            if isinstance(nodes, Sequence) and not isinstance(nodes, (str, bytes, bytearray)):
                return list(nodes)
        if "root" in source:
            return _walk_node(source.get("root"))
        return [source]
    if isinstance(source, Sequence) and not isinstance(source, (str, bytes, bytearray)):
        return list(source)
    return _walk_node(source)


def _walk_node(root: Any) -> list[Any]:
    if root is None:
        return []
    out: list[Any] = []
    stack = [root]
    while stack:
        node = stack.pop()
        out.append(node)
        children = _node_children(node)
        for child in reversed(children):
            stack.append(child)
    return out


def _root_id(source: Any, nodes: Sequence[Any]) -> str:
    root = getattr(source, "root", None)
    if root is not None:
        return _node_id(root)
    if isinstance(source, Mapping):
        raw = source.get("root_id")
        if raw:
            return str(raw)
    for node in nodes:
        if classify_node_role(node) == NODE_ROLE_ROOT:
            return _node_id(node)
    return _node_id(nodes[0])


def _drop_duplicate_root_rows(nodes: Sequence[Any], *, root_id: str) -> list[Any]:
    out: list[Any] = []
    seen_root = False
    for node in nodes:
        node_id = _node_id(node)
        metadata = _node_metadata(node)
        is_root = node_id == str(root_id) or classify_node_role(node, root_id=root_id) == NODE_ROLE_ROOT
        duplicate_root = bool(is_root and seen_root)
        cumulative_root = is_root and (
            _truthy(metadata.get("is_cumulative_merge"))
            or _truthy(metadata.get("is_cumulative_root"))
            or str(metadata.get("row_kind", "")).lower() in {"cumulative_merge", "cumulative_root"}
        )
        if duplicate_root or cumulative_root:
            continue
        out.append(node)
        if is_root:
            seen_root = True
    return out


def _sampling_decisions(
    nodes: Sequence[Any],
    *,
    sampling_policy: str,
    sample_rate: float,
    sample_size: int | None,
    persistent_mask: Mapping[str, bool] | Sequence[bool] | None,
    seed: int,
) -> tuple[dict[str, bool], dict[str, float]]:
    policy = str(sampling_policy)
    node_ids = [_node_id(node) for node in nodes]
    n = len(node_ids)
    if n == 0:
        return {}, {}
    if policy == SAMPLING_FULL_OBS:
        return {node_id: True for node_id in node_ids}, {node_id: 1.0 for node_id in node_ids}

    rate = _probability(sample_rate, name="sample_rate")
    if policy == SAMPLING_FIXED_SIZE_UNIFORM:
        q = int(sample_size) if sample_size is not None else int(math.ceil(rate * n))
        q = max(0, min(n, q))
        rng = random.Random(int(seed))
        selected_ids = set(rng.sample(node_ids, q)) if q > 0 else set()
        propensity = float(q / n) if n > 0 else 0.0
        return (
            {node_id: node_id in selected_ids for node_id in node_ids},
            {node_id: propensity for node_id in node_ids},
        )

    if policy == SAMPLING_BERNOULLI:
        rng = random.Random(int(seed))
        return (
            {node_id: bool(rng.random() < rate) for node_id in node_ids},
            {node_id: rate for node_id in node_ids},
        )

    if policy == SAMPLING_PERSISTENT_MASK:
        if persistent_mask is None:
            raise ValueError("persistent_mask sampling requires persistent_mask")
        selected: dict[str, bool] = {}
        if isinstance(persistent_mask, Mapping):
            selected = {node_id: bool(persistent_mask.get(node_id, False)) for node_id in node_ids}
        else:
            mask = list(persistent_mask)
            if len(mask) != n:
                raise ValueError("persistent_mask length must match node population")
            selected = {node_id: bool(mask[idx]) for idx, node_id in enumerate(node_ids)}
        return selected, {node_id: rate for node_id in node_ids}

    raise ValueError(f"unsupported sampling_policy={sampling_policy!r}")


def _node_id(node: Any) -> str:
    if isinstance(node, Mapping):
        return str(node.get("node_id") or node.get("id") or node.get("row_id") or "")
    return str(getattr(node, "node_id", "") or getattr(node, "id", "") or "")


def _node_metadata(node: Any) -> dict[str, Any]:
    if isinstance(node, Mapping):
        return dict(node.get("metadata", {}) or {})
    return dict(getattr(node, "metadata", {}) or {})


def _node_children(node: Any) -> list[Any]:
    if isinstance(node, Mapping):
        children = node.get("children")
        if children is not None:
            return list(children)
        out = []
        if node.get("left_child") is not None:
            out.append(node.get("left_child"))
        if node.get("right_child") is not None:
            out.append(node.get("right_child"))
        return out
    children = getattr(node, "children", None)
    if children is not None:
        return list(children)
    out = []
    if getattr(node, "left_child", None) is not None:
        out.append(getattr(node, "left_child"))
    if getattr(node, "right_child", None) is not None:
        out.append(getattr(node, "right_child"))
    return out


def _node_depth(node: Any, *, root_id: str) -> int:
    metadata = _node_metadata(node)
    if "depth" in metadata:
        parsed = _optional_float(metadata.get("depth"))
        if parsed is not None:
            return max(0, int(parsed))
    if isinstance(node, Mapping) and "depth" in node:
        parsed = _optional_float(node.get("depth"))
        if parsed is not None:
            return max(0, int(parsed))
    depth = 0
    current = getattr(node, "parent", None)
    while current is not None:
        depth += 1
        if _node_id(current) == str(root_id):
            break
        current = getattr(current, "parent", None)
    return int(depth)


def _has_oracle_payload(node: Any, metadata: Mapping[str, Any]) -> bool:
    return _optional_float(_first_present(node, metadata, "oracle_loss", "loss_oracle")) is not None or (
        _optional_float(
            _first_present(node, metadata, "oracle_target", "oracle_score", "oracle_target_score")
        )
        is not None
    )


def _first_present(node: Any, metadata: Mapping[str, Any], *keys: str) -> Any:
    for key in keys:
        if isinstance(node, Mapping) and key in node:
            return node.get(key)
        if key in metadata:
            return metadata.get(key)
        if not isinstance(node, Mapping) and hasattr(node, key):
            return getattr(node, key)
    return None


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(out):
        return None
    return out


def _probability(value: Any, *, name: str) -> float:
    out = _optional_float(value)
    if out is None or out < 0.0 or out > 1.0:
        raise ValueError(f"{name} must be in [0, 1], got {value!r}")
    return float(out)


def _truthy(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "observed", "sampled", "root", "leaf"}
    return bool(value)


__all__ = [
    "LAW_KIND_IDEMPOTENCE",
    "LAW_KIND_LEAF",
    "LAW_KIND_MERGE",
    "LAW_KIND_ROOT",
    "LocalLawRowAdapterResult",
    "NODE_ROLE_INTERNAL",
    "NODE_ROLE_LEAF",
    "NODE_ROLE_ROOT",
    "SAMPLING_BERNOULLI",
    "SAMPLING_FIXED_SIZE_UNIFORM",
    "SAMPLING_FULL_OBS",
    "SAMPLING_PERSISTENT_MASK",
    "build_local_law_rows",
    "classify_node_role",
    "full_binary_tree_population_size",
]
