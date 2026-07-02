from __future__ import annotations

from typing import Any, Iterable, Mapping, Sequence

from treepo_cdx.audit import LawKind, LocalLawAuditRow
from treepo_cdx.manifest import ArtifactLineage, ManifestRow, RoleTuple, Span
from treepo_cdx.sampling import DocumentSamplingRow


def local_law_row_from_mapping(
    payload: Mapping[str, Any],
    *,
    default_law_kind: str | LawKind = LawKind.C1_LEAF,
) -> LocalLawAuditRow:
    data = dict(payload or {})
    row_id = str(data.get("row_id") or data.get("rowId") or data.get("id") or "")
    if not row_id:
        raise ValueError("local-law mapping requires row_id or id")
    proxy_loss = _required_float(data, "proxy_loss", "proxyLoss", "proxy")
    oracle_loss = _optional_float(data.get("oracle_loss", data.get("oracleLoss", data.get("oracle"))))
    observed = bool(data.get("observed", oracle_loss is not None))
    return LocalLawAuditRow(
        row_id=row_id,
        law_kind=data.get("law_kind") or data.get("lawKind") or default_law_kind,
        proxy_loss=proxy_loss,
        oracle_loss=oracle_loss,
        observed=observed,
        propensity=float(data.get("propensity", data.get("pi", 1.0 if observed else 0.0)) or 0.0),
        effective_propensity=_optional_float(
            data.get("effective_propensity", data.get("effectivePropensity"))
        ),
        node_weight=float(data.get("node_weight", data.get("weight", data.get("lambda", 1.0))) or 0.0),
        depth=int(data.get("depth", 0) or 0),
        metadata=dict(data.get("metadata") or {}),
    )


def local_law_rows_from_mappings(
    rows: Iterable[Mapping[str, Any]],
    *,
    default_law_kind: str | LawKind = LawKind.C1_LEAF,
) -> tuple[LocalLawAuditRow, ...]:
    return tuple(
        local_law_row_from_mapping(row, default_law_kind=default_law_kind)
        for row in rows
    )


def local_law_rows_from_manifest(
    rows: Sequence[ManifestRow],
    *,
    proxy_loss_key: str = "proxy_loss",
    oracle_loss_key: str = "oracle_loss",
    weight_key: str = "node_weight",
    depth_key: str = "depth",
    strict: bool = True,
) -> tuple[LocalLawAuditRow, ...]:
    out: list[LocalLawAuditRow] = []
    for row in rows:
        metadata = dict(row.metadata or {})
        if proxy_loss_key not in metadata:
            if strict:
                raise ValueError(f"manifest row {row.row_id} is missing metadata[{proxy_loss_key!r}]")
            continue
        out.append(
            LocalLawAuditRow(
                row_id=row.row_id,
                law_kind=row.law_kind or LawKind.C1_LEAF,
                proxy_loss=float(metadata[proxy_loss_key]),
                oracle_loss=_optional_float(metadata.get(oracle_loss_key)),
                observed=bool(row.observed),
                propensity=float(row.propensity),
                effective_propensity=row.effective_propensity,
                node_weight=float(metadata.get(weight_key, 1.0)),
                depth=int(metadata.get(depth_key, 0) or 0),
                metadata={
                    "top_level_unit_id": row.top_level_unit_id,
                    "source_unit_id": row.source_unit_id,
                    **metadata,
                },
            )
        )
    return tuple(out)


def manifest_rows_from_document_sampling(
    rows: Sequence[DocumentSamplingRow],
    *,
    artifacts: ArtifactLineage | None = None,
    law_kind: str = "document_sampling",
) -> tuple[ManifestRow, ...]:
    out: list[ManifestRow] = []
    for item in rows:
        out.append(
            ManifestRow(
                row_id=f"document:{item.top_level_unit_id}",
                top_level_unit_id=item.top_level_unit_id,
                fold_id=item.fold_id,
                roles=RoleTuple(chunker="eval", g="eval", oracle="eval"),
                artifacts=artifacts,
                law_kind=law_kind,
                support=Span(0, 0, unit="document"),
                observed=bool(item.observed),
                propensity=float(item.inclusion_probability),
                truth_source="document_truth" if item.truth is not None else "",
                approx_source="document_prediction" if item.prediction is not None else "",
                metadata={
                    "prediction": item.prediction,
                    "predicted_var": item.predicted_var,
                    "truth": item.truth,
                    "split": item.split,
                    **dict(item.metadata or {}),
                },
            )
        )
    return tuple(out)


def _optional_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    return float(value)


def _required_float(data: Mapping[str, Any], *keys: str) -> float:
    for key in keys:
        if key in data and data[key] not in (None, ""):
            return float(data[key])
    raise ValueError(f"mapping requires one of {keys}")


__all__ = [
    "local_law_row_from_mapping",
    "local_law_rows_from_manifest",
    "local_law_rows_from_mappings",
    "manifest_rows_from_document_sampling",
]
