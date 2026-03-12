"""General package-level abstractions for compositional summary learning.

This module is intentionally broader than C-TreePO. It formalizes a common
problem shape:

1. learn a compositional summary object over documents or spans;
2. supervise it either from whole-document labels, sampled substructure labels,
   or both; and
3. optionally attach theorem-backed reduction assumptions to the operator.

The key distinction in the supervision layer is not "LLM vs sketch". It is:

- full-document labels, where the target is observed directly for the whole
  object; and
- randomly sampled substructure labels, where the target is observed on a
  sampled subset of leaves / spans / internal nodes and generally requires
  propensity-aware estimation if we want unbiased risk accounting.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, is_dataclass
from enum import Enum
from typing import Any, Dict, Generic, Optional, Protocol, Sequence, Tuple, TypeVar

from src.core.ops_checks import LawKind, OperatorCapabilityReport
from src.core.provenance import (
    ORACLE_SOURCE,
    TruthLabelSource,
    normalize_truth_label_source,
)
from src.tree.compositional_operator import OperatorAssumptionBundle
from src.tree.theorem_backing import TheoremAssumptionSpec

DocT = TypeVar("DocT")
UnitT = TypeVar("UnitT")
LabelT = TypeVar("LabelT")


def _serialize(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if is_dataclass(value):
        return _serialize(asdict(value))
    if isinstance(value, dict):
        return {str(k): _serialize(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_serialize(v) for v in value]
    return value


class SupervisionChannelKind(str, Enum):
    """How supervision reaches the learner."""

    FULL_DOCUMENT = "full_document"
    SAMPLED_SUBSTRUCTURE = "sampled_substructure"


class LabelAcquisitionMode(str, Enum):
    """Whether labels are complete or acquired through random sampling."""

    COMPLETE_LABELS = "complete_labels"
    RANDOM_SAMPLED_LABELS = "random_sampled_labels"


class SupervisionDeliveryMode(str, Enum):
    """Whether labels are pre-logged or acquired by querying an oracle online."""

    OFFLINE_LOGGED = "offline_logged"
    ONLINE_ORACLE_QUERY = "online_oracle_query"


@dataclass(frozen=True)
class OracleQueryPolicySpec:
    """How an online supervision channel calls and logs oracle labels."""

    name: str
    query_unit_name: str
    selection_strategy: str
    adaptive: bool = False
    budget: Dict[str, Any] = field(default_factory=dict)
    propensity_field_name: str = "propensity"
    logs_realized_propensities: bool = False
    supports_ipw_estimation: bool = False
    notes: Tuple[str, ...] = field(default_factory=tuple)

    def to_dict(self) -> Dict[str, Any]:
        return _serialize(
            {
                "name": self.name,
                "query_unit_name": self.query_unit_name,
                "selection_strategy": self.selection_strategy,
                "adaptive": bool(self.adaptive),
                "budget": self.budget,
                "propensity_field_name": self.propensity_field_name,
                "logs_realized_propensities": bool(self.logs_realized_propensities),
                "supports_ipw_estimation": bool(self.supports_ipw_estimation),
                "notes": tuple(self.notes),
            }
        )


class FullDocumentLabelSource(Protocol, Generic[DocT, LabelT]):
    """Callable source of full-document labels."""

    def label_document(self, document: DocT, **kwargs: Any) -> LabelT:
        ...


class SampledSubstructureLabelSource(Protocol, Generic[DocT, UnitT, LabelT]):
    """Source of randomly sampled substructure labels."""

    def sample_units(
        self,
        document: DocT,
        *,
        budget: Optional[int] = None,
        rng: Optional[Any] = None,
        **kwargs: Any,
    ) -> Sequence[UnitT]:
        ...

    def label_unit(
        self,
        document: DocT,
        unit: UnitT,
        **kwargs: Any,
    ) -> LabelT:
        ...


@dataclass(frozen=True)
class FullDocumentLabelObservation(Generic[LabelT]):
    """Observed whole-document label."""

    document_id: str
    label: LabelT
    truth_label_source: TruthLabelSource = ORACLE_SOURCE
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return _serialize(
            {
                "document_id": self.document_id,
                "label": self.label,
                "truth_label_source": normalize_truth_label_source(self.truth_label_source),
                "metadata": self.metadata,
            }
        )


@dataclass(frozen=True)
class SampledSubstructureLabelObservation(Generic[UnitT, LabelT]):
    """Observed label on a sampled leaf/span/internal node."""

    document_id: str
    unit: UnitT
    label: LabelT
    propensity: Optional[float] = None
    truth_label_source: TruthLabelSource = ORACLE_SOURCE
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def is_propensity_annotated(self) -> bool:
        return self.propensity is not None

    def to_dict(self) -> Dict[str, Any]:
        return _serialize(
            {
                "document_id": self.document_id,
                "unit": self.unit,
                "label": self.label,
                "propensity": self.propensity,
                "truth_label_source": normalize_truth_label_source(self.truth_label_source),
                "metadata": self.metadata,
            }
        )


@dataclass(frozen=True)
class SupervisionChannelSpec:
    """Formal description of one supervision channel in a summary-learning problem."""

    name: str
    kind: SupervisionChannelKind
    target_name: str
    active: bool = True
    label_source: TruthLabelSource = ORACLE_SOURCE
    acquisition_mode: LabelAcquisitionMode = LabelAcquisitionMode.COMPLETE_LABELS
    delivery_mode: SupervisionDeliveryMode = SupervisionDeliveryMode.OFFLINE_LOGGED
    requires_propensity_logging: bool = False
    supports_unbiased_risk: bool = True
    query_policy: Optional[OracleQueryPolicySpec] = None
    targeted_laws: Tuple[LawKind, ...] = field(default_factory=tuple)
    notes: Tuple[str, ...] = field(default_factory=tuple)

    def to_dict(self) -> Dict[str, Any]:
        return _serialize(
            {
                "name": self.name,
                "kind": self.kind,
                "target_name": self.target_name,
                "active": bool(self.active),
                "label_source": normalize_truth_label_source(self.label_source),
                "acquisition_mode": self.acquisition_mode,
                "delivery_mode": self.delivery_mode,
                "requires_propensity_logging": bool(self.requires_propensity_logging),
                "supports_unbiased_risk": bool(self.supports_unbiased_risk),
                "query_policy": (
                    self.query_policy.to_dict() if self.query_policy is not None else None
                ),
                "targeted_laws": tuple(self.targeted_laws),
                "notes": tuple(self.notes),
            }
        )


@dataclass(frozen=True)
class CompositionalLearningProblemSpec:
    """General problem specification for theorem-aware summary learning."""

    name: str
    document_type_name: str
    theorem_domain_name: str
    operator_name: str
    supervision_channels: Tuple[SupervisionChannelSpec, ...]
    theorem_assumptions: Optional[TheoremAssumptionSpec] = None
    operator_assumptions: Optional[OperatorAssumptionBundle] = None
    operator_capabilities: Optional[OperatorCapabilityReport] = None
    notes: Tuple[str, ...] = field(default_factory=tuple)

    @property
    def uses_full_document_labels(self) -> bool:
        return any(
            bool(channel.active)
            and channel.kind == SupervisionChannelKind.FULL_DOCUMENT
            for channel in self.supervision_channels
        )

    @property
    def uses_sampled_substructure_labels(self) -> bool:
        return any(
            bool(channel.active)
            and channel.kind == SupervisionChannelKind.SAMPLED_SUBSTRUCTURE
            for channel in self.supervision_channels
        )

    @property
    def requires_propensity_logging(self) -> bool:
        return any(
            bool(channel.active)
            and bool(channel.requires_propensity_logging)
            for channel in self.supervision_channels
        )

    @property
    def uses_online_oracle_queries(self) -> bool:
        return any(
            bool(channel.active)
            and channel.delivery_mode == SupervisionDeliveryMode.ONLINE_ORACLE_QUERY
            for channel in self.supervision_channels
        )

    @property
    def supports_theorem_backing(self) -> bool:
        if self.theorem_assumptions is not None or self.operator_assumptions is not None:
            return True
        if self.operator_capabilities is not None:
            return bool(
                self.operator_capabilities.theorem_domain_decode_available
                or self.operator_capabilities.evidence_status.value != "proxy_only"
            )
        return False

    @property
    def supports_nested_summaries(self) -> bool:
        if self.theorem_assumptions is not None:
            return bool(
                self.theorem_assumptions.operator_assumptions.tree_nesting_supported
            )
        if self.operator_assumptions is not None:
            return bool(self.operator_assumptions.tree_nesting_supported)
        if self.operator_capabilities is not None:
            return bool(self.operator_capabilities.tree_nesting_supported)
        return False

    def capability_report(self) -> Optional[OperatorCapabilityReport]:
        if self.theorem_assumptions is not None:
            return self.theorem_assumptions.capability_report(self.operator_name)
        if self.operator_assumptions is not None:
            return self.operator_assumptions.capability_report(self.operator_name)
        if self.operator_capabilities is not None:
            return self.operator_capabilities
        return None

    def problem_statement(self) -> str:
        supervision_parts = []
        if self.uses_full_document_labels:
            supervision_parts.append("full-document labels")
        if self.uses_sampled_substructure_labels:
            supervision_parts.append("sampled substructure labels")
        supervision = " + ".join(supervision_parts) if supervision_parts else "no declared labels"
        theorem = "theorem-backed" if self.supports_theorem_backing else "proxy-only"
        return (
            f"{self.name}: learn {self.theorem_domain_name} summaries for "
            f"{self.document_type_name} using {supervision} under a {theorem} "
            f"operator surface."
        )

    def to_dict(self) -> Dict[str, Any]:
        return _serialize(
            {
                "name": self.name,
                "document_type_name": self.document_type_name,
                "theorem_domain_name": self.theorem_domain_name,
                "operator_name": self.operator_name,
                "uses_full_document_labels": self.uses_full_document_labels,
                "uses_sampled_substructure_labels": self.uses_sampled_substructure_labels,
                "requires_propensity_logging": self.requires_propensity_logging,
                "uses_online_oracle_queries": self.uses_online_oracle_queries,
                "supports_theorem_backing": self.supports_theorem_backing,
                "supports_nested_summaries": self.supports_nested_summaries,
                "supervision_channels": tuple(
                    channel.to_dict() for channel in self.supervision_channels
                ),
                "theorem_assumptions": (
                    self.theorem_assumptions.to_dict()
                    if self.theorem_assumptions is not None
                    else None
                ),
                "operator_assumptions": (
                    {
                        "evidence_status": self.operator_assumptions.evidence_status.value,
                        "tree_nesting_supported": bool(
                            self.operator_assumptions.tree_nesting_supported
                        ),
                        "theorem_domain_decode_available": bool(
                            self.operator_assumptions.theorem_domain_decode_available
                        ),
                        "theorem_domain_reencode_available": bool(
                            self.operator_assumptions.theorem_domain_reencode_available
                        ),
                    }
                    if self.operator_assumptions is not None
                    else None
                ),
                "operator_capabilities": (
                    self.operator_capabilities.to_dict()
                    if self.operator_capabilities is not None
                    else None
                ),
                "notes": tuple(self.notes),
            }
        )


def full_document_supervision_channel(
    *,
    name: str = "full_document_labels",
    target_name: str = "document_target",
    active: bool = True,
    label_source: TruthLabelSource = ORACLE_SOURCE,
    delivery_mode: SupervisionDeliveryMode = SupervisionDeliveryMode.OFFLINE_LOGGED,
    query_policy: Optional[OracleQueryPolicySpec] = None,
    notes: Sequence[str] = (),
) -> SupervisionChannelSpec:
    return SupervisionChannelSpec(
        name=str(name),
        kind=SupervisionChannelKind.FULL_DOCUMENT,
        target_name=str(target_name),
        active=bool(active),
        label_source=normalize_truth_label_source(label_source),
        acquisition_mode=LabelAcquisitionMode.COMPLETE_LABELS,
        delivery_mode=delivery_mode,
        requires_propensity_logging=False,
        supports_unbiased_risk=True,
        query_policy=query_policy,
        notes=tuple(str(note) for note in notes),
    )


def sampled_substructure_supervision_channel(
    *,
    name: str = "sampled_substructure_labels",
    target_name: str = "substructure_target",
    active: bool = True,
    label_source: TruthLabelSource = ORACLE_SOURCE,
    delivery_mode: SupervisionDeliveryMode = SupervisionDeliveryMode.OFFLINE_LOGGED,
    query_policy: Optional[OracleQueryPolicySpec] = None,
    targeted_laws: Sequence[LawKind] = (),
    requires_propensity_logging: bool = True,
    supports_unbiased_risk: bool = True,
    notes: Sequence[str] = (),
) -> SupervisionChannelSpec:
    return SupervisionChannelSpec(
        name=str(name),
        kind=SupervisionChannelKind.SAMPLED_SUBSTRUCTURE,
        target_name=str(target_name),
        active=bool(active),
        label_source=normalize_truth_label_source(label_source),
        acquisition_mode=LabelAcquisitionMode.RANDOM_SAMPLED_LABELS,
        delivery_mode=delivery_mode,
        requires_propensity_logging=bool(requires_propensity_logging),
        supports_unbiased_risk=bool(supports_unbiased_risk),
        query_policy=query_policy,
        targeted_laws=tuple(targeted_laws),
        notes=tuple(str(note) for note in notes),
    )


def oracle_query_policy(
    *,
    name: str,
    query_unit_name: str,
    selection_strategy: str,
    adaptive: bool = False,
    budget: Optional[Dict[str, Any]] = None,
    propensity_field_name: str = "propensity",
    logs_realized_propensities: bool = False,
    supports_ipw_estimation: bool = False,
    notes: Sequence[str] = (),
) -> OracleQueryPolicySpec:
    return OracleQueryPolicySpec(
        name=str(name),
        query_unit_name=str(query_unit_name),
        selection_strategy=str(selection_strategy),
        adaptive=bool(adaptive),
        budget=dict(budget or {}),
        propensity_field_name=str(propensity_field_name),
        logs_realized_propensities=bool(logs_realized_propensities),
        supports_ipw_estimation=bool(supports_ipw_estimation),
        notes=tuple(str(note) for note in notes),
    )


__all__ = [
    "LabelAcquisitionMode",
    "SupervisionDeliveryMode",
    "SupervisionChannelKind",
    "OracleQueryPolicySpec",
    "FullDocumentLabelSource",
    "SampledSubstructureLabelSource",
    "FullDocumentLabelObservation",
    "SampledSubstructureLabelObservation",
    "SupervisionChannelSpec",
    "CompositionalLearningProblemSpec",
    "full_document_supervision_channel",
    "sampled_substructure_supervision_channel",
    "oracle_query_policy",
]
