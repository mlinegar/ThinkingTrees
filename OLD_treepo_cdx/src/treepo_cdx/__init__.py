"""treepo_cdx: parallel TreePO/C-TreePO package spine."""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version

from treepo_cdx.audit import (
    InfluenceWeightedAuditOverlap,
    LawKind,
    LocalLawAuditRow,
    compute_influence_weighted_overlap,
)
from treepo_cdx.backends import (
    BackendRuntime,
    StateShapeContract,
    SupervisionSpec,
    backend_capabilities,
)
from treepo_cdx.adapters import (
    local_law_row_from_mapping,
    local_law_rows_from_manifest,
    local_law_rows_from_mappings,
    manifest_rows_from_document_sampling,
)
from treepo_cdx.certificate import (
    UnifiedLearningComponentEvidence,
    UnifiedLearningErrorCertificate,
    build_error_certificate,
)
from treepo_cdx.honesty import (
    HonestChunkingPolicy,
    ThreeLayerHonestyConfig,
    assign_honest_split,
    assign_three_layer_roles,
    role_tuple_for_unit,
)
from treepo_cdx.learning import FitConfig, FitResult, fit
from treepo_cdx.folds import (
    FoldAssignment,
    FoldSpec,
    assign_folds,
    fold_view,
    split_unit_ids,
    stable_fold_id,
    validate_fold_disjointness,
)
from treepo_cdx.local_law import (
    LOCAL_LAW_OBJECTIVE_CORRECTED,
    LOCAL_LAW_OBJECTIVE_MODES,
    LOCAL_LAW_OBJECTIVE_SAMPLED_IPW,
    LocalLawObjectiveSummary,
    corrected_local_law_loss,
    local_law_objective_mean,
    local_law_objective_summary,
    normalize_local_law_objective_mode,
)
from treepo_cdx.manifest import (
    ArtifactLineage,
    ArtifactRef,
    ManifestRow,
    RoleTuple,
    RunManifestContract,
    Span,
    TopLevelUnit,
    manifest_digest,
)
from treepo_cdx.objective import ObjectiveSpec, normalize_objective_spec
from treepo_cdx.release import audit_public_imports, audit_release, audit_static_imports
from treepo_cdx.sampling import DocumentSamplingRow, ObservationUnitKind, SamplingMetadata
from treepo_cdx.sketches import HLLSketchRuntime, hll_fit_summary

try:
    __version__ = version("treepo-cdx")
except (PackageNotFoundError, TypeError, KeyError):  # pragma: no cover
    __version__ = "0.1.0"


__all__ = [
    "__version__",
    "ArtifactLineage",
    "ArtifactRef",
    "BackendRuntime",
    "DocumentSamplingRow",
    "FitConfig",
    "FitResult",
    "FoldAssignment",
    "FoldSpec",
    "HLLSketchRuntime",
    "HonestChunkingPolicy",
    "InfluenceWeightedAuditOverlap",
    "LOCAL_LAW_OBJECTIVE_CORRECTED",
    "LOCAL_LAW_OBJECTIVE_MODES",
    "LOCAL_LAW_OBJECTIVE_SAMPLED_IPW",
    "LawKind",
    "LocalLawAuditRow",
    "LocalLawObjectiveSummary",
    "ManifestRow",
    "ObjectiveSpec",
    "ObservationUnitKind",
    "RoleTuple",
    "RunManifestContract",
    "SamplingMetadata",
    "Span",
    "StateShapeContract",
    "SupervisionSpec",
    "ThreeLayerHonestyConfig",
    "TopLevelUnit",
    "UnifiedLearningComponentEvidence",
    "UnifiedLearningErrorCertificate",
    "assign_folds",
    "assign_honest_split",
    "assign_three_layer_roles",
    "audit_public_imports",
    "audit_release",
    "audit_static_imports",
    "backend_capabilities",
    "build_error_certificate",
    "compute_influence_weighted_overlap",
    "corrected_local_law_loss",
    "fit",
    "fold_view",
    "hll_fit_summary",
    "local_law_objective_mean",
    "local_law_objective_summary",
    "local_law_row_from_mapping",
    "local_law_rows_from_manifest",
    "local_law_rows_from_mappings",
    "manifest_digest",
    "manifest_rows_from_document_sampling",
    "normalize_local_law_objective_mode",
    "normalize_objective_spec",
    "role_tuple_for_unit",
    "split_unit_ids",
    "stable_fold_id",
    "validate_fold_disjointness",
]
