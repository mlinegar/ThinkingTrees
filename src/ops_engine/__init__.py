"""OPS tree building, auditing, and optimization."""

from src.ops_engine.builder import (
    TreeBuilder,
    BuildConfig,
    BuildResult,
    IdentitySummarizer,
    ConcatenatingSummarizer,
    build,
    async_build,
    build_test_tree,
    # Chunking helper (moved from ops_tree.py)
    chunk_binary,
)

# OPS Law Checking Types (unified)
from src.ops_engine.ops_checks import (
    CheckType,
    CheckConfig,
    CheckResult,
    aggregate_check_stats,
)

# Human-in-the-loop Auditing
from src.ops_engine.auditor import (
    Auditor,
    AuditConfig,
    AuditReport,
    AuditCheckResult,
    SimpleScorer,
    AlwaysPassScorer,
    AlwaysFailScorer,
    SamplingStrategy,
    ReviewQueue,
    FlaggedItem,
    ReviewPriority,
    audit_tree,
)

# For optimization, use the registry-based system:
# from src.ops_engine.training_framework.optimizers import get_optimizer

# Score-centric oracle types (new, preferred API)
from src.ops_engine.scoring import (
    OracleScore,
    ScoringOracle,
    SimilarityScorer,
    oracle_as_metric,
    oracle_as_metric_with_feedback,
    normalize_error_to_score,
    score_to_error,
    # Generic bounded scale (domain-agnostic)
    BoundedScale,
    UNIT_SCALE,
    PERCENT_SCALE,
    SYMMETRIC_SCALE,
)

__all__ = [
    # Builder
    "TreeBuilder",
    "BuildConfig",
    "BuildResult",
    "IdentitySummarizer",
    "ConcatenatingSummarizer",
    "build",
    "async_build",
    "build_test_tree",
    # Chunking helper
    "chunk_binary",
    # OPS Law Checking (unified types)
    "CheckType",
    "CheckConfig",
    "CheckResult",
    "aggregate_check_stats",
    # Human-in-the-loop Auditing
    "Auditor",
    "AuditConfig",
    "AuditReport",
    "AuditCheckResult",
    "SimpleScorer",
    "AlwaysPassScorer",
    "AlwaysFailScorer",
    "SamplingStrategy",
    "ReviewQueue",
    "FlaggedItem",
    "ReviewPriority",
    "audit_tree",
    # Score-centric oracle types
    "OracleScore",
    "ScoringOracle",
    "SimilarityScorer",
    "oracle_as_metric",
    "oracle_as_metric_with_feedback",
    "normalize_error_to_score",
    "score_to_error",
    # Generic bounded scale (domain-agnostic)
    "BoundedScale",
    "UNIT_SCALE",
    "PERCENT_SCALE",
    "SYMMETRIC_SCALE",
]
