"""OPS tree building, auditing, and optimization."""

from src.ops_engine.builder import (
    OPSTreeBuilder,
    BuildConfig,
    BuildResult,
    IdentitySummarizer,
    TruncatingSummarizer,
    ConcatenatingSummarizer,
    build_ops_tree,
    build_test_tree,
)

from src.ops_engine.auditor import (
    OPSAuditor,
    AuditConfig,
    AuditReport,
    AuditCheckResult,
    OracleJudge,
    SimpleScorer,  # New, preferred API
    SimpleOracleJudge,  # Deprecated, use SimpleScorer
    AlwaysPassOracle,
    AlwaysFailOracle,
    SamplingStrategy,
    ReviewQueue,
    FlaggedItem,
    ReviewPriority,
    audit_tree,
)

from src.ops_engine.optimizer import (
    OPSOptimizer,
    OptimizationConfig,
    OptimizationResult,
    TrainingExample,
    TrainingDataCollector,
    SummaryMetric,
    create_optimizer,
    optimize_from_reviews,
)

from src.ops_engine.oracle_func_approximation import (
    LearnedOracleFunc,
    OracleFuncReviewEngine,
    OracleFuncConfig,
    OracleFuncTrainingExample,
    OracleFuncTrainingCollector,
    OracleFuncReviewResult,
    OracleFuncMetric,
    ExampleLabel,
    create_oracle_func_reviewer,
    train_oracle_func_from_reviews,
)

# Score-centric oracle types (new, preferred API)
from src.ops_engine.scoring import (
    OracleScore,
    ScoringOracle,
    LegacyOracleAdapter,
    as_scoring_oracle,
    oracle_as_metric,
    oracle_as_metric_with_feedback,
    normalize_error_to_score,
    # Generic bounded scale (domain-agnostic)
    BoundedScale,
    UNIT_SCALE,
    PERCENT_SCALE,
    SYMMETRIC_SCALE,
)

__all__ = [
    # Builder
    "OPSTreeBuilder",
    "BuildConfig",
    "BuildResult",
    "IdentitySummarizer",
    "TruncatingSummarizer",
    "ConcatenatingSummarizer",
    "build_ops_tree",
    "build_test_tree",
    # Auditor
    "OPSAuditor",
    "AuditConfig",
    "AuditReport",
    "AuditCheckResult",
    "OracleJudge",
    "SimpleScorer",  # New, preferred API
    "SimpleOracleJudge",  # Deprecated
    "AlwaysPassOracle",
    "AlwaysFailOracle",
    "SamplingStrategy",
    "ReviewQueue",
    "FlaggedItem",
    "ReviewPriority",
    "audit_tree",
    # Optimizer
    "OPSOptimizer",
    "OptimizationConfig",
    "OptimizationResult",
    "TrainingExample",
    "TrainingDataCollector",
    "SummaryMetric",
    "create_optimizer",
    "optimize_from_reviews",
    # Oracle Function Approximation
    "LearnedOracleFunc",
    "OracleFuncReviewEngine",
    "OracleFuncConfig",
    "OracleFuncTrainingExample",
    "OracleFuncTrainingCollector",
    "OracleFuncReviewResult",
    "OracleFuncMetric",
    "ExampleLabel",
    "create_oracle_func_reviewer",
    "train_oracle_func_from_reviews",
    # Score-centric oracle types (new, preferred API)
    "OracleScore",
    "ScoringOracle",
    "LegacyOracleAdapter",
    "as_scoring_oracle",
    "oracle_as_metric",
    "oracle_as_metric_with_feedback",
    "normalize_error_to_score",
    # Generic bounded scale (domain-agnostic)
    "BoundedScale",
    "UNIT_SCALE",
    "PERCENT_SCALE",
    "SYMMETRIC_SCALE",
]
