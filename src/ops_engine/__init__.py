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
    SimpleScorer,
    AlwaysPassOracle,
    AlwaysFailOracle,
    SamplingStrategy,
    ReviewQueue,
    FlaggedItem,
    ReviewPriority,
    audit_tree,
    create_oracle_from_scorer,
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

# Bootstrap training loop (Paper Section 3.11)
from src.ops_engine.bootstrap_loop import (
    OPSBootstrapTrainer,
    BootstrapConfig,
    BootstrapResult,
    BootstrapIteration,
    run_bootstrap_training,
)

# Unified OPS checks
from src.ops_engine.checks import (
    OPSCheckRunner,
    CheckConfig,
    CheckResult,
    CheckType,
    run_all_checks,
    aggregate_check_stats,
)

# Score-centric oracle types (new, preferred API)
from src.ops_engine.scoring import (
    OracleScore,
    ScoringOracle,
    SimilarityScorer,
    LegacyOracleAdapter,
    as_scoring_oracle,
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

# Top-down initialization (oracle-aligned demo seeding)
from src.ops_engine.initialization import (
    TopDownInitializer,
    OracleAlignedDemo,
    MergeAlignedDemo,
    create_oracle_aligned_demos,
    initialize_summarizer_with_demos,
    create_quick_init_demos,
    run_top_down_initialization,
    train_on_short_docs,
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
    "SimpleScorer",
    "AlwaysPassOracle",
    "AlwaysFailOracle",
    "SamplingStrategy",
    "ReviewQueue",
    "FlaggedItem",
    "ReviewPriority",
    "audit_tree",
    "create_oracle_from_scorer",
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
    # Bootstrap training loop (Paper Section 3.11)
    "OPSBootstrapTrainer",
    "BootstrapConfig",
    "BootstrapResult",
    "BootstrapIteration",
    "run_bootstrap_training",
    # Unified OPS checks
    "OPSCheckRunner",
    "CheckConfig",
    "CheckResult",
    "CheckType",
    "run_all_checks",
    "aggregate_check_stats",
    # Score-centric oracle types (new, preferred API)
    "OracleScore",
    "ScoringOracle",
    "SimilarityScorer",
    "LegacyOracleAdapter",
    "as_scoring_oracle",
    "oracle_as_metric",
    "oracle_as_metric_with_feedback",
    "normalize_error_to_score",
    "score_to_error",
    # Generic bounded scale (domain-agnostic)
    "BoundedScale",
    "UNIT_SCALE",
    "PERCENT_SCALE",
    "SYMMETRIC_SCALE",
    # Top-down initialization (oracle-aligned demo seeding)
    "TopDownInitializer",
    "OracleAlignedDemo",
    "MergeAlignedDemo",
    "create_oracle_aligned_demos",
    "initialize_summarizer_with_demos",
    "create_quick_init_demos",
    "run_top_down_initialization",
    "train_on_short_docs",
]
