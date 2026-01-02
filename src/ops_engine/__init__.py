"""OPS tree building, auditing, and optimization."""

from src.ops_engine.builder import (
    TreeBuilder,
    AsyncTreeBuilder,
    BuildConfig,
    BuildResult,
    IdentitySummarizer,
    TruncatingSummarizer,
    ConcatenatingSummarizer,
    build,
    async_build,
    build_test_tree,
)

# Helper functions for chunking
# Note: Tournament selection is now in TournamentStrategy (src/core/strategy.py)
from src.ops_engine.ops_tree import (
    chunk_binary,
)

from src.ops_engine.auditor import (
    Auditor,
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

# For optimization, use the registry-based system:
# from src.ops_engine.training_framework.optimizers import get_optimizer

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
    BootstrapTrainer,
    BootstrapConfig,
    BootstrapResult,
    BootstrapIteration,
    run_bootstrap_training,
)

# Unified OPS checks
from src.ops_engine.checks import (
    CheckRunner,
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
    oracle_demos,
    initialize_summarizer,
    quick_demos,
    run_top_down_initialization,
    train_on_short_docs,
)

__all__ = [
    # Builder
    "TreeBuilder",
    "AsyncTreeBuilder",
    "BuildConfig",
    "BuildResult",
    "IdentitySummarizer",
    "TruncatingSummarizer",
    "ConcatenatingSummarizer",
    "build",
    "async_build",
    "build_test_tree",
    # Chunking helper (tournament selection is in TournamentStrategy)
    "chunk_binary",
    # Auditor
    "Auditor",
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
    "BootstrapTrainer",
    "BootstrapConfig",
    "BootstrapResult",
    "BootstrapIteration",
    "run_bootstrap_training",
    # Unified OPS checks
    "CheckRunner",
    "CheckConfig",
    "CheckResult",
    "CheckType",
    "run_all_checks",
    "aggregate_check_stats",
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
    # Top-down initialization (oracle-aligned demo seeding)
    "TopDownInitializer",
    "OracleAlignedDemo",
    "MergeAlignedDemo",
    "oracle_demos",
    "initialize_summarizer",
    "quick_demos",
    "run_top_down_initialization",
    "train_on_short_docs",
]
