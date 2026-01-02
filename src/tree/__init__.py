"""
Tree building, auditing, and verification for OPS.

This module provides:
- TreeBuilder: Construct summarization trees from documents
- Auditor: Probabilistic verification of tree quality
- TreeVerifier: OPS law verification at tree nodes
- LabeledTree: Data structures for labeled training trees
"""

# Tree construction
from src.tree.builder import (
    TreeBuilder,
    BuildConfig,
    BuildResult,
    IdentitySummarizer,
    ConcatenatingSummarizer,
    TruncatingSummarizer,
    build,
    async_build,
    build_test_tree,
    chunk_binary,
)

# Tree auditing
from src.tree.auditor import (
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
    get_human_review_queue,
    compute_violation_bound,
    compute_expected_distortion,
    get_audit_statistics,
)

# Tree verification
from src.tree.verification import (
    OracleNodeVerifier,
    NodeVerificationResult,
    TreeVerifier,
    ScorePredictor,
)

# Labeled tree data structures
from src.tree.labeled import (
    LabeledNode,
    LabeledTree,
    LabeledDataset,
)

__all__ = [
    # Builder
    "TreeBuilder",
    "BuildConfig",
    "BuildResult",
    "IdentitySummarizer",
    "ConcatenatingSummarizer",
    "TruncatingSummarizer",
    "build",
    "async_build",
    "build_test_tree",
    "chunk_binary",
    # Auditor
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
    "get_human_review_queue",
    "compute_violation_bound",
    "compute_expected_distortion",
    "get_audit_statistics",
    # Verification
    "OracleNodeVerifier",
    "NodeVerificationResult",
    "TreeVerifier",
    "ScorePredictor",
    # Labeled trees
    "LabeledNode",
    "LabeledTree",
    "LabeledDataset",
]
