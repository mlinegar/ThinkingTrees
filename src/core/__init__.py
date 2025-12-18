"""Core data models and LLM integration."""

from src.core.data_models import (
    OPSNode,
    OPSTree,
    AuditStatus,
    AuditResult,
    create_leaf_node,
    create_internal_node,
    build_tree_from_leaves,
)

from src.core.llm_client import (
    ServerType,
    LLMConfig,
    LLMResponse,
    LLMClient,
    MockLLMClient,
    create_client,
    create_summarizer,
    vllm_client,
    sglang_client,
    openai_client,
)

from src.core.signatures import (
    RecursiveSummary,
    OracleJudge,
    SufficiencyCheck,
    MergeConsistencyCheck,
    Summarizer,
    Judge,
    SufficiencyChecker,
    MergeChecker,
    OracleFuncApproximation,
    OracleFuncReviewer,
)

__all__ = [
    # Data models
    "OPSNode",
    "OPSTree",
    "AuditStatus",
    "AuditResult",
    "create_leaf_node",
    "create_internal_node",
    "build_tree_from_leaves",
    # LLM client
    "ServerType",
    "LLMConfig",
    "LLMResponse",
    "LLMClient",
    "MockLLMClient",
    "create_client",
    "create_summarizer",
    "vllm_client",
    "sglang_client",
    "openai_client",
    # DSPy Signatures
    "RecursiveSummary",
    "OracleJudge",
    "SufficiencyCheck",
    "MergeConsistencyCheck",
    "Summarizer",
    "Judge",
    "SufficiencyChecker",
    "MergeChecker",
    "OracleFuncApproximation",
    "OracleFuncReviewer",
]
