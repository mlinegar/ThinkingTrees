"""Core data models and LLM integration."""

from src.core.data_models import (
    Node,
    Tree,
    AuditStatus,
    AuditResult,
    leaf,
    node,
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

from src.core.strategy import (
    SummarizationStrategy,
    DSPyStrategy,
    BatchedStrategy,
)

from src.core.prompting import (
    PromptBuilders,
    default_summarize_prompt,
    default_merge_prompt,
    parse_numeric_score,
)

from src.core.checkpoints import (
    CheckpointManager,
    CheckpointMetadata,
    CHECKPOINT_VERSION,
)

from src.core.scorers import (
    ScaleScorer,
    PairwiseScorer,
)

from src.core.summarization import (
    GenericSummarizer,
    GenericMerger,
    SummarizationResult,
    create_summarizers,
)

__all__ = [
    # Data models
    "Node",
    "Tree",
    "AuditStatus",
    "AuditResult",
    "leaf",
    "node",
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
    # Strategy interface
    "SummarizationStrategy",
    "DSPyStrategy",
    "BatchedStrategy",
    # Prompting
    "PromptBuilders",
    "default_summarize_prompt",
    "default_merge_prompt",
    "parse_numeric_score",
    # Checkpoints
    "CheckpointManager",
    "CheckpointMetadata",
    "CHECKPOINT_VERSION",
    # Scorers
    "ScaleScorer",
    "PairwiseScorer",
    # Summarization
    "GenericSummarizer",
    "GenericMerger",
    "SummarizationResult",
    "create_summarizers",
]
