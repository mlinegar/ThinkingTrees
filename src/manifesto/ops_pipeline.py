"""
OPS Pipeline for Manifesto RILE scoring.

This module integrates the OPS framework with Manifesto Project data
for evaluating RILE score prediction from summaries.
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, List, Any
import logging
import dspy

from src.core.llm_client import LLMConfig, LLMClient, create_summarizer
from src.ops_engine.builder import OPSTreeBuilder, BuildConfig, BuildResult
from src.ops_engine.auditor import OPSAuditor, AuditConfig, AuditReport, ReviewQueue
from src.core.data_models import OPSTree

from .constants import RILE_RANGE
from .data_loader import ManifestoSample
from .rubrics import RILE_PRESERVATION_RUBRIC, RILE_TASK_CONTEXT
from .signatures import RILEScorer
from .position_oracle import RILEPositionOracle, SimpleRILEOracle


logger = logging.getLogger(__name__)


@dataclass
class ManifestoResult:
    """Result of processing a single manifesto through the pipeline."""

    manifesto_id: str
    party_name: str
    country: str
    year: int

    # Ground truth
    ground_truth_rile: float

    # Predictions
    predicted_rile: Optional[float] = None
    baseline_rile: Optional[float] = None  # RILE from full text (no OPS)

    # OPS tree info
    tree: Optional[OPSTree] = None
    tree_height: Optional[int] = None
    tree_nodes: Optional[int] = None
    tree_leaves: Optional[int] = None

    # Audit info
    audit_report: Optional[AuditReport] = None
    audit_passed: bool = True
    audit_failure_rate: float = 0.0

    # Summaries
    final_summary: str = ""
    summary_length: int = 0
    original_length: int = 0
    compression_ratio: float = 1.0

    # DSPy training support: store intermediate results
    chunks: List[str] = field(default_factory=list)
    leaf_summaries: List[str] = field(default_factory=list)
    processing_time: float = 0.0

    # Errors
    error: Optional[str] = None

    # Detailed analysis
    left_indicators: str = ""
    right_indicators: str = ""
    reasoning: str = ""

    @property
    def prediction_error(self) -> Optional[float]:
        """Absolute error between prediction and ground truth."""
        if self.predicted_rile is not None:
            return abs(self.predicted_rile - self.ground_truth_rile)
        return None

    @property
    def baseline_error(self) -> Optional[float]:
        """Absolute error for baseline (full text) prediction."""
        if self.baseline_rile is not None:
            return abs(self.baseline_rile - self.ground_truth_rile)
        return None


@dataclass
class PipelineConfig:
    """Configuration for the Manifesto OPS pipeline."""

    # Model configuration
    task_model_port: int = 8000      # Port for task model (30b) - summarization & scoring
    auditor_model_port: int = 8001   # Port for auditor model (80b) - audit oracle

    # Chunk settings
    max_chunk_chars: int = 2000

    # Audit settings
    audit_budget: int = 10
    rile_threshold: float = 10.0  # RILE points for audit pass

    # Rubric
    rubric: str = RILE_PRESERVATION_RUBRIC
    task_context: str = RILE_TASK_CONTEXT

    # Options
    run_baseline: bool = True  # Also score full text directly
    use_llm_oracle: bool = False  # Use LLM oracle vs simple keyword oracle


class ManifestoOPSPipeline:
    """
    End-to-end pipeline for Manifesto RILE scoring with OPS.

    Uses:
    - Task model (30b) for summarization and RILE scoring
    - Auditor model (80b) for position preservation auditing
    """

    def __init__(self, config: Optional[PipelineConfig] = None):
        """
        Initialize the pipeline.

        Args:
            config: Pipeline configuration
        """
        self.config = config or PipelineConfig()
        self._initialized = False

        # Will be initialized lazily
        self._task_client = None
        self._auditor_client = None
        self._tree_builder = None
        self._auditor = None
        self._scorer = None

    def _initialize(self):
        """Initialize components lazily."""
        if self._initialized:
            return

        # Task model client (for summarization and scoring)
        task_config = LLMConfig.vllm(
            model="default",
            port=self.config.task_model_port
        )
        self._task_client = LLMClient(task_config)

        # Create summarizer for tree building
        summarizer = create_summarizer(
            client=self._task_client,
            system_prompt="You are a political text summarizer. Preserve all information relevant to left-right political positioning."
        )

        # Tree builder
        build_config = BuildConfig(
            max_chunk_chars=self.config.max_chunk_chars,
            verbose=False
        )
        self._tree_builder = OPSTreeBuilder(
            summarizer=summarizer,
            config=build_config
        )

        # Auditor oracle
        if self.config.use_llm_oracle:
            # Configure DSPy for auditor model
            auditor_config = LLMConfig.vllm(
                model="default",
                port=self.config.auditor_model_port
            )
            self._auditor_client = LLMClient(auditor_config)

            # Set up DSPy LM for auditor
            auditor_lm = dspy.LM(
                f"openai/default",
                api_base=f"http://localhost:{self.config.auditor_model_port}/v1",
                api_key="EMPTY"
            )
            dspy.configure(lm=auditor_lm)

            oracle = RILEPositionOracle(
                threshold=self.config.rile_threshold,
                task_context=self.config.task_context
            )
        else:
            # Use simple keyword-based oracle for testing
            oracle = SimpleRILEOracle(threshold=self.config.rile_threshold)

        # Auditor
        audit_config = AuditConfig(
            sample_budget=self.config.audit_budget,
            discrepancy_threshold=self.config.rile_threshold / RILE_RANGE  # Normalize
        )
        self._review_queue = ReviewQueue()
        self._auditor = OPSAuditor(
            oracle=oracle,
            config=audit_config,
            review_queue=self._review_queue
        )

        # RILE scorer (uses task model)
        task_lm = dspy.LM(
            f"openai/default",
            api_base=f"http://localhost:{self.config.task_model_port}/v1",
            api_key="EMPTY"
        )
        dspy.configure(lm=task_lm)
        self._scorer = RILEScorer()

        self._initialized = True
        logger.info("Pipeline initialized")

    def process_manifesto(
        self,
        sample: ManifestoSample,
        run_baseline: Optional[bool] = None
    ) -> ManifestoResult:
        """
        Process a single manifesto through the pipeline.

        Steps:
        1. Build OPS tree with RILE-preservation rubric
        2. Audit tree for position preservation
        3. Score RILE from root summary
        4. Optionally score RILE from full text (baseline)
        5. Compare to ground truth

        Args:
            sample: ManifestoSample to process
            run_baseline: Override config.run_baseline if specified

        Returns:
            ManifestoResult with all results
        """
        self._initialize()

        run_baseline = run_baseline if run_baseline is not None else self.config.run_baseline

        result = ManifestoResult(
            manifesto_id=sample.manifesto_id,
            party_name=sample.party_name,
            country=sample.country_name,
            year=sample.year,
            ground_truth_rile=sample.rile,
            original_length=len(sample.text)
        )

        try:
            # 1. Build OPS tree
            logger.info(f"Building OPS tree for {sample.manifesto_id}")
            build_result = self._tree_builder.build_from_text(
                sample.text,
                rubric=self.config.rubric
            )

            result.tree = build_result.tree
            result.tree_height = build_result.tree.height
            result.tree_nodes = build_result.nodes_created
            result.tree_leaves = build_result.tree.leaf_count
            result.final_summary = build_result.tree.final_summary
            result.summary_length = len(result.final_summary)
            result.compression_ratio = result.original_length / max(result.summary_length, 1)

            # 2. Audit tree
            logger.info(f"Auditing tree for {sample.manifesto_id}")
            audit_report = self._auditor.audit_tree(build_result.tree)

            result.audit_report = audit_report
            result.audit_passed = audit_report.passed
            result.audit_failure_rate = audit_report.failure_rate

            # 3. Score RILE from summary
            logger.info(f"Scoring RILE from summary for {sample.manifesto_id}")
            score_result = self._scorer(
                text=result.final_summary,
                task_context=self.config.task_context
            )

            result.predicted_rile = float(score_result['rile_score'])
            result.left_indicators = score_result.get('left_indicators', '')
            result.right_indicators = score_result.get('right_indicators', '')
            result.reasoning = score_result.get('reasoning', '')

            # 4. Baseline: score from full text
            if run_baseline:
                logger.info(f"Scoring baseline RILE for {sample.manifesto_id}")
                # For very long texts, truncate to fit context
                text_for_baseline = sample.text[:50000]  # ~12K tokens
                baseline_result = self._scorer(
                    text=text_for_baseline,
                    task_context=self.config.task_context
                )
                result.baseline_rile = float(baseline_result['rile_score'])

        except Exception as e:
            logger.error(f"Error processing {sample.manifesto_id}: {e}")
            result.error = str(e)

        return result

    def process_batch(
        self,
        samples: List[ManifestoSample],
        run_baseline: Optional[bool] = None
    ) -> List[ManifestoResult]:
        """
        Process multiple manifestos.

        Args:
            samples: List of ManifestoSamples
            run_baseline: Override config.run_baseline

        Returns:
            List of ManifestoResults
        """
        results = []
        for i, sample in enumerate(samples):
            logger.info(f"Processing {i+1}/{len(samples)}: {sample.manifesto_id}")
            result = self.process_manifesto(sample, run_baseline)
            results.append(result)

        return results

    def get_review_queue(self) -> ReviewQueue:
        """Get the review queue with flagged audit items."""
        return self._review_queue


class SimplePipeline:
    """
    Simplified pipeline for testing without LLM.

    Uses simple summarizers and keyword-based scoring.
    """

    def __init__(self, max_chunk_chars: int = 2000):
        from src.ops_engine.builder import TruncatingSummarizer

        self.summarizer = TruncatingSummarizer(max_length=500)
        self.oracle = SimpleRILEOracle(threshold=10.0)

        build_config = BuildConfig(max_chunk_chars=max_chunk_chars)
        self._tree_builder = OPSTreeBuilder(
            summarizer=self.summarizer,
            config=build_config
        )

        audit_config = AuditConfig(sample_budget=5)
        self._auditor = OPSAuditor(
            oracle=self.oracle,
            config=audit_config
        )

    def process_manifesto(self, sample: ManifestoSample) -> ManifestoResult:
        """Process with simple summarization (no LLM)."""
        result = ManifestoResult(
            manifesto_id=sample.manifesto_id,
            party_name=sample.party_name,
            country=sample.country_name,
            year=sample.year,
            ground_truth_rile=sample.rile,
            original_length=len(sample.text)
        )

        try:
            # Build tree
            build_result = self._tree_builder.build_from_text(sample.text, rubric="")

            result.tree = build_result.tree
            result.tree_height = build_result.tree.height
            result.tree_nodes = build_result.nodes_created
            result.tree_leaves = build_result.tree.leaf_count
            result.final_summary = build_result.tree.final_summary
            result.summary_length = len(result.final_summary)

            # Audit
            audit_report = self._auditor.audit_tree(build_result.tree)
            result.audit_report = audit_report
            result.audit_passed = audit_report.passed

            # Simple keyword-based RILE estimate
            result.predicted_rile = self.oracle._estimate_rile(result.final_summary)
            result.baseline_rile = self.oracle._estimate_rile(sample.text)

        except Exception as e:
            result.error = str(e)

        return result
