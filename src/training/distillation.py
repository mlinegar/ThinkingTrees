"""Trainer utilities for teacher-student distillation runs."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional
import json
import logging
import math
from difflib import SequenceMatcher

from src.ops_engine.builder import BuildConfig, OPSTreeBuilder, PreferenceScorer
from src.preprocessing.chunker import DocumentChunker, TextChunk, chunk_for_ops

logger = logging.getLogger(__name__)


@dataclass
class DistillationExample:
    """Single training example carrying teacher outputs."""

    text: str
    teacher_logits: Optional[List[float]] = None
    teacher_response: Optional[str] = None
    metadata: Dict[str, object] = field(default_factory=dict)


@dataclass
class DistillationResult:
    """Container for processed distillation results."""

    example: DistillationExample
    student_summary: str
    student_targets: Dict[str, object]
    alignment_metrics: Dict[str, float]


@dataclass
class DistillationConfig:
    """Configuration controlling distillation runs."""

    rubric: str = "Preserve key information and factual grounding."
    max_chunk_chars: int = 2000
    chunk_strategy: str = "sentence"
    teacher_guided: bool = False
    student_only: bool = True


class DistillationTrainer:
    """Runs teacher-student distillation over text datasets."""

    def __init__(
        self,
        builder: OPSTreeBuilder,
        preference_scorer: Optional[PreferenceScorer] = None,
        config: Optional[DistillationConfig] = None,
        build_config: Optional[BuildConfig] = None,
        chunker: Optional[DocumentChunker] = None,
    ) -> None:
        self.config = config or DistillationConfig()
        self.build_config = build_config or builder.config
        self.builder = builder

        if preference_scorer:
            self.builder.preference_scorer = preference_scorer

        # Make sure the builder is aligned with the desired guidance mode
        self.builder.config.teacher_guided = self.config.teacher_guided
        self.builder.config.student_only = self.config.student_only

        self.chunker = chunker

    def load_dataset(self, dataset_path: Path) -> List[DistillationExample]:
        """Load newline-delimited JSON or raw-text datasets."""
        dataset_path = Path(dataset_path)
        if not dataset_path.exists():
            raise FileNotFoundError(dataset_path)

        examples: List[DistillationExample] = []
        for line in dataset_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue

            try:
                payload = json.loads(line)
                text = payload.get("text", "") or ""
                teacher_logits = payload.get("teacher_logits")
                teacher_response = payload.get("teacher_response")
                metadata = {
                    k: v
                    for k, v in payload.items()
                    if k not in {"text", "teacher_logits", "teacher_response"}
                }
            except json.JSONDecodeError:
                text = line
                teacher_logits = None
                teacher_response = None
                metadata = {}

            examples.append(
                DistillationExample(
                    text=text,
                    teacher_logits=teacher_logits,
                    teacher_response=teacher_response,
                    metadata=metadata,
                )
            )

        logger.info("Loaded %d training examples", len(examples))
        return examples

    def _softmax(self, logits: Iterable[float]) -> List[float]:
        logits_list = list(logits)
        if not logits_list:
            return []

        max_logit = max(logits_list)
        exp_logits = [math.exp(logit - max_logit) for logit in logits_list]
        total = sum(exp_logits)
        if total == 0:
            return [0.0 for _ in exp_logits]
        return [value / total for value in exp_logits]

    def _build_student_targets(self, example: DistillationExample) -> Dict[str, object]:
        targets: Dict[str, object] = {}
        if example.teacher_logits is not None:
            targets["probabilities"] = self._softmax(example.teacher_logits)
        if example.teacher_response:
            targets["teacher_response"] = example.teacher_response
        return targets

    def _alignment_metrics(
        self,
        example: DistillationExample,
        student_summary: str,
        student_targets: Dict[str, object],
    ) -> Dict[str, float]:
        metrics: Dict[str, float] = {
            "student_summary_chars": float(len(student_summary)),
        }

        if example.teacher_response:
            matcher = SequenceMatcher(None, example.teacher_response, student_summary)
            metrics["response_overlap"] = matcher.quick_ratio()

        probabilities = student_targets.get("probabilities")
        if probabilities:
            metrics["teacher_confidence"] = max(probabilities)
            metrics["logit_entropy"] = -sum(
                p * math.log(p) for p in probabilities if p > 0
            )

        return metrics

    def _chunk_example(self, text: str) -> List[TextChunk]:
        if self.chunker is not None:
            return self.chunker.chunk(text)

        return chunk_for_ops(
            text=text,
            max_chars=self.build_config.max_chunk_chars,
            strategy=self.config.chunk_strategy,
        )

    def process_example(self, example: DistillationExample) -> DistillationResult:
        """Chunk, summarize, and align a single example."""
        chunks = self._chunk_example(example.text)
        if not chunks:
            raise ValueError("Example produced no chunks")

        build_result = self.builder.build_from_chunks(chunks, rubric=self.config.rubric)
        student_summary = build_result.tree.final_summary

        student_targets = self._build_student_targets(example)
        metrics = self._alignment_metrics(example, student_summary, student_targets)

        logger.debug(
            "Processed example with %d chunks; summary chars=%d", len(chunks), len(student_summary)
        )

        return DistillationResult(
            example=example,
            student_summary=student_summary,
            student_targets=student_targets,
            alignment_metrics=metrics,
        )

    def run(self, dataset_path: Path) -> List[DistillationResult]:
        """Run distillation over a dataset and return processed results."""
        examples = self.load_dataset(dataset_path)
        results: List[DistillationResult] = []
        for example in examples:
            try:
                results.append(self.process_example(example))
            except Exception as exc:  # pragma: no cover - defensive logging
                logger.error("Failed to process example: %s", exc)
        return results
