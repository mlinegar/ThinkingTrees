"""
DSPy Module Wrapper for GenRM Judge.

Wraps GenRMJudge as a DSPy module to enable prompt optimization.
"""

import dspy
from typing import Optional, List, Tuple

from src.ops_engine.training_framework.genrm_preference import GenRMJudge, GenRMResult


class GenRMComparisonSignature(dspy.Signature):
    """
    Signature for GenRM pairwise comparison.

    Optimizable instructions for how GenRM should compare summaries.
    """

    context: str = dspy.InputField(
        desc="Description of what information should be preserved in the summary"
    )
    original_text: str = dspy.InputField(
        desc="The original text being summarized"
    )
    summary_a: str = dspy.InputField(
        desc="First candidate summary to compare"
    )
    summary_b: str = dspy.InputField(
        desc="Second candidate summary to compare"
    )
    law_type: str = dspy.InputField(
        desc="Type of OPS law being evaluated (sufficiency, idempotence, merge)"
    )

    # Output fields
    preference: str = dspy.OutputField(
        desc="Which summary is better: 'A', 'B', or 'tie'"
    )
    reasoning: str = dspy.OutputField(
        desc="Brief explanation of why this summary is preferred"
    )
    score_a: str = dspy.OutputField(
        desc="Helpfulness score for summary A (1-5)"
    )
    score_b: str = dspy.OutputField(
        desc="Helpfulness score for summary B (1-5)"
    )


class GenRMComparisonModule(dspy.Module):
    """
    DSPy module for pairwise comparison.

    Supports two modes:
    - GenRM mode: Uses NVIDIA's specialized GenRM reward model (default)
    - DSPy mode: Uses DSPy ChainOfThought with optimizable prompts (for testing/fallback)

    When using GenRM mode, the comparison uses the specialized reward model API.
    When using DSPy mode, comparison prompts can be optimized via DSPy optimizers.
    """

    def __init__(
        self,
        genrm_judge: Optional[GenRMJudge] = None,
        base_url: Optional[str] = None,
        model_name: Optional[str] = None,
        use_dspy_predictor: bool = False,
    ):
        """
        Initialize comparison module.

        Args:
            genrm_judge: Existing GenRMJudge instance (or create new one)
            base_url: GenRM server URL (None = auto-detect from config)
            model_name: GenRM model name (auto-detected if None)
            use_dspy_predictor: If True, use DSPy ChainOfThought instead of GenRM
        """
        super().__init__()

        self.use_dspy_predictor = use_dspy_predictor

        # Create DSPy predictor for comparison (optimizable via DSPy)
        self.compare = dspy.ChainOfThought(GenRMComparisonSignature)

        # Create GenRM judge (used when use_dspy_predictor=False)
        if not use_dspy_predictor:
            if genrm_judge is not None:
                self.judge = genrm_judge
            else:
                self.judge = GenRMJudge(
                    base_url=base_url,
                    model_name=model_name,
                )
        else:
            self.judge = None

    def forward(
        self,
        context: str,
        original_text: str,
        summary_a: str,
        summary_b: str,
        law_type: str = "sufficiency",
    ) -> dspy.Prediction:
        """
        Compare two summaries.

        Uses either GenRM (specialized reward model) or DSPy predictor
        depending on initialization settings.

        Args:
            context: What information to preserve
            original_text: Original text
            summary_a: First summary
            summary_b: Second summary
            law_type: Type of law (sufficiency, idempotence, merge)

        Returns:
            dspy.Prediction with preference, reasoning, scores
        """
        if self.use_dspy_predictor:
            # Use DSPy ChainOfThought predictor (optimizable)
            result = self.compare(
                context=context,
                original_text=original_text,
                summary_a=summary_a,
                summary_b=summary_b,
                law_type=law_type,
            )

            # Parse scores from string output
            try:
                score_a = float(result.score_a)
            except (ValueError, TypeError):
                score_a = 3.0  # Default middle score

            try:
                score_b = float(result.score_b)
            except (ValueError, TypeError):
                score_b = 3.0

            return dspy.Prediction(
                preference=result.preference,
                reasoning=result.reasoning,
                score_a=str(score_a),
                score_b=str(score_b),
                helpfulness_a=score_a,
                helpfulness_b=score_b,
                ranking_score=1 if result.preference == "A" else (6 if result.preference == "B" else 3),
            )
        else:
            # Use GenRM specialized reward model
            result: GenRMResult = self.judge.compare(
                context=context,
                original_text=original_text,
                summary_a=summary_a,
                summary_b=summary_b,
                law_type=law_type,
            )

            # Convert GenRM result to DSPy prediction format
            preference = result.preferred  # 'A', 'B', or 'tie'

            return dspy.Prediction(
                preference=preference,
                reasoning=result.reasoning,
                score_a=str(result.helpfulness_a),
                score_b=str(result.helpfulness_b),
                helpfulness_a=result.helpfulness_a,
                helpfulness_b=result.helpfulness_b,
                ranking_score=result.ranking_score,
            )

    def compare_batch(
        self,
        comparisons: List[Tuple[str, str, str, str, str]],
    ) -> List[dspy.Prediction]:
        """
        Compare multiple pairs in parallel.

        Args:
            comparisons: List of (context, original, summary_a, summary_b, law_type) tuples

        Returns:
            List of predictions
        """
        from concurrent.futures import ThreadPoolExecutor, as_completed

        results = [None] * len(comparisons)

        with ThreadPoolExecutor(max_workers=10) as executor:
            future_to_idx = {
                executor.submit(
                    self.forward,
                    context=comp[0],
                    original_text=comp[1],
                    summary_a=comp[2],
                    summary_b=comp[3],
                    law_type=comp[4],
                ): idx
                for idx, comp in enumerate(comparisons)
            }

            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                try:
                    results[idx] = future.result()
                except Exception as e:
                    # Return error prediction
                    results[idx] = dspy.Prediction(
                        preference="tie",
                        reasoning=f"Error: {str(e)}",
                        score_a="0",
                        score_b="0",
                        helpfulness_a=0.0,
                        helpfulness_b=0.0,
                        ranking_score=0.0,
                    )

        return results
