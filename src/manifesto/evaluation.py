"""
Evaluation framework for Manifesto RILE scoring.

This module provides metrics and analysis tools for evaluating
the quality of RILE predictions from OPS summaries.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any, Tuple
import math
import json
from pathlib import Path
from datetime import datetime

from .ops_pipeline import ManifestoResult


@dataclass
class EvaluationMetrics:
    """Comprehensive metrics for RILE prediction evaluation."""

    # Sample info
    n_samples: int = 0
    n_errors: int = 0

    # Regression metrics for OPS predictions
    mse: float = 0.0           # Mean squared error
    rmse: float = 0.0          # Root mean squared error
    mae: float = 0.0           # Mean absolute error
    correlation: float = 0.0   # Pearson correlation

    # Baseline metrics (full text prediction)
    baseline_mse: float = 0.0
    baseline_rmse: float = 0.0
    baseline_mae: float = 0.0
    baseline_correlation: float = 0.0

    # Position preservation metrics
    within_5_points: float = 0.0   # % within 5 RILE points
    within_10_points: float = 0.0  # % within 10 RILE points
    within_20_points: float = 0.0  # % within 20 RILE points

    # OPS audit metrics
    mean_audit_failure_rate: float = 0.0
    audit_pass_rate: float = 0.0

    # Compression metrics
    mean_compression_ratio: float = 0.0

    # Improvement metrics (OPS vs baseline)
    mae_improvement: float = 0.0  # Positive = OPS is better
    rmse_improvement: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'n_samples': self.n_samples,
            'n_errors': self.n_errors,
            'ops_metrics': {
                'mse': self.mse,
                'rmse': self.rmse,
                'mae': self.mae,
                'correlation': self.correlation,
            },
            'baseline_metrics': {
                'mse': self.baseline_mse,
                'rmse': self.baseline_rmse,
                'mae': self.baseline_mae,
                'correlation': self.baseline_correlation,
            },
            'position_preservation': {
                'within_5_points': self.within_5_points,
                'within_10_points': self.within_10_points,
                'within_20_points': self.within_20_points,
            },
            'audit_metrics': {
                'mean_failure_rate': self.mean_audit_failure_rate,
                'pass_rate': self.audit_pass_rate,
            },
            'compression': {
                'mean_ratio': self.mean_compression_ratio,
            },
            'improvement': {
                'mae_improvement': self.mae_improvement,
                'rmse_improvement': self.rmse_improvement,
            }
        }


def compute_correlation(x: List[float], y: List[float]) -> float:
    """Compute Pearson correlation coefficient."""
    if len(x) < 2 or len(y) < 2:
        return 0.0

    n = len(x)
    mean_x = sum(x) / n
    mean_y = sum(y) / n

    numerator = sum((xi - mean_x) * (yi - mean_y) for xi, yi in zip(x, y))
    denom_x = math.sqrt(sum((xi - mean_x) ** 2 for xi in x))
    denom_y = math.sqrt(sum((yi - mean_y) ** 2 for yi in y))

    if denom_x == 0 or denom_y == 0:
        return 0.0

    return numerator / (denom_x * denom_y)


class ManifestoEvaluator:
    """
    Evaluator for Manifesto RILE prediction results.

    Computes metrics comparing:
    - OPS predictions vs ground truth
    - Baseline predictions vs ground truth
    - OPS vs baseline performance
    """

    def __init__(self, thresholds: List[float] = None):
        """
        Initialize evaluator.

        Args:
            thresholds: RILE point thresholds for "preserved" metrics
        """
        self.thresholds = thresholds or [5.0, 10.0, 20.0]

    def evaluate(
        self,
        results: List[ManifestoResult],
        split_name: str = "test"
    ) -> EvaluationMetrics:
        """
        Compute all metrics for a set of results.

        Args:
            results: List of ManifestoResults
            split_name: Name of the data split

        Returns:
            EvaluationMetrics with all computed metrics
        """
        metrics = EvaluationMetrics()

        # Filter to results without errors and with predictions
        valid_results = [r for r in results if r.error is None and r.predicted_rile is not None]
        metrics.n_samples = len(valid_results)
        metrics.n_errors = len(results) - len(valid_results)

        if not valid_results:
            return metrics

        # Extract values
        ground_truth = [r.ground_truth_rile for r in valid_results]
        predictions = [r.predicted_rile for r in valid_results]
        baselines = [r.baseline_rile for r in valid_results if r.baseline_rile is not None]

        # OPS prediction metrics
        errors = [abs(p - g) for p, g in zip(predictions, ground_truth)]
        squared_errors = [(p - g) ** 2 for p, g in zip(predictions, ground_truth)]

        metrics.mae = sum(errors) / len(errors)
        metrics.mse = sum(squared_errors) / len(squared_errors)
        metrics.rmse = math.sqrt(metrics.mse)
        metrics.correlation = compute_correlation(predictions, ground_truth)

        # Position preservation thresholds
        metrics.within_5_points = sum(1 for e in errors if e <= 5) / len(errors) * 100
        metrics.within_10_points = sum(1 for e in errors if e <= 10) / len(errors) * 100
        metrics.within_20_points = sum(1 for e in errors if e <= 20) / len(errors) * 100

        # Baseline metrics
        if baselines and len(baselines) == len(valid_results):
            baseline_errors = [abs(b - g) for b, g in zip(baselines, ground_truth)]
            baseline_squared = [(b - g) ** 2 for b, g in zip(baselines, ground_truth)]

            metrics.baseline_mae = sum(baseline_errors) / len(baseline_errors)
            metrics.baseline_mse = sum(baseline_squared) / len(baseline_squared)
            metrics.baseline_rmse = math.sqrt(metrics.baseline_mse)
            metrics.baseline_correlation = compute_correlation(baselines, ground_truth)

            # Improvement metrics (positive = OPS is better)
            metrics.mae_improvement = metrics.baseline_mae - metrics.mae
            metrics.rmse_improvement = metrics.baseline_rmse - metrics.rmse

        # Audit metrics
        audit_results = [r for r in valid_results if r.audit_report is not None]
        if audit_results:
            metrics.mean_audit_failure_rate = sum(r.audit_failure_rate for r in audit_results) / len(audit_results)
            metrics.audit_pass_rate = sum(1 for r in audit_results if r.audit_passed) / len(audit_results) * 100

        # Compression metrics
        compression_results = [r for r in valid_results if r.compression_ratio > 0]
        if compression_results:
            metrics.mean_compression_ratio = sum(r.compression_ratio for r in compression_results) / len(compression_results)

        return metrics

    def error_analysis(
        self,
        results: List[ManifestoResult]
    ) -> Dict[str, Any]:
        """
        Analyze errors by different dimensions.

        Args:
            results: List of ManifestoResults

        Returns:
            Dictionary with error breakdowns
        """
        valid_results = [r for r in results if r.error is None and r.predicted_rile is not None]

        if not valid_results:
            return {}

        # By country
        by_country = {}
        for r in valid_results:
            if r.country not in by_country:
                by_country[r.country] = []
            by_country[r.country].append(r.prediction_error)

        country_mae = {
            country: sum(errors) / len(errors)
            for country, errors in by_country.items()
        }

        # By year decade
        by_decade = {}
        for r in valid_results:
            decade = (r.year // 10) * 10
            if decade not in by_decade:
                by_decade[decade] = []
            by_decade[decade].append(r.prediction_error)

        decade_mae = {
            decade: sum(errors) / len(errors)
            for decade, errors in by_decade.items()
        }

        # By ground truth position (left/center/right)
        by_position = {'left': [], 'center': [], 'right': []}
        for r in valid_results:
            if r.ground_truth_rile < -20:
                by_position['left'].append(r.prediction_error)
            elif r.ground_truth_rile > 20:
                by_position['right'].append(r.prediction_error)
            else:
                by_position['center'].append(r.prediction_error)

        position_mae = {
            pos: sum(errors) / len(errors) if errors else 0.0
            for pos, errors in by_position.items()
        }

        # Worst predictions
        sorted_by_error = sorted(valid_results, key=lambda r: r.prediction_error or 0, reverse=True)
        worst_5 = [
            {
                'id': r.manifesto_id,
                'party': r.party_name,
                'ground_truth': r.ground_truth_rile,
                'predicted': r.predicted_rile,
                'error': r.prediction_error,
            }
            for r in sorted_by_error[:5]
        ]

        return {
            'by_country': country_mae,
            'by_decade': decade_mae,
            'by_position': position_mae,
            'worst_predictions': worst_5,
            'total_samples': len(valid_results),
        }

    def generate_report(
        self,
        results: List[ManifestoResult],
        split_name: str = "test",
        output_path: Optional[Path] = None
    ) -> str:
        """
        Generate a text report of evaluation results.

        Args:
            results: List of ManifestoResults
            split_name: Name of the data split
            output_path: Optional path to save report

        Returns:
            Report as string
        """
        metrics = self.evaluate(results, split_name)
        analysis = self.error_analysis(results)

        lines = [
            "=" * 60,
            f"Manifesto RILE Evaluation Report",
            f"Split: {split_name}",
            f"Generated: {datetime.now().isoformat()}",
            "=" * 60,
            "",
            "SUMMARY",
            "-" * 40,
            f"Samples: {metrics.n_samples}",
            f"Errors: {metrics.n_errors}",
            "",
            "OPS PREDICTION METRICS",
            "-" * 40,
            f"MAE:  {metrics.mae:.2f} RILE points",
            f"RMSE: {metrics.rmse:.2f} RILE points",
            f"Correlation: {metrics.correlation:.3f}",
            "",
            "POSITION PRESERVATION",
            "-" * 40,
            f"Within 5 points:  {metrics.within_5_points:.1f}%",
            f"Within 10 points: {metrics.within_10_points:.1f}%",
            f"Within 20 points: {metrics.within_20_points:.1f}%",
            "",
        ]

        if metrics.baseline_mae > 0:
            lines.extend([
                "BASELINE METRICS (Full Text)",
                "-" * 40,
                f"MAE:  {metrics.baseline_mae:.2f} RILE points",
                f"RMSE: {metrics.baseline_rmse:.2f} RILE points",
                f"Correlation: {metrics.baseline_correlation:.3f}",
                "",
                "IMPROVEMENT (OPS vs Baseline)",
                "-" * 40,
                f"MAE improvement:  {metrics.mae_improvement:+.2f} points",
                f"RMSE improvement: {metrics.rmse_improvement:+.2f} points",
                "",
            ])

        lines.extend([
            "AUDIT METRICS",
            "-" * 40,
            f"Mean failure rate: {metrics.mean_audit_failure_rate:.2%}",
            f"Audit pass rate:   {metrics.audit_pass_rate:.1f}%",
            "",
            "COMPRESSION",
            "-" * 40,
            f"Mean ratio: {metrics.mean_compression_ratio:.1f}x",
            "",
        ])

        if analysis:
            lines.extend([
                "ERROR ANALYSIS BY COUNTRY",
                "-" * 40,
            ])
            for country, mae in sorted(analysis.get('by_country', {}).items()):
                lines.append(f"  {country}: {mae:.2f}")

            lines.extend([
                "",
                "ERROR ANALYSIS BY DECADE",
                "-" * 40,
            ])
            for decade, mae in sorted(analysis.get('by_decade', {}).items()):
                lines.append(f"  {decade}s: {mae:.2f}")

            lines.extend([
                "",
                "ERROR BY POSITION",
                "-" * 40,
            ])
            for pos, mae in analysis.get('by_position', {}).items():
                lines.append(f"  {pos}: {mae:.2f}")

            lines.extend([
                "",
                "WORST PREDICTIONS",
                "-" * 40,
            ])
            for item in analysis.get('worst_predictions', []):
                lines.append(
                    f"  {item['party']}: predicted {item['predicted']:.1f}, "
                    f"actual {item['ground_truth']:.1f} (error: {item['error']:.1f})"
                )

        lines.append("")
        lines.append("=" * 60)

        report = "\n".join(lines)

        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(report)

        return report


def save_results(
    results: List[ManifestoResult],
    output_path: Path,
    include_trees: bool = False
) -> None:
    """
    Save results to JSON file.

    Args:
        results: List of ManifestoResults
        output_path: Path to save JSON
        include_trees: Whether to include tree structures (large)
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    data = {
        'generated_at': datetime.now().isoformat(),
        'n_results': len(results),
        'results': []
    }

    for r in results:
        result_dict = {
            'manifesto_id': r.manifesto_id,
            'party_name': r.party_name,
            'country': r.country,
            'year': r.year,
            'ground_truth_rile': r.ground_truth_rile,
            'predicted_rile': r.predicted_rile,
            'baseline_rile': r.baseline_rile,
            'prediction_error': r.prediction_error,
            'baseline_error': r.baseline_error,
            'tree_height': r.tree_height,
            'tree_nodes': r.tree_nodes,
            'tree_leaves': r.tree_leaves,
            'audit_passed': r.audit_passed,
            'audit_failure_rate': r.audit_failure_rate,
            'summary_length': r.summary_length,
            'original_length': r.original_length,
            'compression_ratio': r.compression_ratio,
            'error': r.error,
            'left_indicators': r.left_indicators,
            'right_indicators': r.right_indicators,
            'reasoning': r.reasoning,
            # Include final_summary and chunks for training/resume
            'final_summary': getattr(r, 'final_summary', None),
            'chunks': getattr(r, 'chunks', None),
        }
        data['results'].append(result_dict)

    with open(output_path, 'w') as f:
        json.dump(data, f, indent=2)


def load_results(input_path: Path) -> List[ManifestoResult]:
    """
    Load results from JSON file.

    Args:
        input_path: Path to JSON file

    Returns:
        List of ManifestoResults
    """
    with open(input_path) as f:
        data = json.load(f)

    results = []
    for r in data['results']:
        result = ManifestoResult(
            manifesto_id=r['manifesto_id'],
            party_name=r['party_name'],
            country=r['country'],
            year=r['year'],
            ground_truth_rile=r['ground_truth_rile'],
            predicted_rile=r.get('predicted_rile'),
            baseline_rile=r.get('baseline_rile'),
            tree_height=r.get('tree_height'),
            tree_nodes=r.get('tree_nodes'),
            tree_leaves=r.get('tree_leaves'),
            audit_passed=r.get('audit_passed', True),
            audit_failure_rate=r.get('audit_failure_rate', 0.0),
            summary_length=r.get('summary_length', 0),
            original_length=r.get('original_length', 0),
            compression_ratio=r.get('compression_ratio', 1.0),
            error=r.get('error'),
            left_indicators=r.get('left_indicators', ''),
            right_indicators=r.get('right_indicators', ''),
            reasoning=r.get('reasoning', ''),
            final_summary=r.get('final_summary', ''),
            chunks=r.get('chunks', []),
        )
        results.append(result)

    return results
