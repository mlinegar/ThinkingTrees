"""
Validate OPS theoretical guarantees using manifesto RILE pipeline.

Uses existing ManifestoPipeline/ManifestoPipelineWithStrategy.
Measures: oracle error vs n_leaves to test log(n) bound.

Theory (from Part II of the analysis):
    For a tree with n leaves, if all OPS laws hold with tolerance δ:
    |f̃(root) - f̃(full_document)| ≤ (1 + log₂ n) × δ

This script empirically validates this bound by:
1. Running manifestos through the pipeline
2. Comparing predicted RILE vs ground truth RILE
3. Checking if error ≤ (1 + log₂ n_leaves) × δ
"""
import argparse
import math
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional
import pandas as pd

import dspy

from src.tasks.manifesto.data_loader import ManifestoDataset
from src.tasks.manifesto.pipeline import ManifestoPipeline
from src.tasks.manifesto.constants import RILE_MIN, RILE_MAX, RILE_RANGE
from src.preprocessing.chunker import chunk_for_ops
from src.config.dspy_config import configure_dspy, create_vllm_lm


@dataclass
class ValidationResult:
    """Result for a single manifesto validation."""
    manifesto_id: str
    n_leaves: int
    text_length: int              # Character count
    rile_ground_truth: float      # From Manifesto Project (expert)
    rile_predicted: float         # From pipeline (LLM on summary)
    error_normalized: float       # |pred - truth| / RILE_RANGE


def run_validation(
    port: int = 8001,
    countries: Optional[List[int]] = None,
    max_samples: Optional[int] = None,
    output_dir: str = "outputs/validation",
    chunk_size: int = 2000,
    delta: float = 0.05,  # 5% tolerance = 10 RILE points
) -> pd.DataFrame:
    """
    Run validation using existing ManifestoPipeline.

    For each manifesto:
    1. Run through pipeline (chunks, summarizes, merges, scores)
    2. Compare pipeline's RILE prediction vs ground truth
    3. Record n_leaves to test logarithmic bound

    Args:
        port: vLLM server port
        countries: List of CMP country codes (default: [11] = Sweden)
        max_samples: Limit number of samples (None = all)
        output_dir: Where to save results
        chunk_size: Max chunk size in characters
        delta: Tolerance for bound check (5% = 10 RILE points)

    Returns:
        DataFrame with validation results
    """
    # Configure DSPy with vLLM
    lm = create_vllm_lm(port=port)
    configure_dspy(lm=lm)
    print(f"Configured DSPy with LM on port {port}")

    # Load data
    countries = countries or [11]  # Sweden default
    dataset = ManifestoDataset(countries=countries, require_text=True)
    print(f"Loaded dataset: {len(dataset)} manifestos from countries {countries}")

    # Create pipeline (uses existing implementation)
    pipeline = ManifestoPipeline(chunk_size=chunk_size)

    results: List[ValidationResult] = []
    samples = list(dataset)
    if max_samples:
        samples = samples[:max_samples]

    print(f"\nProcessing {len(samples)} manifestos...")
    print("-" * 60)

    for i, sample in enumerate(samples):
        print(f"[{i+1}/{len(samples)}] {sample.manifesto_id} ({sample.party_abbrev}, {sample.year})")

        try:
            # Run existing pipeline
            prediction = pipeline(text=sample.text)

            # The pipeline's forward() gives us:
            # - prediction.score (normalized 0-1)
            # - We need to compare vs sample.rile (ground truth -100 to +100)

            # Denormalize prediction back to -100 to +100 scale
            pred_rile = prediction.score * RILE_RANGE + RILE_MIN

            # Compute normalized error
            error = abs(pred_rile - sample.rile) / RILE_RANGE

            # Get n_leaves (same chunking as pipeline uses)
            chunks = chunk_for_ops(sample.text, max_chars=chunk_size, strategy="sentence")
            n_leaves = len(chunks)

            print(f"    n_leaves={n_leaves}, ground_truth={sample.rile:.1f}, "
                  f"predicted={pred_rile:.1f}, error={error:.4f}")

            results.append(ValidationResult(
                manifesto_id=sample.manifesto_id,
                n_leaves=n_leaves,
                text_length=len(sample.text),
                rile_ground_truth=sample.rile,
                rile_predicted=pred_rile,
                error_normalized=error,
            ))

        except Exception as e:
            print(f"    ERROR: {e}")
            continue

    if not results:
        print("No results to analyze!")
        return pd.DataFrame()

    # Convert to DataFrame
    df = pd.DataFrame([vars(r) for r in results])

    # Save raw results
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path / "validation_results.csv", index=False)

    # Analyze: Test logarithmic bound
    df['log_n'] = df['n_leaves'].apply(lambda n: math.log2(n) if n > 1 else 1)
    df['theoretical_bound'] = (1 + df['log_n']) * delta
    df['bound_satisfied'] = df['error_normalized'] <= df['theoretical_bound']

    # Print summary
    print("\n" + "=" * 60)
    print("VALIDATION SUMMARY")
    print("=" * 60)
    print(f"Samples processed: {len(df)}")
    print(f"Mean n_leaves: {df['n_leaves'].mean():.1f}")
    print(f"Mean text length: {df['text_length'].mean():.0f} chars")
    print()
    print(f"RILE Error (normalized 0-1 scale):")
    print(f"  Mean error: {df['error_normalized'].mean():.4f}")
    print(f"  Median error: {df['error_normalized'].median():.4f}")
    print(f"  Max error: {df['error_normalized'].max():.4f}")
    print(f"  Std error: {df['error_normalized'].std():.4f}")
    print()
    print(f"RILE Error (raw -100 to +100 scale):")
    print(f"  Mean error: {df['error_normalized'].mean() * RILE_RANGE:.1f} points")
    print(f"  Max error: {df['error_normalized'].max() * RILE_RANGE:.1f} points")
    print()
    print(f"Theoretical Bound Test (δ = {delta}):")
    print(f"  Bound: error ≤ (1 + log₂ n) × {delta}")
    print(f"  Bound satisfaction rate: {df['bound_satisfied'].mean():.2%}")
    print()
    print(f"Correlation Analysis:")
    print(f"  Correlation(log_n, error): {df['log_n'].corr(df['error_normalized']):.4f}")
    print(f"  Correlation(text_length, error): {df['text_length'].corr(df['error_normalized']):.4f}")

    # Identify outliers (worst cases)
    if len(df) >= 5:
        print()
        print("Worst 5 cases (highest error):")
        worst = df.nlargest(5, 'error_normalized')[
            ['manifesto_id', 'n_leaves', 'rile_ground_truth', 'rile_predicted', 'error_normalized']
        ]
        for _, row in worst.iterrows():
            print(f"  {row['manifesto_id']}: "
                  f"n={row['n_leaves']}, "
                  f"truth={row['rile_ground_truth']:.1f}, "
                  f"pred={row['rile_predicted']:.1f}, "
                  f"err={row['error_normalized']:.4f}")

    # Save full analysis
    df.to_csv(output_path / "validation_analysis.csv", index=False)
    print()
    print(f"Results saved to: {output_path}")

    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Validate OPS theoretical bounds on manifesto RILE pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Quick test (Sweden, 10 samples)
    python validate_theory.py --countries 11 --max-samples 10

    # Full validation (Sweden, Germany, UK)
    python validate_theory.py --countries 11 41 51

    # With custom tolerance
    python validate_theory.py --countries 11 --delta 0.10
        """
    )
    parser.add_argument("--port", type=int, default=8001,
                        help="vLLM server port (default: 8001)")
    parser.add_argument("--countries", type=int, nargs="+", default=[11],
                        help="CMP country codes (default: 11 = Sweden)")
    parser.add_argument("--max-samples", type=int, default=None,
                        help="Limit samples (default: all)")
    parser.add_argument("--output-dir", default="outputs/validation",
                        help="Output directory (default: outputs/validation)")
    parser.add_argument("--chunk-size", type=int, default=2000,
                        help="Chunk size in chars (default: 2000)")
    parser.add_argument("--delta", type=float, default=0.05,
                        help="Tolerance for bound (default: 0.05 = 5%%)")

    args = parser.parse_args()
    run_validation(
        port=args.port,
        countries=args.countries,
        max_samples=args.max_samples,
        output_dir=args.output_dir,
        chunk_size=args.chunk_size,
        delta=args.delta,
    )
