# OPS-RILE Optimization Summary

## Run Details
- **Date**: 2025-12-16
- **Run ID**: overnight_20251216_175823
- **Task Model**: Qwen3-30B-A3B-Thinking (port 8000)
- **Auditor Model**: Qwen3-80B-A3B-Instruct (port 8001)
- **Chunk Size**: 2000 characters
- **Samples**: 50 Swedish manifestos (1960-1988)
- **Iterations**: 5

## Convergence Results

| Iteration | MAE | Std Dev | <5pt | <10pt | <20pt |
|-----------|-----|---------|------|-------|-------|
| 1 (baseline) | 47.37 | 26.90 | 8% | 8% | 24% |
| 2 (optimized) | **26.29** | 20.87 | 14% | 20% | 48% |
| 3 | 26.29 | 20.87 | 14% | 20% | 48% |
| 4 | 26.29 | 20.87 | 14% | 20% | 48% |
| 5 | 26.29 | 20.87 | 14% | 20% | 48% |

### Key Findings
- **DSPy optimization reduced MAE by 44%** (47.37 → 26.29)
- **Convergence after 1 optimization pass** - iterations 3-5 showed no further improvement
- **Accuracy within 20 points doubled** (24% → 48%)

## Error Distribution (Final Iteration)

| Percentile | Error (RILE points) |
|------------|---------------------|
| Min | 2.1 |
| 25th | 13.3 |
| Median | 21.8 |
| 75th | 34.1 |
| Max | 104.2 |

## Best Predictions
| Party | Year | Predicted | Actual | Error |
|-------|------|-----------|--------|-------|
| People's Party | 1976 | 0.0 | -2.1 | 2.1 |
| Social Democratic Labour | 1968 | -45.0 | -42.9 | 2.1 |
| Moderate Coalition Party | 1976 | 0.0 | 2.2 | 2.2 |
| Social Democratic Labour | 1970 | -38.3 | -41.2 | 2.9 |
| Moderate Coalition Party | 1985 | 55.3 | 59.8 | 4.5 |

## Worst Predictions
| Party | Year | Predicted | Actual | Error |
|-------|------|-----------|--------|-------|
| People's Party | 1964 | 55.3 | -48.9 | 104.2 |
| Left Communists Party | 1970 | 55.3 | -40.9 | 96.2 |
| Right Party | 1968 | -38.3 | 25.8 | 64.1 |

### Error Analysis
The worst predictions show a pattern:
1. **Sign errors**: Predicting right-wing when actually left-wing (or vice versa)
2. **Placeholder summaries**: 5/50 samples (10%) had >100x compression, indicating the model output template placeholders instead of actual summaries

## Quality Issues Identified

### Placeholder Outputs
Some samples received placeholder text like `[summary2 content]` instead of actual summaries:
- 5 samples with >100x compression ratio
- These defaulted to RILE=0.0 predictions
- Fix implemented: placeholder detection with retry logic

### Compression Statistics
- Median compression: 10.7x (reasonable)
- Problem threshold: >100x compression indicates failed summarization

## Parallel Processing Performance
- **Before**: ~30-60 seconds per sample (sequential)
- **After**: All 50 samples processed in <2 seconds (parallel batching)
- **Speedup**: ~150x for iteration execution

## Oracle Function Approximation

The oracle approximation learns to predict which RILE predictions are likely wrong.

### Training Data Generated
- **159 total examples** extracted from RILE predictions
- **115 positive** (high error >30 points) - true violations
- **44 negative** (low error <10 points) - good predictions
- Balance ratio: 0.38

### Training Results
- Model: Qwen3-30B-A3B-Thinking
- Training time: ~1.5 minutes
- Bootstrapped 2 full traces from 3 examples
- Checkpoint saved: `data/oracle_func_checkpoints/oracle_func_model_*.json`

### Oracle Statistics
| Metric | Positive (Bad) | Negative (Good) |
|--------|----------------|-----------------|
| Count | 115 | 44 |
| Avg Discrepancy | 0.502 | 0.042 |

### Usage
```python
# Note: LearnedOracleFunc was part of an experimental optimization path
# and has been deprecated. Use the training pipeline instead:
from src.training import JudgeOptimizer

# See src/training/judge_optimization.py for the current approach
```

## Next Steps
1. Run with placeholder detection fix to eliminate template outputs
2. Investigate sign-flip errors (right↔left confusion)
3. Test on larger sample sizes (100-500 manifestos)
4. Use oracle approximation to auto-review low-confidence predictions
5. Consider additional DSPy optimization strategies (MIPROv2)

## Files
- Results: `data/results/manifesto_rile/overnight_20251216_175823/`
- Pipeline: `experiments/manifesto_rile/run_with_optimization.py`
- Overnight script: `scripts/run_overnight_test.sh`
- Oracle test: `experiments/manifesto_rile/test_oracle_approximation.py`
- Oracle training data: `data/results/manifesto_rile/overnight_20251216_175823/oracle_training_data.json`
- Oracle checkpoint: `data/oracle_func_checkpoints/`
