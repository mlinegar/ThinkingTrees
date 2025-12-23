# Oracle Ground Truth Preference Pair Collection

This directory contains scripts for collecting preference pairs using oracle ground truth trees for OPS (Oracle-Preserving Summarization) training.

## Overview

The workflow generates hierarchical ground truth trees where each node (chunk and merge) has been scored by a large oracle model. This enables testing all three OPS laws with known ground truth:

- **Sufficiency**: How well does a summary preserve the oracle value of the original chunk?
- **Idempotence**: How stable is the oracle value when re-summarizing (summarize(summarize(x)) ≈ summarize(x))?
- **Merge**: How well does a summary of merged chunks preserve the parent node's oracle value?

## Architecture

### Phase 0: Generate Oracle Ground Truth Trees

**Script**: `generate_oracle_ground_truth.py`

**Input**: Manifesto documents with ground truth RILE scores
**Output**: Hierarchical trees with oracle scores at all levels

**Process**:
1. Load manifesto documents
2. Chunk text into ~4000 character chunks (paragraph-aware)
3. Build hierarchical merge tree (binary merging)
4. Score all nodes (leaves + internal merges) with large oracle model
5. Save ground truth trees to JSON

**Advantages**:
- Ground truth available for all nodes
- Enables all three OPS laws
- Reusable for multiple preference collection runs
- Can be generated once and shared

### Phase 1: Collect Preference Pairs

**Script**: `collect_preferences_with_ground_truth.py`

**Input**: Ground truth trees from Phase 0
**Output**: Preference pairs with GenRM judgments

**Process**:
1. Load pre-generated ground truth trees
2. For each node:
   - Generate k candidate summaries (varying temperature)
   - For idempotence: re-summarize each candidate
   - For merge: use parent node ground truth
3. Use GenRM to compare all pairs (A vs B)
4. Link preferences to ground truth for validation
5. Save preference pairs in PreferencePair format

**Advantages**:
- Rich preference data with reasoning
- Ground truth linkage for validation
- Supports all three OPS laws
- Position bias mitigation (random swapping)
- Compatible with existing training pipeline

## Usage

### Prerequisites

1. **Oracle server running** (large Nemotron model for ground truth generation):
   ```bash
   # Example: Start oracle server on port 8001
   ./scripts/start_oracle_server.sh --port 8001
   ```

2. **GenRM server running** (for preference judgments):
   ```bash
   # Can be same as oracle if using GenRM as oracle
   # Port 8001 by default
   ```

3. **Summarizer server running** (small model for candidate generation):
   ```bash
   # Example: Start summarizer on port 8000
   ./scripts/start_vllm.sh --port 8000 --model nemotron-nano
   ```

### Step 1: Generate Ground Truth Trees

```bash
python experiments/manifesto_rile/generate_oracle_ground_truth.py \
    --oracle-port 8001 \
    --max-documents 100 \
    --chunk-size 4000 \
    --output-dir data/ground_truth
```

**Key Parameters**:
- `--oracle-port`: Port for oracle model server
- `--max-documents`: Limit number of documents (for testing)
- `--chunk-size`: Maximum characters per chunk
- `--train-only`: Only process training split
- `--output-dir`: Where to save ground truth trees

**Output**:
```
data/ground_truth/
├── index.json                          # Dataset index
├── {manifesto_id}_ground_truth.json    # Individual trees
└── generation_stats_*.json             # Statistics
```

**Ground Truth Tree Structure**:
```json
{
  "manifesto_id": "51320_196410",
  "document_rile": 15.5,
  "nodes": {
    "51320_196410_L0_N0": {
      "chunk_id": "51320_196410_L0_N0",
      "level": 0,
      "text": "...",
      "rile_score": 12.3,
      "reasoning": "...",
      "left_indicators": "...",
      "right_indicators": "..."
    }
  },
  "levels": [
    ["51320_196410_L0_N0", "51320_196410_L0_N1", ...],  // Leaf chunks
    ["51320_196410_L1_N0", ...],                        // Merge level 1
    ["51320_196410_L2_N0"]                              // Root
  ]
}
```

### Step 2: Collect Preference Pairs

#### Sufficiency Preferences

```bash
python experiments/manifesto_rile/collect_preferences_with_ground_truth.py \
    --ground-truth-dir data/ground_truth \
    --law-type sufficiency \
    --genrm-port 8001 \
    --summarizer-port 8000 \
    --k-candidates 4 \
    --output-dir data/preferences
```

#### Idempotence Preferences

```bash
python experiments/manifesto_rile/collect_preferences_with_ground_truth.py \
    --ground-truth-dir data/ground_truth \
    --law-type idempotence \
    --genrm-port 8001 \
    --summarizer-port 8000 \
    --k-candidates 4 \
    --output-dir data/preferences
```

#### Merge Preferences

```bash
python experiments/manifesto_rile/collect_preferences_with_ground_truth.py \
    --ground-truth-dir data/ground_truth \
    --law-type merge \
    --genrm-port 8001 \
    --summarizer-port 8000 \
    --k-candidates 4 \
    --output-dir data/preferences
```

**Key Parameters**:
- `--ground-truth-dir`: Directory with ground truth trees from Step 1
- `--law-type`: Which OPS law to collect (sufficiency/idempotence/merge/all)
- `--genrm-port`: Port for GenRM judge server
- `--summarizer-port`: Port for small model server
- `--k-candidates`: Number of summaries per chunk (default: 4)
- `--temperatures`: Temperatures for diverse generation
- `--max-trees`: Limit number of trees to process
- `--max-chunks-per-tree`: Limit chunks per tree

**Output**:
```
data/preferences/
├── preferences_gt_sufficiency_*.json     # Preference pairs
├── dpo_gt_sufficiency_*.json            # DPO format
├── collection_stats_gt_sufficiency_*.json  # Statistics
├── preferences_gt_idempotence_*.json
├── ...
```

### Quick Test

For a quick end-to-end test:

```bash
./experiments/manifesto_rile/test_ground_truth_workflow.sh
```

This will:
1. Generate ground truth for 5 documents
2. Collect sufficiency preferences
3. Collect idempotence preferences
4. Collect merge preferences
5. Show summary statistics

## Data Structures

### ChunkGroundTruth

```python
@dataclass
class ChunkGroundTruth:
    chunk_id: str                   # Unique identifier
    manifesto_id: str               # Source document
    level: int                      # Tree level (0=leaf, 1+=merge)
    text: str                       # Chunk content
    rile_score: float               # Oracle RILE score
    dimension_scores: Dict          # Future: multi-dimensional
    reasoning: str                  # Oracle reasoning
    left_indicators: str            # LEFT signals
    right_indicators: str           # RIGHT signals
    confidence: float               # Oracle confidence
    left_child_id: str              # For merge nodes
    right_child_id: str             # For merge nodes
```

### ManifestoGroundTruthTree

```python
@dataclass
class ManifestoGroundTruthTree:
    manifesto_id: str
    document_text: str
    document_rile: float            # Original RILE
    nodes: Dict[str, ChunkGroundTruth]
    levels: List[List[str]]         # Tree structure
    num_chunks: int
    num_levels: int
    oracle_model: str
```

### PreferencePair (Existing, Reused)

```python
@dataclass
class PreferencePair:
    pair_id: str
    source_example_id: str          # Links to chunk_id
    original_text: str
    rubric: str
    ground_truth_score: float       # From ChunkGroundTruth
    law_type: str                   # sufficiency/idempotence/merge
    summary_a: str
    summary_b: str
    preferred: str                  # A/B/tie
    reasoning: str                  # GenRM reasoning
    confidence: float               # GenRM confidence
    score_estimate_a: float
    score_estimate_b: float
    judge_model: str
    generation_config_a: Dict
    generation_config_b: Dict
```

## Integration with Existing Pipeline

The new ground truth workflow integrates seamlessly:

1. **Reuses existing code**:
   - `PreferencePair` dataclass (no changes)
   - `GenRMJudge` for preference collection
   - `LeafSummarizer` for candidate generation
   - `chunk_text()` for text splitting
   - `create_rile_scorer()` for oracle

2. **Compatible with training**:
   - Output is `PreferenceDataset` format
   - Can use `train_ops_comparison.py` directly
   - Can generate DPO data with `generate_dpo_data.py`

3. **Extensible**:
   - `dimension_scores` field for future multi-dimensional scoring
   - `GroundTruthDataset` for batch operations
   - Tree structure supports custom OPS laws

## Next Steps After Collection

### Train OPS Comparison Module

```bash
python experiments/manifesto_rile/train_ops_comparison.py \
    --preference-data data/preferences/preferences_gt_sufficiency_*.json \
    --law-type sufficiency \
    --budget heavy \
    --output-dir models/ops_comparison
```

### Generate DPO Data

```bash
python experiments/manifesto_rile/generate_dpo_data.py \
    --comparison-module models/ops_comparison/ops_comparison_*.json \
    --summarizer-port 8000 \
    --judge-port 8000 \
    --output-dir data/dpo
```

### Validate Ground Truth

Analyze how well preferences align with oracle ground truth:

```python
from src.ops_engine.training_framework import (
    PreferenceDataset,
    GroundTruthDataset,
)

# Load data
gt_dataset = GroundTruthDataset.load("data/ground_truth")
pref_dataset = PreferenceDataset.load("data/preferences/preferences_gt_*.json")

# Validate: check if GenRM preferences align with oracle errors
for pair in pref_dataset.pairs:
    node = gt_dataset.get_tree(pair.source_example_id).get_node(pair.source_example_id)
    # Compare pair.preferred with oracle error differences
```

## Design Rationale

### Why Hierarchical Trees?

1. **Enables all OPS laws**: Merge nodes have ground truth for merged content
2. **Efficient**: Generate ground truth once, reuse for multiple experiments
3. **Realistic**: Mirrors actual tree-based summarization systems
4. **Validation**: Can test if preferences align with oracle ground truth

### Why Separate Generation and Collection?

1. **Cost**: Oracle scoring is expensive, do it once
2. **Iteration**: Can try different GenRM parameters without re-scoring
3. **Sharing**: Ground truth trees can be shared across experiments
4. **Debugging**: Can inspect ground truth trees independently

### Why GenRM for Preferences?

1. **Reasoning**: GenRM provides explanations for preferences
2. **Quality**: Top-performing generative reward model
3. **NVIDIA design**: Specifically designed for preference pairs
4. **Future-proof**: Can extend to multi-dimensional scoring

## Troubleshooting

### Oracle server not responding
```bash
# Check if server is running
curl http://localhost:8001/v1/models

# Restart server if needed
./scripts/start_oracle_server.sh --port 8001
```

### Out of memory during ground truth generation
```bash
# Reduce chunk size or process fewer documents
python generate_oracle_ground_truth.py \
    --chunk-size 2000 \
    --max-documents 10
```

### GenRM comparisons failing
```bash
# Check GenRM server
curl http://localhost:8001/v1/models

# Try with smaller batch
python collect_preferences_with_ground_truth.py \
    --max-trees 5 \
    --max-chunks-per-tree 2
```

## Files Reference

### New Files

- `src/ops_engine/training_framework/oracle_ground_truth.py` - Data structures
- `experiments/manifesto_rile/generate_oracle_ground_truth.py` - Ground truth generation
- `experiments/manifesto_rile/collect_preferences_with_ground_truth.py` - Preference collection
- `experiments/manifesto_rile/test_ground_truth_workflow.sh` - End-to-end test
- `experiments/manifesto_rile/README_GROUND_TRUTH.md` - This file

### Existing Files (Reused)

- `src/ops_engine/training_framework/preference.py` - PreferencePair
- `src/ops_engine/training_framework/genrm_preference.py` - GenRM judge
- `src/manifesto/batched_pipeline.py` - chunk_text()
- `src/manifesto/position_oracle.py` - create_rile_scorer()
- `src/manifesto/dspy_summarizer.py` - LeafSummarizer

## Future Extensions

### Multi-Dimensional Scoring

The `dimension_scores` field in `ChunkGroundTruth` is a placeholder for future multi-dimensional scoring:

```python
node.dimension_scores = {
    "rile": 15.5,
    "economic": -10.2,
    "social": 25.8,
    "environmental": 5.0,
}
```

This would enable:
- Fine-grained preference collection per dimension
- Multi-objective optimization
- Dimension-specific law testing

### Custom OPS Laws

The tree structure supports custom laws:

```python
# Example: Transitivity law
# summarize(summarize(summarize(x))) ≈ summarize(x)
# Use nodes at level 0, 1, 2 for testing
```

### Curriculum Learning

Ground truth trees enable curriculum learning:

```python
# Start with easy examples (small oracle error)
# Progress to harder examples (large oracle error)
# Order by confidence or error magnitude
```
