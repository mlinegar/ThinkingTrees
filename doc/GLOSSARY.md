# ThinkingTrees Glossary

This glossary defines the key terms and concepts used throughout the ThinkingTrees codebase.

---

## Core Concepts

### OPS (Oracle-Preserving Summarization)
The core algorithm of ThinkingTrees. OPS builds hierarchical summaries (trees) while preserving task-specific oracle information through each summarization level.

### Oracle
A function that measures how well a summary preserves task-relevant information. The oracle produces reference scores (labels) that we optimize against.

**Example**: For RILE scoring, the oracle extracts political positioning scores from text.

### The Three Laws
The fundamental properties that OPS summaries must satisfy:

1. **Sufficiency Law**: A summary must preserve enough information for the oracle to produce the same score as the original text.

2. **Idempotence Law**: Re-summarizing a summary should produce an equivalent result (no drift over iterations).

3. **Merge Law**: When merging two summaries, the result must preserve the oracle-relevant information from both inputs.

---

## Tree Structure

### Node
A single element in the summarization tree. Contains:
- `content`: The original text (for leaves) or None (for internal nodes)
- `summary`: The summarized text
- `level`: Distance from leaves (0 = leaf, higher = closer to root)
- `children`: Child nodes (empty for leaves)

### Leaf
A node at level 0, created directly from a document chunk. Leaves contain the original chunk text.

### Tree
The complete hierarchical structure with a root node and all descendants.

### Chunk
A segment of the original document, created by the chunking algorithm. Chunks become leaf nodes.

---

## Strategies

### SummarizationStrategy
An abstract interface for how to perform summarization operations. Implementations include:

- **BatchedStrategy**: Uses AsyncBatchLLMClient for high-throughput batched requests
- **DSPyStrategy**: Uses DSPy modules for optimization/training
- **TournamentStrategy**: Generates multiple candidates and selects the best via tournament

### Tournament Selection
A process where multiple candidate summaries compete in pairwise comparisons. The winner of each match advances until a single champion remains. Provides quality improvement at the cost of more LLM calls.

---

## Preference Learning

### Preference Pair
A comparison between two summaries (A and B) for the same input, with a judgment of which is better.

**Fields**:
- `summary_a`, `summary_b`: The two summaries being compared
- `preferred`: "A", "B", or "tie"
- `reasoning`: Explanation for the preference
- `confidence`: How confident the judge is (0-1)
- `oracle_error_a`, `oracle_error_b`: Optional differences from reference score

### GenRM (Generative Reward Model)
NVIDIA's Qwen3-Nemotron-235B-A22B-GenRM model used for preference judgments. It compares two responses and outputs helpfulness scores (1-5) and a ranking score (1-6).

### Ranking Score (GenRM)
A 1-6 scale indicating relative quality:
- 1 = Response 1 is much better than Response 2
- 2 = Response 1 is better
- 3 = Response 1 is slightly better
- 4 = Response 2 is slightly better
- 5 = Response 2 is better
- 6 = Response 2 is much better

### Preference Derivation
The process of converting oracle scores to preferences. Uses the PreferenceEngine to determine which summary better preserves oracle information.

---

## Training Concepts

### DSPy
A framework for programmatic LLM optimization. Used to optimize prompts and in-context examples automatically.

### GEPA (Generalized Preference Alignment)
An optimization approach that aligns model outputs with preferences through iterative refinement.

### Bootstrap Examples
Training examples that demonstrate good behavior. DSPy uses these to construct optimized prompts.

### Oracle-Derived Preference
A preference derived from oracle scores (labels) rather than a judge's opinion. More reliable than judge preferences because it's based on actual task performance.

---

## Batching & Performance

### AsyncBatchLLMClient
An async client that pools LLM requests and sends them in batches to vLLM for optimal GPU utilization.

### BatchTreeOrchestrator
High-level orchestrator for building multiple trees with optimal batching. Processes all documents level-by-level, returning `BuildResult` objects with full tree structures.

**Location**: `src/core/batch_orchestrator.py`

### Automatic Prefix Caching (APC)
vLLM feature that caches KV cache for repeated prompts. Reduces compute for documents with shared system prompts or contexts.

---

## Tasks

### Task
A document analysis objective (e.g., RILE scoring, sentiment analysis). Defines:
- What oracle to use
- What rubric/context to apply
- How to evaluate results

### RILE (Right-Left) Scoring
A political positioning metric ranging from -100 (left) to +100 (right). Used in the manifesto analysis task.

### Rubric
Instructions that guide the summarization process, specifying what information to preserve.

---

## Code Components

### TreeBuilder
The canonical async builder for single documents. Uses a Strategy for summarization operations.

**Location**: `src/ops_engine/builder.py`

### BatchOrchestrator
Multi-document processing coordinator that manages parallel tree construction.

**Location**: `src/core/batch_processor.py`

### PreferenceEngine
Unified engine for deriving preferences from oracle scores. Handles different derivation strategies.

**Location**: `src/ops_engine/training_framework/preference_engine.py`

### SimilarityScorer
Compares two texts by extracting values and computing similarity. Supports caching to avoid redundant LLM calls.

**Location**: `src/ops_engine/scoring.py`

---

## File Naming Conventions

| Pattern | Purpose |
|---------|---------|
| `*_preference.py` | Preference collection/handling |
| `*_dspy.py` | DSPy module wrappers |
| `*_pipeline.py` | End-to-end processing pipelines |
| `*_strategy.py` | Summarization strategy implementations |

---

## Common Abbreviations

| Abbrev | Full Form |
|--------|-----------|
| OPS | Oracle-Preserving Summarization |
| GenRM | Generative Reward Model |
| RILE | Right-Left (political scale) |
| APC | Automatic Prefix Caching |
| DSPy | Declarative Self-improving Python |
| GEPA | Generalized Preference Alignment |
| KV | Key-Value (cache) |
| vLLM | Virtual LLM (serving framework) |
