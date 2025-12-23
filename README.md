# ThinkingTrees: Oracle-Preserving Summarization (OPS)

An implementation of hierarchical summarization with audit capabilities, designed to preserve critical information through recursive compression while maintaining verifiability.

## Project Overview

ThinkingTrees builds **OPS (Oracle-Preserving Summarization)** trees that:
1. Recursively summarize documents from leaves (chunks) to root
2. Preserve task-critical information defined by a "rubric"
3. Enable probabilistic auditing to detect information loss
4. Support human-in-the-loop verification and optimization

## Architecture

```
ThinkingTrees/
├── config/
│   ├── settings.yaml          # Legacy vLLM + generation defaults
│   ├── training.yaml          # Training hyperparameters, RNG seeds, artifacts
│   ├── inference.yaml         # Runtime/generation settings for building trees
│   └── audit.yaml             # Audit sampling policies and output paths
├── data/
│   ├── raw/                   # Input documents (PDF, text, etc.)
│   ├── processed/             # Chunked document JSONs
│   └── trees/                 # Serialized OPS trees
├── src/
│   ├── core/
│   │   ├── llm_client.py      # LLM wrapper (adapted from OmniThink)
│   │   ├── signatures.py      # DSPy signatures for summarization/judging
│   │   └── data_models.py     # OPSNode and OPSTree classes
│   ├── preprocessing/
│   │   └── chunker.py         # Text chunking (adapted from LangExtract)
│   ├── ops_engine/
│   │   ├── builder.py         # Tree construction (leaves → root)
│   │   ├── auditor.py         # Probabilistic audit logic
│   │   └── optimizer.py       # Bootstrap optimization from failures
│   └── utils/
│       └── visualization.py   # Tree visualization and debugging
├── tests/
│   ├── core/
│   │   ├── test_data_models.py
│   │   └── test_llm_client.py
│   ├── preprocessing/
│   │   └── test_chunker.py
│   ├── ops_engine/
│   │   ├── test_builder.py
│   │   └── test_auditor.py
│   └── conftest.py            # Shared pytest fixtures
├── doc/
│   ├── gemini_implementation_plan.txt  # Original implementation plan
│   ├── architecture.md        # Detailed architecture documentation
│   ├── test_plan.md           # Test-driven development plan
│   └── api_reference.md       # API documentation
├── main.py                    # Entry point
└── requirements.txt           # Dependencies
```

## File Mapping: Source → ThinkingTrees

| ThinkingTrees File | Source | Action |
|-------------------|--------|--------|
| `src/core/llm_client.py` | OmniThink `src/tools/lm.py` | ADAPT - Simplified DSPy/OpenAI wrapper |
| `src/core/signatures.py` | OmniThink patterns | ADAPT - New DSPy signatures for OPS |
| `src/core/data_models.py` | New | CREATE - OPSNode/OPSTree classes |
| `src/preprocessing/chunker.py` | LangExtract `langextract/chunking.py` | ADAPT - Simplified chunking |
| `src/ops_engine/builder.py` | New + OmniThink patterns | CREATE - Tree construction |
| `src/ops_engine/auditor.py` | New | CREATE - Audit logic |
| `src/ops_engine/optimizer.py` | OmniThink/DSPy | ADAPT - Bootstrap optimization |
| `src/utils/visualization.py` | New | CREATE - Debug/visualization tools |

## Core Concepts

### OPSNode
The fundamental building block of the summarization tree:
```python
@dataclass
class OPSNode:
    id: str                          # Unique identifier
    level: int                       # 0 = leaf, higher = more summarized
    raw_text_span: Optional[str]     # Original text (for leaves)
    summary: str                     # The summary at this node
    left_child: Optional[OPSNode]    # Left child (binary tree)
    right_child: Optional[OPSNode]   # Right child
    audit_passed: bool               # Did this node pass audit?
    discrepancy_score: float         # 0.0 = perfect, 1.0 = total loss
```

### Key Operations

1. **RecursiveSummary**: Compress content while preserving rubric-defined information
2. **OracleJudge**: Compare two inputs to detect information drift
3. **Audit**: Sample nodes to verify information preservation

## Development Phases

### Phase 1: Foundation (Current)
- [x] Project structure
- [ ] Core data models (OPSNode, OPSTree)
- [ ] Basic text chunking
- [ ] Tree builder (bottom-up construction)

### Phase 2: LLM Integration
- [ ] LLM client with DSPy
- [ ] RecursiveSummary signature
- [ ] Basic summarization pipeline

### Phase 3: Audit & Optimization
- [ ] OracleJudge signature
- [ ] Probabilistic auditor
- [ ] Human review queue
- [ ] Bootstrap optimizer

### Phase 4: Production
- [ ] Visualization tools
- [ ] CLI interface
- [ ] Documentation

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run tests
pytest tests/ -v

# Train preference/distillation pipeline (produces checkpoints + metadata)
python main.py train --config config/training.yaml

# Build a tree from a document
python main.py infer --input data/raw/document.txt --config config/inference.yaml --output data/trees/document.txt

# Audit an existing tree and export a report
python main.py audit --input data/trees/document.txt --config config/audit.yaml --output experiments/audit/report.yaml
```

For a breakdown of expected inputs/outputs per mode and which configs control them, see [docs/ARTIFACTS.md](docs/ARTIFACTS.md).

## Testing Philosophy

We follow Test-Driven Development (TDD):
1. Write tests first that define expected behavior
2. Implement minimal code to pass tests
3. Refactor while maintaining test coverage

Key test categories:
- **Unit tests**: Individual components (nodes, chunks, signatures)
- **Integration tests**: Component interactions (builder + chunker)
- **Property tests**: Invariants (tree structure, summarization properties)

## Dependencies

- `dspy-ai`: Structured LLM prompting
- `pytest`: Testing framework
- `more-itertools`: Iteration utilities (from LangExtract)
- `pyyaml`: Configuration

## References

- Original Implementation Plan: [doc/gemini_implementation_plan.txt](doc/gemini_implementation_plan.txt)
- OmniThink: Hierarchical article generation patterns
- LangExtract: Document chunking and extraction
