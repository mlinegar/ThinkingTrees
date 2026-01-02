# ThinkingTrees Optimization Guide

This guide explains the three optimization paths available in ThinkingTrees and when to use each one.

## Overview

ThinkingTrees provides three distinct optimization strategies:

1. **GenRM Tournament** (Runtime Selection) - Improves summary quality without training
2. **Oracle Approximation** (Task-Specific Training) - Learns to approximate ground truth scoring
3. **Tournament of Tournaments** (Judge Training) - Improves the GenRM judge itself

```
                            ┌─────────────────────────────┐
                            │     Your Documents          │
                            └─────────────┬───────────────┘
                                          │
              ┌───────────────────────────┼───────────────────────────┐
              │                           │                           │
              ▼                           ▼                           ▼
    ┌─────────────────┐       ┌─────────────────┐       ┌─────────────────┐
    │  Path 1: GenRM  │       │ Path 2: Oracle  │       │ Path 3: Judge   │
    │   Tournament    │       │  Approximation  │       │   Training      │
    │                 │       │                 │       │                 │
    │ No training     │       │ Task-specific   │       │ Meta-learning   │
    │ Best-of-k       │       │ Learn oracle    │       │ Better judge    │
    │ selection       │       │ from data       │       │ discrimination  │
    └─────────────────┘       └─────────────────┘       └─────────────────┘
```

---

## Path 1: GenRM Tournament (Runtime Selection)

**Purpose**: Generate better summaries without any training by selecting the best among k candidates.

**How it works**:
1. Generate k candidate summaries at high temperature (default k=4)
2. Run pairwise tournament comparisons using GenRM judge
3. Select the winner through elimination
4. Collect preferences as a FREE byproduct (useful for Path 2 or 3)

**When to use**:
- You want better summaries immediately with no training
- You're willing to trade speed for quality (k times more LLM calls)
- You want to collect preference data for later training
- You're doing inference and have spare compute

**Code**:
```python
from src.core.strategy import TournamentStrategy, BatchedStrategy

# Create base strategy
base_strategy = BatchedStrategy(client=batch_client, rubric=rubric)

# Wrap with tournament selection
tournament = TournamentStrategy(
    base_strategy=base_strategy,
    judge=genrm_judge,
    num_candidates=4,  # Generate 4 candidates, pick best
)

# Build tree with tournament selection
tree = await tree_builder.build(text, strategy=tournament)

# Preference pairs collected automatically
preferences = tournament.get_collected_preferences()
```

**Files**:
- `src/core/strategy.py` - TournamentStrategy class
- `src/ops_engine/training_framework/genrm_preference.py` - GenRMJudge

---

## Path 2: Oracle Approximation (Task-Specific Training)

**Purpose**: Train the summarizer to approximate task-specific ground truth (e.g., RILE scores).

**How it works**:
1. Generate candidate summaries for training documents
2. Score each candidate with the real oracle (e.g., RILE scorer)
3. Derive preferences from oracle errors (lower error = better)
4. Optimize summarizer using DSPy/GEPA to minimize oracle error

**When to use**:
- You have labeled data with ground truth scores
- Your task has a well-defined oracle function
- You want task-specific optimization
- You're doing training/optimization

**Code**:
```python
from src.ops_engine.training_framework.oracle_preference import OraclePreferenceCollector
from src.tasks.manifesto import create_rile_scorer

# Create oracle scorer
oracle = create_rile_scorer()

# Collect preferences from oracle
collector = OraclePreferenceCollector(
    summarizer=summarizer,
    oracle=oracle,
    num_candidates=4,
)

# Process documents and collect preferences
preferences = collector.collect(training_documents)

# Optimize with DSPy
optimizer = dspy.MIPROv2(metric=oracle_metric)
optimized = optimizer.compile(summarizer, trainset=preferences)
```

**Files**:
- `src/ops_engine/training_framework/oracle_preference.py` - OraclePreferenceCollector
- `src/tasks/manifesto/task.py` - ManifestoTask with oracle creation
- `src/training/run_pipeline.py` - Unified training entry point

---

## Path 3: Tournament of Tournaments (Judge Training)

**Purpose**: Meta-optimize the GenRM judge to better discriminate between good and bad summaries.

**How it works**:
1. Collect preferences WITH ground truth scores (from Path 2)
2. Create training data for the judge: "which summary is closer to oracle?"
3. Optimize GenRM prompts using DSPy to predict correct preferences
4. Use the improved judge in Path 1 for better tournament selection

**When to use**:
- You've run Path 2 and have oracle-grounded preferences
- Default GenRM doesn't align well with your task
- You want to improve inference quality for future runs
- You're willing to do meta-optimization

**Code**:
```python
from src.training.judge_optimization import JudgeOptimizer

# Load oracle-grounded preferences (from Path 2)
preferences = load_preferences("data/oracle_preferences.json")

# Create judge optimizer
optimizer = JudgeOptimizer(
    genrm_module=genrm_comparison_module,
    trainset=preferences,
)

# Optimize the judge
optimized_judge = optimizer.optimize()

# Use optimized judge in tournament
tournament = TournamentStrategy(
    base_strategy=base_strategy,
    judge=optimized_judge,  # Now better at your task!
)
```

**Files**:
- `src/training/judge_optimization.py` - JudgeOptimizer
- `src/ops_engine/training_framework/genrm_dspy.py` - GenRMComparisonModule
- `experiments/manifesto_rile/optimize_judge.py` - Judge optimization script

---

## Decision Flowchart

```
                    ┌─────────────────────────┐
                    │   What do you want?     │
                    └───────────┬─────────────┘
                                │
            ┌───────────────────┴───────────────────┐
            │                                       │
            ▼                                       ▼
   ┌─────────────────┐                   ┌─────────────────┐
   │ Better summaries │                   │ Train a model   │
   │   RIGHT NOW      │                   │ for my task     │
   └────────┬────────┘                   └────────┬────────┘
            │                                      │
            ▼                                      │
   ┌─────────────────┐                            │
   │  Use Path 1:    │                            │
   │  GenRM Tournament│                            │
   │                  │                            │
   │ --tournament     │                            │
   └─────────────────┘                            │
                                                   │
                    ┌──────────────────────────────┤
                    │                              │
                    ▼                              ▼
         ┌─────────────────┐            ┌─────────────────┐
         │ Do you have     │            │ Is default      │
         │ labeled data?   │            │ GenRM working?  │
         └────────┬────────┘            └────────┬────────┘
                  │                              │
         ┌────────┴────────┐            ┌────────┴────────┐
         │                 │            │                 │
         ▼                 ▼            ▼                 ▼
     ┌───────┐         ┌───────┐   ┌───────┐         ┌───────┐
     │  Yes  │         │  No   │   │  Yes  │         │  No   │
     └───┬───┘         └───┬───┘   └───┬───┘         └───┬───┘
         │                 │           │                 │
         ▼                 ▼           ▼                 ▼
    ┌──────────┐    ┌──────────┐  ┌──────────┐    ┌──────────┐
    │ Path 2:  │    │ Path 1:  │  │ Done!    │    │ Path 3:  │
    │ Oracle   │    │ Collect  │  │ Use as   │    │ Train    │
    │ Training │    │ prefs    │  │ is       │    │ Judge    │
    └──────────┘    │ then go  │  └──────────┘    └──────────┘
                    │ to Path 2│
                    └──────────┘
```

---

## Combined Workflow (Full Pipeline)

For maximum quality, combine all three paths:

```
1. Initial Setup
   └── Collect oracle preferences (Path 2)
       └── Save to data/oracle_preferences.json

2. Judge Training
   └── Train judge on oracle preferences (Path 3)
       └── Save optimized judge prompts

3. Production Inference
   └── Use optimized judge in tournament (Path 1)
       └── Get better summaries with aligned preferences
```

**Unified training pipeline**:
```bash
# Run the full pipeline
python -m src.training.run_pipeline \
    --task manifesto_rile \
    --phase all \
    --samples 100
```

---

## Performance Considerations

### Path 1: GenRM Tournament
- **Compute**: k times more LLM calls for summarization + tournament comparisons
- **Latency**: Higher (sequential tournament rounds)
- **Quality**: Better summaries, no training required

### Path 2: Oracle Approximation
- **Compute**: Training cost + oracle evaluation on candidates
- **Latency**: Training upfront, then fast inference
- **Quality**: Task-aligned summaries

### Path 3: Judge Training
- **Compute**: Meta-training cost (relatively small)
- **Latency**: Training upfront
- **Quality**: Better tournament selection in Path 1

---

## Canonical Tree Builders

When implementing any of these paths, use the canonical tree builders:

**Single Document**:
```python
from src.ops_engine.builder import TreeBuilder

builder = TreeBuilder(strategy=my_strategy)
tree = await builder.build(text)
```

**Multi-Document (Optimal Batching)**:
```python
from src.core.batch_orchestrator import BatchTreeOrchestrator

orchestrator = BatchTreeOrchestrator(client=batch_client, rubric=rubric)
trees = await orchestrator.build_trees(documents)
```

Note: Use these canonical builders. Legacy builders have been removed.

---

## Configuration

Key settings in `config/settings.yaml`:

```yaml
# Tournament settings
generation:
  summarizer:
    candidate_temperatures: [0.3, 0.5, 0.7, 0.9]  # Diversity for tournament

# Optimization settings
optimization:
  optimization_mode: "joint"  # oracle_only, genrm_only, sequential, joint
  oracle_weight: 0.7
  genrm_weight: 0.3

# GenRM judge settings
generation:
  genrm_judge:
    temperature: 0.6
    top_p: 0.95
```
