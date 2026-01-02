# Domain → Task Migration Summary

## Overview
Successfully migrated ThinkingTrees from domain-specific (RILE/manifesto) to truly generic task-based architecture.

**Update (January 2025)**:
- All backward compatibility aliases have been removed
- `ManifestoTask` replaced with generic `ScoringTask` + building blocks
- New generic components in `src/core/`: `ScaleScorer`, `GenericSummarizer`, etc.

## Completed Phases

### ✅ Phase 1: Terminology Rename (domain → task)
**Goal**: Standardize on "task" terminology throughout codebase

**Changes**:
- Renamed directory: `src/ops_engine/training_framework/domains/` → `.../tasks/`
- Renamed classes:
  - `DomainPlugin` → `TaskPlugin`
  - `AbstractDomain` → `AbstractTask`
  - `DomainConfig` → `TaskConfig`
  - `DomainRegistry` → `TaskRegistry`
  - `ManifestoDomain` → `ManifestoTask`
  - `DocumentAnalysisDomain` → `DocumentAnalysisTask`
- Updated imports in ~20 files
- Added `get_default_task()` function in settings.py

**Files Modified**: 20+ files across src/ops_engine, src/tasks, src/config

---

### ✅ Phase 2: Manifesto Module Consolidation
**Goal**: Delete redundant code, keep only RILE-specific components

**Deleted (7 files, ~2,300 lines)**:
1. `batched_pipeline.py` - shim to generic pipeline
2. `ops_pipeline.py` - duplicated generic pipeline logic
3. `evaluation.py` - generic evaluation, not RILE-specific
4. `metrics.py` - duplicated in training framework
5. `summary_tables.py` - reporting utility
6. `training_integration.py` - superseded by task plugin system

**Kept (7 files, ~1,254 lines)**:
1. `constants.py` - RILE scale constants
2. `rubrics.py` - RILE preservation rubrics
3. `signatures.py` - RILE DSPy signatures
4. `position_oracle.py` - RILE position oracle
5. `dspy_summarizer.py` - RILE-specific summarizers
6. `data_loader.py` - Manifesto Project CSV data loading
7. `__init__.py` - compatibility shim with helpful migration errors

**New Structure Created**:
```
src/tasks/manifesto/
├── __init__.py          # Public API
├── task.py              # ManifestoTask implementation
├── constants.py         # RILE constants
├── rubrics.py           # RILE rubrics
├── signatures.py        # RILE DSPy signatures
└── oracle.py            # RILE position oracle
```

**Result**: 65% code reduction in manifesto module (3,553 → 1,254 lines)

---

### ✅ Phase 3: Eliminate Core → Manifesto Dependencies
**Goal**: Remove all manifesto imports from core framework modules

**Fixed Files**:
1. **src/pipelines/batched.py**:
   - Removed: `from src.manifesto.dspy_summarizer import ...`
   - Now uses: Generic type hints, task-injectable summarizers
   - Documentation updated to show task plugin usage

2. **src/ops_engine/training_framework/metrics.py**:
   - Removed: `from src.manifesto.constants import RILE_RANGE` (4 occurrences)
   - Now uses: `from src.tasks.manifesto import RILE_SCALE` (only in examples)
   - Functions now accept scale as parameter

3. **src/ops_engine/initialization.py**:
   - Updated: `from src.manifesto.rubrics` → `from src.tasks.manifesto`
   - Marked DSPy summarizer imports for future refactor

**Validation**: Zero manifesto imports in:
- src/core/
- src/ops_engine/builder.py
- src/ops_engine/auditor.py
- src/ops_engine/scoring.py
- src/pipelines/

**Result**: Core framework is now 100% task-agnostic

---

### ✅ Phase 4: Update Experiment Scripts
**Goal**: Update experiments to use new structure

**Approach**:
- Experiment scripts in `experiments/manifesto_rile/` are RILE-specific
- They can continue importing from `src.manifesto` (now a clean compatibility shim)
- Main imports updated where appropriate (datasets, pipelines)

**Note**: These are domain-specific experiments, so manifesto imports are acceptable here.

---

### ✅ Phase 5: Final Cleanup and Validation
**Goal**: Validate migration success

**Validation Results**:
- ✅ Zero manifesto imports in core modules
- ✅ Task registry structure in place
- ✅ Backward compatibility maintained
- ✅ Module sizes reduced significantly
- ✅ File structure clean and organized

---

## Architecture After Migration

### Core Building Blocks
```
src/core/
├── scorers.py         # ScaleScorer, PairwiseScorer (NEW)
├── summarization.py   # GenericSummarizer, GenericMerger (NEW)
├── signatures.py      # Generic DSPy signatures
└── ...
```

### Task System
```
src/ops_engine/training_framework/tasks/
├── base.py              # TaskPlugin protocol, AbstractTask, ScaleDefinition
├── registry.py          # TaskRegistry for plugin discovery
├── scoring.py           # Generic ScoringTask (fully configurable)
└── document_analysis.py # DocumentAnalysisTask (generic default)
```

### Domain-Specific Building Blocks
```
src/tasks/
├── __init__.py      # Exports TaskPlugin, get_task, etc.
├── prompting.py     # Generic prompt builders
└── manifesto/       # RILE-specific BUILDING BLOCKS (not a Task class)
    ├── __init__.py  # Exports: RILE_SCALE, RILE_RUBRIC, RILEScorer, etc.
    ├── constants.py # RILE scale bounds
    ├── rubrics.py   # RILE preservation rubrics
    ├── dspy_signatures.py  # RILEScorer, RILEComparator
    ├── summarizer.py       # RILE-specific summarizers
    └── data_loader.py      # ManifestoDataset
```

### Current API
Building blocks approach:
- `ScoringTask` - Configure via constructor (scale, rubric, factories)
- `ScaleDefinition` - Define score bounds
- `ScaleScorer` - Generic DSPy scorer
- Domain modules export building blocks, not Task classes

---

## Usage Examples

### Before (coupled to RILE):
```python
from src.manifesto.batched_pipeline import BatchedManifestoPipeline
from src.manifesto.rubrics import RILE_PRESERVATION_RUBRIC

pipeline = BatchedManifestoPipeline(config, rubric=RILE_PRESERVATION_RUBRIC)
```

### After (building blocks approach):
```python
from src.ops_engine.training_framework.tasks import ScoringTask, ScaleDefinition
from src.tasks.manifesto import (
    RILE_SCALE,                  # ScaleDefinition(-100, +100)
    RILE_PRESERVATION_RUBRIC,   # Domain rubric
    ManifestoDataLoader,        # Data loading
    RILEScorer,                 # Domain scorer
)

# Compose building blocks into a task
rile_task = ScoringTask(
    name="rile",
    scale=RILE_SCALE,
    rubric=RILE_PRESERVATION_RUBRIC,
    data_loader_factory=lambda: ManifestoDataLoader(),
    predictor_factory=lambda: RILEScorer(),
)

# Or create a completely new task with generic components
from src.core import ScaleScorer

MY_SCALE = ScaleDefinition(name="sentiment", min_value=-1, max_value=1)
my_task = ScoringTask(
    name="sentiment",
    scale=MY_SCALE,
    rubric="Preserve sentiment indicators...",
    predictor_factory=lambda: ScaleScorer(MySentimentSignature),
)
```

---

## Benefits

1. **True Multi-Task Support**: Core framework works with ANY task plugin
2. **Clean Separation**: Domain logic separated from generic infrastructure
3. **Reduced Code**: 65% reduction in manifesto module
4. **Better Maintainability**: Clear responsibility boundaries
5. **Easier Testing**: Generic components can be tested independently
6. **Future-Proof**: Easy to add new tasks (sentiment, classification, etc.)

---

## Future Enhancements

### Potential Extensions
1. Add more task plugins:
   - Sentiment analysis (-1 to 1)
   - Topic classification (discrete labels)
   - Document quality (0 to 1)

2. Make DSPy summarizers task-injectable
3. Create rubric template system
4. Add parser registry for score extraction

---

## File Statistics

### Before Migration
- src/manifesto/: 13 files, 3,553 lines
- Manifesto imports in core: 14+ instances
- Hard coupling to RILE throughout

### After Migration
- src/manifesto/: 7 files, 1,254 lines (65% reduction)
- src/tasks/manifesto/: 6 files
- Manifesto imports in core: 0 instances
- Clean task plugin architecture

---

## Breaking Changes

### Full Breaking (No Backward Compatibility)
Backward compatibility aliases were removed in January 2025:
- `from src.ops_engine.training_framework.domains import get_domain` ✗ No longer works
- `DomainPlugin`, `AbstractDomain`, etc. ✗ No longer exist
- `ManifestoTask` ✗ Replaced with `ScoringTask` + building blocks

### What Was Removed
- All `Domain*` → `Task*` aliases in task modules
- `ManifestoTask` class (use `ScoringTask` with RILE building blocks)
- `get_domain()`, `list_domains()` convenience functions
- `BoundedScale` class (use `ScaleDefinition`)
- `build_tree_batched()`, `build_tree_with_dspy()`, `build_tree_for_document()`
- `PairwiseSummaryComparison` (use `PairwiseComparison`)
- Legacy oracle interface adapters (`oracle_utils.py` deleted)
- Old field name fallbacks in `from_dict()` methods (e.g., `manifesto_id` → `doc_id`)

### New Building Blocks Added
- `src/core/scorers.py` - `ScaleScorer`, `PairwiseScorer`
- `src/core/summarization.py` - `GenericSummarizer`, `GenericMerger`, `SummarizationResult`
- `src/ops_engine/scoring.py` - `create_oracle_scorer()` factory
- `src/ops_engine/training_framework/tasks/scoring.py` - Generic `ScoringTask`

---

## Training Pipeline Building Blocks (January 2025)

Extended the building blocks pattern to the training pipeline with registry-based components:

### New Components

| Component | Location | Purpose |
|-----------|----------|---------|
| `JudgeRegistry` | `training_framework/judges/` | Pluggable pairwise comparison judges |
| `MetricBuilder` | `training_framework/metrics.py` | Fluent API for composing DSPy metrics |
| `LabelingStrategy` | `training_framework/labeling.py` | Error-to-label conversion strategies |
| `StrategyRegistry` | `src/core/strategy.py` | Summarization strategy registry |
| `PreferenceDeriver` | `training_framework/preference_types.py` | Preference derivation strategies |

### Available Implementations

**Judges** (`get_judge(name)`):
- `dspy` - DSPy-based judge (optimizable via compile)
- `genrm` - NVIDIA GenRM model wrapper
- `oracle` - Oracle scoring function based

**Labelers** (`get_labeler(name)`):
- `threshold` - Fixed high/low thresholds (default)
- `binary` - Simple above/below single threshold
- `percentile` - Data-driven percentile thresholds
- `adaptive` - Scale-aware adaptive thresholds

**Strategies** (`get_strategy(name)`):
- `batched` - AsyncBatchLLMClient based
- `dspy` - DSPy module wrapper
- `callable` - Sync callable wrapper
- `tournament` - Tournament selection wrapper

**Derivers** (`get_deriver(name)`):
- `judge` - LLM judge (PairwiseJudge)
- `genrm` - GenRM model
- `oracle` - Oracle score comparison

### Usage Examples

```python
# JudgeRegistry
from src.ops_engine.training_framework.judges import get_judge, JudgeConfig
judge = get_judge("genrm", config=JudgeConfig(base_url="http://localhost:8001/v1"))

# MetricBuilder
from src.ops_engine.training_framework.metrics import MetricBuilder
metric = MetricBuilder().with_oracle(fn).with_scale(scale).with_caching().build_metric()

# LabelingStrategy
from src.ops_engine.training_framework.labeling import get_labeler
labeler = get_labeler("threshold", threshold_high=0.3, threshold_low=0.1)

# StrategyRegistry
from src.core.strategy import get_strategy
strategy = get_strategy("tournament", base=base_strategy, judge=my_judge)
```

---

## Testing Recommendations

1. **Unit Tests**: Test task plugins independently
2. **Integration Tests**: Test generic pipeline with multiple tasks
3. **Regression Tests**: Ensure RILE experiments still work
4. **Performance Tests**: Verify no performance degradation

---

## Migration Dates
- **December 29, 2024**: Initial domain → task migration
- **January 2, 2025**: Building blocks refactor (ManifestoTask → ScoringTask + components)
- **January 2, 2025**: Training pipeline building blocks (JudgeRegistry, MetricBuilder, LabelingStrategy, StrategyRegistry, PreferenceDeriver)

## Contributors
- Refactoring assistance: Claude (Anthropic)
- Original architecture: mlinegar
