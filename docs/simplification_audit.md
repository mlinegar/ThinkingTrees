# Simplification Audit

Catalog of complexity, duplication, and anti-patterns across the codebase. Organized by severity.

## Completed

- [x] **1.1** Gutted `_supervision_recovery_tree_learning_errors()` from 180 to 10 lines (smoke guard only)
- [x] **1.2** Defensive string coercion — removed via validation deletion
- [x] **1.3** Giant if/elif dispatch — removed via validation deletion
- [x] **3.1** `_safe_float`/`_safe_int` → shared `src/ctreepo/sim/util.py` (8 files in src/, 11 scripts in progress)
- [x] **3.4** FNO arch resolution → `FNOArchConfig` + `resolve_fno_arch` in `fno_arch_config.py`
- [x] **4.1** Dual FNO params → unified resolution through `resolve_fno_arch`
- [x] **4.2** Silent config overwriting → `apply_comparable_surface_to_mapping` simplified, uses `resolve_fno_arch_from_mapping`
- [x] **4.3** 70+ key allowlist → inverted to deny-list (`TREE_REFERENCE_DENY_KEYS`)
- [x] **4.4** Multiple preservation flags → consolidated `frozen_keys` parameter
- [x] **5.1** Parallel FNO construction → `FNOTokenEncoder` shared by both `FNOCountPredictor` and `FNOCountSketch`
- [x] **5.3** Script-defined constants → extracted to `tree_reference_presets.py` (~360 lines removed from pipeline script)
- [x] **5.2** Core importing from suite → `markov_observed_token_policy.py` moved to `core/`, suite re-exports
- [x] **3.2** `_normalize_simplex_vec/rows` → shared `src/ctreepo/sim/util.py` (3 copies consolidated)
- [x] FNO propagation in pipeline → replaced manual key loop with `resolve_fno_arch_from_mapping`
- [x] `doc_sequence_fno` in FNOCountSketch → constructed via `FNOTokenEncoder`
- [x] **1.2** String coercion soup → 30 patterns replaced with `_gs()`/`_ns()` helpers (14 complex patterns remain)
- [x] **3.3** `_resolve_output_root` / `_resolve_figures_root` → extracted to `suite/common.py`
- [x] `_build_supervision_recovery_scope_config` → clarified comparison branch, compacted package init
- [x] `_safe_float`/`_safe_int` in remaining src/ and scripts/ → 19 more files consolidated (total ~38 files)
- [x] **2.1** `markov_changepoint_ops_count.py` split: classical baselines extracted to `markov_baselines.py` (~1600 lines, 30% reduction)
- [x] **2.1** `markov_neural_operator_baselines.py` split: standalone baselines extracted to `fno_doc_baselines.py` (~1540 lines)
- [ ] **2.1** `full_doc_anchor_diagnostics.py` split — deferred, interleaved metrics/baselines sections too tangled

---

## TIER 1: Harmful Validation Spaghetti

### 1.1 Hardcoded config validation that should be preset defaults

**File:** `scripts/run_markov_optimization_tradeoff_pipeline.py:3920-4105`
**Function:** `_supervision_recovery_tree_learning_errors()`

186 lines of `errors.append()` calls that hardcode expected config values. These are not real validation — they're re-asserting what the preset already defines. If the preset changes, these break. If they disagree with the preset, you get confusing errors.

Examples:
```python
# Why is "unified_v2" hardcoded here when the preset already sets it?
if str(payload.get("tree_batch_runtime_mode", "") or "").strip().lower() != "unified_v2":
    errors.append("tree_batch_runtime_mode must be 'unified_v2'")

# Magic number thresholds with no documented rationale
if stage1_epochs < 8:
    errors.append(f"{scope_kind} tree_stage1_epochs must be at least 8")
if stage2_epochs < 32:
    errors.append(f"{scope_kind} tree_stage2_epochs must be at least 32")

# Hardcoded dimension minimums
if state_dim < 128:
    errors.append(...)
if hidden_dim < 256:
    errors.append(...)
```

**Fix:** Delete this function entirely. If presets define the right values, the validation is redundant. If you need guardrails, define them in the `OPSCountConfig` dataclass with `__post_init__` validation.

### 1.2 Defensive string coercion soup

**File:** `scripts/run_markov_optimization_tradeoff_pipeline.py` (22+ occurrences in single function)

Pattern repeated everywhere:
```python
str(payload.get("key", "") or "").strip().lower()
```

This is 5 operations to read one config field. The `or ""` is redundant when the default is already `""`. The `.strip().lower()` should happen once at deserialization time, not at every access point.

**Fix:** Deserialize config into a typed dataclass once. Access fields as attributes, not via `.get()` chains.

### 1.3 Giant if/elif dispatch chains

**File:** `scripts/run_markov_optimization_tradeoff_pipeline.py:3991-4104`

4-way if/elif/elif/else chain where each branch has 6+ sub-validations. The branches correspond to preset types that are already known at config time.

**Fix:** If you must validate, use a dispatch dict mapping preset name to a small validator function.

---

## TIER 2: Monolithic Files

### 2.1 Mega-scripts that combine parsing + validation + orchestration

| File | Lines | Functions | Conditionals |
|------|-------|-----------|-------------|
| `scripts/run_markov_optimization_tradeoff_pipeline.py` | 11,468 | 149 | 1,743 |
| `scripts/run_tree_neural_full_doc_mig.py` | 13,647 | ~150 | ~1,400 |
| `src/ctreepo/sim/core/markov_neural_operator_baselines.py` | 17,714 | 170 | 1,404 |
| `src/ctreepo/sim/core/full_doc_anchor_diagnostics.py` | 12,114 | 140 | 882 |
| `src/ctreepo/sim/core/markov_changepoint_ops_count.py` | 10,986 | 137 | 738 |
| `src/training/run_pipeline.py` | 20,679 | 141 | 1,743 |

**Fix (per file):**
- `markov_neural_operator_baselines.py`: Extract FNO, DeepONet, MLP, CNN to separate files. The 22 model classes do not need to live together.
- `full_doc_anchor_diagnostics.py`: Split metrics computation from visualization.
- `run_pipeline.py`: Split oracle_trainer, summarizer_trainer, preference_collector, orchestrator.
- Pipeline scripts: Extract config/preset definitions into a shared config module.

### 2.2 Absurdly large generated scripts

| File | Lines |
|------|-------|
| `scripts/build_exact_utility_transport_overnight.py` | 69,143 |

This should be a config file + a generator, not a 69K-line Python script.

---

## TIER 3: Cross-File Duplication

### 3.1 `_safe_float()` / `_safe_int()` duplicated 24+ times

Identical try/except wrappers copy-pasted across:
- `scripts/run_markov_optimization_tradeoff_pipeline.py:2481`
- `src/ctreepo/sim/core/markov_comparison_surface.py:157`
- `src/ctreepo/sim/core/full_doc_anchor_diagnostics.py:415`
- 19+ report scripts

**Fix:** Extract to `src/ctreepo/sim/util.py` (or similar). Import everywhere.

### 3.2 `_normalize_simplex_vec()` / `_normalize_simplex_rows()` in 3 files

Slightly different implementations in:
- `src/ctreepo/sim/core/leaf_local_mixture_utility.py:869-887`
- `src/ctreepo/sim/core/segmented_lda_ctreepo.py:326-343`
- `src/ctreepo/sim/core/tensor_lda_book_benchmark.py:238-263`

### 3.3 `_resolve_device()` in 3 files with incompatible signatures

- `src/ctreepo/sim/core/full_doc_anchor_diagnostics.py:1746` (returns tuple)
- `src/ctreepo/sim/core/lda_tree_recovery_learned.py:257` (returns device)
- `src/ctreepo/sim/core/lda_tree_utility_vector.py:484` (returns device)

### 3.4 FNO arch resolution duplicated 5 times

(Already covered in the FNO/tree unification plan — Step 1 fixes this.)

### 3.5 Field name lists overlap without referencing each other

- `MARKOV_SHARED_COMPARISON_FIELDS` (14 fields) in `markov_comparison_surface.py:42`
- `TREE_REFERENCE_OVERRIDE_KEYS` (70+ fields) in pipeline script:3444

The 14 shared fields are a strict subset of the 70 override keys but defined independently.

---

## TIER 4: Config System Complexity

### 4.1 Dual FNO params with Optional fallback

`fno_width` vs `tree_leaf_fno_width` (+ n_modes, n_layers). The `tree_leaf_fno_*` are Optional overrides that fall back to `fno_*`. This creates the 5-site duplication in Tier 3.4.

(Already covered in the FNO/tree unification plan.)

### 4.2 Silent config overwriting via comparable surface

`apply_comparable_surface_to_mapping()` silently overwrites config fields to force parity between tree and FNO runs. This makes debugging config issues very difficult because the values you set are not the values that run.

**Fix:** Convert to validation-then-error (already in plan Step 6).

### 4.3 70+ key TREE_REFERENCE_OVERRIDE_KEYS

A flat tuple of 70+ string keys that controls which preset values get applied. Adding a new config field requires remembering to add it here too, or the preset silently ignores it.

**Fix:** Invert the pattern. Instead of an allowlist of keys to copy, have the preset be a partial `OPSCountConfig` (or dict) that gets merged. Override keys become "keys to NOT override" (a much shorter deny-list).

### 4.4 Multiple preservation flags create conditional branches

`preserve_fixed_leaf_tokens`, `preserve_schedule`, `preserve_requested_leaf_tokens` — each adds a conditional skip in the override loop. This makes the override behavior depend on a combinatorial explosion of flags.

**Fix:** Use a single `frozen_keys: Set[str]` parameter that lists which keys are frozen for this invocation.

### 4.5 100+ raw string `.get()` calls instead of typed config

Throughout pipeline scripts, config values are accessed via `payload.get("key", "")` with repeated type coercion. The `OPSCountConfig` dataclass exists but is not used consistently — configs often travel as raw dicts.

**Fix:** Deserialize to `OPSCountConfig` (or a subset dataclass) at the boundary. Access as typed attributes throughout.

---

## TIER 5: Architectural Issues

### 5.1 `FNOCountPredictor` and `FNOCountSketch` build FNO independently

Two separate `nn.Module` classes that each construct their own `_NeuralOpFNO`. The canary test exists to verify they produce the same results, but this parity should be structural (shared component), not numerical.

(Already covered in plan Steps 2-3.)

### 5.2 Core imports from suite (inverted dependency)

`src/ctreepo/sim/core/markov_comparison_surface.py` imports from `src/ctreepo/sim/suite/markov_observed_token_policy.py`. Core should not depend on suite.

### 5.3 No clear boundary between "library" and "script" code

The pipeline scripts define constants (TREE_REFERENCE_PRESET_CONFIGS) that other scripts import. This creates a situation where running one script requires importing another 11K-line script.

**Fix:** Extract shared constants/presets into a dedicated config module under `src/`.

---

## Priority Order for Cleanup

1. **Delete `_supervision_recovery_tree_learning_errors()`** (Tier 1.1) — harmful, not helpful
2. **Create `FNOArchConfig` + `resolve_fno_arch`** (Tier 3.4 / plan Step 1) — quick win
3. **Extract `_safe_float`/`_safe_int` to shared util** (Tier 3.1) — quick win
4. **Extract `FNOTokenEncoder`** (Tier 5.1 / plan Steps 2-3) — structural fix
5. **Move preset configs to dedicated module** (Tier 5.3) — reduces coupling
6. **Invert TREE_REFERENCE_OVERRIDE_KEYS to deny-list** (Tier 4.3) — reduces fragility
7. **Deserialize configs to typed dataclass at boundary** (Tier 4.5) — reduces string soup
8. **Split monolithic files** (Tier 2.1) — ongoing, do opportunistically
