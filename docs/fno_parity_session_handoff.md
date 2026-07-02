# FNO Parity Session Handoff — 2026-04-09

## 2026-04-09 Contract Update

This handoff predates the parity-contract repair for supervision-recovery runs. The current authoritative tree-parity path is:

- `scripts/run_markov_optimization_tradeoff_pipeline.py`
- `tree_supervision_source="manifest"`
- `local_estimand_mode="span_mass_ipw_sum"`
- `c2_pair_weighting_mode="pair_ipw_geomean"`

Two consequences:

1. Old standalone depth-discount sweeps are legacy-only and should not be used for parity claims.
2. Current `r100_mass_local_eq_*` conclusions only count if the run summary shows the contract fields above.

For gamma studies, authoritative rows should also expose:

- `depth_discount_gamma`
- `is_authoritative_parity_row`
- `is_authoritative_gamma_row`

See [docs/depth_discount_gamma_session_handoff.md](/home/mlinegar/ThinkingTrees/docs/depth_discount_gamma_session_handoff.md) for the current depth-discount and sampled-C2 status.

## What We Did

Investigated why the tree-at-1-leaf-per-doc doesn't match the standalone FNO, and fixed every source of divergence until we achieved **exact parity** (0.000 gap at all training sizes).

## Key Results

### 1-Leaf Parity (PROVEN)

At 1 leaf per doc (128-token docs, leaf_tokens=128), the tree matches FNO exactly:

| train_docs | FNO | TREE | gap |
|---|---|---|---|
| 1024 | 0.160 | 0.160 | 0.000 |
| 2048 | 0.047 | 0.047 | 0.000 |
| 4096 | 0.000 | 0.000 | 0.000 |

### Multi-Leaf Geometry (from r100 full coverage v3)

| leaf | leaves/doc | FNO | Tree (10k docs) |
|---|---|---|---|
| 128 | 1 | 0.008 | 0.008 (parity) |
| 64 | 2 | 0.008 | 0.010 (close) |
| 32 | 4 | 0.008 | 0.028 |
| 16 | 8 | 0.008 | 0.129 |
| 8 | 16 | 0.008 | 0.072 |

### Multi-Leaf Ablation (root_only vs standard)

Local laws are critical for multi-leaf trees. Without them, the merge operator can't learn:

| leaf | root_only (no laws) | standard (laws) |
|---|---|---|
| 64 (2 leaves) | 0.099 | **0.010** |
| 32 (4 leaves) | 0.496 | **0.033** |
| 16 (8 leaves) | 0.470 | **0.129** |
| 8 (16 leaves) | 0.684 | **0.072** |

## Bugs Found & Fixed

### Critical bugs (caused wrong results)

1. **`predict_canonical_count_from_state` ignored `count_ce` under `use_summary_spec`** — the summary_spec path short-circuited to sigmoid regression before the CE argmax path was reached. Fixed by checking CE first.
   - File: `src/ctreepo/sim/core/markov_neural_operator_baselines.py:6634`

2. **Training CE dispatch bypassed** — both the single-doc path (`use_shared_theorem_surface` check) and the batched path (`_theorem_feature_task_supervision_terms_batched`) used MSE even when `root_supervision_kind == "count_ce"`. Fixed by adding CE dispatch before MSE fallback.
   - File: `src/ctreepo/sim/core/markov_neural_operator_baselines.py:15195` (single-doc)
   - File: `src/ctreepo/sim/core/markov_neural_operator_baselines.py:9623` (batched)

3. **Normalized targets in batched CE loss** — `_theorem_feature_task_supervision_terms_batched` received targets as count/target_scale but tried to look up raw counts in class_index. Fixed by multiplying by `target_scale`.
   - File: `src/ctreepo/sim/core/markov_neural_operator_baselines.py:9631`

4. **Comparable surface source used pre-override config** — the surface snapshot was built from the base config (which had `mse`) instead of the tree-reference-overridden config (which had `count_ce`). The surface then overwrote the tree_config back to `mse`.
   - File: `scripts/run_markov_optimization_tradeoff_pipeline.py:4191`

5. **FNO compare config missing `tree_root_supervision_kind`** — propagated FNO architecture keys but not the string-typed supervision kind.
   - File: `scripts/run_markov_optimization_tradeoff_pipeline.py:9372`

6. **`one_leaf_tree_reference` TOML section not wired to argparse** — the config section existed but wasn't being parsed. Added argparse arguments and config flattening.
   - File: `scripts/run_markov_optimization_tradeoff_pipeline.py:2449` (argparse)
   - File: `scripts/run_markov_optimization_tradeoff_pipeline.py:2249,2270` (config flattening)

7. **`one_leaf_tree_reference` applied to ALL packages** — caused KeyError when mass-matched packages had counts outside the CE class range. Fixed by scoping to `full100` only.
   - File: `scripts/run_markov_optimization_tradeoff_pipeline.py:9315`

### Config alignment fixes (caused silent divergence)

8. **Standalone FNO used different architecture than tree** — FNO hardcoded width/n_modes/n_layers while tree read from config. Fixed: standalone FNO now reads `tree_leaf_fno_*` from config with fallback to legacy defaults.
   - File: `src/ctreepo/sim/core/markov_neural_operator_baselines.py:882`

9. **Standalone FNO used different training hyperparams** — FNO got state_dim=32/hidden_dim=64 from benchmark defaults while tree used 128/512. Fixed: pipeline propagates tree reference's state_dim/hidden_dim/batch_size/lr to FNO config.
   - File: `scripts/run_markov_optimization_tradeoff_pipeline.py:9375`

10. **Locked FNO config dropped tree architecture fields** — Fixed: passes through `tree_leaf_fno_*` fields.
    - File: `src/ctreepo/sim/core/full_doc_anchor_diagnostics.py:1937`

### Validation / infrastructure fixes

11. **Validation rejected `count_ce` and `single_stage`** — relaxed to accept both.
12. **Surface drift check rejected `comparable` vs `exact_collapse`** — excluded `comparison_mode` from comparison.

## New Presets Created

All in `src/ctreepo/sim/core/tree_reference_presets.py`:

| Preset | Purpose |
|---|---|
| `unified_g_fno_parity_canary_v1` | CE + single-stage + no laws — exact FNO parity at 1 leaf |
| `unified_g_ablation_mse_v1` | Canary + MSE (ablation step 1) |
| `unified_g_ablation_two_stage_v1` | + two-stage (ablation step 2) |
| `unified_g_ablation_local_laws_v1` | + local laws = standard (ablation step 3) |
| `unified_g_multi_leaf_root_only_v1` | MSE + single-stage + no laws (multi-leaf baseline) |

## New Config Files

| Config | Purpose |
|---|---|
| `config/markov/tradeoff_pipeline.fno_parity_canary_test.toml` | Quick parity test (full100 only, 3 train sizes) |
| `config/markov/tradeoff_pipeline.supervision_recovery_unified_g_leafgrid_r100_full_coverage.toml` | Full leaf geometry sweep with one_leaf override |
| `config/markov/tradeoff_pipeline.supervision_recovery_unified_g_ablation_ladder.toml` | Template for ablation runs (PLACEHOLDER preset) |
| `config/markov/tradeoff_pipeline.supervision_recovery_unified_g_multi_leaf_ablation.toml` | Template for multi-leaf root-only vs standard comparison |

## What's Running

### Active (as of 2026-04-09 19:00 UTC)

1. **r100_full_coverage_v5** (360 tasks, ~14% done)
   - Output: `outputs/markov_supervision_recovery_unified_g_leafgrid_r100_full_coverage_v5_run_20260409_185903/`
   - Log: `outputs/r100_full_coverage_v5_20260409_185903.log` (empty — stdout buffered)
   - Config: leaf ladder [128,64,32,16,8], packages [full100 + 4 mass-matched], train [1024,4096,10240], seeds [0,1]
   - **Key fix**: `one_leaf_tree_reference` only applies to `full100` at leaf=128 (canary CE preset). Mass-matched packages at leaf=128 use standard MSE preset.

2. **multi_leaf_ablation/root_only** (120 tasks, ~80% done — possibly stuck on 10240-doc tasks)
   - Output: `outputs/markov_multi_leaf_ablation_multi_leaf_root_only_v1_run_20260409_171918/`
   - Single-stage, no local laws, MSE, across all leaf sizes

3. **multi_leaf_ablation/standard** (120 tasks, ~80% done)
   - Output: `outputs/markov_multi_leaf_ablation_full_local_laws_v1_run_20260409_171918/`
   - Standard two-stage + local laws for comparison

### Completed

- `fno_parity_canary_test_run6` — proved exact parity at 1 leaf
- `leafgrid_canary_run_20260409_071119` — canary across 5 packages
- `leafgrid_r100_run_20260409_071119` — original r100 (leaf=8 only)
- `leafgrid_r10_r20_r80_r90_run_20260409_071119` — full supervision rate sweep
- `r100_full_coverage_v3_run_20260409_163154` — first full leaf geometry sweep (has stale leaf=128 mass-matched results due to one_leaf bug)
- Various ablation runs under `outputs/markov_ablation_v2_*` — showed all 4 ablation steps give identical results at 1 leaf

### Superseded (stale, can ignore)

- `r100_full_coverage_v4_run_20260409_175950` — mass-matched leaf=128 tasks crashed with KeyError (one_leaf CE applied too broadly)
- `fno_parity_canary_test_run[1-5]` — earlier iterations before bugs were fixed

## What to Do Next

1. **Wait for r100_full_coverage_v5 to finish** — this is the definitive run with all fixes. Once done, pull the full leaf geometry × supervision rate spread and verify leaf=128 mass-matched results are sane (not 0.65).

2. **Check multi_leaf_ablation completion** — the root_only vs standard comparison at all leaf sizes. Already shows local laws are critical for multi-leaf.

3. **Investigate the mid-leaf degradation** — at 32 tokens (4 leaves), the tree is worst (0.028 at 10k docs). This is the "merge overhead" regime. The merge bottleneck (unified_g_merge_summary_proj: 2×state_dim → 3×fno_width) may be too narrow. Consider:
   - Increasing merge hidden dim
   - Using a wider summary surface
   - Adding skip connections in the merge path

4. **Consider CE for multi-leaf** — at 1 leaf, CE and MSE gave identical results. At multi-leaf, CE might help the root prediction quality. But need to handle the class index carefully (the KeyError we hit).

## Key Files Modified

- `src/ctreepo/sim/core/markov_neural_operator_baselines.py` — CE dispatch in prediction + training, normalized target fix, standalone FNO reads tree config
- `src/ctreepo/sim/core/full_doc_anchor_diagnostics.py` — locked FNO config passes through tree fields
- `src/ctreepo/sim/core/tree_reference_presets.py` — new presets (canary, ablation, multi-leaf root-only)
- `scripts/run_markov_optimization_tradeoff_pipeline.py` — config propagation, validation relaxation, one_leaf reference, surface drift fix, argparse wiring
- `config/markov/tradeoff_pipeline.*.toml` — new and updated experiment configs
