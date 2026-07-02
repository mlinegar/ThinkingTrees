# Classical-HLL parity benchmark (Appendix F reproduction)

Empirical companion to Proposition 1 (`paper/ctreepo/sections/04_theory.tex:77`).
Routes every cell — flat reference, TreePO tree reduction, native HLL,
Apache DataSketches HLL — through the unified `fit()` entry point at
[`parallel/unified_g_v1/src/unified_g_v1/training/fit.py`](../parallel/unified_g_v1/src/unified_g_v1/training/fit.py)
so the comparison is a single CSV join on `FitResult.metrics`.

## One-shot reproduction

```bash
# Install optional extras (gate for the DataSketches reference implementation).
pip install -e treepo[sketches]
python -c "import datasketches; print(datasketches.__class__.__module__)"

# Full sweep; writes outputs/classical_parity/hll/{summary.csv,curve.png,*.json}
# and paper/ctreepo/tables/classical_parity_hll.{md,tex}.
python scripts/run_classical_parity_benchmark.py --out outputs/classical_parity

# Render the paper tables from the CSV (idempotent, safe to re-run).
bash paper/ctreepo/tables/make_tables.sh

# Rebuild the paper PDF with Appendix F.
(cd paper/ctreepo && pdflatex -interaction=nonstopmode main_new.tex)
```

## Smoke test

```bash
# Minimal grid — 8 cells, < 10s on a laptop.
python scripts/run_classical_parity_benchmark.py \
    --out /tmp/cparity_smoke \
    --precisions 8,10 --leaf-counts 1,4 --backends native,datasketches \
    --seeds 0,1 --oracle-kinds analytic,hll_reference \
    --n-val 8 --min-tokens 256 --max-tokens 512
```

## Broad mergeable-sketch suite

The HLL parity script remains the Appendix F reproduction path. The broader
local Python comparison suite is now available through `treepo-bench`:

```bash
# Apache DataSketches is supplied by the optional sketches extra.
pip install -e treepo[sketches]

# Default grid: capacities {small,medium,large} × leaf counts {1,2,4,8,16}.
# This routes through unified_g_v1.fit() by default when run from the repo.
treepo-bench suite classical-sketches --out-root outputs/classical_sketches --jobs 1
treepo-bench report classical-sketches --output-root outputs/classical_sketches

# Paper bundle: run the full official grid, learned overlays, report, and stage
# paper/ctreepo/assets/sketches/{figures,tables}.
python scripts/run_classical_sketches_paper_bundle.py \
    --jobs 32 \
    --seeds 0,1,2 \
    --capacities small,medium,large \
    --leaf-counts 1,2,4,8,16 \
    --include-learned \
    --learned-targets all \
    --learned-variants f,g,fg,gf \
    --learned-epochs 150 \
    --learned-n-train 128 \
    --learned-n-val 48

# Resume the same output root after an interruption. Only use this when the
# code and run configuration are intentionally unchanged.
python scripts/run_classical_sketches_paper_bundle.py \
    --out-root outputs/classical_sketches_paper_resume \
    --skip-existing \
    --learned-variants f,g,fg,gf

# Faster smoke grid.
treepo-bench suite classical-sketches \
    --out-root /tmp/classical_sketches_smoke \
    --jobs 1 --seeds 0 --capacities small,medium --leaf-counts 1,4

# Pure treepo fallback for package-only installs without the local unified_g lane.
treepo-bench suite classical-sketches \
    --out-root /tmp/classical_sketches_treepo \
    --jobs 1 --execution-backend treepo
```

The suite uses Apache DataSketches as the v1 official runtime authority. It
compares native HLL, DataSketches HLL, CPC, Theta, Count-Min, Frequent Items,
KLL, classic DataSketches quantiles, REQ, t-digest, Tuple, and VarOpt against
exact-truth controls and a negative
`sum_leaf_uniques` baseline. Redis HLL and BigQuery HLL++/KLL are cited as
industry references in the paper, but are intentionally not required for
reproducible local tests.

Grid axes are written into every row: `capacity_label`, `n_leaves`,
`leaf_size`, seed, and the concrete sketch-capacity parameters (`lg_k`,
Count-Min buckets, KLL/REQ/t-digest sizes, Tuple/VarOpt budgets). The report
aggregates by family, sketch, query, capacity, and leaf count, and emits a
paper-facing summary plot (`classical_sketches_summary.png`), a gold-floor
excess-error view (`classical_sketches_gold_gap.png`), raw method-separated
plots (`classical_sketches_method_official.png`,
`classical_sketches_method_learned_f.png`,
`classical_sketches_method_learned_g.png`,
`classical_sketches_method_learned_joint.png`), plus one HLL-style detailed
figure per family: `classical_sketches_distinct.png`,
`classical_sketches_frequency.png`, `classical_sketches_quantile.png`,
`classical_sketches_set.png`, and `classical_sketches_sampling.png`. The report
also writes paper-facing
`paper/ctreepo/tables/classical_sketches_grid.{md,tex}` plus compact
learned-overlay tables `classical_sketches_compact.{md,tex}` with seed CIs
when multiple seeds are present. The bundle script stages the same tables and
figures under `paper/ctreepo/assets/sketches/` for Appendix F. GK is
intentionally skipped in the runtime grid because the formal artifact is
sequential-only, not a general arbitrary-tree merge sketch.

The broad grid now has the same orchestration shape as the HLL parity path:
`classical_sketch_grid_task(...)` builds a zero-optimization `TrainerConfig`
and returns a `FitResult`. The JSON output keeps the benchmark rows unchanged
and adds a `unified_g` block with the fit backend, status, metrics, and
artifacts.

## Design

### Everything goes through `fit()`

The classical baseline is *a trainer with zero optimization steps*. The
[`classical_hll_parity_task(...)`](../parallel/unified_g_v1/src/unified_g_v1/sketch/classical_parity.py)
preset builds a `TrainerConfig` whose `trainer` field is a callable
`run_classical_sketch_baseline` that reads the oracle's val examples, encodes
them through a `SketchAdapter`, folds them via `treepo_reduce`, queries
cardinality, and returns a `FitResult` with the same metric schema the
learned-sketch path uses. This follows the
[unified framework's invariant](../parallel/unified_g_v1/src/unified_g_v1/training/fit.py)
that every workload flows through `fit()`.

### `SketchAdapter` Protocol

The adapter Protocol is in
[`treepo/src/treepo/sketches/protocol.py`](../treepo/src/treepo/sketches/protocol.py).
Concrete adapters (HLL-native, HLL-DataSketches) expose a uniform
`update / encode / merge / query / state_equal / serialize /
serialized_size_bytes / memory_bytes` surface.
`treepo_reduce(items_per_leaf, adapter, schedule)` in
[`treepo/src/treepo/sketches/tree_reducer.py`](../treepo/src/treepo/sketches/tree_reducer.py)
is the sketch-agnostic generalization of `reduce_hll_sketches`.

Adding a new classical sketch is one adapter file plus a `make_*_adapter`
factory; no change to the tree reducer or the benchmark runner. The current
DataSketches-backed adapters cover CPC, Theta, Count-Min, Frequent Items, KLL,
classic quantiles, REQ, t-digest, Tuple, and VarOpt.

### Swappable oracle $f^*$

The oracle function is configurable via `oracle_kind` on the preset:

- `oracle_kind="analytic"` (default) — `f*(x) = |set(x)|`, the true distinct
  cardinality. Compares tree and flat HLL against ground truth.
- `oracle_kind="hll_reference"` — `f*(x) = HLL(x).get_estimate()`, the
  classical sketch's own scoring head. Compares tree against flat *HLL
  reference* — the exact signal a learned merge `g` would train against if
  asked to match the classical sketch at every node.

Pass a custom `target_fn: Callable[[Sequence[int]], float]` for arbitrary
oracles.

### Learned `{f, g, fg, gf}` for scalar sketch oracles

[`learned_scalar_sketch_task(...)`](../parallel/unified_g_v1/src/unified_g_v1/sketch/learned_scalar_sketch.py)
is the single-stage primitive (variant in `{f, g}`).
[`learned_sketch_sequence_task(...)`](../parallel/unified_g_v1/src/unified_g_v1/sketch/learned_scalar_sketch.py)
composes single-stage primitives into a multi-stage variant: any non-empty
string in `{f, g}+`. The orchestrator is registered as the
`learned_sketch_sequence` trainer in the `unified_g_v1` trainer registry, so
each call routes through the centralised `fit()` and each per-stage call
recurses through `fit()` again.

The broad grid can add learned companions for distinct counting, frequency,
quantiles, set operations, tuple summaries, and sampling totals via
`--include-learned --learned-targets all`. The default paper bundle uses
`--learned-variants f,g,fg,gf`. The user-facing codenames are
`learned_f_*`, `learned_g_*`, and `learned_joint_*`; exact staged schedules
such as `fg`, `gf`, or `fgf` are kept in the `learned_variant` metadata field.
The canonical schedules are:

- `learned_f_*` — train only the readout `f`; merge is a deterministic
  identity passthrough (mean of left/right child states; leaves bypass the
  merge MLP). Optionally a pretrained merge can be supplied via
  `init_g_from`, in which case the loaded merge weights are used and frozen.
- `learned_g_*` — train only the leaf/merge stack; readout is a fixed
  target-scaled sigmoid of the first state coordinate. Optionally a
  pretrained readout can be supplied via `init_f_from`.
- `learned_joint_*` with `learned_variant=fg` — staged `f`-then-`g`. Stage 1 trains `f` against an
  unlearned (random-init, frozen) merge `g`. Stage 2 freezes the trained `f`
  and trains `g` on top.
- `learned_joint_*` with `learned_variant=gf` — staged `g`-then-`f`. Stage 1 trains `g` against the
  fixed sigmoid readout. Stage 2 freezes the trained `g` and trains `f` on
  top.

Joint single-call training (one optimizer over both components) is not used.
Longer sequences (`fgf`, `gfgf`, …) work without extra plumbing — the
orchestrator iterates the variant string and chains checkpoints between
stages.

### Two HLL implementations behind the Protocol

| adapter | backing lib | merge | state equivalence |
|---|---|---|---|
| `hll_native` | [`treepo.hll.HyperLogLogSketch`](../treepo/src/treepo/hll.py) | register-wise `max` on a `uint8` numpy array | byte-exact (registers identical after any A1–A3 pipeline) |
| `hll_datasketches` | [Apache DataSketches](https://datasketches.apache.org/) `hll_sketch` + `hll_union` | canonical HLL union | estimate-equivalence within 2× HLL RSE (internal list→sparse→dense mode transitions break byte-level determinism but preserve the represented summary) |

The native adapter's merge holds Proposition 1 in its *strongest* form: the
tree-reduced register state equals the flat register state byte-for-byte. The
DataSketches adapter holds the weaker but still-Prop-1-compliant form:
estimates agree within HLL theoretical noise.

## Test suite

- [`treepo/tests/sketches/test_sketch_adapter_contracts.py`](../treepo/tests/sketches/test_sketch_adapter_contracts.py)
  — adapter-level S1/S2/S3/S4 checks (single-leaf identity, schedule invariance, permutation invariance, reference agreement) for both HLL adapters.
- [`treepo/tests/sketches/test_broad_classical_sketches.py`](../treepo/tests/sketches/test_broad_classical_sketches.py)
  — DataSketches adapter checks for stream updates, merges, serialization size,
  exact-truth comparisons, merge-schedule spread, and the
  `treepo-bench suite classical-sketches --jobs 1` smoke path.
- [`parallel/unified_g_v1/tests/test_classical_hll_parity_fit.py`](../parallel/unified_g_v1/tests/test_classical_hll_parity_fit.py)
  — same contract exercised through the `fit()` pipeline end-to-end, plus HLL-as-oracle tests and custom `target_fn` plumbing.
- [`parallel/unified_g_v1/tests/test_classical_sketch_grid_fit.py`](../parallel/unified_g_v1/tests/test_classical_sketch_grid_fit.py)
  — broad classical-sketch grid through `fit()`.
- [`parallel/unified_g_v1/tests/test_learned_scalar_sketch_fit.py`](../parallel/unified_g_v1/tests/test_learned_scalar_sketch_fit.py)
  — generic learned joint scalar sketch path for exact and sketch-reference
  oracles.

Run everything:

```bash
python -m pytest \
    treepo/tests/sketches/test_sketch_adapter_contracts.py \
    treepo/tests/sketches/test_broad_classical_sketches.py \
    parallel/unified_g_v1/tests/test_classical_hll_parity_fit.py \
    parallel/unified_g_v1/tests/test_classical_sketch_grid_fit.py \
    parallel/unified_g_v1/tests/test_learned_scalar_sketch_fit.py
```

## Status matrix

The broad suite reports both an implementation status and a separate
`formal_status` so official library authority and Lean coverage are not
collapsed into one column:

| status | meaning | examples |
|---|---|---|
| `lean_backed` / `lean_backed_family` | Formal local-law coverage exists for the in-repo sketch or its abstract family. | native HLL; DataSketches HLL/Count-Min/KLL as family-aligned rows |
| `official_empirical` | Runtime adapter wraps Apache DataSketches, but no Lean file is claimed for that exact library implementation. | DataSketches HLL, CPC, Theta, Count-Min, Frequent Items, KLL, classic quantiles, REQ, t-digest, Tuple, VarOpt |
| `control` | Exact non-sketch reference used to measure error. | exact set |
| `negative_control` | Deliberately wrong merge baseline. | sum of per-leaf uniques |

Lean-side local laws already exist for HLL
([`HLLIdempotence.lean`](../lean3/FormalProofs/OPT/HLLIdempotence.lean)),
Count-Min ([`CountMinSketch.lean`](../lean3/FormalProofs/OPT/CountMinSketch.lean)),
KLL and GK quantiles
([`KLLLocalLaws.lean`](../lean3/FormalProofs/OPT/KLLLocalLaws.lean),
[`GKLocalLaws.lean`](../lean3/FormalProofs/OPT/GKLocalLaws.lean)). Bloom and
Theta/KMV are natural near-term Lean additions; t-digest needs the approximate
local-law machinery already drafted in
[`ApproximateLocalLaws.lean`](../lean3/FormalProofs/OPT/ApproximateLocalLaws.lean).
