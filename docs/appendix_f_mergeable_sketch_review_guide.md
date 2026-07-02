# Appendix F Mergeable-Sketch Review Guide

This document is a handoff for another LLM or reviewer auditing Appendix F,
the broad mergeable-sketch comparison suite, and the related input simulations.
It is intended to be self-contained: start here, then follow the file paths and
commands below.

## Review Goal

Evaluate whether Appendix F accurately describes the current experiments and
whether the generated plots, tables, benchmark code, learned-sketch code, and
formal-proof crosswalk support the paper claims.

The central paper claim is:

- C-TreePO reduces to classical mergeable sketches when the task oracle is set
  to an official sketch query function and the leaf/merge state is constrained
  to the sketch state.
- The HLL reproduction is the tightest register-space comparison.
- The broader suite compares official Python implementations and learned
  C-TreePO companions across distinct counting, frequency, quantiles, set
  operations, tuple accumulation, and sampling.
- Lean-backed rows, official empirical rows, and controls must stay clearly
  separated.

## Naming: `{learned_f, learned_g, learned_joint}` (Pass 4)

The broad classical-sketch suite collapses to **three method codenames**:

- `learned_f` — single f stage.
- `learned_g` — single g stage.
- `learned_joint` — any schedule that trains both components: `fg`,
  `gf`, `fgf`, `fgfgf`, etc. all roll up to this one codename.

The specific schedule is preserved as a per-row tag (`learned_variant`)
so reports can drill in if needed, but the headline grouping —
plot lines, compact-table columns, METHOD_GROUPS — uses the joint
codename. The short variant keys (used in `--learned-variants`) are
still `f`, `g`, `fg`, `gf`, plus arbitrary `{f, g}+` strings.

The schedule is the variant letters: each letter is one training stage
that updates *that* component while the other is held fixed. The
architecture is **uniform** across variants — `f_head`, `g`, and
`merge_adapter` always exist at the same sizes. "Held fixed" just means
the component's weights have `requires_grad=False`; there is no
identity-merge passthrough or fixed-sigmoid readout edge case. A frozen
component is initialized deterministically from the configured seed (or
loaded from `init_<comp>_from` if supplied).

- `learned_f` — run the **f stage** once. `g` stays at its base
  (deterministic seeded init, frozen — or `init_g_from` if supplied).
- `learned_g` — run the **g stage** once. `f_head` stays at its base
  (deterministic seeded init, frozen — or `init_f_from` if supplied).
- `learned_joint` / `learned_variant=fg` — f stage, then g stage: learn `f` against a base `g`;
  freeze the new `f` and learn `g` against it.
- `learned_joint` / `learned_variant=gf` — g stage, then f stage: learn `g` against a base `f`;
  freeze the new `g` and learn `f` against it.

(HLL parity in [learned_hll_parity.py](parallel/unified_g_v1/src/unified_g_v1/sketch/learned_hll_parity.py)
is a separate module with its own model that *does* use a structured
fixed readout — the differentiable classical HLL estimator over a
register-shaped state. That register-space-constrained variant is the
HLL story specifically; the broad `LearnedScalarSketchMergeModel` does
not have a structured fixed readout.)

Longer strings compose the same way by reading letters left-to-right.
`fgf` is "learn f, then g given that f, then refine f given the new g";
`fgfgf` is the same cadence repeated; any `{f, g}+` string is valid.
Each subsequent same-letter stage **warm-starts** from that component's
most recent prior checkpoint (caller-supplied `init_*_from` if given,
else the previous same-letter stage's output), so `f` and `g` accumulate
improvements across the schedule.

At the end of a run, the artifacts the caller needs for inference are:

- **final `f` model** — the f_head weights written by the *last* f stage
  (or the caller-supplied `init_f_from` if no f stage ran).
- **final `g` model** — symmetric: the g weights from the last g stage
  (or the caller-supplied `init_g_from` if no g stage ran).

Both are exposed as `artifacts["final_f_checkpoint"]` /
`artifacts["final_g_checkpoint"]` on the sequenced `FitResult` and as
`summary["final_f_checkpoint"]` / `summary["final_g_checkpoint"]`. The
helper `unified_g_v1.sketch.load_final_sketch_models` builds a
single-stage `TrainerConfig` whose model has both checkpoints loaded,
suitable for inference or for continuing the schedule.

Checkpoint handoff invariant: if a staged neural backend has modules that
define the shared state interface between `f` and `g`, those modules must
move across stage boundaries from the most recent stage checkpoint. In the
scalar-sketch model, `leaf_adapter` is such a shared-interface module:
component checkpoints still provide `f_head` or `g`/`merge_adapter`, while
`leaf_adapter` is loaded from the last completed stage, whatever side that
stage trained. This prevents the failure mode where a later stage freezes
an `f_head` trained on one leaf representation while silently replacing
that representation with a fresh random adapter.

The shared runner for this pattern is
`unified_g_v1.training.component_ladder.run_component_ladder`. Backends pass
a schedule such as `fgfg`, current component artifacts, shared-interface
artifacts, and a one-stage training callback. The runner handles the artifact
threading and stage directories; model-specific code only decides how to load
components, freeze the complement, optimize the active side, and emit updated
shared artifacts. Use that runner for new LLM, neural-operator, and sketch
ladder implementations instead of reimplementing per-component checkpoint
handoffs.

Aggregate-CSV row prefixes:

- single-letter variants → `learned_f_*` or `learned_g_*` (matches the
  codename).
- multi-letter schedules → `learned_joint_*` (collapses fg, gf, fgf,
  etc.). The specific schedule lives in the row's `learned_variant`
  field.

(Pass 3's `fg` was joint single-call training with one optimizer over
both components; Pass 4 dropped that. Pass-4 "joint" means a
multi-letter schedule of single-component stages — same end result, no
joint optimizer step.)

The cached pre-Pass-4 runs in
`outputs/classical_sketches_paper_gonly_vectorized_all_cpu_20260421_190158/`
have rows labelled `learned_gf_*` (Pass 2 joint training with the wrong
name) and `learned_fg_*` (Pass 3 joint training with f-priority init).
Pass 4 **does not rewrite** these prefixes; cached rows display verbatim
under the matching METHOD_GROUP. Reports that mix pre-Pass-4 and Pass-4
data will see those groups co-mingled.

The Pass-3 silent pareto-overwrite behaviour in
`parallel/unified_g_v1/src/unified_g_v1/training/backends/pytorch_loop.py`
is gone: `best_model.pt` now reflects the most recent epoch
unconditionally. `best_metric_value` and `best_epoch` remain in
`history` and the checkpoint payload for diagnostics only.

The Pass-3 reverse canonicalizer (`_canonical_sketch_name`) is removed
from `treepo/src/treepo/bench/reports/classical_sketches.py`.

The sequenced trainer is registered as `learned_sketch_sequence` in the
`unified_g_v1` trainer registry. The grid wrapper builds an outer
`TrainerConfig` via `learned_sketch_sequence_task(variant=...)`, dispatches
through `unified_g_v1.training.fit.fit()` (which resolves the trainer via
`resolve_trainer`), and the trainer recursively calls `fit()` once per
stage. **All training routes through `fit()`**; each stage produces a
`stage_<i>_<comp>/best_model.pt` checkpoint under the cell output dir.

Pass 4 deliberately does **not** introduce a new contract-runner adapter
in `src/tree/contract_runner.py`. The bench grid is internal
infrastructure where contract-driven dispatch would be ceremony without
a caller. If a paper-facing example later needs to declaratively invoke
sequenced sketch training, a `LearnedSketchSequenceAdapter` can be added
then.

## Environment And Ground Rules

Repository root:

```bash
cd /home/mlinegar/ThinkingTrees
source venv/bin/activate
```

Use `./venv/bin/python` if bare `python` is unavailable.

The worktree is intentionally dirty and contains many unrelated untracked and
modified files. Do not revert, delete, or normalize unrelated files. If you
make changes, only edit files needed for this review.

The current broad-suite run was produced on CPU and completed:

```text
outputs/classical_sketches_paper_gonly_vectorized_all_cpu_20260421_190158
```

The detached launcher for that run is:

```text
outputs/classical_sketches_paper_gonly_vectorized_all_cpu_20260421_190158_launcher
```

## Primary Paper File

Appendix F is here:

```text
paper/ctreepo/appendix/F_classical_parity.tex
```

The appendix is one `\section` with the literal title
`HyperLogLog Parity: Reproduction Details` (label
`app:classical-parity`) containing six subsections, in this order:

1. `Protocol` — native vs DataSketches HLL backends, oracle as a
   swappable knob (analytic and HLL-reference).
2. `Sweep Grid` — the **HLL parity** grid only: 240 classical cells +
   120 learned cells = 360 cells.
3. `Learned Variants in Detail` — definitions of `learned_f`, `learned_g`, and `learned_joint`
   (end-to-end) and `learned_g` (register-space constrained).
4. `The Register-Space Constraint in Detail` — the precision-13
   scaling-with-`L` numbers for both variants. (This subsection is the
   most quantitative part of the HLL story; do not skip it on review.)
5. `Training-Budget Verification` — 30 vs 150 epoch comparison showing
   the `learned_g` plateau is structural, not a budget artifact.
6. `Extension to Other Classical Sketches` — the broad multi-sketch
   suite. The status table, the five included broad-suite figures, and
   the compact learned-overlay table are floats inside or immediately
   following this subsection, not subsections of their own.

The paper preamble includes the staged sketch figure path:

```text
paper/ctreepo/preamble.tex
```

Relevant line:

```tex
\graphicspath{
    ...
    {assets/sketches/figures/}
}
```

The C-TreePO paper driver that currently includes Appendix F is:

```text
paper/ctreepo/main_new.tex
```

Relevant include:

```tex
\input{appendix/F_classical_parity}
```

There is also `paper/ctreepo/main_v2.tex`. If compiling the paper, inspect
which driver is intended before changing either one.

**Note on the appendix `\section` title.** The appendix is titled
`HyperLogLog Parity: Reproduction Details`, but by the
`Extension to Other Classical Sketches` subsection it is the broad
multi-sketch suite (status table, compact table, and all five included
figures). The user's stated goal is to treat HLL and the other
mergeable sketches in a unified way, so reviewers should suggest one of:

- Rename the section to something neutral like
  `Mergeable-Sketch Reproduction Details` and change the label
  `app:classical-parity` to match.
- Restructure so HLL parity and the broad suite are sibling
  subsections under a neutral parent, removing the implication that
  the broad suite is an "extension" rather than a coequal experiment.

Either change is small text-wise but improves the framing the user is
asking about.

## Paper-Facing Wording Constraints

Appendix F should describe the current plots plainly. Avoid wording that
frames a figure as a replacement for another draft or compares it to a prior
layout.

Use direct descriptions instead:

- "Broad classical-sketch comparison suite by sketch family."
- "Raw, unnormalized error curves for the official empirical rows."
- "Learned rows shown as excess RMSE above the official-sketch floor."

Avoid the phrase "normalized to the official-sketch floor" in captions,
even when the y-axis really is `RMSE_learned - min_official RMSE`. The
word "normalized" is the standard signal for **ratio** normalization in
this literature, and the gold-gap plot uses **subtraction**. The
appendix currently uses the phrase
`Learned rows normalized to the official-sketch floor.` at line 175 of
`paper/ctreepo/appendix/F_classical_parity.tex`; change that caption to
`Learned rows shown as excess RMSE above the official-sketch floor.` so
the prose matches the y-axis definition stated immediately afterward.

Also avoid overclaiming from the plots. If a panel has a zero or near-zero
official floor, describe excess RMSE rather than ratio-to-floor normalization.

## Current Generated Paper Assets

Staged figures live here:

```text
paper/ctreepo/assets/sketches/figures/
```

Current staged figures:

```text
paper/ctreepo/assets/sketches/figures/classical_sketches_distinct.png
paper/ctreepo/assets/sketches/figures/classical_sketches_frequency.png
paper/ctreepo/assets/sketches/figures/classical_sketches_gold_gap.png
paper/ctreepo/assets/sketches/figures/classical_sketches_method_learned_f.png
paper/ctreepo/assets/sketches/figures/classical_sketches_method_learned_g.png
paper/ctreepo/assets/sketches/figures/classical_sketches_method_learned_joint.png
paper/ctreepo/assets/sketches/figures/classical_sketches_method_official.png
paper/ctreepo/assets/sketches/figures/classical_sketches_quantile.png
paper/ctreepo/assets/sketches/figures/classical_sketches_sampling.png
paper/ctreepo/assets/sketches/figures/classical_sketches_set.png
paper/ctreepo/assets/sketches/figures/classical_sketches_summary.png
```

Staged tables live here:

```text
paper/ctreepo/assets/sketches/tables/
```

Current staged tables and machine-readable summaries:

```text
paper/ctreepo/assets/sketches/tables/classical_sketches_aggregate.csv
paper/ctreepo/assets/sketches/tables/classical_sketches_aggregate.json
paper/ctreepo/assets/sketches/tables/classical_sketches_compact.md
paper/ctreepo/assets/sketches/tables/classical_sketches_compact.tex
paper/ctreepo/assets/sketches/tables/classical_sketches_grid.md
paper/ctreepo/assets/sketches/tables/classical_sketches_grid.tex
paper/ctreepo/assets/sketches/tables/classical_sketches_report.md
```

The Appendix F compact table uses:

```tex
\input{assets/sketches/tables/classical_sketches_compact.tex}
```

## Current Broad-Suite Run Artifacts

Primary manifest:

```text
outputs/classical_sketches_paper_gonly_vectorized_all_cpu_20260421_190158/paper_bundle_manifest.json
```

Key facts in that manifest:

```text
aggregate_rows: 1095
learned_rows: 780
seeds: 0,1,2
capacities: small,medium,large
leaf_counts: 1,2,4,8,16
include_learned: true
learned_targets: all
learned_variants: fg,g
learned_epochs: 150
learned_n_train: 128
learned_n_val: 48
timestamp_utc: 2026-04-21T19:22:09.235055+00:00
git_sha: f38c8100e9714dbcd1c0d03d50169aab5d280f8d
```

The cached manifest above is a pre-Pass-4 artifact: it has only two
variants (`fg,g` in the manifest field, `learned_gf_*` row prefixes in
the CSV — Pass 2 mis-named joint training as `gf`). Regenerating the
bundle under Pass 4 now writes `f,g,fg,gf`. The Pass-3 reverse
canonicalizer is **removed** under Pass 4: cached `learned_gf_*` rows
display verbatim under the `learned_gf` METHOD_GROUP, even though they
were historically joint training rather than Pass-4 staged g-then-f.
The Pass-2 cache has no `learned_f_*` rows because that variant did
not exist yet; a fresh Pass-4 run is needed to populate that group.

The appendix has a `Sweep Grid` subsection with explicit cell counts
for HLL parity (240 + 120 = 360 cells) but **no equivalent cell count
for the broad suite**. Reviewers should request a one-sentence
addition to the `Extension to Other Classical Sketches` subsection
recording that the broad suite is
`seeds × capacities × leaf_counts × families` = 1095 aggregate rows
(780 of which are learned-companion rows). Without that sentence the
manifest cannot be cross-checked against the paper.

Primary report directory:

```text
outputs/classical_sketches_paper_gonly_vectorized_all_cpu_20260421_190158/reports/classical_sketches/
```

Important report files:

```text
classical_sketches_aggregate.csv
classical_sketches_aggregate.json
classical_sketches_compact.md
classical_sketches_compact.tex
classical_sketches_grid.md
classical_sketches_grid.tex
classical_sketches_report.md
classical_sketches_summary.png
classical_sketches_gold_gap.png
classical_sketches_method_official.png
classical_sketches_method_learned_f.png
classical_sketches_method_learned_g.png
classical_sketches_method_learned_joint.png
classical_sketches_distinct.png
classical_sketches_frequency.png
classical_sketches_quantile.png
classical_sketches_sampling.png
classical_sketches_set.png
```

The launcher files are:

```text
outputs/classical_sketches_paper_gonly_vectorized_all_cpu_20260421_190158_launcher/manifest.json
outputs/classical_sketches_paper_gonly_vectorized_all_cpu_20260421_190158_launcher/job.log
outputs/classical_sketches_paper_gonly_vectorized_all_cpu_20260421_190158_launcher/runner.sh
```

Check status after the fact with:

```bash
./venv/bin/python scripts/long_job.py status \
  --job-root outputs/classical_sketches_paper_gonly_vectorized_all_cpu_20260421_190158_launcher
```

## Quick Artifact Inspection Commands

Pretty-print the bundle manifest:

```bash
./venv/bin/python -m json.tool \
  outputs/classical_sketches_paper_gonly_vectorized_all_cpu_20260421_190158/paper_bundle_manifest.json \
  | sed -n '1,220p'
```

Count aggregate rows and learned variants:

```bash
./venv/bin/python - <<'PY'
import csv
from collections import Counter
from pathlib import Path

p = Path("outputs/classical_sketches_paper_gonly_vectorized_all_cpu_20260421_190158/reports/classical_sketches/classical_sketches_aggregate.csv")
rows = list(csv.DictReader(p.open()))
print("rows", len(rows))
print("families", dict(Counter(r["family"] for r in rows)))
print("learned_f", sum(1 for r in rows if r["sketch"].startswith("learned_f_") and not r["sketch"].startswith("learned_fg_")))
print("learned_g", sum(1 for r in rows if r["sketch"].startswith("learned_g_") and not r["sketch"].startswith("learned_gf_")))
print("learned_joint", sum(1 for r in rows if r["sketch"].startswith(("learned_joint_", "learned_fg_", "learned_gf_"))))
print("columns", rows[0].keys() if rows else [])
PY
```

The `learned_joint` count includes legacy `learned_fg_*` (Pass 3) and
`learned_gf_*` (Pass 2) prefixes alongside the canonical Pass-4
`learned_joint_*` prefix, so cached pre-Pass-4 runs and fresh Pass-4
runs both attribute their multi-letter rows to the same group.

Expected output against the cached pre-Pass-4 run (only `learned_g_*`
and `learned_gf_*` exist; the latter is historically joint training
mis-labelled `gf`):

```text
rows 1095
families {'distinct': 210, 'frequency': 120, 'quantile': 420, 'sampling': 120, 'set': 225}
learned_f 0
learned_g 390
learned_joint 390
```

A fresh Pass-4 regen with `--learned-variants f,g,fg,gf` populates
`learned_f`, `learned_g`, and `learned_joint` (the latter pooling fg,
gf, and any longer schedules).

The `quantile` family is roughly 3.5x the size of the next-largest
family. This is intentional, not a sampling bias: four official
quantile sketches (KLL, classic quantiles, REQ, t-digest) are scored
at multiple quantile levels per cell, so the quantile family
accumulates more rows per cell than the others. Reviewers asking "why
is quantile so over-represented?" should be redirected here.

Each variant has a unique `learned_<variant>_*` row prefix. Pass-4 fresh
runs produce all four (`learned_f_*`, `learned_g_*`, `learned_fg_*`,
`learned_gf_*`); the cached pre-Pass-4 run only has `learned_g_*` and
`learned_gf_*` (where the latter is historically joint training, not
Pass-4 staged g-then-f). Pass 4 does not rewrite the cached prefixes.

List staged assets:

```bash
find paper/ctreepo/assets/sketches -maxdepth 3 -type f | sort
```

Search Appendix F for draft-comparison language:

```bash
rg -n "replacement|prior layout|earlier draft|draft-comparison" \
  paper/ctreepo/appendix/F_classical_parity.tex
```

The command should return no paper-facing hits.

## Reproduction Commands

Install the optional official-sketch dependency:

```bash
./venv/bin/pip install -e 'treepo[sketches]'
./venv/bin/python -c "import datasketches; print(datasketches.__version__ if hasattr(datasketches, '__version__') else datasketches)"
```

Report-only regeneration from the completed output root:

```bash
PYTHONPATH=treepo/src:parallel/unified_g_v1/src \
./venv/bin/python -m treepo.bench.cli report classical-sketches \
  --output-root outputs/classical_sketches_paper_gonly_vectorized_all_cpu_20260421_190158 \
  --tables-dir paper/ctreepo/tables
```

Full broad-suite bundle reproduction:

```bash
PYTHONPATH=treepo/src:parallel/unified_g_v1/src \
./venv/bin/python scripts/run_classical_sketches_paper_bundle.py \
  --out-root outputs/classical_sketches_paper_repro \
  --jobs 32 \
  --seeds 0,1,2 \
  --capacities small,medium,large \
  --leaf-counts 1,2,4,8,16 \
  --include-learned \
  --learned-targets all \
  --learned-variants f,g,fg \
  --learned-epochs 150 \
  --learned-n-train 128 \
  --learned-n-val 48
```

Resume an interrupted broad-suite bundle without recomputing completed cells:

```bash
PYTHONPATH=treepo/src:parallel/unified_g_v1/src \
./venv/bin/python scripts/run_classical_sketches_paper_bundle.py \
  --out-root outputs/classical_sketches_paper_repro \
  --skip-existing \
  --jobs 32 \
  --seeds 0,1,2 \
  --capacities small,medium,large \
  --leaf-counts 1,2,4,8,16 \
  --include-learned \
  --learned-targets all \
  --learned-variants f,g,fg \
  --learned-epochs 150 \
  --learned-n-train 128 \
  --learned-n-val 48
```

Smoke the broad suite:

```bash
PYTHONPATH=treepo/src:parallel/unified_g_v1/src \
./venv/bin/python -m treepo.bench.cli suite classical-sketches \
  --out-root /tmp/classical_sketches_smoke \
  --jobs 1 \
  --seeds 0 \
  --capacities small,medium \
  --leaf-counts 1,4

PYTHONPATH=treepo/src:parallel/unified_g_v1/src \
./venv/bin/python -m treepo.bench.cli report classical-sketches \
  --output-root /tmp/classical_sketches_smoke
```

Legacy HLL parity reproduction:

```bash
PYTHONPATH=treepo/src:parallel/unified_g_v1/src \
./venv/bin/python scripts/run_classical_parity_benchmark.py \
  --out outputs/classical_parity
```

Paper helper script for broad sketches:

```bash
CLASSICAL_SKETCH_SKIP_EXISTING=1 \
paper/ctreepo/scripts/regen_classical_sketches.sh
```

## Test Commands

Focused tests for the broad suite and learned scalar path:

```bash
PYTHONPATH=treepo/src:parallel/unified_g_v1/src \
./venv/bin/python -m pytest \
  treepo/tests/sketches/test_sketch_adapter_contracts.py \
  treepo/tests/sketches/test_broad_classical_sketches.py \
  parallel/unified_g_v1/tests/test_classical_hll_parity_fit.py \
  parallel/unified_g_v1/tests/test_classical_sketch_grid_fit.py \
  parallel/unified_g_v1/tests/test_learned_scalar_sketch_fit.py \
  -q
```

The vectorization smoke tests previously passed for:

```text
parallel/unified_g_v1/tests/test_learned_scalar_sketch_fit.py
parallel/unified_g_v1/tests/test_classical_sketch_grid_fit.py
```

Lint the report code if modified:

```bash
./venv/bin/python -m ruff check treepo/src/treepo/bench/reports/classical_sketches.py
```

## Broad-Suite Code Map

Top-level benchmark runner:

```text
treepo/src/treepo/bench/classical_sketches.py
```

What to inspect there:

- `ClassicalSketchComparisonConfig`: grid parameters, capacity knobs, learned
  options, input sizes, family selection.
- `_token_docs`: Zipf-like integer-token stream generator over
  `universe_size`.
- `_float_docs`: Gaussian-mixture real-valued stream generator for quantiles.
- `_run_distinct`: native HLL, DataSketches HLL, CPC, Theta, exact set,
  negative sum-of-leaf-uniques baseline.
- `_run_frequency`: Count-Min and Frequent Items against exact item counts.
- `_run_quantile`: KLL, classic quantiles, REQ, and t-digest against exact
  quantile/rank truth.
- `_run_set`: Theta union, intersection, and A-not-B set-expression
  cardinalities.
- `_run_sampling`: Tuple accumulator and VarOpt sampling rows.
- `_metric_row`: RMSE, relative RMSE, mean absolute relative error,
  merge-schedule spread, theoretical bound coverage, memory.
- `_attach_official_floors`: computes the best official empirical floor by
  family and query and stores excess RMSE.

Grid builder:

```text
treepo/src/treepo/bench/suites/classical_sketches.py
```

What to inspect there:

- Capacity presets for `small`, `medium`, and `large`.
- Cross product of seeds, capacities, and leaf counts.
- Output directory layout under `out_root/classical_sketches/paper`.
- Whether `execution_backend` routes through `unified_g` or the package-local
  TreePO fallback.

CLI:

```text
treepo/src/treepo/bench/cli.py
```

What to inspect there:

- `treepo-bench suite classical-sketches`.
- `treepo-bench report classical-sketches`.
- Learned flags: `--include-learned`, `--learned-targets`,
  `--learned-variants`, `--learned-epochs`, `--learned-n-train`,
  `--learned-n-val`.

Report and plotting code:

```text
treepo/src/treepo/bench/reports/classical_sketches.py
```

What to inspect there:

- Aggregation from cell-level rows into family/sketch/query/capacity/leaf rows.
- Confidence intervals over seeds.
- Compact table generation.
- Summary plot generation.
- Gold-gap plot generation.
- Method-separated raw plots for official, learned `g`, and learned `g+f`.
- Family-specific plots.
- Staging assumptions for `paper/ctreepo/assets/sketches`.

Bundle script:

```text
scripts/run_classical_sketches_paper_bundle.py
```

What to inspect there:

- One-shot run, report, and asset staging.
- `--skip-existing` resumability.
- Manifest generation.
- Copying report figures and tables to paper assets.

Paper helper script:

```text
paper/ctreepo/scripts/regen_classical_sketches.sh
```

What to inspect there:

- Environment variables for output root, jobs, seeds, capacities, leaf counts,
  learned variants, and skip-existing behavior.

## Official Sketch Adapter Code Map

Common protocol:

```text
treepo/src/treepo/sketches/protocol.py
```

Tree reducer:

```text
treepo/src/treepo/sketches/tree_reducer.py
```

Adapters:

```text
treepo/src/treepo/sketches/adapters/hll_native.py
treepo/src/treepo/sketches/adapters/hll_datasketches.py
treepo/src/treepo/sketches/adapters/datasketches_cardinality.py
treepo/src/treepo/sketches/adapters/datasketches_frequency.py
treepo/src/treepo/sketches/adapters/datasketches_quantiles.py
treepo/src/treepo/sketches/adapters/datasketches_tuple_sampling.py
```

Key interface expected by the benchmark:

```text
update
encode
merge
query
serialize
serialized_size_bytes
memory_bytes
config
```

Review points:

- Each adapter should be locally mergeable through the same reducer surface.
- Official empirical rows should use Apache DataSketches when available.
- Native HLL is the in-repo register implementation.
- Redis HLL and BigQuery HLL++/KLL are narrative industry references, not
  required runtime dependencies for reproducible tests.
- Missing `datasketches` should lead to skipped tests or install guidance, not
  a hard failure for base installs.

## Learned Broad-Suite Code Map

Unified `fit()` wrapper for broad official baselines:

```text
parallel/unified_g_v1/src/unified_g_v1/sketch/classical_sketch_grid.py
```

Generic learned scalar sketch:

```text
parallel/unified_g_v1/src/unified_g_v1/sketch/learned_scalar_sketch.py
```

Learned broad-grid target builder:

```text
parallel/unified_g_v1/src/unified_g_v1/sketch/learned_sketch_grid.py
```

Unified trainer entry point:

```text
parallel/unified_g_v1/src/unified_g_v1/training/fit.py
```

Review points for learned broad-grid rows:

- `learned_f_*` trains only the readout `f`; the merge is a
  deterministic identity passthrough (mean over child states; leaves
  bypass the merge MLP). Optionally a pretrained merge can be supplied
  via `init_g_from`, in which case the loaded merge weights are used
  and frozen.
- `learned_g_*` trains the leaf compressor and merge/local-law machinery but
  uses a fixed non-trainable scalar readout (target-scaled sigmoid of
  state[0]). Optionally a pretrained readout can be supplied via
  `init_f_from`.
- `learned_fg_*` trains both latent merge state `g` and scalar readout
  `f` jointly. `f` is constructed before `g` so it claims the first
  random draws under the configured seed (f-priority init is the only
  coherent ordering since `g` is conceptually a function of `f`).
- In the broad generic path, the fixed readout is a target-scaled sigmoid of
  the first latent coordinate. It is not the HLL classical register readout.
- The HLL-specific learned `g` path is different: it constrains the state to
  HLL register space and applies the differentiable HLL estimator formula.
- Do not conflate broad `learned_g_*` rows with HLL register-space parity rows.

Vectorization code to inspect in `learned_scalar_sketch.py`:

- `_encode_leaf_grid`
- `_merge_pair_batch`
- `_merge_states_batch`
- `predict_scalars`
- `forward_tree`

Expected vectorization behavior:

- Rectangular leaf grids should be encoded as a batch.
- Ready merge nodes should be merged in tensor batches.
- Per-example Python loops should only remain as ragged fallbacks or outer
  orchestration.

## HLL Parity Code Map

Legacy HLL parity script:

```text
scripts/run_classical_parity_benchmark.py
```

HLL parity implementation and report:

```text
parallel/unified_g_v1/src/unified_g_v1/sketch/classical_parity.py
parallel/unified_g_v1/src/unified_g_v1/sketch/learned_hll_parity.py
parallel/unified_g_v1/src/unified_g_v1/sketch/classical_parity_report.py
```

HLL parity paper tables:

```text
paper/ctreepo/tables/classical_parity_hll.md
paper/ctreepo/tables/classical_parity_hll.tex
paper/ctreepo/assets/hll/tables/classical_parity_hll.md
paper/ctreepo/assets/hll/tables/classical_parity_hll.tex
```

Expected output root for the HLL parity path:

```text
outputs/classical_parity/hll/
```

Review points:

- Native HLL should be byte-exact under register-wise max merge.
- DataSketches HLL should be estimate-equivalent, but byte equality is not
  expected because internal representations can transition between modes.
- `oracle_kind="analytic"` uses true distinct cardinality. The paper
  renders this as `analytic` verbatim.
- `oracle_kind="hll_reference"` uses the HLL scoring head as the oracle.
  The paper renders the snake_case identifier as `HLL-reference`
  (hyphenated, capitalized) in prose. Treat the two spellings as the
  same setting; do not introduce a third.
- Learned HLL `g` should be described as register-space constrained only in the
  HLL parity context.

## Input Simulation Families

The broad Appendix F suite uses synthetic local Python inputs, not external
datasets. The important generators are in:

```text
treepo/src/treepo/bench/classical_sketches.py
parallel/unified_g_v1/src/unified_g_v1/sketch/learned_scalar_sketch.py
```

Token documents:

- `_token_docs(config)` samples integer streams from a Zipf-like distribution
  over `universe_size`.
- Default broad-suite token stream sizes are controlled by `n_docs`,
  `min_tokens`, `max_tokens`, `leaf_size`, and `n_leaves`.
- These token streams feed distinct counting, frequency, set, tuple, sampling,
  and learned scalar targets.

Float documents:

- `_float_docs(config)` samples real-valued streams from a Gaussian mixture.
- The base component is `normal(0, 1)`.
- A 15% minority component is shifted by an additional `normal(3.0, 0.8)`
  draw, producing a long right tail. Confirm the share and shift
  parameters at
  `treepo/src/treepo/bench/classical_sketches.py` lines 236 to 243
  before quoting them in the paper.
- These streams feed official quantile sketches and exact quantile truth in
  the package-local broad suite.

Learned scalar target inputs:

- The learned scalar sketch path reuses HLL parity document generation for
  integer-token streams.
- Exact distinct target: `len(set(tokens))`.
- Exact frequency target: count of `focus_token`.
- Exact total weight target: stream length.
- Exact quantile target: empirical quantile of integer tokens.
- Exact set targets: tagged sets split around `universe_size`, then union,
  intersection, or A-not-B cardinality.
- Official reference targets: HLL, CPC, Theta, Count-Min, Frequent Items, KLL,
  classic quantiles, REQ, t-digest, Theta set operations, Tuple accumulator,
  and VarOpt total weight.

Review points:

- Confirm that each Appendix F query head matches the input type used by the
  code.
- Confirm that set-operation learned targets use tagged-set semantics, not
  ordinary untagged distinct counts.
- Confirm that quantile learned targets using integer tokens are described as
  scalar target learning, not identical to the package-local float mixture
  quantile rows unless the code path being discussed is explicit.
- Confirm that `leaf_count` controls the number of chunks and that smaller
  leaves imply more nodes, not necessarily lower runtime.

## Formal-Proof Crosswalk

Primary status crosswalk:

```text
lean3/FormalProofs/OPT/ClassicalSketchLocalLaws.lean
```

Lean-backed sketch files:

```text
lean3/FormalProofs/OPT/HLLIdempotence.lean
lean3/FormalProofs/OPT/CountMinSketch.lean
lean3/FormalProofs/OPT/KLLLocalLaws.lean
lean3/FormalProofs/OPT/GKLocalLaws.lean
lean3/FormalProofs/OPT/BigramSketch.lean
lean3/FormalProofs/OPT/WorkedExampleCMSTree.lean
```

Import registration:

```text
lean3/FormalProofs.lean
```

Review points (the table groups several families per row; the
descriptions below match the actual row layout in
`paper/ctreepo/appendix/F_classical_parity.tex` lines 140 to 159, not a
hypothetical one-family-per-row layout):

- Row `Native HLL`: theorem-backed; in-repo register sketch with
  byte-exact tree/flat check.
- Row `DataSketches HLL`: official empirical; estimate-equivalence
  check, not byte-equivalence (representation modes vary).
- Row `Count-Min, KLL` (combined): theorem-backed and official
  empirical; formal local-law artifacts plus DataSketches adapters.
- Row `GK`: theorem-backed for sequential-merge only; **no broad
  runtime grid row**. Do not list GK among the broad-suite rows.
- Row `CPC, Theta/KMV, Frequent Items, classic quantiles, REQ, t-digest`
  (combined): official empirical only; no formal theorem claimed.
- Row `Tuple, VarOpt` (combined): official empirical only.
- Row `exact set`: control; exact distinct-count ground truth.
- Row `sum of leaf uniques`: negative control; deliberately wrong
  non-mergeable distinct-count baseline.

Bigram/Markov-count sketches are Lean-backed examples (under
`lean3/FormalProofs/OPT/BigramSketch.lean` and
`lean3/FormalProofs/OPT/WorkedExampleCMSTree.lean`) but they are
**not** official DataSketches runtime rows in Appendix F's table; they
appear in the proof crosswalk only.

## Related Simulation And Documentation Files

Appendix F touches the mergeable-sketch story, but the broader C-TreePO
simulation narrative is spread across several files. These are the most useful
files to inspect if asked how the input simulations connect to the paper.

Main empirical paper section:

```text
paper/ctreepo/sections/07_empirical.tex
```

Classical parity and broad-sketch documentation:

```text
docs/classical_parity_benchmark.md
docs/tree_execution_batching.md
docs/mergeable_method_validation_paper_note.md
docs/mergeable_method_validation_report.md
docs/mergeable_ablation_examples.md
```

Learned sketch simulation documentation:

```text
docs/learned_sketch_simulation.md
docs/treepo_regularized_objective.md
docs/treepo_preference_optimization.md
docs/simulation_suite.md
docs/paper_simulation_map.md
```

Lean-side crosswalk (the natural place to verify theorem-backed status
claims against the actual Lean files):

```text
lean3/docs/PAPER_TO_LEAN_MAP.md
```

Learned sketch simulation scripts:

```text
scripts/run_learned_sketch_simulation.py
scripts/run_learned_sketch_sampling_sweep.py
scripts/report_learned_sketch_regularized_objective.py
scripts/plot_learned_sketch_simulation.py
scripts/plot_learned_sketch_sampling_sweep.py
scripts/plot_learned_sketch_distance_to_floor.py
```

In-repo learned sketch implementation:

```text
src/tree/learned_sketch.py
src/tree/learned_sketch_simulation.py
src/tree/hll_merge_learning_simulation.py
```

Mergeable ablation scripts:

```text
scripts/run_mergeable_ablation_simulation.py
scripts/run_mergeable_generalization_sweep.py
scripts/run_mergeable_k_m_sweep.py
scripts/run_mergeable_k_recovery.py
scripts/run_mergeable_param_recovery.py
scripts/plot_mergeable_complexity_ladder.py
scripts/plot_mergeable_ceilings.py
scripts/plot_mergeable_nonlanguage_suite.py
scripts/plot_mergeable_nonlanguage_coverage.py
scripts/plot_mergeable_chunk_quality_sweep.py
```

Topic/LDA-related input simulation docs and code:

```text
docs/tree_topic_simulation_suite.md
docs/tree_relevant_lda_simulation_ladder.md
docs/segmented_lda_ctreepo_end_to_end.md
docs/segment_lda_ops_simulation_spec.md
src/tree/segmented_lda_ctreepo_simulation.py
src/tree/segment_lda_ops_weight_recovery_simulation.py
scripts/run_segmented_lda_ctreepo_simulation.py
scripts/grid_segmented_lda_ctreepo_simulation.py
scripts/run_segment_lda_ops_weight_recovery_simulation.py
```

Markov and other simulation-map references:

```text
docs/paper_simulation_map.md
docs/simulation_suite.md
docs/simulation_theory_alignment_status_20260311.md
docs/markov_alignment_audit.md
```

## Acceptance Checklist For Appendix F

Use this checklist before declaring Appendix F paper-ready.

- The text distinguishes HLL parity from the broad multi-sketch suite.
- The text distinguishes HLL-specific learned `g` register-space constraints
  from generic broad-suite `learned_g_*` fixed-readout rows.
- The text distinguishes theorem-backed, official empirical, and control rows.
- The table does not imply Lean proofs for CPC, Theta, Frequent Items, classic
  quantiles, REQ, t-digest, Tuple, or VarOpt.
- GK is marked theorem-backed for sequential assumptions only and is not
  included as a broad runtime grid row.
- Redis HLL and BigQuery HLL++/KLL are cited only as industry references, not
  required runtime dependencies.
- The "official floor" is described as the best official empirical relative
  RMSE by family and query.
- The "distance to official floor" is described as excess RMSE in the
  paper prose, not a ratio. (The gold-gap figure caption is covered by
  a separate checklist item below.)
- Summary and method plots use direct descriptions and do not compare against
  another draft.
- Exact controls and negative controls are labeled correctly.
- The sum-of-leaf-uniques row is described as a deliberately wrong
  non-mergeable distinct-count baseline.
- Frequency claims acknowledge that Count-Min/Frequent Items are query-specific
  and should be compared against exact item counts or top-k truth.
- Quantile claims distinguish value error/rank truth from cardinality error.
- Set-operation claims use Theta union/intersection/A-not-B, not generic HLL.
- The compact table rows match the aggregate CSV after report regeneration.
- The staged assets match the figure names included by Appendix F.
- The paper compiles with the current `\graphicspath` and table input paths.
- Both the HLL parity sweep and the broad suite enter training through
  the unified `fit()` abstraction (`learned_hll_parity` for HLL,
  `classical_sketch_grid` and `learned_sketch_grid` for the broad
  suite); no separate trainer is introduced for either path.
- The four canonical variant names `learned_f`, `learned_g`,
  `learned_fg`, `learned_gf` (short keys `f`, `g`, `fg`, `gf`) are used
  everywhere in the broad-suite path — paper prose, figure include
  paths and labels, aggregate CSV row prefixes, manifest
  `learned_variants` field. The variant string is interpreted by the
  sequenced trainer in `unified_g_v1`; longer `{f,g}+` sequences
  (`fgf`, etc.) are accepted without extra plumbing and land in
  METHOD_GROUP `learned_other`.
- For each multi-letter variant cell, the cell's output directory
  contains one `stage_<i>_<comp>/best_model.pt` per letter, in order.
- All training routes through the centralised `fit()`. The grid
  wrapper builds a `TrainerConfig` via `learned_sketch_sequence_task`
  and calls `unified_g_v1.training.fit.fit(...)`; the sequenced
  trainer is registered in the `unified_g_v1` trainer registry and
  recursively calls `fit()` once per stage. No bench-internal sketch
  code path bypasses `fit()`.
- The training checkpoint `best_model.pt` reflects the most recent
  epoch's weights, not the best-seen epoch across resumes. A run that
  ends at a worse epoch produces a `best_model.pt` reflecting that
  worse epoch; `best_metric_value` and `best_epoch` are recorded in
  history and the checkpoint payload for diagnostics only.
- The gold-gap figure caption uses "excess RMSE" or equivalent
  subtraction language; not "normalized to" or "ratio to".

## Potential Issues To Watch

Official floors can be zero or near-zero for exact or near-exact families. That
is why the gold-gap plot uses excess RMSE rather than division by the floor.

The broad aggregate CSV currently has no populated `method_class` column; use
the `sketch` prefix and `implementation_status` fields for grouping unless the
report code has been updated.

The bundle manifest records the core run and staged assets. If plots are
regenerated report-only after a bundle run, verify actual file presence in
`paper/ctreepo/assets/sketches/figures/` rather than relying only on the
manifest's staged-figure list.

The generic broad learned `g` path is a fixed scalar-readout experiment, not a
formal claim that all official sketch states have differentiable fixed
readouts.

The HLL register-space result is stronger than the generic broad `learned_g`
result. Keep those paragraphs separate.

The broad suite is CPU-local and deterministic enough for paper regeneration,
but official DataSketches internals can have representation-dependent
serialization. Compare estimates and metrics, not serialized byte equality,
except for native HLL.

## Suggested Review Order

1. Read `paper/ctreepo/appendix/F_classical_parity.tex`.
2. Inspect the staged figures and compact table under
   `paper/ctreepo/assets/sketches/`.
3. Inspect `classical_sketches_aggregate.csv` and confirm row counts and
   family coverage.
4. Read `treepo/src/treepo/bench/classical_sketches.py` to verify input
   generators, truth functions, official rows, and controls.
5. Read `parallel/unified_g_v1/src/unified_g_v1/sketch/learned_scalar_sketch.py`
   to verify learned `g` and learned `g+f` semantics.
6. Read `treepo/src/treepo/bench/reports/classical_sketches.py` to verify plot
   and table semantics.
7. Read `lean3/FormalProofs/OPT/ClassicalSketchLocalLaws.lean` to verify the
   theorem-backed status table.
8. Run the focused tests if code was changed.
9. Compile the paper driver if LaTeX dependencies are available.

## Minimal Reviewer Prompt

If passing this to another LLM, use the following prompt:

```text
You are reviewing Appendix F of the C-TreePO paper in /home/mlinegar/ThinkingTrees.
Start with docs/appendix_f_mergeable_sketch_review_guide.md. Audit whether
paper/ctreepo/appendix/F_classical_parity.tex accurately reflects the broad
mergeable-sketch suite, generated artifacts, input simulations, learned_g and
learned_f and learned_fg variants, official Apache DataSketches baselines, controls, and
Lean-backed status claims. Do not revert unrelated dirty worktree changes.
Prioritize correctness of claims, figure/table consistency, reproducibility,
and paper-facing wording.
```
