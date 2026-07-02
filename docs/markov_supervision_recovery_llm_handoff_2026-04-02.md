# Markov Supervision-Recovery / Family-Grids Handoff 2026-04-02

This note is a handoff for another LLM or human engineer taking over the
current supervision-recovery / family-grids line of work.

It is written to answer four things clearly:

1. what repo and research context we are in
2. what the immediate experimental problem is
3. what has already been run, what is authoritative, and what was stopped
4. what experiment and reporting design is prepared next

## Executive Summary

The immediate working lane is a **controlled Markov full-document
supervision-recovery setting**. We are using that lane because it is cheap,
fast enough to iterate on, and structured enough to isolate supervision and
architecture effects cleanly.

The broader project is not limited to Markov experiments:

- the repo also supports tree/LLM training paths, runtime eval, and C-TreePO
  simulation/theory work
- we have already been consolidating the experiment/reporting API so Markov,
  tree/LLM, runtime, and CTreePO can share one control-plane and reporting
  surface
- the current Markov family-grids work should be treated as a **base empirical
  argument surface**, not the final whole project

The active scientific issue is:

- on the historical `10240`-doc recoverable benchmark,
  `official_fno/full100` performs much better than the historical
  **root-only** `tree_neural/full100`
- but locally supervised tree packages do much better, and in some cases beat
  FNO
- that means the important unresolved question is not whether the tree family
  can work at all, but whether the **root-only tree recipe** was unfairly weak
  or whether there is a real root-only bottleneck

The prepared next step is an **overnight 16-MIG parity grid** that adds
geometry/parity evidence directly into the existing family-grids report lineage
without corrupting the package ladders. That run is prepared in code but has
**not** been launched yet.

## Repo And Project Context

The repo serves two large scopes:

- **C-TreePO / theory / certification**
- **Semantic Forests / systems / training / runtime**

For this lane, the important point is:

- we are currently working in a **simple Markov-style full-doc benchmark**
  setting because it is controlled and cheap
- but the design target is broader: the experiment/reporting API should also
  support tree/LLM training, runtime eval, and future cross-family comparison

Relevant infrastructure:

- canonical experiment layer:
  [src/experiments/contracts.py](/home/mlinegar/ThinkingTrees/src/experiments/contracts.py),
  [src/experiments/reporting.py](/home/mlinegar/ThinkingTrees/src/experiments/reporting.py),
  [src/experiments/adapters.py](/home/mlinegar/ThinkingTrees/src/experiments/adapters.py)
- Markov pipeline:
  [run_markov_optimization_tradeoff_pipeline.py](/home/mlinegar/ThinkingTrees/scripts/run_markov_optimization_tradeoff_pipeline.py)
- tree/FNO capacity runner:
  [run_tree_neural_full_doc_mig.py](/home/mlinegar/ThinkingTrees/scripts/run_tree_neural_full_doc_mig.py)
- current family-grids report merger:
  [report_markov_cohort_compare.py](/home/mlinegar/ThinkingTrees/scripts/report_markov_cohort_compare.py)

## Immediate Problem

### Broad argument we want

The broad report argument is roughly:

- tree and FNO should be compared fairly
- supervision budgets should be explicit
- local supervision and local-law style controls should not be silently merged
  into direct label budgets
- the report should include both older historical grids and newer cohort-based
  or parity-based evidence

### Specific scientific problem right now

We need to explain the apparent contradiction:

- historically, we remember tree and FNO being near parity in some settings
- but in the current historical family-grids summary, the **root-only**
  `tree_neural/full100` row is much worse than `official_fno/full100`

The likely resolution is:

- the bad historical row is specifically a **root-only tree recipe**
- the stronger tree results come from packages with **extra local supervision**
- so the parity claim may still be true in the right regime, but we need to
  show it explicitly and cleanly

### Why this matters

The user’s position is that a parity regime should exist because:

- if the tree leaf is large enough that the whole document effectively fits in
  context, then the tree should collapse toward an FNO-like computation

We are not yet claiming strict architectural identity in the report. The
prepared parity panel is meant to support the **empirical** claim first.

## Current Authoritative Historical Evidence

The main authoritative historical source is:

- [summary.json](/home/mlinegar/ThinkingTrees/outputs/markov_supervision_recovery_strong_tree_autoqueue_refresh_grid_20260329_020856/supervision_recovery/summary.json)

This is the best existing historical supervision-recovery grid. It contains:

- `train_doc_counts = [1024, 4096, 10240]`
- `package_order_len = 23`
- a full historical tree/FNO package sweep

Important `10240`-doc anchor values from that summary:

### Recoverable

- `official_fno/full100 = 0.001953125`
- `official_fno_sumlen/full100 = 0.01171875`
- `tree_neural/full100 = 0.014195965137332678`
- `tree_neural/full0_leaf_full100_internal_count100 = 0.0006374306976795197`

Interpretation:

- root-only tree is much worse than the best root-only FNO
- locally supervised tree can be much better than FNO
- therefore the historical data supports a **tree-family success** story, but
  not a clean **root-only tree parity** story

### Structural

- `official_fno/full100 = 0.005859375`
- `tree_neural/full100 = 0.07287130504846573`
- `tree_neural/full100_leaf_full100_internal_count100 = 0.026150060817599297`

Interpretation:

- structural is much harder
- even the strong locally supervised tree row does not close the structural
  gap the same way recoverable does

## Current Report Lineage

The main combined report lineage we care about is the family-grids line rooted
under:

- [markov_supervision_recovery_strong_tree_autoqueue_refresh_grid_20260329_020856](/home/mlinegar/ThinkingTrees/outputs/markov_supervision_recovery_strong_tree_autoqueue_refresh_grid_20260329_020856)

Current best merged report version:

- [report.pdf](/home/mlinegar/ThinkingTrees/outputs/markov_supervision_recovery_strong_tree_autoqueue_refresh_grid_20260329_020856/tradeoff_report_family_grids_20260401_223500/report.pdf)
- [summary.json](/home/mlinegar/ThinkingTrees/outputs/markov_supervision_recovery_strong_tree_autoqueue_refresh_grid_20260329_020856/tradeoff_report_family_grids_20260401_223500/summary.json)

That report is currently:

- `report_kind = markov_family_grids_with_cohort_compare_v2`
- `historical_grid_present = true`
- `cohort_compare_present = true`
- `geometry_parity_present = null`

It currently includes these figure classes:

- `Dense Full-Doc Anchor`
- `Recoverable Package Ladder`
- `Structural Package Ladder`
- `Recoverable Ordered Families`
- `Structural Ordered Families`
- `Recoverable R10/R20 Cohort Comparison`
- `Structural R10/R20 Cohort Comparison`

Important limitation:

- the report does **not yet** include the new geometry/parity panel
- that is the whole point of the prepared overnight parity grid

## Explicit Cohort Roots

These are the main cohort directories now in play.

### Historical grid / template root

- [markov_supervision_recovery_strong_tree_autoqueue_refresh_grid_20260329_020856](/home/mlinegar/ThinkingTrees/outputs/markov_supervision_recovery_strong_tree_autoqueue_refresh_grid_20260329_020856)

This is the authoritative historical package-grid source.

### R10 cohort root

- [markov_supervision_recovery_r10_local_law_20260331_0206](/home/mlinegar/ThinkingTrees/outputs/markov_supervision_recovery_r10_local_law_20260331_0206)

Artifacts present:

- [supervision_recovery/summary.json](/home/mlinegar/ThinkingTrees/outputs/markov_supervision_recovery_r10_local_law_20260331_0206/supervision_recovery/summary.json)
- [tradeoff_report/summary.json](/home/mlinegar/ThinkingTrees/outputs/markov_supervision_recovery_r10_local_law_20260331_0206/tradeoff_report/summary.json)

Summary facts:

- `status = ready`
- `train_doc_counts = [1024, 4096, 10240]`
- `n_family_rows = 42`

### R20 partial package-capacity root

- [markov_supervision_recovery_r20_local_law_package_capacity_locked_nested_prepared16_retry_20260401_043101](/home/mlinegar/ThinkingTrees/outputs/markov_supervision_recovery_r20_local_law_package_capacity_locked_nested_prepared16_retry_20260401_043101)

This contains:

- completed or partial package-capacity outputs
- [combined_scheduler_status.json](/home/mlinegar/ThinkingTrees/outputs/markov_supervision_recovery_r20_local_law_package_capacity_locked_nested_prepared16_retry_20260401_043101/combined_scheduler_status.json)

Important caveat:

- the R20 launcher was later stopped
- some progress JSON in this root is therefore **stale** and may still say
  `running`
- use it as an artifact snapshot, not as a reliable live-status source

### Stopped root-only diagnosis roots

These two roots were experimental diagnosis runs and were stopped:

- [tree_root_only_parity_grid_base_full100_20260402_000500](/home/mlinegar/ThinkingTrees/outputs/tree_root_only_parity_grid_base_full100_20260402_000500)
- [tree_root_only_parity_grid_base_full100_exploratory_20260402_015334](/home/mlinegar/ThinkingTrees/outputs/tree_root_only_parity_grid_base_full100_exploratory_20260402_015334)

Both have launcher manifests and plan/status files, but neither produced a
completed diagnosis summary useful as main evidence.

Launcher status confirms both are stopped:

- historical heavy diagnosis launcher: `running = false`
- exploratory diagnosis launcher: `running = false`

Their `progress` blobs can still show stale in-flight items because they were
captured before termination.

## What Has Already Been Run

This is the most useful split for another LLM taking over.

### Reused as authoritative evidence

Use these directly:

1. historical supervision-recovery grid summary
   - [summary.json](/home/mlinegar/ThinkingTrees/outputs/markov_supervision_recovery_strong_tree_autoqueue_refresh_grid_20260329_020856/supervision_recovery/summary.json)
2. existing family-grids report lineage
   - [tradeoff_report_family_grids_20260401_223500](/home/mlinegar/ThinkingTrees/outputs/markov_supervision_recovery_strong_tree_autoqueue_refresh_grid_20260329_020856/tradeoff_report_family_grids_20260401_223500)
3. explicit R10 cohort directory
   - [markov_supervision_recovery_r10_local_law_20260331_0206](/home/mlinegar/ThinkingTrees/outputs/markov_supervision_recovery_r10_local_law_20260331_0206)
4. completed partial R20 capacity outputs
   - [markov_supervision_recovery_r20_local_law_package_capacity_locked_nested_prepared16_retry_20260401_043101](/home/mlinegar/ThinkingTrees/outputs/markov_supervision_recovery_r20_local_law_package_capacity_locked_nested_prepared16_retry_20260401_043101)

### Already ran, but not authoritative for the new parity panel

There are older parity-ish outputs such as:

- [fair_parity_run_summary.json](/home/mlinegar/ThinkingTrees/outputs/markov_capacity_v2_confirm_2048_parity_20260326_0129/fair_parity_run_summary.json)
- [fair_parity_run_summary.json](/home/mlinegar/ThinkingTrees/outputs/markov_publication_bundle_no10240_reuse_today_20260324_221500/tree_fno_parity/fair_parity_run_summary.json)
- [tree_fno_fair_parity_summary.json](/home/mlinegar/ThinkingTrees/outputs/markov_optimization_pipeline_20260324_024113_full/oracle_budget_frontier/tree_fno_fair_parity_summary.json)

Why they are not the main evidence for the new report panel:

- some are keyed to different gate train-doc counts
- they are collapsed parity summaries rather than explicit geometry rows
- they are not already in the exact report lineage/shape we now want

### Already ran but stopped / incomplete

- [tree_root_only_parity_grid_base_full100_20260402_000500](/home/mlinegar/ThinkingTrees/outputs/tree_root_only_parity_grid_base_full100_20260402_000500)
- [tree_root_only_parity_grid_base_full100_exploratory_20260402_015334](/home/mlinegar/ThinkingTrees/outputs/tree_root_only_parity_grid_base_full100_exploratory_20260402_015334)

These are not current evidence. They are mostly useful as examples of what not
to relaunch:

- the original diagnosis runner was too serial and under-filled the 16 MIGs
- exploratory mode still used only one active GPU job
- the user explicitly does **not** want to run that path right now

## What Problem We Are Solving Next

The next prepared experiment is designed to answer:

- can root-only tree approach or match the historical full-root FNO on
  recoverable `full100 @ train_docs=10240` when the recipe and leaf geometry
  are fairer?

It is not designed to:

- continue the R10/R20 ladder overnight
- rerun multi-seed package-capacity locked sweeps
- do a broad structural sweep
- flood the report with new package names

The goal is narrower and directly report-relevant:

- add one clean **geometry/parity evidence block** to the existing family-grids
  report

## Constraints And Design Rules

These constraints were chosen explicitly with the user.

### Compute / runtime constraints

- use all `16` MIGs immediately
- one seed per config
- target roughly an overnight run, about `8` hours
- no serial screen/locked tournament for this pass

### Reporting constraints

- do **not** pollute the existing `supervision_recovery.family_rows`
- do **not** encode parity rows as new package ladder items
- keep the historical package ladders intact
- add a separate parity/geometry panel into the same family-grids lineage

### Scientific constraints

- use historical `official_fno/full100` and `official_fno_sumlen/full100` as
  reference lines (verified: FNO reference used seeds `[0, 1]`, parity grid
  uses `seed=0` which is a subset — comparison is valid)
- no FNO reruns in the first overnight base
- the `recoverable` benchmark profile uses `min_tokens=max_tokens=96`, so
  documents are **128 tokens** long
- use `fixed_leaf_tokens=128` as the one-leaf target (96 tokens/leaf on a
  128-token doc = exactly 1 leaf); `fixed_leaf_tokens=64` gives 2 leaves, not 1
- the Lean `one_pass` theorem applies vacuously in the single-leaf regime: L1
  holds because the leaf IS the document, and L2 holds because there are no
  internal nodes — so the tree reduces to exactly the same computation as a flat
  neural operator regardless of law weights
- do not claim strict architectural identity in the report yet; claim empirical
  parity in the one-leaf regime if supported

### Provenance constraints

- compare explicit directories, not mixed report summaries
- report provenance should be directory-based and explicit
- stale stopped-job status files should not be mistaken for active jobs

## Prepared Overnight Experiment

Prepared runner:

- [run_markov_supervision_recovery_parity_grid.py](/home/mlinegar/ThinkingTrees/scripts/run_markov_supervision_recovery_parity_grid.py)

This runner is **implemented but not launched**.

It schedules tree-side geometry jobs plus exact-collapse candidates across
three leaf-granularity tiers on 128-token documents:

**Geometry matrix (18 jobs):**

All five recipes at leaf=16 (8 leaves):

1. `historical_replay`, recoverable, `fixed_leaf_tokens=16`
2. `optimization_fairness`, recoverable, `fixed_leaf_tokens=16`
3. `capacity_fairness`, recoverable, `fixed_leaf_tokens=16`
4. `matched_root`, recoverable, `fixed_leaf_tokens=16`

All five recipes at leaf=64 (2 leaves):

5. `historical_replay`, recoverable, `fixed_leaf_tokens=64`
6. `optimization_fairness`, recoverable, `fixed_leaf_tokens=64`
7. `capacity_fairness`, recoverable, `fixed_leaf_tokens=64`
8. `matched_root`, recoverable, `fixed_leaf_tokens=64`

Geometry sweep for matched_root:

9. `matched_root`, recoverable, `fixed_leaf_tokens=32` (4 leaves)
10. `matched_root`, recoverable, `fixed_leaf_tokens=128` (1 leaf = single-leaf)

Remaining recipes at leaf=128 (single-leaf regime):

11. `historical_replay`, recoverable, `fixed_leaf_tokens=128`
12. `optimization_fairness`, recoverable, `fixed_leaf_tokens=128`
13. `capacity_fairness`, recoverable, `fixed_leaf_tokens=128`
14. `fairfno_matched_root`, recoverable, `fixed_leaf_tokens=128`

Fair-FNO geometry sweep:

15. `fairfno_matched_root`, recoverable, `fixed_leaf_tokens=16`
16. `fairfno_matched_root`, recoverable, `fixed_leaf_tokens=32`
17. `fairfno_matched_root`, recoverable, `fixed_leaf_tokens=64`

Structural confirmation at single-leaf:

18. `matched_root`, structural, `fixed_leaf_tokens=128`

**Exact-collapse candidates (2 jobs):**

19. `exact_collapse_candidate`, recoverable, `fixed_leaf_tokens=128`
20. `exact_collapse_candidate`, structural, `fixed_leaf_tokens=128`

The runner writes:

- `parity_grid_manifest.json`
- `parity_grid_status.json`
- `parity_grid_summary.json`
- `results.jsonl`

### Recipe meanings

These recipe IDs are important:

- `historical_replay`
  - tries to replay the weak historical root-only tree recipe
- `optimization_fairness`
  - same macro-capacity, but switch to a root-oriented training/checkpointing
    regime
- `capacity_fairness`
  - keep the older objective shape, but match the larger state/hidden capacity
- `matched_root`
  - combine the fairness fixes into the candidate clean root-only tree recipe
- `fairfno_matched_root`
  - same matched-root regime, but use the fair-FNO-matched leaf defaults

### What is genuinely new versus rerun

If this overnight run happens:

- **No FNO reruns**
- tree-side only

Strict reruns:

- essentially only `historical_replay + recoverable + fixed_leaf_tokens=16`

New evidence:

- all five recipes at `fixed_leaf_tokens=128` (true single-leaf regime on
  128-token docs)
- all five recipes at `fixed_leaf_tokens=64` (2 leaves)
- geometry sweep at `fixed_leaf_tokens=32` (4 leaves)
- fair-FNO geometry sweep at `fixed_leaf_tokens=16/32/64`
- exact-collapse candidates at `fixed_leaf_tokens=128` (recoverable +
  structural)
- structural confirmation at single-leaf

## Prepared Report Integration

Report merger:

- [report_markov_cohort_compare.py](/home/mlinegar/ThinkingTrees/scripts/report_markov_cohort_compare.py)

The merger now accepts:

- `--cohort LABEL=PATH` for explicit cohort directories
- `--template-report-dir` for the existing family-grids lineage
- `--parity-grid-root` for the new geometry/parity block

The intended next merged report version is:

- `markov_family_grids_with_cohort_compare_v3`

What v3 should preserve:

- dense full-doc anchor
- recoverable / structural package ladders
- recoverable / structural ordered-family plots
- R10 / R20 cohort comparison sections
- parity explainer text

What v3 should add:

- `Recoverable Full100 Geometry / Parity`
- `Structural Full100 Geometry Confirmation`

Important reporting rule:

- parity rows must stay in a separate summary block such as
  `supervision_recovery_parity_grid`
- they must **not** be merged into the package ladder rows

## Why We Are Still Using Markov Right Now

Another LLM taking over should not misread this as “the project is only Markov”.

The actual situation is:

- Markov full-doc supervision-recovery is the current **controlled empirical
  proving ground**
- it gives us fast, structured comparisons for supervision budgets and tree/FNO
  tradeoffs
- we are simultaneously building a canonical experiment/reporting layer that is
  meant to generalize to:
  - tree/LLM training paths
  - runtime eval
  - CTreePO sim suites

The cross-family normalization direction already chosen is:

- direct labels, local supervision, and local-law/verifier controls are
  separate dimensions
- report comparisons must be budget-matched and semantically fair
- report inputs should come from canonical rows/artifacts, not bespoke
  family-specific summary hacks

So the current Markov study is a **base argument surface**, not the end-state.

## Main Open Questions

These are the live research and engineering questions:

1. Is the bad historical `tree_neural/full100` recoverable row mostly a recipe
   fairness artifact?
2. Does one-leaf geometry at `fixed_leaf_tokens=64` materially close the gap?
3. Do fair-FNO-matched leaf defaults matter in the root-only tree regime?
4. Is structural just fundamentally harder in this root-only setting, or does
   the recoverable fix partly transfer?
5. How should the later LLM/tree report panels align to the same budget and
   comparison logic without conflating local control signals with direct labels?

## Practical Next-Step Guidance

If another LLM takes over and the user wants to proceed with the overnight run,
the intended sequence is:

1. confirm the prepared-data root and MIG UUIDs are still correct
2. launch the parity grid, not the old root-only diagnosis runner
3. let it produce a clean `parity_grid_summary.json`
4. merge that root into the existing family-grids report lineage as v3
5. interpret the one-leaf regime as empirical parity evidence only if the data
   supports it

If the user wants reporting work without running anything:

1. use the historical grid and current family-grids v2 as the authoritative
   base
2. do not trust stale stopped-job progress JSON as evidence
3. preserve the package ladders
4. keep cohort comparison and parity/geometry as separate report layers

## Commands Prepared But Not Yet Run

Prepared overnight parity-grid launch shape:

```bash
./venv/bin/python scripts/run_markov_supervision_recovery_parity_grid.py \
  --output-root outputs/markov_supervision_recovery_parity_grid_$(date +%Y%m%d_%H%M%S) \
  --prepared-data-root <prepared-data-root> \
  --mig-uuids "$MIGS"
```

Prepared family-grids v3 merge shape:

```bash
./venv/bin/python scripts/report_markov_cohort_compare.py \
  --cohort r10=<r10_root> \
  --cohort r20=<r20_root> \
  --template-report-dir <family_grids_template_dir> \
  --parity-grid-root <parity_grid_root> \
  --output-dir <new_report_dir>
```

## Bottom Line

The most important context for takeover is:

- the historical package-grid evidence is real and should be preserved
- the old root-only diagnosis runner was the wrong operational shape
- the prepared overnight parity-grid runner is the right next experiment
  because it is:
  - directly relevant to the broad argument
  - uses all 16 MIGs
  - adds evidence to the main report lineage
  - does not corrupt the package ladders
- the repo is broader than Markov, but Markov is the current clean base case
  for the argument we are trying to make
