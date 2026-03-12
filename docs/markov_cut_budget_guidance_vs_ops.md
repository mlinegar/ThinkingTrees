# Why the Cut-Budget “Guidance Grid” Is Not an OPS (C1–C3) Test

This note clarifies a common source of confusion when looking at:

- `outputs/markov_changepoint_cut_budget_guidance_grid.png`

That figure comes from the **cut-budgeted Markov changepoint** simulation (`src/tree/markov_changepoint_cut_budget_simulation.py`).
It is a *chunking / boundary-selection* experiment, not an *OPS honesty / oracle-preserving summarization* experiment.

## What “1 oracle label per leaf/node” means in OPS

In the Lean formalization (and in the C-TreePO paper), “one oracle observation per realized leaf/node” refers to **node-level oracle preservation checks**:

- **C1 / L1 (leaf sufficiency):** the summary of a realized leaf preserves the oracle.
- **C3 / L2 (merge consistency):** the summary produced at an internal merge preserves the oracle of the merged span.
- **C2 / L3 (on-range idempotence):** re-summarizing an on-range summary preserves the oracle.

When these hold (and the merge tree is realized over the fixed partition), Lean proves **zero distortion at the root** (e.g. `FormalProofs.OPT.PreservationTheorems.one_pass`), and in the multi-round setting requires C2/L3 as well.

## What the cut-budget guidance grid is doing instead

The cut-budget simulation is solving a *DP segmentation problem* under a fixed cut budget.
Oracle “guidance” in that simulation means: spend a query budget to reveal **ground-truth boundary labels** for some candidate cut positions (or groups of positions, under the `"tree"` interface), and then override the learned boundary probabilities at those positions.

So, even if you set a budget like “~1 query per leaf”, you are **not** querying the OPS oracle on a *node span* and comparing it to a *node summary* (C1/C3).
You are asking for **boundary-at-position labels**, which is a different interface and a different estimand.

The third panel in that grid (“Lean upper bound Σ|δ| on gap”) is tied to the cut-budget bound in:

- `lean3/FormalProofs/OPT/CutBudgetGuidance.lean`

but it does *not* correspond to OPS C1–C3 checks.

## What to use when you want “OPS-style” simulations

If the intuition you want to test is:

> “With a sufficient mergeable sketch + node-level oracle supervision (C1/C3), we should get perfect root predictions; and with more docs/labels we should converge to the optimum; and adaptive node selection needs IPW/DSL to avoid bias.”

then you want a simulation where the oracle is queried on **realized leaves/internal nodes** and the objects being learned are **mergeable sketches** (not cut positions).

That is exactly what the OPS-count suite implements:

- `src/tree/markov_changepoint_ops_count_simulation.py`
- `scripts/run_markov_changepoint_ops_count_simulation.py`
- `scripts/plot_markov_changepoint_ops_count_grid.py`

It uses a scalar oracle `f⋆(x) = # changepoints` and separates:

- approximation bias (insufficient sketch state / “chunking loss”),
- estimation error (finite docs / finite oracle node labels),
- selection bias from adaptive node sampling (corrected by IPW + DSL / augmented-IPW).

## Repro commands

Single run (writes JSON + CSV summaries):

```bash
cd ThinkingTrees
venv/bin/python scripts/run_markov_changepoint_ops_count_simulation.py \
  --train-docs 1000 --test-docs 1000 \
  --audit-policy fraction --audit-fraction 0.2 \
  --seed 0 --device cpu \
  --json-summary outputs/markov_changepoint_ops_count/train_1000_budget_0.2_seed_0.json \
  --csv-summary outputs/markov_changepoint_ops_count/train_1000_budget_0.2_seed_0.csv
```

Sweep over `{train_docs} × {oracle labels/internal node} × {seed}` (example):

```bash
cd ThinkingTrees
for td in 50 100 200 500 1000 2000; do
  for frac in 0.05 0.1 0.2 0.5 1.0; do
    for seed in 0 1 2 3 4 5 6 7; do
      venv/bin/python scripts/run_markov_changepoint_ops_count_simulation.py \
        --train-docs "$td" --test-docs 1000 \
        --audit-policy fraction --audit-fraction "$frac" \
        --seed "$seed" --device cpu \
        --json-summary "outputs/markov_changepoint_ops_count/train_${td}_budget_${frac}_seed_${seed}.json" \
        --csv-summary "outputs/markov_changepoint_ops_count/train_${td}_budget_${frac}_seed_${seed}.csv"
    done
  done
done
```

Plot a `train_docs × internal_labels_per_leaf` grid from the saved JSON summaries:

```bash
cd ThinkingTrees
venv/bin/python scripts/plot_markov_changepoint_ops_count_grid.py \
  --input-glob "outputs/markov_changepoint_ops_count/train_*_seed_*.json" \
  --output-figure outputs/markov_changepoint_ops_count_grid.png
```

Lean traceability (worked example instantiation for `ExactSketch` + an `L3` counterexample):

```bash
cd ThinkingTrees/lean3
lake build FormalProofs.OPT.MarkovCountSketchExample
```
