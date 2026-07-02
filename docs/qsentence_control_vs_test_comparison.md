# Q-sentence g: comparing test runs against a pre-generated control

Per-dimension Pearson (prediction vs teacher) is the honest composition metric for
the manifesto q-sentence ladder. Pooled Pearson inflates via between-dimension mean
separation, so always compare **per dimension** (rile + domain_1..7).

## Don't re-run the control every time

The gold-children (rate=0) control eval is expensive at deep leaf sizes
(leaf=2 composes g over ~27K nodes / 13 levels). Reuse a pre-generated control
instead of recomputing it for every A/B.

### Canonical saved controls (gold-children, learned g, iter_02)

`outputs/manifesto_qsentence_diffusiongemma_FULL218_leafgrid/dspy/leafq{002,004,008,016}/prediction_records/iter_02_post_eval.jsonl`

Per-dim Pearson of these controls (the collapse worsens with tree depth):

| dim       | leaf2  | leaf4  | leaf8  |
|-----------|--------|--------|--------|
| rile      | -0.283 | -0.165 | -0.184 |
| domain_4  | +0.368 | +0.054 | +0.126 |
| domain_5  | -0.064 | +0.379 | -0.122 |
| (others)  | mixed ~0 | mixed ~0 | mixed ~0 |

(leaf16 is shallower and composes better — rile ~+0.49 — but leaf<=8 is where
exposure-bias collapse bites.)

## The comparator

`scripts/compare_qsentence_per_dim_pearson.py` takes a control + one or more test
runs (run dirs or direct `iter_*_post_eval.jsonl` paths) and prints per-dim Pearson
with both unpaired deltas and **paired** deltas (computed on the docs common to both,
so different eval samples don't confound the comparison). It also emits a verdict
(mean Δ Pearson, dims improved) and optional `--json-out`.

```bash
# Test-only run vs a saved control — no control re-run needed:
./venv/bin/python scripts/compare_qsentence_per_dim_pearson.py \
  --control outputs/manifesto_qsentence_diffusiongemma_FULL218_leafgrid \
  --test    outputs/<your_test_run> \
  --leaf 8 --iter 2 --labels control,mytest \
  --json-out outputs/<your_test_run>/per_dim_pearson_comparison.json
```

Tests: `tests/tasks/test_compare_qsentence_per_dim_pearson.py`.

## A/B driver

`scripts/run_scheduled_sampling_ab_leaf8.sh` runs control(rate0) + sched(rate1) arms
and calls the comparator at the end. To skip the control arm and reuse a saved
control, run only the sched arm of the ladder, then invoke the comparator with
`--control <saved control run>`.
