# IPW CI Simulation Suite

This note describes the simulation families used to validate TreeIPW empirical-Bernstein confidence intervals.

## Why multiple simulation families

A single synthetic generator is not enough. We need several stress axes:

- objective structure: separable vs nonseparable
- sampling design: Bernoulli IPW vs without-replacement (WOR)
- propensity imbalance: balanced to adversarial
- document length variation: fixed vs variable-length chunk counts per doc
- oracle preference functional: additive vs non-additive mergeable-sketch outcomes

The suite now covers all of these axes.

## Population families

## 1) Synthetic numeric populations

Implemented in `src/tree/ipw_simulation.py`:

- `separable`: chunk outcomes from local signal only.
- `nonseparable`: chunk outcomes also depend on document context (mean + dispersion).
- `doc-nonseparable`: doc-level policy quantity drives chunk labels.

This family is good for broad random stress but less interpretable.

## 2) Toy chunking populations

Implemented in `src/tree/ipw_toy_problems.py`:

- granularity:
  - `word`: one word per chunk (interpretable sentence-level chunking)
  - `char`: one character per chunk (extreme fine-grained chunking)
- chunk-importance patterns:
  - `uniform`
  - `front-loaded`
  - `back-loaded`
  - `alternating`
  - `spike` (single critical chunk)
  - `boundary` (critical first/last chunks)
- imbalance profiles:
  - `balanced`
  - `moderate`
  - `severe`
  - `adversarial` (high-signal chunks get low propensity)
- length profiles:
  - `fixed`: constant chunks per document
  - `uniform`: random length in a range
  - `bimodal`: short-doc mode + long-doc mode
  - `long-tail`: many short docs with occasional very long docs
- oracle preference profiles:
  - `legacy-smooth`: legacy smooth doc-policy generator
  - `additive-mean`: additive control (mean local signal only)
  - `topk-spike`: non-additive top-k concentration
  - `quorum-gate`: non-additive threshold/quorum behavior
  - `hybrid-extreme`: non-additive interaction stress case

This family directly addresses “simple sentence, one word/character at a time” and chunking worst cases.

Highly concentrated DGP examples are created by combining:
- `pattern=spike` or `pattern=boundary` (signal concentrated in tiny region),
- `imbalance=adversarial` (critical chunks undersampled),
- `length_profile=bimodal` or `length_profile=long-tail` (adaptive chunking pressure),
- non-additive oracle preferences (`topk-spike`, `quorum-gate`, `hybrid-extreme`).

## Curated mergeable-sketch examples

The CLI also exposes a curated example suite with explicit labels:

- `positive`: sanity/control settings where behavior should be stable.
- `negative`: stress settings where concentration + imbalance should make CIs wider and neff smaller.

Run with:

```bash
source venv/bin/activate
python3 scripts/run_ipw_ci_simulation.py \
  --population-model toy \
  --toy-mergeable-examples \
  --design compare \
  --trials 150 \
  --n-docs 64 \
  --chunks-per-doc 14 \
  --min-chunks-per-doc 4 \
  --max-chunks-per-doc 30 \
  --delta 0.10
```

## Sampling designs

Both families are evaluated under:

- `bernoulli`: two-stage Bernoulli (doc then chunk)
- `wor`: two-stage simple random sampling without replacement

The simulation compares CI coverage and width across designs.

## Coverage interpretation

Given confidence level `1 - delta`:

- target empirical coverage is near `1 - delta`.
- persistent under-coverage implies CI is too optimistic for that regime.
- very high coverage with very wide intervals implies conservative CI.

Track these jointly:

- `violation_coverage`, `preference_coverage`
- `violation_mean_width`, `preference_mean_width`
- effective sample diagnostics (`mean_effective_sample_size`)
- imbalance diagnostics (`min/p10/median joint propensity`, `max_joint_weight`)

## Recommended runs

Quick comparison:

```bash
source venv/bin/activate
python3 scripts/run_ipw_ci_simulation.py \
  --population-model toy \
  --toy-matrix \
  --design compare \
  --scenario all \
  --trials 120 \
  --n-docs 56 \
  --chunks-per-doc 12 \
  --delta 0.10 \
  --enforce-target \
  --coverage-tolerance 0.05
```

Hard stress (adversarial + spike + char):

```bash
source venv/bin/activate
python3 scripts/run_ipw_ci_simulation.py \
  --population-model toy \
  --scenario doc-nonseparable \
  --design compare \
  --granularity char \
  --pattern spike \
  --imbalance adversarial \
  --length-profile long-tail \
  --min-chunks-per-doc 4 \
  --max-chunks-per-doc 40 \
  --trials 250 \
  --n-docs 64 \
  --chunks-per-doc 24 \
  --delta 0.10 \
  --enforce-target \
  --coverage-tolerance 0.03
```

## Limitations and next extension

Current toy generators include variable-length docs, but chunk boundaries are still generated from stylized token streams. Next extension should drive chunk lengths from the real adaptive chunker (`src/preprocessing/chunker.py`) and replay the same coverage matrix to test boundary-placement sensitivity directly.
