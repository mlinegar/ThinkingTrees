# RILE GEPA Control Sanity Artifact

This artifact preserves the historical RILE GEPA control used to diagnose
why GEPA worked in the old RILE setup but not in the later Benoit runs.

## Run

- Run dir: `outputs/manifesto_nested_20260417_045842/dspy_gemma31b_v3_200`
- Model: `google/gemma-4-31B-it`
- API base: `http://localhost:8005/v1`
- Dataset: `/home/mlinegar/ThinkingTrees/outputs/manifesto_nested_20260417_045842/text_pairs_v1_200`
- Optimizer: `gepa` / auto `light`
- GEPA metric calls approx: `572`
- GEPA val cap: `48`
- Pareto val examples observed: `48`
- Reflection minibatch: `3`
- Max tokens: `1024`
- Log elapsed seconds: `2214.0`

## Validation MAE

- Baseline MAE: `34.22037704918033`
- Final MAE: `12.306267759562841`
- Improvement: `21.914109289617485`
- Acceptance threshold: `15.0`
- Acceptance pass: `True`

## Optimized Prompt Excerpt

```text
markdown
# Task: Predict the Manifesto RILE (Right-Left) Score

## Objective
Calculate the RILE score for a given political manifesto excerpt. RILE is a quantitative measure of ideological intensity, defined as the sum of right-coded quasi-sentence shares minus the sum of left-coded quasi-sentence shares.

**Formula: RILE = (% of text dedicated to Right themes) - (% of text dedicated to Left themes)**

## Coding Framework
- **Right-coded themes**: Free market (deregulation, tax cuts, privatization), military expansion, traditional morality, law & order (strong penalties, death penalty, gun rights), nationalism/national way of life.
- **Left-coded themes**: Welfare state (social safety nets, wealth redistribution, student grants/education accessibility), anti-imperialism, labor rights (worker protection, wage growth), peace (disarmament, diplomacy), environmental protection.

## Scoring Guidelines & Calibration (Strict Adherence Required)

### 1. The "Aggressive Dilution" Principle (Critical)
The most frequent error is over-scoring. Political manifestos are predominantly composed of **Neutral/Administrative content**. 
- **Neutral Content includes**: Technical logistics, organizational structures, descriptions of department functions, legislative procedures, budgetary mechanisms, specific parliamentary coalition tactics, and general administrative goals.
- **Calibration**: If a text contains specific dates, budget percentages, or mentions of specific government bodies/committees, these are almost always **Neutral**.
- **Impact**: Final RILE scores are typically very close to 0. 
- **Hard Ceiling**: Avoid scores beyond +/- 20 unless the text is a pure ideological manifesto. In professional policy excerpts, the score usually falls between -10 and +10.

### 2. Share vs. Presence (Quantitative Estimation)
Do not score based on the presence of a keyword. You must estimate the **percentage of the total word count** actually dedicated to the *ideological argument*.
- **Pres...
```

## Interpretation

This control used one excerpt scorer, direct RILE labels, rich directional
GEPA feedback, and a small GEPA validation surface. It should be treated as
the control contract before comparing against Benoit scorer-only or
full-pipeline GEPA.
