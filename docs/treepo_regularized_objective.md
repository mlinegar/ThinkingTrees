# TreePO Regularized Objective

This note makes the optimization problem explicit on both the Lean side and the
simulation side.

## Formal Objective

The Lean definition lives in [RegularizedObjective.lean](/home/mlinegar/ThinkingTrees/lean3/FormalProofs/OPT/RegularizedObjective.lean).

For summarizer `g`, document `x`, reduction depth `R`, tree `T`, oracle `f*`,
summary-cost proxy `cost`, and weights `w`, the population oracle-risk term is

```text
oracleRiskObjective(g)
  = w.distortion * Δ_R_ZR(g, x, R, T, f*)
  + w.summary * E[cost(Z)].
```

The certified regularized objective adds approximate local-law budgets:

```text
certifiedRegularizedObjective(g)
  = oracleRiskObjective(g)
  + w.leaf  * epsLeaf
  + w.merge * epsMerge
  + w.idemp * ((R - 1) * epsIdemp).
```

The key Lean bounds are:

- `oracleRiskObjective_le_of_approx_bundle`
- `certifiedRegularizedObjective_le_of_approx_bundle`

These combine the existing approximate-local-law distortion theorem with the
new regularized objective.

## Default Simulation Weights

The simulation-facing default in Lean is

```text
distortion = 3/4
summary    = 1/8
leaf       = 1/24
merge      = 1/24
idemp      = 1/24
```

Interpretation:

- `0.75` weight on global oracle distortion
- `0.25` total weight on regularization
- inside the regularizer, `0.5` goes to summary-budget pressure
- the remaining `0.5` goes equally to leaf / merge / idempotence law penalties

This is only the starting point. The simulation scripts now support post-hoc
lambda sweeps over the same saved rows.

## Trade-Off Map

The objective above isolates the optimization-side bias terms. The main tradeoffs
are split across existing formal modules:

- compression / approximation bias:
  controlled by `Δ_R_ZR` and the approximate-local-law bounds in
  [ApproximateLocalLaws.lean](/home/mlinegar/ThinkingTrees/lean3/FormalProofs/OPT/ApproximateLocalLaws.lean)
- downstream objective transport:
  controlled by the Lipschitz utility-gap theorems in
  [OracleUtility.lean](/home/mlinegar/ThinkingTrees/lean3/FormalProofs/OPT/OracleUtility.lean)
  and
  [PreferenceBounds.lean](/home/mlinegar/ThinkingTrees/lean3/FormalProofs/OPT/PreferenceBounds.lean)
- audit variance under subsampling:
  controlled by the IPW theorems in
  [TreeIPW.lean](/home/mlinegar/ThinkingTrees/lean3/FormalProofs/DSL/TreeIPW.lean)
- judge bias / variance decomposition:
  controlled by
  [JudgeCalibration.lean](/home/mlinegar/ThinkingTrees/lean3/FormalProofs/DSL/JudgeCalibration.lean)
  and
  `/home/mlinegar/FormalProbability/FormalProbability/DSL/VarianceDecomposition.lean`

So the new regularized objective is the optimization surface, while the
existing probability / information-theory modules still carry the evaluation
and bias-variance guarantees around it.

## Simulation Mapping

The learned-sketch simulation uses the following empirical proxies:

- global error: `learned_relative_rmse`
- summary budget penalty: normalized `learned_memory_bits`
- law penalty:
  - theorem-aligned path: normalized weighted combination of `eps_leaf`,
    `eps_merge`, and `eps_idemp`
  - proxy fallback: `latent_merge_state_mse`

The regularized objective emitted by the simulation is

```text
total
  = (1 - lambda) * global_error
  + lambda * (
      summary_share * summary_budget_penalty
      + (1 - summary_share) * law_penalty
    ).
```

It is useful to name

```text
law_strength = 1 - summary_share.
```

With that convention:

- `law_strength = 0` is the legacy summary-only endpoint
- `law_strength = 1` is the law-only endpoint inside the regularizer
- the current default is `law_strength = 0.5`

Use `simulation_mode=law_backed_learned_sketch` when you want the law penalty to
track decoded approximate local-law terms rather than the latent proxy.

## Scripts

Single run with explicit regularizer settings:

```bash
python3 scripts/run_learned_sketch_simulation.py \
  --simulation-mode law_backed_learned_sketch \
  --regularizer-weight 0.25 \
  --law-strength 0.50 \
  --law-leaf-share 1 \
  --law-merge-share 1 \
  --law-idemp-share 1
```

Multi-seed sweep with the same objective:

```bash
python3 scripts/run_learned_sketch_sampling_sweep.py \
  --simulation-mode law_backed_learned_sketch \
  --regularizer-weight 0.25 \
  --law-strength 0.50
```

Post-hoc lambda and law-strength frontier from a saved artifact:

```bash
python3 scripts/report_learned_sketch_regularized_objective.py \
  --input outputs/learned_sketch_sampling_sweep_summary.json \
  --regularizer-weights 0.0,0.25,0.5,0.75,1.0 \
  --law-strengths 0.0,0.25,0.5,0.75,1.0
```
