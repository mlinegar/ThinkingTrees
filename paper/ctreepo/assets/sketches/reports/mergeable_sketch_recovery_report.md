# Mergeable Sketch Recovery Report
This report uses two completed outputs:
- five-leaf structured grid: `outputs/exact_state_g_then_f_structured_hllfixed_fast_leafsize_20260428_065454`
- fast `fg/gf/gfg` diagnostic: `outputs/mergeable_fg_gfg_diagnostic_fixed_gpu_20260428_170940`

## Figure Index
- [mergeable_exact_recovery_by_leaf.pdf](../figures/mergeable_exact_recovery_by_leaf.pdf) / [mergeable_exact_recovery_by_leaf.png](../figures/mergeable_exact_recovery_by_leaf.png)
- [mergeable_structured_learning_curves.pdf](../figures/mergeable_structured_learning_curves.pdf) / [mergeable_structured_learning_curves.png](../figures/mergeable_structured_learning_curves.png)
- [mergeable_mlp_readout_learning_curves.pdf](../figures/mergeable_mlp_readout_learning_curves.pdf) / [mergeable_mlp_readout_learning_curves.png](../figures/mergeable_mlp_readout_learning_curves.png)
- [quantile_official_tree_leafsize.pdf](../figures/quantile_official_tree_leafsize.pdf) / [quantile_official_tree_leafsize.png](../figures/quantile_official_tree_leafsize.png)

## What The Structured-Zero Result Means
The structured exact-state rows are a handoff/existence sanity check, not a learned neural-decoder claim. In this mode the state is supplied in an exposed numeric sketch space and the readout contains the analytic `f*` path. HLL uses the differentiable HLL estimator directly; additive/frequency/count-min states use the exact coordinate/min readout plus at most a zero-initialized residual. Therefore `fg` can be exactly zero: the green line is not learning `f` from scratch.

## Exact Exposed-State Recovery
| target | schedule | observed leaf sizes | max relative RMSE |
|---|---:|---:|---:|
| HLL registers | C-tree: learned g, fixed f* | 16,32,64,128,256 | 0 |
| HLL registers | C-tree sanity: f-stage -> g (analytic f*) | 64,128 | 0 |
| HLL registers | C-tree sanity: g -> f-stage (analytic f*) | 16,32,64,128,256 | 0 |
| HLL registers | C-tree sanity: g -> f-stage -> g (analytic f*) | 64,128 | 0 |
| Count-Min state | C-tree: learned g, fixed f* | 16,32,64,128,256 | 0 |
| Count-Min state | C-tree sanity: f-stage -> g (analytic f*) | 64,128 | 0 |
| Count-Min state | C-tree sanity: g -> f-stage (analytic f*) | 16,32,64,128,256 | 0 |
| Count-Min state | C-tree sanity: g -> f-stage -> g (analytic f*) | 64,128 | 0 |
| Exact frequency state | C-tree: learned g, fixed f* | 16,32,64,128,256 | 0 |
| Exact frequency state | C-tree sanity: f-stage -> g (analytic f*) | 64,128 | 0 |
| Exact frequency state | C-tree sanity: g -> f-stage (analytic f*) | 16,32,64,128,256 | 0 |
| Exact frequency state | C-tree sanity: g -> f-stage -> g (analytic f*) | 64,128 | 0 |
| Total-weight state | C-tree: learned g, fixed f* | 16,32,64,128,256 | 0 |
| Total-weight state | C-tree sanity: f-stage -> g (analytic f*) | 64,128 | 0 |
| Total-weight state | C-tree sanity: g -> f-stage (analytic f*) | 16,32,64,128,256 | 0 |
| Total-weight state | C-tree sanity: g -> f-stage -> g (analytic f*) | 64,128 | 0 |

## Learned-Readout Stress Rows
These rows use the same exposed state interface but replace the analytic readout with an MLP stress path where applicable. These are the rows to inspect for actual learned-`f` optimization behavior.
| target | schedule | observed leaf sizes | max relative RMSE |
|---|---:|---:|---:|
| HLL registers | C-tree: learned f then g | 64,128 | 0.0722 |
| HLL registers | C-tree: learned g then f | 64,128 | 0.105 |
| HLL registers | C-tree: learned g, then f+g | 64,128 | 0.0517 |
| Count-Min state | C-tree: learned f then g | 64,128 | 0.0542 |
| Count-Min state | C-tree: learned g then f | 64,128 | 0.0662 |
| Count-Min state | C-tree: learned g, then f+g | 64,128 | 0.0597 |
| Exact frequency state | C-tree: learned f then g | 64,128 | 0.00915 |
| Exact frequency state | C-tree: learned g then f | 64,128 | 0.0979 |
| Exact frequency state | C-tree: learned g, then f+g | 64,128 | 0.0106 |
| Total-weight state | C-tree: learned f then g | 64,128 | 0.0118 |
| Total-weight state | C-tree: learned g then f | 64,128 | 0.0733 |
| Total-weight state | C-tree: learned g, then f+g | 64,128 | 0.015 |

## Quantile Tree Results
The current diagnostic has official quantile tree rows only: `16` rows. It has no learned C-tree quantile/projection rows, so quantile should not be included in exact-recovery claims yet.
| sketch | query | min relative RMSE | max relative RMSE |
|---|---:|---:|---:|
| kll_floats_datasketches | rank_at_q0.5 | 0.002772 | 0.002851 |
| kll_floats_datasketches | rank_at_q0.95 | 0.002592 | 0.002833 |
| quantiles_floats_datasketches | rank_at_q0.5 | 0.002583 | 0.002583 |
| quantiles_floats_datasketches | rank_at_q0.95 | 0.002487 | 0.002487 |
| req_floats_datasketches | rank_at_q0.5 | 0.003213 | 0.003326 |
| req_floats_datasketches | rank_at_q0.95 | 0.002487 | 0.002487 |
| tdigest_double_datasketches | rank_at_q0.5 | 0.006391 | 0.006926 |
| tdigest_double_datasketches | rank_at_q0.95 | 0.002462 | 0.002462 |

## Next Data Gap
A full five-leaf `fg/gfg` diagnostic has not been run. The current five-leaf grid covers `g` and `gf`; the `fg/gfg` confirmation is currently at `64,128`.
