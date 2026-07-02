# Classical Sketch Compact Learned Overlay

| family | query | best official | official RMSE | learned f | f RMSE | learned g | g RMSE | learned joint (best) | joint variant | joint RMSE |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| distinct | cardinality | theta_datasketches (L=16) | 0 | -- (L=--) | — | learned_g_exact_distinct_union_state_space (L=16) | 0 | learned_joint_exact_distinct_union_state_space (L=16) | fg | 0 |
| frequency | focus_frequency | -- (L=--) | — | -- (L=--) | — | learned_g_count_min_state_space (L=16) | 0 | learned_joint_count_min_state_space (L=16) | fg | 0 |
| frequency | top5_point_frequency | count_min_datasketches (L=16) | 0 | -- (L=--) | — | -- (L=--) | — | -- (L=--) | -- | — |
| quantile | rank_at_q0.5 | quantiles_floats_datasketches (L=16) | 0.002583 | -- (L=--) | — | -- (L=--) | — | learned_joint_quantiles_reference_q0.5 (L=512) | fg | 0.02422 |
| quantile | rank_at_q0.95 | tdigest_double_datasketches (L=16) | 0.002462 | -- (L=--) | — | -- (L=--) | — | learned_joint_kll_reference_q0.95 (L=32) | fg | 0.007558 |
| sampling | accumulator_summary_sum | tuple_accumulator_datasketches (L=16) | 0 | -- (L=--) | — | -- (L=--) | — | learned_joint_tuple_summary_sum_reference (L=512) | fg | 0.007204 |
| sampling | total_weight | varopt_strings_datasketches (L=16) | 0 | -- (L=--) | — | learned_g_exact_total_weight_state_space (L=16) | 0 | learned_joint_exact_total_weight_state_space (L=16) | fg | 0 |
| set | a_not_b | theta_datasketches (L=16) | 0 | -- (L=--) | — | -- (L=--) | — | learned_joint_exact_set_a_not_b (L=64) | fg | 0.09717 |
| set | intersection | theta_datasketches (L=16) | 0 | -- (L=--) | — | -- (L=--) | — | learned_joint_exact_set_intersection (L=256) | fg | 0.2417 |
| set | union | theta_datasketches (L=16) | 0 | -- (L=--) | — | -- (L=--) | — | learned_joint_exact_set_union (L=512) | fg | 0.03052 |
