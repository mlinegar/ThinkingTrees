# CPU Supervision Tutorials

This is a small tutorial ladder for the supervision-first training surface.
Every example is CPU-only and intentionally simple.

Recommended order:

0. [00 Manual Gradient Descent](./00_manual_gradient_descent.md)
1. [00 Same Average, Local Variation](./00_same_average_local_variation.md)
2. [00 Sampled Local Labels With IPW](./00_sampled_local_ipw.md)
3. [00 Numeric Gradient Descent](./00_numeric_gradient_descent.md)
4. [01 Dense Scalar Regression](./01_dense_scalar_regression.md)
5. [02 Grouped Comparative Supervision](./02_grouped_comparative_supervision.md)
6. [03 Human Preference To Supervision](./03_human_preference_to_supervision.md)
7. [04 Markov-Style CPU Regression](./04_markov_style_cpu_regression.md)

Simulation follow-ups:

8. [05 Easy IPW Mean Simulation](./05_ipw_mean_simulation.md)
9. [06 IPW Regression Simulation](./06_ipw_regression_simulation.md)
10. [07 Markov IPW Simulation](./07_ipw_markov_simulation.md)
11. [08 IPW Variance Tradeoff](./08_ipw_variance_tradeoff.md)
12. [09 Support Failure](./09_support_failure.md)
13. [10 Effective Sample Size And Clipping](./10_effective_sample_size_clipping.md)
14. [11 Online Query Loop](./11_online_query_loop.md)
15. [12 Weighted SGD Equivalence](./12_weighted_sgd_equivalence.md)
16. [13 Scalar, Comparative, And Binary](./13_scalar_comparative_binary_bridge.md)
17. [14 Noise Versus Bias](./14_noise_vs_bias.md)
18. [15 Markov Support Diagnostic](./15_markov_support_diagnostic.md)
19. [16 Joint Tradeoff Matrix](./16_joint_tradeoff_matrix.md)
20. [17 Decision Guide](./17_decision_guide.md)

All commands below should be run from the repo root:

```bash
source venv/bin/activate
```

The scripts print compact JSON summaries so you can see what the canonical
supervision objects produced.

If you want the short operational summary after the examples, read:
- [17 Decision Guide](./17_decision_guide.md)
