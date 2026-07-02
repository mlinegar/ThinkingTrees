# 17 Decision Guide

This is the short operational summary for the tutorial ladder.

The examples keep the true target fixed at `1.0` whenever possible, so the main
question is not "what is the target?" but "what went wrong between the target
and the observed supervision?"

## 1. First Check: Is Support Present?

If any unit or region that matters has propensity `0`, stop.

- Logged propensities are not enough if support is missing.
- No IPW estimator can recover a target from units that are never observed.
- See: [09 Support Failure](./09_support_failure.md)

Rule:
- `support failure` -> redesign the sampling/query policy
- do not try to "fix" this with clipping or a different optimizer

## 2. If Support Holds, Ask What You Care About

### If you care most about unbiasedness

Use Horvitz-Thompson / raw IPW.

- Best when support holds and you want the cleanest unbiased estimator
- Expect higher variance when propensities are very uneven
- See: [05 Easy IPW Mean Simulation](./05_ipw_mean_simulation.md)
- See: [08 IPW Variance Tradeoff](./08_ipw_variance_tradeoff.md)

### If you care more about lower variance / lower RMSE

Use self-normalized IPW or clipped self-normalized IPW.

- Self-normalized IPW trades some bias for lower variance
- Clipping pushes that tradeoff further
- Best when raw IPW weights are too concentrated
- See: [08 IPW Variance Tradeoff](./08_ipw_variance_tradeoff.md)
- See: [10 Effective Sample Size And Clipping](./10_effective_sample_size_clipping.md)

### If propensities are nearly uniform

Naive, HT, and SNIPW will often look similar.

- In that regime the weighting machinery is less important
- See: [08 IPW Variance Tradeoff](./08_ipw_variance_tradeoff.md)

## 3. Check Effective Sample Size

If support holds but ESS is low, expect instability.

Signals:
- a few weights dominate
- clipped ESS is much larger than raw ESS
- repeated-trial IPW estimates have large spread

What to do:
- prefer SNIPW or clipped SNIPW if your goal is lower RMSE
- or redesign the policy so propensities are less extreme

See:
- [10 Effective Sample Size And Clipping](./10_effective_sample_size_clipping.md)
- [15 Markov Support Diagnostic](./15_markov_support_diagnostic.md)

## 4. Separate Sampling Bias From Label Noise

These are different problems.

- `sampling bias`: which units got labeled
- `label noise`: how noisy or inaccurate the labels are once obtained

If labels are noisy but unbiased:
- more samples help
- IPW does not solve the noise problem, it only addresses selection bias

If labels are biased:
- better optimization does not fix the bias

See:
- [14 Noise Versus Bias](./14_noise_vs_bias.md)
- [16 Joint Tradeoff Matrix](./16_joint_tradeoff_matrix.md)

## 5. Online And Offline Are The Same Object

The online case is not a separate theory.

Online loop:
1. choose a unit
2. log its propensity
3. query the oracle
4. update an estimate or model

Offline reuse:
1. read the logged supervision
2. use the same propensities as sample weights

So online querying and offline logged-data training are the same supervision
surface with different timing.

See:
- [11 Online Query Loop](./11_online_query_loop.md)

## 6. Optimization Choice

### Scalar target for one response

Use scalar supervision.

- `ResponseJudgment`
- dense regression / weighted SGD / ridge
- See: [00 Manual Gradient Descent](./00_manual_gradient_descent.md)
- See: [12 Weighted SGD Equivalence](./12_weighted_sgd_equivalence.md)

### Multiple responses with comparable scores

Still start from scalar supervision if you have it.

- Derive grouped comparative judgments when a grouped optimizer needs them
- Derive binary projections only if the optimizer is binary
- See: [13 Scalar, Comparative, And Binary](./13_scalar_comparative_binary_bridge.md)

### Binary optimizer only

Project internally to binary.

- Do not treat pairwise data as the primary stored object
- Binary is an optimizer view, not the canonical supervision surface

## 7. Markov / Local-Label Settings

Use the same decision process.

- If local labels average to the document target, full-document and local
  supervision should agree
- If local labels are sampled with unequal propensities, logged IPW should
  improve calibration
- If block sampling gets too concentrated, ESS will collapse

See:
- [00 Same Average, Local Variation](./00_same_average_local_variation.md)
- [00 Sampled Local Labels With IPW](./00_sampled_local_ipw.md)
- [07 Markov IPW Simulation](./07_ipw_markov_simulation.md)
- [15 Markov Support Diagnostic](./15_markov_support_diagnostic.md)

## 8. Short Decision Tree

1. Does every relevant unit have nonzero propensity?
   If no: redesign the policy.

2. Are weights extremely concentrated / ESS very low?
   If yes: expect HT variance; consider SNIPW, clipping, or a better policy.

3. Is the main problem label noise rather than selection bias?
   If yes: collect better labels or more labels; IPW is not the main lever.

4. Is the optimizer scalar, grouped, or binary?
   Use scalar judgments first, then derive comparative or binary views only as needed.

5. Is the data arriving online?
   Log the same canonical supervision records and reuse them offline.

## 9. Suggested Reading Order

If someone wants the shortest path:

1. [00 Manual Gradient Descent](./00_manual_gradient_descent.md)
2. [00 Sampled Local Labels With IPW](./00_sampled_local_ipw.md)
3. [08 IPW Variance Tradeoff](./08_ipw_variance_tradeoff.md)
4. [09 Support Failure](./09_support_failure.md)
5. [11 Online Query Loop](./11_online_query_loop.md)
6. [13 Scalar, Comparative, And Binary](./13_scalar_comparative_binary_bridge.md)
