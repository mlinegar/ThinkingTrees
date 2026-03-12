# RLHF, DSL, and Bandit Sampling in Tree Summarization

This note captures the current conceptual model for RLHF in the tree-of-summaries
setting, with a design-based supervised learning (DSL) framing, explicit sampling
probabilities, and bandit-style collection. It also outlines how to estimate
uncertainty (standard error) under clustering.

## 1) Two nested DSL problems

There are two linked objectives:

1) Oracle, doc-level target (outer DSL). This is the final goal: capture oracle
   preferences at the document level (or document-level tree score).

2) Node-level surrogate target (inner DSL). We approximate oracle preferences at
   the node level using a judge model. The judge is calibrated on an oracle-
   labeled subset and used to score many nodes cheaply.

The "tournament of tournaments" is an adaptive sampling layer that improves the
judge. It does not change the outer objective; it changes how we collect judge
labels efficiently.

## 2) Sampling hierarchy and feedback loop

```mermaid
flowchart TD
  P[Population of documents] -->|stage 1: sample doc with p_d| D[Document d]
  D -->|stage 2: sample node with p_{n|d}| N[Node n (context)]
  N -->|stage 3: sample action or pair with p_{a|n} or p_{a,b|n}| A[Candidate summary a or pair (a,b)]
  A --> O[Oracle preference y*]
  A --> J[Judge score r_hat]
  O -->|calibrate| J
  O --> L[Log: p_d, p_{n|d}, p_{a|n}, policy_version, features]
  J --> L
  L --> E[IPW evaluation + SE]
  L --> T[Training update (GRPO, DPO, etc.)]
  T -->|policy shift| N
  T -->|policy shift| A
```

Where sampling enters:
- stage 1: document selection
- stage 2: node selection within a doc
- stage 3: candidate selection per node (single or pairwise)

Where sampling exits:
- after labeling (oracle or judge), we log outcomes and propensities, enabling
  design-based inference.

## 3) DSL requirement and minimal change

DSL requires a known, positive inclusion probability for each target unit.
Minimal change: add a small exploration floor (mixture sampling) at adaptive
stages so all units have nonzero support.

Example:
- p_{n|d} = (1 - eps) * p_bandit + eps * p_uniform_or_stratified
- p_{a|n} = (1 - eps_a) * p_policy + eps_a * p_uniform_action

This guarantees bounded weights and enables IPW with finite variance.

## 4) Estimands and IPW

Let each labeled sample i have probability:

    p_i = p_d * p_{n|d} * p_{a|n}

For pairwise:

    p_i = p_d * p_{n|d} * p_{a,b|n}

Weight:

    w_i = 1 / p_i

Hajek (ratio) estimator for a mean reward:

    mu_hat = (sum w_i * y_i) / (sum w_i)

This is unbiased for the target population under correct propensities.

## 5) Bandit and GRPO mapping

Contextual bandit view:
- context: node (plus document context and features)
- action: candidate summary or pair
- reward: oracle or judge preference

GRPO in this setting:
- sample K candidates from policy at node n
- compute advantages relative to the group mean
- update with sum_k A_k * grad log pi(a_k | n)

GRPO is on-policy optimization for the sampled distribution. It does not fix
sampling bias over nodes or docs. Sampling design and IPW handle that.

## 6) Uncertainty and standard error (clustered)

Nodes within a document are correlated. Treat each document as a cluster.

Cluster-robust (sandwich) SE for the Hajek estimator:

    g_i = w_i * (y_i - mu_hat)
    G_d = sum_{i in doc d} g_i
    Var(mu_hat) = (1 / (sum w_i)^2) * (1/(D-1)) * sum_d (G_d^2)
    SE = sqrt(Var)

Alternative: document-level bootstrap (resample docs, recompute mu_hat).

Effective sample size for weighted data:

    n_eff = (sum w)^2 / sum w^2

If n_eff collapses, increase exploration or stratify sampling.

## 7) Judge calibration and bias control

The judge is a measurement device for node-level preferences. Keep a small,
oracle-labeled calibration set sampled with known inclusion probabilities.

Use it to:
- estimate judge bias and variance
- monitor drift across training
- bound surrogate error when reporting final results

## 8) Tournament of tournaments interpretation

The tournament is an adaptive pair-sampling policy to refine the judge:
- it chooses which node pairs and candidate pairs to label
- it is a bandit exploration policy over comparisons

As long as pair selection probabilities are logged and nonzero, IPW correction
is still valid for judge evaluation.

## 9) Practical logging (minimal additions)

For each labeled record, log:
- doc id, node id (depth, subtree id)
- action or pair ids
- outcome (oracle or judge)
- p_d, p_{n|d}, p_{a|n} or p_{a,b|n}
- policy version and sampling policy name

This enables:
- IPW evaluation at node or doc level
- cluster-robust SEs
- off-policy analysis across policy versions

## 10) Generalization story

Doc-level generalization is guaranteed by DSL (known inclusion probabilities +
IPW + clustered SE).

Node-level generalization relies on judge calibration. Use oracle-labeled
subsets to quantify judge error and include it in reporting.

Training optimizes the sampled distribution; evaluation (IPW) re-targets the
population distribution.

## 11) Oracle utility transport (formalized)

Recent Lean results connect oracle-valued utilities to tree sampling without
separability assumptions. See `FormalProofs/OPT/OracleUtility.lean` and
`FormalProofs/DSL/TreeIPW.lean`.

Key statements:
- `expected_utility_bound_pmf` and `expected_utility_bound_pmf_bounded`: expected
  oracle-utility gap is bounded by expected oracle distortion; `expected_utility_bound_ZR`
  and `expected_utility_bound_ZR_summable` give the ZR/multi-round forms.
- `expected_utility_noise_bound_pmf` and `expected_utility_noise_bound_pmf_bounded`:
  sensitivity to noisy truth labels is controlled by `L * dist(fhat x, fstar x)`
  (summability is explicit or automatic under boundedness).
- `ExpectedDocOracleUtility`, `ExpectedTreeOracleUtility`,
  `tree_oracle_utility_gap_bounded`, `tree_oracle_utility_gap_bounded_ipw`:
  tree/IPW recovery of oracle utility gaps via expected tree distortion.
- `utility_gap_unified_gap_pure` plugs oracle-utility gaps into
  `unified_preference_gap_bounded` under the pure-doc distribution.

Calibration note: judge/surrogate error bounds are in
`FormalProofs/DSL/JudgeCalibration.lean` (see `surrogate_bound`).
