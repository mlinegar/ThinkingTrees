# Unified Learning Procedure

Date: 2026-05-18

This note writes the full learning protocol for the C-TreePO / ThinkingTrees
setting where chunking, compression, readout, oracle approximation, and query
selection are all learned or adapted. The central point is:

```text
Honest chunking may depend on learned f/g.
It must not depend on held-out outcomes or residuals from the same evaluation
top-level units used for the final claim.
```

The procedure below is the operational version of:

- document-level honesty from `lean3/FormalProofs/DSL/Honesty.lean`;
- finite document/chunk/manifest contracts from
  `lean3/FormalProofs/DSL/DocumentStructure.lean`;
- the final unified certificate surface from
  `lean3/FormalProofs/DSL/UnifiedLearningCertificate.lean`;
- influence-weighted audit overlap from
  `lean3/FormalProofs/Assumptions.lean`;
- the unified `g`/`f` contract in
  `docs/minimal_unified_gf_contract_2026-05-03.md`;
- the sampled local-law contract in
  `docs/local_law_sampling_contract.md`.

The theorem crosswalk is `docs/unified_learning_theorem_map.md`.

## Units And Objects

The default statistical unit is a top-level case/document:

```text
X_i ~ P_X
```

`X_i` is the unit that receives train/eval roles and the unit over which
document-level generalization or uncertainty claims are made. In the manifesto
setting, `X_i` is a manifesto or manifesto section, depending on the declared
estimand. In a Markov or synthetic panel, `X_i` is one generated sequence or
panel. If a task wants sentence-level, claim-level, or QA-instance-level claims,
then those objects should be renamed as the top-level `X_i`, and all source
document siblings that can leak information should be grouped into the same
fold.

Do not use `X_i` for sub-document text. Sub-document objects are conditional
objects generated from `X_i`:

```text
s_{ij}        raw span or leaf candidate inside X_i
B_i           boundary partition of X_i
v in V_i(B_i) tree node under partition B_i
a             audit/query row, e.g. (i, v, law_type) or (i, v_left, v_right)
```

The IID/exchangeability assumption is:

```text
(X_i, Y_i*) are IID or exchangeable across i.
```

Leaves, nodes, local-law rows, and queried spans inside one `X_i` are not IID
samples from the population. They are dependent derived units. Within-document
or within-tree uncertainty therefore comes from the logged audit/query design
conditional on `X_i`, not from pretending nodes are independent documents.

The target object is a truth function or full-information oracle:

```text
Y_i* = Y*(X_i)
f*(X_i) = target score / utility / preference-relevant readout
```

The learned pipeline at round `t` contains:

```text
C_t      chunk / boundary policy
g_t      learned compression or summary-state operator
f_t      learned readout on states
O_t      learned oracle / judge / nuisance predictor, if distinct from f_t
Q_t      query policy for human, dataset, or trusted-oracle labels
P_t      optional cheap proxy for span scoring and query routing
```

In the strict unified `g/f` lane:

```text
z_leaf = g_t(embed(span), null)
z_node = g_t(z_left, z_right)
y_hat  = f_t(z_root)
```

The chunker may use these learned objects:

```text
B_i = C_t(X_i; g_t, f_t, O_t, P_t, diagnostics_t)
```

That is allowed. The honesty constraint is about which data trained the frozen
objects used to produce `B_i`, not about whether `C_t` is model-aware.

## Chunker Objective

The chunker is an instrumental policy. Its target is not "find natural
paragraphs" or "find all interesting spans" in isolation. Its target is to
choose an admissible partition that lets the frozen tree computation preserve
the downstream target with low cost and auditable uncertainty.

For frozen artifacts `A = (g, f, O, P, Q)`, the chunker chooses:

```text
B_i = C_eta(X_i; A) in admissible_partitions(X_i)
```

A clean population objective is:

```text
J_C(eta; A) =
  E_X [
      downstream_loss(X, B, g, f, Y*)
    + lambda_law  * local_law_residual_mass(X, B, g, f, O)
    + lambda_rad  * certificate_radius(X, B, Q)
    + lambda_cost * compute_or_query_cost(B)
    + lambda_reg  * boundary_regularization(B)
  ]
```

where `B = C_eta(X; A)`. Operationally this objective is estimated on
`r_C=train` top-level units using truth labels, out-of-fold oracle/readout
predictions, and logged local-law samples.

The chunker should allocate resolution where additional boundaries have high
marginal value:

```text
high f/g sensitivity
high predicted local-law residual
high predicted oracle/readout uncertainty
high response-signature disagreement
high query value under the current label budget
```

and keep coarser leaves where the state is stable. In the exact theorem-backed
case, every admissible partition gives the same target state and the chunker is
mostly a cost policy. In the learned approximate case, boundary placement is
part of the error/cost/certificate tradeoff.

The chunker must not make the certificate look good by hiding hard regions.
The query/audit policy must retain positive probability for every consequential
row, and reports must expose the resulting influence-to-propensity ratios.

## What Is Split

The honesty split is over top-level `X_i` cases/documents, not post-hoc chunks.

All derived objects inherit the top-level unit role:

```text
top-level X_i -> chunks -> tree nodes -> summaries -> local-law rows -> labels
```

For document-level scientific claims, splitting chunks from the same document
into train and evaluation is usually leaky because the topology, summaries, and
oracle residuals are coupled through the original document.

Use three document-level roles:

```text
r_C(X_i) in {train, eval}      chunker / boundary adaptation
r_G(X_i) in {train, eval}      g/summarizer/operator training
r_O(X_i) in {train, eval}      oracle/readout/judge training
```

`r_G` is the same role called `summarizer` or `r_S` in the existing
three-layer honesty docs and Lean comments. The name `r_G` is used here because
the unified procedure centers the learned `g` operator.

The final report set is:

```text
E = E_C intersect E_G intersect E_O
```

or an outer held-out fold that plays the same role. Local node/span sampling is
not the honesty split. It is a design-based audit/query sample with logged
propensities.

## Cross-Fit Version

For `K` folds, train artifacts for fold `k` on all `X_i` outside `k`, then
apply the frozen artifacts to fold `k`:

```text
A_-k = train_artifacts({X_i : fold(i) != k})
A_-k = (C_-k, g_-k, f_-k, O_-k, Q_-k, P_-k)

B_i  = C_-k(X_i; g_-k, f_-k, O_-k, P_-k)
T_i  = build_tree(X_i, B_i, g_-k)
y_i  = f_-k(root(T_i))
```

Evaluation labels or residuals from fold `k` do not update `C_-k`, `g_-k`,
`f_-k`, `O_-k`, `Q_-k`, or `P_-k`.

This is the direct analogue of causal-forest honesty:

```text
structure sample      -> trains C/g/f/O/Q
estimation sample     -> estimates loss, gap, calibration, and uncertainty
```

## Per-Round Procedure

Each round operates on frozen snapshots. A round should be reproducible from
its artifact IDs and logged propensities.

### 1. Freeze the Current State

At the start of round `t`, freeze:

```text
A_t = (C_t, g_t, f_t, O_t_online, O_t_eval, Q_t, P_t)
```

`O_t_online` is the adaptive/update view. `O_t_eval` is frozen or out-of-fold
and is used for evaluation, nuisance prediction, calibration diagnostics, and
DR estimators.

### 2. Build Candidate Units

For each top-level `X_i`, use the allowed fold-specific artifacts to construct:

```text
boundaries
leaf spans
tree topology
leaf states
merge states
root state
candidate local-law rows
candidate query rows
```

The candidate local-law rows include:

```text
C1 leaf preservation rows
C2 idempotence rows
C3 merge consistency rows
```

Use the paper numbering in reports. The Lean file maps these to `L1`, `L3`,
and `L2`, respectively.

### 3. Query Truth With Logged Propensities

Apply the query policy `Q_t` to candidate top-level, node, or pair rows.

Every queried row must log:

```text
a                row id, e.g. (i, v, law_type)
Z_a              observed indicator
pi_a             joint propensity
label_source     human | dataset | trusted_oracle | proxy
truth_source     human | dataset, when the row is theorem-facing truth
approx_source    trusted_oracle | O_t | P_t | f_t | none
round_id
fold_id
top_level_unit_id
source_doc_id, if different from top_level_unit_id
node_id / span_id / pair_id
law_type
depth
role tuple       r_C, r_G, r_O
artifact IDs     C, g, f, O, Q, P
```

Human and dataset labels are the clean theorem-facing truth sources. A trusted
oracle can be used operationally, but reports should mark whether it is being
treated as approximate supervision or promoted to a task-specific truth source
under a separate calibration/validation contract.

For local-law training, experiment runners should pass observed masks,
propensities, and node weights into the canonical corrected objective in
`treepo/src/treepo/training/local_law.py`, rather than implementing separate
IPW corrections.

### 4. Enforce Overlap

The query policy must include an exploration floor:

```text
Q_t = (1 - eps) * Q_t_adaptive + eps * Q_base
eps > 0
```

Equivalently:

```text
pi(row) >= eps_row > 0
```

For influence-weighted local-law certificates, consequential rows must not have
unbounded influence-to-propensity ratios:

```text
D_lambda = sum_a lambda(a)^2 / pi(a)
W_lambda = max_a lambda(a) / pi(a)
```

must be finite and reported. This is the formal replacement for a vague
"no hidden needle" assumption. Needles may exist; what is disallowed is a
consequential row with effectively zero chance of being audited.

### 5. Train or Update the Oracle / Readout

Update `O_{t+1}_online` and/or `f_{t+1}` only from oracle/readout-train
top-level units:

```text
r_O(X_i) = train
```

Use truth labels when available. If labels are sampled, use HT/IPW or DR
corrections with the logged propensities. If `O` is not distinct from `f`, this
step is just readout training.

Construct `O_{t+1}_eval` as one of:

```text
frozen validation snapshot
strict out-of-fold predictions
outer-fold model
```

Do not use oracle-eval residuals to update the same evaluation view.

### 6. Train or Update g

Update `g_{t+1}` only from `g`/summarizer-train top-level units:

```text
r_G(X_i) = train
```

The training objective can combine:

```text
root/readout loss
preference loss
corrected local-law loss
contextual response-signature loss
summary budget penalty
stability or idempotence penalty
```

A typical objective is:

```text
L =
  alpha_root    * L_root_or_pref
+ alpha_law     * L_corrected_local_laws
+ alpha_context * L_contextual_signature
+ alpha_budget  * L_budget
+ alpha_stable  * L_stability
```

The theorem-facing local-law component is:

```text
L_corrected_local_laws =
  IPW/CORR(C1 leaf preservation)
+ IPW/CORR(C2 idempotence)
+ IPW/CORR(C3 merge consistency)
```

Rows should carry node weights or influence weights when the certificate is
not uniform.

### 7. Train or Update the Chunker

Update `C_{t+1}` only from chunker-train top-level units:

```text
r_C(X_i) = train
```

The chunker may use learned-model diagnostics, including:

```text
f/g uncertainty
predicted law residuals
predicted oracle residuals
response-signature disagreement
boundary-cost estimates
query-value estimates
```

But any diagnostic used to update `C` should be out-of-fold with respect to the
oracle/readout role. In particular, an eval unit's own truth label or residual
must not decide its reported boundaries.

For eval top-level unit `X_i` in fold `k`, the allowed form is:

```text
B_i = C_-k(X_i; g_-k, f_-k, O_eval_-k, P_-k)
```

The eval unit's raw text/features may be used to route it through the frozen
policy. Its held-out truth label, oracle residual, or observed local-law
failure may not be used to update the policy before that unit is reported.

The forbidden form is:

```text
B_i = C(X_i; residual_i, label_i, law_failure_i)
```

where `residual_i`, `label_i`, or `law_failure_i` were observed on the same
evaluation unit and then used before reporting that unit's metric.

### 8. Train or Update the Query Policy

Update `Q_{t+1}` from train-role feedback only. `Q` can learn where labels are
valuable, but it must retain overlap:

```text
pi >= eps
```

The query policy is part of the frozen artifact tuple for evaluation. Reported
IPW/DR estimates use the propensities from the policy version that actually
selected the queried rows.

### 9. Evaluate Frozen Artifacts

For each held-out top-level unit, use its fold-specific frozen artifacts:

```text
B_i      = C_-k(X_i; g_-k, f_-k, O_eval_-k, P_-k)
T_i      = tree(X_i, B_i, g_-k)
y_hat_i  = f_-k(root(T_i))
loss_i   = ell(y_hat_i, Y_i*)
```

No state update is allowed from these losses before reporting them.

The round target is:

```text
J_t = E[ell(A_t(X), Y*(X))]
```

Estimate it on honest eval `X_i` units using:

```text
HT/IPW estimator
cross-fitted DR estimator
calibrated judge estimate with held-out calibration envelope
```

as appropriate for the available labels.

### 10. Report the Certificate Decomposition

Every report should separate:

```text
point estimate
sampling / IPW / DR radius
calibration gap
local-law residual mass
influence-weighted design effect
clipping or floor bias, if any
split-seed or fold variance
```

The paper-facing envelope is:

```text
|target gap|
<= |honest estimate|
 + statistical radius
 + calibration radius
 + influence-weighted local-law residual bound
 + clipping/floor terms, if used
```

In Lean, this is the theorem
`DSL.unified_learning_final_paper_certificate`; the corresponding
high-probability theorem is
`DSL.unified_learning_final_paper_certificate_high_prob`. The canonical Lean
certificate object is `DSL.UnifiedLearningErrorCertificate` (also exported as
`DSL.CurrentPaperErrorCertificate`). The component-radius provenance object is
`DSL.UnifiedLearningComponentEvidence`. When the local-law radius is supplied by
the influence-weighted audit certificate directly, use
`DSL.unified_learning_abs_gap_le_totalBound_from_influence`.

Training losses, weighted objectives, and train-split local-law curves are
diagnostics. They are not substitutes for honest eval metrics.

## Minimal Pseudocode

```python
for round_id in rounds:
    frozen = freeze(C, g, f, O_online, O_eval, Q, P)

    for fold_id in folds:
        train_units = units[fold != fold_id]
        eval_units = units[fold == fold_id]

        A_minus_k = train_or_load_artifacts(
            train_units=train_units,
            roles=("chunk", "g", "oracle"),
            previous=frozen,
        )

        # Build eval trees with learned, cross-fitted chunking.
        eval_runs = []
        for unit in eval_units:
            boundaries = A_minus_k.C(
                unit.text,
                g=A_minus_k.g,
                f=A_minus_k.f,
                oracle=A_minus_k.O_eval,
                proxy=A_minus_k.P,
            )
            tree = build_tree(unit.text, boundaries, g=A_minus_k.g)
            pred = A_minus_k.f(tree.root_state)
            eval_runs.append((unit, tree, pred, A_minus_k.ids))

        # Query/evaluate with logged propensities and no update from eval loss.
        eval_rows = query_or_score_eval_rows(
            eval_runs,
            policy=A_minus_k.Q,
            log_propensities=True,
        )
        fold_estimate = estimate_honest_metric(eval_rows, method="ipw_or_dr")

    report_round(aggregate_folds())

    # Only after reporting/freeze-safe evaluation do we advance online state.
    C, g, f, O_online, Q, P = update_online_state_from_train_roles()
    O_eval = make_frozen_or_oof_eval_view(O_online)
```

## Required Artifact Fields

A run manifest should include:

```text
run_id
round_id
fold_id
split_seed
top_level_unit_id
source_doc_id, if different from top_level_unit_id
top-level role tuple: r_C, r_G, r_O
C artifact id
g artifact id
f artifact id
O_online artifact id
O_eval artifact id
Q artifact id
P artifact id, if any
chunk boundaries and boundary provenance
tree topology hash
row_id
node/span IDs and support spans
law row type and depth
truth label source
approx label source
observed indicator Z
logged propensity pi
effective propensity after floor
influence weight lambda
loss target and loss value
```

Without these fields, it is difficult to distinguish a real learning gain from
selection on adaptive residuals.

## Cold Start

The first round should be conservative:

1. Use fixed or weakly adaptive chunking.
2. Query enough trusted top-level and local-law labels to train an initial
   `f/g` and optional proxy.
3. Keep span-level adaptive feedback off until a proxy has been trained on
   trusted labels.
4. Turn on honest model-aware chunking only after cross-fitted `f/g` or proxy
   diagnostics are available.

This matches `docs/pipeline_ordering.md`: trusted labels first, cheap
approximators second, adaptive policies third.

## Common Failure Modes

- Updating chunk boundaries from an eval unit's own residual and then reporting
  that unit as held out.
- Treating sampled local-law rows as if they were a train/eval split.
- Omitting propensities for adaptive queries.
- Allowing `pi = 0` for rare but root-relevant node classes.
- Reporting proxy-only scores as final truth claims.
- Comparing adaptive and fixed chunking without matching label/query budgets.
- Letting `O_online` and `O_eval` be the same mutable object during reporting.

## Short Form

The unified learning rule is:

```text
Learn C, g, f, O, and Q on train-role top-level units.
For eval top-level units, freeze cross-fitted C/g/f/O/Q, then let C use those learned
objects to choose boundaries.
Estimate final claims only from held-out truth labels or properly weighted
queries.
Maintain positive query probability for every consequential audit row.
```
