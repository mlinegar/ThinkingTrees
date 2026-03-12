# Markov Theory Alignment Diagnosis (2026-03-11)

This note records the Markov-specific alignment pass between the Lean formalization
and the current simulation suite.

## 1. What Lean now proves

The raw Markov document support is now formalized directly as regime paths in
[MarkovPathDGP.lean](/home/mlinegar/ThinkingTrees/lean3/FormalProofs/OPT/MarkovPathDGP.lean).

The main consequences are:

- `markov_path_local_laws_of_encoded_state`:
  exact path encoding into the endpoint+count sketch induces exact local laws.
- `markov_path_state_exact_on_tree`:
  any downstream utility on the exact Markov sketch state is preserved exactly by tree reduction.
- `markov_path_count_exact_on_tree`:
  changepoint count is preserved exactly on every tree.
- `markov_countOnly_mergeFold_counterexample`:
  the count-only undersupported control is not compositionally sufficient.

Interpretation:

- The exact Markov sketch should work.
- The count-only baseline should not, in general.
- A learned Markov summarizer is only expected to work to the extent that it
  actually learns the approximate local laws; Lean does not imply finite-sample
  monotone improvement with audit fraction or train size.

## 2. Simulation reporting bug that was masking the picture

The old Markov expectation grouping in
[expectations.py](/home/mlinegar/ThinkingTrees/src/ctreepo/sim/expectations.py)
mixed together:

- theorem-relevant local-law runs
- root-only / legacy-weight runs
- different local-law weights
- different DGP regimes

That made the previous Markov family report look misaligned even when the
exact theorem was fine.

The adapter now splits scenarios by:

- `objective_weighting_scheme`
- `objective_parameterization`
- `objective_local_law_weight`
- `theorem_relevant`
- `transition_log_std`
- `min_segments` / `max_segments`
- `min_seg_len` / `max_seg_len`

and only runs the learned-vs-undersupported anchor on theorem-relevant
local-law runs.

## 3. Cleaned Markov result

The rerun artifact is:

- [markov_only_expectations_v2.json](/home/mlinegar/ThinkingTrees/outputs/formal_reruns_20260310_062551/paper_reports/markov_only_expectations_v2.json)
- [markov_only_expectations_v2.md](/home/mlinegar/ThinkingTrees/outputs/formal_reruns_20260310_062551/paper_reports/markov_only_expectations_v2.md)
- [markov_local_law_optimization_triage/README.md](/home/mlinegar/ThinkingTrees/outputs/formal_reruns_20260310_062551/paper_reports/markov_local_law_optimization_triage/README.md)

Summary:

- `n_fail = 0`
- `n_pass = 801`
- `n_warn = 444`
- `n_not_applicable = 1223`

So the hard Markov misalignment was mostly a slice-mixing problem.

## 4. What still goes wrong

After the cleanup, the remaining Markov issue is specific:

- Theorem-relevant neural runs pass the root anchor:
  learned `root_mae` beats the undersupported baseline in every anchored scenario.
- Theorem-relevant neural runs still fail the merge anchor:
  learned `merge_mae` is worse than the count-only undersupported baseline in every anchored scenario.

This is not a contradiction of the Lean theorem. It means:

- the learned model is good enough to improve the root task,
- but it is not learning C3/merge preservation well enough.

## 5. Likely cause

The most suspicious runs are in
`identifiable_zero_learnability/markov_changepoint_ops_count/equivalence/baseline/...`
with:

- `val_docs = 0`
- small `audit_fraction` (for example `0.02`)
- moderate-to-large `local_law_weight`

Examples:

- [seed_0.json](/home/mlinegar/ThinkingTrees/outputs/formal_reruns_20260310_062551/identifiable_zero_learnability/markov_changepoint_ops_count/equivalence/baseline/train_16000/family_neural/rfroot_1/budget_0p02/c3_uniform/c3root_1/lqr_1/llw_0p25/rw_1/scw_0/seed_0.json)
- [seed_0.json](/home/mlinegar/ThinkingTrees/outputs/formal_reruns_20260310_062551/identifiable_zero_learnability/markov_changepoint_ops_count/equivalence/baseline/train_16000/family_neural/rfroot_1/budget_0p02/c3_uniform/c3root_1/lqr_1/llw_0p5/rw_1/scw_0/seed_0.json)

Those runs show:

- strong root performance
- very large `merge_mae`
- large schedule spread
- no validation-based selection signal in the summary

At the time these reruns were produced, the neural trainer in
[markov_changepoint_ops_count.py](/home/mlinegar/ThinkingTrees/src/ctreepo/sim/core/markov_changepoint_ops_count.py)
did fixed-epoch training only; it did not actually perform validation-based
checkpoint selection. In sparse-internal-label settings, that made C3 learning
fragile.

The code path has now been corrected:

- when `val_docs > 0`, the trainer evaluates the held-out optimization objective
  after each epoch, restores the best checkpoint, and records
  `training_selection_*` metadata;
- when `val_docs = 0`, the trainer now explicitly records
  `training_selection_mode = final_epoch_no_validation` instead of silently
  behaving as if a held-out selection step happened.

So the linked triage folder should now be read as a diagnosis of the
pre-fix reruns, not as the current training semantics.

## 6. Current best reading

The Markov family is now aligned with theory in the sense that:

- the exact ceiling behaves as Lean says it should;
- the undersupported count-only baseline behaves as Lean says it can;
- the learned neural lane succeeds on the root task but still has a real C3 problem.

So the next empirical target is not "why does the exact theorem fail?"
It is:

1. why C3/merge supervision is underperforming in the learned neural lane,
2. whether sparse internal-label budgets plus no checkpoint selection are the main cause,
3. whether the anchored paper figure should emphasize root performance and exact-vs-undersupported separation, with merge kept as a known failure mode / appendix diagnostic.
