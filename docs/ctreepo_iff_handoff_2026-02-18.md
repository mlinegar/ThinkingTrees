# C-TreePO IFF Formalization Handoff (2026-02-18)

This note summarizes the Lean work completed on the “both sides of the IFF” thread for the C-TreePO / ThinkingTrees paper, so another LLM can pick up quickly.

## Scope completed

We pushed IFF coverage across three layers:

1. **Measure-theoretic sufficiency (Doob–Dynkin style)**
2. **Local-law inertness characterization (C2/L3)**
3. **Deterministic global↔local bridges (C1/C2/C3 vs A1/A2, with and without surjectivity)**

All key statements are now exposed in `Assumptions.lean` and reflected in the paper crosswalk.

## Main files changed

- `lean3/FormalProofs/OPT/ScoreTransport.lean`
- `lean3/FormalProofs/OPT/ExpectationTheory.lean`
- `lean3/FormalProofs/OPT/GlobalAssumptions.lean`
- `lean3/FormalProofs/Assumptions.lean`
- `lean3/FormalProofs/OPT/MainTheorems.lean`
- `paper/ctreepo/appendix/E_lean_crosswalk.tex`

## Core theorem inventory

### 1) Sufficient-statistic / Doob–Dynkin IFFs

- `oracle_factorization_iff_sigma_subset`  
  (`f*(X)` factors through `Z` iff `σ(f*(X)) ⊆ σ(Z)`)  
  `lean3/FormalProofs/OPT/ScoreTransport.lean:195`

- `oracle_factorization_ae_iff_aestronglyMeasurable`  
  (a.e. factorization iff `AEStronglyMeasurable` over `σ(Z)`)  
  `lean3/FormalProofs/OPT/ScoreTransport.lean:241`

### 2) C2/L3 inertness IFFs

- `L3_iff_RoundInert`  
  (`L3` iff one-step round inertness)  
  `lean3/FormalProofs/OPT/ExpectationTheory.lean:440`

- `L3_implies_ZR_step_inert`  
  (`L3` ⇒ zero one-step normalization term on `ZR`)  
  `lean3/FormalProofs/OPT/ExpectationTheory.lean:457`

### 3) Deterministic global↔local bridges (PMF.pure wrapper)

All deterministic statements use summarizer `fun x => PMF.pure (g_det x)`.

- C1/L1:
  - `L1_deterministic_iff_leafwise`  
    `lean3/FormalProofs/OPT/GlobalAssumptions.lean:673`
  - `A1_iff_L1_for_all_trees`  
    `lean3/FormalProofs/OPT/GlobalAssumptions.lean:724`

- C2/L3 (non-surjective + surjective):
  - `A1_on_summary_range`  
    `lean3/FormalProofs/OPT/GlobalAssumptions.lean:698`
  - `L3_iff_A1_on_summary_range`  
    `lean3/FormalProofs/OPT/GlobalAssumptions.lean:702`
  - `A1_iff_L3_of_surjective`  
    `lean3/FormalProofs/OPT/GlobalAssumptions.lean:747`

- C3/L2:
  - `L2_deterministic_two_leaf_iff_A2_pointwise`  
    `lean3/FormalProofs/OPT/GlobalAssumptions.lean:756`
  - `A2_iff_L2_on_two_leaf_trees`  
    `lean3/FormalProofs/OPT/GlobalAssumptions.lean:793`
  - `A2_iff_L2_on_all_trees_of_A1_A3`  
    `lean3/FormalProofs/OPT/GlobalAssumptions.lean:808`
  - `L2_on_all_trees_iff_two_leaf_trees_of_A1_A3`  
    `lean3/FormalProofs/OPT/GlobalAssumptions.lean:819`

### 4) Surjectivity collapse extensions

- `L1_on_all_trees_iff_L3_of_surjective`  
  `lean3/FormalProofs/OPT/GlobalAssumptions.lean:832`

- `A1_A2_iff_L3_and_L2_on_all_trees_of_A3_surjective`  
  `lean3/FormalProofs/OPT/GlobalAssumptions.lean:847`

- `A1_A2_iff_L3_and_L2_on_two_leaf_trees_of_A3_surjective`  
  `lean3/FormalProofs/OPT/GlobalAssumptions.lean:867`

## Citation-ready master aliases

Added in `MainTheorems.lean`:

- `surjective_global_local_master`  
  alias of `A1_A2_iff_L3_and_L2_on_all_trees_of_A3_surjective`  
  `lean3/FormalProofs/OPT/MainTheorems.lean:238`

- `surjective_global_local_master_two_leaf`  
  two-leaf variant  
  `lean3/FormalProofs/OPT/MainTheorems.lean:242`

## Assumptions-map aliases (paper-facing handles)

All relevant aliases are exposed in `lean3/FormalProofs/Assumptions.lean`, including:

- Doob–Dynkin: `doob_dynkin_oracle_iff`, `doob_dynkin_oracle_ae_iff`
- C2 inertness: `l3_iff_round_inert`, `l3_implies_zr_step_inert`
- Deterministic bridges:
  - `det_l1_iff_leafwise`, `a1_iff_l1_all_trees`
  - `det_l3_iff_inrange`, `A1_OnSummaryRange`, `a1_on_summary_range_iff_l3`, `a1_iff_l3_surjective`
  - `det_l2_two_leaf_iff_a2_pointwise`, `a2_iff_l2_two_leaf_trees`
  - `a2_iff_l2_all_trees_given_a1a3`, `l2_all_trees_iff_two_leaf_trees_given_a1a3`
  - `l1_all_trees_iff_l3_surjective`
  - `a1a2_iff_l3_l2_all_trees_given_a3_surjective`
  - `a1a2_iff_l3_l2_two_leaf_given_a3_surjective`

## Current strongest equivalence picture

- Always (deterministic):
  - `A1 ↔ L1(all trees)`
  - `L3 ↔ A1_on_summary_range`
  - `A2 ↔ L2(two-leaf trees)`
  - `L2(all trees) ⇒ A2`

- With `A1 + A3`:
  - `A2 ↔ L2(all trees)`
  - `L2(all trees) ↔ L2(two-leaf trees)`

- With surjective `g`:
  - `A1 ↔ L3`
  - `L1(all trees) ↔ L3`

- With `A3 + surjective g`:
  - `(A1 ∧ A2) ↔ (L3 ∧ L2(all trees))`
  - `(A1 ∧ A2) ↔ (L3 ∧ L2(two-leaf trees))`

## Paper crosswalk status

Crosswalk rows were added/updated in:

- `paper/ctreepo/appendix/E_lean_crosswalk.tex:18`
- `paper/ctreepo/appendix/E_lean_crosswalk.tex:26`
- `paper/ctreepo/appendix/E_lean_crosswalk.tex:30`
- `paper/ctreepo/appendix/E_lean_crosswalk.tex:38`
- `paper/ctreepo/appendix/E_lean_crosswalk.tex:42`
- `paper/ctreepo/appendix/E_lean_crosswalk.tex:46`
- `paper/ctreepo/appendix/E_lean_crosswalk.tex:50`
- `paper/ctreepo/appendix/E_lean_crosswalk.tex:54`
- `paper/ctreepo/appendix/E_lean_crosswalk.tex:58`
- `paper/ctreepo/appendix/E_lean_crosswalk.tex:62`
- `paper/ctreepo/appendix/E_lean_crosswalk.tex:66`

## Build status

Successful targeted builds were run after changes:

```bash
cd lean3
lake build FormalProofs.OPT.GlobalAssumptions FormalProofs.Assumptions
lake build FormalProofs.OPT.MainTheorems
```

## Suggested next work for another LLM

1. Add a paper-facing proposition text that cites `surjective_global_local_master` directly.
2. Decide whether crosswalk should point to the master alias in `MainTheorems.lean` (instead of underlying theorems in `GlobalAssumptions.lean`).
3. If needed, attempt non-deterministic analogues (currently these global↔local collapse results are deterministic via `PMF.pure`).

