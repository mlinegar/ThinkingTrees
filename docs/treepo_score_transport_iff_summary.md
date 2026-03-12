# TreePO Local Laws, Doob-Dynkin, and Score Transport: IFF Summary

This note summarizes the current Lean status for linking:

1. Tree-local laws (`L1`, `L2`, `L3`)
2. Doob-Dynkin style factorization/`sigma`-algebra statements
3. Score transport/factorization statements

It is intended for handoff to another LLM for deeper theorem synthesis.

## Core Definitions (where to look)

- Local laws:
  - `L1`: `lean3/FormalProofs/OPT/LocalLaws.lean:99`
  - `L2`: `lean3/FormalProofs/OPT/LocalLaws.lean:117`
  - `L3`: `lean3/FormalProofs/OPT/LocalLaws.lean:139`
- Round inertness:
  - `RoundInert`: `lean3/FormalProofs/OPT/ExpectationTheory.lean:409`
- Score-transport side:
  - `ConditionalFactorization'`: `lean3/FormalProofs/OPT/ScoreTransport.lean:62`
  - `OracleSigmaSubset'`: `lean3/FormalProofs/OPT/ScoreTransport.lean:77`
  - `OracleFactorization'`: `lean3/FormalProofs/OPT/ScoreTransport.lean:163`
  - `OracleFactorizationAE'`: `lean3/FormalProofs/OPT/ScoreTransport.lean:206`

## Proven IFF Statements

### Local-law IFFs (support-level characterizations)

- `L1_iff_dist_zero_on_support`:
  - `lean3/FormalProofs/OPT/ExpectationTheory.lean:200`
  - Requires explicit summability hypothesis.
- `L2_iff_dist_zero_on_support`:
  - `lean3/FormalProofs/OPT/ExpectationTheory.lean:249`
  - Requires explicit summability hypothesis.
- `L3_iff_dist_zero_on_support`:
  - `lean3/FormalProofs/OPT/ExpectationTheory.lean:294`
  - Requires explicit summability hypothesis.

Typeclass versions (automatic boundedness via `[BoundedPseudoMetricSpace Y]`):

- `L1_iff_dist_zero_on_support_typeclass`:
  - `lean3/FormalProofs/OPT/ExpectationTheory.lean:340`
- `L2_iff_dist_zero_on_support_typeclass`:
  - `lean3/FormalProofs/OPT/ExpectationTheory.lean:355`
- `L3_iff_dist_zero_on_support_typeclass`:
  - `lean3/FormalProofs/OPT/ExpectationTheory.lean:371`

### Local law to inertness IFF

- `L3_iff_RoundInert`:
  - `lean3/FormalProofs/OPT/ExpectationTheory.lean:440`

### Doob-Dynkin IFFs

- Pointwise factorization IFF:
  - `oracle_factorization_iff_sigma_subset`
  - `lean3/FormalProofs/OPT/ScoreTransport.lean:195`
  - Assumptions: `[Nonempty Y'] [StandardBorelSpace Y']`
- A.e. factorization IFF:
  - `oracle_factorization_ae_iff_aestronglyMeasurable`
  - `lean3/FormalProofs/OPT/ScoreTransport.lean:241`
  - Assumptions: `[Nonempty Y'] [StandardBorelSpace Y'] [SecondCountableTopology Y']`

## New Bridge/Contrapositive Theorems (Score side)

### Transport failure decomposition

- `not_score_transport_implies_cf_or_oracle_factorization_failure`
  - `lean3/FormalProofs/OPT/ScoreTransport.lean:397`
  - If score transport equality fails for action `a`, then either:
    - `ConditionalFactorization'` fails, or
    - `OracleFactorization'` fails.

### Tree-law failure from transport failure (with bridge)

- `not_score_transport_implies_one_local_law_failed_of_bridge`
  - `lean3/FormalProofs/OPT/ScoreTransport.lean:421`
  - Requires:
    - `hBridge : L1 -> L2 -> L3 -> OracleSigmaSubset'`
    - `hCF : ConditionalFactorization'`
  - Conclusion:
    - `¬L1 ∨ ¬L2 ∨ ¬L3`

### Positive bridge to score factorization and its contrapositive

- `local_laws_imply_score_factorization_of_bridge`
  - `lean3/FormalProofs/OPT/ScoreTransport.lean:476`
  - Requires:
    - local laws `L1`, `L2`, `L3`
    - `hBridge : L1 -> L2 -> L3 -> OracleSigmaSubset'`
    - nesting `sigma(Z) <= sigma(X)` (`hσ_ZX`)
    - `hCF`
  - Concludes:
    - `SummaryScore' ... a =ᵐ[μ] (hCF.choose (fstar (X ω)) a)`
- `not_score_factorization_implies_one_local_law_failed_of_bridge`
  - `lean3/FormalProofs/OPT/ScoreTransport.lean:494`
  - Same assumptions (`hBridge`, `hσ_ZX`, `hCF`)
  - If score factorization fails for action `a`, then `¬L1 ∨ ¬L2 ∨ ¬L3`.

## What We Have vs. What We Do Not Have

### We have (conditional near-IFF)

Under `hBridge`, `hσ_ZX`, and `hCF`:

- `(L1 ∧ L2 ∧ L3) -> ScoreFactorization(a)` via
  `local_laws_imply_score_factorization_of_bridge`.
- `¬ScoreFactorization(a) -> (¬L1 ∨ ¬L2 ∨ ¬L3)` via
  `not_score_factorization_implies_one_local_law_failed_of_bridge`.

This is the exact contrapositive pair needed for practical diagnostics.

### We do not have in full generality

- `(¬L1 ∨ ¬L2 ∨ ¬L3) -> ¬ScoreFactorization(a)` is not generally valid without
  stronger identifiability assumptions on score families/actions.
- `¬ScoreTransport` alone does not imply local-law failure unless `hCF` (or an
  equivalent score-side structural assumption) is included.

## Known independence result to keep in mind

- `thm10_1_L3_not_derivable`:
  - `lean3/FormalProofs/OPT/CounterexampleExistence.lean:321`
  - Shows `L3` is independent from the other local requirements in the formalization.

## Main Open Technical Task

Prove a concrete `hBridge` for the intended tree random-variable instantiation:

- choose concrete `X` (raw input RV) and `Z` (tree summary RV),
- show `L1`, `L2`, `L3` imply `OracleSigmaSubset' fstar X Z`,
- then instantiate the new contrapositive theorems directly.

## Suggested Prompt for Another LLM

Use this exact objective:

1. Construct concrete `X` and `Z` for the TreePO reduction process in Lean.
2. Prove `hBridge : L1 g T fstar -> L2 g T fstar -> L3 g fstar -> OracleSigmaSubset' fstar X Z`.
3. Instantiate:
   - `local_laws_imply_score_factorization_of_bridge`
   - `not_score_factorization_implies_one_local_law_failed_of_bridge`
4. State the resulting conditional diagnostic theorem in one corollary.
