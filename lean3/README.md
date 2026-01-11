# FormalProofs (Lean 4)

Lean formalizations for the ThinkingTrees project. The `FormalProofs/Probability` subtree includes a CLT development via characteristic functions and Lévy continuity.

## Build

```bash
lake build FormalProofs.Probability
```

For the full library:

```bash
lake build FormalProofs
```

## CLT Status

- Bounded and finite-variance i.i.d. CLT are formalized.
- User-facing theorems live in `FormalProofs/Probability/CLT.lean`:
  - `central_limit_theorem_iid_finite_variance`
  - `central_limit_theorem_iid_abs_pow3`
  - `central_limit_theorem_iid_bounded`
  - `central_limit_theorem_cdf_iid_bounded`
  - `central_limit_theorem_iid_of_charFunScale`
  - `CharFunCLTScale_of_integrable_sq`
  - `CharFunCLTScale_of_integrable_abs_pow3`
- Analytic infrastructure is in `FormalProofs/Probability/LevyContinuity.lean`.
