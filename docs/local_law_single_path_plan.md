# Single Canonical Local-Law Path — Living Plan (2026-06-25)

Consolidated from two external LLM plans (v1 plumbing, v2 correctness) + the
in-session audit + the user's A2 correction. Goal: one canonical local-law
objective for **every** method/model (FNO neural operator, DSPy/LLM, Markov/sim,
report paths). First deliverable is correctness + unification, NOT new metrics.
Pause Benoit/FNO sweeps until the corrected objective is active.

## Lean contract (the math we must match)

- Outer objective (RootLocalObjective): `(1 − Λ)·rootLoss + Λ·localLawLoss`.
- Local-law loss (AIPW + depth discount):
  `Σ_v γ^depth(v) · [ proxy_loss_v + (R_v/π_v)(oracle_loss_v − proxy_loss_v) ]`,
  normalized by retained node weights (mean conventions).
- Laws (all "distance through f* = 0"):
  - **A1_global → leaf_preservation**: `D f*(g z, z)=0`.
  - **A2_global → merge_preservation** (paper C3): `D f*(A·B, g(gA·gB))=0`, i.e.
    the merge route reads like the **independent reading of the actual parent
    text A·B** — never like the merge-derived parent state.
  - **A3_global → readout factorization** (NOT a substitute for A2):
    `f*(g(gA·gB)) = M(f*gA, f*gB)`, M assoc+comm (Aczél).

## Canonical arithmetic (single source)

- `/home/mlinegar/treepo/src/treepo/training/local_law.py` is the ONLY
  implementation of tensor/scalar AIPW, sampled-IPW, node weights, depth
  discount. (`local_law_objective_from_losses`, `_target_mse`,
  `corrected_local_law_loss_tensor`, `sampled_/observed_uniform_node_ipw_mean_loss`.)
- In-repo `src/training/supervision/local_law_torch.py` and
  `src/core/local_law_adjustment.py` become thin delegating shims (parity-tested).

## Single row/tensor contract (every family emits these; nothing else)

Per node: `prediction`, `proxy_target|proxy_loss`, `oracle_target|oracle_loss`,
`observed`, `propensity`, `depth`, `node_weight`, `law_kind` + metadata
`global_axiom`, `state_kind`, `law_channel`. Family adapters ONLY build rows;
they must not implement their own AIPW / depth weighting / Λ mixing. Role weights
(`root/leaf/merge_weight`) become `node_weight`s; `Λ`/`root_share` stays the
outer convex tradeoff via the public `ObjectiveSpec`/root-local resolver. Collapse
the two `ObjectiveSpec` contracts to that one resolver.

## FNO correction (the load-bearing fix)

- `_a2_term(state)` previously compared `f(states[parent])` to `f(merge(l,r))`;
  since `states[parent]` IS `merge(l,r)` in the forward pass, that residual was
  **identically zero** — a no-op. Prior `a2state` Benoit gains are NOT A2
  evidence (they came from the associativity penalty); rerun after the fix.
- **A2 (merge_preservation) row:**
  - prediction / RHS = `f(merge(state_l, state_r))` (the merge-route readout).
  - proxy_target / LHS = **detached** `f(A·B)` from an independent parent-text
    route (default: pool the node's descendant-leaf RAW embeddings, read through
    the shared leaf encoder + score head as one "big leaf"; reuses cached
    embeddings, always within the fixed-width invariant, and is independent of
    `merge_fno`). Opt-in higher fidelity: re-embed the concatenated parent text
    (subject to no-truncation rules) — for the root at least.
  - oracle_target / LHS = the observed node/root label for A·B when available
    (root = expert mean). When unobserved (interior `None`): observed=0,
    proxy-only.
- **A3 (readout factorization)** = SEPARATE weight `a3_factorization_weight`:
  `f(merge_state) == M(f(l), f(r))` (Aczél phi-form). Reclassify the current
  `a2_mode="readout"` here; it is a projection, not the A2 law.
- **Associativity** (`g_assoc_weight`) stays a separate projection diagnostic.
  Never reported as A2 evidence.

## Sampling & weighting (docs/local_law_sampling_contract.md)

- One logical tree node per row; root counted exactly once. Full binary tree with
  `L` leaves → `2L − 1` rows for all-node supervision.
- Fixed-size uniform sampling logs `π = q/N`; Bernoulli logs its rate; no per-doc
  "at least one node" rule. Persistent masks for rate-grid supervision (R10 = ~10%
  ever labeled, not 10% redrawn per epoch).
- Certificate/audit rows with nonzero influence require positive logged
  propensities. Tensor training may keep unobserved proxy-only rows with π=0 only
  because no division occurs for them.

## Knobs (FNO objective, both f and g phases)

`L = (1 − Λ)·rootLoss + Λ·Σ_{non-root v} γ^depth(v)·w_v·ℓ_v`, ℓ_v = AIPW corrected.
- **Λ = `local_law_weight`** (canonical `ObjectiveSpec`; root_share = 1−Λ).
  Λ=0 → root-only (reference baseline); Λ=1 → pure distributed law. Default 0.5.
- **γ = `gamma_depth`**, Lean convention **depth = max_level − level** (root=depth 0).
  γ=0 → law collapses to root; γ=1 → all-node. Default 1.0.
- `a3_factorization_weight` (separate A3 projection), `g_assoc_weight` (separate
  diagnostic). `g_a2_weight` is now a DEPRECATED no-op (subsumed by Λ).
- Reference corners: root-only Λ=0; all-node Λ=1,γ=1; depth-discounted Λ,γ∈(0,1).
- `f-only` (read whole-doc text through f = root proxy) is a METRIC/target, logged
  per arm — not a training mode.
- OPEN: "delta" third knob (user mentioned γ/δ/Λ) — undefined; revisit.

## Status

DONE (this session, [src/ctreepo/fno_family.py], [tests/test_fno_a2_consistency.py],
[scripts/run_manifesto_qsentence_dspy_ladder.py]):
- `_independent_parent_text_readings` (descendant-leaf pooling → shared f); the real
  merge law `f*(A·B)=f*(g(A)·g(B))`; non-vacuity + unsupervised-interior tests.
- FNO f and g routed through canonical `treepo.training.local_law` (AIPW + γ^depth).
- **γ depth-convention fixed** (root = depth 0; prior `node.level` was inverted).
- **Λ wired** to the canonical `(1−Λ)rootLoss + Λ·lawLoss` split for both f and g.
- CLI: `--fno-local-law-weight`, `--fno-gamma-depth`, `--fno-a3-factorization-weight`;
  `--fno-g-a2-weight` deprecated no-op.
- 30 FNO tests green; Benoit smoke runs at (Λ=0.5, γ=0.5) and defaults.

NEXT (sequenced):
1. **Honest (γ, Λ) sweep** on Benoit econ (8 epochs, warm cache): does turning on the
   laws (Λ>0, γ>0) beat root-only (Λ=0)? Log f-only (root proxy) as the target.
   Correct handoff/memory attribution (prior "a2state win" was the assoc penalty).
2. Shim `local_law_torch.py`/`local_law_adjustment.py` → treepo; collapse
   `ObjectiveSpec` contracts. Parity tests.
3. Central row builder; extend to `embedding_fno._batch_loss` + DSPy/manifesto;
   centralize role classification. Row-population tests (2L−1, root once, q/N,
   persistent masks).

## Assumptions

- `/home/mlinegar/treepo` is the canonical editable `treepo` package.
- Dirty worktree changes are user state; do not revert.
- Correctness/unification first; metrics later.
