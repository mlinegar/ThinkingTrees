# Review: Lean-Aligned Single Canonical Local-Law Objective (2026-06-25)

Self-contained review doc for external LLMs/humans. Goal: critique the audit, the
corrected merge law, the canonical-objective design, and the proposed experiment
**before any training runs**. No prior conversation context required.

Companion files: `docs/local_law_single_path_plan.md` (terse living plan),
`docs/local_law_sampling_contract.md` (sampling/IPW contract), the Lean sources
under `lean3/FormalProofs/OPT/`.

---

## 0. TL;DR

We learn a tree merge `g` whose composition of local readings reconstructs a global
doc label, where the merge weighting is NOT estimated but falls out of local laws,
measured through a readout `f`. Audit found two problems: (1) the FNO
merge-consistency term was **mathematically vacuous** (identically zero), so a prior
"win" was misattributed; (2) there was **no single objective path** — each model
family reimplemented a lossy subset of the Lean objective. We corrected the law to
`f*(A·B) = f*(g(A)·g(B))` (merge route vs an INDEPENDENT reading of the parent text)
and routed the FNO f/g objectives through the one canonical
`treepo.training.local_law` AIPW + depth-discount + convex-split objective, on by
default. 30 FNO tests pass; Benoit econ smoke runs. Next step is a (γ, Λ) sweep
asking whether turning the laws on beats root-only.

---

## 1. Framework and the laws

- Tree of text: leaves = quasi-sentence chunks, internal nodes = merges, root = doc.
- `f` = readout (state → scalar score). `g` = encoder/merge (text or child states →
  state). We want `f(g(tree)) ~ f*(doc)`, `~` = equality THROUGH the readout.
- Two local laws (each measured through `f`):
  - **A1 / leaf preservation** (Lean `A1_global`): `D f*(g z, z) = 0` — a leaf's
    summary reads like the leaf.
  - **A2 / merge preservation** (Lean `A2_global`, paper C3):
    `D f*(A·B, g(g A · g B)) = 0` — the reading of the actual parent text `A·B`
    equals the reading of the merge of the child summaries.
  - (A3 / readout factorization, Lean `A3_global`: `f*(g(gA·gB)) = M(f*gA, f*gB)`,
    M assoc+comm. A SEPARATE law, not a substitute for A2.)
- Lean sources: `lean3/FormalProofs/OPT/GlobalAssumptions.lean` (A1/A2/A3 → local
  laws, `merge_assoc`, `prop3_mergeable_classical`),
  `DiscountedTreeMetaObjective.lean` (γ^depth: root weight 1, each level ×γ;
  γ=0 ⟹ root-only, γ=1 ⟹ all-node), `UnifiedLocalLawAdjustment.lean` +
  `RootLocalObjective.lean` (AIPW corrected node loss + `(1−Λ)root + Λ·law`).

**The canonical objective (the math we must match):**
```
J = (1 − Λ)·rootLoss + Λ·Σ_v γ^depth(v)·w_v·ℓ_v
ℓ_v = proxy_v + (R_v / π_v)·(oracle_v − proxy_v)        (AIPW corrected node loss)
```
At an observed node (R=1, π=1): ℓ = oracle (gold). At an unobserved node (R=0):
ℓ = proxy. depth(v): root = 0.

---

## 2. Audit findings (what was wrong)

### 2a. The FNO merge-consistency term was vacuous (a no-op)

In `src/ctreepo/embedding_fno.py` (`_forward_tree_states`), every internal node's
state is DEFINED as `model.merge(left_child, right_child)` — there is no independent
encoding of the parent's own text. The old `_a2_term` (state mode) computed
`f(states[parent]) − f(merge(l, r))`; since `states[parent]` IS `merge(l, r)`, this
residual is **identically zero**, regardless of f or g. The A/B driver
`scripts/run_fno_benoit_econ_law_ab.sh` ran the "a2state" arm with
`--fno-g-assoc-weight 0.5`, so the reported result (a2state doc Pearson 0.424 vs
control 0.353) was produced by the **associativity penalty**, not the A2 law. The
prior handoff's headline was misattributed.

### 2b. No single objective path

The full Lean chain (AIPW + γ^depth + convex Λ) is faithfully implemented in exactly
ONE place: the sim Markov trainer (`src/training/supervision/local_law_torch.py` +
`src/core/local_law_adjustment.py`, mirrored by canonical
`treepo.training.local_law`). Every model-family substrate reimplemented a lossy
subset:
- **Embedding-FNO** (`src/ctreepo/fno_family.py`) — the substrate running
  Benoit/manifesto/RILE — used bespoke role-weighted MSE. Propensity/observed were
  computed but only written as trace metadata; no γ^depth; no convex Λ.
- **DSPy/LLM** (`manifesto_qsentence_dspy_family.py`) — reward `1−|f(a)−f(b)|`, a
  third encoding; self-declares `"not a corrected or IPW local-law objective"`.
- Two non-interoperating `ObjectiveSpec` contracts; role classification open-coded
  in 4+ sites.

(For the Benoit/manifesto grids, IPW is largely moot — leaves+root observed,
interiors None, no random subsampling — so the live divergences there are weighting
and law-projection, not propensity.)

---

## 3. The corrected merge law (the key fix)

The law is `f*(A·B) = f*(g(A)·g(B))`:
- **RIGHT side** `f*(g(A)·g(B))` = `f(merge(state_A, state_B))` (the merge route).
- **LEFT side** `f*(A·B)` = an INDEPENDENT reading of the parent's OWN text — it must
  NOT come from `merge(l,r)`. We read the parent text through the SAME `f` encoder.

Why this matters: at the root, `f*(A·B)` is the read-the-whole-doc-through-f path —
the number that beats every learned merge (the prior handoff's open problem, "every
merge loses to f-only ~0.526"). Genuine A2 says **make the merge route agree with the
direct text reading**, so enforcing the real law is the principled lever for that gap.
It is also substrate-agnostic: every model has "read this text" (f) and "merge these
summaries" (g), so `f(text) == f(merge(summaries))` is one law for FNO, LLM, and sim.

**AIPW form on the A2 row** (per merge node v, children A,B):
- prediction = `f(merge(s_A, s_B))`
- proxy = detached `f*(A·B)` (independent parent-text read)  ← LEFT side
- oracle = gold(v) when v is observed (root = expert label)
- observed = 1 if v has a score else 0; propensity = 1 (no subsampling here)

So the AIPW corrected loss reduces to **gold supervision at the observed root** and to
**text-read consistency at unsupervised interiors** — one term, the whole g objective.

**Independent parent-text read — implementation choice (open to critique):** we pool
the node's descendant-leaf RAW embeddings and read them through the shared leaf
encoder + score head as one "big leaf" (`_independent_parent_text_readings`). This is
independent of `merge_fno`, always within the fixed-width embedding invariant, and
needs no extra embedding calls. The higher-fidelity alternative is to re-embed each
interior node's concatenated text (true `f*(A·B)`, but costs embeds and can exceed the
2048-token context). We default to pooling; re-embedding is a possible opt-in for the
root. **Reviewers: is pooling an acceptable proxy for f*(A·B), or should the root at
least use re-embedded concatenated text?**

**A2 vs A3 separated:** A2 = merge route vs true parent reading;
A3 = readout factorization `f(merge_state) == M(f(l), f(r))` (`a3_factorization_weight`,
M = Aczél phi-form, assoc+comm by construction); associativity = separate diagnostic
(`g_assoc_weight`, the proven `merge_assoc`). None of these three is reported as
another's evidence.

---

## 4. Canonical objective + knobs (as implemented in the FNO family)

`src/ctreepo/fno_family.py`, both f and g phases, via
`treepo.training.local_law.local_law_objective_from_losses` (no bespoke AIPW/depth
math in the family):

```
L = (1 − Λ)·rootLoss + Λ·Σ_{non-root v} γ^depth(v)·w_v·ℓ_v
```
- **Λ = `local_law_weight`** — the canonical `ObjectiveSpec` convex split
  (root_share = 1 − Λ). Λ=0 → root-only (pure doc-label fit at the root, the
  reference baseline); Λ=1 → pure distributed law. Default 0.5.
- **γ = `gamma_depth`** — depth discount, **Lean convention depth = max_level − level
  (root = depth 0)**. γ=0 → law collapses to root; γ=1 → all-node. Default 1.0.
  (Bug fixed this session: the first cut used `node.level`, which is inverted —
  it up-weighted leaves. Only γ=1 was safe before the fix.)
- f-phase: rootLoss = doc-label fit at root; lawLoss = leaf preservation (A1),
  proxy==oracle==gold (leaves ARE the text). g-phase: rootLoss = root merge gold fit;
  lawLoss = interior merge preservation (A2), proxy = text read, oracle = gold.
- `a3_factorization_weight` (default 0), `g_assoc_weight` (default 0) add separate
  projections to g. `g_a2_weight` is now a DEPRECATED no-op (subsumed by Λ).
- `f-only` (read whole-doc text through f = the root proxy) is a METRIC/target, not a
  training mode — logged per arm as the standing number to beat.

Reference corners: root-only Λ=0; all-node Λ=1,γ=1; depth-discounted Λ,γ∈(0,1).
Note Λ=0 and γ=0 both route to "supervise only the root," so root-only is cleanly Λ=0.

**Design points open to critique:**
- lawLoss EXCLUDES the root (root is the separate (1−Λ) term) to avoid double-counting.
- Λ is applied to BOTH f and g with the same value. Should f and g have separate Λ?
- Default Λ=0.5 (laws on) vs the canonical `ObjectiveSpec` default root_share=1.0
  (root-only). We chose laws-on-by-default per the project's "every piece has local
  laws" stance; the sweep covers Λ=0.
- "delta": the user mentioned γ/δ/Λ; δ is currently undefined. Is there a real third
  structural knob (role-weight ratio? sampling rate? min-propensity floor?)?

---

## 5. What changed (concrete) and what's validated

Changed:
- `src/ctreepo/fno_family.py`: `_independent_parent_text_readings` (new);
  `_a2_term` rewritten to the AIPW root/law-split merge law; `_a3_factorization_term`
  + `_assoc_term` (new, split out); `_root_law_split_objective` (new shared helper);
  `_train_step_loss_f` / `_train_step_loss_g` route through the canonical objective;
  config: `local_law_weight`, `gamma_depth`, `a3_factorization_weight` added,
  `g_a2_weight` deprecated.
- `scripts/run_manifesto_qsentence_dspy_ladder.py`: CLI flags
  `--fno-local-law-weight`, `--fno-gamma-depth`, `--fno-a3-factorization-weight`;
  `--fno-g-a2-weight` deprecated no-op; `--fno-a2-mode` readout reclassified to A3.
- `tests/test_fno_a2_consistency.py`: rewritten — non-vacuity test (old self-comparison
  ≡ 0, corrected term > 0), unsupervised-interior text-consistency test, Λ root-only
  vs pure-law test, γ=0 collapses-law test, A3 factorization tests.

Validated:
- `pytest tests/test_fno_a2_consistency.py tests/test_fno_null_space_law.py
  tests/test_fno_extent_latent.py tests/test_fno_merge_can_learn_average.py` → 30 pass.
- Benoit econ smoke (leaf16, 2 iters, 3 epochs, warm cache) runs end-to-end at
  defaults and at (Λ=0.5, γ=0.5) — the γ-convention fix is exercised.

NOT yet done (rollout remaining):
- Shim TT `local_law_torch.py` / `local_law_adjustment.py` → `treepo.training.local_law`
  (parity tests); collapse the two `ObjectiveSpec` contracts to one.
- Extend the per-node row contract to `embedding_fno._batch_loss` and the DSPy/manifesto
  family; centralize the open-coded role classification.
- Row-population tests (2L−1 nodes, root counted once, q/N propensity, persistent masks).

---

## 6. Proposed experiment (before launch)

On Benoit EXPERT economic mean (a genuinely NON-additive doc label: CV R² ~0.40 from
codes, so averaging cannot satisfy the laws — the decisive test), leaf16, 8 epochs,
warm embed cache, fanned across 4 GPUs:

| Λ | γ | meaning |
|---|---|---|
| 0.0 | — | root-only baseline (the setting that lost to f-only) |
| 0.25, 0.5, 0.75 | 1.0 | all-node law, increasing law share |
| 0.5 | 0.5, 0.25 | depth-discounted (root-dominant law) |
| 1.0 | 1.0 | pure distributed law |

Metrics per arm: doc-reconstruction Pearson/MAE at the final iter; f-only (root proxy)
logged as the standing target. **Hypothesis the corrected law predicts:** an interior
point (Λ>0, γ≤1) beats root-only, because unsupervised interiors now carry the
text-consistency signal root-only lacks. A sanity gate: any "non-additive" target must
be far below the 0.99 additive-rollup Pearson (`scripts/check_rile_reconstruction_sanity.py`)
or the test is void; Benoit econ passes (root non-additive).

### 6a. Result (3-seed replication, 8 epochs, no HPO) — sober

Λ=0.5 fixed, γ∈{0.5,0.25,0.1}, seeds {42,7,123}, vs root-only (Λ=0). Gap =
merge_it2 − f-only (positive = learned merge beats reading the doc through f).

| config | merge (mean±sd) | gap (mean) | note |
|---|---|---|---|
| root_only | 0.353 ± 0.277 | −0.096 | COLLAPSES on seed 123 (merge −0.038) |
| Λ.5 γ.5 | 0.559 ± 0.078 | +0.013 | stable, never collapses (min 0.45) |
| Λ.5 γ.25 | 0.506 ± 0.133 | −0.040 | collapses on seed 7 (−0.134) |
| Λ.5 γ.1 | 0.522 ± 0.126 | −0.024 | also unstable |

Findings: (1) a single-seed run hit merge 0.633 > f-only 0.620 at γ=0.25, but it did
NOT replicate (seed luck). (2) The robust win is REGULARIZATION/STABILITY, not a level
jump: root-only is high-variance and collapses on some seeds; the distributed law
rescues it (seed 123: root −0.038 → Λ.5γ.5 0.600). sd 0.277 → 0.078. (3) γ=0.5 is the
stable sweet spot; γ≤0.25 over-concentrates on the root and removes the stabilizing
interior law (seed 7 collapse). This mechanistically confirms the Lean story: the
distributed local laws regularize the global objective. NOT supported: "merge
decisively beats f-only" — the level margin is within noise at 8 epochs / no HPO.
Next: more epochs + HPO + more seeds to test the level claim; γ=0.5 is the operating
point.

**Questions for reviewers before we run:**
1. Is the corrected law `f*(A·B)=f*(g(A)·g(B))` the right statement, and is
   descendant-leaf-embedding pooling an acceptable realization of the LEFT side
   `f*(A·B)`, or must the root use re-embedded concatenated text?
2. Is excluding the root from lawLoss (separate (1−Λ) term) correct, or should the
   root's A2 row also live inside the discounted law sum?
3. Should f and g carry the same Λ, or separate?
4. Is the (γ, Λ) grid the right comparison, and what is "delta" if it is a real knob?
5. Anything in the AIPW row contract that diverges from
   `docs/local_law_sampling_contract.md` we should fix before scaling to all families?
