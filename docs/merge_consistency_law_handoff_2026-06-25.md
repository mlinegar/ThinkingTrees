# Handoff: through-f merge-consistency laws on the FNO (2026-06-25)

> **SUPERSEDED IN PART (2026-06-25, later same day).** Two claims below are WRONG and
> were corrected:
> 1. **The "a2state beats control (0.424 vs 0.353)" result in §3 was MISATTRIBUTED.**
>    The A2 "state" term compared `f(parent_state)` to `f(merge(l,r))`, but the FNO
>    forward pass DEFINES `parent_state = merge(l,r)`, so that residual was
>    **identically zero — a no-op**. The arm also ran `--fno-g-assoc-weight 0.5`, so
>    the gain came from the ASSOCIATIVITY penalty, not the A2 law.
> 2. **The real merge law is `f*(A·B) = f*(g(A)·g(B))`** — the merge route vs an
>    INDEPENDENT reading of the parent text (NOT the merge-derived state).
> The corrected law is implemented and routed through the single canonical
> `treepo.training.local_law` objective (AIPW + γ^depth + `(1−Λ)root+Λ·law`).
> 3-seed result: the laws REGULARIZE (root-only collapses on some seeds; the law
> rescues it); γ=0.5 is the stable operating point; "merge decisively beats f-only" is
> NOT yet supported (margin within noise at 8 epochs/no HPO).
> **See `docs/local_law_canonical_path_review_2026-06-25.md` (self-contained),
> `docs/local_law_single_path_plan.md` (living plan), and memory
> `project_local_law_single_canonical_path`.** §4's "open problem" is now addressed by
> the corrected law; the rest of this doc (framework, additive-vs-non-additive Benoit
> econ rationale, grid locations) remains valid.

Self-contained resume doc. Goal: learn a tree merge `g` whose composition of local
(leaf/node) readings reconstructs the global doc label, where the merge weighting is NOT
estimated but FALLS OUT of satisfying the local laws — measured THROUGH the readout `f`.

## 1. The core idea / framework

- Tree of text: leaves = quasi-sentence chunks, internal nodes = merges, root = whole doc.
- `f` = readout (state -> scalar/low-dim score). `g` = encoder/merge (text or child states ->
  state). We want `f(g(tree)) ~ f*(doc)` where `f*` is the oracle reading and `~` is equality
  THROUGH the readout (Lean `D f* z x = dist(f*(z), f*(x))`).
- Two laws the user reduced everything to (their words), each measured through f (f as the
  f* proxy, valid only when f is accurate):
  - **Law 1 (leaf sufficiency):** `g(leaf) ~ gold`.            [Lean A1: `D f*(g z, z)=0`]
  - **Law 2 (doc reconstruction):** apply g to all leaves, build the tree; at the doc level
    `g(X) ~ X`.                                                  [Lean A2 at the root]
- Lean source (verified this session): `lean3/FormalProofs/OPT/GlobalAssumptions.lean` +
  `CoreDefinitions.lean`. A3 = the merge factors through the readout: `∃ M, f(g(gu·gv)) =
  M(f(gu), f(gv))`, with `merge_assoc`/`merge_comm` proven (Gibbons 3rd-homomorphism /
  mergeable-sketch family; the Lean comment literally calls it "the C-TreePO-relevant
  theorem").

## 2. The central finding (why everything before failed)

**The merge weight should NOT be estimated.** We tried and refuted explicit weight estimation:
- **Mass/extent latent** (learned scalar "how much" per node, fed to a gated convex merge):
  all arms ~0.135 per-node merge wMAE (~34x over the equal_avg bar 0.00406). The extent
  latent VARIED but never became mass. Convex gate `alpha*l+(1-alpha)*r` is structurally
  trapped BETWEEN the two children, so a salient short child decays toward the mean up the
  tree.
- **f-null-space salience law** (push low-impact content into f's null space so an additive
  merge ignores it): sweep weight {0,0.5,1,2,4} -> 0.136/0.130/0.139/0.147/0.157. Did NOT
  help; monotonically WORSE at higher weight. A surrogate, not the law.
- **A2 law on domain_4:** control 0.136 / a2state 0.139 / a2readout 0.129. All ~0.13.

**WHY they all stalled — the key realization:** the MPDS targets are ADDITIVE by construction.
`domain_k = (count of domain-k quasi-sentences) / total_non_header` — a COUNT RATIO, so the
merge IS mass-weighted averaging and salience == frequency. The RILE Step-0 gate proves global
RILE is a near-exact additive rollup of local codes (Pearson 0.9975;
`scripts/check_rile_reconstruction_sanity.py`). On an additive label, averaging SATISFIES the
laws — so no merge objective can beat the averager. The laws were working; the TARGET was
additive.

**=> The construct question and the two-laws question are the SAME question:** the laws force a
non-trivial (non-additive) merge IFF the doc label X is NOT a near-additive rollup of leaves.

## 3. The decisive experiment (the current frontier)

Use a NON-additive doc label that already exists: **Benoit EXPERT economic mean** (holistic
expert 1-7 score; CV R^2 only 0.40 from CMP codes vs RILE's 0.9975 -> NOT a rollup; "an LLM
reading the text beats the gold-code ceiling" — the salience point, already proven). No new
labeling.

- Grid on disk: `outputs/benoit_llmseg_economic_none/leafq016/labeled_trees.jsonl` (177 docs,
  splits train100/val27/test50). Structure VERIFIED: 75 leaves scored (Law 1 = per-chunk LLM
  econ score), 77 interior merges = None (UNSUPERVISED — g-loss skips None nodes), 1 scored
  merge = ROOT (Law 2 = expert mean, Pearson 1.0 vs `outputs/benoit_qsentence_targets/
  expert_means_raw.json`). Root IS non-additive: leaf-mean 0.504 vs expert root 0.683.
  So this is ALREADY ~the clean "two laws only" objective.

- Run: `outputs/fno_benoit_econ_law_ab_20260625_065231` (3 arms, leaf16, merge_mode=mlp,
  --fno-target-dimension economic). Driver `scripts/run_fno_benoit_econ_law_ab.sh`.

- **RESULT (doc reconstruction, test split):**

  | arm | doc MAE | doc Pearson |
  |-----|--------:|------------:|
  | control (node-MSE averager) | 0.273 | 0.353 |
  | **a2state** (A2 state-merge + assoc) | **0.192** | **0.424** |
  | a2readout (A3 phi-form, 1 scalar) | 0.199 | 0.269 |

  a2state BEATS control (+0.07 Pearson, -30% MAE) — the FIRST merge win, ONLY because the
  label is non-additive (domain_4 arms ~= control). Validates the whole chain.

- **HONEST CAVEAT (the real story — do not overclaim):** iter1 (f-only, BEFORE the g-merge) is
  IDENTICAL across arms at **Pearson 0.526 / MAE 0.177** — the BEST number. iter2 (training the
  g-merge) makes EVERY arm WORSE: control 0.526->0.353, a2state 0.526->0.424, a2readout
  0.526->0.269. So: the FNO merge DEGRADES doc reconstruction (regresses toward averaging on
  its own through the chain of free interior merges); the A2 law MITIGATES the damage (least
  degradation). The merge still LOSES to reading the root through f directly. a2readout (A3
  single-scalar phi) underperformed control — too low-capacity; the full-mlp state-merge is
  what helped. (both/compare design paid off: A3-literal readout-factoring is WORSE here.)
  Single 8-epoch run, no HPO; 0.42 is below the tuned chunk-FNO baseline 0.66-0.73 (NOT
  apples-to-apples — baseline had HPO + llm-span supervision; the valid comparison is
  a2state-vs-control at identical settings = clear win).

## 4. THE OPEN PROBLEM (what to do next)

**Why does ANY merge lose to reading the root through f directly (0.526)?** Recommended next
steps, priority order:

1. **Make the root/A2 signal dominate the merge.** `root_weight >> merge_weight`, more epochs,
   sweep `a2_weight`. The merge self-regresses toward averaging; push harder against it.
2. **Trained salience-weighted leaf-readout pool arm.** Since f-only (effectively a pool over
   leaf readings) WINS, the doc label may be a salience-weighted POOL of leaf readings, not a
   binary-tree merge. Try a trained pool readout (`predict_root_topk` / softmax over leaf
   f-scores, already in `embedding_fno.py`). This directly tests the "push low-info aside /
   salient leaf dominates" intuition at the readout level. THIS MAY BE THE REAL ANSWER.
3. **Give the A3 phi more capacity** — a learned monotone net (still assoc+comm via the Aczel
   form `phi^{-1}(phi(a)+phi(b))`) instead of one scalar offset, IF pursuing readout-factoring.

Recommended: launch #1 + #2 as a small parallel sweep reusing the warm Benoit embed cache.

## 5. What's implemented (all landed, 28+ FNO tests green)

- `src/ctreepo/embedding_fno.py`:
  - Extent latent (`extent_enabled`, `extent_merge_init`, `_split_extent`, `extent_merge`) —
    REFUTED approach, kept off by default; back-compat verified (old ckpts load w/ extent off).
  - `readout_merge(a,b)` = A3 phi-form `sigmoid(logit a + logit b + offset)`, assoc+comm BY
    CONSTRUCTION (Aczel). One param `readout_merge_offset`.
  - Merge modes: `mean`/`gated`/`maxpool`/`mlp`. The non-additive experiments use `mlp`
    (free, non-convex; a salient child can dominate). Channel invariant: leaf_fno 1->1,
    merge_fno 2->1, scalar score head.
- `src/ctreepo/fno_family.py` (`FNOFamilyConfig` + losses):
  - `g_a2_weight`, `a2_mode` {state|readout}, `g_assoc_weight` -> `_a2_term` (Lean A2 loss).
  - `g_null_space_weight` -> `_null_space_term` (REFUTED null-space surrogate; keep off).
  - `extent_enabled`/`extent_merge_init`/`g_depth_lopsided_strength` (REFUTED extent; off).
  - g-loss = node-MSE (+ optional a2/null/lopsided terms). f-loss = leaf/merge/root weighted
    MSE through f.
- CLI (`scripts/run_manifesto_qsentence_dspy_ladder.py`): `--fno-g-a2-weight`, `--fno-a2-mode`,
  `--fno-g-assoc-weight`, `--fno-g-null-space-weight`, `--fno-extent*`, `--embedding-cache-dir`,
  plus the existing `--fno-target-dimension`, `--fno-merge-mode`.
- `scripts/dump_fno_g_node_states.py`: LM-free per-node g-state dump -> generic
  `scripts/eval_qsentence_merge_by_level.py --g-states-jsonl` (per-node merge wMAE vs the
  equal_avg bar 0.00406 / mass_wtd=0 ceiling). NOTE: arch flags MUST match the trained ckpt
  (hidden=32, modes=64, layers=2, head=64 are the ladder defaults).
- `src/ctreepo/embedding_cache.py`: `DiskCachedEmbeddingClient` — embed each chunk ONCE,
  cache to disk (256 shards, atomic merge). The embeddinggemma pass dominates; arms re-embed
  the SAME chunks. Drivers prewarm one point then fan the rest across all 4 GPUs.

## 6. Tests (run from repo root, use `venv/` NOT `.venv/`)

`pytest tests/test_fno_a2_consistency.py tests/test_fno_null_space_law.py
tests/test_fno_extent_latent.py tests/test_fno_merge_can_learn_average.py
tests/test_embedding_cache.py -q`  (28 pass)

Key guards: A3 readout_merge assoc+comm by construction; A2 term zero when consistent,
positive when not; extent off == byte-identical/back-compat; embedding cache hit/miss/persist.

## 7. Operational notes (load-bearing)

- Run long jobs via `scripts/long_job.py launch ... --job-root <root>_launcher`; stop via
  `long_job.py stop --job-root <root>_launcher`. NEVER `pkill` near jobs.
- Embed cache to reuse: `outputs/fno_benoit_econ_law_ab_20260625_065231/embed_cache` (Benoit
  econ chunks) and `outputs/fno_nullspace_sweep_20260625_022747/embed_cache` (MPDS chunks).
- Score per-node merge: dump (matching arch flags) -> `eval_qsentence_merge_by_level.py`.
  Score doc reconstruction (Law 2): read `<run>/fno/leafq016/iteration_history.json`
  -> iterations[-1].split_metrics.test.{internal_f_mae, internal_f_pearson}.
- Sanity gate any NEW target X: it must be FAR below 0.99 additive-rollup Pearson
  (`check_rile_reconstruction_sanity.py`) or the test is void.

## 8. One-line state

The through-f merge-consistency law (Lean A2) BEATS the averager on a non-additive doc label
(Benoit econ: a2state 0.424 vs control 0.353 Pearson) where it CANNOT on additive labels
(domain_4) — proving the laws force non-additive composition. BUT every learned merge still
loses to reading the root through f directly (f-only 0.526), so the open problem is: make the
merge actually IMPROVE on f-only, likely via root-dominant weighting and/or a trained
salience-weighted leaf pool (the doc label may be a pool, not a binary merge).
