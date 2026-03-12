# Segment‑LDA OPS Weight‑Recovery Simulation (Spec)

This note is intended to be paper text later. It defines the **document generator**, the **oracle**,
and the **OPS-style supervision** used in the Segment‑LDA weight‑recovery simulation.

Code pointers:
- Simulation: `src/tree/segment_lda_ops_weight_recovery_simulation.py`
- Runner: `scripts/run_segment_lda_ops_weight_recovery_simulation.py`
- Sweep command builder: `scripts/build_segment_lda_ops_weight_recovery_cmds.py`
- Grid plotter: `scripts/plot_segment_lda_ops_weight_recovery_grid.py`
- Learning-curve plotter: `scripts/plot_segment_lda_ops_weight_recovery_lines.py`
- Lean alignment:
  - Bigram mergeability pattern: `lean3/FormalProofs/OPT/BigramSketch.lean`
  - Topic oracle sketch (added alongside this spec): `lean3/FormalProofs/OPT/TopicBigramOracle.lean`
  - Ridge “large‑N” intuition lemmas: `lean3/FormalProofs/OPT/RidgeRegressionToy.lean`
  - Pipeline glue (ridge consistency ⇒ score consistency): `lean3/FormalProofs/OPT/SegmentLDAPipelineToy.lean`

## 1) Global parameters

- Number of topics: `K`.
- Vocabulary size: `V`.
- Topic–word distributions: `φ₁,…,φ_K`, each a probability vector over `V` words.

We sample `φ_k` from a Dirichlet prior (implementation supports two modes):
- **disjoint**: each topic has a disjoint vocabulary block (sharp topic boundaries).
- **anchored**: each topic has disjoint “anchor words” plus shared background mass (more realistic).

## 2) Per‑document latent structure (LDA + optional segmentation)

For each document `d`:

1) Draw document length `N_d` (tokens), and fix a leaf size `L` (tokens/leaf). This induces a fixed
   leaf partition of the token indices (used to define the OPS tree leaves).

2) Draw a document topic mixture:
   - `π_d ~ Dirichlet(α_doc · 1_K)` (where `α_doc` is `doc_topic_concentration`).

3) Draw a latent topic sequence `z_{d,1:N_d}` using one of two topic processes:

   **(A) Bag‑of‑words LDA** (`topic_process="bag_of_words"`)
   - For each token `t`: `z_{d,t} ~ Categorical(π_d)` i.i.d.

   **(B) Segmented LDA** (`topic_process="segments"`)
   - Choose a number of segments `S_d`.
   - Choose segment boundaries **at leaf boundaries** (optionally), giving segment lengths in
     units of leaves.
   - For each segment `s`: sample `z_s ~ Categorical(π_d)` **conditioned on `z_s ≠ z_{s-1}`**
     (so segment boundaries are true topic changepoints), and set `z_{d,t}=z_s` for all tokens in
     that segment.
   - Segment lengths are random and variable; in code we sample segment lengths (in leaves) from a
     bounded discrete distribution that can be biased toward longer segments via
     `segment_length_power` (weight ∝ `len^p`).

   Segmentation can be biased by a **global boundary location profile**:
   - define a positive weight function `w(t)` over normalized boundary position `t ∈ (0,1)`.
   - sample each boundary location with probability proportional to `w(t)^p`, where `p` is
     `boundary_profile_strength`.
   - profiles supported: `uniform`, `start`, `middle`, `end`, `bimodal`, `random`.

   This creates *globally learnable* structure such as “changepoints tend to occur near the middle”.

4) Emit observed words:
   - for each token `t`: `x_{d,t} ~ Categorical(φ_{z_{d,t}})`.

Only `x` is observed; the topic sequence `z` is latent.

## 3) Oracle (mergeable target)

The oracle is a linear functional of the *topic* sequence over a span:

`f⋆(span) = ⟨θ, c(span)⟩ + λ · ⟨W, b(span)⟩`

where:
- `c(span) ∈ ℕ^K` are topic unigram counts in the span,
- `b(span) ∈ ℕ^{K×K}` are topic bigram counts in the span,
- `θ ∈ ℝ^K` is sparse (only a few topics matter),
- `W ∈ ℝ^{K×K}` is sparse (typically only transitions involving the relevant topics matter),
- `λ ≥ 0` scales the bigram term (“how much boundaries matter”).

### Mergeability / sketch state

To compute `f⋆(u ++ v)` exactly from span-level summaries, we need:
- topic unigram counts on `u` and `v`,
- topic bigram counts on `u` and `v`,
- the last topic id of `u` and first topic id of `v` (to add the cross-boundary bigram).

This is the same boundary-metadata pattern formalized in:
- `lean3/FormalProofs/OPT/BigramSketch.lean` (generic bigram mergeability),
specialized to topic ids (see `TopicBigramOracle.lean`).

## 4) OPS-style supervision and estimation

We build a fixed balanced binary tree over the token sequence with leaf size `L`.

Training supervision:
- **Always** query all leaves (one oracle score per leaf span).
- Query a subset of internal nodes according to:
  - `audit_policy` (e.g. `fraction`, `sqrt`, `log2`, `fixed`, `all`)
  - `audit_strategy` (which internal spans to label):
    - `random`: uniform over internal nodes
    - `active_small`: prioritize 2-leaf internal nodes, then larger nodes
    - `profile`: prioritize 2-leaf internal nodes at leaf-boundary indices with higher empirical
      changepoint rates (estimated globally across training docs from observed words)

Features for regression:
- Either use **true** topic ids (an upper bound), or **infer** per-leaf topics from words via a
  maximum-likelihood assignment under `φ` (then repeat that topic id across tokens in the leaf).

Estimator:
- Fit ridge regression to recover `(θ, λW)` from span features and oracle responses.
  - Optional training-label noise: `y = f⋆(span) + ε`, with `ε ~ Normal(0, σ²)` controlled by
    `oracle_noise_std`. Evaluation remains noiseless.
  - Objective (unweighted): minimize `Σ (y - ⟨β, x⟩)² + ρ‖β‖²` over all labeled spans (leaves + sampled internals).

Evaluation:
- Root distortion / merge distortion diagnostics (OPS semantics).
- Parameter recovery diagnostics (cosines/RMSEs for `θ`, `W` direction, and `λ`).

## 5) Recommended sweeps (grid axes)

Minimal, interpretable grid:
- `train_docs` (more data ⇒ less estimation error),
- internal labels per leaf (e.g. via `audit_fraction`),
- `lambda_multiplier` (need for boundary correction),
- optional: `topic_process` (`bag_of_words` vs `segments`) as a negative control.

Global-structure sweep (to test “learnability” of where boundaries are):
- `boundary_profile` × `boundary_profile_strength`,
- compare `audit_strategy=random` vs `audit_strategy=profile`.
