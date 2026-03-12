# Tree-Relevant LDA Simulation Ladder

This is the current paper-facing ladder for the topic-model simulations.

The guiding principle is simple:

1. start in ordinary bag-of-words LDA, where the exact mergeable statistic is known;
2. measure the cost of compressing that mergeable statistic;
3. only then move to a document model where leaves carry genuine statistical information.

Leaf size is always reported as a **fraction of document length** rather than as a raw token count.
For the default `doc_tokens = 384`, the main grid is:

- `100%` of the document
- `50%` of the document
- `25%` of the document
- `4.17%` of the document (`1/24`, i.e. `16/384`)

## Stage 1: Ordinary LDA Utility Sketches

Document model:

\[
\pi_d \sim \mathrm{Dir}(\alpha),\qquad
z_{d,t}\mid\pi_d \sim \mathrm{Cat}(\pi_d),\qquad
x_{d,t}\mid z_{d,t}=k \sim \mathrm{Cat}(\phi_k).
\]

Primary document-level target:

\[
Y_d = r^\top U c_d.
\]

Mergeable supervision sketch:

\[
u(A) = U c_A,
\]

where \(c_A\) is the bag-of-words histogram on span \(A\) and \(U\) is a fixed
topic-anchored sparse utility matrix. The tree methods use \(u(A)\) because it gives exact
subset labels and merges additively, but the object we ultimately score is still the
full-document scalar \(Y_d\).

Files:

- core: `src/ctreepo/sim/core/lda_tree_utility_vector.py`
- runner: `scripts/run_lda_tree_utility_vector_simulation.py`
- sweep builder: `scripts/build_lda_tree_utility_vector_cmds.py`

Main methods:

- `full_doc_exact_utility`
- `tree_exact_utility`
- `count_svd_ceiling`
- `utility_pca_practical`
- `full_doc_mlp_diag` (appendix diagnostic)

Main claims:

- the exact tree utility sketch must match the full-document objective exactly;
- utility-sketch compression becomes exact at `state_dim = utility_dim`;
- count-sketch compression becomes exact at `state_dim = vocab_size`.

## Stage 2: Leaf-Local-Mixture Utility

Document model:

\[
\pi_d \sim \mathrm{Dir}(\alpha),\qquad
\pi_{d,b} \sim \mathrm{Dir}(\tau \pi_d),
\]

with tokens inside latent base leaf \(b\) drawn iid from \(\pi_{d,b}\).

Primary document-level target:

\[
y_d = \sum_b n_b h(\pi_{d,b}),\qquad
h(\pi)=\theta^\top \pi + \lambda \pi^\top W \pi.
\]

This remains a scalar function of the full document. Leaf labels are only an auxiliary
training signal for the tree method.

Files:

- core: `src/ctreepo/sim/core/leaf_local_mixture_utility.py`
- runner: `scripts/run_leaf_local_mixture_utility_simulation.py`
- sweep builder: `scripts/build_leaf_local_mixture_utility_cmds.py`

Main methods:

- `pooled_doc_wrong_model`
- `leaf_oracle_sum`
- `leaf_infer_sum`
- `leaf_ridge_from_u`
- `coarse_leaf_ridge_from_u`

`leaf_infer_sum` runs per-leaf EM topic inference using the known topic-word matrices
(the same procedure the pooled baseline uses on the full document) and sums
`n_b h(\hat\pi_b)` across leaves. It requires no training labels and provides the
natural tree-based analog of the pooled baseline.

The `latent_leaf_tokens` parameter (default 16) controls the size of each latent base
leaf. The sweep `{16, 32, 64, 96}` tests whether larger leaves (with more tokens for
per-leaf inference) allow leaf-based methods to outperform pooling.

Budget regimes:

- `all_leaves_labeled`
- `fixed_oracle_budget`

The fixed-budget regime is measured in **latent-leaf-label equivalents**, so the
budget remains comparable across coarse and fine evaluation leaf resolutions.

## Formalization

Stage 1 exactness:

- `lean3/FormalProofs/OPT/BagOfWordsLDARecovery.lean`

Stage 2 leaf-local-mixture gap:

- `lean3/FormalProofs/OPT/LeafLocalMixtureUtilityGap.lean`

The key Stage-2 identity is that for
\[
h(\pi)=\theta^\top \pi + \lambda \pi^\top W \pi,
\]
the pooled-vs-leaf gap is exactly the quadratic gap scaled by \(\lambda\).

## Paper Report

- report script: `scripts/report_lda_tree_methods_paper.py`

This report consumes the Stage-1 and Stage-2 output trees directly and keeps the
resolution labels in percent-of-document terms throughout.
