# LDA Aggregate Statistics Formalization

This note pins down the deterministic part of our LDA approximation: how leaf
statistics aggregate into document statistics before we talk about estimation
error, learned compression, or oracle-query budgets.

Lean anchor:

- `lean3/FormalProofs/OPT/BagOfWordsLDARecovery.lean`
- `lean3/FormalProofs/OPT/LDAAggregateStatistics.lean`

Python anchors:

- `src/ctreepo/sim/core/lda_tree_recovery.py`
- `src/ctreepo/sim/core/lda_tree_recovery_learned.py`
- `src/ctreepo/sim/core/segmented_lda_ctreepo.py`
- `src/ctreepo/sim/core/segment_lda_ops_weight_recovery.py`

## 1. Leaf Statistic

For ordinary LDA, the modeled document input is bag-of-words data, not
semantic text. In this file the token-list formulas are a concrete derivation
of the local bag/statistic observation supplied to the theorem-facing `g`: a
leaf can be realized by tokens, but `ldaExactG` consumes the aggregate count
state.

Let a document be split into leaves \(L_1,\ldots,L_m\). Each token realization
\(x_t\) has an observed word \(w_t \in V\) and a hard or soft topic
responsibility vector \(q_t \in \mathbb{R}^K\). In the true-topic path,
\(q_t(k)=1\{z_t=k\}\). In the estimated-topic path, \(q_t\) is the
inferred/simplex topic weight.

For any realized span \(A\), define its bag/statistic observation:

\[
N_A = \sum_{t \in A} 1,
\qquad
C_A(v) = \sum_{t \in A} 1\{w_t=v\},
\]

\[
T_A(k) = \sum_{t \in A} q_t(k),
\qquad
R_A(v,k) = \sum_{t \in A} 1\{w_t=v\} q_t(k).
\]

The outer-product word co-occurrence approximation is:

\[
M_A(v,u) = C_A(v) C_A(u).
\]

In Lean these are `tokenMass`, `wordMass`, `topicMass`,
`wordTopicMass`, and `wordCoocOuter`.

## 2. Merge Rule

For adjacent spans \(A\) and \(B\), the additive fields merge directly:

\[
N_{A \cup B} = N_A + N_B,
\quad
C_{A \cup B} = C_A + C_B,
\quad
T_{A \cup B} = T_A + T_B,
\quad
R_{A \cup B} = R_A + R_B.
\]

The outer-product co-occurrence field gets the two cross terms:

\[
M_{A \cup B}(v,u)
= M_A(v,u) + M_B(v,u)
  + C_A(v) C_B(u) + C_B(v) C_A(u).
\]

The Lean theorem `ldaAggregateStats_append` proves exactly this append/merge
identity. The tree theorem `ldaAggregateTreeStats_eq_full` then proves by
induction that any binary reduction over leaves recovers the full-document
statistic:

\[
\mathrm{TreeAgg}(L_1,\ldots,L_m)
= \mathrm{Stats}(L_1 \Vert \cdots \Vert L_m).
\]

## 3. Topic Proportion Readout

Document topic proportions are a readout from root topic counts:

\[
\hat\pi_d(k) = \frac{T_d(k)}{N_d}.
\]

With prior smoothing:

\[
\hat\pi_d^{(\alpha)}(k)
= \frac{\alpha_k + T_d(k)}{\alpha_0 + N_d}.
\]

The Lean theorems `topicProportion_tree_eq_full` and
`smoothedTopicProportion_tree_eq_full` show that these proportions are identical
whether read from the tree root or from the full document statistic.

If the per-token topic weights are simplex-valued, Lean also proves:

\[
\sum_k T_A(k) = N_A.
\]

This is `sum_topicMass_eq_tokenMass_of_simplex`.

## 4. Leaf Weighted Averages

The additive equalities imply the weighted-average readouts. Let leaf
statistics be

\[
s_\ell = (N_\ell, C_\ell, T_\ell, R_\ell),
\qquad
N = \sum_\ell N_\ell,
\qquad
\lambda_\ell = N_\ell/N.
\]

For word proportions:

\[
\frac{C_d(v)}{N}
= \frac{\sum_\ell C_\ell(v)}{N}
= \sum_\ell \frac{N_\ell}{N}\frac{C_\ell(v)}{N_\ell}
= \sum_\ell \lambda_\ell \hat p_\ell(v).
\]

For topic proportions:

\[
\hat\pi_d(k)
= \frac{T_d(k)}{N}
= \frac{\sum_\ell T_\ell(k)}{N}
= \sum_\ell \frac{N_\ell}{N}\frac{T_\ell(k)}{N_\ell}
= \sum_\ell \lambda_\ell \hat\pi_\ell(k).
\]

For normalized word-topic joint mass:

\[
\hat J_d(v,k)
= \frac{R_d(v,k)}{N}
= \sum_\ell \frac{N_\ell}{N}\frac{R_\ell(v,k)}{N_\ell}.
\]

For the conditional word distribution within topic `k`, the weights are topic
masses rather than token masses. If
\(T_d(k)=\sum_\ell T_\ell(k)\), then

\[
\hat\phi_d(v\mid k)
= \frac{R_d(v,k)}{T_d(k)}
= \sum_\ell \frac{T_\ell(k)}{T_d(k)}
    \frac{R_\ell(v,k)}{T_\ell(k)}.
\]

Lean now proves these statements for bag/stat leaves:

- `ldaBagTreeStats_tokenMass_eq_leaf_sum`
- `ldaBagTreeStats_wordMass_eq_leaf_sum`
- `ldaBagTreeStats_topicMass_eq_leaf_sum`
- `ldaBagTreeStats_wordTopicMass_eq_leaf_sum`
- `lda_topicProportion_eq_tokenWeightedLeafAverage`
- `lda_wordProportion_eq_tokenWeightedLeafAverage`
- `lda_wordTopicJointProportion_eq_tokenWeightedLeafAverage`
- `lda_wordGivenTopicProportion_eq_topicMassWeightedLeafAverage`

The nonzero assumptions in the normalized Lean theorems are exactly the usual
"do not divide by an empty leaf/topic" assumptions. In the simulation setting,
these are guaranteed by positive leaf token counts, and by restricting
conditional topic-word readouts to topics with nonzero expected mass.

For ordinary bag-of-words LDA likelihood with fixed document topic mixture
\(\pi\) and topic-word matrix \(\phi\), define the marginal token probability

\[
m_{\pi,\phi}(v)=\sum_k \pi(k)\phi_k(v).
\]

Ignoring the multinomial coefficient, the bag likelihood factors over leaves:

\[
p(c_d\mid \pi,\phi)
= \prod_v m_{\pi,\phi}(v)^{c_d(v)}
= \prod_\ell \prod_v m_{\pi,\phi}(v)^{c_\ell(v)}.
\]

The cleaner additive statistic is the log-likelihood. Let
\(\ell_{\pi,\phi}(v)=\log m_{\pi,\phi}(v)\) and

\[
\mathrm{LL}_{\pi,\phi}(c)=
  \sum_v c(v)\ell_{\pi,\phi}(v).
\]

Since \(c_d(v)=\sum_\ell c_\ell(v)\),

\[
\mathrm{LL}_{\pi,\phi}(c_d)
= \sum_\ell \mathrm{LL}_{\pi,\phi}(c_\ell).
\]

Normalizing by token count gives the weighted-average form. With
\(N_\ell=\sum_v c_\ell(v)\) and \(N=\sum_\ell N_\ell\),

\[
\overline{\mathrm{LL}}_{\pi,\phi}(c_d)
= \sum_\ell \frac{N_\ell}{N}
  \overline{\mathrm{LL}}_{\pi,\phi}(c_\ell),
\qquad
\overline{\mathrm{LL}}_{\pi,\phi}(c)
= \frac{\mathrm{LL}_{\pi,\phi}(c)}{\sum_v c(v)}.
\]

Lean states the multiplicative form as
`ldaHistogramLikelihood_bagOfWordsTree_eq_leaf_prod` and the shared-`g` form as
`ldaHistogramLikelihood_uniformG_eq_leaf_prod`. It also formalizes the
log-likelihood version directly through:

- `histogramTokenMass`
- `ldaHistogramLogLikelihood`
- `ldaAverageLogLikelihood`
- `histogramTokenMass_bagOfWordsTree_eq_leaf_sum`
- `ldaHistogramLogLikelihood_bagOfWordsTree_eq_leaf_sum`
- `ldaHistogramLogLikelihood_uniformG_eq_leaf_sum`
- `ldaAverageLogLikelihood_bagOfWordsTree_eq_tokenWeightedLeafAverage`
- `ldaAverageLogLikelihood_uniformG_eq_tokenWeightedLeafAverage`

The outer-product word co-occurrence statistic is different: it is not merely a
weighted average of within-leaf outer products. It requires cross-leaf terms,
which are exactly the two extra terms in `mergeLDAAggregateStats`:

\[
C_{A\cup B}(v) C_{A\cup B}(u)
= C_A(v)C_A(u) + C_B(v)C_B(u)
  + C_A(v)C_B(u) + C_B(v)C_A(u).
\]

So the correct claim is: additive LDA sufficient statistics normalize to
weighted averages of leaf readouts; non-additive co-occurrence moments are
still exactly tree-recoverable, but through a merge rule that carries the
cross terms.

## 5. Adjacent Word Co-occurrences

The outer-product matrix above is the bag-of-words co-occurrence approximation.
For adjacent word co-occurrences, plain addition is not enough because the
boundary pair between leaves is missing. The exact sketch carries first/last
boundary words and an adjacent-bigram multiset:

\[
B_{A \cup B}
= B_A + B_B + \{(\mathrm{last}(A), \mathrm{first}(B))\}
\]

when both boundary words exist.

Lean reuses `BigramSketch.lean` and proves `wordBigramTreeSketch_eq_full`: the
boundary-carrying tree sketch recovers the full-document adjacent-word bigram
sketch exactly.

## 6. What This Does And Does Not Claim

This formalizes the bookkeeping layer of our recreated LDA approximation:
leaf-level counts and responsibility-weighted statistics can be merged into
document-level statistics without loss.

It does not prove that a learned topic estimator is statistically consistent,
that a neural compression is lossless, or that per-leaf posterior inference is
exact. Those are separate approximation layers. The result here is the basic
structural claim: if leaves expose the relevant local sufficient statistics,
the document statistic and topic-proportion readout are exactly recoverable by
tree aggregation.

## 7. Oracle/Summary Decomposition And Doc-Level Supervision

The notation should match the theory convention:

\[
\iota_{\mathrm{leaf}} : \text{leaf bag/stat observation} \to X,
\qquad
\iota_{\mathrm{merge}} : X \times X \to X,
\qquad
g : X \to X,
\qquad
\mathrm{Fold}_g : \text{bag-stat tree} \to X,
\qquad
f : X \to Y,
\]

where \(g\) is the shared summarizer/compressor, \(\mathrm{Fold}_g\) is the
induced tree summary, and \(f\) is the oracle/readout. The carrier \(X\)
contains encoded bag/stat leaf observations and intermediate summary states.
Leaf observations and merge material are first placed inside \(X\); then the
identical endomap \(g : X \to X\) is applied at leaves and internal nodes:

\[
f(g(\iota_{\mathrm{leaf}}(s_L))) = f(\mathrm{state}(s_L)),
\qquad
f(g(\iota_{\mathrm{merge}}(s_A,s_B))) =
  f(\mathrm{state}(\mathrm{Merge}(\mathrm{stats}(s_A),\mathrm{stats}(s_B)))).
\]

Lean encodes this with the shared `UniformG` interface, whose fields are
`leafInput : Leaf -> Carrier`, `mergeInput : Carrier -> Carrier -> Carrier`,
and the single shared function `g : Carrier -> Carrier`. The LDA-specialized
evaluator `ldaGTreeEval` applies that same `g` at every node; it does not take
separate leaf and merge summary functions.

For exact LDA, `ldaExactG` chooses an explicit carrier `LDACarrier` with
bag-observation and aggregate-state constructors:

\[
\mathrm{bag}(s) \in X,
\qquad
\mathrm{state}(s) \in X,
\qquad
g(\mathrm{bag}(s))=\mathrm{state}(s),
\qquad
g(\mathrm{state}(s))=\mathrm{state}(s),
\qquad
\iota_{\mathrm{merge}}(u,v)
  = \mathrm{state}(\mathrm{Merge}(\mathrm{stats}(u),\mathrm{stats}(v))).
\]

The main theorem is `ldaGTreeEval_exact_eq_full`:

\[
\mathrm{stats}(\mathrm{Fold}_{g}(T))
= \mathrm{BagFold}(T).
\]

The bridge theorem `ldaBagTreeStats_tokenTreeBags_eq_full` then says that when
a bag-stat tree is generated from token realizations, \(\mathrm{BagFold}(T)\)
coincides with the full-document statistic \(\mathrm{Stats}(S(T))\).

For a document-level LDA oracle \(f^\star\), we make the needed assumption
explicit:

\[
\exists r : S \to Y,\qquad
r(s) = f^\star(s)
\quad \text{for every bag/stat document observation } s.
\]

This says the target factors through the LDA aggregate state. Under that
assumption, Lean proves `lda_exact_summary_recovers_fstar_of_factorization`:

\[
f(\mathrm{Fold}_{g}(T)) = f^\star(\mathrm{BagFold}(T))
\quad \text{for every bag/stat document tree } T.
\]

It also proves `lda_exact_summary_zero_doc_supervision_of_factorization`: the
same exact summary/oracle pair has zero root/doc-level supervision error against
\(f^\star\) on any training or support predicate.

The important distinction is that Lean proves realizability and exact recovery
for this hypothesis class. It does not claim that arbitrary finite root labels
identify the exact pair uniquely, or that a particular optimizer will find it
without additional assumptions. In implementation terms: doc-level supervision
can fit an exact \(g\) summary and \(f\) readout when the model class contains
this aggregate-state factorization and the supervised objective reaches the
zero-loss solution.
