# LDA Tree-Recovery Simulation Spec

This is the simulation spec that should govern the next buildout of the topic-model experiments.

It is intentionally anchored to the end goal:

- prove we can recover ordinary bag-of-words LDA exactly with a tree when no information is lost,
- then measure how much performance is lost when we compress those mergeable summaries,
- then move to local latent structure where leaves become statistically meaningful.

This family should be the entry point for the paper story, the report figures, and the next production sweeps.

Code paths:

- Core simulation: `src/ctreepo/sim/core/lda_tree_recovery.py`
- Learned extension: `src/ctreepo/sim/core/lda_tree_recovery_learned.py`
- Runner: `scripts/run_lda_tree_recovery_simulation.py`
- Learned runner: `scripts/run_lda_tree_recovery_learned_simulation.py`
- Sweep builder: `scripts/build_lda_tree_recovery_cmds.py`
- Learned sweep builder: `scripts/build_lda_tree_recovery_learned_cmds.py`

## 1. End Goal

The end goal is not "show some topic-model curve is green."

It is to establish a clean ladder:

1. **Exact recovery**: the tree architecture can reproduce the full-document LDA answer exactly.
2. **Compression gap**: learned mergeable sketches can approximate that exact answer with finite sketch size.
3. **Locality value**: once documents have genuinely local latent mixtures, leaf-aware modeling buys something real.

If a simulation family does not help answer one of those three questions, it is probably not central.

## 2. Base Family: Exact Bag-of-Words LDA Recovery

Document-level DGP:

\[
\pi_d \sim \mathrm{Dir}(\alpha), \qquad
z_{d,t} \mid \pi_d \sim \mathrm{Cat}(\pi_d), \qquad
x_{d,t} \mid z_{d,t}=k \sim \mathrm{Cat}(\phi_k).
\]

Observed document:

\[
x_d = (x_{d,1}, \dots, x_{d,n_d}).
\]

Exact sufficient statistic:

\[
c_d(v) = \sum_t \mathbf{1}\{x_{d,t}=v\}.
\]

Tree representation:

- partition the document into leaves,
- compute leaf histograms \(c_{d,\ell}\),
- merge by addition:

\[
c_{A \cup B} = c_A + c_B.
\]

This must satisfy:

\[
c_d = \sum_\ell c_{d,\ell},
\]

independent of tree shape or merge schedule.

This is the exact positive control.

### 2.1 Clean Direct-Utility Corollary

A particularly clean special case is a fixed word-weight vector
\(w \in \mathbb{R}^V\) and the histogram-linear utility

\[
u_w(c_d) = w^\top c_d.
\]

This is attractive because subset oracle labels are exact from observed words:
on a leaf \(\ell\),

\[
u_w(c_{d,\ell}) = w^\top c_{d,\ell}.
\]

No latent-topic inference is required to label subsets.

It also remains connected to LDA. If \(\Phi \in \mathbb{R}^{K \times V}\) is the
topic-word table and \(\pi_d\) is the document mixture, then conditional on
\(\pi_d\),

\[
\mathbb{E}[u_w(c_d)\mid \pi_d]
= n_d \, (\Phi w)^\top \pi_d.
\]

So a fixed word-linear utility is an induced linear function of the LDA topic
mixture in expectation, while still admitting exact word-level oracle labels on
every subset.

This is a good paper-facing bridge:

- recover counts exactly,
- note that any fixed \(u_w\) is then exact automatically,
- only then decide whether to present the results as "recovering LDA itself" or
  "recovering a function of LDA topics with exact subset labels."

## 3. Tasks Within The Base Family

There are three tasks, and they should remain clearly separated.

### 3.1 Histogram Recovery

Target:

\[
\hat c_d \approx c_d.
\]

This is the pure mergeability check.

Expected result:

- exact count sketches recover the root histogram with zero error.

### 3.2 Document-Level LDA Quantity Recovery

Use the exact root histogram to recover a document-level LDA object, for example:

\[
\mu_d = \mathbb{E}[\pi_d \mid c_d, \Phi].
\]

Expected result:

- full-document inference and exact tree-count inference agree up to numerical tolerance.

### 3.3 Histogram-Based Utility Recovery

Choose a downstream utility \(u(c_d)\), for example:

- log-likelihood under fixed \((\pi,\Phi)\),
- posterior mean of a linear topic score,
- a simple known functional of \(\mu_d\).

Expected result:

- applying \(u\) to the exact tree root count vector matches applying \(u\) to the full document directly.

## 4. Compared Methods

The comparison ladder should be:

### 4.1 Full-document exact baseline

- input: full document histogram \(c_d\),
- output: exact reference answer for the target.

### 4.2 Exact tree count-merge baseline

- leaf sketch: exact count vector,
- merge: vector addition,
- root readout: same operator as the full-document baseline.

This should match the full-document baseline exactly.

### 4.3 Learned compressed sketch

- leaf encoder: \(E_\psi(x_{d,\ell}) \in \mathbb{R}^m\),
- merge operator: \(M_\eta\),
- root head: \(H_\omega\).

This is the real approximation problem:

\[
\hat y_d = H_\omega(s_{\mathrm{root}}), \qquad
s_{\mathrm{root}} = M_\eta(\cdots M_\eta(s_1, s_2), \dots).
\]

The key gap is performance relative to the exact tree count baseline, not relative to some unrelated heuristic.

For the first production-ready learned family, this should be specialized to an
explicitly additive count sketch:

\[
s_{d,\ell} = A c_{d,\ell}, \qquad
s_d = \sum_\ell s_{d,\ell} = A c_d, \qquad
\hat c_d = B s_d = B A c_d.
\]

This is cleaner than a generic neural merge network because it isolates the
only approximation axis we actually want to study first: compression of the
exact sufficient statistic.

The exactness conditions are then explicit:

- if \(m \ge V\), exact histogram recovery for arbitrary bag-of-words documents
  is in-model;
- more generally, if \(m \ge \mathrm{rank}(X_{\mathrm{train,nodes}})\), the
  sampled training-node histogram matrix can be reconstructed exactly.

The current learned implementation fits this additive sketch with truncated SVD
on the fixed training-node histogram matrix, so the structural compression
ceiling is transparent rather than entangled with optimization noise.

### 4.4 Learned full-document operator

There should also be a learned document-level operator that does not need to
respect tree mergeability:

\[
\hat \mu_d = R_\omega(c_d).
\]

This is the "easy" learned baseline: it receives the full exact histogram, so
it should recover the exact known-topic LDA document operator with enough data.

### 4.5 Deliberately lossy controls

Useful controls include:

- truncated count sketches,
- random projections of counts,
- hard one-topic-per-leaf compression.

These are useful only as lossy baselines, not as the definition of ordinary-LDA recovery.

## 5. Metrics

Primary metrics:

- root count error: \( \lVert \hat c_d - c_d \rVert_1 \) or \( \lVert \hat c_d - c_d \rVert_2 \),
- root utility error: \( |\hat y_d - y_d| \),
- posterior error: \( \lVert \hat \mu_d - \mu_d \rVert_1 \),
- distance to exact-tree ceiling.

Secondary metrics:

- merge consistency error on internal nodes,
- schedule spread across balanced / left-to-right / right-to-left merges,
- oracle-query cost if supervision is budgeted,
- sketch size in bits / floats.
- training-node rank and whether the chosen sketch dimension is large enough to
  represent the sampled node family exactly.

For fixed linear utilities \(u_w(c)=w^\top c\), count recovery already gives a
deterministic utility bound:

\[
|u_w(\hat c_d) - u_w(c_d)|
\le \|w\|_\infty \, \|\hat c_d - c_d\|_1.
\]

So the count-recovery family strictly subsumes that direct-utility case.

The main figure should always preserve the hierarchy:

- full-document exact,
- exact tree,
- full-document learned operator,
- learned compressed,
- lossy controls.

## 6. Fixed-World Requirement

All comparisons should use fixed sampled worlds.

That means:

- same sampled topics \(\Phi\),
- same train/test documents,
- same held-out evaluation set,
- nested train prefixes when increasing train size.

This is already the right direction in the current fixed-world Segment-LDA machinery and should be the default here too.

## 7. Lean Linkage

This family already has the correct exact base theorem:

- [BagOfWordsLDARecovery.lean](/home/mlinegar/ThinkingTrees/lean3/FormalProofs/OPT/BagOfWordsLDARecovery.lean)

The current exact claims are:

- count sketches merge to the full root histogram,
- any histogram-based utility is preserved exactly,
- the bag-of-words LDA document likelihood is preserved exactly.

That is the formal base case for the simulation family.

## 8. Immediate Implementation Path

The next implementation steps should be:

1. build a small exact bag-of-words count-merge simulation family,
2. report full-document vs exact-tree equality directly,
3. add a learned document-level operator on full histograms,
4. add an additive learned count sketch over leaf histograms,
5. sweep fixed worlds over sketch dimension and train size,
6. only then add the local-mixture extension.

The first four of those steps are now implemented by the exact and learned
LDA tree-recovery runners above.

## 9. Local-Mixture Extension

After the exact base family is working, the next extension should be:

\[
\pi_d \sim \mathrm{Dir}(\alpha), \qquad
\pi_{d,\ell} \mid \pi_d \sim \mathrm{Dir}(\tau \pi_d),
\]

with leaf-level utility

\[
y_d^\star = \sum_\ell n_{d,\ell} h(\pi_{d,\ell}).
\]

This is the first setting where pooled full-document counts stop being enough for the target.

That is where "leaves matter" should enter the story.

## 10. What This Family Should Not Try To Do

This family should not initially try to:

- prove that hard one-topic-per-leaf inference recovers ordinary LDA,
- make boundary-drop controls the central story,
- mix plain-LDA recovery and hard segmented topics into one uninterpretable figure.

Those are separate questions. The base family should stay clean.
