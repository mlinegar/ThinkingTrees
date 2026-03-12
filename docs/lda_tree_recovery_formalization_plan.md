# LDA Tree-Recovery Plan

This note fixes the clean story we should build around the LDA simulations and their proof obligations.

The organizing principle is:

1. start with the exact bag-of-words LDA case where a tree should lose nothing,
2. make the tree exact first by merging full count sketches,
3. only then ask how far we can compress those sketches with learned/neural operators,
4. only after that introduce genuinely local latent structure where leaves matter statistically.

## 1. Base Case: Exact Bag-of-Words LDA

Document-level latent structure:

\[
\pi_d \sim \mathrm{Dir}(\alpha), \qquad
z_{d,t} \mid \pi_d \sim \mathrm{Cat}(\pi_d), \qquad
x_{d,t} \mid z_{d,t}=k \sim \mathrm{Cat}(\phi_k).
\]

In this model the document is conditionally exchangeable given \(\pi_d\) and \(\Phi\). The sufficient statistic for the document-level likelihood is the word histogram

\[
c_d(v) = \sum_t \mathbf{1}\{x_{d,t}=v\}.
\]

That gives the first exact tree theorem:

- a leaf sketch that stores exact word counts is mergeable,
- the merge rule is just histogram addition,
- the root histogram equals the full-document histogram regardless of tree shape,
- any downstream operator that depends only on the histogram can be evaluated exactly at the root.

This is the right "recover ordinary LDA" claim.

What it does not claim:

- leaves are statistically useful under plain LDA,
- per-leaf posterior inference is exact,
- boundary corrections matter.

Under ordinary bag-of-words LDA, none of those are the main point.

## 2. Exact Tree Recovery Benchmark

The exact benchmark should have two equivalent views.

Full-document operator:

- input: the full histogram \(c_d\),
- output: either an LDA document-level quantity or a downstream utility \(u(c_d)\).

Exact tree operator:

- leaf sketch: \(s_{d,\ell} = c_{d,\ell}\),
- merge: \(m(s_L, s_R) = s_L + s_R\),
- root output: apply the same operator to the merged root count vector.

Expected result:

- the full-document and tree paths match exactly up to optimization / numerical noise,
- this is a positive control showing the tree architecture itself is not the bottleneck.

Recommended metrics:

- root histogram error,
- posterior-mean mixture error if we fit/infer \(\hat \pi_d\),
- downstream utility error for a histogram-based target.

## 3. Learned Mergeable Sketches

Once the exact count benchmark is in place, the next question is compression rather than correctness.

Replace exact counts by

\[
s_{d,\ell} = E_\psi(x_{d,\ell}), \qquad
s_{A \cup B} = M_\eta(s_A, s_B), \qquad
\hat y_d = H_\omega(s_{\mathrm{root}}).
\]

The empirical question is then:

- how small can the sketch dimension be while still matching the exact count-merge baseline?

The theoretical question is already supported by the existing OPT stack:

- exact local laws give zero distortion,
- approximate local laws give quantitative distortion bounds,
- learned sketches should be evaluated as approximate versions of the exact count sketch.

This is where "neural operators over mergeable sketches" belongs.

## 4. Why Leaves Do Not Yet Matter Under Plain LDA

If the target depends only on the full document histogram, then leaves are only an implementation device.

They matter only because we choose to:

- partition the document for computation,
- compress locally before merging,
- or supervise at leaf/internal nodes rather than at the root.

That is why the plain-LDA page should read as:

- exact count-merge tree recovery is possible,
- learned local compression may hurt,
- smaller leaves may help or hurt only through estimation/compression tradeoffs,
- not because the plain-LDA data-generating process itself has meaningful boundaries.

## 5. First Genuine Extension: Correlated Local Mixtures

The first extension where leaves matter mathematically is not a correlated prior on one document-level mixture.

It is a model with local leaf or segment mixtures:

\[
\pi_d \sim \mathrm{Dir}(\alpha), \qquad
\pi_{d,\ell} \mid \pi_d \sim \mathrm{Dir}(\tau \pi_d),
\]

and then

\[
z_{d,t} \mid \pi_{d,\ell(t)} \sim \mathrm{Cat}(\pi_{d,\ell(t)}).
\]

Now local leaves differ, but remain correlated through the document-level \(\pi_d\).

That makes leaves statistically meaningful without jumping immediately to hard one-topic segments.

Recommended target:

\[
y_d^\star = \sum_\ell n_{d,\ell} h(\pi_{d,\ell}),
\]

with a nonlinear \(h\), for example

\[
h(\pi) = \theta^\top \pi + \lambda \pi^\top W \pi.
\]

Then pooled full-document counts are generally not enough, because

\[
\sum_\ell \omega_\ell h(\pi_{d,\ell}) \neq h\!\left(\sum_\ell \omega_\ell \pi_{d,\ell}\right).
\]

This is the right middle layer between ordinary LDA and hard Segment-LDA.

## 6. Simulation Ladder

The production simulation family should be ordered as follows.

1. Exact bag-of-words LDA recovery from full-document counts.
2. Exact bag-of-words LDA recovery through mergeable tree counts.
3. Learned compressed sketches matched against the exact count benchmark.
4. Correlated local-mixture LDA where leaf-aware recovery becomes statistically useful.
5. Hard Segment-LDA / transition-sensitive targets as the strongest structured case.

Each level should reduce cleanly to the one above when its extra structure is turned off.

## 7. Lean Formalization Ladder

The proof ladder should mirror the simulation ladder.

Already in the repo:

- `FormalProofs.OPT.BigramSketch`
- `FormalProofs.OPT.TopicBigramOracle`
- `FormalProofs.OPT.SegmentLDAPipelineToy`
- `FormalProofs.OPT.ApproximateLocalLaws`

New exact base-case file:

- `FormalProofs.OPT.BagOfWordsLDARecovery`

Its role is to prove:

1. histogram sketches merge exactly under tree reduction,
2. any histogram-based document utility is preserved exactly,
3. bag-of-words LDA document likelihood is therefore preserved exactly.

Planned next Lean file after that:

- a local-mixture bridge showing when pooled document counts cease to be sufficient for the target.

## 8. Figure Plan

The paper-level figure ladder should match the formal one.

1. Full-document ordinary LDA baseline.
2. Exact tree count-merge baseline on the same worlds.
3. Learned sketch approximation curves against the exact tree baseline.
4. Local-mixture extension showing when leaf-aware modeling actually matters.

That gives a coherent story:

- first prove we can reproduce LDA exactly,
- then show where compression costs appear,
- then show where locality becomes genuinely useful.
