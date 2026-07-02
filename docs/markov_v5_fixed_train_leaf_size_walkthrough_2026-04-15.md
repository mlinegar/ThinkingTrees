# Markov V5 Fixed-Train Leaf-Size Walkthrough

This note is a detailed walkthrough of the fixed-train Markov supervision-recovery
plots currently rendered under:

- `outputs/markov_v5_simple_current_plots_20260415_205235`

It is meant to explain four things clearly:

1. the data-generating processes behind the plots
2. the supervision-budget setup and what each panel means
3. how to read the current figures and what they actually show
4. how the generated report files relate to the underlying simulation outputs

The two timestamped plot bundles mentioned in the task,
`markov_v5_simple_current_plots_20260415_204015` and
`markov_v5_simple_current_plots_20260415_205235`, are materially the same for
our purposes. The later bundle regenerates the same views with updated paths and
is the better canonical target for this writeup.

## Artifact Provenance

The current plot bundle is a rendered view over a merged supervision-recovery
summary:

- Plot bundle:
  [`outputs/markov_v5_simple_current_plots_20260415_205235`](../outputs/markov_v5_simple_current_plots_20260415_205235)
- Plot report:
  [`report.md`](../outputs/markov_v5_simple_current_plots_20260415_205235/report.md)
- Machine-readable plot summary:
  [`summary.json`](../outputs/markov_v5_simple_current_plots_20260415_205235/summary.json)
- Upstream merged source:
  [`merged_current_report_summary.json`](../outputs/markov_v5_simple_current_plots_20260415_205235/merged_current_report_summary.json)
- Coverage check:
  [`merged_current_report_coverage.json`](../outputs/markov_v5_simple_current_plots_20260415_205235/merged_current_report_coverage.json)

The merged source summary says:

- `output_root = outputs/markov_v5_simple_fixed10240_quick_20260414_utc`
- `overlay_output_roots = [outputs/markov_v5_simple_leaf128_countonly_repair_20260415_194637]`
- `payload_count = 239`
- `source_job_keys = combined_scheduler_run, combined_scheduler_full_grid_fill, oneleaf_root_budget_fixed10240_simple, oneleaf_local_law_fixed10240_simple`

That matters because this figure bundle is not from one monolithic job. It is a
composed view assembled from:

- the main fixed-10240 quick sweep
- a full-grid fill pass
- a preserved one-leaf root-budget run
- a preserved one-leaf richer local-law run
- a later leaf128 count-only repair overlay

The repair overlay is especially relevant for the blue leaf-mass-equivalent
series, because it filled the missing `leaf128` count-only rows for root shares
`R90` through `R10`.

## The Two DGPs

The figures cover two closely related Markov changepoint-count benchmarks.
Both use the same document length and the same task, but they differ sharply in
state complexity and boundary density.

### Common task

For both DGPs:

- each document has exactly `128` tokens
- each token belongs to a hidden regime
- the document label is the number of regime changes across the full document
- equivalently, the root target is the total number of adjacent token positions
  where the regime flips

This is the quantity implemented by the oracle count in
`src/ctreepo/sim/core/markov_changepoint_ops_count.py`:

- `_oracle_count(doc, start=0, end=n_tok)` counts regime flips on the full span

The generator used here is the `hazard_topic` profile:

- at each token step, the regime either stays the same or switches to a
  uniformly random different regime
- emissions come from regime-specific token distributions
- in these two benchmarks, the emission palettes are disjoint at the regime
  level, so the observation problem is identifiable from tokens

That last point is important. These figures are not mainly about whether the
task is statistically identifiable in the limit. They are about whether the
model family and supervision geometry can learn to exploit that structure under
finite labels and finite tree geometry.

### Benchmark summary

| Case | Canonical key | Legacy alias | Regimes | Vocabulary | Hazard switch prob | Expected changepoints | Interpretation |
| --- | --- | --- | ---: | ---: | ---: | ---: | --- |
| Simple | `recoverable_v5_t128` | none | 4 | 16 | `5 / 127 = 0.03937` | about 5 | Recoverable, lower-complexity target |
| Hard | `structural_core_v2_t128::r12_p079` | `r12_seg10to12` | 12 | 48 | `10 / 127 = 0.07874` | about 10 | Higher-switch, many-regime structural stress case |

The benchmark metadata comes from
`src/ctreepo/sim/core/full_doc_anchor_diagnostics.py`.

### Simple case: `recoverable_v5_t128`

This is the easier benchmark:

- `4` hidden regimes
- `16` observed tokens total
- effectively `4` tokens per regime
- per-token switch probability `0.03937`
- about `5` expected regime changes per 128-token document

Because regimes have disjoint palettes, each observed token strongly identifies
which regime emitted it. The learning problem is then mostly about preserving
the count signal through the supervision and tree geometry rather than
discovering a highly entangled latent representation.

### Hard case: `r12_p079`

This is the harder benchmark and should be treated as the "hard" case for this
writeup:

- canonical key: `structural_core_v2_t128::r12_p079`
- legacy alias: `r12_seg10to12`
- `12` hidden regimes
- `48` observed tokens total
- still `4` observed tokens per regime
- per-token switch probability `0.07874`
- about `10` expected regime changes per 128-token document

Why it is harder:

- the same 128-token document must encode `3x` as many regime identities
- it also carries about `2x` the boundary density
- tree leaves now need to preserve more identity information and more boundary
  events per unit span
- one-leaf or coarse-leaf compression throws away useful local structure much
  faster than in the simple case

So the hard case is not "hard" because the task becomes fundamentally
non-identifiable. It is hard because the supervision and model geometry need to
carry more structured information in the same token budget.

## Experimental Setup

These figures fix the training set size and vary two things:

- the fraction of training documents that receive root labels
- the leaf size used by the tree model

### Fixed training set

Every panel uses:

- `10,240` training documents
- a fixed evaluation split shared across the compared methods

The root-share sweep is:

- `R100 = 10,240` root-labeled docs
- `R90 = 9,216`
- `R80 = 8,192`
- `R70 = 7,168`
- `R60 = 6,144`
- `R50 = 5,120`
- `R40 = 4,096`
- `R30 = 3,072`
- `R20 = 2,048`
- `R10 = 1,024`

### Leaf-size sweep

The x-axis leaf geometries are:

| Leaf tokens | Leaves per 128-token document |
| ---: | ---: |
| 128 | 1 |
| 64 | 2 |
| 32 | 4 |
| 16 | 8 |
| 8 | 16 |

So moving right on the x-axis means using smaller leaves and more internal
structure.

### Supervision variants

The current bundle renders two main comparison families for each DGP:

1. root-only supervision
2. root-only plus equal-total-mass leaf labels

The plotting script also supports a deeper `depth_equal_mass_eq` variant, but it
is not rendered in this bundle.

### What "equal-total-mass" means

This is the most important setup detail after the DGP itself.

The blue series does **not** mean "same number of labels" as the green series.
It means:

- keep the same `10,240` training documents
- keep the same root-label budget as the panel
- reallocate the *missing* root supervision mass to local leaf labels
- choose enough local leaf labels so that total effective supervision mass
  matches the `full100` budget

That logic is implemented through
`build_budgeted_train_supervision_manifest(...)` in
`src/ctreepo/sim/core/markov_changepoint_ops_count.py`, where:

- a full-document label contributes mass `1.0`
- a leaf label contributes span mass equal to
  `leaf_span_tokens / doc_tokens`
- mass is tracked at the document level as
  `effective_full_doc_mass_per_doc`

So a missing root label can be replaced by multiple smaller leaf labels. This is
a mass-equated comparison, not a call-count-equated comparison.

### Why `R100` blue and green coincide

At `R100`, there is no missing root supervision mass to reallocate. The coverage
JSON correctly shows no real leaf-mass-equivalent rows at `R100`, and the plot
script intentionally reuses the root-only curve there so the series coincide by
construction.

## How To Read The Figures

The current bundle contains four figures:

### Recoverable, root-only

![Recoverable root-only](../outputs/markov_v5_simple_current_plots_20260415_205235/figures/recoverable_root_only_leaf_size_fixed_train10240.png)

### Recoverable, equal-total-mass leaf labels

![Recoverable leaf-mass-equivalent](../outputs/markov_v5_simple_current_plots_20260415_205235/figures/recoverable_root_only_leaf_size_fixed_with_leaf_mass_equivalent_train10240.png)

### Hard case, root-only

![Structural hard root-only](../outputs/markov_v5_simple_current_plots_20260415_205235/figures/structural_root_only_leaf_size_fixed_train10240.png)

### Hard case, equal-total-mass leaf labels

![Structural hard leaf-mass-equivalent](../outputs/markov_v5_simple_current_plots_20260415_205235/figures/structural_root_only_leaf_size_fixed_with_leaf_mass_equivalent_train10240.png)

The legend items mean:

- green solid line: best available converged root-only tree surface
- blue dashed line with squares: same root-label budget, but missing supervision
  mass reallocated to count-only leaf labels
- hollow diamond at `leaf128`: one-leaf tree without local laws
- amber dotted horizontal line: official FNO trained on the same training set
  and the same root-label budget as the panel
- red `X` at `leaf128`: the actual official FNO point paired with the leaf128
  no-local-law tree run
- dark teal dashed line: empirical-Bayes lower bound with the DGP family known
- gray triangle: richer one-leaf duplicate-local-label diagnostic

In the current figures, the dark teal empirical-Bayes line sits at essentially
zero MAE. That is not because the learned problem is trivial. It is because the
benchmark uses disjoint token palettes, so under DGP-known inference the regime
identity is effectively exposed by the observed token identity. That makes this
line a lower-bound reference, not a fair learned baseline.

Two reading rules matter:

1. The amber FNO baseline is horizontal because FNO does not depend on leaf
   size.
2. The hollow diamond and the red `X` should coincide when the one-leaf tree is
   in clean parity with the corresponding FNO setup.

In these figures they do coincide. That is a useful sanity check:

- at `leaf128`, the tree without local laws is behaving like the intended
  one-leaf parity canary
- the gains at smaller leaf sizes therefore reflect real benefits from the tree
  geometry rather than a bookkeeping mismatch against FNO

## Result Walkthrough: Recoverable Simple Case

### Quantitative summary

| Root share | Root-labeled docs | Best root-only leaf | Best root-only MAE | FNO MAE | Best leaf-mass-eq leaf | Best leaf-mass-eq MAE | One-leaf richer local-label check |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| R100 | 10,240 | 8 | 0.198 | 0.496 | 8 | 0.198 | 0.255 |
| R90 | 9,216 | 8 | 0.198 | 0.523 | 8 | 0.196 | 0.273 |
| R80 | 8,192 | 8 | 0.199 | 0.508 | 8 | 0.221 | 0.255 |
| R70 | 7,168 | 8 | 0.205 | 0.672 | 8 | 0.209 | 0.268 |
| R60 | 6,144 | 8 | 0.219 | 0.730 | 8 | 0.228 | 0.367 |
| R50 | 5,120 | 8 | 0.251 | 0.707 | 8 | 0.252 | 0.267 |
| R40 | 4,096 | 8 | 0.277 | 0.867 | 64 | 0.312 | 0.381 |
| R30 | 3,072 | 64 | 0.330 | 1.000 | 64 | 0.314 | 0.373 |
| R20 | 2,048 | 64 | 0.342 | 1.098 | 64 | 0.354 | 0.268 |
| R10 | 1,024 | 64 | 0.469 | 1.227 | 64 | 0.428 | 0.277 |

### Main pattern

The simple-case picture is strong and stable:

- the best root-only tree beats the same-budget FNO baseline at every root share
- the best root-only geometry is usually the smallest leaf size, `leaf8`
- only once the root budget becomes very small, at `R30` through `R10`, does
  the best leaf size shift back toward `leaf64`

That is a clean "tree geometry helps" story. The one-leaf parity check already
matches FNO at `leaf128`, and then moving to a real tree with multiple leaves
improves MAE substantially.

### What happens as root labels get scarce

As the root-label budget drops:

- the `leaf128` parity point worsens steadily with FNO
- the multi-leaf tree remains much more stable
- the best root-only tree goes from about `0.198` MAE at `R100` to `0.469` at
  `R10`
- the matched FNO baseline goes from about `0.496` to `1.227`

So even at the lowest root-label budget in this sweep, the best root-only tree
is still much better than the same-budget FNO baseline.

### Why the best leaf size shifts at low budget

In the upper and middle root-budget panels, the smallest leaves win:

- `leaf8` is best from `R100` through `R40`

At low root supervision:

- `leaf64` becomes best at `R30`, `R20`, and `R10`

That likely reflects the bias-variance tradeoff of the tree geometry:

- smaller leaves preserve more local structure
- but they also require the model to coordinate more local predictions and more
  merges
- once root supervision is extremely sparse, a slightly coarser leaf geometry
  can become easier to fit reliably

### What the blue leaf-mass-equivalent series adds

In the simple case, reallocating missing supervision mass to count-only leaf
labels is mostly a second-order effect rather than the main story:

- it helps slightly at `R90`
- it hurts slightly or is nearly neutral for most middle-budget panels
- it helps visibly at `R30`
- it helps most clearly at `R10`, where best MAE improves from `0.469` to
  `0.428`

So the simple-case result is **not** "the tree only works when extra local
labels are added." The root-only tree already wins. The blue series mostly acts
as a mild refinement at low root budgets.

### Important diagnostic: leaf128 duplicate count-only labels

The blue square at `leaf128` is the one-leaf duplicate-count target. It is not a
real tree advantage because there is only one leaf. It is a bookkeeping
diagnostic: what happens if we keep the same one-leaf geometry and add a local
count target on that same whole-document leaf?

In the simple case this is often very poor:

- `R50`: blue `leaf128` is `1.797`
- `R30`: blue `leaf128` is `1.797`
- `R10`: blue `leaf128` is `1.797`

That is much worse than both the green root-only tree and the amber FNO line.
So simply duplicating the document-level count as a local target on a single
leaf is not what explains the multi-leaf improvements.

The gains come from using real tree structure, not from blindly duplicating the
root label.

### The gray triangle diagnostic

The gray triangle is the richer one-leaf duplicate-local-label check. It is
usually much stronger than the bad blue `leaf128` duplicate-count-only square,
and at the lowest root budgets it can even beat the best root-only tree:

- `R20`: gray triangle `0.268` vs best green `0.342`
- `R10`: gray triangle `0.277` vs best green `0.469`

This should **not** be read as a practical baseline. It is a ceiling-style
diagnostic showing that if one provides richer decomposed local side targets on
the one-leaf case, the model can do much better. That tells us the hard part is
not pure capacity; it is learning the right local decomposition under the
available supervision contract.

## Result Walkthrough: Structural Hard Case

### Quantitative summary

| Root share | Root-labeled docs | Best root-only leaf | Best root-only MAE | FNO MAE | Best leaf-mass-eq leaf | Best leaf-mass-eq MAE | One-leaf richer local-label check |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| R100 | 10,240 | 16 | 0.829 | 1.254 | 16 | 0.829 | 1.076 |
| R90 | 9,216 | 32 | 0.850 | 1.410 | 16 | 0.858 | 0.975 |
| R80 | 8,192 | 16 | 0.874 | 1.332 | 8 | 0.881 | 1.018 |
| R70 | 7,168 | 8 | 0.873 | 1.574 | 16 | 0.869 | 0.971 |
| R60 | 6,144 | 8 | 0.940 | 1.652 | 8 | 0.933 | 1.207 |
| R50 | 5,120 | 16 | 0.942 | 1.762 | 8 | 0.929 | 0.933 |
| R40 | 4,096 | 8 | 0.936 | 2.027 | 8 | 0.920 | 1.139 |
| R30 | 3,072 | 8 | 0.983 | 2.227 | 8 | 0.975 | 1.173 |
| R20 | 2,048 | 8 | 1.076 | 2.418 | 8 | 1.097 | 0.942 |
| R10 | 1,024 | 64 | 1.212 | 2.426 | 64 | 1.232 | 0.947 |

### Main pattern

The hard case keeps the same qualitative ordering but compresses the gains:

- the best root-only tree still beats FNO at every root share
- the gap over FNO is still large
- but absolute MAE is much higher than in the simple case
- the best values now sit roughly in the `0.83` to `1.21` range instead of the
  `0.20` to `0.47` range

So the structural hard case does not break the tree family, but it does make the
task materially harder.

### Why the hard case looks different

Compared with the simple case:

- the hard case no longer strongly favors the very smallest leaves in every
  panel
- the best leaf size moves among `32`, `16`, `8`, and then back to `64` at
  `R10`
- performance improvements from reducing leaf size are smaller and more uneven

This is what we should expect when the document contains many more regime
identities and many more boundaries. Very fine leaves preserve local detail, but
they also create more local prediction and merge burden. The optimum is less
extreme and less stable than in the simple case.

### What the blue leaf-mass-equivalent series adds in the hard case

The blue series matters a bit more here than in the simple case, but it is still
not the whole story.

Helpful panels:

- `R70`: `0.873 -> 0.869`
- `R60`: `0.940 -> 0.933`
- `R50`: `0.942 -> 0.929`
- `R40`: `0.936 -> 0.920`
- `R30`: `0.983 -> 0.975`

Non-helpful panels:

- `R90` and `R80` are slightly worse
- `R20` and `R10` are also slightly worse

So in the hard case, count-only local mass reallocation helps most in the middle
root-budget range, but it does not fundamentally change the frontier. The main
story remains that the tree geometry beats FNO and that the hard DGP preserves a
clear difficulty gap relative to the simple benchmark.

### One-leaf diagnostics in the hard case

The same diagnostic pattern appears again:

- the `leaf128` parity point matches FNO
- the blue `leaf128` duplicate-count-only square is bad
- the gray richer-local-label triangle can become meaningfully better than the
  best root-only tree at low budgets

For example:

- `R20`: gray triangle `0.942` vs best green `1.076`
- `R10`: gray triangle `0.947` vs best green `1.212`

That again suggests that some of the remaining gap is about the supervision
factorization, not raw representational impossibility.

### The real hard-case conclusion

The correct conclusion is not "the hard case fails." The correct conclusion is:

- parity at one leaf still looks right
- real multi-leaf tree structure still helps over FNO
- but the structural hard case has a genuine residual difficulty that the
  current root-only setup does not remove

That is exactly the kind of case we should include in the writeup, because it
shows where the method still has headroom.

## Cross-Figure Interpretation

Taken together, the four figures support the following claims.

### 1. The leaf128 parity canary is behaving correctly

Across both DGPs and all root shares:

- the no-local-law one-leaf tree point aligns with the official FNO point

That means the comparison is not being driven by a broken parity setup at the
one-leaf limit.

### 2. The gains are real geometry gains

The big improvements happen when we move from:

- one leaf over the whole document

to:

- multiple leaves and learned tree aggregation

This is strongest in the simple case but still present in the hard case.

### 3. Equal-total-mass local labels are not the main explanation

The blue series is not the reason the tree wins:

- root-only green already wins against FNO
- blue sometimes helps, sometimes barely matters, and sometimes hurts

So the key effect is not "extra local supervision rescues a weak tree." The
tree family is already competitive under root-only supervision.

### 4. Count-only duplicate labels on one leaf are not a good substitute

The poor blue `leaf128` squares show that naive local duplication of the root
count does not explain the benefit of tree structure.

### 5. Richer local decomposition is still informative

The gray triangle diagnostics show that richer local targets could still unlock
better performance, especially in low-root-budget panels and especially in the
hard case.

## How To Read The Generated Report Files

There are four distinct report layers here, and they do different jobs.

### 1. `report.md`

This is the human-readable artifact index.

It gives:

- a timestamp
- the source summary path
- the fixed train-doc count
- the rendered figures
- one bullet block per root share

What it does well:

- quickly shows the four rendered figures
- exposes the figure paths and per-panel availability
- gives one place to click through from the output directory

What it does not do well:

- it is not interpretive
- its `sources:` lines are blank because the merged rows do not currently carry
  useful `source_lineage_label` strings into this view
- it does not explain why the blue series overlaps at `R100`
- it does not explain the parity meaning of the diamond/X coincidence
- it does not explain why the gray triangle is only a diagnostic

So `report.md` is best thought of as a render manifest, not the final narrative.

### 2. `summary.json`

This is the best machine-readable entry point for the current figure bundle.

It contains:

- figure paths
- DGP subtitles
- per-panel leaf-size MAEs
- per-panel FNO values
- secondary series values
- one-leaf diagnostic values

If another script or note needs to reason about these figures, `summary.json`
should usually be the first file to load.

### 3. `merged_current_report_summary.json`

This is the upstream merged source from which the plot summary was derived.

It tells us:

- which output root the bundle came from
- which overlay roots were applied
- which job keys contributed rows
- the raw merged supervision-recovery payload collection

This is the key provenance layer for understanding how the current figure bundle
was assembled.

### 4. `merged_current_report_coverage.json`

This is the coverage audit.

It is especially useful for catching subtleties such as:

- leaf-mass-equivalent rows are absent at `R100`
- all intended root shares and leaf sizes for the main plotted surfaces are now
  present after the repair overlay

Without the coverage JSON, it is easy to misread the `R100` blue curve as a real
separately trained series rather than a by-construction overlap.

## Bottom-Line Reading

If we need a concise scientific reading of this bundle, it is:

- the one-leaf tree/FNO parity canary looks correct
- once we allow real tree structure, the tree family beats the same-budget FNO
  baseline across the entire fixed-10240 root-share sweep
- the simple case shows especially strong gains from smaller leaves
- the hard case still favors the tree, but with a real residual difficulty gap
- equal-total-mass count-only leaf labels sometimes help, but they are not the
  main reason the tree wins
- richer local target factorization remains a meaningful ceiling, especially at
  low root budgets and especially in the hard case

That is why the hard case belongs in the writeup: it prevents the note from
overselling the simple benchmark and makes clear where the current recipe still
has room to improve.
