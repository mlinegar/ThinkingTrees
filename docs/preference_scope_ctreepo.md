# Preference Scope for C-TreePO

Canonical statement:

> C-TreePO supports preferences whose decision-relevant information can be
> represented by a locally composable state. Leaf, merge, and re-summary laws
> certify that the tree stays in the same task-relevant fiber; any preference,
> loss, or utility that factors through that fiber is preserved. The utility
> need not be additive over leaves.

Lean entry points:

- `lean3/FormalProofs/OPT/PreferenceScope.lean`
- `lean3/FormalProofs/OPT/MergeablePreference.lean`
- `lean3/FormalProofs/OPT/MergeableReduction.lean`
- `lean3/FormalProofs/OPT/ClassicalSketchLocalLaws.lean`

## Notation

Use these names when stating the problem:

- `x, y : X`: raw documents or spans, with concatenation `x ++ y`.
- `f* : X -> Y`: the target preference, utility, loss-relevant oracle, or
  decision-relevant readout.
- `g : X -> X`: the learned or hand-written summarizer/reduction operator.
- `sigma : X -> S`: the global task state, meaning the semantic information
  needed by the preference.
- `F : S -> Y`: the downstream readout, so `f* = F ∘ sigma`.
- `tau : X -> L`: a local/sketch state that may be richer than `sigma`.
- `encode : X -> L`, `mergeL : L -> L -> L`, `decode : L -> S`: the local
  state implementation used by a mergeable sketch or tree reducer.

## What "Fiber" Means

A **fiber** is the preimage of one value under a map. The detailed reference
version is in `docs/fiber_definitions_ctreepo.md`; the short version is below.

For any map `m : X -> Y`, the fiber at value `y` is:

```text
{ x : X | m(x) = y }
```

The same-fiber relation is:

```text
x ~_m x'  iff  m(x) = m(x')
```

For a task state map `sigma : X -> S`, the `sigma`-fiber at state value `s` is:

```text
{ x : X | sigma(x) = s }
```

So two documents are in the same `sigma`-fiber when:

```text
sigma(x) = sigma(x')
```

For a non-math audience, read "same fiber" as "same task-relevant bucket." The
documents may differ in many raw details, but those details are invisible to
the chosen state. A preference `f*` factors through `sigma` exactly when it gives
the same answer to every pair of documents in the same `sigma`-fiber:

```text
sigma(x) = sigma(x')  ==>  f*(x) = f*(x')
```

Equivalently, the partition induced by `sigma` refines the partition induced by
`f*`: every state fiber sits inside a preference fiber. This is the precise
meaning of "the decision-relevant information is represented by the state."

Example: if `sigma` is a bag-of-words histogram, then word order is within-fiber
variation. Any preference that depends only on the histogram is constant on
those fibers. A preference that depends on a cross-boundary bigram is not
constant on histogram fibers, because two documents can have the same histogram
but different adjacent-token structure.

In a political-text example, a fiber could be "all manifestos with the same
RILE-relevant evidence state." The texts may use different rhetoric, order, or
stylistic flourishes, but if the chosen state records exactly the policy
evidence used by the score, those differences are within-fiber variation. If the
preference cares about a rhetorical pattern not recorded in the state, then that
preference does not factor through this state.

Lean anchors:

- `MapFiber`: the general preimage/fiber predicate `m(x) = y`.
- `SameMapFiber`: the general relation `m(x) = m(x')`.
- `sameMapFiber_equivalence`: same-fiber is an equivalence relation.
- `sameMapFiber_iff_exists_common_value`: same-fiber means membership in one
  common named fiber.
- `StateFiber`: the set-like predicate `sigma(x) = s`.
- `SameStateFiber`: the relation `sigma(x) = sigma(x')`.
- `PreferenceFiber` and `SamePreferenceFiber`: the analogous target/preference
  fibers.
- `StateFibersRefinePreferenceFibers`: the exact partition-refinement condition
  `sigma(x) = sigma(x') -> f*(x) = f*(x')`.
- `ReadoutRespectsStateFibers`: a readout is constant on same-state pairs.
- `preferenceFactorsThroughState_iff_stateFibersRefinePreferenceFibers`: for
  inhabited readout types, "factors through state" is equivalent to "state
  fibers refine preference fibers."
- `stateFiber_subset_preferenceFiber_of_readout`: a named state fiber maps
  into a named preference fiber when `f*(x) = F(sigma(x))`.
- `sameStateFiber_implies_samePreferenceFiber_of_stateReadout`: same-state
  pairs are same-preference pairs under an explicit state readout.
- `summaryPreservesState_iff_sameStateFiber`: `g` preserves state exactly when
  `g x` stays in the same `sigma`-fiber as `x`.
- `OracleValueFiber`, `SameOracleValueFiber`, and `SameOracleFiber`: exact
  value fibers and the metric zero-distance oracle fibers used by the theorem
  stack.
- `sameOracleFiber_iff_sameOracleValueFiber`: for metric theorem oracles, those
  two oracle-fiber definitions coincide.
- `sameStateFiber_of_sameOracleFiber`: if an oracle identifies `sigma`, then a
  same-oracle-fiber pair is also a same-state-fiber pair.

Reference anchors:

- `Fisher1922` and `Blackwell1953`: sufficient-statistic and
  decision-theoretic sufficiency lineage.
- `Kallenberg2002FMP` / `doobdynkin-readable`: Doob-Dynkin factorization,
  i.e. "depends only through this state."
- `Agarwal2013MergeableTODS`: mergeable summaries as locally composable
  states.

## A Useful Warning from Futer 2013

David Futer's paper, "Fiber detection for state surfaces," is a good analogy
but not literally the same use of the word "fiber." The PDF is stored locally at
`docs/references/futer_2013_fiber_detection_state_surfaces.pdf`.

The bibliography entry is `Futer2013FiberDetection`; the paper appeared in
Algebraic & Geometric Topology 13(5):2799--2807, 2013, DOI
`10.2140/agt.2013.13.2799`.

In that paper, a **fiber** is a topological object: a state surface that is the
fiber surface of a fibration of a link complement over the circle. In this
document, a **state fiber** is an equivalence class: all inputs with the same
`sigma` value.

The shared pattern is the detector schema:

```text
hard global property of object
  <=> simple property of associated reduced/composable certificate
```

For Futer's theorem:

```text
K is fibered with fiber surface S_sigma
  <=> associated reduced graph G'_sigma is a tree
```

For C-TreePO:

```text
document x has target property f*(x)
  <=> state certificate sigma(x) has the corresponding property
```

The first is topological fiber detection. The second is state-fiber / sufficient
state detection. They are not the same theorem, but they support the same style
of explanation: choose the right associated object, then a hard global question
becomes a simple certificate-level question.

Lean anchors:

- `futer2013_theorem1_statement`: abstract statement of the homogeneous
  state-surface theorem.
- `futer2013_corollary2_A_statement` and
  `futer2013_corollary2_B_statement`: Jones-coefficient obstruction statements.
- `detector_problem` and `exact_detector`: generic detector schema.
- `futer2013_theorem1_yields_exact_detector`: Theorem 1 as exact detection by
  tree-ness of the reduced graph.
- `state_factored_detector_exact`: C-TreePO state-factored predicates as exact
  detection by `sigma(x)`.

## Shape of Capturable Preferences

The clean exact shape is:

```text
raw document x
  -> global task state sigma(x)
  -> downstream preference/readout F(sigma(x)) = f*(x)
```

At the preference level:

```text
f*(x) = F(sigma(x))
```

Equivalently, `f*` is constant on the fibers of `sigma`: if
`sigma(x) = sigma(x')`, then `f*(x) = f*(x')`. This is the most basic
preference-shape condition. It says nothing yet about whether the state can be
computed by a tree.

The tree/mergeability condition is separate:

```text
sigma(x ++ y) = merge(sigma(x), sigma(y))
```

If both lines hold, the preference may be nonlinear in the leaves, but it is a
readout of a decomposable state.

The Lean vocabulary is:

- `PreferenceReadoutOfState`: explicit shape `pref x = readout (state x)`.
- `MergeablePreferenceShape`: `sigma` is exactly composable and the preference
  factors through `sigma`.
- `MergeablePreferenceShape.readout_of_mergeFold`: the root preference can be
  read from the folded merge state.
- `StateDecomposesBy`: the global law
  `sigma(x ++ y) = merge(sigma(x), sigma(y))`.
- `SummaryPreservesState`: the `g`-law `sigma(g x) = sigma(x)`.
- `SummaryMergePreservesState`: the two-route `g`-law
  `sigma(g(g x ++ g y)) = sigma(x ++ y)`.
- `ExactComposableState`: leaf encoding plus binary merge recover the same
  state as direct evaluation on the concatenated span.
- `LocalStateRealizesGlobalState`: local state `tau`, `encode`, `mergeL`, and
  `decode` recover the same global state `sigma`.
- `ctreepo_supports_state_factored_preference`: exact C-TreePO local-law
  preservation transports state-factored objectives.
- `exact_mergeable_state_supports_any_downstream_utility`: arbitrary utilities
  on exact mergeable states are preserved.

This is a sufficient-class claim: the framework applies when we can name,
learn, or audit such a state.

## The f/g Statement

In `f*`/`g` terms, an exact certificate says:

```text
f*(x) = F(sigma(x))                      preference factors through state
sigma(g x) = sigma(x)                    summary preserves state
sigma(g(g x ++ g y)) = sigma(x ++ y)     two-route merge preserves state
```

The first line is about the **preference target**. The next two lines are about
the **operator** `g`.

When the theorem-facing oracle is the encoded state,
`f_state(x) := encodedOracle(sigma(x))`, the last two lines imply the usual
global assumptions:

```text
A1_global g f_state
A2_global g f_state
```

Lean anchors:

- `summaryPreservesState_implies_A1_encodedOracle`
- `summaryMergePreservesState_implies_A2_encodedOracle`
- `summaryMergePreservesState_of_preservesState_and_stateDecomposes`
- `summaryPreservesPreference_of_stateReadout`
- `summaryMergePreservesPreference_of_stateReadout`

This gives a compact way to explain why local C-TreePO laws are enough: the
local laws do not need to know the final utility form. They need to certify
that `g` keeps the computation inside the same `sigma`-fiber.

## Global State vs Local State

There are two different states worth naming.

The **global state** `sigma(x)` is the semantic target: the smallest or most
useful theorem-facing object from which `f*(x)` can be read.

The **local state** `tau(x)` is the merge-carried implementation. It may be
richer or differently typed:

```text
encode(x) = tau(x)
mergeL(tau(x), tau(y)) = tau(x ++ y)
decode(tau(x)) = sigma(x)
```

Then a tree fold satisfies:

```text
decode(mergeFold encode mergeL T) = sigma(S T)
f*(S T) = F(decode(mergeFold encode mergeL T))
```

Lean anchors:

- `LocalStateRealizesGlobalState`
- `LocalStateRealizesGlobalState.decode_mergeFold_eq_global`
- `GlobalLocalPreferenceShape`
- `GlobalLocalPreferenceShape.readout_of_local_mergeFold`

This is the mergeable-sketch version of the story. HLL registers, Count-Min
tables, endpoint-aware Markov states, and boundary-token states are local
states. The scalar estimate or decision is a readout after the merge.

## Relation to Additive Separability

Additive separability is a useful special case, not the general boundary.

In Lean:

- `AdditiveComposableState`: `sigma(x ++ y) = sigma(x) + sigma(y)`.
- `AdditiveStateReadout`: the readout itself respects `+`.
- `AdditivelySeparableThroughState`: utility factors through an additive state
  with an additive readout.
- `additivelySeparableThroughState_factorsThroughState`: additive separability
  implies state factorization.

So the inclusion is:

```text
additively separable utility
  ⊂ arbitrary readout of an additive merge state
  ⊂ arbitrary readout of a mergeable state
```

Examples:

- Sum of local scores: additive state and additive readout.
- Threshold/quorum over counts: additive count state, nonlinear readout.
- LDA or histogram scoring: additive histogram state, nonlinear or
  multiplicative readout.
- HLL distinct count: max-register merge state, scalar cardinality readout.
- Markov changepoints: ordered endpoint-aware state, boundary-sensitive readout.

The common point is that the state composes locally. The preference readout can
be nonlinear, discontinuous, thresholded, or interaction-heavy, provided the
state retained the variables needed by that readout.

The additive-separability intuition is therefore still useful, but it attaches
to the **state update**, not necessarily to the final utility. A classical
separable utility merges final values directly:

```text
f*(x ++ y) = f*(x) + f*(y)
```

C-TreePO and mergeable sketches allow the weaker and more useful shape:

```text
sigma(x ++ y) = merge(sigma(x), sigma(y))
f*(x) = F(sigma(x))
```

The first line is a local law for a sufficient statistic. The second line says
the preference lives on that statistic. Additive utility is the case where
`sigma = f*`, `merge = +`, and `F = id`.

## Mergeable Sketch Intuition

Classical mergeable sketches make the state/readout distinction concrete:

- `classical_state_level_mergeable_preference_shape`: merge sketch states first,
  then query/read out at the root.
- `additive_linear_sketch_preference_shape`: additive linear sketches are the
  additive-state case.
- `count_min_state_level_preference_shape`: Count-Min-style counter tables are
  additive state-level sketches.
- `hll_state_level_preference_shape`: HLL register arrays are mergeable by
  pointwise max, not addition.

The scalar answer is usually not the right merge state. For HLL, the distinct
count estimate is the readout; the mergeable state is the register array. For
Count-Min, point queries are readouts; the mergeable state is the table. For
boundary-sensitive text tasks, the root score is the readout; the mergeable
state must contain enough boundary/context information to compute it.

`MergeablePreference.lean` formalizes the narrower scalar-oracle route:

- `additive_scalar_preference_is_mergeable`: additive scalar utilities fit when
  final oracle values themselves merge by `+`.
- `scalar_oracle_concat_witness_not_expressible`: if equal child oracle values
  can lead to different parent oracle values, no single scalar merge operator
  can be well-defined.
- `scalar_threshold_and_not_expressible`: threshold-AND fails after child counts
  are collapsed to Boolean threshold values.
- `scalar_boundary_bigram_not_expressible`: boundary bigrams fail after leaf
  scalars forget boundary tokens.

These are not contradictions with the supported state-level examples. They say:
do not threshold, query, or score too early. Keep the mergeable state until the
root, then read out the preference.

## Supported Nonseparable Examples

- `supported_nonseparable_complementarity`: threshold-AND utility over exact
  left/right counts. This is not additive as a utility, but it is a readout of
  an additive count state.
- `supported_boundary_interaction`: topic unigram plus boundary-bigram state
  carries cross-boundary interaction terms.
- `supported_histogram_state_any_utility`: any utility on a recovered
  bag-of-words histogram is preserved, not only linear word weights.
- `supported_lda_likelihood_histogram_utility`: LDA likelihood is preserved
  because it factors through the histogram state.

These examples are the right way to talk about nonseparability: the utility may
not decompose over leaves, but the sufficient state does.

## Counterexample Classes

Wrong preference target:

- If a preference separates two documents with the same theorem state, it does
  not factor through that state.
- Lean anchor: `preference_not_factored_through_state`.

Wrong state:

- Scalar child distinct counts omit overlap information. The HLL register array
  works; child cardinalities alone do not.
- Lean anchor: `insufficient_scalar_distinct_count_state`.
- Markov count-only summaries omit endpoint information needed for arbitrary
  tree topologies. The endpoint-aware Markov sketch works; the scalar count
  alone does not.
- Lean anchor: `insufficient_markov_count_only_state`.

Wrong operator:

- The state and preference may be right, but the implemented `g` can still
  drift out of the `sigma`-fiber: `sigma(g x) ≠ sigma(x)` or
  `sigma(g(g x ++ g y)) ≠ sigma(x ++ y)`.
- C2/on-range idempotence is not implied by the other local requirements.
- Lean anchors: `c2_idempotence_not_derivable`,
  `c2_independence_counterexample`.

Wrong scalar level:

- A scalar oracle `f*` may not admit a well-defined merge
  `M(f*(x), f*(y))`, even though a richer `sigma` does. This is the precise
  "collapsed too early" failure.
- Lean anchors: `scalar_oracle_concat_witness_not_expressible`,
  `scalar_threshold_and_not_expressible`,
  `scalar_boundary_bigram_not_expressible`.

Everything matters:

- If the oracle is injective, oracle-sufficient compression cannot reduce the
  information needed for the task.
- Lean anchor: `no_compression_when_everything_matters`.

## Practical Diagnostic

For a proposed preference, ask:

1. Can I write `f*(x) = F(sigma(x))`?
2. Is `sigma` small enough to be useful, or did I just choose `sigma = x`?
3. Can `sigma(x ++ y)` be recovered from child states without looking back at
   raw text?
4. If implementation uses a local state `tau`, does
   `decode(mergeFold encode mergeL T) = sigma(S T)`?
5. Does the implemented summarizer `g` preserve `sigma` at leaves, merges, and
   re-summary?

If yes, C-TreePO has a theorem route. If no, the target may still be useful, but
the state must be widened, the merge rule changed, or the preference excluded
from the certificate.
