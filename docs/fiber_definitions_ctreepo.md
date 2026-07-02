# Fiber Definitions for C-TreePO

This note pins down exactly what "fiber" means in the C-TreePO preference-scope
claims, and separates it from the topological usage in Futer's state-surface
paper.

## Exact Definition

For any map

```text
m : X -> Y
```

the fiber of value `y : Y` is the preimage of that value:

```text
Fiber_m(y) = { x : X | m(x) = y }.
```

The associated same-fiber relation is:

```text
x ~_m x'  iff  m(x) = m(x').
```

This relation is an equivalence relation. It partitions the input space into
sets of objects that the map `m` cannot distinguish.

Lean anchors:

- `MapFiber m y x`: `m x = y`.
- `SameMapFiber m x x'`: `m x = m x'`.
- `sameMapFiber_equivalence`: same-fiber is an equivalence relation.
- `sameMapFiber_iff_exists_common_value`: `x` and `x'` are same-fiber exactly
  when they lie in one common named fiber.

## State Fibers

For C-TreePO, the main map is a task state:

```text
sigma : X -> S
```

The state fiber at `s : S` is:

```text
Fiber_sigma(s) = { x : X | sigma(x) = s }.
```

Two documents are in the same state fiber when:

```text
x ~_sigma x'  iff  sigma(x) = sigma(x').
```

Plain-language version: a state fiber is one task-relevant bucket. Documents in
the same bucket may differ in wording, style, length, rhetoric, or raw text, but
those differences are invisible to `sigma`.

Lean anchors:

- `StateFiber sigma s x`
- `SameStateFiber sigma x x'`
- `sameStateFiber_equivalence`
- `sameStateFiber_iff_exists_common_state`
- `sameStateFiber_of_stateFiber`
- `stateFiber_of_sameStateFiber_left`
- `stateFiber_of_sameStateFiber_right`

## Preference Fibers

For a downstream target

```text
f* : X -> Y
```

the preference fiber at value `y` is:

```text
Fiber_f*(y) = { x : X | f*(x) = y }.
```

C-TreePO can preserve this preference through a state `sigma` only when state
fibers are at least as fine as preference fibers:

```text
sigma(x) = sigma(x')  ==>  f*(x) = f*(x').
```

Equivalently, there is a readout

```text
F : S -> Y
```

such that:

```text
f*(x) = F(sigma(x)).
```

The partition language is useful because it says exactly what is ruled out. If
two documents sit in the same `sigma`-fiber but the target prefers one over the
other, then `sigma` is too coarse for that preference.

Lean anchors:

- `PreferenceFiber pref p x`
- `SamePreferenceFiber pref x x'`
- `PreferenceReadoutOfState sigma F pref`
- `PreferenceFactorsThroughState sigma pref`
- `StateFibersRefinePreferenceFibers sigma pref`
- `preferenceFactorsThroughState_iff_stateFibersRefinePreferenceFibers`
- `stateFiber_subset_preferenceFiber_of_readout`
- `sameStateFiber_implies_samePreferenceFiber_of_stateReadout`

## Oracle Fibers

Existing theorem-backed C-TreePO proofs often use a metric-valued oracle

```text
fstar : X -> Y
```

and define same-oracle-fiber by zero distance:

```text
dist(fstar(x), fstar(x')) = 0.
```

For metric oracles, this is the same as equality of oracle values:

```text
fstar(x) = fstar(x').
```

Lean anchors:

- `SameOracleFiber fstar x x'`: metric zero-distance oracle fiber.
- `OracleValueFiber fstar y x`: equality-based named oracle fiber.
- `SameOracleValueFiber fstar x x'`: equality-based same-oracle-value fiber.
- `sameOracleFiber_iff_sameOracleValueFiber`
- `sameStateFiber_of_sameOracleFiber`

## How This Constrains f and g

The target `f*` constrains the state:

```text
f*(x) = F(sigma(x)).
```

This says `sigma` cannot collapse distinctions that `f*` needs. In fiber terms,
state fibers must refine preference fibers.

The summarizer `g` constrains the implementation:

```text
sigma(g x) = sigma(x)
sigma(g(g x ++ g y)) = sigma(x ++ y)
```

These say `g` must keep the tree inside the same state fiber at leaves,
resummaries, and merge nodes.

The mergeable-sketch condition constrains the state update:

```text
sigma(x ++ y) = merge(sigma(x), sigma(y)).
```

If this equation fails, the state is not locally composable as written. The
usual repair is to widen the state. For example, a scalar boundary-bigram score
is not mergeable by itself, but a state containing left endpoint, right endpoint,
and relevant counts can be mergeable.

Lean anchors:

- `SummaryPreservesState sigma g`
- `summaryPreservesState_iff_sameStateFiber`
- `StateDecomposesBy sigma merge`
- `SummaryMergePreservesState sigma g`
- `summaryMergePreservesState_iff_sameStateFiber`
- `MergeablePreferenceShape`
- `GlobalLocalPreferenceShape`
- `LocalStateRealizesGlobalState`

## Local State vs Global State

Sometimes the state that merges locally is not the final task state. Write:

```text
tau : X -> L
decode : L -> S
sigma : X -> S
```

The local state `tau` realizes the global state `sigma` when:

```text
decode(tau(x)) = sigma(x)
tau(x ++ y) = mergeL(tau(x), tau(y)).
```

Then a tree fold over `tau` stays sufficient for the global task state:

```text
decode(mergeFold encode mergeL T) = sigma(S T).
```

This is the classical sketch pattern. HLL registers, Count-Min tables, and
endpoint-aware text states are local states. Cardinality estimates, point
queries, and policy scores are readouts after the merge.

## Relation to Additive Separability

Additive separability is a special case:

```text
f*(x ++ y) = f*(x) + f*(y).
```

C-TreePO needs the broader state-level form:

```text
sigma(x ++ y) = merge(sigma(x), sigma(y))
f*(x) = F(sigma(x)).
```

The final readout `F` may be nonlinear, thresholded, or interaction-sensitive.
The local composability requirement is on `sigma`, not necessarily on `f*`
itself.

## Relation to Futer 2013

Futer's "Fiber detection for state surfaces" uses "fiber" in a topological
sense: a state surface is a fiber surface for a fibration of the link complement.
That is not the same object as a C-TreePO state fiber.

The analogy is detector-shaped, not definition-shaped:

```text
Futer:
  homogeneous state surface is a topological fiber
  iff
  the associated reduced state graph is a tree

C-TreePO:
  document belongs to a task-relevant state/preference class
  iff
  the associated state certificate has the corresponding value
```

Lean anchors:

- `Futer2013.theorem1_statement`
- `Futer2013.theorem1_yields_exact_detector`
- `Futer2013.state_factored_detector_exact`

## References

Bibliography keys in `paper/refs.bib`:

- `Fisher1922`: sufficient statistics as compression for a statistical target.
- `Blackwell1953`: decision-theoretic comparison of experiments.
- `Kallenberg2002FMP` and `doobdynkin-readable`: Doob-Dynkin factorization,
  the measure-theoretic form of "depends only through this state."
- `Agarwal2013MergeableTODS`: mergeable summaries as locally composable states.
- `Futer2013FiberDetection`: topological fiber-surface detection; useful
  analogy, distinct meaning.

Local PDF:

- `docs/references/futer_2013_fiber_detection_state_surfaces.pdf`

External reference for the Futer paper:

- David Futer, "Fiber Detection for State Surfaces," Algebraic & Geometric
  Topology 13(5), 2799--2807, 2013. DOI: `10.2140/agt.2013.13.2799`.
