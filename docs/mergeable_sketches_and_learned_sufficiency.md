# Mergeable Sketches and Learned Sufficiency: a C-TreePO Lineage

A working synthesis of the academic lineage from sufficient statistics through
mergeable summaries through Sketch-Flip-Merge (SFM) into C-TreePO's
learned-sufficiency formulation. Intended as slide-prep material and as a
reference for the paper's "related work" / framing sections.

Companion docs:
- [appendix_f_mergeable_sketch_review_guide.md](appendix_f_mergeable_sketch_review_guide.md) — broad
  empirical suite comparing classical and learned variants.
- [sfm_comparison_setup.md](sfm_comparison_setup.md) — SFM-constrained
  benchmark configuration.
- [learned_sketch_simulation.md](learned_sketch_simulation.md) — HLL parity
  experiments and learned-sketch smoke workflow.
- [compositional_summary_learning.md](compositional_summary_learning.md) —
  manifesto / observable-summary thread.
- Lean: `lean3/FormalProofs/OPT/SketchFlipMergeBridge.lean`,
  `lean3/FormalProofs/OPT/SketchRecovery.lean`,
  `lean3/FormalProofs/OPT/OracleSufficientCompression.lean`,
  `lean3/FormalProofs/OPT/InformationSufficiency.lean`.

---

## 0. TL;DR

C-TreePO sits at the end of a four-step lineage:

1. **Fisher (1922)** — sufficient statistics.
2. **Blackwell (1953)** — sufficiency ordering of experiments.
3. **Mergeable summaries (Agarwal et al., PODS 2012)** — sufficient statistics
   under an associative merge axiom; the algorithmic foundation for streaming
   sketches like HLL, Count-Min, GK quantiles.
4. **Sketch-Flip-Merge (Hehir–Ting–Cormode, KDD 2023)** — mergeable summaries
   under local differential privacy; sufficiency must survive a noisy local
   randomizer.

The unifying primitive across all four is **`f(g(x)) ≈ f(x)`**: a summarizer
`g` that preserves the answer of an oracle / query function `f` on raw inputs.
The tree-merge version is **`f(g(g(u) · g(v))) ≈ f(u · v)`**.

Each generation moves the constraint up one level of abstraction:

| Generation | What's fixed | What's free |
|---|---|---|
| Mergeable summaries | closed-form `f` (cardinality, frequency, quantile) | choice of `f` from a small catalog |
| SFM | `(M, g)` from a small set of channel-compatible pairs | privacy budget ε |
| **C-TreePO** | the **local laws** (C1/C2/C3) | `f`, `g`, the oracle, the readout |

The C-TreePO move: **fix `f` to be whatever oracle the task gives you (LLM
judge, downstream loss, learned scorer), and learn `g` to be an approximate
sufficient statistic for that `f`.** The local laws are the empirical and
Lean-formal certificate that `g` is approximately sufficient. SFM's rigidity
result (one merge route → one oracle, up to equivalence) survives, but
relaxes from a hard limitation to an identifiability statement: when trained
jointly under a λ penalty, `f` drifts from `f*` to land on the closest oracle
that `g` can be sufficient for.

The genuinely new ingredient that LLMs unlock: **`g` is observable**. Classical
sketch states are register tuples; SFM states are noisy bitmaps. An LLM's `g(x)`
is text — readable, auditable, usable as supervision. Verification of
sufficiency goes from algebraic proof to direct inspection plus counterfactual
audit.

---

## 1. The Unifying Frame: Approximate Sufficient Statistics

Setup:

- `Strings` is the input space (documents, multisets of items, traces).
- `Y` is the output space (the task value, equipped with a metric `D`).
- `f* : Strings → Y` is the **true oracle** — what you actually want at the
  root.
- `f : Strings → Y` is your **approximator** of `f*`.
- `g : Strings → Strings` (or `Strings → Sketch`) is the **internal-node
  summarizer**.

The single identity that everything in this document is a special case of:

> **`g` is an approximate sufficient statistic for `f` along the merge route**:
>
> ```
> f(g(g(u) · g(v))) ≈ f(u · v)
> ```

Read this as: knowing `g(x)` is enough to recover `f(x)` up to ε, and the
binary merge `g(g(u) · g(v))` preserves that property at every internal node.

### Why "sufficient statistic"

Fisher (1922): `T(X)` is sufficient for parameter θ iff the conditional
distribution of `X` given `T` does not depend on θ. The C-TreePO analogue
treats `f*` as θ and `g(x)` as `T(x)`: knowing the summary determines the
oracle value, regardless of within-fiber variation.

Blackwell (1953): one experiment dominates another iff the dominated one is a
post-processing of the dominator. An oracle-sufficient `g` is Blackwell-
equivalent to the raw input *for any oracle-measurable task* — equivalent for
that task, strictly more compressed. This is exactly the framing in
`OracleSufficientCompression.lean`:

```
| 1 | shannon_impossibility_full_information       | General lossless compression IS impossible      |
| 2 | zero_oracle_distortion_of_oracle_sufficient  | Oracle-sufficient compression IS achievable     |
| 3 | fiber_representative_oracle_sufficient       | The f*-fiber quotient map achieves it           |
| 4 | no_compression_gain_of_injective_oracle      | When f* is injective, no compression helps      |
| 5 | ctreepo_is_oracle_lossy_not_lossless         | C-TreePO IS lossy w.r.t. full information       |
```

The fiber `{x' : f*(x') = f*(x)}` is the central geometric object. `g` is
required to preserve fiber membership; within-fiber variation is task-
irrelevant noise that `g` is allowed to discard.

### Local laws as approximate sufficiency

The exact merge identity above is too strong. The local-law bundle relaxes it:

- **C1 (monotonicity / leaf sufficiency)**: leaf encoding preserves the oracle
  on single items.
- **C2 (leaf calibration consistency)**: the merge step preserves the oracle
  up to bounded distortion.
- **C3 (smoothness)**: `f` is Lipschitz / locally smooth on the summary space,
  so small `g`-state errors don't blow up at the root.

C2 is the workhorse — `feedback_lean_alignment.md` and the gamma-sweep results
both put C2 at the center. C1 and C3 are gating conditions that prevent
degenerate solutions. Together they package "approximate sufficient statistic"
into an empirically auditable bundle: each law is a measurable constraint on
the trained pair `(f, g)`, and `LocalLawsBundle` is the Lean record that
threads them through the recovery theorems.

---

## 2. Mergeable Summaries (Classical)

The defining paper is Agarwal–Cormode–Huang–Phillips–Wei–Yi, "Mergeable
Summaries" (PODS 2012, journal version TODS 2013). It gives the abstraction
its name and unifies a decade-plus of ad hoc constructions under one template.

### Definition

A **mergeable summary** is a triple `(encode, ⊕, query)`:

- `encode : Item → State` — leaf step.
- `⊕ : State × State → State` — associative, commutative merge.
- `query : State → Y` — readout.

with the guarantee:

```
query(S(A) ⊕ S(B)) ≈ query(A ⊎ B)
```

up to ε, with `|S|` sublinear in `|A| + |B|`.

In our notation: `query` is `f`, the pair `(encode, ⊕)` is `g`, and the
mergeable axiom is the merge sufficiency identity above with **exact** equality
modulo ε.

### Examples

| Sketch | `encode` | `⊕` (merge) | `query` (`f`) | Sufficient for |
|---|---|---|---|---|
| HyperLogLog | hash → leading-zero count → register `j` | register-wise max | `α · m² · (Σ 2^(−Mⱼ))⁻¹` | distinct count |
| Count-Min | row-wise hash → counter increment | element-wise sum | row-min lookup | item frequency |
| Greenwald–Khanna | quantile tuple list | rank-merge | rank query | ε-approximate quantile |
| t-digest | weighted centroid clusters | cluster union | linear interpolation | quantile |
| AMS / F2 | random sign hash → squared sum | element-wise sum | scalar squared norm | F2 / variance |
| Bloom filter | bit positions from `k` hashes | bitwise OR | membership test | set membership |

The pattern across all of these: **`g` is an analytically-derived exact
sufficient statistic for the chosen `f`, and `⊕` is associative on the
nose**. The engineering work in classical sketch design is constructing
`(encode, ⊕)` so that the sufficiency proof goes through.

### The monoid-homomorphism view

The deeper algebraic statement: a streaming sketch is a homomorphism from the
free commutative monoid `(Multisets, ⊎)` to the sketch monoid `(States, ⊕)`,
followed by a query. This is what makes parallel and tree-shaped reductions
correct independent of schedule — the same property C-TreePO needs for
schedule invariance.

Twitter's Algebird / Summingbird (Boykin et al., VLDB 2014) formalized this
exactly: every streaming aggregation is a `Monoid` instance, and the merge
axiom is monoid associativity. Cohen's size-estimation work and Cormode's
streaming surveys make the same point.

### Important caveat: sketches need associativity

Classical sketches require associativity exactly because the stream order is
adversarial — you can't control the schedule. C-TreePO is evaluated on a
*known* tree, so `g` needs only schedule-equivalence along the realized
tree, which is strictly weaker than full monoid associativity. This is one of
the relaxations the next sections build on.

---

## 3. Sketch-Flip-Merge (Mergeable + Local DP)

Hehir–Ting–Cormode, "Sketch-Flip-Merge: Mergeable Sketches for Private
Distinct Counting" (KDD 2023, with subsequent analysis by Gribelyuk et al.)
asks: how does the mergeable abstraction survive when each leaf is privatized
under local differential privacy?

### The setting

Distributed cardinality estimation:
- Each user holds a private item `xᵢ`.
- Goal: estimate `|⋃ᵢ {xᵢ}|`.
- Constraint: each user releases only an ε-locally-DP randomized version of
  their local sketch.

Three operations:

- **Sketch**: hash each item into a register-shaped state (PCSA-like bitmap,
  HLL register array).
- **Flip**: apply randomized response — flip bits with calibrated probability
  to satisfy local DP at budget ε.
- **Merge**: combine the noisy locals into a global sketch, then estimate via
  MLE.

The sufficiency identity is now:

```
f( g( M(g(u)) · M(g(v)) ) ) ≈ f(u · v)
```

where `M` is the privacy channel. **Sufficiency must survive the channel.**

### The 2×2 table

Two local randomizers:
- **`M_sym`**: symmetric RR, every bit flipped with probability `q = 1/(e^ε + 1)`.
- **`M_xor`**: asymmetric RR, asymmetric flip probabilities `p = 1/2`,
  `q = 1/(2 e^ε)`.

Two merges:
- **Deterministic OR / XOR** (bitwise).
- **Randomized merge** (sample-based; merges by drawing from a posterior over
  the merged state).

The paper's main result (Theorem 4.4 / 4.8 family) characterizes which pairs
keep the merged channel in a parametric family the MLE can invert:

| Local | Merge | Merged channel clean? | Theorem |
|---|---|---|---|
| `M_xor` | det. XOR | **Yes** — closed under XOR | 4.4 |
| `M_sym` | randomized merge | **Yes** | 4.8 |
| `M_sym` | det. OR / XOR | No — channel parameters drift with merge count | counterfactual |
| `M_xor` | det. OR | No | counterfactual |

`docs/sfm_comparison_setup.md` and `scripts/run_sfm_comparison.py` reproduce
exactly this table:

- `sfm_xor_detxor_mle` — Theorem 4.4 path (good).
- `sfm_sym_randmerge_mle` — Theorem 4.8 path (good).
- `sym_local_detor_mle`, `sym_local_detxor_mle` — counterfactual broken rows.
- `ours_ridge_sym_local_detor` — learned decoder on a "broken" pair, testing
  whether learning can recover utility even when the closed-form MLE loses
  sufficiency.

### Reframing SFM as a sufficiency-preservation problem

The 2×2 is really a **sufficiency-preservation table**: only matched `(M, g)`
pairs keep the merged state's distribution in a parametric family for which `f`
(the cardinality MLE) is well-defined and unbiased. The "broken" pairs lose
sufficiency in the technical sense — the MLE designed for one family is now
applied to states from another.

This is the precise sense in which SFM extends mergeable sketches: it adds a
noise channel between leaf and merge and asks the same sufficiency question
under that channel. The classical mergeable axiom is the ε=0, no-channel
limiting case.

---

## 4. The Rigidity Theorem: Corollary 4.11 ↔ Lean

The SFM paper's Corollary 4.11 (informally): once you fix one merge route and
one local randomizer, the family of cardinality-style queries you can recover
is essentially unique. You cannot bolt a second target onto the same merge
pipeline without forcing it to be equivalent to the first.

Our Lean restatement (`SketchFlipMergeBridge.lean`, lines 32–70):

```
def SameRouteAltTarget
    (g : Strings → Strings)
    (fstar : Strings → Y)
    (alt : Strings → Strings → Strings) : Prop :=
  ∀ u v : Strings, D fstar (alt u v) (g (g u * g v)) = 0

theorem same_route_two_targets_force_oracle_equiv
    (hA2 : A2_global g fstar)
    (hAlt : SameRouteAltTarget g fstar alt) :
    ∀ u v : Strings, D fstar (u * v) (alt u v) = 0
```

In English: if a single deterministic merge route `g(g(u) · g(v))` preserves
both the concatenation oracle and an alternate target, then the two targets
are oracle-equivalent on every input pair. Triangle inequality on `D` plus the
two zero-distortion hypotheses gives zero distortion on the difference.

The contrapositive (`no_two_distinguished_targets_on_one_route`) says: if any
input pair distinguishes the two targets under `D`, the same-route condition
fails for at least one of them.

### Same shape, two registers

| Setting | Statement |
|---|---|
| Classical mergeable | A merge `⊕` that preserves query `f` cannot also preserve a different query `f'` unless `f ≡ f'` on the relevant inputs. |
| SFM Cor. 4.11 | A `(M, g)` pair tuned for cardinality cannot also produce a clean channel for a second target unless that target is functionally equivalent. |
| C-TreePO Lean | `same_route_two_targets_force_oracle_equiv` — formal, generic, no parametric channel assumption. |

The Lean version is the most general — it makes no assumption about parametric
channels, MLEs, or closed-form readouts. It is a pure consequence of the
distance triangle inequality applied to two zero-distortion conditions on the
same `g`.

### Why this matters

Rigidity is the structural reason **one `g` per `f`**. Classical sketches
honor it by re-engineering `(encode, ⊕)` per query. SFM honors it by changing
`(M, g)` per family. C-TreePO honors it by retraining `g` per task. None of
these is a workaround — they are all the same identifiability statement.

The next two sections show how the C-TreePO formulation **partially relaxes
this rigidity**: when `f` itself is learnable and pulled by a λ penalty, you
don't have to retrain `g` for every conceivable `f*`; instead, `f` drifts to
the closest oracle that the current `g` can be sufficient for.

---

## 5. The C-TreePO Move: Learn `(f, g)`

The transition from SFM to C-TreePO can be stated as a swap of what's fixed
and what's free:

| | `f` (oracle / query) | `g` (summarizer) | What's free | What's fixed |
|---|---|---|---|---|
| Mergeable | closed-form query | engineered exact sufficient stat | choice of query from catalog | closed-form algebra |
| SFM | MLE on noisy channel | engineered, channel-compatible | privacy budget ε | `(M, g)` pair from small set |
| **C-TreePO + LLM** | **learned, λ-projected toward `f*`** | **learned, observable** | **the oracle itself** | **local laws C1/C2/C3** |

The classical regime fixed `f` so that `g` could be designed analytically. SFM
narrowed the design space further by adding a privacy channel. C-TreePO
inverts the constraint pattern: **fix the *axioms* the pair must satisfy, and
let gradient descent find `(f, g)` inside that constraint set**.

### From sketch to local laws

`SketchRecovery.lean` makes the relationship explicit:

```
abbrev sketchSummarizer (op : SketchOperator Strings Sketch) : Summarizer Strings :=
  deterministicSummarizer (summaryFromSketch op)

theorem local_laws_of_sketch
    (op : SketchOperator Strings Sketch) (fstar : Strings → Y) (T : BinTree Strings)
    (h_leaf : SketchLeafPreserving op fstar)
    (h_merge : SketchMergeCompatible op fstar)
    (h_compat : SketchSummaryCompatible op) :
    LocalLawsBundle (sketchSummarizer op) T fstar
```

Read top-down: every classical mergeable sketch (HLL, CMS, GK) is a
`SketchOperator`. Its leaf-preservation, merge-compatibility, and summary-
compatibility hypotheses are exactly the closed-form sufficiency proofs from
the original sketch papers. The theorem packages them into a `LocalLawsBundle`
— the same bundle the learned C-TreePO trainer is required to satisfy
empirically.

So **classical mergeable sketches are the special case of C-TreePO where the
local-law witnesses are closed-form proofs rather than empirical certificates**.
The Lean recovery theorems then apply uniformly to both.

### What you give up, what you gain

Give up:

- Closed-form analyzability of the merged channel.
- Tight a-priori error bounds (HLL's `1.04/√m` is replaced by an empirical
  bound from the local-law audit).
- Provable associativity of `⊕` (replaced by tree-schedule consistency, since
  we evaluate on known trees, not adversarial streams).

Gain:

- `f` is no longer constrained to a closed-form catalog. It can be:
  - a downstream task loss,
  - an LLM judge ("is this argument convincing?"),
  - a learned scorer over preferences,
  - a chained policy reward.
- `g` can exploit task structure that closed-form sketches cannot. HLL doesn't
  know that the items are English sentences; a learned `g` can.
- The local laws (C1/C2/C3) are auditable on real data, so you get an
  empirical sufficiency certificate even when the analytic one is unavailable.

---

## 6. The λ-Penalty Is the Information Bottleneck Knob

The training objective is, schematically:

```
L = L_oracle(f, f*) + λ · L_local_laws(f, g)
```

`L_oracle` measures how well `f` approximates `f*` on raw inputs.
`L_local_laws` measures how well `(f, g)` satisfies C1/C2/C3 — i.e., how
sufficient `g` is for `f`. λ is the trade weight.

### What `f` actually is at the optimum

Under this objective, `f` is **not** a free approximator of `f*`. It is the
**projection of `f*` onto the subspace of `g`-measurable functions**, weighted
by λ:

- **λ = 0**: `f → f*` pointwise. `g` may fail sufficiency badly. This is
  ordinary regression with a free-form summary that doesn't have to commute
  with the merge.
- **λ → ∞**: `f` is forced to factor through `g`'s fibers — `f(x) = h(g(x))`
  for some `h`. `f` may end up far from `f*` because `g`'s coarsening
  collapses task-relevant variation.
- **intermediate λ**: `f` is the closest oracle to `f*` that admits `g` as an
  approximate sufficient statistic, where "closest" is measured by
  `L_oracle` and "approximate sufficiency" by `L_local_laws`.

Equivalently: as λ varies, `f` traces a **Pareto frontier** between

- *oracle fidelity*: `‖f − f*‖`, and
- *sufficiency violation*: `‖f(g(g(u) · g(v))) − f(u·v)‖`.

Classical mergeable sketches sit at the y=0 axis of this frontier (exact
sufficiency, but `f` is fixed by the catalog, so `‖f − f*‖` is "infinite" for
oracles outside the catalog). C-TreePO can land anywhere on the curve.

### Connection to the Information Bottleneck

This is the **operator-valued Information Bottleneck**. Tishby–Pereira–
Bialek's IB asks for a representation `T` such that `I(T; Y)` is high (T
predicts Y) and `I(T; X)` is low (T compresses X), via the Lagrangian

```
L_IB = I(X; T) − β · I(T; Y).
```

The C-TreePO setup has the same shape with `T = g(x)`, `Y = f*(x)`, plus a
**compositional constraint** classical IB lacks:

- IB asks for a single representation.
- C-TreePO asks for a representation **that commutes with a tree merge**.

The local-law penalty `L_local_laws` is the compositional term. So the full
objective is "IB Lagrangian + merge axiom," and λ is the IB knob β with an
extra axis for the merge constraint.

This is more than an analogy — it's an explicit specialization. Setting
`L_local_laws ≡ 0` (no merge constraint) collapses C-TreePO training to
ordinary IB. Adding it back recovers schedule invariance.

### Rigidity, relaxed

Recall: SFM Corollary 4.11 / Lean
`same_route_two_targets_force_oracle_equiv` says one merge route serves one
oracle, up to oracle-equivalence. This does not vanish under the λ formulation
— but it changes character.

Without the λ penalty, you would have to retrain `g` per *unrelated* `f*`. With
the λ penalty, you retrain `(f, g)` jointly, and `f` **drifts** from `f*` to
the closest oracle the current `g` can sufficiently summarize. So the rigidity
becomes the **identifiability theorem** for the joint solution: there is a
unique fiber structure `g` induces, and `f` is the `f*`-best approximator
measurable with respect to it.

Phrased operationally: rigidity stops being a limitation and becomes the
guarantee that the optimization has a well-defined answer.

---

## 7. Observability: What LLMs Add That Classical Sketches Cannot

The genuinely novel ingredient that LLM-shaped `g` brings is **observability of
the summary itself**. This is the move that makes C-TreePO a different kind of
object from a classical sketch, not just a more flexible one.

### Three regimes of `g` opacity

- **Classical sketches**: `g(x)` is a register tuple. Opaque, but algebraically
  analyzable — you can prove things about it.
- **SFM**: `g(x)` is a noisy bitmap. Opaque *and* stochastic — analyzable only
  as a parametric channel.
- **LLM-shaped `g`**: `g(x)` is **natural language**. Readable, auditable,
  human-interpretable.

### What observability buys you

**Counterfactual sufficiency audits.** Sample `x, x'` such that `g(x) ≈ g(x')`
(judged textually or via embedding similarity). Check `f*(x) ≈ f*(x')`. If yes,
`g` is approximately sufficient on this fiber. If no, the fiber is too coarse
or the summary missed a load-bearing fact. This is a sufficiency test that
requires **no gradient information** — pure sampling and reading. Classical
sketches have no analogue because their fibers aren't human-comparable.

**Sufficiency as a readable certificate.** "Does the summary contain the
load-bearing facts?" is a question a human (or another LLM) can answer
directly. The fiber partition `g` induces is inspectable in English. C2 (leaf
calibration) becomes "the summary preserves what matters" in a literal sense.
Manifesto-style human-in-the-loop review (the
`compositional_summary_learning.md` thread) is the productized version of this.

**`g` as supervision, not just compression.** In classical sketches, `g(x)` is
never a training target — only `f(g(x))` vs `f*(x)` is. With LLMs, `g(x)` is a
token sequence you can supervise:

- Distill from a teacher's summary (the
  `feedback_distillation_via_fit.md` pattern: cached LLM labels through
  `node_oracle_predictor`).
- Constrain with stylistic priors (the summary should be coherent, factual,
  bounded length).
- Inject task-specific scaffolding (manifesto templates, structured fields).

This is what `compositional_summary_learning.md` operationalizes: `g`'s text
output is itself the supervision target, in addition to whatever downstream
loss `f(g(·))` produces.

### The verification-protocol shift

Each generation has a different way to certify sufficiency:

| Generation | Sufficiency certificate |
|---|---|
| Mergeable summaries | Algebraic proof that `query(S(A) ⊕ S(B)) = query(A ⊎ B)` exactly. |
| SFM | Parametric proof that the merged channel stays in the family the MLE inverts. |
| C-TreePO + Lean | `LocalLawsBundle` — empirical bounds on C1/C2/C3 plus formal recovery theorems. |
| C-TreePO + LLM | Above, **plus direct human or LLM-judge inspection of `g(x)` itself**. |

The slogan: *classical sketches verified sufficiency by proof; we verify it by
reading.*

---

## 8. The Constraint-Tower Progression

A single picture for the academic lineage:

```
Fisher (1922):  T sufficient for θ iff X | T does not depend on θ.
                                  ↓
Blackwell (1953): sufficiency ordering — S₁ ≥ S₂ iff S₂ is a post-processing of S₁.
                                  ↓
Mergeable summaries (Agarwal et al., PODS 2012):
  + sufficient statistics under an associative merge,
  + sublinear state size,
  + fixed catalog of f.
                                  ↓
Sketch-Flip-Merge (Hehir–Ting–Cormode, KDD 2023):
  + sufficiency under local DP channel M,
  + 2×2 (M, g) compatibility table,
  + Cor. 4.11 rigidity.
                                  ↓
C-TreePO (this work):
  + (f, g) jointly learned,
  + local laws C1/C2/C3 as the certificate,
  + λ-penalty trades oracle fidelity vs sufficiency violation,
  + Lean-formal recovery theorems,
  + LLM-shaped g is observable, enabling counterfactual audits and direct
    supervision on the summary text.
```

Each generation moves the constraint **up one level of abstraction**:

- **Generation 1**: constrain the function (`f` fixed by catalog).
- **Generation 2**: constrain the operator pair (`(M, g)` fixed by table).
- **Generation 3**: constrain only the **axioms** the pair must satisfy.

That progression is the slide-deck spine.

---

## 9. Slide Skeleton

Seven-slide deck that lands the C-TreePO move cleanly. Built bottom-up from
the unifying frame.

### Slide 1 — The Unifying Frame

Title: *Approximate sufficient statistics on a tree.*

One picture: input `x`, summarizer `g`, oracle `f`, identity
`f(g(x)) ≈ f(x)`, then the merge version
`f(g(g(u) · g(v))) ≈ f(u · v)`.

Tagline: *"`g` is sufficient for `f` along the merge route."*

### Slide 2 — Mergeable Summaries (Classical)

Title: *Engineered sufficiency, fixed catalog.*

Examples table: HLL, Count-Min, GK, t-digest. For each: `encode`, `⊕`, `f`.

Punchline: *Classical sketches are the case where `g` is an analytically-
derived exact sufficient statistic for an `f` chosen from a small catalog.*

Reference: Agarwal–Cormode–Huang–Phillips–Wei–Yi, PODS 2012.

### Slide 3 — Sketch-Flip-Merge (Mergeable Under Local DP)

Title: *Sufficiency under a noise channel.*

Show the 2×2 table (`M_sym` / `M_xor` × det. / randomized merge). Highlight
which pairs preserve the merged channel.

Punchline: *SFM is mergeable summaries with a local-DP channel `M` between
leaf and merge. Only matched `(M, g)` pairs preserve sufficiency for the
cardinality MLE.*

Reference: Hehir–Ting–Cormode, KDD 2023.

### Slide 4 — Rigidity: One Merge Route, One Oracle

Title: *Sufficiency uniqueness.*

Show the SFM Corollary 4.11 statement next to the Lean
`same_route_two_targets_force_oracle_equiv` statement, side by side.

Punchline: *A single merge route serves one oracle, up to oracle-equivalence.
This is the same theorem in two registers.*

### Slide 5 — The C-TreePO Move

Title: *Replace the catalog with axioms.*

Show the constraint-tower diagram from Section 8.

Punchline: *Fix `f` to be whatever oracle the task gives you. Learn `g` to be
an approximate sufficient statistic for that `f`. The local laws C1/C2/C3 are
the empirical certificate.*

### Slide 6 — λ Is the IB Knob (Pareto Frontier)

Title: *The trade between oracle fidelity and sufficiency.*

Picture: x-axis `‖f − f*‖`, y-axis `‖f(g(g·u · g·v)) − f(u·v)‖`. λ slides
along the frontier. Mark classical sketches at y=0 (exact sufficiency, but
limited to catalog `f`s). Mark C-TreePO operating points across the curve.

Punchline: *`f` becomes the projection of `f*` onto the subspace of
`g`-measurable functions. λ is the position on the IB-style frontier.*

### Slide 7 — Observability Buys Empirical Sufficiency

Title: *Classical sketches verified sufficiency by proof; we verify it by
reading.*

Picture, single example: input `x` (raw doc), middle row `g(x)` rendered as
text, bottom `f(g(x))` (the prediction). Annotate the middle: *"this is what
`g` thinks is sufficient — read it, sample counterfactuals, audit."*

Punchline: *LLM-shaped `g` is the new ingredient. It turns sufficiency into a
human-auditable property, enables counterfactual fiber tests, and lets the
summary itself be a supervision target.*

---

## 10. Citations

Recommended citation chain for slide 4 / paper related-work:

1. **Fisher, R. A. (1922).** *On the mathematical foundations of theoretical
   statistics.* Phil. Trans. Royal Society A. — Sufficient statistics.
2. **Blackwell, D. (1953).** *Equivalent comparisons of experiments.* Annals
   of Mathematical Statistics. — Sufficiency ordering.
3. **Torgersen, E. (1991).** *Comparison of Statistical Experiments.*
   Cambridge. — Extension of Blackwell's framework.
4. **Tishby, N., Pereira, F. C., Bialek, W. (1999).** *The Information
   Bottleneck Method.* — IB Lagrangian, `I(X;T) − β·I(T;Y)`.
5. **Cohen, E. (1997).** *Size-estimation framework with applications to
   transitive closure and reachability.* J. Computer and System Sciences. —
   Foundational sketch theory.
6. **Flajolet, P., Fusy, É., Gandouet, O., Meunier, F. (2007).** *HyperLogLog:
   the analysis of a near-optimal cardinality estimation algorithm.* — HLL.
7. **Cormode, G., Muthukrishnan, S. (2005).** *An improved data stream summary:
   the count-min sketch and its applications.* J. Algorithms. — Count-Min.
8. **Greenwald, M., Khanna, S. (2001).** *Space-efficient online computation
   of quantile summaries.* SIGMOD. — GK quantiles.
9. **Agarwal, P. K., Cormode, G., Huang, Z., Phillips, J. M., Wei, Z., Yi, K.
   (2013).** *Mergeable summaries.* ACM TODS (PODS 2012). — The canonical
   mergeable-summaries paper.
10. **Boykin, O., Ritchie, S., O'Connell, I., Lin, J. (2014).** *Summingbird: a
    framework for integrating batch and online MapReduce computations.* VLDB.
    — The monoid-homomorphism formalization.
11. **Mitzenmacher, M. (2018).** *A model for learned Bloom filters and
    optimizing by sandwiching.* NeurIPS. — Learned set membership.
12. **Hsu, C.-Y., Indyk, P., Katabi, D., Vakilian, A. (2019).** *Learning-based
    frequency estimation algorithms.* ICLR. — Learned heavy hitters on top of
    Count-Min.
13. **Hehir, J., Ting, D., Cormode, G. (2023).** *Sketch-Flip-Merge: mergeable
    sketches for private distinct counting.* KDD. — SFM.
14. **(C-TreePO, this work)** — `f, g` jointly learned, local laws as the
    sufficiency certificate, Lean-formal recovery, observable summaries.

---

## Appendix A. Lean / Code Pointers

For each section, the canonical entry points in this repo:

| Topic | Lean | Python |
|---|---|---|
| Sufficiency / Blackwell framing | `lean3/FormalProofs/OPT/OracleSufficientCompression.lean` | — |
| Information sufficiency / KLIC | `lean3/FormalProofs/OPT/InformationSufficiency.lean` | — |
| Markov-specific sufficiency | `lean3/FormalProofs/OPT/MarkovSufficiency.lean` | `src/ctreepo/sim/core/markov_changepoint_ops_count.py` |
| Sketch → local laws bridge | `lean3/FormalProofs/OPT/SketchRecovery.lean` | — |
| SFM rigidity / Cor. 4.11 analogue | `lean3/FormalProofs/OPT/SketchFlipMergeBridge.lean` | — |
| HLL parity (register-shaped readout) | — | `parallel/unified_g_v1/src/unified_g_v1/sketch/learned_hll_parity.py` |
| Broad sketch suite (Appendix F) | — | `scripts/run_sfm_comparison.py`, `src/tree/private_sfm_comparison.py` |
| Learned-sketch smoke workflow | — | `src.ctreepo.cli sim suite learned-sketch-smoke` |

## Appendix B. Things to Verify Before Citing

- The exact statement and theorem numbers in the published SFM paper (KDD 2023
  vs the arXiv version) before quoting "Corollary 4.11" — that's the working
  label in our Lean file, but the published numbering may differ.
- The most recent follow-up analyses by Gribelyuk et al. on SFM tightness
  bounds, in case they sharpen the rigidity statement.
- Whether the IB framing is explicit enough in the paper draft to warrant a
  direct cite, or whether it's better treated as an implicit specialization.

## Appendix C. What This Doc Deliberately Does Not Cover

- Full Shannon / mutual-information machinery — `InformationSufficiency.lean`
  intentionally restricts to KLIC and Doob-Dynkin factorization.
- The empirical results from the gamma-sweep, C2-only dominance, or
  ladder-warmstart experiments — those live in the project memory and the
  experiment-status doc, not here.
- Implementation details of the `f / g / fg / gf / fgf` schedule (see
  `appendix_f_mergeable_sketch_review_guide.md`).
- The manifesto thread itself — see `compositional_summary_learning.md`.
