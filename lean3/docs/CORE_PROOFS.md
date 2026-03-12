# Core Proofs (Paper ↔ Lean)

This document is meant to let an *inexperienced* reader do two things:

1. Reconstruct the core arguments **by hand** (pen-and-paper proof skeletons).
2. Jump from each proof step to the **corresponding Lean lemma/definition**.

The Lean code is the source of truth for formal statements; this file is the “guided tour”.

---

## 0. Quick map: main paper results → Lean theorems

The paper’s main theorem statements are in `paper/sections/03_main_theorems.tex`.

| Paper result | Paper label | Lean theorem | Lean file |
|---|---:|---|---|
| Inductive Preservation | `thm:one-pass` | `one_pass` | `../FormalProofs/OPT/PreservationTheorems.lean` |
| Schedule invariance | `cor:schedule` | `schedule_invariance` | `../FormalProofs/OPT/PreservationTheorems.lean` |
| Fold-of-folds invariance | `cor:folds` | `fold_of_folds` | `../FormalProofs/OPT/PreservationTheorems.lean` |
| Multi-round preservation | `thm:multi-round` | `multi_round_proper` | `../FormalProofs/OPT/ExpectationTheory.lean` |
| DPO equivalence | `thm:dpo-equiv` | `dpo_equivalence` | `../FormalProofs/OPT/PreferenceBounds.lean` |
| GRPO-PL equivalence | `thm:grpo-pl` | `grpo_equivalence` | `../FormalProofs/OPT/PreferenceLearning.lean` |
| GRPO-RL equivalence | `thm:grpo-rl` | `grpo_rl_equivalence` | `../FormalProofs/OPT/PreferenceLearning.lean` |
| Unified preference gap | `thm:unified-gap` | `unified_preference_gap_bounded` | `../FormalProofs/OPT/PreferenceBounds.lean` |
| “L3 is substantive” counterexample | `thm:l3-necessary` | `thm10_1_L3_not_derivable` | `../FormalProofs/OPT/CounterexampleExistence.lean` |

For a curated Lean entry point that re-exports the key results with documentation, start from:
`../FormalProofs/OPT/MainTheorems.lean`.

---

## 1. Notation map (paper → Lean)

All core objects are defined in `../FormalProofs/OPT/CoreDefinitions.lean` and
`../FormalProofs/OPT/LocalLaws.lean`.

| Concept | Paper notation | Lean identifier |
|---|---|---|
| Document space (monoid) | `Strings` with `concat` | `Strings` with `[Monoid Strings]` |
| Oracle space (pseudo-metric) | `(Y, d_Y)` | `Y` with `[PseudoMetricSpace Y]` |
| Oracle function | `f* : Strings → Y` | `fstar : Strings → Y` |
| Summarizer (randomized) | `g(x)` is a distribution on strings | `Summarizer Strings := Strings → PMF Strings` |
| Distortion | `d_Y(f*(z), f*(x))` | `D fstar z x := dist (fstar z) (fstar x)` |
| One-step expectation | `E_{z~g(x)}[·]` | `Eg g f x` |
| Tree reduction | reduce a merge tree bottom-up | `reduce g T : PMF Strings` |
| Multi-round reduction | `Z^(R)` | `ZR g x R T : PMF Strings` |
| Realized string of a tree | product of leaves | `S T` |
| Internal-node expectation | expectation under `reduce` | `Egu g T f` |

### Local laws: paper C1/C2/C3 vs Lean L1/L2/L3

The Lean names are in `../FormalProofs/OPT/LocalLaws.lean`:

- Paper **C1 (Sufficiency)** = Lean **L1**: leaf summaries preserve the oracle (in expectation).
- Paper **C3 (Merge consistency)** = Lean **L2**: internal-node merges preserve the oracle (in expectation).
- Paper **C2 (Idempotence / on-range stability)** = Lean **L3**: re-summarizing an on-range string is oracle-preserving (in expectation).

The slightly “scrambled” numbering (C2 ↔ L3) is intentional: in Lean, L1/L2 are the
tree-local laws, and L3 is the global “on-range” law.

---

## 2. Core proof chain (what implies what)

At a very high level, the proof flow is:

1. **Local laws** (`L1`, `L2`, `L3`)
2. ⇒ **preservation of oracle distortion** under tree reduction (`one_pass`, `multi_round_proper`)
3. ⇒ **zero distortion on support** (the nonnegative-expectation argument)
4. ⇒ **expected loss invariance** for oracle-measurable losses and oracle-indexed generators
5. ⇒ **training equivalence** for DPO / GRPO-PL / GRPO-RL
6. and (separately) ⇒ **quantitative gap bound** when distortion is nonzero (`unified_preference_gap_bounded`)

The rest of this document unpacks each arrow with a hand-proof skeleton and Lean anchors.

---

## 3. Inductive (one-pass) preservation

**Paper:** Theorem “Inductive Preservation” (`thm:one-pass`).

**Lean statement:** `one_pass` in `../FormalProofs/OPT/PreservationTheorems.lean`.

### What to prove by hand

Fix a merge tree `T` whose leaves multiply to the document `x = S T`.
Define the property for *any* subtree `u` of `T`:

> `P(u)`: the expected distortion at `u` is zero:  
> `E_{z ~ reduce(g,u)}[ d_Y(f*(z), f*(S u)) ] = 0`.

Then prove `P(u)` for every subtree `u` by structural induction.

### Proof skeleton

1. **Base case (leaf):** `u = leaf b`.
   - `reduce(g, leaf b) = g(b)`.
   - `S(leaf b) = b`.
   - `L1` is *exactly* the claim that `Eg g (fun z => D fstar z b) b = 0`.
2. **Inductive case (node):** `u = node u_L u_R`.
   - `L2` asserts that the expected distortion at each realized internal node is 0, i.e.
     `Egu g (node u_L u_R) (fun z => D fstar z (S (node u_L u_R))) = 0`.
   - This is precisely `P(u)` for internal nodes.
3. Apply the result to the root `u = root T` and rewrite `S (root T) = S T = x`.

### Lean anchors

- The induction is packaged as `nodewise_preservation` (subtree-by-subtree), then specialized to the root in `one_pass`:
  - `nodewise_preservation` in `../FormalProofs/OPT/PreservationTheorems.lean`
  - `one_pass` in `../FormalProofs/OPT/PreservationTheorems.lean`
- The corollaries in the paper are proved immediately from “both sides are 0”:
  - `schedule_invariance` in `../FormalProofs/OPT/PreservationTheorems.lean`
  - `fold_of_folds` in `../FormalProofs/OPT/PreservationTheorems.lean`

---

## 4. Multi-round preservation

**Paper:** Theorem “Multi-round preservation” (`thm:multi-round`).

**Lean statement:** `multi_round_proper` in `../FormalProofs/OPT/ExpectationTheory.lean`.

### What to prove by hand

Let `Z^(R)` be the random output after `R` summarization rounds:
`Z^(1) := reduce(g,T)` and `Z^(R+1) := g(Z^(R))`.

Goal: for all `R ≥ 1`,

> `E[ d_Y(f*(Z^(R)), f*(x)) ] = 0`.

### Proof skeleton (induction on R)

1. **Base (R = 1):** this is exactly the one-pass theorem applied to the root.
2. **Step:** assume `E[ d_Y(f*(Z^(R)), f*(x)) ] = 0`. Consider `Z^(R+1) = g(Z^(R))`.
   - Use the fact that `d_Y ≥ 0`. In measure-theoretic terms:
     `E[nonneg] = 0` forces the integrand to be 0 almost surely.
   - `Z^(R)` lies in the range/support of `g` (because it was created by reductions using `g`).
   - Apply **L3** (on-range idempotence): re-summarizing on-range strings preserves the oracle.
   - Conclude the next-round distortion remains 0 in expectation.

### Lean anchors (and a technical note)

- `multi_round_proper` is the “fully rigorous” statement:
  - `multi_round_proper` in `../FormalProofs/OPT/ExpectationTheory.lean`
- You will also see convenience wrappers:
  - `multi_round_bounded` and `multi_round_typeclass` in `../FormalProofs/OPT/ExpectationTheory.lean`
- **Why the Lean proof looks more technical than the paper:** Lean represents expectations over a `PMF` as `∑'` (a `tsum`), so it must prove *summability*. The `*_proper` versions add an explicit bound on distortion to keep everything axiom-free and summable.

---

## 5. From “expected distortion = 0” to “distortion = 0 on support”

Many downstream equivalence theorems want a hypothesis of the form:

> for all `z` in the support of the summarized distribution and `x` in the support of the original distribution,  
> `dist (fstar z) (fstar x) = 0`.

But preservation is often proved first as an *expectation* statement:
`E[D] = 0`.

### Hand lemma (discrete “E[X]=0 ⇒ X=0 a.s.”)

Let `p` be a discrete distribution and `h : α → ℝ` with `h ≥ 0`.
If `E_p[h] = 0`, then for every `a ∈ support(p)`, we must have `h(a) = 0`.

**Proof:** if some `a` has positive mass and `h(a) > 0`, then the expectation would be strictly positive.

### Lean anchors

This reasoning is done explicitly (by contradiction) in several “via ZR” theorems, e.g.:

- DPO: `dpo_gap_zero_of_local_laws_bounded` in `../FormalProofs/OPT/PreferenceBounds.lean`
- GRPO-PL: `grpo_equivalence_via_ZR` in `../FormalProofs/OPT/PreferenceLearning.lean`
- GRPO-RL: `grpo_rl_equivalence_via_ZR` in `../FormalProofs/OPT/PreferenceLearning.lean`

When reading those proofs, look for the pattern:

1. assume `dist (fstar z) (fstar x) ≠ 0` for a support point `z`,
2. show it implies the `tsum` defining `Exp` is `> 0`,
3. contradict the previously established `Exp (...) = 0`.

---

## 6. Zero distortion ⇒ expected loss invariance (method-agnostic)

This is the core “bridge” between preservation and learning objectives.

### 6.1 Pairwise (generic) version

**Lean:** `expected_loss_eq_of_zero_dist_generic` in `../FormalProofs/OPT/PreferenceLearning.lean`.

**Hypotheses (what you assume by hand):**

1. `h_zero`: all oracle distances between `μ_Z`-support and `μ_X`-support points are 0,
2. `loss` is **oracle-measurable**: `dist(f*(x),f*(x'))=0 → loss x a = loss x' a`,
3. `gen` is **oracle-indexed**: `dist(f*(x),f*(x'))=0 → gen x = gen x'`.

**Conclusion:**

> The expected loss computed under `μ_X` equals the expected loss computed under `μ_Z`.

### Hand proof skeleton

1. Pick a reference point `x₀ ∈ support(μ_X)` (possible because `PMF.support_nonempty`).
2. Show every `x ∈ support(μ_X)` has `dist(f*(x), f*(x₀)) = 0`.
   - Use a fixed `z₀ ∈ support(μ_Z)` plus triangle inequality and the hypothesis `h_zero`.
3. By oracle-indexedness, `gen x = gen x₀` for all `x` in support.
4. By oracle-measurability, `loss x a = loss x₀ a` for all `x` in support and all `a`.
5. Therefore the inner expectation `E_{a~gen x}[loss x a]` is constant over support,
   so the outer expectation is that constant.
6. Repeat the same argument for `μ_Z` and conclude both expectations are equal.

### 6.2 Groupwise version (k-wise losses)

GRPO-style objectives are groupwise. The same idea appears as:

- `expected_group_loss_eq_of_zero_dist` in `../FormalProofs/OPT/PreferenceLearning.lean`

---

## 7. Instantiations: DPO, GRPO-PL, GRPO-RL

Once you have Sections 4–6, the method-specific equivalence theorems are “plug and play”:
prove oracle-measurability / oracle-indexedness, then apply the generic invariance lemma.

### DPO

- **Paper:** Theorem “DPO Equivalence” (`thm:dpo-equiv`).
- **Lean:** `dpo_equivalence` in `../FormalProofs/OPT/PreferenceBounds.lean`.

Proof idea by hand:

1. Local laws ⇒ `E[D(Z^(R), x)] = 0` (multi-round preservation).
2. Nonnegativity ⇒ `dist(f*(z), f*(x)) = 0` for all `z ∈ support(Z^(R))`.
3. DPO loss is oracle-measurable when `pol` and `pol_ref` are oracle-measurable.
4. The pair generator is oracle-indexed.
5. Apply the generic invariance lemma to conclude equality of expected DPO loss.

### GRPO-PL (Plackett–Luce)

- **Paper:** Theorem “GRPO-PL Equivalence” (`thm:grpo-pl`).
- **Lean:** `grpo_equivalence` in `../FormalProofs/OPT/PreferenceLearning.lean`.

`grpo_equivalence_via_ZR` shows the same theorem specialized to the `ZR` distribution.

### GRPO-RL (clipping + KL; DeepSeek-R1 style)

- **Paper:** Theorem “GRPO-RL Equivalence” (`thm:grpo-rl`).
- **Lean:** `grpo_rl_equivalence` in `../FormalProofs/OPT/PreferenceLearning.lean`.

Again, `grpo_rl_equivalence_via_ZR` is the `ZR`-specialized version.

---

## 8. Quantitative bound: the unified preference gap

When distortion is not exactly zero, we want a bound of the form:

> `| E_X[E_gen(X)] - E_Z[E_gen(Z)] | ≤ L · Δ_R`,

where `Δ_R` is the expected oracle distortion between originals and summaries.

### Lean statement

- `unified_preference_gap_bounded` in `../FormalProofs/OPT/PreferenceBounds.lean`.

### Hand proof skeleton (the coupling argument)

Let `μ_X` and `μ_Z` be distributions on documents, and let `E_gen : Strings → ℝ` be the
“inner expected loss” for a fixed document.

1. Write the difference as a **double sum over the product measure**:
   - `E_X[E_gen] - E_Z[E_gen]`
   - `= ∑_x μ_X(x) E_gen(x) - ∑_z μ_Z(z) E_gen(z)`
   - `= ∑_x ∑_z μ_X(x) μ_Z(z) (E_gen(x) - E_gen(z))`.
2. Take absolute values and apply triangle inequality:
   - `|∑_{x,z} μ_X μ_Z (E_gen(x) - E_gen(z))|`
   - `≤ ∑_{x,z} μ_X μ_Z |E_gen(x) - E_gen(z)|`.
3. Apply the **Lipschitz assumption**:
   - `|E_gen(x) - E_gen(z)| ≤ L · dist(f*(x), f*(z))`.
4. Factor out `L` and identify the remaining quantity as `Δ_R`.

### Lean anchors for the proof steps

The Lean proof follows the same steps, but must also manage `tsum` summability:

- Step (1) is `coupling_expansion_bounded` in `../FormalProofs/OPT/PreferenceBounds.lean`.
- Steps (2)–(3) are packaged as `coupling_bound_ineq_bounded` in `../FormalProofs/OPT/PreferenceBounds.lean`.
- The final assembly is `unified_preference_gap_bounded` in `../FormalProofs/OPT/PreferenceBounds.lean`.

---

## 9. Necessity: L3 is independent (“L3 is substantive”)

**Paper:** Theorem “L3 is substantive” (`thm:l3-necessary`).

**Lean:** `thm10_1_L3_not_derivable` in `../FormalProofs/OPT/CounterexampleExistence.lean`.

### Hand proof idea

Construct a summarizer that behaves well on *fresh* inputs, but misbehaves on its own outputs:

- On fresh strings `b`, it returns a canonical representative consistent with the oracle value.
- On some `s ∈ range(g)`, it *flips* (creates a 2-cycle), violating on-range idempotence.

This demonstrates why L3/C2 is a genuine extra condition, not redundant with L1 and L2.

---

## 10. Suggested reading order (Lean)

If you want to follow the proofs directly in Lean, a good order is:

1. `../FormalProofs/OPT/CoreDefinitions.lean`
2. `../FormalProofs/OPT/LocalLaws.lean`
3. `../FormalProofs/OPT/PreservationTheorems.lean`
4. `../FormalProofs/OPT/ExpectationTheory.lean`
5. `../FormalProofs/OPT/PreferenceLearning.lean`
6. `../FormalProofs/OPT/PreferenceBounds.lean`
7. `../FormalProofs/OPT/MainTheorems.lean` (for the polished/curated layer)

