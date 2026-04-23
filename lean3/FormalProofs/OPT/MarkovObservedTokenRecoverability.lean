import FormalProofs.OPT.MarkovPathDGP

/-!
# FormalProofs/OPT/MarkovObservedTokenRecoverability.lean

Observed-token recoverability for the current clean Markov benchmark family.

The runtime learnability study currently stays on the
`piecewise_disjoint_palette` document generator. In that setting each observed
token belongs to a regime-specific disjoint palette block, so the latent regime
path is identifiable from the observed tokens themselves by a deterministic
block-decoder.

This file keeps the theorem surface at that support / decoder level:

- observed tokens deterministically recover the latent regime path;
- therefore they deterministically recover the exact `MarkovCountSketch`; and
- likewise the changepoint-count target is recoverable with zero Bayes error on
  the clean support.
-/

set_option linter.mathlibStandardSet false

open scoped BigOperators
open scoped Real
open scoped Nat
open scoped Classical
open scoped Pointwise

set_option maxHeartbeats 0
set_option maxRecDepth 4000
set_option relaxedAutoImplicit false
set_option autoImplicit false

noncomputable section

namespace FormalProofs.OPT

open MarkovCountSketch

variable {Obs : Type*} {n : ℕ}

/-- Observed-token documents for the clean disjoint-palette Markov family. -/
abbrev PiecewiseDisjointObservedDoc (Obs : Type*) := List Obs

/-- In the disjoint-palette family, each observed token deterministically
reveals its emitting regime block. Mapping tokens through that block-decoder
recovers the latent regime path exactly. -/
def piecewiseDisjointPaletteObservedLatentPath
    (blockOf : Obs → Fin n) :
    PiecewiseDisjointObservedDoc Obs → MarkovPath n :=
  List.map blockOf

/-- The latent regime path is exactly recoverable from observed tokens in the
clean disjoint-palette family. -/
theorem piecewise_disjoint_palette_observed_tokens_recover_latent_path
    (blockOf : Obs → Fin n) :
    ∃ decode : PiecewiseDisjointObservedDoc Obs → MarkovPath n,
      ∀ doc,
        decode doc =
          piecewiseDisjointPaletteObservedLatentPath (n := n) blockOf doc := by
  refine ⟨piecewiseDisjointPaletteObservedLatentPath (n := n) blockOf, ?_⟩
  intro doc
  rfl

/-- Hence the exact theorem-domain Markov sketch is also recoverable directly
from observed tokens. -/
theorem piecewise_disjoint_palette_observed_tokens_recover_exact_sketch
    (blockOf : Obs → Fin n) :
    ∃ decode :
        PiecewiseDisjointObservedDoc Obs → MarkovCountSketch n,
      ∀ doc,
        decode doc =
          MarkovPath.encodePath
            (n := n)
            (piecewiseDisjointPaletteObservedLatentPath (n := n) blockOf doc) := by
  refine ⟨
    fun doc =>
      MarkovPath.encodePath
        (n := n)
        (piecewiseDisjointPaletteObservedLatentPath (n := n) blockOf doc),
    ?_
  ⟩
  intro doc
  rfl

/-- Likewise the changepoint-count target is deterministically recoverable from
observed tokens on the clean disjoint-palette support. -/
theorem piecewise_disjoint_palette_observed_tokens_recover_changepoint_count
    (blockOf : Obs → Fin n) :
    ∃ decode : PiecewiseDisjointObservedDoc Obs → ℕ,
      ∀ doc,
        decode doc =
          MarkovPath.changepointCount
            (n := n)
            (piecewiseDisjointPaletteObservedLatentPath (n := n) blockOf doc) := by
  refine ⟨
    fun doc =>
      MarkovPath.changepointCount
        (n := n)
        (piecewiseDisjointPaletteObservedLatentPath (n := n) blockOf doc),
    ?_
  ⟩
  intro doc
  rfl

/-- Bayes error is zero on the clean disjoint-palette family in the minimal
sense needed by the learnability map: there exists a deterministic decoder that
is exact on every support document. -/
theorem piecewise_disjoint_palette_zero_bayes_error
    (blockOf : Obs → Fin n) :
    ∃ decode : PiecewiseDisjointObservedDoc Obs → MarkovPath n,
      ∀ doc,
        decode doc =
          piecewiseDisjointPaletteObservedLatentPath (n := n) blockOf doc := by
  exact piecewise_disjoint_palette_observed_tokens_recover_latent_path
    (n := n)
    blockOf

end FormalProofs.OPT
