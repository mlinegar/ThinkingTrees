import FormalProofs.Econometrics.Core
import FormalProofs.Econometrics.Assumptions
import FormalProofs.Econometrics.PropensityScore
import FormalProofs.Econometrics.IPWIdentification
import FormalProofs.Econometrics.OLS.GaussMarkov
import FormalProofs.Econometrics.OLS.RSquared
import FormalProofs.Econometrics.Panel.FixedEffects
import FormalProofs.Econometrics.Panel.RandomEffects
import FormalProofs.Econometrics.Panel.Hausman
import FormalProofs.Econometrics.Diagnostics.FunctionalForm
import FormalProofs.Econometrics.Diagnostics.OmittedVariableBias
import FormalProofs.Econometrics.README

/-!
# FormalProofs/Econometrics

Convenience re-exports for the Econometrics subdirectory (local-namespace route).

This umbrella is deliberately NOT imported by the top-level `FormalProofs`
module (see the namespace-clash note there). It is built as its own Lake
target (`FormalProofsEconometrics` in `lakefile.toml`) so the subtree stays
compiled and cannot bitrot while remaining quarantined.

The subtree splits along a namespace fault line: the local
`Econometrics.Core` and `FormalProbability.Econometrics.Core` both declare
`Econometrics.PotentialOutcomes`, so no single environment can contain both.
Modules that (transitively) import the `FormalProbability` route —
`OLS/AsymptoticOLS` and its dependents (via `DSL/AsymptoticTheory`), plus
`Overidentification` — live under the companion umbrella
`FormalProofs/EconometricsSemiparametric.lean` instead.
-/
