import FormalProofs.Econometrics.Core
import FormalProofs.Econometrics.Assumptions
import FormalProofs.Econometrics.PropensityScore
import FormalProofs.Econometrics.IPWIdentification
import FormalProofs.Econometrics.Overidentification
import FormalProofs.Econometrics.Overidentification.CoverageChecklist
import FormalProofs.Econometrics.OLS.GaussMarkov
import FormalProofs.Econometrics.OLS.AsymptoticOLS
import FormalProofs.Econometrics.OLS.Inference
import FormalProofs.Econometrics.OLS.RSquared
import FormalProofs.Econometrics.IV.Identification
import FormalProofs.Econometrics.IV.TwoSLS
import FormalProofs.Econometrics.IV.WeakInstruments
import FormalProofs.Econometrics.Panel.FixedEffects
import FormalProofs.Econometrics.Panel.RandomEffects
import FormalProofs.Econometrics.Panel.Hausman
import FormalProofs.Econometrics.Diagnostics.FunctionalForm
import FormalProofs.Econometrics.Diagnostics.Heteroskedasticity
import FormalProofs.Econometrics.Diagnostics.OmittedVariableBias
import FormalProofs.Econometrics.README

/-!
# FormalProofs/Econometrics

Convenience re-exports for the Econometrics subdirectory.

This umbrella is deliberately NOT imported by the top-level `FormalProofs`
module (see the namespace-clash note there). It is built as its own Lake
target (`FormalProofsEconometrics` in `lakefile.toml`) so the whole subtree
stays compiled and cannot bitrot while remaining quarantined from the
`FormalProbability.Econometrics.*` namespace route.
-/
