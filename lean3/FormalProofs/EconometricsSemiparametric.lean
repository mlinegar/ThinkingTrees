import FormalProofs.Econometrics.OLS.AsymptoticOLS
import FormalProofs.Econometrics.OLS.Inference
import FormalProofs.Econometrics.IV.TwoSLS
import FormalProofs.Econometrics.IV.Identification
import FormalProofs.Econometrics.IV.WeakInstruments
import FormalProofs.Econometrics.Diagnostics.Heteroskedasticity
import FormalProofs.Econometrics.Overidentification
import FormalProofs.Econometrics.Overidentification.CoverageChecklist

/-!
# FormalProofs/EconometricsSemiparametric

The half of the Econometrics subtree that (transitively) uses the
`FormalProbability.Econometrics.*` namespace route: `OLS/AsymptoticOLS`
imports `DSL/AsymptoticTheory` → `FormalProbability.Econometrics`, and
`Overidentification` imports `FormalProbability.Econometrics.GMM` directly.
These declare names (e.g. `Econometrics.PotentialOutcomes`) that clash with
the local `FormalProofs.Econometrics.Core`, so they cannot share an
environment with `FormalProofs/Econometrics.lean`; they are built as the
separate Lake target `FormalProofsEconometricsSemiparametric`.
-/
