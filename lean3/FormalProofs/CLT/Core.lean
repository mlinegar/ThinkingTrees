import FormalProbability.CLT.Core

/-!
# FormalProofs/CLT/Core.lean

Compatibility shim for CLT core probability utilities.

Historically `FormalProofs.CLT.Core` duplicated the declarations now hosted in
`FormalProbability.CLT.Core` (`ProbabilityTheory.expectation`, `IID`, etc.).
This file now re-exports the shared source of truth to avoid duplicate
declaration conflicts when both projects are imported in the same environment.
-/
