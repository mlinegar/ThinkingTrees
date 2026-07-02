import FormalProbability.CLT.Core
import FormalProbability.CLT.ProbabilityLaws
import FormalProbability.CLT.Distributions
import FormalProbability.CLT.GeneratingFunctions
import FormalProbability.CLT.WeakLaw
import FormalProbability.CLT.HellySelection
import FormalProbability.CLT.LevyContinuity
import FormalProbability.CLT.Normal
import FormalProbability.CLT.CLT

/-!
# FormalProofs/CLT

Convenience re-exports for the CLT/probability subtree.

Consolidated 2026-07-02: the per-module re-export stubs formerly under
`FormalProofs/CLT/` (Core, ProbabilityLaws, Distributions, GeneratingFunctions,
WeakLaw, HellySelection, LevyContinuity, Normal) are folded into this umbrella,
which now imports the `FormalProbability.CLT.*` modules directly.  Each stub
file is now a two-line shim importing `FormalProofs.CLT`, keeping the legacy
`FormalProofs.CLT.<Module>` import paths stable.

Original stub doc text:

## From FormalProofs/CLT/Core.lean

Thin re-export of `FormalProbability.CLT.Core`.

## From FormalProofs/CLT/ProbabilityLaws.lean

Thin re-export of `FormalProbability.CLT.ProbabilityLaws`.

## From FormalProofs/CLT/Distributions.lean

Thin re-export of `FormalProbability.CLT.Distributions`.

## From FormalProofs/CLT/GeneratingFunctions.lean

Thin re-export of `FormalProbability.CLT.GeneratingFunctions`.

## From FormalProofs/CLT/WeakLaw.lean

Thin re-export of `FormalProbability.CLT.WeakLaw`.

## From FormalProofs/CLT/HellySelection.lean

Thin re-export of `FormalProbability.CLT.HellySelection`.

## From FormalProofs/CLT/LevyContinuity.lean

Thin re-export of `FormalProbability.CLT.LevyContinuity`.

## From FormalProofs/CLT/Normal.lean

Thin re-export of the standard normal distribution facts from `FormalProbability`.
This keeps the legacy `FormalProofs.CLT.Normal` import path stable without introducing
duplicate declarations in the shared `ProbabilityTheory` namespace.
-/
