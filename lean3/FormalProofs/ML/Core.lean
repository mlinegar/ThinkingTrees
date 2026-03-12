import FormalProbability.ML.Core

/-!
# FormalProofs/ML/Core.lean

Compatibility shim for supervised-learning core definitions.

`FormalProofs.ML.Core` now re-exports `FormalProbability.ML.Core` so shared
declarations in namespace `ML` (`Hypothesis`, `Loss`, `LabeledExample`, etc.)
come from a single module and do not conflict when both repositories are loaded.
-/
