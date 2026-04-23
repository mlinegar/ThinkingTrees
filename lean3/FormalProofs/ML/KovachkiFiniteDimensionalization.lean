import FormalProofs.ML.NeuralOperatorCore
import FormalProbability.ML.KovachkiFiniteDimensionalization

/-!
# FormalProofs/ML/KovachkiFiniteDimensionalization.lean

Compatibility shim.

The reusable Kovachki Lemma 21/22 finite-dimensionalization infrastructure now
lives in `FormalProbability.ML.KovachkiFiniteDimensionalization`. This module
keeps the historical ThinkingTrees import path available while also importing
`FormalProofs.ML.NeuralOperatorCore`, because downstream ThinkingTrees modules
use the local `CompactRealizedCallSet` interface alongside the reusable
finite-dimensionalization declarations.
-/
