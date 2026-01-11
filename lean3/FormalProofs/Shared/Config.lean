import Mathlib.Tactic.Linter

/-!
# FormalProofs/Config.lean

Shared configuration options for all FormalProofs modules.

## Standard Settings
- `linter.mathlibStandardSet false`: Disable mathlib linting for non-standard patterns
- `relaxedAutoImplicit false` and `autoImplicit false`: Require explicit type annotations
- `maxRecDepth 4000`: Allow deeper recursion for complex proofs

## Note on maxHeartbeats
Each file sets `maxHeartbeats` individually (0 or 400000) based on proof complexity.
Files with simple proofs use 0 (unlimited), while files with complex tactic proofs
use 400000 to avoid timeouts.
-/

-- Disable mathlib standard linter (we use custom patterns)
set_option linter.mathlibStandardSet false

-- Require explicit type annotations (no auto-implicit)
set_option relaxedAutoImplicit false
set_option autoImplicit false

-- Allow deeper recursion for complex proofs
set_option maxRecDepth 4000
