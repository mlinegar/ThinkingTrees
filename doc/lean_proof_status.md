# Lean Proof Status (OPT Module)

## Summary
- **Axioms in active build**: 1
- **Sorries in active build**: 0
- **Deprecated file with sorries**: 1 (`FormalProofs/Deprecated/PointwiseLipschitz.lean`)

This means the main `FormalProofs` build is sorry-free, but it does rely on a
single modeling axiom and contains a deprecated file with unresolved `sorry`.

## Active Axiom
- **Name**: `ExpectedGroupLossLipschitz`
- **Location**: `lean3/FormalProofs/OPT/PreferenceBounds.lean`
- **Purpose**: Expected loss Lipschitz in oracle distance (Random Utility Model).
- **Documentation**: `lean3/FormalProofs/Axioms.lean` (re-exports and narrative).

## Deprecated Sorries (Not Imported)
- **File**: `lean3/FormalProofs/Deprecated/PointwiseLipschitz.lean`
- **Reason**: Pointwise Lipschitz fails at ties; expected Lipschitz axiom replaces it.
- **Action**: Keep in a separate repo/branch or delete before release to avoid confusion.

## Consistency Checks To Do Before Launch
- Ensure `FormalProofs.lean` proof-status text matches this report.
- Ensure the paper states the single axiom explicitly and cites RUM/McFadden.
- Ensure `Axioms.lean` is the single source of truth for assumptions.

## Suggested CI For Lean Repo
- `lake build FormalProofs`
- Optional: `lake build FormalProofs.OPT` if you split targets

