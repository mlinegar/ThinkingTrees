import FormalProofs.OPT.PaperTheorems
import FormalProofs.OPT.ExtendedExports

/-!
# Main Theorems (compatibility shim)

The former monolithic curated export surface was split on 2026-07-02:

* `FormalProofs/OPT/PaperTheorems.lean` — the paper-facing surface: exactly
  the declarations cited by the paper's Lean crosswalk
  (`paper/ctreepo/appendix/v13_triangle/E_proof_artifacts.tex`), with a
  minimal import list. Paper readers should start there;
  `lake build FormalProofs.OPT.PaperTheorems` compiles exactly the
  paper-relevant closure.
* `FormalProofs/OPT/ExtendedExports.lean` — every other curated export,
  moved verbatim, keeping the broad OPT import surface.

Both files declare into the `MainTheorems` namespace, so importing this shim
is equivalent to importing the former monolith: all fully-qualified names are
unchanged.
-/
