# Reproducibility

Build instructions for the C-TreePO paper and its formalization.

## Paper

```bash
cd paper/ctreepo
latexmk -pdf -interaction=nonstopmode main_new.tex
```

## Formalization

The Lean artifact is published separately. File paths referenced in
Appendix E are relative to the main repository root (`OPT/...`,
`ML/...`, `DSL/...`). After cloning the Lean repository:

```bash
lake build FormalProofs
```

This builds every theorem the paper cites. The paper-facing export
surface (every crosswalk name in Appendix E that lives in
`OPT/PaperTheorems.lean`, with a minimal import list) can be built on
its own:

```bash
lake build FormalProofs.OPT.PaperTheorems
```

which compiles exactly the paper-relevant dependency closure. Other
individual modules (e.g., `FormalProofs.ML.NeuralOperatorCore`,
`FormalProofs.OPT.NeuralOperatorSpaces`,
`FormalProofs.DSL.LabelRateBounds`) can likewise be built in isolation
with `lake build FormalProofs.<module>` if you only need part of the
dependency graph.

Foundational probability lemmas live in a companion repository; its
build is a dependency of `FormalProofs` and should fetch automatically.
