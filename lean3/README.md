# FormalProofs (Lean 4)

Lean formalization accompanying the ThinkingTrees paper/project. The development is organized into
five main modules:

- `OPT`: oracle-preserving summarization ⇒ preference-learning equivalence + gap bounds (core paper results)
- `DSL`: design-based supervised learning (IPW, honesty, empirical Bernstein wrappers, certificates)
- `CLT`: probability infrastructure (characteristic functions, Lévy continuity, CLT)
- `Econometrics`: identification / propensity-score / IPW foundations
- `ML`: supervised-learning primitives (ERM, decision trees, etc.)

## Build

From this directory:

```bash
lake build FormalProofs
```

If you only care about the OPT “main results” layer:

```bash
lake build FormalProofs.OPT.MainTheorems
```

## Reading / proof reconstruction

If you want to reconstruct the core paper proofs by hand and jump to the corresponding Lean code:

- `docs/CORE_PROOFS.md` (paper → math → Lean, step-by-step proof skeletons)
- `docs/PAPER_TO_LEAN_MAP.md` (theorem-by-theorem map from paper labels to Lean names/files)
- `docs/ASSUMPTION_CROSSWALK.md` (paper assumptions ↔ Lean aliases/files)
- `docs/ADAPTIVE_TREE_OPTIMIZER_TRANSFER.md` (expected-tree and high-probability optimizer-transfer results)
- `docs/README.md` (navigation + build entry points)

For a curated, citation-friendly Lean entry point, start from:
`FormalProofs/OPT/MainTheorems.lean`.

## Dependency: `FormalProbability`

This Lean project depends on a companion library `FormalProbability` (configured in `lakefile.toml`).
In the current monorepo layout, it is expected as a sibling checkout at `../../FormalProbability`
relative to `lean3/`. If you publish a standalone repo, vendor `FormalProbability/` into that repo
or switch the dependency to a pinned `git = ...` requirement.
