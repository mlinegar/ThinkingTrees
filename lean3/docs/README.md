# Documentation (Lean Formalization)

This folder is written for *new* readers. It explains the core proof chain in the paper and
points to the corresponding Lean definitions/lemmas.

## Where to start

- Core proof walkthrough (paper → math → Lean): `lean3/docs/CORE_PROOFS.md`
- Paper-to-Lean result map: `lean3/docs/PAPER_TO_LEAN_MAP.md`
- Assumption crosswalk: `lean3/docs/ASSUMPTION_CROSSWALK.md`
- Adaptive-tree optimizer transfer guide: `lean3/docs/ADAPTIVE_TREE_OPTIMIZER_TRANSFER.md`
- Curated “main theorems” exports (best Lean entry point): `lean3/FormalProofs/OPT/MainTheorems.lean`
- Module overview (OPT): `lean3/FormalProofs/OPT/README.lean`

## How to navigate the Lean code

- The quickest way to find anything is by lemma name:
  - GitHub search, or
  - locally: `rg "lemma_name" -n lean3/FormalProofs`
- In an editor (recommended: VS Code + Lean4 extension):
  - jump-to-definition on a lemma name,
  - hover for docstrings and types,
  - use `#check` / `#print` in scratch files to explore.

## Building locally

From the `lean3/` directory:

```bash
lake build FormalProofs
```

If you only care about the OPT results:

```bash
lake build FormalProofs.OPT.MainTheorems
```

## Dependency: `FormalProbability`

`FormalProofs` depends on a companion Lean library `FormalProbability` (see `lakefile.toml`).
In this monorepo layout, it is expected at `../../FormalProbability` relative to `lean3/`.

If you publish a standalone repo, you will want to vendor `FormalProbability/` into that repo
or switch the dependency to a `git = ...` requirement once the needed modules are published.
