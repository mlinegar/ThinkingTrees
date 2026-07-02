# treepo Publication-Readiness Assessment

_Date: 2026-06-30. Source: three parallel source-surveys over `~/treepo` (the package),
`paper/ctreepo/` (the drafts), and the examples/docs surface across both repos. Draft for review._

## Bottom line

The **`~/treepo` package is close to release-ready**; the **paper (now
`paper/ctreepo/main_v11_systematic_pass.tex`, not the v7 drafts memory still points to) is
structurally complete but has one honest evidence gap**; and the **ThinkingTrees side is where
the un-simplified sprawl lives**. The work that remains is mostly (a) one missing experiment,
(b) a handful of mechanical fixes, and (c) collapsing the ThinkingTrees runner zoo.

---

## 1. What remains before publish-ready

### Paper — the one substantive gap

The abstract/intro claim a **two-tier** result: root-observed benchmark **plus** sampled oracle
calls at tree nodes that "turn compression-induced distortion into a finite-sample estimand with a
reported uncertainty term." But v11 only reports **one-and-a-half tiers**:

- ✅ Root-observed Manifesto scores (macro r = 0.829 per-dim, 0.807 universal) — done
- ✅ Teacher-trace local diagnostics (proxy behavior gap) — done
- ❌ **Sampled node-level certificate** — per-law C1/C2/C3 violation rates + the HT 4-term
  distortion certificate (Prop 5) on real data. Appendix G *specifies* this design but it was
  **not executed**.

The Conclusion (§8) is honest about the tier separation, but the Abstract/Intro read as if the
oracle-call audit was demonstrated. Two paths:

- **Fast:** tighten Abstract/Intro to scope the claim to "design + root-observed + synthetic
  controlled checks" → **publishable as-is**.
- **Strong:** actually run the Appendix G node audit on a Manifesto slice (sampled f* labels at
  leaves/merges, report C1/C2/C3 violation rates + HT certificate). This is the single
  highest-value experiment remaining; it makes the paper's central methodological claim
  *demonstrated* rather than *specified*.

### Package — mechanical blockers (a few hours)

Test suite is **123/129**. The 6 failures are real but small:

- `treepo.methods.dispatch` module is **missing** — 3 tests import `allowed_config_keys()` /
  `run()` from it. Either create the module (it's referenced as public surface) or update the
  tests. Genuine public-API-contract gap, not a flake.
- 3 release-gate failures are just `__pycache__` left in the source tree by the pytest run —
  clean + add to the gate's ignore set.

Everything else on the package is green: no `NotImplementedError`, no TODOs, clean exception
handling, lazy imports, no `OLD_`/duplicate code, license declared (PolyForm-NC-2.0.0-pre.2).

### Stale memory pointer

`project_ctreepo_paper_versions.md` says the canonical drafts are `main_v7_cdx/cld.tex`. The real
canonical is **`main_v11_systematic_pass.tex`** (Apr 30, supersedes v10_evidence_polish). Worth
updating so nobody edits a dead draft.

---

## 2. Where to simplify

The package barely needs it — already unified on one `treepo.fit()`. **The sprawl is entirely on
the ThinkingTrees side:**

- **134 `run_*.py` scripts + 155 shell scripts + 2 legacy sweep runners** with no single training
  entry point.
- **74 TOML configs in `config/markov/` alone** (`tradeoff_pipeline.*` × 20+, `publication_bundle.*`
  × 8, plus `fno_bridge_*`, `supervision_*`, etc.) with no "start here" marker — the v3/v4/
  v4_incremental iteration churn fossilizing into the repo.
- The "canonical surface" is an *inventory file* (`config/runtime_umbrella_entrypoints.yaml`), not
  an API.

Simplification targets, highest leverage first:

1. **Route ThinkingTrees training through `treepo.fit()`** (or document why it can't). Nobody can
   currently tell whether `train_neural_operators.py` ≡
   `treepo.fit({family:"neural_operator", ...})`. If equivalent, the 134 runners should become thin
   configs over one entry point. If diverged, that divergence is itself the thing to fix before
   publication — reviewers/readers will try the package, not the script zoo.
2. **Archive the experimental config/runner fossils** with the `OLD_` convention already in use,
   leaving only the publication-bundle configs live.
3. **Mark one canonical config per task** (`config/markov/README.md` pointing at the recommended
   starting bundle).

---

## 3. Where you need more examples

Package examples are good — 7 runnable `examples/methods/run_*.py` + matched `.toml`, all
instantiating `treepo.fit()`. The gaps:

- **No "local laws on your own data" example anywhere.** This is the conceptual heart of the paper
  (C1/C2/C3), yet there's no runnable script that shows: load custom tree data → annotate
  supervision → set the local-law knobs (`local_law_weight`, `gamma_depth`, Λ) → train → read off
  per-law diagnostics. The laws live in *design docs* (`docs/local_law_single_path_plan.md`) and in
  `treepo.local_law`, but never in a worked example. For a paper whose contribution *is* the
  local-law audit, this is the most important example to add — and it doubles as the artifact
  reviewers will run.
- **No node-audit example** — same gap as the paper's missing experiment. If you build the
  Appendix G audit, ship it as the example.
- **ThinkingTrees has zero user-facing demos** (only `examples/parity/lda_sklearn_comparator.py`, a
  diagnostic). No quickstart doc.

Prioritization: one `examples/local_law/` that goes data → fit → C1/C2/C3 violation report covers
the example gap *and* the paper's evidence gap *and* the package's API-demonstration gap
simultaneously.

---

## 4. Where a unified framework would help (and is possible)

The package already proves the unification is real and achievable: the `FamilyRuntime` Protocol
(`train_f` / `train_g` / `score_roots_with_f` / `validate_artifact`) + `CTreePOLearningSpec` +
`FitResult`, with 6 families (oracle, learnable_constant, classical_sketch, neural_operator/fno,
dspy, llm) all behind one `fit()`. Right abstraction, and done.

Where unification is **incomplete and worth finishing**:

- **ThinkingTrees → package convergence.** The 11 "canonical" scripts + 150 runners are the
  un-migrated tail. The unified `fit()` exists; ThinkingTrees just hasn't been pulled onto it.
  Biggest available consolidation win, and clearly *possible* because the package already absorbed
  the hard part.
- **One metric/evidence schema across tiers.** The paper defines a clean tier taxonomy
  (root-observed / teacher-trace / sampled-certificate). The code doesn't yet emit all three in one
  schema — `treepo.local_law` has the audit rows and `corrected_local_law_loss`, but the
  node-certificate tier isn't wired into `FitResult.metrics`. Unifying the three evidence tiers into
  the `FitResult` schema is both possible and exactly what makes the paper's "don't collapse the
  tiers" discipline enforceable in code.

---

## Suggested punch list (ordered)

1. Decide paper claim scope: tighten Abstract to match evidence **or** run the Appendix G node audit
   (the single highest-value remaining experiment).
2. Fix `treepo.methods.dispatch` (create or de-publicize) + clean `__pycache__` release gate → green
   test suite.
3. Add one `examples/local_law/` end-to-end (data → fit → C1/C2/C3 report) — closes example +
   API-demo gap, and prototypes the audit experiment.
4. Route ThinkingTrees training through `treepo.fit()`; `OLD_`-archive the config/runner fossils;
   mark one canonical config.
5. Update the `project_ctreepo_paper_versions.md` memory to point at v11.

---

## Appendix: survey evidence

### Package (`~/treepo`)

- Structure: `bench` (18 files), `methods` (17), `core` (4), `training` (3), `llm` (3), `tasks` (5),
  plus top-level `learning.py` (public `fit()`), `local_law.py`, `objective.py`, `statistic.py`,
  `state.py`, `release.py`, `certificate.py`, `cli.py`. ~9,075 LoC.
- Core contracts in `src/treepo/methods/contracts.py`: `FamilyRuntime` Protocol,
  `CTreePOLearningSpec` (frozen dataclass), `FitResult`. Public entry `treepo.fit()` at
  `src/treepo/learning.py:54`.
- Tests: 123/129 pass. Failures: missing `treepo.methods.dispatch` (3 tests:
  `test_examples_smoke.py:63`, `test_package_layers.py:107`, `test_fno_family.py:357`) + 3
  release-gate `__pycache__` failures in `test_release_gates.py`.
- No `NotImplementedError`, no TODO/FIXME/XXX, no `OLD_`/duplicate paths. `methods/neural_operator.py`
  is a thin re-export of `methods/fno.py` (intentional alias).
- README + `docs/boundary.md`, `docs/architecture.md`, `docs/training_defaults.md`.

### Paper (`paper/ctreepo/`)

- Canonical draft: `main_v11_systematic_pass.tex` (Apr 30 19:50), supersedes `main_v10_evidence_polish`,
  `main_v10_markov`, `main_v7_cdx/cld`.
- Core claim: oracle-preserving compression for long-document coding; partition into contiguous
  spans, summarize hierarchically via learned state map *g*, audit local validity via three local
  laws (C1 sufficiency, C2 idempotence, C3 merge consistency). Sampled oracle calls with logged
  propensities convert distortion into a finite-sample estimand.
- Theorem ladder: Thm 1 (local laws preserve root), Cor 1 (oracle-compatible readouts), Props 1–5
  (population preference equivalence, population gap bound, corrected node loss unbiased, HT
  distortion unbiased, finite-sample 4-term certificate).
- No TODO/`\todo`/`[CITE]`/comment markers in v11 main or appendices (relics remain only in old
  `sections/v7_cdx/` and `sections/v4/`). All referenced figures/tables exist under `assets/`.
- Evidence gap: sampled node-level certificate (C1/C2/C3 violation rates + HT certificate) is
  designed in Appendix G but not executed. Abstract/Intro may overstate relative to §8.
- Style rules in `paper/ctreepo/STYLE.md` (no "not X but Y", em-dash budget, no throat-clearers,
  one vocabulary per concept, framework name C-TreePO vs object name C-Tree).

### Examples / docs

- Package: 7 runnable `examples/methods/run_*.py` + matched `.toml`, 2 bench YAML examples. Single
  canonical `treepo.fit()`; no parallel runners.
- ThinkingTrees: `examples/` sparse (only `parity/lda_sklearn_comparator.py`); rich internal design
  docs but no user quickstart. 134 `run_*.py` + 155 shell scripts + 2 sweep runners; 74 TOML configs
  under `config/markov/`. "Canonical surface" defined by `config/runtime_umbrella_entrypoints.yaml`
  (11 canonical + 16 sidecar + 2 tool), not an API.
- Onboarding gap: no documented path for "fit a tree with local laws on your own data"; no
  treepo.fit() ↔ ThinkingTrees-training equivalence mapping; no recommended-config marker.
