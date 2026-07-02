# HANDOFF — v14 Polmeth reorganization (`main_v14_polmeth.tex`)

**Date:** 2026-07-02. **Author of pass:** Claude session with Mitchell.
**Audience:** the next LLM/human session resuming work on the C-TreePO paper.
All paths relative to `paper/ctreepo/` unless noted.

---

## 1. What v14 is

v14 is the **Polmeth (political methodology conference) submission** draft. Goals
set by Mitchell and executed in this pass:

- Main text shortened from 53 pp (v13) to **38 pp**; total is 129 pp (appendices grew).
- The **manifesto application is the flagship main-text section** (§6, 14 pp),
  organized as a supervision-level arc: document labels only → expert gold
  quasi-sentence labels at every node.
- All four controlled-example sections (Markov interlude, FNO primer, Markov
  walkthrough, HLL parity) moved to appendices.
- §8 "Main Results" slimmed to 4 headline theorems; the full ladder moved to a
  new theory appendix.
- No compute runs gated this version: the 4 remaining `\fixme`s were converted
  to scope sentences (see §6 below for what is still pending as *runs*).
- New appendix documenting the `treepo` software (`fit`, `FamilyRuntime`,
  vignette, CLI, artifact schema, `treepo-bench`).

**Build state at handoff:** `latexmk -gg -pdf main_v14_polmeth` → clean.
129 pages, **zero** undefined references, **zero** undefined citations,
**zero** duplicate labels, **zero** `fixme` occurrences in v14 sources.

The plan that was executed:
`~/.claude/plans/consider-paper-ctreepo-v12-claims-and-ev-prancy-wind.md`.
The claims ledger `V12_CLAIMS_AND_EVIDENCE.md` header now names v14 as CURRENT.

---

## 2. File layout and lineage (IMPORTANT before editing anything)

v14 = a **merge of two parallel v13 lineages**, both dated 2026-07-02:

| Source | Files taken | Why |
|---|---|---|
| `sections/v13_triangle/`, `appendix/v13_triangle/` | `03_framework`, `08_theory`, `10_estimation`, `B_proofs`, `E_proof_artifacts`, `G_mechanism_checks` | Newer **theory** content: merge triangle (`lem:merge-triangle`), `ContextCompatible`, compositional Lean route (`one_pass_of_local` etc.), updated App E crosswalk. B/E were taken from the **working tree** (they carry uncommitted Lean-audit edits — do not `git checkout` the v13_triangle copies). |
| `sections/v13_reconstruction/`, `appendix/v13_reconstruction/` | everything else, notably `09_manifesto_llm`, `H_benoit_replication`, `K_qsentence_leaf_ladder`, `L_gold_seeding` | Newer **manifesto** writing: six-dim ladder folded in, fixmes resolved/narrowed. |

Verified by diff: the two lineages' diffs exactly partition along this split;
no file needed a hand-merge.

**Rules that bind future edits** (from repo memory/STYLE/CONVENTIONS):
- Non-destructive versioning: never edit `*/v13_*` or earlier from v14 work. A
  substantively new pass should create `main_v15_<tag>.tex` + copied dirs.
- Never rename `fig:/tab:/sec:/app:` labels (CONVENTIONS.md); cross-refs and
  the example-first grep convention depend on them.
- Load `STYLE.md` before any prose edit. Hard rules that mattered in this pass:
  no meta-references to the paper's own structure ("this section", "the
  reader"); named cross-references (`Section~\ref{...}`) are fine; ≤1–2
  em-dashes/section; no "not X but Y"; lead with the claim.

New files created in this pass:
- `appendix/v14_polmeth/AA_theory_details.tex` (input as **Appendix B**)
- `appendix/v14_polmeth/Z_treepo_software.tex` (input as **Appendix R**)
- `main_v14_polmeth.tex`

Deleted from v14 dirs only (content relocated, originals intact in v13 dirs):
`11_verification.tex` (absorbed into App B), `12_related.tex` (folded into
Discussion as a subsection).

---

## 3. Final structure (with start pages at handoff)

Main text:

| § | Title | Label | Pages |
|---|---|---|---|
| 1 | Introduction | `sec:introduction` | 1 |
| 2 | Prior Literature and the Algebraic Target | `sec:mergeable-sketches` | 4 |
| 3 | Framework and Local Laws | `sec:framework` | 7 |
| 4 | Main Results | `sec:theorems` | 12 |
| 5 | The Probabilistic Audit | `sec:estimation` | 17 |
| 6 | Measuring Party Positions from Party Manifestos | `sec:manifesto-deepdive` **and** `sec:manifesto-llm` (both on the umbrella) | 21 |
| 7 | Discussion (incl. "Neighboring Methods" subsection, `sec:related`) | `sec:discussion` | 34 |
| 8 | Conclusion | `sec:conclusion` | 38 |

§6 subsections:

| # | Title | Label |
|---|---|---|
| 6.1 | Rubrics, Surrogates, and the Measurement Problem | `sec:social-measurement` |
| 6.2 | Task, Corpus, and Benchmark | `sec:benoit-task` (new label) |
| 6.3 | Document Labels Only: Headline Parity and Chunk Invariance | `sec:benoit-headline` |
| 6.4 | Optimizing the Prompts: The Alternating f/g Ladder | `sec:rile-fg-ladder` |
| 6.5 | Preference Training on Oracle-Preserving Summaries | `sec:benoit-preference` (new label; holds `ass:pref`) |
| 6.6 | Gold Labels at Every Node: Oracle-Grounded Training | `sec:manifesto-qsentence` **and** `sec:rile-oracle`; subsubsections `sec:qsentence-labels`, `sec:qsentence-harness` |
| 6.7 | Trained Substrates: The Learned Merge Composes Where the Signal Is Dense | `sec:qsentence-results` |
| 6.8 | Seeding the Summarizer with Gold Quasi-Sentence States | `sec:gold-seeding` |
| 6.9 | What the Corpus Teaches the Framework | `sec:manifesto-lessons` (new, all-new prose) |

File-to-section mapping for §6: `09a_social_science_measurement.tex` = umbrella
+ 6.1; `09_manifesto_llm.tex` = 6.2–6.5; `09b_manifesto_qsentence.tex` = 6.6–6.9.

Appendices (input order in `main_v14_polmeth.tex`; letters auto-assigned):

| Letter | Content | File | Top label |
|---|---|---|---|
| A | Notation | `A_notation` | `app:notation` |
| B | **The Full Result Ladder** (new) | `AA_theory_details` | `app:theory-details` |
| C | Full proofs | `B_proofs` | `app:proofs` |
| D | Fixed partition | `C_fixed_partition` | `app:fixed-partition` |
| E | C2 counterexample | `D_counterexample` | `app:counterexample` |
| F | Proof artifacts / Lean crosswalk | `E_proof_artifacts` | `app:proof-artifacts` |
| G | Markov interlude (demoted §) | `sections/.../04_markov_interlude` | `sec:markov-interlude` |
| H | FNO primer (demoted §) | `sections/.../05_fno_primer` | `sec:fno-primer` |
| I | Markov walkthrough (demoted §) | `sections/.../06_markov_walkthrough` | `sec:markov-walkthrough` |
| J | Mechanism checks | `G_mechanism_checks` | `app:mechanism-checks` |
| K | HLL parity (demoted §) | `sections/.../07_hll_parity` | `sec:hll-parity` |
| L | Classical parity | `F_classical_parity` | `app:classical-parity` |
| M | Operator overlap | `I_operator_overlap` | `app:operator-overlap` |
| N | LDA replication | `J_lda_replication` | `app:lda-replication` |
| O | Benoit replication | `H_benoit_replication` | `app:benoit-replication` |
| P | Qsentence leaf ladder | `K_qsentence_leaf_ladder` | `app:qsentence-leaf-ladder` |
| Q | Gold seeding | `L_gold_seeding` | `app:gold-seeding` |
| R | **The `treepo` Software** (new) | `Z_treepo_software` | `app:treepo-software` |

Note the demoted sections still live in `sections/v14_polmeth/` and are
`\input` after `\appendix` — file location is irrelevant to LaTeX; their
`sec:`-prefixed labels were kept per CONVENTIONS (refs now print letters).

---

## 4. What changed, by edit type

### Moved (content unchanged or near-unchanged)
- §§4–7 of v13 (controlled examples) → Appendices G/H/I/K.
- From v13 §8 into App B: `prop:neural-operator-bridge`,
  `prop:law-constrained-no`, the projection-iff discussion, `cor:schedule`,
  `cor:folds`, `thm:multi-round`, `tab:measurability`, the transfer-moduli
  algebra, both worked DPO numeric examples, "Practical Workflow".
- From v13 §3 into App B: the entire Merge Triangle subsection
  (`lem:merge-triangle` + surrounding prose).
- All of v13 §13 "Assumption Check for Gap Bounds" into App B (keeps
  `sec:verification` as a subsection label).
- From v13 §2 into App B: the Lean-backed-instances inventory and "The Lean
  artifact mirrors this split" paragraph (now App B subsection
  `app:lean-literature-layer`).
- From 09b into App R: the three code listings (`FamilyRuntime` Protocol,
  three-call `fit` vignette, dspy/fno CLI).
- v13 §14 "Neighboring Methods" → final subsection of Discussion.

### Rewritten
- **Introduction**: the running-reference paragraph (manifesto is now the
  running reference; Markov is "the controlled anchor") and the entire Paper
  Outline paragraph (new order, appendix routing, software appendix named). A
  new paragraph leads the contributions with the manifesto headline numbers
  (0.829 vs 0.817, ~2 orders of magnitude compute, ±0.027 over 16× leaves).
- **§4 Main Results**: new opening (no longer frames via "the two ends" now
  that Markov/HLL are appendices); keeps `thm:one-pass`, `ass:CF` +
  `thm:pref-equiv`, `thm:unified-gap`, `thm:e2e`, `fig:theorem-deps` (the
  multi-round node line was dropped from the figure), and
  `eq:oracle-projection-objective` with one compact realizability paragraph.
  Closing paragraph forwards to App B.
- **§6.2 opener**: v13's ~1.5 pp of theory glue compressed to 3 paragraphs;
  "From Cardinality to RILE" rewritten as "The rubric oracle" paragraph with
  no dependence on the reader having seen HLL.
- **Conclusion**: second paragraph (the arc walk) rewritten for the new order.
- ~60 cross-reference wording sites `Section~\ref{...}` → `Appendix~\ref{...}`
  for the six demoted labels, incl. sentence-level rewrites of mixed
  Sections-lists in `03_framework:9`, `13_discussion:3`, `14_conclusion`,
  `H_benoit_replication:298`, and `\S\ref` → `App.~\ref` in §3's settings table.

### New prose (did not exist in any v13 tex)
1. **§6 umbrella opener** (`09a`, ~1 paragraph): the two-supervision-levels
   frame and internal roadmap of 6.1–6.9.
2. **§6.4 ¶"Joint training and the sixth dimension"** + the two-track recipe
   paragraph. Content promoted from `docs/manifesto_optimization_writeup.md`
   but **re-verified against the artifact** (see §5 below).
3. **§6.9 "What the Corpus Teaches the Framework"** (~1 pp): three lessons —
   (i) laws must be priced in the objective (the joint-ladder oscillation and
   the LLM-merge collapse are the same failure at two supervision levels);
   (ii) signal density governs learnability (dense ≥ ~0.13 gold share);
   (iii) construct ceilings bound any decomposition (CV R² ordering predicts
   the external-accuracy ordering). Absorbed and replaced v13's two summary
   subsections (09 "Takeaway", 09b "What the Quasi-Sentence Experiments Add").
4. **App B** framing prose (subsection intros; content otherwise moved).
5. **App R** connective prose (artifact-schema and reproduction subsections
   are new writing; listings and treepo-bench description are moved/adapted).

### Fixme conversions (exactly 4, all now scope sentences)
- `K_qsentence_leaf_ladder` (law-weighted LLM merge): collapse rows now
  explicitly "the behavior of the zero-weight objective… an objective failure,
  not a substrate limit"; run named as scoped follow-up.
- `K` (per-dim HPO): transferred economic recipe declared "the result of
  record"; per-dim-tuned 3-seed test reconstruction scoped, "slots into the
  same table".
- `L` ×2 (scheduled-sampling treatment arm; gold→dgemma transfer): declarative
  "remains scoped follow-up work; the artifact inventory above fixes the frame
  its numbers will land in."
- The `\fixme` macro itself is still defined in `preamble.tex:92` — fine.

### Small fixes
- Pre-existing typo in `fig:manifesto-fg-headline` caption: "a agreement
  reference" → "an agreement reference".
- §3's inline notation table (`tab:notation-main`) deleted — it duplicated
  App A and was referenced nowhere; replaced with a pointer to `app:notation`.

---

## 5. Number provenance for the NEW §6.4 claims (verified this session)

Decentralization trajectory in the joint run, leaf **8096** tokens (note: 8096,
not 8192 — that is the actual axis value in the run):

| stage | ext. Pearson | f*-gap |
|---|---|---|
| f¹g⁰ | 0.361 | 0.454 |
| f¹g¹ | 0.461 | 0.326 |
| f²g¹ | 0.343 | 0.484 |
| f²g² | 0.343 | 0.459 |
| f³g² | 0.359 | 0.468 |
| f³g³ | 0.413 | 0.407 |

Environment at the same leaf: 0.782 (f¹g⁰) → 0.722 (f¹g¹) → 0.704 (f³g³).
Other five dims' gaps at leaf 8096: max 0.227 (environment f²g¹), mostly <0.16.
Source, checked directly:
`outputs/manifesto_fg_alternating/combined_benoit_joint_teacher_all6_dspy_fixed_20260425_000122/plots_by_dimension/manifesto_fg_ladder_dimension_rows.csv`
(n_external = 42 for these cells).

**Number-drift trap for future editors:** `docs/manifesto_optimization_writeup.md`
also reports a single-dim decentralization run reaching **0.557** at leaf 256
with an f¹g¹ peak. That was an April partial run with identity-init f. The
final six-dim grid (App O, `tab:fg-ladder-perdim`) says decentralization best =
**0.540** at leaf 256 **peaking at the f¹g⁰ anchor**. v14 quotes only the
App-O-backed 0.540 and makes no claim about which stage the single-dim peak
occurs at. Do not reintroduce 0.557 without re-verifying its run directory.

---

## 6. What remains

### Pending compute runs (unchanged from the ledger; none block v14)
Ledger `V12_CLAIMS_AND_EVIDENCE.md` §3, items 3–6. In value order:
1. **Per-dim-tuned 6-dim FNO reconstruction** (GPU-light, no LLM fleet;
   cached llmspan grids; hours). Updates `tab:app-benoit-6dim-fno` (App P).
   Caution: HPO best values are val-objective; dec 0.679 val vs 0.244 test.
2. **Gold→dgemma transfer eval**, leaves {1, 8, 16} (eval-only, ~1–2 h fleet).
   Lands in App Q "Open Arms".
3. **LLM g-stage with law-reward weights on**, leaf 8 (tests the paper's own
   thesis; flags `--dspy-g-law-{c1,c3a,c3b}-reward-weight`, never run
   nonzero). Lands in App P; §6.7's "specific, fixable cause" sentence and
   §6.9 lesson (i) get their direct test.
4. **Scheduled-sampling rate-1 arm**, leaf 8. Lands in App Q.

Fleet notes (from ledger): dgemma runs need `TT_DSPY_DROP_RESPONSE_FORMAT=1`;
4-GPU round-robin, not affinity; `skip_lm_input_budget_check=True` for audits;
launch via `scripts/long_job.py launch`. GPUs were held by a Julia job
(PID 1689275) at last check.

When a run lands: put numbers in the named App P/Q tables, then revisit the
corresponding scope sentence (grep `scoped follow-up` in
`appendix/v14_polmeth/{K,L}_*.tex` — 4 sites).

### Writing/polish candidates (not done, deliberate)
- **Main text is 38 pp vs the ~30–35 target.** The overage is float-driven,
  mostly §3 (settings table, law cards, two TikZ figures). Trim candidates if
  Mitchell wants more cuts: §3's `tab:settings-comparison` (could shrink to a
  3-row inline list), §2's remaining HLL/collapse paragraph, §6.2's
  "Data and benchmark" (could lose ~½ pp). None attempted — flagged as his call.
- **Full STYLE.md sweep of *unchanged* prose** was not redone (v13 already had
  a pass); only new/edited prose was checked (em-dash counts, "not X but Y",
  meta-references: all clean).
- **Abstract** still describes the v13 ordering ("We demonstrate the mechanism
  on a Markov changepoint…" before the manifesto sentences). It is accurate
  but could be re-weighted manifesto-first for Polmeth. Not changed this pass.
- **Title** unchanged ("C-TreePO: Compression Tree Preference Optimization").
  A Polmeth audience might want a measurement-forward subtitle. Not discussed.
- `fig:theorem-deps` still shows the three-tier graph; fine, but if App B
  reorganizes further, check the figure's theorem refs.
- The joint-run oscillation is prose-only; an optional trajectory figure could
  be rendered from the CSV above (a plot script would need writing; per the
  plan this was explicitly optional).

### Known cosmetic notes
- Refs to demoted single sections print as e.g. "Appendix G" (their labels are
  still `sec:*` — intentional, do not rename).
- `lem:merge-triangle` is referenced from §3 (`03_framework.tex:136`) and
  prints an App-B-numbered lemma; intentional.
- Two em-dashes in App B are inherited from v13 prose, within budget.

---

## 7. How to build and verify

```bash
cd paper/ctreepo
latexmk -pdf -interaction=nonstopmode -halt-on-error main_v14_polmeth
# checks (all should be zero / empty):
grep -aE "Warning.*undefined|multiply defined" main_v14_polmeth.log
grep -rhoE '\\label\{[^}]*\}' sections/v14_polmeth appendix/v14_polmeth | sort | uniq -d
grep -rn fixme sections/v14_polmeth appendix/v14_polmeth
grep -rnE 'Section~\\ref\{sec:(markov-interlude|fno-primer|markov-walkthrough|hll-parity|color-sequences|verification)\}' sections/v14_polmeth appendix/v14_polmeth
pdfinfo main_v14_polmeth.pdf | grep Pages   # 129 at handoff
```

Note: the `.log` needs `grep -a` (binary-safe) — it contains non-UTF8 bytes.

False positives to leave alone: "Section~9-style" in `08_theory` (§4), App B
(`AA_theory_details`), and `B_proofs`/`05_fno_primer` refers to **Kovachki et
al.'s** Section 9, not an internal section.

---

## 8. Git state at handoff

Nothing committed this session. New untracked material: `main_v14_polmeth.tex`,
`sections/v14_polmeth/`, `appendix/v14_polmeth/`, this handoff, plus the v14
build artifacts (`main_v14_polmeth.{aux,log,pdf,...}` — do not commit
artifacts). Pre-existing modified/untracked files from the two v13 work
streams (v13_triangle tracked-but-modified, v13_reconstruction untracked) are
unchanged and must be preserved. `V12_CLAIMS_AND_EVIDENCE.md` header was
updated to point at v14.
