# `main_minimal` Manifesto-First Blueprint (Claude v3)

A stronger combined version of the current `BLUEPRINT.md`. Keeps the
Manifesto-first flow, the five claim tiers, the Hamlet/play analogies,
the Public Formal Anchors, and the wording guardrails. Adds back a
paragraph-level field structure (Purpose / PTS / Reader / Tier /
Public / SRC / Numbers / EQ/FIG/TAB / MOVE / Guards / Cross) so every
unit below is machine-actionable against `sections/v2/` sources and
the current `sections/minimal/*.tex` files.

This is an implementation map, not a second paper outline. Every
entry corresponds to one main paragraph, one theorem block, one
figure/table callout, or one appendix subsection.

---

## 0. Global Commitments

1. **Running example.** Manifesto economic-policy dimension (public
   services vs.\ taxation). Everything else is a validation anchor.
2. **Assumed state.** $S_{\mathrm{econ}}(x)$ is the economic evidence a
   trusted rubric oracle needs for span $x$. Assumed, not observed.
3. **Flow.** Manifesto state → compression trees + local laws →
   guarantees → Manifesto results → Markov / frequency / HLL
   validation anchors → algebraic/representation backstop → audit →
   scope.
4. **Two-tier claim discipline.** The Manifesto/Benoit headline is a
   valid \emph{root-observed} corpus evaluation because the sampled
   manifestos carry full-document expert targets (Tier~1). The
   node-level C1/C2/C3 audit is a \emph{stronger, separate} claim
   (Tier~2). Neither makes the other incomplete; they answer
   different questions. See \S0.c.
5. **Hamlet-anchored analogies (two anchors).** The main text
   carries exactly two classical validation anchors, both queried
   on the same object --- the play of \emph{Hamlet} split at the
   scene level (~20 leaves):
   - **Markov** = register-shift count over
     `{Mourning, Foreboding, Political, Intrigue, Madness, Action,
     Comic}` (ordered boundary state).
   - **HLL** = distinct character \emph{co-appearance pairs}
     (each scene emits the set of speakers sharing the stage;
     each act merges by set-union; HLL's register-array with
     pointwise-max approximates union cardinality without
     storing the union-set). The simpler ``distinct speaking
     roles $\approx 24$'' is kept as an enumerable warm-up --- HLL
     only earns its place once the same pipeline runs on a
     \emph{corpus} of plays, where the union of per-scene
     pair-sets grows with the corpus while the register array
     stays fixed-size.

   Count-Min and Frequent Items are demoted to appendix
   taxonomy notes (App.~F), not main anchors. The state-alignment
   table in \S6.4 therefore has two sketch anchors plus the
   Manifesto row. Exact synthetic DGP detail stays in App.~E.
6. **Figure/table budget.** Two figures and two tables in main;
   everything else pushed. Main figures: `fig:min-plain-tree`,
   `fig:min-state-alignment` (can be promoted from table format if
   space allows). Main tables: `tab:min-benoit-headline`,
   `tab:min-state-alignment`.
7. **Cross-reference discipline.** Use existing `\label{...}` names
   with the `min-` prefix for minimal-specific variants; preserve v2
   labels otherwise.

## 0.a Running-Example Glossary

- $S_{\mathrm{econ}}(x)$: conceptual economic-policy state of span
  $x$; fiscal / welfare / market-intervention evidence, not style.
- $r_{\mathrm{econ}}$: readout from state to seven-point economic
  scale.
- $\fstar_{\mathrm{econ}}(x) = r_{\mathrm{econ}}(S_{\mathrm{econ}}(x))$:
  trusted economic oracle.
- $g(x)$: produced summary / state surrogate for span $x$.
- $f_{\mathrm{econ}}(g(x))$: practical scorer applied to a produced
  summary.
- $\sim_{\mathrm{econ}}$: $u\sim_{\mathrm{econ}}v\iff
  \metric(\fstar_{\mathrm{econ}}(u),\fstar_{\mathrm{econ}}(v))=0$.
- Leaf audit unit: compare $g(b)$ to raw span $b$ under the oracle.
- Merge audit unit: compare $g(g(u)\concat g(v))$ to the raw union
  span.
- C2 audit unit: re-summarize a stored summary and check drift.
- Root / full-document label: full-manifesto expert target (Tier~1
  observation unit).
- Tree-unit / node label: paragraph / section / merge-level local-law
  judgment with logged propensity (Tier~2 observation unit).

## 0.b Guardrails (G1--G11)

Every subsection must stay inside these lines. If a draft sentence
violates one, rewrite it before moving on. These are a merge of the
new Editing Contract with the Wording Guardrails.

- **G1. State existence is assumed, not proved.** Nothing asserts that
  political text actually has a clean $S_{\mathrm{econ}}$. The audit
  estimates how badly that assumption is violated on the realized
  tree.
- **G2. Tier~1 is complete as Tier~1.** The Manifesto/Benoit
  root-observed corpus evaluation is \emph{not} ``incomplete because
  a node audit is missing''. Node-level audits are a stronger tier,
  not a prerequisite.
- **G3. Do not say ``prompt-only evidence''.** The Manifesto result is
  a root-observed corpus evaluation; describe it that way.
- **G4. ``Summary preserves meaning''} must always name the task
  oracle.** Never state preservation without naming the oracle.
- **G5. Scalar child scores do not merge into root scores.** For
  Manifesto (and any scalar, non-homomorphic readout), the merge is
  on \emph{state}, not value. Distinct-count warning applies.
- **G6. DPO/GRPO are application bundles, not manifesto-specific
  theorems.** Say ``DPO/GRPO objectives align under the application
  bundle assumptions'', not ``proved for manifesto data''.
- **G7. Certificate formula uses $C_{\mathrm{meth}}$, not bare $L$.**
  The public `PaperErrorStack` wording is
  $C_{\mathrm{meth}}\cdot\hat\Delta_R + B_{\mathrm{cal}} +
  B_{\mathrm{est}} + B_{\mathrm{clip}}$. Only specialize to $L$ when
  the method constant has been collapsed.
- **G8. Approximation is a design target.** Neural-operator
  approximation supplies a deterministic-realizer route on compact
  realized calls. The stochastic summarizer theorem stack is the
  general formal surface.
- **G9. The framework applies when the audit says it applies.**
  Tasks whose oracle is not locally decomposable will audit badly;
  report that honestly.
- **G10. Local labels are not universal.** They pay only when aligned
  with the root oracle. Markov is the evidence.
- **G11. Frequency sketches are not idempotent.** Count-Min and
  Frequent Items are associative and commutative but \emph{not}
  idempotent --- repeated evidence must count.

## 0.c Five Claim Tiers

Every empirical or theoretical claim in the main text lives in
exactly one tier. A subsection's `Tier:` field names which tier its
load-bearing claim occupies. Do not collapse tiers; do not let a
lower-tier claim stand in for a higher one.

| Tier | Observation Unit | What Is Observed | Claim It Supports | Home |
| --- | --- | --- | --- | --- |
| **T1** Root-observed corpus eval | sampled documents | full-document expert/rubric targets | root prediction agrees with external labels on the sampled corpus | \S5 Manifesto; App.~G |
| **T2** Node-level local-law certificate | sampled tree units | leaf, summary, and merge preservation judgments with logged propensities | C1/C2/C3 distortion is estimated for a realized tree | \S8 audit; App.~H |
| **T3** Local supervision / label-budget substitution | paragraphs, sections, merge spans | application-aligned local oracle labels | smaller units can train or audit the same preservation property | \S8; \S6.1 Markov; future Manifesto quasi-sentence path |
| **T4** Formal theorem stack | Lean objects and theorem hypotheses | assumptions, not labels | conditional preservation, preference alignment, finite-sample certificates | \S4; App.~C; Lean map |
| **T5** Application narrative | domain examples | task descriptions and plausibility | a domain could instantiate the stack | \S9 scope |

**Tier vocabulary.** T1: ``root-observed'', ``full-document labels'',
``external validation'', ``document-sampling unit'', ``corpus-level
evaluation''. T2: ``node-level'', ``tree-unit'', ``local-law audit'',
``logged propensities'', ``realized-tree certificate''. T3: ``local
labels'' only when the sentence also identifies the alignment target
(same oracle as root). T4: prefix with ``backed by the theorem under
assumptions\ldots''. T5: prefix with ``an instance when\ldots''.

## 0.d Public Formal Anchors

The minimal paper cites these when it wants a clean Lean surface.
Source: `lean3/docs/PAPER_TO_LEAN_MAP.md` and
`docs/ctreepo_appendix_proof_audit.md`.

- **Preservation and schedules.**
  - C1 + C3 + context compatibility ⇒ root preservation.
  - + C2 ⇒ repeated-round stability.
  - Public wording: ``structural preservation'' and ``schedule
    invariance''.
- **Preference alignment — `PaperPreferenceStack`.**
  - Exact case (residual $0$): same full and summary argmin sets.
    Anchor: `paper_preference_stack_same_argmin`.
  - Approximate case: exact summary minimizers are $2\varepsilon$
    -optimal for the full objective. Anchor:
    `paper_preference_stack_summary_argmin_full_epsilon`.
- **Error / certificate stack — `PaperErrorStack`.**
  - Formula:
    $|G|\le C_{\mathrm{meth}}\hat\Delta_R + B_{\mathrm{cal}}
    + B_{\mathrm{est}} + B_{\mathrm{clip}}$.
  - Anchor: `paper_error_stack_high_prob`.
- **Applications.**
  - DPO / GRPO are premise packages that instantiate the preference
    stack.
  - Manifesto/RILE: root labels now; local-law labels from
    quasi-sentence aggregation as the planned node-level path.
  - Neural operators certify deterministic realizers; stochastic
    summarizers are covered by the broader PMF theorem stack.

## 0.e Empirical Number Source Map

Every numeric claim in the main text must point back to one of these
files; do not invent numbers.

- **Headline Manifesto table.**
  - Include: `assets/benoit/tables/benoit_comparison_pearson.tex`.
  - Source markdown:
    `assets/benoit/tables/benoit_comparison_pearson.md`.
  - Required numbers: $8$K character tree macro $0.829$; proprietary
    ensemble $0.817$; matched open-weight $0.793$; economic $8$K row
    $0.939$; split-expert economic reference $0.880$.
- **Economic prompt ladder.**
  - Include: `assets/benoit/tables/manifesto_fg_ladder.tex`.
  - Required numbers: $0.885, 0.886, 0.886, 0.879$ for
    $1024$--$8192$ token leaves; drops to $0.861$ at $512$ and
    $0.830$ at $256$.
- **Chunk robustness.**
  - Sources: `benoit_comparison_pearson.*` and `chunk_sweep_per_dim.md`.
  - Keep character chunk sweep and token prompt ladder on separate
    axes; do not mix units.
- **Manifesto quasi-sentence path.**
  - Planned local-law route: $2{,}157$ platforms; $\approx 2.27$M
    coded spans.
  - Not the source of current root-observed correlations.
- **Classical sketches.**
  - Sources:
    - `assets/sketches/tables/classical_sketches_grid.tex`.
    - `assets/sketches/tables/classical_sketches_report.md`.
    - `assets/sketches/tables/classical_sketches_compact.md`.
  - Count-Min `top5_point_frequency`: zero schedule spread; official
    relative RMSE $\approx 0.0527$ (small), $0.0182$ (medium),
    $0.000775$ (large).
  - Frequent Items / Frequent Strings: exact recovery of
    `top5_point_frequency` at medium/large capacity (official RMSE
    $= 0$, schedule spread $= 0$).
  - Weighted updates: supported in
    `treepo/src/treepo/sketches/adapters/datasketches_frequency.py`;
    table benchmark uses repeated unweighted updates (same algebraic
    object when coalesced).
- **HLL.**
  - $\mathrm{RSE}$ for $p=14$ is about $0.81\%$. Write ``under $1\%$''
    in main prose; keep the estimator detail in App.~F.
- **Markov.**
  - Main prose uses the Hamlet register set
    `{Mourning, Foreboding, Political, Intrigue, Madness, Action,
    Comic}` and the Acts~3$\to$4 and Acts~4$\to$5 boundary
    illustrations; App.~E carries the neutral synthetic DGP that
    the actual empirical numbers come from.
- **Hamlet heavy-hitter characters.**
  - Narrative line counts (Hamlet $\approx 1480$, Claudius
    $\approx 540$, Polonius $\approx 360$, Horatio $\approx 280$,
    \ldots) are order-of-magnitude figures drawn from standard
    editions and used only for intuition. They are not cited as
    empirical numbers and must never enter a results table. With
    the two-anchor pivot, these counts only appear in
    App.~F.1 / F.6 taxonomy prose, not in a main-text figure.
- **HLL main-anchor figure.**
  - Figure: `assets/hll/figures/hll_merge_learning_memory_median.png`
    (single-panel memory curve, median over seeds). Source:
    older run under
    `outputs/treepo_fullrun_20260306_194657/figures/cardinality/`.
    Included in `appendix/minimal/F_hll_details.tex` as
    `fig:min-hll-parity`.
  - The earlier five-panel grid
    `assets/hll/figures/hll_parity_curves.png` is retained as a
    reference but no longer included in the minimal build.
- **Figure policy.**
  - Only cite a sketch figure if it exists under
    `assets/sketches/figures/` and has been visually checked. Current
    checked-in evidence is table-based under
    `assets/sketches/tables/`. If adding a main figure, prefer a
    focused frequency-only panel, not the full grid.

## 0.f Move / Compress / Drop Rules

1. Theorem statements stay in main; proofs move to Appendix~C. Keep
   the display equation and a one-paragraph intuition.
2. One headline per setting in main; grids, ablations, per-cell
   tables to Appendices E / F / G.
3. One paragraph on neural operators in main (\S7.4); Kovachki
   paraphrases, transfer moduli, transformer inclusion, and
   operator-overlap diagrams → Appendix~D.
4. Main-text audit = tier definitions + HT estimator definition +
   certificate decomposition. Sampling formulas, reporting template,
   robustness grid → Appendix~H.
5. Drop only repetition, never results.

## 0.g Main-Text Figure / Table / Equation Inventory

Cap: 2 figures, 2 tables, 7 display equations. Main-text figures
are strictly the generic framework schematic; all validation-anchor
figures live in the appendices.

- **Main figures.**
  1. `fig:min-plain-tree` = `01_base.pdf` (\S3.2). Generic
     framework schematic, unchanged.
- **Markov register-arc figure.** `fig:min-markov-registers` =
  `assets/markov/figures/markov_scene_arc_hamlet.pdf`, produced by
  `paper/figures/markov_scene_arc_hamlet.{tex,tikz}` (F-1 output).
  Too dense for a single column, so it lives in App.~E (Markov
  details) and is cross-referenced from \S6.1. Four-register
  reduction (Mourning / Intrigue / Madness / Action) of the
  seven-register main-text set, with the explanation paragraph
  naming Acts 3$\to$4 and 4$\to$5 as the canonical flips.
- **HLL figure.** `fig:min-hll-parity` =
  `assets/hll/figures/hll_merge_learning_memory_median.pdf` lives
  in App.~F, cited from \S6.3.
- **Count-Min / Frequent Items figures.** With the two-anchor
  pivot (\S0.5), `fig:min-sketch-frequency` is \emph{not} a main
  figure. Count-Min / Frequent Items material lives in App.~F as
  taxonomy plus the sketch gallery (App.~F.9).
- `fig:min-audit-pipeline` schematic demotes to App.~H and is no
  longer a candidate main-text figure.
- **Main tables.**
  1. `tab:min-benoit-headline` from
     `assets/benoit/tables/benoit_comparison_pearson.tex` (\S5.2).
  2. `tab:min-state-alignment` (\S6.4): Markov / frequency / HLL /
     Manifesto × (state | merge target | failure mode).
- **Main equations.**
  1. $\fstar_{\mathrm{econ}}(x)=r_{\mathrm{econ}}(S_{\mathrm{econ}}(x))$
     (\S2.2).
  2. C1: $g(b)\sim b$; C2: $g(s)\sim s$ for $s\in\operatorname{range}(g)$;
     C3: $u\concat v\sim g(u\concat v)\sim g(g(u)\concat g(v))$
     (\S3.4).
  3. Root preservation: $\E[\metric(\fstar(Z^{(1)}),\fstar(x))]=0$
     (\S4.2).
  4. Unified gap: $|G_{\mathrm{meth}}|\le C_{\mathrm{meth}}\Delta_R$
     (\S4.5).
  5. `PaperErrorStack`:
     $|G|\le C_{\mathrm{meth}}\hat\Delta_R + B_{\mathrm{cal}}
     + B_{\mathrm{est}} + B_{\mathrm{clip}}$ (\S4.5, \S8.3).
  6. Markov boundary correction:
     $c_L + c_R + \One\{L.\mathrm{last}\ne R.\mathrm{first}\}$ (\S6.1).
  7. Ordered homomorphism: $h(u\concat v)=h(u)\odot h(v)$ (\S7.2).
- **Main-deferred objects.**
  - `fig:local-laws-full` → App.~A.
  - Markov result figures (`markov_simple_leaf_mass.png`,
    `markov_hard_leaf_mass.png`, `markov_budget_split.png`) → App.~E.
  - HLL merge-learning memory curve
    (`hll_merge_learning_memory_median.png`, clearer single-panel
    predecessor of the five-panel `hll_parity_curves.png`) and
    `tab:classical-parity-hll` → App.~F.
  - Count-Min / Frequent Items grid figures → App.~F (only if a
    focused panel is generated; otherwise cite tables).
  - Manifesto ladder heatmaps
    (`manifesto_fg_ladder_benoit_init.png`,
    `manifesto_fg_ladder_raw_init.png`) and per-cell tables → App.~G.
  - Neural-operator notation map, transfer moduli, overlap diagram →
    App.~D.
  - HT per-law formulas, reporting template, stress grid → App.~H.

## 0.h Section Dependency Graph

```
§1 Intro            ─┐
                     │
§2 Manifesto state  ─┼─► §3 Framework ─► §4 Guarantees ─► §5 Manifesto (T1)
                     │                       │                 │
                     │                       ▼                 │
                     │                    §8 Audit (T2) ◄──────┘
                     ▼
                §6 Validation: Markov (T3), Count-Min/FI (T1+T3), HLL (T1)
                     │
                     ▼
                §7 Algebraic backstop ─► §4.5 PaperErrorStack
                                          │
                                          ▼
                                   §8 Audit ─► §9 Scope ─► §10 Conclusion
```

Tiers per section (load-bearing claim): \S1 T5, \S2 T5, \S3 T4, \S4
T4, \S5 T1 (with T3 caveat), \S6.1 T3, \S6.2 T1+T3, \S6.3 T1, \S7 T4,
\S8 T2, \S9 T5, \S10 T5.

---

## Main-Text Blueprint (per-paragraph)

**Field key.**

- **Purpose.** Reader payoff.
- **PTS.** 1--3 claims the paragraph carries.
- **Reader.** PS (political scientist), ML, ST (statistician).
- **Tier.** T1--T5 from \S0.c.
- **Public.** Lean / public anchor when applicable (from \S0.d).
- **SRC.** v2 source file(s) and subsection.
- **Numbers.** Generated-table source path when empirical.
- **EQ / FIG / TAB.** What to display.
- **MOVE.** keep / compress / push:X / drop.
- **Guards.** G-rules from \S0.b.
- **Cross.** Section references.

### 1. Introduction

#### 1.1 Opening Paragraph: One Measurement Problem

- **Purpose:** a first-time reader understands the target before any
  algebra appears.
- **PTS:**
  1. Object: one party manifesto and one economic-policy score from
     the Manifesto/Benoit benchmark.
  2. Score is document-level; evidence is spread across paragraphs
     and sections.
  3. The useful question is whether we can work below whole-document
     granularity without changing the target.
- **Reader:** PS.
- **Tier:** T5.
- **SRC:** v2 \S9a, v2 \S1.2; minimal `01_introduction.tex`
  paragraph 1.
- **MOVE:** keep.
- **Guards:** G1.

#### 1.2 C-TreePO as State Composition

- **Purpose:** introduce the assumed state in plain language before
  any symbols appear.
- **PTS:**
  1. Assume there is $S_{\mathrm{econ}}(x)$; it is conceptual, not
     observed.
  2. A summary (LLM, quasi-sentence aggregate, learned vector) is a
     surrogate for that state.
  3. A compression tree is valid when surrogates preserve oracle-
     visible state through summarization and merging.
- **Reader:** PS + ML.
- **Tier:** T5 (concept), forward-linked to T4 in \S3--\S4.
- **SRC:** v2 \S3.1, v2 \S2.4.
- **EQ:** optional inline
  $\fstar_{\mathrm{econ}}(x)=r_{\mathrm{econ}}(S_{\mathrm{econ}}(x))$;
  full display in \S2.2.
- **MOVE:** keep.
- **Guards:** G1, G4.

#### 1.3 Granularity Reduction Payoff

- **Purpose:** state the reader-facing payoff early and tier-clean.
- **PTS:**
  1. The reported Manifesto/Benoit evidence uses root-level full-
     document labels (T1).
  2. Local preservation (T2) + local labels (T3) become useful when
     the claim is node-level certification or label-budget
     substitution.
  3. Failed local checks become \emph{measured} distortion, not hidden
     engineering error.
- **Reader:** PS.
- **Tier:** T5 (framing); references T1, T2, T3.
- **SRC:** v2 \S1.4--1.5, v2 \S13 audit summary.
- **MOVE:** keep.
- **Guards:** G2, G3, G9, G10.

#### 1.4 Roadmap

- **Purpose:** orient without duplicating the paper outline.
- **PTS:**
  1. Manifesto state first (\S2), then framework (\S3).
  2. Guarantees (\S4), then Manifesto results (\S5).
  3. Markov / frequency / HLL validate (\S6); backstop (\S7); audit
     (\S8); scope (\S9); conclusion (\S10).
- **Reader:** PS + ML.
- **Tier:** T5.
- **SRC:** current minimal `01_introduction.tex` last paragraph.
- **MOVE:** keep.

---

### 2. Running Example: Economic-Policy State

Five subsections. No generic framework definitions yet; those belong
in \S3.

#### 2.1 Define the State

- **Purpose:** make the latent-state assumption explicit and grounded.
- **PTS:**
  1. $S_{\mathrm{econ}}(x)$ holds fiscal / welfare / market-
     intervention evidence.
  2. It may discard style and slogans.
  3. It fails if discarded evidence changes the economic score.
- **Reader:** PS.
- **Tier:** T5.
- **SRC:** v2 \S9a, v2 \S9.1.
- **MOVE:** keep.
- **Guards:** G1.

#### 2.2 Define the Oracle Readout

- **Purpose:** connect state to the seven-point benchmark.
- **PTS:**
  1. $\fstar_{\mathrm{econ}}(x)=r_{\mathrm{econ}}(S_{\mathrm{econ}}(x))$.
  2. Prompt-only experiments approximate the readout with a rubric
     scorer (Gemma-4-31B-IT-NVFP4).
  3. The Manifesto/Benoit setting observes the expert/rubric target
     at the full-document \emph{root} (T1).
- **Reader:** PS.
- **Tier:** T1 anchor.
- **SRC:** v2 \S9.2.
- **EQ:** state/readout display equation.
- **MOVE:** keep.
- **Guards:** G3.

#### 2.3 Span Oracles from Quasi-Sentences

- **Purpose:** explain why local oracle values exist at all in this
  application.
- **PTS:**
  1. Quasi-sentence codings: $\sim 2{,}157$ platforms, $\sim 2.27$M
     coded spans.
  2. Aggregating codes inside a leaf or internal span yields span
     targets that instantiate C1/C3 oracles.
  3. These support the planned T2/T3 node-level certificate path; they
     are not the source of the Tier~1 headline in \S5.
- **Reader:** PS + ML.
- **Tier:** T5 (pointer to T3 path).
- **Numbers:** blueprint-stated $2{,}157$ and $\approx 2.27$M counts.
- **SRC:** v2 \S9.7.
- **MOVE:** keep, carefully.
- **Guards:** G2 (do not retroactively reclassify the Tier~1 headline
  as incomplete).

#### 2.4 Summary as Surrogate State

- **Purpose:** transition from social-science setup to tree laws.
- **PTS:**
  1. Leaf summaries should preserve span state.
  2. Merge summaries should preserve union state.
  3. Re-summarization should not shift readout.
- **Reader:** PS + ML.
- **Tier:** T5.
- **SRC:** v2 \S3.3 (C1/C2/C3 prose), restated under
  $\sim_{\mathrm{econ}}$.
- **Cross:** forward to \S3.4 for equations.
- **Guards:** G4.

#### 2.5 (optional) Why Markov / Frequency / HLL Come Later

- **Purpose:** tell the reader the algebraic anchors are evidence,
  not prerequisites.
- **PTS:** Markov supplies a known-ordered-state mechanism test;
  Count-Min/Frequent Items supply known-unordered-weighted state;
  HLL a known-unordered exact-limit test.
- **MOVE:** optional; drop if \S2 runs long.

---

### 3. Compression Trees and Local Laws

Generic framework, introduced only after the economic example is in
place. All T4 claims.

#### 3.1 Objects

- **Purpose:** set notation with minimal ceremony.
- **PTS:**
  1. $\Strings$, concatenation $\concat$, oracle $\fstar$, metric
     $\metric$.
  2. Summarizer $g$; readout $f$ ($f=\fstar$ closed-form; $f_{
     \mathrm{econ}}$ for Manifesto).
  3. Every $g$-call is local: leaf block or concatenation of two
     child summaries.
- **Reader:** ML + ST.
- **Tier:** T4.
- **SRC:** v2 \S3.1.
- **MOVE:** compress; push full notation table to App.~A.

#### 3.2 Tree Construction

- **Purpose:** define the executable object.
- **PTS:**
  1. Partition $x=b_1\concat\cdots\concat b_k$ into contiguous chunks.
  2. Leaves $s_i=g(b_i)$; internal nodes $s_u=g(s_{u_L}\concat s_{u_R})$;
     only summaries stored.
  3. Observations may be root-level labels (T1) or node-level audit
     labels (T2); raw spans are retained conceptually for the latter.
- **Reader:** ML.
- **Tier:** T4.
- **SRC:** v2 \S3.2.
- **FIG:** `fig:min-plain-tree` = `01_base.pdf`.
- **MOVE:** keep figure; compress prose.
- **Guards:** distinguish T1/T2 observation units.

#### 3.3 Oracle Equivalence and Context Compatibility

- **Purpose:** define ``same enough''.
- **PTS:**
  1. $u\sim v\iff\metric(\fstar(u),\fstar(v))=0$; use
     $\sim_{\mathrm{econ}}$ for the running example.
  2. All preservation is modulo the target, not literal string
     equality --- prevents generic semantic-similarity overclaims.
  3. Assumption~\ref{ass:context}: equivalence survives insertion
     into the same left/right context.
- **Reader:** ML + ST.
- **Tier:** T4.
- **SRC:** v2 \S3.4.
- **EQ:** $\sim$ definition and Assumption~\ref{ass:context} display.
- **Guards:** G9 (name discourse-sensitive failures).

#### 3.4 C1 / C2 / C3

- **Purpose:** make the laws exact.
- **PTS:**
  1. C1 (sufficiency): $g(b)\sim b$ for every realized leaf.
  2. C2 (idempotence): $g(s)\sim s$ for every stored summary.
  3. C3 (merge consistency):
     $u\concat v\sim g(u\concat v)\sim g(g(u)\concat g(v))$.
  4. Global law vs.\ realized law: audits test realized calls, not
     universal statements.
- **Reader:** ML + ST.
- **Tier:** T4.
- **SRC:** v2 \S3.3, equations
  \eqref{eq:leaf_sufficiency}/\eqref{eq:leaf_idempotence}/\eqref{eq:leaf_compatibility}.
- **EQ:** all three.
- **MOVE:** keep equations; `fig:local-laws-full` pushed to App.~A.
- **Guards:** G9.

#### 3.5 Assumption Bundles (one paragraph)

- **Purpose:** name the bundles \S4 consumes.
- **PTS:** Preservation (C1+C3+context) → Recompression (+C2) →
  Optimization (+factorization/measurability) → Certificate
  (+transport/calibration/sampling). Structural (1--2) vs.\ downstream
  (3--4).
- **Tier:** T4.
- **SRC:** v2 \S3.5 + `tab:assumption-bundles`.
- **MOVE:** one paragraph in main; full table to App.~A.

---

### 4. What the Guarantees Say

Theorem statements in main, proofs in App.~C. Target: 1--1.5 pages.
All T4.

#### 4.1 Theorem Bundle Setup

- **Purpose:** avoid a wall of unexplained theorems.
- **PTS:**
  1. Preservation is structural (C1 + C3 + context → root).
  2. Optimization equivalence follows from oracle-measurability.
  3. Audits turn approximate preservation into a finite-sample
     certificate.
- **SRC:** v2 \S8 opener.
- **MOVE:** short intro; optional `fig:min-theorem-deps` schematic.

#### 4.2 Inductive Preservation (Thm.~\ref{thm:one-pass})

- **Purpose:** state the structural result.
- **PTS:**
  1. Realized-valid Preservation Stack ⇒
     $\E[\metric(\fstar(Z^{(1)}),\fstar(x))]=0$.
  2. Adding C2 extends preservation to every re-summarization round
     $Z^{(R)}$.
  3. Expectation covers stochastic summarizers; node-level
     certificates (T2) complement root-observed corpus evaluation
     (T1).
- **EQ:** root preservation display.
- **SRC:** v2 \S8.1 Thm.~\ref{thm:one-pass}, Thm.~\ref{thm:multi-round}.
- **MOVE:** keep statement; push proof to App.~C.
- **Guards:** G2, G8.

#### 4.3 Schedule Invariance

- **Purpose:** link associativity/local laws to chunk schedule.
- **PTS:**
  1. For a fixed partition, any two reduction schedules satisfying
     the same realized local laws produce the same expected oracle
     value at the root.
  2. This is oracle-level, not byte/string equality.
  3. Underwrites leaf-size-robustness diagnostics in \S5.3.
- **SRC:** v2 \S8.1 Cor.~\ref{cor:schedule}, Cor.~\ref{cor:folds}.
- **MOVE:** one corollary in main; fold-of-folds variant to App.~C.

#### 4.4 Preference Alignment --- `PaperPreferenceStack`

- **Purpose:** keep DPO/GRPO result without application sprawl.
- **PTS:**
  1. DPO/GRPO application bundles supply oracle-measurability and
     generator-indexing premises.
  2. `PaperPreferenceStack` residual $0$ gives identical summary /
     full argmins
     (`paper_preference_stack_same_argmin`).
  3. Residual $\varepsilon$ gives full-objective $2\varepsilon$
     optimality for exact summary minimizers
     (`paper_preference_stack_summary_argmin_full_epsilon`).
- **Public:** `PaperPreferenceStack`,
  `paper_preference_stack_same_argmin`,
  `paper_preference_stack_summary_argmin_full_epsilon`.
- **SRC:** v2 \S8.2.
- **MOVE:** keep statements; push proof, transport constants, and
  `tab:measurability` to App.~C.
- **Guards:** G6.

#### 4.5 Gap Bound and `PaperErrorStack`

- **Purpose:** turn approximate preservation into a reported bound.
- **PTS:**
  1. Define $\Delta_R := \E[\metric(\fstar(Z^{(R)}),\fstar(x))]$.
  2. Unified gap: $|G_{\mathrm{meth}}|\le C_{\mathrm{meth}}\cdot\Delta_R$
     under transport Lipschitz + boundedness.
  3. Deployed certificate (`PaperErrorStack`):
     $|G|\le C_{\mathrm{meth}}\hat\Delta_R + B_{\mathrm{cal}}
     + B_{\mathrm{est}} + B_{\mathrm{clip}}$
     (`paper_error_stack_high_prob`).
- **Public:** `PaperErrorStack`, `paper_error_stack_high_prob`.
- **EQ:** both displays.
- **SRC:** v2 \S8.3, v2 \S8.5.
- **MOVE:** keep statements; push DPO worked example and
  method-specific transport constants to App.~C.
- **Guards:** G7 (do not revert to bare $L\hat\Delta_R$).

#### 4.6 Lean Crosswalk Note

- **Purpose:** keep formalization visible but not intrusive.
- **PTS:**
  1. App.~C maps C1/C2/C3 → Lean L1/L3/L2.
  2. State existence is an application assumption.
  3. Lean proves conditional preservation, `PaperPreferenceStack`,
     `PaperErrorStack`.
- **Public:** `PAPER_TO_LEAN_MAP.md`.
- **MOVE:** one paragraph.
- **Guards:** G1.

---

### 5. Manifesto Results (T1)

Target: pipeline paragraph + headline table + robustness + economic
ladder + claim-boundary paragraph. No main-text figure unless space
allows.

#### 5.1 Pipeline Paragraph

- **Purpose:** connect the running economic state to the experiment.
- **PTS:**
  1. Character chunks $c\in\{4,8,16,24,32,64\}$K; dimension-specific
     rubric summaries; recursive pairwise merges; root scored on the
     seven-point economic/rubric scale.
  2. Single model: Gemma-4-31B-IT-NVFP4.
  3. Correlation against Benoit expert-survey targets observed at the
     full-document root on the sampled corpus.
- **Tier:** T1.
- **SRC:** v2 \S9.3.
- **MOVE:** keep.
- **Guards:** G3.

#### 5.2 Headline Six-Dimension Table

- **Purpose:** preserve the main empirical benchmark.
- **PTS:**
  1. $8$K per-dim C-Tree macro $0.829$.
  2. Proprietary ensemble $0.817$; matched open-weight Gemma-3-27B
     $0.793$.
  3. Economic row at $8$K: $r=0.939$; split-expert economic
     reference $0.880$.
- **Tier:** T1.
- **TAB:** `tab:min-benoit-headline`.
- **Numbers:** `assets/benoit/tables/benoit_comparison_pearson.tex`
  (and `.md`).
- **SRC:** v2 \S9.4.
- **MOVE:** keep table.
- **Guards:** G3.

#### 5.3 Granularity / Leaf-Size Robustness

- **Purpose:** make the result about local structure, not only
  correlation.
- **PTS:**
  1. Macro spread $0.027$ across $4$K--$64$K character leaves.
  2. Economic stays high across the same sweep.
  3. Stable root scores as leaves shrink make local auditing
     feasible.
- **Tier:** T1 (diagnostic for T2 feasibility).
- **Numbers:** `assets/benoit/tables/benoit_comparison_pearson.tex` +
  `chunk_sweep_per_dim.md`.
- **SRC:** v2 \S9.4 ``Leaf-size invariance''.
- **MOVE:** keep.
- **Guards:** G2.

#### 5.4 Economic Prompt-Ladder Plateau

- **Purpose:** the single-dimension story this flow centers.
- **PTS:**
  1. Best economic external $r = 0.885, 0.886, 0.886, 0.879$ at
     $1024/2048/4096/8192$-token leaves; $0.007$-wide band around the
     $0.880$ split-expert reference.
  2. Below $1024$ tokens, drops to $0.861$ (leaf $512$) and $0.830$
     (leaf $256$): semantic analogue of the Markov ``leaves wide
     enough'' constraint.
  3. Token axis ≠ character axis; do not mix.
- **Tier:** T1.
- **Numbers:** `assets/benoit/tables/manifesto_fg_ladder.tex`.
- **SRC:** v2 \S9.5.
- **MOVE:** keep short.
- **Guards:** G2.

#### 5.5 Claim Boundary

- **Purpose:** enforce G2 in the main text.
- **PTS:**
  1. Headline is complete as a root-observed external-validation
     result (T1).
  2. Node-level C1/C2/C3 auditing is a stronger, separate tier (T2),
     not a prerequisite for the Tier~1 claim.
  3. Quasi-sentence local-law supervision is the planned T2/T3 path.
- **Tier:** T1 boundary.
- **SRC:** v2 \S9.6--9.7.
- **MOVE:** keep; this is the enforcement paragraph.
- **Guards:** G2, G3.

---

### 6. Validation Anchors (Markov, Frequency, HLL)

Short. One subsection per anchor plus one alignment table.
Appendices E and F carry detail.

#### 6.1 Markov Mechanism (T3)

- **Purpose:** show \emph{why} a learned state must carry boundary
  information.
- **PTS:**
  1. \emph{Hamlet} at scene granularity: each scene carries a
     dominant emotional register from
     `{Mourning, Foreboding, Political, Intrigue, Madness, Action,
     Comic}`. The oracle counts register shifts across adjacent
     scenes.
  2. Sufficient state is `(internal shifts, first register, last
     register)`. Canonical scene-boundary flips:
     3.3 $\to$ 3.4 (Intrigue $\to$ Action: Claudius-at-prayer $\to$
     Polonius-killed) and 4.7 $\to$ 5.1 (Intrigue $\to$ Comic:
     poisoned-foil plot $\to$ gravediggers). Either summary
     dropping its boundary register erases the shift across the
     act boundary.
  3. Budget-split result: $\sim 50\%$ of root labels can be replaced
     by aligned local labels at no cost to root MAE --- when the
     task matches the local laws.
- **Tier:** T3 (observation unit is node labels by design).
- **EQ:** boundary correction
  $c_L + c_R + \One\{L.\mathrm{last}\ne R.\mathrm{first}\}$.
- **FIG/TAB policy:** no main-text figure. The refreshed register-arc
  figure (`fig:min-markov-registers`,
  `assets/markov/figures/markov_scene_arc_hamlet.pdf`) lives in
  App.~E and is cross-referenced from the PTS 2 prose here. The
  textual illustration of the Acts~3$\to$4 / Acts~4$\to$5 boundary
  flips stands alone in main.
- **SRC:** v2 \S4.1--4.3, v2 \S6.
- **MOVE:** compress to one column plus equation; push synthetic
  DGP grids to App.~E. The App.~E DGP still uses neutral labels;
  the Hamlet mapping lives in App.~E.3.
- **Guards:** G10.

#### 6.2 Line-Weighted Character Frequency (Count-Min + Frequent Items)

- **Purpose:** one-paragraph taxonomy pointer. With the
  two-anchor pivot, Count-Min / Frequent Items are \emph{not} main
  anchors --- HLL carries the classical sketch slot. This
  subsection exists only so the main text signposts the
  frequency-sketch family before the appendix details.
- **PTS:**
  1. Frequency-style Hamlet queries (``how many lines does
     \emph{X} speak?'', ``who are the top-5 speakers?'') live in
     the Count-Min / Frequent Items families, not in HLL. State
     is a hash-count table or a heavy-hitter sketch; merge is
     associative + commutative by accumulation; \emph{not}
     idempotent.
  2. These are classical parity anchors, retained as taxonomy
     references so the framework covers point frequency and
     heavy-hitter queries; see App.~F.6 for the full taxonomy
     and App.~F.9 for the sketch gallery.
  3. Horatio and the Ghost remain useful illustrations in the
     appendix taxonomy (Horatio = per-scene-top-$k$ miss; Ghost
     = idempotence-failure case) but do not carry main-text
     weight.
- **Tier:** T1 (appendix tables) + T3 (state interpretation) ---
  but the main-text contribution is a taxonomy pointer, not a
  result.
- **Numbers:** all generated tables / RMSE figures live in App.~F
  (see App.~F.1 / F.9).
- **FIG/TAB policy:** no main-text figure or table. Point readers
  to App.~F.
- **MOVE:** one short paragraph in \S6; full treatment in App.~F.
- **Guards:** G11 (never call these idempotent), G5.

#### 6.3 HLL as the Classical Mergeable-Sketch Anchor (T1)

- **Purpose:** HLL is \emph{the} main-text classical-sketch anchor
  (paired with Markov in \S6.1). It isolates the state-versus-value
  distinction --- scalar cardinality is not mergeable, but a
  bounded register state is --- and pins down the
  noisy-but-valid-readout pattern that the Manifesto analogue
  inherits under a learned state.
- **PTS:**
  1. HLL answers distinct-element cardinality, not line counts or
     character importance. On \emph{Hamlet} the load-bearing query
     is ``how many distinct character co-appearance pairs ever
     share a scene'': each scene emits a set of unordered pairs
     $\{(c_i,c_j): c_i,c_j \text{ both speak}\}$, merging two acts
     unions those sets, and the readout is union cardinality.
     ``Distinct speaking roles'' ($\approx 24$) is the simpler
     warm-up; it is enumerable on one play and is not the
     motivating case.
  2. State is an HLL register array; merge is pointwise register
     max, which exactly instantiates set-union cardinality
     approximation --- not an ad hoc choice.
  3. Why HLL earns its place on this object: the pairs question
     grows as $O(|C|^2)$, and on a \emph{corpus} of plays the
     union-set balloons while the register array stays fixed-size.
     Hamlet alone shows the mechanism; the corpus case shows the
     payoff.
  4. RSE for $p=14$ is about $0.81\%$ --- write ``under $1\%$''.
  5. Valid compression preserves noisy sketch/readout behavior;
     it does not remove estimator noise.
- **Tier:** T1.
- **Numbers:** RSE $=1.04/\sqrt m$; keep detail in App.~F.
- **SRC:** v2 \S7; `appendix/minimal/F_hll_details.tex`.
- **FIG/TAB policy:** no main-text figure. Point the reader to
  `fig:min-hll-parity` in App.~F
  (`assets/hll/figures/hll_merge_learning_memory_median.png`)
  for the learned-merge-vs-exact-HLL memory curve --- the
  clearer single-panel predecessor of the five-panel
  `hll_parity_curves.png`.
- **MOVE:** compress to one paragraph in main; App.~F carries the
  figure, estimator derivation, and implementation-parity details.

#### 6.4 State-Alignment Table and Bridge Back to Manifesto

- **Purpose:** prevent the examples from feeling disconnected.
- **PTS:** three rows --- the two main anchors on one play plus
  the Manifesto analogue --- each row `(query | state | merge
  target | failure mode)`:
  1. \emph{Hamlet} register arc (Markov) --- ordered boundary
     state; hand-written merge with boundary correction; failure
     is a dropped first- or last-register field.
  2. \emph{Hamlet} distinct character co-appearance pairs (HLL)
     --- per-scene set of co-appearing speaker pairs, merged by
     set-union across acts; state is an HLL register array with
     pointwise-max merge approximating union cardinality; failure
     is estimator noise beyond the $1.04/\sqrt m$ floor. The
     simpler ``distinct speaking roles'' is the enumerable
     warm-up.
  3. Manifesto economic policy --- ordered semantic evidence plus
     measurement noise; \emph{learned} state, audited by the same
     C1/C2/C3 machinery the hand-written merges above satisfy by
     construction; failure surfaces as audited distortion, not as
     a wrong closed-form merge.

  A fourth row for Count-Min / Frequent Items lives in App.~F
  only; do not include it here, since \S6.2 is now a taxonomy
  pointer rather than an anchor.
- **TAB:** `tab:min-state-alignment` (main-text table #2).
- **MOVE:** keep.
- **Cross:** back to \S2 (state) and forward to \S8 (audit).

---

### 7. Algebraic and Representation Backstop

Target: 1 page. All T4.

#### 7.1 State Versus Value

- **Purpose:** retain the distinct-count warning without leading
  with algebra.
- **PTS:**
  1. States compose; scalar values often do not.
  2. Distinct-count warning: $|A\cup B|\not=$ a function of $|A|,|B|$.
  3. For Manifesto, child economic scores do not merge into the root
     economic score --- merge is on \emph{state}.
- **SRC:** v2 \S2.2.
- **MOVE:** one paragraph in main; full treatment in App.~B.
- **Guards:** G5.

#### 7.2 Ordered Homomorphism

- **Purpose:** cite the ordered generalization.
- **PTS:**
  1. $h(u\concat v)=h(u)\odot h(v)$.
  2. Associative required; commutative not.
  3. Ordered text behaves like Markov, not HLL, unless the task is
     symmetric.
- **EQ:** $h(u\concat v)=h(u)\odot h(v)$.
- **SRC:** v2 \S2.2 on \citet{Gibbons1996ThirdHomomorphism}.
- **MOVE:** one paragraph.

#### 7.3 Mergeable Reduction Proposition

- **Purpose:** preserve the formal bridge.
- **PTS:**
  1. Strict oracle-value homomorphism is special (requires $\fstar$
     to have a merge).
  2. Classical state-level mergeability is broader.
  3. C-TreePO reduces to the classical sketch when $g$ serializes
     the state --- what HLL exhibits in \S6.3.
- **SRC:** v2 \S2.5, Prop.~\ref{prop:mergeable-reduction}.
- **MOVE:** proposition in main; proof + `tab:sketch-mapping` →
  App.~B.
- **Guards:** G5.

#### 7.4 Learned Representation Note

- **Purpose:** keep FNO / neural-operator / prompt-program story
  visible but deferred.
- **PTS:**
  1. A representation \emph{proposes} a state; local laws define
     what it must preserve; the audit estimates realized violations.
  2. FNOs (Markov), register-based HLL (\S6.3), prompted LLMs (\S5)
     are three instances of the same $f,g$ interface.
  3. Neural operators certify deterministic realizers on compact
     realized calls; the stochastic theorem stack is the general
     surface.
- **SRC:** v2 \S5; v2 \S8 Prop.~\ref{prop:neural-operator-bridge};
  Prop.~\ref{prop:law-constrained-no}.
- **MOVE:** one paragraph. Full bridge, transfer moduli, transformer
  inclusion, overlap diagram → App.~D.
- **Guards:** G8.

---

### 8. Audit and Label Budget (T2 home)

Target: one paragraph per audit component.

#### 8.1 Finite Population of Local Checks

- **Purpose:** operationalize the certificate and anchor the
  tier-distinction in a single place.
- **PTS:**
  1. Root-observed corpus evaluation samples documents and observes
     root targets (T1).
  2. Local-law certification samples leaves, summaries, and merge
     nodes with logged propensities (T2).
  3. HT estimator:
     $\hat\mu_{\mathrm{HT}}=\frac1N\sum_{i\in\mathcal U}\frac{Z_i}{\pi_i}Y_i$.
- **Tier:** T1 + T2 side-by-side.
- **EQ:** HT display.
- **FIG:** optional `fig:min-audit-pipeline` schematic.
- **SRC:** v2 \S10.
- **MOVE:** keep; push per-law HT formulas for
  $\hat p_{\mathrm{suff}},\hat p_{\mathrm{assoc,raw}},
  \hat p_{\mathrm{assoc,joint}},\hat p_{\mathrm{idem}}$ to App.~H.
- **Guards:** G2 (be explicit about the tier distinction).

#### 8.2 Paragraph/Section Labels as Local Measurements

- **Purpose:** state the granularity-reduction benefit.
- **PTS:**
  1. Node-level local labels pose economic-preservation questions at
     paragraph / section / merge-node granularity.
  2. They support certificates (T2) and label-budget substitution
     (T3) when aligned with the root oracle.
  3. They do not replace calibration to the full-document expert
     target.
- **Tier:** T2 + T3.
- **SRC:** v2 \S10.3, v2 \S13.2.
- **MOVE:** keep.
- **Guards:** G10.

#### 8.3 Certificate Decomposition

- **Purpose:** explain the four envelope terms.
- **PTS:**
  1. Empirical distortion $\hat\Delta_R$: audited distance from the
     theorem-eligible subspace.
  2. Judge calibration $B_{\mathrm{cal}}$: irreducible measurement-
     instrument gap.
  3. Sampling envelope $B_{\mathrm{est}}$: finite-sample radius.
  4. Clipping bias $B_{\mathrm{clip}}$: from variance-control
     choices.
- **Public:** `PaperErrorStack`, `paper_error_stack_high_prob`.
- **EQ:**
  $|G|\le C_{\mathrm{meth}}\hat\Delta_R + B_{\mathrm{cal}}
  + B_{\mathrm{est}} + B_{\mathrm{clip}}$.
- **SRC:** v2 \S13.1.
- **MOVE:** keep; push `tab:reporting-template` and stress grid to
  App.~H.
- **Guards:** G7.

---

### 9. Applications and Scope (T5)

Four short paragraphs.

#### 9.1 Application Class

- **Purpose:** generalize cautiously.
- **PTS:**
  1. Long, sectioned documents with rubric targets.
  2. Some datasets observe root labels (T1); others require
     node-level audit labels (T2).
  3. Examples: manifestos, legal, clinical, policy reports.
- **SRC:** v2 \S9a, v2 \S13.4.
- **MOVE:** keep.
- **Guards:** G2.

#### 9.2 Surrogate-Label Risk

- **Purpose:** connect the social-science caution about imperfect
  labels.
- **PTS:**
  1. Correlation alone is not enough for downstream inference
     (\citet{EgamiEtAl2024}).
  2. C-TreePO separates compression error ($\hat\Delta_R$) from
     judge calibration error ($B_{\mathrm{cal}}$) via \S8.3.
- **SRC:** v2 \S9a.
- **MOVE:** keep.

#### 9.3 Neighboring Methods

- **Purpose:** shrink related work to a navigational paragraph.
- **PTS:**
  1. Long context reduces truncation but not certification.
  2. RAG retrieves but does not certify omitted evidence.
  3. Tree compression / fusion lacks task-oracle audit.
- **SRC:** v2 \S12.
- **MOVE:** one paragraph; full citations to App.~I.

#### 9.4 Failure Modes

- **Purpose:** make assumption failures legible.
- **PTS:**
  1. C1 fails when leaves miss evidence.
  2. C3 fails when child summaries drop cross-span tension.
  3. Oracle-measurability fails when annotators respond to style or
     tone outside the oracle target.
- **SRC:** v2 \S13.3.
- **MOVE:** keep.
- **Guards:** G9.

---

### 10. Conclusion

Two short paragraphs.

#### 10.1 Restate Minimal Claim

- **Purpose:** close cleanly.
- **PTS:**
  1. Compression should preserve task-relative state.
  2. Manifesto economic state is the guiding root-observed example
     (T1).
  3. Local laws make the stronger node-level certificate (T2)
     auditable.
- **MOVE:** keep.
- **Guards:** G2.

#### 10.2 Granularity Close

- **Purpose:** end on \emph{why this matters}.
- **PTS:**
  1. Valid local structure moves work from whole documents to
     smaller units.
  2. Approximate validity yields a measured gap, not a silent
     failure.
  3. Markov / frequency / HLL / Manifesto mark the known-ordered,
     known-unordered-weighted, known-unordered-cardinality, and
     learned-state cases.
- **MOVE:** keep.

---

## Appendix Blueprint

Each appendix mirrors the structure of its main-text section, then
expands: every statement gets its proof, every figure its per-cell
grid, every protocol its full detail.

### Appendix A: Notation and Assumption Bundles

- A.1 Manifesto running-example restatement (state / readout /
  oracle / summarizer, under $\sim_{\mathrm{econ}}$). Root-observed
  vs.\ node-level distinction; $S_{\mathrm{econ}}$ assumed, not
  proved.
- A.2 Notation recap table (`tab:notation-main`).
- A.3 Four assumption bundles in full (`tab:assumption-bundles`) ---
  each bundle corresponds to a public theorem stack.
- A.4 Granularity vocabulary (paragraph / section / chapter / book ↔
  leaf / internal-merge / sub-root / root); distinguish document-
  sampling units from tree-audit units.
- A.5 Full three-panel C1/C2/C3 figure `05_full.pdf`.

### Appendix B: Algebraic Background

- B.1 State-level mergeability: encode / merge / query discipline,
  database algebraic aggregates, MUD, full citations.
- B.2 HLL primer: register construction; harmonic-mean estimator
  $\hat n = \alpha_m m^2 / \sum_j 2^{-r_j}$; $\mathrm{RSE}=1.04/\sqrt m$.
- B.3 Ordered text: Gibbons list homomorphism and non-commutative
  merge.
- B.4 State vs.\ value: distinct-count warning (canonical location
  for ``do not merge scalar child scores'').
- B.5 Classical sketch catalog: HLL, Count-Min, KLL, GK, Bloom,
  Theta/KMV, t-digest, REQ, Tuple, VarOpt, CPC.
- B.6 Reduction proof sketch for Prop.~\ref{prop:mergeable-reduction}:
  strict case via A1+A2 through context compat.; classical case via
  bounded sketch state composition. Make clear this is a sufficient
  special case, not the general Manifesto setting.
- **SRC:** v2 \S2; `appendix/B_proofs.tex`.

### Appendix C: Proofs and Formalization

- C.1 Inductive preservation proof (Thm.~\ref{thm:one-pass}). Match
  public wording in \S4; note stochastic summarizers modelled through
  PMFs.
- C.2 Multi-round preservation (Thm.~\ref{thm:multi-round}),
  schedule invariance / fold-of-folds. State equality is oracle
  equality, not literal summary equality.
- C.3 C2 independence: two-token counterexample
  (`appendix/D_counterexample.tex`).
- C.4 `PaperPreferenceStack` proof
  (Thm.~\ref{thm:pref-equiv}) + `tab:measurability`. Cite the stack
  first, then method bundles.
- C.5 Unified gap (Thm.~\ref{thm:unified-gap}) + DPO worked example.
- C.6 `PaperErrorStack` (Thm.~\ref{thm:e2e}): concentration,
  calibration / estimation / clipping decomposition, method-specific
  transport constants for DPO / GRPO-PL / GRPO-RL.
- C.7 Projection ↔ structured-error iff
  (`app:projection-interpretation`).
- C.8 Lean crosswalk: C1/C2/C3 ↔ L1/L3/L2;
  `PaperPreferenceStack`, `PaperErrorStack`, application bundles;
  verify against `lean3/docs/PAPER_TO_LEAN_MAP.md`.

### Appendix D: Neural-Operator Realization

- D.1 Why neural operators are appendix-level: realization /
  approximation machinery, not the first statement of preservation.
- D.2 Equation-(6) architecture and compact realized-call sets ---
  phrase as deterministic realizer certification.
- D.3 Kovachki Lemma 22 / Theorem 11 / Theorem 13 paraphrases and
  uniform-continuity specialization.
- D.4 Transfer moduli $\omega_{\mathrm{leaf}},\omega_{\mathrm{merge}},
  \omega_{\mathrm{idemp}}$ --- tie projection error to the public
  error stack only through stated envelopes.
- D.5 Transformer inclusion (Kovachki Prop.~6 analog).
- D.6 Overlap: $\mathcal N\cap\mathrm{MS}\subseteq\mathrm{ExactLL}_{
  \mathcal N}(T,\fstar)$; finite transformer stacks on the neural-
  operator side.
- D.7 Notation map (`tab:ctreepo-neural-operator-notation`).
- **SRC:** v2 \S5, v2 \S8 propositions,
  `appendix/I_operator_overlap.tex`.
- **Do not** claim a randomized neural-operator theorem unless one
  is added.

### Appendix E: Markov Details

- E.1 DGP (main prose uses emotional-register analogy; here use
  neutral color/register labels): $4$-color vs.\ $12$-color,
  $h=0.039/0.079$, $10{,}240$ training docs. Full `tab:dgp`.
- E.2 Sufficient state (internal shifts, first register, last
  register) and ordered merge; boundary-correction derivation.
- E.3 Relation to the \emph{Hamlet} analogy. Map the
  seven-register Hamlet labels from \S6.1
  (`{Mourning, Foreboding, Political, Intrigue, Madness, Action,
  Comic}`) to the neutral color/register labels used in the
  synthetic DGP (one-to-one when the palette sizes match;
  one-to-many collapse documented otherwise). Record the Acts
  3$\to$4 and Acts 4$\to$5 boundary illustrations as the
  reader-facing anchor, and note explicitly that the empirical
  Markov numbers in \S6.1 come from the synthetic DGP, not from
  any annotated \emph{Hamlet} text corpus --- the play supplies
  intuition only.
- E.4 Architecture: $128$-wide Fourier, $8$ modes, $4$ layers;
  stage-1/stage-2 schedule; shared backbone; parity check.
- E.5 Empirical role of local supervision --- Markov deliberately
  observes node-level labels to test T3.
- E.6 Budget-allocation policies + token-mass accounting.
- E.7 Full grids: `markov_simple_leaf_mass.png`,
  `markov_hard_leaf_mass.png`, `markov_budget_split.png`.
- E.8 Mechanism checks (`appendix/G_mechanism_checks.tex`).
- E.9 Failure mode: dropped boundary state (canonical C3 failure).

### Appendix F: Frequency, HLL, and Classical Sketch Details

- F.1 Line-weighted character frequency (taxonomy, not a main
  anchor).
  - `(character, lines)` updates; Count-Min point queries; Frequent
    Items heavy hitters; repeated-token proxy in the benchmark.
  - Didactic pair on the \emph{Hamlet} object: \textbf{Horatio}
    is the medium-frequency recurring speaker missed by per-scene
    top-$k$ summaries, and \textbf{the Ghost} is the
    clustered-presence speaker whose repeated evidence inside
    1.1/1.4/1.5/3.4 must accumulate because the sketch merge is
    not idempotent. These illustrations live entirely in App.~F
    under the two-anchor pivot.
  - Weighted-update implementation support
    (`treepo/src/treepo/sketches/adapters/datasketches_frequency.py`)
    recorded as provenance, not theorem.
  - Key generated-table evidence: Count-Min RMSE
    $0.0527 \to 0.0182 \to 0.000775$; Frequent Items official RMSE
    $=0$ for `top5_point_frequency` at medium/large capacity.
  - Cross-reference: the main-text HLL anchor is in \S6.3; the
    visual catalog for frequency / cardinality / quantile / set /
    sampling sketches is `fig:sketch-gallery` in App.~F.9.
- F.2 HLL as cardinality backstop --- distinguish distinct speaker
  breadth from character importance / line counts.
- F.3 Classical HLL registers, estimator, RSE. Include $p=14$
  under-$1\%$ RSE.
- F.4 Native byte parity vs.\ DataSketches encoding caveat (list /
  sparse / dense transitions depend on build order).
- F.5 Learned variants: learned-$g$ (classical $f$) vs.\ learned-$g+f$
  (end-to-end); register-space optimization barrier. Evidence for
  representation learning, not proof that arbitrary learned summaries
  are valid.
- F.6 Query taxonomy on the \emph{Hamlet} object.
  - Count-Min: point frequency (``how many lines does Horatio
    speak across the play?''). Frequent Items: heavy hitters
    (``who are the top-5 speakers?'').
  - HLL: distinct-element cardinality under set-union merge. Two
    queries, a warm-up and a load-bearing one:
    - Warm-up: ``how many distinct speaking roles appear''
      ($\approx 24$). Enumerable on one play; HLL is overkill.
      Kept because it motivates the cardinality shape.
    - Load-bearing: ``how many distinct character co-appearance
      pairs ever share a scene''. Per-scene emission is the set
      of unordered speaker pairs on stage; per-act merge is
      set-union; HLL's register-array with pointwise max
      approximates union cardinality without storing the
      union-set. Grows in a corpus of plays --- which is where
      HLL actually saves memory.
  - Learned semantic state: narrative / measurement importance
    (``which characters carry the play's emotional arc?'').
  - ``How many lines does Hamlet speak?'' and ``which characters
    account for most lines?'' are sketch queries and are resolved
    by Count-Min / Frequent Items on the scene-level
    `(character, lines)` stream. ``Which characters matter to the
    play's meaning?'' requires a task-specific oracle and is a
    learned-state question, not a sketch question.
  - Horatio and the Ghost remain the didactic pair: Horatio as the
    per-scene-top-$k$ miss, the Ghost as the idempotence-failure
    case.
- F.7 Full per-cell HLL grid (`tab:classical-parity-hll`), seeds, all
  precisions $p\in\{7,9,11,13\}$, leaf counts
  $L\in\{1,2,4,8,16\}$.
- F.8 HLL merge-learning memory curve
  (`hll_merge_learning_memory_median.png`). Single-panel plot,
  x-axis = HLL memory (bytes), y-axis = relative RMSE, three
  series: \emph{Learned merge (median)}, \emph{Exact HLL},
  \emph{Theory floor}. Learned merge tracks exact HLL across
  $p\in\{6,\ldots,12\}$ (memory $\approx 48$--$3072$ bytes); this
  is the main-axis ``learned merge recovers the classical sketch''
  result. The earlier five-panel `hll_parity_curves.png` stays
  available under `assets/hll/figures/` but is not included ---
  the single-panel predecessor is clearer as a didactic figure.
  The mean-over-seeds companion
  (`hll_merge_learning_memory.png`) can be added if a variance
  story is called for.
- F.9 Sketch task gallery (`fig:sketch-gallery`). Five-panel
  figure assembled from existing PNGs under
  `assets/sketches/figures/`:
  `classical_sketches_frequency.png`,
  `classical_sketches_distinct.png`,
  `classical_sketches_quantile.png`,
  `classical_sketches_set.png`,
  `classical_sketches_sampling.png`. Each panel keeps its existing
  axes; caption names the sketch family per panel (Count-Min /
  Frequent Items, HLL, KLL / GK quantiles, Theta / Bloom set
  membership, VarOpt / reservoir sampling) and the query family
  per panel (point frequency, cardinality, quantiles, set
  membership, weighted sampling). Converts the ``broader classical-
  sketch family'' bullet (CPC, Theta/KMV, KLL, classic quantiles,
  REQ, t-digest, Tuple, VarOpt via `treepo-bench`) from a textual
  list into a visual catalog. Cross-referenced from the \S6.2
  taxonomy pointer; the HLL-panel row pairs with
  `fig:min-hll-parity` in F.8.
- F.10 Learned-variant comparison (`fig:sketch-method-gallery`).
  Two-by-two figure from the four `classical_sketches_method_*.png`
  panels contrasting official / learned-$f$ / learned-$g$ /
  learned-joint. Cross-referenced from \S7.4 as the visual backing
  for the claim that FNOs, register-based HLL, and prompted LLMs
  are three instances of the same $f,g$ interface.
- F.11 Aggregate summary (`fig:sketch-summary`). Twin-panel figure
  combining `classical_sketches_summary.png` and
  `classical_sketches_gold_gap.png` as the ``what the whole bundle
  says'' overview.
- **Figure policy.** F.9--F.11 reuse PNGs already generated by
  `regen_classical_sketches.sh`; no new plot production. Visual
  check: each panel must be legible at the appendix column width
  before inclusion. If a panel is too dense, it stays available
  under `assets/sketches/figures/` but is not included.
- **SRC:** v2 \S7; `appendix/F_classical_parity.tex`; generated
  tables under `assets/sketches/tables/`; generated figures under
  `assets/sketches/figures/`.

### Appendix G: Manifesto Details

- G.1 Corpus and target: Benoit 235 manifestos + 23 Klüver coalition
  agreements; six dimensions; split-expert agreement reference.
  State \emph{document-level} target construction and sampled-corpus
  evaluation.
- G.2 Root-observed pipeline: per-dimension vs.\ joint variants;
  scoring-call reordering. Prefer ``root-observed pipeline'' over
  prompting-mechanics shorthand unless referring to prompting
  mechanics.
- G.3 Main numbers and chunk sweep (`tab:benoit-headline`,
  $c\in\{4,8,16,24,32,64\}$K); $\pm 0.027$ macro across $16\times$
  leaf variation. Cite the generated-table file for every number.
- G.4 Economic prompt ladder: Benoit-init
  (`manifesto_fg_ladder_benoit_init.png`), raw-init
  (`manifesto_fg_ladder_raw_init.png`), per-cell grids. Token leaves
  kept separate from character leaves.
- G.5 Compute accounting.
- G.6 Oracle-grounded quasi-sentence path --- presented as the T2/T3
  local-law route, \emph{not} a missing prerequisite for G.2--G.4.
- G.7 DPO/GRPO connection + Assumption~\ref{ass:pref} (oracle-indexed
  preferences).
- **SRC:** v2 \S9; `appendix/H_benoit_replication.tex`.

### Appendix H: Audit Details

- H.1 Document-sampling units vs.\ tree-audit units and propensities
  --- definitive distinction for root labels vs.\ node labels.
- H.2 HT estimator: full formulas for $\hat p_{\mathrm{suff}},
  \hat p_{\mathrm{assoc,raw}},\hat p_{\mathrm{assoc,joint}},\hat
  p_{\mathrm{idem}}$. Logged marginal propensities + positivity
  required.
- H.3 Calibration and local labels --- local labels must align with
  the same oracle.
- H.4 Envelope monotonicity: $C/\sqrt{n+1}$; global/local separate
  monotonicity.
- H.5 `PaperErrorStack` reporting template
  (`tab:reporting-template`):
  $C_{\mathrm{meth}}\hat\Delta_R + B_{\mathrm{cal}}
  + B_{\mathrm{est}} + B_{\mathrm{clip}}$.
- H.6 Scaling with tree size: union bound
  Eq.~\eqref{eq:error_budget}.
- H.7 Adversarial-sampling robustness: $\pi_{\min}$ / $w_{\max}$
  stress grid.
- H.8 Paragraph/section/chapter labels mapped to audit strata.
- **SRC:** v2 \S10, v2 \S13.1; `appendix/B_proofs.tex`.

### Appendix I: Related Work and Scope

- I.1 Mergeable summaries and data systems:
  \citet{Agarwal2013MergeableTODS, FeldmanEtAl2008MUD,
  GrayEtAl1997DataCube, FlajoletEtAl2007, HeuleEtAl2013,
  CormodeMuthukrishnan2005}.
- I.2 Sufficient statistics: \citet{Fisher1922, Blackwell1953}.
- I.3 Ordered / functional: \citet{Gibbons1996ThirdHomomorphism}.
- I.4 Social-science measurement and surrogate labels:
  \citet{EgamiEtAl2024, BenoitEtAl2025, ManifestoProject2025a}.
- I.5 Long-context / RAG / tree-compression:
  \citet{KnightMarcu2000SentenceCompression,
  BarzilayMcKeown2005SentenceFusion,
  ClarkeLapata2008ILPSentenceCompression,
  KuznetsovaEtAl2014TreeTalk}.
- I.6 Preference learning: \citet{BradleyTerry1952,
  RafailovEtAl2023DPO}.
- I.7 Scope boundary and expected failures.

---

## Revision Checklist (merge with the new BLUEPRINT.md checklist)

Run after any nontrivial edit to the minimal manuscript.

- **Tier separation.**
  - No sentence describes the Manifesto Tier~1 headline as
    ``prompt-only'', ``incomplete'', or ``missing a prerequisite
    audit''. (G2, G3.)
  - Node-level audit language (T2) uses ``tree-unit'', ``logged
    propensities'', ``realized-tree certificate''.
  - Local-label language (T3) always names the alignment target.
  - DPO/GRPO described as application bundles, not manifesto
    theorems. (G6.)
- **Certificate language.**
  - `rg -n 'L\\s*\\\\hat\\{\\\\Delta\\}_R' sections/minimal` returns
    nothing unless a method constant has been explicitly specialized
    to $L$. Otherwise use $C_{\mathrm{meth}}\hat\Delta_R + B_{
    \mathrm{cal}} + B_{\mathrm{est}} + B_{\mathrm{clip}}$. (G7.)
  - `PaperErrorStack` mentioned in any prose that discusses the
    certificate (App.~C/H).
  - `PaperPreferenceStack` mentioned in any prose about summary-vs-
    full argmin equivalence (App.~C).
- **Wording anchors.**
  - Positive: `root-observed`, `full-document expert target`,
    `tree-unit`, `logged propensities`, `local-law certificate`.
  - Negative: any occurrence of the phrases listed in \S0.b Avoid
    list is a bug.
- **Empirical numbers.**
  - All Manifesto numbers trace to `assets/benoit/tables/*`.
  - All sketch numbers trace to `assets/sketches/tables/*`.
  - Character chunk sizes and token leaf sizes never mixed.
- **Sketch framing.**
  - Count-Min / Frequent Items carry the Hamlet line-frequency
    anchor; HLL is the narrower cardinality backstop.
  - No sentence calls Count-Min or Frequent Items idempotent. (G11.)
  - Any cited sketch figure exists under
    `assets/sketches/figures/`; otherwise cite generated tables
    under `assets/sketches/tables/`.
  - Smoke test:
    `pytest treepo/tests/sketches/test_broad_classical_sketches.py -q`.
- **Figure / table budget.**
  - Main text: $\le 2$ figures and $\le 2$ tables (per \S0.g).
- **Cross-references.**
  - Every theorem has a matching proof label in App.~C.
  - Every empirical claim points to exactly one source (table or
    figure), not both.
- **Build.**
  - `cd paper/ctreepo && latexmk -pdf -interaction=nonstopmode
    main_minimal.tex` passes.
  - Log check:
    `rg -n "Undefined|undefined|Citation.*undefined|Reference.*undefined|Label\\(s\\) may have changed"
    main_minimal.log`.
- **Lean / doc crosswalk.**
  - If theorem names or public stacks change, update
    `lean3/docs/PAPER_TO_LEAN_MAP.md` and
    `docs/ctreepo_appendix_proof_audit.md`.
  - Do not edit `docs/ctreepo_python_code_map_for_llms.md` during a
    paper-wording pass.
