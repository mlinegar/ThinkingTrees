# C-TreePO Paper Handoff — 2026-04-29 (v9_polish)

Editorial pass on `main_v8.pdf` that landed as a new versioned draft
[`main_v9_polish.tex`](main_v9_polish.tex) + [`sections/v9_polish/`](sections/v9_polish/).
Picks up from [`HANDOFF_2026-04-16.md`](HANDOFF_2026-04-16.md). The current
handoff covers what changed in the v8 → v9_polish editorial session.

## Build

```bash
cd paper/ctreepo
pdflatex main_v9_polish && bibtex main_v9_polish && pdflatex main_v9_polish && pdflatex main_v9_polish
```

44 pages. Clean: no undefined refs, no duplicate labels, no STYLE.md violations.

## What changed (chronologically)

### 1. Initial editorial pass
Full rewrites of intro and conclusion + line-level STYLE.md compliance pass on
all v8 sections. Plan: [`/home/mlinegar/.claude/plans/consider-our-current-version-abundant-mountain.md`](/home/mlinegar/.claude/plans/consider-our-current-version-abundant-mountain.md).

- Created `main_v9_polish.tex` and `sections/v9_polish/*` (copies of v8 with edits applied).
- Labels stayed `sec:v8-*` / `eq:v8-*` so cross-refs to `appendix/v8/*` still resolve. **No appendix files were modified.**
- STYLE fixes: removed em-dashes, `not only`, meta-references to "the manuscript"/"the reader", throat-clearers, defensive disclaimers, and excess `rather than`s.

### 2. Polmeth-tracking pass
User asked for the abstract to track the [polmeth 2026 proposal](../../docs/proposals/polmeth_2026_ctreepo_v3.tex) more closely.

- Rewrote abstract to lead with "compressed surrogate" framing + DSL (Egami) bridge.
- Updated intro to match (compression-as-design problem; representation-level failure mode that DSL doesn't address).
- Removed all subscripted oracle notation (e.g. `f^*_{\mathrm{econ}}` → `f^*`).
- Threaded plain-English law names: **C1 (sufficiency)**, **C2 (idempotence)**, **C3 (merge consistency)** at every introduction site.

### 3. Supervision-granularity + error decomposition
- Added the "audit at any tree level" point to abstract and intro: an analyst can trade one full-document expert read for sampled reads of paragraphs/sections with comparable accuracy guarantees.
- Added a new intro paragraph that decomposes the certificate into four interpretable terms (sampled local-distortion, calibration, sampling, clipping) and points each at a specific lever.
- Expanded intro roadmap from `Section X gives Y` to a `\paragraph{Map of the paper and appendices.}` block listing all eight appendices A–G plus Lean crosswalk.

### 4. Readability pass on §1, §2, §8
- Retitled §2 from "Manifesto Policy States as the Running Mechanism" to **"Identifying Policy Positions in Political Manifestos"**.
- Rewrote §2 with concrete narrative examples (three-section "Suppose...Suppose a third..." cadence).
- Polished §1 phrasings; normalized "envelope" → "term" for the four certificate components.
- Polished §8 discussion (removed "is not harmless" defensive forms; fixed straw "fix is not necessarily a larger model").
- Polished §8 conclusion (split long sentence; "decentralization is the dimension that breaks" → "where shared compression breaks down").

### 5. v7_cld tone port
User flagged "is the empirical testbed" as awful and asked to port voice from older `v7_cld` draft. The cld/cdx versions only preserved manifesto-section partials (`sections/v7_cld/05_manifesto_results.tex`, `sections/v5_cld/05_manifesto.tex`, `sections/v6_cdx/05_manifesto_results.tex`); no preserved older intro/conclusion to copy from. Tone patterns extracted and applied:

- **First-person plural agent.** "We test", "we score", "we use", "we evaluate".
- **Short declarative sentences in series.** Subject-verb-object, period.
- **Concrete verbs.** "The project collects party platforms, releases document text, and releases hand-coded data" (was "The project releases…").
- **Colons deliver punchy diagnoses.** "split-expert row is a measurement-agreement reference: experts are split into two groups…"
- **Lead with claim, back with scorecard.** "Per-dimension trees reach…Both clear the matched open-weight baseline at 0.793, and the per-dimension tree clears the proprietary 18-score ensemble at 0.817."
- Restored the v7_cld four-verb cadence: "trained, read, sampled, and audited."
- Restored the "PO in C-TreePO" payoff in the abstract and intro.
- Recovered Benoit-pipeline context lost in the v8 → v9 transition: "Their pipeline first summarizes each manifesto–dimension pair into a short English summary, then scores that summary on the seven-point expert scale."

## Files and current state

| File | State |
|---|---|
| [main_v9_polish.tex](main_v9_polish.tex) | wrapper; abstract rewritten in v7_cld voice |
| [sections/v9_polish/01_introduction.tex](sections/v9_polish/01_introduction.tex) | full rewrite (polmeth-tracking, error decomposition, detailed roadmap, v7_cld voice) |
| [sections/v9_polish/02_manifesto_mechanism.tex](sections/v9_polish/02_manifesto_mechanism.tex) | retitled + rewritten with concrete narrative |
| [sections/v9_polish/03_ctree_math.tex](sections/v9_polish/03_ctree_math.tex) | small STYLE edits + plain-English law names |
| [sections/v9_polish/04_objective_theorem_ladder.tex](sections/v9_polish/04_objective_theorem_ladder.tex) | one STYLE edit (cross-ref tightening) |
| [sections/v9_polish/05_audit_certificate.tex](sections/v9_polish/05_audit_certificate.tex) | one defensive disclaimer dropped |
| [sections/v9_polish/06_manifesto_results.tex](sections/v9_polish/06_manifesto_results.tex) | first 4 paragraphs + Takeaways ported from v7_cld; **subsection bodies still in v8 voice** |
| [sections/v9_polish/07_related_scope.tex](sections/v9_polish/07_related_scope.tex) | unchanged from v8 |
| [sections/v9_polish/08_discussion.tex](sections/v9_polish/08_discussion.tex) | multiple polish edits |
| [sections/v9_polish/08_conclusion.tex](sections/v9_polish/08_conclusion.tex) | full rewrite + readability polish |
| [appendix/v8/](appendix/v8/) | **unchanged**, still inputted by `main_v9_polish.tex` |

## Roadmap for the next pass

In rough priority order:

1. **§6 middle subsections** still in v8 voice: `Tree Setup and Splits`, `Single-Dimension Results`, `Universal Summary`, `Local Diagnostics and Node-Label Route`. v7_cld's `sections/v7_cld/05_manifesto_results.tex` has more material to port. Notable phrase to recover: *"marking decentralization as teacher-bounded: the trained scorer fits its teacher trace better than the teacher itself fits gold."* This wasn't ported because v9_polish doesn't have the same teacher-trace numbers laid out — bring those numbers in and the phrase fits naturally.

2. **§3, §4, §5, §7 voice pass.** These are formal sections (math, objective, certificate, related work). They mostly read fine but use the passive "is the…" / "the X is Y" construction the user disliked. A pass to add `we` voice and active verbs would help, especially openers.

3. **Appendices.** `appendix/v8/*.tex` haven't gotten the STYLE pass or the tone pass. They're long and dense; might be worth a separate handoff session.

4. **Label rename** (low priority). All v9_polish files use `sec:v8-*` / `eq:v8-*` / `app:v8-*` labels so cross-refs to `appendix/v8/*` resolve. If a future pass renames the appendix to `appendix/v9_polish/` and renames labels accordingly, do it as a single global sed pass over all files.

5. **Decentralization stress data.** v7_cld had specific per-leaf-size correlation numbers for the decentralization run with concrete diagnoses. v9_polish has a more abstract "the local diagnostic gap flags that breakdown directly" framing. The concrete numbers would be more compelling if they're still accurate against the current data.

## STYLE.md gotchas for the next LLM

Always load [`STYLE.md`](STYLE.md) before any prose edit. Key rules that bit during this session:

- **No `not only X but Y`** — but trailing clarification `is X, not Y` is allowed.
- **Em-dash budget**: at most one or two per section; prefer parens or split sentences. The repo uses LaTeX `---` for em-dash; `--` is en-dash and is fine.
- **No meta-references** to "this paper", "the manuscript", "the reader". Cross-refs to specific sections are fine: `Section~\ref{sec:v8-X}`.
- **No defensive disclaimers**. Don't write `It does not define a new population target` if no one was claiming it did. Just leave the non-claim out.
- **`rather than` budget**: at most one per paragraph. Soft preference but it gets crowded fast.
- **Definitions/Propositions/Theorems contain only formal content**. Move motivation and "plain-language readings" outside the box.
- **No subscripted oracle notation** (user pushback): `f^\ast_{\mathrm{econ}}` is not OK; use plain `f^\ast` and put the dimension name in the surrounding prose.

Verification greps after any prose edit:

```bash
cd paper/ctreepo
grep -nE -- "---" sections/v9_polish/*.tex main_v9_polish.tex
grep -nE " not (only|just|merely) " sections/v9_polish/*.tex main_v9_polish.tex
grep -nE "(this paper|the manuscript|the reader|the empirical testbed)" sections/v9_polish/*.tex main_v9_polish.tex
grep -nE "(it is worth|importantly|crucially|in order to|in terms of|however)" sections/v9_polish/*.tex main_v9_polish.tex
```

All four should return zero hits except for the trailing-clarification variants of `not just` / `not only` (allowed).

## Inspirational source pointers

- [`docs/proposals/polmeth_2026_ctreepo_v3.tex`](../../docs/proposals/polmeth_2026_ctreepo_v3.tex) — the proposal voice the abstract and intro now track. Read first.
- [`sections/v7_cld/05_manifesto_results.tex`](sections/v7_cld/05_manifesto_results.tex) — primary v7_cld tone source. The §6 middle subsections still need this voice ported.
- [`sections/v5_cld/05_manifesto.tex`](sections/v5_cld/05_manifesto.tex) — secondary v7_cld-era voice source.
- [`HANDOFF_2026-04-16.md`](HANDOFF_2026-04-16.md) — earlier handoff, gives the broader paper context, figure inventory, and DGP reference points.

## Versioning convention

Per [`feedback_paper_non_destructive_versioning.md`](/home/mlinegar/.claude/projects/-home-mlinegar-ThinkingTrees/memory/feedback_paper_non_destructive_versioning.md):
**paper edits create a new `main_v<N>_<tag>.tex` + `sections/v<N>_<tag>/...`; never overwrite existing draft files.** This handoff's pass landed as `v9_polish`. The next substantive pass should land as `v10_<tag>` (or `v9_<another-tag>` if it's a parallel branch on top of v9_polish).
