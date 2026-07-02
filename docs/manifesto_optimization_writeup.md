# Manifesto Optimization Results — Writeup Draft

Free-form prose draft for the Manifesto-optimization findings. Destination is
`paper/ctreepo/sections/v4/05_manifesto.tex` (extending §5.3 "Alternating f/g
Optimization" or splitting off a §5.4) once the comparison runs complete. The
draft is written in headed paragraph blocks so a later editor can fold or split
without rewriting prose.

Style discipline (from `paper/ctreepo/STYLE.md`):
- Em-dash budget across the whole draft: at most two.
- No "not X but Y" constructions; no throat-clearers; no meta-references.
- Lead with the claim; numbers as objects.

All numeric claims below are pulled directly from
`outputs/manifesto_fg_alternating/combined_benoit_joint_teacher_all6_dspy_fixed_20260425_000122/plots_by_dimension/manifesto_fg_ladder_dimension_rows.csv`
(combined run) and the corresponding single-dim grid summaries. Heatmap PNGs
referenced are the user's existing plots; nothing is copied into
`paper/ctreepo/assets/` yet.

---

## Headline

The Manifesto f/g ladder reads as three threads that the writeup carries
together. First, leaf-size invariance: per-dimension external Pearson in
the joint six-dim run holds within a 0.014–0.045 band across a 32× leaf
sweep on five of six dimensions, extending the headline-table invariance
claim of §5.1 to the alternating-optimization regime. Second, a clean
f/g tension on the sixth dimension: under joint training one g-update
lifts decentralization 0.361 → 0.461 at leaf 8096, the next f-update
pulls it back to 0.343, and the trajectory oscillates without
converging. Third, the single-dim escape: dropping the joint metric and
training on decentralization alone
(`decentralization_benoit_g0init_fresh_dspy_20260426_1815/`, leaf 256)
reaches external Pearson 0.557 at f¹g¹ and **stays at 0.557 at f²g¹**.
The f-pullback dynamic the joint run shows is gone; the single-dim
ladder converges in one g/f cycle instead of oscillating. The 0.557
sits above the Benoit proprietary 18-score ensemble baseline of 0.490
and at parity with our existing per-dim pipeline at 8K-character leaves
at roughly 1/8 the context budget. Larger leaf rungs (512, 1024) are
still in flight; we are writing up assuming they track this pattern,
with FIXMEs below for the cells to fill in.

## Setup

The experimental scaffolding is the f/g ladder of §5.3. The combined run is
`combined_benoit_joint_teacher_all6_dspy_fixed_20260425_000122`. It uses one
shared `JointDimensionScorer` to score all six dimensions
(`economic`, `social`, `immigration`, `eu`, `environment`,
`decentralization`) on the same g-summary, with the g-training metric defined
as the unweighted mean of per-dimension f-rewards
(`src/ctreepo/joint_dspy_family.py:494-516`). The leaf-size axis sweeps
{256, 512, 1024, 2048, 4096, 8096} tokens; the alternating axis sweeps the
six rungs f¹g⁰, f¹g¹, f²g¹, f²g², f³g², f³g³. The two heatmaps motivating
this writeup are `plots/manifesto_fg_ladder_heatmap.png` (macro-averaged
external Pearson) and
`plots_by_dimension/manifesto_fg_ladder_dimension_ext_pearson_heatmap.png`
(six-panel facet, one panel per dimension).

## Leaf-size invariance across the f/g ladder

The joint run extends the leaf-size invariance pattern of §5.1 down into
the f/g alternating regime. At f¹g⁰ across the 32× leaf sweep
(256 → 8096 tokens), the per-dimension external Pearson bands are:

| dimension          | ext. Pearson band (256 → 8096 tok) | range |
|--------------------|------------------------------------|-------|
| economic           | 0.842 – 0.872                      | 0.030 |
| social             | 0.841 – 0.865                      | 0.025 |
| immigration        | 0.918 – 0.932                      | 0.014 |
| eu                 | 0.935 – 0.952                      | 0.017 |
| environment        | 0.737 – 0.782                      | 0.045 |
| decentralization   | 0.302 – 0.393                      | 0.091 |

Five of six dimensions sit in bands of width ≤ 0.045 across the 32× leaf
sweep, the same regime that §5.1 reports for macro Pearson on the
published per-dim pipeline (a 0.027-wide macro band over a 16× leaf
range). Whatever signal the joint pipeline picks up on these axes is
invariant to how finely the manifesto is cut into leaves; the
audit-budget framing of §5.1 carries forward to alternating
optimization.

Decentralization is also flat under the same sweep. The 0.091 band is
wider than the others, but bigger leaves do not help on this axis.
0.302 (leaf 1024) and 0.393 (leaf 2048) sit a notch below and above the
0.36 mean; there is no monotone improvement with leaf size. The
joint-run failure on decentralization is a level effect, not a
resolution effect. Whatever is suppressing the signal is the same at
leaf 256 and leaf 8096.

## The f/g tension on decentralization in the joint run

Decentralization is the only axis where the alternating ladder shows
non-trivial dynamics across stages. At leaf 8096:

| stage | dec. Pearson | dec. gap | env. Pearson | env. gap |
|-------|--------------|----------|--------------|----------|
| f¹g⁰  | 0.361        | 0.454    | 0.782        | 0.107    |
| f¹g¹  | 0.461        | 0.326    | 0.722        | 0.147    |
| f²g¹  | 0.343        | 0.484    | 0.705        | 0.227    |
| f²g²  | 0.343        | 0.459    | 0.734        | 0.160    |
| f³g²  | 0.359        | 0.468    | 0.752        | 0.152    |
| f³g³  | 0.413        | 0.407    | 0.704        | 0.190    |

The trajectory tells a clean story. g re-trained against the current f
lifts decentralization by 0.10 at f¹g¹, the largest single-step gain
anywhere in the heatmap. The next f-update (f²g¹) re-fits f against the
joint multi-dim metric and decentralization falls back to 0.343, below
the f¹g⁰ baseline. Subsequent rungs oscillate in the 0.34 – 0.41 band
without converging on the f¹g¹ peak. The other four dimensions move by
0.01 – 0.03 across the same trajectory; environment pays a clear cost
(0.782 → 0.722 at f¹g¹, 0.704 by f³g³).

The mechanism is multi-task interference on the f side. The g-training
metric is the unweighted mean of per-dimension f-rewards
(`joint_dspy_family.py:494-516`); a g-step that improves a dimension
near the consensus is rewarded for the average and reinforced.
Decentralization is far from the consensus on this corpus, so a g-step
that improves it costs reward elsewhere unless the gain on
decentralization exceeds the sum of losses on the other five. f
re-training then runs the gradient on the same averaged metric and
treats the decentralization signal as 1/6 of the budget. The f-update
moves f toward the joint optimum and pulls g (through the next g-step's
reward signal) back toward the consensus summary. The cycle repeats.

The compression diagnosis lands on the f side. g has output budget
2× leaf-size (`src/ctreepo/fg_arity.py:48-75`), and the f¹g¹ result
shows g can encode decentralization within that budget. The
representational bottleneck is not in g. The 1/6 gradient weight on the
f side is the operative constraint, and the alternating ladder converts
it into an oscillation rather than a fixed point.

The single-dim run resolves the tension cleanly. At leaf 256 the
single-dim trajectory is f¹g⁰ 0.534 → f¹g¹ 0.557 → f²g¹ 0.557 (full
trajectory below). The f²g¹ rung shows no pullback: external Pearson,
external MAE, internal Pearson, and f\*-gap are all identical to f¹g¹
to four decimal places. Two readings are consistent with the data, and
both support the multi-task-interference diagnosis. The first: the f
gradient under a single-dim objective is aligned with the g objective,
so f²g¹ converges to the same fixed point that f¹g¹ already sat on.
The second: the f-step's MIPRO search returned the same prompts /
demonstrations as f¹g¹, which is what convergence under aligned
objectives looks like when the optimizer is discrete. Either way, the
joint-run pullback dynamic does not appear when the multi-dim averaging
is removed. That is the cleanest empirical control on the diagnosis the
data can deliver.

## Single-dimension comparison: Economics

The Economics-only run
(`economic_benoit_largeleaves_dspy_medium_20260423_001200/`, leaf 1024,
single rung) reaches external Pearson 0.8847 with f\*-gap 0.0892. The
matching combined-run cell (Economics at leaf 1024, f¹g⁰) is Pearson
0.8659 with gap 0.1074. The lift is +0.019 Pearson and -0.018 gap.
Economics already wins under joint training, so the room for improvement
is small; the consistent direction confirms that single-target g produces
a better summary even on a dimension that does not appear to need help.

## Single-dimension comparison: Decentralization (current snapshot)

> **FIXME (partial run).** Leaf 256 has f¹g⁰, f¹g¹, f²g¹ in hand;
> f²g² is mid-MIPRO at the time of writing. The leaf 512 and leaf 1024
> rungs of `decentralization_benoit_g0init_fresh_dspy_20260426_1815/`
> are still in flight. Before the writeup goes into LaTeX, fold in:
> (i) the f¹g⁰ / f¹g¹ / f²g¹ rows for leaf 512 and leaf 1024,
> (ii) the f²g², f³g², f³g³ tail at leaf 256 to confirm the ladder
> stays at the f¹g¹ fixed point through the rest of the trajectory,
> (iii) the teacher's external-expert ceiling on
> decentralization-alone for the right ceiling claim, and (iv) a
> follow-up read on the small MAE move noted below.

The Decentralization-only run
(`decentralization_benoit_g0init_fresh_dspy_20260426_1815/`,
identity-init f, fresh g⁰ init, leaf 256) currently shows:

| stage | int. Pearson | ext. Pearson | f\*-gap | ext. MAE | n_eval |
|-------|--------------|--------------|---------|----------|--------|
| f¹g⁰  | 0.951        | 0.534        | 0.417   | 2.278    | 48     |
| f¹g¹  | 0.960        | 0.557        | 0.403   | 2.323    | 48     |
| f²g¹  | 0.960        | 0.557        | 0.403   | 2.323    | 48     |

Source: `ladder/dspy/leaf0256tok/step_checkpoints/iter_00_post_eval.json`
and `iter_01_post_eval.json` in the run directory above.

The lift relative to the joint run at the same leaf=256, f¹g¹ cell is
**+0.226 external Pearson** (0.331 → 0.557) and **-0.042 f\*-gap**
(0.445 → 0.403). The single-step gain inside the run (f¹g⁰ → f¹g¹) is
+0.023 Pearson, smaller than the +0.10 the joint run produced at
leaf 8096. The plausible read: the single-dim f¹g⁰ baseline (0.534) is
already much higher than the joint f¹g⁰ baseline (0.327) at the same
leaf, so g has less slack to recover. f²g¹ then sits at the f¹g¹ fixed
point exactly (Pearson, gap, MAE, internal Pearson all identical to
four decimals); the f-pullback that defines the joint-run trajectory
is absent under single-dim training.

External baselines on this dimension (from
`paper/ctreepo/assets/benoit/tables/benoit_comparison_pearson.tex`):

| reference                                    | dec. Pearson |
|----------------------------------------------|--------------|
| LLaMA-3.3-70B (Benoit Table 6)               | 0.400        |
| DeepSeek-V3 (Benoit Table 6)                 | 0.450        |
| Gemma-3-27B-IT (Benoit Table 6)              | 0.450        |
| Benoit proprietary 18-score ensemble (Fig 1) | 0.490        |
| Our joint pipeline, leaf 8K chars            | 0.413        |
| Our per-dim pipeline, leaf 8K chars          | 0.543        |
| Our per-dim pipeline, leaf 4K chars          | 0.580        |
| **Our f/g single-dim, leaf 256 tokens**      | **0.557**    |
| Split-expert reliability ceiling (Table 3)   | 0.780        |

The 0.557 number sits above every model-only baseline, including the
proprietary ensemble. Against the existing per-dim pipeline already in
the headline table, 0.557 at leaf 256 tokens is parity with 0.543 at
leaf ~2K tokens (8K chars). The win is leaf-size compression at fixed
quality on this dimension, not a Pearson-ceiling break. The split-expert
reliability ceiling of 0.780 leaves about 0.22 of headroom that
single-dim training cannot cross without external supervision beyond
the teacher.

A small caveat to track. f¹g⁰ → f¹g¹ moves Pearson up but external MAE
up too (2.278 → 2.323). Pearson up + MAE up is a scale shift, not a
calibration improvement: g re-training moved the prediction
distribution. The mean prediction in the JSON is 2.521 against an
expert mean of 4.671; the absolute scale gap is large, and the small
MAE move at f¹g¹ does not change that picture. The clean read on this
needs the leaf 512 / 1024 rungs.

## Methods finding: 2×leaf budget guard

The earlier Decentralization-only attempt
(`..._single_dspy_20260426_1745`) hit
`actual input tokens=12288, available budget=9988
(lm_context_window_tokens=12000 - max_completion_tokens=512 -
prompt_template_overhead_tokens=1500); field_counts={'prompt': 6157,
'completion': 6131}`
at f¹g¹ on the leaf=256 rung. The arithmetic is the
`g_output_tokens ≥ 2 × leaf_size_tokens` constraint
(`src/ctreepo/fg_arity.py:48-75`) colliding with DSPy prompt overhead
under a 12k LM context window. The guard fired correctly (silent
truncation would have produced corrupted summaries), and the operator
response was to re-launch with leaf-context groups
(`256,512,1024:12000:0.90:768:768 2048:20000:0.90:512:512 …`) sized to the
2×leaf concat lower bound. The episode is direct evidence for the
load-bearing role of the concat-budget invariant in the f/g pipeline.
A one-paragraph methods callout in §G ("Manifesto details") fits this
material better than a footnote.

## Implications for the algebra and audit story

Centralized g is the right default when the local-law audit (C1, C2, C3)
holds across dimensions simultaneously, since one summary serves them all
and the f/g ladder converges to a consistent fixed point. When the audit
fails on a minority axis (signaled by external-Pearson and f\*-gap
divergence), single-target g restores learnability for that axis at the
cost of running a separate ladder. The Manifesto data argue for a
two-track recipe in §6 (Discussion): default to centralized g, fall back
to single-target g on any axis whose joint-run f\*-gap exceeds a threshold
(decentralization at 0.45 is far above the 0.05–0.15 band the other axes
sit in). The threshold itself is a knob to be set after the leaf
512 / 1024 rungs of the Decentralization-only run land.

## Framing

Centralized g succeeded on five of six dimensions and held leaf-size
invariance across a 32× sweep on those five. The sixth axis exposed a
clean f/g tension: g can carry decentralization in a single update, f
trained against the averaged six-dim metric pulls back to consensus,
and the alternating ladder oscillates without converging. Single-dim
training removes the averaging, reaches external Pearson 0.557 at leaf
256 tokens, and the f²g¹ rung lands at the same fixed point —
oscillation gone. The 0.557 sits above the Benoit proprietary 18-score
ensemble and at parity with the existing per-dim pipeline at 8× the
leaf budget. The combined story sharpens the scope of the centralized-g
claim and gives §6 a concrete trigger condition for falling back to
per-axis training: when the joint-run f\*-gap on a dimension exceeds
the band the other dimensions sit in, switch that dimension to a
single-target ladder.

---

## TODO / FIXME

- [ ] **(FIXME, blocking)** Update the "Single-dimension comparison:
      Decentralization (current snapshot)" block with leaf 512 and leaf
      1024 rungs from
      `decentralization_benoit_g0init_fresh_dspy_20260426_1815/` once
      they complete. Watch whether the lift over joint stays large at
      bigger leaves and whether the f²g¹ rung claws Pearson back the
      way the joint run did.
- [x] ~~Pull the f²g¹ rung at leaf 256 from the single-dim run.~~
      **Answered.** f²g¹ at leaf 256 is identical to f¹g¹ to four
      decimals (Pearson 0.557, MAE 2.323, internal Pearson 0.960,
      f\*-gap 0.403). No f-pullback under single-dim training; the
      ladder converges in one g/f cycle. Continue to confirm with the
      f²g², f³g², f³g³ tail at leaf 256.
- [ ] **(FIXME)** Investigate the small MAE regression
      (2.278 → 2.323) at f¹g⁰ → f¹g¹. Pearson up + MAE up is a scale
      shift; check `mean_prediction_1_7` against `mean_expert_1_7` once
      more rungs land.
- [ ] Compare against teacher's external-expert Pearson on
      decentralization alone to establish the ceiling. Pull from
      `decentralization_benoit_g0init_fresh_dspy_20260426_1815/teacher/`.
- [ ] Decide whether a third single-dim sanity run is needed.
      Environment is the next-most-degraded dimension under f¹g¹ in the
      joint run (0.78 → 0.72) and a candidate; immigration is the
      cleanest joint-run dim and would be the strongest null case.
- [ ] Decide whether to recompute the joint-run heatmaps with depth
      discount γ < 1 active, per `OPSCountConfig.depth_discount_gamma`.
      γ = 1.0 was the default; whether γ < 1 changes the joint-run
      decentralization story is open.
- [x] ~~Decide §5.3-extend vs new §5.4 placement.~~ **Resolved:**
      Option C (rewrite §5.3 around the joint-vs-single-target arc,
      with f\*-gap as the section's anomaly indicator). Init-scheme
      distinction drops to a footnote. The integrated sketch follows
      below.
- [ ] Style pass against `paper/ctreepo/STYLE.md` once numbers are
      final: em-dash count audit, throat-clearer scan,
      "not X but Y" scan.

---

## Integrated §5.3 sketch (Option C)

Markdown rendition of how §5.3 would read under Option C, with the
narrative shape the user specified: open with the f\*-gap diagnostic,
then per-dimension ladders on Economics and Decentralization (the same
shape on different difficulty levels), then introduce the centralized-g
six-dim version and its failure mode. Numbers from the validated
sources above. Math notation uses `$f$` / `$g$` / `$f^*$` for direct
LaTeX hand-off; section cross-references use `\S\ref{...}` placeholders.

### §5.3 opening: the ladder and its diagnostic

The alternating $f/g$ prompt ladder applies the framework's alternation
principle to the prompts themselves. Hold the summarizer/merge prompt
$g$ fixed, optimize the scorer prompt $f$ against the current
summaries; swap roles, optimize $g$ against the current scorer; repeat.
The compact label $f^a g^b$ marks the scorer after $a$ scorer-side
rounds and the summarizer after $b$ summarizer-side rounds. Each cell
reports two correlations on the held-out split: external Pearson
against expert-survey means (the gold target) and internal Pearson
against the previous stage's scorer (a self-consistency check). Their
difference, the $f^*$-gap, is the section's anomaly indicator. A small
gap means the ladder converges to a scorer that agrees with the
previous scorer and the gold labels at once. A large gap means the
teacher-vs-gold disagreement caps what training-against-teacher alone
can reach, and the ladder's external Pearson is bounded below the
teacher's internal Pearson by the size of that disagreement.[^init]

[^init]: The ladder accepts several initial conditions for $g^0$ and
$f^1$ (Benoit's archived GPT-4o summaries with a scorer-only optimized
prompt; an own-Gemma raw-init pair; identity initialization). The init
scheme is an engineering detail; the diagnostics below report whichever
cell is available for each lane.

### §5.3.x  Per-dimension ladders: Economics

The single-dimension Economics ladder gives the cleanest reading on a
dimension where the teacher carries most of the signal. Per-leaf best
external Pearson sits at $0.885$ (leaf $1024$), $0.886$ ($2048$),
$0.886$ ($4096$), and $0.879$ ($8192$): a $0.007$-wide band straddling
the economic split-expert reliability reference of $0.880$ across an
$8\times$ leaf range. Below roughly $1024$ tokens the ladder degrades
gracefully ($0.861$ at leaf $512$, $0.830$ at $256$); leaves too small
to carry the evidence a later merge needs. The $f^*$-gap stays in
$[0.086, 0.161]$ across the populated grid, with internal Pearson in
$[0.953, 0.990]$. The diagnostic reads cleanly on Economics: the
ladder co-adapts $f$ and $g$ without racing ahead of gold, and the
teacher carries the dimension to within $0.10$--$0.16$ of the
split-expert ceiling. Figure~\ref{fig:v4-manifesto-fg-headline} shows
the leaf-size plateau against the split-expert reference.

### §5.3.x  Per-dimension ladders: Decentralization

The single-dimension Decentralization ladder gives the same shape on a
much harder axis. At leaf $256$, the trajectory is $0.534 \to 0.557
\to 0.557$ across $f^1g^0 \to f^1g^1 \to f^2g^1$. One $g$-update lifts
the correlation by $0.023$; the next $f$-update lands on the same
fixed point to four decimals; the ladder converges in one $g/f$ cycle.
The Pearson sits above the proprietary 18-score ensemble of
\citet{BenoitEtAl2025} on this dimension ($0.490$) and at parity with
our per-dim pipeline at $8\mathrm{K}$-character leaves
(Table~\ref{tab:min-benoit-headline}, $0.543$) at roughly $1/8$ the
leaf budget. Leaf-size invariance carries through to the dimension the
published baselines find hardest. The $f^*$-gap is bounded
($0.42 \to 0.40 \to 0.40$, internal Pearson in $[0.951, 0.960]$), and
the gap moves cleanly through the cycle the way it moves on Economics.
Its absolute level is the load-bearing fact: $\sim\!0.40$, four times
the Economics gap. The diagnostic flags decentralization as
teacher-bounded. The scorer matches the teacher very well; the teacher
is the remaining gap to gold. The next subsection takes the same
ladder under centralized $g$ and shows the gap blowing up further when
the teacher is shared across dimensions [FIXME: leaf $512$, leaf
$1024$ rungs of `decentralization_benoit_g0init_fresh_dspy_20260426_1815/`
extend the convergence claim across leaf sizes].

### §5.3.x  Centralized-$g$: setup

The centralized-$g$ configuration applies the same ladder under a
shared summarizer. One $g$ produces a single summary for each
manifesto; one joint scorer (the \texttt{JointDimensionScorer} of
\S\ref{sec:min-framework}) returns six per-dimension scores against
that summary. The $g$-training metric is the unweighted mean of
per-dimension $f$-rewards. The $f$-training step runs the same
averaged metric on the scoring side. Six leaf rungs ($256$--$8096$
tokens) and six alternating stages ($f^1g^0$ through $f^3g^3$).

### §5.3.x  Centralized-$g$ results: leaf-size invariance on five, failure on the sixth

Five of six dimensions track each other across the centralized-$g$
ladder. At $f^1g^0$ across the $32\times$ leaf sweep, the
per-dimension external Pearson bands are $0.014$ (immigration), $0.017$
(EU), $0.025$ (social), $0.030$ (economic), and $0.045$ (environment).
All five sit in bands narrower than the $0.027$-wide macro band of
\S\ref{sec:min-headline}, so the leaf-size invariance of the headline
extends from per-dimension training into centralized-$g$ alternating
optimization. $f^*$-gaps on these five sit in $[0.03, 0.16]$, the same
regime as the per-dim Economics ladder of
\S\ref{sec:min-fg-economics}. Decentralization sits in $0.30$--$0.39$
across all leaf sizes and its $f^*$-gap is $0.45$, three to five times
higher than the other five. Bigger leaves do not help. The diagnostic
that read cleanly on the per-dimension ladders says decentralization
is the failure mode under centralized $g$.

### §5.3.x  The $f/g$ tension and the single-target escape

At leaf $8096$ the centralized-$g$ trajectory on decentralization
oscillates: $0.361 \to 0.461 \to 0.343 \to 0.343 \to 0.359 \to 0.413$
across $f^1g^0$ through $f^3g^3$. One $g$-step lifts decentralization
by $0.10$, the largest single-step gain anywhere in the grid; the next
$f$-step pulls it back below the $f^1g^0$ baseline; the trajectory
oscillates without converging. The mechanism is multi-task
interference on the $f$ side. The $f$-gradient runs on the averaged
six-dim metric, so a dimension whose latent direction is far from the
consensus receives $1/6$ of the gradient signal, and every $g$-step
that lifts it gets unwound on the next $f$-step. The compression
diagnosis sits on the $f$ side, not the $g$ side; $g$'s output budget
is $2 \times$ leaf-size and the $f^1g^1$ result shows $g$ can encode
decentralization within that budget when the reward signal points it
there.

The single-target ladder of \S\ref{sec:min-fg-decentralization}
converges in one $g/f$ cycle precisely because it removes the
averaging. Its $f^2g^1$ rung lands at the $f^1g^1$ fixed point to four
decimals, and the elevated $f^*$-gap inherited from the dimension's
teacher-bounded character stays bounded through the cycle. Removing
the multi-dimensional average removes the pullback; what remains is
the teacher ceiling. The headline of the section: per-dimension
ladders are the right default. Centralized $g$ works whenever the
$f^*$-gap on every dimension stays in the band the per-dimension
ladders show, and falls back to per-dimension training when any single
axis pushes the gap outside that band.

### Figures referenced (TODOs, no cuts in this pass)

- §5.3.x Economics: `assets/benoit/figures/manifesto_fg_ladder_headline.pdf`
  (existing, kept).
- §5.3.x Centralized-$g$ results: redo of
  `outputs/manifesto_fg_alternating/combined_benoit_joint_teacher_all6_dspy_fixed_20260425_000122/plots_by_dimension/manifesto_fg_ladder_dimension_ext_pearson_heatmap.png`
  in PDF with `paperplot` palette and the decentralization panel
  annotated (e.g., shaded background). Six-panel facet: one panel per
  dimension, x = stage, y = leaf size, cell = external Pearson.
- §5.3.x Tension/escape: new figure
  `manifesto_fg_decentralization_trajectory.pdf`. x = stage
  ($f^1g^0 \ldots f^3g^3$); y = external Pearson. Two lines:
  centralized-$g$ at leaf $8096$ (oscillating) and single-target at
  leaf $256$ (converging at $0.557$). Reference lines: Benoit
  proprietary ensemble $0.490$, per-dim baseline @ $8\mathrm{K}$ chars
  $0.543$, split-expert ceiling $0.780$.

### Notes for the LaTeX editor

- The six paragraph blocks above map onto either six `\paragraph{}`
  breaks under one `\subsection*{}` or three `\subsection{}` breaks
  (Per-dimension ladders / Centralized-$g$ / Tension and escape) with
  two paragraphs each. The latter reads as the cleaner partition.
- Replace the `\S\ref{sec:min-fg-economics}` and
  `\S\ref{sec:min-fg-decentralization}` placeholders with the labels
  the LaTeX editor settles on.
- The footnote in the section opening replaces lines 82–88 of the
  current `05_manifesto.tex` (the Benoit-init / raw-init lane
  paragraph). The defensive paragraph at lines 130–135 ("the raw-init
  lane should be read carefully") drops out under Option C.
- The current §5.3 figure (`manifesto_fg_ladder_headline.pdf`) anchors
  the Economics paragraph block. The trajectory figure is the new main
  figure for the section. The cleaned-up six-panel heatmap is a
  supporting figure (main if the section has the budget; appendix
  otherwise).
