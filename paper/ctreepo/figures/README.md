# Paper Figure Provenance

One-stop index for every figure the minimal C-TreePO paper includes,
plus the standards every generator must follow. Keep this index in
sync with `paper/ctreepo/sections/minimal/BLUEPRINT_claude.md` §0.g.

## Standards (apply to every figure)

- **Format.** Vector PDF for anything with text or axes. PNG only
  for bitmap content (dense heatmaps, images). Matplotlib
  generators save both; `\includegraphics` references the PDF.
- **Typography.** All matplotlib generators call
  `paperplot.rcparams()` from `paper/ctreepo/scripts/paperplot.py`
  at startup. This fixes `pdf.fonttype=42`, serif family, 10pt
  default, 9pt axis labels, 8pt ticks and legend. Don't override
  locally unless a figure has a specific reason.
- **Size.** Use `paperplot.FIGSIZE_ONE_COL` for single-column
  figures (3.5 × 2.2 in) and `paperplot.FIGSIZE_FULL` for
  full-width (7.0 × 3.0 in). Tall variants available for
  legend-heavy plots.
- **Axes.** Prefer the repo's native units. Precision
  ($p=\log_2 m$) on a linear integer axis over "memory in
  bytes"; leaf counts on powers of 2; token budgets on powers
  of 2. If a derived quantity helps a reader, put it in the
  axis *label* (e.g. `Precision p (m=2^p registers)`) rather
  than replacing $p$ with it.
- **Uncertainty.** Don't show a p10--p90 wedge when the
  main-line median tells the didactic story cleanly. Reserve
  shaded bands for plots where variance is itself the point.
- **Color.** Fixed anchor palette in `paperplot.ANCHOR_COLORS`:
  Markov = blue, HLL = orange, Manifesto = green, neural
  operator = purple, theory/baseline = gray. Distinguish series
  within a figure by linestyle (solid / dashed / dotted), not
  color alone, so the figure reads in grayscale.
- **Caption.** Bold lead sentence stating the result, then the
  what-is-plotted sentence, then the interpretation hook.
- **Provenance.** Every figure lists its generator command in
  this README. Generators use fixed seeds.

## Main-text figures

| Label | File (included) | Generator | Last regenerated |
| --- | --- | --- | --- |
| `fig:min-plain-tree` (§3.2) | `doc/old/figures/cld/01_base_plain.pdf` (via `01_base_plain.pdf` on graphicspath) | legacy; no current generator in repo | pre-2026 |

## Appendix figures

| Label | File (included) | Generator | Last regenerated |
| --- | --- | --- | --- |
| `fig:min-markov-registers` (App. E) | `paper/ctreepo/assets/markov/figures/markov_scene_arc_hamlet.pdf` | `cd paper/figures && latexmk -pdf markov_scene_arc_hamlet.tex` | 2026-04-24 |
| `fig:min-hll-parity` (App. F) | `paper/ctreepo/assets/hll/figures/hll_merge_learning_memory_median.pdf` | `venv/bin/python paper/ctreepo/scripts/regen_paper_hll_figure.py` | 2026-04-24 |
| `fig:min-manifesto-ladder` (App. G) | `paper/ctreepo/assets/benoit/figures/manifesto_fg_ladder_benoit_init.png` | `bash paper/ctreepo/scripts/regen_benoit_manifesto.sh` (live-iteration plotter is `scripts/plot_manifesto_fg_ladder_grid.py`; watch-and-refresh driver is `scripts/refresh_manifesto_fg_ladder_plots.py`) | 2026-04-24 |
| `fig:manifesto-fg-ladder-heatmaps` (App. H) | `paper/ctreepo/assets/benoit/figures/manifesto_fg_ladder_{benoit_init,raw_init}.png` | `bash paper/ctreepo/scripts/regen_benoit_manifesto.sh` | 2026-04-24 |
| `fig:manifesto-fg-headline` (§9.4) | `paper/ctreepo/assets/benoit/figures/manifesto_fg_ladder_headline.pdf` | `bash paper/ctreepo/scripts/regen_benoit_manifesto.sh` | 2026-04-24 |
| `tab:manifesto-fg-ladder` (App. H) | `paper/ctreepo/assets/benoit/tables/manifesto_fg_ladder.tex` | `bash paper/ctreepo/scripts/regen_benoit_manifesto.sh` | 2026-04-24 |
| Markov leaf-size publication heatmaps | `paper/ctreepo/assets/markov/figures/markov_leaf_size_fixed_{recoverable,structural}.pdf` | `venv/bin/python paper/ctreepo/scripts/build_markov_figures.py --summary outputs/markov_v3_publication_leaf_size_fixed_docs_20260413_220609/summary.json --output-dir paper/ctreepo/assets/markov/figures/` | 2026-04-24 |

## Retired / reference figures (in assets but not included)

| File | Why kept | Replacement |
| --- | --- | --- |
| `assets/hll/figures/hll_parity_curves.png` | Legacy five-panel $(p, L)$ grid; useful as a reference but less clear as a didactic figure. | `hll_merge_learning_memory_median.pdf` |
| `assets/hll/figures/hll_merge_learning_memory.png` | Mean-over-seeds companion to the median plot. | `hll_merge_learning_memory_median.pdf` (swap if a variance-forward view is needed) |
| `assets/markov/figures/markov_changepoint_combined.pdf` | Abstract four-color register arc used by the v2 paper (`sections/v2/04_markov_interlude.tex`). Kept untouched so the v2 build still compiles. | `markov_scene_arc_hamlet.pdf` is the Hamlet-labelled replacement used by the minimal paper. |

## Regen commands

All commands run from the repository root with the repo venv.

```bash
# HLL main-anchor figure (App. F)
venv/bin/python paper/ctreepo/scripts/regen_paper_hll_figure.py

# Markov register-arc figure (App. E)
cd paper/figures && latexmk -pdf markov_scene_arc_hamlet.tex \
    && cp markov_scene_arc_hamlet.pdf \
       ../ctreepo/assets/markov/figures/markov_scene_arc_hamlet.pdf

# Classical sketch bundle (App. F.9 gallery, if promoted)
bash paper/ctreepo/scripts/regen_classical_sketches.sh

# Manifesto f/g ladder publication figures and table (App. H, §9.4)
bash paper/ctreepo/scripts/regen_benoit_manifesto.sh

# Markov leaf-size publication heatmaps (App. E / §8 leaf-size view)
venv/bin/python paper/ctreepo/scripts/build_markov_figures.py \
    --summary outputs/markov_v3_publication_leaf_size_fixed_docs_20260413_220609/summary.json \
    --output-dir paper/ctreepo/assets/markov/figures/
```

## Adding a new figure

1. Write the generator in `paper/ctreepo/scripts/`. Import
   `paperplot` and call `paperplot.rcparams()` before any plotting.
2. Use `paperplot.save(fig, stem)` so the PDF and PNG land together.
3. Add an `\includegraphics{<name>.pdf}` call in the appropriate
   `.tex` file with a bold-lead caption.
4. Add a row to the table above with label, path, generator
   command, and today's date.
5. Update `BLUEPRINT_claude.md` §0.g figure inventory to match.
