# Manifesto f/g Ladder Publication Pass — Handoff (2026-04-24)

## What this pass did

Built a publication-figure suite for the manifesto f/g alternating ladder
(plus the older Markov v3 leaf-size data), wired it into `main_new.tex`,
and refreshed the assets that `main_minimal.tex` already references.

**Headline data point landed by the Apr 24 retry run** (raw-init lane,
leaf 256, stage f¹g¹): external Pearson **r = 0.909**, internal-external
gap **0.056** — beats the Benoit 2025 split-expert reference of r = 0.880
on the held-out split.

## Where the changes landed

The pass focused on `main_new.tex`. `main_minimal.tex` picks up the
**regenerated asset files** transitively (it `\includegraphics`'s the
same PNGs and `\input`'s the same `.tex` table) but does NOT yet have:

- The new main-text headline figure (`manifesto_fg_ladder_headline.pdf`).
- The raw-init heatmap subfigure (`manifesto_fg_ladder_raw_init.png`).
  `main_minimal` currently shows only `manifesto_fg_ladder_benoit_init.png`.
- The Markov leaf-size publication heatmaps
  (`markov_leaf_size_fixed_{recoverable,structural}.pdf`).
- The narrative resolutions for the four `\fixme{}` blocks that flagged
  the 2026-04-24 raw-init refresh as pending — `main_minimal` doesn't
  have these `\fixme{}` blocks at all (the minimal narrative is leaner),
  so nothing to delete; just consider whether the new numbers warrant
  expanding the minimal app G claims.

Both papers build cleanly with the new assets:
- `main_new.tex` — 88 pages, 4.5 MB, no warnings, no missing includes.
- `main_minimal.tex` — 29 pages, 0.9 MB, no warnings, no missing includes.

## New files (paper-side tooling)

All under `paper/ctreepo/scripts/` so the asset tree owns its
generators (sibling to `regen_paper_hll_figure.py`, `regen_benoit.sh`).

| File | Purpose |
|---|---|
| `paperplot.py` | **Extended.** Added `heatmap_with_reference()` shared helper (cell grid + bolded colorbar tick + missing-cell hatching + inherited-cell `=` overlay; supports `vcenter` for diverging maps via `TwoSlopeNorm`). Reused by manifesto and Markov figures. |
| `benoit_references.py` | Single source for `BENOIT_SPLIT_EXPERT_R = 0.880`, `PARITY_GAP = 0.0`, `ECONOMIC_PEARSON_AXIS = (0.75, 0.92)`, `ECONOMIC_GAP_AXIS = (-0.05, 0.20)`, `HEADLINE_CELL` dict (raw_init/256/f1g1/r=0.909/gap=0.056). Renderers hard-fail on drift. |
| `_ladder_io.py` | `LadderCell` dataclass (23 CSV columns), `load_cells()`, `dedupe()`, `collapse_inherited()`, `per_leaf_best()`, `assert_headline()`, `as_matrix()`, `inherited_mask()`. Lane assignment is explicit (caller passes `--csv` + `--lane`). |
| `build_manifesto_fg_figures.py` | Renders **F1** (`manifesto_fg_ladder_benoit_init`), **F2** (`manifesto_fg_ladder_raw_init`), **F3** (`manifesto_fg_ladder_headline`). YlGn for external Pearson, RdBu_r centered at 0 for gap. |
| `render_ladder_table.py` | Auto-regenerates `assets/benoit/tables/manifesto_fg_ladder.tex`. Booktabs, lane banners, anchor marker on lane's smallest-leaf anchor stage, bolded best ext-r and tightest gap per lane. |
| `build_markov_figures.py` | Renders **F5** (`markov_leaf_size_fixed_recoverable`), **F6** (`markov_leaf_size_fixed_structural`) from the older Markov v3 `summary.json`. Blues cmap (Markov anchor), bolded average-FNO reference tick. |
| `regen_benoit_manifesto.sh` | One-shot regen recipe for all manifesto figures + table. Cited by `figures/README.md`. |

## Generated assets

```
paper/ctreepo/assets/benoit/figures/
  manifesto_fg_ladder_benoit_init.{pdf,png}   # F1, App. H left subfigure
  manifesto_fg_ladder_raw_init.{pdf,png}      # F2, App. H right subfigure
  manifesto_fg_ladder_headline.{pdf,png}      # F3, §9.4 main-text figure (main_new only)
paper/ctreepo/assets/benoit/tables/
  manifesto_fg_ladder.tex                     # F4, auto-regenerated, 34 collapsed rows
paper/ctreepo/assets/markov/figures/
  markov_leaf_size_fixed_recoverable.{pdf,png}    # F5
  markov_leaf_size_fixed_structural.{pdf,png}     # F6
```

## TeX edits in main_new

| File | Change |
|---|---|
| `sections/v2/09_manifesto_llm.tex` | Inserted `fig:manifesto-fg-headline` figure float after the leaf-size-robustness paragraph. Resolved 2 of 4 `\fixme{}` blocks (the two referencing the Apr 24 refresh). Folded the raw-init vignette into the main narrative. |
| `appendix/H_benoit_replication.tex` | Resolved all 4 `\fixme{}` blocks. Updated raw-init subfigure caption (no longer says "leaves 1024–8192 still populating"). Updated table caption to describe the auto-regenerated structure (no more "31 evals collapsed to 15"). Tightened "What remains" paragraph from 3 items to 2. |
| `figures/README.md` | Updated `fig:min-manifesto-ladder` row to cite `regen_benoit_manifesto.sh`; added rows for the heatmap pair, headline figure, table, and Markov leaf-size figures; new bash invocations under "Regen commands". |

Two `\fixme{}` blocks remain in `09_manifesto_llm.tex`, both unrelated
to the Apr 24 refresh: (1) "pilot is economic-dimension only;
multi-dimension extension pending"; (2) "absolute-r comparison to the
§9.4 per-dim headline … is deferred". These belong to a future pass.

## What main_minimal needs to reach parity

`main_minimal.tex` already includes the regenerated F1 figure and F4
table via `appendix/minimal/G_manifesto_details.tex`. To bring it
fully in line:

1. **Add F2 (raw-init heatmap) as a paired subfigure in App. G.**
   Currently `G_manifesto_details.tex:45-60` has a single `\includegraphics`
   for `manifesto_fg_ladder_benoit_init.png`. Convert to a paired
   `subfigure` block matching the structure in
   `appendix/H_benoit_replication.tex:348-397`. The raw-init lane lands
   the headline cell (r=0.909) and is worth showing.

2. **Optionally add F3 (headline line plot)** to `sections/minimal/09_applications_scope.tex`
   or near `sections/minimal/06_main_results.tex` as a single-panel
   summary. The minimal paper currently relies on the App. G heatmap
   alone to carry the leaf-robustness claim; a one-glance line plot
   would be a small, low-risk addition.

3. **Add Markov F5/F6 references** somewhere in `appendix/minimal/E_markov_details.tex`.
   The minimal Markov appendix currently only includes
   `markov_scene_arc_hamlet.pdf`. Adding the leaf-size publication
   heatmaps would mirror the Manifesto appendix's coverage.

4. **Update `G_manifesto_details.tex` figure caption source string**
   (line 58) from `scripts/plot_manifesto_fg_ladder_grid.py` to
   `bash paper/ctreepo/scripts/regen_benoit_manifesto.sh`.

5. **Refresh the per-leaf best numbers in `G_manifesto_details.tex:38-40`**
   to reflect the auto-regenerated table. Spot-check the prose against
   the `manifesto_fg_ladder.tex` Benoit-init rows.

6. **Update the table caption at `G_manifesto_details.tex:65-71`** —
   currently says "Raw-init lane (bottom) reports the populated anchor
   and initial rows only." With the Apr 24 refresh, the raw-init lane
   now spans leaves 256–8096 at the f¹g⁰ and f¹g¹ stages plus deeper
   cells. Adjust language.

## How to regenerate

```bash
# Manifesto figures + table (uses the live bundle CSVs + Apr 24 retry)
bash paper/ctreepo/scripts/regen_benoit_manifesto.sh

# Markov leaf-size publication figures (from older v3 bundle)
venv/bin/python paper/ctreepo/scripts/build_markov_figures.py \
  --summary outputs/markov_v3_publication_leaf_size_fixed_docs_20260413_220609/summary.json \
  --output-dir paper/ctreepo/assets/markov/figures/

# Both papers
cd paper/ctreepo && latexmk -pdf -interaction=nonstopmode main_new.tex
cd paper/ctreepo && latexmk -pdf -interaction=nonstopmode main_minimal.tex
```

## Style decisions worth knowing

- All publication generators call `paperplot.rcparams()` (serif, 10pt,
  pdf.fonttype=42, tight savefig). The live-iteration plotter
  `scripts/plot_manifesto_fg_ladder_grid.py` does NOT and must not — it
  is the user's daily-iteration tool and stays unchanged.

- Anchor colors per `paperplot.ANCHOR_COLORS`: Manifesto = `#1b7837`
  (green), Markov = `#1f77b4` (blue), theory/baseline = grey. Series
  within a figure differ by **linestyle (solid/dashed/dotted), not
  color** — figures must read in grayscale.

- Heatmap cmaps: external-quality panels use `YlGn` (Manifesto) or
  `Blues` (Markov) — sequential, ramping toward the topic anchor color.
  Signed gap / structural-error panels use `RdBu_r` with `vcenter=0`
  (TwoSlopeNorm) so 0 always reads as white.

- Reference values: the `r=0.880` Benoit split-expert correlation lives
  in `benoit_references.py` and is bolded on every Manifesto colorbar.
  The Markov figures bold the average FNO root MAE drawn from the
  per-panel `official_fno_root_mae` field of the v3 summary. Both
  reference-line treatments share the same bolded-tick convention via
  `paperplot.heatmap_with_reference(..., bold_reference=True)`.

- The renderer hard-fails if the headline cell drifts: see
  `_ladder_io.assert_headline()`. Pass `--skip-headline-check` to bypass
  during exploratory work, but do NOT bypass when regenerating
  publication assets — `appendix/H_benoit_replication.tex` and
  `sections/v2/09_manifesto_llm.tex` cite r=0.909 and gap=0.056
  inline.

## Data sources used

```
outputs/manifesto_fg_alternating/
  benoit_grid_plots_benoit_init/manifesto_fg_ladder_grid_rows.csv      # Benoit-init lane
  benoit_grid_plots_raw_init/manifesto_fg_ladder_grid_rows.csv         # raw-init Apr 23 bundle
  economic_benoit_g0init_largeleaves_retry_20260424_085154/plots/manifesto_fg_ladder_grid_rows.csv  # raw-init Apr 24 retry (large leaves)
outputs/markov_v3_publication_leaf_size_fixed_docs_20260413_220609/
  summary.json                                                          # both Markov scopes
```

The regen shell script and the Markov build CLI both reference these
exact paths.

## Known follow-ups (not addressed in this pass)

1. **Multi-dimension extension.** Pilot is economic-only; the
   `\fixme{}` at `09_manifesto_llm.tex:343-347` flags this. When all
   six Benoit dimensions land, F3 needs a dimension facet (small
   multiples) or a per-dim band overlay.

2. **Apples-to-apples baseline.** The `\fixme{}` at
   `09_manifesto_llm.tex:344-347` defers the matched-leaf
   fixed-prompt baseline at the same prompt-search budget. When that
   run lands, App. H's "Apples-to-apples comparison" paragraph can be
   tightened.

3. **Fresh Markov overnight bundle.** User opted to use the older Apr
   13 `markov_v3_publication_leaf_size_fixed_docs_*` bundle this pass.
   When the overnight bundle (was preflighting `capacity` at session
   start) finishes and writes a newer `summary.json`, point
   `build_markov_figures.py --summary` at it and re-run. No code
   changes expected.

4. **Live-iteration plotter parity.** `scripts/plot_manifesto_fg_ladder_grid.py`
   could be refactored to share `paperplot.heatmap_with_reference()`
   for visual consistency with the publication renderer. Optional;
   the iteration plot is for the user, not the paper, so a different
   visual fingerprint is acceptable.

5. **Leaf-token typo: 8096 vs 8192.** The Apr 24 retry run labels its
   largest leaf as 8096 tokens; earlier runs use 8192. Both numbers
   appear in the auto-generated table. Cosmetic; doesn't affect the
   leaf-robustness claim. Worth correcting at the source if a future
   retry happens.

## Files NOT to touch

- `scripts/plot_manifesto_fg_ladder_grid.py` — daily-iteration tool.
- `scripts/render_manifesto_fg_plot_bundles.py` — bundle preset driver
  for live work; the paper-side renderer writes elsewhere.
- `scripts/refresh_manifesto_fg_ladder_plots.py` — watch-and-refresh
  driver feeding the bundle dirs.

These three drive the live workflow that produces the per-row CSVs the
publication renderer consumes; keeping them stable is what makes the
auto-regen safe.
