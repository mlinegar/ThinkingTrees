# Paper Asset Conventions (C-TreePO)

Every figure and table that appears in `main_new.tex` lives under
`paper/ctreepo/assets/<example>/{figures,tables}/`. There is one
sub-directory per running example:

- **`markov/`** — the Markov-changepoint walkthrough (sections 4 and 6)
- **`hll/`** — the HyperLogLog parity experiment (section 7, appendix F)
- **`benoit/`** — the Manifesto / Benoit-2026 replication (section 9, appendix H)

A new running example follows the same template: create
`assets/<name>/{figures,tables}/`, add `paper/ctreepo/scripts/regen_<name>.sh`,
list `<name>` in `regen_assets.sh`'s default `EXAMPLES` array, and reference
its assets via the conventions below.

## File-name convention

| Asset type | Pattern | Example |
|---|---|---|
| Headline figure | `<example>_<label>.{pdf,png}` | `assets/benoit/figures/benoit_chunk_invariance.pdf` |
| Table (LaTeX + Markdown) | `<example>_<label>.tex` and `<example>_<label>.md` | `assets/benoit/tables/benoit_comparison_pearson.tex` |
| Per-variant grid | `<example>_<label>_<variant>.{tex,md}` | `assets/benoit/tables/benoit_comparison_T0.2_N3.tex` |

Use `_` as the field separator inside a file name; reserve `-` for inside a
single field (e.g. `markov_simple-leaf-mass.png` is wrong — use
`markov_simple_leaf_mass.png`).

The `.md` companion to every `.tex` table is intentional: reviewers and the
companion-repo README read the markdown; the paper inputs the `.tex`. Both are
emitted by the same generator so they cannot drift.

## LaTeX label convention

| LaTeX object | Label pattern | Example |
|---|---|---|
| Figure | `\label{fig:<example>-<label>}` | `\label{fig:benoit-chunk-invariance}` |
| Table | `\label{tab:<example>-<label>}` | `\label{tab:benoit-headline}` |
| Section | `\label{sec:<example>-<topic>}` | `\label{sec:benoit-headline}` |
| Appendix | `\label{app:<example>-replication}` | `\label{app:benoit-replication}` |

Labels read top-down (example then label) so a paper-wide grep for `benoit`
turns up everything in one shot.

## Including assets in LaTeX

```tex
% Figure
\begin{figure}[t]
    \centering
    \includegraphics[width=0.95\linewidth]{benoit_chunk_invariance.pdf}
    \caption{...}
    \label{fig:benoit-chunk-invariance}
\end{figure}

% Table
\begin{table}[t]
    \centering
    \small
    \input{assets/benoit/tables/benoit_comparison_pearson.tex}
    \caption{...}
    \label{tab:benoit-headline}
\end{table}
```

`\graphicspath` (in `preamble.tex`) lists `assets/markov/figures/`,
`assets/hll/figures/`, and `assets/benoit/figures/`, so figures are referenced
by *base name only* — never by a relative path.

Tables are explicitly `\input{assets/<example>/tables/<file>.tex}` because
LaTeX has no `\inputpath` analog.

## Caption convention

Every figure and table caption ends with a one-line pointer to the relevant
appendix when the appendix carries the full per-cell data:

> ... (full per-cell table with standard errors, coverage, and additional
> ablations in Appendix~\ref{app:benoit-replication}.)

The intent is that a reader who wants to dig in always finds the full grid in
exactly one place per example: `appendix/<letter>_<example>_replication.tex`.

## Regenerating assets

```bash
# all examples (markov, hll, benoit)
bash paper/ctreepo/regen_assets.sh

# just one example
bash paper/ctreepo/regen_assets.sh benoit
bash paper/ctreepo/regen_assets.sh markov hll
```

The master driver delegates to per-example sub-scripts in
`paper/ctreepo/scripts/`. Each sub-script:

1. Reads from `outputs/` (the source of truth for raw run results).
2. Calls the generator scripts (`scripts/comparison_table.py`,
   `scripts/render_*.py`, etc.).
3. Stages canonical-named copies under
   `paper/ctreepo/assets/<example>/{figures,tables}/`.

The generator scripts write to the example's *native* output directory first
(e.g. `outputs/classical_parity/hll/`) and then the sub-script copies them to
the canonical asset names. This keeps generator scripts decoupled from the
paper layout while letting the paper own its own naming.

## Adding a new example

1. `mkdir paper/ctreepo/assets/<name>/{figures,tables}`
2. Create `paper/ctreepo/scripts/regen_<name>.sh` modelled on the existing
   sub-scripts.
3. Add `<name>` to the `EXAMPLES` default array in `regen_assets.sh`.
4. Add `assets/<name>/figures/` to the `\graphicspath` in `preamble.tex`.
5. New section/appendix LaTeX uses `fig:<name>-…` / `tab:<name>-…` labels and
   `\input{assets/<name>/tables/…}` for tables.

## Things not covered here

- The exact column ordering inside generator scripts is the generator's
  responsibility (e.g. `_DIM_ORDER` in `scripts/comparison_table.py`). This
  doc only standardizes the file-system and label layer, not the table-content
  schema.
- Citations to other papers are managed in `paper/refs.bib`; the
  example-name prefix scheme does not extend to bibkeys.
- One-off figures used only in slide decks live under `paper/figures/`
  (legacy) and should be promoted to the asset tree if they enter the paper.
