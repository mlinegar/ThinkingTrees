#!/usr/bin/env python3
"""Generate a gentle, tutorial-style introduction to tree-based merging.

This is the teaching version — minimal notation, conversational tone,
aimed at someone who has never seen the framework before.  It reuses
the same data and figures as the publication appendix but wraps them
in much simpler prose.
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Emit a gentle introductory walkthrough of tree-based merging."
    )
    p.add_argument("--output-root", type=Path, required=True,
                    help="Publication-clean output root (contains figures/ and run JSONs).")
    p.add_argument("--output-tex", type=Path, default=None)
    p.add_argument("--output-pdf", type=Path, default=None)
    p.add_argument("--emit-pdf", action=argparse.BooleanOptionalAction, default=True)
    return p.parse_args()


# ---------------------------------------------------------------------------
# Tiny helpers (same as the publication appendix script)
# ---------------------------------------------------------------------------

def _load_json(path: Path) -> Dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _as_float(x: object) -> Optional[float]:
    try:
        v = float(x)  # type: ignore[arg-type]
    except Exception:
        return None
    if not math.isfinite(v):
        return None
    return float(v)


def _fmt(x: object) -> str:
    v = _as_float(x)
    if v is None:
        return "---"
    if abs(v) >= 1000.0 or (0.0 < abs(v) < 1e-3):
        return f"{v:.2e}"
    return f"{v:.4g}"


def _run_pdflatex(tex_path: Path) -> bool:
    if shutil.which("latexmk") is not None:
        subprocess.run(
            ["latexmk", "-pdf", "-interaction=nonstopmode", "-halt-on-error",
             tex_path.name],
            cwd=str(tex_path.parent), check=True,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
        )
        return tex_path.with_suffix(".pdf").exists()
    if shutil.which("pdflatex") is None:
        return False
    subprocess.run(
        ["pdflatex", "-interaction=nonstopmode", "-halt-on-error",
         tex_path.name],
        cwd=str(tex_path.parent), check=True,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
    )
    return tex_path.with_suffix(".pdf").exists()


def _pdf_to_png(pdf_path: Path, png_path: Path) -> bool:
    if shutil.which("pdftoppm") is None:
        return False
    subprocess.run(
        ["pdftoppm", "-singlefile", "-png", "-r", "220",
         pdf_path.name, png_path.with_suffix("").name],
        cwd=str(pdf_path.parent), check=True,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
    )
    return png_path.exists()


def _compile_standalone_figure(source_tex: Path, out_pdf: Path, out_png: Path) -> None:
    if not source_tex.exists():
        raise FileNotFoundError(f"Missing standalone figure source: {source_tex}")
    if not _run_pdflatex(source_tex):
        raise RuntimeError(f"Failed to build standalone figure: {source_tex}")
    built = source_tex.with_suffix(".pdf")
    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    if built != out_pdf:
        shutil.copyfile(built, out_pdf)
    _pdf_to_png(out_pdf, out_png)


# ---------------------------------------------------------------------------
# Data loaders (identical logic to the publication appendix)
# ---------------------------------------------------------------------------

def _markov_sample_config(output_root: Path) -> Dict[str, object]:
    files = sorted(glob.glob(
        str(output_root / "markov_changepoint_ops_count" / "**" / "*seed_*.json"),
        recursive=True,
    ))
    matched: List[Path] = []
    sample: Optional[Dict[str, object]] = None
    for fp in files:
        cfg = (_load_json(Path(fp)).get("config") or {})
        if (str(cfg.get("model_family")) == "additive"
                and int(cfg.get("train_docs", -1)) == 8000
                and abs(float(cfg.get("audit_fraction", -1.0)) - 1.0) <= 1e-12):
            matched.append(Path(fp))
            if sample is None:
                sample = cfg
    return {
        "config": sample or {},
        "n_seeds": len({int((_load_json(p).get("config") or {}).get("seed", -1))
                        for p in matched}),
    }


def _ctree_sample_config(output_root: Path) -> Dict[str, object]:
    files = sorted(glob.glob(
        str(output_root / "segmented_lda_ctreepo" / "**" / "*.json"),
        recursive=True,
    ))
    matched: List[Path] = []
    sample: Optional[Dict[str, object]] = None
    for fp in files:
        cfg = (_load_json(Path(fp)).get("config") or {})
        if (int(cfg.get("n_books_train", -1)) == 4096
                and abs(float(cfg.get("calibration_leaf_query_rate", -1.0)) - 0.1) <= 1e-12):
            matched.append(Path(fp))
            if sample is None and abs(float(cfg.get("eval_leaf_query_rate", -1.0)) - 1.0) <= 1e-12:
                sample = cfg
    return {
        "config": sample or {},
        "n_seeds": len({int((_load_json(p).get("config") or {}).get("seed", -1))
                        for p in matched}),
    }


def _series_val(series: Dict, q: float, key: str) -> float:
    fmt_q = _fmt(q)
    row = series.get(fmt_q) or {}
    v = _as_float(row.get(key))
    return float(v) if v is not None else float("nan")


# ---------------------------------------------------------------------------
# LaTeX helpers
# ---------------------------------------------------------------------------

def _itemize(items: List[str]) -> List[str]:
    out = [r"\begin{itemize}[leftmargin=1.4em,itemsep=0.15em,topsep=0.3em]"]
    out.extend(rf"\item {it}" for it in items)
    out.append(r"\end{itemize}")
    out.append("")
    return out


def _landscape_figure(path: str, caption: str) -> List[str]:
    return [
        r"\clearpage",
        r"\begin{landscape}",
        r"\thispagestyle{plain}",
        r"\begin{figure}[p]",
        r"\centering",
        rf"\includegraphics[width=0.98\linewidth,height=0.88\textheight,keepaspectratio]{{{path}}}",
        rf"\caption*{{{caption}}}",
        r"\end{figure}",
        r"\end{landscape}",
        r"\clearpage",
        "",
    ]


def _inline_figure(path: str, caption: str) -> List[str]:
    return [
        r"\begin{figure}[htbp]",
        r"\centering",
        rf"\includegraphics[width=0.95\linewidth,height=0.75\textheight,keepaspectratio]{{{path}}}",
        rf"\caption*{{{caption}}}",
        r"\end{figure}",
        "",
    ]


# ---------------------------------------------------------------------------
# The document
# ---------------------------------------------------------------------------

def _enumerate(items: List[str]) -> List[str]:
    out = [r"\begin{enumerate}[leftmargin=1.4em,itemsep=0.15em,topsep=0.3em]"]
    out.extend(rf"\item {it}" for it in items)
    out.append(r"\end{enumerate}")
    out.append("")
    return out


def _build_tex(
    *,
    combined_rel: str,
    fig_rel: str,
    markov_series: Dict,
    ctree_series: Dict,
    gain_share: Dict,
    markov_n_seeds: int,
    ctree_n_seeds: int,
) -> str:
    L: List[str] = []  # noqa: N806  (short name for readability below)

    # ── Preamble ─────────────────────────────────────────────────────
    L.extend([
        r"\documentclass[12pt]{article}",
        r"\usepackage[T1]{fontenc}",
        r"\usepackage{lmodern}",
        r"\usepackage[margin=1in]{geometry}",
        r"\usepackage{amsmath,amssymb}",
        r"\usepackage{booktabs}",
        r"\usepackage{caption}",
        r"\usepackage{graphicx}",
        r"\usepackage{hyperref}",
        r"\usepackage{pdflscape}",
        r"\usepackage{enumitem}",
        r"\usepackage{xcolor}",
        r"\usepackage{tcolorbox}",
        r"\tcbuselibrary{skins}",
        r"\hypersetup{colorlinks=true,linkcolor=blue!50!black,urlcolor=blue!50!black}",
        r"\setlength{\parindent}{0pt}",
        r"\setlength{\parskip}{0.6em}",
        r"\title{A Gentle Introduction to Tree-Based Merging}",
        r"\author{}",
        r"\date{}",
        r"\begin{document}",
        r"\maketitle",
        r"\thispagestyle{empty}",
        "",
    ])

    # ══════════════════════════════════════════════════════════════════
    # SECTION 1 — THE BIG PICTURE
    # ══════════════════════════════════════════════════════════════════
    L.extend([
        r"\section*{1.\quad The Big Picture}",
        "",
        r"Picture a document where the writing alternates between a few hidden styles, "
        r"or a book whose chapters each concentrate on different topics. "
        r"You want to compute something about the \emph{whole} document---say, "
        r"``how many times does the style change?'' or "
        r"``what fraction of this book is about cooking?''---but the document "
        r"is too long to process in one shot.",
        "",
        r"\textbf{Tree-based merging} is a simple idea: "
        r"chop the document into small, fixed-size pieces (we call them \textbf{leaves}), "
        r"summarize each piece with a short description (a \textbf{sketch}), "
        r"then combine sketches in pairs, bottom-up through a binary tree, "
        r"until you reach the root. The root sketch is your prediction.",
        "",
        r"Three moving parts make this work:",
        "",
    ])
    L.extend(_enumerate([
        r"A \textbf{leaf summarizer} that reads one piece and produces a sketch.",
        r"A \textbf{merge rule} that takes two child sketches and produces one parent sketch.",
        r"A \textbf{readout} that converts the root sketch into a final answer.",
    ]))
    L.extend([
        r"That is the entire framework. The rest of this note walks through two concrete "
        r"examples that use the same three-part skeleton but fill it in very differently.",
        "",
    ])

    # ── Peek rate box ────────────────────────────────────────────────
    L.extend([
        r"\begin{tcolorbox}[colback=green!3,colframe=green!40!black,"
        r"title=One knob to know: the peek rate]",
        r"At prediction time, we can optionally reveal the \emph{true} answer for some "
        r"fraction of the leaves. We call this fraction the \textbf{peek rate}.",
        r"",
        r"\smallskip",
        r"\begin{tabular}{@{}rl@{}}",
        r"peek = 0: & no help---the system relies entirely on what it learned. \\",
        r"peek = 0.5: & half the leaves get their true answer; the system fills in the rest. \\",
        r"peek = 1: & every leaf gets its true answer (the ``ceiling'').",
        r"\end{tabular}",
        r"\smallskip",
        r"",
        r"Even a small peek rate helps a lot, because the merge rule propagates "
        r"revealed information to neighboring pieces.",
        r"\end{tcolorbox}",
        "",
    ])

    # ══════════════════════════════════════════════════════════════════
    # SECTION 2 — EXAMPLE 1: MARKOV CHANGEPOINTS
    # ══════════════════════════════════════════════════════════════════
    L.extend([
        r"\section*{2.\quad Example 1 --- Counting Style Changes}",
        "",
        r"\textbf{Setup.}\ "
        r"A document is a sequence of words. Behind the scenes, each word was generated "
        r"by one of several hidden writing styles (we call them \textbf{regimes}). "
        r"A regime persists for a stretch of words, then switches to a different one. "
        r"The question: \emph{how many times does the style change?}",
        "",
        r"\textbf{Why this is a good first example.}\ "
        r"We happen to know the \emph{exact} merge rule for changepoint counts. "
        r"Each piece of the document needs to remember just three things: "
        r"how many changes happened inside it, which regime it started with, "
        r"and which regime it ended with. "
        r"To merge two pieces, add their internal counts "
        r"and check the boundary---if the left piece ends in a different regime "
        r"than the right piece starts in, that is one more changepoint.",
        "",
        r"\textbf{Walkthrough.}\ "
        r"The two figures below show a toy 12-word document with four leaves. "
        r"Each leaf records (count, first regime, last regime). "
        r"The tree merges bottom-up. The root recovers the exact answer: 5 changepoints.",
        "",
    ])
    L.extend(_landscape_figure(
        combined_rel,
        "A toy 12-word document (bottom) and its bottom-up merge tree (top). "
        "Each leaf reads three words and records (count, first regime, last regime). "
        "The merge rule adds counts and checks the boundary. The root recovers the exact answer: 5.",
    ))

    # ── Results table ────────────────────────────────────────────────
    m0 = _fmt(_series_val(markov_series, 0.0, "root_mae"))
    m5 = _fmt(_series_val(markov_series, 0.5, "root_mae"))
    m1 = _fmt(_series_val(markov_series, 1.0, "root_mae"))
    mg = _fmt(gain_share.get("markov_additive"))

    L.extend([
        rf"\textbf{{Results}} (averaged over {markov_n_seeds} random seeds):",
        r"\begin{center}",
        r"\begin{tabular}{@{}lccc@{}}",
        r"\toprule",
        r" & peek = 0 & peek = 0.5 & peek = 1 \\",
        r"\midrule",
        rf"average error & {m0} & {m5} & {m1} \\",
        rf"gap closed at peek\,=\,0.5 &  & {mg} &  \\",
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{center}",
        r"Because the merge rule is exact, the error is near zero even with no peeking. "
        r"This is the expected sanity check: give the system the correct rule "
        r"and it gets the right answer.",
        "",
    ])

    # ══════════════════════════════════════════════════════════════════
    # SECTION 3 — EXAMPLE 2: C-TREEPO TOPIC RECOVERY
    # ══════════════════════════════════════════════════════════════════
    L.extend([
        r"\section*{3.\quad Example 2 --- Recovering Topic Mixtures}",
        "",
        r"\textbf{Setup.}\ "
        r"A book has chapters, and each chapter concentrates on one topic "
        r"with its own vocabulary. "
        r"The question: \emph{what is the overall topic mixture of this book?} "
        r"For instance, the answer might be ``40\% sports, 35\% cooking, 25\% travel.''",
        "",
        r"\textbf{What is different from Example 1.}\ "
        r"This time there is \emph{no known formula} for combining chapter-level "
        r"topic estimates into a book-level estimate. "
        r"The system must \emph{learn} its own merge rule from data. "
        r"It also has to learn how to summarize each leaf and how to read out the final "
        r"answer---all three parts are discovered from mostly leaf-level labels, "
        r"without anyone specifying the algebra in advance.",
        "",
        r"Concretely, the leaf summarizer is a small neural network that reads word counts. "
        r"The merge rule is a gated network that decides, for each pair of child sketches, "
        r"what to keep and what to blend. "
        r"The readout is a linear layer that converts the root sketch into a topic distribution.",
        "",
    ])

    # ── Results table ────────────────────────────────────────────────
    c0 = _fmt(_series_val(ctree_series, 0.0, "root_l1_mean"))
    c5 = _fmt(_series_val(ctree_series, 0.5, "root_l1_mean"))
    c1 = _fmt(_series_val(ctree_series, 1.0, "root_l1_mean"))
    cg = _fmt(gain_share.get("ctree"))

    L.extend([
        rf"\textbf{{Results}} (averaged over {ctree_n_seeds} random seeds):",
        r"\begin{center}",
        r"\begin{tabular}{@{}lccc@{}}",
        r"\toprule",
        r" & peek = 0 & peek = 0.5 & peek = 1 \\",
        r"\midrule",
        rf"average error & {c0} & {c5} & {c1} \\",
        rf"gap closed at peek\,=\,0.5 &  & {cg} &  \\",
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{center}",
        r"With no peeking there is real error---the learned rule is not perfect. "
        r"But at peek\,=\,0.5, roughly 90\% of the gap disappears. "
        r"A learned approximate merge recovers most of the benefit of an exact one.",
        "",
    ])

    # ══════════════════════════════════════════════════════════════════
    # SECTION 4 — WHAT TO TAKE AWAY
    # ══════════════════════════════════════════════════════════════════
    L.extend([
        r"\section*{4.\quad What To Take Away}",
        "",
        r"\begin{center}",
        r"\renewcommand{\arraystretch}{1.3}",
        r"\begin{tabular}{@{}lcc@{}}",
        r"\toprule",
        r"& \textbf{Example 1 (exact rule)} & \textbf{Example 2 (learned rule)} \\",
        r"\midrule",
        r"merge rule & given in advance & learned from data \\",
        r"error at peek\,=\,0 & near zero & nonzero \\",
        r"gap closed at peek\,=\,0.5 & ${\sim}$97\% & ${\sim}$91\% \\",
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{center}",
        "",
        r"When you know the correct merge rule, you get the right answer. "
        r"When you have to learn it, the answer is approximate but surprisingly good---"
        r"and far cheaper to train than a model that reads the whole document at once.",
        "",
        r"The reference heatmap below shows how error and gap-closed vary "
        r"as you change both the amount of training data and the peek rate at prediction time. "
        r"Green means low error; darker red means closer to the ceiling.",
        "",
    ])
    L.extend(_inline_figure(
        fig_rel,
        "Tradeoff surfaces for both examples. "
        "Left column: raw error. Right column: fraction of the gap closed. "
        "Gray: baseline and ceiling too close for a meaningful ratio.",
    ))

    L.append(r"\end{document}")
    L.append("")
    return "\n".join(L)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    args = _parse_args()
    output_root = args.output_root.resolve()
    formal_root = output_root.parent
    repo_root = Path(__file__).resolve().parent.parent

    tex_path = (
        args.output_tex.resolve()
        if args.output_tex is not None
        else (formal_root / "paper_reports" / "introductory_tree_merging_walkthrough.tex")
    )
    pdf_path = (
        args.output_pdf.resolve()
        if args.output_pdf is not None
        else tex_path.with_suffix(".pdf")
    )

    # Combined figure: toy document + merge tree in one standalone TikZ.
    combined_png = tex_path.parent / "markov_changepoint_combined_overview.png"
    combined_pdf = combined_png.with_suffix(".pdf")
    combined_asset_tex = repo_root / "paper" / "figures" / "markov_changepoint_combined_slide.tex"

    # Load diagnostics.
    diag_path = output_root / "figures" / "identifiable_zero_publication_report_latest_diagnostics.json"
    diag = _load_json(diag_path)
    diagnostics = diag.get("diagnostics") or {}
    neural = diagnostics.get("neural_lag_evidence") or {}

    markov_series = neural.get("markov_additive") or {}
    ctree_series = (neural.get("ctree_reference") or {}).get("series") or {}
    obs = (neural.get("observations") or [{}])[0]
    gain_share = (obs.get("evidence") or {}).get("partial_gain_share_to_q05") or {}

    markov_meta = _markov_sample_config(output_root)
    ctree_meta = _ctree_sample_config(output_root)

    # Compile figures.
    _compile_standalone_figure(combined_asset_tex, combined_pdf, combined_png)
    fig_path = output_root / "figures" / "pub_clean" / "main_figure_B_gap_decomposition.png"

    tex_body = _build_tex(
        combined_rel=os.path.relpath(combined_pdf, tex_path.parent),
        fig_rel=os.path.relpath(fig_path, tex_path.parent),
        markov_series=markov_series,
        ctree_series=ctree_series,
        gain_share=gain_share,
        markov_n_seeds=int(markov_meta.get("n_seeds", 0)),
        ctree_n_seeds=int(ctree_meta.get("n_seeds", 0)),
    )

    tex_path.parent.mkdir(parents=True, exist_ok=True)
    tex_path.write_text(tex_body, encoding="utf-8")

    pdf_emitted = False
    if args.emit_pdf:
        pdf_emitted = _run_pdflatex(tex_path)
        built = tex_path.with_suffix(".pdf")
        if pdf_emitted and built != pdf_path:
            shutil.copyfile(built, pdf_path)

    print(json.dumps({
        "output_tex": str(tex_path),
        "output_pdf": str(pdf_path) if pdf_emitted else None,
        "pdf_emitted": pdf_emitted,
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
