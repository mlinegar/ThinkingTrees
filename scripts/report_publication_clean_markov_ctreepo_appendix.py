#!/usr/bin/env python3
"""Write an appendix-style walkthrough for the publication-clean Markov/C-TreePO story."""

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
from typing import Dict, Iterable, List, Optional

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle


REGIME_COLORS = {
    "A": "#4C78A8",
    "B": "#F58518",
    "C": "#54A24B",
    "D": "#E45756",
}
REGIME_VOCAB = {
    "A": ["mist", "lake", "reed"],
    "B": ["rust", "ember", "brick"],
    "C": ["fern", "moss", "leaf"],
    "D": ["plum", "rose", "wine"],
}


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Emit appendix-ready Markov additive vs C-TreePO walkthrough.")
    p.add_argument("--output-root", type=Path, required=True)
    p.add_argument("--output-tex", type=Path, default=None)
    p.add_argument("--output-markdown", type=Path, default=None)
    p.add_argument("--output-pdf", type=Path, default=None)
    p.add_argument("--emit-pdf", action=argparse.BooleanOptionalAction, default=True)
    return p.parse_args()


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
        return "nan"
    if abs(v) >= 1000.0 or (0.0 < abs(v) < 1e-3):
        return f"{v:.3e}"
    return f"{v:.6g}"


def _run_pandoc(md_path: Path, pdf_path: Path) -> bool:
    if shutil.which("pandoc") is None or shutil.which("pdflatex") is None:
        return False
    subprocess.run(
        [
            "pandoc",
            str(md_path.name),
            "-o",
            str(pdf_path.name),
            "--pdf-engine=pdflatex",
        ],
        cwd=str(md_path.parent),
        check=True,
    )
    return True


def _run_pdflatex(tex_path: Path) -> bool:
    if shutil.which("latexmk") is not None:
        subprocess.run(
            [
                "latexmk",
                "-pdf",
                "-interaction=nonstopmode",
                "-halt-on-error",
                tex_path.name,
            ],
            cwd=str(tex_path.parent),
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        return tex_path.with_suffix(".pdf").exists()
    if shutil.which("pdflatex") is None:
        return False
    subprocess.run(
        [
            "pdflatex",
            "-interaction=nonstopmode",
            "-halt-on-error",
            tex_path.name,
        ],
        cwd=str(tex_path.parent),
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    return tex_path.with_suffix(".pdf").exists()


def _pdf_to_png(pdf_path: Path, png_path: Path) -> bool:
    if shutil.which("pdftoppm") is None:
        return False
    out_prefix = png_path.with_suffix("")
    subprocess.run(
        [
            "pdftoppm",
            "-singlefile",
            "-png",
            "-r",
            "220",
            pdf_path.name,
            out_prefix.name,
        ],
        cwd=str(pdf_path.parent),
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    return png_path.exists()


def _write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _markov_dgp_tikz_source() -> str:
    return r"""\documentclass[tikz,border=4mm]{standalone}
\usepackage[T1]{fontenc}
\usepackage{amsmath,amssymb}
\usepackage{helvet}
\renewcommand{\familydefault}{\sfdefault}
\usepackage{xcolor}
\usepackage{tikz}
\usetikzlibrary{positioning,calc,fit,backgrounds,decorations.pathreplacing}
\definecolor{regA}{HTML}{4C78A8}
\definecolor{regB}{HTML}{F58518}
\definecolor{regC}{HTML}{54A24B}
\definecolor{regD}{HTML}{E45756}
\begin{document}
\begin{tikzpicture}[x=1cm,y=1cm]
  \tikzset{
    title/.style={font=\bfseries\fontsize{22}{24}\selectfont, anchor=west},
    subtitle/.style={font=\fontsize{11}{13}\selectfont, text=black!65, anchor=west},
    sectionhead/.style={font=\bfseries\fontsize{17}{19}\selectfont, anchor=west},
    body/.style={font=\fontsize{11}{13}\selectfont, text=black!85, anchor=west},
    smallbody/.style={font=\fontsize{10}{12}\selectfont, text=black!65, anchor=west},
    callout/.style={draw=black!35, rounded corners=4pt, inner sep=4mm, line width=0.9pt},
    tokenbox/.style={draw=black!20, rounded corners=2pt, minimum width=1.06cm, minimum height=0.72cm, inner sep=1pt, font=\fontsize{11}{12}\selectfont},
    chip/.style={rounded corners=3pt, minimum width=1.85cm, minimum height=0.72cm, inner sep=2pt},
  }

  \node[title] at (0, 8.65) {Markov Changepoint DGP: What Generates One Document?};
  \node[subtitle] at (0, 8.15) {Latent regimes stay constant for stretches, regime-specific vocabularies emit words, and the oracle target counts flips.};

  \node[sectionhead] at (0.2, 7.45) {Step 1. Sample the latent regime path};
  \node[body] at (0.25, 6.93) {Latent path: $z_{1:T}$ is piecewise constant.};
  \node[body] at (0.25, 6.48) {Changepoint rule: count $1$ exactly when $z_t \neq z_{t+1}$.};
  \node[body] at (0.25, 5.82) {$f^\star(x)=\sum_{t=1}^{T-1}\mathbf{1}\{z_t \neq z_{t+1}\}$};
  \begin{scope}[on background layer]
    \node[callout, draw=regA!70!black, fill=regA!5, fit={(0.0,5.35) (6.3,7.75)}] {};
  \end{scope}

  \node[sectionhead] at (6.9, 7.45) {Step 2. Each regime emits its own vocabulary};
  \node[body] at (6.95, 6.93) {Token model: $x_t \mid z_t=r \sim \mathrm{Cat}(\phi_r)$.};
  \node[smallbody] at (6.95, 6.48) {Each color below shows a toy high-probability vocabulary for one latent regime.};
  \node[chip, draw=regA!85!black, fill=regA!18, anchor=west] at (6.95, 5.75) {};
  \node[chip, draw=regB!85!black, fill=regB!18, anchor=west] at (9.02, 5.75) {};
  \node[chip, draw=regC!85!black, fill=regC!18, anchor=west] at (11.09, 5.75) {};
  \node[chip, draw=regD!85!black, fill=regD!18, anchor=west] at (13.16, 5.75) {};
  \node[font=\bfseries\fontsize{11}{12}\selectfont, anchor=west] at (7.18, 5.92) {Regime A};
  \node[font=\fontsize{10}{12}\selectfont, anchor=west, text=black!80] at (7.18, 5.55) {mist, lake, reed};
  \node[font=\bfseries\fontsize{11}{12}\selectfont, anchor=west] at (9.25, 5.92) {Regime B};
  \node[font=\fontsize{10}{12}\selectfont, anchor=west, text=black!80] at (9.25, 5.55) {rust, ember, brick};
  \node[font=\bfseries\fontsize{11}{12}\selectfont, anchor=west] at (11.32, 5.92) {Regime C};
  \node[font=\fontsize{10}{12}\selectfont, anchor=west, text=black!80] at (11.32, 5.55) {fern, moss, leaf};
  \node[font=\bfseries\fontsize{11}{12}\selectfont, anchor=west] at (13.39, 5.92) {Regime D};
  \node[font=\fontsize{10}{12}\selectfont, anchor=west, text=black!80] at (13.39, 5.55) {plum, rose, wine};
  \begin{scope}[on background layer]
    \node[callout, draw=regB!80!black, fill=regB!6, fit={(6.7,5.28) (15.15,7.75)}] {};
  \end{scope}

  \node[sectionhead] at (0, 4.72) {Step 3. One toy document drawn from the DGP};
  \node[body] at (0, 4.28) {Top row = observed word token. Bottom strip = latent regime. Dashed boundaries mark the 5 true changepoints.};
  \node[font=\fontsize{11}{13}\selectfont, anchor=east, align=right] at (1.05, 3.15) {observed\\word};
  \node[font=\fontsize{11}{13}\selectfont, anchor=east, align=right] at (1.05, 2.38) {latent\\regime};

  \foreach \word/\reg/\col/\x in {
    mist/A/regA/1.5,
    lake/A/regA/2.63,
    reed/A/regA/3.76,
    rust/B/regB/4.89,
    brick/B/regB/6.02,
    fern/C/regC/7.15,
    moss/C/regC/8.28,
    plum/D/regD/9.41,
    wine/D/regD/10.54,
    lake/A/regA/11.67,
    mist/A/regA/12.80,
    ember/B/regB/13.93
  }{
    \node[tokenbox, draw=\col!85!black, anchor=west] at (\x, 2.8) {\word};
    \fill[\col] (\x, 1.98) rectangle ++(1.05, 0.5);
    \node[font=\bfseries\fontsize{11}{12}\selectfont, text=white] at (\x+0.525, 2.23) {\reg};
  }

  \foreach \n/\x in {1/2.025,2/3.155,3/4.285,4/5.415,5/6.545,6/7.675,7/8.805,8/9.935,9/11.065,10/12.195,11/13.325,12/14.455}{
    \node[font=\fontsize{9}{10}\selectfont, text=black!55] at (\x, 1.74) {\n};
  }
  \foreach \x in {4.89,7.15,9.41,11.67,13.93}{
    \draw[dashed, line width=0.8pt, black!55] (\x, 1.55) -- (\x, 4.02);
  }

  \node[callout, draw=regC!80!black, fill=regC!6, anchor=west, text width=3.8cm] at (11.8, 3.55) {\textbf{Oracle answer on this toy document}\\[0.8mm]$f^\star(x)=5$ because there are $5$ regime flips.};
  \node[body] at (4.75, 1.24) {$z_{1:T} = (A,A,A,B,B,C,C,D,D,A,A,B)$};

  \node[sectionhead] at (0, 0.56) {Step 4. The simulation fixes a leaf partition before building the tree};
  \node[smallbody] at (0, 0.16) {These four gray brackets are the toy leaves. The next figure uses the same four spans to show the exact merge sketch.};
  \draw[decorate, decoration={brace, amplitude=4pt}, black!50] (1.5,-0.12) -- (4.81,-0.12);
  \draw[decorate, decoration={brace, amplitude=4pt}, black!50] (4.89,-0.12) -- (8.20,-0.12);
  \draw[decorate, decoration={brace, amplitude=4pt}, black!50] (8.28,-0.12) -- (11.59,-0.12);
  \draw[decorate, decoration={brace, amplitude=4pt}, black!50] (11.67,-0.12) -- (14.98,-0.12);
  \node[font=\fontsize{10}{12}\selectfont, text=black!75] at (3.16, -0.65) {L1 = tokens 1--3};
  \node[font=\fontsize{10}{12}\selectfont, text=black!75] at (6.55, -0.65) {L2 = tokens 4--6};
  \node[font=\fontsize{10}{12}\selectfont, text=black!75] at (9.94, -0.65) {L3 = tokens 7--9};
  \node[font=\fontsize{10}{12}\selectfont, text=black!75] at (13.33, -0.65) {L4 = tokens 10--12};
\end{tikzpicture}
\end{document}
"""


def _markov_merge_tikz_source() -> str:
    return r"""\documentclass[tikz,border=4mm]{standalone}
\usepackage[T1]{fontenc}
\usepackage{amsmath,amssymb}
\usepackage{helvet}
\renewcommand{\familydefault}{\sfdefault}
\usepackage{xcolor}
\usepackage{tikz}
\usetikzlibrary{positioning,calc,fit,backgrounds,arrows.meta,decorations.pathreplacing}
\definecolor{regA}{HTML}{4C78A8}
\definecolor{regB}{HTML}{F58518}
\definecolor{regC}{HTML}{54A24B}
\definecolor{regD}{HTML}{E45756}
\begin{document}
\begin{tikzpicture}[x=1cm,y=1cm, >=Latex]
  \tikzset{
    title/.style={font=\bfseries\fontsize{22}{24}\selectfont, anchor=west},
    subtitle/.style={font=\fontsize{11}{13}\selectfont, text=black!65, anchor=west},
    sectionhead/.style={font=\bfseries\fontsize{17}{19}\selectfont, anchor=west},
    body/.style={font=\fontsize{11}{13}\selectfont, text=black!85, anchor=west},
    smallbody/.style={font=\fontsize{10}{12}\selectfont, text=black!65, anchor=west},
    callout/.style={draw=black!35, rounded corners=4pt, inner sep=4mm, line width=0.9pt},
    summary/.style={draw=black!45, rounded corners=4pt, minimum width=2.6cm, minimum height=1.15cm, align=center, inner sep=3pt},
  }

  \node[title] at (0, 8.45) {Exact Markov Sketch: Why The Tree Can Merge Without Losing Information};
  \node[subtitle] at (0, 7.95) {Each span stores changepoints inside the span plus the first and last latent regimes, and that is enough to merge exactly.};

  \node[sectionhead] at (0.2, 7.1) {Exact sketch stored at every node};
  \node[body] at (0.25, 6.52) {$S(u) = \big(c(u), a(u), b(u)\big)$};
  \node[smallbody] at (0.25, 6.03) {$c(u)$ = changepoints inside the span, $a(u)$ = first regime, $b(u)$ = last regime.};
  \begin{scope}[on background layer]
    \node[callout, draw=regA!70!black, fill=regA!5, fit={(0.0,5.55) (6.45,7.35)}] {};
  \end{scope}

  \node[sectionhead] at (7.0, 7.1) {Exact merge rule};
  \node[body] at (7.05, 6.52) {$S(u_L)\otimes S(u_R)=\big(c_L+c_R+\mathbf{1}\{b_L\neq a_R\},\, a_L,\, b_R\big)$};
  \node[smallbody] at (7.05, 6.03) {There is only one boundary correction: add $1$ if the left span ends in a different regime than the right span begins.};
  \begin{scope}[on background layer]
    \node[callout, draw=regB!80!black, fill=regB!6, fit={(6.8,5.55) (15.3,7.35)}] {};
  \end{scope}

  \node[sectionhead] at (0, 4.95) {Same toy document, now reduced to four fixed leaves};
  \node[smallbody] at (0, 4.55) {Top strip shows the latent regimes only. Each gray bracket is one leaf, and the box below it is that leaf's exact summary.};
  \node[font=\fontsize{11}{13}\selectfont, anchor=east, align=right] at (1.05, 3.72) {latent\\regime};
  \foreach \reg/\col/\x in {
    A/regA/1.5,
    A/regA/2.63,
    A/regA/3.76,
    B/regB/4.89,
    B/regB/6.02,
    C/regC/7.15,
    C/regC/8.28,
    D/regD/9.41,
    D/regD/10.54,
    A/regA/11.67,
    A/regA/12.80,
    B/regB/13.93
  }{
    \fill[\col] (\x, 3.45) rectangle ++(1.05, 0.5);
    \node[font=\bfseries\fontsize{11}{12}\selectfont, text=white] at (\x+0.525, 3.70) {\reg};
  }
  \foreach \n/\x in {1/2.025,2/3.155,3/4.285,4/5.415,5/6.545,6/7.675,7/8.805,8/9.935,9/11.065,10/12.195,11/13.325,12/14.455}{
    \node[font=\fontsize{9}{10}\selectfont, text=black!55] at (\x, 3.15) {\n};
  }
  \draw[decorate, decoration={brace, amplitude=4pt}, black!50] (1.5,2.92) -- (4.81,2.92);
  \draw[decorate, decoration={brace, amplitude=4pt}, black!50] (4.89,2.92) -- (8.20,2.92);
  \draw[decorate, decoration={brace, amplitude=4pt}, black!50] (8.28,2.92) -- (11.59,2.92);
  \draw[decorate, decoration={brace, amplitude=4pt}, black!50] (11.67,2.92) -- (14.98,2.92);
  \node[font=\fontsize{10}{12}\selectfont, text=black!75] at (3.16, 2.52) {L1 = tokens 1--3};
  \node[font=\fontsize{10}{12}\selectfont, text=black!75] at (6.55, 2.52) {L2 = tokens 4--6};
  \node[font=\fontsize{10}{12}\selectfont, text=black!75] at (9.94, 2.52) {L3 = tokens 7--9};
  \node[font=\fontsize{10}{12}\selectfont, text=black!75] at (13.33, 2.52) {L4 = tokens 10--12};

  \node[summary, fill=black!3] (l1) at (2.7, 1.45) {\textbf{L1}\\[0.3mm]$S=(0,A,A)$};
  \node[summary, fill=black!3] (l2) at (6.1, 1.45) {\textbf{L2}\\[0.3mm]$S=(1,B,C)$};
  \node[summary, fill=black!3] (l3) at (9.5, 1.45) {\textbf{L3}\\[0.3mm]$S=(1,C,D)$};
  \node[summary, fill=black!3] (l4) at (12.9, 1.45) {\textbf{L4}\\[0.3mm]$S=(1,A,B)$};

  \node[summary, fill=regA!10] (m12) at (4.4, -0.15) {\textbf{M12}\\[0.3mm]$S=(2,A,C)$};
  \node[summary, fill=regA!10] (m34) at (11.2, -0.15) {\textbf{M34}\\[0.3mm]$S=(3,C,B)$};
  \node[summary, fill=regC!10] (root) at (7.8, -1.85) {\textbf{Root}\\[0.3mm]$S=(5,A,B)$};

  \draw[black!45, line width=0.9pt] (l1.south) -- (m12.north west);
  \draw[black!45, line width=0.9pt] (l2.south) -- (m12.north east);
  \draw[black!45, line width=0.9pt] (l3.south) -- (m34.north west);
  \draw[black!45, line width=0.9pt] (l4.south) -- (m34.north east);
  \draw[black!45, line width=0.9pt] (m12.south) -- (root.north west);
  \draw[black!45, line width=0.9pt] (m34.south) -- (root.north east);

  \node[font=\fontsize{10}{11}\selectfont, text=black!70] at (4.4, 0.55) {$0 + 1 + 1 = 2$ because $A \neq B$};
  \node[font=\fontsize{10}{11}\selectfont, text=black!70] at (11.2, 0.55) {$1 + 1 + 1 = 3$ because $D \neq A$};
  \node[font=\bfseries\fontsize{10}{11}\selectfont, text=black!80] at (7.8, -1.0) {$2 + 3 + 0 = 5$ because $C = C$};

  \node[smallbody] at (0, -2.95) {Interpretation: the root summary gives the exact oracle answer $5$, so this tree loses no information about the changepoint-count target.};
\end{tikzpicture}
\end{document}
"""


def _emit_tikz_slide_figure(tex_path: Path, pdf_path: Path, png_path: Path, source: str) -> None:
    _write_text(tex_path, source)
    if not _run_pdflatex(tex_path):
        raise RuntimeError(f"Failed to build TeX figure: {tex_path}")
    built_pdf = tex_path.with_suffix(".pdf")
    if built_pdf != pdf_path:
        shutil.copyfile(built_pdf, pdf_path)
    if not _pdf_to_png(pdf_path, png_path):
        raise RuntimeError(f"Failed to convert PDF preview for figure: {pdf_path}")


def _compile_standalone_figure_asset(source_tex: Path, out_pdf: Path, out_png: Path) -> None:
    if not source_tex.exists():
        raise FileNotFoundError(f"Missing standalone figure source: {source_tex}")
    if not _run_pdflatex(source_tex):
        raise RuntimeError(f"Failed to build standalone figure asset: {source_tex}")
    built_pdf = source_tex.with_suffix(".pdf")
    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    if built_pdf != out_pdf:
        shutil.copyfile(built_pdf, out_pdf)
    if not _pdf_to_png(out_pdf, out_png):
        raise RuntimeError(f"Failed to emit PNG preview for standalone figure: {out_pdf}")


def _markov_sample_config(output_root: Path) -> Dict[str, object]:
    files = sorted(
        glob.glob(str(output_root / "markov_changepoint_ops_count" / "**" / "*seed_*.json"), recursive=True)
    )
    matched: List[Path] = []
    sample: Optional[Dict[str, object]] = None
    for fp in files:
        payload = _load_json(Path(fp))
        cfg = payload.get("config") or {}
        if (
            str(cfg.get("model_family")) == "additive"
            and int(cfg.get("train_docs", -1)) == 8000
            and abs(float(cfg.get("audit_fraction", -1.0)) - 1.0) <= 1e-12
        ):
            matched.append(Path(fp))
            if sample is None:
                sample = cfg
    return {
        "config": sample or {},
        "n_seed_files": len(matched),
        "n_seeds": len({int((_load_json(path).get("config") or {}).get("seed", -1)) for path in matched}),
    }


def _ctree_sample_config(output_root: Path) -> Dict[str, object]:
    files = sorted(glob.glob(str(output_root / "segmented_lda_ctreepo" / "**" / "*.json"), recursive=True))
    matched: List[Path] = []
    sample: Optional[Dict[str, object]] = None
    for fp in files:
        payload = _load_json(Path(fp))
        cfg = payload.get("config") or {}
        if (
            int(cfg.get("n_books_train", -1)) == 4096
            and abs(float(cfg.get("calibration_leaf_query_rate", -1.0)) - 0.1) <= 1e-12
        ):
            matched.append(Path(fp))
            if sample is None and abs(float(cfg.get("eval_leaf_query_rate", -1.0)) - 1.0) <= 1e-12:
                sample = cfg
    return {
        "config": sample or {},
        "n_seed_files": len(matched),
        "n_seeds": len({int((_load_json(path).get("config") or {}).get("seed", -1)) for path in matched}),
    }


def _series_value(series: Dict[str, Dict[str, float]], q: float, key: str) -> float:
    row = series.get(_fmt(q)) or {}
    v = _as_float(row.get(key))
    return float(v) if v is not None else float("nan")


def _paragraph(lines: List[str], *parts: str) -> None:
    lines.extend(parts)
    lines.append("")


def _tex_escape(text: object) -> str:
    s = str(text)
    return (
        s.replace("\\", r"\textbackslash{}")
        .replace("&", r"\&")
        .replace("%", r"\%")
        .replace("$", r"\$")
        .replace("#", r"\#")
        .replace("_", r"\_")
        .replace("{", r"\{")
        .replace("}", r"\}")
        .replace("~", r"\textasciitilde{}")
        .replace("^", r"\textasciicircum{}")
    )


def _tt(value: object) -> str:
    return r"\texttt{" + _tex_escape(value) + "}"


def _tex_itemize(items: List[str]) -> List[str]:
    out = [r"\begin{itemize}[leftmargin=1.5em,itemsep=0.2em,topsep=0.35em]"]
    out.extend([rf"\item {item}" for item in items])
    out.append(r"\end{itemize}")
    out.append("")
    return out


def _tex_enumerate(items: List[str]) -> List[str]:
    out = [r"\begin{enumerate}[leftmargin=1.5em,itemsep=0.2em,topsep=0.35em]"]
    out.extend([rf"\item {item}" for item in items])
    out.append(r"\end{enumerate}")
    out.append("")
    return out


def _appendix_tex_figure(path: str, caption: str, *, landscape: bool = True) -> List[str]:
    if landscape:
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
    return [
        r"\begin{figure}[htbp]",
        r"\centering",
        rf"\includegraphics[width=0.98\linewidth,height=0.80\textheight,keepaspectratio]{{{path}}}",
        rf"\caption*{{{caption}}}",
        r"\end{figure}",
        "",
    ]


def _build_appendix_tex(
    *,
    generated: str,
    output_root: Path,
    diag_path: Path,
    dgp_rel: str,
    merge_rel: str,
    fig_rel: str,
    markov_cfg: Dict[str, object],
    ctree_cfg: Dict[str, object],
    markov_meta: Dict[str, object],
    ctree_meta: Dict[str, object],
    markov_series: Dict[str, Dict[str, float]],
    ctree_series: Dict[str, Dict[str, float]],
    gain_share: Dict[str, object],
) -> str:
    lines: List[str] = [
        r"\documentclass[11pt]{article}",
        r"\usepackage[T1]{fontenc}",
        r"\usepackage{lmodern}",
        r"\usepackage[margin=0.9in]{geometry}",
        r"\usepackage{amsmath,amssymb}",
        r"\usepackage{booktabs}",
        r"\usepackage{caption}",
        r"\usepackage{graphicx}",
        r"\usepackage{hyperref}",
        r"\usepackage{pdflscape}",
        r"\usepackage{enumitem}",
        r"\usepackage{xcolor}",
        r"\hypersetup{colorlinks=true,linkcolor=blue!50!black,urlcolor=blue!50!black}",
        r"\setlength{\parindent}{0pt}",
        r"\setlength{\parskip}{0.55em}",
        r"\title{Appendix Walkthrough: Markov Additive and C-TreePO}",
        r"\date{}",
        r"\begin{document}",
        r"\maketitle",
        rf"\textbf{{Generated:}} {_tt(generated)}\\",
        rf"\textbf{{Clean publication root:}} \path{{{output_root}}}\\",
        rf"\textbf{{Diagnostics source:}} \path{{{diag_path}}}",
        "",
        r"\section*{1. Why Start Here}",
        "This appendix is meant to be read like a lecture, not like a result dump.",
        "The teaching order is deliberate. We start with the Markov additive family because it is the cleanest theorem-matched example: the oracle target is simple, the exact mergeable statistic is explicit, and the additive learned family is given the correct merge rule. Then we move to C-TreePO, which no longer knows the exact merge rule \\emph{a priori} but can still recover the same qualitative story quickly once it gets a modest amount of supervision.",
        "The intended message is not that the two raw error scales should be compared directly. They should not. The intended message is:",
        "",
    ]
    lines.extend(
        _tex_enumerate(
            [
                "Markov additive is the closest empirical proxy to the exact DGP-plus-correct-merge-rule story.",
                "C-TreePO is a richer approximate system, but with some labels it closes most of its own gap quickly.",
                "That makes the Markov example the simplest lecture version, and C-TreePO the next approximation step.",
            ]
        )
    )

    lines.extend(
        [
            r"\section*{2. The Markov Example}",
            r"\subsection*{2.1 Data-generating process}",
            r"A document is a token sequence $x = (x_1, \ldots, x_T)$ with a latent regime path $z_1, \ldots, z_T \in \{1,\ldots,K\}$. In the fixed publication slice:",
            "",
        ]
    )
    lines.extend(
        _tex_itemize(
            [
                rf"number of latent regimes: {_tt(markov_cfg.get('n_regimes'))}",
                rf"vocabulary size: {_tt(markov_cfg.get('vocab_size'))}",
                rf"token count per document: {_tt(markov_cfg.get('min_tokens'))} to {_tt(markov_cfg.get('max_tokens'))}",
                rf"realized leaf size: {_tt(markov_cfg.get('fixed_leaf_tokens'))} tokens",
            ]
        )
    )
    lines.extend(
        [
            r"The latent regime is piecewise constant. A changepoint occurs exactly when the regime flips between adjacent tokens. The oracle target is therefore",
            r"\[",
            r"f^\star(x) = \sum_{t=1}^{T-1} \mathbf{1}\{z_t \neq z_{t+1}\}.",
            r"\]",
            r"In plain language: count the true changepoints.",
            r"A concrete way to read the DGP is:",
            "",
        ]
    )
    lines.extend(
        _tex_enumerate(
            [
                r"draw a latent regime path $z_{1:T}$ that stays constant for stretches and occasionally flips,",
                r"for each token position $t$, emit the observed word $x_t$ from the regime-specific vocabulary attached to $z_t$,",
                r"cut the document into fixed leaves, and",
                r"summarize each leaf by the exact sketch $S=(\text{count}, \text{first}, \text{last})$ before merging upward.",
            ]
        )
    )
    lines.append(
        r"A small toy slide helps fix the objects before looking at the large sweep. The first figure below is only about the DGP: what is latent, what is observed, and what the oracle counts."
    )
    lines.append("")
    lines.extend(
        _appendix_tex_figure(
            dgp_rel,
            "Toy Markov DGP: latent regime path, regime-specific vocabularies, observed words, oracle target, and fixed leaf partition.",
        )
    )
    lines.extend(
        [
            r"\subsection*{2.2 Exact mergeable statistic}",
            r"For any span $u$, the exact sketch keeps three pieces of information:",
            r"\[",
            r"S(u) = (c(u), a(u), b(u)),",
            r"\]",
            r"where $c(u)$ is the number of changepoints inside the span, $a(u)$ is the first regime in the span, and $b(u)$ is the last regime in the span. If we merge a left child and a right child, the correct merge rule is",
            r"\[",
            r"S(u_L) \otimes S(u_R) = \big(c_L + c_R + \mathbf{1}\{b_L \neq a_R\},\; a_L,\; b_R\big).",
            r"\]",
            r"This is the worked example formalized in \texttt{lean3/FormalProofs/OPT/MarkovCountSketchExample.lean}: the exact theorem-domain state is \texttt{MarkovCountSketch.empty} or \texttt{MarkovCountSketch.nonempty count first last}, the monoid product is the merge law above, and the root oracle reads out the \texttt{count} coordinate.",
            r"In the simulation code, the helper \texttt{\_ExactState(count, first, last)} uses the same exact semantics, and the additive learned family stores the same three semantics in numeric form as a tensor \texttt{[count\_norm, first\_one\_hot, last\_one\_hot]}. In the plotted publication slice, \texttt{feature\_mode=full}, so the endpoint channels are copied exactly from the leaf features and only the normalized count coordinate is fit from labels. So when this appendix writes $S=(\mathrm{count}, \mathrm{first}, \mathrm{last})$, it is naming the theorem-facing coordinates, not claiming that the implementation literally stores a symbolic tuple at runtime.",
            r"The second toy slide uses the same four leaves and shows the exact merge arithmetic explicitly.",
            "",
        ]
    )
    lines.extend(
        _appendix_tex_figure(
            merge_rel,
            "Toy exact merge sketch: each leaf stores $S=(\\mathrm{count}, \\mathrm{first}, \\mathrm{last})$, and the root recovers the exact changepoint count.",
        )
    )
    lines.extend(
        [
            r"\subsection*{2.3 What is fixed, and what is learned}",
            r"The additive learned family is the closest learned family to that exact construction, but the roles of the model and the tree should be separated carefully.",
            "",
        ]
    )
    lines.extend(
        _tex_itemize(
            [
                r"\textbf{Fixed before training:} the contiguous leaf partition, the binary tree above those leaves, the oracle target at the root, and the exact upward merge rule $S(u_L)\otimes S(u_R)$.",
                r"\textbf{Learned during training:} in the plotted additive slice, only the leaf count coordinate is fit. Because \texttt{feature\_mode=full}, the first/last endpoint channels are copied exactly from the realized leaf endpoints rather than relearned.",
                r"\textbf{Consequence:} this family is not inventing the tree algebra from scratch. It only has to fit the correct leaf summaries well enough that the fixed recursion returns the right root count.",
            ]
        )
    )
    lines.extend(
        [
            r"This is also why the additive lane uses a closed-form label fit for the leaf encoder rather than the weighted neural objective used by the neural family.",
            r"\subsection*{2.4 Which labels does training actually see?}",
            "",
            r"On the plotted training slice, the learner is given the following queried information:",
            "",
        ]
    )
    lines.extend(
        _tex_itemize(
            [
                rf"training documents: {_tt(markov_cfg.get('train_docs'))}",
                rf"held-out test documents: {_tt(markov_cfg.get('test_docs'))}",
                rf"leaf-label query rate: {_tt(markov_cfg.get('leaf_query_rate'))}",
                r"internal-node audit fraction: \texttt{1.0}",
                rf"optional root query during training: {_tt(markov_cfg.get('include_root_query'))}",
                rf"main objective weight: {_tt(markov_cfg.get('task_objective_weight'))} on root loss, with local-law weight {_tt(markov_cfg.get('local_law_weight'))}",
            ]
        )
    )
    lines.extend(
        [
            r"In plain language: the learner sees the words in each training leaf, together with whatever leaf labels, internal audits, and optional root labels the experiment pays for. In this additive slice the endpoint channels are already exact, so those queried labels are used mainly to fit the leaf count coordinate; the tree above the leaves is not re-learned.",
            r"The main downstream test is held-out root MAE. That is the number we should treat as the operational score.",
            rf"The table below is read from the clean publication slice and aggregates over {_tt(markov_meta.get('n_seeds'))} seeds.",
            r"\subsection*{2.5 What we see}",
            r"\begin{center}",
            r"\begin{tabular}{@{}lccc@{}}",
            r"\toprule",
            r"Markov additive fixed slice & $q_{\mathrm{infer}}=0$ & $q_{\mathrm{infer}}=0.5$ & $q_{\mathrm{infer}}=1.0$ \\",
            r"\midrule",
            rf"root MAE & {_fmt(_series_value(markov_series, 0.0, 'root_mae'))} & {_fmt(_series_value(markov_series, 0.5, 'root_mae'))} & {_fmt(_series_value(markov_series, 1.0, 'root_mae'))} \\",
            rf"share of total gap already closed by $q_{{\mathrm{{infer}}}}=0.5$ &  & {_fmt(gain_share.get('markov_additive'))} &  \\",
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{center}",
            r"The lecture reading is straightforward. Markov additive is already extremely close to the exact ceiling at $q_{\mathrm{infer}} = 0$. By $q_{\mathrm{infer}} = 0.5$ it has closed essentially all of the remaining gap, and by $q_{\mathrm{infer}} = 1$ it reaches the observed ceiling exactly. That is exactly what we would hope to see from a theorem-matched family with the correct merge rule.",
            r"So the clean pedagogical message is: this family is not just doing well; it is the sanity-check control. It is the empirical version of ``the DGP really is mergeable, and we told the learner the right algebra.''",
            r"\section*{3. The C-TreePO Example}",
            r"\subsection*{3.1 Data-generating process}",
            r"Now move one step away from the theorem-matched Markov control. In the segmented-LDA C-TreePO simulation, each book has a global topic mixture, but it is realized through contiguous segments with topic-specific concentrations. In the fixed publication slice:",
            "",
        ]
    )
    lines.extend(
        _tex_itemize(
            [
                rf"number of topics: {_tt(ctree_cfg.get('n_topics'))}",
                rf"vocabulary size: {_tt(ctree_cfg.get('vocab_size'))}",
                rf"training books: {_tt(ctree_cfg.get('n_books_train'))}",
                rf"held-out test books: {_tt(ctree_cfg.get('n_books_test'))}",
                rf"segments per book: {_tt(ctree_cfg.get('min_segments'))} to {_tt(ctree_cfg.get('max_segments'))}",
                rf"segment length: {_tt(ctree_cfg.get('min_seg_tokens'))} to {_tt(ctree_cfg.get('max_seg_tokens'))} tokens",
                rf"realized leaf size: {_tt(ctree_cfg.get('fixed_leaf_tokens'))} tokens",
            ]
        )
    )
    lines.extend(
        [
            r"Formally, topic-word distributions satisfy $\phi_k \sim \mathrm{Dirichlet}(\beta)$, each book draws a root mixture $w_b \sim \mathrm{Dirichlet}(\alpha)$, each segment chooses a dominant topic and then draws a concentrated local mixture around it. The target is the root topic mixture of the whole book.",
            r"\subsection*{3.2 Why this is harder}",
            r"Here we are no longer handing the learner an exact closed-form merge rule analogous to the changepoint count sketch. Instead, C-TreePO has to:",
            "",
        ]
    )
    lines.extend(
        _tex_enumerate(
            [
                r"estimate or inherit a topic-word matrix $\hat\phi$,",
                r"estimate leaf topic mixtures from leaf word counts,",
                r"calibrate those estimates from queried training leaves, and",
                r"optionally spend evaluation-time oracle queries on leaves and internal nodes.",
            ]
        )
    )
    lines.extend(
        [
            r"So this is the right next lecture step: the theorem-matched Markov example is the easy exact control, and C-TreePO is the approximate system that has to recover the same qualitative behavior from partial supervision.",
            r"\subsection*{3.3 What is being supervised}",
            "",
        ]
    )
    lines.extend(
        _tex_itemize(
            [
                rf"topic estimator in the clean publication slice: {_tt(ctree_cfg.get('topic_phi_estimator'))}",
                rf"leaf-theta estimator: {_tt(ctree_cfg.get('leaf_theta_estimator'))}",
                rf"learn-time calibration rate: {_tt(ctree_cfg.get('calibration_leaf_query_rate'))}",
                rf"decision-time leaf query rate in the full-guidance anchor: {_tt(ctree_cfg.get('eval_leaf_query_rate'))}",
                rf"decision-time internal query rate in the full-guidance anchor: {_tt(ctree_cfg.get('eval_internal_query_rate'))}",
                rf"internal-node query design: {_tt(ctree_cfg.get('eval_internal_query_design'))}",
            ]
        )
    )
    lines.extend(
        [
            r"The main downstream test here is held-out root $L^1$ error on the recovered root topic mixture.",
            rf"The table below is read from the clean publication slice and aggregates over {_tt(ctree_meta.get('n_seeds'))} seeds.",
            r"\subsection*{3.4 What we see}",
            r"\begin{center}",
            r"\begin{tabular}{@{}lccc@{}}",
            r"\toprule",
            r"C-TreePO fixed slice & $q_{\mathrm{infer}}=0$ & $q_{\mathrm{infer}}=0.5$ & $q_{\mathrm{infer}}=1.0$ \\",
            r"\midrule",
            rf"root $L^1$ & {_fmt(_series_value(ctree_series, 0.0, 'root_l1_mean'))} & {_fmt(_series_value(ctree_series, 0.5, 'root_l1_mean'))} & {_fmt(_series_value(ctree_series, 1.0, 'root_l1_mean'))} \\",
            rf"share of total gap already closed by $q_{{\mathrm{{infer}}}}=0.5$ &  & {_fmt(gain_share.get('ctree'))} &  \\",
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{center}",
            r"This is the approximation story we want. C-TreePO does not start at the exact ceiling. But with a modest learn-time calibration rate and moderate decision-time oracle visibility, it closes most of its own remaining gap quickly. In this fixed slice, by $q_{\mathrm{infer}}=0.5$ it has already closed about ninety percent of the total gap to its observed ceiling.",
            r"That is the appendix-friendly statement: once the model gets at least a few labels, the approximate tree system can move quickly toward the oracle behavior, even though it is solving a richer end-to-end problem than the exact Markov count sketch.",
            r"\section*{4. How To Compare The Two Stories Carefully}",
            r"The raw Markov and C-TreePO scores should not be compared numerically. Markov uses root MAE on changepoint count; C-TreePO uses root $L^1$ on topic-mixture recovery. Those are different units.",
            r"The right comparison is conceptual and step-by-step:",
            "",
        ]
    )
    lines.extend(
        _tex_enumerate(
            [
                r"Markov additive is the simplest theorem-facing control because the oracle and merge rule are both explicit.",
                r"It behaves exactly the way the lecture story says it should behave: near ceiling already, and perfect once decision-time visibility is full.",
                r"C-TreePO is the next approximation step: it does not know the exact merge rule, but after modest calibration it closes most of its own gap quickly.",
                r"Therefore the appendix story should read as a progression from exact mergeable control to approximate tree policy, not as a single raw-number horse race.",
            ]
        )
    )
    lines.extend(
        [
            r"\section*{5. Appendix-Ready Summary Paragraph}",
            r"A simple way to present the clean publication slice is to start from the Markov additive control. In that example the oracle target is just the number of changepoints, and the correct mergeable statistic is explicit: each span needs only its internal changepoint count together with its first and last regimes, so merging two children adds the two counts and a single boundary-correction term. The additive family is therefore the closest learned proxy to the true DGP plus the correct merge rule, and empirically it behaves that way: at learn-time full it is already near the exact ceiling before any decision-time intervention, and moderate decision-time visibility removes almost all remaining error. C-TreePO should then be introduced as the next approximation step rather than as a direct raw-number competitor. In the segmented-LDA benchmark it must estimate leaf topic mixtures, calibrate them from partial labels, and then use a limited evaluation-time oracle budget; nevertheless, with a modest amount of learn-time calibration it closes most of its own remaining gap quickly. The pedagogical lesson is therefore that the theorem-matched Markov example provides the clean control, while C-TreePO shows that a richer approximate tree system can recover the same qualitative oracle-guidance story with only modest supervision.",
            r"\section*{6. Reference Figure}",
            r"The paired raw/normalized tradeoff figure from the clean publication slice is reproduced here for reference.",
            "",
        ]
    )
    lines.extend(
        _appendix_tex_figure(
            fig_rel,
            "Clean publication Figure B: paired raw and normalized gap decomposition for the Markov additive control and the C-TreePO reference lane.",
            landscape=False,
        )
    )
    lines.append(r"\end{document}")
    lines.append("")
    return "\n".join(lines)


def _latex_full_page_image(path: str, caption: str) -> List[str]:
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


def _lighten_hex(color: str, factor: float = 0.78) -> str:
    color = color.lstrip("#")
    r = int(color[0:2], 16)
    g = int(color[2:4], 16)
    b = int(color[4:6], 16)
    rr = int(round(255 - (255 - r) * factor))
    gg = int(round(255 - (255 - g) * factor))
    bb = int(round(255 - (255 - b) * factor))
    return f"#{rr:02x}{gg:02x}{bb:02x}"


def _token_regime_sequence() -> List[str]:
    return ["A", "A", "A", "B", "B", "C", "C", "D", "D", "A", "A", "B"]


def _token_words() -> List[str]:
    return [
        "mist",
        "lake",
        "reed",
        "rust",
        "brick",
        "fern",
        "moss",
        "plum",
        "wine",
        "lake",
        "mist",
        "ember",
    ]


def _leaf_summaries(regimes: List[str], leaf_size: int = 4) -> List[Dict[str, object]]:
    out: List[Dict[str, object]] = []
    for i in range(0, len(regimes), leaf_size):
        span = regimes[i : i + leaf_size]
        count = sum(1 for a, b in zip(span[:-1], span[1:]) if a != b)
        out.append(
            {
                "label": f"L{1 + i // leaf_size}",
                "start": i + 1,
                "end": i + leaf_size,
                "count": count,
                "first": span[0],
                "last": span[-1],
            }
        )
    return out


def _merge_summary(left: Dict[str, object], right: Dict[str, object], label: str) -> Dict[str, object]:
    correction = 0 if str(left["last"]) == str(right["first"]) else 1
    return {
        "label": label,
        "count": int(left["count"]) + int(right["count"]) + correction,
        "first": str(left["first"]),
        "last": str(right["last"]),
        "correction": correction,
    }


def _draw_round_box(
    ax: plt.Axes,
    x: float,
    y: float,
    w: float,
    h: float,
    *,
    face: str,
    edge: str = "#555555",
    lw: float = 1.2,
    radius: float = 0.015,
) -> FancyBboxPatch:
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle=f"round,pad=0.01,rounding_size={radius}",
        linewidth=lw,
        edgecolor=edge,
        facecolor=face,
    )
    ax.add_patch(patch)
    return patch


def _draw_markov_dgp_figure(out_png: Path, out_pdf: Path) -> None:
    regimes = _token_regime_sequence()
    words = _token_words()
    leaves = _leaf_summaries(regimes, leaf_size=3)

    fig, ax = plt.subplots(figsize=(16.2, 8.9))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    ax.text(
        0.03,
        0.965,
        "Markov Changepoint DGP: What Generates One Document?",
        fontsize=23,
        fontweight="bold",
        ha="left",
        va="top",
    )
    ax.text(
        0.03,
        0.922,
        "Latent regimes stay constant for stretches, regime-specific vocabularies emit words, and the oracle target counts flips.",
        fontsize=12.4,
        color="#444444",
        ha="left",
        va="top",
    )

    _draw_round_box(ax, 0.03, 0.71, 0.37, 0.18, face="#f6f7fb", edge="#6b7280", lw=1.3)
    ax.text(0.045, 0.872, "Step 1. Sample the latent regime path", fontsize=13.2, fontweight="bold", ha="left", va="top")
    ax.text(0.045, 0.834, r"Latent path: $z_{1:T}$ is piecewise constant.", fontsize=11.2, ha="left", va="top")
    ax.text(0.045, 0.800, r"Changepoint rule: count 1 exactly when $z_t \neq z_{t+1}$.", fontsize=11.2, ha="left", va="top")
    ax.text(0.045, 0.758, r"Oracle target: $f^\star(x) = \sum_{t=1}^{T-1} \mathbf{1}\{z_t \neq z_{t+1}\}$", fontsize=12.1, ha="left", va="top")

    _draw_round_box(ax, 0.43, 0.71, 0.54, 0.18, face="#fff8eb", edge="#cf8b1b", lw=1.3)
    ax.text(0.445, 0.872, "Step 2. Each regime emits its own vocabulary", fontsize=13.2, fontweight="bold", ha="left", va="top")
    ax.text(0.445, 0.838, r"Token model: $x_t \mid z_t=r \sim \mathrm{Cat}(\phi_r)$.", fontsize=11.2, ha="left", va="top")
    ax.text(0.445, 0.800, "Each color below names a toy high-probability vocabulary for that regime.", fontsize=10.7, ha="left", va="top")

    legend_y = 0.715
    legend_w = 0.12
    legend_h = 0.055
    for idx, regime in enumerate(["A", "B", "C", "D"]):
        x = 0.445 + idx * 0.13
        col = REGIME_COLORS[regime]
        _draw_round_box(ax, x, legend_y, legend_w, legend_h, face=_lighten_hex(col, 0.75), edge=col, lw=1.5)
        ax.text(x + 0.012, legend_y + legend_h - 0.011, f"Regime {regime}", fontsize=10.4, fontweight="bold", ha="left", va="top")
        ax.text(
            x + 0.012,
            legend_y + 0.011,
            ", ".join(REGIME_VOCAB[regime]),
            fontsize=9.1,
            ha="left",
            va="bottom",
            color="#333333",
        )

    ax.text(0.03, 0.655, "Step 3. One toy document drawn from the DGP", fontsize=13.5, fontweight="bold", ha="left")
    ax.text(0.03, 0.626, "Top row = observed word token. Bottom strip = latent regime. Dashed boundaries mark the 5 true changepoints.", fontsize=11.1, color="#555555", ha="left")

    x0 = 0.12
    token_w = 0.068
    word_y = 0.50
    word_h = 0.078
    regime_h = 0.052
    regime_y = 0.441
    for i, (word, regime) in enumerate(zip(words, regimes)):
        x = x0 + i * token_w
        col = REGIME_COLORS[regime]
        _draw_round_box(ax, x, word_y, token_w - 0.004, word_h, face="#ffffff", edge=col, lw=1.4, radius=0.01)
        ax.text(x + 0.5 * (token_w - 0.004), word_y + 0.5 * word_h, word, fontsize=11.0, ha="center", va="center")
        ax.add_patch(Rectangle((x, regime_y), token_w - 0.004, regime_h, linewidth=0.8, edgecolor="#ffffff", facecolor=col))
        ax.text(x + 0.5 * (token_w - 0.004), regime_y + 0.5 * regime_h, regime, fontsize=12.0, color="white", ha="center", va="center", fontweight="bold")
        ax.text(x + 0.5 * (token_w - 0.004), word_y - 0.018, str(i + 1), fontsize=10.0, color="#555555", ha="center", va="top")

    ax.text(0.025, word_y + 0.5 * word_h, "observed\nword", fontsize=10.8, ha="left", va="center")
    ax.text(0.018, regime_y + 0.5 * regime_h, "latent\nregime", fontsize=10.8, ha="left", va="center")
    ax.text(0.43, 0.405, r"$z_{1:T} = (A,A,A,B,B,C,C,D,D,A,A,B)$", fontsize=12.6, ha="center", va="center")

    for i in range(len(regimes) - 1):
        if regimes[i] != regimes[i + 1]:
            x = x0 + (i + 1) * token_w - 0.002
            ax.plot([x, x], [regime_y - 0.03, word_y + word_h + 0.022], color="#444444", linestyle=(0, (3, 3)), linewidth=1.1)
            ax.scatter([x], [word_y + word_h + 0.033], s=30, color="#444444", zorder=5)

    _draw_round_box(ax, 0.73, 0.586, 0.22, 0.068, face="#eef8ec", edge="#5b8a3c", lw=1.2)
    ax.text(0.742, 0.632, "Oracle answer on this toy document", fontsize=10.7, fontweight="bold", ha="left", va="top")
    ax.text(0.742, 0.604, r"$f^\star(x)=5$ because there are 5 regime flips.", fontsize=11.0, ha="left", va="top")

    ax.text(0.03, 0.335, "Step 4. The simulation fixes a leaf partition before building the tree", fontsize=13.5, fontweight="bold", ha="left")
    ax.text(0.03, 0.308, "These gray brackets show the four leaves in the toy example. The appendix's exact-merge figure uses the same four spans.", fontsize=11.1, color="#555555", ha="left")

    bracket_y = 0.258
    for j, leaf in enumerate(leaves):
        start_x = x0 + j * 3 * token_w
        end_x = x0 + (j + 1) * 3 * token_w - 0.004
        mid_x = 0.5 * (start_x + end_x)
        ax.plot([start_x, end_x], [bracket_y, bracket_y], color="#666666", linewidth=1.3)
        ax.plot([start_x, start_x], [bracket_y, bracket_y + 0.012], color="#666666", linewidth=1.3)
        ax.plot([end_x, end_x], [bracket_y, bracket_y + 0.012], color="#666666", linewidth=1.3)
        ax.text(mid_x, bracket_y - 0.015, f"{leaf['label']} = tokens {leaf['start']}-{leaf['end']}", fontsize=10.9, ha="center", va="top")

    _draw_round_box(ax, 0.03, 0.07, 0.94, 0.115, face="#f7f7f7", edge="#9aa0a6", lw=1.1)
    ax.text(0.045, 0.166, "Takeaway", fontsize=12.4, fontweight="bold", ha="left", va="top")
    ax.text(
        0.045,
        0.135,
        "This Markov example is the lecture-friendly control: the DGP is explicit, the oracle target is changepoint count, and the next figure shows the exact mergeable sketch that recovers that target without loss.",
        fontsize=11.2,
        ha="left",
        va="top",
        color="#333333",
    )

    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=260, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)


def _draw_markov_exact_merge_figure(out_png: Path, out_pdf: Path) -> None:
    regimes = _token_regime_sequence()
    leaves = _leaf_summaries(regimes, leaf_size=3)
    left_parent = _merge_summary(leaves[0], leaves[1], "M12")
    right_parent = _merge_summary(leaves[2], leaves[3], "M34")
    root = _merge_summary(left_parent, right_parent, "Root")

    fig, ax = plt.subplots(figsize=(16.2, 9.4))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    ax.text(
        0.03,
        0.965,
        "Exact Markov Sketch: Why The Tree Can Merge Without Losing Information",
        fontsize=22,
        fontweight="bold",
        ha="left",
        va="top",
    )
    ax.text(
        0.03,
        0.922,
        "Slide version: each span stores changepoints inside the span plus the first and last latent regimes, and that is enough to merge exactly.",
        fontsize=13.0,
        color="#444444",
        ha="left",
        va="top",
    )

    _draw_round_box(ax, 0.03, 0.775, 0.38, 0.12, face="#eef6ff", edge="#4C78A8", lw=1.3)
    ax.text(0.045, 0.877, "Exact sketch stored at every node", fontsize=13.0, fontweight="bold", ha="left", va="top")
    ax.text(0.045, 0.842, r"$S(u) = (c(u), a(u), b(u))$", fontsize=13.8, ha="left", va="top")
    ax.text(0.045, 0.807, r"$c(u)$ = changepoints inside the span,  $a(u)$ = first regime,  $b(u)$ = last regime", fontsize=10.9, ha="left", va="top")

    _draw_round_box(ax, 0.45, 0.755, 0.52, 0.14, face="#fff7e8", edge="#d28a1f", lw=1.3)
    ax.text(0.465, 0.877, "Exact merge rule", fontsize=13.0, fontweight="bold", ha="left", va="top")
    ax.text(
        0.465,
        0.842,
        r"$S(u_L) \otimes S(u_R) = \left(c_L + c_R + \mathbf{1}\{b_L \neq a_R\},\, a_L,\, b_R\right)$",
        fontsize=12.0,
        ha="left",
        va="top",
    )
    ax.text(0.465, 0.804, "You only need one boundary correction: add 1 if the left span ends in a different regime than the right span begins.", fontsize=10.9, ha="left", va="top")

    ax.text(0.03, 0.695, "Same toy document, now reduced to four fixed leaves", fontsize=13.4, fontweight="bold", ha="left")
    ax.text(0.03, 0.666, "Top strip shows the latent regimes only. Each gray bracket is one leaf, and the box below it is that leaf's exact summary.", fontsize=11.0, color="#555555", ha="left")

    x0 = 0.12
    token_w = 0.068
    regime_h = 0.055
    regime_y = 0.586
    for i, regime in enumerate(regimes):
        x = x0 + i * token_w
        col = REGIME_COLORS[regime]
        ax.add_patch(Rectangle((x, regime_y), token_w - 0.004, regime_h, linewidth=0.8, edgecolor="#ffffff", facecolor=col))
        ax.text(x + 0.5 * (token_w - 0.004), regime_y + 0.5 * regime_h, regime, fontsize=12.0, color="white", ha="center", va="center", fontweight="bold")
        ax.text(x + 0.5 * (token_w - 0.004), regime_y - 0.018, str(i + 1), fontsize=10.0, color="#555555", ha="center", va="top")

    ax.text(0.028, regime_y + 0.5 * regime_h, "latent\nregime", fontsize=10.8, ha="left", va="center")

    bracket_y = 0.545
    for j, leaf in enumerate(leaves):
        start_x = x0 + j * 3 * token_w
        end_x = x0 + (j + 1) * 3 * token_w - 0.004
        mid_x = 0.5 * (start_x + end_x)
        ax.plot([start_x, end_x], [bracket_y, bracket_y], color="#666666", linewidth=1.3)
        ax.plot([start_x, start_x], [bracket_y, bracket_y + 0.012], color="#666666", linewidth=1.3)
        ax.plot([end_x, end_x], [bracket_y, bracket_y + 0.012], color="#666666", linewidth=1.3)
        ax.text(mid_x, bracket_y - 0.014, f"{leaf['label']} = tokens {leaf['start']}-{leaf['end']}", fontsize=10.8, ha="center", va="top")

    def draw_summary_box(xc: float, y: float, summary: Dict[str, object], *, face: str, title: str) -> None:
        box_w = 0.16
        box_h = 0.074
        _draw_round_box(ax, xc - 0.5 * box_w, y, box_w, box_h, face=face, edge="#666666", lw=1.2)
        ax.text(xc, y + box_h - 0.015, title, fontsize=10.3, fontweight="bold", ha="center", va="top")
        ax.text(
            xc,
            y + 0.028,
            f"S = ({summary['count']}, {summary['first']}, {summary['last']})",
            fontsize=10.4,
            ha="center",
            va="center",
            family="monospace",
        )

    leaf_xs = [0.15, 0.36, 0.58, 0.79]
    leaf_y = 0.38
    parent_y = 0.21
    root_y = 0.045
    box_h = 0.074

    for x, leaf in zip(leaf_xs, leaves):
        draw_summary_box(x, leaf_y, leaf, face="#f7f7f7", title=str(leaf["label"]))

    left_x = 0.255
    right_x = 0.685
    draw_summary_box(left_x, parent_y, left_parent, face="#eef6ff", title="M12")
    draw_summary_box(right_x, parent_y, right_parent, face="#eef6ff", title="M34")
    draw_summary_box(0.47, root_y, root, face="#eef8ec", title="Root")

    def connect(x0c: float, y0: float, x1c: float, y1: float) -> None:
        ax.plot([x0c, x1c], [y0, y1], color="#888888", linewidth=1.5)

    for x in leaf_xs[:2]:
        connect(x, leaf_y, left_x, parent_y + box_h)
    for x in leaf_xs[2:]:
        connect(x, leaf_y, right_x, parent_y + box_h)
    connect(left_x, parent_y, 0.47, root_y + box_h)
    connect(right_x, parent_y, 0.47, root_y + box_h)

    ax.text(
        left_x,
        parent_y + box_h + 0.02,
        "0 + 1 + 1 = 2  (join adds 1 because A != B)",
        fontsize=9.8,
        ha="center",
        va="bottom",
        color="#555555",
        bbox=dict(boxstyle="round,pad=0.16", facecolor="white", edgecolor="none", alpha=0.92),
    )
    ax.text(
        right_x,
        parent_y + box_h + 0.02,
        "1 + 1 + 1 = 3  (join adds 1 because D != A)",
        fontsize=9.8,
        ha="center",
        va="bottom",
        color="#555555",
        bbox=dict(boxstyle="round,pad=0.16", facecolor="white", edgecolor="none", alpha=0.92),
    )
    ax.text(
        0.47,
        root_y + box_h + 0.02,
        "2 + 3 + 0 = 5  (root join adds 0 because C = C)",
        fontsize=10.3,
        ha="center",
        va="bottom",
        color="#444444",
        fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.16", facecolor="white", edgecolor="none", alpha=0.94),
    )

    ax.text(
        0.03,
        0.012,
        "Interpretation: the root summary gives the exact oracle answer 5, so this tree loses no information about the changepoint-count target. "
        "That is why the Markov additive family is the theorem-matched control in the clean publication report.",
        fontsize=10.6,
        ha="left",
        va="bottom",
        color="#444444",
    )

    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=260, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    args = _parse_args()
    output_root = args.output_root.resolve()
    formal_root = output_root.parent
    repo_root = Path(__file__).resolve().parent.parent

    tex_path = (
        args.output_tex.resolve()
        if args.output_tex is not None
        else (
            args.output_markdown.resolve().with_suffix(".tex")
            if args.output_markdown is not None
            else (formal_root / "paper_reports" / "appendix_markov_additive_ctreepo_walkthrough.tex")
        )
    )
    pdf_path = args.output_pdf.resolve() if args.output_pdf is not None else tex_path.with_suffix(".pdf")
    dgp_png = tex_path.parent / "appendix_markov_changepoint_dgp_overview.png"
    dgp_pdf = dgp_png.with_suffix(".pdf")
    merge_png = tex_path.parent / "appendix_markov_changepoint_exact_merge_overview.png"
    merge_pdf = merge_png.with_suffix(".pdf")
    dgp_asset_tex = repo_root / "paper" / "figures" / "markov_changepoint_dgp_slide.tex"
    merge_asset_tex = repo_root / "paper" / "figures" / "markov_changepoint_exact_merge_slide.tex"

    diag_path = output_root / "figures" / "identifiable_zero_publication_report_latest_diagnostics.json"
    diag = _load_json(diag_path)
    diagnostics = diag.get("diagnostics") or {}
    neural = diagnostics.get("neural_lag_evidence") or {}
    fixed_slice = diagnostics.get("fixed_slice") or {}

    markov_series = neural.get("markov_additive") or {}
    ctree_series = (neural.get("ctree_reference") or {}).get("series") or {}
    obs = (neural.get("observations") or [{}])[0]
    gain_share = (obs.get("evidence") or {}).get("partial_gain_share_to_q05") or {}

    markov_meta = _markov_sample_config(output_root)
    ctree_meta = _ctree_sample_config(output_root)
    markov_cfg = markov_meta.get("config") or {}
    ctree_cfg = ctree_meta.get("config") or {}

    generated = datetime.now(timezone.utc).replace(microsecond=0).isoformat()

    _compile_standalone_figure_asset(dgp_asset_tex, dgp_pdf, dgp_png)
    _compile_standalone_figure_asset(merge_asset_tex, merge_pdf, merge_png)
    fig_path = output_root / "figures" / "pub_clean" / "main_figure_B_gap_decomposition.png"
    fig_rel = os.path.relpath(fig_path, tex_path.parent)
    dgp_rel = os.path.relpath(dgp_pdf, tex_path.parent)
    merge_rel = os.path.relpath(merge_pdf, tex_path.parent)

    tex_body = _build_appendix_tex(
        generated=generated,
        output_root=output_root,
        diag_path=diag_path,
        dgp_rel=dgp_rel,
        merge_rel=merge_rel,
        fig_rel=fig_rel,
        markov_cfg=markov_cfg,
        ctree_cfg=ctree_cfg,
        markov_meta=markov_meta,
        ctree_meta=ctree_meta,
        markov_series=markov_series,
        ctree_series=ctree_series,
        gain_share=gain_share,
    )

    tex_path.parent.mkdir(parents=True, exist_ok=True)
    tex_path.write_text(tex_body, encoding="utf-8")

    pdf_emitted = False
    if bool(args.emit_pdf):
        pdf_emitted = _run_pdflatex(tex_path)
        built_pdf = tex_path.with_suffix(".pdf")
        if pdf_emitted and built_pdf != pdf_path:
            shutil.copyfile(built_pdf, pdf_path)

    print(
        json.dumps(
            {
                "deprecated_output_markdown_arg_used": bool(args.output_markdown is not None),
                "output_tex": str(tex_path),
                "output_pdf": str(pdf_path) if pdf_emitted else None,
                "pdf_emitted": bool(pdf_emitted),
                "figure_pdf_paths": [str(dgp_pdf), str(merge_pdf)],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
