#!/usr/bin/env python3
"""Build a narrative Beamer deck for the Markov additive / C-TreePO appendix walkthrough."""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from _appendix_markov_toy import (
    REGIME_COLORS,
    REGIME_VOCAB,
    leaf_summaries,
    merge_summary,
    token_regime_sequence,
    token_words,
)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build the appendix narrative Beamer deck.")
    p.add_argument("--output-root", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, default=None)
    p.add_argument("--emit-pdf", action=argparse.BooleanOptionalAction, default=True)
    return p.parse_args()


def _load_json(path: Path) -> Dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _as_float(x: object) -> Optional[float]:
    try:
        v = float(x)  # type: ignore[arg-type]
    except Exception:
        return None
    if not math.isfinite(v):
        return None
    return float(v)


def _fmt_key(x: object) -> str:
    v = _as_float(x)
    if v is None:
        return str(x)
    if abs(v - round(v)) <= 1e-12:
        return str(int(round(v)))
    return f"{v:.6g}"


def _fmt_num(x: object) -> str:
    v = _as_float(x)
    if v is None:
        return "n/a"
    if abs(v) <= 1e-12:
        return "0"
    av = abs(v)
    if abs(v - round(v)) <= 1e-9 and av < 1_000_000:
        return f"{int(round(v)):,}"
    if av >= 1000.0 or av < 1e-3:
        return f"{v:.2e}"
    if av < 0.01:
        return f"{v:.4f}".rstrip("0").rstrip(".")
    if av < 0.1:
        return f"{v:.3f}".rstrip("0").rstrip(".")
    if av < 1.0:
        return f"{v:.3f}".rstrip("0").rstrip(".")
    return f"{v:.3g}"


def _pct_str(x: object) -> str:
    v = _as_float(x)
    if v is None:
        return "n/a"
    return f"{100.0 * v:.1f}\\%"


def _series_value(series: Dict[str, Dict[str, object]], q: float, key: str) -> float:
    row = series.get(_fmt_key(q)) or {}
    v = _as_float(row.get(key))
    return float(v) if v is not None else float("nan")


def _gap_share(q0: float, q05: float, q1: float, fallback: object = None) -> float:
    fb = _as_float(fallback)
    denom = q0 - q1
    if math.isfinite(q0) and math.isfinite(q05) and math.isfinite(q1) and abs(denom) > 1e-12:
        return max(min((q0 - q05) / denom, 1.0), -10.0)
    return float(fb) if fb is not None else float("nan")


def _relative_tex_path(from_dir: Path, target: Path) -> str:
    return os.path.relpath(target.resolve(), from_dir.resolve())


def _run_latex(tex_path: Path) -> bool:
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

    for _ in range(2):
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


def _overlay(spec: Optional[str]) -> str:
    return f"<{spec}>" if spec else ""


def _reg_name(letter: str) -> str:
    return f"reg{letter}"


def _document_positions(n_tokens: int) -> List[float]:
    x0 = 1.45
    dx = 1.11
    return [x0 + i * dx for i in range(n_tokens)]


def _markov_document_tikz(
    *,
    words_overlay: Optional[str],
    regimes_overlay: Optional[str],
    boundary_overlays: Sequence[Optional[str]],
    oracle_overlay: Optional[str],
    leaf_overlay: Optional[str],
) -> str:
    regimes = token_regime_sequence()
    words = token_words()
    leaves = leaf_summaries(regimes, leaf_size=3)
    positions = _document_positions(len(words))
    boundaries = [i for i in range(len(regimes) - 1) if regimes[i] != regimes[i + 1]]

    lines: List[str] = [
        r"\begin{tikzpicture}[x=0.74cm,y=0.74cm,font=\sffamily]",
        r"\begin{scope}[on background layer]",
        r"\node[panel, draw=black!18, fill=bgPage, fit={(0.05,-2.25) (15.00,2.25)}] {};",
        r"\end{scope}",
        r"\node[anchor=east, align=right, font=\fontsize{10.3}{12.0}\selectfont] at (0.92, 1.30) {observed\\word};",
    ]
    if regimes_overlay:
        lines.append(
            rf"\node{_overlay(regimes_overlay)}[anchor=east, align=right, font=\fontsize{{10.3}}{{12.0}}\selectfont] at (0.92, 0.46) {{latent\\state}};"
        )

    for i, (word, regime, x) in enumerate(zip(words, regimes, positions), start=1):
        reg = _reg_name(regime)
        lines.append(rf"\node[docslot, draw={reg}!85!black, anchor=west] at ({x:.2f}, 1.02) {{}};")
        if words_overlay:
            lines.append(
                rf"\node{_overlay(words_overlay)}[font=\fontsize{{9.0}}{{10.4}}\selectfont] at ({x + 0.515:.3f}, 1.30) {{{word}}};"
            )
        if regimes_overlay:
            lines.append(rf"\path{_overlay(regimes_overlay)}[fill={reg}] ({x:.2f}, 0.20) rectangle ++(1.03, 0.52);")
            lines.append(
                rf"\node{_overlay(regimes_overlay)}[font=\bfseries\fontsize{{10.6}}{{12.0}}\selectfont, text=white] at ({x + 0.515:.3f}, 0.46) {{{regime}}};"
            )
        lines.append(
            rf"\node[font=\fontsize{{8.2}}{{9.4}}\selectfont, text=black!55] at ({x + 0.515:.3f}, -0.02) {{{i}}};"
        )

    if regimes_overlay:
        seq = ",".join(regimes)
        lines.append(
            rf"\node{_overlay(regimes_overlay)}[anchor=west, font=\fontsize{{10.4}}{{12.4}}\selectfont] at (4.05, -0.54) {{$z_{{1:T}}=({seq})$}};"
        )

    for idx, boundary_idx in enumerate(boundaries):
        spec = boundary_overlays[idx] if idx < len(boundary_overlays) else None
        if not spec:
            continue
        x = positions[boundary_idx + 1]
        lines.append(
            rf"\draw{_overlay(spec)}[densely dashed, black!60, line width=0.8pt] ({x:.2f}, 0.06) -- ({x:.2f}, 1.88);"
        )
        lines.append(rf"\path{_overlay(spec)}[fill=black!60] ({x:.2f}, 1.98) circle (2.2pt);")

    if oracle_overlay:
        lines.append(
            rf"\node{_overlay(oracle_overlay)}[callout, draw=cGold!65!black, fill=cGoldFill!82, anchor=south west, text width=4.05cm, font=\fontsize{{8.4}}{{9.8}}\selectfont] at (9.55, 1.88) {{\textbf{{Oracle target}}\\[0.35mm]count hidden state flips\\[0.35mm]$f^\star(x)=\sum_{{t=1}}^{{T-1}}\mathbf{{1}}\{{z_t\neq z_{{t+1}}\}}$\\[0.35mm]Here: $f^\star(x)=5$.}};"
        )

    if leaf_overlay:
        bracket_y = -1.08
        for leaf in leaves:
            left = positions[int(leaf["start"]) - 1]
            right = positions[int(leaf["end"]) - 1] + 1.03
            mid = 0.5 * (left + right)
            label = str(leaf["label"])
            lines.append(
                rf"\draw{_overlay(leaf_overlay)}[black!55, line width=1.0pt] ({left:.2f},{bracket_y:.2f}) -- ({left:.2f},{bracket_y - 0.25:.2f});"
            )
            lines.append(
                rf"\draw{_overlay(leaf_overlay)}[black!55, line width=1.0pt] ({left:.2f},{bracket_y - 0.25:.2f}) -- ({right:.2f},{bracket_y - 0.25:.2f});"
            )
            lines.append(
                rf"\draw{_overlay(leaf_overlay)}[black!55, line width=1.0pt] ({right:.2f},{bracket_y:.2f}) -- ({right:.2f},{bracket_y - 0.25:.2f});"
            )
            lines.append(
                rf"\node{_overlay(leaf_overlay)}[font=\bfseries\fontsize{{9.2}}{{10.2}}\selectfont, text=black!75] at ({mid:.2f}, {bracket_y - 0.58:.2f}) {{{label}}};"
            )
        lines.append(
            rf"\node{_overlay(leaf_overlay)}[anchor=west, text=black!65, font=\fontsize{{8.9}}{{10.8}}\selectfont] at (0.25,-1.92) {{The tree topology is fixed first; learning only chooses what each leaf emits.}};"
        )

    lines.append(r"\end{tikzpicture}")
    return "\n".join(lines)


def _state_node(
    *,
    name: str,
    x: float,
    y: float,
    count: object,
    first: object,
    last: object,
    face: str,
    draw: str,
    overlay: Optional[str],
) -> str:
    return (
        rf"\node{_overlay(overlay)}[statebox, fill={face}, draw={draw}] ({name}) at ({x:.2f},{y:.2f}) "
        rf"{{\textbf{{{name}}}\\[-0.05em]\scriptsize $c={count}$\\[-0.12em]\tiny $a={first},\ b={last}$}};"
    )


def _markov_merge_tikz(
    *,
    leaf_overlays: Sequence[Optional[str]],
    parent_overlays: Sequence[Optional[str]],
    root_overlay: Optional[str],
) -> str:
    regimes = token_regime_sequence()
    leaves = leaf_summaries(regimes, leaf_size=3)
    left_parent = merge_summary(leaves[0], leaves[1], "P12")
    right_parent = merge_summary(leaves[2], leaves[3], "P34")
    root = merge_summary(left_parent, right_parent, "Root")
    positions = _document_positions(len(regimes))

    lines: List[str] = [
        r"\begin{tikzpicture}[x=0.66cm,y=0.66cm,font=\sffamily,>=Latex]",
        r"\begin{scope}[on background layer]",
        r"\node[panel, draw=black!18, fill=bgPage, fit={(0.05,-2.05) (14.90,5.15)}] {};",
        r"\end{scope}",
    ]

    leaf_xy = [(1.80, 1.05), (4.85, 1.05), (9.65, 1.05), (12.70, 1.05)]
    for idx, (leaf, (x, y)) in enumerate(zip(leaves, leaf_xy)):
        lines.append(
            _state_node(
                name=str(leaf["label"]),
                x=x,
                y=y,
                count=leaf["count"],
                first=leaf["first"],
                last=leaf["last"],
                face="white",
                draw="black!45",
                overlay=leaf_overlays[idx] if idx < len(leaf_overlays) else None,
            )
        )

    if len(parent_overlays) > 0 and parent_overlays[0]:
        lines.append(
            _state_node(
                name="P12",
                x=3.32,
                y=3.00,
                count=left_parent["count"],
                first=left_parent["first"],
                last=left_parent["last"],
                face="cBlueFill!70",
                draw="cBlue!70!black",
                overlay=parent_overlays[0],
            )
        )
    if len(parent_overlays) > 1 and parent_overlays[1]:
        lines.append(
            _state_node(
                name="P34",
                x=11.18,
                y=3.00,
                count=right_parent["count"],
                first=right_parent["first"],
                last=right_parent["last"],
                face="cBlueFill!70",
                draw="cBlue!70!black",
                overlay=parent_overlays[1],
            )
        )
    if root_overlay:
        lines.append(
            _state_node(
                name="Root",
                x=7.25,
                y=4.70,
                count=root["count"],
                first=root["first"],
                last=root["last"],
                face="cGreenFill!75",
                draw="cGreen!65!black",
                overlay=root_overlay,
            )
        )

    if len(parent_overlays) > 0 and parent_overlays[0]:
        lines.extend(
            [
                rf"\draw{_overlay(parent_overlays[0])}[->, line width=0.9pt, draw=black!55] (L1.north) -- (P12.south west);",
                rf"\draw{_overlay(parent_overlays[0])}[->, line width=0.9pt, draw=black!55] (L2.north) -- (P12.south east);",
                rf"\node{_overlay(parent_overlays[0])}[mergeeq] at (3.32, 3.88) {{$0 + 1 + 1 = 2$}};",
            ]
        )
    if len(parent_overlays) > 1 and parent_overlays[1]:
        lines.extend(
            [
                rf"\draw{_overlay(parent_overlays[1])}[->, line width=0.9pt, draw=black!55] (L3.north) -- (P34.south west);",
                rf"\draw{_overlay(parent_overlays[1])}[->, line width=0.9pt, draw=black!55] (L4.north) -- (P34.south east);",
                rf"\node{_overlay(parent_overlays[1])}[mergeeq] at (11.18, 3.88) {{$1 + 1 + 1 = 3$}};",
            ]
        )
    if root_overlay:
        lines.extend(
            [
                rf"\draw{_overlay(root_overlay)}[->, line width=0.9pt, draw=black!55] (P12.north) -- (Root.south west);",
                rf"\draw{_overlay(root_overlay)}[->, line width=0.9pt, draw=black!55] (P34.north) -- (Root.south east);",
                rf"\node{_overlay(root_overlay)}[mergeeq, font=\bfseries\fontsize{{10.0}}{{11.6}}\selectfont] at (7.25, 5.45) {{$2 + 3 + 0 = 5$}};",
                rf"\node{_overlay(root_overlay)}[callout, draw=cGreen!65!black, fill=cGreenFill!85, anchor=west, text width=3.30cm] at (11.10, 4.35) {{\textbf{{Exact recovery}}\\[0.4mm]The root count equals the oracle answer. Once the leaf state is right, the upper-tree recursion is forced.}};",
            ]
        )

    lines.append(r"\node[anchor=east, align=right, font=\fontsize{9.6}{11.4}\selectfont] at (0.95, -0.62) {latent\\state};")
    for i, (regime, x) in enumerate(zip(regimes, positions), start=1):
        reg = _reg_name(regime)
        lines.append(rf"\path[fill={reg}] ({x:.2f}, -0.98) rectangle ++(1.03, 0.52);")
        lines.append(
            rf"\node[font=\bfseries\fontsize{{10.2}}{{11.8}}\selectfont, text=white] at ({x + 0.515:.3f}, -0.72) {{{regime}}};"
        )
        lines.append(
            rf"\node[font=\fontsize{{8.0}}{{9.2}}\selectfont, text=black!55] at ({x + 0.515:.3f}, -1.18) {{{i}}};"
        )

    for leaf in leaves:
        left = positions[int(leaf["start"]) - 1]
        right = positions[int(leaf["end"]) - 1] + 1.03
        mid = 0.5 * (left + right)
        lines.append(rf"\draw[black!55, line width=0.9pt] ({left:.2f},-1.38) -- ({left:.2f},-1.62);")
        lines.append(rf"\draw[black!55, line width=0.9pt] ({left:.2f},-1.62) -- ({right:.2f},-1.62);")
        lines.append(rf"\draw[black!55, line width=0.9pt] ({right:.2f},-1.38) -- ({right:.2f},-1.62);")
        lines.append(
            rf"\node[font=\bfseries\fontsize{{9.0}}{{10.2}}\selectfont, text=black!75] at ({mid:.2f}, -1.95) {{{leaf['label']}}};"
        )

    lines.append(r"\end{tikzpicture}")
    return "\n".join(lines)


def _ctree_pipeline_tikz(
    *,
    stage_overlays: Sequence[Optional[str]],
    show_merge_callout: bool,
    show_query_arrows: bool,
) -> str:
    stage_specs = list(stage_overlays) + [None] * max(0, 5 - len(stage_overlays))
    lines: List[str] = [
        r"\begin{tikzpicture}[x=0.82cm,y=0.82cm,font=\sffamily,>=Latex]",
        r"\begin{scope}[on background layer]",
        r"\node[panel, draw=black!18, fill=bgPage, fit={(0.0,-0.92) (14.20,2.18)}] {};",
        r"\end{scope}",
        rf"\node{_overlay(stage_specs[0])}[stagebox, fill=cBlueFill!85, draw=cBlue!70!black, text width=1.65cm] (wc) at (1.10,0.65) {{\textbf{{Word counts}}\\[-0.05em]\tiny per-leaf bag of words}};",
        rf"\node{_overlay(stage_specs[1])}[stagebox, fill=cGoldFill!90, draw=cGold!75!black, text width=1.95cm] (spec) at (3.95,0.65) {{\textbf{{Spectral $\hat\phi$}}\\[-0.05em]\tiny estimate topic--word map once}};",
        rf"\node{_overlay(stage_specs[2])}[stagebox, fill=cTealFill!90, draw=cTeal!75!black, text width=1.95cm] (leaf) at (6.95,0.65) {{\textbf{{Leaf mixtures}}\\[-0.05em]\tiny infer topic weights from counts}};",
        rf"\node{_overlay(stage_specs[3])}[stagebox, fill=cOrangeFill!95, draw=cOrange!75!black, text width=1.95cm] (merge) at (10.05,0.65) {{\textbf{{Learned merge}}\\[-0.05em]\tiny tree-structured aggregation}};",
        rf"\node{_overlay(stage_specs[4])}[stagebox, fill=cGreenFill!90, draw=cGreen!70!black, text width=1.55cm] (root) at (13.05,0.65) {{\textbf{{Root $\hat w$}}\\[-0.05em]\tiny topic-mixture estimate}};",
    ]
    arrows = [
        ("wc", "spec", stage_specs[1]),
        ("spec", "leaf", stage_specs[2]),
        ("leaf", "merge", stage_specs[3]),
        ("merge", "root", stage_specs[4]),
    ]
    for src, dst, spec in arrows:
        if spec:
            lines.append(rf"\draw{_overlay(spec)}[->, line width=1.0pt, draw=black!65] ({src}.east) -- ({dst}.west);")

    if show_merge_callout:
        lines.append(
            r"\node[badgebox, fill=cOrangeFill!90, draw=cOrange!75!black, anchor=north] at (10.05,-0.34) {no explicit merge rule};"
        )
    if show_query_arrows:
        lines.extend(
            [
                r"\draw[densely dashed, ->, line width=0.8pt, draw=cPurple!80!black] (6.95,1.62) -- (6.95,1.02);",
                r"\draw[densely dashed, ->, line width=0.8pt, draw=cPurple!80!black] (10.05,1.62) -- (10.05,1.02);",
                r"\node[badgebox, fill=cPurpleFill!88, draw=cPurple!80!black] at (8.50,1.92) {$q_{\mathrm{infer}}$ may query leaf and internal nodes};",
            ]
        )

    lines.append(r"\end{tikzpicture}")
    return "\n".join(lines)


def _result_cards_tikz(
    *,
    metric_label: str,
    q0: float,
    q05: float,
    q1: float,
    gap_share: float,
    q_train: object,
    train_docs: object,
    left_title: str,
    takeaway: str,
    context_note: str,
) -> str:
    lines = [
        r"\begin{tikzpicture}[x=1cm,y=1cm,font=\sffamily]",
        r"\begin{scope}[on background layer]",
        r"\node[panel, draw=black!18, fill=bgPage, fit={(0.00,0.00) (12.55,5.90)}] {};",
        r"\end{scope}",
        r"\node[anchor=west, font=\bfseries\fontsize{12.0}{13.8}\selectfont] at (0.30,5.42) {" + left_title + r"};",
        r"\node[resultcard, fill=cBlueFill!72, draw=cBlue!70!black] at (2.10,4.00) {" + "\n".join(
            [
                r"$q_{\mathrm{infer}}=0$\\[0.15em]",
                metric_label + r"\\[0.10em]",
                r"\Large\bfseries " + _fmt_num(q0),
            ]
        ) + r"};",
        r"\node[resultcard, fill=cGoldFill!85, draw=cGold!75!black] at (6.25,4.00) {" + "\n".join(
            [
                r"$q_{\mathrm{infer}}=0.5$\\[0.15em]",
                metric_label + r"\\[0.10em]",
                r"\Large\bfseries " + _fmt_num(q05),
            ]
        ) + r"};",
        r"\node[resultcard, fill=cGreenFill!85, draw=cGreen!70!black] at (10.40,4.00) {" + "\n".join(
            [
                r"$q_{\mathrm{infer}}=1$\\[0.15em]",
                metric_label + r"\\[0.10em]",
                r"\Large\bfseries " + _fmt_num(q1),
            ]
        ) + r"};",
        r"\node[badgebox, fill=cGreenFill!85, draw=cGreen!70!black] at (6.25,2.72) {"
        + _pct_str(gap_share)
        + r" of the total gap is closed by $q_{\mathrm{infer}}=0.5$};",
        r"\node[callout, draw=cBlue!70!black, fill=white, anchor=north west, text width=5.45cm] at (0.35,2.18) {\textbf{Takeaway}\\[0.45mm]\footnotesize "
        + takeaway
        + r"};",
        r"\node[callout, draw=black!22, fill=white, anchor=north west, text width=5.45cm] at (6.65,2.18) {\textbf{Context}\\[0.45mm]\footnotesize "
        + rf"Learn-time visibility: $q_{{\mathrm{{train}}}}={_fmt_num(q_train)}$. "
        + rf"Training documents: {_fmt_num(train_docs)}. "
        + context_note
        + r"};",
        r"\end{tikzpicture}",
    ]
    return "\n".join(lines)


def _title_frame() -> str:
    body = "\n".join(
        [
            r"\begin{frame}[plain]",
            r"\begin{tikzpicture}[remember picture,overlay]",
            r"\fill[bgPage] (current page.south west) rectangle (current page.north east);",
            r"\node[anchor=north west, align=left, text width=12.25cm] at ([xshift=0.65cm,yshift=-0.55cm]current page.north west) {"
            r"{\fontsize{22}{26}\selectfont\bfseries\color{fg}From Exact Mergeability to Approximate Tree Policies}\\[0.35em]"
            r"{\fontsize{12.5}{15}\selectfont\color{fgSub}Narrative Beamer deck for the Markov additive / C-TreePO appendix walkthrough}"
            r"};",
            r"\node[callout, draw=cBlue!70!black, fill=white, anchor=north west, text width=5.95cm] at ([xshift=0.75cm,yshift=-2.20cm]current page.north west) {"
            r"\textbf{What this deck does}\\[0.5mm]"
            r"It turns the appendix into a lecture path: first the exact control, then the learned tree policy, then the normalized comparison that ties them together."
            r"};",
            r"\node[callout, draw=cOrange!75!black, fill=cOrangeFill!88, anchor=north west, text width=5.95cm] at ([xshift=7.05cm,yshift=-2.20cm]current page.north west) {"
            r"\textbf{Reading rule}\\[0.5mm]"
            r"First ask what exact state would make merging lossless. Then ask what is given, what is learned, and where learn-time vs decision-time oracle visibility enters."
            r"};",
            r"\node[callout, draw=black!20, fill=white, anchor=south west, text width=12.25cm] at ([xshift=0.75cm,yshift=0.75cm]current page.south west) {"
            r"\textbf{Fixed framing for the Markov toy}\\[0.5mm]"
            r"Colors are our visualization of latent state IDs in the synthetic DGP. The oracle target is defined on hidden state transitions / regime flips; the learner only sees emitted words and whatever labels we choose to query."
            r"};",
            r"\end{tikzpicture}",
            r"\end{frame}",
            "",
        ]
    )
    return body


def _shared_operator_frame() -> str:
    body = "\n".join(
        [
            r"\begin{columns}[T,onlytextwidth]",
            r"\begin{column}{0.38\textwidth}",
            r"\begin{block}{Invariant skeleton}",
            r"\small",
            r"$\phi:\mathrm{Span}\to S,\quad g:S\times S\to S,\quad \rho:S\to\mathcal{O}$\\[0.45em]",
            r"Read every slide with the same checklist: what summary state would make merging lossless, how siblings combine, and what is read out at the root.",
            r"\end{block}",
            r"\end{column}",
            r"\begin{column}{0.60\textwidth}",
            r"\begin{block}{Markov additive: exact control}",
            r"\small",
            r"$\phi$ returns the exact leaf sketch $(\mathrm{count},\mathrm{first},\mathrm{last})$.\\[0.35em]",
            r"$g$ is the exact boundary-corrected merge law.\\[0.35em]",
            r"$\rho$ reads off the root count, which equals the oracle changepoint total.",
            r"\end{block}",
            r"\begin{alertblock}{C-TreePO: learned tree policy}",
            r"\small",
            r"Estimate $\hat\phi$ once, infer leaf mixtures from counts, and learn the upward merge from supervision.\\[0.35em]",
            r"The tree skeleton stays fixed, but there is no explicit merge rule to hand the system.",
            r"\end{alertblock}",
            r"\end{column}",
            r"\end{columns}",
            r"{\scriptsize\color{fgSub}\textbf{Comparison rule:} raw error within a family; normalized progress toward each family's own ceiling across families.}",
        ]
    )
    return _beamer_frame("Shared Operator Type", body)


def _markov_colors_frame() -> str:
    chips: List[str] = [
        r"\begin{tikzpicture}[x=1cm,y=1cm,font=\sffamily]",
        r"\begin{scope}[on background layer]",
        r"\node[panel, draw=black!18, fill=bgPage, fit={(0.00,0.00) (12.45,4.75)}] {};",
        r"\end{scope}",
        r"\node[callout, draw=cBlue!70!black, fill=white, anchor=north west, text width=5.65cm] at (0.35,4.45) {"
        r"\textbf{Start with the hidden world}\\[0.5mm]"
        r"The Markov toy starts from a latent path $z_{1:T}$. The colors below are our visualization of latent state IDs; they are not observed labels fed to the learner."
        r"};",
    ]
    chip_xy = [(6.55, 3.55), (9.40, 3.55), (6.55, 2.05), (9.40, 2.05)]
    for (state, vocab), (x, y) in zip(REGIME_VOCAB.items(), chip_xy):
        reg = _reg_name(state)
        chips.append(
            rf"\node[chipbox, fill={reg}!16, draw={reg}!85!black, anchor=west] at ({x:.2f},{y:.2f}) {{\textbf{{State {state}}}\\[-0.05em]\tiny {', '.join(vocab)}}};"
        )
    chips.extend(
        [
            r"\node[callout, draw=cGold!75!black, fill=cGoldFill!88, anchor=west, text width=5.50cm] at (0.35,1.35) {"
            r"\textbf{Oracle framing}\\[0.5mm]"
            r"The oracle target is defined on hidden state transitions / regime flips. It does not need semantic labels like ``blue means topic A''; it only needs the hidden path well enough to count changes."
            r"};",
            r"\node[callout, draw=black!20, fill=white, anchor=west, text width=5.40cm] at (6.55,0.72) {"
            r"\textbf{What the learner sees}\\[0.5mm]"
            r"Only the emitted words and whatever labels we decide to query at train time or decision time."
            r"};",
            r"\end{tikzpicture}",
        ]
    )
    body = "\n".join(
        [
            r"{\small ``Here are the colors'' is purely explanatory. They visualize the hidden state IDs in the synthetic DGP so we can talk through what the oracle counts before we ever talk about learning.}",
            r"\vspace{0.08cm}",
            "\n".join(chips),
        ]
    )
    return _beamer_frame("Markov DGP I: Here Are the Colors", body)


def _markov_observed_tokens_frame() -> str:
    body = "\n".join(
        [
            r"{\small First reveal the token positions and the emitted words. Overlay 1 gives the slots; overlay 2 fills them with observed words. The hidden-state row is still absent here.}",
            r"\vspace{0.05cm}",
            r"\begin{center}",
            _markov_document_tikz(
                words_overlay="2-",
                regimes_overlay=None,
                boundary_overlays=[],
                oracle_overlay=None,
                leaf_overlay=None,
            ),
            r"\end{center}",
        ]
    )
    return _beamer_frame("Markov DGP II: Observed Tokens", body)


def _markov_latent_frame() -> str:
    body = "\n".join(
        [
            r"{\small Now add the hidden row. The colors are ground truth for explanation only: they visualize latent state IDs so we can see what the DGP did, not what the learner directly observes.}",
            r"\vspace{0.05cm}",
            r"\begin{center}",
            _markov_document_tikz(
                words_overlay="1-",
                regimes_overlay="1-",
                boundary_overlays=[],
                oracle_overlay=None,
                leaf_overlay=None,
            ),
            r"\end{center}",
        ]
    )
    return _beamer_frame("Markov DGP III: Reveal the Latent States", body)


def _markov_boundaries_frame() -> str:
    body = "\n".join(
        [
            r"{\small Each dashed line is one latent-state flip. This frame builds the five changepoints one pause at a time before we reveal the oracle answer.}",
            r"\vspace{0.05cm}",
            r"\begin{center}",
            _markov_document_tikz(
                words_overlay="1-",
                regimes_overlay="1-",
                boundary_overlays=["1-", "2-", "3-", "4-", "5-"],
                oracle_overlay=None,
                leaf_overlay=None,
            ),
            r"\end{center}",
        ]
    )
    return _beamer_frame("Markov DGP IV: Changepoints Appear One by One", body)


def _markov_oracle_frame() -> str:
    body = "\n".join(
        [
            r"{\small Once the hidden path is fixed, the oracle target is just the changepoint count. The learner still sees only the words; the oracle score is defined on the hidden transitions.}",
            r"\vspace{0.05cm}",
            r"\begin{center}",
            _markov_document_tikz(
                words_overlay="1-",
                regimes_overlay="1-",
                boundary_overlays=["1-", "1-", "1-", "1-", "1-"],
                oracle_overlay="1-",
                leaf_overlay=None,
            ),
            r"\end{center}",
        ]
    )
    return _beamer_frame("Markov DGP V: The Oracle Target", body)


def _markov_leaves_frame() -> str:
    body = "\n".join(
        [
            r"{\small Before any fitting happens, the simulator fixes the four leaves. The additive learner will have to emit the right summary from each leaf span, but it does not get to redesign the tree.}",
            r"\vspace{0.05cm}",
            r"\begin{center}",
            _markov_document_tikz(
                words_overlay="1-",
                regimes_overlay="1-",
                boundary_overlays=["1-", "1-", "1-", "1-", "1-"],
                oracle_overlay="1-",
                leaf_overlay="1-",
            ),
            r"\end{center}",
        ]
    )
    return _beamer_frame("Markov DGP VI: Fix the Leaves Before Learning", body)


def _markov_leaf_state_frame() -> str:
    body = "\n".join(
        [
            r"{\small Each leaf emits the exact sufficient state $S(u)=(c(u),a(u),b(u))$: changepoints inside the span, first hidden state, last hidden state. This frame fills the four leaves one at a time.}",
            r"\vspace{0.05cm}",
            r"\begin{center}",
            _markov_merge_tikz(
                leaf_overlays=["1-", "2-", "3-", "4-"],
                parent_overlays=[],
                root_overlay=None,
            ),
            r"\end{center}",
        ]
    )
    return _beamer_frame("Exact Markov Sketch I: Leaf State", body)


def _markov_pairwise_merge_frame() -> str:
    body = "\n".join(
        [
            r"{\small The exact merge rule is",
            r"\[",
            r"S(u_L)\otimes S(u_R)=\bigl(c_L+c_R+\mathbf{1}\{b_L\neq a_R\},\ a_L,\ b_R\bigr).",
            r"\]",
            r"The only extra term is the boundary correction when the left span ends in a different hidden state than the right span begins.}",
            r"\vspace{-0.08cm}",
            r"\begin{center}",
            _markov_merge_tikz(
                leaf_overlays=["1-", "1-", "1-", "1-"],
                parent_overlays=["1-", "2-"],
                root_overlay=None,
            ),
            r"\end{center}",
        ]
    )
    return _beamer_frame("Exact Markov Sketch II: Pairwise Merges", body)


def _markov_root_merge_frame() -> str:
    body = "\n".join(
        [
            r"{\small At the root, the same exact recursion lands on the oracle answer. That is why Markov additive is the theorem-matched control: once the leaf state is right, the rest of the tree is forced.}",
            r"\vspace{0.02cm}",
            r"\begin{center}",
            _markov_merge_tikz(
                leaf_overlays=["1-", "1-", "1-", "1-"],
                parent_overlays=["1-", "1-"],
                root_overlay="1-",
            ),
            r"\end{center}",
        ]
    )
    return _beamer_frame("Exact Markov Sketch III: Root Recovery", body)


def _markov_results_frame(markov_q0: float, markov_q05: float, markov_q1: float, markov_gap: float, q_train: object, train_docs: object) -> str:
    body = _result_cards_tikz(
        metric_label=r"Root MAE",
        q0=markov_q0,
        q05=markov_q05,
        q1=markov_q1,
        gap_share=markov_gap,
        q_train=q_train,
        train_docs=train_docs,
        left_title=r"Result: theorem-matched control near the ceiling",
        takeaway=r"The additive control is already at its ceiling up to tiny residual error. By $q_{\mathrm{infer}}=0.5$, almost all remaining error is gone, exactly as the sketch theory predicts.",
        context_note=r"This is the clean control with the correct merge law built in.",
    )
    return _beamer_frame("Markov Result: Near-Ceiling Recovery", body)


def _ctree_pipeline_frame_a() -> str:
    body = "\n".join(
        [
            r"{\small Now keep the tree skeleton and remove the exact algebra. This frame reveals the C-TreePO pipeline left to right, one stage at a time.}",
            r"\vspace{0.06cm}",
            r"\begin{center}",
            _ctree_pipeline_tikz(
                stage_overlays=["1-", "2-", "3-", "4-", "5-"],
                show_merge_callout=False,
                show_query_arrows=False,
            ),
            r"\end{center}",
            r"\begin{block}{Read left to right}",
            r"\small",
            r"Counts are the observable input at the leaves. C-TreePO estimates $\hat\phi$ once, infers leaf mixtures, learns how information should merge up the tree, and reads out a root estimate $\hat w$.",
            r"\end{block}",
        ]
    )
    return _beamer_frame("C-TreePO I: The Inference Pipeline", body)


def _ctree_pipeline_frame_b() -> str:
    body = "\n".join(
        [
            r"{\small The sharp contrast with the Markov control is structural: there is no explicit merge rule. C-TreePO must estimate $\hat\phi$, infer leaf mixtures, and learn the merge from supervision.}",
            r"\vspace{0.06cm}",
            r"\begin{center}",
            _ctree_pipeline_tikz(
                stage_overlays=["1-", "1-", "1-", "1-", "1-"],
                show_merge_callout=True,
                show_query_arrows=False,
            ),
            r"\end{center}",
            r"\begin{columns}[T,onlytextwidth]",
            r"\begin{column}{0.48\textwidth}",
            r"\begin{block}{Estimated once}",
            r"\small",
            r"$\hat\phi$ is learned globally, then reused when each leaf mixture is inferred from its observed counts.",
            r"\end{block}",
            r"\end{column}",
            r"\begin{column}{0.48\textwidth}",
            r"\begin{alertblock}{Still learned}",
            r"\small",
            r"The merge is not given by theorem. Leaf summaries and internal aggregation both have to be learned from partial supervision, with no explicit merge rule.",
            r"\end{alertblock}",
            r"\end{column}",
            r"\end{columns}",
        ]
    )
    return _beamer_frame("C-TreePO II: Where the Unknowns Live", body)


def _ctree_pipeline_frame_c() -> str:
    body = "\n".join(
        [
            r"{\small Decision-time visibility $q_{\mathrm{infer}}$ can query leaf summaries and internal nodes during evaluation. The tree can ask for guidance, but it still does not inherit an exact algebraic merge law.}",
            r"\vspace{0.06cm}",
            r"\begin{center}",
            _ctree_pipeline_tikz(
                stage_overlays=["1-", "1-", "1-", "1-", "1-"],
                show_merge_callout=True,
                show_query_arrows=True,
            ),
            r"\end{center}",
            r"\begin{block}{$q_{\mathrm{infer}}$ intervention points}",
            r"\small",
            r"Decision-time oracle queries can touch leaf summaries and internal nodes during evaluation. They reveal selected local state, but they do not replace the learned merge with the exact Markov algebra.",
            r"\end{block}",
        ]
    )
    return _beamer_frame("C-TreePO III: Decision-Time Queries", body)


def _ctree_results_frame(ctree_q0: float, ctree_q05: float, ctree_q1: float, ctree_gap: float, q_train: object, train_docs: object) -> str:
    body = _result_cards_tikz(
        metric_label=r"Root $L^1$",
        q0=ctree_q0,
        q05=ctree_q05,
        q1=ctree_q1,
        gap_share=ctree_gap,
        q_train=q_train,
        train_docs=train_docs,
        left_title=r"Result: approximate tree policy recovers the same qualitative story",
        takeaway=r"C-TreePO does not start at the ceiling, but modest decision-time visibility closes most of the remaining gap. The same qualitative guidance story survives even when the merge must be learned.",
        context_note=r"Harder setting: estimate $\hat\phi$, infer leaves, learn the merge, then optionally query during evaluation.",
    )
    return _beamer_frame("C-TreePO Result: Most of the Gap Closes Quickly", body)


def _progression_frame(markov_gap: float, ctree_gap: float) -> str:
    body = "\n".join(
        [
            r"{\small This comparison is structural, not a raw-number horse race. Read it as a move from an exact theorem-matched control to an approximate learned tree policy.}",
            r"\vspace{0.10cm}",
            r"\begin{columns}[T,onlytextwidth]",
            r"\begin{column}{0.48\textwidth}",
            r"\begin{block}{Markov additive}",
            r"\small",
            r"Merge is exact and given \emph{a priori}. Leaf state is theorem-matched. Gap closed by $q_{\mathrm{infer}}=0.5$: "
            + _pct_str(markov_gap)
            + r".",
            r"\end{block}",
            r"\end{column}",
            r"\begin{column}{0.48\textwidth}",
            r"\begin{alertblock}{C-TreePO}",
            r"\small",
            r"Same tree type, but estimate $\hat\phi$, infer leaf mixtures, and learn the merge. Gap closed by $q_{\mathrm{infer}}=0.5$: "
            + _pct_str(ctree_gap)
            + r".",
            r"\end{alertblock}",
            r"\end{column}",
            r"\end{columns}",
            r"\begin{center}",
            r"\begin{tikzpicture}[x=1cm,y=1cm,font=\sffamily,>=Latex]",
            r"\node[badgebox, fill=cGreenFill!88, draw=cGreen!70!black] (mid) at (0,0) {exact control $\rightarrow$ approximate policy};",
            r"\end{tikzpicture}",
            r"\end{center}",
            r"\begin{block}{Interpretation}",
            r"\small",
            r"The Markov example validates the formalism when the exact state and merge law are known. C-TreePO then asks how much of the same oracle-guidance story survives once the algebra disappears and has to be approximated from supervision.",
            r"\end{block}",
        ]
    )
    return _beamer_frame("Progression: Exact Control to Approximate Policy", body)


def _gap_frame(gap_rel: str) -> str:
    body = "\n".join(
        [
            r"{\scriptsize\color{fgSub}\textbf{Reading rule:} the normalized panel is the safe cross-family comparison because each family is measured against its own ceiling.}",
            r"\vspace{0.02cm}",
            r"\begin{center}",
            rf"\includegraphics[width=0.95\linewidth,height=0.78\textheight,keepaspectratio,trim=28 34 28 54,clip]{{{gap_rel}}}",
            r"\end{center}",
        ]
    )
    return _beamer_frame("Gap Decomposition and the Normalized Comparison", body)


def _beamer_frame(title: str, body: str) -> str:
    return "\n".join([rf"\begin{{frame}}[t]{{{title}}}", body, r"\end{frame}", ""])


def _build_deck_tex(
    *,
    generated: str,
    gap_rel: str,
    markov_q0: float,
    markov_q05: float,
    markov_q1: float,
    ctree_q0: float,
    ctree_q05: float,
    ctree_q1: float,
    markov_gap: float,
    ctree_gap: float,
    markov_q_train: object,
    markov_train_docs: object,
    ctree_q_train: object,
    ctree_train_docs: object,
) -> Tuple[str, List[str]]:
    frame_titles = [
        "Appendix-to-Slides Reading Rule",
        "Shared Operator Type",
        "Markov DGP I: Here Are the Colors",
        "Markov DGP II: Observed Tokens",
        "Markov DGP III: Reveal the Latent States",
        "Markov DGP IV: Changepoints Appear One by One",
        "Markov DGP V: The Oracle Target",
        "Markov DGP VI: Fix the Leaves Before Learning",
        "Exact Markov Sketch I: Leaf State",
        "Exact Markov Sketch II: Pairwise Merges",
        "Exact Markov Sketch III: Root Recovery",
        "Markov Result: Near-Ceiling Recovery",
        "C-TreePO I: The Inference Pipeline",
        "C-TreePO II: Where the Unknowns Live",
        "C-TreePO III: Decision-Time Queries",
        "C-TreePO Result: Most of the Gap Closes Quickly",
        "Progression: Exact Control to Approximate Policy",
        "Gap Decomposition and the Normalized Comparison",
    ]

    parts: List[str] = [
        r"\documentclass[aspectratio=169,11pt]{beamer}",
        r"\usepackage[T1]{fontenc}",
        r"\usepackage{lmodern}",
        r"\usepackage{amsmath,amssymb,booktabs,array}",
        r"\usepackage{graphicx}",
        r"\usepackage{tikz}",
        r"\usetikzlibrary{arrows.meta,positioning,calc,fit,backgrounds,decorations.pathreplacing}",
        rf"\definecolor{{fg}}{{HTML}}{{1A1A2E}}",
        rf"\definecolor{{fgSub}}{{HTML}}{{6C6C8A}}",
        rf"\definecolor{{bgPage}}{{HTML}}{{F4F4F9}}",
        rf"\definecolor{{cBlue}}{{HTML}}{{4A7FB5}}",
        rf"\definecolor{{cBlueFill}}{{HTML}}{{D6E4F0}}",
        rf"\definecolor{{cGold}}{{HTML}}{{B8963E}}",
        rf"\definecolor{{cGoldFill}}{{HTML}}{{F5E6C8}}",
        rf"\definecolor{{cTeal}}{{HTML}}{{3A8A8A}}",
        rf"\definecolor{{cTealFill}}{{HTML}}{{D0EFEF}}",
        rf"\definecolor{{cPurple}}{{HTML}}{{7B5EA7}}",
        rf"\definecolor{{cPurpleFill}}{{HTML}}{{E3D5F0}}",
        rf"\definecolor{{cGreen}}{{HTML}}{{4A8A5A}}",
        rf"\definecolor{{cGreenFill}}{{HTML}}{{D4EDDA}}",
        rf"\definecolor{{cOrange}}{{HTML}}{{C07830}}",
        rf"\definecolor{{cOrangeFill}}{{HTML}}{{FDE8D0}}",
        rf"\definecolor{{regA}}{{HTML}}{{{REGIME_COLORS['A'].lstrip('#')}}}",
        rf"\definecolor{{regB}}{{HTML}}{{{REGIME_COLORS['B'].lstrip('#')}}}",
        rf"\definecolor{{regC}}{{HTML}}{{{REGIME_COLORS['C'].lstrip('#')}}}",
        rf"\definecolor{{regD}}{{HTML}}{{{REGIME_COLORS['D'].lstrip('#')}}}",
        r"\setbeamertemplate{navigation symbols}{}",
        r"\setbeamercolor{normal text}{fg=fg,bg=white}",
        r"\setbeamercolor{frametitle}{fg=fg,bg=white}",
        r"\setbeamercolor{block title}{fg=white,bg=cBlue}",
        r"\setbeamercolor{block body}{fg=fg,bg=cBlueFill!45}",
        r"\setbeamercolor{block title alerted}{fg=white,bg=cOrange}",
        r"\setbeamercolor{block body alerted}{fg=fg,bg=cOrangeFill!65}",
        r"\setbeamerfont{frametitle}{series=\bfseries,size=\Large}",
        r"\setbeamersize{text margin left=0.55cm,text margin right=0.55cm}",
        r"\setbeamertemplate{footline}{%",
        r"  \leavevmode\hbox{%",
        r"    \begin{beamercolorbox}[wd=.78\paperwidth,ht=2.8ex,dp=1.1ex,leftskip=.55cm]{author in head/foot}%",
        r"      {\color{fgSub}\scriptsize\insertsectionhead}%",
        r"    \end{beamercolorbox}%",
        r"    \begin{beamercolorbox}[wd=.22\paperwidth,ht=2.8ex,dp=1.1ex,rightskip=.55cm plus1fil]{date in head/foot}%",
        r"      {\color{fgSub}\scriptsize\insertframenumber/\inserttotalframenumber}%",
        r"    \end{beamercolorbox}}%",
        r"}",
        r"\renewcommand{\familydefault}{\sfdefault}",
        r"\tikzset{",
        r"  panel/.style={rounded corners=4pt, line width=0.9pt},",
        r"  docslot/.style={rectangle, draw, rounded corners=2pt, minimum width=1.03cm, minimum height=0.56cm, inner sep=0pt, fill=white},",
        r"  chipbox/.style={rounded corners=4pt, minimum width=2.35cm, minimum height=0.95cm, inner sep=5pt, align=left},",
        r"  callout/.style={rounded corners=4pt, line width=0.8pt, inner sep=6pt, align=left},",
        r"  statebox/.style={rounded corners=4pt, line width=0.8pt, minimum width=2.15cm, minimum height=1.08cm, inner sep=4pt, align=center},",
        r"  stagebox/.style={rounded corners=4pt, line width=0.8pt, minimum height=0.92cm, inner xsep=6pt, inner ysep=4pt, align=center},",
        r"  resultcard/.style={rounded corners=5pt, line width=0.9pt, minimum width=3.25cm, minimum height=2.05cm, inner sep=6pt, align=center},",
        r"  badgebox/.style={rounded corners=4pt, line width=0.8pt, inner xsep=6pt, inner ysep=4pt, font=\sffamily\scriptsize\bfseries, align=center},",
        r"  mergeeq/.style={rounded corners=3pt, fill=white, inner xsep=4pt, inner ysep=2pt, font=\fontsize{9.2}{10.8}\selectfont, text=black!65},",
        r"  note/.style={font=\fontsize{8.2}{9.8}\selectfont\itshape, text=fgSub},",
        r"}",
        r"\title{From Exact Mergeability to Approximate Tree Policies}",
        r"\subtitle{Narrative Beamer version of the Markov additive / C-TreePO appendix walkthrough}",
        r"\author{}",
        rf"\date{{Generated {generated}}}",
        r"\begin{document}",
        "",
        _title_frame(),
        r"\section{Shared Operator Type}",
        _shared_operator_frame(),
        r"\section{Markov Control}",
        _markov_colors_frame(),
        _markov_observed_tokens_frame(),
        _markov_latent_frame(),
        _markov_boundaries_frame(),
        _markov_oracle_frame(),
        _markov_leaves_frame(),
        _markov_leaf_state_frame(),
        _markov_pairwise_merge_frame(),
        _markov_root_merge_frame(),
        _markov_results_frame(
            markov_q0=markov_q0,
            markov_q05=markov_q05,
            markov_q1=markov_q1,
            markov_gap=markov_gap,
            q_train=markov_q_train,
            train_docs=markov_train_docs,
        ),
        r"\section{C-TreePO}",
        _ctree_pipeline_frame_a(),
        _ctree_pipeline_frame_b(),
        _ctree_pipeline_frame_c(),
        _ctree_results_frame(
            ctree_q0=ctree_q0,
            ctree_q05=ctree_q05,
            ctree_q1=ctree_q1,
            ctree_gap=ctree_gap,
            q_train=ctree_q_train,
            train_docs=ctree_train_docs,
        ),
        r"\section{Comparison}",
        _progression_frame(markov_gap=markov_gap, ctree_gap=ctree_gap),
        _gap_frame(gap_rel=gap_rel),
        r"\end{document}",
        "",
    ]
    return "\n".join(parts), frame_titles


def main() -> int:
    args = _parse_args()
    output_root = args.output_root.resolve()
    formal_root = output_root.parent
    out_dir = args.out_dir.resolve() if args.out_dir is not None else (formal_root / "paper_reports" / "appendix_narrative_slide_deck")
    out_dir.mkdir(parents=True, exist_ok=True)

    diag_path = output_root / "figures" / "identifiable_zero_publication_report_latest_diagnostics.json"
    if not diag_path.exists():
        raise FileNotFoundError(f"Missing diagnostics JSON: {diag_path}")

    gap_pdf = output_root / "figures" / "pub_clean" / "main_figure_B_gap_decomposition.pdf"
    gap_png = output_root / "figures" / "pub_clean" / "main_figure_B_gap_decomposition.png"
    if gap_pdf.exists():
        gap_path = gap_pdf
    elif gap_png.exists():
        gap_path = gap_png
    else:
        raise FileNotFoundError(
            f"Missing gap decomposition figure; checked {gap_pdf} and {gap_png}"
        )

    diag = _load_json(diag_path)
    diagnostics = diag.get("diagnostics") or {}
    neural = diagnostics.get("neural_lag_evidence") or {}
    fixed_slice = neural.get("fixed_slice") or {}
    evidence = ((neural.get("observations") or [{}])[0].get("evidence") or {})

    markov_series = neural.get("markov_additive") or {}
    ctree_series = (neural.get("ctree_reference") or {}).get("series") or {}

    markov_q0 = _series_value(markov_series, 0.0, "root_mae")
    markov_q05 = _series_value(markov_series, 0.5, "root_mae")
    markov_q1 = _series_value(markov_series, 1.0, "root_mae")
    ctree_q0 = _series_value(ctree_series, 0.0, "root_l1_mean")
    ctree_q05 = _series_value(ctree_series, 0.5, "root_l1_mean")
    ctree_q1 = _series_value(ctree_series, 1.0, "root_l1_mean")

    gain_share = evidence.get("partial_gain_share_to_q05") or {}
    markov_gap = _gap_share(markov_q0, markov_q05, markov_q1, gain_share.get("markov_additive"))
    ctree_gap = _gap_share(ctree_q0, ctree_q05, ctree_q1, gain_share.get("ctree"))

    markov_fixed = fixed_slice.get("markov") or {}
    ctree_fixed = fixed_slice.get("ctree") or {}

    markov_q_train = _as_float(markov_fixed.get("learn_time_oracle_visibility"))
    markov_train_docs = _as_float(markov_fixed.get("train_docs"))
    ctree_q_train = _as_float(ctree_fixed.get("learn_time_oracle_visibility"))
    ctree_train_docs = _as_float(ctree_fixed.get("train_docs"))

    tex_path = out_dir / "appendix_narrative_slide_deck.tex"
    pdf_path = out_dir / "appendix_narrative_slide_deck.pdf"
    gap_rel = _relative_tex_path(out_dir, gap_path)
    generated = datetime.now(timezone.utc).replace(microsecond=0).isoformat()

    tex_body, frame_titles = _build_deck_tex(
        generated=generated,
        gap_rel=gap_rel,
        markov_q0=markov_q0,
        markov_q05=markov_q05,
        markov_q1=markov_q1,
        ctree_q0=ctree_q0,
        ctree_q05=ctree_q05,
        ctree_q1=ctree_q1,
        markov_gap=markov_gap,
        ctree_gap=ctree_gap,
        markov_q_train=markov_q_train,
        markov_train_docs=markov_train_docs,
        ctree_q_train=ctree_q_train,
        ctree_train_docs=ctree_train_docs,
    )

    tex_path.write_text(tex_body, encoding="utf-8")

    pdf_emitted = False
    if bool(args.emit_pdf):
        pdf_emitted = _run_latex(tex_path)

    summary = {
        "generated": generated,
        "output_root": str(output_root),
        "diag_path": str(diag_path),
        "deck_tex": str(tex_path),
        "deck_pdf": str(pdf_path) if pdf_emitted else None,
        "pdf_emitted": bool(pdf_emitted),
        "slide_count": len(frame_titles),
        "frame_titles": frame_titles,
        "reused_asset_paths": [str(gap_path)],
        "markov_metrics": {
            "q_train": markov_q_train,
            "train_docs": markov_train_docs,
            "q0_root_mae": markov_q0,
            "q05_root_mae": markov_q05,
            "q1_root_mae": markov_q1,
            "gap_share_to_q05": markov_gap,
        },
        "ctree_metrics": {
            "q_train": ctree_q_train,
            "train_docs": ctree_train_docs,
            "q0_root_l1": ctree_q0,
            "q05_root_l1": ctree_q05,
            "q1_root_l1": ctree_q1,
            "gap_share_to_q05": ctree_gap,
        },
    }
    (out_dir / "appendix_narrative_slide_deck_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
