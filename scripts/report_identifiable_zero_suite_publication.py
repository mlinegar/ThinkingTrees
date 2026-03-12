#!/usr/bin/env python3
"""Generate a publication-oriented markdown/PDF report for identifiable-zero suite outputs.

This report is intentionally narrative: it emphasizes ceilings/attainability,
clean ablations, and filtered views that avoid calibration underdetermination.

It expects figures + JSON reports produced by the plotting scripts (typically in
<output-root>/figures/pub/).
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from pathlib import Path
from typing import Dict, List, Optional


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build publication-style report for identifiable-zero suite outputs.")
    p.add_argument("--output-root", type=Path, required=True, help="Suite output root.")
    p.add_argument(
        "--figures-subdir",
        type=str,
        default="pub",
        help="Subdirectory under <output-root>/figures containing publication figures.",
    )
    p.add_argument(
        "--output-markdown",
        type=Path,
        default=None,
        help="Markdown report path (default: <output-root>/figures/identifiable_zero_publication_report.md).",
    )
    p.add_argument(
        "--output-pdf",
        type=Path,
        default=None,
        help="PDF report path (default: same stem as markdown).",
    )
    p.add_argument("--emit-pdf", action=argparse.BooleanOptionalAction, default=True)
    return p.parse_args()


def _load_json(path: Path) -> Dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _fmt(x: object) -> str:
    try:
        v = float(x)  # type: ignore[arg-type]
    except Exception:
        return "nan"
    if not (v == v):
        return "nan"
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


def _md_img(rel_path: str, *, width: str = "100%") -> str:
    # Pandoc supports attribute syntax in markdown.
    return f"![]({rel_path}){{width={width}}}"


def _maybe_img(figures_dir: Path, rel_path: str) -> Optional[str]:
    if (figures_dir / rel_path).exists():
        return _md_img(rel_path)
    return None


def main() -> int:
    args = _parse_args()
    output_root = args.output_root.resolve()
    figures = output_root / "figures"
    if not figures.exists():
        raise SystemExit(f"figures directory not found: {figures}")

    pub = figures / str(args.figures_subdir)
    if not pub.exists():
        raise SystemExit(f"publication figures dir not found: {pub}")

    md_path = (
        args.output_markdown.resolve()
        if args.output_markdown is not None
        else (figures / "identifiable_zero_publication_report.md")
    )
    pdf_path = args.output_pdf.resolve() if args.output_pdf is not None else md_path.with_suffix(".pdf")

    # Load plot diagnostics (if present).
    seg_true = _load_json(pub / "segment_ops_ceilings_true_lam1_report.json")
    seg_emb = _load_json(pub / "segment_ops_ceilings_embedding_spectral_lam1_report.json")
    ctree = _load_json(pub / "ctreepo_ceilings_train1024_cal0p05_report.json")
    frontier = _load_json(pub / "ctreepo_frontier_train1024_cal0p05_report.json")
    calreg = _load_json(pub / "ctreepo_calibration_regression_report.json")

    seg_true_diag = ((seg_true.get("diagnostics") or {}).get("full_audit") or {}) if seg_true else {}
    seg_emb_diag = ((seg_emb.get("diagnostics") or {}).get("full_audit") or {}) if seg_emb else {}
    ctree_diag = ((ctree.get("diagnostics") or {}).get("full_guidance") or {}) if ctree else {}

    lines: List[str] = []
    lines.append("# Identifiable-Zero Sims (Publication-Oriented, Filtered)")
    lines.append("")
    lines.append(f"- Output root: `{output_root}`")
    lines.append(f"- Figures: `{figures}`")
    lines.append(f"- Publication figures subdir: `{pub}`")
    lines.append("")
    lines.append("## Narrative (What these sims are checking)")
    lines.append("")
    lines.append(
        "These simulations are designed to mirror the core C-TreePO / Semantic Forests logic in a controlled setting:"
    )
    lines.append("")
    lines.append(
        "- **Ceilings are explicit**: each family includes an oracle/exact construction with ~0 distortion (theoretical maximum)."
    )
    lines.append(
        "- **Gaps are decomposed**: when we are above the ceiling, we can attribute the gap to upstream estimation (topics/phi), calibration, or guidance budget."
    )
    lines.append(
        "- **Budgets are meaningful**: we vary audit/guidance/query rates and show when additional oracle access closes the gap (and when it cannot)."
    )
    lines.append("")
    lines.append("## Segment-LDA OPS (Operator Sketch Ceiling vs Upstream Estimation)")
    lines.append("")
    if seg_true_diag:
        lines.append(
            f"- Full-audit diagnostic (phi=true, lambda=1): ridge root MAE `{_fmt(seg_true_diag.get('ridge_root_mae'))}` vs exact `{_fmt(seg_true_diag.get('exact_root_mae'))}`."
        )
    if seg_emb_diag:
        lines.append(
            f"- Full-audit diagnostic (phi=embedding_spectral, lambda=1): ridge root MAE `{_fmt(seg_emb_diag.get('ridge_root_mae'))}` vs exact `{_fmt(seg_emb_diag.get('exact_root_mae'))}`."
        )
    lines.append("")
    img = _maybe_img(pub, "segment_ops_focus_true_lam1.png")
    if img:
        lines.append(img.replace("![](", "![](pub/"))
        lines.append("")
    img = _maybe_img(pub, "segment_ops_ceilings_true_lam1.png")
    if img:
        lines.append("\\newpage")
        lines.append("")
        lines.append(img.replace("![](", "![](pub/"))
        lines.append("")
    img = _maybe_img(pub, "segment_ops_focus_embedding_spectral_lam1.png")
    if img:
        lines.append("\\newpage")
        lines.append("")
        lines.append(img.replace("![](", "![](pub/"))
        lines.append("")
    img = _maybe_img(pub, "segment_ops_ceilings_embedding_spectral_lam1.png")
    if img:
        lines.append("\\newpage")
        lines.append("")
        lines.append(img.replace("![](", "![](pub/"))
        lines.append("")
    lines.append(
        "Interpretation: `exact` is the sketch ceiling; `ridge_true_topics` is the best-case downstream estimator given the same budgets; the remaining gap is upstream topic/phi estimation + inference."
    )
    lines.append("")
    lines.append("## Segmented-LDA C-TreePO (Ablations + Guidance Frontier)")
    lines.append("")
    if ctree_diag:
        lines.append(
            f"- Full-guidance diagnostic (train=1024, cal=0.05): budgeted root L1 `{_fmt(ctree_diag.get('estimated_calibrated_budgeted_root_l1'))}` vs oracle tree `{_fmt(ctree_diag.get('oracle_tree_root_l1'))}`."
        )
    if frontier:
        lines.append(f"- Frontier rows (after filters): `{frontier.get('n_rows_after_filters', 'nan')}`.")
    lines.append("")
    img = _maybe_img(pub, "ctreepo_ceilings_train1024_cal0p05.png")
    if img:
        lines.append(img.replace("![](", "![](pub/"))
        lines.append("")
    img = _maybe_img(pub, "ctreepo_frontier_train1024_cal0p05.png")
    if img:
        lines.append("\\newpage")
        lines.append("")
        lines.append(img.replace("![](", "![](pub/"))
        lines.append("")
    lines.append(
        "Interpretation: the ablation panel shows how much is gained by (i) calibrated estimates and (ii) **budgeted guidance** (querying a small fraction of nodes) relative to the oracle tree ceiling."
    )
    lines.append("")
    lines.append("## Calibration Failure Regime (Separated Out)")
    lines.append("")
    if calreg:
        lines.append(
            f"- Fraction with calibrated regression delta(root L1) > 0.25: `{_fmt(calreg.get('fraction_delta_gt_0p25'))}` (across filtered rows in this plot)."
        )
        lines.append(
            f"- Affine-map underdetermination threshold shown at k+1 = `{calreg.get('underdetermined_threshold_k_plus_1')}`."
        )
    lines.append("")
    img = _maybe_img(pub, "ctreepo_calibration_regression.png")
    if img:
        lines.append(img.replace("![](", "![](pub/"))
        lines.append("")
    lines.append(
        "Interpretation: very small calibration sample sizes can make the affine calibration ill-conditioned, producing large regressions. The main C-TreePO figures above filter to avoid this regime (so guidance effects are not confounded by calibration underdetermination)."
    )
    lines.append("")
    lines.append("## Markov (OPS Count) Narrative")
    lines.append("")
    lines.append("Markov narrative figures are included if present.")
    lines.append("")
    mk_neural0 = _load_json(pub / "markov_narrative_neural_rootq0_report.json")
    mk_neural1 = _load_json(pub / "markov_narrative_neural_rootq1_report.json")
    mk_add0 = _load_json(pub / "markov_narrative_additive_rootq0_report.json")
    mk_add1 = _load_json(pub / "markov_narrative_additive_rootq1_report.json")
    mk_neural1_diag = ((mk_neural1.get("diagnostics") or {}).get("full_audit") or {}) if mk_neural1 else {}
    mk_add1_diag = ((mk_add1.get("diagnostics") or {}).get("full_audit") or {}) if mk_add1 else {}
    if mk_neural1_diag or mk_add1_diag:
        lines.append(
            f"- Full-audit diagnostic (neural): learned root MAE `{_fmt(mk_neural1_diag.get('learned_root_mae'))}` vs exact `{_fmt(mk_neural1_diag.get('exact_root_mae'))}`."
        )
        lines.append(
            f"- Full-audit diagnostic (additive): learned root MAE `{_fmt(mk_add1_diag.get('learned_root_mae'))}` vs exact `{_fmt(mk_add1_diag.get('exact_root_mae'))}`."
        )
        lines.append("")
        lines.append(
            "Interpretation: additive reaches the ceiling once there is enough *local* supervision (either leaf labels or internal-node labels), while the unstructured neural merger remains far from the ceiling even at full audit (and exhibits large schedule dependence)."
        )
        lines.append("")
    for rel in [
        "markov_narrative_neural_rootq0.png",
        "markov_narrative_neural_rootq1.png",
        "markov_narrative_additive_rootq0.png",
        "markov_narrative_additive_rootq1.png",
    ]:
        img = _maybe_img(pub, rel)
        if img:
            lines.append("\\newpage")
            lines.append("")
            lines.append(img.replace("![](", "![](pub/"))
            lines.append("")

    lines.append("## Takeaways (What this says about the framework)")
    lines.append("")
    lines.append(
        "- These suites validate the **ceiling story**: when the oracle/guidance budget is full, oracle/exact constructions reach ~0 error (so failures elsewhere are not plotting artifacts)."
    )
    lines.append(
        "- They isolate **where TreePO-style guarantees help**: budgeted guidance closes the gap when the remaining error is due to local violations that are discoverable by auditing."
    )
    lines.append(
        "- They also show **what guidance cannot fix**: upstream representation/estimation error (e.g. topic/phi estimation) and underdetermined calibration can dominate unless separately budgeted or stabilized."
    )
    lines.append("")

    md_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    pdf_emitted = False
    if bool(args.emit_pdf):
        try:
            pdf_emitted = _run_pandoc(md_path, pdf_path)
        except Exception:
            pdf_emitted = False

    print(
        json.dumps(
            {
                "output_markdown": str(md_path),
                "output_pdf": str(pdf_path) if pdf_emitted else None,
                "pdf_emitted": bool(pdf_emitted),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
