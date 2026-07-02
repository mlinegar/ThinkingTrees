from __future__ import annotations

import importlib
from pathlib import Path

from src.ctreepo.contracts import run_manifest_metadata


def test_v7_asset_manifest_resolves_figures_tables_and_contracts(tmp_path: Path) -> None:
    cli = importlib.import_module("scripts.audit_v7_paper_assets")
    paper_root = tmp_path / "paper" / "ctreepo"
    (paper_root / "sections").mkdir(parents=True)
    (paper_root / "assets" / "figures").mkdir(parents=True)
    (paper_root / "assets" / "tables").mkdir(parents=True)
    (paper_root / "preamble.tex").write_text(
        "\\graphicspath{\n    {assets/figures/}\n}\n",
        encoding="utf-8",
    )
    (paper_root / "main.tex").write_text(
        "\\input{sections/body}\n",
        encoding="utf-8",
    )
    (paper_root / "sections" / "body.tex").write_text(
        "\\includegraphics{contracted.pdf}\n"
        "\\includegraphics{legacy.pdf}\n"
        "\\input{assets/tables/contracted_table.tex}\n",
        encoding="utf-8",
    )
    (paper_root / "assets" / "figures" / "contracted.pdf").write_bytes(b"%PDF-1.4\n")
    (paper_root / "assets" / "figures" / "legacy.pdf").write_bytes(b"%PDF-1.4\n")
    (paper_root / "assets" / "tables" / "contracted_table.tex").write_text(
        "a & b \\\\\n",
        encoding="utf-8",
    )
    run_manifest = run_manifest_metadata(
        run_id="fixture.figure",
        domain="fixture",
        role="asset_fixture",
        backend="test",
        status="completed",
        f_init="f0",
        g_init="g0",
        audit_results={"ok": True},
        quarantine={"classification": "valid_treebundle_v1"},
        publication_ready=True,
    )
    (paper_root / "assets" / "figures" / "run_manifest.json").write_text(
        cli.json.dumps(run_manifest),
        encoding="utf-8",
    )

    manifest = cli.build_asset_manifest(
        paper_root=paper_root,
        tex_paths=[paper_root / "main.tex"],
    )

    assert manifest["summary"]["asset_count"] == 3
    by_ref = {entry["asset_reference"]: entry for entry in manifest["assets"]}
    assert by_ref["contracted.pdf"]["quarantine_status"]["classification"] == "valid_treebundle_v1"
    assert by_ref["legacy.pdf"]["quarantine_status"]["classification"] == "valid_treebundle_v1"
    assert by_ref["assets/tables/contracted_table.tex"]["quarantine_status"]["classification"] == "missing_contract"
    assert manifest["summary"]["missing_count"] == 0
    assert manifest["summary"]["missing_or_unknown_contract_count"] == 1


def test_v7_asset_manifest_reports_missing_asset(tmp_path: Path) -> None:
    cli = importlib.import_module("scripts.audit_v7_paper_assets")
    paper_root = tmp_path / "paper" / "ctreepo"
    paper_root.mkdir(parents=True)
    (paper_root / "preamble.tex").write_text(
        "\\graphicspath{\n    {assets/figures/}\n}\n",
        encoding="utf-8",
    )
    (paper_root / "main.tex").write_text(
        "\\includegraphics{missing_figure.pdf}\n",
        encoding="utf-8",
    )

    manifest = cli.build_asset_manifest(
        paper_root=paper_root,
        tex_paths=[paper_root / "main.tex"],
    )

    assert manifest["summary"]["asset_count"] == 1
    assert manifest["summary"]["missing_count"] == 1
    assert manifest["assets"][0]["quarantine_status"]["classification"] == "missing_contract"
