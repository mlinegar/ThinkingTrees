from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.ctreepo.ladder import (
    LadderStageContext,
    LadderStageOutput,
    continue_ladder,
    run_component_ladder,
)


def _artifact_stage(context: LadderStageContext) -> LadderStageOutput:
    artifact = context.stage_dir / f"{context.component}_{context.index}.pt"
    artifact.write_text("artifact\n", encoding="utf-8")
    return LadderStageOutput(
        component_artifact=artifact,
        shared_artifacts={"interface": artifact},
        result={"component": context.component},
        metrics={"stage": float(context.index)},
    )


def test_schedule_f_updates_only_f_and_records_inputs(tmp_path: Path) -> None:
    result = run_component_ladder(
        schedule="f",
        output_dir=tmp_path,
        train_stage=_artifact_stage,
        initial_component_artifacts={"f": "f0", "g": "g0"},
        initial_shared_artifacts={"interface": "i0"},
        allowed_components=frozenset({"f", "g"}),
    )

    assert result.component_artifacts["f"] == tmp_path / "stage_0_f" / "f_0.pt"
    assert result.component_artifacts["g"] == "g0"
    assert result.shared_artifacts["interface"] == tmp_path / "stage_0_f" / "f_0.pt"

    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    assert manifest["status"] == "completed"
    assert manifest["schedule"] == ["f"]
    assert manifest["component_artifacts"]["f"].endswith("stage_0_f/f_0.pt")
    assert manifest["component_artifacts"]["g"] == "g0"
    assert manifest["stages"][0]["input_component_artifacts"] == {"f": "f0", "g": "g0"}
    assert manifest["stages"][0]["input_shared_artifacts"] == {"interface": "i0"}


def test_continue_ladder_g_receives_latest_f_from_previous_manifest(tmp_path: Path) -> None:
    first = run_component_ladder(
        schedule="f",
        output_dir=tmp_path / "run_f",
        train_stage=_artifact_stage,
        initial_component_artifacts={"f": "f0", "g": "g0"},
        initial_shared_artifacts={"interface": "i0"},
        allowed_components=frozenset({"f", "g"}),
    )
    calls: list[tuple[str, dict[str, object], dict[str, object]]] = []

    def train_g(context: LadderStageContext) -> LadderStageOutput:
        calls.append(
            (
                context.component,
                dict(context.component_artifacts),
                dict(context.shared_artifacts),
            )
        )
        return _artifact_stage(context)

    second = continue_ladder(
        previous_manifest=first.manifest_path,
        schedule="g",
        output_dir=tmp_path / "run_g",
        train_stage=train_g,
        allowed_components=frozenset({"f", "g"}),
    )

    latest_f = str(tmp_path / "run_f" / "stage_0_f" / "f_0.pt")
    assert calls == [("g", {"f": latest_f, "g": "g0"}, {"interface": latest_f})]
    assert second.component_artifacts["f"] == latest_f
    assert second.component_artifacts["g"] == tmp_path / "run_g" / "stage_0_g" / "g_0.pt"

    manifest = json.loads(second.manifest_path.read_text(encoding="utf-8"))
    assert manifest["previous_manifest"] == str(first.manifest_path)
    assert manifest["stages"][0]["input_component_artifacts"]["f"] == latest_f
    assert manifest["stages"][0]["input_component_artifacts"]["g"] == "g0"


def test_schedule_fgf_final_artifacts_are_from_last_f_and_middle_g(tmp_path: Path) -> None:
    calls: list[tuple[int, str, dict[str, object], dict[str, object]]] = []

    def train_stage(context: LadderStageContext) -> LadderStageOutput:
        calls.append(
            (
                context.index,
                context.component,
                dict(context.component_artifacts),
                dict(context.shared_artifacts),
            )
        )
        return _artifact_stage(context)

    result = run_component_ladder(
        schedule="fgf",
        output_dir=tmp_path,
        train_stage=train_stage,
        initial_component_artifacts={"f": "f0", "g": "g0"},
        initial_shared_artifacts={"interface": "i0"},
        allowed_components=frozenset({"f", "g"}),
    )

    stage0_f = tmp_path / "stage_0_f" / "f_0.pt"
    stage1_g = tmp_path / "stage_1_g" / "g_1.pt"
    stage2_f = tmp_path / "stage_2_f" / "f_2.pt"
    assert result.schedule == ("f", "g", "f")
    assert result.component_artifacts["f"] == stage2_f
    assert result.component_artifacts["g"] == stage1_g
    assert calls[0] == (0, "f", {"f": "f0", "g": "g0"}, {"interface": "i0"})
    assert calls[1][1] == "g"
    assert calls[1][2]["f"] == stage0_f
    assert calls[1][3]["interface"] == stage0_f
    assert calls[2][1] == "f"
    assert calls[2][2]["g"] == stage1_g
    assert calls[2][3]["interface"] == stage1_g


def test_continue_ladder_fg_from_g_manifest_is_g_plus_fg(tmp_path: Path) -> None:
    first = run_component_ladder(
        schedule="g",
        output_dir=tmp_path / "run_g",
        train_stage=_artifact_stage,
        initial_component_artifacts={"f": "f0", "g": "g0"},
        initial_shared_artifacts={"interface": "i0"},
        allowed_components=frozenset({"f", "g"}),
    )
    calls: list[tuple[int, str, dict[str, object], dict[str, object]]] = []

    def train_stage(context: LadderStageContext) -> LadderStageOutput:
        calls.append(
            (
                context.index,
                context.component,
                dict(context.component_artifacts),
                dict(context.shared_artifacts),
            )
        )
        return _artifact_stage(context)

    second = continue_ladder(
        previous_manifest=first.manifest_path,
        schedule="fg",
        output_dir=tmp_path / "run_gfg",
        train_stage=train_stage,
        allowed_components=frozenset({"f", "g"}),
    )

    recovered_g = str(tmp_path / "run_g" / "stage_0_g" / "g_0.pt")
    stage0_f = tmp_path / "run_gfg" / "stage_0_f" / "f_0.pt"
    stage1_g = tmp_path / "run_gfg" / "stage_1_g" / "g_1.pt"
    assert calls[0] == (0, "f", {"f": "f0", "g": recovered_g}, {"interface": recovered_g})
    assert calls[1] == (
        1,
        "g",
        {"f": stage0_f, "g": recovered_g},
        {"interface": stage0_f},
    )
    assert second.schedule == ("f", "g")
    assert second.component_artifacts["f"] == stage0_f
    assert second.component_artifacts["g"] == stage1_g

    manifest = json.loads(second.manifest_path.read_text(encoding="utf-8"))
    assert manifest["previous_manifest"] == str(first.manifest_path)
    assert manifest["stages"][0]["input_component_artifacts"]["g"] == recovered_g
    assert manifest["stages"][1]["input_component_artifacts"]["f"].endswith(
        "run_gfg/stage_0_f/f_0.pt"
    )


def test_component_ladder_rejects_invalid_schedule_component(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="unsupported components"):
        run_component_ladder(
            schedule="fx",
            output_dir=tmp_path,
            train_stage=_artifact_stage,
            allowed_components=frozenset({"f", "g"}),
        )
