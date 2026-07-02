from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import dspy
import pytest

from src.tasks.manifesto.dimensions import BENOIT_DIMENSIONS, PolicyDimension


def _score_with_feedback_type():
    try:
        from dspy.teleprompt.gepa import gepa_utils
    except Exception:  # noqa: BLE001
        pytest.skip("DSPy GEPA utilities are unavailable")
    score_with_feedback = getattr(gepa_utils, "ScoreWithFeedback", None)
    if score_with_feedback is None:
        pytest.skip("ScoreWithFeedback not available in this DSPy version")
    return score_with_feedback


def _load_phase3_module():
    root = Path(__file__).resolve().parents[2]
    path = root / "scripts" / "phase3_full_pipeline_optimize.py"
    spec = importlib.util.spec_from_file_location("phase3_full_pipeline_optimize_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_scalar_metric_supports_mae_and_rank_modes():
    module = _load_phase3_module()
    gold = dspy.Example(expert_mean=5.0)
    pred = dspy.Prediction(score=3.0)

    mae = module._metric(gold, pred, mode="mae")
    rank = module._metric(gold, pred, mode="rank")

    assert mae == pytest.approx(1.0 - 2.0 / 6.0)
    assert rank == pytest.approx(mae - 0.25)


def test_rich_gepa_metric_reports_parse_failure_and_feedback_content():
    ScoreWithFeedback = _score_with_feedback_type()
    module = _load_phase3_module()
    spec = BENOIT_DIMENSIONS[PolicyDimension.DECENTRALIZATION]
    metric = module._make_gepa_metric(spec, mode="rank", feedback_mode="rich")
    gold = dspy.Example(expert_mean=5.5)
    pred = dspy.Prediction(score=None, reasoning="unclear", summary="regional autonomy")

    result = metric(gold, pred)

    assert isinstance(result, ScoreWithFeedback)
    assert result.score == 0.0
    assert "Parse/NA failure" in result.feedback
    assert "decentralization" in result.feedback


def test_rich_gepa_metric_includes_direction_rank_side_and_anchors():
    ScoreWithFeedback = _score_with_feedback_type()
    module = _load_phase3_module()
    spec = BENOIT_DIMENSIONS[PolicyDimension.ECONOMIC]
    metric = module._make_gepa_metric(spec, mode="rank", feedback_mode="rich")
    gold = dspy.Example(expert_mean=2.0)
    pred = dspy.Prediction(score=6.0, reasoning="tax cuts", summary="tax reduction")

    result = metric(gold, pred)

    assert isinstance(result, ScoreWithFeedback)
    assert "Direction of correction: lower" in result.feedback
    assert "Rank-side check" in result.feedback
    assert spec.anchor_low in result.feedback
    assert spec.anchor_high in result.feedback


def test_component_warm_start_accepts_scorer_only_artifact(tmp_path):
    module = _load_phase3_module()
    from src.tasks.manifesto.dimension_scorer import DimensionScorer

    spec = BENOIT_DIMENSIONS[PolicyDimension.DECENTRALIZATION]
    scorer = DimensionScorer(spec)
    import json

    path = tmp_path / "optimized_scorer.json"
    scorer.save(path)
    state = json.loads(path.read_text())
    state["predictor"]["signature"]["instructions"] = "warm scorer instruction"
    path.write_text(json.dumps(state))

    pipeline = module.DimensionFullPipeline(PolicyDimension.DECENTRALIZATION)
    loaded = module._warm_start_pipeline(
        pipeline,
        init_program=None,
        init_scorer=path,
        init_g=None,
        init_g_legacy_leaf=None,
    )

    assert loaded == {"scorer": str(path)}
    assert pipeline.scorer.signature.instructions == "warm scorer instruction"


def test_component_warm_start_accepts_full_program_g_artifact(tmp_path):
    module = _load_phase3_module()
    from src.tasks.manifesto.pipeline import UnifiedManifestoG

    g = UnifiedManifestoG()
    g_state = g.dump_state()["summarize"]
    g_state["signature"]["instructions"] = "warm g instruction"
    path = tmp_path / "program_like.json"
    path.write_text(__import__("json").dumps({"g.summarize": g_state}))

    pipeline = module.DimensionFullPipeline(PolicyDimension.DECENTRALIZATION)
    loaded = module._warm_start_pipeline(
        pipeline,
        init_program=None,
        init_scorer=None,
        init_g=path,
        init_g_legacy_leaf=None,
    )

    assert loaded == {"g": str(path)}
    assert pipeline.g.summarize.signature.instructions == "warm g instruction"


def test_component_warm_start_transplants_legacy_leaf_instruction(tmp_path):
    module = _load_phase3_module()
    path = tmp_path / "leaf_summarizer_final.json"
    path.write_text(__import__("json").dumps({
        "summarize": {
            "signature": {
                "instructions": "legacy leaf instruction for unified g",
            },
            "demos": [],
            "traces": [],
        },
    }))

    pipeline = module.DimensionFullPipeline(
        PolicyDimension.DECENTRALIZATION,
        optimize_scope="f",
    )
    loaded = module._warm_start_pipeline(
        pipeline,
        init_program=None,
        init_scorer=None,
        init_g=None,
        init_g_legacy_leaf=path,
    )

    assert loaded == {"g_legacy_leaf_instruction": str(path)}
    assert pipeline.g._module.summarize.signature.instructions == "legacy leaf instruction for unified g"


def test_init_dir_falls_back_to_legacy_leaf_when_no_unified_g(tmp_path):
    module = _load_phase3_module()
    leaf = tmp_path / "leaf_summarizer_final.json"
    leaf.write_text(__import__("json").dumps({
        "summarize": {"signature": {"instructions": "legacy leaf"}},
    }))
    args = type("Args", (), {
        "init_dir": tmp_path,
        "init_program": None,
        "init_scorer": None,
        "init_g": None,
        "init_g_legacy_leaf": None,
    })()

    paths = module._resolve_init_paths(args)

    assert paths["g"] is None
    assert paths["g_legacy_leaf"] == leaf


def test_init_dir_prefers_final_component_artifacts(tmp_path):
    module = _load_phase3_module()
    for filename in [
        "optimized_program.json",
        "final_program.json",
        "optimized_scorer.json",
        "scorer_final.json",
        "optimized_unified_g.json",
        "unified_g_final.json",
    ]:
        (tmp_path / filename).write_text("{}")
    args = type("Args", (), {
        "init_dir": tmp_path,
        "init_program": None,
        "init_scorer": None,
        "init_g": None,
        "init_g_legacy_leaf": None,
    })()

    paths = module._resolve_init_paths(args)

    assert paths["program"] == tmp_path / "final_program.json"
    assert paths["scorer"] == tmp_path / "scorer_final.json"
    assert paths["g"] == tmp_path / "unified_g_final.json"


def test_init_dir_can_prefer_optimized_component_artifacts(tmp_path):
    module = _load_phase3_module()
    for filename in [
        "optimized_program.json",
        "final_program.json",
        "optimized_scorer.json",
        "scorer_final.json",
        "optimized_unified_g.json",
        "unified_g_final.json",
    ]:
        (tmp_path / filename).write_text("{}")
    args = type("Args", (), {
        "init_dir": tmp_path,
        "init_artifact_kind": "optimized",
        "init_program": None,
        "init_scorer": None,
        "init_g": None,
        "init_g_legacy_leaf": None,
    })()

    paths = module._resolve_init_paths(args)

    assert paths["program"] == tmp_path / "optimized_program.json"
    assert paths["scorer"] == tmp_path / "optimized_scorer.json"
    assert paths["g"] == tmp_path / "optimized_unified_g.json"


def test_init_dir_components_only_skips_inferred_program_artifact(tmp_path):
    module = _load_phase3_module()
    for filename in [
        "optimized_program.json",
        "optimized_scorer.json",
        "optimized_unified_g.json",
    ]:
        (tmp_path / filename).write_text("{}")
    args = type("Args", (), {
        "init_dir": tmp_path,
        "init_artifact_kind": "optimized",
        "init_components_only": True,
        "init_program": None,
        "init_scorer": None,
        "init_g": None,
        "init_g_legacy_leaf": None,
    })()

    paths = module._resolve_init_paths(args)

    assert paths["program"] is None
    assert paths["scorer"] == tmp_path / "optimized_scorer.json"
    assert paths["g"] == tmp_path / "optimized_unified_g.json"


def test_partial_init_resolution_allows_learned_f_with_fresh_g(tmp_path):
    module = _load_phase3_module()
    scorer = tmp_path / "optimized_scorer.json"
    scorer.write_text("{}")
    args = type("Args", (), {
        "init_dir": None,
        "init_program": None,
        "init_scorer": scorer,
        "init_g": None,
        "init_g_legacy_leaf": None,
    })()

    paths = module._resolve_init_paths(args)

    assert paths["scorer"] == scorer
    assert paths["g"] is None
    assert paths["program"] is None


def test_partial_init_resolution_allows_learned_g_with_fresh_f(tmp_path):
    module = _load_phase3_module()
    g = tmp_path / "optimized_unified_g.json"
    g.write_text("{}")
    args = type("Args", (), {
        "init_dir": None,
        "init_program": None,
        "init_scorer": None,
        "init_g": g,
        "init_g_legacy_leaf": None,
    })()

    paths = module._resolve_init_paths(args)

    assert paths["g"] == g
    assert paths["scorer"] is None
    assert paths["program"] is None


def test_component_warm_start_allows_no_paths():
    module = _load_phase3_module()
    pipeline = module.DimensionFullPipeline(PolicyDimension.DECENTRALIZATION)

    loaded = module._warm_start_pipeline(
        pipeline,
        init_program=None,
        init_scorer=None,
        init_g=None,
        init_g_legacy_leaf=None,
    )

    assert loaded == {}


def test_evaluate_reports_prediction_jsonl_path(tmp_path):
    module = _load_phase3_module()

    class ToyProgram:
        def __call__(self, text):
            return dspy.Prediction(score={"a": 5.0, "b": 4.0, "c": 3.0, "d": 2.0}[text])

    examples = [
        dspy.Example(text="a", expert_mean=5.0, manifesto_id="m1").with_inputs("text"),
        dspy.Example(text="b", expert_mean=4.0, manifesto_id="m2").with_inputs("text"),
        dspy.Example(text="c", expert_mean=3.0, manifesto_id="m3").with_inputs("text"),
        dspy.Example(text="d", expert_mean=2.0, manifesto_id="m4").with_inputs("text"),
    ]

    report = module._evaluate(
        ToyProgram(),
        examples,
        "optimized_dev",
        tmp_path,
        "decentralization",
    )

    pred_path = tmp_path / "per_mfesto_optimized_dev.jsonl"
    assert report["prediction_path"] == str(pred_path)
    assert report["n_examples_requested"] == 4
    assert pred_path.exists()
