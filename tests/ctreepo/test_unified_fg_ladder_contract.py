from __future__ import annotations

import inspect
import json
from pathlib import Path

import pytest
import torch

from treepo.training.local_law import (
    local_law_training_objective_mean as local_law_objective_mean,
)
from src.core.batch_transport import DEFAULT_BATCH_MAX_CONCURRENT, DEFAULT_BATCH_SIZE
from src.ctreepo.alternating import (
    family_default_f,
    family_default_g,
    family_expected_bundle,
    family_resolve_init,
    family_share_state_axes,
    family_supported_inits,
    run_alternating_family,
)
from src.ctreepo.contracts import (
    CTreePOLearningSpec,
    CTreePOProgramSpec,
    LEAF_UNIT_TEXT_TOKEN,
    SOURCE_KIND_RAW_INPUT,
    tree_bundle_metadata,
)
from src.ctreepo.distillation import write_labeled_trees_jsonl
from src.ctreepo.dspy_family import DSPyFamily, DSPyFamilyConfig
from src.ctreepo.embedding_fno import (
    EmbeddingCoordinateFNOTreeRegressor,
    _prepare_trees,
)
from src.ctreepo.fg_arity import two_child_lm_budget_report
from src.ctreepo.fno_family import FNOFamily, FNOFamilyConfig
from src.ctreepo.joint_dspy_family import JointDSPyFamily, JointDSPyFamilyConfig
from src.ctreepo.learning import (
    continue_ladder,
    preflight,
    run_family_runtime_ladder,
    schedule_from_max_iterations,
)
from src.ctreepo.runtime import reduce_tree, score, trace_tree
from src.ctreepo.trl_family import TRLFamily, TRLFamilyConfig
from src.tasks.manifesto.dimension_scorer import DimensionScorer
from src.tasks.manifesto.dimensions import PolicyDimension, get_dimension
from src.tree.labeled import LabeledNode, LabeledTree
from src.tree.full_tree_ipw import local_law_observations_from_state_tree
from src.tree.state_tree import (
    state_tree_skeleton_from_labeled_tree,
    state_tree_trace_metrics,
    update_state_tree_node,
)


class _FakeEmbeddingClient:
    def __init__(self, dim: int = 5) -> None:
        self.dim = int(dim)

    def embed_texts(self, texts):
        return [
            [float((len(str(text)) + idx) % 7) for idx in range(self.dim)]
            for text in texts
        ]


def _tiny_tree(doc_id: str, *, split: str = "test", score: float = 4.0) -> LabeledTree:
    text = (
        f"{doc_id} left policy evidence about investment and jobs. "
        f"{doc_id} right policy evidence about taxation and welfare."
    )
    tree = LabeledTree(
        doc_id=doc_id,
        document_text=text,
        document_score=float(score),
        metadata={
            "split": split,
            "expert_score_1_7": float(score) + 0.1,
            "teacher_score_1_7": float(score),
        },
        label_source="test",
    )
    left = LabeledNode(
        node_id="leaf_0",
        doc_id=doc_id,
        level=0,
        text=f"{doc_id} left policy evidence about investment and jobs.",
        score=float(score) - 0.2,
        metadata={"teacher_summary": "left summary", "target_summary": "left summary"},
    )
    right = LabeledNode(
        node_id="leaf_1",
        doc_id=doc_id,
        level=0,
        text=f"{doc_id} right policy evidence about taxation and welfare.",
        score=float(score) + 0.2,
        metadata={"teacher_summary": "right summary", "target_summary": "right summary"},
    )
    root = LabeledNode(
        node_id="root",
        doc_id=doc_id,
        level=1,
        text=text,
        score=float(score),
        left_child_id="leaf_0",
        right_child_id="leaf_1",
        metadata={"teacher_summary": "root summary", "target_summary": "root summary"},
    )
    tree.add_node(left)
    tree.add_node(right)
    tree.add_node(root)
    return tree


def _write_manifesto_tree_bundle_manifest(bundle_dir: Path, *, dimension: str = "economic") -> None:
    bundle_dir.mkdir(parents=True, exist_ok=True)
    meta = tree_bundle_metadata(
        domain="manifesto_rile",
        leaf_unit=LEAF_UNIT_TEXT_TOKEN,
        source_kind=SOURCE_KIND_RAW_INPUT,
        dimension=dimension,
        target_scale="normalized_1_7",
        leaf_policy={"topology_axis": "size_tokens", "leaf_size_tokens": [8]},
    )
    (bundle_dir / "manifest.json").write_text(
        json.dumps({"config": meta}, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def test_dspy_score_roots_reduces_supplied_tree_with_trained_g(monkeypatch) -> None:
    family = DSPyFamily(
        config=DSPyFamilyConfig(
            dimension="economic",
            num_threads=1,
            leaf_size_tokens=4,
            max_completion_tokens=8,
            lm_config={"model": "fake", "api_base": "http://localhost", "api_key": "EMPTY"},
        )
    )
    g_program = object()
    calls: list[str] = []

    monkeypatch.setattr(family, "_load_g_program", lambda artifact: g_program)
    # A trained (non-passthrough) f is required to exercise the g reduction:
    # teacher-passthrough f short-circuits to the cached teacher root score
    # with zero LM calls (covered by the next test).
    monkeypatch.setattr(family, "_load_f_program", lambda artifact: object())
    monkeypatch.setattr(
        family, "_apply_f_normalized", lambda _program, *, response: 0.5
    )

    def fake_apply_g(_program, *, prompt: str) -> str:
        calls.append(prompt)
        if prompt.startswith("Summarize"):
            return f"generated_leaf_state_{len(calls)}"
        return "generated_root_state"

    monkeypatch.setattr(family, "_apply_g", fake_apply_g)

    scores = family.score_roots_with_f(
        f="trained_f.json",
        g="trained_g.json",
        trees=[_tiny_tree("doc", score=4.0)],
    )

    assert scores == [pytest.approx(family._target_from_scorer_norm(0.5))]
    assert len(calls) == 3
    assert calls[0].startswith("Summarize the following leaf span")
    assert calls[1].startswith("Summarize the following leaf span")
    assert calls[2].startswith("Merge the two child summaries")
    assert "generated_leaf_state_1" in calls[2]
    assert "generated_leaf_state_2" in calls[2]
    assert all("root summary" not in prompt for prompt in calls)


def test_dspy_teacher_passthrough_keeps_cached_teacher_root(monkeypatch) -> None:
    family = DSPyFamily(
        config=DSPyFamilyConfig(
            dimension="economic",
            num_threads=1,
            leaf_size_tokens=4,
            max_completion_tokens=8,
            lm_config={"model": "fake", "api_base": "http://localhost", "api_key": "EMPTY"},
        )
    )

    monkeypatch.setattr(
        family,
        "_load_g_program",
        lambda artifact: family.TEACHER_PASSTHROUGH,
    )
    monkeypatch.setattr(
        family,
        "_load_f_program",
        lambda artifact: family.TEACHER_PASSTHROUGH,
    )

    def fail_apply_g(*_args, **_kwargs):
        raise AssertionError("teacher_passthrough should not call trained g")

    monkeypatch.setattr(family, "_apply_g", fail_apply_g)

    scores = family.score_roots_with_f(
        f=family.TEACHER_PASSTHROUGH,
        g=family.TEACHER_PASSTHROUGH,
        trees=[_tiny_tree("doc", score=4.0)],
    )

    assert scores == [pytest.approx(4.0)]


def test_joint_dspy_score_roots_reduces_supplied_tree_with_trained_g(monkeypatch) -> None:
    family = JointDSPyFamily(
        config=JointDSPyFamilyConfig(
            dimensions=("economic", "social"),
            num_threads=1,
            leaf_size_tokens=4,
            max_completion_tokens=8,
            lm_config={"model": "fake", "api_base": "http://localhost", "api_key": "EMPTY"},
        )
    )
    tree = _tiny_tree("joint_doc", score=4.0)
    root = tree.get_node("root")
    assert root is not None
    root.dimension_scores = {"economic": 4.0, "social": 5.0}
    root.metadata["teacher_dimension_scores_1_7"] = {"economic": 4.0, "social": 5.0}
    g_program = object()
    calls: list[str] = []

    monkeypatch.setattr(family, "_load_g_program", lambda artifact: g_program)
    monkeypatch.setattr(
        family,
        "_load_f_program",
        lambda artifact: family.TEACHER_PASSTHROUGH,
    )

    def fake_apply_g(_program, *, prompt: str) -> str:
        calls.append(prompt)
        if prompt.startswith("Summarize"):
            return f"joint_leaf_state_{len(calls)}"
        return "joint_root_state"

    monkeypatch.setattr(family, "_apply_g", fake_apply_g)

    by_dim = family.score_roots_with_f_by_dimension(
        f=family.TEACHER_PASSTHROUGH,
        g="trained_joint_g.json",
        trees=[tree],
    )

    assert by_dim["economic"] == [pytest.approx(4.0)]
    assert by_dim["social"] == [pytest.approx(5.0)]
    assert len(calls) == 3
    assert calls[0].startswith("Summarize the following leaf span")
    assert calls[1].startswith("Summarize the following leaf span")
    assert calls[2].startswith("Merge the two child summaries")
    assert "joint_leaf_state_1" in calls[2]
    assert "joint_leaf_state_2" in calls[2]
    assert all("root summary" not in prompt for prompt in calls)


class _ToyAlternatingFamily:
    name = "toy"

    def __init__(self) -> None:
        self.calls = []

    def train_f(self, *, f_init, g, traces, output_dir, iteration):
        self.calls.append(("f", iteration, f_init, g))
        output_dir.mkdir(parents=True, exist_ok=True)
        artifact = output_dir / f"f_iter_{iteration:02d}.json"
        artifact.write_text('{"kind": "f"}\n', encoding="utf-8")
        return str(artifact)

    def train_g(self, *, g_init, f, traces, output_dir, iteration):
        self.calls.append(("g", iteration, g_init, f))
        output_dir.mkdir(parents=True, exist_ok=True)
        artifact = output_dir / f"g_iter_{iteration:02d}.json"
        artifact.write_text('{"kind": "g"}\n', encoding="utf-8")
        return str(artifact)

    def score_roots_with_f(self, *, f, g, trees):
        return [float(tree.document_score) for tree in trees]

    def full_tree_traces_with_f_g(self, *, f, g, trees):
        out = []
        for tree in trees:
            trace = state_tree_skeleton_from_labeled_tree(
                tree,
                method_family="toy",
                state_kind="toy_summary",
                split=str((tree.metadata or {}).get("split", "") or ""),
            )
            for node in trace.traverse_preorder():
                target = float(node.metadata.get("target", 0.0))
                update_state_tree_node(
                    trace,
                    node.id,
                    rendered=f"{f}:{g}:{node.id}",
                    state={"f": f, "g": g},
                    metadata={
                        "prediction": target,
                        "proxy_loss": 0.0,
                        "observed": False,
                        "propensity": 0.0,
                    },
                )
            out.append(trace)
        return out


class _ToyVectorAlternatingFamily(_ToyAlternatingFamily):
    name = "toy_vector"

    def score_roots_with_f_by_dimension(self, *, f, g, trees):
        return {
            "economic": [float(tree.document_score) for tree in trees],
            "social": [float(tree.document_score) + 1.0 for tree in trees],
        }

    def full_tree_traces_with_f_g_by_dimension(self, *, f, g, trees):
        by_dim = {"economic": [], "social": []}
        for dim in by_dim:
            for tree in trees:
                trace = state_tree_skeleton_from_labeled_tree(
                    tree,
                    method_family="toy_vector",
                    state_kind=f"toy_{dim}_summary",
                    split=str((tree.metadata or {}).get("split", "") or ""),
                )
                for node in trace.traverse_preorder():
                    target = float(node.metadata.get("target", 0.0))
                    update_state_tree_node(
                        trace,
                        node.id,
                        rendered=f"{dim}:{node.id}",
                        metadata={
                            "dimension": dim,
                            "prediction": target,
                            "proxy_loss": 0.0,
                            "observed": False,
                            "propensity": 0.0,
                        },
                    )
                by_dim[dim].append(trace)
        return by_dim


class _ToyRootOnlyVectorFamily:
    name = "toy_root_vector"

    def score_roots_with_f_by_dimension(self, *, f, g, trees):
        return {
            "economic": [float(tree.document_score) for tree in trees],
            "social": [float(tree.document_score) + 1.0 for tree in trees],
        }


def test_fno_default_shape_contract_is_embedding_axis_with_fixed_channels() -> None:
    pytest.importorskip("neuralop")

    cfg = FNOFamilyConfig(
        leaf_size_tokens=512,
        embedding_max_length_tokens=2048,
        effective_embedding_dim=768,
    )
    assert cfg.chunks_per_leaf == 1
    assert cfg.effective_embedding_dim == 768
    assert cfg.summary_dim == 768
    assert cfg.state_dim == 1536

    model = EmbeddingCoordinateFNOTreeRegressor(
        embedding_dim=768,
        hidden_channels=4,
        n_modes=8,
        n_layers=1,
        head_hidden_dim=8,
        target_min=1.0,
        target_max=7.0,
    )
    leaf_inputs = torch.randn(3, 768)
    leaf_states = model.encode_leaves(leaf_inputs)
    assert tuple(leaf_states.shape) == (3, 1, 768)
    merged = model.merge(leaf_states[:2], leaf_states[1:])
    assert tuple(merged.shape) == (2, 1, 768)

    future = FNOFamilyConfig(
        leaf_size_tokens=4096,
        embedding_max_length_tokens=2048,
        effective_embedding_dim=1536,
    )
    assert future.chunks_per_leaf == 2
    assert future.summary_dim == 1536
    assert future.state_dim == 3072

    with pytest.raises(ValueError, match="state_dim must be at least 2 \\* summary_dim"):
        FNOFamilyConfig(
            leaf_size_tokens=512,
            embedding_max_length_tokens=2048,
            effective_embedding_dim=768,
            summary_dim=768,
            state_dim=1024,
        )


def test_fno_family_declares_shared_ladder_setup() -> None:
    pytest.importorskip("neuralop")

    family = FNOFamily(
        config=FNOFamilyConfig(
            hidden_channels=4,
            n_modes=4,
            n_layers=1,
            head_hidden_dim=8,
            leaf_size_tokens=8,
            embedding_max_length_tokens=None,
            effective_embedding_dim=8,
        ),
        embedding_client=_FakeEmbeddingClient(dim=8),
        device=torch.device("cpu"),
    )

    assert family_default_f(family) == "identity"
    assert family_default_g(family) == "raw_concat"
    assert family_share_state_axes(family) == frozenset({"f", "g"})
    assert family_supported_inits(family) == {
        "f": frozenset({"identity", "artifact"}),
        "g": frozenset({"identity", "raw_concat", "artifact"}),
    }
    assert (
        family_resolve_init(family, kind="f", spec="artifact:/tmp/fno_f.pt")
        == "/tmp/fno_f.pt"
    )
    assert family_resolve_init(family, kind="g", spec="raw_concat") == "identity"
    expected = family_expected_bundle(family)
    assert expected["leaf_unit"] == LEAF_UNIT_TEXT_TOKEN
    assert expected["summary_dim"] == 8
    assert expected["state_dim_min"] == 16


def test_dspy_target_only_trace_has_proxy_rows_without_oracle_rows() -> None:
    family = DSPyFamily(
        config=DSPyFamilyConfig(
            num_threads=1,
            leaf_size_tokens=4,
            lm_context_window_tokens=64,
            max_completion_tokens=8,
            prompt_template_overhead_tokens=8,
            local_law_weight=1.0,
        )
    )

    trace = family.full_tree_traces_with_f_g(
        f=DSPyFamily.TEACHER_PASSTHROUGH,
        g=DSPyFamily.RAW_CONCAT,
        trees=[_tiny_tree("dspy-proxy")],
    )[0]
    observations = local_law_observations_from_state_tree(trace)
    metrics = state_tree_trace_metrics([trace])

    assert len(observations) == trace.node_count
    assert metrics["count_proxy_rows"] == trace.node_count
    assert metrics["count_oracle_rows"] == 0
    assert metrics["count_observed_nodes"] == 0
    assert all(row.oracle_loss is None for row in observations)


def test_dspy_explicit_oracle_metadata_trace_has_observed_row() -> None:
    tree = _tiny_tree("dspy-oracle")
    root = tree.get_node(str(tree.levels[-1][0]))
    assert root is not None
    root.metadata.update(
        {
            "oracle_target": float(root.score),
            "observed": True,
            "sampled": True,
            "propensity": 0.5,
            "label_source": "oracle",
        }
    )
    family = DSPyFamily(
        config=DSPyFamilyConfig(
            num_threads=1,
            leaf_size_tokens=4,
            lm_context_window_tokens=64,
            max_completion_tokens=8,
            prompt_template_overhead_tokens=8,
            local_law_weight=1.0,
        )
    )

    trace = family.full_tree_traces_with_f_g(
        f=DSPyFamily.TEACHER_PASSTHROUGH,
        g=DSPyFamily.RAW_CONCAT,
        trees=[tree],
    )[0]
    observed = [row for row in local_law_observations_from_state_tree(trace) if row.observed]

    assert len(observed) == 1
    assert observed[0].propensity == pytest.approx(0.5)
    assert observed[0].oracle_loss is not None


def test_fno_target_only_trace_has_proxy_rows_without_oracle_rows() -> None:
    family = FNOFamily(
        config=FNOFamilyConfig(
            hidden_channels=4,
            n_modes=4,
            n_layers=1,
            head_hidden_dim=8,
            embedding_max_length_tokens=None,
            effective_embedding_dim=None,
            leaf_size_tokens=8,
            state_dim=None,
            summary_dim=None,
        ),
        embedding_client=_FakeEmbeddingClient(dim=8),
        device=torch.device("cpu"),
    )

    trace = family.full_tree_traces_with_f_g(
        f="identity",
        g="raw_concat",
        trees=[_tiny_tree("fno-proxy")],
    )[0]
    observations = local_law_observations_from_state_tree(trace)
    metrics = state_tree_trace_metrics([trace])

    assert len(observations) == trace.node_count
    assert metrics["count_proxy_rows"] == trace.node_count
    assert metrics["count_oracle_rows"] == 0
    assert metrics["count_observed_nodes"] == 0
    assert all(row.oracle_loss is None for row in observations)


def test_fno_prepare_trees_raises_instead_of_truncating_oversized_leaf() -> None:
    text = " ".join(f"oversized_token_{idx}" for idx in range(60))
    tree = LabeledTree(
        doc_id="oversized",
        document_text=text,
        document_score=4.0,
        metadata={"split": "test", "expert_score_1_7": 4.0},
    )
    tree.add_node(
        LabeledNode(
            node_id="leaf_0",
            doc_id="oversized",
            level=0,
            text=text,
            score=4.0,
        )
    )

    with pytest.raises(RuntimeError, match="needs .* embedding chunks"):
        _prepare_trees(
            [tree],
            embedding_client=_FakeEmbeddingClient(),
            embedding_max_tokens=4,
            chunks_per_leaf=1,
            enforce_no_truncation=True,
        )


def test_dspy_actual_record_budget_guard_hard_errors_before_optimizer() -> None:
    family = DSPyFamily(
        config=DSPyFamilyConfig(
            leaf_size_tokens=1,
            lm_context_window_tokens=16,
            max_completion_tokens=4,
            prompt_template_overhead_tokens=1,
            lm_config={"model": "openai/test", "api_base": "http://localhost:9/v1"},
        )
    )

    with pytest.raises(RuntimeError, match="DSPy no-truncation guard failed"):
        family._check_training_record_budgets(
            [
                {
                    "prompt": " ".join(f"budget_token_{idx}" for idx in range(50)),
                    "response": "short",
                }
            ],
            role="f",
        )


def test_dspy_family_uses_batched_lm_transport_by_default() -> None:
    from src.core.dspy_batch_client import BatchedDSPyLM

    family = DSPyFamily(
        config=DSPyFamilyConfig(
            leaf_size_tokens=1,
            lm_context_window_tokens=16,
            max_completion_tokens=4,
            prompt_template_overhead_tokens=1,
            lm_config={
                "model": "openai/test-model",
                "api_base": "http://localhost:9/v1",
                "api_key": "EMPTY",
                "max_tokens": 4,
            },
        )
    )

    lm = family._ensure_lm()

    try:
        assert isinstance(lm, BatchedDSPyLM)
        assert lm._batch_api_bases == ["http://localhost:9/v1"]
        assert lm._batch_size == DEFAULT_BATCH_SIZE
    finally:
        lm.close()


def test_dspy_f_init_modes_resolve_explicitly() -> None:
    family = DSPyFamily(
        config=DSPyFamilyConfig(
            leaf_size_tokens=1,
            lm_context_window_tokens=16,
            max_completion_tokens=4,
            prompt_template_overhead_tokens=1,
            lm_config={"model": "openai/test", "api_base": "http://localhost:9/v1"},
            f_init_path="",
        )
    )

    assert family._load_f_program("teacher_passthrough") == family.TEACHER_PASSTHROUGH
    bare = family._load_f_program("bare_scorer")
    pretuned_fallback = family._load_f_program("pretuned_scorer")

    assert bare.__class__.__name__ == "DimensionScorer"
    assert pretuned_fallback.__class__.__name__ == "DimensionScorer"


def test_dspy_f_init_path_directory_adapts_full_doc_program(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import dspy

    class FakeFullDocProgram:
        max_output_tokens = 7

        def __call__(self, **kwargs):
            return self.forward(**kwargs)

        def forward(self, dimension: str, task_context: str, document: str):
            assert dimension == "environment"
            assert task_context
            assert document == "climate and growth summary"
            return dspy.Prediction(score="5.5")

    program_dir = tmp_path / "dspy_program"
    program_dir.mkdir()
    (program_dir / "program.pkl").write_bytes(b"placeholder")
    monkeypatch.setattr(dspy, "load", lambda path: FakeFullDocProgram())

    family = DSPyFamily(
        config=DSPyFamilyConfig(
            leaf_size_tokens=1,
            lm_context_window_tokens=16,
            max_completion_tokens=4,
            prompt_template_overhead_tokens=1,
            lm_config={"model": "openai/test", "api_base": "http://localhost:9/v1"},
            dimension="environment",
            f_init_path=str(program_dir),
        )
    )

    loaded = family._load_f_program("pretuned_scorer")

    assert loaded.__class__.__name__ != "DimensionScorer"
    assert family._program_accepts_summary_input(loaded)
    assert loaded(summary="climate and growth summary").score == "5.5"


def test_dspy_native_objective_separates_scorer_and_target_bounds() -> None:
    family = DSPyFamily(
        config=DSPyFamilyConfig(
            leaf_size_tokens=1,
            lm_context_window_tokens=16,
            max_completion_tokens=4,
            prompt_template_overhead_tokens=1,
            lm_config={"model": "openai/test", "api_base": "http://localhost:9/v1"},
            target_min=0.0,
            target_max=10.0,
            scorer_output_min=1.0,
            scorer_output_max=7.0,
        )
    )

    assert family._normalize_scorer_output(4.0) == pytest.approx(0.5)
    assert family._target_from_scorer_norm(0.5) == pytest.approx(5.0)


def test_dimension_scorer_loads_legacy_score_key_as_predictor() -> None:
    scorer = DimensionScorer(get_dimension(PolicyDimension.ECONOMIC))
    legacy_state = {"score": scorer.dump_state()["predictor"]}

    restored = DimensionScorer(get_dimension(PolicyDimension.ECONOMIC))
    restored.load_state(legacy_state)

    assert callable(restored.predictor)
    assert not callable(getattr(restored, "score", None))


def test_trl_validate_artifact_requires_hf_load_markers(tmp_path: Path) -> None:
    family = TRLFamily(
        config=TRLFamilyConfig(
            leaf_size_tokens=8,
            max_completion_tokens=32,
            lm_context_window_tokens=128,
            prompt_template_overhead_tokens=16,
        )
    )
    model_dir = tmp_path / "hf_model"
    model_dir.mkdir()
    (model_dir / "config.json").write_text("{}\n", encoding="utf-8")

    family.validate_artifact(kind="f", artifact=str(model_dir))

    bad_dir = tmp_path / "bad_model"
    bad_dir.mkdir()
    with pytest.raises(RuntimeError, match="no HuggingFace load markers"):
        family.validate_artifact(kind="g", artifact=str(bad_dir))


def test_alternating_ladder_writes_step_checkpoints_with_current_artifacts(tmp_path: Path) -> None:
    trees = [_tiny_tree(f"doc_{idx}", score=3.5 + idx * 0.1) for idx in range(4)]
    family = _ToyAlternatingFamily()

    records = run_alternating_family(
        family=family,
        f_init="f0",
        g_init="g0",
        traces=trees,
        eval_trees=trees,
        max_iterations=2,
        axis_kind="leaf_size_tokens",
        axis_value=8,
        leaf_size_tokens=8,
        output_dir=tmp_path,
    )

    assert [record.stage_name for record in records] == ["fg", "fgf", "fgfg"]
    post_train = json.loads(
        (tmp_path / "step_checkpoints" / "iter_01_post_train.json").read_text(
            encoding="utf-8"
        )
    )
    assert post_train["phase"] == "post_train"
    assert post_train["trained"] == "f"
    assert post_train["f_artifact"].endswith("iter_01_train_f/f_iter_01.json")
    assert post_train["g_artifact"] == "g0"

    latest = json.loads(
        (tmp_path / "step_checkpoints" / "latest.json").read_text(encoding="utf-8")
    )
    assert latest["phase"] == "post_eval"
    assert latest["iteration"] == 2
    assert latest["g_artifact"].endswith("iter_02_train_g/g_iter_02.json")
    assert (tmp_path / "full_tree_traces" / "iter_00_train.jsonl").exists()
    assert (tmp_path / "full_tree_traces" / "iter_00_eval.jsonl").exists()
    assert (tmp_path / "full_tree_traces" / "iter_02_eval_metrics.json").exists()
    assert "full_tree_traces_iter_02_eval_jsonl" in latest["trace_artifacts"]
    assert records[0].extra["trace_artifacts"]["full_tree_traces_iter_00_train_jsonl"].endswith(
        "full_tree_traces/iter_00_train.jsonl"
    )
    assert records[0].extra["trace_metrics"]["train"]["count_proxy_rows"] == 12
    assert records[1].f_artifact == post_train["f_artifact"]
    assert family.calls[0] == ("f", 1, "f0", "g0")
    assert family.calls[1][0:3] == ("g", 2, "g0")
    assert family.calls[1][3] == post_train["f_artifact"]


def test_learning_facade_threads_current_f_into_next_g(tmp_path: Path) -> None:
    trees = [_tiny_tree(f"doc_{idx}", score=3.5 + idx * 0.1) for idx in range(4)]
    family = _ToyAlternatingFamily()

    result = run_family_runtime_ladder(
        family=family,
        f_init="f0",
        g_init="g0",
        traces=trees,
        eval_trees=trees,
        schedule="fg",
        axis_kind="leaf_size_tokens",
        axis_value=8,
        leaf_size_tokens=8,
        output_dir=tmp_path,
    )

    assert [record["stage_name"] for record in result.history] == ["fg", "fgf", "fgfg"]
    assert family.calls[0] == ("f", 1, "f0", "g0")
    assert family.calls[1][0:3] == ("g", 2, "g0")
    assert family.calls[1][3].endswith("iter_01_train_f/f_iter_01.json")
    assert result.artifacts["f"].endswith("iter_01_train_f/f_iter_01.json")
    assert result.artifacts["g"].endswith("iter_02_train_g/g_iter_02.json")
    assert result.artifacts["full_tree_traces_iter_00_train_jsonl"].endswith(
        "full_tree_traces/iter_00_train.jsonl"
    )
    assert result.artifacts["full_tree_traces_iter_02_eval_jsonl"].endswith(
        "full_tree_traces/iter_02_eval.jsonl"
    )
    assert result.history[0]["extra"]["trace_metrics"]["eval"]["count_proxy_rows"] == 12

    manifest = json.loads((tmp_path / "ladder_manifest.json").read_text(encoding="utf-8"))
    assert manifest["stages"][1]["component"] == "g"
    assert manifest["shared_artifacts"]["full_tree_traces_iter_02_eval_jsonl"].endswith(
        "full_tree_traces/iter_02_eval.jsonl"
    )
    assert manifest["stages"][1]["input_component_artifacts"]["f"].endswith(
        "iter_01_train_f/f_iter_01.json"
    )


def test_alternating_ladder_writes_vector_dimension_trace_artifacts(tmp_path: Path) -> None:
    trees = [_tiny_tree(f"doc_{idx}", score=3.5 + idx * 0.1) for idx in range(4)]
    family = _ToyVectorAlternatingFamily()

    records = run_alternating_family(
        family=family,
        f_init="f0",
        g_init="g0",
        traces=trees,
        eval_trees=trees,
        max_iterations=0,
        axis_kind="leaf_size_tokens",
        axis_value=8,
        leaf_size_tokens=8,
        output_dir=tmp_path,
    )

    assert (tmp_path / "full_tree_traces" / "iter_00_eval_economic.jsonl").exists()
    assert (tmp_path / "full_tree_traces" / "iter_00_eval_social.jsonl").exists()
    assert "full_tree_traces_iter_00_eval_economic_jsonl" in records[0].extra["trace_artifacts"]
    assert records[0].extra["trace_metrics"]["eval"]["economic"]["count_proxy_rows"] == 12


def test_alternating_ladder_vector_trace_fallback_keeps_full_topology(tmp_path: Path) -> None:
    trees = [_tiny_tree(f"doc_{idx}", score=3.5 + idx * 0.1) for idx in range(4)]
    for tree in trees:
        root = tree.get_node("root")
        assert root is not None
        root.dimension_scores = {"economic": float(tree.document_score), "social": 5.0}
    family = _ToyRootOnlyVectorFamily()

    records = run_alternating_family(
        family=family,
        f_init="f0",
        g_init="g0",
        traces=trees,
        eval_trees=trees,
        max_iterations=0,
        axis_kind="leaf_size_tokens",
        axis_value=8,
        leaf_size_tokens=8,
        output_dir=tmp_path,
    )

    payload = json.loads(
        (tmp_path / "full_tree_traces" / "iter_00_eval_economic.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()[0]
    )
    assert payload["node_count"] == 3
    assert records[0].extra["trace_metrics"]["eval"]["economic"]["count_nodes"] == 12
    assert records[0].extra["trace_metrics"]["eval"]["economic"]["count_proxy_rows"] == 4


def test_learning_facade_continue_ladder_tracks_g_plus_fg(tmp_path: Path) -> None:
    trees = [_tiny_tree(f"doc_{idx}", score=3.5 + idx * 0.1) for idx in range(4)]
    family = _ToyAlternatingFamily()

    first = run_family_runtime_ladder(
        family=family,
        f_init="f0",
        g_init="g0",
        traces=trees,
        eval_trees=trees,
        schedule="g",
        axis_kind="leaf_size_tokens",
        axis_value=8,
        leaf_size_tokens=8,
        output_dir=tmp_path / "g",
    )
    second = continue_ladder(
        first.manifest_path,
        schedule="fg",
        output_dir=tmp_path / "gfg",
        spec=CTreePOLearningSpec(
            space_kind="labeled_tree",
            family="toy",
            schedule="fg",
            train_data=trees,
            eval_data=trees,
            axis={"axis_kind": "leaf_size_tokens", "axis_value": 8, "leaf_size_tokens": 8},
            backend_config={"family_runtime": family},
        ),
    )

    assert second.summary["previous_manifest"] == str(first.manifest_path)
    assert second.summary["schedule_prefix"] == "g"
    assert second.summary["combined_schedule"] == "gfg"
    assert family.calls[0] == ("g", 1, "g0", "f0")
    assert family.calls[1][0:3] == ("f", 1, "f0")
    assert family.calls[1][3].endswith("g/iter_01_train_g/g_iter_01.json")
    assert family.calls[2][0:3] == ("g", 2, family.calls[1][3])
    assert family.calls[2][3].endswith("gfg/iter_01_train_f/f_iter_01.json")

    manifest = json.loads((tmp_path / "gfg" / "ladder_manifest.json").read_text())
    assert manifest["previous_manifest"] == str(first.manifest_path)
    assert manifest["metadata"]["combined_schedule"] == "gfg"


def test_schedule_from_max_iterations_preserves_legacy_order() -> None:
    assert schedule_from_max_iterations(0) == ""
    assert schedule_from_max_iterations(3) == "fgf"
    assert schedule_from_max_iterations(4, first_train_side="g") == "gfgf"


def test_alternating_ladder_size_axis_smoke_with_fake_trees(tmp_path: Path) -> None:
    pytest.importorskip("neuralop")

    import scripts.run_alternating_ladder as cli

    fg_dir = tmp_path / "fg_grid"
    write_labeled_trees_jsonl(
        fg_dir / "leaf0008tok" / "labeled_trees.jsonl",
        [_tiny_tree(f"doc_{idx}", score=3.5 + idx * 0.2) for idx in range(4)],
    )
    _write_manifesto_tree_bundle_manifest(fg_dir)
    output_dir = tmp_path / "alternating"

    rc = cli.main(
        [
            "--families",
            "fno",
            "--teacher-dir",
            str(fg_dir),
            "--output-dir",
            str(output_dir),
            "--leaf-size-tokens",
            "8",
            "--max-iterations",
            "2",
            "--embedding-backend",
            "hashing",
            "--hashing-embedding-dim",
            "8",
            "--fno-hidden-channels",
            "4",
            "--fno-n-modes",
            "4",
            "--fno-n-layers",
            "1",
            "--fno-head-hidden-dim",
            "8",
            "--fno-epochs",
            "1",
            "--fno-batch-size",
            "2",
            "--fno-device",
            "cpu",
            "--embedding-dim",
            "8",
        ]
    )

    assert rc == 0
    summary = json.loads((output_dir / "grid_summary.json").read_text(encoding="utf-8"))
    assert summary["topology_axis"] == "leaf_size_tokens"
    assert summary["leaf_grid"] is None
    assert summary["leaf_size_tokens"] == [8]
    assert summary["per_row_paths"] == ["fno/leaf0008tok/iteration_history.json"]
    assert {row["axis_kind"] for row in summary["rows"]} == {"leaf_size_tokens"}
    assert {row["leaf_size_tokens"] for row in summary["rows"]} == {8}
    assert summary["run_manifest"]["schema_version"] == "ctreepo.run_manifest.v1"
    assert summary["run_manifest"]["role"] == "fg_ladder_runner"
    assert summary["run_manifest"]["input_contracts"][0]["kind"] == "tree_bundle"

    history = json.loads(
        (output_dir / "fno" / "leaf0008tok" / "iteration_history.json").read_text(
            encoding="utf-8"
        )
    )
    assert history["row_label"] == "leaf0008tok"
    assert history["leaf_count"] is None
    assert history["leaf_size_tokens"] == 8
    assert [row["stage_name"] for row in history["iterations"]] == ["fg", "fgf", "fgfg"]

    latest = json.loads(
        (
            output_dir
            / "fno"
            / "leaf0008tok"
            / "step_checkpoints"
            / "latest.json"
        ).read_text(encoding="utf-8")
    )
    assert latest["phase"] == "post_eval"
    assert latest["artifact_validation"]["g"] == "passed"
    assert latest["g_artifact"].endswith(".pt")


def test_alternating_ladder_tree_bundle_cli_alias(tmp_path: Path) -> None:
    import scripts.run_alternating_ladder as cli

    write_labeled_trees_jsonl(
        tmp_path / "bundle" / "leaf0008tok" / "labeled_trees.jsonl",
        [_tiny_tree("bundle_doc", score=4.0)],
    )
    _write_manifesto_tree_bundle_manifest(tmp_path / "bundle")
    args = cli.parse_args(["--tree-bundle", str(tmp_path / "bundle")])
    assert args.fg_grid_dir == tmp_path / "bundle"
    assert args.used_deprecated_tree_bundle_alias is False
    loaded = cli._load_leaf_size_trees(args.fg_grid_dir, 8)
    assert loaded is not None
    assert [tree.doc_id for tree in loaded] == ["bundle_doc"]

    legacy_args = cli.parse_args(["--teacher-dir", str(tmp_path / "legacy")])
    assert legacy_args.fg_grid_dir == tmp_path / "legacy"
    assert legacy_args.used_deprecated_tree_bundle_alias is True


def test_two_child_lm_budget_report_exposes_preflight_failure() -> None:
    report = two_child_lm_budget_report(
        family_name="dspy",
        leaf_size_tokens=1024,
        lm_context_window_tokens=40000,
        max_completion_tokens=1024,
        prompt_template_overhead_tokens=1500,
    )

    assert report.ok is False
    assert report.required_g_input_tokens == 2048
    assert report.required_g_output_tokens == 2048
    assert report.max_completion_tokens == 1024
    assert report.minimum_context_window_tokens == 5596
    assert any("max_completion_tokens=1024" in item for item in report.violations)
    with pytest.raises(RuntimeError, match="max_completion_tokens=1024"):
        report.raise_for_error()


def test_learning_spec_dspy_preflight_fails_without_teacher_load() -> None:
    invalid = CTreePOLearningSpec(
        space_kind="labeled_tree",
        family="dspy",
        schedule="f",
        axis={"leaf_size_tokens": 1024},
        backend_config={
            "lm_context_window_tokens": 40000,
            "max_completion_tokens": 1024,
            "prompt_template_overhead_tokens": 1500,
        },
    )

    with pytest.raises(RuntimeError, match="max_completion_tokens=1024"):
        preflight(invalid)

    valid = invalid.with_schedule("g")
    valid = CTreePOLearningSpec(
        space_kind=valid.space_kind,
        family=valid.family,
        schedule=valid.schedule,
        initial_artifacts=valid.initial_artifacts,
        train_data=valid.train_data,
        eval_data=valid.eval_data,
        backend_config={
            "lm_context_window_tokens": 40000,
            "max_completion_tokens": 0,
            "prompt_template_overhead_tokens": 1500,
        },
        axis=valid.axis,
    )
    assert preflight(valid)["budget_report"]["ok"] is True


def test_runtime_facade_classical_hll_smoke() -> None:
    spec = CTreePOProgramSpec(
        space_kind="sketch_state",
        family="",
        method_id="hll",
        backend_config={"precision": 8, "backend": "native"},
    )

    root = reduce_tree(spec, [[1, 2, 3], [3, 4], [5, 6]], schedule="balanced")

    assert score(spec, root) == pytest.approx(6.0, rel=0.25)


def test_runtime_facade_classical_hll_emits_full_tree_trace() -> None:
    spec = CTreePOProgramSpec(
        space_kind="sketch_state",
        family="",
        method_id="hll",
        backend_config={"precision": 8, "backend": "native"},
    )

    trace = trace_tree(
        spec,
        [[1, 2], [2, 3], [4]],
        schedule="balanced",
        doc_id="sketch_doc",
        split="test",
        targets={"root": 4.0},
    )

    assert trace.root.metadata["doc_id"] == "sketch_doc"
    assert trace.root.metadata["state_kind"] == "classical_sketch_state"
    assert trace.root.metadata["prediction"] == pytest.approx(4.0, rel=0.25)
    assert trace.root.metadata["target"] == pytest.approx(4.0)
    assert trace.node_count == 5


def _annotate_hll_trace_losses(
    trace,
    leaf_inputs,
    *,
    observed_node_ids: set[str],
    propensity: float = 1.0,
) -> None:
    cache: dict[str, set[int]] = {}

    def exact_items(node) -> set[int]:
        node_id = str(node.id)
        if node_id in cache:
            return cache[node_id]
        if node.is_leaf:
            leaf_index = int(dict(node.span or {}).get("leaf_index", 0))
            items = set(int(x) for x in leaf_inputs[leaf_index])
        else:
            items = set()
            for child in node.children:
                items.update(exact_items(child))
        cache[node_id] = items
        return items

    for node in trace.traverse_preorder():
        prediction = float(node.metadata["prediction"])
        target = float(len(exact_items(node)))
        loss = float((prediction - target) ** 2)
        observed = str(node.id) in observed_node_ids
        metadata = {
            "proxy_target": target,
            "proxy_loss": loss,
            "observed": bool(observed),
            "sampled": bool(observed),
            "propensity": float(propensity) if observed else 0.0,
        }
        if observed:
            metadata.update(
                {
                    "oracle_target": target,
                    "oracle_loss": loss,
                }
            )
        update_state_tree_node(trace, str(node.id), metadata=metadata)


def test_runtime_facade_hll_root_only_trace_has_dense_proxy_sparse_oracle_rows() -> None:
    spec = CTreePOProgramSpec(
        space_kind="sketch_state",
        family="",
        method_id="hll",
        backend_config={"precision": 8, "backend": "native"},
    )
    leaf_inputs = [[1, 2], [2, 3], [4]]
    trace = trace_tree(
        spec,
        leaf_inputs,
        schedule="balanced",
        doc_id="hll_root_only",
        split="train",
    )
    _annotate_hll_trace_losses(trace, leaf_inputs, observed_node_ids={str(trace.root.id)})

    observations = local_law_observations_from_state_tree(trace)
    metrics = state_tree_trace_metrics([trace])

    assert trace.node_count == 5
    assert len(observations) == trace.node_count
    assert metrics["count_proxy_rows"] == trace.node_count
    assert metrics["count_oracle_rows"] == 1
    assert metrics["count_observed_nodes"] == 1
    assert local_law_objective_mean(observations) >= 0.0
    assert local_law_objective_mean(observations, objective_mode="sampled_ipw") >= 0.0


def test_runtime_facade_target_only_trace_is_proxy_only() -> None:
    spec = CTreePOProgramSpec(
        space_kind="sketch_state",
        family="",
        method_id="hll",
        backend_config={"precision": 8, "backend": "native"},
    )
    trace = trace_tree(
        spec,
        [[1, 2], [2, 3], [4]],
        schedule="balanced",
        doc_id="hll_proxy_only_targets",
        split="train",
        targets={
            "leaf_0": 1.0,
            "leaf_1": 2.0,
            "leaf_2": 3.0,
            "merge_1_0": 4.0,
            "root": 5.0,
        },
    )

    observations = local_law_observations_from_state_tree(trace)
    metrics = state_tree_trace_metrics([trace])

    assert len(observations) == trace.node_count
    assert metrics["count_proxy_rows"] == trace.node_count
    assert metrics["count_oracle_rows"] == 0
    assert metrics["count_observed_nodes"] == 0
    assert all(row.oracle_loss is None for row in observations)
    assert all(not row.observed for row in observations)


def test_runtime_facade_explicit_oracle_target_trace_is_observed() -> None:
    spec = CTreePOProgramSpec(
        space_kind="sketch_state",
        family="",
        method_id="hll",
        backend_config={"precision": 8, "backend": "native"},
    )
    trace = trace_tree(
        spec,
        [[1, 2], [2, 3]],
        schedule="balanced",
        doc_id="hll_explicit_oracle",
        split="train",
        targets={
            "root": {
                "target": 5.0,
                "oracle_target": 5.0,
                "observed": True,
                "sampled": True,
                "propensity": 0.5,
                "label_source": "oracle",
            }
        },
    )

    observations = local_law_observations_from_state_tree(trace)
    observed = [row for row in observations if row.observed]
    metrics = state_tree_trace_metrics([trace])

    assert len(observed) == 1
    assert observed[0].propensity == pytest.approx(0.5)
    assert observed[0].oracle_loss is not None
    assert metrics["count_oracle_rows"] == 1
    assert metrics["count_observed_nodes"] == 1


def test_runtime_facade_hll_sampled_node_trace_uses_observed_rows_for_ipw() -> None:
    spec = CTreePOProgramSpec(
        space_kind="sketch_state",
        family="",
        method_id="hll",
        backend_config={"precision": 8, "backend": "native"},
    )
    leaf_inputs = [[1, 2], [2, 3], [4]]
    trace = trace_tree(
        spec,
        leaf_inputs,
        schedule="balanced",
        doc_id="hll_sampled_nodes",
        split="train",
    )
    observed_ids = [
        str(node.id)
        for node in list(trace.traverse_preorder())
        if not bool(node.metadata.get("is_root", False))
    ][:2]
    _annotate_hll_trace_losses(
        trace,
        leaf_inputs,
        observed_node_ids=set(observed_ids),
        propensity=0.5,
    )

    observations = local_law_observations_from_state_tree(trace)
    observed_losses = [
        float(row.oracle_loss)
        for row in observations
        if bool(row.observed) and row.oracle_loss is not None
    ]
    expected_hajek = float(sum(observed_losses) / len(observed_losses))
    metrics = state_tree_trace_metrics([trace])

    assert len(observations) == trace.node_count
    assert metrics["count_proxy_rows"] == trace.node_count
    assert metrics["count_oracle_rows"] == 2
    assert metrics["count_observed_nodes"] == 2
    assert local_law_objective_mean(observations, objective_mode="sampled_ipw") == pytest.approx(
        expected_hajek
    )


def test_alternating_ladder_dspy_preflight_fails_before_teacher_load(tmp_path: Path) -> None:
    import scripts.run_alternating_ladder as cli

    with pytest.raises(SystemExit, match="DSPy budget preflight failed"):
        cli.main(
            [
                "--families",
                "dspy",
                "--teacher-dir",
                str(tmp_path / "missing_teacher_grid"),
                "--output-dir",
                str(tmp_path / "out"),
                "--leaf-size-tokens",
                "1024",
                "--dspy-max-tokens",
                "1024",
                "--dspy-lm-context-tokens",
                "40000",
                "--preflight-only",
            ]
        )


def test_alternating_ladder_dspy_preflight_only_passes_without_teacher_files(tmp_path: Path) -> None:
    import scripts.run_alternating_ladder as cli

    _write_manifesto_tree_bundle_manifest(tmp_path / "missing_teacher_grid")
    rc = cli.main(
        [
            "--families",
            "dspy",
            "--teacher-dir",
            str(tmp_path / "missing_teacher_grid"),
            "--output-dir",
            str(tmp_path / "out"),
            "--leaf-size-tokens",
            "8096",
            "--dspy-max-tokens",
            "0",
            "--dspy-lm-context-tokens",
            "40000",
            "--dspy-prompt-overhead-tokens",
            "1500",
            "--preflight-only",
        ]
    )

    assert rc == 0


def test_alternating_ladder_dspy_batch_transport_cli_defaults() -> None:
    import scripts.run_alternating_ladder as cli

    args = cli.parse_args(
        [
            "--families",
            "dspy",
            "--leaf-size-tokens",
            "8",
            "--dspy-model",
            "openai/test-model",
            "--dspy-api-base",
            "http://localhost:9/v1",
        ]
    )
    family = cli._build_dspy_family(args, leaf_size_tokens=8)

    assert family.config.lm_transport == "batch"
    assert family.config.batch_max_concurrent == DEFAULT_BATCH_MAX_CONCURRENT
    assert family.config.batch_size == DEFAULT_BATCH_SIZE
    assert family.config.f_init_mode == "pretuned_scorer"
    assert tuple(family.config.root_label_sources) == ()


def test_alternating_ladder_local_law_default_objective_cli() -> None:
    import scripts.run_alternating_ladder as cli

    args = cli.parse_args(
        [
            "--families",
            "dspy",
            "--leaf-size-tokens",
            "8",
            "--root-label-sources",
            "stored_summary",
        ]
    )
    family = cli._build_dspy_family(args, leaf_size_tokens=8)

    assert family.config.local_law_weight is None
    assert cli._effective_root_anchor_weight(args) == pytest.approx(0.75)
    assert cli._effective_local_law_weight(args) == pytest.approx(0.25)
    summary = cli._objective_summary(args)
    assert summary["root_share"] == pytest.approx(0.75)
    assert summary["local_law_weight"] == pytest.approx(0.25)
    assert summary["local_law_component_weights"] == pytest.approx(
        {
            "leaf_preservation": 0.25 / 3.0,
            "merge_preservation": 0.25 / 3.0,
            "on_range_idempotence": 0.25 / 3.0,
        }
    )


def test_alternating_ladder_local_law_weight_is_canonical_cli() -> None:
    import scripts.run_alternating_ladder as cli

    args = cli.parse_args(
        [
            "--families",
            "dspy",
            "--leaf-size-tokens",
            "8",
            "--root-label-sources",
            "stored_summary",
            "--local-law-weight",
            "0.25",
        ]
    )
    family = cli._build_dspy_family(args, leaf_size_tokens=8)

    assert family.config.local_law_weight == pytest.approx(0.25)
    assert cli._effective_root_anchor_weight(args) == pytest.approx(0.75)
    assert cli._effective_local_law_weight(args) == pytest.approx(0.25)


def test_alternating_ladder_rejects_gold_standard_lambda_cli() -> None:
    import scripts.run_alternating_ladder as cli

    with pytest.raises(SystemExit):
        cli.parse_args(
            [
                "--families",
                "dspy",
                "--leaf-size-tokens",
                "8",
                "--root-label-sources",
                "stored_summary",
                "--gold-standard-lambda",
                "0.75",
            ]
        )


def test_alternating_ladder_rejects_removed_objective_cli_flags() -> None:
    import scripts.run_alternating_ladder as cli

    with pytest.raises(SystemExit):
        cli.parse_args(["--full-doc-anchor-weight", "1.0"])

    with pytest.raises(SystemExit):
        cli.parse_args(["--teacher-node-lambda", "0.25"])


def test_dspy_family_config_exposes_only_local_law_weight_objective_knob() -> None:
    params = set(inspect.signature(DSPyFamilyConfig).parameters)

    assert "local_law_weight" in params
    assert "gold_standard_lambda" not in params
    assert "full_doc_anchor_weight" not in params
    assert "full_doc_anchor_mode" not in params
    assert "full_doc_anchor_target" not in params
    assert "teacher_node_lambda" not in params


def test_dspy_family_local_law_one_does_not_require_anchor_records() -> None:
    family = DSPyFamily(
        config=DSPyFamilyConfig(
            root_label_sources=("stored_summary",),
            local_law_weight=1.0,
            max_completion_tokens=1024,
        )
    )

    family._assert_full_doc_anchor_records_present(
        [
            {
                "weight": 1.0,
                "metadata": {
                    "law_role": "leaf_g",
                    "example_weight": 1.0,
                },
            }
        ],
        role="g",
    )


def test_alternating_ladder_dspy_initial_artifacts_are_explicit() -> None:
    import scripts.run_alternating_ladder as cli

    args = cli.parse_args(["--families", "dspy", "--leaf-size-tokens", "8"])

    assert cli._initial_f_artifact("dspy", args) == "pretuned_scorer"
    assert cli._initial_g_artifact("dspy", args) == "raw_concat"
    assert cli._initial_f_artifact("fno", args) == "identity"


def test_learning_facade_fno_fake_tree_smoke(tmp_path: Path) -> None:
    pytest.importorskip("neuralop")

    from src.ctreepo.fno_family import FNOFamily, FNOFamilyConfig

    trees = [_tiny_tree(f"doc_{idx}", score=3.5 + idx * 0.2) for idx in range(4)]
    family = FNOFamily(
        config=FNOFamilyConfig(
            hidden_channels=4,
            n_modes=4,
            n_layers=1,
            head_hidden_dim=8,
            epochs_per_iteration=1,
            batch_size=2,
            leaf_size_tokens=8,
            embedding_max_length_tokens=None,
            effective_embedding_dim=8,
        ),
        embedding_client=_FakeEmbeddingClient(dim=8),
        device=torch.device("cpu"),
    )

    result = run_family_runtime_ladder(
        family=family,
        f_init="identity",
        g_init="identity",
        traces=trees,
        eval_trees=trees,
        schedule="fg",
        axis_kind="leaf_size_tokens",
        axis_value=8,
        leaf_size_tokens=8,
        output_dir=tmp_path,
    )

    assert result.artifacts["g"].endswith(".pt")
    latest = json.loads((tmp_path / "step_checkpoints" / "latest.json").read_text())
    assert latest["artifact_validation"]["g"] == "passed"
