from __future__ import annotations

import json
from pathlib import Path

import pytest

from treepo_cdx import (
    ArtifactLineage,
    ArtifactRef,
    DocumentSamplingRow,
    FoldSpec,
    LawKind,
    LocalLawAuditRow,
    ManifestRow,
    ObjectiveSpec,
    ObservationUnitKind,
    RunManifestContract,
    SamplingMetadata,
    Span,
    StateShapeContract,
    SupervisionSpec,
    ThreeLayerHonestyConfig,
    TopLevelUnit,
    UnifiedLearningComponentEvidence,
    assign_folds,
    backend_capabilities,
    build_error_certificate,
    compute_influence_weighted_overlap,
    fit,
    hll_fit_summary,
    local_law_objective_mean,
    local_law_rows_from_manifest,
    manifest_rows_from_document_sampling,
    normalize_objective_spec,
    role_tuple_for_unit,
    split_unit_ids,
)
from treepo_cdx.audit import corrected_losses_from_rows
from treepo_cdx.honesty import assign_three_layer_roles
from treepo_cdx.release import audit_release


def _lineage() -> ArtifactLineage:
    return ArtifactLineage(
        chunker="chunker:v1",
        g="g:v1",
        f="f:v1",
        oracle_online="oracle:online",
        oracle_eval="oracle:eval",
        query_policy="query:v1",
        proxy="proxy:v1",
    )


def test_manifest_round_trips_and_validates() -> None:
    manifest = RunManifestContract(
        run_id="run_1",
        top_level_units=(TopLevelUnit(unit_id="doc_1", length=100),),
        artifacts=(
            ArtifactRef("chunker:v1"),
            ArtifactRef("g:v1"),
            ArtifactRef("f:v1"),
            ArtifactRef("oracle:online"),
            ArtifactRef("oracle:eval"),
            ArtifactRef("query:v1"),
            ArtifactRef("proxy:v1"),
        ),
        rows=(
            ManifestRow(
                row_id="row_1",
                top_level_unit_id="doc_1",
                fold_id="fold_0",
                split_seed=7,
                roles=role_tuple_for_unit("doc_1", ThreeLayerHonestyConfig(enabled=True, split_seed=7)),
                artifacts=_lineage(),
                law_kind="c1_leaf",
                support=Span(0, 50, unit="char"),
                observed=True,
                propensity=0.5,
                truth_source="fixture_oracle",
                approx_source="fixture_proxy",
            ),
        ),
    )
    report = manifest.validate()
    assert report.ok, report.to_dict()
    restored = RunManifestContract.from_dict(json.loads(json.dumps(manifest.to_dict())))
    assert restored == manifest
    assert len(restored.digest) == 64


def test_manifest_rejects_missing_parent_and_invalid_observed_propensity() -> None:
    with pytest.raises(ValueError, match="positive propensity"):
        ManifestRow(
            row_id="bad",
            top_level_unit_id="doc_1",
            support=Span(0, 1),
            observed=True,
            propensity=0.0,
        )

    manifest = RunManifestContract(
        run_id="run_1",
        top_level_units=(TopLevelUnit(unit_id="doc_1", length=10),),
        rows=(
            ManifestRow(
                row_id="row_1",
                top_level_unit_id="missing",
                support=Span(0, 1),
                observed=False,
                propensity=0.0,
            ),
        ),
    )
    report = manifest.validate(require_artifacts=False)
    assert not report.ok
    assert "missing top-level unit" in report.errors[0]


def test_objective_rejects_additive_oracle_gap_terms() -> None:
    spec = ObjectiveSpec(
        objective_family="root_plus_laws",
        local_law_component_weights={"c1": 0.2, "c3": 0.3},
    )
    payload = spec.to_dict()
    assert set(payload["terms"]) == {"root", "local_law_corrected"}
    assert payload["local_law_component_weights"]["leaf_preservation"] == pytest.approx(0.2)
    assert payload["local_law_weight"] == pytest.approx(0.5)

    with pytest.raises(ValueError, match="oracle_gap"):
        normalize_objective_spec({"terms": {"oracle_gap": {"weight": 1.0}}})
    with pytest.raises(ValueError, match="legacy public objective fields"):
        normalize_objective_spec({"gap_weight": 1.0})


def test_audit_rows_compute_corrected_losses_and_overlap() -> None:
    rows = [
        LocalLawAuditRow(
            row_id="r0",
            law_kind=LawKind.C1_LEAF,
            proxy_loss=0.4,
            observed=False,
            propensity=0.0,
            node_weight=1.0,
        ),
        LocalLawAuditRow(
            row_id="r1",
            law_kind="c3",
            proxy_loss=0.4,
            oracle_loss=0.1,
            observed=True,
            propensity=0.5,
            node_weight=2.0,
        ),
    ]
    assert corrected_losses_from_rows(rows) == pytest.approx([0.4, -0.2])
    overlap = compute_influence_weighted_overlap(rows)
    assert overlap.D_lambda == pytest.approx(1.0 / 1e-12 + 4.0 / 0.5)
    assert overlap.W_lambda == pytest.approx(1.0 / 1e-12)
    assert overlap.effective_sample_size > 0.0


def test_local_law_objective_supports_corrected_and_sampled_modes() -> None:
    rows = [
        LocalLawAuditRow(
            row_id="r0",
            law_kind=LawKind.C1_LEAF,
            proxy_loss=0.4,
            observed=False,
            propensity=0.0,
            node_weight=1.0,
        ),
        LocalLawAuditRow(
            row_id="r1",
            law_kind="c3",
            proxy_loss=0.4,
            oracle_loss=0.1,
            observed=True,
            propensity=0.5,
            node_weight=2.0,
        ),
    ]
    assert local_law_objective_mean(rows) == pytest.approx(0.0)
    assert local_law_objective_mean(rows, objective_mode="sampled_ipw") == pytest.approx(0.1)


def test_folds_are_stable_and_disjoint() -> None:
    spec = FoldSpec(n_folds=3, seed=7, eval_fold=1, namespace="fixture")
    assignments = assign_folds(["doc_1", "doc_2", "doc_3", "doc_4"], spec)
    assert assignments == assign_folds(["doc_1", "doc_2", "doc_3", "doc_4"], spec)
    assert spec.artifact_id.startswith("folds:")
    split = split_unit_ids(assignments)
    assert set(split["train"]).isdisjoint(set(split["eval"]))


def test_manifest_and_sampling_rows_adapt_to_objective_rows() -> None:
    manifest_row = ManifestRow(
        row_id="law_1",
        top_level_unit_id="doc_1",
        law_kind="c1_leaf",
        support=Span(0, 10),
        observed=True,
        propensity=0.5,
        metadata={"proxy_loss": 0.4, "oracle_loss": 0.1, "node_weight": 2.0},
    )
    local_rows = local_law_rows_from_manifest((manifest_row,))
    assert local_rows[0].corrected_loss() == pytest.approx(-0.2)

    sampling_rows = manifest_rows_from_document_sampling(
        (
            DocumentSamplingRow(
                top_level_unit_id="doc_1",
                observed=True,
                inclusion_probability=0.25,
                prediction=0.3,
                truth=0.2,
                fold_id="fold_0",
            ),
        )
    )
    assert sampling_rows[0].row_id == "document:doc_1"
    assert sampling_rows[0].propensity == pytest.approx(0.25)


def test_backend_capabilities_are_small_and_serializable() -> None:
    class DummyRuntime:
        def state_shape_contract(self) -> StateShapeContract:
            return StateShapeContract(state_family="hll", shape=(128,), dtype="float32")

        def supported_supervisions(self) -> tuple[SupervisionSpec, str]:
            return (
                SupervisionSpec(name="local_law", requires_oracle=True),
                "root",
            )

    caps = backend_capabilities(DummyRuntime())
    assert caps["state_shape_contract"]["state_family"] == "hll"
    assert [item["name"] for item in caps["supported_supervisions"]] == ["local_law", "root"]


def test_hll_sketch_fit_uses_existing_native_hll(tmp_path: Path) -> None:
    summary = hll_fit_summary(((1, 2, 3), (3, 4, 5)), precision=4)
    assert summary["true_cardinality"] == pytest.approx(5.0)
    assert summary["capabilities"]["state_shape_contract"]["shape"] == [16]

    result = fit(
        {
            "mode": "hll_sketch",
            "leaf_token_lists": [[1, 2, 3], [3, 4, 5]],
            "precision": 4,
        },
        output_dir=tmp_path / "hll",
    )
    assert result.status == "ok"
    assert result.metrics["true_cardinality"] == pytest.approx(5.0)
    assert Path(result.artifacts["fit_result_json"]).exists()


def test_certificate_sampling_and_honesty() -> None:
    cert = build_error_certificate(
        reported_estimate=0.2,
        component_evidence=[
            UnifiedLearningComponentEvidence(component="local_law", radius=0.1),
            UnifiedLearningComponentEvidence(component="calibration", radius=0.2),
            UnifiedLearningComponentEvidence(component="estimation", radius=0.3),
            UnifiedLearningComponentEvidence(component="clipping", radius=0.4),
        ],
    )
    assert cert.radius_sum == pytest.approx(1.0)
    assert cert.total_bound == pytest.approx(1.2)

    sampling = SamplingMetadata(
        document_propensity=0.5,
        unit_propensity=0.25,
        label_propensity=1.0,
        unit_kind=ObservationUnitKind.LEAF,
    )
    assert sampling.effective_joint_propensity() == pytest.approx(0.125)
    assert sampling.ipw_weight() == pytest.approx(8.0)

    cfg = ThreeLayerHonestyConfig(enabled=True, split_seed=11)
    assert assign_three_layer_roles("doc_1", cfg) == assign_three_layer_roles("doc_1", cfg)


def test_fit_runtime_uses_existing_runtime_pattern(tmp_path: Path) -> None:
    dataset = tmp_path / "tiny.jsonl"
    dataset.write_text(
        json.dumps(
            {
                "_id": "lbv2-1",
                "domain": "law",
                "sub_domain": "contracts",
                "difficulty": "easy",
                "length": "short",
                "question": "Which option is supported?",
                "choice_A": "No evidence",
                "choice_B": "The contract was signed.",
                "choice_C": "The contract expired.",
                "choice_D": "The contract was void.",
                "answer": "B",
                "context": "The contract was signed.",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    result = fit(
        {
            "experiment_id": "fit_runtime_smoke",
            "benchmark": {"family": "longbench_v2", "dataset": str(dataset), "split": "fixture"},
            "methods": ["full_context"],
            "scorer": {"kind": "mock", "model": "deterministic-overlap"},
            "oracle": {"kind": "benchmark_labels"},
            "runtime_defaults": {"mock": True, "max_output_tokens": 4},
        },
        output_dir=tmp_path / "fit",
    )
    assert result.status == "ok"
    assert result.metrics["n"] == pytest.approx(1.0)
    assert Path(result.artifacts["json_out"]).exists()
    assert Path(result.artifacts["fit_result_json"]).exists()
    assert result.manifest_path is not None
    manifest = RunManifestContract.from_dict(json.loads(Path(result.manifest_path).read_text(encoding="utf-8")))
    report = manifest.validate()
    assert report.ok, report.to_dict()
    assert len(manifest.rows) == 1


def test_fit_local_law_path_is_self_contained(tmp_path: Path) -> None:
    result = fit(
        {
            "mode": "local_law",
            "local_law_rows": [
                {
                    "row_id": "r0",
                    "law_kind": "c1_leaf",
                    "proxy_loss": 0.4,
                    "observed": False,
                    "propensity": 0.0,
                    "node_weight": 1.0,
                },
                {
                    "row_id": "r1",
                    "law_kind": "c3",
                    "proxy_loss": 0.4,
                    "oracle_loss": 0.1,
                    "observed": True,
                    "propensity": 0.5,
                    "node_weight": 2.0,
                },
            ],
        },
        output_dir=tmp_path / "local_law",
    )
    assert result.status == "ok"
    assert result.metrics["objective"] == pytest.approx(0.0)
    assert Path(result.artifacts["fit_result_json"]).exists()


def test_release_audit_keeps_package_import_light() -> None:
    report = audit_release()
    assert report["ok"], report
