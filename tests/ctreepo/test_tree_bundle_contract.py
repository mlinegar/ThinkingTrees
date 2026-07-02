from __future__ import annotations

import pytest

from src.ctreepo.contracts import (
    LAW_ID_LEAF_PRESERVATION,
    LEAF_UNIT_STREAM_ITEM,
    LEAF_UNIT_SYNTHETIC_ATOM,
    LEAF_UNIT_TEXT_TOKEN,
    REDUCER_CONTRACT_BOTTOM_UP,
    RUN_MANIFEST_SCHEMA_VERSION,
    OBJECTIVE_SCHEMA_VERSION,
    SOURCE_KIND_EXTERNAL_STATE,
    SOURCE_KIND_RAW_INPUT,
    SOURCE_KIND_SYNTHETIC_ORACLE,
    STATE_CONTRACT_ORACLE_STATE,
    TREE_BUNDLE_SCHEMA_VERSION,
    RunManifest,
    TreeBundleManifest,
    normalize_objective_spec,
    normalize_run_manifest,
    objective_metadata,
    objective_spec_digest,
    run_manifest_metadata,
    sketch_tree_bundle_metadata,
    normalize_tree_bundle_manifest,
    tree_bundle_input_contract,
    tree_bundle_metadata,
    validate_objective_spec,
    validate_run_manifest,
    validate_tree_bundle_manifest,
)


def test_tree_bundle_contract_text_token_raw_default() -> None:
    meta = tree_bundle_metadata(
        domain="manifesto_rile",
        leaf_unit=LEAF_UNIT_TEXT_TOKEN,
        source_kind=SOURCE_KIND_RAW_INPUT,
        dimension="economic",
        target_scale="normalized_1_7",
        leaf_policy={"topology_axis": "size_tokens", "leaf_size_tokens": [256]},
    )

    assert meta["schema_version"] == TREE_BUNDLE_SCHEMA_VERSION
    assert meta["tree_representation"] == "partition_tree"
    assert meta["source_kind"] == SOURCE_KIND_RAW_INPUT
    assert meta["state_contract"] == "raw_concat"
    assert meta["reducer_contract"] == REDUCER_CONTRACT_BOTTOM_UP
    assert meta["tree_bundle_manifest"]["leaf_unit"] == LEAF_UNIT_TEXT_TOKEN
    assert meta["tree_bundle_kind"] == "raw_manifesto_token_tree"
    assert meta["tree_text_source"] == "aligned_text"


def test_tree_bundle_contract_normalizes_legacy_external_summary() -> None:
    normalized = normalize_tree_bundle_manifest(
        {
            "tree_bundle_kind": "external_summary_token_tree",
            "tree_text_source": "existing_summary",
            "external_state_producer": "g_benoit",
            "dimension": "environment",
            "expert_target_scale": "normalized_1_7",
        }
    )

    assert normalized["schema_version"] == TREE_BUNDLE_SCHEMA_VERSION
    assert normalized["source_kind"] == SOURCE_KIND_EXTERNAL_STATE
    assert normalized["state_contract"] == "external_passthrough"
    assert normalized["external_state_producer"] == "g_benoit"
    assert normalized["dimension"] == "environment"
    assert normalized["target_scale"] == "normalized_1_7"


def test_tree_bundle_contract_supports_non_text_tree_units() -> None:
    stream = TreeBundleManifest(
        domain="classical_sketch",
        leaf_unit=LEAF_UNIT_STREAM_ITEM,
        source_kind=SOURCE_KIND_RAW_INPUT,
        state_dim=64,
        summary_dim=32,
    )
    synthetic = TreeBundleManifest(
        domain="markov",
        leaf_unit=LEAF_UNIT_SYNTHETIC_ATOM,
        source_kind=SOURCE_KIND_SYNTHETIC_ORACLE,
        state_contract=STATE_CONTRACT_ORACLE_STATE,
    )

    assert stream.to_dict()["leaf_unit"] == LEAF_UNIT_STREAM_ITEM
    assert synthetic.to_dict()["source_kind"] == SOURCE_KIND_SYNTHETIC_ORACLE


def test_tree_bundle_contract_enforces_state_dim_capacity() -> None:
    with pytest.raises(ValueError, match="state_dim must be at least 2 \\* summary_dim"):
        TreeBundleManifest(
            domain="classical_sketch",
            leaf_unit=LEAF_UNIT_STREAM_ITEM,
            source_kind=SOURCE_KIND_RAW_INPUT,
            state_dim=63,
            summary_dim=32,
        )


def test_tree_bundle_contract_exempts_explicit_oracle_state_capacity() -> None:
    meta = TreeBundleManifest(
        domain="classical_sketch",
        leaf_unit=LEAF_UNIT_STREAM_ITEM,
        source_kind=SOURCE_KIND_RAW_INPUT,
        state_contract=STATE_CONTRACT_ORACLE_STATE,
        state_dim=32,
        summary_dim=32,
    )

    assert meta.to_dict()["state_contract"] == STATE_CONTRACT_ORACLE_STATE


def test_sketch_tree_bundle_metadata_records_fg_lineage() -> None:
    meta = sketch_tree_bundle_metadata(
        family="distinct",
        query="estimate",
        sketch="learned_g_exact_distinct",
        state_contract="bottom_up_g",
        summary_dim=16,
        state_dim=32,
        f_init="official_oracle",
        g_init="raw_concat",
        schedule="fg",
    )

    manifest = meta["tree_bundle_manifest"]
    assert manifest["domain"] == "classical_sketch"
    assert manifest["leaf_unit"] == LEAF_UNIT_STREAM_ITEM
    assert manifest["reducer_contract"] == REDUCER_CONTRACT_BOTTOM_UP
    assert manifest["f_lineage"]["init"] == "official_oracle"
    assert manifest["g_lineage"]["init"] == "raw_concat"
    assert manifest["metadata"]["schedule"] == "fg"


def test_tree_bundle_validator_checks_expected_fields() -> None:
    meta = tree_bundle_metadata(
        domain="manifesto_rile",
        leaf_unit=LEAF_UNIT_TEXT_TOKEN,
        source_kind=SOURCE_KIND_RAW_INPUT,
        dimension="economic",
    )

    validate_tree_bundle_manifest(
        meta,
        expected_domain="manifesto_rile",
        expected_leaf_unit=LEAF_UNIT_TEXT_TOKEN,
        expected_source_kind=SOURCE_KIND_RAW_INPUT,
        expected_dimension="economic",
    )
    with pytest.raises(ValueError, match="source_kind mismatch"):
        validate_tree_bundle_manifest(
            meta,
            expected_source_kind=SOURCE_KIND_EXTERNAL_STATE,
        )


def test_run_manifest_wraps_tree_bundle_input_and_lineage() -> None:
    tree = tree_bundle_metadata(
        domain="manifesto_rile",
        leaf_unit=LEAF_UNIT_TEXT_TOKEN,
        source_kind=SOURCE_KIND_RAW_INPUT,
        dimension="economic",
    )
    run = run_manifest_metadata(
        run_id="manifesto.test",
        domain="manifesto_rile",
        role="fg_ladder_runner",
        backend="dspy",
        status="completed",
        tree_bundle=tree,
        f_init="official_oracle",
        g_init="raw_concat",
        f_lineage={"init": "official_oracle"},
        g_lineage={"init": "raw_concat"},
        schedule="fg",
        audit_results={"ok": True},
        publication_ready=True,
    )

    assert run["schema_version"] == RUN_MANIFEST_SCHEMA_VERSION
    assert run["input_contracts"][0]["kind"] == "tree_bundle"
    assert run["input_contracts"][0]["digest"]
    parsed = validate_run_manifest(
        run,
        expected_domain="manifesto_rile",
        require_tree_bundle=True,
        require_lineage=True,
        require_publication_ready=True,
    )
    assert parsed.publication_ready is True


def test_run_manifest_supports_non_tree_exploratory_runs() -> None:
    run = RunManifest(
        run_id="runtime.smoke",
        domain="runtime_eval",
        role="longbench_runtime",
        backend="runtime",
        status="planned",
        metadata={"expected_input_contract": "runtime_eval_manifest"},
    ).to_dict()

    assert normalize_run_manifest(run)["schema_version"] == RUN_MANIFEST_SCHEMA_VERSION
    validate_run_manifest(run, expected_domain="runtime_eval")


def test_run_manifest_publication_ready_rejects_bad_quarantine() -> None:
    tree = tree_bundle_metadata(
        domain="classical_sketch",
        leaf_unit=LEAF_UNIT_STREAM_ITEM,
        source_kind=SOURCE_KIND_RAW_INPUT,
        state_dim=64,
        summary_dim=32,
        include_legacy_manifesto_aliases=False,
    )
    run = run_manifest_metadata(
        run_id="sketch.bad",
        domain="classical_sketch",
        role="paper_bundle",
        backend="treepo",
        tree_bundle=tree,
        audit_results={"ok": True},
        quarantine={"classification": "missing_contract"},
        publication_ready=True,
    )

    with pytest.raises(ValueError, match="quarantine classification"):
        validate_run_manifest(run, require_publication_ready=True)


def test_tree_bundle_input_contract_has_stable_digest() -> None:
    tree = tree_bundle_metadata(
        domain="manifesto_rile",
        leaf_unit=LEAF_UNIT_TEXT_TOKEN,
        source_kind=SOURCE_KIND_RAW_INPUT,
    )
    first = tree_bundle_input_contract(tree)
    second = tree_bundle_input_contract(tree["tree_bundle_manifest"])

    assert first["digest"] == second["digest"]


def test_objective_spec_canonical_shape_and_digest() -> None:
    objective = objective_metadata(
        objective_family="manifesto_alternating_ladder",
        local_law_estimator="proxy_only",
        local_law_weight=0.25,
        root_share=0.75,
        local_law_component_weights={LAW_ID_LEAF_PRESERVATION: 0.25},
    )

    assert objective["schema_version"] == OBJECTIVE_SCHEMA_VERSION
    assert set(objective["terms"]) >= {"root", "local_law_corrected"}
    assert "oracle_gap" not in objective["terms"]
    assert objective["terms"]["root"]["weight"] == pytest.approx(0.75)
    assert objective_spec_digest(objective) == objective_spec_digest(dict(objective))
    validate_objective_spec(objective, require_canonical_public_names=True)


def test_objective_spec_rejects_legacy_gap_aliases_as_objective_terms() -> None:
    with pytest.raises(ValueError, match="legacy or non-canonical objective terms"):
        normalize_objective_spec(
            {
                "objective_family": "legacy",
                "terms": {
                    "calibration": {"weight": 0.4},
                    "oracle_recovery": {"metric": "legacy_name"},
                },
            }
        )


def test_oracle_state_objective_needs_only_nominal_lambda_contract() -> None:
    objective = objective_metadata(
        objective_family="markov_publication_bundle",
        local_law_estimator="oracle_state",
        local_law_weight=0.5,
        root_share=0.5,
        local_law_component_weights={LAW_ID_LEAF_PRESERVATION: 0.5},
    )

    validate_objective_spec(objective)
    assert objective["local_law_weight"] == pytest.approx(0.5)
    assert ("relia" + "bility") not in objective


def test_objective_spec_rejects_legacy_public_fields() -> None:
    for field in (
        "lambda_local_law",
        "lambda",
        "lambda_" + "effective",
        "lambda_" + "eff",
        "lambda_nominal",
        "root_weight",
        "task_weight",
        "task_objective_weight",
        "local_law_weights",
        "gap_weight",
        "oracle_gap_weight",
        "proxy_weights",
        "relia" + "bility",
    ):
        payload = {
            "objective_family": "root_only",
            "local_law_estimator": "none",
            "root_share": 1.0,
            "local_law_component_weights": {},
            field: 0.0 if field != "proxy_weights" else {},
        }
        with pytest.raises(ValueError, match=field):
            normalize_objective_spec(payload)


def test_objective_spec_preserves_zero_local_law_weight() -> None:
    objective = normalize_objective_spec(
        {
            "objective_family": "root_only",
            "local_law_estimator": "none",
            "local_law_weight": 0.0,
            "root_share": 1.0,
            "local_law_component_weights": {},
        }
    )

    assert objective["local_law_weight"] == pytest.approx(0.0)
    assert objective["root_share"] == pytest.approx(1.0)


def test_run_manifest_records_objective_digest_and_requires_objective_when_publication_ready() -> None:
    run = run_manifest_metadata(
        run_id="manifesto.objective",
        domain="manifesto_rile",
        role="fg_ladder_runner",
        backend="dspy",
        status="completed",
        objective=objective_metadata(
            objective_family="manifesto_alternating_ladder",
            local_law_estimator="proxy_only",
            root_share=1.0,
            local_law_component_weights={LAW_ID_LEAF_PRESERVATION: 1.0},
        ),
        audit_results={"ok": True},
        publication_ready=True,
    )

    assert run["objective"]["schema_version"] == OBJECTIVE_SCHEMA_VERSION
    assert run["objective_digest"] == objective_spec_digest(run["objective"])
    validate_run_manifest(run, require_publication_ready=True, require_objective=True)

    missing = RunManifest(
        run_id="missing.objective",
        domain="manifesto_rile",
        role="fg_ladder_runner",
        backend="dspy",
        status="completed",
        audit_results={"ok": True},
        publication_ready=True,
    ).to_dict()
    with pytest.raises(ValueError, match="requires ObjectiveSpec"):
        validate_run_manifest(missing, require_publication_ready=True)
