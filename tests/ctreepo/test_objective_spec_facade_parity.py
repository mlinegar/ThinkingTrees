"""Parity + centralization guard for `contracts.ObjectiveSpec`.

Part of "fully move the objective contract onto treepo" (master plan Phase 2/6).
The ThinkingTrees `contracts.ObjectiveSpec` keeps its public `ctreepo.objective.v1`
serialization shape and digest for back-compat, but the estimator vocabulary and
the root/local resolver are now sourced from `treepo.objective` (single source of
truth). These tests pin the serialization behavior (so any future schema flip is
deliberate) and assert the centralization invariant.
"""

from __future__ import annotations

import pytest

from src.ctreepo import contracts as C
from treepo import objective as T


# --- behavior characterization (must stay green across the facade refactor) ---


def test_objective_spec_serializes_with_treepo_schema() -> None:
    # The objective contract now serializes with the canonical treepo schema.
    spec = C.ObjectiveSpec(objective_family="root_only")
    payload = spec.to_dict()
    assert payload["schema_version"] == "treepo.objective.v1"
    assert C.OBJECTIVE_SCHEMA_VERSION == "treepo.objective.v1"
    assert set(payload) >= {
        "schema_version",
        "objective_family",
        "local_law_estimator",
        "local_law_weight",
        "root_share",
        "local_law_component_weights",
        "terms",
        "metadata",
    }
    # ctreepo shape does NOT expose the upstream-only convex flag.
    assert "allow_nonconvex_objective" not in payload
    assert payload["terms"]["root"]["metric"] == "root_loss"


def test_objective_spec_digest_is_self_consistent() -> None:
    spec = C.ObjectiveSpec(
        objective_family="local_law",
        local_law_estimator=C.LOCAL_LAW_ESTIMATOR_CORRECTED,
        local_law_weight=0.5,
        root_share=0.5,
        local_law_component_weights={C.LAW_ID_ON_RANGE_IDEMPOTENCE: 1.0},
    )
    payload = spec.to_dict()
    assert C.objective_spec_digest(payload) == C.objective_spec_digest(dict(payload))
    assert C.normalize_objective_spec(payload)["schema_version"] == "treepo.objective.v1"


def test_legacy_ctreepo_objective_payload_still_accepted_on_read() -> None:
    # Back-compat: historical ctreepo.objective.v1 payloads still validate.
    legacy = {
        "schema_version": "ctreepo.objective.v1",
        "objective_family": "root_only",
        "local_law_estimator": "none",
        "root_share": 1.0,
        "local_law_component_weights": {},
    }
    spec = C.validate_objective_spec(legacy)
    assert spec.schema_version == "ctreepo.objective.v1"


def test_default_objective_for_run_still_accepts_non_convex() -> None:
    # data-prep default is intentionally non-convex (root_share=0, no local law);
    # the facade must keep accepting it.
    prep = C.default_objective_for_run(role="split", backend="data_prep")
    assert prep["root_share"] == pytest.approx(0.0)
    run = C.default_objective_for_run(role="train", backend="neural")
    assert run["root_share"] == pytest.approx(1.0)


def test_legacy_law_aliases_still_rejected_on_public_api() -> None:
    # contracts public API stays strict: legacy c1/c2/c3 ids are not accepted.
    with pytest.raises(ValueError):
        C.canonical_law_component_weights({"c2": 1.0})


# --- centralization invariants (the "move to treepo" win) ---


def test_estimator_vocabulary_sourced_from_treepo() -> None:
    for name in (
        "LOCAL_LAW_ESTIMATOR_NONE",
        "LOCAL_LAW_ESTIMATOR_PROXY_ONLY",
        "LOCAL_LAW_ESTIMATOR_CORRECTED",
        "LOCAL_LAW_ESTIMATOR_ORACLE_STATE",
        "LOCAL_LAW_ESTIMATOR_ORACLE_EXACT",
        "LOCAL_LAW_ESTIMATOR_EXTERNAL_PASSTHROUGH",
    ):
        assert getattr(C, name) == getattr(T, name), f"{name} drifted from treepo"


def test_root_local_resolver_is_the_treepo_one() -> None:
    assert C.resolve_root_local_objective_weights is T.resolve_root_local_objective_weights
    resolved = C.resolve_root_local_objective_weights(
        local_law_weight=0.5,
        active_laws=("merge_preservation",),
    )
    assert resolved.root_share == pytest.approx(0.5)
    assert resolved.local_law_weight == pytest.approx(0.5)
