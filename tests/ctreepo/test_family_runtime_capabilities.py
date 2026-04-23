from __future__ import annotations

from src.ctreepo.sim.core.markov_alignment_spec import markov_alignment_spec
from src.ctreepo.sim.learning_problem import (
    build_local_law_learning_problem,
    build_local_law_runtime_capability,
)
from src.ctreepo.sim.local_law_learnability import (
    LocalLawPolicyEvaluation,
    LocalLawRunSummary,
    PolicyRole,
    SupportBudgetSummary,
)


def _summary(*, family: str, exact_family: str = "") -> LocalLawRunSummary:
    metadata = {"exact_family": exact_family} if exact_family else {}
    return LocalLawRunSummary(
        family=family,
        dgp="demo",
        oracle_name="oracle_g",
        study_role="diagnostics_and_learned",
        split_ids={"train": "train", "val": "val", "test": "test"},
        support_budget=SupportBudgetSummary(
            train_docs=8,
            val_docs=4,
            test_docs=4,
            leaf_query_rate=1.0,
            internal_query_rate=1.0,
            root_query_rate=1.0,
        ),
        selection={"selected_candidate": exact_family or "learned_g"},
        policies={
            "learned_g": LocalLawPolicyEvaluation(
                name="learned_g",
                role=PolicyRole.CANDIDATE_G,
                split_metrics={"val": {"objective": {"task_weight": 1.0}}},
            ),
        },
        counterexamples=[],
        thresholds={"c1_tau": 0.2, "c2_tau": 0.2, "c3_tau": 0.2},
        suite_role="support_scaling",
        metadata=metadata,
    )


def test_markov_exact_runtime_capability_marks_symbolic_lane_claim_bearing() -> None:
    problem = build_local_law_learning_problem(_summary(family="markov_ops_count", exact_family="exact"))
    runtime = dict(build_local_law_runtime_capability(_summary(family="markov_ops_count", exact_family="exact")))

    assert "runtime_capabilities" not in problem
    assert runtime["claim_status_by_surface"]["symbolic_exact"] == "claim_bearing"
    assert runtime["claim_status_by_surface"]["chat_openai"] == "research_only"
    assert runtime["claim_status_by_surface"]["diffusion_generate"] == "infrastructure_only"
    assert runtime["recommended_engines"][0] == "symbolic_local"


def test_general_family_runtime_capability_is_conservative_by_default() -> None:
    problem = build_local_law_learning_problem(_summary(family="tree_relevant_lda_local_law"))
    runtime = dict(build_local_law_runtime_capability(_summary(family="tree_relevant_lda_local_law")))

    assert "runtime_capabilities" not in problem
    assert runtime["claim_status_by_surface"]["chat_openai"] == "research_only"
    assert runtime["claim_status_by_surface"]["diffusion_generate"] == "infrastructure_only"
    assert runtime["claim_status_by_surface"]["symbolic_exact"] == "not_applicable"


def test_markov_alignment_spec_publishes_runtime_capabilities_per_surface() -> None:
    spec = markov_alignment_spec()
    surfaces = {surface["name"]: surface for surface in spec["surfaces"]}

    observed = dict(surfaces["markov_observed_token"]["runtime_capabilities"])
    ladder = dict(surfaces["markov_full_doc_anchor_ladder"]["runtime_capabilities"])

    assert observed["claim_status_by_surface"]["symbolic_exact"] == "claim_bearing"
    assert ladder["claim_status_by_surface"]["chat_openai"] == "research_only"
