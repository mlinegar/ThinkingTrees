from __future__ import annotations

from typing import Dict, Tuple

from src.core.protocols import format_merge_input
from src.tree.auditor import AuditConfig, Auditor, SamplingStrategy, SimpleScorer
from src.tree.state_tree import StateNode, StateTree, state_tree_to_text_tree
from src.tree.state_tree_auditor import StateTreeAuditor


def test_state_tree_auditor_matches_legacy_auditor_checks() -> None:
    left = StateNode(level=0, span="alpha beta", state="s1", rendered="alpha beta")
    right = StateNode(level=0, span="gamma delta", state="s2", rendered="gamma delta")
    root_span = format_merge_input(str(left.span), str(right.span))
    root = StateNode(level=1, span=root_span, state="root", rendered=root_span, left_child=left, right_child=right)
    left.parent = root
    right.parent = root
    tree = StateTree(root=root, metadata={"document_id": "doc_state_tree"})

    cfg = AuditConfig(
        sample_budget=10,
        sampling_strategy=SamplingStrategy.RANDOM,
        sampling_probability=1.0,
        discrepancy_threshold=0.1,
        audit_leaves=True,
        audit_internal=True,
        audit_idempotence=False,
        audit_substitution=False,
        random_seed=0,
    )

    oracle = SimpleScorer()

    legacy_tree = state_tree_to_text_tree(tree, rubric="Objective.", metadata=None)
    legacy_report = Auditor(oracle, config=cfg).audit_tree(legacy_tree)

    state_report = StateTreeAuditor(oracle, config=cfg).audit_tree(tree, rubric="Objective.")

    def index(report) -> Dict[Tuple[str, str], float]:
        out: Dict[Tuple[str, str], float] = {}
        for check in list(getattr(report, "checks", []) or []):
            out[(str(check.node_id), str(check.check_type))] = float(check.discrepancy_score)
        return out

    assert index(state_report) == index(legacy_report)

