from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

import pytest

from src.diffusion.backends import DiffusionBatchResponse, DiffusionGeneration
from src.training.supervision.types import SupervisionDataset
from src.tree.auditor import SimpleScorer
from src.tree.treepo_stack import (
    OracleLaneSpec,
    TreePOContractSpec,
    TreePOLocalLawConfig,
    TreePOModelSpec,
    build_treepo_stack,
)
from src.tree.treepo_supervision import TreePOSupervisionSpec


class _QueueBackend:
    def __init__(self, queued_outputs: Sequence[Sequence[str]]) -> None:
        self.backend_name = "mock_backend"
        self._queued_outputs = [list(batch) for batch in queued_outputs]
        self.calls: List[Dict[str, Any]] = []

    def generate(
        self,
        texts: Sequence[str] | str,
        sampling_params: Optional[Mapping[str, Any]] = None,
        engine_options: Optional[Mapping[str, Any]] = None,
    ) -> DiffusionBatchResponse:
        prompts = [texts] if isinstance(texts, str) else list(texts)
        outputs = self._queued_outputs.pop(0)
        assert len(outputs) == len(prompts)
        self.calls.append(
            {
                "texts": prompts,
                "sampling_params": dict(sampling_params or {}),
                "engine_options": dict(engine_options or {}),
            }
        )
        return DiffusionBatchResponse(
            generations=[
                DiffusionGeneration(input_text=prompt, output_text=output)
                for prompt, output in zip(prompts, outputs)
            ],
            latency_seconds=0.01,
            request_payload={"text": prompts},
            raw_response={"text": outputs},
        )


def test_can_run_text_stack_without_oracle_when_verification_disabled(tmp_path: Path) -> None:
    backend = _QueueBackend([["leaf-1", "leaf-2", "leaf-3"], ["merge-12"], ["merge-root"]])
    contract = TreePOContractSpec(
        rubric="Objective.",
        local_law_config=TreePOLocalLawConfig(
            enable_l1=False,
            enable_l2=False,
            enable_l3=False,
            enable_substitution=False,
        ),
        oracle_lane=None,
        supervision_source=None,
    )

    stack = build_treepo_stack(
        TreePOModelSpec(kind="diffusion_backend", backend=backend, surface="generate"),
        contract,
    )

    result = stack.run_fixed_binary(
        ["alpha", "beta", "gamma"],
        document_id="doc_001",
        supervision=TreePOSupervisionSpec(
            mode="requests",
            doc_sample_probability=1.0,
            unit_selector="all",
            max_units=2,
            output_dir=str(tmp_path),
            random_seed=0,
        ),
    )

    assert result.tree.final_rendered == "merge-root"
    assert result.tree.root.audit.get("law_checks") is None
    meta = result.tree.metadata.get("treepo_supervision", {})
    assert meta.get("dataset_path")
    dataset_path = Path(str(meta["dataset_path"]))
    assert dataset_path.exists()
    dataset = SupervisionDataset.load(dataset_path)
    assert 0 < len(dataset.response_judgments) <= 2
    assert all(j.response_signal_value is None for j in dataset.response_judgments)


def test_supervision_label_now_calls_oracle_and_persists_labels(tmp_path: Path) -> None:
    backend = _QueueBackend([["leaf-1", "leaf-2"], ["merge-root"]])
    contract = TreePOContractSpec(
        rubric="Objective.",
        local_law_config=TreePOLocalLawConfig(enable_l1=False, enable_l2=False, enable_l3=False),
        oracle_lane=None,
        supervision_source=None,
    )
    stack = build_treepo_stack(
        TreePOModelSpec(kind="diffusion_backend", backend=backend, surface="generate"),
        contract,
    )
    result = stack.run_fixed_binary(
        ["alpha", "beta"],
        document_id="doc_002",
        supervision=TreePOSupervisionSpec(
            mode="label_now",
            doc_sample_probability=1.0,
            unit_selector="root",
            max_units=1,
            output_dir=str(tmp_path),
            random_seed=0,
        ),
        supervision_oracle=SimpleScorer(),
    )
    meta = result.tree.metadata.get("treepo_supervision", {})
    dataset_path = Path(str(meta["dataset_path"]))
    dataset = SupervisionDataset.load(dataset_path)
    assert len(dataset.response_judgments) == 1
    assert dataset.response_judgments[0].response_signal_value is not None
    assert 0.0 <= float(dataset.response_judgments[0].response_signal_value) <= 1.0


def test_supervision_probability_can_skip_and_records_metadata(tmp_path: Path) -> None:
    backend = _QueueBackend([["leaf-1"], ["merge-root"]])
    contract = TreePOContractSpec(
        rubric="Objective.",
        local_law_config=TreePOLocalLawConfig(enable_l1=False, enable_l2=False, enable_l3=False),
    )
    stack = build_treepo_stack(
        TreePOModelSpec(kind="diffusion_backend", backend=backend, surface="generate"),
        contract,
    )
    result = stack.run_fixed_binary(
        ["alpha"],
        document_id="doc_003",
        supervision=TreePOSupervisionSpec(
            mode="requests",
            doc_sample_probability=0.0,
            output_dir=str(tmp_path),
            random_seed=0,
        ),
    )
    meta = result.tree.metadata.get("treepo_supervision", {})
    assert meta.get("skipped") is True


def test_markov_supervision_can_label_root_without_online_oracle(tmp_path: Path) -> None:
    from src.diffusion.markov_toy import encode_markov_path

    stack = build_treepo_stack(
        TreePOModelSpec(kind="markov_toy_exact"),
        TreePOContractSpec(
            rubric="Markov objective.",
            oracle_lane=OracleLaneSpec(kind="markov_exact"),
        ),
    )
    leaf_spans = [["a", "a"], ["b", "b"], ["a"]]
    expected_root = encode_markov_path([token for chunk in leaf_spans for token in chunk]).changepoints

    result = stack.run_fixed_binary(
        leaf_spans,
        document_id="markov_doc_001",
        supervision=TreePOSupervisionSpec(
            mode="label_now",
            labeler_kind="markov_toy_changepoints",
            doc_sample_probability=1.0,
            unit_selector="root",
            max_units=1,
            output_dir=str(tmp_path),
            random_seed=0,
            response_signal_min=0.0,
            response_signal_max=10.0,
        ),
    )

    meta = result.tree.metadata.get("treepo_supervision", {})
    dataset_path = Path(str(meta["dataset_path"]))
    dataset = SupervisionDataset.load(dataset_path)
    assert len(dataset.response_judgments) == 1
    assert dataset.response_judgments[0].response_signal_value == pytest.approx(float(expected_root))


def test_treepo_supervision_records_doc_propensity_and_ipw_flag() -> None:
    from src.tree.state_tree import StateNode, StateTree
    from src.tree.treepo_supervision import build_supervision_dataset_from_state_tree

    root = StateNode(level=0, span="alpha", state="leaf", rendered="leaf")
    tree = StateTree(root=root, metadata={})
    spec = TreePOSupervisionSpec(
        mode="requests",
        doc_sample_probability=0.25,
        unit_selector="root",
        max_units=1,
        random_seed=0,
    )
    dataset = build_supervision_dataset_from_state_tree(tree, rubric="Objective.", spec=spec, document_id="doc_prop")
    assert len(dataset.response_judgments) == 1
    row = dataset.response_judgments[0]
    assert row.sampling.document_propensity == pytest.approx(0.25)
    assert row.sampling.supports_ipw_estimation is True


def test_treepo_supervision_level_weighted_sampling_sets_unit_propensity() -> None:
    import hashlib
    import random

    from src.stats.sampling import pps_inclusion_probabilities, systematic_pps_sample_indices
    from src.tree.state_tree import StateNode, StateTree
    from src.tree.treepo_supervision import build_supervision_dataset_from_state_tree

    left = StateNode(level=0, span="A", state="a", rendered="a")
    right = StateNode(level=0, span="B", state="b", rendered="b")
    root = StateNode(level=1, span="A B", state="ab", rendered="ab", left_child=left, right_child=right)
    left.parent = root
    right.parent = root
    tree = StateTree(root=root, metadata={})

    candidates = list(tree.traverse_preorder())
    weights = []
    max_level = max(int(node.level) for node in candidates)
    for node in candidates:
        weights.append((int(node.level) + 1) / float(max_level + 1))
    total = sum(weights)
    weights = [w / total for w in weights]
    inclusion = pps_inclusion_probabilities(weights, 1)

    payload = "0:doc_level:treepo_units"
    derived = int(hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16], 16)
    rng = random.Random(derived)
    expected_indices = systematic_pps_sample_indices(inclusion, 1, rng=rng)
    assert len(expected_indices) == 1
    expected_node = candidates[expected_indices[0]]
    expected_prob = float(inclusion[expected_indices[0]])

    spec = TreePOSupervisionSpec(
        mode="requests",
        doc_sample_probability=1.0,
        unit_selector="all",
        max_units=1,
        random_seed=0,
        sampling_strategy="level_weighted",
    )
    dataset = build_supervision_dataset_from_state_tree(tree, rubric="Objective.", spec=spec, document_id="doc_level")
    assert len(dataset.response_judgments) == 1
    row = dataset.response_judgments[0]
    assert row.response_id == expected_node.id
    assert row.sampling.unit_propensity == pytest.approx(expected_prob)


def test_label_supervision_dataset_can_label_subset_and_sets_label_propensity() -> None:
    from src.tree.auditor import SimpleScorer
    from src.tree.state_tree import StateNode, StateTree
    from src.tree.treepo_supervision import (
        TreePOSupervisionSpec,
        build_supervision_dataset_from_state_tree,
        label_supervision_dataset,
    )

    nodes = [
        StateNode(level=0, span=f"doc {idx}", state=f"s{idx}", rendered=f"s{idx}", metadata={"leaf_index": idx})
        for idx in range(10)
    ]
    # Single-node tree is enough; unit_selector="all" would only select root, so use a fake internal root.
    root = StateNode(level=1, span="root", state="root", rendered="root", left_child=nodes[0], right_child=nodes[1])
    nodes[0].parent = root
    nodes[1].parent = root
    tree = StateTree(root=root, metadata={})

    dataset = build_supervision_dataset_from_state_tree(
        tree,
        rubric="Objective.",
        spec=TreePOSupervisionSpec(
            mode="requests",
            doc_sample_probability=1.0,
            unit_selector="all",
            max_units=10,
            random_seed=0,
        ),
        document_id="doc_label_subset",
    )
    assert len(dataset.response_judgments) == 3  # root + 2 leaves in this tiny tree
    assert all(row.response_signal_value is None for row in dataset.response_judgments)

    label_supervision_dataset(
        dataset,
        oracle=SimpleScorer(),
        max_labels=1,
        random_seed=0,
    )

    labeled = [row for row in dataset.response_judgments if row.response_signal_value is not None]
    assert len(labeled) == 1
    assert labeled[0].sampling.label_propensity == pytest.approx(1.0 / 3.0)


def test_treepo_supervision_comparative_requests_can_be_labeled_later() -> None:
    from src.tree.state_tree import StateNode, StateTree
    from src.tree.treepo_supervision import TreePOSupervisionSpec, build_supervision_dataset_from_state_tree, label_supervision_dataset

    left = StateNode(level=0, span="alpha", state="a", rendered="sum(alpha)")
    right = StateNode(level=0, span="beta", state="b", rendered="sum(beta)")
    root = StateNode(level=1, span="alpha beta", state="ab", rendered="sum(alpha beta)", left_child=left, right_child=right)
    left.parent = root
    right.parent = root
    tree = StateTree(root=root, metadata={})

    dataset = build_supervision_dataset_from_state_tree(
        tree,
        rubric="Objective.",
        spec=TreePOSupervisionSpec(
            mode="requests",
            supervision_kind="comparative",
            doc_sample_probability=1.0,
            unit_selector="root",
            max_units=1,
            random_seed=0,
        ),
        document_id="doc_cmp",
    )
    assert len(dataset.response_judgments) == 0
    assert len(dataset.comparative_judgments) == 1
    record = dataset.comparative_judgments[0]
    assert len(record.candidates) == 2
    assert all(c.rank is None for c in record.candidates)

    label_supervision_dataset(dataset, oracle=SimpleScorer(), max_labels=1, random_seed=0)
    record2 = dataset.comparative_judgments[0]
    assert all(c.rank is not None for c in record2.candidates)
    assert record2.sampling.supports_ipw_estimation is True
