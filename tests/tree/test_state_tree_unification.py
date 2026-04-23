from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Sequence

import pytest

from src.core.engines import EngineSurface
from src.core.inference_engine import build_inference_engine
from src.tree.async_operator import AsyncFromInferenceEngine, MarkovToyOperator
from src.tree.state_tree import StateNode, StateTree
from src.tree.state_tree_runner import run_fixed_binary_state_tree
from src.tree.state_tree_verifiers import MarkovExactVerifier


@dataclass(frozen=True)
class _DummyState:
    x: int
    y: str = "ok"


class _Unknown:
    def __repr__(self) -> str:  # pragma: no cover - repr is what we want to see
        return "<unknown>"


def test_state_tree_to_dict_is_json_safe_for_dataclasses_tensors_and_unknown() -> None:
    try:
        import torch
    except Exception:  # pragma: no cover
        torch = None

    node = StateNode(
        level=0,
        span="leaf",
        state=_DummyState(x=3),
        rendered="dummy",
        metadata={
            "tensor": torch.tensor([1, 2, 3]) if torch is not None else "no_torch",
            "unknown": _Unknown(),
        },
    )
    tree = StateTree(root=node, metadata={"extra": _Unknown()})
    payload = tree.to_dict()

    assert payload["root_id"] == node.id
    encoded = payload["nodes"][node.id]
    assert encoded["state"] == {"x": 3, "y": "ok"}
    assert isinstance(encoded["metadata"]["unknown"], str)
    assert isinstance(payload["metadata"]["extra"], str)
    if torch is not None:
        tensor = encoded["metadata"]["tensor"]
        assert tensor["type"] == "torch.Tensor"
        assert tensor["shape"] == [3]


class _QueueOperator:
    def __init__(self, queued_outputs: Sequence[str]) -> None:
        self.name = "queue_operator"
        self._queue = list(queued_outputs)
        self.calls: List[Dict[str, Any]] = []

    async def aencode(self, span: str, **kwargs: Any) -> str:
        self.calls.append({"op": "encode", "span": span})
        return self._queue.pop(0)

    async def amerge(self, left_state: str, right_state: str, **kwargs: Any) -> str:
        self.calls.append({"op": "merge", "left": left_state, "right": right_state})
        return self._queue.pop(0)

    def combine(self, left_span: str, right_span: str, **_: Any) -> str:
        return f"{left_span}||{right_span}"

    async def adecode(self, state: str, **_: Any) -> str:
        return state

    def capability_report(self):
        from src.core.ops_checks import EvidenceStatus, LawCapabilityReport, LawKind, OperatorCapabilityReport

        return OperatorCapabilityReport(
            operator_name=self.name,
            evidence_status=EvidenceStatus.PROXY_ONLY,
            latent_mergeability_enforced=False,
            tree_nesting_supported=True,
            theorem_domain_decode_available=True,
            theorem_domain_reencode_available=True,
            leaf_law=LawCapabilityReport(
                law_kind=LawKind.L1_LEAF,
                available=True,
                evidence_status=EvidenceStatus.PROXY_ONLY,
                exact=False,
            ),
            merge_law=LawCapabilityReport(
                law_kind=LawKind.L2_MERGE,
                available=True,
                evidence_status=EvidenceStatus.PROXY_ONLY,
                exact=False,
            ),
            idempotence_law=LawCapabilityReport(
                law_kind=LawKind.L3_IDEMPOTENCE,
                available=True,
                evidence_status=EvidenceStatus.PROXY_ONLY,
                exact=False,
            ),
        )


def test_fixed_binary_state_tree_runner_matches_diffusion_prototype_shape() -> None:
    operator = _QueueOperator(
        [
            "leaf-1",
            "leaf-2",
            "leaf-3",
            "merge-12",
            "merge-root",
            "refine-1",
            "refine-2",
        ]
    )

    result = run_fixed_binary_state_tree(
        operator,  # type: ignore[arg-type]
        ["alpha", "beta", "gamma"],
        rubric="Keep exact theorem content.",
        refine_rounds=2,
    )

    assert result.tree.final_rendered == "refine-2"
    assert result.tree.node_count == 5
    assert len(result.operations) == 5
    assert result.operations[1].operation == "merge_level"
    assert result.operations[1].carried_node_ids
    assert result.operations[-1].round_index == 2


def test_state_tree_runner_populates_parent_child_labels() -> None:
    operator = _QueueOperator(
        [
            "leaf-1",
            "leaf-2",
            "leaf-3",
            "merge-12",
            "merge-root",
        ]
    )

    result = run_fixed_binary_state_tree(
        operator,  # type: ignore[arg-type]
        ["alpha", "beta", "gamma"],
        rubric="Objective.",
        refine_rounds=0,
    )

    root = result.tree.root
    assert root.metadata.get("child_side") == "root"
    assert root.metadata.get("range_label") == "0:2"
    assert root.metadata.get("leaf_start_index") == 0
    assert root.metadata.get("leaf_end_index") == 2

    nodes = list(result.tree.traverse_preorder())
    leaves = [node for node in nodes if node.is_leaf]
    assert {leaf.metadata.get("leaf_index") for leaf in leaves} == {0, 1, 2}
    for leaf in leaves:
        idx = int(leaf.metadata["leaf_index"])
        assert leaf.metadata.get("leaf_start_index") == idx
        assert leaf.metadata.get("leaf_end_index") == idx
        assert leaf.metadata.get("range_label") == f"{idx}:{idx}"
        assert str(leaf.metadata.get("parent_id") or "")
        assert leaf.metadata.get("child_side") in {"left", "right"}

    internal = [node for node in nodes if not node.is_leaf]
    merge_12 = [node for node in internal if node.metadata.get("range_label") == "0:1"]
    assert len(merge_12) == 1
    assert merge_12[0].metadata.get("child_side") == "left"
    assert str(merge_12[0].metadata.get("parent_id") or "")


def test_refine_prefers_operator_aresummarize_over_decode_encode() -> None:
    class _Operator:
        def __init__(self) -> None:
            self.name = "prefers_aresummarize"
            self.calls: List[str] = []

        async def aencode(self, span: str, **_: Any) -> str:
            self.calls.append("encode")
            return "leaf"

        async def amerge(self, left_state: str, right_state: str, **_: Any) -> str:
            self.calls.append("merge")
            return f"{left_state}+{right_state}"

        def combine(self, left_span: str, right_span: str, **_: Any) -> str:
            return f"{left_span}||{right_span}"

        async def adecode(self, state: str, **_: Any) -> str:
            self.calls.append("decode")
            return state

        async def aresummarize(self, state: str, **_: Any) -> str:
            self.calls.append("aresummarize")
            return f"refined({state})"

        def capability_report(self):
            from src.core.ops_checks import EvidenceStatus, LawCapabilityReport, LawKind, OperatorCapabilityReport

            return OperatorCapabilityReport(
                operator_name=self.name,
                evidence_status=EvidenceStatus.PROXY_ONLY,
                latent_mergeability_enforced=False,
                tree_nesting_supported=True,
                theorem_domain_decode_available=True,
                theorem_domain_reencode_available=True,
                leaf_law=LawCapabilityReport(
                    law_kind=LawKind.L1_LEAF,
                    available=True,
                    evidence_status=EvidenceStatus.PROXY_ONLY,
                    exact=False,
                ),
                merge_law=LawCapabilityReport(
                    law_kind=LawKind.L2_MERGE,
                    available=True,
                    evidence_status=EvidenceStatus.PROXY_ONLY,
                    exact=False,
                ),
                idempotence_law=LawCapabilityReport(
                    law_kind=LawKind.L3_IDEMPOTENCE,
                    available=True,
                    evidence_status=EvidenceStatus.PROXY_ONLY,
                    exact=False,
                ),
            )

    operator = _Operator()
    result = run_fixed_binary_state_tree(
        operator,  # type: ignore[arg-type]
        ["alpha"],
        refine_rounds=1,
    )
    assert result.tree.final_rendered == "refined(leaf)"
    assert "aresummarize" in operator.calls
    assert "decode" not in operator.calls


def test_refine_falls_back_to_decode_encode_when_aresummarize_unavailable() -> None:
    class _Operator:
        def __init__(self) -> None:
            self.name = "fallback_decode_encode"
            self.calls: List[str] = []
            self._encode_outputs = ["leaf", "reencoded"]

        async def aencode(self, span: str, **_: Any) -> str:
            self.calls.append("encode")
            return self._encode_outputs.pop(0)

        async def amerge(self, left_state: str, right_state: str, **_: Any) -> str:
            self.calls.append("merge")
            return f"{left_state}+{right_state}"

        def combine(self, left_span: str, right_span: str, **_: Any) -> str:
            return f"{left_span}||{right_span}"

        async def adecode(self, state: str, **_: Any) -> str:
            self.calls.append("decode")
            return state

        async def aresummarize(self, state: str, **_: Any) -> str:
            self.calls.append("aresummarize")
            raise NotImplementedError("no dedicated resummary")

        def capability_report(self):
            from src.core.ops_checks import EvidenceStatus, LawCapabilityReport, LawKind, OperatorCapabilityReport

            return OperatorCapabilityReport(
                operator_name=self.name,
                evidence_status=EvidenceStatus.PROXY_ONLY,
                latent_mergeability_enforced=False,
                tree_nesting_supported=True,
                theorem_domain_decode_available=True,
                theorem_domain_reencode_available=True,
                leaf_law=LawCapabilityReport(
                    law_kind=LawKind.L1_LEAF,
                    available=True,
                    evidence_status=EvidenceStatus.PROXY_ONLY,
                    exact=False,
                ),
                merge_law=LawCapabilityReport(
                    law_kind=LawKind.L2_MERGE,
                    available=True,
                    evidence_status=EvidenceStatus.PROXY_ONLY,
                    exact=False,
                ),
                idempotence_law=LawCapabilityReport(
                    law_kind=LawKind.L3_IDEMPOTENCE,
                    available=True,
                    evidence_status=EvidenceStatus.PROXY_ONLY,
                    exact=False,
                ),
            )

    operator = _Operator()
    result = run_fixed_binary_state_tree(
        operator,  # type: ignore[arg-type]
        ["alpha"],
        refine_rounds=1,
    )
    assert result.tree.final_rendered == "reencoded"
    assert operator.calls.count("aresummarize") == 1
    assert "decode" in operator.calls
    assert operator.calls.count("encode") == 2


def test_refine_skips_when_no_aresummarize_and_no_decode_encode() -> None:
    class _Operator:
        name = "no_refine"

        async def aencode(self, span: str, **_: Any) -> str:
            return "leaf"

        async def amerge(self, left_state: str, right_state: str, **_: Any) -> str:
            return f"{left_state}+{right_state}"

        def combine(self, left_span: str, right_span: str, **_: Any) -> str:
            return f"{left_span}||{right_span}"

        async def adecode(self, state: str, **_: Any) -> str:
            raise NotImplementedError("no decode")

        def capability_report(self):
            from src.core.ops_checks import EvidenceStatus, LawCapabilityReport, LawKind, OperatorCapabilityReport

            return OperatorCapabilityReport(
                operator_name=self.name,
                evidence_status=EvidenceStatus.PROXY_ONLY,
                latent_mergeability_enforced=False,
                tree_nesting_supported=True,
                theorem_domain_decode_available=False,
                theorem_domain_reencode_available=False,
                leaf_law=LawCapabilityReport(
                    law_kind=LawKind.L1_LEAF,
                    available=True,
                    evidence_status=EvidenceStatus.PROXY_ONLY,
                    exact=False,
                ),
                merge_law=LawCapabilityReport(
                    law_kind=LawKind.L2_MERGE,
                    available=True,
                    evidence_status=EvidenceStatus.PROXY_ONLY,
                    exact=False,
                ),
                idempotence_law=LawCapabilityReport(
                    law_kind=LawKind.L3_IDEMPOTENCE,
                    available=False,
                    evidence_status=EvidenceStatus.PROXY_ONLY,
                    exact=False,
                ),
            )

    result = run_fixed_binary_state_tree(
        _Operator(),  # type: ignore[arg-type]
        ["alpha"],
        refine_rounds=2,
    )
    assert result.tree.final_rendered == "leaf"
    assert result.tree.metadata["refine_skipped"] is True
    assert result.tree.metadata["refine_skip_reason"] == "operator_missing_aresummarize_and_decode_encode"
    assert all(op.operation != "refine_round" for op in result.operations)


def test_markov_operator_unified_runner_matches_exact_root_state_and_verifier() -> None:
    operator = MarkovToyOperator()
    verifier = MarkovExactVerifier()

    # Two chunks of a path.
    leaf_spans = [["A", "A"], ["B", "B"]]
    result = run_fixed_binary_state_tree(
        operator,  # type: ignore[arg-type]
        leaf_spans,
        refine_rounds=1,  # skipped (no decode/L3)
        verifiers=[verifier],
    )

    root_state = result.tree.root.state
    assert root_state is not None
    assert getattr(root_state, "changepoints") == 1

    audit = result.tree.root.audit.get("law_checks", {}).get(verifier.name, {})
    assert audit["l2_merge"]["passed"] is True


class _FakeResponse:
    def __init__(self, payload: Any) -> None:
        self._payload = payload

    def raise_for_status(self) -> None:
        return None

    def json(self) -> Any:
        return self._payload


class _FakeSession:
    def __init__(self, payload: Any) -> None:
        self.payload = payload
        self.calls: List[Dict[str, Any]] = []

    def post(self, url: str, json: Dict[str, Any], timeout: float) -> _FakeResponse:
        self.calls.append({"url": url, "json": json, "timeout": timeout})
        return _FakeResponse(self.payload)


@pytest.mark.anyio
async def test_async_from_inference_engine_routes_chat_and_diffusion_surfaces() -> None:
    # Chat surface (mocked).
    chat_engine = build_inference_engine(
        "sglang",
        surface=EngineSurface.CHAT_OPENAI,
        base_url="http://localhost:30000/v1",
        model="default",
        mock=True,
    )
    seen_chat: List[Any] = []
    orig_chat = chat_engine.aexecute

    async def wrapped_chat(request):
        seen_chat.append(request)
        return await orig_chat(request)

    chat_engine.aexecute = wrapped_chat  # type: ignore[assignment]

    chat_op = AsyncFromInferenceEngine(chat_engine)
    _ = await chat_op.aencode("hello", rubric="rubric")
    assert seen_chat[0].surface is EngineSurface.CHAT_OPENAI

    # Diffusion surface (fake HTTP session).
    session = _FakeSession({"text": "ok"})
    diff_engine = build_inference_engine(
        "sglang",
        surface=EngineSurface.DIFFUSION_GENERATE,
        base_url="http://localhost:30000",
        model="default",
        session=session,
    )
    seen_diff: List[Any] = []
    orig_diff = diff_engine.aexecute

    async def wrapped_diff(request):
        seen_diff.append(request)
        return await orig_diff(request)

    diff_engine.aexecute = wrapped_diff  # type: ignore[assignment]

    diff_op = AsyncFromInferenceEngine(diff_engine)
    _ = await diff_op.aencode("hello", rubric="rubric")
    assert seen_diff[0].surface is EngineSurface.DIFFUSION_GENERATE
