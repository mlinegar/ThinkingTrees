from __future__ import annotations

from src.runtime.adapters.ruler import _string_match_all, _string_match_part
from src.runtime.contracts import RunPhaseSpec, RunSpec, expand_units
from src.runtime.loop import _extract_context_and_question, _run_problem
from src.runtime.contracts import ModelResponse, NodeContract, ProblemSpec, RunUnit, RuntimeConfig
from src.runtime.memory import TokenCounter, chunk_text_tokens
from src.runtime.repair import SimpleRepairPolicy
from src.runtime.verifier import DeterministicVerifier


def test_expand_units_runtime_grid_cartesian_product():
    spec = RunSpec(
        run_id="r1",
        created_utc="now",
        output_dir="outputs/evals",
        benchmark={"name": "ruler_synthetic"},
        model={},
        runtime_defaults={},
        phases=[
            RunPhaseSpec(
                phase_id="P0",
                tasks=["t1", "t2"],
                lengths=[10],
                seeds=[0, 1],
                num_samples=5,
                split="validation",
                modes=["m1", "m2"],
                runtime_grid={"leaf_memory_tokens": [64, 128], "merge_memory_tokens": [256]},
            )
        ],
    )

    units = expand_units(spec)
    # tasks(2) * lengths(1) * seeds(2) * modes(2) * grid(2*1)=16
    assert len(units) == 16
    assert any(u.runtime_overrides.get("leaf_memory_tokens") == 64 for u in units)
    assert any(u.runtime_overrides.get("leaf_memory_tokens") == 128 for u in units)


def test_ruler_string_match_all():
    preds = ["A B C", "foo bar"]
    refs = [["a", "b"], ["bar"]]
    assert _string_match_all(preds, refs) == 100.0


def test_ruler_string_match_part():
    preds = ["the answer is Paris", "missing"]
    refs = [["paris"], ["rome", "athens"]]
    assert _string_match_part(preds, refs) == 50.0


def test_extract_context_and_question_niah():
    p = ProblemSpec(
        problem_id="p1",
        input_text="Intro...\nCONTEXT\nWhat are all the things?",
        metadata={"ruler_task_type": "niah"},
    )
    ctx, q = _extract_context_and_question(p)
    assert "CONTEXT" in ctx
    assert q.startswith("What ")


def test_runtime_full_uses_leaf_and_merge_budgets_for_max_tokens():
    class DummyAdapter:
        def build_contract(self, problem: ProblemSpec) -> NodeContract:
            return NodeContract(objective="dummy", max_input_tokens=8192, max_output_tokens=9999)

    class SpyBackbone:
        def __init__(self):
            self.max_tokens_calls: list[int] = []

        def generate(self, messages, *, max_tokens: int, **kwargs) -> ModelResponse:  # type: ignore[no-untyped-def]
            self.max_tokens_calls.append(int(max_tokens))
            return ModelResponse(text="ok", model_id="spy", prompt_tokens=0, completion_tokens=0, latency_ms=0.0)

    counter = TokenCounter()
    verifier = DeterministicVerifier(counter)
    repair = SimpleRepairPolicy()

    runtime = RuntimeConfig(
        mode="runtime_full",
        cap_tokens=512,
        safety_tokens=0,
        max_output_tokens=30,
        chunk_tokens=10,
        overlap_tokens=0,
        leaf_memory_tokens=10,
        merge_memory_tokens=20,
        verifier_enabled=True,
        repair_enabled=True,
    )

    unit = RunUnit(
        run_id="r1",
        unit_id="u000001",
        phase_id="P0",
        benchmark="b",
        task_id="t",
        split="validation",
        max_seq_length=10,
        seed=0,
        num_samples=1,
        mode="runtime_full",
    )

    problem = ProblemSpec(problem_id="p1", input_text=("x " * 100).strip(), query="Q?", references=["ref"])
    chunks = chunk_text_tokens(
        problem.input_text, counter=counter, chunk_tokens=runtime.chunk_tokens, overlap_tokens=runtime.overlap_tokens
    )
    assert len(chunks) >= 2

    backbone = SpyBackbone()
    pred, cost, events = _run_problem(
        problem=problem,
        adapter=DummyAdapter(),
        unit=unit,
        runtime=runtime,
        backbone=backbone,  # type: ignore[arg-type]
        counter=counter,
        verifier=verifier,
        repair=repair,
    )

    assert pred == "ok"
    assert cost.get("n_calls", 0) == len(backbone.max_tokens_calls)
    assert backbone.max_tokens_calls[-1] == runtime.max_output_tokens  # answer step
    assert backbone.max_tokens_calls[: len(chunks)] == [runtime.leaf_memory_tokens] * len(chunks)
    assert backbone.max_tokens_calls[len(chunks) : -1] == [runtime.merge_memory_tokens] * (len(chunks) - 1)
