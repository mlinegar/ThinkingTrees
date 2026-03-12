from __future__ import annotations

from argparse import Namespace

from src.training.preference.oracle_reward import (
    create_local_law_summary_reward_func,
    create_oracle_alignment_reward_func,
)
from src.training.run_pipeline import build_default_grpo_reward_funcs


def test_oracle_alignment_reward_uses_reference_score() -> None:
    def oracle_predict(text: str) -> float:
        text = str(text).strip().lower()
        if "left" in text:
            return 0.10
        if "right" in text:
            return 0.90
        return 0.50

    reward = create_oracle_alignment_reward_func(
        oracle_predict,
        error_scale=1.0,
        neutral_reward=0.5,
        min_completion_chars=0,
    )
    scores = reward(
        completions=["left policy summary", "right policy summary"],
        prompts=["p1", "p2"],
        reference_score=[0.10, 0.90],
    )
    assert scores[0] == 1.0
    assert scores[1] == 1.0


def test_oracle_alignment_reward_falls_back_to_original_text() -> None:
    def oracle_predict(text: str) -> float:
        rendered = str(text).strip().lower()
        if "doc-left" in rendered or "left" in rendered:
            return 0.20
        if "doc-right" in rendered or "right" in rendered:
            return 0.80
        return 0.50

    reward = create_oracle_alignment_reward_func(
        oracle_predict,
        error_scale=1.0,
        neutral_reward=0.5,
        min_completion_chars=0,
    )
    scores = reward(
        completions=["left summary", "right summary"],
        prompts=["p1", "p2"],
        original_text=["doc-left", "doc-right"],
    )
    assert scores[0] == 1.0
    assert scores[1] == 1.0


def test_build_default_grpo_reward_funcs_returns_callable_and_metadata() -> None:
    class DummyTask:
        @staticmethod
        def create_oracle_scorer():
            return lambda text: 0.5 if text else 0.0

    args = Namespace(
        grpo_reward_error_scale=1.0,
        grpo_reward_neutral=0.5,
        grpo_reward_min_completion_chars=0,
        grpo_reward_short_penalty=0.1,
        grpo_reward_cache_size=32,
    )
    reward_funcs, metadata = build_default_grpo_reward_funcs(task=DummyTask(), args=args)
    assert len(reward_funcs) == 1
    reward_values = reward_funcs[0](
        completions=["ok summary"],
        prompts=["prompt"],
        reference_score=[0.5],
    )
    assert reward_values == [1.0]
    assert metadata["reward_backend"] == "task_oracle_scorer"
    assert metadata["reward_mode"] == "oracle_alignment"


def test_local_law_reward_prefers_on_target_and_stable_completion() -> None:
    def oracle_predict(text: str) -> float:
        rendered = str(text).strip().lower()
        if "src_left" in rendered:
            return -20.0
        if "off_target" in rendered:
            return 35.0
        if "good_completion" in rendered:
            return -18.0
        return -20.0

    reward = create_local_law_summary_reward_func(
        oracle_predict,
        c1_threshold_raw=10.0,
        c2_threshold_raw=6.0,
        neutral_raw=0.0,
        min_completion_chars=0,
    )
    values = reward(
        completions=["good_completion", "off_target"],
        prompts=["p1", "p2"],
        reference_score=[-20.0, -20.0],
        input_text=["src_left", "src_left"],
        hop=[2, 2],
    )
    assert values[0] > values[1]
    assert values[0] > 0.7
    assert values[1] < 0.4


def test_local_law_reward_penalizes_side_flip() -> None:
    def oracle_predict(text: str) -> float:
        rendered = str(text).strip().lower()
        if "source_left" in rendered:
            return -12.0
        if "flipped_right" in rendered:
            return 8.0
        return -12.0

    reward = create_local_law_summary_reward_func(
        oracle_predict,
        c1_threshold_raw=20.0,
        c2_threshold_raw=20.0,
        same_side_weight=0.5,
        c1_weight=0.3,
        c2_weight=0.2,
        neutral_raw=0.0,
        min_completion_chars=0,
    )
    values = reward(
        completions=["source_left", "flipped_right"],
        prompts=["p1", "p2"],
        reference_score=[-12.0, -12.0],
        input_text=["source_left", "source_left"],
        hop=[2, 2],
    )
    assert values[0] > values[1]


def test_local_law_reward_parse_failure_gets_min_objective_reward() -> None:
    def oracle_predict(text: str) -> float:
        rendered = str(text).strip().lower()
        if "bad_completion" in rendered:
            return float("nan")
        if "source_text" in rendered:
            return -20.0
        return -18.0

    reward = create_local_law_summary_reward_func(
        oracle_predict,
        parse_failure_reward=0.0,
        neutral_reward=0.25,
        min_completion_chars=0,
    )
    values = reward(
        completions=["bad_completion", "good_completion"],
        prompts=["p1", "p2"],
        reference_score=[-20.0, -20.0],
        input_text=["source_text", "source_text"],
        hop=[2, 2],
    )
    assert values[0] == 0.0
    assert values[1] > values[0]
