from __future__ import annotations

import argparse

import pytest

from src.training.run_pipeline import (
    apply_preference_collection_aliases,
    enforce_large_model_only_flags,
    should_collect_phase1_preferences,
)


def _args(*, enable_genrm: bool = False, optimize_judge: bool = False, tournament_of_tournaments: bool = False) -> argparse.Namespace:
    return argparse.Namespace(
        enable_genrm=enable_genrm,
        optimize_judge=optimize_judge,
        tournament_of_tournaments=tournament_of_tournaments,
    )


def test_enforce_large_model_only_flags_allows_modern_defaults() -> None:
    enforce_large_model_only_flags(_args())


@pytest.mark.parametrize(
    ("kwargs", "expected_flag"),
    [
        ({"enable_genrm": True}, "--enable-genrm"),
        ({"optimize_judge": True}, "--optimize-judge"),
        ({"tournament_of_tournaments": True}, "--tournament-of-tournaments"),
    ],
)
def test_enforce_large_model_only_flags_blocks_legacy_paths(kwargs: dict[str, bool], expected_flag: str) -> None:
    with pytest.raises(ValueError, match="local-law bootstrap .* no GenRM"):
        enforce_large_model_only_flags(_args(**kwargs))
    try:
        enforce_large_model_only_flags(_args(**kwargs))
    except ValueError as exc:
        assert expected_flag in str(exc)


def test_apply_preference_collection_aliases_maps_modern_cli_flags() -> None:
    args = argparse.Namespace(
        preference_init_samples=17,
        preference_init_candidates=5,
        preference_tree_concurrency=9,
        preference_sample_seed=123,
        preference_incremental_sampling=True,
        genrm_init_samples=8,
        genrm_init_candidates=4,
        genrm_tree_concurrency=None,
    )
    apply_preference_collection_aliases(args)
    assert args.genrm_init_samples == 17
    assert args.genrm_init_candidates == 5
    assert args.genrm_tree_concurrency == 9
    assert args.genrm_sample_seed == 123
    assert args.genrm_incremental_sampling is True


def test_apply_preference_collection_aliases_sets_defaults_when_unset() -> None:
    args = argparse.Namespace(
        preference_init_samples=None,
        preference_init_candidates=None,
        preference_tree_concurrency=None,
        preference_sample_seed=None,
        preference_incremental_sampling=None,
    )
    apply_preference_collection_aliases(args)
    assert args.genrm_sample_seed == 42
    assert args.genrm_incremental_sampling is False


def test_should_collect_phase1_preferences_requires_modern_training_need() -> None:
    args = argparse.Namespace(
        train_generator=False,
        enable_unified_training=False,
        train_comparison_module=False,
        interleaved_final_opt=False,
    )
    assert should_collect_phase1_preferences(args, interleaved_optimize=False) is False

    args.train_generator = True
    assert should_collect_phase1_preferences(args, interleaved_optimize=False) is True

    args.interleaved_final_opt = False
    assert should_collect_phase1_preferences(args, interleaved_optimize=True) is False

    args.interleaved_final_opt = True
    assert should_collect_phase1_preferences(args, interleaved_optimize=True) is True
