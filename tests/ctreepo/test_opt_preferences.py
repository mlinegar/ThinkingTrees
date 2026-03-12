from __future__ import annotations

from src.ctreepo.opt.preferences import (
    derive_preference_from_losses,
    derive_preference_from_scores,
    derive_preference_from_utilities,
)


def test_derive_preference_from_losses() -> None:
    out = derive_preference_from_losses(1.0, 2.0)
    assert out.preferred == "A"
    assert 0.5 < out.confidence <= 0.99

    out = derive_preference_from_losses(2.0, 1.0)
    assert out.preferred == "B"

    out = derive_preference_from_losses(1.0, 1.01, tie_margin=0.1)
    assert out.preferred == "tie"
    assert out.confidence == 0.5


def test_derive_preference_from_utilities() -> None:
    out = derive_preference_from_utilities(3.0, 1.0)
    assert out.preferred == "A"

    out = derive_preference_from_utilities(1.0, 3.0)
    assert out.preferred == "B"

    out = derive_preference_from_utilities(1.0, 1.01, tie_margin=0.1)
    assert out.preferred == "tie"


def test_derive_preference_from_scores() -> None:
    out = derive_preference_from_scores(reference=0.0, score_a=0.1, score_b=0.5)
    assert out.preferred == "A"
    assert out.loss_a is not None and out.loss_b is not None
    assert out.loss_a < out.loss_b

