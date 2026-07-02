from __future__ import annotations

import math

import pytest

from src.tasks.manifesto.rile_reconstruction import pearson_or_nan


def test_pearson_or_nan_matches_reconstruction_script_contract() -> None:
    assert pearson_or_nan([1.0, 2.0, 3.0], [1.0, 2.0, 3.0]) == pytest.approx(1.0)
    assert pearson_or_nan([1.0, 2.0, 3.0], [3.0, 2.0, 1.0]) == pytest.approx(-1.0)
    assert math.isnan(pearson_or_nan([1.0], [1.0]))
    assert math.isnan(pearson_or_nan([1.0, 1.0], [1.0, 2.0]))
