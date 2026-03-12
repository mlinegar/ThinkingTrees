"""Tests for output parsing utilities."""

from src.core.output_parser import (
    NormalizedOutputAccessor,
    find_matching_key,
    get_field,
    normalize_output_keys,
)


class FakePrediction:
    """Minimal stand-in for DSPy Prediction behavior."""

    def __init__(self, store):
        self._store = dict(store)
        self._completions = None

    def keys(self):
        return self._store.keys()

    def __getitem__(self, key):
        return self._store[key]

    def __getattr__(self, name):
        try:
            return self._store[name]
        except KeyError as exc:
            raise AttributeError(name) from exc


def test_find_matching_key_prefers_mapping_keys_over_internal_dict():
    pred = FakePrediction({"score": -42.0, "reasoning": "test"})
    assert find_matching_key(pred, "score") == "score"


def test_get_field_reads_prediction_style_store():
    pred = FakePrediction({"score": -15.5})
    assert get_field(pred, "score", default=0.0) == -15.5


def test_accessor_reads_prediction_style_score():
    pred = FakePrediction({"score": 12.0})
    accessor = NormalizedOutputAccessor(pred)
    assert accessor.get("score", 0.0) == 12.0


def test_normalize_output_keys_with_prediction_style_object():
    pred = FakePrediction({"Score": 7.5, "Reasoning": "ok"})
    normalized = normalize_output_keys(pred, expected_fields=["score", "reasoning"])
    assert normalized == {"score": 7.5, "reasoning": "ok"}

