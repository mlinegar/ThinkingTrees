import math

from src.tasks.manifesto.corpus_metrics import compute_corpus_pearson_r


def test_constant_predictions_do_not_become_perfect_pearson():
    report = compute_corpus_pearson_r(
        pred=[2.0, 2.0, 2.0, 2.0],
        true=[1.0, 2.0, 3.0, 4.0],
    )

    assert report.pearson_r == 0.0
    assert report.pearson_ci_low == 0.0
    assert report.pearson_ci_high == 0.0
    assert report.pearson_defined is False
    assert report.undefined_reason == "constant predictions"


def test_constant_targets_do_not_become_perfect_pearson():
    report = compute_corpus_pearson_r(
        pred=[1.0, 2.0, 3.0, 4.0],
        true=[2.0, 2.0, 2.0, 2.0],
    )

    assert report.pearson_r == 0.0
    assert report.pearson_defined is False
    assert report.undefined_reason == "constant targets"


def test_tied_spearman_uses_average_ranks():
    report = compute_corpus_pearson_r(
        pred=[1.0, 1.0, 2.0, 3.0],
        true=[1.0, 2.0, 2.0, 3.0],
    )

    assert report.pearson_defined is True
    assert report.spearman_defined is True
    assert math.isclose(report.spearman_r, 0.8333333333333333)
