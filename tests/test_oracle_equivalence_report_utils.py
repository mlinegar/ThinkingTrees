from src.ctreepo.sim.cli.report import identifiable_zero_publication_clean as publication_clean_report


def _load_module():
    return publication_clean_report


def test_apply_exact_fixed_slice():
    mod = _load_module()
    rows = [
        {"train_docs": 12000, "q": 0.1},
        {"train_docs": 12000, "q": 0.2},
        {"train_docs": 4096, "q": 0.1},
    ]
    got = mod._apply_exact_fixed_slice(rows, key="train_docs", value=12000)
    assert len(got) == 2
    assert all(int(r["train_docs"]) == 12000 for r in got)


def test_normalized_gap_anchor_points():
    mod = _load_module()
    baseline = 2.0
    ceiling = 0.5
    assert abs(mod._normalized_gap(baseline, baseline, ceiling) - 1.0) <= 1e-12
    assert abs(mod._normalized_gap(ceiling, baseline, ceiling) - 0.0) <= 1e-12


def test_normalization_validity_and_anchors():
    mod = _load_module()
    # Valid lane: baseline > ceiling.
    vals = [2.0, 1.0, 0.5]
    out, valid, den = mod._normalize_series(vals, baseline=2.0, ceiling=0.5, eps_den=1e-12)
    assert valid is True
    assert abs(den - 1.5) <= 1e-12
    assert abs(out[0] - 1.0) <= 1e-12
    assert abs(out[-1] - 0.0) <= 1e-12

    # Invalid lane: denominator <= eps.
    out_bad, valid_bad, den_bad = mod._normalize_series([0.1, 0.1], baseline=0.1, ceiling=0.1, eps_den=1e-12)
    assert valid_bad is False
    assert abs(den_bad) <= 1e-12
    assert len(out_bad) == 2
    assert all(not (x == x) for x in out_bad)  # NaNs


def test_norm_display_na_for_invalid_lane():
    mod = _load_module()
    assert mod._fmt_norm_display(0.33, valid=False) == "N/A"
    assert mod._fmt_norm_display(0.33, valid=True) != "N/A"


def test_frontier_is_monotone_nonincreasing():
    mod = _load_module()
    points = [(0.2, 0.8), (0.1, 1.0), (0.2, 0.6), (0.4, 0.7), (0.5, 0.3)]
    f = mod._frontier_from_points(points)
    ys = f["best_error"]
    assert len(ys) > 0
    assert all(float(ys[i + 1]) <= float(ys[i]) + 1e-12 for i in range(len(ys) - 1))
