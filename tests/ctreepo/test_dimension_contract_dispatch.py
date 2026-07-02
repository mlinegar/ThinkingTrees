"""Pinning tests for the generalized state/summary 2x rule and helpers.

Covers ``check_state_summary_invariant`` and its embedding/sketch
specializations, plus the additive ``BundleAwareFamilyRuntime`` helpers in
``src.ctreepo.alternating``.

These tests are independent of any specific family runtime; they exercise the
contract surface itself.
"""

from __future__ import annotations

import pytest

from src.ctreepo.alternating import (
    BundleAwareFamilyRuntime,
    INIT_SPEC_RAW_PREFIXES,
    InitSpec,
    family_default_f,
    family_default_g,
    family_expected_bundle,
    family_resolve_init,
    family_share_state_axes,
    family_supported_inits,
    parse_init_spec,
)
from src.ctreepo.fg_arity import (
    StateSummaryReport,
    check_state_summary_invariant,
    check_two_child_embedding_budget,
    check_two_child_sketch_budget,
    state_summary_report,
)


# ---------------------------------------------------------------------------
# state/summary invariant
# ---------------------------------------------------------------------------


class TestStateSummaryInvariant:
    def test_embedding_strict_passes_when_state_dim_doubles_summary(self):
        check_state_summary_invariant(
            family_name="fno",
            state_kind="embedding",
            state_dim=1536,
            summary_dim=768,
            g_in_dim=1536,
            g_out_dim=1536,
        )

    def test_embedding_state_dim_below_2x_summary_raises(self):
        with pytest.raises(RuntimeError) as exc:
            check_state_summary_invariant(
                family_name="fno",
                state_kind="embedding",
                state_dim=768,
                summary_dim=768,
                g_in_dim=1536,
                g_out_dim=1536,
            )
        assert "state_dim=768 < 2 * summary_dim = 1536" in str(exc.value)

    def test_embedding_g_out_below_2x_raises(self):
        with pytest.raises(RuntimeError) as exc:
            check_state_summary_invariant(
                family_name="fno",
                state_kind="embedding",
                state_dim=1536,
                summary_dim=768,
                g_in_dim=1536,
                g_out_dim=768,
            )
        assert "g_out_dim=768 < 2 * summary_dim = 1536" in str(exc.value)

    def test_sketch_state_strict_passes(self):
        check_state_summary_invariant(
            family_name="ctreepo",
            state_kind="sketch_state",
            state_dim=256,
            summary_dim=128,
        )

    def test_sketch_lossy_native_requires_explicit_optin(self):
        with pytest.raises(RuntimeError) as exc:
            check_state_summary_invariant(
                family_name="hll",
                state_kind="sketch_state_lossy_native",
                state_dim=128,
                summary_dim=128,
                allow_lossy_native=False,
            )
        assert "requires allow_lossy_native=True" in str(exc.value)

    def test_sketch_lossy_native_with_optin_passes(self):
        check_state_summary_invariant(
            family_name="hll",
            state_kind="sketch_state_lossy_native",
            state_dim=128,
            summary_dim=128,
            allow_lossy_native=True,
        )

    def test_invalid_state_kind_rejected(self):
        with pytest.raises(ValueError):
            check_state_summary_invariant(
                family_name="x",
                state_kind="not_a_kind",
                state_dim=1,
                summary_dim=1,
            )

    def test_zero_dims_rejected(self):
        with pytest.raises(RuntimeError):
            check_state_summary_invariant(
                family_name="x",
                state_kind="embedding",
                state_dim=0,
                summary_dim=0,
            )

    def test_report_collects_all_violations(self):
        report = state_summary_report(
            family_name="fno",
            state_kind="embedding",
            state_dim=100,
            summary_dim=200,
            g_in_dim=100,
            g_out_dim=100,
        )
        assert isinstance(report, StateSummaryReport)
        assert not report.ok
        # Three violations: state_dim, g_in, g_out.
        assert len(report.violations) == 3


# ---------------------------------------------------------------------------
# specialized embedding/sketch checks
# ---------------------------------------------------------------------------


class TestEmbeddingBudgetCheck:
    def test_passes_when_g_doubles_summary(self):
        check_two_child_embedding_budget(
            family_name="fno",
            summary_dim=768,
            g_in_dim=1536,
            g_out_dim=1536,
        )

    def test_default_state_dim_falls_back_to_g_out(self):
        # state_dim defaults to g_out_dim, which is 1536 >= 2*768.
        check_two_child_embedding_budget(
            family_name="fno",
            summary_dim=768,
            g_in_dim=1536,
            g_out_dim=1536,
        )

    def test_raises_when_g_out_below_2x(self):
        with pytest.raises(RuntimeError):
            check_two_child_embedding_budget(
                family_name="fno",
                summary_dim=768,
                g_in_dim=1536,
                g_out_dim=768,
            )


class TestSketchBudgetCheck:
    def test_passes_when_g_doubles_summary(self):
        check_two_child_sketch_budget(
            family_name="ctreepo_sketch",
            summary_units=64,
            g_in_units=128,
            g_out_units=128,
        )

    def test_lossy_native_optin_passes(self):
        check_two_child_sketch_budget(
            family_name="hll",
            summary_units=128,
            g_in_units=128,
            g_out_units=128,
            allow_lossy_native=True,
        )

    def test_lossy_native_default_rejected(self):
        with pytest.raises(RuntimeError):
            check_two_child_sketch_budget(
                family_name="ctreepo_sketch",
                summary_units=64,
                g_in_units=64,
                g_out_units=64,
            )


# ---------------------------------------------------------------------------
# init-spec grammar
# ---------------------------------------------------------------------------


class TestParseInitSpec:
    @pytest.mark.parametrize("sentinel", sorted(INIT_SPEC_RAW_PREFIXES))
    def test_all_sentinels_parse(self, sentinel):
        parsed = parse_init_spec(sentinel)
        assert parsed == InitSpec(kind="sentinel", value=sentinel)
        assert parsed.raw == sentinel

    def test_none_returns_none(self):
        assert parse_init_spec(None) is None
        assert parse_init_spec("") is None
        assert parse_init_spec("   ") is None

    def test_oracle_prefix_parsed(self):
        parsed = parse_init_spec("oracle:hll_exact")
        assert parsed == InitSpec(kind="oracle", value="hll_exact")
        assert parsed.raw == "oracle:hll_exact"

    def test_artifact_prefix_parsed(self):
        parsed = parse_init_spec("artifact:/tmp/f_v3.json")
        assert parsed == InitSpec(kind="artifact", value="/tmp/f_v3.json")
        assert parsed.raw == "artifact:/tmp/f_v3.json"

    def test_unknown_prefix_rejected(self):
        with pytest.raises(ValueError):
            parse_init_spec("magic:foo")

    def test_unknown_sentinel_rejected(self):
        with pytest.raises(ValueError):
            parse_init_spec("nonsense_sentinel")

    def test_oracle_missing_value_rejected(self):
        with pytest.raises(ValueError):
            parse_init_spec("oracle:")

    def test_case_insensitive_sentinel(self):
        parsed = parse_init_spec("Raw_Concat")
        assert parsed == InitSpec(kind="sentinel", value="raw_concat")


# ---------------------------------------------------------------------------
# family helper fallbacks
# ---------------------------------------------------------------------------


class _LegacyFamilyShim:
    """Stand-in for an existing family that doesn't implement BundleAware."""

    name = "legacy"


class _BundleAwareFamilyShim:
    name = "bundle_aware"

    @property
    def default_f(self) -> str:
        return "oracle:hll_exact"

    @property
    def default_g(self) -> str:
        return "raw_concat"

    def expected_bundle(self):
        return {"leaf_unit": "stream_item", "domain": ("classical_sketch",)}

    def supported_inits(self):
        return {
            "f": frozenset({"identity", "raw", "oracle", "artifact"}),
            "g": frozenset({"raw_concat", "oracle", "artifact"}),
        }

    def resolve_init(self, *, kind: str, spec: str):
        return f"resolved:{kind}:{spec}"

    def share_state_axes(self):
        return frozenset({"f", "g"})


class TestFamilyHelpers:
    def test_legacy_family_falls_back_to_identity_and_raw_concat(self):
        family = _LegacyFamilyShim()
        assert family_default_f(family) == "identity"
        assert family_default_g(family) == "raw_concat"
        assert family_expected_bundle(family) == {}
        assert family_share_state_axes(family) == frozenset()

    def test_legacy_family_supported_inits_is_wildcard(self):
        family = _LegacyFamilyShim()
        supported = family_supported_inits(family)
        assert "f" in supported and "g" in supported
        assert "raw_concat" in supported["g"]
        assert "oracle" in supported["f"]

    def test_bundle_aware_family_returns_declared_defaults(self):
        family = _BundleAwareFamilyShim()
        assert family_default_f(family) == "oracle:hll_exact"
        assert family_default_g(family) == "raw_concat"
        assert family_share_state_axes(family) == frozenset({"f", "g"})

    def test_bundle_aware_family_expected_bundle_threaded_through(self):
        family = _BundleAwareFamilyShim()
        bundle = family_expected_bundle(family)
        assert bundle["leaf_unit"] == "stream_item"
        assert bundle["domain"] == ("classical_sketch",)

    def test_bundle_aware_resolve_init_dispatches_to_family(self):
        family = _BundleAwareFamilyShim()
        result = family_resolve_init(family, kind="g", spec="oracle:hll_max_merge")
        assert result == "resolved:g:oracle:hll_max_merge"

    def test_legacy_resolve_init_returns_parsed_initspec(self):
        family = _LegacyFamilyShim()
        result = family_resolve_init(family, kind="g", spec="raw_concat")
        assert isinstance(result, InitSpec)
        assert result.value == "raw_concat"

    def test_legacy_resolve_init_none_input_returns_none(self):
        family = _LegacyFamilyShim()
        assert family_resolve_init(family, kind="f", spec=None) is None

    def test_bundle_aware_protocol_isinstance_check(self):
        # Runtime-checkable protocol: structural isinstance lookup works.
        assert isinstance(_BundleAwareFamilyShim(), BundleAwareFamilyRuntime)
        # Legacy shim is missing the new properties, so it should not match.
        assert not isinstance(_LegacyFamilyShim(), BundleAwareFamilyRuntime)
