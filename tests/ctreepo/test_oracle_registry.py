"""Pinning tests for the oracle registry and ``OracleFamilyRuntime``.

Covers:

- Every default oracle (``type_oracle``, ``hll_exact``, ``hll_max_merge``,
  ``markov_changepoint_count``, ``leaf_local_mixture_target``) is registered
  with consistent ``OracleSpec`` fields aligned to the TreeBundle v1
  vocabulary.
- The native callables behave the same way they did inline.
- The thin re-exports at historical call sites (``src.tree.learned_sketch``,
  ``src.ctreepo.sim.core.markov_changepoint_ops_count``,
  ``src.ctreepo.sim.core.leaf_local_mixture_utility``) resolve to the
  registry entries.
- ``OracleFamilyRuntime`` implements both ``FamilyRuntime`` and
  ``BundleAwareFamilyRuntime`` and threads through ``score_tree`` adapters.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from src.ctreepo.alternating import (
    BundleAwareFamilyRuntime,
    FamilyRuntime,
    family_default_f,
    family_default_g,
    family_expected_bundle,
    family_resolve_init,
    family_share_state_axes,
    family_supported_inits,
)
from src.ctreepo.contracts import (
    LEAF_UNIT_STREAM_ITEM,
    LEAF_UNIT_SYNTHETIC_ATOM,
)
from src.ctreepo.oracles import (
    OracleSpec,
    get_oracle,
    has_oracle,
    list_oracles,
    register_oracle,
)
from src.ctreepo.oracles.runtime import OracleFamilyRuntime
from src.ctreepo.oracles.markov import (
    markov_changepoint_count,
    markov_changepoint_count_for_doc,
)
from src.ctreepo.oracles.sketches import (
    SPIKE_THRESHOLD,
    hll_exact_count,
    hll_max_merge,
    type_oracle,
)


# ---------------------------------------------------------------------------
# Registry contents
# ---------------------------------------------------------------------------


class TestRegistryContents:
    def test_all_canonical_oracles_registered(self):
        names = set(list_oracles())
        assert {
            "type_oracle",
            "hll_exact",
            "hll_max_merge",
            "markov_changepoint_count",
            "leaf_local_mixture_target",
        } <= names

    def test_get_oracle_returns_consistent_spec(self):
        spec = get_oracle("type_oracle")
        assert isinstance(spec, OracleSpec)
        assert spec.name == "type_oracle"
        assert spec.domain == "classical_sketch"
        assert spec.leaf_unit == LEAF_UNIT_SYNTHETIC_ATOM
        assert callable(spec.f_callable)
        assert callable(spec.score_tree)

    def test_unknown_name_raises_keyerror_listing_available(self):
        with pytest.raises(KeyError) as exc:
            get_oracle("does_not_exist")
        message = str(exc.value)
        assert "does_not_exist" in message
        assert "type_oracle" in message  # available list

    def test_register_duplicate_rejected_without_replace(self):
        spec = OracleSpec(
            name="type_oracle",
            domain="classical_sketch",
            leaf_unit=LEAF_UNIT_SYNTHETIC_ATOM,
            f_callable=lambda *a, **kw: None,
        )
        with pytest.raises(ValueError):
            register_oracle(spec)

    def test_register_duplicate_with_replace_overrides(self):
        original = get_oracle("type_oracle")
        try:
            sentinel = object()
            spec = OracleSpec(
                name="type_oracle",
                domain="classical_sketch",
                leaf_unit=LEAF_UNIT_SYNTHETIC_ATOM,
                f_callable=lambda *a, **kw: sentinel,
            )
            register_oracle(spec, replace=True)
            assert get_oracle("type_oracle").f_callable() is sentinel
        finally:
            register_oracle(original, replace=True)

    def test_has_oracle(self):
        assert has_oracle("type_oracle")
        assert not has_oracle("not_a_real_oracle")


# ---------------------------------------------------------------------------
# Native callable behavior preserved
# ---------------------------------------------------------------------------


class TestNativeCallables:
    def test_type_oracle_default_threshold_matches_legacy(self):
        # Legacy SPIKE_THRESHOLD=0.90 — values below stay below.
        indicators = [0.0, 0.5, 0.91, 0.95]
        positions = [0, 1, 2, 3]
        out = type_oracle(indicators, positions, n_types=2)
        # Spikes are at positions 2 (type 0) and 3 (type 1), so each type
        # gets one count.
        assert out == [1.0, 1.0]
        assert SPIKE_THRESHOLD == pytest.approx(0.90)

    def test_type_oracle_explicit_threshold(self):
        out = type_oracle([0.6, 0.6], [0, 1], n_types=2, threshold=0.5)
        assert out == [1.0, 1.0]

    def test_hll_exact_count(self):
        assert hll_exact_count([1, 2, 2, 3, 3, 3]) == 3
        assert hll_exact_count([]) == 0

    def test_hll_max_merge_returns_pure_function(self):
        class _FakeSketch:
            def __init__(self, regs):
                self.regs = list(regs)

            def copy(self):
                return _FakeSketch(self.regs)

            def merge(self, other):
                self.regs = [max(a, b) for a, b in zip(self.regs, other.regs)]
                return self

        left = _FakeSketch([1, 0, 3])
        right = _FakeSketch([2, 5, 1])
        merged = hll_max_merge(left, right)
        assert merged.regs == [2, 5, 3]
        # Pure function: left untouched.
        assert left.regs == [1, 0, 3]

    def test_markov_changepoint_count(self):
        assert markov_changepoint_count([0, 0, 1, 1, 0]) == 2
        assert markov_changepoint_count([0]) == 0
        assert markov_changepoint_count([]) == 0

    def test_markov_changepoint_count_for_doc_slice(self):
        class _FakeDoc:
            def __init__(self, regimes):
                self.token_regimes = list(regimes)

        doc = _FakeDoc([0, 0, 1, 1, 0, 0])
        assert markov_changepoint_count_for_doc(doc, start=0, end=6) == 2
        # slice off the back half: [0, 0, 1] has one transition.
        assert markov_changepoint_count_for_doc(doc, start=0, end=3) == 1

    def test_lda_target_lifted_signature_matches_re_export(self):
        # The re-export at sim/core/leaf_local_mixture_utility._true_doc_target
        # must delegate to the canonical implementation in oracles.lda. We
        # check that with a degenerate empty-doc fixture: zero spans -> zero
        # leaves -> sum is 0.0.
        from src.ctreepo.oracles.lda import leaf_local_mixture_target
        from src.ctreepo.sim.core.leaf_local_mixture_utility import (
            LeafLocalMixtureDoc,
            _true_doc_target,
        )

        doc = LeafLocalMixtureDoc(
            tokens=tuple(),
            topics=tuple(),
            global_topic_weights=tuple(),
            local_topic_weights=tuple(),
            latent_section_spans=tuple(),
            latent_section_block_spans=tuple(),
            atomic_block_tokens=0,
        )
        theta = np.zeros((1, 1))
        W_base = np.zeros((1, 1))
        canonical = leaf_local_mixture_target(
            doc, theta=theta, W_base=W_base, lambda_multiplier=1.0
        )
        legacy = _true_doc_target(
            doc, theta=theta, W_base=W_base, lambda_multiplier=1.0
        )
        assert canonical == legacy == 0.0


# ---------------------------------------------------------------------------
# Thin re-exports
# ---------------------------------------------------------------------------


class TestReExports:
    def test_learned_sketch_type_oracle_is_registry_callable(self):
        from src.tree.learned_sketch import type_oracle as legacy_type_oracle

        assert legacy_type_oracle is type_oracle
        assert legacy_type_oracle is get_oracle("type_oracle").f_callable

    def test_markov_oracle_count_delegates(self):
        from src.ctreepo.sim.core.markov_changepoint_ops_count import (
            _changepoint_count,
            _oracle_count,
        )

        class _FakeDoc:
            def __init__(self, regimes):
                self.token_regimes = list(regimes)

        assert _changepoint_count([0, 1, 0]) == 2
        doc = _FakeDoc([0, 0, 1, 0])
        assert _oracle_count(doc, start=0, end=4) == 2

    def test_leaf_local_mixture_re_export(self):
        # Imported above already; this just pins the public attribute path.
        from src.ctreepo.sim.core.leaf_local_mixture_utility import _true_doc_target

        assert callable(_true_doc_target)


# ---------------------------------------------------------------------------
# OracleFamilyRuntime
# ---------------------------------------------------------------------------


class TestOracleFamilyRuntime:
    def test_unknown_oracle_raises_at_construction(self):
        with pytest.raises(KeyError):
            OracleFamilyRuntime("nonexistent_oracle")

    def test_implements_protocols_structurally(self):
        family = OracleFamilyRuntime("hll_exact")
        # Both protocols are runtime-checkable.
        assert isinstance(family, FamilyRuntime)
        assert isinstance(family, BundleAwareFamilyRuntime)

    def test_default_f_uses_oracle_namespace(self):
        family = OracleFamilyRuntime("hll_exact")
        assert family_default_f(family) == "oracle:hll_exact"

    def test_default_g_is_raw_concat_for_f_only_oracles(self):
        family = OracleFamilyRuntime("hll_exact")
        assert family_default_g(family) == "raw_concat"

    def test_default_g_is_oracle_for_g_carrying_oracles(self):
        family = OracleFamilyRuntime("hll_max_merge")
        # hll_max_merge has a g_callable, so default_g should be itself.
        assert family_default_g(family) == "oracle:hll_max_merge"

    def test_expected_bundle_threads_through(self):
        family = OracleFamilyRuntime("hll_exact")
        bundle = family_expected_bundle(family)
        assert bundle["domain"] == "classical_sketch"
        assert bundle["leaf_unit"] == LEAF_UNIT_STREAM_ITEM

    def test_train_f_is_no_op_returning_oracle_handle(self):
        family = OracleFamilyRuntime("hll_exact")
        artifact = family.train_f(
            f_init=None,
            g=None,
            traces=[],
            output_dir=Path("/tmp/oracle_train_smoke"),
            iteration=1,
        )
        assert artifact == "oracle:hll_exact"
        family.validate_artifact(kind="f", artifact=artifact)

    def test_train_g_no_op_for_f_only_oracle_returns_raw_concat(self):
        family = OracleFamilyRuntime("hll_exact")
        artifact = family.train_g(
            g_init=None,
            f=None,
            traces=[],
            output_dir=Path("/tmp/oracle_train_smoke"),
            iteration=2,
        )
        assert artifact == "raw_concat"

    def test_score_roots_with_f_uses_score_tree_adapter(self):
        family = OracleFamilyRuntime("hll_exact")

        class _FakeTree:
            def __init__(self, tokens):
                self.tokens = tokens

        trees = [_FakeTree([1, 2, 2]), _FakeTree([5, 5, 5, 5]), _FakeTree([])]
        results = family.score_roots_with_f(f="oracle:hll_exact", g=None, trees=trees)
        assert results == [2.0, 1.0, 0.0]

    def test_score_roots_with_f_handles_missing_attrs_gracefully(self):
        family = OracleFamilyRuntime("hll_exact")

        class _Bare:
            pass

        # No tokens or leaves on the tree -> adapter raises -> family returns None.
        results = family.score_roots_with_f(f=None, g=None, trees=[_Bare()])
        assert results == [None]

    def test_resolve_init_via_helper_requires_prefixed_grammar(self):
        # The init-spec grammar enforces "oracle:<name>" — bare oracle names
        # without a prefix are rejected at parse time so users get a clear
        # error from the unified runner CLI.
        family = OracleFamilyRuntime("hll_exact")
        assert family_resolve_init(family, kind="f", spec="oracle:hll_exact") == "oracle:hll_exact"
        with pytest.raises(ValueError):
            family_resolve_init(family, kind="f", spec="hll_exact")

    def test_resolve_init_directly_on_family_is_permissive(self):
        # When called directly (not via the helper), the family accepts the
        # bare oracle name as a convenience for programmatic callers that
        # already know they have an oracle handle.
        family = OracleFamilyRuntime("hll_exact")
        assert family.resolve_init(kind="f", spec="hll_exact") == "oracle:hll_exact"
        assert family.resolve_init(kind="f", spec="oracle:hll_exact") == "oracle:hll_exact"

    def test_resolve_init_rejects_other_oracle(self):
        family = OracleFamilyRuntime("hll_exact")
        with pytest.raises(ValueError):
            family_resolve_init(family, kind="f", spec="oracle:type_oracle")

    def test_share_state_axes_empty_for_oracles(self):
        family = OracleFamilyRuntime("hll_exact")
        assert family_share_state_axes(family) == frozenset()

    def test_supported_inits_g_set_depends_on_g_callable_presence(self):
        f_only = OracleFamilyRuntime("hll_exact")
        assert "raw_concat" in family_supported_inits(f_only)["g"]
        with_g = OracleFamilyRuntime("hll_max_merge")
        assert "oracle" in family_supported_inits(with_g)["g"]
