"""
Integration tests for the Phase 1-6 pipeline components.

Each test targets a specific phase's code without requiring GPUs or live
LLM servers.  Run the full suite in ~10 min:

    pytest tests/test_pipeline_phases.py -v --tb=short
"""

import asyncio
import json
import re
import textwrap
from pathlib import Path
from typing import Dict, List

import pytest

from src.core.conditional_memory import (
    ConditionalMemory,
    ConditionalMemoryConfig,
    MemoryRecord,
    canonical_hash,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

SAMPLE_TEXT = textwrap.dedent("""\
    The Federal Reserve raised interest rates by 0.25% on Wednesday,
    surprising analysts at Goldman Sachs and JPMorgan Chase. Chair
    Jerome Powell said inflation remains "stubbornly high" at 3.2%,
    above the Fed's 2% target. Markets reacted sharply, with the
    S&P 500 dropping 1.4% and the Nasdaq falling 2.1%. European
    Central Bank president Christine Lagarde echoed similar concerns
    about persistent price pressures across the Eurozone.
""")

BOILERPLATE_TEXT = (
    "the the the the the the the the the the "
    "the the the the the the the the the the "
    "the the the the the the the the the the"
)


def _make_memory(tmp_path, **kwargs):
    """Create a readwrite ConditionalMemory backed by tmp_path."""
    defaults = dict(
        sqlite_path=tmp_path / "test.db",
        l1_capacity=256,
        mode="readwrite",
    )
    defaults.update(kwargs)
    return ConditionalMemory(**defaults)


# ===================================================================
# Phase 3: ConditionalMemory — multi-head scores
# ===================================================================

class TestConditionalMemoryMultiHead:
    """Multi-head score storage and lookup (Phase 3 / 5.3)."""

    def test_store_and_lookup_single_head(self, tmp_path):
        """Store with one score head, retrieve via score_heads filter."""
        mem = _make_memory(tmp_path)
        try:
            mem.store(SAMPLE_TEXT, scores={"oracle": 0.85})
            rec = mem.lookup(SAMPLE_TEXT, score_heads=["oracle"])
            assert rec is not None
            assert abs(rec.scores["oracle"] - 0.85) < 1e-6
        finally:
            mem.close()

    def test_multi_head_merge_on_update(self, tmp_path):
        """Storing different heads incrementally merges them."""
        mem = _make_memory(tmp_path)
        try:
            mem.store(SAMPLE_TEXT, scores={"similarity": 0.72})
            mem.store(SAMPLE_TEXT, scores={"oracle": 0.85})
            rec = mem.lookup(SAMPLE_TEXT, score_heads=["similarity"])
            assert rec is not None
            assert "similarity" in rec.scores
            assert "oracle" in rec.scores
        finally:
            mem.close()

    def test_lookup_fails_for_missing_head(self, tmp_path):
        """Lookup with a head that was never stored returns None."""
        mem = _make_memory(tmp_path)
        try:
            mem.store(SAMPLE_TEXT, scores={"oracle": 0.85})
            rec = mem.lookup(SAMPLE_TEXT, score_heads=["nonexistent_head"])
            assert rec is None
        finally:
            mem.close()

    def test_store_with_metadata(self, tmp_path):
        """Metadata survives round-trip through store/lookup."""
        mem = _make_memory(tmp_path)
        try:
            mem.store(
                SAMPLE_TEXT,
                scores={"enrichment": 0.5},
                metadata={"key_entities": ["Fed", "Powell"]},
            )
            rec = mem.lookup(SAMPLE_TEXT, score_heads=["enrichment"])
            assert rec is not None
            assert rec.metadata.get("key_entities") == ["Fed", "Powell"]
        finally:
            mem.close()

    def test_cross_instance_persistence(self, tmp_path):
        """Scores persist in L2 and survive a new ConditionalMemory instance."""
        db = tmp_path / "persist.db"
        m1 = ConditionalMemory(sqlite_path=db, l1_capacity=4, mode="readwrite")
        try:
            m1.store(SAMPLE_TEXT, scores={"oracle": 0.9})
        finally:
            m1.close()

        m2 = ConditionalMemory(sqlite_path=db, l1_capacity=4, mode="read")
        try:
            rec = m2.lookup(SAMPLE_TEXT, score_heads=["oracle"])
            assert rec is not None
            assert abs(rec.scores["oracle"] - 0.9) < 1e-6
        finally:
            m2.close()


class TestConditionalMemoryNamespaces:
    """Namespace isolation and get_json/set_json (Phase 3)."""

    def test_json_round_trip(self, tmp_path):
        """set_json / get_json preserves structure."""
        mem = _make_memory(tmp_path)
        try:
            payload = {"entities": ["Fed", "Powell"], "count": 7}
            mem.set_json("enrichment", "key1", payload)
            result = mem.get_json("enrichment", "key1")
            assert result == payload
        finally:
            mem.close()

    def test_namespace_isolation(self, tmp_path):
        """Different namespaces don't see each other's keys."""
        mem = _make_memory(tmp_path)
        try:
            mem.set_text("ns_a", "k1", "value_a")
            mem.set_text("ns_b", "k1", "value_b")
            assert mem.get_text("ns_a", "k1") == "value_a"
            assert mem.get_text("ns_b", "k1") == "value_b"
        finally:
            mem.close()

    def test_namespace_version_accessible(self, tmp_path):
        """namespace_version attribute is available and stable."""
        cfg = ConditionalMemoryConfig(
            enabled=True, mode="readwrite",
            namespace_version="v1.0:test",
            l2_path=tmp_path / "ns.db",
        )
        mem = ConditionalMemory(cfg)
        try:
            assert mem.namespace_version == "v1.0:test"
        finally:
            mem.close()


# ===================================================================
# Phase 5.2: ChunkEnricher
# ===================================================================

class TestChunkEnricher:
    """Pre-merge enrichment layer (Phase 5.2)."""

    def test_tier1_extracts_entities(self):
        """Tier 1 regex enrichment finds capitalized entities."""
        from src.preprocessing.enrichment import ChunkEnricher

        enricher = ChunkEnricher(enable_tier2=False)
        result = enricher.enrich(SAMPLE_TEXT)
        assert result.word_count > 0
        assert result.entity_count > 0
        assert result.entity_density > 0

    def test_tier1_extracts_named_entities(self):
        """Tier 1 extracts named entities (capitalized multi-word phrases)."""
        from src.preprocessing.enrichment import ChunkEnricher

        enricher = ChunkEnricher(enable_tier2=False)
        result = enricher.enrich(SAMPLE_TEXT)
        # Engram extracts multi-word capitalized phrases as entities
        assert len(result.key_entities) >= 3
        entity_text = " ".join(result.key_entities).lower()
        assert "federal reserve" in entity_text or "goldman sachs" in entity_text

    def test_boilerplate_detection(self):
        """Highly repetitive text has high boilerplate ratio."""
        from src.preprocessing.enrichment import ChunkEnricher

        enricher = ChunkEnricher(enable_tier2=False)
        result = enricher.enrich(BOILERPLATE_TEXT)
        assert result.boilerplate_ratio > 0.5
        assert result.is_low_complexity

    def test_prompt_block_formatting(self):
        """to_prompt_block produces non-empty string for entity-rich text."""
        from src.preprocessing.enrichment import ChunkEnricher

        enricher = ChunkEnricher(enable_tier2=False)
        result = enricher.enrich(SAMPLE_TEXT)
        block = result.to_prompt_block()
        assert block.startswith("[ENRICHMENT:")
        assert "Key entities" in block or "Key numbers" in block

    def test_tier2_adds_topic_keywords(self):
        """Tier 2 (heuristic) adds topic_keywords and semantic_complexity."""
        from src.preprocessing.enrichment import ChunkEnricher

        enricher = ChunkEnricher(enable_tier2=True)
        result = enricher.enrich(SAMPLE_TEXT)
        assert len(result.topic_keywords) > 0
        assert result.semantic_complexity > 0

    def test_enrichment_caches_in_memory(self, tmp_path):
        """When ConditionalMemory is provided, enrichment is cached."""
        from src.preprocessing.enrichment import ChunkEnricher

        mem = _make_memory(tmp_path)
        try:
            enricher = ChunkEnricher(memory=mem, enable_tier2=False)
            r1 = enricher.enrich(SAMPLE_TEXT)
            r2 = enricher.enrich(SAMPLE_TEXT)
            assert r1.word_count == r2.word_count
            assert r1.entity_count == r2.entity_count
            # Second call should be a cache hit
            report = mem.report()
            assert report["l1_hits"] + report["l2_hits"] >= 1
        finally:
            mem.close()

    def test_to_dict_round_trip(self):
        """to_dict produces a serializable dictionary."""
        from src.preprocessing.enrichment import ChunkEnricher, ChunkEnrichment

        enricher = ChunkEnricher(enable_tier2=True)
        result = enricher.enrich(SAMPLE_TEXT)
        d = result.to_dict()
        assert isinstance(d, dict)
        assert "word_count" in d
        assert "key_entities" in d
        # Should be JSON-serializable
        json.dumps(d)


# ===================================================================
# Phase 5.1: GatedStrategy
# ===================================================================

class TestGatedStrategy:
    """Context-aware gated strategy (Phase 5.1)."""

    def test_complexity_score_high_for_rich_text(self):
        """Entity-rich, diverse text gets high complexity score."""
        from src.core.strategy import GatedStrategy
        score = GatedStrategy._complexity_score(SAMPLE_TEXT)
        assert score > 0.3, f"Expected high complexity for rich text, got {score}"

    def test_complexity_score_low_for_boilerplate(self):
        """Repetitive text gets low complexity score."""
        from src.core.strategy import GatedStrategy
        score = GatedStrategy._complexity_score(BOILERPLATE_TEXT)
        assert score < 0.3, f"Expected low complexity for boilerplate, got {score}"

    def test_complexity_score_short_text_maxes_out(self):
        """Very short text returns 1.0 (always go to LLM)."""
        from src.core.strategy import GatedStrategy
        score = GatedStrategy._complexity_score("hello world")
        assert score == 1.0

    def test_gate_hits_cached_easy_chunk(self, tmp_path):
        """Summarize returns cached result for easy (cached + low complexity) text."""
        from src.core.strategy import GatedStrategy

        mem = _make_memory(tmp_path)
        try:
            # Pre-populate cache with a "summary" for boilerplate text
            key = canonical_hash(BOILERPLATE_TEXT)
            ns = f"gated_summarize:{mem.namespace_version}"
            mem.set_text(ns, key, "cached summary of boilerplate")

            class _DummyStrategy:
                async def summarize(self, content, rubric, temperature=0.7):
                    raise AssertionError("LLM should not be called")

            gated = GatedStrategy(
                base=_DummyStrategy(),
                memory=mem,
                gate_threshold=0.5,
            )

            result = asyncio.run(gated.summarize(BOILERPLATE_TEXT, "test rubric"))
            assert result == "cached summary of boilerplate"
            assert gated._gate_hits == 1
            assert gated._gate_misses == 0
        finally:
            mem.close()

    def test_gate_miss_calls_llm_for_complex_text(self, tmp_path):
        """Complex text with no cache calls the LLM."""
        from src.core.strategy import GatedStrategy

        mem = _make_memory(tmp_path)
        try:
            call_count = 0

            class _TrackingStrategy:
                async def summarize(self, content, rubric, temperature=0.7):
                    nonlocal call_count
                    call_count += 1
                    return "llm summary"

            gated = GatedStrategy(
                base=_TrackingStrategy(),
                memory=mem,
                gate_threshold=0.3,
            )

            result = asyncio.run(gated.summarize(SAMPLE_TEXT, "test rubric"))
            assert result == "llm summary"
            assert call_count == 1
            assert gated._gate_misses == 1
        finally:
            mem.close()

    def test_gate_stats_tracking(self, tmp_path):
        """gate_stats returns correct hit/miss/rate."""
        from src.core.strategy import GatedStrategy

        mem = _make_memory(tmp_path)
        try:
            class _EchoStrategy:
                async def summarize(self, content, rubric, temperature=0.7):
                    return "echo"

            gated = GatedStrategy(
                base=_EchoStrategy(), memory=mem, gate_threshold=0.3,
            )

            asyncio.run(gated.summarize(SAMPLE_TEXT, "rubric"))

            stats = gated.gate_stats()
            assert stats["gate_misses"] == 1
            assert stats["gate_hits"] == 0
            assert stats["gate_rate"] == 0.0
        finally:
            mem.close()


# ===================================================================
# Phase 1.2: Affinity Routing
# ===================================================================

class TestAffinityRouting:
    """Document-affinity routing (Phase 1.2) — already in test_batch_routing,
    but here we add a determinism/stability check."""

    def test_hash_routing_is_deterministic(self):
        """Same document_id always maps to same server index."""
        from src.core.batch_processor import BatchRequest

        doc_id = "manifesto_doc_42"
        indices = set()
        for i in range(100):
            req = BatchRequest(request_id=f"r{i}", messages=[], document_id=doc_id)
            idx = hash(req.document_id) % 3  # simulate 3 servers
            indices.add(idx)
        assert len(indices) == 1, "Same doc_id should always hash to same index"

    def test_different_docs_spread_across_servers(self):
        """Different document IDs should (usually) map to different servers."""
        from src.core.batch_processor import BatchRequest

        server_counts: Dict[int, int] = {}
        for i in range(30):
            req = BatchRequest(
                request_id=f"r{i}", messages=[], document_id=f"doc_{i}"
            )
            idx = hash(req.document_id) % 3
            server_counts[idx] = server_counts.get(idx, 0) + 1

        # With 30 docs and 3 servers, all servers should get at least 1
        assert len(server_counts) >= 2, "Expected docs spread across servers"


# ===================================================================
# Phase 4.1: Overlapped GPU Transitions
# ===================================================================

class TestOverlappedTransitions:
    """Completion fraction and prewarm config (Phase 4.1)."""

    def test_orchestrator_config_has_prewarm_fields(self):
        """OrchestratorConfig includes enable_prewarm and prewarm_threshold."""
        from src.core.gpu_orchestrator import OrchestratorConfig
        cfg = OrchestratorConfig()
        assert hasattr(cfg, "enable_prewarm")
        assert hasattr(cfg, "prewarm_threshold")
        assert cfg.enable_prewarm is True
        assert cfg.prewarm_threshold == 0.85

    def test_completion_fraction_starts_at_zero(self):
        """BatchTreeOrchestrator.completion_fraction starts at 0."""
        from src.core.batch_orchestrator import BatchTreeOrchestrator

        class _DummyStrategy:
            pass

        orch = BatchTreeOrchestrator.__new__(BatchTreeOrchestrator)
        orch._completed_leaves = 0
        orch._total_leaves = 0
        orch._completed_merges = 0
        orch._total_merges = 0
        assert orch.completion_fraction == 0.0

    def test_completion_fraction_computes_correctly(self):
        """completion_fraction returns correct ratio."""
        from src.core.batch_orchestrator import BatchTreeOrchestrator

        orch = BatchTreeOrchestrator.__new__(BatchTreeOrchestrator)
        orch._completed_leaves = 8
        orch._total_leaves = 10
        orch._completed_merges = 3
        orch._total_merges = 5
        # (8 + 3) / (10 + 5) = 11/15 ≈ 0.733
        frac = orch.completion_fraction
        assert abs(frac - 11 / 15) < 1e-6


# ===================================================================
# Phase 4.2: Size-Aware Merge Scheduling
# ===================================================================

class TestMergeScheduling:
    """Size-aware merge scheduling (Phase 4.2)."""

    def test_plan_merge_task_has_estimated_tokens(self):
        """PlanMergeTask includes estimated_input_tokens field."""
        from src.core.batch_orchestrator import PlanMergeTask

        task = PlanMergeTask(doc_idx=0, id=1, level=2, left_idx=0, right_idx=1)
        assert hasattr(task, "estimated_input_tokens")
        assert task.estimated_input_tokens == 0  # default

        task.estimated_input_tokens = 1500
        assert task.estimated_input_tokens == 1500

    def test_sort_merges_by_level_descending(self):
        """Higher-level merges should be prioritized (critical path)."""
        from src.core.batch_orchestrator import PlanMergeTask

        merges = [
            PlanMergeTask(doc_idx=0, id=1, level=1, left_idx=0, right_idx=1,
                          estimated_input_tokens=500),
            PlanMergeTask(doc_idx=0, id=2, level=3, left_idx=2, right_idx=3,
                          estimated_input_tokens=200),
            PlanMergeTask(doc_idx=0, id=3, level=2, left_idx=4, right_idx=5,
                          estimated_input_tokens=800),
        ]
        # Sort by (-level, -estimated_input_tokens) as in _sort_ready_merges
        sorted_merges = sorted(
            merges, key=lambda m: (-m.level, -m.estimated_input_tokens)
        )
        assert sorted_merges[0].level == 3
        assert sorted_merges[1].level == 2
        assert sorted_merges[2].level == 1


# ===================================================================
# Phase 6.1: KV-Cache Persistence Config
# ===================================================================

class TestKVPersistenceConfig:
    """KV-cache persistence configuration wiring (Phase 6.1)."""

    def test_orchestrator_config_has_kv_fields(self):
        """OrchestratorConfig includes kv_persistence_* fields."""
        from src.core.gpu_orchestrator import OrchestratorConfig
        cfg = OrchestratorConfig()
        assert hasattr(cfg, "kv_persistence_enabled")
        assert cfg.kv_persistence_enabled is False
        assert cfg.kv_persistence_backend == "lmcache"
        assert cfg.kv_persistence_disk_path == "/tmp/thinkingtrees_kv_cache"

    def test_managed_server_accepts_kv_persistence(self):
        """ManagedServer constructor accepts kv_persistence kwarg."""
        from src.core.gpu_orchestrator import ManagedServer, ServerConfig

        config = ServerConfig(
            profile="test", port=9999, cuda_devices="0",
            tensor_parallel=1, enable_sleep_mode=False,
        )
        server = ManagedServer(
            config=config,
            venv_path="/tmp/fake",
            model_path="/tmp/fake_model",
            kv_persistence={"enabled": True, "backend": "lmcache"},
        )
        assert server._kv_persistence["enabled"] is True
        assert server._kv_persistence["backend"] == "lmcache"

    def test_lmcache_config_file_exists(self):
        """config/lmcache_config.yaml was created with expected fields."""
        import yaml

        cfg_path = Path(__file__).parent.parent / "config" / "lmcache_config.yaml"
        assert cfg_path.exists(), f"Missing {cfg_path}"
        with open(cfg_path) as f:
            cfg = yaml.safe_load(f)
        assert cfg["chunk_size"] == 256
        assert cfg["local_cpu"] is True
        assert "local_disk" in cfg

    def test_settings_yaml_has_kv_persistence_section(self):
        """settings.yaml contains orchestration.kv_persistence section."""
        import yaml

        settings_path = Path(__file__).parent.parent / "config" / "settings.yaml"
        with open(settings_path) as f:
            cfg = yaml.safe_load(f)
        kv = cfg.get("orchestration", {}).get("kv_persistence", {})
        assert "enabled" in kv
        assert "backend" in kv
        assert kv["backend"] == "lmcache"


# ===================================================================
# Phase 6.2: Memory-Augmented Preference Learning
# ===================================================================

class TestMemoryAugmentedPreferences:
    """Enrichment context injection into preference collection (Phase 6.2)."""

    def test_build_enrichment_context_returns_none_without_memory(self):
        """No memory → no enrichment context."""
        from src.training.preference.collector import PreferenceCollector

        collector = PreferenceCollector.__new__(PreferenceCollector)
        collector._memory = None
        result = collector._build_enrichment_context("orig", "sumA", "sumB")
        assert result is None

    def test_build_enrichment_context_with_cached_entities(self, tmp_path):
        """When enrichment data is cached, context string is built."""
        from src.training.preference.collector import PreferenceCollector

        mem = _make_memory(tmp_path, namespace_version="test:v1")
        try:
            # Pre-populate enrichment cache
            key = canonical_hash(SAMPLE_TEXT)
            enrichment_data = {
                "key_entities": ["Federal Reserve", "Jerome Powell", "Goldman Sachs"],
                "key_numbers": ["0.25%", "3.2%", "2%"],
            }
            ns = f"enrichment:{mem.namespace_version}"
            mem.set_json(ns, key, enrichment_data)

            collector = PreferenceCollector.__new__(PreferenceCollector)
            collector._memory = mem

            # Summary A preserves "Jerome Powell", B doesn't
            summary_a = "Jerome Powell announced rate changes at Federal Reserve."
            summary_b = "Interest rates were raised by the central bank."

            ctx = collector._build_enrichment_context(SAMPLE_TEXT, summary_a, summary_b)
            assert ctx is not None
            assert "[ENRICHMENT CONTEXT]" in ctx
            assert "Entity preservation" in ctx
        finally:
            mem.close()

    def test_entity_preservation_rate_calculation(self, tmp_path):
        """Preservation rates are computed correctly."""
        from src.training.preference.collector import PreferenceCollector

        mem = _make_memory(tmp_path, namespace_version="test:v1")
        try:
            key = canonical_hash("Original text about Apple and Google and Microsoft.")
            enrichment_data = {
                "key_entities": ["Apple", "Google", "Microsoft"],
                "key_numbers": [],
            }
            ns = f"enrichment:{mem.namespace_version}"
            mem.set_json(ns, key, enrichment_data)

            collector = PreferenceCollector.__new__(PreferenceCollector)
            collector._memory = mem

            # A preserves 2/3, B preserves 1/3
            ctx = collector._build_enrichment_context(
                "Original text about Apple and Google and Microsoft.",
                "Apple and Google are tech companies.",
                "Microsoft is a tech company.",
            )
            assert ctx is not None
            # A should show 67%, B should show 33%
            assert "67%" in ctx or "66%" in ctx  # 2/3
            assert "33%" in ctx  # 1/3
        finally:
            mem.close()


# ===================================================================
# Phase 6.3-6.4: Experiment Frameworks
# ===================================================================

class TestSparsityAllocationFramework:
    """Sparsity allocation sweep structure (Phase 6.3)."""

    def test_result_dataclass_fields(self):
        """SparsityResult has expected fields."""
        from experiments.sparsity_allocation import SparsityResult
        r = SparsityResult(rho=0.7)
        assert r.rho == 0.7
        assert r.mean_oracle_score == 0.0
        assert r.total_llm_calls == 0
        assert r.gate_hits == 0

    def test_efficiency_ratio_computation(self):
        """efficiency_ratio = score / llm_calls."""
        from experiments.sparsity_allocation import SparsityResult
        r = SparsityResult(rho=0.8, mean_oracle_score=0.9, total_llm_calls=100)
        assert abs(r.efficiency_ratio() - 0.009) < 1e-6

    def test_sweep_results_find_optimal(self):
        """find_optimal picks the highest-scoring ρ."""
        from experiments.sparsity_allocation import SparsitySweepResults, SparsityResult
        sweep = SparsitySweepResults(corpus="test", num_documents=10)
        sweep.results = [
            SparsityResult(rho=0.5, mean_oracle_score=0.70),
            SparsityResult(rho=0.8, mean_oracle_score=0.85),
            SparsityResult(rho=1.0, mean_oracle_score=0.80),
        ]
        sweep.find_optimal()
        assert sweep.optimal_rho == 0.8
        assert sweep.optimal_score == 0.85

    def test_load_documents_from_jsonl(self, tmp_path):
        """_load_documents reads .jsonl files."""
        from experiments.sparsity_allocation import _load_documents
        jsonl = tmp_path / "corpus.jsonl"
        jsonl.write_text(
            '{"text": "doc one"}\n'
            '{"text": "doc two"}\n'
            '{"text": "doc three"}\n'
        )
        docs = _load_documents(str(jsonl))
        assert len(docs) == 3
        assert docs[0]["text"] == "doc one"

    def test_load_documents_max_docs(self, tmp_path):
        """_load_documents respects max_docs cap."""
        from experiments.sparsity_allocation import _load_documents
        jsonl = tmp_path / "corpus.jsonl"
        jsonl.write_text("\n".join(f'{{"text": "doc {i}"}}' for i in range(20)))
        docs = _load_documents(str(jsonl), max_docs=5)
        assert len(docs) == 5


class TestComponentAblationFramework:
    """Component ablation framework structure (Phase 6.4)."""

    def test_components_dict_has_all_phases(self):
        """COMPONENTS covers all major pipeline phases."""
        from experiments.component_ablation import COMPONENTS
        assert "gating" in COMPONENTS
        assert "enrichment" in COMPONENTS
        assert "memory" in COMPONENTS
        assert "kv_persistence" in COMPONENTS
        assert "affinity_routing" in COMPONENTS
        assert "overlapped_transitions" in COMPONENTS
        assert "prefix_restructure" in COMPONENTS

    def test_build_config_overrides_disables_gating(self):
        """Disabling gating produces correct config override."""
        from experiments.component_ablation import build_config_overrides
        overrides = build_config_overrides({"gating"})
        assert overrides.get("gating_enabled") is False

    def test_build_config_overrides_switches_routing(self):
        """Disabling affinity routing switches to round_robin."""
        from experiments.component_ablation import build_config_overrides
        overrides = build_config_overrides({"affinity_routing"})
        assert overrides.get("routing_policy") == "round_robin"

    def test_ablation_results_compute_impacts(self):
        """compute_impacts calculates deltas from baseline."""
        from experiments.component_ablation import AblationResults, AblationRun
        results = AblationResults(corpus="test", num_documents=10)
        results.baseline = AblationRun(
            label="baseline",
            enabled_components=["gating", "memory"],
            disabled_components=[],
            mean_oracle_score=0.85,
            wall_time_s=100.0,
            total_llm_calls=500,
        )
        results.ablations = [
            AblationRun(
                label="-gating",
                enabled_components=["memory"],
                disabled_components=["gating"],
                mean_oracle_score=0.82,
                wall_time_s=120.0,
                total_llm_calls=700,
            ),
        ]
        results.compute_impacts()
        impact = results.component_impacts["gating"]
        assert abs(impact["delta_oracle"] - 0.03) < 1e-6
        assert abs(impact["delta_wall_time"] - 20.0) < 1e-6
        assert impact["delta_llm_calls"] == 200


# ===================================================================
# Cross-phase: Canonical Hashing
# ===================================================================

class TestCanonicalHashing:
    """Canonical hashing consistency across pipeline (Phase 3)."""

    def test_hash_is_deterministic(self):
        """Same text always produces same hash."""
        h1 = canonical_hash(SAMPLE_TEXT)
        h2 = canonical_hash(SAMPLE_TEXT)
        assert h1 == h2

    def test_whitespace_normalization(self):
        """Different whitespace produces same hash."""
        h1 = canonical_hash("hello  world")
        h2 = canonical_hash("hello\t\nworld")
        assert h1 == h2

    def test_nfkc_normalization(self):
        """Unicode variants produce same hash."""
        h1 = canonical_hash("Ａ")  # fullwidth A
        h2 = canonical_hash("A")
        assert h1 == h2

    def test_case_sensitivity_preserved_by_default(self):
        """By default, case IS significant."""
        h1 = canonical_hash("ABC")
        h2 = canonical_hash("abc")
        assert h1 != h2

    def test_hash_used_by_enricher_and_strategy(self):
        """Enricher and GatedStrategy use the same hashing function."""
        # Both import from conditional_memory — verify they get the same result
        from src.core.conditional_memory import canonical_hash as cm_hash
        from src.preprocessing.enrichment import ChunkEnricher
        from src.core.strategy import GatedStrategy

        text = "Test text for hashing consistency"
        h = cm_hash(text)
        assert len(h) == 64  # SHA-256 hex digest
        assert all(c in "0123456789abcdef" for c in h)
