# Conditional Memory for ThinkingTrees: Lessons from Engram

**Source paper:** Cheng et al. (2026). *Conditional Memory via Scalable Lookup: A New Axis of Sparsity for Large Language Models.* DeepSeek-AI / Peking University. arXiv:2601.07372v1.

**Scope:** This document translates Engram's architectural insights about conditional memory into a concrete implementation plan for the ThinkingTrees OPS pipeline. Each section maps an Engram concept to a ThinkingTrees improvement, with exact file paths, code-level specifications, and expected impact.

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Background: Engram's Core Ideas](#2-background-engrams-core-ideas)
3. [Current ThinkingTrees Memory Landscape](#3-current-thinkingtrees-memory-landscape)
4. [Workstream 1: Unified ConditionalMemory Module](#4-workstream-1-unified-conditionalmemory-module)
5. [Workstream 2: Chunk Canonicalization](#5-workstream-2-chunk-canonicalization)
6. [Workstream 3: Context-Aware Gated Strategy Selection](#6-workstream-3-context-aware-gated-strategy-selection)
7. [Workstream 4: Pre-Merge Enrichment Layer](#7-workstream-4-pre-merge-enrichment-layer)
8. [Workstream 5: Sparsity Allocation Experiments](#8-workstream-5-sparsity-allocation-experiments)
9. [Workstream 6: Deterministic Batch Planning with Prefetch](#9-workstream-6-deterministic-batch-planning-with-prefetch)
10. [Workstream 7: Zipfian Tiered Cache Hierarchy](#10-workstream-7-zipfian-tiered-cache-hierarchy)
11. [Workstream 8: Multi-Head Oracle Scoring](#11-workstream-8-multi-head-oracle-scoring)
12. [Workstream 9: Memory-Augmented Preference Learning](#12-workstream-9-memory-augmented-preference-learning)
13. [Workstream 10: Component Sensitivity Analysis](#13-workstream-10-component-sensitivity-analysis)
14. [Implementation Roadmap](#14-implementation-roadmap)
15. [Measurement & Evaluation Framework](#15-measurement--evaluation-framework)
16. [Appendix: Engram-ThinkingTrees Concept Mapping](#16-appendix-engram-thinkingtrees-concept-mapping)

---

## 1. Executive Summary

### The Core Insight

Engram's central thesis: **conditional memory** (O(1) static lookup) and **conditional computation** (dynamic neural processing) are structurally complementary primitives. Allocating ~20-25% of model capacity to static memory — designed as a first-class architectural component — yields outsized gains not only on knowledge retrieval tasks but, *surprisingly*, on reasoning tasks too. The mechanism: by offloading stereotyped local pattern reconstruction to memory, the system frees up computational depth for genuine compositional reasoning.

### Why This Matters for ThinkingTrees

ThinkingTrees currently treats every oracle call, every merge operation, and every chunk the same way: through full LLM inference. This is analogous to a pure-MoE model that lacks a native knowledge lookup primitive, "wasting valuable sequential depth on trivial operations that could otherwise be allocated to higher-level reasoning" (Engram §1).

The ThinkingTrees pipeline has seven independent, inconsistently designed caches that die with each process. There is no persistent memory, no context-aware gating, no principled allocation between computation and lookup. This plan addresses all of these gaps.

### Expected Impact

| Metric | Current | Projected | Mechanism |
|--------|---------|-----------|-----------|
| Oracle calls per document (repeat corpus) | 100% | 50-70% | Persistent ConditionalMemory + canonicalization |
| LLM calls for tree building | 100% | 60-75% | Gated strategy selection (skip easy merges) |
| Cross-run startup cost | Full recompute | Near-zero for seen documents | Persistent L2 memory |
| Merge quality (oracle accuracy) | Baseline | +5-15% | Pre-merge enrichment frees LLM for reasoning |
| Pipeline throughput | Baseline | +20-40% | Deterministic batch planning + prefetch |

---

## 2. Background: Engram's Core Ideas

### 2.1 The Computation vs. Memory Duality

Language modeling involves two qualitatively different sub-tasks:
- **Compositional reasoning**: dynamic, context-dependent, requires deep computation
- **Knowledge retrieval**: static, local, stereotyped (named entities, formulaic phrases, boilerplate)

Standard Transformers lack a native knowledge lookup primitive and are forced to *simulate* retrieval through computation. Engram resolves this by introducing a parallel memory pathway: hashed N-gram embeddings that provide O(1) lookup for static patterns.

**Key result (Table 1):** Engram-27B outperforms iso-parameter, iso-FLOPs MoE-27B across ALL benchmarks, with the largest gains in reasoning (+5.0 BBH) and code/math (+3.0 HumanEval), not just knowledge retrieval.

### 2.2 The U-Shaped Scaling Law

Given a fixed parameter budget, the optimal allocation between MoE experts (computation) and Engram embeddings (memory) follows a U-shaped curve (Figure 3, left). The optimum is at ρ ≈ 75-80% computation, 20-25% memory. This is stable across scales.

### 2.3 Context-Aware Gating

Retrieved memory vectors are modulated by a context-dependent gate α_t ∈ (0,1) computed via scaled dot-product attention between the hidden state (query) and the retrieved embedding (key). When memory contradicts context, the gate suppresses it to near-zero (Equation 4).

### 2.4 Effective Depth Increase

Engram's most important mechanistic finding: by handling static patterns at early layers, the model achieves deeper effective representations earlier. Layer 5 of Engram-27B corresponds functionally to layer 12 of MoE-27B (Figure 4, CKA analysis). The network is "deeper" for reasoning without actually adding layers.

### 2.5 Infrastructure-Aware Efficiency

Because Engram uses deterministic hash-based addressing (unlike MoE's dynamic routing), retrieval indices are known before the forward pass. This enables:
- **Prefetching**: Asynchronous retrieval from host memory, overlapping with computation
- **Zipfian caching**: Multi-level cache hierarchy exploiting power-law access patterns
- **Result**: 100B-parameter table offloaded to host memory incurs < 3% throughput overhead

---

## 3. Current ThinkingTrees Memory Landscape

### 3.1 Existing Cache Inventory

| System | File | Hash | Eviction | Persistent | Thread-Safe | Stats |
|--------|------|------|----------|-----------|-------------|-------|
| SimilarityScorer | `src/core/scoring.py:335-412` | SHA-256 | LRU (1-4K) | No | Yes (lock) | Yes |
| OraclePredictionCache | `src/training/metrics/metrics.py:928-981` | String key | LRU (OrderedDict, 10K) | No | Yes (lock) | Yes |
| PreferenceCollector | `src/training/preference/collector.py:247-279` | MD5 | None (unbounded) | No | No | No |
| VLLMEmbeddingClient | `src/training/embedding_proxy.py:393-480` | stable_text_key | None (unbounded) | No | No | No |
| GenRMDSPyJudge | `src/training/preference/genrm_dspy.py:156-319` | (rubric,law_type) tuple | None (unbounded) | No | No | No |
| LLMClient | `src/core/llm_client.py:165-290` | Message hash | LRU (OrderedDict, 10K) | No | Yes | Yes |
| AdaptiveChunkMemory | `src/preprocessing/chunker.py:125-217` | doc_id | Truncate to 2048 | No | No | No |
| CheckpointManager | `src/core/checkpoints.py` | Phase name | Disk (JSON+pickle) | Yes | File-based | No |

### 3.2 Key Problems

1. **No persistence across runs.** Every cache except CheckpointManager dies with the process. For corpora with repetitive structure (political manifestos, legal documents, financial reports), this means recomputing the same oracle scores, embeddings, and summaries from scratch every run.

2. **Inconsistent hashing.** Three different hashing strategies (SHA-256, MD5, `stable_text_key`) produce incompatible cache keys. The same text hashed by different functions produces different keys, preventing cache sharing across subsystems.

3. **No context-aware gating.** Cached values are returned unconditionally whenever the key matches. There's no mechanism to suppress stale or contextually inappropriate cached results (analogous to Engram's α_t gate).

4. **No cache hierarchy.** All caches are flat in-memory dicts. There's no hot/warm/cold tiering to exploit access frequency patterns.

5. **No cross-subsystem coordination.** The SimilarityScorer, OraclePredictionCache, and LLMClient cache independently. A text scored by one may not be found by another even if the information is equivalent.

6. **No unified observability.** Only 3 of 8 caches track hit/miss statistics. There's no dashboard or aggregated view of cache effectiveness.

---

## 4. Workstream 1: Unified ConditionalMemory Module

### 4.1 Motivation (Engram §2, §2.5)

Engram's key design principle: memory is a *first-class architectural primitive*, not scattered caching heuristics. It has one coherent module with:
- Deterministic O(1) addressing via multi-head hashing
- Context-aware gating that dynamically modulates trust
- Multi-branch integration that provides redundant views
- Tiered storage that exploits Zipfian access patterns

### 4.2 Design

#### New file: `src/core/conditional_memory.py`

```python
"""
Conditional Memory module for ThinkingTrees.

Inspired by Engram (Cheng et al., 2026), this module provides a unified,
persistent, context-gated memory system for the OPS pipeline.

Architecture:
    L1 (hot, in-memory) → L2 (warm, SQLite) → L3 (miss → compute)

Each entry stores multi-head scores and context embeddings for gating.
Lookup is O(1) via canonical text hashing.
"""

import hashlib
import sqlite3
import threading
import unicodedata
import re
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

@dataclass
class MemoryEntry:
    """A single memory entry, analogous to an Engram embedding slot."""
    canonical_hash: str                    # Deterministic address (Engram §2.2)
    original_text_preview: str             # First 200 chars for debugging
    scores: Dict[str, float]              # Multi-head scores (Engram §2.4)
    context_embedding: Optional[List[float]]  # For context-aware gating (Engram §2.3)
    summary: Optional[str]                # Cached summary if available
    access_count: int = 0                 # For Zipfian tier promotion (Engram §2.5)
    created_at: float = 0.0              # Timestamp
    last_accessed: float = 0.0           # For LRU within tiers

@dataclass
class MemoryStats:
    """Unified statistics across all tiers."""
    l1_hits: int = 0
    l2_hits: int = 0
    misses: int = 0
    gate_suppressions: int = 0          # Times context gate rejected a match
    total_entries_l1: int = 0
    total_entries_l2: int = 0

    @property
    def hit_rate(self) -> float:
        total = self.l1_hits + self.l2_hits + self.misses
        return (self.l1_hits + self.l2_hits) / total if total > 0 else 0.0

    @property
    def l1_hit_rate(self) -> float:
        total = self.l1_hits + self.l2_hits + self.misses
        return self.l1_hits / total if total > 0 else 0.0

class ConditionalMemory:
    """
    Unified memory system for ThinkingTrees.

    Maps text → MemoryEntry via canonical hashing (Engram §2.2).
    Context-aware gating (Engram §2.3) optionally suppresses stale entries.
    Tiered storage (Engram §2.5) exploits Zipfian access patterns.

    Thread-safe for concurrent use across async pipeline stages.

    Parameters
    ----------
    l1_capacity : int
        Maximum entries in the hot in-memory tier. Default 4096.
    l2_path : Optional[Path]
        Path to SQLite database for persistent warm tier.
        If None, operates as in-memory-only (backward compatible).
    gate_threshold : float
        Minimum cosine similarity between stored context embedding and
        current context for a cached entry to be returned. Set to 0.0
        to disable gating (always trust cache). Default 0.0.
    promotion_threshold : int
        Access count at which an L2 entry is promoted to L1. Default 3.
    """

    def __init__(
        self,
        l1_capacity: int = 4096,
        l2_path: Optional[Path] = None,
        gate_threshold: float = 0.0,
        promotion_threshold: int = 3,
    ):
        self.l1_capacity = l1_capacity
        self.l2_path = l2_path
        self.gate_threshold = gate_threshold
        self.promotion_threshold = promotion_threshold

        # L1: hot in-memory tier (OrderedDict for O(1) LRU)
        self._l1: OrderedDict[str, MemoryEntry] = OrderedDict()
        self._lock = threading.Lock()

        # L2: warm persistent tier (SQLite)
        self._l2: Optional[sqlite3.Connection] = None
        if l2_path is not None:
            self._init_l2(l2_path)

        # Statistics
        self.stats = MemoryStats()

    # --- Public API ---

    def lookup(
        self,
        text: str,
        context_embedding: Optional[List[float]] = None,
        score_heads: Optional[List[str]] = None,
    ) -> Optional[MemoryEntry]:
        """
        O(1) lookup with optional context-aware gating.

        Parameters
        ----------
        text : str
            The text to look up. Canonicalized before hashing.
        context_embedding : Optional[List[float]]
            Current context vector. If provided and gate_threshold > 0,
            the cached entry is suppressed if cosine similarity between
            stored and current context falls below threshold.
        score_heads : Optional[List[str]]
            If specified, only return entry if it has ALL requested heads.

        Returns
        -------
        Optional[MemoryEntry]
            The cached entry, or None on miss / gate suppression.
        """
        key = self.canonical_hash(text)

        # L1 lookup
        with self._lock:
            entry = self._l1.get(key)
            if entry is not None:
                # Check gate
                if not self._passes_gate(entry, context_embedding):
                    self.stats.gate_suppressions += 1
                    return None
                # Check required heads
                if score_heads and not all(h in entry.scores for h in score_heads):
                    return None
                # LRU touch
                self._l1.move_to_end(key)
                entry.access_count += 1
                self.stats.l1_hits += 1
                return entry

        # L2 lookup (outside L1 lock to avoid holding lock during I/O)
        if self._l2 is not None:
            entry = self._l2_lookup(key)
            if entry is not None:
                if not self._passes_gate(entry, context_embedding):
                    self.stats.gate_suppressions += 1
                    return None
                if score_heads and not all(h in entry.scores for h in score_heads):
                    return None
                entry.access_count += 1
                self.stats.l2_hits += 1
                # Promote to L1 if frequently accessed (Zipfian)
                if entry.access_count >= self.promotion_threshold:
                    self._promote_to_l1(entry)
                return entry

        self.stats.misses += 1
        return None

    def store(
        self,
        text: str,
        scores: Dict[str, float],
        context_embedding: Optional[List[float]] = None,
        summary: Optional[str] = None,
    ) -> None:
        """
        Store a new entry or update an existing one.

        Multi-head scores are merged (updated) if the entry already exists,
        not replaced. This allows incremental enrichment by different scorers.
        """
        key = self.canonical_hash(text)
        entry = MemoryEntry(
            canonical_hash=key,
            original_text_preview=text[:200],
            scores=scores,
            context_embedding=context_embedding,
            summary=summary,
            access_count=1,
            created_at=time.time(),
            last_accessed=time.time(),
        )

        with self._lock:
            if key in self._l1:
                # Merge scores (multi-head update)
                self._l1[key].scores.update(scores)
                if summary:
                    self._l1[key].summary = summary
                self._l1.move_to_end(key)
            else:
                # Evict oldest if at capacity
                while len(self._l1) >= self.l1_capacity:
                    evicted_key, evicted = self._l1.popitem(last=False)
                    # Demote to L2 (Engram §2.5 cache hierarchy)
                    if self._l2 is not None:
                        self._l2_store(evicted)
                self._l1[key] = entry

        # Always persist to L2 for cross-run durability
        if self._l2 is not None:
            self._l2_store(entry)

    # --- Canonical Hashing (Engram §2.2 Tokenizer Compression) ---

    @staticmethod
    def canonical_hash(text: str) -> str:
        """
        Deterministic canonical hash for text.

        Applies Engram-inspired normalization before hashing:
        1. NFKC Unicode normalization (same as Engram §2.2)
        2. Whitespace collapsing
        3. Case folding
        4. SHA-256 for stable, deterministic output

        This maps surface variants ("Apple", "apple", " apple") to the
        same key, analogous to Engram's tokenizer compression achieving
        23% vocabulary reduction.
        """
        # NFKC normalization (Engram uses this for tokenizer compression)
        text = unicodedata.normalize('NFKC', text)
        # Collapse whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        # Case fold
        text = text.casefold()
        # Deterministic hash
        return hashlib.sha256(text.encode('utf-8')).hexdigest()

    # --- Context-Aware Gating (Engram §2.3) ---

    def _passes_gate(
        self,
        entry: MemoryEntry,
        context_embedding: Optional[List[float]],
    ) -> bool:
        """
        Context-aware gate: suppress cached entry if context has shifted.

        Analogous to Engram's Equation 4:
            α_t = σ(RMSNorm(h_t)^T RMSNorm(k_t) / √d)

        Here we use cosine similarity as a simplified gate.
        If α < gate_threshold, the memory is suppressed.
        """
        if self.gate_threshold <= 0.0:
            return True  # Gating disabled
        if context_embedding is None or entry.context_embedding is None:
            return True  # No context to gate on

        # Cosine similarity
        dot = sum(a * b for a, b in zip(entry.context_embedding, context_embedding))
        norm_a = sum(a * a for a in entry.context_embedding) ** 0.5
        norm_b = sum(b * b for b in context_embedding) ** 0.5
        if norm_a == 0 or norm_b == 0:
            return True
        similarity = dot / (norm_a * norm_b)

        return similarity >= self.gate_threshold

    # --- L2 Persistent Tier (SQLite) ---

    def _init_l2(self, path: Path) -> None:
        """Initialize SQLite database for persistent warm tier."""
        path.parent.mkdir(parents=True, exist_ok=True)
        self._l2 = sqlite3.connect(str(path), check_same_thread=False)
        self._l2.execute("""
            CREATE TABLE IF NOT EXISTS memory (
                canonical_hash TEXT PRIMARY KEY,
                original_text_preview TEXT,
                scores_json TEXT,
                context_embedding_json TEXT,
                summary TEXT,
                access_count INTEGER DEFAULT 0,
                created_at REAL,
                last_accessed REAL
            )
        """)
        self._l2.execute("""
            CREATE INDEX IF NOT EXISTS idx_access_count
            ON memory(access_count DESC)
        """)
        self._l2.commit()

    def _l2_lookup(self, key: str) -> Optional[MemoryEntry]:
        """Look up an entry in the persistent tier."""
        ...  # Standard SQLite SELECT, deserialize JSON fields

    def _l2_store(self, entry: MemoryEntry) -> None:
        """Store or update an entry in the persistent tier."""
        ...  # Standard SQLite INSERT OR REPLACE, serialize JSON fields

    def _promote_to_l1(self, entry: MemoryEntry) -> None:
        """Promote a frequently accessed L2 entry to L1."""
        with self._lock:
            if len(self._l1) >= self.l1_capacity:
                evicted_key, evicted = self._l1.popitem(last=False)
                if self._l2 is not None:
                    self._l2_store(evicted)
            self._l1[entry.canonical_hash] = entry

    # --- Observability ---

    def report(self) -> Dict[str, Any]:
        """Return unified statistics for all tiers."""
        return {
            "l1_hits": self.stats.l1_hits,
            "l2_hits": self.stats.l2_hits,
            "misses": self.stats.misses,
            "gate_suppressions": self.stats.gate_suppressions,
            "hit_rate": self.stats.hit_rate,
            "l1_hit_rate": self.stats.l1_hit_rate,
            "l1_entries": len(self._l1),
            "l2_entries": self._l2_count() if self._l2 else 0,
        }
```

### 4.3 Integration Points

The ConditionalMemory replaces or wraps all 7 existing caches:

| Existing Cache | Integration Strategy |
|----------------|---------------------|
| `SimilarityScorer._cache` | Delegate to `ConditionalMemory.lookup/store` with head name `"similarity"` |
| `OraclePredictionCache` | Delegate to `ConditionalMemory.lookup/store` with head name `"oracle"` |
| `PreferenceCollector._oracle_cache` | Use `ConditionalMemory.lookup` for oracle scores |
| `VLLMEmbeddingClient._cache` | Store embeddings as `context_embedding` in `MemoryEntry` |
| `GenRMDSPyJudge._prompt_cache` | Separate concern — keep as-is (prompt-level, not text-level) |
| `LLMClient._cache` | Keep as-is for response-level caching; augment with ConditionalMemory for semantic-level |
| `AdaptiveChunkMemory` | Extend `MemoryEntry` with a `feedback_signals` field for adaptive chunking signals |

### 4.4 Migration Plan

**Phase 1 (Non-breaking):** Create `ConditionalMemory` as a new module. Wire it into `run_pipeline.py` as an *additional* layer alongside existing caches. Measure hit rates and verify correctness without changing existing behavior.

**Phase 2 (Gradual replacement):** One by one, refactor existing caches to delegate to `ConditionalMemory`:
1. `SimilarityScorer` → add `memory: Optional[ConditionalMemory]` parameter
2. `OraclePredictionCache` → wrap `ConditionalMemory.lookup` with LRU interface
3. `PreferenceCollector` → inject shared `ConditionalMemory` instance

**Phase 3 (Cleanup):** Remove redundant cache implementations, consolidate hashing to `canonical_hash`.

### 4.5 Configuration

Add to `config/settings.yaml`:

```yaml
memory:
  enabled: true
  l1_capacity: 4096
  l2_path: "memory/conditional_memory.db"   # Persistent across runs
  gate_threshold: 0.0                        # 0.0 = disabled, 0.85 = strict
  promotion_threshold: 3                     # Access count for L2→L1 promotion
  canonicalize: true                         # NFKC + whitespace + casefold
```

### 4.6 Expected Impact

- **Repeat corpus (e.g., re-running manifestos):** 50-80% of oracle calls served from L2 cache
- **Within-run deduplication:** 10-20% fewer oracle calls from canonicalization collapsing surface variants
- **Cross-subsystem sharing:** Oracle scores computed by one subsystem immediately available to all others

---

## 5. Workstream 2: Chunk Canonicalization

### 5.1 Motivation (Engram §2.2, Appendix C)

Engram's tokenizer compression maps surface variants to canonical IDs. Example: 163 whitespace variants → single canonical token. 54 variants of "a" (including "A", "á", "ä", etc.) → single canonical token. This achieves 23% vocabulary reduction, directly translating to 23% fewer hash collisions and higher cache hit rates.

ThinkingTrees chunks the same text differently depending on boundary alignment, whitespace, and Unicode normalization. Two chunks containing the same political phrase but with different leading whitespace get different hashes and are treated as completely different texts.

### 5.2 Design

#### New function in `src/preprocessing/chunker.py`:

```python
def canonicalize_chunk(text: str) -> str:
    """
    Canonicalize chunk text for cache key generation.

    Applies Engram-inspired normalization (§2.2, Appendix C):
    1. NFKC Unicode normalization
    2. Whitespace collapsing (all whitespace → single space)
    3. Strip leading/trailing whitespace

    NOTE: Does NOT casefold. Casefolding is applied only for cache keys
    (in ConditionalMemory.canonical_hash), not for text passed to LLMs.
    The LLM sees the original text; only the cache key is canonicalized.
    """
    import unicodedata
    import re
    text = unicodedata.normalize('NFKC', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text
```

#### Integration in `chunk_for_ops` and `AdaptiveChunkingConfig`:

```python
# In chunk_for_ops():
chunks = []
for span in raw_spans:
    canonical = canonicalize_chunk(span.text)
    chunks.append(TextChunk(
        text=canonical,           # Canonicalized for LLM + cache
        original_text=span.text,  # Preserve original for audit trail
        ...
    ))
```

#### TextChunk extension in `src/core/data_models.py`:

```python
@dataclass
class TextChunk:
    text: str                              # Canonicalized text (sent to LLM)
    original_text: Optional[str] = None    # Pre-canonicalization text (audit trail)
    char_start: int = 0
    char_end: int = 0
    token_count: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)
```

### 5.3 Files to Modify

| File | Change |
|------|--------|
| `src/preprocessing/chunker.py` | Add `canonicalize_chunk()`, apply in `chunk_for_ops()` |
| `src/core/data_models.py` | Add `original_text` field to `TextChunk` |
| `src/core/scoring.py` | Use `ConditionalMemory.canonical_hash` instead of inline SHA-256 |
| `src/training/metrics/metrics.py` | Use `ConditionalMemory.canonical_hash` for OraclePredictionCache keys |
| `src/training/preference/collector.py` | Use `ConditionalMemory.canonical_hash` instead of inline MD5 |
| `src/training/embedding_proxy.py` | Use `ConditionalMemory.canonical_hash` instead of `_stable_text_key` |

### 5.4 Expected Impact

- **Cache hit rate improvement:** 10-20% based on Engram's 23% vocabulary compression ratio, adjusted down because ThinkingTrees operates on chunk-level text (less surface variation than individual tokens)
- **Zero compute cost:** Pure preprocessing, no LLM calls needed
- **Immediate win:** Can be deployed independently before the full ConditionalMemory module

---

## 6. Workstream 3: Context-Aware Gated Strategy Selection

### 6.1 Motivation (Engram §2.3, §6.5, Figure 7)

Engram's gating mechanism (α_t ∈ (0,1)) dynamically decides how much to trust retrieved memory based on alignment between the static lookup and the current dynamic context. Figure 7 shows the gate pattern: high activation on stereotyped patterns ("Alexander the Great", "By the way"), near-zero activation on novel compositional content.

In ThinkingTrees, strategy selection is a static choice at config time: you pick `BatchedStrategy`, `DSPyStrategy`, or `TournamentStrategy` and it applies uniformly to all merge operations. But merge operations vary enormously in difficulty:

- **Easy merges (high α for memory path):** Combining two adjacent chunks about the same topic, both containing similar entities and sentiments. A template-based merge or lightweight model would suffice.
- **Hard merges (low α for memory path, high for computation):** Reconciling two chunks with contradictory political positions, or merging a factual chunk with a rhetorical chunk. These require full LLM reasoning.

### 6.2 Design

#### New file: `src/core/gated_strategy.py`

```python
"""
Context-aware gated strategy selection for tree building.

Inspired by Engram's gating mechanism (§2.3, Equation 4).
Dynamically routes merge operations between:
  - "Memory" path: cheap template/cache-based merging (α ≈ 0)
  - "Computation" path: full LLM-based merging (α ≈ 1)

The gate is trained to predict merge difficulty from input features.
"""

from dataclasses import dataclass
from typing import Optional
import math

@dataclass
class MergeDifficultySignal:
    """Features for estimating merge difficulty."""
    embedding_similarity: float        # Cosine sim between left/right chunks
    entity_overlap: float             # Jaccard coefficient of named entities
    oracle_score_variance: float      # Variance of children's oracle scores
    level: int                        # Tree level (0=leaf merges, higher=harder)
    left_length: int                  # Token count of left child
    right_length: int                 # Token count of right child

class GatedStrategy:
    """
    Routes merge operations between cheap and expensive strategies.

    Gate function (analogous to Engram Equation 4):
        α = σ(w · features + b)
    where features = MergeDifficultySignal and σ is sigmoid.

    When α < threshold:
        Use cheap_strategy (template merge, cached result, small model)
    When α ≥ threshold:
        Use expensive_strategy (full LLM merge via BatchedStrategy)
    """

    def __init__(
        self,
        cheap_strategy: SummarizationStrategy,
        expensive_strategy: SummarizationStrategy,
        gate_threshold: float = 0.5,
        memory: Optional[ConditionalMemory] = None,
    ):
        self.cheap_strategy = cheap_strategy
        self.expensive_strategy = expensive_strategy
        self.gate_threshold = gate_threshold
        self.memory = memory

        # Gate weights (initially heuristic, can be learned)
        self.gate_weights = {
            'embedding_similarity': -2.0,    # High sim → easy → low α
            'entity_overlap': -1.5,          # High overlap → easy
            'oracle_score_variance': 3.0,    # High variance → hard → high α
            'level': 0.5,                    # Higher levels → harder
            'length_ratio': 1.0,             # Asymmetric lengths → harder
        }
        self.gate_bias = 0.0

    def compute_gate(self, signal: MergeDifficultySignal) -> float:
        """Compute gate value α ∈ (0, 1)."""
        length_ratio = abs(signal.left_length - signal.right_length) / (
            max(signal.left_length, signal.right_length) + 1
        )
        z = (
            self.gate_weights['embedding_similarity'] * signal.embedding_similarity
            + self.gate_weights['entity_overlap'] * signal.entity_overlap
            + self.gate_weights['oracle_score_variance'] * signal.oracle_score_variance
            + self.gate_weights['level'] * signal.level
            + self.gate_weights['length_ratio'] * length_ratio
            + self.gate_bias
        )
        return 1.0 / (1.0 + math.exp(-z))  # sigmoid

    async def merge(self, left: Node, right: Node, rubric: str) -> str:
        """Route merge to cheap or expensive strategy based on gate."""
        signal = self._extract_signal(left, right)
        alpha = self.compute_gate(signal)

        if alpha < self.gate_threshold:
            # Memory/template path
            # First: check ConditionalMemory for cached merge result
            if self.memory is not None:
                combined_text = left.summary + " ||| " + right.summary
                entry = self.memory.lookup(combined_text)
                if entry and entry.summary:
                    return entry.summary

            # Fallback: cheap strategy
            return await self.cheap_strategy.merge(left, right, rubric)
        else:
            # Full computation path
            result = await self.expensive_strategy.merge(left, right, rubric)

            # Store in memory for future lookups
            if self.memory is not None:
                combined_text = left.summary + " ||| " + right.summary
                self.memory.store(
                    text=combined_text,
                    scores={},
                    summary=result,
                )

            return result

    def _extract_signal(self, left: Node, right: Node) -> MergeDifficultySignal:
        """Extract difficulty features from two nodes."""
        # ... compute embedding similarity, entity overlap, etc.
        ...
```

#### Cheap Strategy: Template-based Merge

```python
class TemplateMergeStrategy:
    """
    Lightweight merge strategy for easy/routine merges.

    Uses extractive or template-based summarization instead of
    full LLM inference. Suitable when children are topically similar
    and structurally simple.
    """

    TEMPLATE = (
        "{left_summary}\n\n"
        "Additionally: {right_key_points}"
    )

    async def merge(self, left: Node, right: Node, rubric: str) -> str:
        """Template-based merge (no LLM call)."""
        # Simple extractive approach: keep the longer summary,
        # append key sentences from the shorter one
        if len(left.summary) >= len(right.summary):
            base, supplement = left.summary, right.summary
        else:
            base, supplement = right.summary, left.summary

        # Extract first and last sentences from supplement
        sentences = supplement.split('. ')
        key_points = '. '.join(sentences[:2] + sentences[-1:])

        return f"{base}\n\n{key_points}"
```

### 6.3 Integration in Tree Builder

Modify `src/tree/builder.py` to accept `GatedStrategy`:

```python
class TreeBuilder:
    def __init__(
        self,
        strategy: SummarizationStrategy,  # Can now be GatedStrategy
        ...
    ):
        self.strategy = strategy
```

No changes needed to `TreeBuilder` itself — `GatedStrategy` implements the same `SummarizationStrategy` protocol. It's purely a strategy injection.

### 6.4 Gate Weight Learning

Initially use heuristic weights (as above). Over time, collect (features, actual_quality_delta) pairs by comparing gated vs full-LLM merge quality. Use logistic regression to learn optimal gate weights.

Store learning data in `ConditionalMemory`:
```python
self.memory.store(
    text=combined_text,
    scores={
        "gate_alpha": alpha,
        "cheap_quality": cheap_oracle_score,
        "expensive_quality": expensive_oracle_score,
    },
    summary=result,
)
```

### 6.5 Expected Impact

- **30-50% reduction in LLM merge calls** for typical manifesto processing, where many leaf-level merges combine topically adjacent chunks
- **Quality preservation:** Gate ensures hard merges still get full LLM treatment
- **Progressive improvement:** Gate weights improve as the system processes more documents

### 6.6 Files to Create/Modify

| File | Action |
|------|--------|
| `src/core/gated_strategy.py` | Create: GatedStrategy, TemplateMergeStrategy, MergeDifficultySignal |
| `src/core/strategy.py` | Add GatedStrategy to strategy registry |
| `src/tree/builder.py` | No changes (protocol-compatible) |
| `config/settings.yaml` | Add `strategy.gated.enabled`, `strategy.gated.threshold` |
| `src/training/run_pipeline.py` | Wire GatedStrategy when config enables it |

---

## 7. Workstream 4: Pre-Merge Enrichment Layer

### 7.1 Motivation (Engram §6.1, §6.1.2, Figure 4, Table 3)

Engram's deepest mechanistic finding: by injecting static knowledge at early layers, the model's shallow layers become functionally equivalent to deeper layers of the baseline. The "Diana, Princess of Wales" example (Table 3) shows that an LLM without Engram needs 6 layers of attention + FFN to resolve a named entity — work that could be handled by a single lookup.

In ThinkingTrees, the merge LLM call must simultaneously:
1. **Parse content** — understand what the chunks say
2. **Identify entities** — find named entities, key terms, policy positions
3. **Evaluate importance** — determine what's preservation-critical
4. **Compose summary** — synthesize into coherent text

Steps 1 and 2 are static pattern recognition that could be pre-computed cheaply, freeing the LLM to focus on steps 3 and 4 (the genuine reasoning).

### 7.2 Design

#### New file: `src/preprocessing/enrichment.py`

```python
"""
Pre-merge enrichment layer for ThinkingTrees.

Inspired by Engram's effective depth increase mechanism (§6.1).
By pre-computing static features (entities, key phrases, topic labels)
before the merge step, the merge LLM can focus on reasoning about
information preservation rather than basic content understanding.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional

@dataclass
class EnrichmentMetadata:
    """Static features attached to leaf nodes before merging."""
    entities: List[str] = field(default_factory=list)
    key_phrases: List[str] = field(default_factory=list)
    topic_label: str = ""
    sentiment_polarity: float = 0.0            # -1.0 to 1.0
    static_pattern_ratio: float = 0.0          # Fraction of boilerplate/formulaic content
    estimated_oracle_range: tuple = (0.0, 0.0) # (low, high) oracle score estimate
    word_count: int = 0
    unique_entity_count: int = 0

class ChunkEnricher:
    """
    Attaches structured metadata to chunks before tree building.

    Three enrichment tiers (from cheapest to most expensive):

    Tier 1 (regex/heuristic, ~0ms per chunk):
        - Word count
        - Basic entity patterns (capitalized multi-word sequences)
        - Boilerplate detection (repeated section headers, etc.)

    Tier 2 (embedding model, ~5ms per chunk):
        - Topic clustering via embedding similarity to reference topics
        - Semantic key phrases via TF-IDF against corpus background
        - Sentiment polarity via embedding-space projection

    Tier 3 (small LLM, ~50ms per chunk):
        - Full NER (named entity recognition)
        - Key phrase extraction with importance ranking
        - Oracle score range estimation

    Default: Tier 1 + Tier 2. Tier 3 only if embedding model is available.
    """

    def __init__(
        self,
        tier: int = 2,
        embedding_client: Optional[VLLMEmbeddingClient] = None,
        topic_centroids: Optional[Dict[str, List[float]]] = None,
        memory: Optional[ConditionalMemory] = None,
    ):
        self.tier = tier
        self.embedding_client = embedding_client
        self.topic_centroids = topic_centroids or {}
        self.memory = memory

    def enrich(self, chunks: List[TextChunk]) -> List[TextChunk]:
        """
        Attach enrichment metadata to each chunk.

        Checks ConditionalMemory first — if a chunk has been enriched
        before, reuse the cached metadata. Only compute for novel chunks.
        """
        results = []
        novel_indices = []

        for i, chunk in enumerate(chunks):
            if self.memory:
                entry = self.memory.lookup(chunk.text)
                if entry and 'enrichment' in entry.scores:
                    chunk.metadata['enrichment'] = entry.metadata.get('enrichment', {})
                    results.append(chunk)
                    continue
            novel_indices.append(i)
            results.append(chunk)

        # Batch-enrich novel chunks
        if novel_indices:
            novel_chunks = [chunks[i] for i in novel_indices]
            enrichments = self._batch_enrich(novel_chunks)
            for idx, enrichment in zip(novel_indices, enrichments):
                results[idx].metadata['enrichment'] = enrichment
                # Cache for future runs
                if self.memory:
                    self.memory.store(
                        text=chunks[idx].text,
                        scores={'enrichment': 1.0},  # Flag as enriched
                        summary=None,
                    )

        return results

    def _batch_enrich(self, chunks: List[TextChunk]) -> List[EnrichmentMetadata]:
        """Compute enrichment for a batch of chunks."""
        # Tier 1: Regex/heuristic (always)
        enrichments = [self._tier1_enrich(c) for c in chunks]

        # Tier 2: Embedding-based (if available)
        if self.tier >= 2 and self.embedding_client:
            self._tier2_enrich_batch(chunks, enrichments)

        return enrichments

    def _tier1_enrich(self, chunk: TextChunk) -> EnrichmentMetadata:
        """Heuristic enrichment (no model calls)."""
        import re
        text = chunk.text

        # Basic entity detection: capitalized multi-word sequences
        entities = re.findall(r'\b(?:[A-Z][a-z]+\s+){1,3}[A-Z][a-z]+\b', text)

        # Key phrases: first occurrence of quoted terms or bold/italic markers
        key_phrases = re.findall(r'"([^"]+)"', text)

        # Boilerplate detection: very short lines, repeated patterns
        lines = text.split('\n')
        boilerplate_lines = sum(1 for l in lines if len(l.strip()) < 20)
        static_ratio = boilerplate_lines / max(len(lines), 1)

        return EnrichmentMetadata(
            entities=list(set(entities))[:20],
            key_phrases=key_phrases[:10],
            static_pattern_ratio=static_ratio,
            word_count=len(text.split()),
            unique_entity_count=len(set(entities)),
        )

    def _tier2_enrich_batch(
        self,
        chunks: List[TextChunk],
        enrichments: List[EnrichmentMetadata],
    ) -> None:
        """Embedding-based enrichment (batch API call)."""
        texts = [c.text for c in chunks]
        embeddings = self.embedding_client.embed_texts(texts)

        for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
            # Topic classification via nearest centroid
            if self.topic_centroids:
                best_topic = ""
                best_sim = -1.0
                for topic, centroid in self.topic_centroids.items():
                    sim = cosine_similarity(embedding, centroid)
                    if sim > best_sim:
                        best_sim = sim
                        best_topic = topic
                enrichments[i].topic_label = best_topic
```

### 7.3 Integration into Merge Prompts

Modify the merge prompt template to include enrichment data:

```python
# In src/core/strategy.py or src/core/signatures.py:

ENRICHED_MERGE_TEMPLATE = """
## Left Child
**Topics:** {left_topics}
**Key Entities:** {left_entities}
**Key Phrases:** {left_phrases}
**Content:** {left_summary}

## Right Child
**Topics:** {right_topics}
**Key Entities:** {right_entities}
**Key Phrases:** {right_phrases}
**Content:** {right_summary}

## Rubric
{rubric}

## Task
Merge the two summaries while preserving the entities and phrases listed above.
Focus on maintaining the political/analytical content, not on re-identifying
what the text is about (that information is provided above).
"""
```

This mirrors Engram's architecture: the enrichment layer (analogous to Engram at layer 2) pre-computes the "static pattern" component, so the merge LLM (analogous to the deeper transformer layers) can focus entirely on compositional reasoning about what to preserve and how to integrate.

### 7.4 Expected Impact

- **+5-15% oracle accuracy on merges** (based on Engram's +5.0 BBH improvement from freeing up reasoning depth)
- **Better entity preservation** — entities are explicitly listed in the merge prompt, reducing the chance of dropping them
- **Faster convergence** in DSPy optimization — richer merge prompts give the optimizer a better starting point

### 7.5 Files to Create/Modify

| File | Action |
|------|--------|
| `src/preprocessing/enrichment.py` | Create: ChunkEnricher, EnrichmentMetadata |
| `src/core/data_models.py` | Ensure `TextChunk.metadata` is Dict[str, Any] (already is) |
| `src/core/strategy.py` | Add enriched merge template variant |
| `src/core/signatures.py` | Add DSPy signature that includes enrichment fields |
| `src/tree/builder.py` | Call `enricher.enrich()` on chunks before building |
| `src/training/run_pipeline.py` | Wire enricher, pass to builder |

---

## 8. Workstream 5: Sparsity Allocation Experiments

### 8.1 Motivation (Engram §3, §3.1, Figure 3)

Engram's Sparsity Allocation problem asks: given a fixed total parameter budget, how should capacity be distributed between MoE experts (computation) and Engram memory (lookup)? The answer is a robust U-shaped curve with optimum at ρ ≈ 75-80%.

ThinkingTrees has an analogous allocation problem: given a fixed total LLM token budget for processing a corpus, how should tokens be distributed across pipeline phases?

### 8.2 Design

#### Allocation Ratio Definition

Let `T_total` = total LLM tokens available for processing a corpus. Define allocation ratio ρ:

```
T_computation = ρ · T_total       # Tree building + oracle scoring + optimization
T_memory      = (1-ρ) · T_total   # Building and querying persistent chunk memory
```

At ρ = 1.0 (current default), every oracle call and merge call goes to the LLM. At ρ < 1.0, some token budget is spent upfront building a lookup table of `chunk → (score, summary)` mappings from a subset of documents, which subsequent documents can query via O(1) lookup.

#### Experimental Protocol

```python
# New file: experiments/sparsity_allocation.py

"""
Sparsity Allocation Experiment for ThinkingTrees.

Sweeps the allocation ratio ρ ∈ {0.5, 0.6, 0.7, 0.8, 0.9, 1.0}
and measures final oracle accuracy vs. total LLM tokens consumed.

Hypothesis (from Engram §3.1): There exists a U-shaped optimum
where hybrid allocation outperforms pure computation (ρ=1.0).
"""

ALLOCATION_RATIOS = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

def run_allocation_experiment(
    corpus: List[DocumentSample],
    total_token_budget: int,
    allocation_ratio: float,
) -> AllocationResult:
    """
    Run pipeline with a specific allocation ratio.

    Phase 1 (Memory building): Process first N documents with full
    LLM oracle scoring, storing all results in ConditionalMemory.
    Token budget = (1 - ρ) * T_total.

    Phase 2 (Memory-augmented processing): Process remaining documents
    using ConditionalMemory for cache hits, LLM for misses.
    Token budget = ρ * T_total.
    """
    memory = ConditionalMemory(l2_path=Path(f"memory/alloc_{allocation_ratio}.db"))

    # Determine memory-building budget
    memory_budget = int((1 - allocation_ratio) * total_token_budget)
    compute_budget = int(allocation_ratio * total_token_budget)

    # Phase 1: Build memory from first documents
    n_memory_docs = estimate_docs_for_budget(memory_budget)
    memory_docs = corpus[:n_memory_docs]
    for doc in memory_docs:
        # Full processing with all results stored in memory
        result = process_document(doc, memory=memory, store_all=True)

    # Phase 2: Process remaining with memory assistance
    eval_docs = corpus[n_memory_docs:]
    results = []
    for doc in eval_docs:
        result = process_document(doc, memory=memory, budget=compute_budget)
        results.append(result)

    return AllocationResult(
        ratio=allocation_ratio,
        oracle_accuracy=mean_oracle_accuracy(results),
        total_tokens_used=memory_budget + sum(r.tokens for r in results),
        cache_hit_rate=memory.stats.hit_rate,
    )
```

### 8.3 Metrics to Track

| Metric | Measures | Expected Pattern |
|--------|----------|-----------------|
| Oracle MAE vs ρ | Final prediction quality | U-shaped (best at ρ ≈ 0.75-0.85) |
| Total tokens vs ρ | Compute efficiency | Monotonically decreasing as ρ decreases (more caching) |
| Throughput (docs/min) vs ρ | Processing speed | Increases as ρ decreases (fewer LLM calls) |
| Cache hit rate vs ρ | Memory utilization | Higher at lower ρ (more memory investment) |

### 8.4 Files to Create

| File | Description |
|------|-------------|
| `experiments/sparsity_allocation.py` | Main experiment script |
| `experiments/allocation_analysis.py` | Plot U-shaped curves, fit scaling law |

---

## 9. Workstream 6: Deterministic Batch Planning with Prefetch

### 9.1 Motivation (Engram §2.5, §6.4, Table 4, Figure 2)

Engram's key system insight: because retrieval addresses are deterministic (known from token IDs before the forward pass), the system can *prefetch* embeddings asynchronously while earlier layers compute. Result: 100B parameters offloaded to host memory with < 3% throughput overhead.

In ThinkingTrees, the binary merge tree structure is fully deterministic. Given N leaf chunks, the complete merge schedule (which pairs merge at each level) is known before any LLM call. But `AsyncTreeBuilder` currently operates reactively — it waits for level L results before constructing level L+1 merge requests.

### 9.2 Design

#### Merge Schedule Pre-computation

```python
# New method in src/tree/builder.py:

@dataclass
class MergeOp:
    """A single planned merge operation."""
    level: int
    left_idx: int       # Index of left child in level's node list
    right_idx: int      # Index of right child in level's node list
    result_idx: int     # Where the result goes in next level

@dataclass
class MergeSchedule:
    """Complete pre-computed merge schedule for a tree."""
    levels: List[List[MergeOp]]
    total_merges: int
    height: int

    @classmethod
    def from_leaf_count(cls, n_leaves: int) -> 'MergeSchedule':
        """
        Pre-compute the complete merge schedule.

        Given N leaves, the binary merge tree is deterministic:
        - Level 0: N leaves
        - Level 1: ceil(N/2) merges
        - Level 2: ceil(ceil(N/2)/2) merges
        - ...
        - Level H: 1 root

        This is O(N) computation, done once before any LLM calls.
        Analogous to Engram's deterministic hash-based addressing
        that enables prefetching (§2.5).
        """
        levels = []
        current_count = n_leaves
        level = 0
        while current_count > 1:
            ops = []
            result_idx = 0
            for i in range(0, current_count, 2):
                if i + 1 < current_count:
                    ops.append(MergeOp(level, i, i+1, result_idx))
                    result_idx += 1
                else:
                    # Odd one out: pass through
                    ops.append(MergeOp(level, i, -1, result_idx))
                    result_idx += 1
            levels.append(ops)
            current_count = result_idx
            level += 1

        return cls(
            levels=levels,
            total_merges=sum(len(l) for l in levels),
            height=len(levels),
        )
```

#### Prompt Prefetching

```python
class PrefetchingTreeBuilder(TreeBuilder):
    """
    Tree builder with deterministic batch planning and prompt prefetch.

    Inspired by Engram's prefetch-and-overlap strategy (§2.5, Figure 2b):
    - Pre-compute full merge schedule (deterministic)
    - While level L executes on LLM, prepare prompts for level L+1
    - Overlap prompt construction with LLM inference
    """

    async def build_tree_prefetched(
        self,
        chunks: List[TextChunk],
        rubric: str,
    ) -> Tree:
        # Phase 1: Deterministic schedule (O(N), no LLM calls)
        schedule = MergeSchedule.from_leaf_count(len(chunks))

        # Phase 2: Execute with prefetching
        current_nodes = [self._create_leaf(c) for c in chunks]

        for level_idx, level_ops in enumerate(schedule.levels):
            next_level_ops = (
                schedule.levels[level_idx + 1]
                if level_idx + 1 < len(schedule.levels)
                else None
            )

            # Start prefetching prompt templates for NEXT level
            # while current level executes
            prefetch_task = None
            if next_level_ops is not None:
                prefetch_task = asyncio.create_task(
                    self._prefetch_prompts(next_level_ops, rubric)
                )

            # Execute current level merges (LLM calls)
            next_nodes = await self._execute_level(
                current_nodes, level_ops, rubric
            )

            # Ensure prefetch is complete
            if prefetch_task:
                await prefetch_task

            current_nodes = next_nodes

        return self._assemble_tree(current_nodes[0])

    async def _prefetch_prompts(
        self,
        ops: List[MergeOp],
        rubric: str,
    ) -> None:
        """
        Pre-construct prompt templates for the next level.

        The static portions (rubric, task context, system instructions)
        are identical across all merges. By pre-computing them, we
        maximize vLLM prefix cache hits (config: enable_prefix_caching=true).
        """
        # Pre-hash the static prompt prefix for vLLM prefix caching
        static_prefix = self._build_static_prefix(rubric)
        # Store in thread-local for use by _execute_level
        self._cached_prefix = static_prefix
```

### 9.3 Integration with vLLM Prefix Caching

ThinkingTrees already has `enable_prefix_caching: true` in `config/settings.yaml`. The prefetching strategy maximizes its effectiveness:

1. **Static prefix reuse:** All merge prompts share the same rubric, task context, and system instructions. By ensuring these are byte-identical across all merge calls, vLLM's prefix caching (APC) computes the KV cache for this prefix once and reuses it.

2. **Prompt ordering:** Process merges within a level in order (left-to-right), so adjacent merges share the most prefix overlap.

### 9.4 Expected Impact

- **20-40% throughput improvement** from prefetch overlap and better prefix cache utilization
- **Zero quality impact** — purely a systems optimization
- **Compounding benefit** with the gated strategy (Workstream 3): fewer LLM calls + each call is faster

### 9.5 Files to Modify

| File | Change |
|------|--------|
| `src/tree/builder.py` | Add `MergeSchedule`, `PrefetchingTreeBuilder` |
| `src/core/batch_orchestrator.py` | Update to use merge schedule for cross-document batching |
| `config/settings.yaml` | Add `tree.prefetch_enabled: true` |

---

## 10. Workstream 7: Zipfian Tiered Cache Hierarchy

### 10.1 Motivation (Engram §2.5)

Natural language N-grams follow a Zipfian distribution: a small fraction of patterns accounts for the vast majority of accesses. Engram exploits this with a multi-level cache hierarchy: GPU HBM (fastest) → Host DRAM → NVMe SSD (largest).

ThinkingTrees corpus data is similarly Zipfian. In political manifestos:
- ~500 common policy phrases cover ~80% of all oracle evaluations
- ~50 boilerplate patterns (headers, section intros) cover ~90% of "easy" chunks
- The long tail of novel/rare content requires ~10% of oracle calls

### 10.2 Design

This is largely implemented by the `ConditionalMemory` module (Workstream 1). This workstream adds the Zipfian analysis and automatic tier management.

#### Access Frequency Tracking

```python
# In src/core/conditional_memory.py:

class ZipfianTierManager:
    """
    Manages automatic promotion/demotion between cache tiers
    based on observed access frequency.

    Periodically analyzes access patterns and adjusts tier boundaries:
    - Top 10% most accessed → L1 (in-memory hot)
    - Next 40% → L2 (SQLite warm)
    - Bottom 50% → L2 only (no promotion)

    Inspired by Engram's Multi-Level Cache Hierarchy (§2.5).
    """

    def __init__(self, memory: ConditionalMemory, rebalance_interval: int = 1000):
        self.memory = memory
        self.rebalance_interval = rebalance_interval
        self._ops_since_rebalance = 0

    def maybe_rebalance(self) -> None:
        """Check if it's time to rebalance tier assignments."""
        self._ops_since_rebalance += 1
        if self._ops_since_rebalance >= self.rebalance_interval:
            self._rebalance()
            self._ops_since_rebalance = 0

    def _rebalance(self) -> None:
        """
        Promote frequently accessed L2 entries to L1,
        demote infrequently accessed L1 entries to L2.
        """
        # Get access counts from L2
        l2_entries = self.memory._l2_get_top_entries(
            limit=self.memory.l1_capacity,
            order_by="access_count DESC"
        )

        # Promote top entries to L1
        for entry in l2_entries:
            if entry.access_count >= self.memory.promotion_threshold:
                self.memory._promote_to_l1(entry)

    def access_distribution_report(self) -> Dict:
        """
        Report Zipfian statistics for the current corpus.

        Returns the fraction of total accesses attributed to the
        top 1%, 5%, 10%, 20% of entries (expected to follow power law).
        """
        ...
```

### 10.3 Corpus-Specific Warm-Up

For known corpora (e.g., Manifesto Project), pre-seed the cache with common patterns:

```python
# New file: scripts/warmup_memory.py

"""
Pre-seed ConditionalMemory with known high-frequency patterns.

For the manifesto corpus, this includes:
- Common policy phrases and their typical RILE scores
- Standard section headers and boilerplate patterns
- Party-family-specific terminology
"""

MANIFESTO_WARMUP_PATTERNS = [
    # (pattern, expected_rile_range)
    ("economic growth and free market", (20.0, 60.0)),
    ("social justice and equality", (-60.0, -20.0)),
    ("national security and defense", (10.0, 50.0)),
    ("environmental protection", (-40.0, -10.0)),
    # ... hundreds more from corpus analysis
]
```

### 10.4 Expected Impact

- **80% of oracle lookups served from L1** after warmup (Zipfian prediction)
- **Near-zero startup cost** for repeat corpora (L2 persists across runs)
- **Automatic adaptation** — tier boundaries adjust to actual access patterns

---

## 11. Workstream 8: Multi-Head Oracle Scoring

### 11.1 Motivation (Engram §2.2, §2.4)

Engram uses K independent hash heads per N-gram order to reduce collision noise and provide redundancy. The heads are concatenated into a richer, more robust representation.

ThinkingTrees scores each node on a single dimension (RILE score or preservation score). A single noisy score can mislead optimization and auditing — there's no way to diagnose *why* a score is bad.

### 11.2 Design

#### Multi-Head OracleScore

```python
# Extension to src/core/scoring.py:

@dataclass
class MultiHeadOracleScore:
    """
    Multi-head oracle score for richer signal.

    Instead of a single scalar, provides K independent quality
    dimensions. Analogous to Engram's multi-head hashing (§2.2)
    which provides redundancy and richer representations.

    Heads:
        primary:    Task-specific score (e.g., RILE position)
        entity:     Named entity preservation (recall vs original)
        sentiment:  Sentiment polarity preservation
        coverage:   Topic/section coverage completeness
    """
    primary: float              # Head 1: task-specific (existing oracle)
    entity_recall: float = -1.0 # Head 2: fraction of original entities preserved
    sentiment_delta: float = -1.0  # Head 3: |sentiment(summary) - sentiment(original)|
    coverage: float = -1.0      # Head 4: fraction of topics/sections represented

    @property
    def available_heads(self) -> Dict[str, float]:
        """Return only heads that have been computed (not -1.0)."""
        heads = {"primary": self.primary}
        if self.entity_recall >= 0:
            heads["entity_recall"] = self.entity_recall
        if self.sentiment_delta >= 0:
            heads["sentiment_delta"] = self.sentiment_delta
        if self.coverage >= 0:
            heads["coverage"] = self.coverage
        return heads

    def aggregate(self, weights: Optional[Dict[str, float]] = None) -> float:
        """
        Weighted aggregation of available heads.

        Analogous to Engram's branch-specific gating (§2.4, Equation 6)
        where independent gates modulate a shared value.
        """
        if weights is None:
            weights = {"primary": 1.0}  # Default: just use primary
        heads = self.available_heads
        total_weight = sum(weights.get(h, 0.0) for h in heads)
        if total_weight == 0:
            return self.primary
        return sum(weights.get(h, 0.0) * v for h, v in heads.items()) / total_weight
```

#### Cheap Head Computation

Entity recall and coverage can be computed *without LLM calls*:

```python
def compute_entity_recall(original: str, summary: str) -> float:
    """Compute entity preservation without LLM (regex-based)."""
    import re
    original_entities = set(re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+\b', original))
    summary_entities = set(re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+\b', summary))
    if not original_entities:
        return 1.0
    return len(original_entities & summary_entities) / len(original_entities)

def compute_coverage(original: str, summary: str, n_topics: int = 5) -> float:
    """Compute topic coverage via keyword overlap."""
    from collections import Counter
    original_words = Counter(original.lower().split())
    summary_words = set(summary.lower().split())
    # Top N keywords by frequency in original
    top_keywords = [w for w, _ in original_words.most_common(n_topics * 3)
                    if len(w) > 4][:n_topics]
    if not top_keywords:
        return 1.0
    return sum(1 for kw in top_keywords if kw in summary_words) / len(top_keywords)
```

### 11.3 Integration

Store multi-head scores in ConditionalMemory:

```python
memory.store(
    text=chunk_text,
    scores={
        "primary": oracle_score,
        "entity_recall": entity_recall,
        "sentiment_delta": sentiment_delta,
        "coverage": coverage,
    },
    ...
)
```

The auditor can then provide richer diagnostics:

```python
# In src/tree/auditor.py:
def diagnose_violation(self, node: Node) -> str:
    """Explain WHY a node fails, not just that it fails."""
    scores = node.metadata.get('multi_head_scores', {})
    issues = []
    if scores.get('entity_recall', 1.0) < 0.5:
        issues.append(f"Entity loss: only {scores['entity_recall']:.0%} of entities preserved")
    if scores.get('coverage', 1.0) < 0.5:
        issues.append(f"Topic gap: only {scores['coverage']:.0%} of topics covered")
    if scores.get('sentiment_delta', 0.0) > 0.3:
        issues.append(f"Sentiment drift: Δ={scores['sentiment_delta']:.2f}")
    return "; ".join(issues) if issues else "Primary score deviation only"
```

### 11.4 Expected Impact

- **Richer audit diagnostics** — know *why* a node fails, not just *that* it fails
- **More robust optimization** — DSPy can optimize against multiple complementary objectives
- **Cheap additional signal** — entity_recall and coverage cost zero LLM calls

### 11.5 Files to Modify

| File | Change |
|------|--------|
| `src/core/scoring.py` | Add MultiHeadOracleScore, head computation functions |
| `src/tree/auditor.py` | Add `diagnose_violation()` using multi-head scores |
| `src/training/metrics/metrics.py` | Support multi-head metric computation |
| `src/core/conditional_memory.py` | Store multi-head scores as Dict[str, float] (already designed) |

---

## 12. Workstream 9: Memory-Augmented Preference Learning

### 12.1 Motivation (Engram §4.2, Table 1)

Engram's most surprising finding: memory improved *reasoning* more than knowledge retrieval (BBH +5.0 vs MMLU +3.4). The mechanism: by offloading static patterns, the network has more effective depth for compositional reasoning.

ThinkingTrees' GenRM tournament pipeline spends significant compute evaluating candidates that are obviously poor — summaries that drop all named entities, or summaries that are trivially just the input text repeated.

### 12.2 Design

#### Memory-Based Candidate Pre-Filter

```python
# Extension to src/training/preference/collector.py:

class MemoryAugmentedPreferenceCollector(PreferenceCollector):
    """
    Preference collector with memory-based pre-filtering.

    Before sending candidates to the expensive GenRM judge,
    uses ConditionalMemory to instantly reject obviously bad candidates.
    This frees up GenRM capacity for genuinely competitive comparisons.

    Analogous to Engram relieving early layers from static pattern
    reconstruction so deeper layers can focus on reasoning (§6.1).
    """

    def __init__(
        self,
        memory: ConditionalMemory,
        rejection_threshold: float = 0.3,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.memory = memory
        self.rejection_threshold = rejection_threshold

    async def collect_preferences(
        self,
        candidates: List[str],
        original_text: str,
        rubric: str,
    ) -> List[PreferencePair]:
        """Collect preferences with memory-based pre-filtering."""

        # Phase 1: Memory-based instant quality check
        # (analogous to Engram's O(1) lookup)
        viable_candidates = []
        rejected_count = 0

        for candidate in candidates:
            # Check entity recall (free, no LLM)
            entity_recall = compute_entity_recall(original_text, candidate)
            if entity_recall < self.rejection_threshold:
                rejected_count += 1
                continue

            # Check memory for known-bad patterns
            entry = self.memory.lookup(candidate)
            if entry and entry.scores.get('quality', 1.0) < self.rejection_threshold:
                rejected_count += 1
                continue

            viable_candidates.append(candidate)

        # Phase 2: Full GenRM evaluation on remaining candidates
        # (analogous to Engram's deep computation path)
        if len(viable_candidates) < 2:
            # Not enough candidates survived filtering
            viable_candidates = candidates[:2]  # Fallback to first two

        preferences = await super().collect_preferences(
            viable_candidates, original_text, rubric
        )

        # Store quality signals in memory for future filtering
        for pref in preferences:
            self.memory.store(
                text=pref.text_a,
                scores={"quality": 1.0 if pref.winner == "a" else 0.0},
            )
            self.memory.store(
                text=pref.text_b,
                scores={"quality": 1.0 if pref.winner == "b" else 0.0},
            )

        return preferences
```

### 12.3 Expected Impact

- **20-40% fewer GenRM calls** from filtering obviously bad candidates
- **Better preference pairs** — GenRM compares genuinely competitive candidates, producing more informative gradients for DPO/GRPO training
- **Progressive improvement** — as memory accumulates quality signals, filtering becomes more precise

---

## 13. Workstream 10: Component Sensitivity Analysis

### 13.1 Motivation (Engram §6.3, Figure 6)

Engram's ablation methodology cleanly reveals which capabilities depend on memory vs computation. Result: factual knowledge collapses (29% retained) while reading comprehension barely changes (93% retained).

ThinkingTrees has no systematic understanding of which pipeline components contribute most to final oracle accuracy.

### 13.2 Design

#### Ablation Experiment Matrix

| ID | Ablation | What it Tests | Expected Sensitivity |
|----|----------|---------------|---------------------|
| A1 | Fixed chunks (disable adaptive) | Adaptive chunking contribution | Low-Medium |
| A2 | Reduce tree to 2 levels max | Diminishing returns of depth | Medium |
| A3 | Extractive merge (first+last sentence) | Abstractive summarization value | High |
| A4 | Skip optimization (zero-shot) | DSPy optimization contribution | Medium-High |
| A5 | Halve audit budget | Audit bound tightness | Low |
| A6 | Disable prefix caching | Raw vs cached throughput | Medium (speed only) |
| A7 | Random strategy (ignore rubric) | Rubric contribution | Very High |
| A8 | Remove enrichment layer | Enrichment value (after WS4) | Medium |
| A9 | Disable context-aware gating | Gating value (after WS3) | Medium |
| A10 | Clear ConditionalMemory | Memory contribution (after WS1) | High for repeat corpus |

#### Measurement Protocol

```python
# New file: experiments/sensitivity_analysis.py

"""
Component Sensitivity Analysis for ThinkingTrees.

Inspired by Engram's ablation methodology (§6.3, Figure 6).
For each component, measure "retained performance" =
    (ablated_accuracy / full_accuracy) × 100%.

A component with low retained performance is load-bearing.
A component with high retained performance may be over-invested.
"""

@dataclass
class AblationResult:
    ablation_id: str
    ablation_description: str
    full_pipeline_mae: float          # Full pipeline MAE (lower is better)
    ablated_mae: float                # Ablated pipeline MAE
    retained_performance: float       # (1 - ablated_mae/full_mae) * 100 or equiv
    throughput_ratio: float           # ablated_throughput / full_throughput

def run_sensitivity_suite(
    corpus: List[DocumentSample],
    n_runs: int = 3,  # Average over multiple runs for stability
) -> List[AblationResult]:
    """Run all ablations and report retained performance."""

    # Baseline: full pipeline
    baseline = run_full_pipeline(corpus)

    results = []
    for ablation in ABLATION_CONFIGS:
        ablated = run_ablated_pipeline(corpus, ablation)
        results.append(AblationResult(
            ablation_id=ablation.id,
            ablation_description=ablation.description,
            full_pipeline_mae=baseline.mae,
            ablated_mae=ablated.mae,
            retained_performance=compute_retained(baseline, ablated),
            throughput_ratio=ablated.throughput / baseline.throughput,
        ))

    return results
```

### 13.3 Visualization

Produce a horizontal bar chart (Figure 6 style) showing retained performance per component, ordered from most to least sensitive. This immediately reveals where compute is well-spent and where it's wasted.

### 13.4 Expected Findings (Hypotheses)

Based on the Engram parallel:
- **High sensitivity (< 60% retained):** Oracle scoring, rubric content, tree building merges — these are the "factual knowledge" analog, core to the pipeline's function
- **Medium sensitivity (60-80% retained):** DSPy optimization, adaptive chunking — useful but not essential
- **Low sensitivity (> 80% retained):** Audit budget beyond minimum, tree depth beyond 3 levels — like reading comprehension, the backbone (LLM) handles these regardless

### 13.5 Files to Create

| File | Description |
|------|-------------|
| `experiments/sensitivity_analysis.py` | Ablation runner |
| `experiments/sensitivity_plots.py` | Figure 6-style visualization |

---

## 14. Implementation Roadmap

### Phase 1: Foundations (Weeks 1-2)

**Goal:** Non-breaking additions that provide immediate value.

| Priority | Workstream | Effort | Dependencies |
|----------|-----------|--------|-------------|
| P0-a | WS2: Chunk Canonicalization | 1-2 days | None |
| P0-b | WS1: ConditionalMemory (L1 only, no SQLite) | 3-4 days | None |
| P0-c | WS10: Component Sensitivity Analysis | 2-3 days | None |

**Deliverables:**
- `canonical_hash()` function deployed across all cache paths
- In-memory ConditionalMemory wired as additional layer
- First sensitivity analysis results guiding where to invest compute
- Measured cache hit rate improvement from canonicalization

### Phase 2: Memory System (Weeks 3-4)

**Goal:** Persistent, tiered memory that accumulates knowledge across runs.

| Priority | Workstream | Effort | Dependencies |
|----------|-----------|--------|-------------|
| P1-a | WS1: ConditionalMemory (SQLite L2 tier) | 2-3 days | Phase 1 |
| P1-b | WS7: Zipfian Tier Manager | 2 days | WS1 L2 |
| P1-c | WS1: Migrate existing caches to ConditionalMemory | 3-4 days | WS1 L2 |

**Deliverables:**
- Persistent cross-run memory (SQLite-backed)
- Automatic L2→L1 promotion based on access frequency
- Unified cache statistics dashboard
- Existing caches delegating to ConditionalMemory

### Phase 3: Compute Optimization (Weeks 5-7)

**Goal:** Reduce unnecessary LLM calls without sacrificing quality.

| Priority | Workstream | Effort | Dependencies |
|----------|-----------|--------|-------------|
| P2-a | WS3: Gated Strategy Selection | 4-5 days | WS1 (for memory-backed cheap strategy) |
| P2-b | WS4: Pre-Merge Enrichment Layer | 3-4 days | Embedding model available |
| P2-c | WS6: Deterministic Batch Planning | 3-4 days | None |

**Deliverables:**
- GatedStrategy routing easy merges to template/cache path
- Enrichment metadata attached to chunks before merging
- Merge schedule pre-computation with prompt prefetching
- Measured LLM call reduction and throughput improvement

### Phase 4: Advanced Features (Weeks 8-10)

**Goal:** Richer signals and systematic evaluation.

| Priority | Workstream | Effort | Dependencies |
|----------|-----------|--------|-------------|
| P3-a | WS8: Multi-Head Oracle Scoring | 3-4 days | WS1 (for multi-head storage) |
| P3-b | WS9: Memory-Augmented Preference Learning | 2-3 days | WS1, WS8 |
| P3-c | WS5: Sparsity Allocation Experiments | 3-4 days | WS1 (for memory-augmented processing) |
| P3-d | WS1: Context-Aware Gating (embedding-based) | 2-3 days | Embedding model |

**Deliverables:**
- Multi-head oracle diagnostics ("entity loss" vs "sentiment drift" vs "topic gap")
- Memory-based pre-filtering in preference collection
- U-shaped allocation curve for the manifesto corpus
- Context-aware gating that suppresses stale/mismatched cached values

### Phase 5: Integration & Optimization (Weeks 11-12)

**Goal:** End-to-end integration, measurement, documentation.

| Task | Effort | Dependencies |
|------|--------|-------------|
| Full pipeline integration test | 2-3 days | All phases |
| Gate weight learning from collected data | 2 days | Phase 3 data |
| Corpus-specific warmup scripts | 1-2 days | Phase 2 |
| Documentation and AGENTS.md updates | 1 day | All phases |
| Re-run sensitivity analysis with all features | 2 days | All phases |

---

## 15. Measurement & Evaluation Framework

### 15.1 Key Metrics

| Metric | Baseline Source | Target | How Measured |
|--------|----------------|--------|-------------|
| Cache hit rate (L1) | 0% (no persistent cache) | 60-80% on repeat corpus | `ConditionalMemory.stats.l1_hit_rate` |
| Cache hit rate (L1+L2) | 0% | 80-95% on repeat corpus | `ConditionalMemory.stats.hit_rate` |
| Oracle calls per document | Current count | -30% to -50% | Token counter in pipeline |
| LLM merge calls per document | Current count | -30% to -50% | Gate routing statistics |
| Pipeline throughput (docs/min) | Current | +20% to +40% | Wall clock time |
| Oracle MAE (primary metric) | Current | Same or better | Standard eval |
| Entity recall (new metric) | Not tracked | > 0.85 average | Multi-head scoring |
| Gate suppression rate | N/A | < 10% | Context-aware gating stats |

### 15.2 A/B Testing Protocol

For each workstream, run the pipeline with and without the feature on the same corpus (split by document for statistical power). Report:
- **Quality:** Oracle MAE, entity recall, coverage
- **Efficiency:** LLM tokens consumed, wall clock time, cache hit rate
- **Robustness:** Variance across runs, sensitivity to hyperparameters

### 15.3 Logging Additions

Add to `src/training/run_pipeline.py`:

```python
# At the end of each pipeline run:
logger.info("=== ConditionalMemory Report ===")
logger.info(json.dumps(memory.report(), indent=2))

logger.info("=== Gate Routing Report ===")
logger.info(f"Cheap merges: {gate_stats.cheap_count} ({gate_stats.cheap_pct:.1f}%)")
logger.info(f"Expensive merges: {gate_stats.expensive_count} ({gate_stats.expensive_pct:.1f}%)")

logger.info("=== Multi-Head Score Distribution ===")
for head_name, scores in multi_head_scores.items():
    logger.info(f"  {head_name}: mean={np.mean(scores):.3f}, std={np.std(scores):.3f}")
```

---

## 16. Appendix: Engram-ThinkingTrees Concept Mapping

| Engram Concept | Engram Reference | ThinkingTrees Analog | Workstream |
|----------------|-----------------|---------------------|------------|
| Conditional memory (first-class primitive) | §1, §2 | ConditionalMemory module | WS1 |
| Tokenizer compression (NFKC + casefold) | §2.2, Appendix C | Chunk canonicalization | WS2 |
| Multi-head hashing | §2.2, Eq. 2 | Multi-head oracle scoring | WS8 |
| Context-aware gating (α_t) | §2.3, Eq. 4 | Gated strategy selection | WS3 |
| Context-aware gating (cache suppression) | §2.3 | ConditionalMemory gate_threshold | WS1 |
| Multi-branch integration | §2.4 | Multi-strategy routing | WS3 |
| Deterministic addressing + prefetch | §2.5, Figure 2b | Deterministic batch planning | WS6 |
| Zipfian cache hierarchy | §2.5 | Tiered L1/L2 with promotion | WS7 |
| U-shaped allocation law | §3.1, Figure 3 | Sparsity allocation experiments | WS5 |
| Infinite memory regime scaling | §3.2 | Persistent L2 scaling | WS1/WS7 |
| Effective depth increase | §6.1, Figure 4 | Pre-merge enrichment layer | WS4 |
| Layer sensitivity sweep | §6.2, Figure 5 | Component sensitivity analysis | WS10 |
| Sensitivity analysis (retained performance) | §6.3, Figure 6 | Ablation experiment matrix | WS10 |
| Gating visualization | §6.5, Figure 7 | Gate routing statistics + visualization | WS3 |
| Offloading to host memory | §6.4, Table 4 | SQLite persistent tier | WS1 |
| Memory-augmented preference learning | §4.2 (reasoning gains) | Pre-filter for GenRM tournaments | WS9 |

---

## References

- Cheng, X., Zeng, W., Dai, D., et al. (2026). Conditional Memory via Scalable Lookup: A New Axis of Sparsity for Large Language Models. arXiv:2601.07372v1.
- ThinkingTrees ARCHITECTURE.md
- ThinkingTrees AGENTS.md
- ThinkingTrees config/settings.yaml
