from concurrent.futures import ThreadPoolExecutor
import sqlite3

from src.core.conditional_memory import ConditionalMemory, ConditionalMemoryConfig, canonical_hash


def test_canonical_hash_is_mild_by_default():
    # Whitespace canonicalization
    assert canonical_hash("A\tB\nC") == canonical_hash("A B C")
    # NFKC canonicalization (e.g., fullwidth → ASCII)
    assert canonical_hash("Ａ") == canonical_hash("A")
    # No casefold by default
    assert canonical_hash("ABC") != canonical_hash("abc")
    # No accent stripping by default
    assert canonical_hash("résumé") != canonical_hash("resume")


def test_conditional_memory_persists_across_instances(tmp_path):
    db_path = tmp_path / "memory.db"
    memory_1 = ConditionalMemory(sqlite_path=db_path, l1_capacity=2, mode="readwrite")
    try:
        memory_1.set_text("ns", "k_text", "value_1")
        memory_1.set_json("ns", "k_json", {"x": 1, "y": [2, 3]})
    finally:
        memory_1.close()

    memory_2 = ConditionalMemory(sqlite_path=db_path, l1_capacity=2, mode="read")
    try:
        assert memory_2.get_text("ns", "k_text") == "value_1"
        assert memory_2.get_json("ns", "k_json") == {"x": 1, "y": [2, 3]}

        # Read-only mode should not persist writes.
        memory_2.set_text("ns", "k_text_2", "value_2")
        assert memory_2.get_text("ns", "k_text_2") is None
    finally:
        memory_2.close()


def test_conditional_memory_l1_l2_hit_accounting(tmp_path):
    db_path = tmp_path / "memory.db"
    memory_1 = ConditionalMemory(sqlite_path=db_path, l1_capacity=1, mode="readwrite")
    try:
        memory_1.set_text("ns", "k", "v")
    finally:
        memory_1.close()

    memory_2 = ConditionalMemory(sqlite_path=db_path, l1_capacity=1, mode="read")
    try:
        assert memory_2.stats.l1_hits == 0
        assert memory_2.stats.l2_hits == 0

        assert memory_2.get_text("ns", "k") == "v"
        assert memory_2.stats.l2_hits == 1
        assert memory_2.stats.l1_hits == 0

        assert memory_2.get_text("ns", "k") == "v"
        assert memory_2.stats.l1_hits == 1
    finally:
        memory_2.close()


def test_conditional_memory_max_l2_entries_eviction(tmp_path):
    db_path = tmp_path / "memory.db"
    config = ConditionalMemoryConfig(
        enabled=True,
        mode="readwrite",
        l1_capacity=1,
        l2_path=db_path,
        max_l2_entries=2,
    )
    memory = ConditionalMemory(config)
    try:
        memory.set_text("ns", "k1", "v1")
        memory.set_text("ns", "k2", "v2")
        memory.set_text("ns", "k3", "v3")
    finally:
        memory.close()

    reader = ConditionalMemory(sqlite_path=db_path, l1_capacity=1, mode="read")
    try:
        assert reader.l2_size == 2
        values = {k: reader.get_text("ns", k) for k in ("k1", "k2", "k3")}
        assert sum(1 for v in values.values() if v is not None) == 2
    finally:
        reader.close()


def test_conditional_memory_thread_safety(tmp_path):
    db_path = tmp_path / "memory.db"
    memory = ConditionalMemory(sqlite_path=db_path, l1_capacity=256, mode="readwrite")

    def _worker(worker_id: int) -> None:
        for j in range(10):
            key = f"k{worker_id}:{j}"
            value = f"v{worker_id}:{j}"
            memory.set_text("ns", key, value)
            assert memory.get_text("ns", key) == value

    try:
        with ThreadPoolExecutor(max_workers=8) as executor:
            list(executor.map(_worker, range(8)))
    finally:
        memory.close()


def test_conditional_memory_l2_shards_round_trip(tmp_path):
    config = ConditionalMemoryConfig(
        enabled=True,
        mode="readwrite",
        l1_capacity=32,
        l2_path=tmp_path / "memory.db",
        l2_shards=4,
    )
    writer = ConditionalMemory(config)
    try:
        for idx in range(32):
            writer.set_text("ns", f"k{idx}", f"v{idx}")
    finally:
        writer.close()

    shard_paths = config.resolved_l2_paths()
    assert len(shard_paths) == 4
    assert all(path.exists() for path in shard_paths)

    per_shard_counts = []
    for path in shard_paths:
        conn = sqlite3.connect(str(path))
        try:
            row = conn.execute("SELECT COUNT(*) FROM entries").fetchone()
            per_shard_counts.append(int(row[0]) if row else 0)
        finally:
            conn.close()
    assert sum(per_shard_counts) == 32
    assert sum(1 for count in per_shard_counts if count > 0) >= 2

    reader = ConditionalMemory(
        ConditionalMemoryConfig(
            enabled=True,
            mode="read",
            l1_capacity=16,
            l2_path=tmp_path / "memory.db",
            l2_shards=4,
        )
    )
    try:
        for idx in range(32):
            assert reader.get_text("ns", f"k{idx}") == f"v{idx}"
        assert reader.l2_size == 32
    finally:
        reader.close()
