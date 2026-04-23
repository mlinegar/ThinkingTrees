from __future__ import annotations

from datetime import UTC, datetime

from src.core.autotune_probe_cache import (
    AUTOTUNE_PROBE_CACHE_VERSION,
    ProbeCacheEntry,
    ProbeCacheStore,
    ProbeRunProfile,
    build_probe_cache_key,
    classify_device_signature,
)


def test_build_probe_cache_key_is_stable_for_same_inputs() -> None:
    payload = {
        "model_signature": {"model_class": "FNOCountSketch", "state_dim": 32},
        "pack_mode": "fixed_fused",
        "topology_signature": "n4:leaf(8,8,8,8):merge(16,32,32)",
        "probe_mode": "train",
        "device_class_signature": {
            "device_name": "NVIDIA A100-SXM4-40GB MIG 1g.24gb",
            "total_memory_bytes": 24 * 1024 ** 3,
            "compute_capability": (8, 0),
            "is_mig": True,
            "mig_profile": "1g.24gb",
        },
    }

    key_a = build_probe_cache_key(**payload)
    key_b = build_probe_cache_key(**payload)

    assert key_a == key_b


def test_probe_cache_entry_rejects_old_versions() -> None:
    entry_payload = {
        "cache_key": "abc",
        "cache_version": AUTOTUNE_PROBE_CACHE_VERSION - 1,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "model_signature": {"model_class": "FNOCountSketch"},
        "pack_mode": "fixed_fused",
        "topology_signature": "n4",
        "probe_mode": "eval",
        "device_class_signature": {"device_name": "cpu"},
        "selected_docs_cap": 64,
        "run_profile": {
            "probe_mode": "eval",
            "topology_signature": "n4",
            "selected_docs_cap": 64,
            "heuristic_docs_cap": 32,
            "max_candidate_docs": 128,
            "target_fraction": 0.55,
            "cache_key": "abc",
            "cache_hit": False,
            "total_wall_time_s": 1.0,
            "stop_reason": "accepted",
            "cached_source_wall_time_s": 0.0,
            "candidate_profiles": [],
        },
    }

    assert ProbeCacheEntry.from_dict(entry_payload) is None


def test_probe_cache_store_roundtrip(tmp_path) -> None:
    store = ProbeCacheStore(root_dir=tmp_path / "probe_cache")
    device_signature = classify_device_signature(
        device_name="NVIDIA A100-SXM4-40GB MIG 1g.24gb",
        total_memory_bytes=24 * 1024 ** 3,
        capability=(8, 0),
    )
    cache_key = build_probe_cache_key(
        model_signature={"model_class": "FNOCountSketch", "state_dim": 32},
        pack_mode="fixed_fused",
        topology_signature="n4:leaf(8,8,8,8):merge(16,32,32)",
        probe_mode="train",
        device_class_signature=device_signature,
    )
    run_profile = ProbeRunProfile(
        probe_mode="train",
        topology_signature="n4:leaf(8,8,8,8):merge(16,32,32)",
        selected_docs_cap=64,
        heuristic_docs_cap=32,
        max_candidate_docs=128,
        target_fraction=0.78,
        cache_key=cache_key,
        cache_hit=False,
        total_wall_time_s=1.25,
        stop_reason="target_fraction_exceeded",
    )
    entry = ProbeCacheEntry(
        cache_key=cache_key,
        cache_version=AUTOTUNE_PROBE_CACHE_VERSION,
        created_at_utc=datetime.now(UTC).isoformat(),
        model_signature={"model_class": "FNOCountSketch", "state_dim": 32},
        pack_mode="fixed_fused",
        topology_signature="n4:leaf(8,8,8,8):merge(16,32,32)",
        probe_mode="train",
        device_class_signature=device_signature,
        selected_docs_cap=64,
        run_profile=run_profile,
    )

    path = store.put(entry)
    loaded = store.get(cache_key)

    assert path.exists()
    assert loaded is not None
    assert loaded.cache_key == cache_key
    assert loaded.selected_docs_cap == 64
    assert loaded.run_profile.stop_reason == "target_fraction_exceeded"
