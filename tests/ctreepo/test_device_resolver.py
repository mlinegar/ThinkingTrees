from __future__ import annotations

from src.ctreepo.sim.core import device_resolver


def test_resolve_devices_auto_falls_back_to_cpu_without_nvidia_smi(
    monkeypatch,
) -> None:
    def _raise(*args, **kwargs):
        raise FileNotFoundError("nvidia-smi not found")

    monkeypatch.setattr(device_resolver.subprocess, "run", _raise)

    assert device_resolver.resolve_devices(device_mode="auto") == [""]
    assert device_resolver.resolve_devices(device_mode="gpu") == []


def test_build_worker_env_sets_thread_defaults_and_visible_device(
    monkeypatch,
) -> None:
    monkeypatch.delenv("OMP_NUM_THREADS", raising=False)
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)

    env = device_resolver.build_worker_env("MIG-test", use_cuda=True)

    assert env["OMP_NUM_THREADS"] == "1"
    assert env["PYTHONUNBUFFERED"] == "1"
    assert env["CUDA_VISIBLE_DEVICES"] == "MIG-test"


def test_parse_mig_layout_from_nvidia_smi_listing() -> None:
    listing = "\n".join(
        [
            "GPU 0: GPU0 (UUID: GPU-0)",
            "  MIG 1g.24gb Device 0: (UUID: MIG-0-0)",
            "  MIG 1g.24gb Device 1: (UUID: MIG-0-1)",
            "GPU 1: GPU1 (UUID: GPU-1)",
            "  MIG 1g.24gb Device 0: (UUID: MIG-1-0)",
        ]
    )

    layout = device_resolver.parse_mig_layout_from_nvidia_smi_listing(listing)

    assert layout == [
        {"gpu_index": 0, "gpu_uuid": "GPU-0", "mig_uuid": "MIG-0-0"},
        {"gpu_index": 0, "gpu_uuid": "GPU-0", "mig_uuid": "MIG-0-1"},
        {"gpu_index": 1, "gpu_uuid": "GPU-1", "mig_uuid": "MIG-1-0"},
    ]
