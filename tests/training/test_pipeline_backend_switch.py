import argparse

import pytest

from src.training.run_pipeline import (
    apply_inference_backend_defaults,
    resolve_inference_backend_config,
)


def _args(**overrides):
    base = {
        "task_backend": None,
        "genrm_backend": None,
        "routing_policy": None,
        "backend_fallback": None,
        "sglang_venv_path": None,
        "port": 8000,
        "genrm_port": 8001,
    }
    base.update(overrides)
    return argparse.Namespace(**base)


def test_backend_switch_uses_settings_defaults_and_adjusts_ports():
    settings = {
        "inference": {
            "backend": {
                "task_backend": "sglang",
                "genrm_backend": "sglang",
                "fallback_backend": "vllm",
                "routing_policy": "document_affinity",
                "metrics_poll_seconds": 1.5,
                "sglang_venv_path": "/home/mlinegar/sglang-env",
            }
        },
        "sglang": {"port": 30000, "genrm_port": 30001},
    }
    args = _args()

    cfg = resolve_inference_backend_config(args, settings)
    apply_inference_backend_defaults(args, cfg, settings)

    assert args.task_backend == "sglang"
    assert args.genrm_backend == "sglang"
    assert args.routing_policy == "document_affinity"
    assert args.port == 30000
    assert args.genrm_port == 30001


def test_backend_switch_preserves_explicit_ports():
    settings = {
        "inference": {"backend": {"task_backend": "sglang", "genrm_backend": "sglang"}},
        "sglang": {"port": 30000, "genrm_port": 30001},
    }
    args = _args(task_backend="sglang", genrm_backend="sglang", port=31000, genrm_port=31001)

    cfg = resolve_inference_backend_config(args, settings)
    apply_inference_backend_defaults(args, cfg, settings)

    assert args.port == 31000
    assert args.genrm_port == 31001


def test_backend_switch_rejects_registered_but_unsupported_engine():
    settings = {
        "inference": {"backend": {"task_backend": "openai", "genrm_backend": "sglang"}},
    }
    args = _args()

    with pytest.raises(ValueError, match="not supported in training pipeline task backend selection"):
        resolve_inference_backend_config(args, settings)


def test_backend_switch_rejects_unknown_engine() -> None:
    settings = {"inference": {"backend": {}}}
    args = _args(task_backend="definitely_not_an_engine")

    with pytest.raises(ValueError, match="Unsupported engine"):
        resolve_inference_backend_config(args, settings)
