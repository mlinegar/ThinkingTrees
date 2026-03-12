"""Tests for profile-aware vLLM runtime flag resolution."""

from src.core.vllm_runtime import resolve_vllm_runtime_flags


def test_resolve_vllm_runtime_flags_profile_overrides():
    vllm_cfg = {
        "runtime": {
            "enforce_eager_default": False,
            "enforce_eager_profiles": ["glm-4.6"],
            "api_server_count": 2,
            "mm_processor_cache_gb": 3.5,
            "profile_overrides": {
                "qwen-vl-235b": {
                    "limit_mm_per_prompt": {"image": 8, "video": 2},
                    "allowed_media_domains": ["cdn.example.com", "img.example.com"],
                    "interleave_mm_strings": True,
                }
            },
        }
    }

    regular = resolve_vllm_runtime_flags(vllm_cfg=vllm_cfg, profile="nemotron-30b-nvfp4")
    assert regular.enforce_eager is False
    assert regular.api_server_count == 2
    assert regular.mm_processor_cache_gb == 3.5
    assert regular.limit_mm_per_prompt is None

    glm = resolve_vllm_runtime_flags(vllm_cfg=vllm_cfg, profile="glm-4.6")
    assert glm.enforce_eager is True

    vl = resolve_vllm_runtime_flags(vllm_cfg=vllm_cfg, profile="qwen-vl-235b")
    assert vl.limit_mm_per_prompt == {"image": 8, "video": 2}
    assert vl.allowed_media_domains == ("cdn.example.com", "img.example.com")
    assert vl.interleave_mm_strings is True
    cli_args = vl.to_cli_args()
    assert "--limit-mm-per-prompt" in cli_args
    assert "--allowed-media-domains" in cli_args
    assert "--interleave-mm-strings" in cli_args


def test_resolve_vllm_runtime_flags_explicit_eager_override():
    vllm_cfg = {
        "runtime": {
            "enforce_eager_default": True,
            "profile_overrides": {
                "nemotron-30b-nvfp4": {
                    "enforce_eager": False,
                    "disable_frontend_multiprocessing": True,
                    "extra_flags": ["--disable-log-stats"],
                }
            },
        }
    }

    flags = resolve_vllm_runtime_flags(vllm_cfg=vllm_cfg, profile="nemotron-30b-nvfp4")
    assert flags.enforce_eager is False
    assert flags.disable_frontend_multiprocessing is True
    assert flags.extra_flags == ("--disable-log-stats",)

    cli_args = flags.to_cli_args()
    assert "--enforce-eager" not in cli_args
    assert "--disable-frontend-multiprocessing" in cli_args
    assert "--disable-log-stats" in cli_args
