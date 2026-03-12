from src.config.settings import get_inference_backend_config


def test_inference_backend_config_preserves_none_fallback():
    settings = {
        "inference": {
            "backend": {
                "task_backend": "sglang",
                "genrm_backend": "sglang",
                "fallback_backend": "none",
            }
        }
    }
    cfg = get_inference_backend_config(settings)
    assert cfg["task_backend"] == "sglang"
    assert cfg["genrm_backend"] == "sglang"
    assert cfg["fallback_backend"] == "none"


def test_inference_backend_config_normalizes_disabled_fallback(monkeypatch):
    monkeypatch.setenv("TT_FALLBACK_BACKEND", "disabled")
    cfg = get_inference_backend_config({})
    assert cfg["fallback_backend"] == "none"
