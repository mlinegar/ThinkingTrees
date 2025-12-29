"""
LLM utility helpers.
"""

import requests


def get_vllm_model_name(port: int, fallback: str = "default") -> str:
    """
    Query vLLM server to get the actual model name.

    Args:
        port: vLLM server port
        fallback: Fallback model name if query fails

    Returns:
        Model name from server or fallback
    """
    try:
        response = requests.get(
            f"http://localhost:{port}/v1/models",
            timeout=5,
        )
        response.raise_for_status()
        model_info = response.json()
        if model_info.get("data") and len(model_info["data"]) > 0:
            return model_info["data"][0]["id"]
    except Exception:
        pass
    return fallback
