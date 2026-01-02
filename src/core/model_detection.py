"""
Centralized model detection for vLLM servers.

This module consolidates model auto-detection logic that was previously
duplicated in llm_client.py, batch_processor.py, and llm_utils.py.
"""

import logging
from typing import Optional

import aiohttp
import requests

logger = logging.getLogger(__name__)


def detect_model_sync(
    base_url: str,
    fallback: str = "default",
    timeout: float = 5.0,
) -> str:
    """
    Synchronously detect vLLM model name from server.

    Args:
        base_url: Base URL of the vLLM server (e.g., "http://localhost:8000/v1")
        fallback: Model name to return if detection fails
        timeout: Request timeout in seconds

    Returns:
        Detected model ID or fallback
    """
    try:
        response = requests.get(f"{base_url}/models", timeout=timeout)
        response.raise_for_status()
        data = response.json()
        if data.get("data") and len(data["data"]) > 0:
            model_id = data["data"][0]["id"]
            logger.debug(f"Auto-detected model: {model_id}")
            return model_id
    except requests.RequestException as e:
        logger.debug(f"Failed to auto-detect model from {base_url}: {e}")
    except (KeyError, IndexError, ValueError) as e:
        logger.debug(f"Failed to parse model response: {e}")
    return fallback


async def detect_model_async(
    base_url: str,
    fallback: str = "default",
    timeout: float = 5.0,
) -> str:
    """
    Asynchronously detect vLLM model name from server.

    Args:
        base_url: Base URL of the vLLM server (e.g., "http://localhost:8000/v1")
        fallback: Model name to return if detection fails
        timeout: Request timeout in seconds

    Returns:
        Detected model ID or fallback
    """
    try:
        timeout_config = aiohttp.ClientTimeout(total=timeout)
        async with aiohttp.ClientSession(timeout=timeout_config) as session:
            async with session.get(f"{base_url}/models") as resp:
                if resp.status == 200:
                    data = await resp.json()
                    if data.get("data") and len(data["data"]) > 0:
                        model_id = data["data"][0]["id"]
                        logger.info(f"Auto-detected model: {model_id}")
                        return model_id
    except aiohttp.ClientError as e:
        logger.warning(f"Failed to auto-detect model from {base_url}: {e}")
    except (KeyError, IndexError, ValueError) as e:
        logger.warning(f"Failed to parse model response: {e}")
    return fallback


def detect_model_from_port(
    port: int = 8000,
    host: str = "localhost",
    fallback: str = "default",
    timeout: float = 5.0,
) -> str:
    """
    Convenience function to detect model from host:port.

    Args:
        port: vLLM server port
        host: vLLM server host
        fallback: Model name to return if detection fails
        timeout: Request timeout in seconds

    Returns:
        Detected model ID or fallback
    """
    base_url = f"http://{host}:{port}/v1"
    return detect_model_sync(base_url, fallback, timeout)
