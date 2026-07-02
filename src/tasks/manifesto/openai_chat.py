from __future__ import annotations

from typing import Any

import requests

DEFAULT_MAIN_MODEL = "/mnt/data/models/nvidia/Qwen3.5-397B-A17B-NVFP4"


def http_error_detail(exc: Exception) -> str:
    if isinstance(exc, requests.HTTPError) and exc.response is not None:
        body = str(exc.response.text or "").strip().replace("\n", " ")
        if len(body) > 400:
            body = body[:400] + "..."
        return f"status={exc.response.status_code} body={body}"
    return str(exc)


class OpenAIChatClient:
    """Minimal OpenAI-compatible chat client used by Manifesto scripts."""

    def __init__(
        self,
        *,
        base_url: str,
        model: str,
        api_key: str = "EMPTY",
        timeout_seconds: float = 180.0,
        enable_thinking: bool = False,
        max_connections: int | None = None,
    ):
        self.base_url = str(base_url).rstrip("/")
        self.model = str(model)
        self.api_key = str(api_key)
        self.timeout_seconds = float(timeout_seconds)
        self.enable_thinking = bool(enable_thinking)
        self._session = requests.Session()
        if max_connections is not None:
            adapter = requests.adapters.HTTPAdapter(
                pool_connections=max(1, int(max_connections)),
                pool_maxsize=max(1, int(max_connections)),
                max_retries=0,
            )
            self._session.mount("http://", adapter)
            self._session.mount("https://", adapter)

    def chat(
        self,
        *,
        system: str,
        user: str,
        temperature: float,
        max_tokens: int,
    ) -> str:
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": float(temperature),
            "max_tokens": int(max_tokens),
            "chat_template_kwargs": {
                "enable_thinking": bool(self.enable_thinking),
            },
        }
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        response = self._session.post(
            f"{self.base_url}/chat/completions",
            json=payload,
            headers=headers,
            timeout=self.timeout_seconds,
        )
        response.raise_for_status()
        data = response.json()
        choices = data.get("choices") or []
        if not choices:
            return ""
        message = choices[0].get("message") or {}
        return str(message.get("content") or "").strip()


__all__ = ["DEFAULT_MAIN_MODEL", "OpenAIChatClient", "http_error_detail"]
