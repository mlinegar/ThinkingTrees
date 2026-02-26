from __future__ import annotations

import time
from dataclasses import dataclass, replace
from typing import Any, Dict, List, Optional

from src.core.llm_client import LLMClient, LLMConfig, MockLLMClient
from src.runtime.contracts import ModelResponse


@dataclass(frozen=True)
class BackboneConfig:
    base_url: str = "http://localhost:8000/v1"
    model: str = "default"
    api_key: str = "EMPTY"
    temperature: float = 0.0
    timeout: float = 120.0


class BackboneAdapter:
    """Thin wrapper around `src.core.llm_client.LLMClient` with a stable interface."""

    def __init__(
        self,
        *,
        config: BackboneConfig,
        mock: bool = False,
        enable_cache: bool = True,
    ):
        resolved_model = config.model
        if resolved_model == "default" and "api.openai.com" not in config.base_url:
            from src.core.model_detection import detect_model_sync

            resolved_model = detect_model_sync(config.base_url, fallback="default", timeout=2.0)

        self.config = replace(config, model=resolved_model)
        llm_cfg = LLMConfig(
            base_url=self.config.base_url,
            model=self.config.model,
            api_key=self.config.api_key,
            temperature=self.config.temperature,
            timeout=self.config.timeout,
        )
        if mock:
            self.client = MockLLMClient(llm_cfg)
        else:
            self.client = LLMClient(llm_cfg, enable_cache=enable_cache)

    def model_id(self) -> str:
        return self.config.model

    def generate(
        self,
        messages: List[Dict[str, str]],
        *,
        max_tokens: int,
        temperature: Optional[float] = None,
        stop: Optional[List[str]] = None,
        extra: Optional[Dict[str, Any]] = None,
    ) -> ModelResponse:
        start = time.time()
        kwargs: Dict[str, Any] = {
            "max_tokens": int(max_tokens),
            "temperature": float(self.config.temperature if temperature is None else temperature),
        }
        if stop:
            kwargs["stop"] = stop
        if extra:
            kwargs.update(extra)

        resp = self.client.chat(messages, **kwargs)
        latency_ms = (time.time() - start) * 1000.0
        return ModelResponse(
            text=resp.content,
            model_id=resp.model,
            prompt_tokens=getattr(resp, "prompt_tokens", 0) or 0,
            completion_tokens=getattr(resp, "completion_tokens", 0) or 0,
            latency_ms=latency_ms,
            raw=None,
        )
