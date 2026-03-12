"""Tests for parser router contracts and dispatch behavior."""

from src.parsers.router import (
    PARSER_ROUTER_CONTRACT_VERSION,
    PARSER_ROUTER_RESPONSE_TYPE,
    ParserRouter,
    ParserRouterConfig,
    ExternalJSONHintProcessor,
)


def _sample_with_hint():
    return {
        "doc_id": "doc-1",
        "modality": "text",
        "text": "old page text",
        "pages": ["old page text"],
        "metadata": {
            "source_path": "dummy.pdf",
            "parser_feedback": {
                "axis_hints": [
                    {
                        "axis_unit": "page",
                        "start": 0,
                        "end": 1,
                        "source": "parser:pdf_needs_ocr",
                        "action": "ocr_first_then_vision_embedding",
                        "recommended_processors": ["ocr"],
                    }
                ]
            },
            "page_assets": [
                {
                    "page_index": 0,
                    "page_number": 1,
                    "page_uri": "pdf://dummy.pdf#page=1",
                    "image_refs": ["xref:12"],
                    "image_count": 1,
                }
            ],
            "page_char_ranges": [[0, 13]],
            "axis_char_ranges": {"page": [[0, 13]]},
        },
    }


def test_external_processor_requires_contract_in_strict_mode(monkeypatch):
    processor = ExternalJSONHintProcessor("ocr", endpoint="http://example.test")

    def _fake_post_json(_url, _payload, *, timeout_seconds):
        assert timeout_seconds == 5
        return {"status": "ok", "page_text": "new"}  # missing contract keys

    monkeypatch.setattr("src.parsers.router._post_json", _fake_post_json)
    result = processor.process(
        sample=_sample_with_hint(),
        hint=_sample_with_hint()["metadata"]["parser_feedback"]["axis_hints"][0],
        hint_index=0,
        timeout_seconds=5,
        max_retries=0,
        retry_backoff_seconds=0.0,
        strict_contracts=True,
        contract_version=PARSER_ROUTER_CONTRACT_VERSION,
    )

    assert result.status == "invalid_response_contract"


def test_external_processor_accepts_valid_contract_and_updates_text(monkeypatch):
    processor = ExternalJSONHintProcessor("ocr", endpoint="http://example.test")
    sample = _sample_with_hint()

    def _fake_post_json(_url, payload, *, timeout_seconds):
        assert payload["contract_version"] == PARSER_ROUTER_CONTRACT_VERSION
        assert payload["request_type"]
        assert payload["sample"]["page_assets"][0]["page_uri"].startswith("pdf://")
        assert timeout_seconds == 5
        return {
            "contract_version": PARSER_ROUTER_CONTRACT_VERSION,
            "response_type": PARSER_ROUTER_RESPONSE_TYPE,
            "action": "ocr",
            "status": "ok",
            "page_text": "new page text",
        }

    monkeypatch.setattr("src.parsers.router._post_json", _fake_post_json)
    result = processor.process(
        sample=sample,
        hint=sample["metadata"]["parser_feedback"]["axis_hints"][0],
        hint_index=0,
        timeout_seconds=5,
        max_retries=0,
        retry_backoff_seconds=0.0,
        strict_contracts=True,
        contract_version=PARSER_ROUTER_CONTRACT_VERSION,
    )

    assert result.status == "applied"
    assert sample["pages"][0] == "new page text"
    assert sample["text"] == "new page text"


def test_external_processor_retries_then_succeeds(monkeypatch):
    processor = ExternalJSONHintProcessor("ocr", endpoint="http://example.test")
    sample = _sample_with_hint()
    calls = {"count": 0}

    def _fake_post_json(_url, _payload, *, timeout_seconds):
        assert timeout_seconds == 5
        calls["count"] += 1
        if calls["count"] == 1:
            raise RuntimeError("temporary failure")
        return {
            "contract_version": PARSER_ROUTER_CONTRACT_VERSION,
            "response_type": PARSER_ROUTER_RESPONSE_TYPE,
            "action": "ocr",
            "status": "ok",
            "page_text": "retry success",
        }

    monkeypatch.setattr("src.parsers.router._post_json", _fake_post_json)
    result = processor.process(
        sample=sample,
        hint=sample["metadata"]["parser_feedback"]["axis_hints"][0],
        hint_index=0,
        timeout_seconds=5,
        max_retries=2,
        retry_backoff_seconds=0.0,
        strict_contracts=True,
        contract_version=PARSER_ROUTER_CONTRACT_VERSION,
    )

    assert result.status == "applied"
    assert calls["count"] == 2
    assert result.metadata.get("attempt_count") == 2


def test_router_dispatches_actions_and_updates_summary(monkeypatch):
    sample = _sample_with_hint()

    def _fake_post_json(_url, _payload, *, timeout_seconds):
        assert timeout_seconds == 5
        return {
            "contract_version": PARSER_ROUTER_CONTRACT_VERSION,
            "response_type": PARSER_ROUTER_RESPONSE_TYPE,
            "action": "ocr",
            "status": "ok",
            "page_text": "router text",
        }

    monkeypatch.setattr("src.parsers.router._post_json", _fake_post_json)

    router = ParserRouter(
        ParserRouterConfig(
            enabled=True,
            enabled_processors=("ocr",),
            ocr_endpoint="http://example.test",
            timeout_seconds=5,
            max_concurrency=2,
            max_retries=1,
            retry_backoff_seconds=0.0,
            strict_contracts=True,
        )
    )
    summary = router.route_sample(sample)

    assert summary["actions_attempted"] == 1
    assert summary["applied"] == 1
    assert summary["errors"] == 0
    assert sample["metadata"]["parser_router"]["last_run"]["summary"]["applied"] == 1
