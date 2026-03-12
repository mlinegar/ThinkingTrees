from src.runtime.backbone import BackboneAdapter, BackboneConfig


def test_runtime_backbone_mock_parity_vllm_vs_sglang():
    messages = [{"role": "user", "content": "Summarize this document."}]

    vllm_adapter = BackboneAdapter(
        config=BackboneConfig(base_url="http://localhost:8000/v1", model="mock-model"),
        mock=True,
        enable_cache=False,
    )
    sglang_adapter = BackboneAdapter(
        config=BackboneConfig(base_url="http://localhost:30000/v1", model="mock-model"),
        mock=True,
        enable_cache=False,
    )

    vllm_resp = vllm_adapter.generate(messages, max_tokens=64)
    sglang_resp = sglang_adapter.generate(messages, max_tokens=64)

    assert vllm_resp.text == sglang_resp.text
    assert vllm_resp.model_id == "mock"
    assert sglang_resp.model_id == "mock"
