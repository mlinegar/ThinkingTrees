from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

import pytest

from src.core.engines import EngineRegistry, EngineSurface, EngineType
from src.core.url_utils import normalize_generate_base_url
from src.diffusion.backends import DiffusionBatchResponse, DiffusionGeneration
from src.training.supervision.types import SupervisionDataset
from src.tree.treepo_stack import (
    OracleLaneSpec,
    SupervisionSourceSpec,
    TreePOContractSpec,
    TreePOLocalLawConfig,
    TreePOModelSpec,
    build_treepo_stack,
)


class _QueueBackend:
    def __init__(self, queued_outputs: Sequence[Sequence[str]]) -> None:
        self.backend_name = "mock_backend"
        self._queued_outputs = [list(batch) for batch in queued_outputs]
        self.calls: List[Dict[str, Any]] = []

    def generate(
        self,
        texts: Sequence[str] | str,
        sampling_params: Optional[Mapping[str, Any]] = None,
        engine_options: Optional[Mapping[str, Any]] = None,
    ) -> DiffusionBatchResponse:
        prompts = [texts] if isinstance(texts, str) else list(texts)
        outputs = self._queued_outputs.pop(0)
        assert len(outputs) == len(prompts)
        self.calls.append(
            {
                "texts": prompts,
                "sampling_params": dict(sampling_params or {}),
                "engine_options": dict(engine_options or {}),
            }
        )
        return DiffusionBatchResponse(
            generations=[
                DiffusionGeneration(input_text=prompt, output_text=output)
                for prompt, output in zip(prompts, outputs)
            ],
            latency_seconds=0.01,
            request_payload={"text": prompts},
            raw_response={"text": outputs},
        )


class _DeterministicEmbeddingClient:
    def __init__(self, dim: int = 2) -> None:
        self.dim = int(dim)

    def resolve_model(self) -> str:
        return "fake-embedding-model"

    def embed_texts(self, texts):
        out = []
        for text in [str(t) for t in texts]:
            n = float(len(text))
            s = float(sum(ord(ch) for ch in text) % 100)
            vec = [n, s]
            if self.dim > 2:
                vec = vec + [0.0] * (self.dim - 2)
            out.append([float(x) for x in vec[: self.dim]])
        return out


def _simple_contract(*, rubric: str = "Keep exact theorem content.") -> TreePOContractSpec:
    return TreePOContractSpec(
        rubric=rubric,
        oracle_lane=OracleLaneSpec(
            kind="provided_scoring_oracle",
            import_path="src.tree.auditor:SimpleScorer",
        ),
    )


def test_surface_selection_prefers_generate_and_falls_back_to_chat() -> None:
    contract = _simple_contract()

    stack_generate = build_treepo_stack(
        TreePOModelSpec(engine="sglang", surface="generate", prefer_generate=True),
        contract,
    )
    assert stack_generate.surface is EngineSurface.DIFFUSION_GENERATE
    assert stack_generate.surface_fallback_reason is None

    stack_fallback = build_treepo_stack(
        TreePOModelSpec(engine="vllm", surface="generate", prefer_generate=True),
        contract,
    )
    assert stack_fallback.surface is EngineSurface.CHAT_OPENAI
    assert stack_fallback.surface_fallback_reason == "engine_missing_generate_surface"


def test_model_spec_engine_auto_infers_engine_from_base_url() -> None:
    contract = _simple_contract()

    stack_vllm = build_treepo_stack(
        TreePOModelSpec(engine="auto", base_url="http://localhost:8000/v1", surface="generate", prefer_generate=True),
        contract,
    )
    assert stack_vllm.engine is EngineType.VLLM
    assert stack_vllm.surface is EngineSurface.CHAT_OPENAI
    assert stack_vllm.surface_fallback_reason == "engine_missing_generate_surface"

    stack_sglang = build_treepo_stack(
        TreePOModelSpec(engine="auto", base_url="http://localhost:30000/v1", surface="generate", prefer_generate=True),
        contract,
    )
    assert stack_sglang.engine is EngineType.SGLANG
    assert stack_sglang.surface is EngineSurface.DIFFUSION_GENERATE
    # generate-first: strip /v1 for generate base_url construction
    assert stack_sglang.base_url == "http://localhost:30000"


def test_generate_base_url_normalization_avoids_double_generate_suffix() -> None:
    contract = _simple_contract()

    spec = EngineRegistry.resolve(EngineType.SGLANG)
    generate_path = str(spec.diffusion_generate_path or "/generate")
    default_full = spec.default_base_url(surface=EngineSurface.DIFFUSION_GENERATE)
    assert default_full is not None
    expected_root = normalize_generate_base_url(default_full, generate_path=generate_path)

    stack_default = build_treepo_stack(TreePOModelSpec(engine="sglang", surface="generate"), contract)
    backend = stack_default.inference_engine.backend
    assert stack_default.base_url == expected_root
    assert backend.base_url == expected_root
    assert backend.generate_path == generate_path
    assert f"{backend.base_url}{backend.generate_path}" == default_full

    stack_with_suffix = build_treepo_stack(
        TreePOModelSpec(engine="sglang", surface="generate", base_url=default_full),
        contract,
    )
    backend2 = stack_with_suffix.inference_engine.backend
    assert backend2.base_url == expected_root
    assert f"{backend2.base_url}{backend2.generate_path}" == default_full

    stack_with_root = build_treepo_stack(
        TreePOModelSpec(engine="sglang", surface="generate", base_url=expected_root),
        contract,
    )
    backend3 = stack_with_root.inference_engine.backend
    assert backend3.base_url == expected_root
    assert f"{backend3.base_url}{backend3.generate_path}" == default_full


def test_supervision_source_csv_builds_and_saves_supervision_dataset(tmp_path: Path) -> None:
    csv_path = tmp_path / "data.csv"
    csv_path.write_text("id,text,label\n1,alpha,0.25\n2,beta,0.75\n")
    save_path = tmp_path / "supervision_dataset.json"

    stack = build_treepo_stack(
        TreePOModelSpec(engine="sglang", surface="generate"),
        TreePOContractSpec(
            rubric="Document score objective.",
            supervision_source=SupervisionSourceSpec(
                kind="csv",
                path=str(csv_path),
                text_column="text",
                label_column="label",
                example_id_column="id",
                rubric="Document score objective.",
                save_path=str(save_path),
            ),
            oracle_lane=OracleLaneSpec(
                kind="provided_scoring_oracle",
                import_path="src.tree.auditor:SimpleScorer",
            ),
        ),
    )
    assert stack.capabilities["supervision_dataset_path"] == str(save_path)
    assert save_path.exists()
    loaded = SupervisionDataset.load(save_path)
    assert len(loaded.response_judgments) == 2
    assert loaded.response_judgments[0].response == "alpha"
    assert loaded.response_judgments[0].response_signal_value == pytest.approx(0.25)


def test_embedding_proxy_oracle_lane_trains_and_persists_artifact(tmp_path: Path) -> None:
    csv_path = tmp_path / "train.csv"
    csv_path.write_text("text,label\nalpha,0.0\nbeta,0.25\ngamma,0.75\ndelta,1.0\n")
    dataset_path = tmp_path / "supervision.json"
    proxy_path = tmp_path / "proxy.json"

    stack = build_treepo_stack(
        TreePOModelSpec(engine="sglang", surface="generate"),
        TreePOContractSpec(
            rubric="Proxy objective.",
            oracle_scale_min=0.0,
            oracle_scale_max=1.0,
            supervision_source=SupervisionSourceSpec(
                kind="csv",
                path=str(csv_path),
                text_column="text",
                label_column="label",
                rubric="Proxy objective.",
                save_path=str(dataset_path),
            ),
            oracle_lane=OracleLaneSpec(
                kind="embedding_proxy",
                embedding_client=_DeterministicEmbeddingClient(),
                ridge_lambda=1e-8,
                proxy_model_id="unit_test_embedding_proxy",
                proxy_artifact_path=str(proxy_path),
                value_name="doc_score",
            ),
        ),
    )

    assert proxy_path.exists()
    assert stack.oracle is not None
    same = stack.oracle.score("alpha", "alpha", rubric="").score
    diff = stack.oracle.score("alpha", "delta", rubric="").score
    assert same >= diff


def test_end_to_end_stack_smoke_runs_state_tree_and_attaches_law_checks(tmp_path: Path) -> None:
    csv_path = tmp_path / "train.csv"
    csv_path.write_text("text,label\nalpha,0.0\nbeta,0.25\ngamma,0.75\ndelta,1.0\n")
    dataset_path = tmp_path / "supervision.json"
    proxy_path = tmp_path / "proxy.json"

    backend = _QueueBackend(
        [
            ["leaf-1", "leaf-2", "leaf-3"],
            ["merge-12"],
            ["merge-root"],
        ]
    )

    contract = TreePOContractSpec(
        rubric="Objective.",
        oracle_scale_min=0.0,
        oracle_scale_max=1.0,
        local_law_config=TreePOLocalLawConfig(enable_l3=False),
        supervision_source=SupervisionSourceSpec(
            kind="csv",
            path=str(csv_path),
            text_column="text",
            label_column="label",
            rubric="Objective.",
            save_path=str(dataset_path),
        ),
        oracle_lane=OracleLaneSpec(
            kind="embedding_proxy",
            embedding_client=_DeterministicEmbeddingClient(),
            ridge_lambda=1e-8,
            proxy_model_id="unit_test_embedding_proxy",
            proxy_artifact_path=str(proxy_path),
        ),
    )

    stack = build_treepo_stack(
        TreePOModelSpec(kind="diffusion_backend", backend=backend, surface="generate"),
        contract,
    )
    result = stack.run_fixed_binary(["alpha", "beta", "gamma"])
    assert result.tree.final_rendered == "merge-root"
    assert backend.calls
    assert result.tree.root.audit.get("law_checks")
    verifier_slot = result.tree.root.audit["law_checks"]
    assert "text_auditor_adapter" in verifier_slot
    assert "l2_merge" in verifier_slot["text_auditor_adapter"]


def test_local_law_config_maps_sampling_probability_into_audit_config() -> None:
    cfg = TreePOLocalLawConfig(sampling_probability=0.25)
    audit_cfg = cfg.to_audit_config()
    assert audit_cfg.sampling_probability == pytest.approx(0.25)
