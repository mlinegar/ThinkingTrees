"""Paper-facing contract examples for the unified C-TreePO framework."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence


from src.training.config_sections import RunConfig, TrainConfig, ValidationConfig
from src.tree import (
    TreePOContractSpec,
    TreePOLocalLawConfig,
    TreePOModelSpec,
    TreePOResourceSpec,
    fit_treepo_contract,
)


ARTIFACT_VERSION = "paper_unified_contract_examples_v1"
CONTRACT_CHOICES = ("all", "text_contract", "symbolic_contract", "learned_contract")


class DeterministicEmbeddingClient:
    """Tiny local embedding client used for paper smoke artifacts."""

    def __init__(self, dim: int = 4) -> None:
        self.dim = int(max(2, dim))

    def resolve_model(self) -> str:
        return "deterministic-paper-demo-embedding"

    def embed_texts(self, texts: Sequence[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        for index, text in enumerate(str(item) for item in texts):
            length = float(len(text))
            char_sum = float(sum(ord(ch) for ch in text) % 997)
            vowels = float(sum(1 for ch in text.lower() if ch in "aeiou"))
            vec = [length, char_sum / 997.0, vowels, float(index)]
            if self.dim > len(vec):
                vec.extend([0.0] * (self.dim - len(vec)))
            vectors.append([float(value) for value in vec[: self.dim]])
        return vectors


class QueueBackend:
    """Deterministic local generation backend for the text contract."""

    backend_name = "paper_demo_deterministic_generate_backend"

    def __init__(self, queued_outputs: Sequence[Sequence[str]]) -> None:
        self._queued_outputs = [list(batch) for batch in queued_outputs]
        self.calls: list[dict[str, Any]] = []

    def generate(
        self,
        texts: Sequence[str] | str,
        sampling_params: Optional[Mapping[str, Any]] = None,
        engine_options: Optional[Mapping[str, Any]] = None,
    ) -> Any:
        from src.diffusion.backends import DiffusionBatchResponse, DiffusionGeneration

        prompts = [texts] if isinstance(texts, str) else list(texts)
        if not self._queued_outputs:
            raise RuntimeError("QueueBackend received more generate calls than expected")
        outputs = self._queued_outputs.pop(0)
        if len(outputs) != len(prompts):
            raise RuntimeError(
                f"QueueBackend expected {len(prompts)} outputs, received {len(outputs)}"
            )
        self.calls.append(
            {
                "n_prompts": len(prompts),
                "sampling_params": dict(sampling_params or {}),
                "engine_options": dict(engine_options or {}),
            }
        )
        return DiffusionBatchResponse(
            generations=[
                DiffusionGeneration(input_text=prompt, output_text=output)
                for prompt, output in zip(prompts, outputs)
            ],
            latency_seconds=0.0,
            request_payload={"text": prompts},
            raw_response={"text": outputs},
        )


def _write_json(path: Path, payload: Mapping[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return path


def _write_markdown_summary(path: Path, manifest: Mapping[str, Any]) -> Path:
    lines = [
        "# Paper Unified Contract Examples",
        "",
        f"Artifact version: `{manifest['artifact_version']}`",
        "",
        "These deterministic examples exercise the same contract-driven TreePO runner across text, symbolic, and learned-state settings.",
        "",
        "## Artifacts",
    ]
    for contract_id, summary in dict(manifest.get("contracts", {})).items():
        artifacts = dict(summary.get("artifacts", {}))
        lines.append(f"- `{contract_id}`: `{artifacts.get('summary')}`")
    lines.extend(
        [
            "",
            "## Reproduce",
            "",
            "```bash",
            "python scripts/run_paper_unified_examples.py --output-dir outputs/paper_unified_examples",
            "```",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines))
    return path


def _select_contracts(values: Iterable[str]) -> tuple[str, ...]:
    requested = tuple(str(item).strip().lower().replace("-", "_") for item in values)
    if not requested or "all" in requested:
        return ("text_contract", "symbolic_contract", "learned_contract")
    unknown = sorted(set(requested) - set(CONTRACT_CHOICES))
    if unknown:
        raise ValueError(f"Unknown contract(s): {unknown}")
    return tuple(dict.fromkeys(requested))


def build_contract_specs(*, seed: int, learned_docs: int, learned_steps: int) -> list[Dict[str, Any]]:
    text_backend = QueueBackend(
        [
            [
                "Alice paid Bob ten dollars.",
                "Bob refunded five dollars after failure.",
                "A five dollar balance remains disputed.",
            ],
            ["Alice paid ten dollars; Bob refunded five."],
            ["Alice and Bob dispute a remaining five dollar refund."],
        ]
    )
    embedding_client = DeterministicEmbeddingClient(dim=4)

    text_leaves = [
        "Alice paid Bob $10 on Tuesday.",
        "Bob refunded Alice $5 after the delivery failed.",
        "Alice says the remaining $5 is still disputed.",
    ]

    return [
        {
            "contract": TreePOContractSpec(
                contract_id="text_contract",
                objective_kind="node_summary_distillation",
                state_semantics="natural_language_summary",
                rubric="Preserve names, amounts, and the dispute status.",
                oracle_scale_min=0.0,
                oracle_scale_max=1.0,
                local_law_config=TreePOLocalLawConfig(enable_l3=False),
                operator_requirements={"surface": "generate", "tree_indexed_outputs": True},
                oracle_requirements={"scalar_score": True, "node_level_labels": True},
            ),
            "model": TreePOModelSpec(surface="generate"),
            "data": {
                "document_id": "paper_text_contract_demo",
                "leaf_spans": text_leaves,
                "document_text": "\n\n".join(text_leaves),
                "document_score": 42.0,
                "split": "train",
                "window_size": 48,
                "root_summary": "Alice and Bob dispute the remaining five dollars after a partial refund.",
                "resummary_target": "Partial refund complete; remaining five dollars disputed.",
                "sampling_params": {"max_tokens": 96, "temperature": 0.0},
                "teacher_model_spec": {"kind": "paper_demo_teacher"},
            },
            "supervision": {
                "ridge_lambda": 1e-8,
                "rows": [
                    {"text": "Alice paid Bob $10 on Tuesday.", "label": 0.2},
                    {"text": "Bob refunded Alice $5.", "label": 0.6},
                    {"text": "The remaining refund is disputed.", "label": 0.9},
                ],
            },
            "resources": {
                "generation": TreePOResourceSpec(kind="object", value=text_backend),
                "embedding": TreePOResourceSpec(kind="object", value=embedding_client),
            },
        },
        {
            "contract": TreePOContractSpec(
                contract_id="symbolic_contract",
                objective_kind="symbolic_state_reduction",
                state_semantics="finite_state_boundary_summary",
                rubric="Preserve the exact finite-state boundary and change-count summary.",
                local_law_config=TreePOLocalLawConfig(),
                operator_requirements={"exact_reference": True, "binary_tree_reduction": True},
                oracle_requirements={"theorem_backed": True},
                theorem_domain={"family": "finite_state_sequence", "reference": "local_law_formalization"},
            ),
            "model": TreePOModelSpec(),
            "data": {
                "document_id": "paper_symbolic_contract_demo",
                "sequence": "a a b b a c c b".split(),
                "leaf_size": 2,
            },
            "supervision": {},
            "resources": {},
        },
        {
            "contract": TreePOContractSpec(
                contract_id="learned_contract",
                objective_kind="local_law_recovery",
                state_semantics="learned_state_summary",
                rubric="Recover a compact tree state whose decoded summary obeys local laws.",
                local_law_config=TreePOLocalLawConfig(),
                operator_requirements={"trainable_tree_operator": True, "decoded_summary": True},
                oracle_requirements={"sampled_node_labels": True, "local_law_metrics": True},
            ),
            "model": TreePOModelSpec(),
            "data": {
                "n_docs": int(learned_docs),
                "steps": int(learned_steps),
                "n_tokens": 32,
                "state_dim": 4,
                "target_k": 4,
                "hidden_dim": 16,
                "chunk_size": 4,
                "eval_docs": 16,
            },
            "supervision": {},
            "resources": {},
        },
    ]


def run_contracts(
    *,
    output_dir: Path,
    contracts: Sequence[str],
    seed: int,
    learned_docs: int,
    learned_steps: int,
) -> Dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    selected = set(_select_contracts(contracts))
    payload: Dict[str, Any] = {
        "artifact_version": ARTIFACT_VERSION,
        "output_dir": str(output_dir),
        "seed": int(seed),
        "contracts": {},
    }
    run = RunConfig(output_dir=output_dir, seed=int(seed))
    train = TrainConfig(train_splits=("train",), steps=int(learned_steps))
    validation = ValidationConfig(enabled=False, val_splits=(), eval_every=max(1, int(learned_steps) // 2))

    for spec in build_contract_specs(
        seed=seed,
        learned_docs=learned_docs,
        learned_steps=learned_steps,
    ):
        contract = spec["contract"]
        contract_id = str(contract.contract_id)
        if contract_id not in selected:
            continue
        result = fit_treepo_contract(
            contract=contract,
            model=spec["model"],
            run=run,
            train=train,
            validation=validation,
            data=spec["data"],
            supervision=spec["supervision"],
            resources=spec["resources"],
            output_dir=output_dir,
        )
        payload["contracts"][contract_id] = result.to_dict()

    manifest_path = _write_json(output_dir / "manifest.json", payload)
    payload["manifest_path"] = str(manifest_path)
    _write_json(output_dir / "manifest.json", payload)
    _write_markdown_summary(output_dir / "README.md", payload)
    return payload


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run deterministic paper-facing examples through contract-driven TreePO.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/paper_unified_examples"))
    parser.add_argument(
        "--contracts",
        nargs="+",
        choices=CONTRACT_CHOICES,
        default=("all",),
        help="Contract IDs to run.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--learned-docs", type=int, default=48)
    parser.add_argument("--learned-steps", type=int, default=12)
    parser.add_argument("--json", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    manifest = run_contracts(
        output_dir=Path(args.output_dir),
        contracts=tuple(args.contracts),
        seed=int(args.seed),
        learned_docs=int(args.learned_docs),
        learned_steps=int(args.learned_steps),
    )
    if bool(args.json):
        print(json.dumps(manifest, indent=2, sort_keys=True))
    else:
        print(f"Wrote {manifest['manifest_path']}")
    return 0

