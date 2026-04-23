from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from src.tree import (
    ResolvedTreePOResources,
    TextGenerationDistillationAdapter,
    TreePOContractAdapter,
    TreePOContractSpec,
    TreePOModelSpec,
    TreePOResourceSpec,
    find_contract_setup_bypasses,
    fit_treepo_contract,
    resolve_treepo_contract_adapter,
    resolve_treepo_contract_route,
    resolve_treepo_resources,
)


def test_paper_unified_examples_cli_smoke(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[2]
    output_dir = tmp_path / "paper_examples"

    completed = subprocess.run(
        [
            sys.executable,
            "scripts/run_paper_unified_examples.py",
            "--output-dir",
            str(output_dir),
            "--learned-docs",
            "8",
            "--learned-steps",
            "2",
            "--no-json",
        ],
        cwd=repo_root,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    manifest = json.loads((output_dir / "manifest.json").read_text())
    assert manifest["artifact_version"] == "paper_unified_contract_examples_v1"
    assert set(manifest["contracts"]) == {
        "text_contract",
        "symbolic_contract",
        "learned_contract",
    }

    manifest_text = json.dumps(manifest)
    assert "markov_exact" not in manifest_text
    assert "learned_sketch" not in manifest_text
    assert "mergeable_sketch" not in manifest_text

    text = manifest["contracts"]["text_contract"]
    assert text["resolved_model_class"] == "generative_tree_operator"
    assert text["resolved_supervision_source"] == "labeled_tree_artifact"
    assert text["route"]["adapter_class"] == "TextGenerationDistillationAdapter"
    assert "generation" in text["route"]["resource_kinds"]
    assert "embedding" in text["route"]["resource_kinds"]
    assert text["capabilities"]["uses_distillation_fit"] is True
    assert text["metrics"]["g_sft_record_count"] > 0
    assert text["metrics"]["f_embedding_example_count"] > 0
    assert Path(text["artifacts"]["labeled_trees"]).exists()
    assert (output_dir / "text_contract" / "g_student" / "g_sft_train.jsonl").exists()
    assert (output_dir / "text_contract" / "f_student" / "f_embedding_proxy.json").exists()

    symbolic = manifest["contracts"]["symbolic_contract"]
    assert symbolic["resolved_model_class"] == "exact_symbolic_operator"
    assert symbolic["route"]["adapter_class"] == "SymbolicReferenceAdapter"
    assert symbolic["metrics"]["root_matches_reference"] is True
    assert Path(symbolic["artifacts"]["state_tree"]).exists()

    learned = manifest["contracts"]["learned_contract"]
    assert learned["resolved_model_class"] == "learned_tree_operator"
    assert learned["route"]["adapter_class"] == "LearnedStateSummaryAdapter"
    assert learned["metrics"]["training_steps"] == 2
    assert learned["metrics"]["ablation_summaries"]
    assert len(learned["metrics"]["local_law_series"]) == 2
    assert learned["metrics"]["local_law_series"][-1]["step"] == 1
    assert Path(learned["artifacts"]["summary"]).exists()


def test_treepo_contract_route_resolution() -> None:
    text = resolve_treepo_contract_route(
        TreePOContractSpec(
            contract_id="text_contract",
            rubric="text",
            objective_kind="node_summary_distillation",
            state_semantics="natural_language_summary",
        )
    )
    assert text.resolved_model_class == "generative_tree_operator"
    assert text.resolved_supervision_source == "labeled_tree_artifact"

    symbolic = resolve_treepo_contract_route(
        TreePOContractSpec(
            contract_id="symbolic_contract",
            rubric="symbolic",
            objective_kind="symbolic_state_reduction",
            state_semantics="finite_state_boundary_summary",
        )
    )
    assert symbolic.resolved_model_class == "exact_symbolic_operator"
    assert symbolic.resolved_supervision_source == "theorem_backed_reference"

    learned = resolve_treepo_contract_route(
        TreePOContractSpec(
            contract_id="learned_contract",
            rubric="learned",
            objective_kind="local_law_recovery",
            state_semantics="learned_state_summary",
        )
    )
    assert learned.resolved_model_class == "learned_tree_operator"
    assert learned.resolved_supervision_source == "local_law_oracle_queries"

    labeled = resolve_treepo_contract_route(
        TreePOContractSpec(
            contract_id="artifact_contract",
            rubric="artifact",
            objective_kind="labeled_tree_distillation",
            state_semantics="artifact_summary_state",
        )
    )
    assert labeled.resolved_model_class == "artifact_distillation"
    assert labeled.resolved_supervision_source == "labeled_tree_artifact"
    assert labeled.adapter_class == "LabeledTreeDistillationAdapter"


class _AlwaysAdapter(TreePOContractAdapter):
    adapter_key = "always"
    resolved_model_class = "always_model"
    resolved_supervision_source = "always_source"

    def supports(self, contract, model, data, resources: ResolvedTreePOResources) -> bool:
        return True


def test_treepo_contract_registry_rejects_no_match_and_ambiguous_match() -> None:
    with pytest.raises(ValueError, match="No TreePO contract adapter"):
        resolve_treepo_contract_route(
            TreePOContractSpec(
                contract_id="unknown",
                rubric="unknown",
                objective_kind="unknown_objective",
                state_semantics="unknown_state",
            )
        )

    ambiguous = TreePOContractSpec(
        contract_id="ambiguous",
        rubric="ambiguous",
        objective_kind="node_summary_distillation",
        state_semantics="natural_language_summary",
    )
    with pytest.raises(ValueError, match="Multiple TreePO contract adapters"):
        resolve_treepo_contract_adapter(
            ambiguous,
            adapters=(TextGenerationDistillationAdapter(), _AlwaysAdapter()),
        )

    preferred = TreePOContractSpec(
        contract_id="preferred",
        rubric="preferred",
        objective_kind="node_summary_distillation",
        state_semantics="natural_language_summary",
        metadata={"adapter_preference": "text_generation_distillation"},
    )
    adapter, route, _ = resolve_treepo_contract_adapter(
        preferred,
        adapters=(TextGenerationDistillationAdapter(), _AlwaysAdapter()),
    )
    assert adapter.adapter_key == "text_generation_distillation"
    assert route.adapter_class == "TextGenerationDistillationAdapter"


def test_treepo_contract_resources_are_resolved_centrally() -> None:
    generation = object()
    embedding = object()
    resources = resolve_treepo_resources(
        {
            "generation": TreePOResourceSpec(kind="object", value=generation),
            "embedding": TreePOResourceSpec(kind="object", value=embedding),
        }
    )
    assert resources.get("generation") is generation
    assert resources.get("embedding") is embedding
    assert resources.to_dict()["resource_specs"]["generation"]["kind"] == "object"


def test_treepo_contract_missing_required_resource_errors(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="requires resource 'generation'"):
        fit_treepo_contract(
            contract=TreePOContractSpec(
                contract_id="text_missing_resource",
                rubric="text",
                objective_kind="node_summary_distillation",
                state_semantics="natural_language_summary",
            ),
            model=TreePOModelSpec(surface="generate"),
            data={"leaf_spans": ["alpha", "beta"]},
            resources={},
            output_dir=tmp_path,
        )


def test_paper_script_has_no_implementation_family_routing() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    findings = find_contract_setup_bypasses(
        [repo_root / "scripts" / "run_paper_unified_examples.py"]
    )
    assert findings == {}
