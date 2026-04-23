#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.structured_config import load_structured_config, write_structured_config


PIPELINE_SCRIPT = REPO_ROOT / "scripts" / "run_markov_optimization_tradeoff_pipeline.py"
LONG_JOB_SCRIPT = REPO_ROOT / "scripts" / "long_job.py"
DEFAULT_PYTHON_BIN = (
    REPO_ROOT / "venv" / "bin" / "python"
    if (REPO_ROOT / "venv" / "bin" / "python").exists()
    else Path(sys.executable)
)


def _utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def _rel(path: Path) -> str:
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def _resolve_repo_relative_path(raw: str | Path) -> Path:
    path = Path(raw).expanduser()
    if path.is_absolute():
        return path.resolve()
    return (REPO_ROOT / path).resolve()


@dataclass(frozen=True)
class GridJobSpec:
    key: str
    label: str
    category: str
    config_relpath: str
    comparisons: tuple[str, ...]
    groups: tuple[str, ...]
    template_overrides: Mapping[str, Any] | None = None
    notes: tuple[str, ...] = ()

    @property
    def config_path(self) -> Path:
        return (REPO_ROOT / self.config_relpath).resolve()

    @property
    def is_template(self) -> bool:
        return bool(self.template_overrides)


@dataclass(frozen=True)
class AxisCoverageSpec:
    key: str
    label: str
    question: str
    side_a: str
    side_b: str
    required_job_keys: tuple[str, ...]
    alternative_job_sets: tuple[tuple[str, ...], ...] = ()
    selection_expectation: str = "default"
    notes: tuple[str, ...] = ()


MULTILEAF_ENDPOINT_LEAF_TOKEN_LADDER = [64, 32, 16, 8]
TWO_LEAF_TOKEN_LADDER = [64]
FULL_COVERAGE_LEAF_TOKEN_LADDER = [128, 64, 32, 16, 8]
DEPTH_EQUAL_REDISTRIBUTION_LEAF_TOKEN_LADDER = [32, 16, 8]
SMALL_TRAIN_DOC_LADDER = [1024, 4096]
LARGE_TRAIN_DOC_LADDER = [10240]
XLARGE_TRAIN_DOC_LADDER = [20480]
LARGE_TRAIN_TUNING_ROOT_PACKAGES = [
    "full10",
    "full20",
    "full30",
    "full40",
    "full50",
    "full70",
]
LARGE_TRAIN_TUNING_LEAF_ONLY_PACKAGES = [
    "r0_leaf_mass_eq_100p0",
    "r10_leaf_mass_eq_90p0",
    "r30_leaf_mass_eq_70p0",
    "r70_leaf_mass_eq_30p0",
    "r80_leaf_mass_eq_20p0",
    "r90_leaf_mass_eq_10p0",
]
PUBLICATION_FULLVAL_ROOT_ONLY_PACKAGES = [
    "full100",
    "full90",
    "full80",
    "full70",
    "full50",
]
PUBLICATION_FULLVAL_LEAF_ONLY_PACKAGES = [
    "r90_leaf_mass_eq_10p0",
    "r80_leaf_mass_eq_20p0",
    "r70_leaf_mass_eq_30p0",
    "r50_leaf_mass_eq_50p0",
]
PUBLICATION_FULLVAL_DEPTH_EQUAL_PACKAGES = [
    "r90_depth_equal_mass_eq_10p0",
    "r80_depth_equal_mass_eq_20p0",
    "r70_depth_equal_mass_eq_30p0",
    "r50_depth_equal_mass_eq_50p0",
]
PUBLICATION_FULLVAL_FULL_LOCAL_LAW_PACKAGES = [
    "full100",
    "root100_extra_leaffull100_internalcount100",
]
PUBLICATION_FIXED_DOC_ROOT_SHARES = [100, 90, 80, 70, 50, 20, 10]
PUBLICATION_FIXED_DOC_ROOT_ONLY_PACKAGES = [
    f"root{int(root_share)}" for root_share in PUBLICATION_FIXED_DOC_ROOT_SHARES
]
PUBLICATION_FIXED_DOC_ONELEAF_LOCAL_LAW_PACKAGES = [
    f"root{int(root_share)}_extra_leaffull100_internalcount100"
    for root_share in PUBLICATION_FIXED_DOC_ROOT_SHARES
]
PUBLICATION_FIXED_DOC_ONELEAF_LEAF_MASS_PACKAGES = [
    "r90_leaf_mass_eq_10p0",
    "r80_leaf_mass_eq_20p0",
    "r70_leaf_mass_eq_30p0",
    "r50_leaf_mass_eq_50p0",
    "r20_leaf_mass_eq_80p0",
    "r10_leaf_mass_eq_90p0",
]
PUBLICATION_FULLVAL_R100_SUPERSET_LOCAL10_PACKAGES = [
    "root100",
    "root100_extra_local10",
]
PUBLICATION_FULLVAL_SEEDS = [0, 1, 2]
PUBLICATION_XLARGE_SEEDS = [0]
STRUCTURAL_ONELEAF_RESCUE_PACKAGES = [
    "full90",
    "full80",
    "full50",
    "full10",
]
STRUCTURAL_ONELEAF_RESCUE_ANCHOR_PACKAGES = ["full100"]
STRUCTURAL_ONELEAF_RESCUE_DOCS = [10240, 20480]
STRUCTURAL_ONELEAF_RESCUE_SEEDS = [0]
V3_T128_SURFACE_OVERRIDES: dict[str, Any] = {
    "tradeoff_pipeline.supervision_recovery_recoverable_benchmark": "recoverable_v5_t128",
    "tradeoff_pipeline.supervision_recovery_structural_grid": "structural_core_v2_t128",
    "tradeoff_pipeline.supervision_recovery_structural_cell": "r12_p079",
}
ONE_LEAF_FNO_PARITY_OVERRIDES: dict[str, Any] = {
    "tradeoff_pipeline.one_leaf_tree_reference.mode": "preset",
    "tradeoff_pipeline.one_leaf_tree_reference.preset": "fno_parity_canary",
}
PUBLICATION_FULLVAL_OVERRIDES: dict[str, Any] = {
    **V3_T128_SURFACE_OVERRIDES,
    "tradeoff_pipeline.tree_stage1_screen_doc_limit": 0,
    "tradeoff_pipeline.tree_stage1_final_exact_doc_limit": 0,
    "tradeoff_pipeline.exact_metric_final_doc_limit": 0,
    "tradeoff_pipeline.tree_stage1_resume_if_available": False,
}


def _merged_overrides(*parts: Mapping[str, Any]) -> dict[str, Any]:
    merged: dict[str, Any] = {}
    for part in parts:
        merged.update(dict(part))
    return merged


def _set_nested(mapping: dict[str, Any], dotted_key: str, value: Any) -> None:
    parts = [part for part in str(dotted_key).split(".") if part]
    if not parts:
        raise ValueError(f"invalid dotted key: {dotted_key!r}")
    cursor = mapping
    for part in parts[:-1]:
        next_value = cursor.get(part)
        if not isinstance(next_value, dict):
            next_value = {}
            cursor[part] = next_value
        cursor = next_value
    cursor[parts[-1]] = value


def materialize_job_config_payload(spec: GridJobSpec) -> dict[str, Any]:
    payload = load_structured_config(spec.config_path)
    if not spec.template_overrides:
        return payload
    rendered = json.loads(json.dumps(payload))
    for dotted_key, value in spec.template_overrides.items():
        _set_nested(rendered, str(dotted_key), value)
    return rendered


def _pipeline_command(config_path: Path, output_root: Path, python_bin: Path) -> list[str]:
    return [
        str(python_bin),
        str(PIPELINE_SCRIPT),
        "--config",
        str(config_path),
        "--output-root",
        str(output_root),
    ]


def _long_job_command(
    *,
    name: str,
    description: str,
    job_root: Path,
    command: Sequence[str],
    python_bin: Path,
    launch_backend: str,
    replace_existing: bool,
    env_assignments: Sequence[str],
) -> list[str]:
    long_job_python = str(python_bin)
    cmd = [
        long_job_python,
        str(LONG_JOB_SCRIPT),
        "launch",
        "--name",
        name,
        "--description",
        description,
        "--job-root",
        str(job_root),
        "--cwd",
        str(REPO_ROOT),
        "--python-bin",
        long_job_python,
        "--launch-backend",
        str(launch_backend),
        "--replace-existing" if replace_existing else "--no-replace-existing",
    ]
    for env_item in env_assignments:
        cmd.extend(["--env", str(env_item)])
    cmd.append("--")
    cmd.extend(str(item) for item in command)
    return cmd


def _read_json_dict(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return dict(payload) if isinstance(payload, dict) else {}


def _query_existing_long_job_status(
    *,
    job_root: Path,
    python_bin: Path,
) -> dict[str, Any]:
    manifest_path = job_root / "manifest.json"
    if not manifest_path.exists():
        return {}
    if not python_bin.exists() and os.path.sep in str(python_bin):
        return {}
    result = subprocess.run(
        [
            str(python_bin),
            str(LONG_JOB_SCRIPT),
            "status",
            "--job-root",
            str(job_root),
            "--tail-lines",
            "0",
        ],
        capture_output=True,
        text=True,
        check=False,
        cwd=str(REPO_ROOT),
    )
    if int(result.returncode) != 0:
        return {
            "status_query_failed": True,
            "status_query_returncode": int(result.returncode),
            "status_query_stderr": str(result.stderr or "").strip(),
        }
    try:
        payload = json.loads(result.stdout or "{}")
    except Exception:
        return {
            "status_query_failed": True,
            "status_query_returncode": 0,
            "status_query_stderr": "invalid_json",
        }
    return dict(payload) if isinstance(payload, dict) else {}


def inspect_existing_job_state(
    *,
    job_root: Path,
    output_root: Path,
    python_bin: Path,
) -> dict[str, Any]:
    manifest_path = job_root / "manifest.json"
    scheduler_status_path = output_root / "scheduler_status.json"
    progress_status_path = output_root / "experiment_status.json"
    scheduler_payload = (
        _read_json_dict(scheduler_status_path)
        if scheduler_status_path.exists()
        else _read_json_dict(progress_status_path)
    )
    scheduler_state = str(scheduler_payload.get("state", "") or "").strip().lower()
    status_payload = _query_existing_long_job_status(
        job_root=job_root,
        python_bin=python_bin,
    )
    running = bool(status_payload.get("running", False))
    state = "not_launched"
    if running:
        state = "running"
    elif scheduler_state == "completed":
        state = "completed"
    elif scheduler_state in {"failed", "cancelled", "canceled"}:
        state = scheduler_state
    elif scheduler_state:
        state = scheduler_state
    elif manifest_path.exists():
        state = "stopped"
    return {
        "state": state,
        "manifest_path": str(manifest_path) if manifest_path.exists() else "",
        "job_root": str(job_root),
        "output_root": str(output_root),
        "running": running,
        "scheduler_state": scheduler_state,
        "scheduler_status_path": (
            str(scheduler_status_path)
            if scheduler_status_path.exists()
            else (
                str(progress_status_path)
                if progress_status_path.exists()
                else ""
            )
        ),
        "status_payload": status_payload,
    }


JOB_SPECS: tuple[GridJobSpec, ...] = (
    GridJobSpec(
        key="v3_main_grid",
        label="Canonical v3 comparison grid",
        category="baseline",
        config_relpath="config/markov/tradeoff_pipeline.v3.toml",
        template_overrides={
            "tradeoff_pipeline.supervision_recovery_leaf_token_ladder": list(
                MULTILEAF_ENDPOINT_LEAF_TOKEN_LADDER
            ),
            "tradeoff_pipeline.tree_stage1_artifact_root": (
                "outputs/_stage1_artifacts/markov_comparison_grid_v3_multileaf_endpoints"
            ),
        },
        comparisons=(
            "Root-only R100 versus true-superset local-label lanes at +10%, +15%, and +20% local rates (`comparison_grid_v3`).",
            "Leaf geometries 64 / 32 / 16 / 8 on 128-token documents, so the comparison stays genuinely multi-level while still hitting the two-leaf endpoint.",
            "Depth-discount gamma sweep 1.0 and 0.9.",
            "Train-doc ladder 1024 / 4096 / 10240 with seeds 0 / 1.",
            "Recoverable `recoverable_v5_t128` plus structural `structural_core_v2_t128` at cell `r12_p079`.",
            "Tree preset `comparison_grid_v3` aliases the current unified-g full-local-laws recipe; the exact-collapse `full100` 1-leaf FNO anchor is handled separately in the dedicated one-leaf surfaces.",
        ),
        groups=("initial_grid", "core"),
        notes=(
            "This is materialized from the checked-in v3 config with a multi-level endpoint ladder added.",
            "This is the canonical baseline grid that the focused follow-ups refine.",
        ),
    ),
    GridJobSpec(
        key="superset_gamma_t128",
        label="Superset gamma sweep at true-128",
        category="core",
        config_relpath="config/markov/tradeoff_pipeline.supervision_recovery_unified_g_superset_gamma_t128.toml",
        template_overrides={
            "tradeoff_pipeline.supervision_recovery_leaf_token_ladder": list(
                MULTILEAF_ENDPOINT_LEAF_TOKEN_LADDER
            ),
            "tradeoff_pipeline.tree_stage1_artifact_root": (
                "outputs/_stage1_artifacts/markov_supervision_recovery_unified_g_superset_gamma_t128_multileaf_endpoints"
            ),
        },
        comparisons=(
            "Root-only R100 versus true-superset local-label lanes at +10%, +15%, and +20% local rates.",
            "Leaf geometries 64 / 32 / 16 / 8 on 128-token documents, with the two-leaf endpoint explicit and all local-label lanes remaining genuinely multi-level.",
            "Depth-discount gamma sweep 1.0 / 0.9 / 0.75.",
            "Train-doc count 10240 with seed 0.",
            "Tree preset `standard_tree` for the current unified-g full-laws surface; the exact-collapse `full100` 1-leaf FNO anchor is handled separately in the dedicated one-leaf surfaces.",
        ),
        groups=("initial_grid", "core"),
        notes=(
            "This is materialized from the checked-in gamma config with a multi-level endpoint ladder added.",
        ),
    ),
    GridJobSpec(
        key="mass_matched_gamma_t128",
        label="Mass-matched gamma sweep at true-128",
        category="core",
        config_relpath="config/markov/tradeoff_pipeline.supervision_recovery_unified_g_leafgrid_r100_gamma_t128_lossnorm.toml",
        template_overrides={
            "tradeoff_pipeline.supervision_recovery_leaf_token_ladder": list(
                MULTILEAF_ENDPOINT_LEAF_TOKEN_LADDER
            ),
            "tradeoff_pipeline.tree_stage1_artifact_root": (
                "outputs/_stage1_artifacts/markov_supervision_recovery_t128_gamma_lossnorm_multileaf_endpoints"
            ),
        },
        comparisons=(
            "Root-only R100 versus mass-matched local-label lanes at +10%, +15%, and +20% local rates.",
            "Leaf geometries 64 / 32 / 16 / 8 on 128-token documents, with the two-leaf endpoint explicit and all local-label lanes remaining genuinely multi-level.",
            "Depth-discount gamma sweep 1.0 / 0.9 / 0.75.",
            "Train-doc count 10240 with seed 0.",
            "Runs on the loss-normalization-fixed stage-1 artifact lineage; the exact-collapse `full100` 1-leaf FNO anchor is handled separately in the dedicated one-leaf surfaces.",
        ),
        groups=("initial_grid", "core"),
        notes=(
            "This is materialized from the checked-in gamma config with a multi-level endpoint ladder added.",
        ),
    ),
    GridJobSpec(
        key="full100_leaf_ladder_standard",
        label="Standard full100 leaf-law ladder",
        category="leaf_law_followup",
        config_relpath="config/markov/tradeoff_pipeline.supervision_recovery_unified_g_full100_gamma09_leaf_ladder_t128.toml",
        template_overrides=_merged_overrides(
            {
                "tradeoff_pipeline.supervision_recovery_leaf_token_ladder": list(
                    FULL_COVERAGE_LEAF_TOKEN_LADDER
                ),
                "tradeoff_pipeline.tree_stage1_artifact_root": (
                    "outputs/_stage1_artifacts/markov_supervision_recovery_unified_g_full100_gamma09_leaf_ladder_t128_endpoints"
                ),
            },
            ONE_LEAF_FNO_PARITY_OVERRIDES,
        ),
        comparisons=(
            "Full100 root-only supervision only.",
            "Leaf ladder 128 / 64 / 32 / 16 / 8 on 128-token docs, so the exact-collapse FNO anchor is explicit.",
            "Gamma fixed at 0.9.",
            "Train-doc count 10240 with seed 0.",
            "Standard unified-g full-laws preset on the multi-leaf points; the 1-leaf `full100` anchor uses `fno_parity_canary` to stay exactly comparable to FNO.",
        ),
        groups=("initial_grid", "leaf_law_followups"),
        notes=(
            "This is materialized from the checked-in full100 ladder config with the 128-token endpoint added.",
        ),
    ),
    GridJobSpec(
        key="full100_leaf_ladder_half_c1",
        label="Half-C1 full100 leaf-law ladder",
        category="leaf_law_followup",
        config_relpath="config/markov/tradeoff_pipeline.supervision_recovery_unified_g_half_leaf_law_gamma09_leaf_ladder_t128.toml",
        template_overrides=_merged_overrides(
            {
                "tradeoff_pipeline.supervision_recovery_leaf_token_ladder": list(
                    FULL_COVERAGE_LEAF_TOKEN_LADDER
                ),
                "tradeoff_pipeline.tree_stage1_artifact_root": (
                    "outputs/_stage1_artifacts/markov_supervision_recovery_unified_g_half_leaf_law_gamma09_leaf_ladder_t128_endpoints"
                ),
            },
            ONE_LEAF_FNO_PARITY_OVERRIDES,
        ),
        comparisons=(
            "Full100 root-only supervision only.",
            "Leaf ladder 128 / 64 / 32 / 16 / 8 on 128-token docs, so the exact-collapse FNO anchor is explicit.",
            "Gamma fixed at 0.9.",
            "Train-doc count 10240 with seed 0.",
            "Half-C1 preset cuts the effective C1 weight in half on the multi-leaf points while the 1-leaf `full100` anchor stays on `fno_parity_canary` for exact FNO comparability.",
        ),
        groups=("initial_grid", "leaf_law_followups"),
        notes=(
            "This is materialized from the checked-in half-C1 ladder config with the 128-token endpoint added.",
        ),
    ),
    GridJobSpec(
        key="superset_leaf32_c1half",
        label="Leaf32 superset half-C1 follow-up",
        category="leaf_law_followup",
        config_relpath="config/markov/tradeoff_pipeline.supervision_recovery_unified_g_superset_leaf32_gamma09_c1half.toml",
        comparisons=(
            "Root-only R100 versus true-superset +10% local-label lane.",
            "Single geometry: leaf size 32 on 128-token docs.",
            "Gamma fixed at 0.9.",
            "Train-doc count 10240 with seed 0.",
            "Half-C1 preset isolates whether the +10% superset gain is overly dependent on the leaf-law term.",
        ),
        groups=("initial_grid", "leaf_law_followups"),
    ),
    GridJobSpec(
        key="superset_leaf32_leafratehalf",
        label="Leaf32 superset reduced-leaf-rate follow-up",
        category="leaf_law_followup",
        config_relpath="config/markov/tradeoff_pipeline.supervision_recovery_unified_g_superset_leaf32_gamma09_leafratehalf.toml",
        comparisons=(
            "Root-only R100 versus standard +10% superset and reduced-leaf-rate +5%-leaf / +10%-internal lane.",
            "Single geometry: leaf size 32 on 128-token docs.",
            "Gamma fixed at 0.9.",
            "Train-doc count 10240 with seed 0.",
            "Standard preset retained so the only changed factor is the leaf-label rate.",
        ),
        groups=("initial_grid", "leaf_law_followups"),
    ),
    GridJobSpec(
        key="preset_ablation_canary",
        label="Preset ablation step 1: FNO-parity canary",
        category="preset_ablation",
        config_relpath="config/markov/tradeoff_pipeline.supervision_recovery_unified_g_ablation_ladder.toml",
        template_overrides={
            "tradeoff_pipeline.tree_reference.preset": "fno_parity_canary",
            "tradeoff_pipeline.structural_tree_reference.preset": "fno_parity_canary",
        },
        comparisons=(
            "Full100 only at exact 1-leaf geometry (128-token docs, leaf size 128).",
            "Train-doc ladder 1024 / 4096 / 10240 with seed 0.",
            "Step 1 of the canary → standard ladder: CE root supervision, single-stage training, no local laws.",
        ),
        groups=("initial_grid", "preset_ablation"),
        notes=("This job is materialized from the PLACEHOLDER template at launch time.",),
    ),
    GridJobSpec(
        key="preset_ablation_mse_only",
        label="Preset ablation step 2: CE→MSE only",
        category="preset_ablation",
        config_relpath="config/markov/tradeoff_pipeline.supervision_recovery_unified_g_ablation_ladder.toml",
        template_overrides={
            "tradeoff_pipeline.tree_reference.preset": "mse_only",
            "tradeoff_pipeline.structural_tree_reference.preset": "mse_only",
        },
        comparisons=(
            "Full100 only at exact 1-leaf geometry (128-token docs, leaf size 128).",
            "Train-doc ladder 1024 / 4096 / 10240 with seed 0.",
            "Adds MSE regression on top of the canary while keeping single-stage training and no local laws.",
        ),
        groups=("initial_grid", "preset_ablation"),
        notes=("This job is materialized from the PLACEHOLDER template at launch time.",),
    ),
    GridJobSpec(
        key="preset_ablation_two_stage_no_laws",
        label="Preset ablation step 3: add two-stage schedule",
        category="preset_ablation",
        config_relpath="config/markov/tradeoff_pipeline.supervision_recovery_unified_g_ablation_ladder.toml",
        template_overrides={
            "tradeoff_pipeline.tree_reference.preset": "two_stage_no_laws",
            "tradeoff_pipeline.structural_tree_reference.preset": "two_stage_no_laws",
        },
        comparisons=(
            "Full100 only at exact 1-leaf geometry (128-token docs, leaf size 128).",
            "Train-doc ladder 1024 / 4096 / 10240 with seed 0.",
            "Keeps local laws off, but adds the standard two-stage schedule to isolate schedule effects.",
        ),
        groups=("initial_grid", "preset_ablation"),
        notes=("This job is materialized from the PLACEHOLDER template at launch time.",),
    ),
    GridJobSpec(
        key="preset_ablation_full_laws",
        label="Preset ablation step 4: add full local laws",
        category="preset_ablation",
        config_relpath="config/markov/tradeoff_pipeline.supervision_recovery_unified_g_ablation_ladder.toml",
        template_overrides={
            "tradeoff_pipeline.tree_reference.preset": "full_laws",
            "tradeoff_pipeline.structural_tree_reference.preset": "full_laws",
        },
        comparisons=(
            "Full100 only at exact 1-leaf geometry (128-token docs, leaf size 128).",
            "Train-doc ladder 1024 / 4096 / 10240 with seed 0.",
            "Final ladder step: standard unified-g recipe on the exact 1-leaf surface.",
            "This is a protocol comparison, not a law-validity comparison: at `leaf128` there are no non-root leaf/internal supervision targets, so the local-law terms are structurally inactive even though the preset carries their weights.",
        ),
        groups=("initial_grid", "preset_ablation"),
        notes=("This job is materialized from the PLACEHOLDER template at launch time.",),
    ),
    GridJobSpec(
        key="multileaf_root_only",
        label="Multi-leaf protocol ablation: root-only",
        category="multileaf_protocol",
        config_relpath="config/markov/tradeoff_pipeline.supervision_recovery_unified_g_multi_leaf_ablation.toml",
        template_overrides={
            "tradeoff_pipeline.tree_reference.preset": "multileaf_root_only",
            "tradeoff_pipeline.structural_tree_reference.preset": "multileaf_root_only",
        },
        comparisons=(
            "Full100 only across leaf geometries 128 / 64 / 32 / 16 / 8.",
            "Train-doc ladder 1024 / 4096 / 10240 with seeds 0 / 1.",
            "Single-stage, no-local-laws multi-leaf baseline to isolate architecture from training protocol.",
            "The template already keeps the 1-leaf point on the FNO-parity canary preset.",
        ),
        groups=("initial_grid", "multileaf_protocol"),
        notes=("This job is materialized from the PLACEHOLDER template at launch time.",),
    ),
    GridJobSpec(
        key="multileaf_full_laws",
        label="Multi-leaf protocol ablation: standard full laws",
        category="multileaf_protocol",
        config_relpath="config/markov/tradeoff_pipeline.supervision_recovery_unified_g_multi_leaf_ablation.toml",
        template_overrides={
            "tradeoff_pipeline.tree_reference.preset": "standard_tree",
            "tradeoff_pipeline.structural_tree_reference.preset": "standard_tree",
        },
        comparisons=(
            "Full100 only across leaf geometries 128 / 64 / 32 / 16 / 8.",
            "Train-doc ladder 1024 / 4096 / 10240 with seeds 0 / 1.",
            "Current standard two-stage + full-local-laws multi-leaf surface.",
            "Directly comparable to the root-only multi-leaf baseline above.",
        ),
        groups=("initial_grid", "multileaf_protocol"),
        notes=("This job is materialized from the PLACEHOLDER template at launch time.",),
    ),
    GridJobSpec(
        key="small_train_multileaf_root_only",
        label="Small-train multi-leaf root-only law sanity",
        category="small_train_local_law",
        config_relpath="config/markov/tradeoff_pipeline.supervision_recovery_unified_g_multi_leaf_ablation.toml",
        template_overrides={
            "tradeoff_pipeline.tree_reference.preset": "multileaf_root_only",
            "tradeoff_pipeline.structural_tree_reference.preset": "multileaf_root_only",
            "tradeoff_pipeline.supervision_recovery_train_docs": list(
                SMALL_TRAIN_DOC_LADDER
            ),
            "tradeoff_pipeline.supervision_recovery_seeds": [0],
            "tradeoff_pipeline.supervision_recovery_leaf_token_ladder": list(
                MULTILEAF_ENDPOINT_LEAF_TOKEN_LADDER
            ),
            "tradeoff_pipeline.tree_stage1_artifact_root": (
                "outputs/_stage1_artifacts/markov_supervision_recovery_unified_g_small_train_multileaf_root_only"
            ),
        },
        comparisons=(
            "Full100 only across the multi-leaf geometries 64 / 32 / 16 / 8.",
            "Small-train bring-up: 1024 / 4096 docs with seed 0.",
            "Root-only baseline for law-validity checks, where leaf/internal supervision is intentionally absent but the tree geometry is real.",
            "The exact 1-leaf FNO anchor is handled separately by `preset_ablation_canary`, because local-law terms are structurally inactive at `leaf128`.",
        ),
        groups=("small_train_local_law",),
        notes=(
            "This is the first scientifically meaningful local-law baseline surface; it excludes `leaf128` by design.",
        ),
    ),
    GridJobSpec(
        key="quick_two_leaf_root_only",
        label="Quick two-leaf root-only baseline",
        category="local_law_quickcheck",
        config_relpath="config/markov/tradeoff_pipeline.supervision_recovery_unified_g_multi_leaf_ablation.toml",
        template_overrides={
            "tradeoff_pipeline.tree_reference.preset": "multileaf_root_only",
            "tradeoff_pipeline.structural_tree_reference.preset": "multileaf_root_only",
            "tradeoff_pipeline.supervision_recovery_train_docs": list(
                SMALL_TRAIN_DOC_LADDER
            ),
            "tradeoff_pipeline.supervision_recovery_seeds": [0],
            "tradeoff_pipeline.supervision_recovery_leaf_token_ladder": list(
                TWO_LEAF_TOKEN_LADDER
            ),
            "tradeoff_pipeline.tree_stage1_artifact_root": (
                "outputs/_stage1_artifacts/markov_supervision_recovery_unified_g_quick_two_leaf_root_only"
            ),
        },
        comparisons=(
            "Full100 only at the first nondegenerate tree geometry: leaf64 on 128-token docs.",
            "Small-train bring-up: 1024 / 4096 docs with seed 0.",
            "Root-only baseline for the first real local-law setting, where the tree has exactly two leaves per document.",
            "Use this directly against the two-leaf local-label jobs before scaling up to deeper ladders.",
        ),
        groups=("local_law_quickcheck",),
        notes=(
            "This is the first quick sanity surface after the one-leaf canary: exactly two leaves per document.",
        ),
    ),
    GridJobSpec(
        key="one_leaf_duplicate_local_full_laws",
        label="Small-train one-leaf duplicate-local no-harm check",
        category="small_train_local_law",
        config_relpath="config/markov/tradeoff_pipeline.supervision_recovery_unified_g_ablation_ladder.toml",
        template_overrides={
            "tradeoff_pipeline.tree_reference.preset": "full_laws",
            "tradeoff_pipeline.structural_tree_reference.preset": "full_laws",
            "tradeoff_pipeline.supervision_recovery_packages": [
                "root100_extra_leaffull100_internalcount100"
            ],
            "tradeoff_pipeline.supervision_recovery_train_docs": list(
                SMALL_TRAIN_DOC_LADDER
            ),
            "tradeoff_pipeline.supervision_recovery_seeds": [0],
            "tradeoff_pipeline.tree_stage1_artifact_root": (
                "outputs/_stage1_artifacts/markov_supervision_recovery_unified_g_one_leaf_duplicate_local_full_laws"
            ),
        },
        comparisons=(
            "Exact 1-leaf geometry only: 128-token docs with leaf size 128.",
            "Small-train bring-up: 1024 / 4096 docs with seed 0.",
            "Uses the full local-supervision package `root100_extra_leaffull100_internalcount100`, so the single available leaf gets duplicate local supervision on top of full root supervision.",
            "This is a no-harm redundancy check, not a full law-validity proof: merge/internal structure is still degenerate at 1 leaf, so the question here is whether extra local labels hurt parity.",
        ),
        groups=("small_train_local_law",),
        notes=(
            "The canonical FNO anchor for this surface is still `preset_ablation_canary` on `full100`.",
        ),
    ),
    GridJobSpec(
        key="quick_two_leaf_full100_local_full_laws",
        label="Quick two-leaf full100 + full local laws",
        category="local_law_quickcheck",
        config_relpath="config/markov/tradeoff_pipeline.supervision_recovery_unified_g_multi_leaf_ablation.toml",
        template_overrides={
            "tradeoff_pipeline.tree_reference.preset": "standard_tree",
            "tradeoff_pipeline.structural_tree_reference.preset": "standard_tree",
            "tradeoff_pipeline.supervision_recovery_packages": [
                "root100_extra_leaffull100_internalcount100"
            ],
            "tradeoff_pipeline.supervision_recovery_train_docs": list(
                SMALL_TRAIN_DOC_LADDER
            ),
            "tradeoff_pipeline.supervision_recovery_seeds": [0],
            "tradeoff_pipeline.supervision_recovery_leaf_token_ladder": list(
                TWO_LEAF_TOKEN_LADDER
            ),
            "tradeoff_pipeline.tree_stage1_artifact_root": (
                "outputs/_stage1_artifacts/markov_supervision_recovery_unified_g_quick_two_leaf_full100_local_full_laws"
            ),
        },
        comparisons=(
            "Full root supervision plus full leaf full-sketch and full internal count labels at leaf64.",
            "Small-train bring-up: 1024 / 4096 docs with seed 0.",
            "This is the maximal two-leaf local-law activation check: if local supervision can work, it should be obvious here.",
            "Compare directly to `quick_two_leaf_root_only` and the one-leaf canary anchor.",
        ),
        groups=("local_law_quickcheck",),
        notes=(
            "Uses `root100_extra_leaffull100_internalcount100`, so local supervision is guaranteed to be materially present on the two-leaf tree.",
        ),
    ),
    GridJobSpec(
        key="quick_two_leaf_r100_superset_local10",
        label="Quick two-leaf R100 +10% superset local laws",
        category="local_law_quickcheck",
        config_relpath="config/markov/tradeoff_pipeline.superset_multileaf_lossnorm_ablation.toml",
        template_overrides={
            "tradeoff_pipeline.tree_reference.preset": "standard_tree",
            "tradeoff_pipeline.structural_tree_reference.preset": "standard_tree",
            "tradeoff_pipeline.supervision_recovery_packages": [
                "root100",
                "root100_extra_local10",
            ],
            "tradeoff_pipeline.supervision_recovery_train_docs": list(
                SMALL_TRAIN_DOC_LADDER
            ),
            "tradeoff_pipeline.supervision_recovery_seeds": [0],
            "tradeoff_pipeline.supervision_recovery_depth_discount_gammas": [1.0],
            "tradeoff_pipeline.supervision_recovery_leaf_token_ladder": list(
                TWO_LEAF_TOKEN_LADDER
            ),
            "tradeoff_pipeline.tree_stage1_artifact_root": (
                "outputs/_stage1_artifacts/markov_supervision_recovery_unified_g_quick_two_leaf_r100_superset_local10"
            ),
        },
        comparisons=(
            "Root100 baseline versus root100 + 10% leaf/internal count labels at leaf64.",
            "Small-train bring-up: 1024 / 4096 docs with seed 0 and gamma 1.0.",
            "This is the first direct check that the milder R100 superset local-label surface works, not just the maximal full-local package.",
            "FNO remains the full100 leaf128 reference emitted from the root100 lane.",
        ),
        groups=("local_law_quickcheck",),
        notes=(
            "This is the first quick validation of the scientifically relevant R100 local-law surface.",
        ),
    ),
    GridJobSpec(
        key="small_train_multileaf_full_laws",
        label="Small-train multi-leaf full-laws sanity",
        category="small_train_local_law",
        config_relpath="config/markov/tradeoff_pipeline.supervision_recovery_unified_g_multi_leaf_ablation.toml",
        template_overrides={
            "tradeoff_pipeline.tree_reference.preset": "standard_tree",
            "tradeoff_pipeline.structural_tree_reference.preset": "standard_tree",
            "tradeoff_pipeline.supervision_recovery_packages": [
                "root100_extra_leaffull100_internalcount100"
            ],
            "tradeoff_pipeline.supervision_recovery_train_docs": list(
                SMALL_TRAIN_DOC_LADDER
            ),
            "tradeoff_pipeline.supervision_recovery_seeds": [0],
            "tradeoff_pipeline.supervision_recovery_leaf_token_ladder": list(
                MULTILEAF_ENDPOINT_LEAF_TOKEN_LADDER
            ),
            "tradeoff_pipeline.tree_stage1_artifact_root": (
                "outputs/_stage1_artifacts/markov_supervision_recovery_unified_g_small_train_multileaf_full_laws"
            ),
        },
        comparisons=(
            "Full root supervision plus full leaf full-sketch and all internal count labels across the multi-leaf geometries 64 / 32 / 16 / 8.",
            "Small-train bring-up: 1024 / 4096 docs with seed 0.",
            "Standard two-stage + full-local-laws surface, now on geometries where leaf/internal supervision is actually present by construction.",
            "Use this against the root-only multi-leaf job, with `preset_ablation_canary` as the exact 1-leaf FNO anchor.",
        ),
        groups=("small_train_local_law",),
        notes=(
            "This is the correct first-pass local-law sanity surface; it excludes the law-degenerate `leaf128` point.",
        ),
    ),
    GridJobSpec(
        key="small_train_r100_superset_local10",
        label="Small-train R100 +10% superset local-law sanity",
        category="small_train_local_law",
        config_relpath="config/markov/tradeoff_pipeline.superset_multileaf_lossnorm_ablation.toml",
        template_overrides={
            "tradeoff_pipeline.tree_reference.preset": "standard_tree",
            "tradeoff_pipeline.structural_tree_reference.preset": "standard_tree",
            "tradeoff_pipeline.supervision_recovery_packages": [
                "root100",
                "root100_extra_local10",
            ],
            "tradeoff_pipeline.supervision_recovery_train_docs": list(
                SMALL_TRAIN_DOC_LADDER
            ),
            "tradeoff_pipeline.supervision_recovery_seeds": [0],
            "tradeoff_pipeline.supervision_recovery_depth_discount_gammas": [1.0],
            "tradeoff_pipeline.supervision_recovery_leaf_token_ladder": list(
                MULTILEAF_ENDPOINT_LEAF_TOKEN_LADDER
            ),
            "tradeoff_pipeline.tree_stage1_artifact_root": (
                "outputs/_stage1_artifacts/markov_supervision_recovery_unified_g_small_train_r100_superset_local10"
            ),
        },
        comparisons=(
            "Root100 baseline versus root100 + 10% leaf/internal count labels across 64 / 32 / 16 / 8.",
            "Small-train bring-up: 1024 / 4096 docs with seed 0 and gamma 1.0.",
            "This is the first scaled-up R100 local-law sanity surface after the two-leaf quickcheck.",
            "Use this to confirm the mild superset local-label path behaves sensibly before broader r100 grids.",
        ),
        groups=("small_train_local_law",),
        notes=(
            "This job is about the scientifically relevant `r100 + local` surface, not just the maximal full-local package.",
        ),
    ),
    GridJobSpec(
        key="redistribution_quickcheck",
        label="Quick root/node redistribution sanity",
        category="redistribution",
        config_relpath="config/markov/tradeoff_pipeline.supervision_recovery_unified_g_leafgrid_r100_full_coverage.toml",
        template_overrides={
            "tradeoff_pipeline.supervision_recovery_packages": [
                "redistribution_r100_coarse"
            ],
            "tradeoff_pipeline.supervision_recovery_train_docs": list(
                SMALL_TRAIN_DOC_LADDER
            ),
            "tradeoff_pipeline.supervision_recovery_seeds": [0],
            "tradeoff_pipeline.supervision_recovery_depth_discount_gammas": [1.0],
            "tradeoff_pipeline.supervision_recovery_leaf_token_ladder": list(
                TWO_LEAF_TOKEN_LADDER
            ),
            "tradeoff_pipeline.tree_stage1_artifact_root": (
                "outputs/_stage1_artifacts/markov_supervision_recovery_unified_g_redistribution_quickcheck"
            ),
        },
        comparisons=(
            "Fixed total supervision mass at 100%, redistributed between root-only review and covered-token node supervision.",
            "Coarse split ladder: 100/0, 80/20, 50/50, 20/80, and 0/100 root/node mass shares.",
            "Two-leaf geometry only: leaf64 on 128-token docs, so the first nondegenerate merge tree is active while the split is easy to inspect.",
            "Small-train bring-up: 1024 / 4096 docs with seed 0 and gamma 1.0.",
            "Full100 remains in the bundle, so the canonical `official_fno` base is emitted from the same run.",
        ),
        groups=("redistribution", "redistribution_quickcheck"),
        notes=(
            "Uses the new geometry-aware redistribution packages, so `root50_nodes50` means the same 50/50 mass split at this geometry, not just a reused label rate.",
        ),
    ),
    GridJobSpec(
        key="redistribution_small_train",
        label="Small-train root/node redistribution grid",
        category="redistribution",
        config_relpath="config/markov/tradeoff_pipeline.supervision_recovery_unified_g_leafgrid_r100_full_coverage.toml",
        template_overrides={
            "tradeoff_pipeline.supervision_recovery_packages": [
                "redistribution_r100"
            ],
            "tradeoff_pipeline.supervision_recovery_train_docs": list(
                SMALL_TRAIN_DOC_LADDER
            ),
            "tradeoff_pipeline.supervision_recovery_seeds": [0],
            "tradeoff_pipeline.supervision_recovery_depth_discount_gammas": [1.0],
            "tradeoff_pipeline.supervision_recovery_leaf_token_ladder": list(
                MULTILEAF_ENDPOINT_LEAF_TOKEN_LADDER
            ),
            "tradeoff_pipeline.tree_stage1_artifact_root": (
                "outputs/_stage1_artifacts/markov_supervision_recovery_unified_g_redistribution_small_train"
            ),
        },
        comparisons=(
            "Fixed total supervision mass at 100%, redistributed between root-only review and covered-token node supervision.",
            "Full decile ladder: 100/0, 90/10, 80/20, 70/30, 60/40, 50/50, 40/60, 30/70, 20/80, 10/90, and 0/100 root/node mass shares.",
            "Multi-leaf geometries 64 / 32 / 16 / 8 on 128-token docs, with the split defined by covered-token mass rather than a geometry-drifting label rate.",
            "Small-train bring-up: 1024 / 4096 docs with seed 0 and gamma 1.0.",
            "Full100 remains in the bundle, so the canonical `official_fno` base is emitted from the same run.",
        ),
        groups=("redistribution", "redistribution_small_train"),
        notes=(
            "This is the first proper root-vs-nodes redistribution surface: total mass stays fixed at 100% while the root/node split changes exactly across the whole leaf ladder.",
        ),
    ),
    GridJobSpec(
        key="root_budget_ladder_small_train",
        label="Small-train root-budget ladder",
        category="depth_redistribution",
        config_relpath="config/markov/tradeoff_pipeline.supervision_recovery_unified_g_leafgrid_r100_full_coverage.toml",
        template_overrides=_merged_overrides(
            V3_T128_SURFACE_OVERRIDES,
            {
                "tradeoff_pipeline.supervision_recovery_packages": [
                    "root_ladder_deciles"
                ],
                "tradeoff_pipeline.supervision_recovery_train_docs": list(
                    SMALL_TRAIN_DOC_LADDER
                ),
                "tradeoff_pipeline.supervision_recovery_seeds": [0],
                "tradeoff_pipeline.supervision_recovery_depth_discount_gammas": [1.0],
                "tradeoff_pipeline.supervision_recovery_leaf_token_ladder": list(
                    FULL_COVERAGE_LEAF_TOKEN_LADDER
                ),
                "tradeoff_pipeline.tree_stage1_artifact_root": (
                    "outputs/_stage1_artifacts/markov_supervision_recovery_unified_g_root_budget_ladder_small_train"
                ),
            },
        ),
        comparisons=(
            "Root-only baseline ladder at 100 / 90 / 80 / 70 / 60 / 50 / 40 / 30 / 20 / 10 percent reviewed docs.",
            "Geometries 128 / 64 / 32 / 16 / 8 on 128-token documents, so the pure root-supervision baseline is available at every tree depth including exact-collapse leaf128.",
            "Small-train bring-up: 1024 / 4096 docs with seed 0 and gamma 1.0.",
            "Each root-only package still emits its direct FNO analogue, so this is the baseline ladder for asking how much root review local supervision can replace.",
        ),
        groups=("depth_redistribution", "depth_redistribution_root_ladder"),
        notes=(
            "This is the clean root-only comparator surface for the new depth-aware mass-preserving study.",
        ),
    ),
    GridJobSpec(
        key="mass_preserving_leaf_only_small_train",
        label="Small-train mass-preserving leaf-only grid",
        category="depth_redistribution",
        config_relpath="config/markov/tradeoff_pipeline.supervision_recovery_unified_g_leafgrid_r100_full_coverage.toml",
        template_overrides=_merged_overrides(
            V3_T128_SURFACE_OVERRIDES,
            {
                "tradeoff_pipeline.supervision_recovery_packages": [
                    "mass_preserving_leaf_only_deciles"
                ],
                "tradeoff_pipeline.supervision_recovery_train_docs": list(
                    SMALL_TRAIN_DOC_LADDER
                ),
                "tradeoff_pipeline.supervision_recovery_seeds": [0],
                "tradeoff_pipeline.supervision_recovery_depth_discount_gammas": [1.0],
                "tradeoff_pipeline.supervision_recovery_leaf_token_ladder": list(
                    MULTILEAF_ENDPOINT_LEAF_TOKEN_LADDER
                ),
                "tradeoff_pipeline.tree_stage1_artifact_root": (
                    "outputs/_stage1_artifacts/markov_supervision_recovery_unified_g_mass_preserving_leaf_only_small_train"
                ),
            },
        ),
        comparisons=(
            "Root-share decile ladder with total supervision mass fixed at 100%, and every non-root token of that budget placed on leaves only.",
            "Packages include the matching root-only baselines (`root90`, `root80`, ... `root10`) next to the mass-preserving alternatives (`root90_leaf10`, ... `root0_leaf100`).",
            "Multi-leaf geometries 64 / 32 / 16 / 8, where leaf-only local supervision is always meaningful and 64-token leaves realize the simple 50/50 root-vs-leaf endpoint.",
            "Small-train bring-up: 1024 / 4096 docs with seed 0 and gamma 1.0.",
        ),
        groups=("depth_redistribution", "depth_redistribution_leaf_only"),
        notes=(
            "This is the direct `50/50/0`-style family generalized to the full root-share decile ladder.",
        ),
    ),
    GridJobSpec(
        key="mass_preserving_depth_equal_small_train",
        label="Small-train mass-preserving depth-equal grid",
        category="depth_redistribution",
        config_relpath="config/markov/tradeoff_pipeline.supervision_recovery_unified_g_leafgrid_r100_full_coverage.toml",
        template_overrides=_merged_overrides(
            V3_T128_SURFACE_OVERRIDES,
            {
                "tradeoff_pipeline.supervision_recovery_packages": [
                    "mass_preserving_levels_equal_deciles"
                ],
                "tradeoff_pipeline.supervision_recovery_train_docs": list(
                    SMALL_TRAIN_DOC_LADDER
                ),
                "tradeoff_pipeline.supervision_recovery_seeds": [0],
                "tradeoff_pipeline.supervision_recovery_depth_discount_gammas": [1.0],
                "tradeoff_pipeline.supervision_recovery_leaf_token_ladder": list(
                    DEPTH_EQUAL_REDISTRIBUTION_LEAF_TOKEN_LADDER
                ),
                "tradeoff_pipeline.tree_stage1_artifact_root": (
                    "outputs/_stage1_artifacts/markov_supervision_recovery_unified_g_mass_preserving_depth_equal_small_train"
                ),
            },
        ),
        comparisons=(
            "Root-share decile ladder with total supervision mass fixed at 100%, and the non-root budget spread evenly over leaves plus every available non-root merge depth.",
            "This realizes geometry-aware profiles such as 50/25/25 at leaf32, 25/25/25/25 at leaf16, and equal multi-level depth splits at leaf8.",
            "Packages include the matching root-only baselines (`root90`, `root80`, ... `root10`) next to the depth-equal mass-preserving alternatives (`root90_levels_equal10`, ... `root0_levels_equal100`).",
            "Small-train bring-up: 1024 / 4096 docs with seed 0 and gamma 1.0.",
        ),
        groups=("depth_redistribution", "depth_redistribution_levels_equal"),
        notes=(
            "This is the first v3 surface that explicitly answers whether upper summary levels help more than leaf-only redistribution at the same total supervision mass.",
        ),
    ),
    GridJobSpec(
        key="root_budget_ladder_large_train",
        label="10240-doc root-budget ladder",
        category="depth_redistribution_large_train",
        config_relpath="config/markov/tradeoff_pipeline.supervision_recovery_unified_g_leafgrid_r100_full_coverage.toml",
        template_overrides=_merged_overrides(
            V3_T128_SURFACE_OVERRIDES,
            {
                "tradeoff_pipeline.supervision_recovery_packages": [
                    "root_ladder_deciles"
                ],
                "tradeoff_pipeline.supervision_recovery_train_docs": list(
                    LARGE_TRAIN_DOC_LADDER
                ),
                "tradeoff_pipeline.supervision_recovery_seeds": [0],
                "tradeoff_pipeline.supervision_recovery_depth_discount_gammas": [1.0],
                "tradeoff_pipeline.supervision_recovery_leaf_token_ladder": list(
                    FULL_COVERAGE_LEAF_TOKEN_LADDER
                ),
                "tradeoff_pipeline.tree_stage1_artifact_root": (
                    "outputs/_stage1_artifacts/markov_supervision_recovery_unified_g_root_budget_ladder_large_train"
                ),
            },
        ),
        comparisons=(
            "Root-only baseline ladder at 100 / 90 / 80 / 70 / 60 / 50 / 40 / 30 / 20 / 10 percent reviewed docs.",
            "Geometries 128 / 64 / 32 / 16 / 8 on 128-token documents, so the pure root-supervision baseline is available at every tree depth including exact-collapse leaf128.",
            "Large-train follow-up: 10240 docs with seed 0 and gamma 1.0.",
            "Each root-only package still emits its direct FNO analogue, so this is the 10240-doc comparator ladder for asking how much root review local supervision can replace.",
        ),
        groups=("depth_redistribution_large_train", "depth_redistribution_large_root_ladder"),
        notes=(
            "This is the large-train root-only comparator surface for the depth-aware mass-preserving study.",
        ),
    ),
    GridJobSpec(
        key="mass_preserving_leaf_only_large_train",
        label="10240-doc mass-preserving leaf-only grid",
        category="depth_redistribution_large_train",
        config_relpath="config/markov/tradeoff_pipeline.supervision_recovery_unified_g_leafgrid_r100_full_coverage.toml",
        template_overrides=_merged_overrides(
            V3_T128_SURFACE_OVERRIDES,
            {
                "tradeoff_pipeline.supervision_recovery_packages": [
                    "mass_preserving_leaf_only_deciles"
                ],
                "tradeoff_pipeline.supervision_recovery_train_docs": list(
                    LARGE_TRAIN_DOC_LADDER
                ),
                "tradeoff_pipeline.supervision_recovery_seeds": [0],
                "tradeoff_pipeline.supervision_recovery_depth_discount_gammas": [1.0],
                "tradeoff_pipeline.supervision_recovery_leaf_token_ladder": list(
                    MULTILEAF_ENDPOINT_LEAF_TOKEN_LADDER
                ),
                "tradeoff_pipeline.tree_stage1_artifact_root": (
                    "outputs/_stage1_artifacts/markov_supervision_recovery_unified_g_mass_preserving_leaf_only_large_train"
                ),
            },
        ),
        comparisons=(
            "Root-share decile ladder with total supervision mass fixed at 100%, and every non-root token of that budget placed on leaves only.",
            "Packages include the matching root-only baselines (`root90`, `root80`, ... `root10`) next to the mass-preserving alternatives (`root90_leaf10`, ... `root0_leaf100`).",
            "Multi-leaf geometries 64 / 32 / 16 / 8, where leaf-only local supervision is always meaningful and 64-token leaves realize the simple 50/50 root-vs-leaf endpoint.",
            "Large-train follow-up: 10240 docs with seed 0 and gamma 1.0.",
        ),
        groups=("depth_redistribution_large_train", "depth_redistribution_large_leaf_only"),
        notes=(
            "This is the direct `50/50/0`-style family at 10240 docs across the full root-share decile ladder.",
        ),
    ),
    GridJobSpec(
        key="mass_preserving_depth_equal_large_train",
        label="10240-doc mass-preserving depth-equal grid",
        category="depth_redistribution_large_train",
        config_relpath="config/markov/tradeoff_pipeline.supervision_recovery_unified_g_leafgrid_r100_full_coverage.toml",
        template_overrides=_merged_overrides(
            V3_T128_SURFACE_OVERRIDES,
            {
                "tradeoff_pipeline.supervision_recovery_packages": [
                    "mass_preserving_levels_equal_deciles"
                ],
                "tradeoff_pipeline.supervision_recovery_train_docs": list(
                    LARGE_TRAIN_DOC_LADDER
                ),
                "tradeoff_pipeline.supervision_recovery_seeds": [0],
                "tradeoff_pipeline.supervision_recovery_depth_discount_gammas": [1.0],
                "tradeoff_pipeline.supervision_recovery_leaf_token_ladder": list(
                    DEPTH_EQUAL_REDISTRIBUTION_LEAF_TOKEN_LADDER
                ),
                "tradeoff_pipeline.tree_stage1_artifact_root": (
                    "outputs/_stage1_artifacts/markov_supervision_recovery_unified_g_mass_preserving_depth_equal_large_train"
                ),
            },
        ),
        comparisons=(
            "Root-share decile ladder with total supervision mass fixed at 100%, and the non-root budget spread evenly over leaves plus every available non-root merge depth.",
            "This realizes geometry-aware profiles such as 50/25/25 at leaf32, 25/25/25/25 at leaf16, and equal multi-level depth splits at leaf8.",
            "Packages include the matching root-only baselines (`root90`, `root80`, ... `root10`) next to the depth-equal mass-preserving alternatives (`root90_levels_equal10`, ... `root0_levels_equal100`).",
            "Large-train follow-up: 10240 docs with seed 0 and gamma 1.0.",
        ),
        groups=("depth_redistribution_large_train", "depth_redistribution_large_levels_equal"),
        notes=(
            "This is the 10240-doc depth-equal follow-up surface once the current max-internal-depth drift bug is fixed.",
        ),
    ),
    GridJobSpec(
        key="root_budget_ladder_large_train_longschedule",
        label="10240-doc root-budget long-schedule tuning",
        category="depth_redistribution_large_train_tuning",
        config_relpath="config/markov/tradeoff_pipeline.supervision_recovery_unified_g_leafgrid_r100_full_coverage.toml",
        template_overrides=_merged_overrides(
            V3_T128_SURFACE_OVERRIDES,
            {
                "tradeoff_pipeline.supervision_recovery_packages": list(
                    LARGE_TRAIN_TUNING_ROOT_PACKAGES
                ),
                "tradeoff_pipeline.supervision_recovery_train_docs": list(
                    LARGE_TRAIN_DOC_LADDER
                ),
                "tradeoff_pipeline.supervision_recovery_seeds": [0],
                "tradeoff_pipeline.supervision_recovery_depth_discount_gammas": [1.0],
                "tradeoff_pipeline.supervision_recovery_leaf_token_ladder": list(
                    FULL_COVERAGE_LEAF_TOKEN_LADDER
                ),
                "tradeoff_pipeline.supervision_epochs": 52,
                "tradeoff_pipeline.tree_training_schedule": "two_stage",
                "tradeoff_pipeline.tree_stage1_epochs": 12,
                "tradeoff_pipeline.tree_stage2_epochs": 40,
                "tradeoff_pipeline.tree_stage1_artifact_root": (
                    "outputs/_stage1_artifacts/markov_supervision_recovery_unified_g_root_budget_ladder_large_train_longschedule"
                ),
            },
        ),
        comparisons=(
            "Focused 10240-doc rerun of the root-budget cells that showed the clearest non-monotone regressions versus 4096 docs.",
            "Uses the same package semantics and geometry ladder as the main root-budget study, but lifts the tree schedule from 10+30 epochs to a 12+40 long-schedule follow-up.",
            "Targets the current offenders first: full10 / full20 / full30 / full40 / full50 / full70 across the full geometry ladder.",
            "This is a tuning surface for checking whether the large-train regressions are optimization-limited rather than scientific.",
        ),
        groups=("depth_redistribution_large_train_tuning",),
        notes=(
            "This is intentionally narrower than the full 10240 root-budget ladder so we can test whether the known regression cells recover under a stronger schedule before widening the rerun.",
        ),
    ),
    GridJobSpec(
        key="mass_preserving_leaf_only_large_train_longschedule",
        label="10240-doc leaf-only long-schedule tuning",
        category="depth_redistribution_large_train_tuning",
        config_relpath="config/markov/tradeoff_pipeline.supervision_recovery_unified_g_leafgrid_r100_full_coverage.toml",
        template_overrides=_merged_overrides(
            V3_T128_SURFACE_OVERRIDES,
            {
                "tradeoff_pipeline.supervision_recovery_packages": list(
                    LARGE_TRAIN_TUNING_LEAF_ONLY_PACKAGES
                ),
                "tradeoff_pipeline.supervision_recovery_train_docs": list(
                    LARGE_TRAIN_DOC_LADDER
                ),
                "tradeoff_pipeline.supervision_recovery_seeds": [0],
                "tradeoff_pipeline.supervision_recovery_depth_discount_gammas": [1.0],
                "tradeoff_pipeline.supervision_recovery_leaf_token_ladder": list(
                    MULTILEAF_ENDPOINT_LEAF_TOKEN_LADDER
                ),
                "tradeoff_pipeline.supervision_epochs": 52,
                "tradeoff_pipeline.tree_training_schedule": "two_stage",
                "tradeoff_pipeline.tree_stage1_epochs": 12,
                "tradeoff_pipeline.tree_stage2_epochs": 40,
                "tradeoff_pipeline.tree_stage1_artifact_root": (
                    "outputs/_stage1_artifacts/markov_supervision_recovery_unified_g_mass_preserving_leaf_only_large_train_longschedule"
                ),
            },
        ),
        comparisons=(
            "Focused 10240-doc rerun of the leaf-only mass-preserving cells that regressed most clearly versus 4096 docs, especially the very low-root-share settings.",
            "Keeps the same leaf-only semantics and geometry ladder, but upgrades the tree schedule from 10+30 epochs to 12+40.",
            "Targets the most informative local-only packages first: root0_leaf100, root10_leaf90, root30_leaf70, plus stronger-root control lanes root70_leaf30 / root80_leaf20 / root90_leaf10.",
            "This is the first large-train tuning pass for asking whether the apparent 10240 regressions are just undertraining at the hard low-root end.",
        ),
        groups=("depth_redistribution_large_train_tuning",),
        notes=(
            "Depth-equal tuning is intentionally excluded from this first pass until the relaunch of the main 10240 depth-equal bundle finishes, so we tune against complete data rather than a partial landing set.",
        ),
    ),
    GridJobSpec(
        key="oneleaf_root_budget_publication_fullval",
        label="Publication rerun: one-leaf root-budget with full-val selection",
        category="publication_followup",
        config_relpath="config/markov/tradeoff_pipeline.supervision_recovery_unified_g_leafgrid_r100_full_coverage.toml",
        template_overrides=_merged_overrides(
            PUBLICATION_FULLVAL_OVERRIDES,
            {
                "tradeoff_pipeline.supervision_recovery_packages": ["root_ladder_deciles"],
                "tradeoff_pipeline.supervision_recovery_train_docs": list(
                    SMALL_TRAIN_DOC_LADDER + LARGE_TRAIN_DOC_LADDER
                ),
                "tradeoff_pipeline.supervision_recovery_seeds": list(
                    PUBLICATION_FULLVAL_SEEDS
                ),
                "tradeoff_pipeline.supervision_recovery_depth_discount_gammas": [1.0],
                "tradeoff_pipeline.supervision_recovery_leaf_token_ladder": [128],
                "tradeoff_pipeline.tree_stage1_artifact_root": (
                    "outputs/_stage1_artifacts/markov_supervision_recovery_publication_oneleaf_root_budget_fullval"
                ),
            },
        ),
        comparisons=(
            "Exact-collapse leaf128 root-budget ladder at full100 / full90 / ... / full10 with one-tree identity routing and full-validation checkpoint selection.",
            "Train-doc ladder 1024 / 4096 / 10240 with seeds 0 / 1 / 2, so the one-leaf tree can be compared directly against its FNO analogue without screened-validation noise.",
            "This is the publication rerun for the previously contaminated one-leaf root-share surface.",
        ),
        groups=("publication_fullval", "publication_oneleaf"),
        notes=(
            "Uses full validation rather than a 128-doc screen and disables stage-1 artifact reuse so the resulting one-leaf curves are publication-safe.",
        ),
    ),
    GridJobSpec(
        key="oneleaf_root_budget_longschedule_fill_fullval",
        label="Fixed-doc plot filler: one-leaf root-budget long-schedule",
        category="publication_followup",
        config_relpath="config/markov/tradeoff_pipeline.supervision_recovery_unified_g_leafgrid_r100_full_coverage.toml",
        template_overrides=_merged_overrides(
            PUBLICATION_FULLVAL_OVERRIDES,
            ONE_LEAF_FNO_PARITY_OVERRIDES,
            {
                "tradeoff_pipeline.supervision_recovery_packages": list(
                    PUBLICATION_FIXED_DOC_ROOT_ONLY_PACKAGES
                ),
                "tradeoff_pipeline.supervision_recovery_train_docs": [10240],
                "tradeoff_pipeline.supervision_recovery_seeds": [0],
                "tradeoff_pipeline.supervision_recovery_depth_discount_gammas": [1.0],
                "tradeoff_pipeline.supervision_recovery_leaf_token_ladder": [128],
                "tradeoff_pipeline.tree_training_schedule": "two_stage",
                "tradeoff_pipeline.tree_stage1_epochs": 12,
                "tradeoff_pipeline.tree_stage2_epochs": 40,
                "tradeoff_pipeline.tree_stage1_artifact_root": (
                    "outputs/_stage1_artifacts/markov_supervision_recovery_publication_oneleaf_root_budget_longschedule_fill_fullval"
                ),
            },
        ),
        comparisons=(
            "Targeted `leaf128` long-schedule fill for the fixed-10240 publication leaf-size plots.",
            "Runs the one-leaf exact-collapse root-only ladder at R100 / R90 / R80 / R70 / R50 / R20 / R10 with seed 0.",
            "This is not a new headline surface; it exists so the alternate root-only schedule can be shown honestly at `leaf128` when desired.",
        ),
        groups=("publication_plot_fillers",),
        notes=(
            "This is a narrow plot-filler rerun rather than a full publication bundle.",
        ),
    ),
    GridJobSpec(
        key="oneleaf_local_law_root_sweep_fullval",
        label="Fixed-doc plot filler: one-leaf duplicate-local root sweep",
        category="publication_followup",
        config_relpath="config/markov/tradeoff_pipeline.supervision_recovery_unified_g_ablation_ladder.toml",
        template_overrides=_merged_overrides(
            PUBLICATION_FULLVAL_OVERRIDES,
            {
                "tradeoff_pipeline.tree_reference.preset": "full_laws",
                "tradeoff_pipeline.structural_tree_reference.preset": "full_laws",
                "tradeoff_pipeline.supervision_recovery_packages": list(
                    PUBLICATION_FIXED_DOC_ONELEAF_LOCAL_LAW_PACKAGES
                ),
                "tradeoff_pipeline.supervision_recovery_train_docs": [10240],
                "tradeoff_pipeline.supervision_recovery_seeds": list(
                    PUBLICATION_FULLVAL_SEEDS
                ),
                "tradeoff_pipeline.supervision_recovery_depth_discount_gammas": [1.0],
                "tradeoff_pipeline.supervision_recovery_leaf_token_ladder": [128],
                "tradeoff_pipeline.tree_stage1_artifact_root": (
                    "outputs/_stage1_artifacts/markov_supervision_recovery_publication_oneleaf_local_law_root_sweep_fullval"
                ),
            },
        ),
        comparisons=(
            "Targeted one-leaf duplicate-local-label sweep for the fixed-10240 publication leaf-size plots.",
            "Runs `full100 / full90 / full80 / full70 / full50 / full20 / full10` with the `leaf_full100_internal_count100` one-leaf package.",
            "This is the redundancy/no-harm one-leaf local-label probe across the same root-share panels as the root-only ladder.",
        ),
        groups=("publication_plot_fillers",),
        notes=(
            "This fills the missing `leaf128` duplicate-local column in the fixed-doc publication plots.",
        ),
    ),
    GridJobSpec(
        key="oneleaf_leaf_mass_root_sweep_fullval",
        label="Fixed-doc plot filler: one-leaf equal-mass leaf-only root sweep",
        category="publication_followup",
        config_relpath="config/markov/tradeoff_pipeline.supervision_recovery_unified_g_leafgrid_r100_full_coverage.toml",
        template_overrides=_merged_overrides(
            PUBLICATION_FULLVAL_OVERRIDES,
            {
                "tradeoff_pipeline.supervision_recovery_packages": list(
                    PUBLICATION_FIXED_DOC_ONELEAF_LEAF_MASS_PACKAGES
                ),
                "tradeoff_pipeline.supervision_recovery_train_docs": [10240],
                "tradeoff_pipeline.supervision_recovery_seeds": [0],
                "tradeoff_pipeline.supervision_recovery_depth_discount_gammas": [1.0],
                "tradeoff_pipeline.supervision_recovery_leaf_token_ladder": [128],
                "tradeoff_pipeline.tree_stage1_artifact_root": (
                    "outputs/_stage1_artifacts/markov_supervision_recovery_publication_oneleaf_leaf_mass_root_sweep_fullval"
                ),
            },
        ),
        comparisons=(
            "Targeted one-leaf count-only equal-total-mass sweep for the fixed-10240 publication leaf-size plots.",
            "Runs the one-leaf leaf-mass packages at R90 / R80 / R70 / R50 / R20 / R10 with seed 0.",
            "This fills the missing `leaf128` point for the blue equal-total-mass comparison line without rerunning the larger multileaf grid.",
        ),
        groups=("publication_plot_fillers",),
        notes=(
            "This is a narrow plot-filler rerun rather than a new headline surface.",
        ),
    ),
    GridJobSpec(
        key="root_budget_publication_multileaf_fullval",
        label="Publication rerun: multileaf root-budget focus with full-val selection",
        category="publication_followup",
        config_relpath="config/markov/tradeoff_pipeline.supervision_recovery_unified_g_leafgrid_r100_full_coverage.toml",
        template_overrides=_merged_overrides(
            PUBLICATION_FULLVAL_OVERRIDES,
            {
                "tradeoff_pipeline.supervision_recovery_packages": list(
                    PUBLICATION_FULLVAL_ROOT_ONLY_PACKAGES
                ),
                "tradeoff_pipeline.supervision_recovery_train_docs": [4096, 10240],
                "tradeoff_pipeline.supervision_recovery_seeds": list(
                    PUBLICATION_FULLVAL_SEEDS
                ),
                "tradeoff_pipeline.supervision_recovery_depth_discount_gammas": [1.0],
                "tradeoff_pipeline.supervision_recovery_leaf_token_ladder": [64, 32],
                "tradeoff_pipeline.tree_stage1_artifact_root": (
                    "outputs/_stage1_artifacts/markov_supervision_recovery_publication_root_budget_multileaf_fullval"
                ),
            },
        ),
        comparisons=(
            "Root-only multileaf controls at full100 / full90 / full80 / full70 / full50 on the strongest current geometries 64 / 32.",
            "Focused on 4096 / 10240 docs with seeds 0 / 1 / 2, to measure how much root budget alone can be reduced once checkpoint selection uses the full validation split.",
            "This is the clean root-only control surface for the publication reruns.",
        ),
        groups=("publication_fullval", "publication_multileaf"),
        notes=(
            "This rerun targets the regimes that currently look closest to the publishable story: shallower trees with high but not full root supervision.",
        ),
    ),
    GridJobSpec(
        key="leaf_only_publication_focus_fullval",
        label="Publication rerun: leaf-only redistribution focus with full-val selection",
        category="publication_followup",
        config_relpath="config/markov/tradeoff_pipeline.supervision_recovery_unified_g_leafgrid_r100_full_coverage.toml",
        template_overrides=_merged_overrides(
            PUBLICATION_FULLVAL_OVERRIDES,
            {
                "tradeoff_pipeline.supervision_recovery_packages": list(
                    PUBLICATION_FULLVAL_LEAF_ONLY_PACKAGES
                ),
                "tradeoff_pipeline.supervision_recovery_train_docs": [4096, 10240],
                "tradeoff_pipeline.supervision_recovery_seeds": list(
                    PUBLICATION_FULLVAL_SEEDS
                ),
                "tradeoff_pipeline.supervision_recovery_depth_discount_gammas": [1.0],
                "tradeoff_pipeline.supervision_recovery_leaf_token_ladder": [64, 32],
                "tradeoff_pipeline.tree_stage1_artifact_root": (
                    "outputs/_stage1_artifacts/markov_supervision_recovery_publication_leaf_only_fullval"
                ),
            },
        ),
        comparisons=(
            "Mass-preserving leaf-only redistribution at root90/80/70/50 with matching local shares 10/20/30/50.",
            "Focused on 64 / 32-token leaves, 4096 / 10240 docs, and seeds 0 / 1 / 2 to test whether leaf-only local supervision can reliably replace part of the root budget.",
            "This is the main publication follow-up for the `50/50/0` family after removing screened-validation checkpoint selection.",
        ),
        groups=("publication_fullval", "publication_multileaf"),
        notes=(
            "These are the leaf-only settings that currently look most competitive in the fresh multileaf report.",
        ),
    ),
    GridJobSpec(
        key="depth_equal_publication_focus_fullval",
        label="Publication rerun: depth-equal redistribution focus with full-val selection",
        category="publication_followup",
        config_relpath="config/markov/tradeoff_pipeline.supervision_recovery_unified_g_leafgrid_r100_full_coverage.toml",
        template_overrides=_merged_overrides(
            PUBLICATION_FULLVAL_OVERRIDES,
            {
                "tradeoff_pipeline.supervision_recovery_packages": list(
                    PUBLICATION_FULLVAL_DEPTH_EQUAL_PACKAGES
                ),
                "tradeoff_pipeline.supervision_recovery_train_docs": [4096, 10240],
                "tradeoff_pipeline.supervision_recovery_seeds": list(
                    PUBLICATION_FULLVAL_SEEDS
                ),
                "tradeoff_pipeline.supervision_recovery_depth_discount_gammas": [1.0],
                "tradeoff_pipeline.supervision_recovery_leaf_token_ladder": [32, 16, 8],
                "tradeoff_pipeline.tree_stage1_artifact_root": (
                    "outputs/_stage1_artifacts/markov_supervision_recovery_publication_depth_equal_fullval"
                ),
            },
        ),
        comparisons=(
            "Mass-preserving depth-equal redistribution at root90/80/70/50 with the non-root budget spread over all available non-root depths.",
            "Focused on 32 / 16 / 8-token leaves, 4096 / 10240 docs, and seeds 0 / 1 / 2, where the current fresh results suggest depth-aware redistribution can help on structural recovery.",
            "This is the publication rerun for the `50/25/25` and deeper equal-depth families under full-validation checkpointing.",
        ),
        groups=("publication_fullval", "publication_multileaf"),
        notes=(
            "These settings are selected from the strongest currently completed depth-equal cells rather than from the full decile ladder.",
        ),
    ),
    GridJobSpec(
        key="local_law_publication_fullval",
        label="Publication rerun: full local-law multileaf focus with full-val selection",
        category="publication_followup",
        config_relpath="config/markov/tradeoff_pipeline.supervision_recovery_unified_g_multi_leaf_ablation.toml",
        template_overrides=_merged_overrides(
            PUBLICATION_FULLVAL_OVERRIDES,
            {
                "tradeoff_pipeline.tree_reference.preset": "standard_tree",
                "tradeoff_pipeline.structural_tree_reference.preset": "standard_tree",
                "tradeoff_pipeline.supervision_recovery_packages": list(
                    PUBLICATION_FULLVAL_FULL_LOCAL_LAW_PACKAGES
                ),
                "tradeoff_pipeline.supervision_recovery_train_docs": [4096, 10240],
                "tradeoff_pipeline.supervision_recovery_seeds": list(
                    PUBLICATION_FULLVAL_SEEDS
                ),
                "tradeoff_pipeline.supervision_recovery_depth_discount_gammas": [1.0],
                "tradeoff_pipeline.supervision_recovery_leaf_token_ladder": [64, 32, 16, 8],
                "tradeoff_pipeline.tree_stage1_artifact_root": (
                    "outputs/_stage1_artifacts/markov_supervision_recovery_publication_local_law_fullval"
                ),
            },
        ),
        comparisons=(
            "Publication rerun of the strongest full-local-law versus root-only multileaf control surface.",
            "Compares `full100` against `root100_extra_leaffull100_internalcount100` on 64 / 32 / 16 / 8-token leaves.",
            "Uses 4096 / 10240 docs, seeds 0 / 1 / 2, and full-validation checkpoint selection rather than the old screened-validation path.",
        ),
        groups=("publication_fullval", "publication_multileaf", "publication_local_law"),
        notes=(
            "This is the clean publication rerun for the main local-law validity claim.",
        ),
    ),
    GridJobSpec(
        key="r100_superset_local10_publication_fullval",
        label="Publication rerun: R100 +10% superset local-law focus with full-val selection",
        category="publication_followup",
        config_relpath="config/markov/tradeoff_pipeline.superset_multileaf_lossnorm_ablation.toml",
        template_overrides=_merged_overrides(
            PUBLICATION_FULLVAL_OVERRIDES,
            {
                "tradeoff_pipeline.tree_reference.preset": "standard_tree",
                "tradeoff_pipeline.structural_tree_reference.preset": "standard_tree",
                "tradeoff_pipeline.supervision_recovery_packages": list(
                    PUBLICATION_FULLVAL_R100_SUPERSET_LOCAL10_PACKAGES
                ),
                "tradeoff_pipeline.supervision_recovery_train_docs": [4096, 10240],
                "tradeoff_pipeline.supervision_recovery_seeds": list(
                    PUBLICATION_FULLVAL_SEEDS
                ),
                "tradeoff_pipeline.supervision_recovery_depth_discount_gammas": [1.0],
                "tradeoff_pipeline.supervision_recovery_leaf_token_ladder": [64, 32, 16, 8],
                "tradeoff_pipeline.tree_stage1_artifact_root": (
                    "outputs/_stage1_artifacts/markov_supervision_recovery_publication_r100_superset_local10_fullval"
                ),
            },
        ),
        comparisons=(
            "Publication rerun of the mild `root100 + 10%` superset local-label surface against pure `root100`.",
            "Runs the multileaf ladder 64 / 32 / 16 / 8 at 4096 / 10240 docs with seeds 0 / 1 / 2.",
            "Uses the same full-validation checkpoint selection policy as the fresh publication root-budget and redistribution reruns.",
        ),
        groups=("publication_fullval", "publication_multileaf", "publication_local_law"),
        notes=(
            "This is the clean publication rerun for the milder `r100 + local` claim.",
        ),
    ),
    GridJobSpec(
        key="root_budget_ladder_xlarge_train",
        label="20480-doc root-budget ladder",
        category="depth_redistribution_xlarge",
        config_relpath="config/markov/tradeoff_pipeline.supervision_recovery_unified_g_leafgrid_r100_full_coverage.toml",
        template_overrides=_merged_overrides(
            V3_T128_SURFACE_OVERRIDES,
            {
                "tradeoff_pipeline.supervision_recovery_packages": ["root_ladder_deciles"],
                "tradeoff_pipeline.supervision_recovery_train_docs": list(XLARGE_TRAIN_DOC_LADDER),
                "tradeoff_pipeline.supervision_recovery_seeds": [0],
                "tradeoff_pipeline.supervision_recovery_depth_discount_gammas": [1.0],
                "tradeoff_pipeline.supervision_recovery_leaf_token_ladder": list(
                    FULL_COVERAGE_LEAF_TOKEN_LADDER
                ),
                "tradeoff_pipeline.tree_stage1_artifact_root": (
                    "outputs/_stage1_artifacts/markov_supervision_recovery_unified_g_root_budget_ladder_xlarge_train"
                ),
            },
        ),
        comparisons=(
            "Root-only decile ladder at 20480 docs on the t128 recoverable and structural publication surface.",
            "Geometries 128 / 64 / 32 / 16 / 8 with direct FNO comparators preserved at the one-leaf endpoint.",
            "Seed 0 first-pass xlarge extension for asking whether more data stabilizes the low-root regimes.",
        ),
        groups=("depth_redistribution_xlarge", "overnight_xlarge"),
        notes=(
            "This is the 20480-doc extension of the root-budget comparator ladder.",
        ),
    ),
    GridJobSpec(
        key="mass_preserving_leaf_only_xlarge_train",
        label="20480-doc mass-preserving leaf-only grid",
        category="depth_redistribution_xlarge",
        config_relpath="config/markov/tradeoff_pipeline.supervision_recovery_unified_g_leafgrid_r100_full_coverage.toml",
        template_overrides=_merged_overrides(
            V3_T128_SURFACE_OVERRIDES,
            {
                "tradeoff_pipeline.supervision_recovery_packages": [
                    "mass_preserving_leaf_only_deciles"
                ],
                "tradeoff_pipeline.supervision_recovery_train_docs": list(
                    XLARGE_TRAIN_DOC_LADDER
                ),
                "tradeoff_pipeline.supervision_recovery_seeds": [0],
                "tradeoff_pipeline.supervision_recovery_depth_discount_gammas": [1.0],
                "tradeoff_pipeline.supervision_recovery_leaf_token_ladder": list(
                    MULTILEAF_ENDPOINT_LEAF_TOKEN_LADDER
                ),
                "tradeoff_pipeline.tree_stage1_artifact_root": (
                    "outputs/_stage1_artifacts/markov_supervision_recovery_unified_g_mass_preserving_leaf_only_xlarge_train"
                ),
            },
        ),
        comparisons=(
            "Leaf-only mass-preserving decile ladder at 20480 docs on the t128 publication surface.",
            "Multi-leaf geometries 64 / 32 / 16 / 8, where the non-root budget lives entirely on leaves.",
            "Seed 0 xlarge follow-up for the direct `50/50/0` family.",
        ),
        groups=("depth_redistribution_xlarge", "overnight_xlarge"),
        notes=(
            "This is the 20480-doc extension of the leaf-only redistribution study.",
        ),
    ),
    GridJobSpec(
        key="mass_preserving_depth_equal_xlarge_train",
        label="20480-doc mass-preserving depth-equal grid",
        category="depth_redistribution_xlarge",
        config_relpath="config/markov/tradeoff_pipeline.supervision_recovery_unified_g_leafgrid_r100_full_coverage.toml",
        template_overrides=_merged_overrides(
            V3_T128_SURFACE_OVERRIDES,
            {
                "tradeoff_pipeline.supervision_recovery_packages": [
                    "mass_preserving_levels_equal_deciles"
                ],
                "tradeoff_pipeline.supervision_recovery_train_docs": list(
                    XLARGE_TRAIN_DOC_LADDER
                ),
                "tradeoff_pipeline.supervision_recovery_seeds": [0],
                "tradeoff_pipeline.supervision_recovery_depth_discount_gammas": [1.0],
                "tradeoff_pipeline.supervision_recovery_leaf_token_ladder": list(
                    DEPTH_EQUAL_REDISTRIBUTION_LEAF_TOKEN_LADDER
                ),
                "tradeoff_pipeline.tree_stage1_artifact_root": (
                    "outputs/_stage1_artifacts/markov_supervision_recovery_unified_g_mass_preserving_depth_equal_xlarge_train"
                ),
            },
        ),
        comparisons=(
            "Depth-equal mass-preserving decile ladder at 20480 docs on the t128 publication surface.",
            "Geometries 32 / 16 / 8, where the local mass is spread over leaves and all available non-root merge depths.",
            "Seed 0 xlarge follow-up for the `50/25/25` and deeper equal-depth families.",
        ),
        groups=("depth_redistribution_xlarge", "overnight_xlarge"),
        notes=(
            "This is the 20480-doc extension of the depth-aware redistribution study.",
        ),
    ),
    GridJobSpec(
        key="oneleaf_root_budget_publication_xlarge",
        label="Publication xlarge: one-leaf root-budget with full-val selection",
        category="publication_xlarge",
        config_relpath="config/markov/tradeoff_pipeline.supervision_recovery_unified_g_leafgrid_r100_full_coverage.toml",
        template_overrides=_merged_overrides(
            PUBLICATION_FULLVAL_OVERRIDES,
            {
                "tradeoff_pipeline.supervision_recovery_packages": ["root_ladder_deciles"],
                "tradeoff_pipeline.supervision_recovery_train_docs": list(
                    XLARGE_TRAIN_DOC_LADDER
                ),
                "tradeoff_pipeline.supervision_recovery_seeds": list(
                    PUBLICATION_XLARGE_SEEDS
                ),
                "tradeoff_pipeline.supervision_recovery_depth_discount_gammas": [1.0],
                "tradeoff_pipeline.supervision_recovery_leaf_token_ladder": [128],
                "tradeoff_pipeline.tree_stage1_artifact_root": (
                    "outputs/_stage1_artifacts/markov_supervision_recovery_publication_oneleaf_root_budget_xlarge"
                ),
            },
        ),
        comparisons=(
            "Exact-collapse leaf128 root-budget ladder at 20480 docs with matched-root one-leaf routing and full-validation checkpoint selection.",
            "Seed 0 xlarge publication extension for the most direct tree-vs-FNO root-budget comparison.",
        ),
        groups=("publication_xlarge", "publication_oneleaf_xlarge", "overnight_xlarge"),
        notes=(
            "This extends the corrected one-leaf publication surface beyond 10240 docs.",
        ),
    ),
    GridJobSpec(
        key="oneleaf_root_budget_longschedule_fill_xlarge",
        label="Fixed-doc plot filler xlarge: one-leaf root-budget long-schedule",
        category="publication_xlarge",
        config_relpath="config/markov/tradeoff_pipeline.supervision_recovery_unified_g_leafgrid_r100_full_coverage.toml",
        template_overrides=_merged_overrides(
            PUBLICATION_FULLVAL_OVERRIDES,
            ONE_LEAF_FNO_PARITY_OVERRIDES,
            {
                "tradeoff_pipeline.supervision_recovery_packages": list(
                    PUBLICATION_FIXED_DOC_ROOT_ONLY_PACKAGES
                ),
                "tradeoff_pipeline.supervision_recovery_train_docs": list(
                    XLARGE_TRAIN_DOC_LADDER
                ),
                "tradeoff_pipeline.supervision_recovery_seeds": [0],
                "tradeoff_pipeline.supervision_recovery_depth_discount_gammas": [1.0],
                "tradeoff_pipeline.supervision_recovery_leaf_token_ladder": [128],
                "tradeoff_pipeline.tree_training_schedule": "two_stage",
                "tradeoff_pipeline.tree_stage1_epochs": 12,
                "tradeoff_pipeline.tree_stage2_epochs": 40,
                "tradeoff_pipeline.tree_stage1_artifact_root": (
                    "outputs/_stage1_artifacts/markov_supervision_recovery_publication_oneleaf_root_budget_longschedule_fill_xlarge"
                ),
            },
        ),
        comparisons=(
            "Targeted `leaf128` long-schedule fill at 20480 docs for the fixed-doc publication leaf-size plots.",
            "Runs the same R100 / R90 / R80 / R70 / R50 / R20 / R10 one-leaf exact-collapse ladder with seed 0.",
        ),
        groups=("publication_plot_fillers",),
        notes=(
            "This is a narrow xlarge plot-filler rerun rather than a new main publication surface.",
        ),
    ),
    GridJobSpec(
        key="oneleaf_local_law_root_sweep_xlarge",
        label="Fixed-doc plot filler xlarge: one-leaf duplicate-local root sweep",
        category="publication_xlarge",
        config_relpath="config/markov/tradeoff_pipeline.supervision_recovery_unified_g_ablation_ladder.toml",
        template_overrides=_merged_overrides(
            PUBLICATION_FULLVAL_OVERRIDES,
            {
                "tradeoff_pipeline.tree_reference.preset": "full_laws",
                "tradeoff_pipeline.structural_tree_reference.preset": "full_laws",
                "tradeoff_pipeline.supervision_recovery_packages": list(
                    PUBLICATION_FIXED_DOC_ONELEAF_LOCAL_LAW_PACKAGES
                ),
                "tradeoff_pipeline.supervision_recovery_train_docs": list(
                    XLARGE_TRAIN_DOC_LADDER
                ),
                "tradeoff_pipeline.supervision_recovery_seeds": list(
                    PUBLICATION_XLARGE_SEEDS
                ),
                "tradeoff_pipeline.supervision_recovery_depth_discount_gammas": [1.0],
                "tradeoff_pipeline.supervision_recovery_leaf_token_ladder": [128],
                "tradeoff_pipeline.tree_stage1_artifact_root": (
                    "outputs/_stage1_artifacts/markov_supervision_recovery_publication_oneleaf_local_law_root_sweep_xlarge"
                ),
            },
        ),
        comparisons=(
            "Targeted one-leaf duplicate-local-label sweep at 20480 docs for the fixed-doc publication leaf-size plots.",
            "Runs the same one-leaf `leaf_full100_internal_count100` root-share ladder with seed 0.",
        ),
        groups=("publication_plot_fillers",),
        notes=(
            "This fills the missing xlarge `leaf128` duplicate-local column for the same root-share panels.",
        ),
    ),
    GridJobSpec(
        key="oneleaf_leaf_mass_root_sweep_xlarge",
        label="Fixed-doc plot filler xlarge: one-leaf equal-mass leaf-only root sweep",
        category="publication_xlarge",
        config_relpath="config/markov/tradeoff_pipeline.supervision_recovery_unified_g_leafgrid_r100_full_coverage.toml",
        template_overrides=_merged_overrides(
            PUBLICATION_FULLVAL_OVERRIDES,
            {
                "tradeoff_pipeline.supervision_recovery_packages": list(
                    PUBLICATION_FIXED_DOC_ONELEAF_LEAF_MASS_PACKAGES
                ),
                "tradeoff_pipeline.supervision_recovery_train_docs": list(
                    XLARGE_TRAIN_DOC_LADDER
                ),
                "tradeoff_pipeline.supervision_recovery_seeds": [0],
                "tradeoff_pipeline.supervision_recovery_depth_discount_gammas": [1.0],
                "tradeoff_pipeline.supervision_recovery_leaf_token_ladder": [128],
                "tradeoff_pipeline.tree_stage1_artifact_root": (
                    "outputs/_stage1_artifacts/markov_supervision_recovery_publication_oneleaf_leaf_mass_root_sweep_xlarge"
                ),
            },
        ),
        comparisons=(
            "Targeted one-leaf count-only equal-total-mass sweep at 20480 docs for the fixed-doc publication leaf-size plots.",
            "Runs the one-leaf leaf-mass packages at R90 / R80 / R70 / R50 / R20 / R10 with seed 0.",
            "This fills the missing xlarge `leaf128` point for the blue equal-total-mass comparison line without rerunning the larger multileaf grid.",
        ),
        groups=("publication_plot_fillers",),
        notes=(
            "This is a narrow xlarge plot-filler rerun rather than a new headline surface.",
        ),
    ),
    GridJobSpec(
        key="root_budget_publication_multileaf_xlarge",
        label="Publication xlarge: multileaf root-budget focus",
        category="publication_xlarge",
        config_relpath="config/markov/tradeoff_pipeline.supervision_recovery_unified_g_leafgrid_r100_full_coverage.toml",
        template_overrides=_merged_overrides(
            PUBLICATION_FULLVAL_OVERRIDES,
            {
                "tradeoff_pipeline.supervision_recovery_packages": list(
                    PUBLICATION_FULLVAL_ROOT_ONLY_PACKAGES
                ),
                "tradeoff_pipeline.supervision_recovery_train_docs": list(
                    XLARGE_TRAIN_DOC_LADDER
                ),
                "tradeoff_pipeline.supervision_recovery_seeds": list(
                    PUBLICATION_XLARGE_SEEDS
                ),
                "tradeoff_pipeline.supervision_recovery_depth_discount_gammas": [1.0],
                "tradeoff_pipeline.supervision_recovery_leaf_token_ladder": [64, 32],
                "tradeoff_pipeline.tree_stage1_artifact_root": (
                    "outputs/_stage1_artifacts/markov_supervision_recovery_publication_root_budget_multileaf_xlarge"
                ),
            },
        ),
        comparisons=(
            "Root-only multileaf controls at full100 / full90 / full80 / full70 / full50 on 64 / 32-token leaves at 20480 docs.",
            "Seed 0 xlarge publication extension for the strongest current root-only multileaf surface.",
        ),
        groups=("publication_xlarge", "publication_multileaf_xlarge", "overnight_xlarge"),
        notes=(
            "This extends the clean multileaf root-budget publication surface to 20480 docs.",
        ),
    ),
    GridJobSpec(
        key="leaf_only_publication_focus_xlarge",
        label="Publication xlarge: leaf-only redistribution focus",
        category="publication_xlarge",
        config_relpath="config/markov/tradeoff_pipeline.supervision_recovery_unified_g_leafgrid_r100_full_coverage.toml",
        template_overrides=_merged_overrides(
            PUBLICATION_FULLVAL_OVERRIDES,
            {
                "tradeoff_pipeline.supervision_recovery_packages": list(
                    PUBLICATION_FULLVAL_LEAF_ONLY_PACKAGES
                ),
                "tradeoff_pipeline.supervision_recovery_train_docs": list(
                    XLARGE_TRAIN_DOC_LADDER
                ),
                "tradeoff_pipeline.supervision_recovery_seeds": list(
                    PUBLICATION_XLARGE_SEEDS
                ),
                "tradeoff_pipeline.supervision_recovery_depth_discount_gammas": [1.0],
                "tradeoff_pipeline.supervision_recovery_leaf_token_ladder": [64, 32],
                "tradeoff_pipeline.tree_stage1_artifact_root": (
                    "outputs/_stage1_artifacts/markov_supervision_recovery_publication_leaf_only_xlarge"
                ),
            },
        ),
        comparisons=(
            "Leaf-only mass-preserving publication focus at 20480 docs for the current strongest root90/80/70/50 settings.",
            "Seed 0 xlarge extension of the main `50/50/0` publication family.",
        ),
        groups=("publication_xlarge", "publication_multileaf_xlarge", "overnight_xlarge"),
        notes=(
            "This extends the strongest leaf-only publication cells to 20480 docs.",
        ),
    ),
    GridJobSpec(
        key="depth_equal_publication_focus_xlarge",
        label="Publication xlarge: depth-equal redistribution focus",
        category="publication_xlarge",
        config_relpath="config/markov/tradeoff_pipeline.supervision_recovery_unified_g_leafgrid_r100_full_coverage.toml",
        template_overrides=_merged_overrides(
            PUBLICATION_FULLVAL_OVERRIDES,
            {
                "tradeoff_pipeline.supervision_recovery_packages": list(
                    PUBLICATION_FULLVAL_DEPTH_EQUAL_PACKAGES
                ),
                "tradeoff_pipeline.supervision_recovery_train_docs": list(
                    XLARGE_TRAIN_DOC_LADDER
                ),
                "tradeoff_pipeline.supervision_recovery_seeds": list(
                    PUBLICATION_XLARGE_SEEDS
                ),
                "tradeoff_pipeline.supervision_recovery_depth_discount_gammas": [1.0],
                "tradeoff_pipeline.supervision_recovery_leaf_token_ladder": [32, 16, 8],
                "tradeoff_pipeline.tree_stage1_artifact_root": (
                    "outputs/_stage1_artifacts/markov_supervision_recovery_publication_depth_equal_xlarge"
                ),
            },
        ),
        comparisons=(
            "Depth-equal publication focus at 20480 docs for the current strongest root90/80/70/50 settings.",
            "Seed 0 xlarge extension of the `50/25/25` and deeper publication families.",
        ),
        groups=("publication_xlarge", "publication_multileaf_xlarge", "overnight_xlarge"),
        notes=(
            "This extends the strongest depth-equal publication cells to 20480 docs.",
        ),
    ),
    GridJobSpec(
        key="local_law_publication_xlarge",
        label="Publication xlarge: full local-law multileaf focus",
        category="publication_xlarge",
        config_relpath="config/markov/tradeoff_pipeline.supervision_recovery_unified_g_multi_leaf_ablation.toml",
        template_overrides=_merged_overrides(
            PUBLICATION_FULLVAL_OVERRIDES,
            {
                "tradeoff_pipeline.tree_reference.preset": "standard_tree",
                "tradeoff_pipeline.structural_tree_reference.preset": "standard_tree",
                "tradeoff_pipeline.supervision_recovery_packages": list(
                    PUBLICATION_FULLVAL_FULL_LOCAL_LAW_PACKAGES
                ),
                "tradeoff_pipeline.supervision_recovery_train_docs": list(
                    XLARGE_TRAIN_DOC_LADDER
                ),
                "tradeoff_pipeline.supervision_recovery_seeds": list(
                    PUBLICATION_XLARGE_SEEDS
                ),
                "tradeoff_pipeline.supervision_recovery_depth_discount_gammas": [1.0],
                "tradeoff_pipeline.supervision_recovery_leaf_token_ladder": [64, 32, 16, 8],
                "tradeoff_pipeline.tree_stage1_artifact_root": (
                    "outputs/_stage1_artifacts/markov_supervision_recovery_publication_local_law_xlarge"
                ),
            },
        ),
        comparisons=(
            "Full local-law versus root-only multileaf publication surface at 20480 docs.",
            "Seed 0 xlarge extension of the main local-law validity claim.",
        ),
        groups=("publication_xlarge", "publication_local_law_xlarge", "publication_multileaf_xlarge", "overnight_xlarge"),
        notes=(
            "This extends the strongest full local-law publication cells to 20480 docs.",
        ),
    ),
    GridJobSpec(
        key="r100_superset_local10_publication_xlarge",
        label="Publication xlarge: R100 +10% superset local-law focus",
        category="publication_xlarge",
        config_relpath="config/markov/tradeoff_pipeline.superset_multileaf_lossnorm_ablation.toml",
        template_overrides=_merged_overrides(
            PUBLICATION_FULLVAL_OVERRIDES,
            {
                "tradeoff_pipeline.tree_reference.preset": "standard_tree",
                "tradeoff_pipeline.structural_tree_reference.preset": "standard_tree",
                "tradeoff_pipeline.supervision_recovery_packages": list(
                    PUBLICATION_FULLVAL_R100_SUPERSET_LOCAL10_PACKAGES
                ),
                "tradeoff_pipeline.supervision_recovery_train_docs": list(
                    XLARGE_TRAIN_DOC_LADDER
                ),
                "tradeoff_pipeline.supervision_recovery_seeds": list(
                    PUBLICATION_XLARGE_SEEDS
                ),
                "tradeoff_pipeline.supervision_recovery_depth_discount_gammas": [1.0],
                "tradeoff_pipeline.supervision_recovery_leaf_token_ladder": [64, 32, 16, 8],
                "tradeoff_pipeline.tree_stage1_artifact_root": (
                    "outputs/_stage1_artifacts/markov_supervision_recovery_publication_r100_superset_local10_xlarge"
                ),
            },
        ),
        comparisons=(
            "R100 plus +10% superset local-label publication surface at 20480 docs.",
            "Seed 0 xlarge extension of the mild `r100 + local` claim.",
        ),
        groups=("publication_xlarge", "publication_local_law_xlarge", "publication_multileaf_xlarge", "overnight_xlarge"),
        notes=(
            "This extends the strongest mild-superset publication cells to 20480 docs.",
        ),
    ),
    GridJobSpec(
        key="structural_oneleaf_matched_root_v2_rescue",
        label="Structural one-leaf rescue: matched-root v2",
        category="structural_oneleaf_rescue",
        config_relpath="config/markov/tradeoff_pipeline.supervision_recovery_unified_g_leafgrid_r100_full_coverage.toml",
        template_overrides=_merged_overrides(
            PUBLICATION_FULLVAL_OVERRIDES,
            {
                "tradeoff_pipeline.tree_reference.preset": "root_only_matched_v2",
                "tradeoff_pipeline.structural_tree_reference.preset": "structural_root_only_matched_v2",
                "tradeoff_pipeline.supervision_recovery_packages": list(
                    STRUCTURAL_ONELEAF_RESCUE_PACKAGES
                ),
                "tradeoff_pipeline.supervision_recovery_train_docs": list(
                    STRUCTURAL_ONELEAF_RESCUE_DOCS
                ),
                "tradeoff_pipeline.supervision_recovery_seeds": list(
                    STRUCTURAL_ONELEAF_RESCUE_SEEDS
                ),
                "tradeoff_pipeline.supervision_recovery_depth_discount_gammas": [1.0],
                "tradeoff_pipeline.supervision_recovery_leaf_token_ladder": [128],
                "tradeoff_pipeline.tree_stage1_artifact_root": (
                    "outputs/_stage1_artifacts/markov_supervision_recovery_structural_oneleaf_rescue_matched_v2"
                ),
            },
        ),
        comparisons=(
            "Structural one-leaf partial-root rescue matrix using the structural matched-root v2 preset on `full90 / full80 / full50 / full10`.",
            "Runs the exact-collapse `leaf128` surface at 10240 / 20480 docs with seed 0.",
            "Recoverable scope keeps the recoverable matched-root v2 recipe; structural scope uses the structural matched-root v2 recipe.",
        ),
        groups=("structural_oneleaf_rescue",),
        notes=(
            "This is a targeted structural debug surface, not a publication-best surface.",
        ),
    ),
    GridJobSpec(
        key="structural_oneleaf_matched_root_v3_rescue",
        label="Structural one-leaf rescue: matched-root v3",
        category="structural_oneleaf_rescue",
        config_relpath="config/markov/tradeoff_pipeline.supervision_recovery_unified_g_leafgrid_r100_full_coverage.toml",
        template_overrides=_merged_overrides(
            PUBLICATION_FULLVAL_OVERRIDES,
            {
                "tradeoff_pipeline.tree_reference.preset": "root_only_matched",
                "tradeoff_pipeline.structural_tree_reference.preset": "structural_root_only_matched",
                "tradeoff_pipeline.supervision_recovery_packages": list(
                    STRUCTURAL_ONELEAF_RESCUE_PACKAGES
                ),
                "tradeoff_pipeline.supervision_recovery_train_docs": list(
                    STRUCTURAL_ONELEAF_RESCUE_DOCS
                ),
                "tradeoff_pipeline.supervision_recovery_seeds": list(
                    STRUCTURAL_ONELEAF_RESCUE_SEEDS
                ),
                "tradeoff_pipeline.supervision_recovery_depth_discount_gammas": [1.0],
                "tradeoff_pipeline.supervision_recovery_leaf_token_ladder": [128],
                "tradeoff_pipeline.tree_stage1_artifact_root": (
                    "outputs/_stage1_artifacts/markov_supervision_recovery_structural_oneleaf_rescue_matched_v3"
                ),
            },
        ),
        comparisons=(
            "Structural one-leaf partial-root rescue matrix using the stronger structural matched-root v3 preset on `full90 / full80 / full50 / full10`.",
            "Runs the exact-collapse `leaf128` surface at 10240 / 20480 docs with seed 0.",
            "Recoverable scope keeps the recoverable matched-root v3 recipe; structural scope uses the structural matched-root v3 recipe.",
        ),
        groups=("structural_oneleaf_rescue",),
        notes=(
            "This is the main structural rescue surface for checking whether extra capacity fixes the flat partial-root structural curve.",
        ),
    ),
    GridJobSpec(
        key="structural_oneleaf_recoverable_recipe_v3_rescue",
        label="Structural one-leaf rescue: recoverable recipe on structural task",
        category="structural_oneleaf_rescue",
        config_relpath="config/markov/tradeoff_pipeline.supervision_recovery_unified_g_leafgrid_r100_full_coverage.toml",
        template_overrides=_merged_overrides(
            PUBLICATION_FULLVAL_OVERRIDES,
            {
                "tradeoff_pipeline.tree_reference.preset": "root_only_matched",
                "tradeoff_pipeline.structural_tree_reference.preset": "root_only_matched",
                "tradeoff_pipeline.supervision_recovery_packages": list(
                    STRUCTURAL_ONELEAF_RESCUE_PACKAGES
                ),
                "tradeoff_pipeline.supervision_recovery_train_docs": list(
                    STRUCTURAL_ONELEAF_RESCUE_DOCS
                ),
                "tradeoff_pipeline.supervision_recovery_seeds": list(
                    STRUCTURAL_ONELEAF_RESCUE_SEEDS
                ),
                "tradeoff_pipeline.supervision_recovery_depth_discount_gammas": [1.0],
                "tradeoff_pipeline.supervision_recovery_leaf_token_ladder": [128],
                "tradeoff_pipeline.tree_stage1_artifact_root": (
                    "outputs/_stage1_artifacts/markov_supervision_recovery_structural_oneleaf_rescue_recoverable_recipe_v3"
                ),
            },
        ),
        comparisons=(
            "Structural one-leaf partial-root rescue matrix that deliberately reuses the recoverable matched-root v3 recipe on the structural benchmark.",
            "Runs `full90 / full80 / full50 / full10` at `leaf128`, 10240 / 20480 docs, seed 0.",
            "This isolates whether the structural failure is driven by the structural preset itself rather than by the benchmark/task.",
        ),
        groups=("structural_oneleaf_rescue",),
        notes=(
            "This is diagnostic-only and should not be treated as a publication surface unless it clearly rescues the structural task.",
        ),
    ),
    GridJobSpec(
        key="structural_oneleaf_canary_anchor_rescue",
        label="Structural one-leaf rescue: canary anchor",
        category="structural_oneleaf_rescue",
        config_relpath="config/markov/tradeoff_pipeline.supervision_recovery_unified_g_leafgrid_r100_full_coverage.toml",
        template_overrides=_merged_overrides(
            PUBLICATION_FULLVAL_OVERRIDES,
            {
                "tradeoff_pipeline.tree_reference.preset": "fno_parity_canary",
                "tradeoff_pipeline.structural_tree_reference.preset": "fno_parity_canary",
                "tradeoff_pipeline.one_leaf_tree_reference.preset": "fno_parity_canary",
                "tradeoff_pipeline.supervision_recovery_packages": list(
                    STRUCTURAL_ONELEAF_RESCUE_ANCHOR_PACKAGES
                ),
                "tradeoff_pipeline.supervision_recovery_train_docs": list(
                    STRUCTURAL_ONELEAF_RESCUE_DOCS
                ),
                "tradeoff_pipeline.supervision_recovery_seeds": list(
                    STRUCTURAL_ONELEAF_RESCUE_SEEDS
                ),
                "tradeoff_pipeline.supervision_recovery_depth_discount_gammas": [1.0],
                "tradeoff_pipeline.supervision_recovery_leaf_token_ladder": [128],
                "tradeoff_pipeline.tree_stage1_artifact_root": (
                    "outputs/_stage1_artifacts/markov_supervision_recovery_structural_oneleaf_rescue_canary_anchor"
                ),
            },
        ),
        comparisons=(
            "Structural one-leaf sanity ceiling using the exact canary `full100` anchor on the same benchmark surface.",
            "Runs the exact-collapse `leaf128` surface at 10240 / 20480 docs with seed 0.",
            "This is the structural rescue reference point for checking whether the task itself is compatible with one-leaf root-only learning.",
        ),
        groups=("structural_oneleaf_rescue",),
        notes=(
            "This is the sanity ceiling for the structural one-leaf rescue pass.",
        ),
    ),
    GridJobSpec(
        key="mass_matched_full_coverage",
        label="Optional mass-matched full leaf-coverage sweep",
        category="coverage",
        config_relpath="config/markov/tradeoff_pipeline.supervision_recovery_unified_g_leafgrid_r100_full_coverage.toml",
        template_overrides=V3_T128_SURFACE_OVERRIDES,
        comparisons=(
            "Root-only R100 versus mass-matched +5% / +10% / +15% / +20% local-label lanes.",
            "Leaf geometries 128 / 64 / 32 / 16 / 8 on 128-token docs.",
            "Train-doc ladder 1024 / 4096 / 10240 with seeds 0 / 1.",
            "Uses the one-leaf FNO-parity canary for the 128-token exact-collapse anchor.",
        ),
        groups=("coverage", "all_curated"),
        notes=("This is optional and not part of the default initial_grid group.",),
    ),
)


JOB_BY_KEY: dict[str, GridJobSpec] = {job.key: job for job in JOB_SPECS}
CHECK_BASICS_KEYS: tuple[str, ...] = (
    "superset_gamma_t128",
    "mass_matched_gamma_t128",
    "preset_ablation_canary",
    "preset_ablation_full_laws",
)
SCIENTIFIC_FOLLOWUP_KEYS: tuple[str, ...] = (
    "v3_main_grid",
    "full100_leaf_ladder_standard",
    "full100_leaf_ladder_half_c1",
    "superset_leaf32_c1half",
    "superset_leaf32_leafratehalf",
)
PROTOCOL_FOLLOWUP_KEYS: tuple[str, ...] = (
    "preset_ablation_mse_only",
    "preset_ablation_two_stage_no_laws",
    "multileaf_root_only",
    "multileaf_full_laws",
)
AFTER_BASICS_KEYS: tuple[str, ...] = (
    SCIENTIFIC_FOLLOWUP_KEYS + PROTOCOL_FOLLOWUP_KEYS
)
GROUPS: dict[str, tuple[str, ...]] = {
    "check_basics": CHECK_BASICS_KEYS,
    "local_law_quickcheck": (
        "preset_ablation_canary",
        "one_leaf_duplicate_local_full_laws",
        "quick_two_leaf_root_only",
        "quick_two_leaf_full100_local_full_laws",
        "quick_two_leaf_r100_superset_local10",
    ),
    "small_train_local_law": (
        "preset_ablation_canary",
        "one_leaf_duplicate_local_full_laws",
        "small_train_multileaf_root_only",
        "small_train_multileaf_full_laws",
        "small_train_r100_superset_local10",
    ),
    "redistribution_quickcheck": (
        "redistribution_quickcheck",
    ),
    "redistribution_small_train": (
        "redistribution_small_train",
    ),
    "redistribution": (
        "redistribution_quickcheck",
        "redistribution_small_train",
    ),
    "depth_redistribution_root_ladder": (
        "root_budget_ladder_small_train",
    ),
    "depth_redistribution_leaf_only": (
        "mass_preserving_leaf_only_small_train",
    ),
    "depth_redistribution_levels_equal": (
        "mass_preserving_depth_equal_small_train",
    ),
    "depth_redistribution": (
        "root_budget_ladder_small_train",
        "mass_preserving_leaf_only_small_train",
        "mass_preserving_depth_equal_small_train",
    ),
    "depth_redistribution_large_root_ladder": (
        "root_budget_ladder_large_train",
    ),
    "depth_redistribution_large_leaf_only": (
        "mass_preserving_leaf_only_large_train",
    ),
    "depth_redistribution_large_levels_equal": (
        "mass_preserving_depth_equal_large_train",
    ),
    "depth_redistribution_large_train_stable": (
        "root_budget_ladder_large_train",
        "mass_preserving_leaf_only_large_train",
    ),
    "depth_redistribution_large_train": (
        "root_budget_ladder_large_train",
        "mass_preserving_leaf_only_large_train",
        "mass_preserving_depth_equal_large_train",
    ),
    "depth_redistribution_large_train_tuning": (
        "root_budget_ladder_large_train_longschedule",
        "mass_preserving_leaf_only_large_train_longschedule",
    ),
    "depth_redistribution_xlarge": (
        "root_budget_ladder_xlarge_train",
        "mass_preserving_leaf_only_xlarge_train",
        "mass_preserving_depth_equal_xlarge_train",
    ),
    "publication_oneleaf": (
        "oneleaf_root_budget_publication_fullval",
    ),
    "publication_oneleaf_xlarge": (
        "oneleaf_root_budget_publication_xlarge",
    ),
    "publication_plot_fillers": (
        "oneleaf_root_budget_longschedule_fill_fullval",
        "oneleaf_local_law_root_sweep_fullval",
        "oneleaf_leaf_mass_root_sweep_fullval",
        "oneleaf_root_budget_longschedule_fill_xlarge",
        "oneleaf_local_law_root_sweep_xlarge",
        "oneleaf_leaf_mass_root_sweep_xlarge",
    ),
    "publication_local_law": (
        "local_law_publication_fullval",
        "r100_superset_local10_publication_fullval",
    ),
    "publication_local_law_xlarge": (
        "local_law_publication_xlarge",
        "r100_superset_local10_publication_xlarge",
    ),
    "structural_oneleaf_rescue": (
        "structural_oneleaf_matched_root_v2_rescue",
        "structural_oneleaf_matched_root_v3_rescue",
        "structural_oneleaf_recoverable_recipe_v3_rescue",
        "structural_oneleaf_canary_anchor_rescue",
    ),
    "publication_multileaf": (
        "root_budget_publication_multileaf_fullval",
        "leaf_only_publication_focus_fullval",
        "depth_equal_publication_focus_fullval",
        "local_law_publication_fullval",
        "r100_superset_local10_publication_fullval",
    ),
    "publication_multileaf_xlarge": (
        "root_budget_publication_multileaf_xlarge",
        "leaf_only_publication_focus_xlarge",
        "depth_equal_publication_focus_xlarge",
        "local_law_publication_xlarge",
        "r100_superset_local10_publication_xlarge",
    ),
    "publication_fullval": (
        "oneleaf_root_budget_publication_fullval",
        "root_budget_publication_multileaf_fullval",
        "leaf_only_publication_focus_fullval",
        "depth_equal_publication_focus_fullval",
        "local_law_publication_fullval",
        "r100_superset_local10_publication_fullval",
    ),
    "publication_xlarge": (
        "oneleaf_root_budget_publication_xlarge",
        "root_budget_publication_multileaf_xlarge",
        "leaf_only_publication_focus_xlarge",
        "depth_equal_publication_focus_xlarge",
        "local_law_publication_xlarge",
        "r100_superset_local10_publication_xlarge",
    ),
    "overnight_xlarge": (
        "root_budget_ladder_xlarge_train",
        "mass_preserving_leaf_only_xlarge_train",
        "mass_preserving_depth_equal_xlarge_train",
        "oneleaf_root_budget_publication_xlarge",
        "root_budget_publication_multileaf_xlarge",
        "leaf_only_publication_focus_xlarge",
        "depth_equal_publication_focus_xlarge",
        "local_law_publication_xlarge",
        "r100_superset_local10_publication_xlarge",
    ),
    "scientific_followups": SCIENTIFIC_FOLLOWUP_KEYS,
    "protocol_followups": PROTOCOL_FOLLOWUP_KEYS,
    "after_basics": AFTER_BASICS_KEYS,
    "initial_grid": CHECK_BASICS_KEYS + AFTER_BASICS_KEYS,
    "core": tuple(job.key for job in JOB_SPECS if "core" in job.groups),
    "leaf_law_followups": tuple(job.key for job in JOB_SPECS if "leaf_law_followups" in job.groups),
    "preset_ablation": tuple(job.key for job in JOB_SPECS if "preset_ablation" in job.groups),
    "multileaf_protocol": tuple(job.key for job in JOB_SPECS if "multileaf_protocol" in job.groups),
    "coverage": tuple(job.key for job in JOB_SPECS if "coverage" in job.groups),
    "all_curated": tuple(job.key for job in JOB_SPECS),
}

AXIS_COVERAGE_SPECS: tuple[AxisCoverageSpec, ...] = (
    AxisCoverageSpec(
        key="superset_semantics",
        label="Package Semantics: Superset",
        question="Does adding local labels on top of full root supervision help beyond root-only R100?",
        side_a="`root100` baseline",
        side_b="`root100_extra_local10/15/20` true-superset lanes",
        required_job_keys=("v3_main_grid", "superset_gamma_t128"),
        alternative_job_sets=(("superset_gamma_t128",),),
    ),
    AxisCoverageSpec(
        key="mass_matched_semantics",
        label="Package Semantics: Mass-Matched",
        question="At matched total supervision mass, do local labels help beyond root-only R100?",
        side_a="`root100` baseline",
        side_b="`root100_mass_local10/15/20` mass-matched lanes",
        required_job_keys=("mass_matched_gamma_t128",),
    ),
    AxisCoverageSpec(
        key="root_node_redistribution",
        label="Root/Node Redistribution",
        question="At fixed total supervision mass 100%, how does performance move as we shift mass from the root to covered-token nodes?",
        side_a="Root-heavy splits such as `root90_nodes10` and `root80_nodes20`",
        side_b="Node-heavy splits such as `root20_nodes80` and `root0_nodes100`",
        required_job_keys=("redistribution_small_train",),
        alternative_job_sets=(("redistribution_quickcheck",),),
        selection_expectation="optional",
        notes=(
            "This is distinct from the older rate-based mass-matched packages: these jobs use geometry-aware node-mass targets so `root50_nodes50` means the same 50/50 split at every leaf size.",
        ),
    ),
    AxisCoverageSpec(
        key="root_budget_ladder",
        label="Root Budget Ladder",
        question="Do we have the root-only reviewed-doc ladder that local supervision is supposed to replace?",
        side_a="Dense root review: `root100`",
        side_b="Sparse root review: `root90/80/.../10`",
        required_job_keys=("root_budget_ladder_small_train",),
        alternative_job_sets=(("root_budget_ladder_large_train",),),
        selection_expectation="optional",
        notes=(
            "This is the comparator ladder for asking how much explicit root review is actually necessary before local labels are added.",
        ),
    ),
    AxisCoverageSpec(
        key="mass_preserving_leaf_only",
        label="Mass-Preserving Leaf-Only",
        question="At fixed total supervision mass 100%, how far can leaf-only local supervision replace missing root review?",
        side_a="Root-only baselines such as `root50`",
        side_b="Leaf-only redistribution such as `root50_leaf50`",
        required_job_keys=("mass_preserving_leaf_only_small_train",),
        alternative_job_sets=(("mass_preserving_leaf_only_large_train",),),
        selection_expectation="optional",
        notes=(
            "This is the direct `50/50/0`-style family across the whole root-share decile ladder.",
        ),
    ),
    AxisCoverageSpec(
        key="mass_preserving_depth_equal",
        label="Mass-Preserving Depth-Equal",
        question="At fixed total supervision mass 100%, does spreading local mass over deeper summary levels beat leaf-only redistribution?",
        side_a="Leaf-only redistribution such as `root50_leaf50`",
        side_b="Depth-equal redistribution such as `root50_levels_equal50`",
        required_job_keys=("mass_preserving_leaf_only_small_train", "mass_preserving_depth_equal_small_train"),
        alternative_job_sets=(
            ("mass_preserving_depth_equal_small_train",),
            ("mass_preserving_leaf_only_large_train", "mass_preserving_depth_equal_large_train"),
            ("mass_preserving_depth_equal_large_train",),
        ),
        selection_expectation="optional",
        notes=(
            "This realizes geometry-aware profiles like `50/25/25` at leaf32 and `25/25/25/25` at leaf16.",
        ),
    ),
    AxisCoverageSpec(
        key="geometry_endpoints",
        label="Geometry Endpoints",
        question="Do the tree results span the full one-leaf to many-leaf geometry ladder with an explicit FNO anchor?",
        side_a="`leaf128` exact-collapse endpoint with `fno_parity_canary` on `full100`",
        side_b="`leaf64/32/16/8` multi-leaf geometries on 128-token docs",
        required_job_keys=(
            "v3_main_grid",
            "superset_gamma_t128",
            "mass_matched_gamma_t128",
            "full100_leaf_ladder_standard",
            "full100_leaf_ladder_half_c1",
        ),
        alternative_job_sets=(
            ("superset_gamma_t128", "preset_ablation_canary"),
            ("mass_matched_gamma_t128", "preset_ablation_canary"),
            ("superset_gamma_t128", "mass_matched_gamma_t128", "preset_ablation_canary"),
            ("full100_leaf_ladder_standard",),
            ("full100_leaf_ladder_half_c1",),
            ("preset_ablation_canary", "small_train_multileaf_root_only"),
            ("preset_ablation_canary", "small_train_multileaf_full_laws"),
            ("root_budget_ladder_small_train",),
            ("root_budget_ladder_small_train", "mass_preserving_leaf_only_small_train"),
            ("root_budget_ladder_small_train", "mass_preserving_depth_equal_small_train"),
            ("root_budget_ladder_large_train",),
            ("root_budget_ladder_large_train", "mass_preserving_leaf_only_large_train"),
            ("root_budget_ladder_large_train", "mass_preserving_depth_equal_large_train"),
        ),
        notes=(
            "Local-label comparison grids now start at `leaf64`; the one-leaf endpoint is carried by dedicated `full100` exact-collapse surfaces.",
        ),
    ),
    AxisCoverageSpec(
        key="gamma_sweep",
        label="Depth Discount Gamma",
        question="How sensitive are recoveries to discounting deeper local laws?",
        side_a="No extra discounting: `gamma=1.0`",
        side_b="More aggressive discounting: `gamma=0.9/0.75`",
        required_job_keys=("superset_gamma_t128", "mass_matched_gamma_t128"),
        alternative_job_sets=(
            ("superset_gamma_t128",),
            ("mass_matched_gamma_t128",),
        ),
    ),
    AxisCoverageSpec(
        key="law_weighting",
        label="Local-Law Weighting",
        question="Are gains robust to weakening the C1 leaf-law contribution?",
        side_a="Standard full-laws recipe",
        side_b="Half-C1 variants",
        required_job_keys=(
            "full100_leaf_ladder_standard",
            "full100_leaf_ladder_half_c1",
            "superset_leaf32_c1half",
        ),
    ),
    AxisCoverageSpec(
        key="leaf_rate_allocation",
        label="Leaf/Internal Allocation",
        question="Is the superset gain driven by leaf-label rate or by having extra local labels at all?",
        side_a="Standard `+10% leaf / +10% internal` superset lane",
        side_b="Reduced-leaf-rate `+5% leaf / +10% internal` superset lane",
        required_job_keys=("superset_leaf32_leafratehalf",),
    ),
    AxisCoverageSpec(
        key="official_fno_base",
        label="Official FNO Base",
        question="Do we have a direct exact-collapse `full100` tree-vs-`official_fno` comparison in the selected batch?",
        side_a="Tree `full100` at `leaf128` with the exact one-leaf parity configuration",
        side_b="Canonical `official_fno` baseline emitted from the same `full100` package",
        required_job_keys=("preset_ablation_canary",),
        alternative_job_sets=(
            ("preset_ablation_full_laws",),
            ("v3_main_grid",),
            ("full100_leaf_ladder_standard",),
            ("redistribution_quickcheck",),
            ("redistribution_small_train",),
            ("root_budget_ladder_small_train",),
            ("mass_preserving_leaf_only_small_train",),
            ("mass_preserving_depth_equal_small_train",),
            ("root_budget_ladder_large_train",),
            ("mass_preserving_leaf_only_large_train",),
            ("mass_preserving_depth_equal_large_train",),
        ),
        notes=(
            "`full100` automatically emits the canonical `official_fno` baseline because its package spec has `run_fno=True`.",
        ),
    ),
    AxisCoverageSpec(
        key="one_leaf_canary_vs_standard",
        label="One-Leaf Canary vs Standard Protocol",
        question="Do we have the minimal side-by-side comparison between the exact FNO-parity canary and the standard 1-leaf training recipe?",
        side_a="`fno_parity_canary`",
        side_b="`full_laws` at the same `full100`, `leaf128` surface",
        required_job_keys=("preset_ablation_canary", "preset_ablation_full_laws"),
        notes=(
            "This is a one-leaf protocol comparison only. At `leaf128`, local-law terms are structurally inactive because there are no non-root local targets.",
        ),
    ),
    AxisCoverageSpec(
        key="one_leaf_protocol",
        label="One-Leaf Protocol Ablation",
        question="Which training-choice change breaks or preserves 1-leaf FNO parity, before we move to law-valid multi-leaf surfaces?",
        side_a="`fno_parity_canary`",
        side_b="`full_laws`, with `mse_only` and `two_stage_no_laws` bridge steps",
        required_job_keys=(
            "preset_ablation_canary",
            "preset_ablation_mse_only",
            "preset_ablation_two_stage_no_laws",
            "preset_ablation_full_laws",
        ),
        notes=(
            "This axis is about optimizer/protocol choices on the one-leaf surface. It should not be interpreted as a local-law-validity check.",
        ),
    ),
    AxisCoverageSpec(
        key="local_law_validity",
        label="Local-Law Validity",
        question="Do we have a multi-leaf root-only versus full-laws comparison where leaf/internal supervision is actually present?",
        side_a="Root-only multi-leaf `full100` on `leaf64/32/16/8`",
        side_b="A real local-label package on the same geometries, not `full100` with zero local targets",
        required_job_keys=(
            "small_train_multileaf_root_only",
            "small_train_multileaf_full_laws",
        ),
        alternative_job_sets=(
            ("quick_two_leaf_root_only", "quick_two_leaf_full100_local_full_laws"),
            ("multileaf_root_only", "multileaf_full_laws"),
        ),
        notes=(
            "This is the correct surface for checking whether the local laws themselves help. The one-leaf `leaf128` jobs are excluded because the law terms are structurally inactive there.",
        ),
    ),
    AxisCoverageSpec(
        key="one_leaf_duplicate_local_no_harm",
        label="One-Leaf Duplicate Local No-Harm",
        question="If we add duplicate local supervision to the exact 1-leaf surface, does it preserve parity instead of making the model worse?",
        side_a="`preset_ablation_canary` on `full100`",
        side_b="`one_leaf_duplicate_local_full_laws` on `root100_extra_leaffull100_internalcount100`",
        required_job_keys=(
            "preset_ablation_canary",
            "one_leaf_duplicate_local_full_laws",
        ),
        notes=(
            "This is a redundancy stress test, not a full local-law-validity proof. At `leaf128`, there is still no nontrivial merge tree.",
        ),
    ),
    AxisCoverageSpec(
        key="multileaf_protocol",
        label="Multi-Leaf Protocol Ablation",
        question="Across the full geometry ladder, how much do protocol choices matter relative to the merge architecture itself?",
        side_a="`multileaf_root_only`",
        side_b="`multileaf_full_laws`",
        required_job_keys=("multileaf_root_only", "multileaf_full_laws"),
    ),
    AxisCoverageSpec(
        key="mass_full_coverage",
        label="Mass-Matched Full Coverage",
        question="Do we want the denser mass-matched coverage run with the extra +5% lane and the full train-doc ladder in one bundle?",
        side_a="Default mass-matched gamma/core coverage",
        side_b="Optional full-coverage mass-matched sweep including `+5%`",
        required_job_keys=("mass_matched_full_coverage",),
        selection_expectation="optional",
    ),
)


def selected_job_specs(
    *,
    group_names: Sequence[str],
    explicit_job_keys: Sequence[str],
) -> list[GridJobSpec]:
    ordered_keys: list[str] = []
    seen: set[str] = set()

    def _append(key: str) -> None:
        if key in seen:
            return
        if key not in JOB_BY_KEY:
            raise ValueError(
                f"unknown job key {key!r}; valid keys are {sorted(JOB_BY_KEY)}"
            )
        seen.add(key)
        ordered_keys.append(key)

    for group_name in group_names:
        if group_name not in GROUPS:
            raise ValueError(
                f"unknown group {group_name!r}; valid groups are {sorted(GROUPS)}"
            )
        for key in GROUPS[group_name]:
            _append(key)
    for key in explicit_job_keys:
        _append(key)
    return [JOB_BY_KEY[key] for key in ordered_keys]


def _generated_config_path(output_root_base: Path, spec: GridJobSpec) -> Path:
    return output_root_base / "_generated_configs" / f"{spec.key}.toml"


def _build_axis_coverage(jobs: Sequence[GridJobSpec]) -> list[dict[str, Any]]:
    selected_keys = {job.key for job in jobs}
    coverage: list[dict[str, Any]] = []
    for spec in AXIS_COVERAGE_SPECS:
        candidate_sets = (tuple(spec.required_job_keys),) + tuple(spec.alternative_job_sets)
        matched_set: tuple[str, ...] | None = None
        selected_required: tuple[str, ...] = ()
        partial_hit = False
        for job_set in candidate_sets:
            selected_in_set = tuple(key for key in job_set if key in selected_keys)
            if len(selected_in_set) == len(job_set):
                matched_set = tuple(job_set)
                selected_required = selected_in_set
                break
            if selected_in_set:
                partial_hit = True
                if not selected_required:
                    selected_required = selected_in_set
        required_keys = matched_set or tuple(spec.required_job_keys)
        if spec.selection_expectation == "optional":
            status = (
                "selected_optional"
                if matched_set is not None
                else "available_optional"
            )
        else:
            if matched_set is not None:
                status = "ready"
            elif partial_hit or selected_required:
                status = "partial"
            else:
                status = "missing"
        coverage.append(
            {
                "key": spec.key,
                "label": spec.label,
                "question": spec.question,
                "side_a": spec.side_a,
                "side_b": spec.side_b,
                "required_jobs": list(required_keys),
                "selected_jobs": list(selected_required),
                "selection_expectation": spec.selection_expectation,
                "status": status,
                "notes": list(spec.notes),
            }
        )
    return coverage


def build_launch_plan(
    *,
    group_names: Sequence[str],
    explicit_job_keys: Sequence[str],
    output_root_base: Path,
    python_bin: Path,
    launch_backend: str,
    replace_existing: bool,
    env_assignments: Sequence[str],
    inspect_existing_launchers: bool = False,
) -> dict[str, Any]:
    jobs = selected_job_specs(
        group_names=group_names,
        explicit_job_keys=explicit_job_keys,
    )
    axis_coverage = _build_axis_coverage(jobs)
    plan_jobs: list[dict[str, Any]] = []
    for spec in jobs:
        output_root = output_root_base / spec.key
        job_root = output_root_base / "_launchers" / spec.key
        effective_config_path = (
            _generated_config_path(output_root_base, spec)
            if spec.is_template
            else spec.config_path
        )
        pipeline_cmd = _pipeline_command(
            effective_config_path,
            output_root,
            python_bin,
        )
        launcher_cmd = _long_job_command(
            name=f"markov_v3_initial_grid__{spec.key}",
            description=spec.label,
            job_root=job_root,
            command=pipeline_cmd,
            python_bin=python_bin,
            launch_backend=launch_backend,
            replace_existing=replace_existing,
            env_assignments=env_assignments,
        )
        existing_state = (
            inspect_existing_job_state(
                job_root=job_root,
                output_root=output_root,
                python_bin=python_bin,
            )
            if inspect_existing_launchers
            else None
        )
        plan_jobs.append(
            {
                "key": spec.key,
                "label": spec.label,
                "category": spec.category,
                "config": _rel(spec.config_path),
                "effective_config": _rel(effective_config_path),
                "uses_generated_config": bool(spec.is_template),
                "groups": list(spec.groups),
                "comparisons": list(spec.comparisons),
                "notes": list(spec.notes),
                "output_root": _rel(output_root),
                "launcher_job_root": _rel(job_root),
                "pipeline_command": pipeline_cmd,
                "pipeline_command_shell": shlex.join(pipeline_cmd),
                "launcher_command": launcher_cmd,
                "launcher_command_shell": shlex.join(launcher_cmd),
                "existing_state": existing_state,
            }
        )
    return {
        "selection_groups": list(group_names),
        "explicit_jobs": list(explicit_job_keys),
        "output_root_base": _rel(output_root_base),
        "python_bin": _rel(python_bin) if python_bin.is_absolute() else str(python_bin),
        "job_count": len(plan_jobs),
        "axis_coverage": axis_coverage,
        "jobs": plan_jobs,
    }


def materialize_generated_configs(
    plan: Mapping[str, Any],
) -> list[Path]:
    written: list[Path] = []
    for job in list(plan.get("jobs") or []):
        key = str(job.get("key", "") or "")
        spec = JOB_BY_KEY[key]
        if not spec.is_template:
            continue
        config_path = (REPO_ROOT / str(job["effective_config"])).resolve()
        payload = materialize_job_config_payload(spec)
        write_structured_config(config_path, payload)
        written.append(config_path)
    return written


def launch_plan_jobs(
    plan: Mapping[str, Any],
    *,
    python_bin: Path,
    skip_running: bool,
    skip_completed: bool,
    fail_fast: bool,
) -> dict[str, Any]:
    base_root = _resolve_repo_relative_path(str(plan.get("output_root_base", "")))
    materialized_configs = [
        str(path) for path in materialize_generated_configs(plan)
    ]
    launched: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    failed: list[dict[str, Any]] = []
    for job in list(plan.get("jobs") or []):
        job_key = str(job.get("key", "") or "")
        job_root = _resolve_repo_relative_path(str(job.get("launcher_job_root", "")))
        output_root = _resolve_repo_relative_path(str(job.get("output_root", "")))
        existing_state = dict(job.get("existing_state") or {})
        if not existing_state:
            existing_state = inspect_existing_job_state(
                job_root=job_root,
                output_root=output_root,
                python_bin=python_bin,
            )
        state = str(existing_state.get("state", "") or "not_launched")
        if bool(skip_running) and state == "running":
            skipped.append(
                {
                    "key": job_key,
                    "reason": "running",
                    "job_root": str(job_root),
                    "output_root": str(output_root),
                    "existing_state": existing_state,
                }
            )
            continue
        if bool(skip_completed) and state == "completed":
            skipped.append(
                {
                    "key": job_key,
                    "reason": "completed",
                    "job_root": str(job_root),
                    "output_root": str(output_root),
                    "existing_state": existing_state,
                }
            )
            continue
        result = subprocess.run(
            list(job["launcher_command"]),
            capture_output=True,
            text=True,
            check=False,
            cwd=str(REPO_ROOT),
        )
        if int(result.returncode) != 0:
            failed.append(
                {
                    "key": job_key,
                    "returncode": int(result.returncode),
                    "job_root": str(job_root),
                    "output_root": str(output_root),
                    "stdout": str(result.stdout or "").strip(),
                    "stderr": str(result.stderr or "").strip(),
                }
            )
            if bool(fail_fast):
                break
            continue
        launch_payload: dict[str, Any]
        try:
            parsed_stdout = json.loads(result.stdout or "{}")
        except Exception:
            parsed_stdout = {}
        launch_payload = dict(parsed_stdout) if isinstance(parsed_stdout, dict) else {}
        launched.append(
            {
                "key": job_key,
                "job_root": str(job_root),
                "output_root": str(output_root),
                "manifest_path": str(launch_payload.get("manifest_path", "") or ""),
                "pid": int(launch_payload.get("pid", 0) or 0),
                "launch_backend": str(launch_payload.get("launch_backend", "") or ""),
            }
        )
    summary = {
        "base_root": str(base_root),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "selection_groups": list(plan.get("selection_groups") or []),
        "explicit_jobs": list(plan.get("explicit_jobs") or []),
        "skip_running": bool(skip_running),
        "skip_completed": bool(skip_completed),
        "fail_fast": bool(fail_fast),
        "materialized_configs": materialized_configs,
        "launched_jobs": launched,
        "skipped_jobs": skipped,
        "failed_jobs": failed,
        "launched_count": len(launched),
        "skipped_count": len(skipped),
        "failed_count": len(failed),
    }
    summary_path = base_root / "launch_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary


def _render_text_plan(plan: Mapping[str, Any]) -> str:
    lines = [
        f"Markov v3 initial-grid launch plan: {int(plan.get('job_count', 0))} jobs",
        f"Groups: {', '.join(plan.get('selection_groups') or []) or '(none)'}",
        f"Output root base: {plan.get('output_root_base', '')}",
        "",
    ]
    axis_coverage = list(plan.get("axis_coverage") or [])
    if axis_coverage:
        lines.append("Axis coverage:")
        for axis in axis_coverage:
            label = str(axis.get("label") or "")
            status = str(axis.get("status") or "")
            lines.append(f"- {label} [{status}]")
            lines.append(f"  question: {axis.get('question', '')}")
            lines.append(f"  side A: {axis.get('side_a', '')}")
            lines.append(f"  side B: {axis.get('side_b', '')}")
            selected_jobs = list(axis.get("selected_jobs") or [])
            required_jobs = list(axis.get("required_jobs") or [])
            if selected_jobs:
                lines.append(f"  jobs: {', '.join(selected_jobs)}")
            elif required_jobs:
                lines.append(f"  jobs: {', '.join(required_jobs)}")
            for note in list(axis.get("notes") or []):
                lines.append(f"  note: {note}")
        lines.append("")
    for index, job in enumerate(list(plan.get("jobs") or []), start=1):
        lines.append(f"{index}. {job['key']} [{job['category']}]")
        lines.append(f"   {job['label']}")
        lines.append(f"   config: {job['config']}")
        if bool(job.get("uses_generated_config")):
            lines.append(f"   generated config: {job['effective_config']}")
        existing_state = dict(job.get("existing_state") or {})
        if existing_state:
            lines.append(f"   current state: {existing_state.get('state', 'unknown')}")
            scheduler_status_path = str(
                existing_state.get("scheduler_status_path", "") or ""
            ).strip()
            if scheduler_status_path:
                lines.append(f"   status file: {_rel(Path(scheduler_status_path))}")
            manifest_path = str(existing_state.get("manifest_path", "") or "").strip()
            if manifest_path:
                lines.append(f"   launcher manifest: {_rel(Path(manifest_path))}")
        lines.append("   comparisons:")
        for item in list(job.get("comparisons") or []):
            lines.append(f"     - {item}")
        for note in list(job.get("notes") or []):
            lines.append(f"   note: {note}")
        lines.append(f"   output: {job['output_root']}")
        lines.append(f"   launch cmd: {job['launcher_command_shell']}")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Prepare or launch the curated Markov v3 initial-grid batch. "
            "Defaults to plan-only output; use --launch to actually start detached jobs."
        )
    )
    parser.add_argument(
        "--group",
        action="append",
        default=[],
        help=f"Select a predefined job group. Valid groups: {', '.join(sorted(GROUPS))}",
    )
    parser.add_argument(
        "--job",
        action="append",
        default=[],
        help="Append an individual job key on top of the selected groups.",
    )
    parser.add_argument(
        "--list-groups",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="List available groups and exit.",
    )
    parser.add_argument(
        "--list-jobs",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="List available job keys and exit.",
    )
    parser.add_argument(
        "--base-root",
        "--output-root-base",
        dest="output_root_base",
        type=Path,
        default=REPO_ROOT / "outputs" / f"markov_v3_initial_grid_{_utc_stamp()}",
        help=(
            "Persistent base folder for this launch batch. Reusing the same path lets "
            "you resume stopped jobs and keep all per-job outputs/manifests together."
        ),
    )
    parser.add_argument("--python-bin", type=Path, default=DEFAULT_PYTHON_BIN)
    parser.add_argument(
        "--launch",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Actually launch the detached jobs via scripts/long_job.py.",
    )
    parser.add_argument(
        "--launch-backend",
        choices=("auto", "double_fork", "systemd"),
        default="auto",
    )
    parser.add_argument(
        "--replace-existing",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Pass --replace-existing through to long_job launch.",
    )
    parser.add_argument(
        "--skip-running",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="When reusing a base folder, skip jobs whose long_job launcher is still running.",
    )
    parser.add_argument(
        "--skip-completed",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="When reusing a base folder, skip jobs whose scheduler_status.json already reports completed.",
    )
    parser.add_argument(
        "--fail-fast",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Stop launching further jobs after the first launch failure. Default is to continue.",
    )
    parser.add_argument(
        "--env",
        action="append",
        default=[],
        help="Optional KEY=VALUE environment assignment forwarded to long_job.",
    )
    parser.add_argument(
        "--write-plan",
        type=Path,
        default=None,
        help="Optional JSON plan path. A sibling .md file is written alongside it.",
    )
    parser.add_argument(
        "--json",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Print the plan as JSON instead of text.",
    )
    return parser.parse_args(list(argv) if argv is not None else None)


def _print_groups() -> None:
    for name in sorted(GROUPS):
        print(f"{name}: {', '.join(GROUPS[name])}")


def _print_jobs() -> None:
    for job in JOB_SPECS:
        print(f"{job.key}: {job.label}")


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    if args.list_groups:
        _print_groups()
        return 0
    if args.list_jobs:
        _print_jobs()
        return 0

    group_names = list(args.group or [])
    if not group_names and not list(args.job or []):
        group_names = ["check_basics"]

    python_bin = args.python_bin.expanduser()
    if not python_bin.is_absolute():
        python_bin = (REPO_ROOT / python_bin).resolve()

    plan = build_launch_plan(
        group_names=group_names,
        explicit_job_keys=list(args.job or []),
        output_root_base=args.output_root_base.resolve(),
        python_bin=python_bin,
        launch_backend=str(args.launch_backend),
        replace_existing=bool(args.replace_existing),
        env_assignments=list(args.env or []),
        inspect_existing_launchers=True,
    )

    rendered_text = _render_text_plan(plan)
    write_plan_path = (
        args.write_plan.resolve()
        if args.write_plan is not None
        else (
            args.output_root_base.resolve() / "launch_plan.json"
            if bool(args.launch)
            else None
        )
    )
    if write_plan_path is not None:
        write_path = write_plan_path
        write_path.parent.mkdir(parents=True, exist_ok=True)
        write_path.write_text(
            json.dumps(plan, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        write_path.with_suffix(".md").write_text(rendered_text, encoding="utf-8")

    if bool(args.json):
        print(json.dumps(plan, indent=2, sort_keys=True))
    else:
        print(rendered_text, end="")

    if not bool(args.launch):
        return 0

    launch_summary = launch_plan_jobs(
        plan,
        python_bin=python_bin,
        skip_running=bool(args.skip_running),
        skip_completed=bool(args.skip_completed),
        fail_fast=bool(args.fail_fast),
    )
    if bool(args.json):
        print(json.dumps(launch_summary, indent=2, sort_keys=True))
    else:
        print(
            (
                "\nLaunch summary:\n"
                f"- launched: {int(launch_summary.get('launched_count', 0))}\n"
                f"- skipped: {int(launch_summary.get('skipped_count', 0))}\n"
                f"- failed: {int(launch_summary.get('failed_count', 0))}\n"
                f"- summary: {_rel(Path(str(launch_summary['base_root'])) / 'launch_summary.json')}\n"
            ),
            end="",
        )
    return 1 if int(launch_summary.get("failed_count", 0) or 0) > 0 else 0


if __name__ == "__main__":
    raise SystemExit(main())
