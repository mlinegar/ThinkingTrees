#!/usr/bin/env python3
"""Run only the R0 (zero root labels) tasks for the allocation grid.

Usage: CUDA_VISIBLE_DEVICES=3 python scripts/run_r0_fill.py
"""
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
VENV_PYTHON = str(REPO / "venv" / "bin" / "python")
PIPELINE = str(REPO / "scripts" / "run_markov_optimization_tradeoff_pipeline.py")
BASE_CONFIG = str(REPO / "outputs" / "markov_v5_simple_fixed10240_quick_20260414_utc" / "_generated_configs" / "combined_scheduler_allocation_policy_grid.toml")
OUTPUT = str(REPO / "outputs" / "markov_v5_r0_fill")

# Run the pipeline with supervision_recovery phase only
cmd = [
    VENV_PYTHON, PIPELINE,
    "--preset", "standard",
    "--phases", "supervision_recovery",
    "--device-mode", "cuda",
    "--train-docs", "10240",
    "--output-root", OUTPUT,
    "--selection-config", BASE_CONFIG,
    "--max-workers", "1",
]

print(f"Running: {' '.join(cmd)}")
result = subprocess.run(cmd, cwd=str(REPO))
sys.exit(result.returncode)
