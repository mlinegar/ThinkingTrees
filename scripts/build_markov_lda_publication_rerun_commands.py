#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Dict, Iterable, List, Mapping, Optional

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.ctreepo.sim.local_law_backfill import load_or_backfill_local_law_payload
from src.ctreepo.sim.objective_backfill import safe_objective_backfill


def _iter_json_files(root: Path) -> Iterable[Path]:
    return sorted(path for path in root.rglob("*.json") if path.is_file())


def _load_json(path: Path) -> Optional[Dict[str, Any]]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    return payload if isinstance(payload, dict) else None


def _is_run_payload(payload: Mapping[str, Any]) -> bool:
    if "artifact_id" in payload and "payload" in payload:
        return False
    if isinstance(payload.get("local_law_learnability"), Mapping) and not any(
        key in payload for key in ("methods", "metrics", "local_law", "stage3")
    ):
        return False
    return any(
        key in payload
        for key in (
            "family",
            "config",
            "methods",
            "metrics",
            "local_law",
            "stage3",
            "objective",
        )
    )


def _first_run_payload(root: Path) -> Optional[Dict[str, Any]]:
    for path in _iter_json_files(root):
        payload = _load_json(path)
        if payload is None:
            continue
        if _is_run_payload(payload):
            return payload
    return None


def _root_needs_rerun(root: Path) -> bool:
    for path in _iter_json_files(root):
        payload = _load_json(path)
        if payload is None:
            continue
        if not _is_run_payload(payload):
            continue
        if isinstance(payload.get("objective"), Mapping) and dict(payload.get("objective", {}) or {}):
            objective_ok = True
        else:
            objective_ok = safe_objective_backfill(payload) is not None
        if isinstance(payload.get("local_law"), Mapping) and dict(payload.get("local_law", {}) or {}):
            local_ok = load_or_backfill_local_law_payload(payload, source_path=str(path)) is not None
        else:
            local_ok = True
        if not objective_ok or not local_ok:
            return True
    return False


def _lda_stage3_command(root: Path, payload: Mapping[str, Any]) -> str:
    cfg = dict(payload.get("config", {}) or {})
    return (
        "./scripts/run_tree_relevant_lda_stage3_overnight.sh "
        f"--output-root outputs/{root.name}_rerun_$(date +%Y%m%d_%H%M%S) "
        f"--train-docs {int(cfg.get('train_docs', 512))} "
        f"--test-docs {int(cfg.get('test_docs', 512))} "
        f"--latent-leaf-tokens {int(cfg.get('latent_leaf_tokens', 64))}"
    )


def _lda_followup_command(root: Path, payload: Mapping[str, Any]) -> str:
    cfg = dict(payload.get("config", {}) or {})
    return (
        "./scripts/run_tree_relevant_lda_followup_overnight.sh "
        f"--output-root outputs/{root.name}_rerun_$(date +%Y%m%d_%H%M%S) "
        f"--train-docs {int(cfg.get('train_docs', 512))} "
        f"--test-docs {int(cfg.get('test_docs', 512))}"
    )


def _markov_longrun_command(root: Path) -> str:
    return (
        f"OUT_ROOT=outputs/{root.name}_rerun_$(date +%Y%m%d_%H%M%S) "
        "./scripts/run_markov_local_law_learnability_longrun.sh"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build publication rerun commands for Markov/LDA roots that remain unsafe to backfill.")
    parser.add_argument("roots", nargs="+", type=Path)
    parser.add_argument("--output-script", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    commands: List[str] = []
    manifest_rows: List[Dict[str, Any]] = []
    for root in args.roots:
        root = Path(root)
        exists = root.exists()
        payload = _first_run_payload(root) if exists else None
        needs_rerun = bool(exists and _root_needs_rerun(root))
        command = ""
        kind = ""
        if needs_rerun and payload is not None:
            name = root.name
            if "tree_relevant_lda_stage3" in name:
                command = _lda_stage3_command(root, payload)
                kind = "lda_stage3"
            elif "tree_relevant_lda_followup" in name:
                command = _lda_followup_command(root, payload)
                kind = "lda_followup"
            elif name.startswith("markov_"):
                command = _markov_longrun_command(root)
                kind = "markov"
        if command:
            commands.append(command)
        manifest_rows.append(
            {
                "root": str(root),
                "exists": bool(exists),
                "needs_rerun": bool(needs_rerun),
                "kind": str(kind),
                "command": str(command),
            }
        )

    script_lines = ["#!/usr/bin/env bash", "set -euo pipefail", ""]
    if commands:
        script_lines.extend(commands)
    else:
        script_lines.append("echo 'No Markov/LDA publication reruns are currently required.'")
    args.output_script.parent.mkdir(parents=True, exist_ok=True)
    args.output_script.write_text("\n".join(script_lines) + "\n", encoding="utf-8")
    args.output_script.chmod(0o755)
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(
        json.dumps({"roots": [str(x) for x in args.roots], "rows": manifest_rows}, indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"output_script": str(args.output_script), "commands": commands}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
