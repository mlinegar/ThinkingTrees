#!/usr/bin/env python3
"""Config-driven launcher for the manifesto q-sentence f/g ladder.

ONE launcher + a YAML bundle replaces the ~40 hand-written shell drivers that
each wrapped ``scripts/run_manifesto_qsentence_dspy_ladder.py`` with a
copy-pasted argv. A comparison is now declared as data (a bundle listing one or
more "runs"), not encoded in a bespoke ``.sh``.

Each bundle run is one invocation of the canonical ladder runner. The launcher
translates the run's YAML fields into the runner's EXACT flag spelling, sets any
per-run environment (``TT_DSPY_DROP_RESPONSE_FORMAT``, ``CUDA_VISIBLE_DEVICES``,
...), and executes them (or, with ``--dry-run``, prints the exact invocations
without running — the primary verification path, no GPU/LLM needed).

Bundle schema (``version: 1``)::

    version: 1
    # Optional: fields merged into EVERY run (run-level fields override these).
    defaults:
      family: fno
      grid_dir: outputs/manifesto_qsentence_dspy_labeled_grid
      max_iterations: 2
      env:
        TT_DSPY_DROP_RESPONSE_FORMAT: "1"
    runs:
      - name: dgemma_full_leafgrid
        family: dspy
        leaf_qsentences: "2,4,8,16"
        supervision: default            # comma-list; FNO-only levels error on dspy
        grid_dir: outputs/manifesto_qsentence_dspy_labeled_grid
        output_dir: outputs/manifesto_qsentence_diffusiongemma_full_leafgrid
        target_dimensions: all          # dspy vector head
        # target_dimension: rile        # fno scalar head (per-run, run once per dim)
        env:
          TT_DSPY_DROP_RESPONSE_FORMAT: "1"
          CUDA_VISIBLE_DEVICES: "0"
        dspy:                           # nested = --dspy-<key>
          model: openai/RedHatAI/diffusiongemma-26B-A4B-it-NVFP4
          api_base: http://localhost:8004/v1,http://localhost:8005/v1
          optimizer: gepa
        fno: {}                         # nested = --fno-<key>
        embedding: {}                   # nested = --embedding-<key>
        extra_args: ["--verbose"]       # verbatim, appended last

Field mapping:

- Known scalar/grid fields (``family``, ``leaf_qsentences``, ``supervision``,
  ``grid_dir`` -> ``--fg-grid-dir``, ``output_dir``, ``max_iterations``,
  ``target_dimensions``, ``target_dimension`` -> ``--fno-target-dimension``,
  ``first_train_side``, ``initial_f_degree``, ``initial_g_degree``,
  ``stage_naming``, ``train_split``, ``eval_split``, ``max_eval_trees``,
  ``eval_sample_seed``, ``tokenizer_model``, plus boolean flags ``verbose``,
  ``fail_on_row_error``, ``preflight_only``, ``include_identity_targets``) each
  translate to their canonical runner flag.
- ``dspy:``, ``fno:``, ``embedding:`` are nested maps whose keys become
  ``--dspy-<key>`` / ``--fno-<key>`` / ``--embedding-<key>`` (underscores ->
  dashes). Boolean values become presence flags (``true`` adds the flag,
  ``false`` omits it).
- ``env:`` is a per-run environment overlay (strings). ``defaults.env`` merges
  under each run's ``env``.
- ``extra_args:`` is a verbatim list appended after the mapped flags, for the
  long tail of runner flags without a dedicated schema key.

Unknown top-level or run keys are a hard error (fail loudly). This is the whole
contract: no silent drops.
"""

from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
LADDER_RUNNER = PROJECT_ROOT / "scripts" / "run_manifesto_qsentence_dspy_ladder.py"

# Top-level bundle keys.
_ALLOWED_TOP_KEYS = {"version", "defaults", "runs"}

# Run keys that map to a single runner flag (value rendered as-is). The mapping
# is the ONLY source of truth for spelling; keep it aligned with the runner's
# argparse. ``grid_dir`` -> ``--fg-grid-dir`` and ``target_dimension`` ->
# ``--fno-target-dimension`` are the two renames worth calling out.
_SCALAR_FLAG_MAP: Dict[str, str] = {
    "family": "--family",
    "leaf_qsentences": "--leaf-qsentences",
    "supervision": "--supervision",
    "grid_dir": "--fg-grid-dir",
    "output_dir": "--output-dir",
    "max_iterations": "--max-iterations",
    "target_dimensions": "--target-dimensions",
    "target_dimension": "--fno-target-dimension",
    "first_train_side": "--first-train-side",
    "initial_f_degree": "--initial-f-degree",
    "initial_g_degree": "--initial-g-degree",
    "stage_naming": "--stage-naming",
    "train_split": "--train-split",
    "eval_split": "--eval-split",
    "max_eval_trees": "--max-eval-trees",
    "eval_sample_seed": "--eval-sample-seed",
    "target_min": "--target-min",
    "target_max": "--target-max",
    "tokenizer_model": "--tokenizer-model",
    "embedding_backend": "--embedding-backend",
}

# Run keys that map to a boolean presence flag (rendered only when truthy).
_BOOL_FLAG_MAP: Dict[str, str] = {
    "verbose": "--verbose",
    "fail_on_row_error": "--fail-on-row-error",
    "preflight_only": "--preflight-only",
    "include_identity_targets": "--include-identity-targets",
}

# Nested-map keys -> flag prefix. Each child key becomes ``<prefix>-<child>``.
_NESTED_PREFIX_MAP: Dict[str, str] = {
    "dspy": "--dspy",
    "fno": "--fno",
    "embedding": "--embedding",
}

# Remaining recognized run keys (not translated to flags directly).
_OTHER_RUN_KEYS = {"name", "env", "extra_args"}

_ALLOWED_RUN_KEYS = (
    set(_SCALAR_FLAG_MAP)
    | set(_BOOL_FLAG_MAP)
    | set(_NESTED_PREFIX_MAP)
    | _OTHER_RUN_KEYS
)


class BundleError(ValueError):
    """Raised for any malformed bundle (unknown keys, wrong types, ...)."""


def _render_scalar(value: Any) -> str:
    """Render a scalar YAML value as a single argv token."""
    if isinstance(value, bool):
        # Booleans belong on the bool-flag / nested-flag paths, not here.
        raise BundleError(f"boolean value {value!r} is not a scalar flag argument")
    return str(value)


def _nested_flags(prefix: str, mapping: Mapping[str, Any]) -> List[str]:
    """Translate ``{prefix: {child: value}}`` into ``--prefix-child value`` argv.

    Booleans become presence flags: ``true`` emits ``--prefix-child`` with no
    value; ``false`` emits nothing. Underscores in child keys become dashes.
    """
    if not isinstance(mapping, Mapping):
        raise BundleError(
            f"nested block {prefix!r} must be a mapping, got {type(mapping).__name__}"
        )
    argv: List[str] = []
    for child, value in mapping.items():
        flag = f"{prefix}-{str(child).replace('_', '-')}"
        if isinstance(value, bool):
            if value:
                argv.append(flag)
            continue
        argv.extend([flag, _render_scalar(value)])
    return argv


def _merge_defaults(defaults: Mapping[str, Any], run: Mapping[str, Any]) -> Dict[str, Any]:
    """Merge ``defaults`` under ``run`` (run wins). ``env`` merges key-by-key."""
    merged: Dict[str, Any] = dict(defaults)
    merged.update({k: v for k, v in run.items() if k != "env"})
    default_env = dict(defaults.get("env") or {})
    run_env = dict(run.get("env") or {})
    default_env.update(run_env)
    if default_env:
        merged["env"] = default_env
    return merged


def build_run_argv(run: Mapping[str, Any]) -> List[str]:
    """Translate one (already-merged) run mapping into a runner argv list.

    Order: scalar flags (in schema order) -> nested blocks -> bool flags ->
    extra_args verbatim. Deterministic so dry-run output is stable/diffable.
    """
    unknown = sorted(set(run) - _ALLOWED_RUN_KEYS)
    if unknown:
        raise BundleError(
            f"run {run.get('name', '<unnamed>')!r}: unknown key(s) {unknown}; "
            f"allowed: {sorted(_ALLOWED_RUN_KEYS)}"
        )
    argv: List[str] = []
    for key, flag in _SCALAR_FLAG_MAP.items():
        if key in run and run[key] is not None:
            argv.extend([flag, _render_scalar(run[key])])
    for key, prefix in _NESTED_PREFIX_MAP.items():
        if key in run and run[key]:
            argv.extend(_nested_flags(prefix, run[key]))
    for key, flag in _BOOL_FLAG_MAP.items():
        if run.get(key):
            argv.append(flag)
    extra = run.get("extra_args") or []
    if not isinstance(extra, (list, tuple)):
        raise BundleError(
            f"run {run.get('name', '<unnamed>')!r}: extra_args must be a list"
        )
    argv.extend(str(tok) for tok in extra)
    return argv


def load_bundle(path: Path) -> Dict[str, Any]:
    """Load + validate a bundle YAML file's top-level structure."""
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, Mapping):
        raise BundleError(f"{path}: top level must be a mapping")
    unknown = sorted(set(raw) - _ALLOWED_TOP_KEYS)
    if unknown:
        raise BundleError(
            f"{path}: unknown top-level key(s) {unknown}; "
            f"allowed: {sorted(_ALLOWED_TOP_KEYS)}"
        )
    runs = raw.get("runs")
    if not isinstance(runs, Sequence) or isinstance(runs, (str, bytes)) or not runs:
        raise BundleError(f"{path}: 'runs' must be a non-empty list")
    for i, run in enumerate(runs):
        if not isinstance(run, Mapping):
            raise BundleError(f"{path}: runs[{i}] must be a mapping")
    defaults = raw.get("defaults") or {}
    if not isinstance(defaults, Mapping):
        raise BundleError(f"{path}: 'defaults' must be a mapping")
    return dict(raw)


def expand_bundle(bundle: Mapping[str, Any]) -> List[Dict[str, Any]]:
    """Return one ``{name, env, argv}`` cell per run, defaults merged in."""
    defaults = bundle.get("defaults") or {}
    cells: List[Dict[str, Any]] = []
    for run in bundle["runs"]:
        merged = _merge_defaults(defaults, run)
        name = merged.get("name", f"run_{len(cells)}")
        env = {str(k): str(v) for k, v in (merged.get("env") or {}).items()}
        argv = build_run_argv(merged)
        cells.append({"name": name, "env": env, "argv": argv})
    return cells


def _format_invocation(cell: Mapping[str, Any]) -> str:
    """Render a copy-pasteable ``ENV=v python runner ...`` line for a cell."""
    env_prefix = " ".join(
        f"{k}={shlex.quote(v)}" for k, v in sorted(cell["env"].items())
    )
    tokens = [sys.executable, str(LADDER_RUNNER), *cell["argv"]]
    cmd = " ".join(shlex.quote(t) for t in tokens)
    return f"{env_prefix + ' ' if env_prefix else ''}{cmd}"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bundle", type=Path, help="Path to a bundle YAML file.")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the exact runner invocation(s) without executing them.",
    )
    parser.add_argument(
        "--only",
        default=None,
        help="Comma-list of run names to include (skip the rest). Default: all.",
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Keep launching remaining runs if one exits non-zero.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    bundle = load_bundle(args.bundle)
    cells = expand_bundle(bundle)
    if args.only:
        wanted = {n.strip() for n in str(args.only).split(",") if n.strip()}
        unknown = wanted - {c["name"] for c in cells}
        if unknown:
            raise SystemExit(
                f"--only names not in bundle: {sorted(unknown)}; "
                f"available: {[c['name'] for c in cells]}"
            )
        cells = [c for c in cells if c["name"] in wanted]

    print(
        f"# bundle {args.bundle} -> {len(cells)} run(s): "
        f"{[c['name'] for c in cells]}",
        file=sys.stderr,
    )
    failures: List[str] = []
    for cell in cells:
        line = _format_invocation(cell)
        if args.dry_run:
            print(f"# run: {cell['name']}")
            print(line)
            continue
        print(f"# run: {cell['name']}", file=sys.stderr)
        print(line, file=sys.stderr)
        env = dict(os.environ)
        env.update(cell["env"])
        proc = subprocess.run(
            [sys.executable, str(LADDER_RUNNER), *cell["argv"]],
            cwd=str(PROJECT_ROOT),
            env=env,
        )
        if proc.returncode != 0:
            failures.append(cell["name"])
            msg = f"run {cell['name']!r} exited with code {proc.returncode}"
            if args.continue_on_error:
                print(f"# WARN: {msg} (continuing)", file=sys.stderr)
            else:
                print(f"# ERROR: {msg}", file=sys.stderr)
                return proc.returncode
    if failures:
        print(f"# {len(failures)} run(s) failed: {failures}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
