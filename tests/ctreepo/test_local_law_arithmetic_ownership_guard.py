"""Repo-wide ownership guard for canonical local-law arithmetic.

Phase 0 of ``docs/local_law_single_path_master_plan.md`` requires that the only
implementation of AIPW / sampled-IPW / depth-discount / root-local convex mixing
lives upstream in ``treepo.training.local_law`` and ``treepo.objective``. This
guard complements the narrower fragment check in
``tests/ctreepo/test_local_law_source_guards.py``:

- It bans any ThinkingTrees ``src/`` module from *defining* a canonical local-law
  arithmetic function (re-implementing it instead of importing the shim).
- It bans ``src/`` imports of archived trees (``treepo._research``,
  ``treepo_cdx``, ``OLD_*``) so the archive-import guard fails once those paths
  are retired in Phase 7.

If a future change legitimately needs to add a new canonical helper, add it
upstream and import it; do not extend the allowlist here without updating the
master plan's "canonical home" table.
"""

from __future__ import annotations

import re
from pathlib import Path


_REPO_ROOT = Path(__file__).resolve().parents[2]
_SRC_ROOT = _REPO_ROOT / "src"


# Canonical arithmetic that must be defined ONLY upstream in
# ``treepo.training.local_law`` / ``treepo.local_law`` / ``treepo.objective``.
# ThinkingTrees modules must import these, never redefine them.
_CANONICAL_ARITHMETIC_DEFS = (
    "corrected_local_law_loss",
    "corrected_local_law_loss_tensor",
    "corrected_local_law_target_mse",
    "local_law_objective_from_losses",
    "local_law_objective_target_mse",
    "local_law_training_objective_mean",
    "aggregate_local_law_training_rows",
    "sampled_uniform_node_ipw_mean_loss",
    "observed_uniform_node_ipw_mean_loss",
    "depth_discount",
    "_depth_discount_weights",
    "resolve_root_local_objective_weights",
)

# No ThinkingTrees ``src/`` module may define these. (Empty allowlist on purpose:
# the canonical home is upstream ``treepo``.)
_DEFINITION_ALLOWLIST: dict[str, set[str]] = {}

_ARCHIVED_IMPORT_PATTERNS = (
    re.compile(r"\btreepo\._research\b"),
    re.compile(r"\btreepo_cdx\b"),
    re.compile(r"(?:from|import)\s+OLD_"),
)


def _src_python_files() -> list[Path]:
    return [
        path
        for path in _SRC_ROOT.rglob("*.py")
        if "__pycache__" not in path.parts
    ]


def test_canonical_local_law_arithmetic_defined_only_upstream() -> None:
    offenders: list[str] = []
    for path in _src_python_files():
        rel = path.relative_to(_REPO_ROOT).as_posix()
        text = path.read_text(encoding="utf-8")
        for name in _CANONICAL_ARITHMETIC_DEFS:
            if re.search(rf"^\s*def {re.escape(name)}\s*\(", text, flags=re.MULTILINE):
                if rel in _DEFINITION_ALLOWLIST.get(name, set()):
                    continue
                offenders.append(f"{rel} defines canonical helper {name!r}")
    assert not offenders, (
        "Canonical local-law arithmetic must live upstream in treepo, not be "
        "re-implemented in ThinkingTrees src/. Offenders: " + "; ".join(offenders)
    )


def test_src_does_not_import_archived_local_law_trees() -> None:
    offenders: list[str] = []
    for path in _src_python_files():
        rel = path.relative_to(_REPO_ROOT).as_posix()
        text = path.read_text(encoding="utf-8")
        for pattern in _ARCHIVED_IMPORT_PATTERNS:
            if pattern.search(text):
                offenders.append(f"{rel} matches archived-import pattern {pattern.pattern!r}")
    assert not offenders, (
        "ThinkingTrees src/ must not import archived trees (treepo._research, "
        "treepo_cdx, OLD_*). Offenders: " + "; ".join(offenders)
    )


def test_canonical_helpers_are_importable_upstream() -> None:
    # Positive control: the canonical homes actually export the contract.
    from treepo.training import local_law as training_local_law
    from treepo import objective as treepo_objective

    for name in (
        "corrected_local_law_loss_tensor",
        "local_law_objective_from_losses",
        "local_law_objective_target_mse",
        "local_law_training_objective_mean",
        "aggregate_local_law_training_rows",
        "LocalLawTrainingRow",
    ):
        assert hasattr(training_local_law, name), f"treepo.training.local_law missing {name}"
    assert hasattr(treepo_objective, "resolve_root_local_objective_weights")
    assert hasattr(treepo_objective, "ObjectiveSpec")
