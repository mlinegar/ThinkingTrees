"""Alternating f/g optimization trampoline for the manifesto distillation ladder.

The ladder name counts alternations starting from supplied ``(f_init, g_init)``:

- ``k = 0`` -> ``fg``       (no training; evaluate ``f_init`` on ``g_init``'s output)
- ``k = 1`` -> ``fgf``      (produce ``f1`` from ``(f_init, g_init)``; g unchanged)
- ``k = 2`` -> ``fgfg``     (produce ``g1`` from ``(f1, g_init)``; f unchanged)
- ``k = 3`` -> ``fgfgf``    (produce ``f2`` from ``(f1, g1)``)
- ``k = 4`` -> ``fgfgfg``   (produce ``g2`` from ``(f2, g1)``)
- ... alternating odd -> train f, even -> train g.

**Training signal for g**: when training ``g_k``, the scoring / reward function
is the *current* student ``f_k``, NOT the teacher or the gold expert. This is
the whole point of the alternation: f and g co-adapt. The f-vs-f* gap between
"what our f says" (internal) and "what the expert says" (external) must be
measured at every iteration to surface reward-hacking.

This module defines:
- ``FamilyRuntime``: the protocol every backend family implements.
- ``stage_name_for_iteration(k)``: maps integer k to a stage id.
- ``stage_label_for_iteration(k)``: human-readable power notation.
- ``run_alternating_family(...)``: the shared loop.
- ``IterationRecord``: the output schema.
"""

from __future__ import annotations

import json
import logging
import math
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Mapping,
    Optional,
    Protocol,
    Sequence,
    runtime_checkable,
)

LOGGER = logging.getLogger(__name__)


def _json_write_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f"{path.name}.tmp")
    tmp_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    tmp_path.replace(path)


def _artifact_for_manifest(artifact: Any) -> Optional[str]:
    if artifact is None:
        return None
    return str(artifact)


def _normalize_first_train_side(value: str) -> str:
    side = str(value or "f").strip().lower()
    if side not in {"f", "g"}:
        raise ValueError(f"first_train_side must be 'f' or 'g', got {value!r}")
    return side


def stage_powers_for_iteration(
    k: int,
    *,
    first_train_side: str = "f",
    initial_f_degree: int = 1,
    initial_g_degree: int = 1,
) -> tuple[int, int]:
    """Return the ``(f_degree, g_degree)`` at iteration ``k``."""
    if k < 0:
        raise ValueError(f"iteration must be >= 0, got {k}")
    f_degree = int(initial_f_degree)
    g_degree = int(initial_g_degree)
    if f_degree < 0 or g_degree < 0:
        raise ValueError(
            "initial degrees must be >= 0, got "
            f"f={initial_f_degree!r} g={initial_g_degree!r}"
        )
    next_side = _normalize_first_train_side(first_train_side)
    for _ in range(int(k)):
        if next_side == "f":
            f_degree += 1
            next_side = "g"
        else:
            g_degree += 1
            next_side = "f"
    return f_degree, g_degree


def _write_step_checkpoint(
    *,
    output_dir: Path,
    family: FamilyRuntime,
    axis_kind: str,
    axis_value: int,
    leaf_count: Optional[int],
    leaf_size_tokens: Optional[int],
    iteration: int,
    stage_name: str,
    stage_label: Optional[str],
    f_degree: Optional[int],
    g_degree: Optional[int],
    trained: str,
    phase: str,
    f_artifact: Any,
    g_artifact: Any,
    iteration_dir: Optional[Path] = None,
    split_metrics: Optional[Mapping[str, "SplitMetrics"]] = None,
    error: Optional[str] = None,
    artifact_validation: Optional[Mapping[str, Any]] = None,
) -> Path:
    checkpoints_dir = Path(output_dir) / "step_checkpoints"
    payload: Dict[str, Any] = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "family": str(family.name),
        "axis_kind": str(axis_kind),
        "axis_value": int(axis_value),
        "leaf_count": int(leaf_count) if leaf_count is not None else None,
        "leaf_size_tokens": int(leaf_size_tokens) if leaf_size_tokens is not None else None,
        "iteration": int(iteration),
        "stage_name": str(stage_name),
        "stage_label": str(stage_label) if stage_label is not None else None,
        "f_degree": int(f_degree) if f_degree is not None else None,
        "g_degree": int(g_degree) if g_degree is not None else None,
        "trained": str(trained),
        "phase": str(phase),
        "f_artifact": _artifact_for_manifest(f_artifact),
        "g_artifact": _artifact_for_manifest(g_artifact),
        "iteration_dir": str(iteration_dir) if iteration_dir is not None else None,
        "error": error,
        "artifact_validation": dict(artifact_validation or {}),
    }
    if split_metrics is not None:
        payload["split_metrics"] = {
            str(name): asdict(metrics) for name, metrics in split_metrics.items()
        }
    checkpoint_path = checkpoints_dir / f"iter_{int(iteration):02d}_{phase}.json"
    payload["checkpoint_path"] = str(checkpoint_path)
    _json_write_atomic(checkpoint_path, payload)
    _json_write_atomic(checkpoints_dir / "latest.json", payload)
    LOGGER.info("Wrote alternating step checkpoint %s", checkpoint_path)
    return checkpoint_path


def _legacy_stage_name_for_iteration(k: int) -> str:
    if k < 0:
        raise ValueError(f"iteration must be >= 0, got {k}")
    if k == 0:
        return "fg"
    tail = "".join("f" if i % 2 == 0 else "g" for i in range(k))
    return "fg" + tail


def stage_name_for_iteration(
    k: int,
    *,
    first_train_side: str = "f",
    initial_f_degree: int = 1,
    initial_g_degree: int = 1,
    naming: str = "legacy",
) -> str:
    """Map iteration ``k`` to a stage identifier.

    ``naming="legacy"`` preserves the historical ``fg -> fgf -> fgfg`` labels
    for the canonical ``(f^1, g^1)`` / f-first ladder. Any other setup falls
    back to compact power ids like ``f1g0``.
    """
    mode = str(naming or "legacy").strip().lower()
    if mode not in {"legacy", "powers"}:
        raise ValueError(f"naming must be 'legacy' or 'powers', got {naming!r}")
    side = _normalize_first_train_side(first_train_side)
    if (
        mode == "legacy"
        and side == "f"
        and int(initial_f_degree) == 1
        and int(initial_g_degree) == 1
    ):
        return _legacy_stage_name_for_iteration(k)
    f_degree, g_degree = stage_powers_for_iteration(
        k,
        first_train_side=side,
        initial_f_degree=initial_f_degree,
        initial_g_degree=initial_g_degree,
    )
    return f"f{f_degree}g{g_degree}"


def stage_label_for_iteration(
    k: int,
    *,
    first_train_side: str = "f",
    initial_f_degree: int = 1,
    initial_g_degree: int = 1,
) -> str:
    """Return human-readable power notation for iteration ``k``."""
    f_degree, g_degree = stage_powers_for_iteration(
        k,
        first_train_side=first_train_side,
        initial_f_degree=initial_f_degree,
        initial_g_degree=initial_g_degree,
    )
    return f"f^{f_degree} g^{g_degree}"


def trains_f_at_iteration(k: int, *, first_train_side: str = "f") -> bool:
    """Return True if iteration ``k`` trains f."""
    if k < 1:
        return False
    side = _normalize_first_train_side(first_train_side)
    return (k % 2 == 1) if side == "f" else (k % 2 == 0)


def trains_g_at_iteration(k: int, *, first_train_side: str = "f") -> bool:
    """Return True if iteration ``k`` trains g."""
    if k < 1:
        return False
    side = _normalize_first_train_side(first_train_side)
    return (k % 2 == 1) if side == "g" else (k % 2 == 0)


@runtime_checkable
class FamilyRuntime(Protocol):
    """Contract every alternating-optimization backend family must satisfy.

    Families own their artifact types: ``FArtifact`` / ``GArtifact`` are opaque
    Any handles that the family constructs and consumes. The trampoline only
    threads them through.
    """

    #: Short family name, e.g. ``"dspy"`` / ``"trl"`` / ``"fno"``.
    name: str

    def train_f(
        self,
        *,
        f_init: Any,
        g: Any,
        traces: Sequence[Any],
        output_dir: Path,
        iteration: int,
    ) -> Any:
        """Train f for one iteration. Returns the new f artifact."""
        ...

    def train_g(
        self,
        *,
        g_init: Any,
        f: Any,
        traces: Sequence[Any],
        output_dir: Path,
        iteration: int,
    ) -> Any:
        """Train g for one iteration. Returns the new g artifact.

        ``f`` here is the *current* student f; the family must use it as the
        scoring / reward signal for g training, NOT the teacher.
        """
        ...

    def score_roots_with_f(
        self,
        *,
        f: Any,
        g: Any,
        trees: Sequence[Any],
    ) -> List[Optional[float]]:
        """Apply ``f(g(tree_root))`` for each tree; return 1-7 predictions.

        Returns a list aligned with ``trees``; entries may be ``None`` for
        trees the family cannot score (missing text, failed inference, etc.).
        """
        ...

    def validate_artifact(self, *, kind: str, artifact: Any) -> None:
        """Raise if a returned artifact cannot be reloaded for a future step."""
        ...


def _validate_family_artifact(
    family: FamilyRuntime,
    *,
    kind: str,
    artifact: Any,
) -> Optional[str]:
    validator = getattr(family, "validate_artifact", None)
    if not callable(validator):
        return None
    validator(kind=str(kind), artifact=artifact)
    return "passed"


@dataclass
class IterationRecord:
    """One row of the alternating-optimization output history."""

    iteration: int
    stage_name: str          # "fg", "fgf", "fgfg", ...
    family: str
    trained: str             # "none", "f", "g"
    stage_label: Optional[str] = None
    f_degree: Optional[int] = None
    g_degree: Optional[int] = None
    axis_kind: str = "leaf_count"
    axis_value: int = 0
    leaf_count: Optional[int] = None
    leaf_size_tokens: Optional[int] = None
    f_artifact: Optional[str] = None
    g_artifact: Optional[str] = None
    #: Per-split metrics. Keys are split names ("train", "val", "test", "all").
    split_metrics: Dict[str, "SplitMetrics"] = field(default_factory=dict)
    extra: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SplitMetrics:
    """Pearson + MAE vs teacher-f (internal) and vs gold expert (external)."""

    n: int
    #: Internal: how closely current f agrees with the teacher f at the root.
    internal_f_pearson: Optional[float] = None
    internal_f_mae_1_7: Optional[float] = None
    #: External: Pearson / MAE vs gold expert score (the paper-facing metric).
    external_expert_pearson: Optional[float] = None
    external_expert_mae_1_7: Optional[float] = None
    #: ``internal_f_pearson - external_expert_pearson``. Positive = f is
    #: drifting from expert signal while still agreeing with the teacher.
    f_star_gap: Optional[float] = None
    mean_prediction_1_7: Optional[float] = None
    mean_teacher_1_7: Optional[float] = None
    mean_expert_1_7: Optional[float] = None


def _safe_float(value: Any) -> Optional[float]:
    try:
        if value is None:
            return None
        f = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(f):
        return None
    return f


def _pearson_and_mae(
    preds: Sequence[Optional[float]],
    truths: Sequence[Optional[float]],
) -> Dict[str, Optional[float]]:
    """Compute Pearson r + MAE on paired lists, dropping entries with None on either side."""
    from src.tasks.manifesto.corpus_metrics import compute_corpus_pearson_r

    paired = [
        (float(p), float(t))
        for p, t in zip(preds, truths)
        if p is not None and t is not None
    ]
    if len(paired) < 4:
        return {
            "n": len(paired),
            "pearson_r": None,
            "mae_1_7": None,
            "mean_prediction_1_7": float(sum(p for p, _ in paired) / len(paired)) if paired else None,
            "mean_truth_1_7": float(sum(t for _, t in paired) / len(paired)) if paired else None,
        }
    ps, ts = zip(*paired)
    corr = compute_corpus_pearson_r(ps, ts).as_dict()
    mae = sum(abs(p - t) for p, t in paired) / len(paired)
    return {
        "n": len(paired),
        "pearson_r": corr.get("pearson_r"),
        "mae_1_7": float(mae),
        "mean_prediction_1_7": float(sum(ps) / len(ps)),
        "mean_truth_1_7": float(sum(ts) / len(ts)),
    }


def _tree_split(tree: Any) -> str:
    metadata = getattr(tree, "metadata", None) or {}
    return str(metadata.get("split") or "unknown").lower()


def _teacher_root_score(tree: Any) -> Optional[float]:
    """Extract the teacher f's root score from a LabeledTree."""
    metadata = getattr(tree, "metadata", None) or {}
    root_level = getattr(tree, "levels", None) or []
    if root_level:
        for node_id in reversed(root_level[-1]):
            node = tree.get_node(str(node_id)) if hasattr(tree, "get_node") else None
            if node is not None:
                s = _safe_float(getattr(node, "score", None))
                if s is not None:
                    return s
                nm = node.metadata or {}
                return _safe_float(nm.get("teacher_score_1_7"))
    return _safe_float(metadata.get("teacher_score_1_7") or getattr(tree, "document_score", None))


def _expert_root_score(tree: Any) -> Optional[float]:
    metadata = getattr(tree, "metadata", None) or {}
    return _safe_float(metadata.get("expert_score_1_7"))


def evaluate_iteration(
    *,
    family: FamilyRuntime,
    f: Any,
    g: Any,
    trees: Sequence[Any],
    splits: Sequence[str] = ("all", "train", "val", "test"),
) -> Dict[str, SplitMetrics]:
    """Produce the per-split metric dict for one iteration."""
    preds = family.score_roots_with_f(f=f, g=g, trees=list(trees))
    if len(preds) != len(trees):
        raise RuntimeError(
            f"{family.name}.score_roots_with_f returned {len(preds)} predictions "
            f"for {len(trees)} trees"
        )
    tree_splits = [_tree_split(t) for t in trees]
    teacher_scores = [_teacher_root_score(t) for t in trees]
    expert_scores = [_expert_root_score(t) for t in trees]

    out: Dict[str, SplitMetrics] = {}
    for split in splits:
        if split == "all":
            idxs = list(range(len(trees)))
        else:
            idxs = [i for i, s in enumerate(tree_splits) if s == split.lower()]
        if not idxs:
            out[split] = SplitMetrics(n=0)
            continue
        split_preds = [preds[i] for i in idxs]
        split_teacher = [teacher_scores[i] for i in idxs]
        split_expert = [expert_scores[i] for i in idxs]
        internal = _pearson_and_mae(split_preds, split_teacher)
        external = _pearson_and_mae(split_preds, split_expert)
        gap: Optional[float] = None
        if internal["pearson_r"] is not None and external["pearson_r"] is not None:
            gap = float(internal["pearson_r"]) - float(external["pearson_r"])
        out[split] = SplitMetrics(
            n=int(internal["n"]),
            internal_f_pearson=internal["pearson_r"],
            internal_f_mae_1_7=internal["mae_1_7"],
            external_expert_pearson=external["pearson_r"],
            external_expert_mae_1_7=external["mae_1_7"],
            f_star_gap=gap,
            mean_prediction_1_7=internal["mean_prediction_1_7"],
            mean_teacher_1_7=internal["mean_truth_1_7"],
            mean_expert_1_7=external["mean_truth_1_7"],
        )
    return out


def run_alternating_family(
    *,
    family: FamilyRuntime,
    f_init: Any,
    g_init: Any,
    traces: Sequence[Any],
    eval_trees: Sequence[Any],
    max_iterations: int,
    axis_value: int,
    output_dir: Path,
    axis_kind: str = "leaf_count",
    leaf_count: Optional[int] = None,
    leaf_size_tokens: Optional[int] = None,
    first_train_side: str = "f",
    initial_f_degree: int = 1,
    initial_g_degree: int = 1,
    stage_naming: str = "legacy",
    artifact_namer: Optional[Callable[[str, int], Optional[str]]] = None,
) -> List[IterationRecord]:
    """Run the alternating loop for one ``(family, axis_value)`` row.

    Returns one ``IterationRecord`` per ``k in {0, 1, ..., max_iterations}``.

    ``artifact_namer(kind, iteration)`` lets callers customize on-disk artifact
    paths; when ``None``, artifacts inherit the family's own conventions.
    """
    if max_iterations < 0:
        raise ValueError(f"max_iterations must be >= 0, got {max_iterations}")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    first_train_side = _normalize_first_train_side(first_train_side)

    records: List[IterationRecord] = []
    f_current: Any = f_init
    g_current: Any = g_init

    for k in range(0, max_iterations + 1):
        stage = stage_name_for_iteration(
            k,
            first_train_side=first_train_side,
            initial_f_degree=initial_f_degree,
            initial_g_degree=initial_g_degree,
            naming=stage_naming,
        )
        stage_label = stage_label_for_iteration(
            k,
            first_train_side=first_train_side,
            initial_f_degree=initial_f_degree,
            initial_g_degree=initial_g_degree,
        )
        f_degree, g_degree = stage_powers_for_iteration(
            k,
            first_train_side=first_train_side,
            initial_f_degree=initial_f_degree,
            initial_g_degree=initial_g_degree,
        )
        axis_label = (
            f"leaf{int(axis_value):04d}tok"
            if str(axis_kind) == "leaf_size_tokens"
            else f"leaf_{int(axis_value):03d}"
        )
        trains_f = trains_f_at_iteration(k, first_train_side=first_train_side)
        trains_g = trains_g_at_iteration(k, first_train_side=first_train_side)
        train_side = "f" if trains_f else "g" if trains_g else "none"
        LOGGER.info(
            "[%s %s] iteration %d (%s / %s)", family.name, axis_label, k, stage, stage_label
        )
        trained = "none"
        error: Optional[str] = None
        iter_dir: Optional[Path] = None
        artifact_validation: Dict[str, Any] = {}
        try:
            if trains_f:
                iter_dir = output_dir / f"iter_{k:02d}_train_f"
                iter_dir.mkdir(parents=True, exist_ok=True)
                f_current = family.train_f(
                    f_init=f_current,
                    g=g_current,
                    traces=traces,
                    output_dir=iter_dir,
                    iteration=k,
                )
                trained = "f"
                artifact_validation["f"] = _validate_family_artifact(
                    family, kind="f", artifact=f_current
                )
            elif trains_g:
                iter_dir = output_dir / f"iter_{k:02d}_train_g"
                iter_dir.mkdir(parents=True, exist_ok=True)
                g_current = family.train_g(
                    g_init=g_current,
                    f=f_current,
                    traces=traces,
                    output_dir=iter_dir,
                    iteration=k,
                )
                trained = "g"
                artifact_validation["g"] = _validate_family_artifact(
                    family, kind="g", artifact=g_current
                )
        except NotImplementedError as exc:
            LOGGER.warning(
                "[%s %s] iteration %d train_%s not implemented: %s",
                family.name, axis_label, k, train_side, exc,
            )
            error = f"NotImplementedError: {exc}"
            trained = "skipped"
        except RuntimeError as exc:
            LOGGER.warning(
                "[%s %s] iteration %d train_%s hard-error: %s",
                family.name, axis_label, k, train_side, exc,
            )
            error = f"RuntimeError: {exc}"
            trained = "skipped"
        except Exception as exc:
            LOGGER.exception(
                "[%s %s] iteration %d train/validate_%s failed",
                family.name,
                axis_label,
                k,
                train_side,
            )
            error = f"{type(exc).__name__}: {exc}"

        _write_step_checkpoint(
            output_dir=output_dir,
            family=family,
            axis_kind=axis_kind,
            axis_value=int(axis_value),
            leaf_count=leaf_count,
            leaf_size_tokens=leaf_size_tokens,
            iteration=k,
            stage_name=stage,
            stage_label=stage_label,
            f_degree=f_degree,
            g_degree=g_degree,
            trained=trained,
            phase="post_train",
            f_artifact=f_current,
            g_artifact=g_current,
            iteration_dir=iter_dir,
            error=error,
            artifact_validation=artifact_validation,
        )

        if error is None:
            try:
                split_metrics = evaluate_iteration(
                    family=family, f=f_current, g=g_current, trees=eval_trees,
                )
            except NotImplementedError as exc:
                LOGGER.warning(
                    "[%s %s] iteration %d evaluation not implemented: %s",
                    family.name, axis_label, k, exc,
                )
                split_metrics = {}
                error = error or f"evaluation NotImplementedError: {exc}"
            except Exception as exc:
                LOGGER.exception(
                    "[%s %s] iteration %d evaluation failed after %s training",
                    family.name,
                    axis_label,
                    k,
                    trained,
                )
                split_metrics = {}
                error = error or f"evaluation {type(exc).__name__}: {exc}"
        else:
            split_metrics = {}

        extra: Dict[str, Any] = {}
        if error is not None:
            extra["error"] = error
        record = IterationRecord(
            iteration=k,
            stage_name=stage,
            stage_label=stage_label,
            family=family.name,
            f_degree=f_degree,
            g_degree=g_degree,
            axis_kind=str(axis_kind),
            axis_value=int(axis_value),
            leaf_count=int(leaf_count) if leaf_count is not None else None,
            leaf_size_tokens=int(leaf_size_tokens) if leaf_size_tokens is not None else None,
            trained=trained,
            f_artifact=_artifact_for_manifest(f_current),
            g_artifact=_artifact_for_manifest(g_current),
            split_metrics=split_metrics,
            extra=extra,
        )
        records.append(record)
        _write_step_checkpoint(
            output_dir=output_dir,
            family=family,
            axis_kind=axis_kind,
            axis_value=int(axis_value),
            leaf_count=leaf_count,
            leaf_size_tokens=leaf_size_tokens,
            iteration=k,
            stage_name=stage,
            stage_label=stage_label,
            f_degree=f_degree,
            g_degree=g_degree,
            trained=trained,
            phase="post_eval",
            f_artifact=f_current,
            g_artifact=g_current,
            iteration_dir=iter_dir,
            split_metrics=split_metrics,
            error=error,
            artifact_validation=artifact_validation,
        )
        # If training was skipped due to NotImplementedError, subsequent
        # iterations would produce the same error; stop early but keep
        # the records we did gather.
        if error is not None and trained == "skipped":
            break
        if error is not None and not split_metrics:
            break
    return records
