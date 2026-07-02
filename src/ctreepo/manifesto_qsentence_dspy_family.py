"""DSPy f/g family for Manifesto Project quasi-sentence supervision.

This family keeps the alternating ladder contract from ``DSPyFamily`` and
``JointDSPyFamily`` but swaps the supervision source: every labeled node target
is derived exactly from descendant Manifesto Project CMP quasi-sentence labels.
The learned ``g`` program produces compact CMP policy states; the learned ``f``
program predicts compact aggregate targets from those states.
"""

from __future__ import annotations

import json
import logging
import random
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

from src.ctreepo.dspy_family import (
    DSPyFamily,
    DSPyFamilyConfig,
    _clamp01,
    _parse_first_float,
    _root_node,
    _root_text,
)
from src.tasks.manifesto.span_targets import (
    COMPACT_TARGET_DIMENSIONS,
    parse_compact_scores_json,
)
from src.tree.labeled import LabeledNode, LabeledTree

LOGGER = logging.getLogger(__name__)


try:  # Keep import of this module cheap for tests that do not exercise DSPy.
    import dspy as _dspy
except Exception:  # pragma: no cover - exercised only in non-DSPy envs.
    _dspy = None


QSENTENCE_TARGET_SIGNATURE_DOC = (
    "Read the compact Manifesto Project CMP target vector from a candidate "
    "policy-state summary. The summary may be written in ANY format -- canonical "
    "JSON, a flat or dotted-key object, a salience/domains map, a list of policy "
    "objects, RILE given as a direction word (left/right/pos/neg/center), or "
    "free-form prose. Interpret the policy intent it expresses and emit the "
    "compact vector; do not reject it for using a non-canonical schema. If the "
    "summary states it has no scorable policy content (e.g. an empty or "
    "'not relevant' span), return the neutral state rile=0.5 with all domain "
    "shares 0.\n\n"
    "Return only a JSON object with keys `rile`, `domain_1`, `domain_2`, "
    "`domain_3`, `domain_4`, `domain_5`, `domain_6`, and `domain_7`. Each value "
    "must be a number in [0, 1]. `rile` is normalized from raw RILE [-100,100] "
    "into [0,1] (left=0, center=0.5, right=1); each domain value is the share of "
    "non-headline quasi-sentences in that CMP domain."
)


if _dspy is not None:

    class ManifestoQSentenceTargetSignature(_dspy.Signature):
        __doc__ = QSENTENCE_TARGET_SIGNATURE_DOC

        summary: str = _dspy.InputField(desc="Compact policy state or summary to score")
        scores_json: str = _dspy.OutputField(
            desc=(
                "Strict JSON object with numeric [0,1] keys: rile, domain_1, "
                "domain_2, domain_3, domain_4, domain_5, domain_6, domain_7"
            )
        )


    class ManifestoQSentenceTargetScorer(_dspy.Module):
        """DSPy module for compact quasi-sentence aggregate target prediction."""

        def __init__(self, *, max_output_tokens: int = 256) -> None:
            super().__init__()
            self.max_output_tokens = int(max_output_tokens)
            self.predictor = _dspy.Predict(ManifestoQSentenceTargetSignature)

        def load_state(self, state: Any) -> None:
            compat_state = dict(state)
            if "predictor" not in compat_state and "scores_json" in compat_state:
                compat_state["predictor"] = compat_state["scores_json"]
            super().load_state(compat_state)

        def forward(self, summary: str) -> dict[str, Any]:
            predictor = getattr(self, "predictor", None)
            if not callable(predictor):
                predictor = _dspy.Predict(ManifestoQSentenceTargetSignature)
                self.predictor = predictor
            result = predictor(
                summary=str(summary or ""),
                config={"max_tokens": int(self.max_output_tokens)},
            )
            raw = str(getattr(result, "scores_json", "") or "")
            scores = parse_compact_scores_json(raw)
            if not scores:
                scores = parse_compact_scores_json(str(result))
            return {
                "scores_json": json.dumps(scores, sort_keys=True) if scores else raw,
                "scores": scores,
            }

else:
    ManifestoQSentenceTargetSignature = None

    class ManifestoQSentenceTargetScorer:  # type: ignore[no-redef]
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            raise RuntimeError("dspy is required for ManifestoQSentenceTargetScorer")


@dataclass
class ManifestoQSentenceDSPyFamilyConfig(DSPyFamilyConfig):
    """Config for the quasi-sentence compact-target DSPy family."""

    dimension: str = "manifesto_qsentence"
    target_dimensions: Sequence[str] = field(
        default_factory=lambda: tuple(COMPACT_TARGET_DIMENSIONS)
    )
    # Empty default means "start from a bare compact-target scorer" rather than
    # inheriting the Benoit dimension scorer warm start from ``DSPyFamily``.
    f_init_path: Optional[str] = ""
    # During g optimization, reward the candidate state primarily by directly
    # parsing its compact CMP targets. The f-proxy component remains useful for
    # the alternating contract, but cannot rescue unparseable state text.
    g_direct_parse_reward_weight: float = 0.75
    g_f_proxy_reward_weight: float = 0.25
    # Q-sentence LLM comparisons are conference-quality runs: parser, adapter,
    # and transport exceptions must abort instead of becoming objective noise.
    strict_optimizer_errors: bool = True
    # Eval/inference can see rare blank or malformed generations from local
    # servers. Retry those, but never accept an unparseable compact state.
    g_inference_retries: int = 2
    # Conference-quality runs can require that even transient malformed g
    # outputs are treated as failed runs, rather than hidden by retries.
    fail_fast_on_invalid_g_state: bool = False
    # f is an LLM readout that can interpret a non-canonical g state, so by
    # default a non-empty g output is accepted (the f scorer reads intent
    # downstream). Set True to restore the strict gate that requires all 8
    # compact dimensions to be structurally parseable from g's raw output.
    g_require_canonical_state: bool = False
    # Scheduled sampling (DAgger-style) to close the train/eval exposure-bias gap:
    # at train time the merge prompt normally feeds g the GOLD child summaries, but
    # at eval (_generate_root_state) g consumes its OWN generated child states. With
    # a nonzero rate, a fraction of merge nodes substitute g's generated child state
    # for the gold child summary during g-training, so g learns to merge the noisier
    # states it will actually see. This is NOT an averaging/mean-preserving
    # constraint: g stays a fully general learned merge; only the INPUT distribution
    # is matched to eval. Rate 0.0 = legacy gold-children behavior.
    g_scheduled_sampling_rate: float = 0.0
    # Linear ramp of the rate across alternating iterations: effective rate at
    # iteration i is min(rate, rate_start + i * rate_ramp_per_iter). Start low so
    # early g sees mostly gold children, then increasingly its own.
    g_scheduled_sampling_rate_start: float = 0.0
    g_scheduled_sampling_ramp_per_iter: float = 0.0
    # Lopsidedness-weighted C2 calibration. The dim/RILE targets are ratios with
    # a per-node ``total_non_header`` denominator, so the correct merge is the
    # MASS-WEIGHTED mean of child ratios, not the equal-average. Most merge nodes
    # (level-1 leaf pairs) have near-balanced sibling masses where equal-average
    # ~= mass-weighted, so an UNWEIGHTED node loss lets g collapse to averaging.
    # With strength>0 each g node's reward (and the per-level eval) is weighted by
    # ``1 + strength * lopsidedness`` (lopsidedness = |m_l-m_r|/(m_l+m_r) in [0,1])
    # so deep lopsided merges -- where mass-weighting strictly beats averaging --
    # dominate the gradient. This is laws-as-supervision (C2 to each node's gold
    # ratio), NOT a closed-form averaging constraint: g must recover mass to win.
    # 0.0 = legacy unweighted behavior.
    g_lopsidedness_weight_strength: float = 0.0
    # ALL FOUR paper local laws as g-training reward terms (not just C2). The
    # base direct+proxy reward IS C2 (g's state reads, through f, as the gold
    # node target -> idempotence). These add the other three, each measured as an
    # f-readout discrepancy 1 - mean|f(a) - f(b)| (same metric as the auditor):
    #   C1 sufficiency (leaf):  g's leaf state reads like the raw leaf span.
    #   C3a joint faith (merge):g's merge reads like the f-readout of u@v concat.
    #   C3b compositionality:   merging g's OWN children reads like merging gold
    #                           children. C3b is the deployment/exposure-bias law;
    #                           it is ALSO addressed by scheduled sampling (which
    #                           trains g on its own generated children). Enabling
    #                           the reward term here costs extra g+f calls/record,
    #                           so default 0; prefer g_scheduled_sampling_rate for
    #                           the exposure-bias law unless you want it in-objective.
    # Weights are relative; combined with the C2 base via normalization at use.
    g_law_c1_reward_weight: float = 0.0
    g_law_c3a_reward_weight: float = 0.0
    g_law_c3b_reward_weight: float = 0.0

    def __post_init__(self) -> None:
        super().__post_init__()
        retries = int(self.g_inference_retries)
        if retries < 0:
            raise ValueError(
                f"g_inference_retries must be >= 0, got {self.g_inference_retries!r}"
            )
        self.g_inference_retries = retries
        for name in (
            "g_scheduled_sampling_rate",
            "g_scheduled_sampling_rate_start",
            "g_scheduled_sampling_ramp_per_iter",
        ):
            val = float(getattr(self, name))
            if not (0.0 <= val <= 1.0) and name != "g_scheduled_sampling_ramp_per_iter":
                raise ValueError(f"{name} must be in [0, 1], got {val!r}")
            setattr(self, name, val)


def _active_target_dimensions(config: ManifestoQSentenceDSPyFamilyConfig) -> List[str]:
    requested = [str(dim) for dim in config.target_dimensions if str(dim)]
    requested_set = set(requested)
    return [dim for dim in COMPACT_TARGET_DIMENSIONS if dim in requested_set]


def _summary_target(
    node: Optional[LabeledNode],
    *,
    include_identity_targets: bool = False,
) -> Optional[str]:
    if node is None:
        return None
    metadata = dict(node.metadata or {})
    for key in (
        "target_summary",
        "teacher_summary",
        "teacher_leaf_summary",
        "teacher_merge_summary",
        "summary",
    ):
        value = metadata.get(key)
        if value is not None and str(value).strip():
            return str(value).strip()
    if include_identity_targets and str(node.text or "").strip():
        return str(node.text)
    return None


def _prediction_scores(pred: Any) -> Dict[str, float]:
    if isinstance(pred, Mapping):
        if isinstance(pred.get("scores"), Mapping):
            parsed = parse_compact_scores_json(pred.get("scores"))
            if parsed:
                return parsed
        parsed = parse_compact_scores_json(pred.get("scores_json"))
        if parsed:
            return parsed
        return parse_compact_scores_json(pred)
    raw_scores = getattr(pred, "scores", None)
    if isinstance(raw_scores, Mapping):
        parsed = parse_compact_scores_json(raw_scores)
        if parsed:
            return parsed
    parsed = parse_compact_scores_json(getattr(pred, "scores_json", None))
    if parsed:
        return parsed
    return parse_compact_scores_json(str(pred))


def _node_target_scores(node: Optional[LabeledNode]) -> Dict[str, float]:
    if node is None:
        return {}
    out: Dict[str, float] = {}
    raw = getattr(node, "dimension_scores", None)
    if isinstance(raw, Mapping):
        for dim, value in raw.items():
            parsed = _parse_first_float(value)
            if parsed is not None:
                out[str(dim)] = _clamp01(float(parsed))
    metadata = dict(node.metadata or {})
    for key in ("target_dimension_scores_0_1", "teacher_dimension_scores_1_7", "dimension_scores"):
        raw_meta = metadata.get(key)
        if isinstance(raw_meta, Mapping):
            for dim, value in raw_meta.items():
                parsed = _parse_first_float(value)
                if parsed is not None:
                    out.setdefault(str(dim), _clamp01(float(parsed)))
    if not out:
        summary = _summary_target(node, include_identity_targets=False)
        out.update(parse_compact_scores_json(summary))
    return out


def _node_mass(node: Optional[LabeledNode]) -> Optional[float]:
    """Subtree mass (``total_non_header`` quasi-sentence count) for a node.

    The dim/RILE targets are ratios with denominator ``total_non_header``
    (``span_targets.targets_from_counts``), so a correct merge is the
    MASS-WEIGHTED mean of child ratios, not the equal-average. Mass is read
    from the node's compact CMP state (teacher summary / metadata).
    """
    if node is None:
        return None
    metadata = dict(node.metadata or {})
    # Direct metadata first (cheap), then parse the teacher summary JSON.
    for key in ("total_non_header", "n_non_header", "node_mass"):
        value = metadata.get(key)
        parsed = _parse_first_float(value) if value is not None else None
        if parsed is not None and parsed > 0:
            return float(parsed)
    for key in ("teacher_summary", "target_summary", "summary"):
        raw = metadata.get(key)
        if not raw:
            continue
        try:
            payload = json.loads(str(raw))
        except (TypeError, ValueError):
            continue
        for container in (payload, payload.get("cmp_state") if isinstance(payload, Mapping) else None):
            if isinstance(container, Mapping):
                value = container.get("total_non_header")
                parsed = _parse_first_float(value) if value is not None else None
                if parsed is not None and parsed > 0:
                    return float(parsed)
    return None


def _node_lopsidedness(tree: LabeledTree, node: LabeledNode) -> float:
    """Sibling-mass imbalance for a merge node, in [0, 1].

    ``0`` => children carry equal mass (equal-average is correct); ``->1`` =>
    one subtree dwarfs its sibling (equal-average is maximally wrong, the merge
    must mass-weight). Leaves and merges with unknown mass return ``0.0`` so
    they contribute the neutral weight ``1.0`` (see ``_lopsidedness_weight``).
    """
    if int(node.level) == 0:
        return 0.0
    left = tree.get_node(str(node.left_child_id)) if node.left_child_id else None
    right = tree.get_node(str(node.right_child_id)) if node.right_child_id else None
    if left is None or right is None or left.node_id == right.node_id:
        return 0.0
    ml, mr = _node_mass(left), _node_mass(right)
    if ml is None or mr is None or (ml + mr) <= 0:
        return 0.0
    return float(abs(ml - mr) / (ml + mr))


def _lopsidedness_weight(lopsidedness: float, *, strength: float) -> float:
    """Map sibling-mass lopsidedness in [0,1] to a positive record weight.

    ``strength=0`` => every node weight 1.0 (legacy unweighted behavior).
    ``strength>0`` => weight ``1 + strength * lopsidedness`` so lopsided deep
    merges (where mass-weighting beats equal-averaging) dominate the C2
    calibration gradient instead of being drowned by near-balanced leaf pairs.
    """
    s = max(0.0, float(strength))
    lop = min(1.0, max(0.0, float(lopsidedness)))
    return float(1.0 + s * lop)


def _tree_target_scores(tree: LabeledTree) -> Dict[str, float]:
    root = _root_node(tree)
    if root is not None:
        scores = _node_target_scores(root)
        if scores:
            return scores
    metadata = dict(tree.metadata or {})
    for key in ("target_dimension_scores_0_1", "teacher_dimension_scores_1_7", "dimension_scores"):
        raw = metadata.get(key)
        if isinstance(raw, Mapping):
            parsed = parse_compact_scores_json(raw)
            if parsed:
                return parsed
            out = {}
            for dim, value in raw.items():
                parsed_value = _parse_first_float(value)
                if parsed_value is not None:
                    out[str(dim)] = _clamp01(float(parsed_value))
            if out:
                return out
    score = _parse_first_float(getattr(tree, "document_score", None))
    return {"rile": _clamp01(float(score))} if score is not None else {}


def _scores_json(scores: Mapping[str, Any], dims: Sequence[str]) -> str:
    payload = {}
    for dim in dims:
        parsed = _parse_first_float(scores.get(dim))
        if parsed is not None:
            payload[str(dim)] = _clamp01(float(parsed))
    return json.dumps(payload, sort_keys=True)


def _mean(values: Sequence[float]) -> Optional[float]:
    finite = [float(value) for value in values if value is not None]
    return float(sum(finite) / len(finite)) if finite else None


def _record_key(row: Mapping[str, Any]) -> Dict[str, Any]:
    meta = dict(row.get("metadata") or {})
    return {
        "doc_id": meta.get("doc_id"),
        "node_id": meta.get("node_id"),
        "split": meta.get("split"),
        "level": meta.get("level"),
        "law_role": meta.get("law_role"),
        "is_leaf": meta.get("is_leaf"),
        "target_scores": dict(meta.get("target_scores") or {}),
    }


class ManifestoQSentenceDSPyFamily(DSPyFamily):
    """Alternating DSPy family for compact CMP quasi-sentence targets."""

    name: str = "dspy"

    def __init__(self, *, config: ManifestoQSentenceDSPyFamilyConfig) -> None:
        super().__init__(config=config)
        self.config: ManifestoQSentenceDSPyFamilyConfig = config

    def _active_dimensions(self) -> List[str]:
        return _active_target_dimensions(self.config)

    def _g_signature(self):
        import dspy

        instructions = (
            "Generate a compact Manifesto Project CMP policy state for "
            "quasi-sentence distillation. Inputs are either raw quasi-sentence "
            "text spans or child CMP policy states. Return only valid JSON. The "
            "JSON must contain `cmp_state.compact_targets` with numeric [0,1] "
            "keys `rile`, `domain_1`, `domain_2`, `domain_3`, `domain_4`, "
            "`domain_5`, `domain_6`, and `domain_7`. `rile` is normalized from "
            "raw RILE [-100,100] to [0,1]; `domain_i` is the share of "
            "non-headline quasi-sentences whose CMP code starts with digit i. "
            "Do not rename, redefine, or describe the domains with invented "
            "labels. Include `total_non_header`, `left_count`, `right_count`, "
            "and `top_codes` when inferable so later merges can aggregate "
            "weighted counts. Do not invent policy evidence not supported by "
            "the input. If the span or child states carry NO scorable policy "
            "content, do not fabricate scores: emit the neutral state "
            "`{\"cmp_state\": {\"compact_targets\": {\"rile\": 0.5, \"domain_1\": 0, "
            "\"domain_2\": 0, \"domain_3\": 0, \"domain_4\": 0, \"domain_5\": 0, "
            "\"domain_6\": 0, \"domain_7\": 0}}}` (equivalently set "
            "`not_relevant: true`)."
        )

        class CTreePOQSentenceGSignature(dspy.Signature):
            __doc__ = instructions

            prompt: str = dspy.InputField(desc="Raw quasi-sentence span or child states")
            completion: str = dspy.OutputField(
                desc=(
                    "Valid JSON CMP state with cmp_state.compact_targets keys: "
                    "rile, domain_1, domain_2, domain_3, domain_4, domain_5, "
                    "domain_6, domain_7"
                )
            )

        return CTreePOQSentenceGSignature

    def _new_target_scorer(self, *, max_output_tokens: Optional[int] = None) -> Any:
        return ManifestoQSentenceTargetScorer(
            max_output_tokens=max_output_tokens or int(self.config.max_completion_tokens)
        )

    def _default_f_init_path(self) -> Optional[Path]:
        if self.config.f_init_path is None or not str(self.config.f_init_path):
            return None
        return Path(str(self.config.f_init_path))

    def _load_f_program(self, artifact: Any) -> Any:
        if artifact == self.TEACHER_PASSTHROUGH:
            return self.TEACHER_PASSTHROUGH
        if artifact in (None, "identity"):
            scorer = self._new_target_scorer()
            default_path = self._default_f_init_path()
            if default_path is not None and default_path.exists():
                if default_path.is_dir() and (default_path / "program.pkl").exists():
                    import dspy

                    return dspy.load(str(default_path))
                try:
                    scorer.load(str(default_path))
                    LOGGER.info("Loaded q-sentence compact scorer from %s", default_path)
                except Exception as exc:
                    LOGGER.warning(
                        "Failed to load q-sentence compact scorer from %s: %s; "
                        "using a bare scorer",
                        default_path,
                        exc,
                    )
            return scorer
        path = Path(str(artifact))
        if not path.exists():
            LOGGER.warning(
                "Q-sentence DSPy f artifact %s missing; using a bare scorer", path
            )
            return self._new_target_scorer()
        if path.is_dir() and (path / "program.pkl").exists():
            import dspy

            return dspy.load(str(path))
        scorer = self._new_target_scorer()
        scorer.load(str(path))
        return scorer

    def validate_artifact(self, *, kind: str, artifact: Any) -> None:
        if artifact in (None, "identity", self.TEACHER_PASSTHROUGH):
            return
        kind = str(kind)
        path = Path(str(artifact))
        if not path.exists():
            raise RuntimeError(f"Q-sentence DSPy {kind} artifact does not exist: {path}")
        if kind == "f":
            if path.is_dir():
                if not (path / "program.pkl").exists():
                    raise RuntimeError(
                        f"Q-sentence DSPy f program directory is missing program.pkl: {path}"
                    )
                import dspy

                loaded = dspy.load(str(path))
                if not callable(loaded):
                    raise RuntimeError(f"Q-sentence DSPy f program is not callable: {path}")
                return
            scorer = self._new_target_scorer()
            scorer.load(str(path))
            if not callable(getattr(scorer, "predictor", None)):
                raise RuntimeError(f"Q-sentence DSPy f state is not callable: {path}")
            return
        if kind == "g":
            return super().validate_artifact(kind=kind, artifact=artifact)
        raise ValueError(f"unknown DSPy artifact kind: {kind!r}")

    def _apply_f_scores(self, f_program: Any, *, response: str) -> Dict[str, float]:
        if f_program == self.TEACHER_PASSTHROUGH:
            return {}
        self._assert_lm_input_budget(
            label="q-sentence f inference",
            fields={"summary": str(response or "")},
        )
        import dspy

        lm = self._ensure_lm()
        try:
            with dspy.context(lm=lm, max_errors=1, provide_traceback=True):
                result = f_program(summary=str(response or ""))
            return _prediction_scores(result)
        except Exception as exc:
            raise RuntimeError("Q-sentence compact target scorer call failed") from exc

    def _leaf_prompt(self, node: LabeledNode) -> str:
        return (
            "Convert this Manifesto Project quasi-sentence span into a compact "
            "CMP policy state. Preserve the CMP code signal, RILE direction, "
            "and domain salience.\n\nSPAN:\n"
            f"{str(node.text or '')}"
        )

    def _merge_prompt(self, *, left_state: str, right_state: Optional[str]) -> str:
        if right_state is None:
            return (
                "Promote this only child CMP policy state as the parent state. "
                "Do not duplicate its counts or salience.\n\nCHILD_STATE:\n"
                f"{left_state}"
            )
        return (
            "Merge these two child CMP policy states into one compact parent "
            "state. Preserve aggregate RILE direction and CMP domain salience.\n\n"
            f"LEFT_STATE:\n{left_state}\n\nRIGHT_STATE:\n{right_state}"
        )

    def _child_state_text(
        self,
        child: Optional[LabeledNode],
        *,
        state_override: Optional[Mapping[str, str]],
    ) -> str:
        """Gold child summary, or a g-generated state when scheduled sampling
        selected this child (override map carries node_id -> generated state)."""
        if child is None:
            return ""
        if state_override is not None:
            override = state_override.get(str(child.node_id))
            if override:
                return str(override)
        return _summary_target(child, include_identity_targets=True) or ""

    def _g_prompt_for_node(
        self,
        tree: LabeledTree,
        node: LabeledNode,
        *,
        state_override: Optional[Mapping[str, str]] = None,
    ) -> str:
        if int(node.level) == 0:
            return self._leaf_prompt(node)
        left = tree.get_node(str(node.left_child_id)) if node.left_child_id else None
        right = tree.get_node(str(node.right_child_id)) if node.right_child_id else None
        left_text = self._child_state_text(left, state_override=state_override)
        right_text: Optional[str]
        if right is None or (left is not None and right.node_id == left.node_id):
            right_text = None
        else:
            right_text = self._child_state_text(right, state_override=state_override)
        return self._merge_prompt(left_state=left_text, right_state=right_text)

    def _qsentence_f_records(self, trees: Sequence[LabeledTree]) -> List[Dict[str, Any]]:
        dims = self._active_dimensions()
        records: List[Dict[str, Any]] = []
        for tree in trees:
            split = str((tree.metadata or {}).get("split", "") or "")
            for node in tree.nodes.values():
                summary = _summary_target(
                    node,
                    include_identity_targets=bool(self.config.include_identity_targets),
                )
                if not summary:
                    continue
                scores = _node_target_scores(node)
                if not scores:
                    continue
                records.append(
                    {
                        "summary": summary,
                        "scores_json": _scores_json(scores, dims),
                        "metadata": {
                            "doc_id": tree.doc_id,
                            "node_id": node.node_id,
                            "split": split,
                            "level": int(node.level),
                            "law_role": "leaf_f" if int(node.level) == 0 else "merge_f",
                            "target_scores": {
                                dim: float(scores[dim]) for dim in dims if dim in scores
                            },
                        },
                    }
                )
        return records

    def _scheduled_sampling_rate(self, *, iteration: int) -> float:
        """Effective scheduled-sampling rate for this alternating iteration.

        When neither a ramp start nor a per-iteration ramp is configured, the
        cap (``g_scheduled_sampling_rate``) is used as a FLAT rate. When a ramp
        is configured, the rate climbs from ``rate_start`` by ``ramp_per_iter``
        each iteration, clamped to the cap.
        """
        cfg = self.config
        cap = float(getattr(cfg, "g_scheduled_sampling_rate", 0.0))
        if cap <= 0.0:
            return 0.0
        start = float(getattr(cfg, "g_scheduled_sampling_rate_start", 0.0))
        ramp = float(getattr(cfg, "g_scheduled_sampling_ramp_per_iter", 0.0))
        if start <= 0.0 and ramp <= 0.0:
            return cap  # flat-rate mode: cap is the rate
        rate = start + max(0, int(iteration)) * ramp
        return float(min(cap, max(0.0, rate)))

    def _build_scheduled_override(
        self,
        tree: LabeledTree,
        *,
        generated_states: Mapping[str, str],
        rate: float,
        seed: int,
    ) -> Dict[str, str]:
        """Per-tree map of child node_id -> g-generated state, selecting each node
        independently with probability ``rate`` (seeded, deterministic). Empty
        generated states are skipped so a failed g call falls back to gold."""
        if rate <= 0.0 or not generated_states:
            return {}
        rng = random.Random(int(seed) ^ (hash(str(tree.doc_id)) & 0xFFFFFFFF))
        override: Dict[str, str] = {}
        for node_id, state in generated_states.items():
            if not str(state or "").strip():
                continue
            if rng.random() < rate:
                override[str(node_id)] = str(state)
        return override

    def _qsentence_g_records(
        self,
        trees: Sequence[LabeledTree],
        *,
        g_program: Any = None,
        iteration: int = 0,
    ) -> List[Dict[str, Any]]:
        dims = self._active_dimensions()
        rate = self._scheduled_sampling_rate(iteration=iteration)
        use_scheduled = (
            rate > 0.0
            and g_program is not None
            and g_program not in (self.TEACHER_PASSTHROUGH, self.RAW_CONCAT)
        )
        if use_scheduled:
            LOGGER.info(
                "Scheduled sampling ON for g-training iter %d: rate=%.3f "
                "(merge prompts mix g's own generated child states with gold)",
                int(iteration),
                rate,
            )
        records: List[Dict[str, Any]] = []
        # Scheduled sampling needs g's OWN generated state at every node. Generate
        # them for ALL trees in ONE level-synchronous batched pass (saturates the
        # whole fleet) instead of a per-tree sequential loop, which pinned one GPU
        # and deadlocked on long chains. See _generate_all_node_states_batched.
        generated_by_tree: List[Dict[str, str]] = []
        if use_scheduled:
            generated_by_tree = self._generate_all_node_states_batched(
                g_program=g_program, trees=list(trees)
            )
        for tree_idx, tree in enumerate(trees):
            split = str((tree.metadata or {}).get("split", "") or "")
            override: Mapping[str, str] = {}
            if use_scheduled:
                generated_states = (
                    generated_by_tree[tree_idx]
                    if tree_idx < len(generated_by_tree)
                    else {}
                )
                override = self._build_scheduled_override(
                    tree,
                    generated_states=generated_states,
                    rate=rate,
                    seed=int(self.config.record_sample_seed) * 7919 + tree_idx,
                )
            for node in tree.nodes.values():
                target = _summary_target(
                    node,
                    include_identity_targets=bool(self.config.include_identity_targets),
                )
                if not target:
                    continue
                scores = _node_target_scores(node)
                if not scores:
                    continue
                prompt = self._g_prompt_for_node(
                    tree, node, state_override=override or None
                )
                # Context for the C1/C3a law reward terms (read through f at
                # scoring time). C1 (leaf): g's state should read like the raw
                # span text. C3a (merge): g's merge should read like the f-readout
                # of the child concat (== the merge prompt itself).
                c1_raw_text = str(node.text or "") if int(node.level) == 0 else ""
                c3a_concat = prompt if int(node.level) > 0 else ""
                records.append(
                    {
                        "prompt": prompt,
                        "completion": target,
                        "metadata": {
                            "doc_id": tree.doc_id,
                            "node_id": node.node_id,
                            "split": split,
                            "level": int(node.level),
                            "is_leaf": int(node.level) == 0,
                            "c1_raw_text": c1_raw_text,
                            "c3a_concat": c3a_concat,
                            "scheduled_sampling_rate": float(rate) if use_scheduled else 0.0,
                            "used_generated_children": bool(
                                use_scheduled
                                and (
                                    str(node.left_child_id or "") in override
                                    or str(node.right_child_id or "") in override
                                )
                            ),
                            "target_scores": {
                                dim: float(scores[dim]) for dim in dims if dim in scores
                            },
                            "lopsidedness": _node_lopsidedness(tree, node),
                        },
                    }
                )
        return records

    def _check_qsentence_f_record_budgets(self, records: Sequence[Mapping[str, Any]]) -> None:
        for idx, row in enumerate(records):
            self._assert_lm_input_budget(
                label=f"q-sentence f training record {idx}",
                fields={
                    "summary": str(row.get("summary") or ""),
                    "scores_json": str(row.get("scores_json") or ""),
                },
            )

    def _check_qsentence_g_record_budgets(self, records: Sequence[Mapping[str, Any]]) -> None:
        for idx, row in enumerate(records):
            self._assert_lm_input_budget(
                label=f"q-sentence g training record {idx}",
                fields={
                    "prompt": str(row.get("prompt") or ""),
                    "completion": str(row.get("completion") or ""),
                },
            )

    def _score_vector_reward(
        self,
        *,
        predicted: Mapping[str, Any],
        target: Mapping[str, Any],
        penalize_missing: bool = False,
    ) -> float:
        rewards: List[float] = []
        for dim in self._active_dimensions():
            if dim not in target:
                continue
            if dim not in predicted:
                if penalize_missing:
                    rewards.append(0.0)
                continue
            p = _parse_first_float(predicted.get(dim))
            t = _parse_first_float(target.get(dim))
            if p is None or t is None:
                if penalize_missing:
                    rewards.append(0.0)
                continue
            rewards.append(max(0.0, 1.0 - abs(_clamp01(p) - _clamp01(t))))
        return float(sum(rewards) / len(rewards)) if rewards else 0.0

    def _g_reward_weights(self) -> tuple[float, float]:
        direct = max(0.0, float(self.config.g_direct_parse_reward_weight))
        proxy = max(0.0, float(self.config.g_f_proxy_reward_weight))
        total = direct + proxy
        if total <= 0.0:
            return 1.0, 0.0
        return direct / total, proxy / total

    def _f_readout_agreement(
        self, f_program: Any, *, a: str, b: str
    ) -> Optional[float]:
        """Reward 1 - mean_dim |f(a) - f(b)| in [0,1]; None if a readout is empty.

        The same f-readout discrepancy the auditor uses for the paper's laws,
        turned into a maximize-able reward. Returns None when either side yields
        no scorable readout so the caller can skip (not penalize) the term.
        """
        fa = self._apply_f_scores(f_program, response=str(a or ""))
        fb = self._apply_f_scores(f_program, response=str(b or ""))
        if not fa or not fb:
            return None
        diffs: List[float] = []
        for dim in self._active_dimensions():
            av, bv = fa.get(dim), fb.get(dim)
            pa, pb = _parse_first_float(av), _parse_first_float(bv)
            if pa is None or pb is None:
                continue
            diffs.append(abs(_clamp01(pa) - _clamp01(pb)))
        if not diffs:
            return None
        return float(max(0.0, 1.0 - sum(diffs) / len(diffs)))

    def _law_reward_weights(self) -> Dict[str, float]:
        return {
            "c1": max(0.0, float(getattr(self.config, "g_law_c1_reward_weight", 0.0))),
            "c3a": max(0.0, float(getattr(self.config, "g_law_c3a_reward_weight", 0.0))),
            "c3b": max(0.0, float(getattr(self.config, "g_law_c3b_reward_weight", 0.0))),
        }

    def _score_g_candidate_state(
        self,
        *,
        summary: str,
        target: Mapping[str, Any],
        f_program: Any,
        law_context: Optional[Mapping[str, Any]] = None,
    ) -> float:
        direct_weight, proxy_weight = self._g_reward_weights()
        parsed_state = parse_compact_scores_json(summary)
        # --- C2 idempotence: g's state reads as the gold node target. ---
        direct_reward = self._score_vector_reward(
            predicted=parsed_state,
            target=target,
            penalize_missing=True,
        )
        proxy_reward = 0.0
        if proxy_weight > 0.0:
            proxy_reward = self._score_vector_reward(
                predicted=self._apply_f_scores(f_program, response=summary),
                target=target,
                penalize_missing=True,
            )
        if not parsed_state:
            c2_reward = float(proxy_weight * proxy_reward)
        else:
            c2_reward = float(direct_weight * direct_reward + proxy_weight * proxy_reward)

        # --- C1 / C3a / C3b law reward terms (paper's other three laws). ---
        # Each is an f-readout AGREEMENT (1 - |f(a)-f(b)|), so the convex blend
        # below keeps the score in [0,1]. Terms with no context or empty readout
        # are skipped, and their weight is redistributed to C2 (re-normalized).
        law_weights = self._law_reward_weights()
        active: Dict[str, float] = {}
        ctx = dict(law_context or {})
        summary_text = str(summary or "")
        if parsed_state and (law_weights["c1"] or law_weights["c3a"] or law_weights["c3b"]):
            # C1 sufficiency (leaf): f(g(b)) ~= f(raw span).
            raw = str(ctx.get("c1_raw_text") or "")
            if law_weights["c1"] and raw:
                r = self._f_readout_agreement(f_program, a=summary_text, b=raw)
                if r is not None:
                    active["c1"] = (law_weights["c1"], r)
            # C3a joint faithfulness (merge): f(g(u@v)) ~= f(u@v concat). The
            # concat (child-state concatenation == the merge prompt) is read
            # DIRECTLY through f, NOT re-merged by g (matches the auditor).
            concat = str(ctx.get("c3a_concat") or "")
            if law_weights["c3a"] and concat:
                r = self._f_readout_agreement(f_program, a=summary_text, b=concat)
                if r is not None:
                    active["c3a"] = (law_weights["c3a"], r)
            # C3b compositionality (merge): f(g(u@v)) ~= f(g(g(u)@g(v))).
            # Needs g re-summaries of the children; only when a g_program and the
            # raw child states are supplied in context.
            g_prog = ctx.get("g_program")
            child_concat = str(ctx.get("c3b_child_state_concat") or "")
            if law_weights["c3b"] and g_prog is not None and child_concat:
                try:
                    g_gu_gv = self._apply_g(g_program=g_prog, prompt=child_concat)
                except BaseException:
                    g_gu_gv = ""
                if g_gu_gv:
                    r = self._f_readout_agreement(f_program, a=summary_text, b=g_gu_gv)
                    if r is not None:
                        active["c3b"] = (law_weights["c3b"], r)

        if not active:
            return c2_reward
        # Convex blend: C2 gets weight 1.0, each active law gets its configured
        # weight; normalize so the result stays comparable to the pure-C2 scale.
        total_w = 1.0 + sum(w for w, _ in active.values())
        blended = c2_reward + sum(w * r for w, r in active.values())
        return float(blended / total_w)

    def _cap_qsentence_records(
        self,
        records: Sequence[Dict[str, Any]],
        *,
        role: str,
    ) -> List[Dict[str, Any]]:
        """Apply ``max_train_records`` to q-sentence node records.

        The base family's ``_cap_training_records`` stratifies by ``law_role``
        metadata the q-sentence record builders do not emit, so this family
        uses a seeded uniform sample instead. Without a cap, GEPA's auto
        budget scales with the FULL node count (e.g. ~276K examples / 1.1M
        rollouts on the 140-doc grid at leaf=1).
        """
        raw_cap = self.config.max_train_records
        if raw_cap is None or int(raw_cap) <= 0 or len(records) <= int(raw_cap):
            return list(records)
        rng = random.Random(int(self.config.record_sample_seed) * 2 + (role == "g"))
        sampled = rng.sample(list(records), int(raw_cap))
        LOGGER.info(
            "Capped q-sentence %s records %d -> %d (max_train_records)",
            role,
            len(records),
            len(sampled),
        )
        return sampled

    def _qsentence_training_record_summary(
        self,
        records: Sequence[Mapping[str, Any]],
        *,
        role: str,
        pre_cap_records: Sequence[Mapping[str, Any]],
    ) -> Dict[str, Any]:
        def summarize(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
            by_role: Dict[str, int] = {}
            by_split: Dict[str, int] = {}
            by_level: Dict[str, int] = {}
            doc_ids: set[str] = set()
            for row in rows:
                meta = dict(row.get("metadata") or {})
                doc_id = meta.get("doc_id")
                if doc_id is not None:
                    doc_ids.add(str(doc_id))
                law_role = str(meta.get("law_role") or ("g_node" if role == "g" else "f_node"))
                by_role[law_role] = int(by_role.get(law_role, 0)) + 1
                split = str(meta.get("split") or "unknown")
                by_split[split] = int(by_split.get(split, 0)) + 1
                level = str(meta.get("level") if meta.get("level") is not None else "unknown")
                by_level[level] = int(by_level.get(level, 0)) + 1
            return {
                "count": int(len(rows)),
                "tree_count": int(len(doc_ids)),
                "by_law_role": dict(sorted(by_role.items())),
                "by_split": dict(sorted(by_split.items())),
                "by_level": dict(sorted(by_level.items(), key=lambda item: item[0])),
            }

        pre = summarize(pre_cap_records)
        post = summarize(records)
        cap_value = self.config.max_train_records
        cap = None if cap_value is None or int(cap_value) <= 0 else int(cap_value)
        cap_applied = int(pre["count"]) != int(post["count"])
        inclusion_probability = (
            float(post["count"]) / float(pre["count"]) if int(pre["count"]) > 0 else 0.0
        )
        direct_weight, proxy_weight = self._g_reward_weights()
        return {
            "role": str(role),
            **post,
            "record_cap": {
                "max_train_records": cap,
                "applied": bool(cap_applied),
                "pre_cap_count": int(pre["count"]),
                "post_cap_count": int(post["count"]),
                "pre_cap_by_law_role": pre["by_law_role"],
                "post_cap_by_law_role": post["by_law_role"],
                "sample_seed": int(self.config.record_sample_seed),
                "selection_policy": (
                    "deterministic_uniform_without_replacement_over_qsentence_node_records"
                ),
                "inclusion_probability": float(inclusion_probability),
            },
            "objective": {
                "label_source": "manifesto_qsentence_cmp_annotations_v1",
                "target_dimensions": self._active_dimensions(),
                "target_scale": "normalized_0_1",
                "gold_label_kind": (
                    "observed_gold_g_completion" if role == "g" else "observed_gold_f_target"
                ),
                "g_direct_parse_reward_weight": float(direct_weight) if role == "g" else None,
                "g_f_proxy_reward_weight": float(proxy_weight) if role == "g" else None,
            },
            "local_law_contract": {
                "estimand": "training_subset_only_not_ipw_estimator",
                "target_population": "logical_labeled_tree_nodes_once",
                "sample_unit": "node_record",
                "root_guaranteed": False,
                "propensity_recorded": True,
                "note": (
                    "This metadata describes which gold qsentence node labels were "
                    "used to compile the DSPy program. It is not a corrected or IPW "
                    "local-law objective."
                ),
            },
        }

    def _write_qsentence_training_record_artifacts(
        self,
        *,
        output_dir: Path,
        iteration: int,
        role: str,
        records: Sequence[Mapping[str, Any]],
        pre_cap_records: Sequence[Mapping[str, Any]],
    ) -> None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        summary = self._qsentence_training_record_summary(
            records,
            role=role,
            pre_cap_records=pre_cap_records,
        )
        summary_path = output_dir / f"{role}_qs_training_records_summary_iter_{iteration:02d}.json"
        summary_path.write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        keys_path = output_dir / f"{role}_qs_selected_record_keys_iter_{iteration:02d}.jsonl"
        with keys_path.open("w", encoding="utf-8") as handle:
            for row in records:
                handle.write(json.dumps(_record_key(row), sort_keys=True) + "\n")

    def train_f(
        self,
        *,
        f_init: Any,
        g: Any,
        traces: Sequence[LabeledTree],
        output_dir: Path,
        iteration: int,
    ) -> Any:
        import dspy

        f_program = self._load_f_program(f_init)
        if f_program == self.TEACHER_PASSTHROUGH:
            f_program = self._load_f_program("identity")

        pre_cap_records = self._qsentence_f_records(traces)
        records = self._cap_qsentence_records(pre_cap_records, role="f")
        self._write_qsentence_training_record_artifacts(
            output_dir=output_dir,
            iteration=iteration,
            role="f",
            records=records,
            pre_cap_records=pre_cap_records,
        )
        self._check_qsentence_f_record_budgets(records)
        train_examples = [
            dspy.Example(
                summary=str(row.get("summary") or ""),
                scores_json=str(row.get("scores_json") or "{}"),
            ).with_inputs("summary")
            for row in records
        ]
        output_dir.mkdir(parents=True, exist_ok=True)
        if not train_examples:
            LOGGER.warning("No q-sentence f training examples; saving bare scorer")
            artifact_path = Path(output_dir) / f"f_qsentence_dspy_iter_{iteration:02d}"
            f_program.save(str(artifact_path), save_program=True)
            return str(artifact_path)

        def metric(gold: Any, pred: Any, trace: Any = None, *unused: Any, **kwargs: Any) -> float:
            target = parse_compact_scores_json(getattr(gold, "scores_json", "{}"))
            predicted = _prediction_scores(pred)
            return self._score_vector_reward(
                predicted=predicted,
                target=target,
                penalize_missing=True,
            )

        lm = self._ensure_lm()
        with dspy.context(lm=lm, max_errors=1, provide_traceback=True):
            compiled = self._compile(
                program=f_program,
                metric=metric,
                trainset=train_examples,
                valset=train_examples,
            )
        artifact_path = Path(output_dir) / f"f_qsentence_dspy_iter_{iteration:02d}"
        compiled.save(str(artifact_path), save_program=True)
        return str(artifact_path)

    def train_g(
        self,
        *,
        g_init: Any,
        f: Any,
        traces: Sequence[LabeledTree],
        output_dir: Path,
        iteration: int,
    ) -> Any:
        import dspy

        f_program = self._load_f_program(f)
        if f_program == self.TEACHER_PASSTHROUGH:
            f_program = self._load_f_program("identity")

        g_program = self._load_g_program(g_init)
        if g_program == self.TEACHER_PASSTHROUGH:
            g_program = dspy.Predict(self._g_signature())

        # Scheduled sampling uses the CURRENT g (the program we are training from)
        # to generate the child states it will be fed at eval. Iteration 0 has no
        # prior learned g (passthrough -> bare Predict), so generated children are
        # only used once the rate ramps in on later iterations.
        pre_cap_records = self._qsentence_g_records(
            traces, g_program=g_program, iteration=int(iteration)
        )
        records = self._cap_qsentence_records(pre_cap_records, role="g")
        self._write_qsentence_training_record_artifacts(
            output_dir=output_dir,
            iteration=iteration,
            role="g",
            records=records,
            pre_cap_records=pre_cap_records,
        )
        self._check_qsentence_g_record_budgets(records)
        lop_strength = float(getattr(self.config, "g_lopsidedness_weight_strength", 0.0))
        train_examples = [
            dspy.Example(
                prompt=str(row.get("prompt") or ""),
                completion=str(row.get("completion") or ""),
                target_scores_json=json.dumps(
                    (row.get("metadata") or {}).get("target_scores") or {},
                    sort_keys=True,
                ),
                lopsidedness_weight=_lopsidedness_weight(
                    float((row.get("metadata") or {}).get("lopsidedness") or 0.0),
                    strength=lop_strength,
                ),
                c1_raw_text=str((row.get("metadata") or {}).get("c1_raw_text") or ""),
                c3a_concat=str((row.get("metadata") or {}).get("c3a_concat") or ""),
            ).with_inputs("prompt")
            for row in records
        ]
        law_weights = self._law_reward_weights()
        if any(law_weights.values()):
            LOGGER.info(
                "g law-reward terms ON: C1=%.3f C3a=%.3f C3b=%.3f (plus C2 base)",
                law_weights["c1"], law_weights["c3a"], law_weights["c3b"],
            )
        output_dir.mkdir(parents=True, exist_ok=True)
        if not train_examples:
            LOGGER.warning("No q-sentence g training examples; saving bare g")
            artifact_path = Path(output_dir) / f"g_qsentence_dspy_iter_{iteration:02d}.json"
            g_program.save(str(artifact_path))
            return str(artifact_path)

        def metric(gold: Any, pred: Any, trace: Any = None, *unused: Any, **kwargs: Any) -> float:
            summary = str(getattr(pred, "completion", "") or "")
            if not summary:
                return 0.0
            try:
                target = json.loads(str(getattr(gold, "target_scores_json", "{}") or "{}"))
            except json.JSONDecodeError:
                target = {}
            reward = self._score_g_candidate_state(
                summary=summary,
                target=target,
                f_program=f_program,
                law_context={
                    "c1_raw_text": str(getattr(gold, "c1_raw_text", "") or ""),
                    "c3a_concat": str(getattr(gold, "c3a_concat", "") or ""),
                    # g_program/c3b_child_state_concat omitted: C3b defaults off
                    # (use g_scheduled_sampling_rate for the exposure-bias law).
                },
            )
            # Lopsidedness weighting (C2-calibration emphasis): scale this node's
            # reward by 1 + strength*lopsidedness so deep lopsided merges -- where
            # mass-weighting strictly beats equal-averaging -- dominate the
            # objective instead of being drowned by near-balanced leaf pairs.
            weight = _parse_first_float(getattr(gold, "lopsidedness_weight", 1.0))
            return float(reward * (weight if weight is not None else 1.0))

        lm = self._ensure_lm()
        with dspy.context(lm=lm, max_errors=1, provide_traceback=True):
            compiled = self._compile(
                program=g_program,
                metric=metric,
                trainset=train_examples,
                valset=train_examples,
            )
        artifact_path = Path(output_dir) / f"g_qsentence_dspy_iter_{iteration:02d}.json"
        compiled.save(str(artifact_path))
        return str(artifact_path)

    def _missing_compact_state_dims(self, generated: str) -> List[str]:
        parsed = parse_compact_scores_json(generated)
        dims = self._active_dimensions()
        if not parsed:
            return list(dims)
        missing: List[str] = []
        for dim in dims:
            if _parse_first_float(parsed.get(dim)) is None:
                missing.append(dim)
        return missing

    @staticmethod
    def _preview_text(value: Any, *, limit: int = 500) -> str:
        text = str(value or "").replace("\n", " ").strip()
        if len(text) <= limit:
            return text
        return text[: limit - 3] + "..."

    def _apply_g(self, g_program: Any, *, prompt: str) -> str:
        if g_program in (self.TEACHER_PASSTHROUGH, self.RAW_CONCAT):
            return str(super()._apply_g(g_program, prompt=prompt) or "").strip()

        attempts = max(1, int(getattr(self.config, "g_inference_retries", 0)) + 1)
        fail_fast = bool(getattr(self.config, "fail_fast_on_invalid_g_state", False))
        failures: List[str] = []
        last_exc: Optional[BaseException] = None
        for attempt in range(1, attempts + 1):
            try:
                generated = str(super()._apply_g(g_program, prompt=prompt) or "").strip()
            except Exception as exc:
                last_exc = exc
                failures.append(f"attempt {attempt}: {type(exc).__name__}: {exc}")
                if fail_fast:
                    message = (
                        "Q-sentence g call failed with fail-fast enabled; "
                        f"attempt={attempt}/{attempts}; "
                        f"prompt_preview={self._preview_text(prompt)!r}; "
                        f"failures={failures}"
                    )
                    raise RuntimeError(message) from exc
                if attempt < attempts:
                    LOGGER.warning(
                        "Q-sentence g call failed on attempt %d/%d; retrying: %s",
                        attempt,
                        attempts,
                        exc,
                    )
                    continue
                break

            # f is an LLM readout: it can interpret a non-canonical state string.
            # So a NON-EMPTY g output is acceptable here -- we do not reject it
            # for failing the strict compact-target schema; the f readout will
            # parse intent downstream. Only EMPTY output is a real failure worth
            # retrying. Set ``g_require_canonical_state=True`` to restore the old
            # strict gate (all 8 dims must be structurally parseable).
            require_canonical = bool(getattr(self.config, "g_require_canonical_state", False))
            missing = self._missing_compact_state_dims(generated) if require_canonical else []
            if generated and not missing:
                return generated
            reason = (
                "empty completion"
                if not generated
                else f"missing compact target dimension(s): {', '.join(missing)}"
            )
            failures.append(
                f"attempt {attempt}: {reason}; "
                f"generated_preview={self._preview_text(generated)!r}"
            )
            if fail_fast:
                raise RuntimeError(
                    "Q-sentence g call returned invalid compact state with "
                    f"fail-fast enabled; attempt={attempt}/{attempts}; "
                    f"reason={reason}; prompt_preview={self._preview_text(prompt)!r}; "
                    f"generated_preview={self._preview_text(generated)!r}"
                )
            if attempt < attempts:
                LOGGER.warning(
                    "Q-sentence g call returned invalid compact state on attempt %d/%d; retrying: %s",
                    attempt,
                    attempts,
                    reason,
                )

        message = (
            f"Q-sentence g call produced no valid compact state after {attempts} attempt(s); "
            f"prompt_preview={self._preview_text(prompt)!r}; failures={failures}"
        )
        if last_exc is not None:
            raise RuntimeError(message) from last_exc
        raise RuntimeError(message)

    def _generate_all_node_states(
        self, *, g_program: Any, tree: LabeledTree
    ) -> Dict[str, str]:
        """Run g bottom-up on its OWN generated child states and return the
        full node_id -> generated-state map (eval-time distribution). Used both
        for root extraction and for scheduled-sampling g-training inputs."""
        state_by_node: Dict[str, str] = {}
        for level_ids in getattr(tree, "levels", None) or []:
            for node_id in level_ids:
                node = tree.get_node(str(node_id)) if hasattr(tree, "get_node") else None
                if node is None:
                    continue
                if int(node.level) == 0:
                    prompt = self._leaf_prompt(node)
                else:
                    left = tree.get_node(str(node.left_child_id)) if node.left_child_id else None
                    right = tree.get_node(str(node.right_child_id)) if node.right_child_id else None
                    left_state = (
                        state_by_node.get(str(left.node_id))
                        if left is not None
                        else ""
                    )
                    if right is None or (left is not None and right.node_id == left.node_id):
                        right_state = None
                    else:
                        right_state = state_by_node.get(str(right.node_id), "")
                    prompt = self._merge_prompt(
                        left_state=str(left_state or ""),
                        right_state=right_state,
                    )
                generated = self._apply_g(g_program, prompt=prompt)
                state_by_node[str(node.node_id)] = generated
        return state_by_node

    def _generate_all_node_states_resilient(
        self, *, g_program: Any, tree: LabeledTree
    ) -> Dict[str, str]:
        """Like ``_generate_all_node_states`` but tolerant of invalid g outputs.

        For audit/diagnostic use only: a node whose g call produces no valid
        compact state stores ``""`` (a failed merge) and the chain continues,
        rather than raising. Deployment paths must keep using the strict
        ``_generate_all_node_states``.
        """
        state_by_node: Dict[str, str] = {}
        for level_ids in getattr(tree, "levels", None) or []:
            for node_id in level_ids:
                node = tree.get_node(str(node_id)) if hasattr(tree, "get_node") else None
                if node is None:
                    continue
                if int(node.level) == 0:
                    prompt = self._leaf_prompt(node)
                else:
                    left = tree.get_node(str(node.left_child_id)) if node.left_child_id else None
                    right = tree.get_node(str(node.right_child_id)) if node.right_child_id else None
                    left_state = state_by_node.get(str(left.node_id)) if left is not None else ""
                    if right is None or (left is not None and right.node_id == left.node_id):
                        right_state = None
                    else:
                        right_state = state_by_node.get(str(right.node_id), "")
                    prompt = self._merge_prompt(left_state=str(left_state or ""), right_state=right_state)
                try:
                    generated = self._apply_g(g_program, prompt=prompt)
                except BaseException:
                    # Any g-call failure (adapter parse error, timeout, etc.)
                    # degrades to an empty state -> the chain continues instead
                    # of propagating and deadlocking worker threads.
                    generated = ""
                state_by_node[str(node.node_id)] = generated
        return state_by_node

    def _generate_root_state(self, *, g_program: Any, tree: LabeledTree) -> str:
        root = _root_node(tree)
        if root is None:
            return str(tree.document_text or "")
        if g_program == self.TEACHER_PASSTHROUGH:
            return _summary_target(root, include_identity_targets=True) or _root_text(tree)

        state_by_node = self._generate_all_node_states(g_program=g_program, tree=tree)
        return (
            state_by_node.get(str(root.node_id))
            or _summary_target(root, include_identity_targets=True)
            or _root_text(tree)
        )

    # ------------------------------------------------------------------
    # Local-law violation audit (paper sec:min-framework).
    #
    # All laws are oracle-equivalence checks read THROUGH the f readout:
    # violation := mean_dim |f(a)[dim] - f(b)[dim]|, i.e. discrepancy in the
    # scored state, not in raw strings. Per node:
    #   C1  Sufficiency  (leaf b):        d(f(g(b)),        f(b))
    #   C2  Idempotence  (stored s):      d(f(g(s)),        f(s))
    #   C3a Joint faithfulness (merge):   d(f(u@v),         f(g(u@v)))
    #   C3b Compositionality   (merge):   d(f(g(u@v)),      f(g(g(u)@g(v))))
    # where u,v are the child STATES (gold or learned-g) and u@v is the merge
    # prompt concatenation. C3b is the deployment-relevant merge law; C3a/C3b
    # together are the two-layer split of the paper's 3-term C3 chain.
    # ------------------------------------------------------------------
    def _f_scores_cached(
        self, f_program: Any, *, text: str, cache: Dict[str, Dict[str, float]]
    ) -> Dict[str, float]:
        key = str(text or "")
        hit = cache.get(key)
        if hit is not None:
            return hit
        scores: Dict[str, float] = {}
        if key.strip():
            try:
                raw = self._apply_f_scores(f_program, response=key)
                scores = {str(k): float(v) for k, v in raw.items() if v is not None}
            except Exception:
                # Audit must be fault-tolerant: a failed f readout yields a
                # missing (empty) score, recorded as a None violation, never a crash.
                scores = {}
        cache[key] = scores
        return scores

    def _readout_discrepancy(
        self, a: Mapping[str, float], b: Mapping[str, float], *, dims: Sequence[str]
    ) -> Dict[str, Optional[float]]:
        """Per-dimension |f(a) - f(b)|; None where either readout is missing."""
        out: Dict[str, Optional[float]] = {}
        for dim in dims:
            av = a.get(dim)
            bv = b.get(dim)
            if av is None or bv is None:
                out[dim] = None
            else:
                out[dim] = abs(float(av) - float(bv))
        return out

    def audit_local_law_violations(
        self,
        *,
        f_program: Any,
        g_program: Any,
        tree: LabeledTree,
        use_gold_states: bool,
    ) -> List[Dict[str, Any]]:
        """Per-node C1/C2/C3a/C3b violations for one tree.

        ``use_gold_states=True`` audits the GOLD node summaries (sanity: the
        gold states should be near-equivalent under f, so violations ~0).
        Otherwise it audits the LEARNED-g states actually produced bottom-up.
        Returns one row per node with the laws applicable to its node type.
        """
        dims = self._active_dimensions()
        f_cache: Dict[str, Dict[str, float]] = {}

        def f_of(text: str) -> Dict[str, float]:
            return self._f_scores_cached(f_program, text=text, cache=f_cache)

        # State at each node: gold summary, or learned-g generated state.
        if use_gold_states:
            state_by_node = {
                str(n.node_id): (_summary_target(n, include_identity_targets=True) or "")
                for n in tree.nodes.values()
            }
        else:
            state_by_node = self._generate_all_node_states_resilient(
                g_program=g_program, tree=tree
            )

        invalid_g_calls = {"count": 0}

        def g_of(prompt: str) -> str:
            if g_program == self.TEACHER_PASSTHROUGH:
                # gold-state audit re-summarization is a no-op identity proxy
                return prompt
            # In an audit, an unparseable g output is a DATA POINT (g produced no
            # valid state), not a fatal error. Record it and return "" so the
            # law violation for that node reads as "no readout" rather than
            # crashing the whole audit.
            # Catch broadly: a DSPy adapter parse error, transport timeout, or
            # any LM-call failure must degrade to "no state" (a data point), never
            # propagate (which deadlocked worker threads on hung futures).
            try:
                return self._apply_g(g_program, prompt=prompt)
            except BaseException:
                invalid_g_calls["count"] += 1
                return ""

        leaf_size = int((tree.metadata or {}).get("leaf_qsentences") or 0)
        rows: List[Dict[str, Any]] = []
        for node in tree.nodes.values():
            level = int(node.level)
            node_id = str(node.node_id)
            state = state_by_node.get(node_id, "")
            base = {
                "doc_id": str(tree.doc_id),
                "node_id": node_id,
                "level": level,
                "leaf_qsentences": leaf_size,
                "is_leaf": level == 0,
            }
            if level == 0:
                # C1: d(f(g(b)), f(b)) -- summary reads same as raw span.
                raw = str(node.text or "")
                gb = state if state else g_of(self._leaf_prompt(node))
                rows.append({
                    **base, "law": "C1_sufficiency",
                    "violation": self._readout_discrepancy(f_of(gb), f_of(raw), dims=dims),
                })
                # C2: d(f(g(s)), f(s)) -- idempotence on the stored leaf summary.
                if state:
                    gs = g_of(self._merge_prompt(left_state=state, right_state=None))
                    rows.append({
                        **base, "law": "C2_idempotence",
                        "violation": self._readout_discrepancy(f_of(gs), f_of(state), dims=dims),
                    })
                continue
            # Merge node: gather child states.
            left = tree.get_node(str(node.left_child_id)) if node.left_child_id else None
            right = tree.get_node(str(node.right_child_id)) if node.right_child_id else None
            u = state_by_node.get(str(left.node_id), "") if left is not None else ""
            if right is None or (left is not None and right.node_id == left.node_id):
                v = None
            else:
                v = state_by_node.get(str(right.node_id), "")
            uv_concat = self._merge_prompt(left_state=u, right_state=v)
            g_uv = state if state else g_of(uv_concat)  # g(u@v) == this node's state
            # C3a Joint faithfulness: d(f(u@v), f(g(u@v))).
            rows.append({
                **base, "law": "C3a_joint_faithfulness",
                "violation": self._readout_discrepancy(f_of(uv_concat), f_of(g_uv), dims=dims),
            })
            # C3b Compositionality: d(f(g(u@v)), f(g(g(u)@g(v)))).
            # g(u), g(v) are re-summaries of the child states; merge those.
            g_u = g_of(self._merge_prompt(left_state=u, right_state=None)) if u else ""
            g_v = g_of(self._merge_prompt(left_state=v, right_state=None)) if v else None
            g_gu_gv = g_of(self._merge_prompt(left_state=g_u, right_state=g_v))
            rows.append({
                **base, "law": "C3b_compositionality",
                "violation": self._readout_discrepancy(f_of(g_uv), f_of(g_gu_gv), dims=dims),
            })
        # Diagnostic: how many g calls in THIS tree produced no valid state
        # (g failure rate is itself a depth-dependent collapse signal).
        for r in rows:
            r["tree_invalid_g_calls"] = int(invalid_g_calls["count"])
        return rows

    def _generate_root_states_batched(
        self,
        *,
        g_program: Any,
        trees: Sequence[LabeledTree],
    ) -> List[str]:
        """Compute q-sentence root states for all trees level-synchronously."""
        tree_list = list(trees)
        if not tree_list:
            return []
        if g_program == self.TEACHER_PASSTHROUGH:
            states: List[str] = []
            for tree in tree_list:
                root = _root_node(tree)
                if root is None:
                    states.append(str(tree.document_text or ""))
                else:
                    states.append(
                        _summary_target(root, include_identity_targets=True)
                        or _root_text(tree)
                    )
            return states

        state_by_tree = self._generate_all_node_states_batched(
            g_program=g_program, trees=tree_list
        )
        roots = [_root_node(tree) for tree in tree_list]
        out: List[str] = []
        for tree, root, states in zip(tree_list, roots, state_by_tree, strict=True):
            if root is None:
                out.append(str(tree.document_text or ""))
            else:
                out.append(
                    states.get(str(root.node_id))
                    or _summary_target(root, include_identity_targets=True)
                    or _root_text(tree)
                )
        return out

    def _generate_all_node_states_batched(
        self,
        *,
        g_program: Any,
        trees: Sequence[LabeledTree],
    ) -> List[Dict[str, str]]:
        """Generate EVERY node's g state for all trees level-synchronously.

        Returns one ``{node_id: state}`` dict per input tree. This is the
        canonical multi-tree LM path: nodes WITHIN a tree are walked bottom-up
        (a merge needs its children first, so within-tree order is sequential),
        but the per-level WAVE pools nodes across ALL trees and runs them through
        a ``ThreadPoolExecutor`` (workers=``num_threads``), so the whole fleet
        stays saturated instead of pinning one GPU per tree. Callers that need
        per-node states (scheduled sampling, the g-state dumper) MUST use this
        instead of a ``for tree: _generate_all_node_states(tree)`` loop, which
        serializes trees and idles the fleet.
        """
        tree_list = list(trees)
        if not tree_list:
            return []
        if g_program == self.TEACHER_PASSTHROUGH:
            # Gold/passthrough: no LM calls; return gold summaries per node.
            out_states: List[Dict[str, str]] = []
            for tree in tree_list:
                out_states.append({
                    str(n.node_id): (_summary_target(n, include_identity_targets=True) or "")
                    for n in tree.nodes.values()
                })
            return out_states

        state_by_tree: List[Dict[str, str]] = [{} for _ in tree_list]
        tree_levels = [list(getattr(tree, "levels", None) or []) for tree in tree_list]
        max_height = max((len(levels) for levels in tree_levels), default=0)
        total_nodes = sum(len(level) for levels in tree_levels for level in levels)
        workers = max(1, int(self.config.num_threads or 1))
        LOGGER.info(
            "Q-sentence g eval pass: %d trees, %d nodes, %d levels, workers=%d",
            len(tree_list),
            total_nodes,
            max_height,
            workers,
        )
        if total_nodes <= 0:
            return [str(tree.document_text or "") for tree in tree_list]

        completed = 0
        progress_every = max(1, total_nodes // 20)
        last_progress_log = time.monotonic()

        def log_progress(*, force: bool = False, level_index: int = 0, wave_size: int = 0) -> None:
            nonlocal last_progress_log
            now = time.monotonic()
            if (
                force
                or completed == total_nodes
                or completed % progress_every == 0
                or now - last_progress_log >= 30.0
            ):
                LOGGER.info(
                    "Q-sentence g eval progress: %d/%d nodes (level %d/%d, wave=%d)",
                    completed,
                    total_nodes,
                    level_index + 1,
                    max_height,
                    wave_size,
                )
                last_progress_log = now

        for level_index in range(max_height):
            wave: List[tuple[int, str, str]] = []
            for tree_index, tree in enumerate(tree_list):
                levels = tree_levels[tree_index]
                if level_index >= len(levels):
                    continue
                states = state_by_tree[tree_index]
                for node_id in levels[level_index]:
                    node = tree.get_node(str(node_id)) if hasattr(tree, "get_node") else None
                    if node is None:
                        continue
                    if int(node.level) == 0:
                        prompt = self._leaf_prompt(node)
                    else:
                        left = tree.get_node(str(node.left_child_id)) if node.left_child_id else None
                        right = tree.get_node(str(node.right_child_id)) if node.right_child_id else None
                        left_state = (
                            states.get(str(left.node_id))
                            if left is not None
                            else ""
                        )
                        if right is None or (left is not None and right.node_id == left.node_id):
                            right_state = None
                        else:
                            right_state = states.get(str(right.node_id), "")
                        prompt = self._merge_prompt(
                            left_state=str(left_state or ""),
                            right_state=right_state,
                        )
                    wave.append((tree_index, str(node.node_id), prompt))
            if not wave:
                continue

            def run_wave_item(item: tuple[int, str, str]) -> tuple[int, str, str]:
                tree_index, node_id, prompt = item
                generated = self._apply_g(g_program, prompt=prompt)
                return tree_index, node_id, generated

            if len(wave) == 1 or workers == 1:
                for item in wave:
                    tree_index, node_id, generated = run_wave_item(item)
                    state_by_tree[tree_index][node_id] = generated
                    completed += 1
                    log_progress(level_index=level_index, wave_size=len(wave))
            else:
                pool = ThreadPoolExecutor(max_workers=min(workers, len(wave)))
                shutdown_started = False
                future_to_item = {
                    pool.submit(run_wave_item, item): item
                    for item in wave
                }
                try:
                    for future in as_completed(future_to_item):
                        try:
                            tree_index, node_id, generated = future.result()
                        except Exception:
                            for pending in future_to_item:
                                if pending is not future:
                                    pending.cancel()
                            shutdown_started = True
                            pool.shutdown(wait=False, cancel_futures=True)
                            raise
                        state_by_tree[tree_index][node_id] = generated
                        completed += 1
                        log_progress(level_index=level_index, wave_size=len(wave))
                finally:
                    if not shutdown_started:
                        pool.shutdown(wait=True, cancel_futures=False)
            log_progress(force=True, level_index=level_index, wave_size=len(wave))

        return state_by_tree

    def score_roots_with_f_by_dimension(
        self,
        *,
        f: Any,
        g: Any,
        trees: Sequence[LabeledTree],
    ) -> Dict[str, List[Optional[float]]]:
        f_program = self._load_f_program(f)
        g_program = self._load_g_program(g)
        dims = self._active_dimensions()
        out: Dict[str, List[Optional[float]]] = {dim: [] for dim in dims}

        tree_list = list(trees)
        root_states = self._generate_root_states_batched(g_program=g_program, trees=tree_list)

        def score_tree(tree: LabeledTree, summary: str) -> Dict[str, Optional[float]]:
            target_scores = _tree_target_scores(tree)
            if f_program == self.TEACHER_PASSTHROUGH or not summary:
                return {dim: target_scores.get(dim) for dim in dims}
            predicted = self._apply_f_scores(f_program, response=summary)
            return {dim: predicted.get(dim) for dim in dims}

        max_workers = max(1, min(len(tree_list), int(self.config.num_threads or 1)))
        if max_workers == 1:
            rows = [
                score_tree(tree, root_states[idx])
                for idx, tree in enumerate(tree_list)
            ]
        else:
            with ThreadPoolExecutor(max_workers=max_workers) as pool:
                rows = [None] * len(tree_list)
                future_to_index = {
                    pool.submit(score_tree, tree, root_states[idx]): idx
                    for idx, tree in enumerate(tree_list)
                }
                progress_every = max(1, len(tree_list) // 20)
                for completed, future in enumerate(as_completed(future_to_index), start=1):
                    idx = future_to_index[future]
                    rows[idx] = future.result()
                    if completed == len(tree_list) or completed % progress_every == 0:
                        LOGGER.info(
                            "Q-sentence root eval progress: %d/%d trees",
                            completed,
                            len(tree_list),
                        )
                rows = [row or {} for row in rows]
        for row in rows:
            for dim in dims:
                out[dim].append(row.get(dim))
        return out

    def score_roots_with_f(
        self,
        *,
        f: Any,
        g: Any,
        trees: Sequence[LabeledTree],
    ) -> List[Optional[float]]:
        by_dim = self.score_roots_with_f_by_dimension(f=f, g=g, trees=trees)
        rows: List[Optional[float]] = []
        for idx in range(len(trees)):
            values = [
                float(preds[idx])
                for preds in by_dim.values()
                if idx < len(preds) and preds[idx] is not None
            ]
            rows.append(_mean(values))
        return rows
