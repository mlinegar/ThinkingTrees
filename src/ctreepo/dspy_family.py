"""DSPy backend family for the alternating f/g optimization loop.

In this family, ``f`` and ``g`` are each compiled DSPy programs:
- ``g`` = ``dspy.Predict(CTreePOGSignature)`` compiled via MIPROv2/GEPA/etc.
- ``f`` = ``dspy.Predict(CTreePOFSignature)`` likewise.

Both serialize to JSON on disk; the family's artifact type is the JSON path.

Key alternation detail: when training ``g``, the optimizer's metric is
``f_current(response=candidate_summary).score`` — higher is better. This means
DSPy's optimizer (GEPA/MIPRO) searches prompts for g that maximize the
current student f's score, not string similarity to the teacher summary.
This is what makes alternation alternation rather than parallel fit.

Identity init convention:
- ``f_init = "identity"`` / ``None`` -> use the tree's ``teacher_score_1_7`` as
  the prediction at k=0. Equivalent to "start from teacher f".
- ``g_init = "identity"`` / ``None`` -> use the tree's ``teacher_summary`` as
  the generated summary at k=0. Equivalent to "start from teacher g".

Warmstart from a prior compiled iterate (passing ``dspy.Program.load(prev)``
as ``student=`` to ``optimizer.compile``) is NOT yet wired in this first pass;
the DSPy optimizer API varies across versions and the lift warrants its own
integration check. Left as a followup.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

from src.ctreepo.alternating import FamilyRuntime
from src.ctreepo.fg_arity import check_two_child_lm_budget
from src.tree.labeled import LabeledNode, LabeledTree

LOGGER = logging.getLogger(__name__)


def _root_node(tree: LabeledTree) -> Optional[LabeledNode]:
    levels = getattr(tree, "levels", None) or []
    for level_ids in reversed(levels):
        for node_id in level_ids:
            node = tree.get_node(str(node_id)) if hasattr(tree, "get_node") else None
            if node is not None:
                return node
    return None


def _root_text(tree: LabeledTree) -> str:
    root = _root_node(tree)
    if root is None:
        return ""
    meta = root.metadata or {}
    return str(
        meta.get("teacher_summary")
        or meta.get("target_summary")
        or root.text
        or tree.document_text
        or ""
    )


def _teacher_root_score(tree: LabeledTree) -> Optional[float]:
    root = _root_node(tree)
    if root is not None and root.score is not None:
        try:
            return float(root.score)
        except (TypeError, ValueError):
            pass
    metadata = tree.metadata or {}
    for key in ("teacher_score_1_7", "document_score"):
        val = metadata.get(key)
        if val is not None:
            try:
                return float(val)
            except (TypeError, ValueError):
                continue
    return None


def _parse_first_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    s = str(value)
    m = re.search(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", s)
    if not m:
        return None
    try:
        return float(m.group(0))
    except ValueError:
        return None


def _clamp01(x: float) -> float:
    return max(0.0, min(1.0, float(x)))


@dataclass
class DSPyFamilyConfig:
    """Config for the DSPy alternating family."""

    #: DSPy optimizer to use ("gepa", "mipro", "bootstrap_fewshot", ...).
    optimizer: str = "mipro"
    #: Budget for GEPA/MIPRO's `auto` knob ("light", "medium", "heavy").
    budget: str = "light"
    num_threads: int = 4
    target_min: float = 1.0
    target_max: float = 7.0
    #: LM configuration dict passed to dspy.LM(...).
    lm_config: Dict[str, Any] = field(default_factory=dict)
    #: Include train+val examples even when identity targets would cause
    #: circular supervision (inherited from existing record builders).
    include_identity_targets: bool = False
    #: Size of every leaf in tokens. This is the load-bearing axis of the
    #: size-based restructure: leaves are exactly this many EmbeddingGemma
    #: tokens (except possibly the last leaf of a document). g at merge
    #: time must hold 2 × this many tokens. Set at family construction.
    leaf_size_tokens: int = 512
    #: Total LM context window in tokens. The budget check asserts:
    #:   ``2 * leaf_size_tokens + max_completion_tokens + prompt_template_overhead_tokens``
    #:   ``<= lm_context_window_tokens``.
    #: Default 12000 matches the vLLM server's ``--max-model-len 12000``.
    lm_context_window_tokens: int = 12000
    #: Upper bound on the LM's generated output per call (tokens).
    max_completion_tokens: int = 512
    #: Conservative estimate of DSPy signature template + demo stacking
    #: overhead (tokens) that sits on top of the raw prompt + completion in
    #: the actual LM request. Tune by inspection if MIPRO reports OOC.
    prompt_template_overhead_tokens: int = 1500
    #: Tokenizer used for exact record-level and inference-time budget checks.
    tokenizer_model_path: str = "/mnt/data/models/google/embeddinggemma-300m"
    #: Policy dimension (economic, social, immigration, eu, environment,
    #: decentralization). Drives the rubric / scoring context injected into
    #: the DSPy signatures so the student has teacher-level task framing
    #: even when unoptimized.
    dimension: str = "economic"
    #: Path to a pre-tuned DSPy scorer (``DimensionScorer`` compiled via GEPA
    #: v2). Loaded as ``f_init`` at k=0 so the starting point is teacher-grade
    #: (~0.83 Pearson on Benoit summaries) instead of a bare ``dspy.Predict``.
    #: When ``None``, defaults to ``outputs/phase1_gepa_v2_rank/<dim>/optimized_scorer.json``.
    #: Set to empty string to force a bare scorer (not recommended).
    f_init_path: Optional[str] = None


class DSPyFamily(FamilyRuntime):
    """DSPy alternating family: text-in/text-out g, text-in/scalar-out f.

    Artifact semantics:
    - ``f`` artifact: path to a saved DSPy Program JSON, OR the sentinel string
      ``"teacher_passthrough"`` which reads teacher scores directly from the
      tree at k=0.
    - ``g`` artifact: same pattern; the ``"teacher_passthrough"`` sentinel at
      k=0 routes to the tree's ``teacher_summary`` at the root.
    """

    name: str = "dspy"

    TEACHER_PASSTHROUGH: str = "teacher_passthrough"

    def __init__(self, *, config: DSPyFamilyConfig) -> None:
        self.config = config
        self._lm = None
        self._dspy_cache_configured = False
        # Pure config check: must hold for every iteration, so evaluate once.
        self._check_two_leaf_budget_config()

    # ------------------------------------------------------------------
    # Budget enforcement (two-leaf concatenation invariant)
    # ------------------------------------------------------------------

    def _check_two_leaf_budget_config(self) -> None:
        """Raise immediately if the LM can't hold 2 × leaf_size_tokens of input
        AND emit at least 2 × leaf_size_tokens of output (so g can pass through
        a literal concatenation of two children if it wants to). No-truncation
        invariant expressed as a pure config relationship.
        """
        check_two_child_lm_budget(
            family_name="DSPyFamily",
            leaf_size_tokens=int(self.config.leaf_size_tokens),
            lm_context_window_tokens=int(self.config.lm_context_window_tokens),
            max_completion_tokens=int(self.config.max_completion_tokens),
            prompt_template_overhead_tokens=int(
                self.config.prompt_template_overhead_tokens
            ),
        )

    def _available_input_budget(self) -> int:
        return (
            int(self.config.lm_context_window_tokens)
            - int(self.config.max_completion_tokens)
            - int(self.config.prompt_template_overhead_tokens)
        )

    def _count_tokens(self, text: str) -> int:
        from src.preprocessing.leaf_size_utils import count_tokens

        return int(
            count_tokens(
                str(text or ""),
                model_path=str(self.config.tokenizer_model_path),
            )
        )

    def _assert_lm_input_budget(
        self,
        *,
        label: str,
        fields: Mapping[str, str],
    ) -> None:
        """Hard-error before any DSPy optimizer / LM call that would truncate.

        This is intentionally record-level: the fast config check catches the
        canonical two-leaf case, while this scans the actual prompt/response
        text that DSPy will send or place in demos.
        """
        counts = {
            str(key): self._count_tokens(str(value or ""))
            for key, value in dict(fields).items()
        }
        total = int(sum(counts.values()))
        budget = int(self._available_input_budget())
        if total > budget:
            raise RuntimeError(
                f"DSPy no-truncation guard failed for {label}: actual input "
                f"tokens={total}, available budget={budget} "
                f"(lm_context_window_tokens={self.config.lm_context_window_tokens} "
                f"- max_completion_tokens={self.config.max_completion_tokens} "
                f"- prompt_template_overhead_tokens="
                f"{self.config.prompt_template_overhead_tokens}); "
                f"field_counts={counts}. Reduce leaf_size_tokens, shorten "
                "teacher summaries, or run with a larger verified LM context."
            )

    def _check_training_record_budgets(
        self,
        records: Sequence[Mapping[str, Any]],
        *,
        role: str,
    ) -> None:
        for idx, row in enumerate(records):
            prompt = str(row.get("prompt") or "")
            if role == "f":
                self._assert_lm_input_budget(
                    label=f"f training record {idx}",
                    fields={
                        "prompt": prompt,
                        "response": str(row.get("response") or ""),
                    },
                )
            elif role == "g":
                self._assert_lm_input_budget(
                    label=f"g training record {idx}",
                    fields={
                        "prompt": prompt,
                        "completion": str(row.get("completion") or ""),
                    },
                )
            else:
                raise ValueError(f"unknown DSPy budget role: {role!r}")

    # ------------------------------------------------------------------
    # LM / signature helpers
    # ------------------------------------------------------------------

    def _ensure_lm(self) -> Any:
        import dspy

        if self._lm is not None:
            return self._lm
        if not self.config.lm_config:
            raise ValueError(
                "DSPyFamilyConfig.lm_config must be populated before training; "
                "set model / api_base / api_key via DSPyFamilyConfig(lm_config=...)"
            )
        if not self._dspy_cache_configured:
            # MIPRO/GEPA already parallelize candidate evaluation. DSPy's
            # sqlite-backed disk cache can exhaust file descriptors and lock
            # under long local-vLLM runs, so keep cache state in-process here.
            try:
                dspy.configure_cache(enable_disk_cache=False, enable_memory_cache=True)
            except Exception as exc:
                LOGGER.warning("Failed to configure DSPy cache: %s", exc)
            self._dspy_cache_configured = True
        self._lm = dspy.LM(**dict(self.config.lm_config))
        return self._lm

    def _g_signature(self):
        import dspy

        # Pull the dimension's rubric + scoring context so the g signature
        # carries teacher-grade task instructions even in the unoptimized
        # bare-Predict case. Without this, the instruction was a single
        # sentence — the bootstrap student had no idea what dimension to
        # summarize for, and MIPRO had to rediscover that from scratch.
        from src.tasks.manifesto.dimensions import PolicyDimension, get_preservation_rubric
        from src.tasks.manifesto.scoring_contexts import get_scoring_context

        dim = PolicyDimension(self.config.dimension)
        rubric = str(get_preservation_rubric(dim) or "").strip()
        context = str(get_scoring_context(dim) or "").strip()
        instructions = (
            "Generate a summary of the given political text that preserves all "
            f"information relevant to the {dim.value} dimension of Benoit's 1-7 "
            "policy scale. The summary will later be scored by a separate "
            "scoring model against expert annotations, so it must preserve "
            "every signal that distinguishes high vs low positions on this "
            "dimension.\n\n"
            f"{context}\n\n{rubric}"
        )

        class CTreePOGSignature(dspy.Signature):
            __doc__ = instructions

            prompt: str = dspy.InputField(desc="Input text or child summary/summaries to summarize")
            completion: str = dspy.OutputField(desc="Dimension-preserving summary")

        return CTreePOGSignature

    def _default_f_init_path(self) -> Optional[Path]:
        """Resolve the default GEPA-v2 tuned scorer path for the configured dimension.

        Returns ``None`` if the artifact is absent (forces a bare scorer fallback).
        """
        if self.config.f_init_path is not None:
            if not self.config.f_init_path:
                return None  # explicit empty string = do not load
            return Path(self.config.f_init_path)
        root = Path(__file__).resolve().parent.parent.parent
        candidate = root / "outputs" / "phase1_gepa_v2_rank" / str(self.config.dimension) / "optimized_scorer.json"
        return candidate if candidate.exists() else None

    def _new_dimension_scorer(self, *, max_output_tokens: Optional[int] = None):
        """Instantiate a fresh :class:`DimensionScorer` for the configured dimension."""
        from src.tasks.manifesto.dimension_scorer import DimensionScorer
        from src.tasks.manifesto.dimensions import PolicyDimension, get_dimension

        dim_spec = get_dimension(PolicyDimension(self.config.dimension))
        return DimensionScorer(
            dimension=dim_spec,
            max_output_tokens=max_output_tokens or int(self.config.max_completion_tokens),
        )

    def _load_f_program(self, artifact: Any):
        """Return a loaded ``DimensionScorer``, or the ``TEACHER_PASSTHROUGH`` sentinel.

        Semantics:
        - ``None``, ``"identity"``: load the dimension's GEPA-v2 optimized scorer if
          present; otherwise return a bare ``DimensionScorer``. This is the default
          f_init and makes k=0 a teacher-grade scorer (not a bare ``Predict``).
        - ``TEACHER_PASSTHROUGH``: explicit passthrough — callers read teacher
          root scores from tree metadata. No LM call.
        - path to a scorer JSON: load the scorer from that path.
        """
        if artifact == self.TEACHER_PASSTHROUGH:
            return self.TEACHER_PASSTHROUGH
        if artifact in (None, "identity"):
            scorer = self._new_dimension_scorer()
            default_path = self._default_f_init_path()
            if default_path is not None:
                try:
                    scorer.load(str(default_path))
                    LOGGER.info(
                        "Loaded GEPA-v2 tuned scorer from %s for dimension=%s",
                        default_path, self.config.dimension,
                    )
                except Exception as exc:
                    LOGGER.warning(
                        "Failed to load GEPA scorer from %s: %s; using bare DimensionScorer",
                        default_path, exc,
                    )
            else:
                LOGGER.info(
                    "No pretuned scorer found for dimension=%s; using bare DimensionScorer",
                    self.config.dimension,
                )
            return scorer
        path = Path(str(artifact))
        if not path.exists():
            LOGGER.warning(
                "DSPy f artifact %s missing; falling back to bare DimensionScorer", path
            )
            return self._new_dimension_scorer()
        if path.is_dir() and (path / "program.pkl").exists():
            import dspy

            return dspy.load(str(path))
        scorer = self._new_dimension_scorer()
        try:
            scorer.load(str(path))
        except KeyError as exc:
            LOGGER.warning(
                "DSPy f artifact %s is not compatible with DimensionScorer "
                "state loading (%s); falling back to bare DimensionScorer",
                path,
                exc,
            )
        return scorer

    def _load_g_program(self, artifact: Any):
        if artifact in (None, "identity", self.TEACHER_PASSTHROUGH):
            return self.TEACHER_PASSTHROUGH
        import dspy

        path = Path(str(artifact))
        if not path.exists():
            LOGGER.warning("DSPy g artifact %s missing; falling back to teacher passthrough", path)
            return self.TEACHER_PASSTHROUGH
        program = dspy.Predict(self._g_signature())
        program.load(str(path))
        return program

    def _apply_g(self, g_program: Any, *, prompt: str) -> str:
        """Run g on a prompt and return the generated completion text."""
        if g_program == self.TEACHER_PASSTHROUGH:
            return ""  # sentinel; caller falls back to the tree's teacher_summary
        self._assert_lm_input_budget(
            label="g inference prompt",
            fields={"prompt": prompt},
        )
        import dspy

        lm = self._ensure_lm()
        try:
            with dspy.context(lm=lm):
                result = g_program(prompt=prompt)
            return str(getattr(result, "completion", "") or "")
        except Exception as exc:
            LOGGER.warning("DSPy g call failed: %s", exc)
            return ""

    def _apply_f_normalized(
        self,
        f_program: Any,
        *,
        prompt: str = "",
        response: str,
    ) -> Optional[float]:
        """Score ``response`` with the DimensionScorer; return normalized [0,1].

        ``prompt`` is accepted for backward compatibility but ignored — the
        DimensionScorer carries the dimension's scoring context internally
        (rubric, 1-7 scale, expert framing) via its frozen task_context.
        This is the fix for the previous bug where passing g's long
        "Summarize..." prompt inflated the LM call to 10977 tokens.
        """
        if f_program == self.TEACHER_PASSTHROUGH:
            return None
        # Only the response (the candidate summary) is the variable input here.
        self._assert_lm_input_budget(
            label="f inference (summary only)",
            fields={"summary": response},
        )
        import dspy

        lm = self._ensure_lm()
        try:
            with dspy.context(lm=lm):
                # DimensionScorer returns {"score": float|None, "reasoning": str}.
                # Use the module call path so compiled DSPy programs run with
                # their optimizer-managed state instead of bypassing wrappers.
                result = f_program(summary=response)
            raw = result.get("score") if isinstance(result, dict) else None
        except Exception as exc:
            LOGGER.warning("DimensionScorer call failed: %s", exc)
            return None
        if raw is None:
            return None
        # Normalize 1-7 raw score into [0, 1] so the alternating trampoline's
        # scaling pipeline (target_min + span * normalized) produces target-
        # range predictions.
        lo = float(self.config.target_min)
        hi = float(self.config.target_max)
        span = max(1e-9, hi - lo)
        return _clamp01((float(raw) - lo) / span)

    # ------------------------------------------------------------------
    # Record building (mirrors existing helpers in the grid script)
    # ------------------------------------------------------------------

    def _g_records(self, trees: Sequence[LabeledTree]) -> List[Dict[str, Any]]:
        from src.ctreepo.distillation import build_g_sft_records

        return build_g_sft_records(
            list(trees),
            include_identity_targets=self.config.include_identity_targets,
        )

    def _f_records(self, trees: Sequence[LabeledTree]) -> List[Dict[str, Any]]:
        from src.ctreepo.distillation import build_f_lm_regression_records

        return build_f_lm_regression_records(
            list(trees),
            include_identity_targets=self.config.include_identity_targets,
            target_min=float(self.config.target_min),
            target_max=float(self.config.target_max),
        )

    # ------------------------------------------------------------------
    # Optimization
    # ------------------------------------------------------------------

    def _compile(
        self,
        *,
        program: Any,
        metric: Callable[..., float],
        trainset: Sequence[Any],
        valset: Sequence[Any],
    ) -> Any:
        import dspy

        optimizer_name = self.config.optimizer.strip().lower()
        if optimizer_name == "gepa":
            optimizer = dspy.GEPA(
                metric=metric,
                reflection_lm=dspy.settings.lm,
                auto=self.config.budget,
                num_threads=int(self.config.num_threads),
                use_wandb=False,
                use_mlflow=False,
            )
        elif optimizer_name == "mipro":
            optimizer = dspy.MIPROv2(
                metric=metric,
                auto=self.config.budget,
                num_threads=int(self.config.num_threads),
            )
        elif optimizer_name in {"bootstrap_fewshot", "bootstrap"}:
            # Cap demo stacking to keep context under budget. DSPy's default
            # of max_bootstrapped_demos=4 stacks large demo (prompt, response,
            # score) tuples — for merge-level records, each demo alone can be
            # >1k tokens and four of them overflow a 12k context window.
            optimizer = dspy.BootstrapFewShot(
                metric=metric,
                max_bootstrapped_demos=2,
                max_labeled_demos=2,
            )
        else:
            raise ValueError(f"unsupported DSPy optimizer: {self.config.optimizer!r}")
        try:
            return optimizer.compile(program, trainset=list(trainset), valset=list(valset))
        except TypeError:
            return optimizer.compile(program, trainset=list(trainset))

    # ------------------------------------------------------------------
    # FamilyRuntime protocol
    # ------------------------------------------------------------------

    def train_f(
        self,
        *,
        f_init: Any,
        g: Any,
        traces: Sequence[LabeledTree],
        output_dir: Path,
        iteration: int,
    ) -> Any:
        """**Strengthen** the current f via MIPRO/GEPA over its existing state.

        Warmstart invariant (see feedback_never_reset_between_rungs.md):
        the current ``f_init`` (which is a ``DimensionScorer`` — typically
        already loaded with GEPA-v2 state for the configured dimension) is
        passed as the ``program`` to ``optimizer.compile(...)`` so each rung
        refines the previous, never resets it.

        Supervision records come from ``build_f_lm_regression_records`` —
        they carry ``response`` = the node's summary and ``score`` = the
        teacher's 1-7 score normalized to [0, 1].

        Metric: rewards the predicted score matching the target score —
        ``1 - |pred - target|``. This is a distillation-style metric that
        correctly rewards fidelity, not high absolute scores.
        """
        import dspy

        # Warmstart: load the current f program (DimensionScorer) so the
        # optimizer refines IT rather than starting from a fresh bare Predict.
        f_program = self._load_f_program(f_init)
        if f_program == self.TEACHER_PASSTHROUGH:
            # Upgrade passthrough to a loaded DimensionScorer so we have
            # something to optimize.
            f_program = self._load_f_program("identity")

        records = self._f_records(traces)
        self._check_training_record_budgets(records, role="f")
        train_examples = [
            dspy.Example(
                summary=str(row.get("response") or ""),
                score=str(float(row.get("score", 0.5))),
            ).with_inputs("summary")
            for row in records
        ]
        if not train_examples:
            LOGGER.warning("No f training examples; skipping f compile")
            output_dir.mkdir(parents=True, exist_ok=True)
            path = Path(output_dir) / "f_dspy_noop.json"
            path.write_text("{}\n", encoding="utf-8")
            return str(path)

        def metric(gold: Any, pred: Any, trace: Any = None, *unused: Any, **kwargs: Any) -> float:
            target = _parse_first_float(getattr(gold, "score", None))
            # DimensionScorer.forward returns a dict; dspy Module calls return
            # a Prediction wrapper whose ``.score`` attribute holds the raw
            # 1-7 string. Parse whichever form appears and normalize to [0,1]
            # so the diff is comparable to the target.
            raw_score = getattr(pred, "score", None)
            if raw_score is None and isinstance(pred, dict):
                raw_score = pred.get("score")
            predicted_raw = _parse_first_float(raw_score)
            if target is None or predicted_raw is None:
                return 0.0
            lo = float(self.config.target_min)
            hi = float(self.config.target_max)
            span = max(1e-9, hi - lo)
            predicted_norm = _clamp01((predicted_raw - lo) / span)
            target_norm = _clamp01(target)
            return max(0.0, 1.0 - abs(predicted_norm - target_norm))

        lm = self._ensure_lm()
        with dspy.context(lm=lm):
            compiled = self._compile(
                program=f_program,
                metric=metric,
                trainset=train_examples,
                valset=train_examples,
            )
        output_dir.mkdir(parents=True, exist_ok=True)
        artifact_path = Path(output_dir) / f"f_dspy_iter_{iteration:02d}"
        compiled.save(str(artifact_path), save_program=True)
        return str(artifact_path)

    def validate_artifact(self, *, kind: str, artifact: Any) -> None:
        """Hard-check that a returned artifact can be reloaded by this family."""
        if artifact in (None, "identity", self.TEACHER_PASSTHROUGH):
            return
        kind = str(kind)
        path = Path(str(artifact))
        if not path.exists():
            raise RuntimeError(f"DSPy {kind} artifact does not exist: {path}")
        import dspy

        if path.is_dir():
            if not (path / "program.pkl").exists():
                raise RuntimeError(
                    f"DSPy {kind} program directory is missing program.pkl: {path}"
                )
            program = dspy.load(str(path))
            if kind == "f" and not callable(getattr(program, "predictor", None)):
                raise RuntimeError(
                    f"DSPy f program at {path} does not expose a callable predictor"
                )
            return
        if kind == "f":
            scorer = self._new_dimension_scorer()
            scorer.load(str(path))
            if not callable(getattr(scorer, "predictor", None)):
                raise RuntimeError(
                    f"DSPy f state at {path} does not expose a callable predictor"
                )
            return
        if kind == "g":
            program = dspy.Predict(self._g_signature())
            program.load(str(path))
            return
        raise ValueError(f"unknown DSPy artifact kind: {kind!r}")

    def train_g(
        self,
        *,
        g_init: Any,
        f: Any,
        traces: Sequence[LabeledTree],
        output_dir: Path,
        iteration: int,
    ) -> Any:
        """**Strengthen** the current g using f_current as the scoring judge.

        Two invariants:

        1. **Warmstart** (see feedback_never_reset_between_rungs.md): the
           current ``g_init`` program is the ``program`` arg to
           ``optimizer.compile`` — not a fresh ``Predict``.
        2. **Agreement metric** (not raw f-score): the reward for a candidate
           summary is ``1 - |f_current(candidate) - target| / scale`` where
           ``target`` is the ground-truth score known for the node (teacher
           score on the node's span, or expert score at the root). This
           rewards *fidelity* — g should produce summaries that let f
           correctly recover the known score — not summaries that merely
           make f output a high absolute number (which would just be reward
           hacking).
        """
        import dspy

        f_program = self._load_f_program(f)
        if f_program == self.TEACHER_PASSTHROUGH:
            # Upgrade to the loaded DimensionScorer so we have a real judge.
            f_program = self._load_f_program("identity")

        # Warmstart: load current g.
        g_program = self._load_g_program(g_init)
        if g_program == self.TEACHER_PASSTHROUGH:
            g_program = dspy.Predict(self._g_signature())

        records = self._g_records(traces)
        self._check_training_record_budgets(records, role="g")

        # Build target lookup so the metric can find each example's ground-
        # truth target score by node_id. Record metadata carries the raw
        # teacher score in "target_score_raw"; fall back to 4.0 (dimension
        # midpoint) when absent.
        target_by_node: Dict[str, float] = {}
        for row in records:
            meta = row.get("metadata") or {}
            node_id = str(meta.get("node_id") or "")
            raw = meta.get("target_score_raw")
            if node_id and raw is not None:
                try:
                    target_by_node[node_id] = float(raw)
                except (TypeError, ValueError):
                    pass

        train_examples = [
            dspy.Example(
                prompt=str(row.get("prompt") or ""),
                completion=str(row.get("completion") or ""),
                node_id=str((row.get("metadata") or {}).get("node_id") or ""),
                target_raw=float(
                    (row.get("metadata") or {}).get("target_score_raw") or 4.0
                ),
            ).with_inputs("prompt")
            for row in records
        ]
        if not train_examples:
            LOGGER.warning("No g training examples; skipping g compile")
            output_dir.mkdir(parents=True, exist_ok=True)
            path = Path(output_dir) / "g_dspy_noop.json"
            path.write_text("{}\n", encoding="utf-8")
            return str(path)

        lm = self._ensure_lm()
        lo = float(self.config.target_min)
        hi = float(self.config.target_max)
        span = max(1e-9, hi - lo)

        def metric(gold: Any, pred: Any, trace: Any = None, *unused: Any, **kwargs: Any) -> float:
            summary = str(getattr(pred, "completion", "") or "")
            if not summary:
                return 0.0
            if f_program == self.TEACHER_PASSTHROUGH:
                # Fallback: lexical similarity to teacher's reference summary.
                from difflib import SequenceMatcher

                reference = str(getattr(gold, "completion", "") or "")
                if not reference:
                    return 0.0
                return float(SequenceMatcher(None, reference, summary).ratio())

            # Ground-truth target for this example (raw 1-7 scale).
            target_raw = _parse_first_float(getattr(gold, "target_raw", None))
            if target_raw is None:
                node_id = str(getattr(gold, "node_id", "") or "")
                target_raw = target_by_node.get(node_id)
            if target_raw is None:
                target_raw = (lo + hi) / 2.0  # neutral midpoint if unknown
            target_norm = _clamp01((float(target_raw) - lo) / span)

            # Score the candidate summary with the current f. _apply_f_normalized
            # already normalizes DimensionScorer's 1-7 output to [0, 1].
            predicted_norm = self._apply_f_normalized(
                f_program, response=summary
            )
            if predicted_norm is None:
                return 0.0
            # Reward = how close f's score on the candidate comes to the known
            # target. Maximum = 1 (perfect agreement), minimum = 0.
            return max(0.0, 1.0 - abs(float(predicted_norm) - target_norm))

        with dspy.context(lm=lm):
            compiled = self._compile(
                program=g_program,
                metric=metric,
                trainset=train_examples,
                valset=train_examples,
            )
        output_dir.mkdir(parents=True, exist_ok=True)
        artifact_path = Path(output_dir) / f"g_dspy_iter_{iteration:02d}.json"
        compiled.save(str(artifact_path))
        return str(artifact_path)

    def score_roots_with_f(
        self,
        *,
        f: Any,
        g: Any,
        trees: Sequence[LabeledTree],
    ) -> List[Optional[float]]:
        """Score each tree's root: apply g to root text, then f to the result.

        Teacher-passthrough sentinels use the tree's existing teacher summary
        and/or teacher score. This makes the k=0 (identity) row meaningful
        without an LM call: predictions equal the teacher's root scores.
        """
        f_program = self._load_f_program(f)
        g_program = self._load_g_program(g)
        span = float(self.config.target_max - self.config.target_min)
        # No silent truncation at inference time: _apply_g and
        # _apply_f_normalized tokenize actual inputs before any LM call.
        preds: List[Optional[float]] = []
        for tree in trees:
            root_prompt = _root_text(tree)
            if g_program == self.TEACHER_PASSTHROUGH or not root_prompt:
                summary = root_prompt
            else:
                generated = self._apply_g(g_program, prompt=root_prompt)
                summary = generated or root_prompt
            if f_program == self.TEACHER_PASSTHROUGH or not summary:
                preds.append(_teacher_root_score(tree))
                continue
            normalized = self._apply_f_normalized(
                f_program, prompt=root_prompt, response=summary
            )
            if normalized is None:
                preds.append(None)
            else:
                preds.append(self.config.target_min + span * float(normalized))
        return preds
