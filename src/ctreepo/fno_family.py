"""FNO backend family for the alternating f/g optimization loop.

In this family, f and g share a single ``EmbeddingCoordinateFNOTreeRegressor``
instance because the model's parameters partition cleanly: ``leaf_fno +
leaf_norm + score_head`` are the f-path, ``merge_fno`` is the g-path. The
``FArtifact`` / ``GArtifact`` returned to the alternating trampoline is the
same underlying state_dict at each step — we just toggle which half trains
via ``freeze_for_{f,g}_training()``.

Supervision:
- ``train_f``: MSE between student f-predictions and teacher f-scores at every
  node (per-node supervision, unchanged from the legacy FNO trainer).
- ``train_g``: MSE between student-f applied to student-g-merged state and
  teacher f-scores at merge nodes. The scoring signal flows through the
  frozen current student f exactly as the alternating-semantics rule requires.

Identity init (``EmbeddingCoordinateFNOTreeRegressor.initialize_as_identity``)
makes the k=0 (``fg``) iteration a neutral baseline: every prediction is 4.0
(the 1-7 midpoint), consistent with ``leaf_fno = id`` and ``merge = avg``.
"""

from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F

from treepo.training.local_law import (
    local_law_objective_from_losses,
    local_law_objective_target_mse,
)

from src.ctreepo.alternating import FamilyRuntime
from src.ctreepo.embedding_fno import (
    EmbeddingCoordinateFNOTreeRegressor,
    _PreparedTree,
    _forward_tree_states,
    _ordered_nodes,
    _prepare_trees,
)
from src.tree.labeled import LabeledNode, LabeledTree
from src.tree.state_tree import (
    StateTree,
    explicit_oracle_trace_kwargs,
    local_law_trace_metadata,
    state_tree_skeleton_from_labeled_tree,
    state_tree_trace_metrics,
    update_state_tree_node,
    write_state_trees_jsonl,
)

LOGGER = logging.getLogger(__name__)

_FNO_IDENTITY_STATE_SENTINELS = frozenset({"identity", "raw_concat"})


def _is_identity_state_artifact(artifact: Any) -> bool:
    if artifact is None:
        return True
    return str(artifact).strip().lower() in _FNO_IDENTITY_STATE_SENTINELS


def _normalize_01(value: float, *, lo: float, hi: float) -> float:
    span = float(hi - lo)
    if span <= 0.0:
        return 0.5
    return max(0.0, min(1.0, (float(value) - float(lo)) / span))


def _denormalize(value: float, *, lo: float, hi: float) -> float:
    return float(lo) + max(0.0, min(1.0, float(value))) * float(hi - lo)


@dataclass
class FNOFamilyConfig:
    hidden_channels: int = 32
    n_modes: int = 64
    n_layers: int = 2
    head_hidden_dim: int = 64
    target_min: float = 1.0
    target_max: float = 7.0
    #: Merge baseline mode for g. "mean" (default) anchors g to 0.5*(left+right)
    #: + FNO residual. "gated" learns a per-dimension gate alpha(left,right) so
    #: the baseline is alpha*left+(1-alpha)*right + residual — lets g ROUTE a
    #: sparse on-topic child's signal instead of averaging it away (targets the
    #: eu-style learned-merge failure). "maxpool" = non-convex per-dim
    #: max(left,right) baseline (no params) so the strongest child signal
    #: survives to the root — for dims whose doc label tracks the MAX on-topic
    #: leaf, not the mean. Invariant preserved: merge(a,a)=a.
    merge_mode: str = "mean"
    merge_gate_hidden_dim: int = 64
    #: Root readout for the DOC-level prediction. "mean_root" (default) =
    #: predict on the composed root state. "topk" = mean of the k highest LEAF
    #: scores. "softmax" = temperature softmax pool over leaf scores. Non-default
    #: modes target dims whose doc label tracks the MAX on-topic leaf (eu:
    #: top1-leaf r=0.79 ~= ceiling) instead of the mean-composed root.
    root_readout: str = "mean_root"
    root_readout_k: int = 1
    root_readout_attn_temp: float = 0.2
    #: Learned "extent" latent (mass-aware general g). When True, every node state
    #: carries an extra scalar coordinate the merge gate reads, so g can weight
    #: children by information density instead of being structurally mass-blind.
    #: Requires merge_mode="gated". The extent is a FREE latent (laws-only: nothing
    #: supervises it against the true text mass). See embedding_fno.py merge/extent.
    extent_enabled: bool = False
    #: Init of the extent-merge head when extent_enabled. "additive" = parent extent
    #: warm-starts at m_l+m_r (the mass-weighted prior basin); "neutral" = parent
    #: extent starts at 0 (the constant-extent / equal-averaging collapse basin).
    extent_merge_init: str = "neutral"
    #: g-loss reweighting: multiply each merge node's loss weight by
    #: 1 + strength*(depth_norm * lopsidedness), concentrating the gradient on the
    #: deep, lopsided merges where mass-weighting strictly beats equal-averaging and
    #: where the extent latent is identifiable. 0.0 = flat (legacy). lopsidedness =
    #: |m_l-m_r|/(m_l+m_r) from child total_non_header; depth_norm = level/max_level.
    g_depth_lopsided_strength: float = 0.0
    #: f-null-space salience law (no explicit merge weight). For each binary merge,
    #: a child's IMPACT is the leave-one-out readout change |f(parent) - f(sibling)|
    #: (parent-without-child = the other child). Low-impact children are penalized
    #: for carrying f-visible signal: weight*(1-impact)*(f(child)-f_neutral)^2, so
    #: negligible content is pushed into f's null space (reads neutral) and an
    #: ADDITIVE/free merge ignores it automatically — salience from geometry, not a
    #: coefficient. Pairs with merge_mode="mlp" (non-convex, salient child can
    #: dominate). Trained in the f phase (leaf encoder + readout unfrozen) so the
    #: gradient reshapes LEAF GEOMETRY; the merge is measured but frozen during f.
    #: 0.0 = off. See feedback_fno_extent_latent... (why explicit weight estimation
    #: failed) and the deregulation reframe (ideology/salience != mass).
    g_null_space_weight: float = 0.0
    #: Lean A2 merge-consistency law (the PRINCIPLED objective, replacing the
    #: null-space surrogate). A2: D f*(u.v, g(g u . g v)) = 0, i.e. the parent's
    #: reading equals the merge of the child readings, THROUGH f. Trained as
    #: |f(parent_state) - <child-merge>| where the merge is:
    #:   "state"   -> M implicit = f(merge_state(s_l,s_r))  (A2-direct; commutativity
    #:                from the symmetrized merge; associativity via g_assoc_weight).
    #:   "readout" -> M(f(l),f(r)) = phi^{-1}(phi f l + phi f r) on the scalar
    #:                readouts (A3-literal; assoc+comm BY CONSTRUCTION, Aczel form).
    #: 0.0 = off. Pair with merge_mode="mlp".
    #: Lambda = the canonical ObjectiveSpec convex split (root_share = 1 - Lambda).
    #: BOTH the f and g objectives are
    #:   (1 - Lambda)*rootLoss + Lambda*Sum_{non-root v} gamma^depth(v)*w_v*l_v,
    #: i.e. (1 - Lambda) on directly fitting the doc label at the ROOT and Lambda on
    #: the distributed local laws (f: leaf preservation A1; g: merge preservation A2
    #: -- the merge route vs the independent parent-text read / gold, via the AIPW
    #: corrected loss). Lambda=0 -> root-only (the reference baseline, the setting
    #: where the merge regresses to averaging); Lambda=1 -> pure distributed law.
    local_law_weight: float = 0.5
    #: DEPRECATED (no-op): the g law is now governed by ``local_law_weight`` (the
    #: convex split), not a standalone A2 weight. Kept so existing --fno-g-a2-weight
    #: flags/configs do not crash. Has no effect on the objective.
    g_a2_weight: float = 1.0
    a2_mode: str = "state"  # retained for back-compat; A2 is always the state law
    #: A3 readout-FACTORIZATION projection, SEPARATE from the A2 merge law.
    #: Penalizes (f(merge_state) - M(f(l), f(r)))^2 with M the Aczel phi-form
    #: (assoc+comm by construction). This is the Lean A3 factorization, NOT A2.
    #: 0 = off.
    a3_factorization_weight: float = 0.0
    #: Associativity penalty: |f(m(m(a,b),c)) - f(m(a,m(b,c)))| over sampled 3-leaf
    #: triples (the proven Lean merge_assoc). A separate projection DIAGNOSTIC,
    #: never reported as A2 evidence. 0 = off.
    g_assoc_weight: float = 0.0
    #: Lean depth discount gamma^depth in the canonical objective
    #: (DiscountedTreeMetaObjective): root weight 1, each level x gamma. 1.0 = flat
    #: sum. Applied to BOTH f and g through ``local_law_objective_from_losses``.
    gamma_depth: float = 1.0
    #: Number of epochs per ``train_f`` / ``train_g`` call.
    epochs_per_iteration: int = 8
    batch_size: int = 2
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    grad_clip_norm: float = 1.0
    #: Relative weighting of leaf / merge / root nodes in f-training (single-model MSE).
    root_weight: float = 1.0
    leaf_weight: float = 0.5
    merge_weight: float = 0.5
    #: Balanced-leaf-loss knob. For sparse leaf targets where most leaves sit at
    #: a NEUTRAL value and only a minority are informative, plain MSE drives the
    #: head to that constant neutral (degenerate mean). When > 1.0, leaves whose
    #: NORMALIZED target deviates from ``leaf_pos_neutral`` by more than
    #: ``leaf_pos_threshold`` (the informative minority) get their f-loss weight
    #: multiplied by this factor, counteracting the neutral-class gradient
    #: dominance. Default 1.0 = unchanged. Applies to f-training leaves only.
    #: ``leaf_pos_neutral`` is the neutral target: 0.0 for MPDS sparse categorical
    #: leaves (~90% zeros), 0.5 for Benoit LLM-span leaves (~86% at the 0.5
    #: "neutral/4-of-7" midpoint). Set it to match the data's neutral mode.
    leaf_pos_weight: float = 1.0
    leaf_pos_threshold: float = 1e-6
    leaf_pos_neutral: float = 0.0
    #: If True, run ``initialize_as_identity()`` on first use.
    identity_init: bool = True
    seed: int = 42
    #: Configured leaf size in tokenizer tokens. This is the canonical row axis.
    leaf_size_tokens: int = 512
    #: Embedding model's max input length in tokens. When set, ``_prepare_trees``
    #: chunks a leaf into fixed slots if leaf_size_tokens exceeds this value,
    #: and otherwise hard-errors on any malformed oversized leaf.
    #: Default 2048 matches Google EmbeddingGemma-300m's max_position_embeddings.
    #: Set to ``None`` to disable the no-truncation enforcement (legacy / smoke).
    embedding_max_length_tokens: Optional[int] = 2048
    tokenizer_model_path: str = "/mnt/data/models/google/embeddinggemma-300m"
    #: Optional expected FNO spatial width after any within-leaf concatenation.
    #: For the EmbeddingGemma defaults this is 768. If a future embedding model
    #: has max length smaller than leaf_size_tokens, callers should set this to
    #: ``ceil(leaf_size_tokens / embedding_max_length_tokens) * base_embedding_dim``.
    effective_embedding_dim: Optional[int] = 768
    #: Canonical TreeBundle state-shape contract. ``summary_dim`` is the input
    #: representation width; ``state_dim`` must be wide enough to carry a pure
    #: concatenation of two child summaries/states. The legacy
    #: ``EmbeddingCoordinateFNOTreeRegressor`` still uses averaged
    #: embedding-width states internally, but publication-facing configs must
    #: satisfy this contract before they are accepted.
    summary_dim: Optional[int] = None
    state_dim: Optional[int] = None

    def __post_init__(self) -> None:
        if int(self.leaf_size_tokens) <= 0:
            raise ValueError(
                f"leaf_size_tokens must be positive, got {self.leaf_size_tokens}"
            )
        if (
            self.embedding_max_length_tokens is not None
            and int(self.embedding_max_length_tokens) <= 0
        ):
            raise ValueError(
                "embedding_max_length_tokens must be positive when set, "
                f"got {self.embedding_max_length_tokens}"
            )
        if (
            self.effective_embedding_dim is not None
            and int(self.effective_embedding_dim) <= 0
        ):
            raise ValueError(
                "effective_embedding_dim must be positive when set, "
                f"got {self.effective_embedding_dim}"
            )
        if self.summary_dim is None and self.effective_embedding_dim is not None:
            self.summary_dim = int(self.effective_embedding_dim)
        if self.state_dim is None and self.summary_dim is not None:
            self.state_dim = 2 * int(self.summary_dim)
        if self.summary_dim is not None and int(self.summary_dim) <= 0:
            raise ValueError(f"summary_dim must be positive, got {self.summary_dim}")
        if self.state_dim is not None and int(self.state_dim) <= 0:
            raise ValueError(f"state_dim must be positive, got {self.state_dim}")
        if self.summary_dim is not None and self.state_dim is not None:
            if int(self.state_dim) < 2 * int(self.summary_dim):
                raise ValueError(
                    "state_dim must be at least 2 * summary_dim for canonical "
                    f"TreeBundle/FNO configs, got state_dim={self.state_dim}, "
                    f"summary_dim={self.summary_dim}"
                )

    @property
    def chunks_per_leaf(self) -> int:
        if self.embedding_max_length_tokens is None:
            return 1
        return max(
            1,
            int(
                math.ceil(
                    int(self.leaf_size_tokens) / float(int(self.embedding_max_length_tokens))
                )
            ),
        )


class FNOFamily(FamilyRuntime):
    """Alternating-optimization family using a shared EmbeddingCoordinateFNOTreeRegressor.

    The ``f_init`` and ``g_init`` handed to ``run_alternating_family`` are
    path strings pointing at a saved state_dict (or ``None`` / ``"identity"``
    to trigger identity initialization). After every training iteration we
    write a new state_dict snapshot and return its path as the next artifact.
    """

    name: str = "fno"

    def __init__(
        self,
        *,
        config: FNOFamilyConfig,
        embedding_client: Any,
        device: Optional[torch.device] = None,
    ) -> None:
        self.config = config
        self.embedding_client = embedding_client
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self._model: Optional[EmbeddingCoordinateFNOTreeRegressor] = None
        self._embedding_dim: Optional[int] = None
        self._prepared_cache: Dict[
            Tuple[int, ...], Tuple[List[_PreparedTree], int]
        ] = {}
        self._last_full_tree_traces: List[StateTree[Any, Any]] = []

    # ------------------------------------------------------------------
    # BundleAwareFamilyRuntime protocol
    # ------------------------------------------------------------------

    @property
    def default_f(self) -> str:
        return "identity"

    @property
    def default_g(self) -> str:
        # FNO's merge path is initialized as the identity/average no-op, which
        # is the raw-concat-equivalent default in the shared ladder vocabulary.
        return "raw_concat"

    def expected_bundle(self) -> Mapping[str, Any]:
        expected: Dict[str, Any] = {
            "leaf_unit": "text_token",
            "state_kind": (
                "embedding",
                "embedding_coordinate",
                "embedding_coordinate_fno_state",
            ),
        }
        if self.config.summary_dim is not None:
            expected["summary_dim"] = int(self.config.summary_dim)
        if self.config.state_dim is not None:
            expected["state_dim_min"] = int(self.config.state_dim)
        return expected

    def supported_inits(self) -> Mapping[str, frozenset[str]]:
        return {
            "f": frozenset({"identity", "artifact"}),
            "g": frozenset({"identity", "raw_concat", "artifact"}),
        }

    def resolve_init(self, *, kind: str, spec: str) -> Any:
        axis = str(kind).strip().lower()
        text = str(spec).strip()
        lowered = text.lower()
        if axis not in {"f", "g"}:
            raise ValueError(f"FNO init kind must be 'f' or 'g', got {kind!r}")
        if not text:
            raise ValueError("FNO init spec must be non-empty")
        if lowered.startswith("artifact:"):
            path = text.partition(":")[2].strip()
            if not path:
                raise ValueError(f"FNO artifact init is missing a path: {spec!r}")
            return path
        if lowered == "identity":
            return "identity"
        if axis == "g" and lowered == "raw_concat":
            return "identity"
        supported = sorted(self.supported_inits().get(axis, frozenset()))
        raise ValueError(
            f"FNO {axis}-init {spec!r} is unsupported; expected one of {supported} "
            "or artifact:<path>"
        )

    def share_state_axes(self) -> frozenset[str]:
        return frozenset({"f", "g"})

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _ensure_model(self, embedding_dim: int) -> EmbeddingCoordinateFNOTreeRegressor:
        if self._model is not None and self._embedding_dim == embedding_dim:
            return self._model
        if (
            self.config.effective_embedding_dim is not None
            and int(embedding_dim) != int(self.config.effective_embedding_dim)
        ):
            raise RuntimeError(
                "FNO effective embedding dimension mismatch: prepared trees produced "
                f"D_eff={embedding_dim}, but config expected "
                f"{self.config.effective_embedding_dim}. Check embedding dim, "
                "embedding_max_length_tokens, and chunks_per_leaf."
            )
        torch.manual_seed(int(self.config.seed))
        model = EmbeddingCoordinateFNOTreeRegressor(
            embedding_dim=embedding_dim,
            hidden_channels=self.config.hidden_channels,
            n_modes=self.config.n_modes,
            n_layers=self.config.n_layers,
            head_hidden_dim=self.config.head_hidden_dim,
            target_min=self.config.target_min,
            target_max=self.config.target_max,
            merge_mode=self.config.merge_mode,
            merge_gate_hidden_dim=self.config.merge_gate_hidden_dim,
            root_readout=self.config.root_readout,
            root_readout_k=self.config.root_readout_k,
            root_readout_attn_temp=self.config.root_readout_attn_temp,
            extent_enabled=self.config.extent_enabled,
            extent_merge_init=self.config.extent_merge_init,
        ).to(self.device)
        if self.config.identity_init:
            model.initialize_as_identity()
        self._model = model
        self._embedding_dim = int(embedding_dim)
        return model

    def _load_state(self, artifact: Any) -> None:
        if self._model is None:
            raise RuntimeError("model not initialized yet")
        if _is_identity_state_artifact(artifact):
            if self.config.identity_init:
                self._model.initialize_as_identity()
            return
        path = Path(str(artifact))
        if not path.exists():
            LOGGER.warning("FNO artifact %s missing; keeping current state", path)
            return
        state = torch.load(path, map_location=self.device, weights_only=False)
        # neuralop's FNO.state_dict() injects a non-parameter ``_metadata`` key;
        # drop it so strict loading passes.
        if isinstance(state, dict):
            state = {k: v for k, v in state.items() if k != "_metadata"}
        self._model.load_state_dict(state)

    def _save_state(self, output_dir: Path, tag: str) -> Path:
        assert self._model is not None
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        path = output_dir / f"fno_state_{tag}.pt"
        torch.save(self._model.state_dict(), path)
        return path

    def validate_artifact(self, *, kind: str, artifact: Any) -> None:
        """Hard-check that an FNO state snapshot can be loaded."""
        if _is_identity_state_artifact(artifact):
            return
        path = Path(str(artifact))
        if not path.exists():
            raise RuntimeError(f"FNO {kind} artifact does not exist: {path}")
        state = torch.load(path, map_location=self.device, weights_only=False)
        if not isinstance(state, dict):
            raise RuntimeError(f"FNO {kind} artifact is not a state dict: {path}")
        state = {k: v for k, v in state.items() if k != "_metadata"}
        if self._model is not None:
            self._model.load_state_dict(state)

    def _prepare(self, trees: Sequence[LabeledTree]) -> Tuple[List[_PreparedTree], int]:
        # Key by the identity of each tree, not the wrapping list: callers
        # (e.g. evaluate_iteration's `list(trees)`) pass fresh list objects per
        # call, and a dead list's id() can be recycled, so an id(trees) key
        # both misses the cache on every evaluation (re-embedding the whole
        # split) and risks serving prepared tensors for the wrong trees. The
        # per-tree ids stay valid because each cached _PreparedTree holds a
        # reference to its tree.
        key = tuple(id(tree) for tree in trees)
        if key in self._prepared_cache:
            return self._prepared_cache[key]
        prepared, embedding_dim = _prepare_trees(
            list(trees),
            embedding_client=self.embedding_client,
            embedding_max_tokens=self.config.embedding_max_length_tokens,
            chunks_per_leaf=int(self.config.chunks_per_leaf),
            tokenizer_model_path=str(self.config.tokenizer_model_path),
            enforce_no_truncation=self.config.embedding_max_length_tokens is not None,
        )
        # Keep the prepared leaf-coordinate tensors resident on the FNO device.
        # The same prepared objects are reused across f/g stages and repeated
        # evaluations, so paying the host->device transfer once avoids a copy in
        # every tree forward pass.
        for item in prepared:
            item.leaf_embeddings = item.leaf_embeddings.to(self.device)
        self._prepared_cache[key] = (prepared, int(embedding_dim))
        return prepared, int(embedding_dim)

    # ------------------------------------------------------------------
    # Training loops
    # ------------------------------------------------------------------

    def _train_step_loss_f(
        self,
        model: EmbeddingCoordinateFNOTreeRegressor,
        item: _PreparedTree,
    ) -> torch.Tensor:
        """Leaf-preservation (A1) f loss via the canonical local-law objective.

        Every scored node is an OBSERVED row with proxy==oracle==gold (leaves are the
        text, so there is no proxy/oracle gap here); the canonical objective applies
        the Lean gamma^depth discount and weight-normalizes. The non-default
        root-readout pool and the f-null-space term stay as extra weighted terms.
        """
        lo, hi = self.config.target_min, self.config.target_max
        states = _forward_tree_states(model, item, device=self.device)
        # Collect all scored-node states; evaluate predict + the canonical objective
        # in ONE batched pass (no per-node kernel launches / autograd blowup).
        node_states: List[torch.Tensor] = []
        targets: List[float] = []
        weights: List[float] = []
        levels: List[float] = []
        is_root_flags: List[float] = []
        max_level = max((int(lv_i) for lv_i, _ in enumerate(item.tree.levels or [])), default=0)
        # When a non-default root readout is active, the root term is computed from
        # the LEAF score distribution (top-k / softmax pool) rather than the
        # composed root state — handled separately below so f trains to make the
        # top leaves accurate, not the mean-composed root.
        use_leaf_readout = self.config.root_readout != "mean_root"
        root_readout_term: Optional[torch.Tensor] = None
        for node_id in item.node_order:
            node = item.tree.get_node(node_id)
            if node is None or node.score is None:
                continue
            is_root = str(node_id) == str(item.root_node_id)
            is_leaf = int(node.level) == 0
            weight = (
                self.config.root_weight
                if is_root
                else (self.config.leaf_weight if is_leaf else self.config.merge_weight)
            )
            if weight <= 0.0:
                continue
            if is_root and use_leaf_readout:
                leaf_ids = list(item.tree.levels[0] if item.tree.levels else [])
                leaf_states = torch.cat([states[str(lid)] for lid in leaf_ids], dim=0)
                root_pred = model.predict_root_topk(
                    leaf_states,
                    mode=self.config.root_readout,
                    k=self.config.root_readout_k,
                    attn_temp=self.config.root_readout_attn_temp,
                )
                root_tgt = _normalize_01(float(node.score), lo=lo, hi=hi)
                root_readout_term = float(weight) * (root_pred - root_tgt) ** 2
                continue
            target_norm = _normalize_01(float(node.score), lo=lo, hi=hi)
            # Balanced leaf loss: upweight the rare INFORMATIVE leaves (those that
            # deviate from the neutral target) so the neutral majority can't drag
            # the head to a constant predictor. leaf_pos_neutral=0.0 recovers the
            # old "nonzero" rule (MPDS); 0.5 targets Benoit's 0.5-neutral leaves.
            if (
                is_leaf
                and self.config.leaf_pos_weight != 1.0
                and abs(target_norm - self.config.leaf_pos_neutral) > self.config.leaf_pos_threshold
            ):
                weight = weight * self.config.leaf_pos_weight
            node_states.append(states[str(node_id)])
            targets.append(target_norm)
            weights.append(float(weight))
            levels.append(float(node.level))
            is_root_flags.append(1.0 if is_root else 0.0)
        if not node_states:
            if root_readout_term is not None:
                return root_readout_term
            return torch.zeros((), dtype=torch.float32, device=self.device)
        preds = model.predict_normalized(torch.cat(node_states, dim=0))
        target_t = torch.tensor(targets, dtype=torch.float32, device=self.device)
        weight_t = torch.tensor(weights, dtype=torch.float32, device=self.device)
        # Canonical A1 leaf-preservation via the root/law split: the root row is the
        # doc-label fit (weight 1-Lambda); leaves/merges are the distributed law
        # (weight Lambda, gamma^depth). proxy==oracle==gold (leaves ARE the text, so
        # no proxy/oracle gap). Single path with g / other families.
        node_loss = self._root_law_split_objective(
            predictions=preds,
            proxy_targets=target_t,
            oracle_targets=target_t,
            observed=torch.ones_like(target_t),
            levels=torch.tensor(levels, dtype=torch.float32, device=self.device),
            node_weights=weight_t,
            is_root=torch.tensor(is_root_flags, dtype=torch.bool, device=self.device),
            max_level=max_level,
        )
        if root_readout_term is not None:
            # Fold the leaf-readout root term in with the per-node terms (it was one
            # node's contribution; keep the overall scale ~unchanged).
            node_loss = (node_loss * len(node_states) + root_readout_term) / (len(node_states) + 1)
        null_w = float(getattr(self.config, "g_null_space_weight", 0.0) or 0.0)
        if null_w > 0.0:
            # f-null-space salience law: shapes the LEAF ENCODER (unfrozen in f) to
            # push low-impact content into f's null space, so the additive/free merge
            # ignores it without estimating any weight. See _null_space_term.
            node_loss = node_loss + null_w * self._null_space_term(model, states, item)
        return node_loss

    def _train_step_loss_g(
        self,
        model: EmbeddingCoordinateFNOTreeRegressor,
        item: _PreparedTree,
    ) -> torch.Tensor:
        """Canonical g objective: the A2 merge-preservation local-law loss.

        leaf_fno, leaf_norm, and score_head are frozen; the gradient only flows to
        merge_fno, so this optimizes g against current f's reading. The merge-law
        AIPW objective (``_a2_term``) IS the g loss: at observed merges it reduces to
        gold supervision (root-weighted), at unsupervised interiors it enforces
        consistency with the independent parent-text reading. A3 readout
        factorization and the associativity diagnostic are separate weighted
        projections. root_weight >> merge_weight (via node_weight) lets the holistic
        root drive the shared merge operator.
        """
        states = _forward_tree_states(model, item, device=self.device)
        # The merge-preservation law (root/law split via Lambda) IS the g objective.
        loss = self._a2_term(model, states, item)
        a3_w = float(getattr(self.config, "a3_factorization_weight", 0.0) or 0.0)
        if a3_w > 0.0:
            loss = loss + a3_w * self._a3_factorization_term(model, states, item)
        assoc_w = float(getattr(self.config, "g_assoc_weight", 0.0) or 0.0)
        if assoc_w > 0.0:
            loss = loss + assoc_w * self._assoc_term(model, states, item)
        return loss

    def _independent_parent_text_readings(
        self,
        model: EmbeddingCoordinateFNOTreeRegressor,
        item: _PreparedTree,
        node_ids: Sequence[str],
    ) -> torch.Tensor:
        """Read each interior node's OWN text, independent of the merge route.

        This is the LEFT side of the merge-consistency law f*(A.B) = f*(g(A).g(B)):
        the oracle reading of the parent's actual concatenated text A.B, which must
        NOT come from merge(s_l, s_r) (that is the RIGHT side). We approximate
        f*(A.B) by pooling the node's descendant-leaf RAW embeddings (reusing the
        already-cached leaf embeddings, no re-embed) and reading them through the
        SAME leaf encoder + score head as a single "big leaf". At the root this is
        the read-the-whole-doc-through-f path that beats every merge. During
        g-training the leaf encoder is frozen, so this target is constant and the
        gradient flows into the merge to make it agree with the text reading.
        """
        leaf_ids = [str(lid) for lid in (item.tree.levels[0] if item.tree.levels else [])]
        leaf_row = {lid: idx for idx, lid in enumerate(leaf_ids)}

        def _descendant_leaf_rows(nid: str) -> List[int]:
            node = item.tree.get_node(nid)
            if node is None:
                return []
            if int(node.level) == 0:
                row = leaf_row.get(str(nid))
                return [row] if row is not None else []
            rows: List[int] = []
            for child in (node.left_child_id, node.right_child_id):
                if child is not None:
                    rows.extend(_descendant_leaf_rows(str(child)))
            return rows

        pooled: List[torch.Tensor] = []
        emb = item.leaf_embeddings.to(self.device)
        for nid in node_ids:
            rows = _descendant_leaf_rows(str(nid))
            if not rows:
                # Fallback: read the node's own leaf embedding if it is a leaf.
                rows = [leaf_row[str(nid)]] if str(nid) in leaf_row else [0]
            idx = torch.tensor(sorted(set(rows)), dtype=torch.long, device=self.device)
            pooled.append(emb.index_select(0, idx).mean(dim=0, keepdim=True))
        pooled_emb = torch.cat(pooled, dim=0)  # (n_nodes, D)
        text_states = model.encode_leaves(pooled_emb)  # (n_nodes, 1, D[+1])
        return model.predict_normalized(text_states)  # (n_nodes,)

    def _root_law_split_objective(
        self,
        *,
        predictions: torch.Tensor,
        proxy_targets: torch.Tensor,
        oracle_targets: torch.Tensor,
        observed: torch.Tensor,
        levels: torch.Tensor,
        node_weights: torch.Tensor,
        is_root: torch.Tensor,
        max_level: int,
    ) -> torch.Tensor:
        """Canonical RootLocalObjective split: (1-Lambda)*rootLoss + Lambda*lawLoss.

        rootLoss = canonical AIPW objective over the ROOT row(s) (doc-label fit).
        lawLoss  = canonical AIPW objective over the NON-root rows, depth-discounted
                   by gamma^depth with the LEAN convention depth = max_level - level
                   (root = depth 0, leaves deepest). Lambda = ``local_law_weight``.
        Each side is built by ``treepo.training.local_law`` -- no bespoke AIPW/depth
        math here. Returns 0 when a side is empty (its share contributes nothing).
        """
        lam = float(self.config.local_law_weight)
        gamma = float(self.config.gamma_depth)
        proxy_loss = (predictions - proxy_targets) ** 2
        oracle_loss = (predictions - oracle_targets) ** 2
        # Lean depth: root = 0, each level below the root adds 1.
        depths = (float(max_level) - levels.to(dtype=torch.float32)).clamp(min=0.0)
        zero = torch.zeros((), dtype=predictions.dtype, device=predictions.device)

        def _obj(mask: torch.Tensor) -> torch.Tensor:
            idx = mask.nonzero(as_tuple=False).reshape(-1)
            if idx.numel() == 0:
                return zero
            return local_law_objective_from_losses(
                proxy_loss=proxy_loss.index_select(0, idx),
                oracle_loss=oracle_loss.index_select(0, idx),
                observed=observed.index_select(0, idx),
                propensity=torch.ones_like(observed).index_select(0, idx),
                depths=depths.index_select(0, idx),
                node_weights=node_weights.index_select(0, idx),
                gamma_depth=gamma,
            )

        root_obj = _obj(is_root)
        law_obj = _obj(~is_root)
        return (1.0 - lam) * root_obj + lam * law_obj

    def _a2_term(
        self,
        model: EmbeddingCoordinateFNOTreeRegressor,
        states: Dict[str, torch.Tensor],
        item: _PreparedTree,
    ) -> torch.Tensor:
        """A2 merge-preservation law as the canonical root/law-split objective.

        The law is f*(A.B) = f*(g(A).g(B)), measured through f. Per merge node v with
        children (A, B):
          prediction = f(merge(s_A, s_B))            # RIGHT side, the merge route
          proxy      = detached f*(A.B) text read    # LEFT side, independent of merge
          oracle     = gold(v) when v is observed    # the true label for A.B
          observed   = 1 if v has a score else 0
        These rows go through ``_root_law_split_objective``: the ROOT merge is the
        doc-label fit (weight 1-Lambda), interior merges are the distributed law
        (weight Lambda, gamma^depth). corrected = proxy + R/pi*(oracle - proxy): the
        observed root reduces to gold supervision; unsupervised interiors reduce to
        consistency with the independent parent-text reading. (The previous impl
        compared f(parent_state) to f(merge(l,r)); since parent_state IS merge(l,r)
        that was identically zero -- a no-op. This is the real law.) Returns 0 when
        there are no binary merge nodes.
        """
        lo, hi = self.config.target_min, self.config.target_max
        lop_strength = float(getattr(self.config, "g_depth_lopsided_strength", 0.0) or 0.0)
        max_level = max((int(lv_i) for lv_i, _ in enumerate(item.tree.levels or [])), default=0)
        _node_lopsidedness = None
        if lop_strength > 0.0:
            from src.ctreepo.manifesto_qsentence_dspy_family import (
                _node_lopsidedness as _nl,
            )

            _node_lopsidedness = _nl
        parent_ids: List[str] = []
        preds: List[torch.Tensor] = []
        observed: List[float] = []
        oracle: List[float] = []
        levels: List[float] = []
        weights: List[float] = []
        is_root: List[float] = []
        for node_id in item.node_order:
            node = item.tree.get_node(node_id)
            if node is None or int(node.level) == 0:
                continue
            lid = str(node.left_child_id) if node.left_child_id else None
            rid = str(node.right_child_id) if node.right_child_id else None
            if not (lid and rid and lid != rid and lid in states and rid in states):
                continue
            if str(node_id) not in states:
                continue
            root_node = str(node_id) == str(item.root_node_id)
            weight = self.config.root_weight if root_node else self.config.merge_weight
            if weight <= 0.0:
                continue
            if _node_lopsidedness is not None:
                depth_norm = (int(node.level) / max_level) if max_level > 0 else 0.0
                lop = float(_node_lopsidedness(item.tree, node))
                weight *= 1.0 + lop_strength * depth_norm * lop
            # RIGHT side: f(merge(s_l, s_r)). states[node] IS that merge route.
            preds.append(states[str(node_id)])
            has_score = node.score is not None
            observed.append(1.0 if has_score else 0.0)
            oracle.append(_normalize_01(float(node.score), lo=lo, hi=hi) if has_score else 0.0)
            levels.append(float(node.level))
            weights.append(float(weight))
            is_root.append(1.0 if root_node else 0.0)
            parent_ids.append(str(node_id))
        if not parent_ids:
            return torch.zeros((), dtype=torch.float32, device=self.device)
        pred = model.predict_normalized(torch.cat(preds, dim=0))
        # LEFT side proxy: detached independent parent-text reading f*(A.B).
        proxy = self._independent_parent_text_readings(model, item, parent_ids).detach()
        dev = self.device
        return self._root_law_split_objective(
            predictions=pred,
            proxy_targets=proxy,
            oracle_targets=torch.tensor(oracle, dtype=torch.float32, device=dev),
            observed=torch.tensor(observed, dtype=torch.float32, device=dev),
            levels=torch.tensor(levels, dtype=torch.float32, device=dev),
            node_weights=torch.tensor(weights, dtype=torch.float32, device=dev),
            is_root=torch.tensor(is_root, dtype=torch.bool, device=dev),
            max_level=max_level,
        )

    def _a3_factorization_term(
        self,
        model: EmbeddingCoordinateFNOTreeRegressor,
        states: Dict[str, torch.Tensor],
        item: _PreparedTree,
    ) -> torch.Tensor:
        """A3 readout-FACTORIZATION projection (separate from A2).

        Lean A3: f*(g(gA.gB)) = M(f*gA, f*gB) for an assoc+comm M. Penalizes
        (f(merge_state) - M(f(l), f(r)))^2 with M the Aczel phi-form
        (``readout_merge``, assoc+comm by construction). This checks the merge route
        FACTORS through the scalar readout; it does NOT reference the parent text, so
        it is a distinct law from A2 (do not report one as the other).
        """
        lefts: List[torch.Tensor] = []
        rights: List[torch.Tensor] = []
        node_ids: List[str] = []
        for node_id in item.node_order:
            node = item.tree.get_node(node_id)
            if node is None or int(node.level) == 0:
                continue
            lid = str(node.left_child_id) if node.left_child_id else None
            rid = str(node.right_child_id) if node.right_child_id else None
            if lid and rid and lid != rid and lid in states and rid in states and str(node_id) in states:
                node_ids.append(str(node_id))
                lefts.append(states[lid])
                rights.append(states[rid])
        if not node_ids:
            return torch.zeros((), dtype=torch.float32, device=self.device)
        l = torch.cat(lefts, dim=0)
        r = torch.cat(rights, dim=0)
        f_merge = model.predict_normalized(model.merge(l, r))
        m = model.readout_merge(model.predict_normalized(l), model.predict_normalized(r))
        return ((f_merge - m) ** 2).mean()

    def _assoc_term(
        self,
        model: EmbeddingCoordinateFNOTreeRegressor,
        states: Dict[str, torch.Tensor],
        item: _PreparedTree,
    ) -> torch.Tensor:
        """Merge associativity projection DIAGNOSTIC (the proven Lean merge_assoc):
        f(m(m(a,b),c)) == f(m(a,m(b,c))). Separate from A2/A3; never A2 evidence."""
        lefts: List[torch.Tensor] = []
        rights: List[torch.Tensor] = []
        for node_id in item.node_order:
            node = item.tree.get_node(node_id)
            if node is None or int(node.level) == 0:
                continue
            lid = str(node.left_child_id) if node.left_child_id else None
            rid = str(node.right_child_id) if node.right_child_id else None
            if lid and rid and lid != rid and lid in states and rid in states:
                lefts.append(states[lid])
                rights.append(states[rid])
        if not lefts:
            return torch.zeros((), dtype=torch.float32, device=self.device)
        l = torch.cat(lefts, dim=0)
        r = torch.cat(rights, dim=0)
        c = torch.cat(lefts[1:] + lefts[:1], dim=0)  # rotate to get a third state
        left_assoc = model.predict_normalized(model.merge(model.merge(l, r), c))
        right_assoc = model.predict_normalized(model.merge(l, model.merge(r, c)))
        return ((left_assoc - right_assoc) ** 2).mean()

    def _null_space_term(
        self,
        model: EmbeddingCoordinateFNOTreeRegressor,
        states: Dict[str, torch.Tensor],
        item: _PreparedTree,
    ) -> torch.Tensor:
        """f-null-space salience law (no explicit merge weight).

        impact(child) = |f(parent) - f(sibling)|  (leave-one-out: parent without a
        child reduces to the OTHER child for a binary merge). A low-impact child is
        penalized for carrying any f-visible signal, pushing negligible content to
        read NEUTRAL (f's null space). f_neutral = f(zeros). All readouts are scalar
        per node (single-dim head), so impact/neutral-gap are scalars in [0,1].

        Trained in f (encoder + readout unfrozen) so the gradient reshapes the LEAF
        ENCODER geometry — the merge is measured but frozen during f. Returns 0 when
        there are no binary merge nodes.
        """
        parents: List[torch.Tensor] = []
        lefts: List[torch.Tensor] = []
        rights: List[torch.Tensor] = []
        for node_id in item.node_order:
            node = item.tree.get_node(node_id)
            if node is None or int(node.level) == 0:
                continue
            lid = str(node.left_child_id) if node.left_child_id else None
            rid = str(node.right_child_id) if node.right_child_id else None
            if lid and rid and lid != rid and lid in states and rid in states and str(node_id) in states:
                parents.append(states[str(node_id)])
                lefts.append(states[lid])
                rights.append(states[rid])
        if not parents:
            return torch.zeros((), dtype=torch.float32, device=self.device)
        p = torch.cat(parents, dim=0)
        l = torch.cat(lefts, dim=0)
        r = torch.cat(rights, dim=0)
        f_p = model.predict_normalized(p)
        f_l = model.predict_normalized(l)
        f_r = model.predict_normalized(r)
        f_neutral = model.predict_normalized(torch.zeros_like(p[:1]))
        # impact: how much removing each child moves the parent reading.
        impact_l = (f_p - f_r).abs().clamp(0.0, 1.0)
        impact_r = (f_p - f_l).abs().clamp(0.0, 1.0)
        # low-impact children pay for reading far from neutral.
        pen_l = (1.0 - impact_l) * (f_l - f_neutral) ** 2
        pen_r = (1.0 - impact_r) * (f_r - f_neutral) ** 2
        return torch.cat([pen_l, pen_r]).mean()

    def _run_training(
        self,
        prepared: Sequence[_PreparedTree],
        *,
        mode: str,
        output_dir: Path,
    ) -> Path:
        model = self._model
        assert model is not None
        if mode == "f":
            model.freeze_for_f_training()
            loss_fn = self._train_step_loss_f
        elif mode == "g":
            model.freeze_for_g_training()
            loss_fn = self._train_step_loss_g
        else:
            raise ValueError(f"mode must be 'f' or 'g', got {mode!r}")

        trainable = [p for p in model.parameters() if p.requires_grad]
        if not trainable:
            LOGGER.warning("No trainable params for mode=%s; skipping", mode)
            model.unfreeze_all()
            return self._save_state(output_dir, f"{mode}_noop")

        # g-training has no gradients at leaf_count=1 (no merge nodes exist).
        # Detect this by counting merge-level nodes across the prepared batch;
        # if none, the training is a no-op by construction.
        if mode == "g":
            merge_node_count = sum(
                1
                for item in prepared
                for node_id in item.node_order
                if (item.tree.get_node(node_id) is not None
                    and int(item.tree.get_node(node_id).level) > 0)
            )
            if merge_node_count == 0:
                LOGGER.info(
                    "No merge nodes in %d trees (leaf_count=1); g-training is a no-op",
                    len(prepared),
                )
                model.unfreeze_all()
                return self._save_state(output_dir, "g_noop_no_merge_nodes")

        optimizer = torch.optim.AdamW(
            trainable,
            lr=float(self.config.learning_rate),
            weight_decay=float(self.config.weight_decay),
        )
        model.train()
        batch_size = max(1, int(self.config.batch_size))
        n = len(prepared)
        train_losses: List[float] = []
        for epoch in range(int(self.config.epochs_per_iteration)):
            order = list(range(n))
            torch.manual_seed(int(self.config.seed) + epoch)
            random_perm = torch.randperm(n).tolist()
            epoch_losses: List[float] = []
            for start in range(0, n, batch_size):
                optimizer.zero_grad()
                batch_losses: List[torch.Tensor] = []
                for idx in random_perm[start : start + batch_size]:
                    per_loss = loss_fn(model, prepared[idx])
                    # Skip per-tree losses that have no autograd graph. For
                    # g-training, a tree with only leaves (1-leaf trees, short
                    # docs) produces a zero tensor with no grad_fn; calling
                    # backward() on such a batch raises. Skip these cases so
                    # mixed batches (some with merges, some without) train
                    # only on the trees that actually contribute gradients.
                    if per_loss.requires_grad and per_loss.grad_fn is not None:
                        batch_losses.append(per_loss)
                if not batch_losses:
                    continue
                loss = torch.stack(batch_losses).mean()
                loss.backward()
                if self.config.grad_clip_norm > 0:
                    torch.nn.utils.clip_grad_norm_(trainable, float(self.config.grad_clip_norm))
                optimizer.step()
                epoch_losses.append(float(loss.detach().cpu().item()))
            if epoch_losses:
                train_losses.append(sum(epoch_losses) / len(epoch_losses))
                LOGGER.debug(
                    "FNO %s epoch %d/%d mean_loss=%.6f",
                    mode, epoch + 1, self.config.epochs_per_iteration, train_losses[-1],
                )
        model.unfreeze_all()

        ckpt_path = self._save_state(output_dir, mode)
        history_path = Path(output_dir) / f"fno_{mode}_training_losses.json"
        history_path.write_text(
            json.dumps({"mode": mode, "epoch_mean_losses": train_losses}, indent=2) + "\n",
            encoding="utf-8",
        )
        return ckpt_path

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
        prepared, embedding_dim = self._prepare(traces)
        self._ensure_model(embedding_dim)
        # Load the state the model should start from. For FNO the combined
        # (f, g) state lives in one checkpoint; prefer g (the most-recent
        # artifact) since f_init may be stale relative to the current g.
        self._load_state(g if not _is_identity_state_artifact(g) else f_init)
        return self._run_training(prepared, mode="f", output_dir=Path(output_dir))

    def train_g(
        self,
        *,
        g_init: Any,
        f: Any,
        traces: Sequence[LabeledTree],
        output_dir: Path,
        iteration: int,
    ) -> Any:
        prepared, embedding_dim = self._prepare(traces)
        self._ensure_model(embedding_dim)
        # Load the most recent f checkpoint so the frozen f-path matches the
        # current student f during g training.
        self._load_state(f if not _is_identity_state_artifact(f) else g_init)
        return self._run_training(prepared, mode="g", output_dir=Path(output_dir))

    @torch.no_grad()
    def full_tree_traces_with_f_g(
        self,
        *,
        f: Any,
        g: Any,
        trees: Sequence[LabeledTree],
    ) -> List[StateTree[Any, Any]]:
        """Return full per-node FNO traces for the supplied trees."""

        prepared, embedding_dim = self._prepare(trees)
        self._ensure_model(embedding_dim)
        # Prefer g as the most recent artifact; both are the same state_dict.
        self._load_state(g if not _is_identity_state_artifact(g) else f)
        model = self._model
        assert model is not None
        model.eval()
        traces_out: List[StateTree[Any, Any]] = []
        for item in prepared:
            states = _forward_tree_states(model, item, device=self.device)
            trace = state_tree_skeleton_from_labeled_tree(
                item.tree,
                method_family="fno",
                state_kind="embedding_coordinate_fno_state",
                split=item.split,
            )
            for node_id in item.node_order:
                node = item.tree.get_node(str(node_id))
                state = states.get(str(node_id))
                if node is None or state is None:
                    continue
                pred_norm = float(
                    model.predict_normalized(state).detach().cpu().reshape(()).item()
                )
                pred_raw = _denormalize(
                    pred_norm, lo=self.config.target_min, hi=self.config.target_max
                )
                is_root = str(node_id) == str(item.root_node_id)
                is_leaf = int(node.level) == 0
                # Unsupervised merges (merge_supervision="none") carry node.score=None:
                # g learns their aggregation freely. Still emit the prediction (root/leaf
                # records and downstream metrics need it), but skip the target/law trace.
                if node.score is None:
                    update_state_tree_node(
                        trace,
                        str(node_id),
                        rendered=str(node.text or ""),
                        state=state.detach().cpu(),
                        metadata={
                            "prediction": float(pred_raw),
                            "readout_prediction": float(pred_raw),
                            "prediction_normalized": float(pred_norm),
                            "target": None,
                        },
                    )
                    continue
                target_raw = float(node.score)
                proxy_loss = float((pred_raw - target_raw) ** 2)
                oracle_kwargs = explicit_oracle_trace_kwargs(getattr(node, "metadata", {}) or {})
                law_metadata = local_law_trace_metadata(
                    prediction=float(pred_raw),
                    proxy_target=float(target_raw),
                    proxy_loss=float(proxy_loss),
                    oracle_target=oracle_kwargs["oracle_target"],
                    oracle_loss=oracle_kwargs["oracle_loss"],
                    observed=bool(oracle_kwargs["observed"]),
                    sampled=bool(oracle_kwargs["sampled"]),
                    propensity=oracle_kwargs["propensity"],
                    law_channel="root" if is_root else ("leaf" if is_leaf else "merge"),
                    state_kind="embedding_coordinate_fno_state",
                    label_source=str(oracle_kwargs["label_source"] or "proxy_score"),
                )
                update_state_tree_node(
                    trace,
                    str(node_id),
                    rendered=str(node.text or ""),
                    state=state.detach().cpu(),
                    metadata={
                        "prediction": float(pred_raw),
                        "readout_prediction": float(pred_raw),
                        "prediction_normalized": float(pred_norm),
                        "target": float(target_raw),
                        **law_metadata,
                    },
                )
            traces_out.append(trace)
        self._last_full_tree_traces = list(traces_out)
        return traces_out

    @torch.no_grad()
    def score_roots_with_f(
        self,
        *,
        f: Any,
        g: Any,
        trees: Sequence[LabeledTree],
    ) -> List[Optional[float]]:
        traces = self.full_tree_traces_with_f_g(f=f, g=g, trees=trees)
        pred_by_doc: Dict[str, Optional[float]] = {}
        for trace in traces:
            root_meta = dict(trace.root.metadata or {})
            doc_id = str(root_meta.get("doc_id", trace.metadata.get("doc_id", "")) or "")
            raw = root_meta.get("prediction")
            try:
                pred_by_doc[doc_id] = None if raw is None else float(raw)
            except (TypeError, ValueError):
                pred_by_doc[doc_id] = None
        return [pred_by_doc.get(str(tree.doc_id)) for tree in trees]

    @torch.no_grad()
    def score_root_text_readings_with_f(
        self,
        *,
        f: Any,
        g: Any,
        trees: Sequence[LabeledTree],
    ) -> List[Optional[float]]:
        """f-only baseline: read the WHOLE doc text directly through f.

        Returns, per tree, f applied to the root's INDEPENDENT text reading f*(A.B)
        (pool of all leaf embeddings through the shared encoder), NOT the composed
        merge root state. This is the standing target the learned merge must beat --
        the merge-consistency law at the root says f(merge_root) should match this.
        Denormalized to the target range to match ``score_roots_with_f``.
        """
        prepared, embedding_dim = self._prepare(trees)
        self._ensure_model(embedding_dim)
        self._load_state(g if not _is_identity_state_artifact(g) else f)
        model = self._model
        assert model is not None
        model.eval()
        pred_by_doc: Dict[str, Optional[float]] = {}
        for item in prepared:
            proxy_norm = self._independent_parent_text_readings(
                model, item, [str(item.root_node_id)]
            )
            value = _denormalize(
                float(proxy_norm.detach().cpu().reshape(-1)[0].item()),
                lo=self.config.target_min,
                hi=self.config.target_max,
            )
            pred_by_doc[str(item.tree.doc_id)] = float(value)
        return [pred_by_doc.get(str(tree.doc_id)) for tree in trees]

    def export_last_full_tree_traces(
        self,
        output_root: str | Path,
        *,
        split: str = "predict",
    ) -> Dict[str, Any]:
        """Persist the most recent full-tree traces emitted by ``score_roots_with_f``."""

        root = Path(output_root)
        root.mkdir(parents=True, exist_ok=True)
        trace_path = root / f"full_tree_traces_{split}.jsonl"
        metrics_path = root / f"full_tree_metrics_{split}.json"
        write_state_trees_jsonl(self._last_full_tree_traces, trace_path)
        metrics = state_tree_trace_metrics(self._last_full_tree_traces)
        metrics_path.write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return {
            "full_tree_traces_jsonl": str(trace_path),
            "full_tree_metrics_json": str(metrics_path),
        }
