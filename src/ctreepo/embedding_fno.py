"""Embedding-coordinate FNO distillation for tree-indexed labels.

This backend treats each embedding vector as a 1D signal over its coordinate
axis.  The leaf-count grid controls tree topology only; the FNO resolution is
the embedding dimension, e.g. 1024 for Qwen3-Embedding-0.6B.

FNO channel invariant (load-bearing):
- ``leaf_fno`` (f): ``in_channels=1``, ``out_channels=1``; operates on
  ``(batch, 1, embedding_dim)``.
- ``merge_fno`` (g): ``in_channels=2``, ``out_channels=1``; operates on
  ``(batch, 2, embedding_dim)`` = concat of two child embeddings along a new
  channel axis, producing a single embedding-dim-wide output.

This invariant is what makes ``merge(concat(a, b))`` literally "concatenate two
embeddings and produce one embedding", and what lets identity init
(``merge(a, a) = a``) be well-defined. ``state_channels`` is intentionally NOT
a tunable parameter of this module; ``hidden_channels`` inside the FNO is.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
import math
import random
import time
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.tasks.manifesto.corpus_metrics import compute_corpus_pearson_r
from src.training.config_sections import (
    OptimizerConfig,
    RunConfig,
    RuntimeConfig,
    TestConfig,
    TrainConfig,
    ValidationConfig,
    config_to_dict,
)
from src.tree.labeled import LabeledNode, LabeledTree
from src.tree.state_tree import (
    explicit_oracle_trace_kwargs,
    local_law_trace_metadata,
    state_tree_skeleton_from_labeled_tree,
    state_tree_trace_metrics,
    update_state_tree_node,
    write_state_trees_jsonl,
)

import logging as _logging
_LOGGER = _logging.getLogger(__name__)


@dataclass(frozen=True, kw_only=True)
class EmbeddingFNOModelConfig:
    hidden_channels: int = 32
    n_modes: int = 64
    n_layers: int = 2
    head_hidden_dim: int = 64
    target_min: float = 1.0
    target_max: float = 7.0
    # Merge baseline mode. "mean" (default, backward-compatible) anchors g to
    # 0.5*(left+right) + FNO residual. "gated" replaces the fixed mean with a
    # learned per-dimension gate alpha(left,right) so the baseline is
    # alpha*left + (1-alpha)*right + FNO residual. The gate lets g ROUTE signal
    # from the on-topic child instead of averaging it away — targeted at the
    # eu-style learned-merge failure where a sparse on-topic minority gets
    # diluted by mean composition. Invariant preserved: merge(a,a)=a for any
    # alpha, and the gate is symmetric (see EmbeddingCoordinateFNOTreeRegressor).
    # "maxpool" uses a non-convex per-dim max(left,right) baseline (no extra
    # params) so the strongest child signal survives to the root — for dims
    # whose doc label tracks the MAX on-topic leaf, not the mean.
    merge_mode: str = "mean"
    merge_gate_hidden_dim: int = 64
    # Root readout: how the DOC-level prediction is read off. "mean_root"
    # (default) = predict_normalized(composed root state). "topk" = mean of the k
    # highest LEAF scores. "softmax" = temperature softmax pool over leaf scores.
    # The non-default modes bypass the mean-composed root for dims whose doc label
    # tracks the MAX on-topic leaf (eu: top1-leaf r=0.79 ~= ceiling).
    root_readout: str = "mean_root"
    root_readout_k: int = 1
    root_readout_attn_temp: float = 0.2


@dataclass(frozen=True, kw_only=True)
class EmbeddingFNOObjectiveConfig:
    root_weight: float = 1.0
    leaf_weight: float = 0.5
    merge_weight: float = 0.5


@dataclass(frozen=True, kw_only=True)
class EmbeddingFNOTrainConfig:
    run: RunConfig = field(default_factory=RunConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    optimizer: OptimizerConfig = field(
        default_factory=lambda: OptimizerConfig(
            learning_rate=1e-3,
            weight_decay=1e-4,
            optimizer="adamw",
            grad_clip_norm=1.0,
        )
    )
    validation: ValidationConfig = field(default_factory=ValidationConfig)
    test: TestConfig = field(default_factory=TestConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)
    model: EmbeddingFNOModelConfig = field(default_factory=EmbeddingFNOModelConfig)
    objective: EmbeddingFNOObjectiveConfig = field(default_factory=EmbeddingFNOObjectiveConfig)


@dataclass
class EmbeddingFNOFitResult:
    output_dir: str
    embedding_dim: int
    train_count: int
    val_count: int
    test_count: int
    metrics: Dict[str, Any]
    artifacts: Dict[str, str]
    config: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return config_to_dict(self)


@dataclass
class _PreparedTree:
    tree: LabeledTree
    split: str
    leaf_embeddings: torch.Tensor
    node_order: List[str]
    leaf_ranges: Dict[str, Tuple[int, int]]
    root_node_id: str


class EmbeddingCoordinateFNOTreeRegressor(nn.Module):
    """Tree model whose leaf and merge operators are FNOs over embedding dims."""

    def __init__(
        self,
        *,
        embedding_dim: int,
        hidden_channels: int,
        n_modes: int,
        n_layers: int,
        head_hidden_dim: int,
        target_min: float,
        target_max: float,
        merge_mode: str = "mean",
        merge_gate_hidden_dim: int = 64,
        root_readout: str = "mean_root",
        root_readout_k: int = 1,
        root_readout_attn_temp: float = 0.2,
        extent_enabled: bool = False,
        extent_merge_init: str = "neutral",
    ) -> None:
        super().__init__()
        from neuralop.models import FNO

        self.embedding_dim = int(embedding_dim)
        self.target_min = float(target_min)
        self.target_max = float(target_max)
        self.merge_mode = str(merge_mode)
        self.root_readout = str(root_readout)
        self.root_readout_k = int(root_readout_k)
        self.root_readout_attn_temp = float(root_readout_attn_temp)
        modes = max(1, min(int(n_modes), int(embedding_dim)))
        self.leaf_norm = nn.LayerNorm(int(embedding_dim))
        # f operator: (B, 1, embedding_dim) -> (B, 1, embedding_dim).
        self.leaf_fno = FNO(
            n_modes=(modes,),
            in_channels=1,
            out_channels=1,
            hidden_channels=int(hidden_channels),
            n_layers=int(n_layers),
        )
        # g operator: (B, 2, embedding_dim) -> (B, 1, embedding_dim).
        self.merge_fno = FNO(
            n_modes=(modes,),
            in_channels=2,
            out_channels=1,
            hidden_channels=int(hidden_channels),
            n_layers=int(n_layers),
        )
        self.score_head = nn.Sequential(
            nn.Linear(int(embedding_dim), int(head_hidden_dim)),
            nn.GELU(),
            nn.Linear(int(head_hidden_dim), 1),
        )
        # A3-literal readout merge M(a,b)=sigmoid(logit a + logit b + offset): the
        # single learned offset of the associative+commutative phi-form (see
        # readout_merge). Always present (tiny); only used when the A2 readout mode
        # is active. Init 0 -> M(a,b)=sigmoid(logit a + logit b) (a sum-in-logit
        # baseline), which the loss moves.
        self.readout_merge_offset = nn.Parameter(torch.zeros(()))
        # Gated merge baseline (merge_mode="gated"): a per-dimension gate
        # alpha(left,right) in [0,1] replaces the fixed 0.5 mean. Built from
        # symmetric features [left+right, |left-right|] so the gate magnitude is
        # order-stable; the final logit is anti-symmetrized so that swapping
        # left<->right maps alpha -> 1-alpha (the baseline alpha*l+(1-alpha)*r is
        # then permutation-consistent). merge(a,a)=a holds for ANY alpha. At init
        # the final layer is zeroed so alpha=0.5 (recovers the mean baseline).
        # Learned "extent" latent (mass-aware general g). When enabled, every node
        # state carries an EXTRA scalar coordinate at index ``embedding_dim`` (the
        # stored state is width D+1). The extent is a FREE latent: the leaf encoder
        # emits it from leaf content; the merge propagates it; NOTHING supervises it
        # against the true text mass. Its only purpose is to let the state-blend gate
        # weight children by "how much" (information density), which the two child
        # state vectors alone cannot encode (the correct merge weight is the mass
        # ratio N_l/(N_l+N_r), not a function of (s_l,s_r)). CRITICAL: the extent is
        # EXCISED before every FNO/score op and re-injected only as a gate feature +
        # a propagated scalar, so leaf_fno/merge_fno/score_head/merge_gate-output all
        # stay width-D (channel invariant intact; old checkpoints load when off).
        self.extent_enabled = bool(extent_enabled)
        self.extent_merge_init = str(extent_merge_init)
        # Per-merge gate sees 2*D state features (+2 child extents when enabled).
        gate_in = 2 * int(embedding_dim) + (2 if self.extent_enabled else 0)
        self.merge_gate = None
        if self.merge_mode == "gated":
            self.merge_gate = nn.Sequential(
                nn.Linear(gate_in, int(merge_gate_hidden_dim)),
                nn.GELU(),
                nn.Linear(int(merge_gate_hidden_dim), int(embedding_dim)),
            )
        elif self.merge_mode == "maxpool":
            # No extra params: baseline = per-dim max(left,right) + FNO residual.
            # NON-convex, NON-additive composition — the strongest child's signal
            # on each embedding dim survives all the way to the root instead of
            # being averaged down. Motivated by the eu leaf diagnostic: the doc
            # signal lives in the MAX / top-k on-topic q-sentence (max-leaf r=0.79
            # ~= the 0.78 ceiling), while sum/count rolls up WORSE. Convex gating
            # (above) cannot express this — its output is bounded between the two
            # children, so a single strong leaf decays toward the mean up the tree.
            pass
        elif self.merge_mode == "mlp":
            # EXPERIMENTAL — intentionally BREAKS the FNO channel invariant for g.
            # A free, high-capacity learnable merge: concat([l,r]) (B,2D) -> MLP ->
            # (B,D). No mean/max/gated prior; the FNO merge_fno is UNUSED in this
            # mode. Tests whether a general learnable function (the "neural-operator
            # is universal" hypothesis) can find the eu composition that the
            # hand-picked baselines (mean/gated/maxpool all failed) cannot — or
            # whether the signal is already gone from the learned leaf STATES
            # (vs the LLM leaf scores, which roll up to 0.78). Init makes the MLP
            # output the mean baseline 0.5*(l+r) (residual final layer zeroed +
            # explicit averaging skip), so it WARM-STARTS exactly where the working
            # 'mean' merge starts and learns away from there. Only enabled when a
            # caller opts in; default stays 'mean' so the invariant holds elsewhere.
            self.merge_mlp = nn.Sequential(
                nn.Linear(2 * int(embedding_dim), int(merge_gate_hidden_dim)),
                nn.GELU(),
                nn.Linear(int(merge_gate_hidden_dim), int(merge_gate_hidden_dim)),
                nn.GELU(),
                nn.Linear(int(merge_gate_hidden_dim), int(embedding_dim)),
            )
        elif self.merge_mode != "mean":
            raise ValueError(
                f"unknown merge_mode {self.merge_mode!r}; expected "
                "'mean', 'gated', 'maxpool', or 'mlp'"
            )
        if not hasattr(self, "merge_mlp"):
            self.merge_mlp = None
        # Extent heads (only built when enabled). The extent rides the gate, so the
        # gate must exist — extent is meaningful only with the per-dim gated merge,
        # which is where "how much each child weighs" can steer the blend.
        self.leaf_extent_head = None
        self.extent_merge_head = None
        if self.extent_enabled:
            if self.merge_gate is None:
                raise ValueError(
                    "extent_enabled=True requires merge_mode='gated' (the extent "
                    "latent steers the per-dim gate; mean/maxpool/mlp have no gate "
                    "to read it). Pass --fno-merge-mode gated."
                )
            # Leaf extent: a scalar emitted from the raw leaf embedding. Learned, not
            # fed the true mass. tanh-bounded so the latent stays O(1) and the gate
            # features are well-scaled.
            self.leaf_extent_head = nn.Sequential(
                nn.Linear(int(embedding_dim), int(merge_gate_hidden_dim)),
                nn.GELU(),
                nn.Linear(int(merge_gate_hidden_dim), 1),
            )
            # Extent merge: parent extent from the two child extents + a summary of
            # the two child states. init='additive' warm-starts the parent extent at
            # m_l+m_r (the mass-weighted prior basin) by zeroing the residual MLP and
            # adding an explicit sum skip; init='neutral' zeroes the head so the
            # untrained parent extent is 0 everywhere (the constant-extent basin that
            # collapses to equal-averaging — the pure-laws arm A control).
            self.extent_merge_head = nn.Sequential(
                nn.Linear(2 + 2 * int(embedding_dim), int(merge_gate_hidden_dim)),
                nn.GELU(),
                nn.Linear(int(merge_gate_hidden_dim), 1),
            )
            # Both inits zero the final residual layer; 'additive' adds m_l+m_r via an
            # explicit skip in extent_merge() (see below), 'neutral' does not.
            with torch.no_grad():
                final = self.extent_merge_head[-1]
                final.weight.zero_()
                final.bias.zero_()
            if self.extent_merge_init not in ("neutral", "additive"):
                raise ValueError(
                    f"unknown extent_merge_init {self.extent_merge_init!r}; "
                    "expected 'neutral' or 'additive'"
                )
        _LOGGER.info(
            "FNO invariant: embedding_dim=%d, leaf_fno 1->1, merge_fno 2->1, "
            "hidden_channels=%d, n_modes=%d, n_layers=%d, head_hidden_dim=%d, merge_mode=%s",
            int(embedding_dim), int(hidden_channels), int(n_modes),
            int(n_layers), int(head_hidden_dim), self.merge_mode,
        )

    def encode_leaves(self, embeddings: torch.Tensor) -> torch.Tensor:
        # Residual bypass around the FNO: at identity init (zeroed FNO weights
        # and leaf_norm weight=1/bias=0), the output equals the raw input
        # embedding. Trained, the FNO learns a residual on top of the embedding.
        raw = embeddings.unsqueeze(1)
        normalized = self.leaf_norm(embeddings).unsqueeze(1)
        state = raw + self.leaf_fno(normalized)  # (B, 1, D)
        if not self.extent_enabled:
            return state
        # Append the learned extent scalar as the (D+1)-th coordinate. The FNO above
        # only ever sees the D state coords; the extent is computed separately from
        # the leaf embedding and never enters the spectral conv.
        extent = self.leaf_extent_head(embeddings).unsqueeze(1)  # (B, 1, 1)
        return torch.cat([state, extent], dim=-1)  # (B, 1, D+1)

    def _split_extent(
        self, state: torch.Tensor
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Split a stored node state into (D state coords, extent scalar | None).

        When extent is disabled the state is width D and extent is None.
        """
        if not self.extent_enabled:
            return state, None
        d = self.embedding_dim
        return state[..., :d], state[..., d : d + 1]

    def extent_merge(
        self,
        m_left: torch.Tensor,
        m_right: torch.Tensor,
        s_left: torch.Tensor,
        s_right: torch.Tensor,
    ) -> torch.Tensor:
        """Combine two child extents into the parent extent (B, 1, 1).

        Reads both child extents and a summary of the two child STATES (so extent
        propagation can depend on content). 'additive' init adds an explicit
        m_l+m_r skip so the untrained parent extent is the mass-weighted prior;
        'neutral' has no skip (untrained parent extent = 0, the collapse basin).
        """
        feats = torch.cat(
            [
                m_left.squeeze(1),
                m_right.squeeze(1),
                s_left.squeeze(1),
                s_right.squeeze(1),
            ],
            dim=-1,
        )  # (B, 2 + 2D)
        resid = self.extent_merge_head(feats).unsqueeze(1)  # (B, 1, 1)
        if self.extent_merge_init == "additive":
            return m_left + m_right + resid
        return resid

    def merge(self, left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
        # Residual bypass around the merge FNO. The FNO (2->1) always learns a
        # residual on top of a content-independent-or-gated baseline.
        #
        # When the extent latent is on, split it off FIRST so every op below
        # (gate, merge_fno) operates only on the D state coords; the extent is fed
        # to the gate as a feature and propagated by extent_merge(), then the parent
        # extent is concatenated back. The FNO never sees the extent coordinate.
        left_s, m_left = self._split_extent(left)
        right_s, m_right = self._split_extent(right)
        if self.merge_gate is not None:
            # Gated baseline: alpha*left + (1-alpha)*right, per embedding dim.
            # left/right are (B, 1, D); flatten the singleton channel for the gate.
            l = left_s.squeeze(1)
            r = right_s.squeeze(1)
            # Symmetric state features; anti-symmetrize so swap(l,r)->alpha->1-alpha.
            sym = [l + r, (l - r).abs()]
            if self.extent_enabled:
                # Extent features: append (m_l, m_r) to feats_lr and (m_r, m_l) to
                # feats_rl so the anti-symmetrization still maps swap -> 1-alpha
                # while letting alpha depend on which child has more extent.
                ml = m_left.squeeze(1)
                mr = m_right.squeeze(1)
                feats_lr = torch.cat(sym + [ml, mr], dim=-1)
                feats_rl = torch.cat([r + l, (r - l).abs(), mr, ml], dim=-1)
            else:
                feats_lr = torch.cat(sym, dim=-1)
                feats_rl = torch.cat([r + l, (r - l).abs()], dim=-1)
            logit = self.merge_gate(feats_lr) - self.merge_gate(feats_rl)
            alpha = torch.sigmoid(logit).unsqueeze(1)  # (B, 1, D), alpha(a,a)->0.5
            avg = alpha * left_s + (1.0 - alpha) * right_s
        elif self.merge_mode == "maxpool":
            # Per-dim elementwise max baseline (non-convex; preserves the strongest
            # child signal up the tree). Commutative -> permutation-symmetric;
            # max(a,a)=a -> invariant holds.
            avg = torch.maximum(left_s, right_s)
        elif self.merge_mlp is not None:
            # Free learnable merge: mean warm-start + MLP residual over concat.
            # The merge_fno is UNUSED here. At init the MLP final layer is zeroed
            # so this equals 0.5*(l+r); merge(a,a)=a holds at init. NOT symmetric
            # by construction once trained (the model can learn order-sensitivity
            # — acceptable for this probe; trees have a fixed child order).
            l = left_s.squeeze(1)
            r = right_s.squeeze(1)
            resid = self.merge_mlp(torch.cat([l, r], dim=-1)).unsqueeze(1)
            return 0.5 * (left_s + right_s) + resid
        else:
            # Fixed mean baseline (backward-compatible default).
            avg = 0.5 * (left_s + right_s)
        residual = self.merge_fno(torch.cat([left_s, right_s], dim=1))
        merged_state = avg + residual  # (B, 1, D)
        if not self.extent_enabled:
            return merged_state
        m_parent = self.extent_merge(m_left, m_right, left_s, right_s)  # (B,1,1)
        return torch.cat([merged_state, m_parent], dim=-1)  # (B, 1, D+1)

    def predict_normalized(self, state: torch.Tensor) -> torch.Tensor:
        # Score reads only the D state coords; the extent latent (if present) is a
        # routing variable for the merge gate and is never scored.
        state_s, _ = self._split_extent(state)
        flat = state_s.squeeze(1)
        logits = self.score_head(flat).reshape(-1)
        return torch.sigmoid(logits)

    def readout_merge(self, y_left: torch.Tensor, y_right: torch.Tensor) -> torch.Tensor:
        """A3-literal merge on the SCALAR readouts: M(a,b) = phi^{-1}(phi(a)+phi(b)).

        By Aczel's theorem every continuous, associative, commutative, strictly
        monotone operator on an interval has this form for a homeomorphism phi, so
        this M is ASSOCIATIVE + COMMUTATIVE BY CONSTRUCTION (the proven Lean
        merge_assoc / merge_comm). We parameterize phi in logit space so it is
        exactly invertible on (0,1): phi(y) = w * logit(y) + b (w>0), hence
            M(a,b) = sigmoid( logit(a) + logit(b) + b/w ).
        The single learned offset c = b/w spans the family from sum-like (c<0, two
        mid values reinforce downward) to a learned neutral; w cancels in M but
        keeps phi a proper homeomorphism. Inputs/outputs in (0,1). Only enabled when
        a2_readout_merge is on; otherwise this method is unused.
        """
        eps = 1e-6
        a = y_left.clamp(eps, 1.0 - eps)
        b = y_right.clamp(eps, 1.0 - eps)
        la = torch.log(a) - torch.log1p(-a)
        lb = torch.log(b) - torch.log1p(-b)
        return torch.sigmoid(la + lb + self.readout_merge_offset)

    def predict_raw(self, state: torch.Tensor) -> torch.Tensor:
        norm = self.predict_normalized(state)
        return self.target_min + norm * (self.target_max - self.target_min)

    def predict_root_topk(
        self,
        leaf_states: torch.Tensor,
        *,
        mode: str = "mean_root",
        k: int = 1,
        attn_temp: float = 1.0,
    ) -> torch.Tensor:
        """Read a DOC-level normalized prediction from the LEAF score distribution.

        The mean-composed root state averages a peaked signal away (the eu
        diagnostic: top1-leaf score r=0.79 ~= ceiling, top-k DECREASING). These
        readouts use the leaf SCORES directly instead of the composed root state:
        - "topk":    mean of the k highest leaf scores (k=1 -> the single max leaf)
        - "softmax": temperature-weighted softmax pool over leaf scores
                     (attn_temp->0 approximates max; ->inf approximates mean).
        ``leaf_states`` is (n_leaves, 1, D). Differentiable; trains the f leaf head
        to make the top leaves accurate. ``mode="mean_root"`` is handled by the
        caller (uses the composed root state, not this method).
        """
        scores = self.predict_normalized(leaf_states)  # (n_leaves,)
        if scores.numel() == 0:
            return scores.new_zeros(())
        if mode == "topk":
            kk = max(1, min(int(k), int(scores.numel())))
            top = torch.topk(scores, kk, largest=True).values
            return top.mean()
        if mode == "softmax":
            temp = max(1e-4, float(attn_temp))
            weights = torch.softmax(scores / temp, dim=0)
            return (weights * scores).sum()
        raise ValueError(f"unknown root_readout mode {mode!r}; expected 'topk' or 'softmax'")

    @torch.no_grad()
    def initialize_as_identity(self) -> None:
        """Set weights so the f/g paths reduce to the invariant's baseline.

        After this call:
        - ``encode_leaves(x)`` equals ``x.unsqueeze(1)`` for any ``x``.
        - ``merge(a, b)`` equals ``0.5 * (a + b)`` for any ``a``, ``b``.
        - ``predict_normalized`` returns 0.5 (mid-range), so ``predict_raw``
          returns ``target_min + 0.5 * (target_max - target_min)``.

        Subsequent training lets the FNOs learn residual corrections on top
        of these baselines, and the score head to move away from 0.5.

        Only the FINAL layer of each module is zeroed; hidden layers keep
        their default random init (zero-init-last-layer trick). Zeroing every
        parameter — the previous behavior — created a mutual gradient
        deadlock (zero hidden activations x zero downstream weights), so f
        and g could only ever learn constants: observed empirically as root
        predictions exactly equal across documents, drifting from 0.5 to the
        target mean via the sole trainable final bias.
        """

        def _zero_fno_output_layer(fno: nn.Module) -> None:
            # neuralop FNO: output = projection (ChannelMLP) over block
            # features; zeroing its last conv makes the FNO output exactly 0
            # while leaving every other parameter trainable from step 1.
            last = fno.projection.fcs[-1]
            nn.init.zeros_(last.weight)
            if last.bias is not None:
                nn.init.zeros_(last.bias)

        _zero_fno_output_layer(self.leaf_fno)
        _zero_fno_output_layer(self.merge_fno)
        if self.merge_gate is not None:
            # Zero the gate's final layer so its logit (and the anti-symmetrized
            # difference) is 0 -> alpha=0.5 -> the gated baseline reduces to the
            # mean baseline at init. The gate learns to route away from 0.5.
            gate_final = self.merge_gate[-1]
            assert isinstance(gate_final, nn.Linear)
            if self.extent_enabled:
                # Extent path must stay differentiable at init. If we fully zero the
                # gate's final layer, alpha=0.5 exactly but d(alpha)/d(extent)=0, so
                # the leaf-extent head NEVER receives gradient during g-training and
                # the extent collapses to a constant (= equal-averaging) — a
                # permanent deadlock confirmed empirically. Fix: seed the final
                # layer's EXTENT input columns (the last 2 hidden units are NOT
                # extent-specific, so instead we keep the final weights tiny-random
                # rather than zero). Tiny so alpha stays ~0.5 at init (mean warm
                # start preserved to ~1e-3), but the extent gradient path is live
                # from step 1. The STATE columns of the gate's FIRST layer dominate
                # alpha once leaf extents are still ~0, so this does not bias routing.
                nn.init.normal_(gate_final.weight, std=1e-3)
                nn.init.zeros_(gate_final.bias)
            else:
                nn.init.zeros_(gate_final.weight)
                nn.init.zeros_(gate_final.bias)
        if self.merge_mlp is not None:
            # Zero the MLP residual's final layer so merge = 0.5*(l+r) at init
            # (warm-start = the working mean baseline); learns away from there.
            mlp_final = self.merge_mlp[-1]
            assert isinstance(mlp_final, nn.Linear)
            nn.init.zeros_(mlp_final.weight)
            nn.init.zeros_(mlp_final.bias)
        nn.init.ones_(self.leaf_norm.weight)
        nn.init.zeros_(self.leaf_norm.bias)
        final = self.score_head[-1]
        assert isinstance(final, nn.Linear)
        nn.init.zeros_(final.weight)
        nn.init.zeros_(final.bias)
        if self.leaf_extent_head is not None:
            # Seed the leaf extent head's final layer TINY-random (not zero) so leaf
            # extents start slightly nonzero. With exactly-zero extents the gate's
            # extent-input columns receive no gradient (0 input -> 0 weight grad),
            # reinforcing the deadlock; a tiny nonzero extent keeps that path live.
            # The extent never enters the score (predict_normalized slices it off),
            # so a nonzero init does not perturb the doc-level warm start; it only
            # gives the merge gate a usable extent signal to learn from.
            le_final = self.leaf_extent_head[-1]
            assert isinstance(le_final, nn.Linear)
            nn.init.normal_(le_final.weight, std=1e-2)
            nn.init.zeros_(le_final.bias)

    def _set_requires_grad(self, module: nn.Module, flag: bool) -> None:
        for p in module.parameters():
            p.requires_grad = bool(flag)

    def freeze_for_f_training(self) -> None:
        """Train f-path params (leaf_fno + leaf_norm + score_head); freeze g (merge_fno + gate)."""
        self._set_requires_grad(self.leaf_fno, True)
        self._set_requires_grad(self.leaf_norm, True)
        self._set_requires_grad(self.score_head, True)
        self._set_requires_grad(self.merge_fno, False)
        if self.merge_gate is not None:
            self._set_requires_grad(self.merge_gate, False)
        if self.merge_mlp is not None:
            self._set_requires_grad(self.merge_mlp, False)
        # The leaf extent is part of the leaf STATE (emitted at encode time), so it
        # trains with f; the merge-side extent head is frozen during f.
        if self.leaf_extent_head is not None:
            self._set_requires_grad(self.leaf_extent_head, True)
        if self.extent_merge_head is not None:
            self._set_requires_grad(self.extent_merge_head, False)
        # A3 readout-merge offset is a merge param -> frozen during f.
        self.readout_merge_offset.requires_grad = False

    def freeze_for_g_training(self) -> None:
        """Train g-path params (merge_fno + gate + mlp); freeze leaf_fno, leaf_norm, score_head."""
        self._set_requires_grad(self.leaf_fno, False)
        self._set_requires_grad(self.leaf_norm, False)
        self._set_requires_grad(self.score_head, False)
        self._set_requires_grad(self.merge_fno, True)
        if self.merge_gate is not None:
            self._set_requires_grad(self.merge_gate, True)
        if self.merge_mlp is not None:
            self._set_requires_grad(self.merge_mlp, True)
        # g learns BOTH how to combine extents (extent_merge_head) AND how to shape
        # the leaf extents it consumes (leaf_extent_head) — if the leaf extent were
        # frozen at its identity-init 0, every child would have extent 0 and the gate
        # would see no extent signal (the collapse the A/B is built to detect). So
        # the leaf extent head trains in g too.
        if self.leaf_extent_head is not None:
            self._set_requires_grad(self.leaf_extent_head, True)
        if self.extent_merge_head is not None:
            self._set_requires_grad(self.extent_merge_head, True)
        # A3 readout-merge offset is a merge param -> trains with g.
        self.readout_merge_offset.requires_grad = True

    def unfreeze_all(self) -> None:
        self._set_requires_grad(self, True)


def _normalize_score(value: float, *, target_min: float, target_max: float) -> float:
    span = float(target_max) - float(target_min)
    if span <= 0.0:
        return 0.5
    return max(0.0, min(1.0, (float(value) - float(target_min)) / span))


def _denormalize_score(value: float, *, target_min: float, target_max: float) -> float:
    return float(target_min) + max(0.0, min(1.0, float(value))) * (float(target_max) - float(target_min))


def _tree_split(tree: LabeledTree) -> str:
    return str((tree.metadata or {}).get("split", "") or "")


def _select_trees(trees: Sequence[LabeledTree], splits: Sequence[str]) -> List[LabeledTree]:
    keys = {str(value).lower() for value in splits}
    return [tree for tree in trees if _tree_split(tree).lower() in keys]


def _ordered_nodes(tree: LabeledTree) -> List[LabeledNode]:
    out: List[LabeledNode] = []
    seen: set[str] = set()
    for level_ids in list(tree.levels or []):
        for node_id in level_ids:
            node = tree.get_node(str(node_id))
            if node is not None and str(node.node_id) not in seen:
                out.append(node)
                seen.add(str(node.node_id))
    for node in sorted(tree.nodes.values(), key=lambda item: (int(item.level), str(item.node_id))):
        if str(node.node_id) not in seen:
            out.append(node)
            seen.add(str(node.node_id))
    return out


def _node_leaf_ranges(tree: LabeledTree) -> Dict[str, Tuple[int, int]]:
    leaves = list(tree.levels[0] if tree.levels else [])
    leaf_index = {str(node_id): idx for idx, node_id in enumerate(leaves)}
    memo: Dict[str, Tuple[int, int]] = {}

    def visit(node_id: str) -> Tuple[int, int]:
        node_id = str(node_id)
        if node_id in memo:
            return memo[node_id]
        node = tree.get_node(node_id)
        if node is None:
            raise ValueError(f"missing node id {node_id!r} in tree {tree.doc_id!r}")
        if int(node.level) == 0 or not node.left_child_id:
            idx = leaf_index[node_id]
            memo[node_id] = (idx, idx + 1)
            return memo[node_id]
        left = visit(str(node.left_child_id))
        right = visit(str(node.right_child_id or node.left_child_id))
        memo[node_id] = (min(left[0], right[0]), max(left[1], right[1]))
        return memo[node_id]

    for node in _ordered_nodes(tree):
        visit(str(node.node_id))
    return memo


def _prepare_trees(
    trees: Sequence[LabeledTree],
    *,
    embedding_client: Any,
    embedding_max_tokens: Optional[int] = None,
    chunks_per_leaf: int = 1,
    tokenizer_model_path: Optional[str] = None,
    enforce_no_truncation: bool = True,
) -> Tuple[List[_PreparedTree], int]:
    """Embed each leaf into a fixed-width coordinate vector.

    Per the no-truncation invariant: if a leaf is larger than the embedding
    model's max token length, split it into fixed chunk slots and concatenate
    those embeddings along the FNO spatial axis. This preserves the 1-channel
    f / 2-channel g invariant while allowing ``D_eff = K * D`` for future
    smaller-context embedding backends.
    """
    prepared: List[_PreparedTree] = []
    embedding_dim: Optional[int] = None
    chunks_per_leaf = max(1, int(chunks_per_leaf))
    if enforce_no_truncation and embedding_max_tokens is not None:
        from src.preprocessing.leaf_size_utils import (
            assert_no_truncation,
            char_windows_from_token_budget,
        )
    else:
        assert_no_truncation = None  # type: ignore[assignment]
        char_windows_from_token_budget = None  # type: ignore[assignment]
    for tree in trees:
        leaves = [tree.get_node(str(node_id)) for node_id in (tree.levels[0] if tree.levels else [])]
        leaf_nodes = [node for node in leaves if node is not None]
        if not leaf_nodes:
            continue
        leaf_texts = [str(node.text or "") for node in leaf_nodes]
        leaf_chunks: List[List[str]] = []
        for idx, text in enumerate(leaf_texts):
            if char_windows_from_token_budget is not None and embedding_max_tokens is not None:
                windows = char_windows_from_token_budget(
                    text,
                    int(embedding_max_tokens),
                    model_path=tokenizer_model_path,
                )
                chunks = [text[int(start): int(end)] for start, end in windows]
            else:
                chunks = [text]
            if len(chunks) > chunks_per_leaf:
                raise RuntimeError(
                    f"silent truncation in _prepare_trees: tree={tree.doc_id!r} leaf_idx={idx} "
                    f"needs {len(chunks)} embedding chunks but chunks_per_leaf={chunks_per_leaf}. "
                    "Increase leaf_size_tokens/chunks_per_leaf alignment or reduce leaf size."
                )
            if assert_no_truncation is not None and embedding_max_tokens is not None:
                for chunk_idx, chunk in enumerate(chunks):
                    try:
                        assert_no_truncation(
                            chunk,
                            max_tokens=int(embedding_max_tokens),
                            model_path=tokenizer_model_path,
                        )
                    except RuntimeError as exc:
                        raise RuntimeError(
                            f"silent truncation in _prepare_trees: tree={tree.doc_id!r} "
                            f"leaf_idx={idx} chunk_idx={chunk_idx} would overflow "
                            f"embedding_max_tokens={embedding_max_tokens}. Underlying error: {exc}"
                        ) from exc
            leaf_chunks.append(chunks or [""])
        flat_chunks = [chunk for chunks in leaf_chunks for chunk in chunks]
        chunk_embeddings = embedding_client.embed_texts(flat_chunks)
        if not chunk_embeddings:
            continue
        base_dim = int(len(chunk_embeddings[0]))
        if any(int(len(vec)) != base_dim for vec in chunk_embeddings):
            raise ValueError(f"embedding dimension changed across chunks for tree {tree.doc_id!r}")
        leaf_vectors: List[List[float]] = []
        cursor = 0
        zero = [0.0] * base_dim
        for chunks in leaf_chunks:
            count = len(chunks)
            parts = [list(vec) for vec in chunk_embeddings[cursor: cursor + count]]
            cursor += count
            while len(parts) < chunks_per_leaf:
                parts.append(list(zero))
            leaf_vectors.append([value for part in parts for value in part])
        tensor = torch.tensor(leaf_vectors, dtype=torch.float32)
        if tensor.ndim != 2:
            raise ValueError(f"embedding client returned non-matrix embeddings for {tree.doc_id!r}")
        if embedding_dim is None:
            embedding_dim = int(tensor.shape[1])
        elif int(tensor.shape[1]) != int(embedding_dim):
            raise ValueError(
                f"embedding dimension changed from {embedding_dim} to {tensor.shape[1]} "
                f"for tree {tree.doc_id!r}"
            )
        node_order = [str(node.node_id) for node in _ordered_nodes(tree)]
        root_node_id = str(tree.levels[-1][0]) if tree.levels and tree.levels[-1] else node_order[-1]
        prepared.append(
            _PreparedTree(
                tree=tree,
                split=_tree_split(tree),
                leaf_embeddings=tensor,
                node_order=node_order,
                leaf_ranges=_node_leaf_ranges(tree),
                root_node_id=root_node_id,
            )
        )
    if embedding_dim is None:
        raise ValueError("No trees could be embedded")
    return prepared, int(embedding_dim)


def _device_from_runtime(runtime: RuntimeConfig) -> torch.device:
    requested = str(runtime.device or "auto").lower()
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)


def _forward_tree_states(
    model: EmbeddingCoordinateFNOTreeRegressor,
    item: _PreparedTree,
    *,
    device: torch.device,
) -> Dict[str, torch.Tensor]:
    leaf_ids = list(item.tree.levels[0] if item.tree.levels else [])
    leaf_states = model.encode_leaves(item.leaf_embeddings.to(device))
    states: Dict[str, torch.Tensor] = {}
    for idx, node_id in enumerate(leaf_ids):
        states[str(node_id)] = leaf_states[idx : idx + 1]
    # Merge per LEVEL: every merge node at level L depends only on children at
    # levels < L, so all merges at a level can be evaluated in ONE batched
    # ``merge`` call. The trees are balanced (depth ~log(leaves)), so this turns
    # O(nodes) sequential tiny GPU ops into O(depth) batched ops — identical math
    # (merge is per-node independent given children), just vectorized.
    levels = list(item.tree.levels or [])
    for level in levels[1:]:
        merge_nodes = []
        lefts: List[torch.Tensor] = []
        rights: List[torch.Tensor] = []
        for node_id in level:
            if str(node_id) in states:
                continue
            node = item.tree.get_node(node_id)
            if node is None:
                continue
            merge_nodes.append(str(node_id))
            lefts.append(states[str(node.left_child_id)])
            rights.append(states[str(node.right_child_id or node.left_child_id)])
        if not merge_nodes:
            continue
        merged = model.merge(torch.cat(lefts, dim=0), torch.cat(rights, dim=0))
        for offset, node_id in enumerate(merge_nodes):
            states[node_id] = merged[offset : offset + 1]
    # Fallback for any node not covered by `levels` (defensive; preserves the
    # original topological pass so behavior is unchanged on irregular trees).
    for node_id in item.node_order:
        if str(node_id) in states:
            continue
        node = item.tree.get_node(node_id)
        if node is None:
            continue
        left = states[str(node.left_child_id)]
        right = states[str(node.right_child_id or node.left_child_id)]
        states[str(node_id)] = model.merge(left, right)
    return states


def _node_weight(
    node: LabeledNode,
    *,
    root_node_id: str,
    objective: EmbeddingFNOObjectiveConfig,
) -> float:
    if str(node.node_id) == str(root_node_id):
        return float(objective.root_weight)
    if int(node.level) == 0:
        return float(objective.leaf_weight)
    return float(objective.merge_weight)


def _batch_loss(
    model: EmbeddingCoordinateFNOTreeRegressor,
    batch: Sequence[_PreparedTree],
    *,
    device: torch.device,
    cfg: EmbeddingFNOTrainConfig,
) -> torch.Tensor:
    losses: List[torch.Tensor] = []
    target_min = float(cfg.model.target_min)
    target_max = float(cfg.model.target_max)
    for item in batch:
        states = _forward_tree_states(model, item, device=device)
        for node_id in item.node_order:
            node = item.tree.get_node(node_id)
            if node is None:
                continue
            weight = _node_weight(node, root_node_id=item.root_node_id, objective=cfg.objective)
            if weight <= 0.0:
                continue
            pred_norm = model.predict_normalized(states[node_id]).reshape(())
            target_norm = torch.tensor(
                _normalize_score(float(node.score), target_min=target_min, target_max=target_max),
                dtype=torch.float32,
                device=device,
            )
            losses.append(float(weight) * F.mse_loss(pred_norm, target_norm))
    if not losses:
        return torch.zeros((), dtype=torch.float32, device=device)
    return torch.stack(losses).mean()


@torch.no_grad()
def _evaluate_split(
    model: EmbeddingCoordinateFNOTreeRegressor,
    items: Sequence[_PreparedTree],
    *,
    device: torch.device,
    cfg: EmbeddingFNOTrainConfig,
    output_path: Optional[Path] = None,
    full_tree_trace_path: Optional[Path] = None,
) -> Dict[str, Any]:
    model.eval()
    target_min = float(cfg.model.target_min)
    target_max = float(cfg.model.target_max)
    rows: List[Dict[str, Any]] = []
    root_preds: List[float] = []
    root_targets: List[float] = []
    root_experts: List[float] = []
    node_errors: List[float] = []
    leaf_errors: List[float] = []
    merge_errors: List[float] = []
    root_errors: List[float] = []
    full_tree_traces = []

    for item in items:
        states = _forward_tree_states(model, item, device=device)
        trace = state_tree_skeleton_from_labeled_tree(
            item.tree,
            method_family="embedding_fno",
            state_kind="embedding_fno_state",
            split=item.split,
        )
        expert_score = (item.tree.metadata or {}).get("expert_score_1_7")
        try:
            expert_value = float(expert_score)
        except (TypeError, ValueError):
            expert_value = float("nan")
        for node_id in item.node_order:
            node = item.tree.get_node(node_id)
            if node is None:
                continue
            is_root = str(node_id) == str(item.root_node_id)
            is_leaf = int(node.level) == 0
            readout = str(getattr(cfg.model, "root_readout", "mean_root"))
            if is_root and readout != "mean_root":
                # Read the doc prediction from the LEAF score distribution (must
                # MATCH the training-time root readout, else train/eval mismatch).
                leaf_ids = list(item.tree.levels[0] if item.tree.levels else [])
                leaf_states = torch.cat([states[str(lid)] for lid in leaf_ids], dim=0)
                pred_norm = float(
                    model.predict_root_topk(
                        leaf_states,
                        mode=readout,
                        k=int(getattr(cfg.model, "root_readout_k", 1)),
                        attn_temp=float(getattr(cfg.model, "root_readout_attn_temp", 0.2)),
                    ).detach().cpu().reshape(()).item()
                )
            else:
                pred_norm = float(model.predict_normalized(states[node_id]).detach().cpu().reshape(()).item())
            pred_raw = _denormalize_score(pred_norm, target_min=target_min, target_max=target_max)
            target_raw = float(node.score)
            error = abs(float(pred_raw) - float(target_raw))
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
                node_weight=float(_node_weight(node, root_node_id=item.root_node_id, objective=cfg.objective)),
                law_channel="root" if is_root else ("leaf" if is_leaf else "merge"),
                state_kind="embedding_fno_state",
                label_source=str(oracle_kwargs["label_source"] or "proxy_score"),
            )
            node_errors.append(error)
            if is_root:
                root_errors.append(error)
                root_preds.append(float(pred_raw))
                root_targets.append(float(target_raw))
                if math.isfinite(expert_value):
                    root_experts.append(float(expert_value))
            elif is_leaf:
                leaf_errors.append(error)
            else:
                merge_errors.append(error)
            lo, hi = item.leaf_ranges.get(str(node_id), (0, 0))
            update_state_tree_node(
                trace,
                str(node_id),
                rendered=str(node.text or ""),
                state=states[node_id].detach().cpu(),
                metadata={
                    "prediction": float(pred_raw),
                    "readout_prediction": float(pred_raw),
                    "prediction_normalized": float(pred_norm),
                    "target": float(target_raw),
                    "target_1_7": float(target_raw),
                    **law_metadata,
                    "leaf_range": [int(lo), int(hi)],
                    "expert_score_1_7": expert_value if math.isfinite(expert_value) else None,
                },
            )
            rows.append(
                {
                    "doc_id": item.tree.doc_id,
                    "split": item.split,
                    "node_id": str(node_id),
                    "level": int(node.level),
                    "is_leaf": bool(is_leaf),
                    "is_root": bool(is_root),
                    "leaf_range": [int(lo), int(hi)],
                    "target_1_7": target_raw,
                    "prediction_1_7": float(pred_raw),
                    "abs_error_1_7": float(error),
                    "expert_score_1_7": expert_value if math.isfinite(expert_value) else None,
                }
            )
        full_tree_traces.append(trace)

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(config_to_dict(row), sort_keys=True) + "\n")
    if full_tree_trace_path is not None:
        write_state_trees_jsonl(full_tree_traces, full_tree_trace_path)

    def _mae(values: Sequence[float]) -> Optional[float]:
        return float(np.mean(values)) if values else None

    root_report = compute_corpus_pearson_r(root_preds, root_targets).as_dict() if len(root_preds) >= 4 else {"n": len(root_preds)}
    root_report["mae_1_7"] = _mae(root_errors)
    expert_report: Dict[str, Any]
    if len(root_experts) == len(root_preds) and len(root_preds) >= 4:
        expert_report = compute_corpus_pearson_r(root_preds, root_experts).as_dict()
    else:
        expert_report = {"n": min(len(root_preds), len(root_experts))}
    if root_experts and len(root_experts) == len(root_preds):
        expert_report["mae_1_7"] = float(np.mean([abs(p - t) for p, t in zip(root_preds, root_experts)]))
    else:
        expert_report["mae_1_7"] = None

    return {
        "count_trees": int(len(items)),
        "count_nodes": int(len(rows)),
        "node_mae_1_7": _mae(node_errors),
        "leaf_mae_1_7": _mae(leaf_errors),
        "merge_mae_1_7": _mae(merge_errors),
        "root_teacher_report": root_report,
        "root_expert_report": expert_report,
        "prediction_path": str(output_path) if output_path is not None else None,
        "full_tree_trace_path": (
            str(full_tree_trace_path) if full_tree_trace_path is not None else None
        ),
        "full_tree_trace_metrics": state_tree_trace_metrics(full_tree_traces),
    }


def fit_embedding_fno_node_regressor(
    labeled_trees: Sequence[LabeledTree],
    *,
    embedding_client: Any,
    config: Optional[EmbeddingFNOTrainConfig] = None,
) -> EmbeddingFNOFitResult:
    """Fit an embedding-coordinate FNO against tree node labels."""

    if embedding_client is None:
        raise ValueError("embedding_client is required for embedding-FNO fitting")
    cfg = config or EmbeddingFNOTrainConfig()
    output_dir = Path(cfg.run.output_dir or "outputs/embedding_fno_fit")
    output_dir.mkdir(parents=True, exist_ok=True)

    random.seed(int(cfg.run.seed))
    np.random.seed(int(cfg.run.seed))
    torch.manual_seed(int(cfg.run.seed))
    device = _device_from_runtime(cfg.runtime)

    train_trees = _select_trees(labeled_trees, cfg.train.train_splits)
    val_trees = _select_trees(labeled_trees, cfg.validation.val_splits) if cfg.validation.enabled else []
    test_trees = _select_trees(labeled_trees, cfg.test.test_splits) if cfg.test.enabled else []
    if not train_trees:
        train_trees = list(labeled_trees)

    all_selected = list(train_trees) + list(val_trees) + list(test_trees)
    prepared_all, embedding_dim = _prepare_trees(all_selected, embedding_client=embedding_client)
    by_doc_split = {(item.tree.doc_id, item.split): item for item in prepared_all}
    train_items = [by_doc_split[(tree.doc_id, _tree_split(tree))] for tree in train_trees if (tree.doc_id, _tree_split(tree)) in by_doc_split]
    val_items = [by_doc_split[(tree.doc_id, _tree_split(tree))] for tree in val_trees if (tree.doc_id, _tree_split(tree)) in by_doc_split]
    test_items = [by_doc_split[(tree.doc_id, _tree_split(tree))] for tree in test_trees if (tree.doc_id, _tree_split(tree)) in by_doc_split]

    model = EmbeddingCoordinateFNOTreeRegressor(
        embedding_dim=embedding_dim,
        hidden_channels=int(cfg.model.hidden_channels),
        n_modes=int(cfg.model.n_modes),
        n_layers=int(cfg.model.n_layers),
        head_hidden_dim=int(cfg.model.head_hidden_dim),
        target_min=float(cfg.model.target_min),
        target_max=float(cfg.model.target_max),
        merge_mode=str(cfg.model.merge_mode),
        merge_gate_hidden_dim=int(cfg.model.merge_gate_hidden_dim),
        root_readout=str(cfg.model.root_readout),
        root_readout_k=int(cfg.model.root_readout_k),
        root_readout_attn_temp=float(cfg.model.root_readout_attn_temp),
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg.optimizer.learning_rate),
        weight_decay=float(cfg.optimizer.weight_decay),
    )
    grad_clip = float(cfg.optimizer.grad_clip_norm or 0.0)
    best_val = float("inf")
    best_epoch = -1
    losses: List[Dict[str, Any]] = []
    start = time.time()

    for epoch in range(int(cfg.train.epochs)):
        model.train()
        order = list(range(len(train_items)))
        if cfg.train.shuffle:
            random.shuffle(order)
        epoch_losses: List[float] = []
        for start_idx in range(0, len(order), int(max(1, cfg.train.batch_size))):
            batch = [train_items[idx] for idx in order[start_idx : start_idx + int(max(1, cfg.train.batch_size))]]
            optimizer.zero_grad()
            loss = _batch_loss(model, batch, device=device, cfg=cfg)
            loss.backward()
            if grad_clip > 0.0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            epoch_losses.append(float(loss.detach().cpu().item()))
        epoch_payload: Dict[str, Any] = {
            "epoch": int(epoch),
            "train_loss": float(np.mean(epoch_losses)) if epoch_losses else 0.0,
        }
        if val_items and ((epoch + 1) % int(max(1, cfg.validation.eval_every)) == 0 or epoch == int(cfg.train.epochs) - 1):
            val_metrics = _evaluate_split(model, val_items, device=device, cfg=cfg)
            val_mae = val_metrics.get("node_mae_1_7")
            epoch_payload["val_node_mae_1_7"] = val_mae
            if val_mae is not None and float(val_mae) < best_val:
                best_val = float(val_mae)
                best_epoch = int(epoch)
                torch.save(model.state_dict(), output_dir / "embedding_fno_best.pt")
        losses.append(epoch_payload)

    if best_epoch < 0:
        torch.save(model.state_dict(), output_dir / "embedding_fno_best.pt")
    torch.save(model.state_dict(), output_dir / "embedding_fno_final.pt")

    metrics = {
        "train": _evaluate_split(
            model,
            train_items,
            device=device,
            cfg=cfg,
            output_path=output_dir / "node_predictions_train.jsonl",
            full_tree_trace_path=output_dir / "full_tree_traces_train.jsonl",
        ),
        "val": _evaluate_split(
            model,
            val_items,
            device=device,
            cfg=cfg,
            output_path=output_dir / "node_predictions_val.jsonl",
            full_tree_trace_path=output_dir / "full_tree_traces_val.jsonl",
        ) if val_items else {},
        "test": _evaluate_split(
            model,
            test_items,
            device=device,
            cfg=cfg,
            output_path=output_dir / "node_predictions_test.jsonl",
            full_tree_trace_path=output_dir / "full_tree_traces_test.jsonl",
        ) if test_items else {},
        "losses": losses,
        "best_epoch": int(best_epoch),
        "best_val_node_mae_1_7": None if not math.isfinite(best_val) else float(best_val),
        "training_time_seconds": float(time.time() - start),
    }
    canonical_trace_path = output_dir / "full_tree_traces_test.jsonl"
    if not test_items and val_items:
        canonical_trace_path = output_dir / "full_tree_traces_val.jsonl"
    if not test_items and not val_items:
        canonical_trace_path = output_dir / "full_tree_traces_train.jsonl"
    artifacts = {
        "best_checkpoint": str(output_dir / "embedding_fno_best.pt"),
        "final_checkpoint": str(output_dir / "embedding_fno_final.pt"),
        "train_predictions": str(output_dir / "node_predictions_train.jsonl"),
        "val_predictions": str(output_dir / "node_predictions_val.jsonl"),
        "test_predictions": str(output_dir / "node_predictions_test.jsonl"),
        "train_full_tree_traces": str(output_dir / "full_tree_traces_train.jsonl"),
        "val_full_tree_traces": str(output_dir / "full_tree_traces_val.jsonl"),
        "test_full_tree_traces": str(output_dir / "full_tree_traces_test.jsonl"),
        "full_tree_traces_jsonl": str(canonical_trace_path),
        "metrics": str(output_dir / "embedding_fno_metrics.json"),
        "full_tree_metrics_json": str(output_dir / "embedding_fno_metrics.json"),
    }
    result = EmbeddingFNOFitResult(
        output_dir=str(output_dir),
        embedding_dim=int(embedding_dim),
        train_count=len(train_items),
        val_count=len(val_items),
        test_count=len(test_items),
        metrics=metrics,
        artifacts=artifacts,
        config=config_to_dict(cfg),
    )
    (output_dir / "embedding_fno_metrics.json").write_text(
        json.dumps(result.to_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return result


__all__ = [
    "EmbeddingCoordinateFNOTreeRegressor",
    "EmbeddingFNOFitResult",
    "EmbeddingFNOModelConfig",
    "EmbeddingFNOObjectiveConfig",
    "EmbeddingFNOTrainConfig",
    "fit_embedding_fno_node_regressor",
]
