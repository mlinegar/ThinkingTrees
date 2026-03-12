"""
CTreePO model: learned mergeable sketches over multilingual embeddings.

Learns a compact sketch (default 32-dim) from Qwen3-Embedding-8B vectors
(4096-dim) that captures political position. The sketch merges bottom-up
through the document's tree structure (same topology as the text merge) and
predicts target scores (RILE, etc.) via linear readout heads.

Architecture:
    LeafProjector:  embedding (4096) -> sketch (d)
    GatedMerge:     (sketch_L, sketch_R) -> sketch_parent
    ReadoutHead:    sketch -> scalar score

The GatedMerge uses a soft gate + residual, making it approximately
associative when the gate is near 0.5. Any regularizers in this module are
proxy-only heuristics; they are not Lean local-law certificates.

Requires: torch >= 2.0
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except ImportError:
    raise ImportError(
        "PyTorch is required for CTreePO model. "
        "Install with: pip install torch>=2.0.0"
    )

from src.core.ops_checks import (
    EvidenceStatus,
    LawCapabilityReport,
    LawKind,
    OperatorCapabilityReport,
)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CTreePOConfig:
    """Architecture and training hyperparameters for CTreePO."""

    embedding_dim: int = 4096
    sketch_dim: int = 32
    hidden_dim: int = 256
    merge_type: str = "gated"  # "gated" | "mlp" | "avg"
    head_names: tuple = ("rile",)

    # Target scale: RILE is [-100, +100], internally normalized to [0, 1].
    target_min: float = -100.0
    target_max: float = 100.0


# ---------------------------------------------------------------------------
# Model components
# ---------------------------------------------------------------------------


class LeafProjector(nn.Module):
    """Projects high-dim embeddings to compact sketch space.

    Linear(embedding_dim, hidden_dim) -> LayerNorm -> ReLU -> Linear(hidden_dim, sketch_dim)
    """

    def __init__(self, embedding_dim: int, sketch_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, sketch_dim),
        )

    def forward(self, embedding: torch.Tensor) -> torch.Tensor:
        return self.net(embedding)


class GatedMerge(nn.Module):
    """Gated combination of two child sketches.

    gate = sigmoid(W_gate @ [left; right] + b_gate)
    merged = gate * left + (1 - gate) * right + residual([left; right])

    The gate provides soft attention to whichever child carries more signal.
    The residual connection allows learning corrections beyond weighted average.
    Approximately associative when gate ~ 0.5.
    """

    def __init__(self, sketch_dim: int):
        super().__init__()
        self.gate = nn.Linear(2 * sketch_dim, sketch_dim)
        self.residual = nn.Linear(2 * sketch_dim, sketch_dim)

    def forward(self, left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
        cat = torch.cat([left, right], dim=-1)
        g = torch.sigmoid(self.gate(cat))
        return g * left + (1 - g) * right + self.residual(cat)


class MLPMerge(nn.Module):
    """MLP merge (same structure as learned_sketch.py).

    cat(left, right) -> Linear(2d, hidden) -> ReLU -> Linear(hidden, d)
    """

    def __init__(self, sketch_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2 * sketch_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, sketch_dim),
        )

    def forward(self, left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([left, right], dim=-1))


class AvgMerge(nn.Module):
    """Simple average merge (baseline)."""

    def forward(self, left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
        return 0.5 * (left + right)


class ResidualGatedMerge(nn.Module):
    """More expressive gated merge with residual MLP correction."""

    def __init__(self, sketch_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.gate = nn.Linear(2 * sketch_dim, sketch_dim)
        self.residual = nn.Sequential(
            nn.Linear(2 * sketch_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, sketch_dim),
        )
        self.norm = nn.LayerNorm(sketch_dim)

    def forward(self, left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
        cat = torch.cat([left, right], dim=-1)
        g = torch.sigmoid(self.gate(cat))
        mixed = g * left + (1 - g) * right
        return self.norm(mixed + self.residual(cat))


class BilinearMerge(nn.Module):
    """Merge with explicit pairwise interactions between child sketches."""

    def __init__(self, sketch_dim: int):
        super().__init__()
        self.cross = nn.Bilinear(sketch_dim, sketch_dim, sketch_dim)
        self.fuse = nn.Linear(3 * sketch_dim, sketch_dim)
        self.norm = nn.LayerNorm(sketch_dim)

    def forward(self, left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
        cross = self.cross(left, right)
        cat = torch.cat([left, right, cross], dim=-1)
        return self.norm(self.fuse(cat))


class ReadoutHead(nn.Module):
    """Linear readout from sketch to scalar, mapped to target scale."""

    def __init__(self, sketch_dim: int, target_min: float = -100.0, target_max: float = 100.0):
        super().__init__()
        self.linear = nn.Linear(sketch_dim, 1)
        self.target_min = target_min
        self.target_max = target_max

    def forward(self, sketch: torch.Tensor) -> torch.Tensor:
        """Returns scalar prediction on [target_min, target_max] scale."""
        raw = torch.sigmoid(self.linear(sketch))  # [0, 1]
        return self.target_min + raw * (self.target_max - self.target_min)

    def forward_normalized(self, sketch: torch.Tensor) -> torch.Tensor:
        """Returns prediction on [0, 1] scale (for loss computation)."""
        return torch.sigmoid(self.linear(sketch))


# ---------------------------------------------------------------------------
# Full model
# ---------------------------------------------------------------------------


_MERGE_CLASSES = {
    "gated": GatedMerge,
    "mlp": MLPMerge,
    "avg": AvgMerge,
    "residual_gated": ResidualGatedMerge,
    "bilinear": BilinearMerge,
}


class CTreePOModel(nn.Module):
    """Complete proxy-only CTreePO model: project -> merge -> readout.

    Usage:
        model = CTreePOModel(CTreePOConfig())
        leaf_sketch = model.encode_leaf(embedding_tensor)
        parent_sketch = model.merge(left_sketch, right_sketch)
        rile_pred = model.predict(root_sketch, "rile")
    """

    def __init__(self, config: CTreePOConfig):
        super().__init__()
        self.config = config
        self.evidence_status = EvidenceStatus.PROXY_ONLY

        self.leaf_projector = LeafProjector(
            config.embedding_dim, config.sketch_dim, config.hidden_dim
        )

        merge_cls = _MERGE_CLASSES.get(config.merge_type)
        if merge_cls is None:
            raise ValueError(
                f"Unknown merge_type={config.merge_type!r}, "
                f"expected one of {list(_MERGE_CLASSES.keys())}"
            )
        if config.merge_type == "avg":
            self.merge_module = merge_cls()
        elif config.merge_type == "mlp":
            self.merge_module = merge_cls(
                config.sketch_dim,
                hidden_dim=max(64, int(config.hidden_dim // 2)),
            )
        elif config.merge_type == "residual_gated":
            self.merge_module = merge_cls(
                config.sketch_dim,
                hidden_dim=max(64, int(config.hidden_dim)),
            )
        else:
            self.merge_module = merge_cls(config.sketch_dim)

        self.heads = nn.ModuleDict()
        for name in config.head_names:
            self.heads[name] = ReadoutHead(
                config.sketch_dim, config.target_min, config.target_max
            )

    def encode_leaf(self, embedding: torch.Tensor) -> torch.Tensor:
        """Project an embedding vector to sketch space."""
        return self.leaf_projector(embedding)

    def merge(self, left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
        """Merge two child sketches into a parent sketch."""
        return self.merge_module(left, right)

    def predict(self, sketch: torch.Tensor, head: str = "rile") -> torch.Tensor:
        """Predict target score on original scale from sketch."""
        return self.heads[head](sketch)

    def predict_normalized(self, sketch: torch.Tensor, head: str = "rile") -> torch.Tensor:
        """Predict target score on [0, 1] scale from sketch."""
        return self.heads[head].forward_normalized(sketch)

    def predict_confidence(self, sketch: torch.Tensor, head: str = "rile") -> torch.Tensor:
        """Confidence proxy in [0,1] based on distance from center uncertainty band."""
        pred_norm = self.predict_normalized(sketch, head=head)
        return 1.0 - 2.0 * torch.abs(pred_norm - 0.5)

    def predict_interval(
        self,
        sketch: torch.Tensor,
        head: str = "rile",
        *,
        z_score: float = 1.96,
        min_std: float = 0.5,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return (mean, lower, upper, std) on target scale.

        Uncertainty is a heuristic interval derived from Bernoulli variance on
        normalized predictions. This gives a conservative, architecture-agnostic
        uncertainty proxy without changing checkpoint format.
        """
        mean = self.predict(sketch, head=head)
        pred_norm = torch.clamp(self.predict_normalized(sketch, head=head), min=1e-6, max=1.0 - 1e-6)
        span = float(self.config.target_max - self.config.target_min)
        std_norm = torch.sqrt(pred_norm * (1.0 - pred_norm))
        std = torch.clamp(std_norm * span, min=float(min_std))
        lower = torch.clamp(mean - float(z_score) * std, min=self.config.target_min, max=self.config.target_max)
        upper = torch.clamp(mean + float(z_score) * std, min=self.config.target_min, max=self.config.target_max)
        return mean, lower, upper, std

    def capability_report(self) -> OperatorCapabilityReport:
        """Structured local-law capability report for the core architecture."""
        return OperatorCapabilityReport(
            operator_name="ctreepo",
            evidence_status=self.evidence_status,
            latent_mergeability_enforced=False,
            tree_nesting_supported=True,
            theorem_domain_decode_available=False,
            theorem_domain_reencode_available=False,
            exact_reduction_supported=False,
            leaf_law=LawCapabilityReport(
                law_kind=LawKind.L1_LEAF,
                available=False,
                evidence_status=self.evidence_status,
                exact=False,
                notes="No theorem-domain summary/decode path is exposed by the base model.",
            ),
            merge_law=LawCapabilityReport(
                law_kind=LawKind.L2_MERGE,
                available=False,
                evidence_status=self.evidence_status,
                exact=False,
                notes="Recursive latent merges are supported, but not certified against theorem-domain spans.",
            ),
            idempotence_law=LawCapabilityReport(
                law_kind=LawKind.L3_IDEMPOTENCE,
                available=False,
                evidence_status=self.evidence_status,
                exact=False,
                notes="The base architecture has no decode/re-encode loop for theorem-backed re-summary checks.",
            ),
            notes=(
                "Tree nesting is available in latent space through encode_leaf/merge.",
                "Regularizers in this module remain proxy-only until a supplied theorem-domain codec/certificate is attached.",
            ),
        )

    def local_law_capabilities(self) -> Dict[str, object]:
        """Backwards-compatible dictionary view of the capability report."""
        return self.capability_report().to_dict()


# ---------------------------------------------------------------------------
# Loss utilities
# ---------------------------------------------------------------------------


def normalize_target(value: float, target_min: float = -100.0, target_max: float = 100.0) -> float:
    """Map target value from [target_min, target_max] to [0, 1]."""
    span = target_max - target_min
    if span == 0:
        return 0.5
    return (value - target_min) / span


def denormalize_prediction(value: float, target_min: float = -100.0, target_max: float = 100.0) -> float:
    """Map prediction from [0, 1] back to [target_min, target_max]."""
    return target_min + value * (target_max - target_min)


def associativity_penalty(
    model: CTreePOModel,
    sketches: Sequence[torch.Tensor],
    n_triplets: int = 4,
) -> torch.Tensor:
    """Proxy-only merge associativity regularizer over random triplets.

    This improves empirical stability but is not a Lean law witness.
    """
    if len(sketches) < 3:
        return torch.tensor(0.0)

    n = len(sketches)
    penalty = torch.tensor(0.0)
    count = 0

    for _ in range(min(n_triplets, n * (n - 1) * (n - 2) // 6)):
        indices = torch.randperm(n)[:3]
        a, b, c = sketches[indices[0]], sketches[indices[1]], sketches[indices[2]]

        left_first = model.merge(model.merge(a, b), c)
        right_first = model.merge(a, model.merge(b, c))
        penalty = penalty + ((left_first - right_first) ** 2).sum()
        count += 1

    return penalty / max(count, 1)


def readout_aggregation_penalty(
    model: CTreePOModel,
    parent_sketch: torch.Tensor,
    left_sketch: torch.Tensor,
    right_sketch: torch.Tensor,
    left_weight: float,
    head: str = "rile",
) -> torch.Tensor:
    """Proxy-only penalty: parent readout tracks weighted child readouts.

    left_weight = len(left_text) / (len(left_text) + len(right_text))
    """
    parent_pred = model.predict_normalized(parent_sketch, head)
    left_pred = model.predict_normalized(left_sketch, head)
    right_pred = model.predict_normalized(right_sketch, head)
    expected = left_weight * left_pred + (1 - left_weight) * right_pred
    return ((parent_pred - expected) ** 2).sum()


def consistency_penalty(
    model: CTreePOModel,
    parent_sketch: torch.Tensor,
    left_sketch: torch.Tensor,
    right_sketch: torch.Tensor,
    left_weight: float,
    head: str = "rile",
) -> torch.Tensor:
    """Deprecated alias for ``readout_aggregation_penalty``."""
    return readout_aggregation_penalty(
        model,
        parent_sketch,
        left_sketch,
        right_sketch,
        left_weight,
        head=head,
    )


def contrastive_loss(
    sketches: List[torch.Tensor],
    targets: List[float],
    tau: float = 0.1,
    similarity_threshold: float = 10.0,
) -> torch.Tensor:
    """Cross-language contrastive loss.

    Documents with similar RILE (|rile_i - rile_j| < threshold) are positives.
    """
    n = len(sketches)
    if n < 2:
        return torch.tensor(0.0)

    mat = torch.stack(sketches, dim=0)  # (n, d)
    mat = F.normalize(mat, dim=1)
    sims = mat @ mat.T / tau  # (n, n)

    loss = torch.tensor(0.0)
    count = 0

    for i in range(n):
        # Find positives: similar RILE
        positives = [
            j for j in range(n)
            if j != i and abs(targets[i] - targets[j]) < similarity_threshold
        ]
        if not positives:
            continue

        for j in positives:
            # InfoNCE: -log(exp(sim(i,j)) / sum_k exp(sim(i,k)))
            numerator = sims[i, j]
            denominator = torch.logsumexp(
                torch.cat([sims[i, :i], sims[i, i + 1:]]), dim=0
            )
            loss = loss - (numerator - denominator)
            count += 1

    return loss / max(count, 1)
