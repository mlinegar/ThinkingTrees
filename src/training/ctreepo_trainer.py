"""
Training loop for CTreePO: learned mergeable sketches over multilingual embeddings.

Trains a CTreePOModel on manifesto documents with RILE supervision at the root
(dataset ground truth) and optionally at internal nodes (LLM oracle, cached).

Usage (from scripts/train_ctreepo.py):
    trainer = CTreePOTrainer(config)
    trainer.prepare_data(manifesto_ids)
    result = trainer.train()
"""

from __future__ import annotations

import json
import logging
import math
import random
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

try:
    import torch
    import torch.nn.functional as F
    import torch.optim as optim
except ImportError:
    raise ImportError("PyTorch required. Install with: pip install torch>=2.0.0")

from src.tree.ctreepo_model import (
    CTreePOConfig,
    CTreePOModel,
    associativity_penalty,
    contrastive_loss,
    normalize_target,
    readout_aggregation_penalty,
)
from src.tree.embedding_tree import (
    EmbeddingTreeNode,
    build_tree_from_text,
    collect_sketches,
    forward_ctreepo,
    get_root_sketch,
)
from src.core.provenance import (
    DATASET_SOURCE,
    normalize_truth_label_source,
)
from src.tree.compositional_learning import (
    CompositionalLearningProblemSpec,
    SupervisionDeliveryMode,
    full_document_supervision_channel,
    oracle_query_policy,
    sampled_substructure_supervision_channel,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Training config
# ---------------------------------------------------------------------------


@dataclass
class CTreePOTrainingConfig:
    """Training hyperparameters for CTreePO."""

    # Model
    model: CTreePOConfig = field(default_factory=CTreePOConfig)

    # Data
    window_size: int = 1200
    window_overlap: int = 150
    merge_drift_threshold: Optional[float] = None  # 0.03 to enable

    # Training
    batch_size: int = 4
    n_epochs: int = 50
    lr: float = 1e-3
    weight_decay: float = 1e-4
    optimizer: str = "adamw"   # adam | adamw
    scheduler: str = "cosine"  # none | cosine | linear
    min_lr: float = 1e-5
    warmup_epochs: int = 3
    grad_clip_norm: float = 1.0
    early_stopping_patience: int = 12
    early_stopping_min_delta: float = 0.0
    uncertainty_z_score: float = 1.96
    min_interval_std: float = 0.5
    n_audit: int = 5
    seed: int = 42

    # Loss weights
    root_weight: float = 1.0
    audit_weight: float = 0.5
    leaf_audit_weight: float = 0.0
    pseudo_weight: float = 0.1
    assoc_weight: float = 0.01
    contrastive_weight: float = 0.1
    consistency_weight: float = 0.05
    local_law_violation_threshold: float = 10.0
    require_local_law_supervision: bool = False

    # Evaluation
    eval_every: int = 5
    device: str = "auto"


# ---------------------------------------------------------------------------
# Training result
# ---------------------------------------------------------------------------


@dataclass
class TrainingResult:
    """Result of a CTreePO training run."""

    config: Dict[str, Any] = field(default_factory=dict)
    train_losses: List[float] = field(default_factory=list)
    eval_metrics: List[Dict[str, Any]] = field(default_factory=list)
    best_epoch: int = 0
    best_root_mae: float = float("inf")
    stopped_early: bool = False
    epochs_completed: int = 0
    training_time_seconds: float = 0.0
    local_law_summary: Dict[str, Any] = field(default_factory=dict)
    compositional_learning_problem: Dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Eval metrics
# ---------------------------------------------------------------------------


@dataclass
class CTreePOEvalMetrics:
    epoch: int
    root_mae: float              # MAE on original RILE scale [-100, +100]
    root_mse: float
    root_mae_normalized: float
    interval_coverage_95: float
    interval_mean_width_95: float
    confidence_calibration_error: float
    node_oracle_label_rate: float
    node_oracle_mae: float
    leaf_oracle_mae: float
    merge_oracle_mae: float
    leaf_violation_rate: float
    merge_violation_rate: float
    leaf_oracle_count: int
    merge_oracle_count: int
    n_docs: int
    per_doc: List[Dict[str, Any]] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------


class CTreePOTrainer:
    """Trains a CTreePOModel on manifesto data with embedding trees."""

    def __init__(
        self,
        config: CTreePOTrainingConfig,
        embedding_client: Any = None,
        node_oracle_predictor: Optional[Callable[[str], float]] = None,
        node_oracle_source_kind: str = "none",
        node_oracle_source_spec: Optional[str] = None,
    ):
        self.config = config
        self.embedding_client = embedding_client
        self.node_oracle_predictor = node_oracle_predictor
        self.node_oracle_source_kind = str(node_oracle_source_kind or "none").strip().lower() or "none"
        self.node_oracle_source_spec = (
            str(node_oracle_source_spec).strip() if node_oracle_source_spec else None
        )
        self.model = CTreePOModel(config.model)
        self.device = self._resolve_device(getattr(config, "device", "auto"))
        self.model.to(self.device)
        self.rng = random.Random(config.seed)
        self._node_oracle_cache: Dict[str, float] = {}
        self._warned_missing_leaf_labels = False
        self._warned_missing_merge_labels = False

        # Pre-built trees (cached after prepare_data)
        self.train_trees: List[Tuple[List[EmbeddingTreeNode], float, str]] = []  # (nodes, rile, doc_id)
        self.val_trees: List[Tuple[List[EmbeddingTreeNode], float, str]] = []

    def _predict_node_oracle_score(self, text: str) -> Optional[float]:
        if self.node_oracle_predictor is None:
            return None
        rendered = str(text or "")
        if not rendered.strip():
            return None
        cached = self._node_oracle_cache.get(rendered)
        if cached is not None:
            return float(cached)
        try:
            score = float(self.node_oracle_predictor(rendered))
        except Exception as exc:
            logger.warning("Node oracle predictor failed on span length=%d: %s", len(rendered), exc)
            return None
        self._node_oracle_cache[rendered] = score
        return score

    def _label_tree_nodes_with_oracle_scores(
        self,
        nodes: List[EmbeddingTreeNode],
        *,
        doc_id: str,
    ) -> Dict[str, int]:
        leaf_count = 0
        merge_count = 0
        if self.node_oracle_predictor is None:
            return {"leaf": 0, "merge": 0, "total": 0}

        for node in nodes:
            score = self._predict_node_oracle_score(node.text_span)
            if score is None:
                continue
            node.oracle_scores["rile"] = float(score)
            if node.is_leaf:
                leaf_count += 1
            else:
                merge_count += 1

        total = leaf_count + merge_count
        if total > 0:
            logger.info(
                "Labeled %s with node oracle scores: leaves=%d internal=%d",
                doc_id,
                leaf_count,
                merge_count,
            )
        return {"leaf": leaf_count, "merge": merge_count, "total": total}

    def _tree_local_law_summary(
        self,
        trees: Sequence[Tuple[List[EmbeddingTreeNode], float, str]],
    ) -> Dict[str, Any]:
        capability_report = (
            self.model.capability_report().to_dict()
            if hasattr(self.model, "capability_report")
            else None
        )
        total_nodes = 0
        total_leaves = 0
        total_internal = 0
        labeled_leaves = 0
        labeled_internal = 0

        for nodes, _rile, _doc_id in trees:
            total_nodes += len(nodes)
            for node in nodes:
                if node.is_leaf:
                    total_leaves += 1
                    if "rile" in node.oracle_scores:
                        labeled_leaves += 1
                else:
                    total_internal += 1
                    if "rile" in node.oracle_scores:
                        labeled_internal += 1

        compositional_learning_problem = self._compositional_learning_problem(
            total_leaves=int(total_leaves),
            total_internal=int(total_internal),
            labeled_leaves=int(labeled_leaves),
            labeled_internal=int(labeled_internal),
            operator_capabilities=capability_report,
        )

        return {
            "node_oracle_predictor_attached": bool(self.node_oracle_predictor is not None),
            "node_label_source_kind": str(self.node_oracle_source_kind),
            "node_label_source_spec": self.node_oracle_source_spec,
            "total_nodes": int(total_nodes),
            "total_leaves": int(total_leaves),
            "total_internal": int(total_internal),
            "labeled_leaves": int(labeled_leaves),
            "labeled_internal": int(labeled_internal),
            "leaf_label_rate": (
                float(labeled_leaves) / float(total_leaves) if total_leaves > 0 else 0.0
            ),
            "merge_label_rate": (
                float(labeled_internal) / float(total_internal) if total_internal > 0 else 0.0
            ),
            "requested_weights": {
                "root_weight": float(self.config.root_weight),
                "leaf_audit_weight": float(self.config.leaf_audit_weight),
                "merge_audit_weight": float(self.config.audit_weight),
                "consistency_weight": float(self.config.consistency_weight),
                "assoc_weight": float(self.config.assoc_weight),
                "contrastive_weight": float(self.config.contrastive_weight),
            },
            "operator_capabilities": capability_report,
            "require_local_law_supervision": bool(
                getattr(self.config, "require_local_law_supervision", False)
            ),
            "objective": {
                "root_supervision": bool(float(self.config.root_weight) > 0.0),
                "leaf_supervision": bool(
                    float(self.config.leaf_audit_weight) > 0.0 and labeled_leaves > 0
                ),
                "merge_supervision": bool(
                    float(self.config.audit_weight) > 0.0 and labeled_internal > 0
                ),
                "idempotence_supervision": False,
                "proxy_readout_aggregation_penalty": bool(
                    float(self.config.consistency_weight) > 0.0
                ),
                "proxy_associativity_penalty": bool(float(self.config.assoc_weight) > 0.0),
            },
            "violation_threshold_raw": float(self.config.local_law_violation_threshold),
            "compositional_learning_problem": compositional_learning_problem,
            "notes": [
                "Leaf/internal node supervision is only active when node-span oracle labels are attached.",
                "Preferred local-law path: task-provided exact span oracle via --local-law-oracle task, or an explicit mechanical callback via --local-law-oracle.",
                "Model-backed teacher labeling is a fallback label source, not a requirement for neural-operator training.",
                "CTreePO remains a proxy-only operator: idempotence is not directly supervised because there is no theorem-domain decode/resummary path.",
                "See operator_capabilities for the architecture-level theorem/proxy split.",
            ],
        }

    def _compositional_learning_problem(
        self,
        *,
        total_leaves: int,
        total_internal: int,
        labeled_leaves: int,
        labeled_internal: int,
        operator_capabilities: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        capability_report = (
            self.model.capability_report()
            if hasattr(self.model, "capability_report")
            else None
        )
        sampled_active = bool(labeled_leaves > 0 or labeled_internal > 0)
        sampled_label_source = normalize_truth_label_source(
            self.node_oracle_source_kind,
            default=(
                "oracle"
                if bool(self.node_oracle_predictor is not None)
                else "unknown"
            ),
        )
        targeted_laws: List[Any] = []
        if float(self.config.leaf_audit_weight) > 0.0:
            from src.core.ops_checks import LawKind

            targeted_laws.append(LawKind.L1_LEAF)
        if float(self.config.audit_weight) > 0.0:
            from src.core.ops_checks import LawKind

            targeted_laws.append(LawKind.L2_MERGE)

        problem = CompositionalLearningProblemSpec(
            name="ctreepo_local_law_training",
            document_type_name="documents",
            theorem_domain_name="span_summary_objects",
            operator_name=(
                capability_report.operator_name
                if capability_report is not None
                else type(self.model).__name__
            ),
            operator_capabilities=capability_report,
            supervision_channels=(
                full_document_supervision_channel(
                    name="root_document_labels",
                    target_name="document_target",
                    active=bool(float(self.config.root_weight) > 0.0),
                    label_source=DATASET_SOURCE,
                    notes=(
                        "Whole-document labels supervise the root prediction directly.",
                    ),
                ),
                sampled_substructure_supervision_channel(
                    name="sampled_node_labels",
                    target_name="node_span_target",
                    active=sampled_active,
                    label_source=sampled_label_source,
                    delivery_mode=(
                        SupervisionDeliveryMode.ONLINE_ORACLE_QUERY
                        if bool(self.node_oracle_predictor is not None)
                        else SupervisionDeliveryMode.OFFLINE_LOGGED
                    ),
                    query_policy=(
                        oracle_query_policy(
                            name="node_oracle_query_policy",
                            query_unit_name="tree_nodes",
                            selection_strategy="all_observed_tree_nodes",
                            adaptive=False,
                            budget={
                                "observed_labeled_leaves": int(labeled_leaves),
                                "observed_labeled_internal": int(labeled_internal),
                            },
                            propensity_field_name="propensity",
                            logs_realized_propensities=False,
                            supports_ipw_estimation=False,
                            notes=(
                                "Current trainer queries the attached node oracle callback while preparing trees.",
                                "This is not yet an IPW-logged online policy; the shared schema leaves room for that upgrade.",
                            ),
                        )
                        if bool(self.node_oracle_predictor is not None)
                        else None
                    ),
                    targeted_laws=tuple(targeted_laws),
                    requires_propensity_logging=False,
                    supports_unbiased_risk=False,
                    notes=(
                        "Leaf and internal-node labels are attached to sampled tree units during training.",
                        "Current trainer uses sampled node supervision for optimization but does not persist per-sample propensities.",
                        "Idempotence is not directly supervised because there is no theorem-domain decode/resummary path.",
                    ),
                ),
            ),
            notes=(
                f"observed_labeled_leaves={int(labeled_leaves)}/{int(total_leaves)}",
                f"observed_labeled_internal={int(labeled_internal)}/{int(total_internal)}",
                "This spec is backend-agnostic: it records supervision channels separately from theorem-backed operator assumptions.",
            ),
        )
        payload = problem.to_dict()
        if operator_capabilities is not None and payload.get("operator_capabilities") is None:
            payload["operator_capabilities"] = dict(operator_capabilities)
        return payload

    def _validate_required_local_law_supervision(self) -> None:
        if not bool(getattr(self.config, "require_local_law_supervision", False)):
            return

        train_summary = self._tree_local_law_summary(self.train_trees)
        missing: List[str] = []
        if float(self.config.leaf_audit_weight) > 0.0 and int(train_summary["labeled_leaves"]) <= 0:
            missing.append("leaf oracle labels")
        if float(self.config.audit_weight) > 0.0 and int(train_summary["labeled_internal"]) <= 0:
            missing.append("internal-node oracle labels")

        if not missing:
            return

        source_status = (
            "attached" if bool(train_summary["node_oracle_predictor_attached"]) else "missing"
        )
        raise ValueError(
            "Local-law supervision was required but inactive for the training split: "
            f"missing {', '.join(missing)}; "
            f"node_label_source={str(train_summary.get('node_label_source_kind', 'none'))}; "
            f"node_oracle_predictor={source_status}; "
            f"leaf_audit_weight={float(self.config.leaf_audit_weight):.6g}; "
            f"merge_audit_weight={float(self.config.audit_weight):.6g}; "
            f"labeled_leaves={int(train_summary['labeled_leaves'])}; "
            f"labeled_internal={int(train_summary['labeled_internal'])}. "
            "Attach a task-provided local-law oracle, supply an explicit callback, explicitly opt into model-backed teacher labeling, "
            "or set the corresponding local-law weights to zero."
        )

    @staticmethod
    def _resolve_device(requested: Any) -> torch.device:
        raw = str(requested or "auto").strip().lower()
        if raw and raw != "auto":
            return torch.device(raw)
        if torch.cuda.is_available():
            try:
                _ = torch.zeros(1, device="cuda")
                return torch.device("cuda")
            except Exception:
                logger.warning("CUDA auto-detected but allocation failed; falling back to CPU.")
        return torch.device("cpu")

    def prepare_trees_from_samples(
        self,
        samples: List[Any],
        split: str = "train",
    ) -> int:
        """Build embedding trees for a set of ManifestoSample objects.

        Args:
            samples: List of ManifestoSample (must have .text and .rile).
            split: "train" or "val".

        Returns:
            Number of trees built.
        """
        if self.embedding_client is None:
            raise ValueError("embedding_client required for prepare_trees_from_samples")

        trees = []
        for sample in samples:
            try:
                nodes = build_tree_from_text(
                    text=sample.text,
                    embedding_client=self.embedding_client,
                    window_size=self.config.window_size,
                    window_overlap=self.config.window_overlap,
                    merge_drift_threshold=self.config.merge_drift_threshold,
                )
                self._label_tree_nodes_with_oracle_scores(nodes, doc_id=str(sample.manifesto_id))
                trees.append((nodes, float(sample.rile), sample.manifesto_id))
                logger.info(
                    "Built tree for %s: %d nodes, %d leaves, RILE=%.1f",
                    sample.manifesto_id,
                    len(nodes),
                    sum(1 for n in nodes if n.is_leaf),
                    sample.rile,
                )
            except Exception as e:
                logger.error("Failed to build tree for %s: %s", getattr(sample, "manifesto_id", "?"), e)

        if split == "train":
            self.train_trees = trees
        else:
            self.val_trees = trees

        return len(trees)

    def prepare_trees_from_precomputed(
        self,
        embeddings_by_doc: Dict[str, Tuple[List[List[float]], List[Tuple[int, int]], str, float]],
        split: str = "train",
    ) -> int:
        """Build trees from pre-computed embeddings (no embedding client needed).

        Args:
            embeddings_by_doc: {doc_id: (embeddings, windows, text, rile)}
            split: "train" or "val"
        """
        from src.tree.embedding_tree import build_embedding_tree

        trees = []
        for doc_id, (embs, windows, text, rile) in embeddings_by_doc.items():
            nodes = build_embedding_tree(text, embs, windows)
            self._label_tree_nodes_with_oracle_scores(nodes, doc_id=str(doc_id))
            trees.append((nodes, rile, doc_id))

        if split == "train":
            self.train_trees = trees
        else:
            self.val_trees = trees
        return len(trees)

    def train_step(
        self,
        batch_trees: List[Tuple[List[EmbeddingTreeNode], float, str]],
        optimizer: optim.Optimizer,
    ) -> float:
        """One training step over a batch of document trees."""
        self.model.train()
        optimizer.zero_grad()

        total_loss = torch.tensor(0.0, device=self.device)
        n_terms = 0
        cfg = self.config

        all_root_sketches: List[torch.Tensor] = []
        all_root_targets: List[float] = []

        for nodes, rile, doc_id in batch_trees:
            # Forward pass: set sketch on each node
            forward_ctreepo(self.model, nodes)

            root_sketch = get_root_sketch(nodes)
            all_root_sketches.append(root_sketch)
            all_root_targets.append(rile)

            # --- Root loss (dataset ground truth) ---
            pred_norm = self.model.predict_normalized(root_sketch, "rile")
            target_norm = normalize_target(
                rile, cfg.model.target_min, cfg.model.target_max
            )
            target_t = torch.tensor([target_norm], dtype=torch.float32, device=pred_norm.device)
            root_loss = ((pred_norm - target_t) ** 2).sum()
            total_loss = total_loss + cfg.root_weight * root_loss
            n_terms += 1

            # --- Leaf preservation loss (C1-style span supervision when labels exist) ---
            leaf_indices = [
                i for i, node in enumerate(nodes)
                if node.is_leaf and "rile" in node.oracle_scores
            ]
            if leaf_indices and float(cfg.leaf_audit_weight) > 0.0:
                n_to_sample = min(cfg.n_audit, len(leaf_indices))
                sampled = self.rng.sample(leaf_indices, k=n_to_sample)
                for idx in sampled:
                    node = nodes[idx]
                    pred = self.model.predict_normalized(node.sketch, "rile")
                    target = normalize_target(
                        node.oracle_scores["rile"],
                        cfg.model.target_min,
                        cfg.model.target_max,
                    )
                    target_t = torch.tensor([target], dtype=torch.float32, device=pred.device)
                    leaf_loss = ((pred - target_t) ** 2).sum()
                    total_loss = total_loss + cfg.leaf_audit_weight * leaf_loss
                    n_terms += 1
            elif float(cfg.leaf_audit_weight) > 0.0 and not self._warned_missing_leaf_labels:
                logger.warning(
                    "CTreePO leaf local-law weight is positive, but no leaf oracle scores were attached. "
                    "Leaf preservation supervision is inactive."
                )
                self._warned_missing_leaf_labels = True

            # --- Merge preservation loss (C3-style span supervision when labels exist) ---
            internal_indices = [
                i for i, node in enumerate(nodes)
                if not node.is_leaf and node.oracle_scores
            ]
            if internal_indices:
                n_to_sample = min(cfg.n_audit, len(internal_indices))
                sampled = self.rng.sample(internal_indices, k=n_to_sample)
                for idx in sampled:
                    node = nodes[idx]
                    if "rile" in node.oracle_scores:
                        pred = self.model.predict_normalized(node.sketch, "rile")
                        target = normalize_target(
                            node.oracle_scores["rile"],
                            cfg.model.target_min,
                            cfg.model.target_max,
                        )
                        target_t = torch.tensor([target], dtype=torch.float32, device=pred.device)
                        audit_loss = ((pred - target_t) ** 2).sum()
                        total_loss = total_loss + cfg.audit_weight * audit_loss
                        n_terms += 1
            elif float(cfg.audit_weight) > 0.0 and not self._warned_missing_merge_labels:
                logger.warning(
                    "CTreePO merge local-law weight is positive, but no internal-node oracle scores were attached. "
                    "Merge preservation supervision is inactive."
                )
                self._warned_missing_merge_labels = True

            # --- Proxy-only readout aggregation loss ---
            if cfg.consistency_weight > 0:
                for node in nodes:
                    if not node.is_leaf and node.children is not None:
                        left_idx, right_idx = node.children
                        if left_idx != right_idx:  # skip promoted odd nodes
                            left_node = nodes[left_idx]
                            right_node = nodes[right_idx]
                            left_len = left_node.text_len
                            right_len = right_node.text_len
                            w = left_len / max(left_len + right_len, 1)
                            c_loss = readout_aggregation_penalty(
                                self.model, node.sketch,
                                left_node.sketch, right_node.sketch,
                                left_weight=w, head="rile",
                            )
                            total_loss = total_loss + cfg.consistency_weight * c_loss
                            n_terms += 1

            # --- Associativity regularization ---
            if cfg.assoc_weight > 0:
                leaf_sketches, _ = collect_sketches(nodes)
                if len(leaf_sketches) >= 3:
                    a_loss = associativity_penalty(self.model, leaf_sketches, n_triplets=4)
                    total_loss = total_loss + cfg.assoc_weight * a_loss
                    n_terms += 1

        # --- Cross-document contrastive loss ---
        if cfg.contrastive_weight > 0 and len(all_root_sketches) >= 2:
            c_loss = contrastive_loss(
                all_root_sketches, all_root_targets,
                tau=0.1, similarity_threshold=15.0,
            )
            total_loss = total_loss + cfg.contrastive_weight * c_loss
            n_terms += 1

        if n_terms > 0:
            loss_val = total_loss / n_terms
            loss_val.backward()
            grad_clip = float(getattr(cfg, "grad_clip_norm", 0.0) or 0.0)
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=grad_clip)
            optimizer.step()
            return loss_val.item()
        return 0.0

    @torch.no_grad()
    def evaluate(
        self,
        trees: Optional[List[Tuple[List[EmbeddingTreeNode], float, str]]] = None,
        epoch: int = 0,
    ) -> CTreePOEvalMetrics:
        """Evaluate model on a set of document trees."""
        self.model.eval()
        if trees is None:
            trees = self.val_trees if self.val_trees else self.train_trees

        errors: List[float] = []
        sq_errors: List[float] = []
        norm_errors: List[float] = []
        covered_95: List[float] = []
        width_95: List[float] = []
        confidence_errors: List[float] = []
        all_node_oracle_errors: List[float] = []
        leaf_oracle_errors: List[float] = []
        merge_oracle_errors: List[float] = []
        per_doc: List[Dict[str, Any]] = []
        total_nodes = 0
        total_labeled_nodes = 0
        violation_threshold = float(getattr(self.config, "local_law_violation_threshold", 10.0) or 10.0)

        for nodes, rile, doc_id in trees:
            forward_ctreepo(self.model, nodes)
            root_sketch = get_root_sketch(nodes)
            pred = self.model.predict(root_sketch, "rile")
            pred_norm = self.model.predict_normalized(root_sketch, "rile")
            confidence = self.model.predict_confidence(root_sketch, "rile")
            _, lower95, upper95, std95 = self.model.predict_interval(
                root_sketch,
                "rile",
                z_score=float(self.config.uncertainty_z_score),
                min_std=float(self.config.min_interval_std),
            )
            pred_val = pred.item()
            error = abs(pred_val - rile)
            errors.append(error)
            sq_errors.append(error ** 2)
            target_norm = normalize_target(
                rile,
                self.config.model.target_min,
                self.config.model.target_max,
            )
            norm_error = abs(float(pred_norm.item()) - float(target_norm))
            norm_errors.append(norm_error)
            lower_val = float(lower95.item())
            upper_val = float(upper95.item())
            width_val = max(0.0, float(upper_val - lower_val))
            width_95.append(width_val)
            covered_95.append(1.0 if (lower_val <= float(rile) <= upper_val) else 0.0)
            proxy_accuracy = max(0.0, min(1.0, 1.0 - norm_error))
            confidence_errors.append(abs(float(confidence.item()) - proxy_accuracy))
            doc_leaf_oracle_count = 0
            doc_merge_oracle_count = 0
            for node in nodes:
                total_nodes += 1
                if node.sketch is None or "rile" not in node.oracle_scores:
                    continue
                total_labeled_nodes += 1
                node_pred = self.model.predict(node.sketch, "rile")
                node_error = abs(float(node_pred.item()) - float(node.oracle_scores["rile"]))
                all_node_oracle_errors.append(node_error)
                if node.is_leaf:
                    doc_leaf_oracle_count += 1
                    leaf_oracle_errors.append(node_error)
                else:
                    doc_merge_oracle_count += 1
                    merge_oracle_errors.append(node_error)
            per_doc.append({
                "doc_id": doc_id,
                "rile_true": rile,
                "rile_pred": round(pred_val, 2),
                "abs_error": round(error, 2),
                "pred_norm": round(float(pred_norm.item()), 4),
                "confidence": round(float(confidence.item()), 4),
                "pred_interval_95": [round(lower_val, 2), round(upper_val, 2)],
                "pred_std_proxy": round(float(std95.item()), 4),
                "in_interval_95": bool(lower_val <= float(rile) <= upper_val),
                "oracle_labeled_leaves": int(doc_leaf_oracle_count),
                "oracle_labeled_internal": int(doc_merge_oracle_count),
            })

        mae = float(np.mean(errors)) if errors else 0.0
        mse = float(np.mean(sq_errors)) if sq_errors else 0.0
        mae_norm = float(np.mean(norm_errors)) if norm_errors else 0.0
        coverage_95 = float(np.mean(covered_95)) if covered_95 else 0.0
        mean_width_95 = float(np.mean(width_95)) if width_95 else 0.0
        conf_cal_err = float(np.mean(confidence_errors)) if confidence_errors else 0.0
        node_oracle_mae = float(np.mean(all_node_oracle_errors)) if all_node_oracle_errors else float("nan")
        leaf_oracle_mae = float(np.mean(leaf_oracle_errors)) if leaf_oracle_errors else float("nan")
        merge_oracle_mae = float(np.mean(merge_oracle_errors)) if merge_oracle_errors else float("nan")
        leaf_violation_rate = (
            float(np.mean(np.asarray(leaf_oracle_errors, dtype=np.float64) > violation_threshold))
            if leaf_oracle_errors
            else float("nan")
        )
        merge_violation_rate = (
            float(np.mean(np.asarray(merge_oracle_errors, dtype=np.float64) > violation_threshold))
            if merge_oracle_errors
            else float("nan")
        )
        node_oracle_label_rate = (
            float(total_labeled_nodes) / float(total_nodes) if total_nodes > 0 else 0.0
        )

        return CTreePOEvalMetrics(
            epoch=epoch,
            root_mae=mae,
            root_mse=mse,
            root_mae_normalized=mae_norm,
            interval_coverage_95=coverage_95,
            interval_mean_width_95=mean_width_95,
            confidence_calibration_error=conf_cal_err,
            node_oracle_label_rate=node_oracle_label_rate,
            node_oracle_mae=node_oracle_mae,
            leaf_oracle_mae=leaf_oracle_mae,
            merge_oracle_mae=merge_oracle_mae,
            leaf_violation_rate=leaf_violation_rate,
            merge_violation_rate=merge_violation_rate,
            leaf_oracle_count=len(leaf_oracle_errors),
            merge_oracle_count=len(merge_oracle_errors),
            n_docs=len(trees),
            per_doc=per_doc,
        )

    def _make_optimizer(self) -> optim.Optimizer:
        cfg = self.config
        name = str(getattr(cfg, "optimizer", "adamw") or "adamw").strip().lower()
        if name == "adam":
            return optim.Adam(
                self.model.parameters(),
                lr=float(cfg.lr),
                weight_decay=float(cfg.weight_decay),
            )
        return optim.AdamW(
            self.model.parameters(),
            lr=float(cfg.lr),
            weight_decay=float(cfg.weight_decay),
        )

    def _make_scheduler(self, optimizer: optim.Optimizer):
        cfg = self.config
        mode = str(getattr(cfg, "scheduler", "none") or "none").strip().lower()
        if mode in {"", "none", "off"}:
            return None

        total_epochs = max(1, int(cfg.n_epochs))
        warmup_epochs = max(0, min(total_epochs - 1, int(cfg.warmup_epochs)))
        base_lr = max(1e-12, float(cfg.lr))
        min_lr = max(0.0, min(float(cfg.min_lr), base_lr))
        min_ratio = min_lr / base_lr

        def _lr_lambda(epoch_idx: int) -> float:
            epoch_idx = int(max(0, epoch_idx))
            if warmup_epochs > 0 and epoch_idx < warmup_epochs:
                return float(epoch_idx + 1) / float(warmup_epochs)

            denom = max(1, total_epochs - warmup_epochs - 1)
            progress = min(1.0, max(0.0, float(epoch_idx - warmup_epochs) / float(denom)))
            if mode == "linear":
                return float(min_ratio + (1.0 - min_ratio) * (1.0 - progress))
            if mode == "cosine":
                cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
                return float(min_ratio + (1.0 - min_ratio) * cosine)
            return 1.0

        return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=_lr_lambda)

    # -----------------------------------------------------------------
    # Uncertainty-guided audit sampling (Phase 3)
    # -----------------------------------------------------------------

    @torch.no_grad()
    def select_audit_nodes(
        self,
        nodes: List[EmbeddingTreeNode],
        n_audit: int = 5,
        exploration_fraction: float = 0.3,
    ) -> List[int]:
        """Select internal nodes for oracle audit, preferring uncertain sketches.

        Instead of random sampling, this uses sketch readout confidence to
        prioritise nodes where the model is least sure.  A fraction of the
        budget is reserved for random exploration to avoid getting stuck.

        Args:
            nodes: Unified tree nodes (must have sketch set via forward_ctreepo).
            n_audit: Total number of nodes to select.
            exploration_fraction: Fraction of budget for random sampling.

        Returns:
            List of node indices to audit.
        """
        self.model.eval()
        internal_indices = [
            i for i, node in enumerate(nodes) if not node.is_leaf
        ]
        if not internal_indices or n_audit <= 0:
            return []

        n_to_select = min(n_audit, len(internal_indices))
        n_explore = max(1, int(n_to_select * exploration_fraction))
        n_uncertain = n_to_select - n_explore

        # Compute confidence for each internal node
        confidences: List[Tuple[int, float]] = []
        for idx in internal_indices:
            node = nodes[idx]
            if node.sketch is None:
                confidences.append((idx, 0.0))  # no sketch → definitely audit
                continue
            conf = float(self.model.predict_confidence(node.sketch, "rile").item())
            confidences.append((idx, conf))

        # Sort by ascending confidence (lowest first = most uncertain)
        confidences.sort(key=lambda x: x[1])

        # Take top-n_uncertain most uncertain
        selected = set()
        for idx, _ in confidences[:n_uncertain]:
            selected.add(idx)

        # Random exploration from remaining
        remaining = [idx for idx, _ in confidences if idx not in selected]
        if remaining and n_explore > 0:
            explore = self.rng.sample(remaining, k=min(n_explore, len(remaining)))
            selected.update(explore)

        return list(selected)

    @torch.no_grad()
    def populate_sketch_scores(
        self,
        nodes: List[EmbeddingTreeNode],
        head: str = "rile",
    ) -> None:
        """Fill sketch_scores and sketch_confidence on all nodes in-place.

        After ``forward_ctreepo(model, nodes)`` has set sketches, this
        computes readout predictions and confidence for every node, writing
        them into the unified node fields.

        Args:
            nodes: Tree nodes with sketches already set.
            head: Which readout head to use.
        """
        self.model.eval()
        for node in nodes:
            if node.sketch is None:
                continue
            pred = self.model.predict(node.sketch, head)
            node.sketch_scores[head] = round(pred.item(), 2)
            node.sketch_confidence = float(self.model.predict_confidence(node.sketch, head).item())

    def train(self, output_dir: Optional[Path] = None) -> TrainingResult:
        """Run the full training loop.

        Args:
            output_dir: If set, save checkpoints and metrics here.

        Returns:
            TrainingResult with training history.
        """
        if not self.train_trees:
            raise ValueError("No training trees. Call prepare_trees_from_samples() first.")

        cfg = self.config
        self._validate_required_local_law_supervision()
        optimizer = self._make_optimizer()
        scheduler = self._make_scheduler(optimizer)

        if output_dir:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)

        result = TrainingResult(config=asdict(cfg))
        result.local_law_summary = self._tree_local_law_summary(self.train_trees + self.val_trees)
        result.compositional_learning_problem = dict(
            result.local_law_summary.get("compositional_learning_problem", {}) or {}
        )
        best_mae = float("inf")
        best_epoch = 0
        best_epoch_metrics: Optional[Dict[str, Any]] = None
        epochs_since_improve = 0
        start_time = time.time()

        logger.info(
            "Starting CTreePO training: %d train docs, %d val docs, %d epochs (device=%s)",
            len(self.train_trees), len(self.val_trees), cfg.n_epochs,
            self.device,
        )

        for epoch in range(cfg.n_epochs):
            # Shuffle training data
            indices = list(range(len(self.train_trees)))
            self.rng.shuffle(indices)

            epoch_losses: List[float] = []
            for batch_start in range(0, len(indices), cfg.batch_size):
                batch_idx = indices[batch_start:batch_start + cfg.batch_size]
                batch = [self.train_trees[i] for i in batch_idx]
                loss = self.train_step(batch, optimizer)
                epoch_losses.append(loss)

            avg_loss = float(np.mean(epoch_losses)) if epoch_losses else 0.0
            result.train_losses.append(avg_loss)
            result.epochs_completed = int(epoch + 1)
            current_lr = float(optimizer.param_groups[0]["lr"])

            # Evaluation
            if (epoch + 1) % cfg.eval_every == 0 or epoch == cfg.n_epochs - 1:
                eval_trees = self.val_trees if self.val_trees else self.train_trees
                metrics = self.evaluate(eval_trees, epoch=epoch)

                metrics_dict = {
                    "epoch": epoch,
                    "train_loss": round(avg_loss, 6),
                    "learning_rate": round(current_lr, 10),
                    "root_mae": round(metrics.root_mae, 2),
                    "root_mse": round(metrics.root_mse, 2),
                    "root_mae_norm": round(metrics.root_mae_normalized, 6),
                    "interval_coverage_95": round(metrics.interval_coverage_95, 4),
                    "interval_mean_width_95": round(metrics.interval_mean_width_95, 4),
                    "confidence_calibration_error": round(metrics.confidence_calibration_error, 6),
                    "local_law": {
                        "node_oracle_label_rate": round(metrics.node_oracle_label_rate, 6),
                        "node_oracle_mae": round(metrics.node_oracle_mae, 4)
                        if np.isfinite(metrics.node_oracle_mae)
                        else None,
                        "leaf_oracle_mae": round(metrics.leaf_oracle_mae, 4)
                        if np.isfinite(metrics.leaf_oracle_mae)
                        else None,
                        "merge_oracle_mae": round(metrics.merge_oracle_mae, 4)
                        if np.isfinite(metrics.merge_oracle_mae)
                        else None,
                        "leaf_violation_rate": round(metrics.leaf_violation_rate, 6)
                        if np.isfinite(metrics.leaf_violation_rate)
                        else None,
                        "merge_violation_rate": round(metrics.merge_violation_rate, 6)
                        if np.isfinite(metrics.merge_violation_rate)
                        else None,
                        "leaf_oracle_count": int(metrics.leaf_oracle_count),
                        "merge_oracle_count": int(metrics.merge_oracle_count),
                        "violation_threshold_raw": float(cfg.local_law_violation_threshold),
                    },
                    "per_doc": metrics.per_doc,
                }
                result.eval_metrics.append(metrics_dict)

                logger.info(
                    "Epoch %d/%d: loss=%.4f lr=%.3e MAE=%.2f MSE=%.2f cov95=%.3f width95=%.2f cce=%.3f node_label_rate=%.3f leaf_viol=%s merge_viol=%s",
                    epoch + 1, cfg.n_epochs, avg_loss,
                    current_lr,
                    metrics.root_mae, metrics.root_mse,
                    metrics.interval_coverage_95,
                    metrics.interval_mean_width_95,
                    metrics.confidence_calibration_error,
                    metrics.node_oracle_label_rate,
                    (
                        f"{metrics.leaf_violation_rate:.3f}"
                        if np.isfinite(metrics.leaf_violation_rate)
                        else "na"
                    ),
                    (
                        f"{metrics.merge_violation_rate:.3f}"
                        if np.isfinite(metrics.merge_violation_rate)
                        else "na"
                    ),
                )

                if metrics.root_mae < (best_mae - float(cfg.early_stopping_min_delta)):
                    best_mae = metrics.root_mae
                    best_epoch = epoch
                    best_epoch_metrics = dict(metrics_dict)
                    epochs_since_improve = 0
                    if output_dir:
                        torch.save(self.model.state_dict(), output_dir / "best.pt")
                        logger.info("  -> New best (MAE=%.2f), saved checkpoint", best_mae)
                else:
                    epochs_since_improve += 1
                    patience = int(max(0, cfg.early_stopping_patience))
                    if patience > 0 and epochs_since_improve >= patience:
                        result.stopped_early = True
                        logger.info(
                            "Early stopping at epoch %d (no MAE improvement for %d evals; best epoch=%d MAE=%.2f)",
                            epoch + 1,
                            epochs_since_improve,
                            best_epoch + 1,
                            best_mae,
                        )
                        if scheduler is not None:
                            scheduler.step()
                        break

            if scheduler is not None:
                scheduler.step()

        result.best_epoch = best_epoch
        result.best_root_mae = best_mae
        result.training_time_seconds = time.time() - start_time

        logger.info(
            "Training complete: best MAE=%.2f at epoch %d (%.1fs)",
            best_mae, best_epoch + 1, result.training_time_seconds,
        )

        if output_dir:
            # Save final model and training results
            torch.save(self.model.state_dict(), output_dir / "final.pt")
            if best_epoch_metrics is not None:
                (output_dir / "best_metrics.json").write_text(
                    json.dumps(best_epoch_metrics, indent=2, default=str),
                    encoding="utf-8",
                )
            (output_dir / "training_result.json").write_text(
                json.dumps(asdict(result), indent=2, default=str),
                encoding="utf-8",
            )

        return result


# ---------------------------------------------------------------------------
# Sketch extraction (inference mode)
# ---------------------------------------------------------------------------


@torch.no_grad()
def extract_root_sketch(
    model: CTreePOModel,
    text: str,
    embedding_client: Any,
    window_size: int = 1200,
    window_overlap: int = 150,
) -> Tuple[torch.Tensor, float]:
    """Extract root sketch and RILE prediction for a document.

    Returns:
        (root_sketch, rile_prediction)
    """
    model.eval()
    nodes = build_tree_from_text(
        text, embedding_client, window_size, window_overlap
    )
    forward_ctreepo(model, nodes)
    root = get_root_sketch(nodes)
    rile = model.predict(root, "rile").item()
    return root, rile
