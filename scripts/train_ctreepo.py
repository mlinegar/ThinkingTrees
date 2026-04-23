#!/usr/bin/env python3
"""
CLI entry point for CTreePO training and evaluation.

Trains a CTreePOModel (learned mergeable sketches) on manifesto documents,
using multilingual embeddings from Qwen3-Embedding-8B and RILE supervision.

Usage:
    # Pilot test with 5 parties (requires embedding server on port 8003)
    ./venv/bin/python scripts/train_ctreepo.py --pilot

    # Custom IDs
    ./venv/bin/python scripts/train_ctreepo.py \
        --train-ids 11320_199809 33320_199603 31320_199705 \
        --val-ids 51620_199705 41521_199809

    # Full training with ManifestoDataset
    ./venv/bin/python scripts/train_ctreepo.py \
        --countries 11 33 31 51 41 \
        --party-families 30 50 60 \
        --train-end-year 1995 --val-end-year 2005
"""

from __future__ import annotations

import argparse
import importlib
import json
import logging
import random
import sys
from pathlib import Path
from typing import List, Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.training.reproducibility import (
    configure_reproducibility,
    write_reproducibility_manifest,
)

logger = logging.getLogger(__name__)


# Late-1990s pilot test set — train on all 5 for smoke testing
# (with only 5 docs, a train/val split leaves one side unrepresented)
PILOT_TRAIN_IDS = [
    "11320_199809",   # SAP (Sweden) RILE=-3.5
    "33320_199603",   # PSOE (Spain) RILE=-4.1
    "31320_199705",   # PS (France) RILE=-13.3
    "51620_199705",   # Conservatives (UK) RILE=+25.7
    "41521_199410",   # CDU/CSU (Germany) RILE=+26.8
]
PILOT_VAL_IDS: list[str] = []  # all docs in train for pilot smoke test


def _load_oracle_callable(spec: str):
    if ":" not in str(spec):
        raise ValueError(
            f"Oracle spec must be 'module.path:function_name', got {spec!r}"
        )
    module_path, func_name = str(spec).rsplit(":", 1)
    module = importlib.import_module(module_path)
    fn = getattr(module, func_name)
    if not callable(fn):
        raise TypeError(f"{spec!r} is not callable")
    return fn


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Train CTreePO (learned mergeable sketches) on manifesto data.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Data source
    data_group = parser.add_argument_group("data")
    data_group.add_argument("--pilot", action="store_true",
                            help="Use pilot test set (5 parties, late 1990s)")
    data_group.add_argument("--train-ids", nargs="+", default=None,
                            help="Manifesto IDs for training")
    data_group.add_argument("--val-ids", nargs="+", default=None,
                            help="Manifesto IDs for validation")
    data_group.add_argument("--countries", type=int, nargs="+", default=None,
                            help="Country codes to include")
    data_group.add_argument("--party-families", type=int, nargs="+", default=None,
                            help="Party family codes to include")
    data_group.add_argument("--train-end-year", type=int, default=1995)
    data_group.add_argument("--val-end-year", type=int, default=2005)
    data_group.add_argument(
        "--task",
        type=str,
        default="manifesto_rile",
        help="Task/plugin name used to resolve task-provided local-law oracles.",
    )
    data_group.add_argument(
        "--labeled-tree-artifacts",
        type=Path,
        nargs="+",
        default=None,
        help=(
            "Offline labeled-tree JSON/JSONL artifact(s) from Stage 0 teacher tracing. "
            "When set, manifesto ID selection is skipped and node labels are replayed without live teacher calls."
        ),
    )
    data_group.add_argument(
        "--labeled-tree-train-splits",
        type=str,
        nargs="+",
        default=["train"],
        help="Split names from labeled-tree artifacts used for tree-operator training.",
    )
    data_group.add_argument(
        "--labeled-tree-val-splits",
        type=str,
        nargs="+",
        default=["val"],
        help="Split names from labeled-tree artifacts used for validation.",
    )

    # Model
    model_group = parser.add_argument_group("model")
    model_group.add_argument("--embedding-dim", type=int, default=None,
                             help="Embedding dimensionality override (auto-detected when omitted)")
    model_group.add_argument("--sketch-dim", type=int, default=32)
    model_group.add_argument("--hidden-dim", type=int, default=256)
    model_group.add_argument(
        "--merge-type",
        choices=["gated", "mlp", "avg", "residual_gated", "bilinear"],
        default="gated",
    )
    model_group.add_argument(
        "--tree-model-version",
        choices=["legacy", "v2"],
        default="v2",
        help="Shared CTreePO trainer surface version to use for new training runs.",
    )

    # Training
    train_group = parser.add_argument_group("training")
    train_group.add_argument("--epochs", type=int, default=50)
    train_group.add_argument("--lr", type=float, default=1e-3)
    train_group.add_argument("--optimizer", choices=["adam", "adamw"], default="adamw")
    train_group.add_argument("--scheduler", choices=["none", "cosine", "linear"], default="cosine")
    train_group.add_argument("--min-lr", type=float, default=1e-5)
    train_group.add_argument("--warmup-epochs", type=int, default=3)
    train_group.add_argument("--grad-clip-norm", type=float, default=1.0)
    train_group.add_argument("--early-stopping-patience", type=int, default=12)
    train_group.add_argument("--early-stopping-min-delta", type=float, default=0.0)
    train_group.add_argument("--eval-every", type=int, default=5)
    train_group.add_argument("--uncertainty-z-score", type=float, default=1.96)
    train_group.add_argument("--min-interval-std", type=float, default=0.5)
    train_group.add_argument("--batch-size", type=int, default=4)
    train_group.add_argument("--window-size", type=int, default=1200)
    train_group.add_argument("--window-overlap", type=int, default=150)
    train_group.add_argument("--seed", type=int, default=42)
    train_group.add_argument("--device", type=str, default="auto", help="cpu | cuda | cuda:N | auto")
    train_group.add_argument("--root-weight", type=float, default=1.0)
    train_group.add_argument("--leaf-audit-weight", type=float, default=0.0)
    train_group.add_argument("--merge-audit-weight", type=float, default=0.5)
    train_group.add_argument(
        "--idempotence-weight",
        type=float,
        default=0.0,
        help="Latent proxy C2/L3 term: ||g(z,z)-z||^2. This is not theorem-domain resummary supervision.",
    )
    train_group.add_argument("--local-law-violation-threshold", type=float, default=10.0)
    train_group.add_argument(
        "--require-local-law-supervision",
        action="store_true",
        help="Fail if positive local-law weights are requested but no node oracle labels are attached on the training split.",
    )

    # Embedding
    emb_group = parser.add_argument_group("embedding")
    emb_group.add_argument("--embedding-url", type=str, default=None)
    emb_group.add_argument("--embedding-model", type=str, default=None)

    # Local-law label source
    law_group = parser.add_argument_group("local-law label source")
    law_group.add_argument(
        "--local-law-oracle",
        "--local-law-oracle-module",
        type=str,
        dest="local_law_oracle_spec",
        default=None,
        help=(
            "Node-span label source. Use 'task' for the task/teacher-provided oracle, "
            "or module.path:function_name for an explicit callback. Preferred path is an "
            "exact mechanical task oracle when the setting supplies one."
        ),
    )
    law_group.add_argument(
        "--local-law-teacher-port",
        "--local-law-score-port",
        type=int,
        dest="local_law_score_port",
        default=None,
        help="Optional model-backed teacher endpoint for node-span labeling. Fallback only; disabled by default for this workflow.",
    )
    law_group.add_argument(
        "--local-law-teacher-model",
        "--local-law-score-model",
        type=str,
        dest="local_law_score_model",
        default=None,
        help="Optional model id override for the model-backed teacher labeler backend.",
    )
    law_group.add_argument(
        "--local-law-teacher-max-tokens",
        "--local-law-score-max-tokens",
        type=int,
        dest="local_law_score_max_tokens",
        default=64,
        help="Max tokens for the model-backed teacher labeler.",
    )
    law_group.add_argument(
        "--local-law-teacher-temperature",
        "--local-law-score-temperature",
        type=float,
        dest="local_law_score_temperature",
        default=0.0,
        help="Temperature for the model-backed teacher labeler.",
    )
    law_group.add_argument(
        "--allow-model-based-local-law-labeling",
        "--allow-model-based-local-law-scoring",
        action="store_true",
        dest="allow_model_based_local_law_scoring",
        help="Explicitly opt into model-backed local-law labeling. Without this flag, prefer --local-law-oracle task or an explicit exact callback.",
    )
    law_group.add_argument(
        "--online-local-law-supervision",
        action="store_true",
        help="Queue sampled node-label requests through FeedbackStore instead of blocking tree preparation.",
    )
    law_group.add_argument(
        "--feedback-store",
        type=Path,
        default=None,
        help="Durable JSON FeedbackStore path for online local-law supervision.",
    )
    law_group.add_argument(
        "--online-teacher-worker",
        choices=["off", "on"],
        default="off",
        help="Run a non-blocking in-process worker that answers pending online requests with the configured node oracle.",
    )
    law_group.add_argument(
        "--online-human-only",
        action="store_true",
        help="Alias for online local-law supervision with no teacher worker; humans answer the shared feedback queue.",
    )
    law_group.add_argument("--online-leaf-query-budget-per-epoch", type=int, default=16)
    law_group.add_argument("--online-merge-query-budget-per-epoch", type=int, default=16)
    law_group.add_argument("--online-worker-concurrency", type=int, default=4)

    # Output
    parser.add_argument("--output-dir", type=Path, default=None,
                        help="Output directory (default: outputs/ctreepo/<run_id>)")
    parser.add_argument("--verbose", action="store_true")

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    from src.tasks import get_task
    from src.training.local_law_oracles import (
        normalize_local_law_oracle_spec,
        resolve_task_local_law_oracle,
    )

    args.local_law_oracle_spec = normalize_local_law_oracle_spec(
        getattr(args, "local_law_oracle_spec", None)
    )
    labeled_tree_artifact_paths = [Path(path) for path in (args.labeled_tree_artifacts or [])]
    if bool(args.online_human_only):
        args.online_local_law_supervision = True
        args.online_teacher_worker = "off"

    if labeled_tree_artifact_paths and (
        args.local_law_oracle_spec
        or args.local_law_score_port is not None
        or bool(args.online_local_law_supervision)
    ):
        logger.error(
            "--labeled-tree-artifacts is an offline replay path; do not combine it with "
            "--local-law-oracle, --local-law-teacher-port, or --online-local-law-supervision."
        )
        return 2

    if (
        args.local_law_oracle_spec
        and str(args.local_law_oracle_spec).strip().lower() != "task"
        and args.local_law_score_port is not None
    ):
        logger.error(
            "Choose one local-law supervision source: --local-law-oracle <module.path:function_name> or --local-law-teacher-port, not both."
        )
        return 2

    try:
        task = get_task(str(args.task))
    except Exception as exc:
        logger.error("Failed to load task %r: %s", args.task, exc)
        return 2

    node_oracle_predictor = None
    node_oracle_source_kind = "none"
    node_oracle_source_spec = None
    task_oracle_requested = str(args.local_law_oracle_spec or "").strip().lower() == "task"
    if args.local_law_oracle_spec and not task_oracle_requested:
        try:
            node_oracle_predictor = _load_oracle_callable(args.local_law_oracle_spec)
        except Exception as exc:
            logger.error("Failed to load local-law oracle module %r: %s", args.local_law_oracle_spec, exc)
            return 2
        node_oracle_source_kind = "oracle_callback"
        node_oracle_source_spec = str(args.local_law_oracle_spec)
        logger.info("Explicit local-law oracle callback enabled: %s", args.local_law_oracle_spec)
    elif task_oracle_requested or args.local_law_score_port is not None:
        describe_local_law_oracle = getattr(task, "describe_local_law_oracle", None)
        described_oracle = describe_local_law_oracle() if callable(describe_local_law_oracle) else None
        local_law_oracle_meta = dict(described_oracle) if isinstance(described_oracle, dict) else {}
        if (
            bool(local_law_oracle_meta.get("model_backed"))
            and not args.allow_model_based_local_law_scoring
        ):
            logger.error(
                "Task-provided model-backed local-law labeling is disabled for this workflow. "
                "Pass --allow-model-based-local-law-labeling to opt in intentionally."
            )
            return 2
        try:
            resolution = resolve_task_local_law_oracle(
                task,
                backend_port=args.local_law_score_port,
                backend_model=args.local_law_score_model,
                max_tokens=args.local_law_score_max_tokens,
                temperature=args.local_law_score_temperature,
                strict_parse=True,
            )
        except Exception as exc:
            logger.error(
                "Failed to construct task-provided local-law oracle for task %r: %s",
                args.task,
                exc,
            )
            return 2
        if resolution is None:
            if task_oracle_requested:
                logger.error(
                    "Task %r does not provide a node-span local-law oracle. "
                    "Use --local-law-oracle <module.path:function_name> instead.",
                    args.task,
                )
                return 2
            logger.error(
                "Task %r does not provide a task-scoped node-span oracle, so --local-law-teacher-port "
                "cannot be used here. Use --local-law-oracle <module.path:function_name> for an explicit callback.",
                args.task,
            )
            return 2
        resolution_meta = dict(resolution.metadata)
        node_oracle_predictor = resolution.predictor
        node_oracle_source_kind = str(resolution.source_kind)
        node_oracle_source_spec = resolution.source_spec
        logger.info(
            "Task-provided local-law oracle enabled: task=%s kind=%s spec=%s exact=%s model_backed=%s",
            args.task,
            node_oracle_source_kind,
            node_oracle_source_spec or "none",
            bool(resolution_meta.get("exact")),
            bool(resolution_meta.get("model_backed")),
        )
    else:
        logger.info("Local-law node-span labels disabled; local-law supervision will remain inactive.")

    if bool(args.online_local_law_supervision):
        if str(args.online_teacher_worker) == "on" and node_oracle_predictor is None:
            logger.error(
                "--online-teacher-worker on requires --local-law-oracle or a task/model-backed local-law oracle."
            )
            return 2
        if node_oracle_predictor is None:
            node_oracle_source_kind = "human"
            node_oracle_source_spec = "feedback_store"
        logger.info(
            "Online local-law supervision enabled: teacher_worker=%s human_only=%s",
            args.online_teacher_worker,
            bool(args.online_human_only),
        )

    if labeled_tree_artifact_paths:
        node_oracle_source_kind = "labeled_tree"
        node_oracle_source_spec = ";".join(str(path) for path in labeled_tree_artifact_paths)
        logger.info(
            "Offline labeled-tree distillation enabled: %d artifact path(s)",
            len(labeled_tree_artifact_paths),
        )

    # ------------------------------------------------------------------
    # Resolve output dir
    # ------------------------------------------------------------------
    if args.output_dir is None:
        import datetime
        run_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output_dir = PROJECT_ROOT / "outputs" / "ctreepo" / run_id
    args.output_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Output dir: %s", args.output_dir)
    applied_repro = configure_reproducibility(int(args.seed))

    online_node_oracle_queue = None
    feedback_store_path = args.feedback_store or (args.output_dir / "online_feedback_store.json")
    if bool(args.online_local_law_supervision):
        from src.feedback.store import FeedbackStore
        from src.training.online_node_oracle import (
            OnlineNodeOracleQueue,
            OnlineNodeOracleQueueConfig,
        )

        feedback_store = FeedbackStore(
            storage_path=feedback_store_path,
            autosave=True,
            load_existing=True,
        )
        online_node_oracle_queue = OnlineNodeOracleQueue(
            store=feedback_store,
            config=OnlineNodeOracleQueueConfig(
                leaf_budget_per_epoch=int(args.online_leaf_query_budget_per_epoch),
                merge_budget_per_epoch=int(args.online_merge_query_budget_per_epoch),
                source_kind=str(node_oracle_source_kind),
                source_spec=node_oracle_source_spec,
            ),
            rng=random.Random(int(args.seed)),
        )

    # ------------------------------------------------------------------
    # Resolve manifesto IDs
    # ------------------------------------------------------------------
    train_ids: List[str] = []
    val_ids: List[str] = []

    if labeled_tree_artifact_paths:
        logger.info("Using labeled-tree artifacts; skipping manifesto ID selection.")
    elif args.pilot:
        train_ids = list(PILOT_TRAIN_IDS)
        val_ids = list(PILOT_VAL_IDS)
        logger.info("Pilot mode: %d train + %d val", len(train_ids), len(val_ids))
    elif args.train_ids:
        train_ids = args.train_ids
        val_ids = args.val_ids or []
    elif args.countries or args.party_families:
        from src.tasks.manifesto.data_loader import ManifestoDataset
        ds = ManifestoDataset(countries=args.countries)
        t_ids, v_ids, te_ids = ds.create_temporal_split(
            train_end_year=args.train_end_year,
            val_end_year=args.val_end_year,
        )
        # Filter by party family if specified
        if args.party_families:
            pf_set = set(args.party_families)
            def _filter_by_family(ids):
                return [
                    mid for mid in ids
                    if (s := ds.get_sample(mid)) is not None and s.party_family in pf_set
                ]
            train_ids = _filter_by_family(t_ids)
            val_ids = _filter_by_family(v_ids)
        else:
            train_ids = t_ids
            val_ids = v_ids
        logger.info("Dataset: %d train + %d val", len(train_ids), len(val_ids))
    else:
        logger.error("Specify --pilot, --train-ids, or --countries")
        return 2

    if not train_ids and not labeled_tree_artifact_paths:
        logger.error("No training IDs found")
        return 2

    # ------------------------------------------------------------------
    # Load samples
    # ------------------------------------------------------------------
    if labeled_tree_artifact_paths:
        train_samples = []
        val_samples = []
        logger.info("Manifesto sample loading skipped for labeled-tree replay.")
    else:
        from src.tasks.manifesto.data_loader import ManifestoDataset

        ds = ManifestoDataset()
        train_samples = [s for mid in train_ids if (s := ds.get_sample(mid)) is not None]
        val_samples = [s for mid in val_ids if (s := ds.get_sample(mid)) is not None]

        logger.info("Loaded %d train samples, %d val samples", len(train_samples), len(val_samples))

        for s in train_samples + val_samples:
            logger.info("  %s: %s (%s) RILE=%.1f (%d chars)",
                         s.manifesto_id, s.party_abbrev, s.country_name, s.rile, len(s.text))

    # ------------------------------------------------------------------
    # Set up embedding client
    # ------------------------------------------------------------------
    from src.config.settings import get_embedding_model, get_embedding_url, load_settings
    from src.training.embedding_proxy import VLLMEmbeddingClient

    settings = load_settings()
    api_base = (args.embedding_url or get_embedding_url(settings)).rstrip("/")
    model_name = args.embedding_model or get_embedding_model(settings) or None

    client = VLLMEmbeddingClient(
        api_base=api_base,
        model=model_name,
        timeout_seconds=60.0,
        batch_size=32,
    )
    try:
        resolved = client.resolve_model()
        logger.info("Embedding model: %s", resolved)
    except Exception as e:
        logger.error("Embedding server not reachable (%s). Start with: ./scripts/start_embedding_server.sh", e)
        return 1

    detected_embedding_dim: Optional[int] = None
    if args.embedding_dim is None:
        try:
            probe = client.embed_texts(["ctreepo_embedding_dim_probe"])
            if probe and probe[0]:
                detected_embedding_dim = int(len(probe[0]))
                logger.info("Detected embedding dimension: %d", detected_embedding_dim)
        except Exception as e:
            logger.warning("Could not auto-detect embedding dimension: %s", e)

    resolved_embedding_dim = (
        int(args.embedding_dim)
        if args.embedding_dim is not None
        else int(detected_embedding_dim or 4096)
    )

    # ------------------------------------------------------------------
    # Build config and trainer
    # ------------------------------------------------------------------
    import torch
    from src.tree.ctreepo_model import ctreepo_config_from_mapping
    from src.training.config_sections import (
        OptimizerConfig,
        RunConfig,
        RuntimeConfig,
        TrainConfig,
        ValidationConfig,
    )
    from src.training.ctreepo_trainer import (
        CTreePOTrainer,
        CTreePOTrainingConfig,
        LocalLawSupervisionConfig,
        OnlineLocalLawSupervisionConfig,
        TreeOperatorDataConfig,
        TreeOperatorEvaluationConfig,
        TreeOperatorObjectiveConfig,
    )

    model_config = ctreepo_config_from_mapping(
        {
            "sketch_dim": int(args.sketch_dim),
            "hidden_dim": int(args.hidden_dim),
            "merge_type": str(args.merge_type),
            "head_names": ("rile",),
            "tree_model_version": str(args.tree_model_version),
        },
        embedding_dim=int(resolved_embedding_dim),
    )
    train_config = CTreePOTrainingConfig(
        model=model_config,
        data=TreeOperatorDataConfig(
            window_size=args.window_size,
            window_overlap=args.window_overlap,
        ),
        run=RunConfig(output_dir=args.output_dir, seed=args.seed),
        train=TrainConfig(batch_size=args.batch_size, epochs=args.epochs),
        optimizer=OptimizerConfig(
            learning_rate=args.lr,
            optimizer=args.optimizer,
            scheduler=args.scheduler,
            min_learning_rate=args.min_lr,
            warmup_epochs=args.warmup_epochs,
            grad_clip_norm=args.grad_clip_norm,
            weight_decay=1e-4,
        ),
        validation=ValidationConfig(
            eval_every=args.eval_every,
            early_stopping_patience=args.early_stopping_patience,
            early_stopping_min_delta=args.early_stopping_min_delta,
        ),
        runtime=RuntimeConfig(device=args.device),
        objective=TreeOperatorObjectiveConfig(
            root_weight=args.root_weight,
            leaf_audit_weight=args.leaf_audit_weight,
            merge_audit_weight=args.merge_audit_weight,
            idempotence_weight=args.idempotence_weight,
            local_law_violation_threshold=args.local_law_violation_threshold,
        ),
        supervision=LocalLawSupervisionConfig(
            require_local_law_supervision=args.require_local_law_supervision,
            online=OnlineLocalLawSupervisionConfig(
                enabled=bool(args.online_local_law_supervision),
                teacher_worker=(
                    bool(args.online_local_law_supervision)
                    and str(args.online_teacher_worker) == "on"
                    and not bool(args.online_human_only)
                ),
                worker_concurrency=int(args.online_worker_concurrency),
            ),
        ),
        evaluation=TreeOperatorEvaluationConfig(
            uncertainty_z_score=args.uncertainty_z_score,
            min_interval_std=args.min_interval_std,
        ),
    )
    trainer = CTreePOTrainer(
        config=train_config,
        embedding_client=client,
        node_oracle_predictor=node_oracle_predictor,
        node_oracle_source_kind=node_oracle_source_kind,
        node_oracle_source_spec=node_oracle_source_spec,
        online_node_oracle_queue=online_node_oracle_queue,
        online_teacher_worker=(
            bool(args.online_local_law_supervision)
            and str(args.online_teacher_worker) == "on"
            and not bool(args.online_human_only)
        ),
        online_worker_concurrency=int(args.online_worker_concurrency),
    )
    logger.info("Training device: %s", trainer.device)
    repro_manifest_path = write_reproducibility_manifest(
        args.output_dir,
        seed=int(args.seed),
        cli_args=vars(args),
        config=train_config,
        applied=applied_repro,
        extra={
            "task": str(args.task),
            "resolved_embedding_model": str(resolved),
            "embedding_api_base": str(api_base),
            "embedding_dim": int(resolved_embedding_dim),
            "train_manifesto_ids": list(train_ids),
            "val_manifesto_ids": list(val_ids),
            "labeled_tree_artifacts": [str(path) for path in labeled_tree_artifact_paths],
            "labeled_tree_train_splits": list(args.labeled_tree_train_splits or []),
            "labeled_tree_val_splits": list(args.labeled_tree_val_splits or []),
            "distillation_contract": (
                {
                    "train_targets": ["tree_operator"],
                    "student_model_class": "ctreepo_embedding_tree",
                    "supervision_source": "labeled_tree_artifact",
                    "teacher_model_spec": None,
                }
                if labeled_tree_artifact_paths
                else None
            ),
            "local_law_label_source": {
                "kind": str(node_oracle_source_kind),
                "spec": node_oracle_source_spec,
            },
            "online_local_law_supervision": {
                "enabled": bool(args.online_local_law_supervision),
                "feedback_store": str(feedback_store_path)
                if bool(args.online_local_law_supervision)
                else None,
                "teacher_worker": str(args.online_teacher_worker),
                "leaf_budget_per_epoch": int(args.online_leaf_query_budget_per_epoch),
                "merge_budget_per_epoch": int(args.online_merge_query_budget_per_epoch),
                "worker_concurrency": int(args.online_worker_concurrency),
            },
        },
    )
    logger.info("Reproducibility manifest: %s", repro_manifest_path)

    # ------------------------------------------------------------------
    # Prepare/train trees
    # ------------------------------------------------------------------
    if labeled_tree_artifact_paths:
        from src.ctreepo.distillation import (
            DistillationContractConfig,
            DistillationTrainConfig,
            fit as fit_distillation_student,
            load_labeled_trees,
        )

        labeled_trees = []
        for path in labeled_tree_artifact_paths:
            loaded = load_labeled_trees(path)
            logger.info("Loaded %d labeled tree(s) from %s", len(loaded), path)
            labeled_trees.extend(loaded)
        if not labeled_trees:
            logger.error("No labeled trees loaded from %s", labeled_tree_artifact_paths)
            return 2
        logger.info(
            "Fitting C-TreePO tree-operator model from %d labeled tree artifact(s)",
            len(labeled_trees),
        )
        fit_result = fit_distillation_student(
            labeled_trees,
            DistillationTrainConfig(
                contract=DistillationContractConfig(
                    train_targets=("tree_operator",),
                    student_model_class="ctreepo_embedding_tree",
                    supervision_source="labeled_tree_artifact",
                ),
                run=RunConfig(output_dir=args.output_dir, seed=args.seed),
                train=TrainConfig(
                    train_splits=tuple(args.labeled_tree_train_splits or ["train"]),
                    batch_size=args.batch_size,
                    epochs=args.epochs,
                ),
                validation=ValidationConfig(
                    val_splits=tuple(args.labeled_tree_val_splits or ["val"]),
                    eval_every=args.eval_every,
                ),
            ),
            embedding_client=client,
            trainer=trainer,
        )
        result = fit_result.trained_artifact
        if result is None:
            logger.error("Distillation fit did not return a trained tree-operator artifact")
            return 1
        logger.info(
            "Tree-operator distillation fit consumed %d train tree(s), %d val tree(s)",
            fit_result.train_count,
            fit_result.val_count,
        )
    else:
        logger.info("Building embedding trees...")
        n_train = trainer.prepare_trees_from_samples(train_samples, split="train")
        n_val = trainer.prepare_trees_from_samples(val_samples, split="val") if val_samples else 0
        logger.info("Built %d train trees, %d val trees", n_train, n_val)
        result = trainer.train(output_dir=args.output_dir)

    # ------------------------------------------------------------------
    # Final evaluation on all data (using best checkpoint)
    # ------------------------------------------------------------------
    best_path = args.output_dir / "best.pt"
    if best_path.exists():
        trainer.model.load_state_dict(
            torch.load(best_path, map_location=trainer.device, weights_only=True)
        )
        logger.info("Loaded best checkpoint from %s", best_path)

    all_trees = trainer.train_trees + trainer.val_trees
    final_metrics = trainer.evaluate(all_trees, epoch=result.best_epoch)

    print("")
    print("=" * 60)
    print("  CTreePO Training Complete")
    print("=" * 60)
    print(f"  Best epoch: {result.best_epoch + 1}")
    print(f"  Best root MAE: {result.best_root_mae:.2f}")
    print(f"  Interval coverage (95%% proxy): {final_metrics.interval_coverage_95:.3f}")
    print(f"  Interval mean width (95%% proxy): {final_metrics.interval_mean_width_95:.2f}")
    if torch.isfinite(torch.tensor(final_metrics.leaf_oracle_mae)):
        print(
            "  Local-law span metrics: "
            f"leaf_mae={final_metrics.leaf_oracle_mae:.2f} "
            f"merge_mae={final_metrics.merge_oracle_mae:.2f} "
            f"leaf_violation_rate={final_metrics.leaf_violation_rate:.3f} "
            f"merge_violation_rate={final_metrics.merge_violation_rate:.3f}"
        )
    else:
        print("  Local-law span metrics: inactive (no node-span oracle labels attached)")
    print(f"  Training time: {result.training_time_seconds:.1f}s")
    print("")
    print("  Per-document predictions:")
    for doc in final_metrics.per_doc:
        err = doc["abs_error"]
        mark = "OK" if err < 15 else "MISS"
        print(f"    {doc['doc_id']}: true={doc['rile_true']:+.1f} pred={doc['rile_pred']:+.1f} err={err:.1f} {mark}")
    print("")
    print(f"  Output: {args.output_dir}")
    print("=" * 60)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
