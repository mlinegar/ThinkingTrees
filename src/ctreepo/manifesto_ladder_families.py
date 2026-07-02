from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

from src.ctreepo.fg_arity import auto_g_output_tokens
from src.tasks.manifesto.expert_scale import (
    EXPERT_SCALE_NORMALIZED_1_7,
    expert_scale_bounds,
)


def root_label_sources(args: Any) -> Tuple[str, ...]:
    raw = str(getattr(args, "root_label_sources", "") or "").strip()
    if not raw:
        return tuple()
    sources: list[str] = []
    for part in raw.split(","):
        source = part.strip().lower().replace("-", "_")
        if not source:
            continue
        if source in {"stored", "summary"}:
            source = "stored_summary"
        if source == "raw":
            source = "raw_document"
        if source not in {"stored_summary", "raw_document"}:
            raise ValueError(
                f"unknown root label source {part!r}; expected stored_summary or raw_document"
            )
        if source not in sources:
            sources.append(source)
    return tuple(sources)


def resolved_target_bounds(args: Any) -> Tuple[float, float]:
    if getattr(args, "target_min", None) is not None and getattr(args, "target_max", None) is not None:
        return float(args.target_min), float(args.target_max)
    if str(getattr(args, "dimension", "")).strip().lower() == "environment":
        default_min, default_max = expert_scale_bounds(
            dimension="environment",
            scale=EXPERT_SCALE_NORMALIZED_1_7,
        )
    else:
        default_min, default_max = 1.0, 7.0
    target_min = float(default_min if getattr(args, "target_min", None) is None else args.target_min)
    target_max = float(default_max if getattr(args, "target_max", None) is None else args.target_max)
    return target_min, target_max


def build_dspy_family(args: Any, *, leaf_size_tokens: int) -> Any:
    max_tokens = auto_g_output_tokens(
        int(args.dspy_max_tokens),
        leaf_size_tokens=int(leaf_size_tokens),
    )
    lm_config: Dict[str, Any] = {}
    if args.dspy_model:
        lm_config["model"] = str(args.dspy_model)
    if args.dspy_api_base:
        lm_config["api_base"] = str(args.dspy_api_base)
    if args.dspy_api_key:
        lm_config["api_key"] = str(args.dspy_api_key)
    lm_config["max_tokens"] = int(max_tokens)
    if str(args.dimension).strip().lower() in {"combined", "joint", "all", "all6"}:
        from src.ctreepo.joint_dspy_family import JointDSPyFamily, JointDSPyFamilyConfig

        target_min, target_max = resolved_target_bounds(args)
        joint_f_init_path = (
            str(args.dspy_f_init_path)
            if args.dspy_f_init_path is not None
            else "outputs/phase2/joint_gepa/optimized_program.json"
        )
        return JointDSPyFamily(
            config=JointDSPyFamilyConfig(
                optimizer=str(args.dspy_optimizer),
                budget=str(args.dspy_budget),
                num_threads=int(args.dspy_num_threads),
                target_min=target_min,
                target_max=target_max,
                scorer_output_min=float(args.scorer_output_min),
                scorer_output_max=float(args.scorer_output_max),
                lm_config=lm_config,
                lm_transport=str(args.dspy_lm_transport),
                batch_max_concurrent=int(args.dspy_batch_max_concurrent),
                batch_size=int(args.dspy_batch_size),
                batch_timeout=float(args.dspy_batch_timeout),
                batch_request_timeout=float(args.dspy_batch_request_timeout),
                batch_await_response_timeout=args.dspy_batch_await_response_timeout,
                batch_routing_policy=str(args.dspy_batch_routing_policy),
                mipro_num_candidates=args.dspy_mipro_num_candidates,
                mipro_num_trials=args.dspy_mipro_num_trials,
                mipro_max_bootstrapped_demos=args.dspy_mipro_max_bootstrapped_demos,
                mipro_max_labeled_demos=args.dspy_mipro_max_labeled_demos,
                mipro_minibatch_size=int(args.dspy_mipro_minibatch_size),
                mipro_minibatch_full_eval_steps=int(args.dspy_mipro_minibatch_full_eval_steps),
                max_train_records=(
                    None
                    if int(args.dspy_max_train_records) <= 0
                    else int(args.dspy_max_train_records)
                ),
                record_sample_seed=int(args.seed),
                leaf_size_tokens=int(leaf_size_tokens),
                lm_context_window_tokens=int(args.dspy_lm_context_tokens),
                max_completion_tokens=int(max_tokens),
                prompt_template_overhead_tokens=int(args.dspy_prompt_overhead_tokens),
                tokenizer_model_path=str(args.embedding_model),
                dimension="combined",
                f_init_path=joint_f_init_path,
                f_init_mode=str(args.dspy_f_init_mode),
                root_label_sources=root_label_sources(args),
                root_label_target=str(args.root_label_target),
                local_law_weight=args.local_law_weight,
                node_weight_normalization=str(args.node_weight_normalization),
            )
        )

    from src.ctreepo.dspy_family import DSPyFamily, DSPyFamilyConfig

    target_min, target_max = resolved_target_bounds(args)
    return DSPyFamily(
        config=DSPyFamilyConfig(
            optimizer=str(args.dspy_optimizer),
            budget=str(args.dspy_budget),
            num_threads=int(args.dspy_num_threads),
            target_min=target_min,
            target_max=target_max,
            scorer_output_min=float(args.scorer_output_min),
            scorer_output_max=float(args.scorer_output_max),
            lm_config=lm_config,
            lm_transport=str(args.dspy_lm_transport),
            batch_max_concurrent=int(args.dspy_batch_max_concurrent),
            batch_size=int(args.dspy_batch_size),
            batch_timeout=float(args.dspy_batch_timeout),
            batch_request_timeout=float(args.dspy_batch_request_timeout),
            batch_await_response_timeout=args.dspy_batch_await_response_timeout,
            batch_routing_policy=str(args.dspy_batch_routing_policy),
            mipro_num_candidates=args.dspy_mipro_num_candidates,
            mipro_num_trials=args.dspy_mipro_num_trials,
            mipro_max_bootstrapped_demos=args.dspy_mipro_max_bootstrapped_demos,
            mipro_max_labeled_demos=args.dspy_mipro_max_labeled_demos,
            mipro_minibatch_size=int(args.dspy_mipro_minibatch_size),
            mipro_minibatch_full_eval_steps=int(args.dspy_mipro_minibatch_full_eval_steps),
            max_train_records=(
                None
                if int(args.dspy_max_train_records) <= 0
                else int(args.dspy_max_train_records)
            ),
            record_sample_seed=int(args.seed),
            leaf_size_tokens=int(leaf_size_tokens),
            lm_context_window_tokens=int(args.dspy_lm_context_tokens),
            max_completion_tokens=int(max_tokens),
            prompt_template_overhead_tokens=int(args.dspy_prompt_overhead_tokens),
            tokenizer_model_path=str(args.embedding_model),
            dimension=str(args.dimension),
            f_init_path=str(args.dspy_f_init_path) if args.dspy_f_init_path is not None else None,
            f_init_mode=str(args.dspy_f_init_mode),
            root_label_sources=root_label_sources(args),
            root_label_target=str(args.root_label_target),
            local_law_weight=args.local_law_weight,
            node_weight_normalization=str(args.node_weight_normalization),
        )
    )


__all__ = ["build_dspy_family", "resolved_target_bounds", "root_label_sources"]
