#!/usr/bin/env python3
"""Build xargs-friendly command lists for segmented-LDA C-TreePO sweeps."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, List


def _parse_items(text: str) -> List[str]:
    out: List[str] = []
    for raw in str(text).replace(",", " ").split():
        x = raw.strip()
        if x:
            out.append(x)
    return out


def _parse_ints(text: str) -> List[int]:
    return [int(x) for x in _parse_items(text)]


def _parse_floats(text: str) -> List[float]:
    return [float(x) for x in _parse_items(text)]


def _fmt_float(x: float) -> str:
    s = f"{float(x):.6g}"
    return s.replace("-", "m").replace(".", "p")


def _iter_commands(
    *,
    python_bin: str,
    train_docs: Iterable[int],
    seeds: Iterable[int],
    calibration_rates: Iterable[float],
    eval_internal_rates: Iterable[float],
    eval_leaf_rates: Iterable[float],
    output_root: Path,
    topic_phi_estimators: Iterable[str],
    topic_phi_docs_values: Iterable[int],
    leaf_theta_estimators: Iterable[str],
    topic_processes: Iterable[str],
    n_topics: int,
    vocab_size: int,
    min_segments: int,
    max_segments: int,
    min_seg_tokens: int,
    max_seg_tokens: int,
    fixed_leaf_tokens: int,
    n_books_test: int,
    alpha_topic: float,
    beta_word: float,
    segment_concentration: float,
    segment_background: float,
    calibration_policy: str,
    eval_internal_query_design: str,
    spectral_svd_dim_extra: int,
    spectral_max_leaves: int,
    spectral_kmeans_inits: int,
    spectral_kmeans_max_iter: int,
    tlda_delta: float,
    tlda_rate_constant: float,
    tlda_sigmaK_floor: float,
    topic_phi_permute: bool,
    online_tensor_lda_burn_in_docs: int,
    online_tensor_lda_batch_docs: int,
    online_tensor_lda_passes: int,
    online_tensor_lda_lr: float,
    online_tensor_lda_grad_clip_norm: float,
    embedding_topic_svd_dim_extra: int,
    embedding_topic_kmeans_inits: int,
    embedding_topic_kmeans_max_iter: int,
    embedding_topic_assignment_temperature: float,
    embedding_topic_ppmi_shift: float,
    neural_topic_base_estimator: str,
    neural_topic_seed_fraction_default: float,
    neural_topic_seed_fractions: Iterable[float],
    neural_topic_hidden_dim: int,
    neural_topic_steps: int,
    neural_topic_lr: float,
    neural_topic_weight_decay: float,
    neural_topic_mix_samples: int,
    neural_topic_mix_temperature: float,
    neural_topic_operator_boost: float,
    neural_topic_seed_llm_min_weight: float,
    neural_topic_seed_llm_max_weight: float,
    neural_topic_similarity_temperature: float,
    neural_topic_ridge: float,
    selection_audit_trials: int,
    leaf_theta_rf_n_estimators: int,
    leaf_theta_rf_max_depth: int,
    leaf_theta_rf_min_samples_leaf: int,
    leaf_theta_mlp_hidden_dim: int,
    leaf_theta_mlp_epochs: int,
    leaf_theta_mlp_batch_size: int,
    leaf_theta_mlp_lr: float,
    leaf_theta_mlp_weight_decay: float,
    device: str,
    cuda_device: int | None,
    torch_threads: int,
    skip_existing: bool,
) -> List[str]:
    cmds: List[str] = []
    script = "scripts/run_segmented_lda_ctreepo_simulation.py"

    estimator_values = [str(x).strip() for x in topic_phi_estimators if str(x).strip()]
    if not estimator_values:
        raise ValueError("topic_phi_estimators must be non-empty")
    docs_values = [int(x) for x in topic_phi_docs_values]
    if not docs_values:
        docs_values = [0]
    theta_values = [str(x).strip().lower() for x in leaf_theta_estimators if str(x).strip()]
    if not theta_values:
        theta_values = ["lstsq"]
    for theta in theta_values:
        if theta not in {"lstsq", "rf", "mlp", "sklearn_lda"}:
            raise ValueError("leaf_theta_estimators must be a subset of {'lstsq','rf','mlp','sklearn_lda'}")
    topic_process_values = [str(x).strip().lower() for x in topic_processes if str(x).strip()]
    if not topic_process_values:
        topic_process_values = ["segments"]
    for proc in topic_process_values:
        if proc not in {"segments", "bag_of_words"}:
            raise ValueError("topic_processes must be a subset of {'segments','bag_of_words'}")
    seed_frac_values = [float(x) for x in neural_topic_seed_fractions]
    if not seed_frac_values:
        seed_frac_values = [float(neural_topic_seed_fraction_default)]
    if any((not (0.0 < float(x) <= 1.0)) for x in seed_frac_values):
        raise ValueError("neural_topic_seed_fractions must be in (0, 1]")
    default_seed_frac = float(neural_topic_seed_fraction_default)

    for proc in topic_process_values:
        proc_prefix = ""
        if len(topic_process_values) > 1 or str(proc) != "segments":
            proc_prefix = f"tp_{str(proc)}/"
        for theta in theta_values:
            theta_prefix = ""
            if len(theta_values) > 1 or str(theta) != "lstsq":
                theta_prefix = f"theta_{str(theta)}/"
            for est in estimator_values:
                est_norm = str(est).strip().lower()
                is_neural = bool(est_norm.startswith("neural_"))
                for phi_docs in docs_values:
                    seed_fracs = seed_frac_values if is_neural else [float(default_seed_frac)]
                    for seed_frac in seed_fracs:
                        for td in train_docs:
                            for cal in calibration_rates:
                                for el in eval_leaf_rates:
                                    for ei in eval_internal_rates:
                                        for seed in seeds:
                                            docs_component = ""
                                            if len(docs_values) > 1 or int(phi_docs) != 0:
                                                docs_component = f"/docs_{int(phi_docs)}"
                                            seed_component = ""
                                            if is_neural and (
                                                len(seed_fracs) > 1 or float(seed_frac) != float(default_seed_frac)
                                            ):
                                                seed_component = f"/seedfrac_{_fmt_float(float(seed_frac))}"
                                            sub = (
                                                f"{proc_prefix}{theta_prefix}phi_{str(est)}{docs_component}{seed_component}"
                                                f"/train_{int(td)}"
                                                f"/lt_{int(fixed_leaf_tokens)}"
                                                f"/cal_{_fmt_float(cal)}"
                                                f"/leaf_{_fmt_float(el)}"
                                                f"/int_{_fmt_float(ei)}"
                                            )
                                            base = output_root / sub / f"seed_{int(seed)}"
                                            out_json = base.with_suffix(".json")
                                            out_csv = base.with_suffix(".csv")
                                            if skip_existing and out_json.exists() and out_csv.exists():
                                                continue

                                            perm_flag = (
                                                "--topic-phi-permute" if bool(topic_phi_permute) else "--no-topic-phi-permute"
                                            )
                                            cmd = (
                                                f"{python_bin} -u {script} "
                                                f"--topic-phi-estimator {str(est)} "
                                                f"--topic-phi-docs {int(phi_docs)} "
                                                f"--n-topics {int(n_topics)} --vocab-size {int(vocab_size)} "
                                                f"--min-segments {int(min_segments)} --max-segments {int(max_segments)} "
                                                f"--min-seg-tokens {int(min_seg_tokens)} --max-seg-tokens {int(max_seg_tokens)} "
                                                f"--fixed-leaf-tokens {int(fixed_leaf_tokens)} "
                                                f"--alpha-topic {float(alpha_topic)} --beta-word {float(beta_word)} "
                                                f"--topic-process {str(proc)} "
                                                f"--segment-concentration {float(segment_concentration)} "
                                                f"--segment-background {float(segment_background)} "
                                                f"--leaf-theta-estimator {str(theta)} "
                                                f"--leaf-theta-rf-n-estimators {int(leaf_theta_rf_n_estimators)} "
                                                f"--leaf-theta-rf-max-depth {int(leaf_theta_rf_max_depth)} "
                                                f"--leaf-theta-rf-min-samples-leaf {int(leaf_theta_rf_min_samples_leaf)} "
                                                f"--leaf-theta-mlp-hidden-dim {int(leaf_theta_mlp_hidden_dim)} "
                                                f"--leaf-theta-mlp-epochs {int(leaf_theta_mlp_epochs)} "
                                                f"--leaf-theta-mlp-batch-size {int(leaf_theta_mlp_batch_size)} "
                                                f"--leaf-theta-mlp-lr {float(leaf_theta_mlp_lr)} "
                                                f"--leaf-theta-mlp-weight-decay {float(leaf_theta_mlp_weight_decay)} "
                                                f"--n-books-train {int(td)} --n-books-test {int(n_books_test)} "
                                                f"--calibration-leaf-query-rate {float(cal)} "
                                                f"--calibration-policy {calibration_policy} "
                                                f"--eval-leaf-query-rate {float(el)} "
                                                f"--eval-internal-query-rate {float(ei)} "
                                                f"--eval-internal-query-design {eval_internal_query_design} "
                                                f"--spectral-svd-dim-extra {int(spectral_svd_dim_extra)} "
                                                f"--spectral-max-leaves {int(spectral_max_leaves)} "
                                                f"--spectral-kmeans-inits {int(spectral_kmeans_inits)} "
                                                f"--spectral-kmeans-max-iter {int(spectral_kmeans_max_iter)} "
                                                f"--tlda-delta {float(tlda_delta)} "
                                                f"--tlda-rate-constant {float(tlda_rate_constant)} "
                                                f"--tlda-sigmaK-floor {float(tlda_sigmaK_floor)} "
                                                f"{perm_flag} "
                                                f"--online-tensor-lda-burn-in-docs {int(online_tensor_lda_burn_in_docs)} "
                                                f"--online-tensor-lda-batch-docs {int(online_tensor_lda_batch_docs)} "
                                                f"--online-tensor-lda-passes {int(online_tensor_lda_passes)} "
                                                f"--online-tensor-lda-lr {float(online_tensor_lda_lr)} "
                                                f"--online-tensor-lda-grad-clip-norm {float(online_tensor_lda_grad_clip_norm)} "
                                                f"--embedding-topic-svd-dim-extra {int(embedding_topic_svd_dim_extra)} "
                                                f"--embedding-topic-kmeans-inits {int(embedding_topic_kmeans_inits)} "
                                                f"--embedding-topic-kmeans-max-iter {int(embedding_topic_kmeans_max_iter)} "
                                                f"--embedding-topic-assignment-temperature {float(embedding_topic_assignment_temperature)} "
                                                f"--embedding-topic-ppmi-shift {float(embedding_topic_ppmi_shift)} "
                                                f"--neural-topic-base-estimator {neural_topic_base_estimator} "
                                                f"--neural-topic-seed-fraction {float(seed_frac)} "
                                                f"--neural-topic-hidden-dim {int(neural_topic_hidden_dim)} "
                                                f"--neural-topic-steps {int(neural_topic_steps)} "
                                                f"--neural-topic-lr {float(neural_topic_lr)} "
                                                f"--neural-topic-weight-decay {float(neural_topic_weight_decay)} "
                                                f"--neural-topic-mix-samples {int(neural_topic_mix_samples)} "
                                                f"--neural-topic-mix-temperature {float(neural_topic_mix_temperature)} "
                                                f"--neural-topic-operator-boost {float(neural_topic_operator_boost)} "
                                                f"--neural-topic-seed-llm-min-weight {float(neural_topic_seed_llm_min_weight)} "
                                                f"--neural-topic-seed-llm-max-weight {float(neural_topic_seed_llm_max_weight)} "
                                                f"--neural-topic-similarity-temperature {float(neural_topic_similarity_temperature)} "
                                                f"--neural-topic-ridge {float(neural_topic_ridge)} "
                                                f"--selection-audit-trials {int(selection_audit_trials)} "
                                                f"--device {str(device)} --torch-threads {int(torch_threads)} "
                                                f"--seed {int(seed)} "
                                                f"--json-summary {out_json} --csv-summary {out_csv}"
                                            )
                                            if cuda_device is not None:
                                                cmd += f" --cuda-device {int(cuda_device)}"
                                            cmds.append(cmd)
    return cmds


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build segmented-LDA C-TreePO sweep command list.")
    p.add_argument("--python-bin", type=str, default="venv/bin/python")
    p.add_argument("--out-cmds", type=str, default="logs/segmented_lda_ctreepo_cmds.txt")
    p.add_argument("--output-root", type=str, default="outputs/segmented_lda_ctreepo")

    p.add_argument("--train-docs", type=str, default="64 128 256 512")
    p.add_argument("--seeds", type=str, default="0 1 2 3 4 5 6 7")
    p.add_argument("--calibration-rates", type=str, default="0 0.05 0.1 0.25 0.5")
    p.add_argument("--eval-leaf-rates", type=str, default="0")
    p.add_argument("--eval-internal-rates", type=str, default="0 0.05 0.1 0.25 0.5 1.0")
    p.add_argument("--leaf-theta-estimator", choices=["lstsq", "rf", "mlp", "sklearn_lda"], default="lstsq")
    p.add_argument(
        "--leaf-theta-estimators",
        type=str,
        default="",
        help="Optional space/comma list of leaf-theta estimators. If set, overrides --leaf-theta-estimator.",
    )
    p.add_argument("--leaf-theta-rf-n-estimators", type=int, default=200)
    p.add_argument("--leaf-theta-rf-max-depth", type=int, default=16)
    p.add_argument("--leaf-theta-rf-min-samples-leaf", type=int, default=5)
    p.add_argument("--leaf-theta-mlp-hidden-dim", type=int, default=128)
    p.add_argument("--leaf-theta-mlp-epochs", type=int, default=10)
    p.add_argument("--leaf-theta-mlp-batch-size", type=int, default=256)
    p.add_argument("--leaf-theta-mlp-lr", type=float, default=1e-3)
    p.add_argument("--leaf-theta-mlp-weight-decay", type=float, default=1e-4)
    p.add_argument("--topic-process", choices=["segments", "bag_of_words"], default="segments")
    p.add_argument(
        "--topic-processes",
        type=str,
        default="",
        help="Optional space/comma list of topic-process modes. If set, overrides --topic-process.",
    )

    p.add_argument(
        "--topic-phi-estimator",
        choices=[
            "true",
            "noisy_theory",
            "tensor_lda",
            "online_tensor_lda",
            "embedding_spectral",
            "spectral_numpy",
            "neural_ctreepo",
            "neural_mergeable_sketch",
            "neural_hybrid",
            "neural_embedding_hybrid",
        ],
        default="spectral_numpy",
    )
    p.add_argument(
        "--topic-phi-estimators",
        type=str,
        default="",
        help="Optional space/comma list of topic-phi-estimators. If set, overrides --topic-phi-estimator.",
    )
    p.add_argument("--topic-phi-docs", type=int, default=0)
    p.add_argument(
        "--topic-phi-docs-grid",
        type=str,
        default="",
        help="Optional space/comma list of topic_phi_docs values. If set, overrides --topic-phi-docs.",
    )
    p.add_argument("--n-topics", type=int, default=4)
    p.add_argument("--vocab-size", type=int, default=256)
    p.add_argument("--min-segments", type=int, default=6)
    p.add_argument("--max-segments", type=int, default=6)
    p.add_argument("--min-seg-tokens", type=int, default=24)
    p.add_argument("--max-seg-tokens", type=int, default=48)
    p.add_argument("--fixed-leaf-tokens", type=int, default=32)
    p.add_argument("--alpha-topic", type=float, default=0.20)
    p.add_argument("--beta-word", type=float, default=0.10)
    p.add_argument("--segment-concentration", type=float, default=80.0)
    p.add_argument("--segment-background", type=float, default=2.0)
    p.add_argument("--n-books-test", type=int, default=2000)
    p.add_argument("--calibration-policy", choices=["uniform", "entropy"], default="uniform")
    p.add_argument("--eval-internal-query-design", choices=["none", "uniform", "risk"], default="risk")

    p.add_argument("--spectral-svd-dim-extra", type=int, default=2)
    p.add_argument("--spectral-max-leaves", type=int, default=4000)
    p.add_argument("--spectral-kmeans-inits", type=int, default=6)
    p.add_argument("--spectral-kmeans-max-iter", type=int, default=60)
    p.add_argument("--tlda-delta", type=float, default=0.10)
    p.add_argument("--tlda-rate-constant", type=float, default=1.0)
    p.add_argument("--tlda-sigmaK-floor", type=float, default=1e-6)
    p.add_argument("--topic-phi-permute", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--online-tensor-lda-burn-in-docs", type=int, default=0)
    p.add_argument("--online-tensor-lda-batch-docs", type=int, default=32)
    p.add_argument("--online-tensor-lda-passes", type=int, default=1)
    p.add_argument("--online-tensor-lda-lr", type=float, default=0.1)
    p.add_argument("--online-tensor-lda-grad-clip-norm", type=float, default=1.0)
    p.add_argument("--embedding-topic-svd-dim-extra", type=int, default=4)
    p.add_argument("--embedding-topic-kmeans-inits", type=int, default=8)
    p.add_argument("--embedding-topic-kmeans-max-iter", type=int, default=80)
    p.add_argument("--embedding-topic-assignment-temperature", type=float, default=0.35)
    p.add_argument("--embedding-topic-ppmi-shift", type=float, default=1.0)
    p.add_argument("--neural-topic-base-estimator", type=str, default="tensor_lda")
    p.add_argument("--neural-topic-seed-fraction", type=float, default=0.35)
    p.add_argument(
        "--neural-topic-seed-fractions",
        type=str,
        default="",
        help="Optional space/comma list of neural_topic_seed_fraction values (applies only when estimator is neural_ctreepo).",
    )
    p.add_argument("--neural-topic-hidden-dim", type=int, default=48)
    p.add_argument("--neural-topic-steps", type=int, default=60)
    p.add_argument("--neural-topic-lr", type=float, default=3e-3)
    p.add_argument("--neural-topic-weight-decay", type=float, default=1e-4)
    p.add_argument("--neural-topic-mix-samples", type=int, default=128)
    p.add_argument("--neural-topic-mix-temperature", type=float, default=1.0)
    p.add_argument("--neural-topic-operator-boost", type=float, default=1.4)
    p.add_argument("--neural-topic-seed-llm-min-weight", type=float, default=0.2)
    p.add_argument("--neural-topic-seed-llm-max-weight", type=float, default=0.55)
    p.add_argument("--neural-topic-similarity-temperature", type=float, default=0.15)
    p.add_argument("--neural-topic-ridge", type=float, default=1e-3)
    p.add_argument("--selection-audit-trials", type=int, default=0)
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--cuda-device", type=int, default=None)
    p.add_argument("--torch-threads", type=int, default=0)

    p.add_argument("--skip-existing", action=argparse.BooleanOptionalAction, default=True)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    out_cmds = Path(args.out_cmds)
    out_cmds.parent.mkdir(parents=True, exist_ok=True)
    Path(args.output_root).mkdir(parents=True, exist_ok=True)

    topic_phi_estimators = _parse_items(args.topic_phi_estimators) if str(args.topic_phi_estimators).strip() else [str(args.topic_phi_estimator)]
    topic_phi_docs_values = _parse_ints(args.topic_phi_docs_grid) if str(args.topic_phi_docs_grid).strip() else [int(args.topic_phi_docs)]
    neural_topic_seed_fractions = _parse_floats(args.neural_topic_seed_fractions) if str(args.neural_topic_seed_fractions).strip() else [float(args.neural_topic_seed_fraction)]
    leaf_theta_estimators = (
        _parse_items(args.leaf_theta_estimators)
        if str(args.leaf_theta_estimators).strip()
        else [str(args.leaf_theta_estimator)]
    )
    topic_processes = (
        _parse_items(args.topic_processes)
        if str(args.topic_processes).strip()
        else [str(args.topic_process)]
    )

    cmds = _iter_commands(
        python_bin=str(args.python_bin),
        train_docs=_parse_ints(args.train_docs),
        seeds=_parse_ints(args.seeds),
        calibration_rates=_parse_floats(args.calibration_rates),
        eval_internal_rates=_parse_floats(args.eval_internal_rates),
        eval_leaf_rates=_parse_floats(args.eval_leaf_rates),
        output_root=Path(args.output_root),
        topic_phi_estimators=topic_phi_estimators,
        topic_phi_docs_values=topic_phi_docs_values,
        leaf_theta_estimators=leaf_theta_estimators,
        topic_processes=topic_processes,
        n_topics=int(args.n_topics),
        vocab_size=int(args.vocab_size),
        min_segments=int(args.min_segments),
        max_segments=int(args.max_segments),
        min_seg_tokens=int(args.min_seg_tokens),
        max_seg_tokens=int(args.max_seg_tokens),
        fixed_leaf_tokens=int(args.fixed_leaf_tokens),
        n_books_test=int(args.n_books_test),
        alpha_topic=float(args.alpha_topic),
        beta_word=float(args.beta_word),
        segment_concentration=float(args.segment_concentration),
        segment_background=float(args.segment_background),
        calibration_policy=str(args.calibration_policy),
        eval_internal_query_design=str(args.eval_internal_query_design),
        spectral_svd_dim_extra=int(args.spectral_svd_dim_extra),
        spectral_max_leaves=int(args.spectral_max_leaves),
        spectral_kmeans_inits=int(args.spectral_kmeans_inits),
        spectral_kmeans_max_iter=int(args.spectral_kmeans_max_iter),
        tlda_delta=float(args.tlda_delta),
        tlda_rate_constant=float(args.tlda_rate_constant),
        tlda_sigmaK_floor=float(args.tlda_sigmaK_floor),
        topic_phi_permute=bool(args.topic_phi_permute),
        online_tensor_lda_burn_in_docs=int(args.online_tensor_lda_burn_in_docs),
        online_tensor_lda_batch_docs=int(args.online_tensor_lda_batch_docs),
        online_tensor_lda_passes=int(args.online_tensor_lda_passes),
        online_tensor_lda_lr=float(args.online_tensor_lda_lr),
        online_tensor_lda_grad_clip_norm=float(args.online_tensor_lda_grad_clip_norm),
        embedding_topic_svd_dim_extra=int(args.embedding_topic_svd_dim_extra),
        embedding_topic_kmeans_inits=int(args.embedding_topic_kmeans_inits),
        embedding_topic_kmeans_max_iter=int(args.embedding_topic_kmeans_max_iter),
        embedding_topic_assignment_temperature=float(args.embedding_topic_assignment_temperature),
        embedding_topic_ppmi_shift=float(args.embedding_topic_ppmi_shift),
        neural_topic_base_estimator=str(args.neural_topic_base_estimator),
        neural_topic_seed_fraction_default=float(args.neural_topic_seed_fraction),
        neural_topic_seed_fractions=neural_topic_seed_fractions,
        neural_topic_hidden_dim=int(args.neural_topic_hidden_dim),
        neural_topic_steps=int(args.neural_topic_steps),
        neural_topic_lr=float(args.neural_topic_lr),
        neural_topic_weight_decay=float(args.neural_topic_weight_decay),
        neural_topic_mix_samples=int(args.neural_topic_mix_samples),
        neural_topic_mix_temperature=float(args.neural_topic_mix_temperature),
        neural_topic_operator_boost=float(args.neural_topic_operator_boost),
        neural_topic_seed_llm_min_weight=float(args.neural_topic_seed_llm_min_weight),
        neural_topic_seed_llm_max_weight=float(args.neural_topic_seed_llm_max_weight),
        neural_topic_similarity_temperature=float(args.neural_topic_similarity_temperature),
        neural_topic_ridge=float(args.neural_topic_ridge),
        selection_audit_trials=int(args.selection_audit_trials),
        leaf_theta_rf_n_estimators=int(args.leaf_theta_rf_n_estimators),
        leaf_theta_rf_max_depth=int(args.leaf_theta_rf_max_depth),
        leaf_theta_rf_min_samples_leaf=int(args.leaf_theta_rf_min_samples_leaf),
        leaf_theta_mlp_hidden_dim=int(args.leaf_theta_mlp_hidden_dim),
        leaf_theta_mlp_epochs=int(args.leaf_theta_mlp_epochs),
        leaf_theta_mlp_batch_size=int(args.leaf_theta_mlp_batch_size),
        leaf_theta_mlp_lr=float(args.leaf_theta_mlp_lr),
        leaf_theta_mlp_weight_decay=float(args.leaf_theta_mlp_weight_decay),
        device=str(args.device),
        cuda_device=(int(args.cuda_device) if args.cuda_device is not None else None),
        torch_threads=int(args.torch_threads),
        skip_existing=bool(args.skip_existing),
    )

    out_cmds.write_text("\n".join(cmds) + ("\n" if cmds else ""), encoding="utf-8")
    print(f"wrote_cmds | {out_cmds} | n_commands={len(cmds)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
