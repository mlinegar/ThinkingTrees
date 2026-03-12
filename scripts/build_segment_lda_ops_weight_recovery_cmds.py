#!/usr/bin/env python3
"""Build xargs-friendly command lists for Segment-LDA OPS weight-recovery sweeps."""

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
    test_docs: int,
    audit_fractions: Iterable[float],
    topic_phi_docs: Iterable[int],
    topic_phi_estimators: Iterable[str],
    topic_processes: Iterable[str],
    lambda_multipliers: Iterable[float],
    seeds: Iterable[int],
    output_root: Path,
    topic_source: str,
    feature_inference: str,
    n_topics: int,
    vocab_size: int,
    min_tokens: int,
    max_tokens: int,
    leaf_tokens: int,
    device: str,
    cuda_device: int | None,
    torch_threads: int,
    run_all_feature_modes: bool,
    skip_existing: bool,
) -> List[str]:
    cmds: List[str] = []
    script = "scripts/run_segment_lda_ops_weight_recovery_simulation.py"

    all_feature_flag = "--run-all-feature-modes" if bool(run_all_feature_modes) else "--no-run-all-feature-modes"

    for est in topic_phi_estimators:
        for proc in topic_processes:
            for td in train_docs:
                for af in audit_fractions:
                    for phi_docs in topic_phi_docs:
                        for lam in lambda_multipliers:
                            for seed in seeds:
                                sub = (
                                    f"phi_{est}"
                                    f"/proc_{proc}"
                                    f"/train_{int(td)}"
                                    f"/audit_{_fmt_float(af)}"
                                    f"/phi_docs_{int(phi_docs)}"
                                    f"/lam_{_fmt_float(lam)}"
                                )
                                base = output_root / sub / f"seed_{int(seed)}"
                                out_json = base.with_suffix(".json")
                                out_csv = base.with_suffix(".csv")
                                if skip_existing and out_json.exists() and out_csv.exists():
                                    continue

                                cmd = (
                                    f"{python_bin} -u {script} "
                                    f"--n-topics {int(n_topics)} --vocab-size {int(vocab_size)} "
                                    f"--min-tokens {int(min_tokens)} --max-tokens {int(max_tokens)} "
                                    f"--leaf-tokens {int(leaf_tokens)} "
                                    f"--topic-process {proc} "
                                    f"--lambda-multiplier {float(lam)} "
                                    f"--topic-source {topic_source} "
                                    f"--feature-inference {feature_inference} "
                                    f"--audit-policy fraction --audit-fraction {float(af)} "
                                    f"--topic-phi-estimator {est} --topic-phi-docs {int(phi_docs)} "
                                    f"--device {str(device)} --torch-threads {int(torch_threads)} "
                                    f"{all_feature_flag} "
                                    f"--train-docs {int(td)} --test-docs {int(test_docs)} "
                                    f"--seed {int(seed)} "
                                    f"--json-summary {out_json} --csv-summary {out_csv}"
                                )
                                if cuda_device is not None:
                                    cmd += f" --cuda-device {int(cuda_device)}"
                                cmds.append(cmd)
    return cmds


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build Segment-LDA OPS weight-recovery sweep command list.")
    p.add_argument("--python-bin", type=str, default="venv/bin/python")
    p.add_argument("--out-cmds", type=str, default="logs/segment_lda_ops_weight_recovery_cmds.txt")
    p.add_argument("--output-root", type=str, default="outputs/segment_lda_ops_weight_recovery")

    p.add_argument("--train-docs", type=str, default="100 200 500 1000 2000")
    p.add_argument("--test-docs", type=int, default=2000)
    p.add_argument("--audit-fractions", type=str, default="0.05 0.1 0.2 0.5 1.0")
    p.add_argument("--topic-phi-docs", type=str, default="0")
    p.add_argument(
        "--topic-phi-estimators",
        type=str,
        default=(
            "true noisy_theory tensor_lda online_tensor_lda embedding_spectral "
            "neural_ctreepo neural_mergeable_sketch neural_hybrid neural_embedding_hybrid"
        ),
        help="Space-separated list.",
    )
    p.add_argument("--topic-processes", type=str, default="segments bag_of_words", help="Space-separated list.")
    p.add_argument("--lambda-multipliers", type=str, default="0 0.25 1.0", help="Space-separated list.")
    p.add_argument("--seeds", type=str, default="0 1 2 3 4 5 6 7")

    p.add_argument("--topic-source", choices=["true", "infer"], default="infer")
    p.add_argument("--feature-inference", choices=["hard", "soft"], default="hard")

    p.add_argument("--n-topics", type=int, default=8)
    p.add_argument("--vocab-size", type=int, default=512)
    p.add_argument("--min-tokens", type=int, default=384)
    p.add_argument("--max-tokens", type=int, default=384)
    p.add_argument("--leaf-tokens", type=int, default=16)
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--cuda-device", type=int, default=None)
    p.add_argument("--torch-threads", type=int, default=0)

    p.add_argument("--run-all-feature-modes", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--skip-existing", action=argparse.BooleanOptionalAction, default=True)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    out_cmds = Path(args.out_cmds)
    out_cmds.parent.mkdir(parents=True, exist_ok=True)
    Path(args.output_root).mkdir(parents=True, exist_ok=True)

    cmds = _iter_commands(
        python_bin=str(args.python_bin),
        train_docs=_parse_ints(args.train_docs),
        test_docs=int(args.test_docs),
        audit_fractions=_parse_floats(args.audit_fractions),
        topic_phi_docs=_parse_ints(args.topic_phi_docs),
        topic_phi_estimators=_parse_items(args.topic_phi_estimators),
        topic_processes=_parse_items(args.topic_processes),
        lambda_multipliers=_parse_floats(args.lambda_multipliers),
        seeds=_parse_ints(args.seeds),
        output_root=Path(args.output_root),
        topic_source=str(args.topic_source),
        feature_inference=str(args.feature_inference),
        n_topics=int(args.n_topics),
        vocab_size=int(args.vocab_size),
        min_tokens=int(args.min_tokens),
        max_tokens=int(args.max_tokens),
        leaf_tokens=int(args.leaf_tokens),
        device=str(args.device),
        cuda_device=(int(args.cuda_device) if args.cuda_device is not None else None),
        torch_threads=int(args.torch_threads),
        run_all_feature_modes=bool(args.run_all_feature_modes),
        skip_existing=bool(args.skip_existing),
    )

    out_cmds.write_text("\n".join(cmds) + ("\n" if cmds else ""), encoding="utf-8")
    print(f"wrote_cmds | {out_cmds} | n_commands={len(cmds)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
