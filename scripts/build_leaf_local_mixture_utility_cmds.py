#!/usr/bin/env python3
"""Build xargs-friendly command lists for the Stage-2 leaf-local-mixture family."""

from __future__ import annotations

import argparse
from fractions import Fraction
from pathlib import Path
from typing import Iterable, List

DEFAULT_LAW_WEIGHT = 1.0 / 3.0


def _parse_items(text: str) -> List[str]:
    out: List[str] = []
    for raw in str(text).replace(",", " ").split():
        item = raw.strip()
        if item:
            out.append(item)
    return out


def _parse_ints(text: str) -> List[int]:
    return [int(x) for x in _parse_items(text)]


def _parse_floats(text: str) -> List[float]:
    return [float(x) for x in _parse_items(text)]


def _fmt_float(x: float) -> str:
    return f"{float(x):.6g}".replace("-", "m").replace(".", "p")


def _fraction_dir_label(text: str) -> str:
    frac = Fraction(str(text))
    pct = 100.0 * float(frac)
    if abs(pct - round(pct)) <= 1e-9:
        return f"doc_{int(round(pct))}pct"
    return f"doc_{str(text).replace('/', 'of').replace('.', 'p')}"


def _apply_law_objective_path_labels(
    base: Path,
    *,
    law_set_id: str,
    local_law_weight: float,
) -> Path:
    if str(law_set_id).strip() and str(law_set_id).strip() != "all":
        base = base / f"lawset_{str(law_set_id).strip()}"
    if abs(float(local_law_weight) - 0.5) > 1e-12:
        base = base / f"llw_{_fmt_float(local_law_weight)}"
    return base


def _iter_commands(
    *,
    python_bin: str,
    output_root: Path,
    leaf_fractions: Iterable[str],
    doc_topic_concentrations: Iterable[float],
    taus: Iterable[float],
    quadratic_utility_weights: Iterable[float],
    budgets: Iterable[float],
    budget_regimes: Iterable[str],
    latent_leaf_tokens_list: Iterable[int],
    local_law_weights: Iterable[float],
    law_set_id: str,
    seeds: Iterable[int],
    skip_existing: bool,
    doc_tokens: int,
    train_docs: int,
    test_docs: int,
) -> List[str]:
    script = "scripts/run_leaf_local_mixture_utility_simulation.py"
    cmds: List[str] = []
    local_law_weight_values = [float(x) for x in local_law_weights]
    if not local_law_weight_values:
        local_law_weight_values = [0.5]
    for latent_leaf_tok in latent_leaf_tokens_list:
        if int(doc_tokens) % int(latent_leaf_tok) != 0:
            continue
        for leaf_frac in leaf_fractions:
            frac_val = float(Fraction(str(leaf_frac)))
            eval_leaf_tok = max(1, min(int(doc_tokens), int(round(float(doc_tokens) * frac_val))))
            if int(doc_tokens) % int(eval_leaf_tok) != 0:
                continue
            if int(eval_leaf_tok) % int(latent_leaf_tok) != 0:
                continue
            leaf_label = _fraction_dir_label(leaf_frac)
            for alpha in doc_topic_concentrations:
                for tau in taus:
                    for qweight in quadratic_utility_weights:
                        for regime in budget_regimes:
                            active_budgets = (
                                [0.0] if str(regime) == "all_leaves_labeled" else list(budgets)
                            )
                            for budget in active_budgets:
                                for local_law_weight in local_law_weight_values:
                                    for seed in seeds:
                                        base = (
                                            output_root
                                            / f"llt_{latent_leaf_tok}"
                                            / leaf_label
                                            / f"dtc_{alpha:g}"
                                            / f"tau_{tau:g}"
                                            / f"qweight_{qweight:g}"
                                            / str(regime)
                                            / f"budget_{budget:g}"
                                        )
                                        base = _apply_law_objective_path_labels(
                                            base,
                                            law_set_id=str(law_set_id).strip() or "all",
                                            local_law_weight=float(local_law_weight),
                                        )
                                        base = base / f"seed_{seed}"
                                        out_json = base.with_suffix(".json")
                                        out_csv = base.with_suffix(".csv")
                                        if skip_existing and out_json.exists() and out_csv.exists():
                                            continue
                                        cmd = (
                                            f"{python_bin} -u {script} "
                                            f"--doc-tokens {int(doc_tokens)} "
                                            f"--doc-topic-concentration {float(alpha)} "
                                            f"--latent-leaf-tokens {int(latent_leaf_tok)} "
                                            f"--leaf-fraction {leaf_frac} "
                                            f"--local-mixture-concentration {float(tau)} "
                                            f"--quadratic-utility-weight {float(qweight)} "
                                            f"--budget-regime {regime} "
                                            f"--leaf-label-budget {float(budget if budget > 0.0 else 8.0)} "
                                            f"--law-set-id {str(law_set_id).strip() or 'all'} "
                                            f"--local-law-weight {float(local_law_weight)} "
                                            f"--train-docs {int(train_docs)} --test-docs {int(test_docs)} "
                                            f"--seed {int(seed)} "
                                            f"--json-summary {out_json} --csv-summary {out_csv}"
                                        )
                                        cmds.append(cmd)
    return cmds


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Build command lists for Stage-2 leaf-local-mixture sweeps."
    )
    p.add_argument("--python-bin", type=str, default="venv/bin/python")
    p.add_argument("--out-cmds", type=str, default="logs/leaf_local_mixture_utility_cmds.txt")
    p.add_argument("--output-root", type=str, default="outputs/leaf_local_mixture_utility")
    p.add_argument("--latent-leaf-tokens", type=str, default="16")
    p.add_argument("--leaf-fractions", type=str, default="1 1/2 1/4 1/24")
    p.add_argument("--doc-topic-concentrations", type=str, default="0.2 0.6 1.5")
    p.add_argument("--local-mixture-concentrations", type=str, default="64 8 1 0.25")
    p.add_argument("--quadratic-utility-weights", type=str, default="0 1 2")
    p.add_argument("--budget-regimes", type=str, default="all_leaves_labeled fixed_oracle_budget")
    p.add_argument("--leaf-label-budgets", type=str, default="2 4 8 16 24")
    p.add_argument("--local-law-weights", type=str, default="0.5")
    p.add_argument("--law-set-id", type=str, default="all")
    p.add_argument("--seeds", type=str, default="0 1 2 3 4")
    p.add_argument("--doc-tokens", type=int, default=384)
    p.add_argument("--train-docs", type=int, default=512)
    p.add_argument("--test-docs", type=int, default=256)
    p.add_argument("--skip-existing", action=argparse.BooleanOptionalAction, default=True)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    out_cmds = Path(args.out_cmds)
    out_cmds.parent.mkdir(parents=True, exist_ok=True)
    Path(args.output_root).mkdir(parents=True, exist_ok=True)
    cmds = _iter_commands(
        python_bin=str(args.python_bin),
        output_root=Path(args.output_root),
        leaf_fractions=_parse_items(args.leaf_fractions),
        doc_topic_concentrations=_parse_floats(args.doc_topic_concentrations),
        taus=_parse_floats(args.local_mixture_concentrations),
        quadratic_utility_weights=_parse_floats(args.quadratic_utility_weights),
        budgets=_parse_floats(args.leaf_label_budgets),
        budget_regimes=_parse_items(args.budget_regimes),
        latent_leaf_tokens_list=_parse_ints(args.latent_leaf_tokens),
        local_law_weights=_parse_floats(args.local_law_weights),
        law_set_id=str(args.law_set_id).strip() or "all",
        seeds=_parse_ints(args.seeds),
        skip_existing=bool(args.skip_existing),
        doc_tokens=int(args.doc_tokens),
        train_docs=int(args.train_docs),
        test_docs=int(args.test_docs),
    )
    out_cmds.write_text("\n".join(cmds) + ("\n" if cmds else ""), encoding="utf-8")
    print(f"wrote_cmds | {out_cmds} | n_commands={len(cmds)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
