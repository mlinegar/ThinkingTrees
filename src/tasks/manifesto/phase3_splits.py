from __future__ import annotations

import random
from pathlib import Path
from typing import Any, Mapping, Sequence

from src.tasks.manifesto import ManifestoDataset
from src.tasks.manifesto.dimensions import PolicyDimension
from src.tasks.manifesto.expert_benchmarks import (
    benoit_ensemble_mean,
    load_benoit_expert_means,
    load_benoit_llm_scores,
    load_benoit_mp_crosswalk,
)


def stratified_take(
    records: list[dict[str, Any]],
    n: int,
    rng: random.Random,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if n <= 0:
        return [], list(records)
    if n > len(records):
        raise ValueError(f"Cannot take {n} records from pool of {len(records)}")
    ordered = sorted(records, key=lambda r: (float(r["label"]), str(r["manifesto_id"])))
    selected_ids: set[int] = set()
    selected: list[dict[str, Any]] = []
    for i in range(n):
        start = i * len(ordered) // n
        end = (i + 1) * len(ordered) // n
        bucket = ordered[start:end] or ordered
        choices = [r for r in bucket if id(r) not in selected_ids]
        if not choices:
            choices = [r for r in ordered if id(r) not in selected_ids]
        rec = rng.choice(choices)
        selected.append(rec)
        selected_ids.add(id(rec))
    remaining = [r for r in records if id(r) not in selected_ids]
    rng.shuffle(selected)
    rng.shuffle(remaining)
    return selected, remaining


def split_records(
    records: list[dict[str, Any]],
    *,
    train_n: int,
    dev_n: int,
    test_n: int,
    split_strategy: str,
    rng: random.Random,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    needed = train_n + dev_n + test_n
    if len(records) < needed:
        raise SystemExit(
            f"Not enough records: have {len(records)}, need train={train_n}+"
            f"dev={dev_n}+test={test_n}={needed}"
        )
    pool = list(records)
    if split_strategy == "random":
        rng.shuffle(pool)
        return pool[:train_n], pool[train_n : train_n + dev_n], pool[train_n + dev_n : needed]
    if split_strategy != "label-stratified":
        raise ValueError(f"unknown split strategy: {split_strategy}")
    train, pool = stratified_take(pool, train_n, rng)
    dev, pool = stratified_take(pool, dev_n, rng)
    test, pool = stratified_take(pool, test_n, rng)
    return train, dev, test


def build_phase3_records(
    dim: PolicyDimension,
    train_pool: str,
    mp_data_dir: Path,
    train_n: int,
    dev_n: int,
    test_n: int,
    seed: int,
    split_strategy: str = "random",
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """Build phase-3 train/dev/test records with text and labels."""

    ds = ManifestoDataset(data_dir=mp_data_dir, require_text=True)
    crosswalk = load_benoit_mp_crosswalk()
    benoit_to_py = {
        row.manifesto: (int(row.party), int(row.year))
        for row in crosswalk.itertuples()
    }
    py_to_mid: dict[tuple[int, int], str] = {}
    for mid in ds.get_all_ids():
        sample = ds.get_sample(mid)
        if sample is None:
            continue
        py_to_mid[(int(sample.party_id), int(sample.year))] = mid

    experts = load_benoit_expert_means(dim)
    test_records = []
    for row in experts.itertuples():
        key = benoit_to_py.get(str(row.manifesto))
        if key is None:
            continue
        mid = py_to_mid.get(key)
        if mid is None:
            continue
        sample = ds.get_sample(mid)
        if sample is None or not sample.text:
            continue
        test_records.append(
            {
                "manifesto_id": mid,
                "benoit_key": str(row.manifesto),
                "text": sample.text,
                "label": float(row.expert_mean_1_7),
                "party": key[0],
                "year": key[1],
            }
        )

    rng = random.Random(seed)

    if train_pool == "expert-split":
        ow_scores = load_benoit_llm_scores(kind="openweight", dimension=dim)
        ow_ens = benoit_ensemble_mean(ow_scores)
        ow_lookup = {row.manifesto: float(row.score_llm_mean) for row in ow_ens.itertuples()}
        all_records = list(test_records)
        train_records, dev_records, test_records = split_records(
            all_records,
            train_n=train_n,
            dev_n=dev_n,
            test_n=test_n,
            split_strategy=split_strategy,
            rng=rng,
        )

        def with_openweight_label(records: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
            out: list[dict[str, Any]] = []
            for rec in records:
                row = dict(rec)
                label = ow_lookup.get(row["benoit_key"])
                if label is not None:
                    row["label"] = float(label)
                    row["label_source"] = "benoit_openweight_ensemble"
                else:
                    row["label_source"] = "benoit_expert_mean_fallback"
                out.append(row)
            return out

        return (
            with_openweight_label(train_records),
            with_openweight_label(dev_records),
            [dict(row, label_source="benoit_expert_mean") for row in test_records],
        )

    if train_pool == "openweight":
        scores = load_benoit_llm_scores(kind="openweight", dimension=dim)
        ensemble = benoit_ensemble_mean(scores)
        train_lookup = {row.manifesto: float(row.score_llm_mean) for row in ensemble.itertuples()}
    elif train_pool == "expert":
        train_lookup = {row.manifesto: float(row.expert_mean_1_7) for row in experts.itertuples()}
    else:
        raise ValueError(train_pool)

    test_keys = set(row["benoit_key"] for row in test_records)
    train_records = []
    for bkey, label in train_lookup.items():
        if bkey in test_keys:
            continue
        key = benoit_to_py.get(bkey)
        if key is None:
            continue
        mid = py_to_mid.get(key)
        if mid is None:
            continue
        sample = ds.get_sample(mid)
        if sample is None or not sample.text:
            continue
        train_records.append(
            {
                "manifesto_id": mid,
                "benoit_key": bkey,
                "text": sample.text,
                "label": float(label),
                "party": key[0],
                "year": key[1],
                "label_source": f"benoit_{train_pool}",
            }
        )

    train, dev, _unused = split_records(
        train_records,
        train_n=train_n,
        dev_n=dev_n,
        test_n=0,
        split_strategy=split_strategy,
        rng=rng,
    )
    if split_strategy == "label-stratified":
        test, _remaining_test = stratified_take(test_records, test_n, rng)
    else:
        test = test_records[:test_n]
    return train, dev, [dict(row, label_source="benoit_expert_mean") for row in test]


def build_phase3_examples(
    dim: PolicyDimension,
    train_pool: str,
    mp_data_dir: Path,
    train_n: int,
    dev_n: int,
    test_n: int,
    seed: int,
    split_strategy: str = "random",
):
    import dspy

    train_records, dev_records, test_records = build_phase3_records(
        dim,
        train_pool,
        mp_data_dir,
        train_n,
        dev_n,
        test_n,
        seed,
        split_strategy=split_strategy,
    )

    def ex(record: Mapping[str, Any]):
        return dspy.Example(
            text=record["text"],
            expert_mean=record["label"],
            manifesto_id=record["manifesto_id"],
        ).with_inputs("text")

    return [ex(r) for r in train_records], [ex(r) for r in dev_records], [ex(r) for r in test_records]


__all__ = [
    "build_phase3_examples",
    "build_phase3_records",
    "split_records",
    "stratified_take",
]
