#!/usr/bin/env python3
"""
Fetch raw manifesto text from the Manifesto Project API for the full list
of manifestos that appear in Benoit's replication archive.

Runs once. Resolves each Benoit `(party, year)` to an MP `(party, date_code)`
via the MPDS metadata already in the repo, then hits the MP API for the
sentence-level text and reassembles to plain-text per manifesto.

Prereqs
-------
- outside_data/Manifesto_Project/manifesto_apikey.txt  (API key)
- outside_data/Manifesto_Project/MPDataset_MPDS2025a.csv  (for date lookup)
- data/examples/benoit_dataverse/data_mp.rda           (Benoit manifest list)

Output
------
- data/raw/manifesto_corpus_benoit/texts/{party}_{date}.txt
- data/raw/manifesto_corpus_benoit/manifesto_maindataset.csv  (subset of MPDS matching the fetched set; lets us point ManifestoDataset at this dir)
- data/raw/manifesto_corpus_benoit/fetch_manifest.json        (per-key status)

Usage
-----
    python scripts/fetch_mp_text.py \\
        --version 2025-1 \\
        --output-dir data/raw/manifesto_corpus_benoit \\
        --rate 0.2
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import pandas as pd
import pyreadr
import requests

project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

logger = logging.getLogger(__name__)

API_BASE = "https://manifesto-project.wzb.eu/tools"
TEXTS_ENDPOINT = f"{API_BASE}/api_texts_and_annotations.json"
VERSIONS_ENDPOINT = f"{API_BASE}/api_list_metadata_versions.json"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--api-key-file", type=Path,
                   default=project_root / "outside_data" / "Manifesto_Project" / "manifesto_apikey.txt")
    p.add_argument("--mpds-csv", type=Path,
                   default=project_root / "outside_data" / "Manifesto_Project" / "MPDataset_MPDS2025a.csv")
    p.add_argument("--benoit-mp-rda", type=Path,
                   default=project_root / "data" / "examples" / "benoit_dataverse" / "data_mp.rda")
    p.add_argument("--benoit-experts-rda", type=Path,
                   default=project_root / "data" / "examples" / "benoit_dataverse" / "data_experts.rda")
    p.add_argument("--output-dir", type=Path,
                   default=project_root / "data" / "raw" / "manifesto_corpus_benoit")
    p.add_argument("--version", type=str, default="2025-1",
                   help="MP corpus version. Use 'latest' to auto-discover.")
    p.add_argument("--rate", type=float, default=0.2, help="Sleep seconds between API calls")
    p.add_argument("--timeout", type=float, default=30.0)
    p.add_argument("--force", action="store_true", help="Re-fetch even if text file already exists")
    p.add_argument("--dry-run", action="store_true", help="List the keys to fetch; skip API calls")
    p.add_argument("--max-keys", type=int, default=None, help="Cap on number of manifestos (debug)")
    p.add_argument("--all-mpds", action="store_true",
                   help="Fetch every MPDS (party, date) with valid metadata, not just Benoit's 233.")
    p.add_argument("--log-level", type=str, default="INFO")
    return p.parse_args()


def resolve_version(api_key: str, version: str, timeout: float) -> str:
    if version != "latest":
        return version
    r = requests.get(VERSIONS_ENDPOINT, params={"api_key": api_key}, timeout=timeout)
    r.raise_for_status()
    versions = r.json()["versions"]
    return versions[-1]


def build_key_list_benoit(
    benoit_mp_rda: Path,
    benoit_experts_rda: Path,
    mpds_csv: Path,
) -> pd.DataFrame:
    """Return a DataFrame of Benoit's 233 manifestos with columns:
    key, party, date, manifesto, year, partyname, country, source."""
    benoit_mp = pyreadr.read_r(str(benoit_mp_rda))["data_mp"]
    keys = benoit_mp[["manifesto", "country", "year", "party", "date", "partyname"]].dropna(
        subset=["party", "date"]
    ).copy()
    keys["party"] = keys["party"].astype(int)
    keys["date"] = keys["date"].astype(int)
    keys["year"] = keys["year"].astype(int)
    keys["country"] = keys["country"].astype(int)
    keys["key"] = keys["party"].astype(str) + "_" + keys["date"].astype(str)
    keys["source"] = "benoit_mp"

    experts = pyreadr.read_r(str(benoit_experts_rda))["data_experts"]
    missing = set(experts["manifesto"].dropna().unique()) - set(keys["manifesto"].dropna().unique())
    if missing:
        logger.info("manifestos in data_experts but not in data_mp: %d", len(missing))
        for mfesto in sorted(missing):
            logger.warning("No MP row for Benoit manifesto=%r; will skip unless resolvable", mfesto)

    return keys.reset_index(drop=True).drop_duplicates("key")


def build_key_list_all_mpds(mpds_csv: Path) -> pd.DataFrame:
    """Return every (party, date) in MPDS 2025a metadata."""
    df = pd.read_csv(mpds_csv, low_memory=False).dropna(subset=["party", "date"]).copy()
    df["party"] = df["party"].astype(int)
    df["date"] = df["date"].astype(int)
    df["year"] = (df["date"] // 100).astype(int)
    if "country" in df:
        df["country"] = df["country"].astype(int)
    df["key"] = df["party"].astype(str) + "_" + df["date"].astype(str)
    df["manifesto"] = df.get("partyname", "").astype(str) + " " + df["year"].astype(str)
    if "partyname" not in df:
        df["partyname"] = ""
    df["source"] = "mpds"
    return df[["key", "party", "date", "year", "country", "partyname", "manifesto", "source"]].drop_duplicates("key").reset_index(drop=True)


def fetch_one(
    key: str,
    api_key: str,
    version: str,
    timeout: float,
) -> dict:
    r = requests.get(
        TEXTS_ENDPOINT,
        params={"keys[]": key, "api_key": api_key, "version": version},
        timeout=timeout,
    )
    r.raise_for_status()
    return r.json()


def reassemble_text(items: list) -> str:
    """Join sentence-level `text` fields into a single newline-separated string."""
    return "\n".join((it.get("text") or "").strip() for it in items if it.get("text"))


def main() -> int:
    args = parse_args()
    logging.basicConfig(level=args.log_level.upper(), format="%(asctime)s %(levelname)s %(name)s | %(message)s")

    api_key = args.api_key_file.read_text().strip()
    version = resolve_version(api_key, args.version, args.timeout)
    logger.info("Using MP corpus version %s", version)

    if args.all_mpds:
        keys_df = build_key_list_all_mpds(args.mpds_csv)
        logger.info("Full MPDS key list: %d rows", len(keys_df))
    else:
        keys_df = build_key_list_benoit(args.benoit_mp_rda, args.benoit_experts_rda, args.mpds_csv)
        logger.info("Benoit key list: %d rows", len(keys_df))

    texts_dir = args.output_dir / "texts"
    texts_dir.mkdir(parents=True, exist_ok=True)

    manifest_rows: list[dict] = []
    to_fetch = keys_df if args.max_keys is None else keys_df.head(args.max_keys)
    logger.info("Planning to fetch %d manifestos into %s", len(to_fetch), texts_dir)

    if args.dry_run:
        print(to_fetch[["key", "manifesto", "partyname", "year"]].to_string(index=False))
        return 0

    t0 = time.time()
    n_ok = n_skip = n_miss = n_err = 0

    for i, row in enumerate(to_fetch.itertuples(), start=1):
        text_path = texts_dir / f"{row.key}.txt"
        if text_path.exists() and text_path.stat().st_size > 0 and not args.force:
            n_skip += 1
            manifest_rows.append(
                {"key": row.key, "manifesto": row.manifesto, "partyname": row.partyname,
                 "year": row.year, "status": "skip_exists", "n_items": None,
                 "text_chars": text_path.stat().st_size}
            )
            continue

        try:
            data = fetch_one(row.key, api_key, version, args.timeout)
        except requests.HTTPError as e:
            logger.error("[%d/%d] %s: HTTP %s", i, len(to_fetch), row.key, e.response.status_code)
            manifest_rows.append(
                {"key": row.key, "manifesto": row.manifesto, "partyname": row.partyname,
                 "year": row.year, "status": f"http_error_{e.response.status_code}",
                 "n_items": None, "text_chars": 0}
            )
            n_err += 1
            continue
        except requests.RequestException as e:
            logger.error("[%d/%d] %s: request error %s", i, len(to_fetch), row.key, e)
            manifest_rows.append(
                {"key": row.key, "manifesto": row.manifesto, "partyname": row.partyname,
                 "year": row.year, "status": f"request_error", "n_items": None, "text_chars": 0}
            )
            n_err += 1
            continue

        if data.get("missing_items"):
            logger.warning("[%d/%d] %s: MP API reports missing (%s)", i, len(to_fetch), row.key, data["missing_items"])

        items_outer = data.get("items") or []
        items = items_outer[0].get("items") if items_outer and isinstance(items_outer[0], dict) else []
        if not items:
            logger.warning("[%d/%d] %s: no text items returned", i, len(to_fetch), row.key)
            manifest_rows.append(
                {"key": row.key, "manifesto": row.manifesto, "partyname": row.partyname,
                 "year": row.year, "status": "no_items", "n_items": 0, "text_chars": 0}
            )
            n_miss += 1
            time.sleep(args.rate)
            continue

        text = reassemble_text(items)
        text_path.write_text(text, encoding="utf-8")
        manifest_rows.append(
            {"key": row.key, "manifesto": row.manifesto, "partyname": row.partyname,
             "year": row.year, "status": "ok", "n_items": len(items),
             "text_chars": len(text)}
        )
        n_ok += 1
        if i % 10 == 0 or i == len(to_fetch):
            logger.info("[%d/%d] ok=%d skip=%d miss=%d err=%d elapsed=%.1fs",
                        i, len(to_fetch), n_ok, n_skip, n_miss, n_err, time.time() - t0)
        time.sleep(args.rate)

    # Write the manifest
    manifest_path = args.output_dir / "fetch_manifest.json"
    manifest = {
        "run": {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "mp_version": version,
            "n_fetched": n_ok,
            "n_skipped": n_skip,
            "n_missing": n_miss,
            "n_error": n_err,
            "elapsed_seconds": round(time.time() - t0, 1),
        },
        "rows": manifest_rows,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2))

    # Build the MPDS-like maindataset CSV for ManifestoDataset compatibility.
    # Use every key that has a non-empty text file on disk — this includes
    # everything we just fetched plus everything from prior runs (skip_exists).
    try:
        mpds_full = pd.read_csv(args.mpds_csv, low_memory=False)
        keep_keys = {p.stem for p in texts_dir.glob("*.txt") if p.stat().st_size > 0}
        mpds_full["_key"] = mpds_full["party"].astype(str) + "_" + mpds_full["date"].astype(str)
        subset = mpds_full[mpds_full["_key"].isin(keep_keys)].drop(columns="_key")
        out_csv = args.output_dir / "manifesto_maindataset.csv"
        subset.to_csv(out_csv, index=False)
        logger.info("Wrote MPDS subset CSV (%d rows) to %s", len(subset), out_csv)
    except Exception as e:  # noqa: BLE001
        logger.warning("Could not build MPDS subset: %s", e)

    logger.info("DONE: ok=%d skip=%d miss=%d err=%d in %.1fs",
                n_ok, n_skip, n_miss, n_err, time.time() - t0)
    logger.info("Artifacts in %s", args.output_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
