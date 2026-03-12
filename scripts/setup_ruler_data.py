#!/usr/bin/env python3
"""
Provision required RULER synthetic source assets.

Creates (if missing):
- outside_data/RULER/scripts/data/synthetic/json/PaulGrahamEssays.json
- outside_data/RULER/scripts/data/synthetic/json/squad.json
- outside_data/RULER/scripts/data/synthetic/json/hotpotqa.json
"""

from __future__ import annotations

import argparse
import html
import json
import logging
import os
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from html.parser import HTMLParser
from pathlib import Path
from typing import Iterable, List, Optional, Tuple
from urllib.error import URLError
from urllib.request import Request, urlopen


logger = logging.getLogger("setup_ruler_data")


SQUAD_URLS = [
    "https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v2.0.json",
]

HOTPOT_URLS = [
    # Original upstream URL (kept first for provenance; often unavailable).
    "https://curtis.ml.cmu.edu/datasets/hotpot/hotpot_dev_distractor_v1.json",
    # Working mirror validated in this environment.
    "https://huggingface.co/datasets/namlh2004/hotpotqa/resolve/main/hotpot_dev_distractor_v1.json",
]


class _HTMLTextExtractor(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self._parts: List[str] = []

    def handle_data(self, data: str) -> None:
        if data:
            self._parts.append(data)

    def text(self) -> str:
        return " ".join(p.strip() for p in self._parts if p and p.strip())


def _download_bytes(url: str, timeout: float = 30.0) -> bytes:
    req = Request(url, headers={"User-Agent": "ThinkingTrees-RULER-Setup/1.0"})
    with urlopen(req, timeout=max(5.0, float(timeout))) as resp:
        return resp.read()


def _download_to_file(urls: Iterable[str], output_path: Path, *, force: bool) -> Tuple[bool, str]:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists() and not force:
        return True, f"exists: {output_path}"

    last_error = "unknown error"
    for url in urls:
        try:
            payload = _download_bytes(url)
            with tempfile.NamedTemporaryFile(
                mode="wb",
                dir=str(output_path.parent),
                delete=False,
                prefix=f".{output_path.name}.",
                suffix=".tmp",
            ) as tmp:
                tmp.write(payload)
                tmp_path = Path(tmp.name)
            tmp_path.replace(output_path)
            return True, f"downloaded: {url}"
        except Exception as exc:
            last_error = f"{url}: {exc}"
            continue
    return False, last_error


def _fetch_paulgraham_text(url: str, timeout: float = 30.0) -> Tuple[bool, str]:
    try:
        raw = _download_bytes(url, timeout=timeout)
        # Try UTF-8 first; fall back to latin-1 replacement.
        try:
            decoded = raw.decode("utf-8", errors="strict")
        except Exception:
            decoded = raw.decode("latin-1", errors="replace")

        if url.lower().endswith(".html") or "<html" in decoded[:1000].lower():
            parser = _HTMLTextExtractor()
            parser.feed(decoded)
            text = html.unescape(parser.text())
        else:
            text = decoded
        text = " ".join(str(text).split())
        if not text:
            return False, f"{url}: empty payload"
        return True, text
    except Exception as exc:
        return False, f"{url}: {exc}"


def _build_paulgraham_json(
    *,
    urls_file: Path,
    output_path: Path,
    force: bool,
    max_workers: int = 16,
    min_successes: int = 100,
) -> Tuple[bool, str]:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists() and not force:
        return True, f"exists: {output_path}"
    if not urls_file.exists():
        return False, f"missing urls file: {urls_file}"

    urls = [line.strip() for line in urls_file.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not urls:
        return False, f"no URLs found in {urls_file}"

    texts: List[str] = []
    failures: List[str] = []
    with ThreadPoolExecutor(max_workers=max(1, int(max_workers))) as pool:
        futures = {pool.submit(_fetch_paulgraham_text, url): url for url in urls}
        for fut in as_completed(futures):
            ok, payload = fut.result()
            if ok:
                texts.append(payload)
            else:
                failures.append(payload)

    if len(texts) < min_successes:
        preview = "; ".join(failures[:5]) if failures else "no failure details"
        return (
            False,
            f"insufficient successful essay downloads ({len(texts)}/{len(urls)}). failures: {preview}",
        )

    combined = "\n".join(texts).strip()
    if not combined:
        return False, "combined essay text is empty"

    payload = {"text": combined}
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=str(output_path.parent),
        delete=False,
        prefix=f".{output_path.name}.",
        suffix=".tmp",
    ) as tmp:
        json.dump(payload, tmp, ensure_ascii=False)
        tmp_path = Path(tmp.name)
    tmp_path.replace(output_path)

    detail = f"downloaded essays: {len(texts)}/{len(urls)}"
    if failures:
        detail += f" (failures: {len(failures)})"
    return True, detail


def _validate_json_file(path: Path, *, required_key: Optional[str] = None) -> Tuple[bool, str]:
    if not path.exists():
        return False, f"missing: {path}"
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        return False, f"invalid json ({path}): {exc}"
    if required_key is not None:
        if not isinstance(payload, dict):
            return False, f"expected JSON object in {path}"
        if required_key not in payload:
            return False, f"missing key '{required_key}' in {path}"
        if not str(payload.get(required_key, "")).strip():
            return False, f"key '{required_key}' empty in {path}"
    return True, "ok"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Provision RULER synthetic benchmark source JSON files.")
    parser.add_argument("--ruler-dir", type=Path, default=Path("outside_data/RULER"))
    parser.add_argument("--force", action="store_true", help="Re-download and overwrite existing files.")
    parser.add_argument("--skip-paulgraham", action="store_true")
    parser.add_argument("--skip-squad", action="store_true")
    parser.add_argument("--skip-hotpot", action="store_true")
    parser.add_argument("--max-workers", type=int, default=16)
    parser.add_argument("--min-essay-successes", type=int, default=100)
    parser.add_argument("--verbose", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    ruler_dir = Path(args.ruler_dir).resolve()
    json_dir = ruler_dir / "scripts" / "data" / "synthetic" / "json"
    json_dir.mkdir(parents=True, exist_ok=True)

    failures: List[str] = []

    if not args.skip_squad:
        ok, detail = _download_to_file(SQUAD_URLS, json_dir / "squad.json", force=bool(args.force))
        logger.info("squad.json: %s", detail)
        if not ok:
            failures.append(f"squad.json: {detail}")
        else:
            valid, vdetail = _validate_json_file(json_dir / "squad.json")
            if not valid:
                failures.append(f"squad.json validation: {vdetail}")

    if not args.skip_hotpot:
        ok, detail = _download_to_file(HOTPOT_URLS, json_dir / "hotpotqa.json", force=bool(args.force))
        logger.info("hotpotqa.json: %s", detail)
        if not ok:
            failures.append(f"hotpotqa.json: {detail}")
        else:
            valid, vdetail = _validate_json_file(json_dir / "hotpotqa.json")
            if not valid:
                failures.append(f"hotpotqa.json validation: {vdetail}")

    if not args.skip_paulgraham:
        urls_file = json_dir / "PaulGrahamEssays_URLs.txt"
        ok, detail = _build_paulgraham_json(
            urls_file=urls_file,
            output_path=json_dir / "PaulGrahamEssays.json",
            force=bool(args.force),
            max_workers=int(args.max_workers),
            min_successes=int(args.min_essay_successes),
        )
        logger.info("PaulGrahamEssays.json: %s", detail)
        if not ok:
            failures.append(f"PaulGrahamEssays.json: {detail}")
        else:
            valid, vdetail = _validate_json_file(
                json_dir / "PaulGrahamEssays.json",
                required_key="text",
            )
            if not valid:
                failures.append(f"PaulGrahamEssays.json validation: {vdetail}")

    if failures:
        logger.error("RULER setup incomplete:")
        for item in failures:
            logger.error("  - %s", item)
        return 1

    logger.info("RULER synthetic data setup complete: %s", json_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

