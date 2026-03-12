#!/usr/bin/env python3
"""
Download Congressional data sources for replication work.

Targets:
  - Volden & Wiseman (Center for Effective Lawmaking / TheLawmakers.org)
  - Congressional Bills Project (Adler & Wilkerson)
"""

import argparse
import datetime as dt
import hashlib
import json
import logging
import re
import sys
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

USER_AGENT = "ThinkingTreesDownloader/1.0 (+https://thelawmakers.org)"
LAWMAKERS_PAGE_ID = 542
LAWMAKERS_PAGE_URL = f"https://thelawmakers.org/wp-json/wp/v2/pages/{LAWMAKERS_PAGE_ID}"
CBP_DOWNLOAD_PAGE = "http://www.congressionalbills.org/download.html"

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download Volden/Wiseman labels and Congressional Bills text files."
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("data/raw/congressional_bills"),
        help="Base output directory for downloaded files.",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=20,
        help="Network timeout in seconds (default: 20).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite files if they already exist.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show downloads without fetching files.",
    )

    parser.add_argument(
        "--skip-lawmakers",
        action="store_true",
        help="Skip downloading the Lawmakers data.",
    )
    parser.add_argument(
        "--lawmakers-url",
        type=str,
        default=None,
        help="Direct URL for the House data file (overrides auto-discovery).",
    )
    parser.add_argument(
        "--include-sles",
        action="store_true",
        help="Also download SLES individual data if present on the Lawmakers page.",
    )

    parser.add_argument(
        "--skip-cbp",
        action="store_true",
        help="Skip downloading the Congressional Bills Project files.",
    )
    parser.add_argument(
        "--cbp-download-page",
        type=str,
        default=CBP_DOWNLOAD_PAGE,
        help="CBP download page URL (default: congressionalbills.org/download.html).",
    )
    parser.add_argument(
        "--cbp-titles-url",
        type=str,
        default=None,
        help="Direct URL for the Bill Titles file.",
    )
    parser.add_argument(
        "--cbp-summaries-url",
        type=str,
        default=None,
        help="Direct URL for the Bill Summaries file.",
    )
    parser.add_argument(
        "--require-cbp",
        action="store_true",
        help="Fail if CBP downloads cannot be resolved.",
    )

    return parser.parse_args()


def fetch_text(url: str, timeout: int) -> str:
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=timeout) as resp:
        return resp.read().decode("utf-8", errors="replace")


def fetch_json(url: str, timeout: int) -> Dict:
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=timeout) as resp:
        return json.load(resp)


def extract_links(html: str) -> List[str]:
    return re.findall(r'href=["\\\']([^"\\\']+)["\\\']', html)


def resolve_lawmakers_urls(page_url: str, timeout: int) -> Tuple[str, Optional[str]]:
    data = fetch_json(page_url, timeout)
    content = data.get("content", {}).get("rendered", "")
    links = extract_links(content)
    xlsx_links = [l for l in links if l.lower().endswith(".xlsx")]

    if not xlsx_links:
        raise RuntimeError("No XLSX links found on the Lawmakers data page.")

    def score_link(link: str) -> int:
        link_lower = link.lower()
        score = 0
        if "celhouse" in link_lower or "house" in link_lower:
            score += 10
        if "revised" in link_lower or "update" in link_lower:
            score += 2
        return score

    house_url = max(xlsx_links, key=score_link)
    sles_url = next((l for l in xlsx_links if "sles" in l.lower()), None)
    return house_url, sles_url


def resolve_cbp_urls(
    download_page: str,
    timeout: int,
    titles_url: Optional[str],
    summaries_url: Optional[str],
) -> Tuple[Optional[str], Optional[str]]:
    if titles_url and summaries_url:
        return titles_url, summaries_url

    html = fetch_text(download_page, timeout)
    links = extract_links(html)
    absolute_links = [urllib.parse.urljoin(download_page, l) for l in links]

    def pick_link(keyword: str) -> Optional[str]:
        matches = [l for l in absolute_links if keyword in l.lower()]
        if not matches:
            return None
        matches.sort(key=lambda x: len(x), reverse=True)
        return matches[0]

    titles_candidate = titles_url or pick_link("title")
    summaries_candidate = summaries_url or pick_link("summary")
    return titles_candidate, summaries_candidate


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_file(url: str, dest: Path, timeout: int, overwrite: bool, dry_run: bool) -> Optional[Dict]:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and not overwrite:
        logger.info("Skip existing file: %s", dest)
        return None

    if dry_run:
        logger.info("Dry run: %s -> %s", url, dest)
        return None

    tmp_path = dest.with_suffix(dest.suffix + ".partial")
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as resp, tmp_path.open("wb") as handle:
            while True:
                chunk = resp.read(1024 * 1024)
                if not chunk:
                    break
                handle.write(chunk)
    except Exception:
        if tmp_path.exists():
            tmp_path.unlink()
        raise

    tmp_path.replace(dest)
    meta = {
        "url": url,
        "path": str(dest),
        "bytes": dest.stat().st_size,
        "sha256": sha256_file(dest),
        "downloaded_at": dt.datetime.utcnow().isoformat(timespec="seconds") + "Z",
    }
    return meta


def write_metadata(path: Path, meta: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(meta, handle, indent=2, sort_keys=True)
        handle.write("\n")


def main() -> int:
    args = parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    base_dir = args.out_dir

    if not args.skip_lawmakers:
        lawmakers_dir = base_dir / "thelawmakers"
        try:
            if args.lawmakers_url:
                house_url = args.lawmakers_url
                sles_url = None
            else:
                house_url, sles_url = resolve_lawmakers_urls(LAWMAKERS_PAGE_URL, args.timeout)

            house_name = Path(urllib.parse.urlparse(house_url).path).name
            house_dest = lawmakers_dir / house_name
            meta = download_file(house_url, house_dest, args.timeout, args.overwrite, args.dry_run)
            if meta:
                write_metadata(house_dest.with_suffix(house_dest.suffix + ".meta.json"), meta)
                logger.info("Downloaded House data to %s", house_dest)

            if args.include_sles and sles_url:
                sles_name = Path(urllib.parse.urlparse(sles_url).path).name
                sles_dest = lawmakers_dir / sles_name
                meta = download_file(sles_url, sles_dest, args.timeout, args.overwrite, args.dry_run)
                if meta:
                    write_metadata(sles_dest.with_suffix(sles_dest.suffix + ".meta.json"), meta)
                    logger.info("Downloaded SLES data to %s", sles_dest)
        except Exception as exc:
            logger.error("Failed to download Lawmakers data: %s", exc)
            return 1

    if not args.skip_cbp:
        cbp_dir = base_dir / "congressional_bills_project"
        try:
            titles_url, summaries_url = resolve_cbp_urls(
                args.cbp_download_page,
                args.timeout,
                args.cbp_titles_url,
                args.cbp_summaries_url,
            )
        except Exception as exc:
            logger.error("Failed to fetch CBP download page: %s", exc)
            titles_url, summaries_url = None, None

        if not titles_url or not summaries_url:
            message = (
                "Could not resolve CBP file URLs. "
                "Provide --cbp-titles-url and --cbp-summaries-url."
            )
            if args.require_cbp:
                logger.error(message)
                return 1
            logger.warning(message)
            return 0

        titles_name = Path(urllib.parse.urlparse(titles_url).path).name
        summaries_name = Path(urllib.parse.urlparse(summaries_url).path).name

        meta = download_file(titles_url, cbp_dir / titles_name, args.timeout, args.overwrite, args.dry_run)
        if meta:
            write_metadata((cbp_dir / titles_name).with_suffix(Path(titles_name).suffix + ".meta.json"), meta)
            logger.info("Downloaded Bill Titles to %s", cbp_dir / titles_name)

        meta = download_file(
            summaries_url, cbp_dir / summaries_name, args.timeout, args.overwrite, args.dry_run
        )
        if meta:
            write_metadata((cbp_dir / summaries_name).with_suffix(Path(summaries_name).suffix + ".meta.json"), meta)
            logger.info("Downloaded Bill Summaries to %s", cbp_dir / summaries_name)

    return 0


if __name__ == "__main__":
    sys.exit(main())
