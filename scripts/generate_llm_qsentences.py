#!/usr/bin/env python3
"""Stage 1 of the LLM-recreation experiment: have the LLM (gemma) re-segment each
manifesto doc into its OWN quasi-sentences ("simulated" segmentation), at roughly
the GOLD granularity, and write a corpus CSV in the same shape as
``manifesto_corpus_df.csv`` so the existing q-sentence grid builder can consume it
via ``--corpus-csv``.

The point: the gold human q-sentence segmentation is an oracle not available at
deploy time. This produces the LLM's own segmentation so we can test whether
g(f(llm qsents)) reaches g(f(gold qsents)) (g aligned with gold as supervision).

Reuses the same BatchedDSPyLM client as score_benoit_chunks.py (gemma-4-31B on
ports 8010-8013). Matches gold granularity: we tell the LLM the target number of
quasi-sentences (= the doc's gold qsentence count) so the simulated tree is
structurally comparable for a clean A/B.
"""
from __future__ import annotations

import argparse
import logging
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.tasks.manifesto.span_annotations import (  # noqa: E402
    DEFAULT_QSENTENCE_CORPUS,
    load_manifesto_qsentences,
)

LOGGER = logging.getLogger("generate_llm_qsentences")


def _char_windows(text: str, window_chars: int) -> List[str]:
    """Split text into <=window_chars windows on whitespace boundaries (no mid-word cut)."""
    if len(text) <= window_chars:
        return [text]
    windows: List[str] = []
    i = 0
    n = len(text)
    while i < n:
        end = min(i + window_chars, n)
        if end < n:
            # back off to the last whitespace so we don't split a token
            sp = text.rfind(" ", i + window_chars // 2, end)
            if sp > i:
                end = sp
        windows.append(text[i:end])
        i = end
    return windows


def _segment_window(text: str, target_n: int, predictor, dim_lm) -> List[str]:
    import dspy
    try:
        with dspy.context(lm=dim_lm):
            res = predictor(document=text, target_count=str(int(max(1, target_n))))
        raw = str(getattr(res, "quasi_sentences", "") or "")
    except Exception as exc:  # noqa: BLE001
        LOGGER.debug("window segmentation failed: %s", exc)
        return []
    parts = [p.strip(" \t-•").strip() for p in raw.splitlines()]
    return [p for p in parts if p]


def _split_into_quasi_sentences(
    text: str, target_n: int, predictor, dim_lm, *, window_chars: int = 12000
) -> List[str]:
    """Split a doc into ~target_n quasi-sentences, WINDOWING long docs so each LLM
    call stays within context. Long manifestos (up to ~1.4M chars / ~1900 gold
    q-sentences) overflow a single prompt; we split into <=window_chars windows,
    segment each with a proportional target count, and concatenate in order."""
    windows = _char_windows(text, window_chars)
    if len(windows) == 1:
        return _segment_window(text, target_n, predictor, dim_lm)
    out: List[str] = []
    total_chars = max(1, len(text))
    for w in windows:
        wn = max(1, round(target_n * len(w) / total_chars))
        out.extend(_segment_window(w, wn, predictor, dim_lm))
    return out


def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--corpus-csv", type=Path, default=DEFAULT_QSENTENCE_CORPUS)
    p.add_argument("--benoit-targets", type=Path,
                   default=PROJECT_ROOT / "outputs/benoit_qsentence_targets/expert_means_raw.json")
    p.add_argument("--output-csv", type=Path,
                   default=PROJECT_ROOT / "outputs/benoit_llmseg/manifesto_corpus_llmseg.csv")
    p.add_argument("--model", default="openai/nvidia/Gemma-4-31B-IT-NVFP4")
    p.add_argument("--api-base",
                   default="http://localhost:8010/v1,http://localhost:8011/v1,"
                           "http://localhost:8012/v1,http://localhost:8013/v1")
    p.add_argument("--api-key", default="EMPTY")
    p.add_argument("--max-concurrent", type=int, default=128)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--num-threads", type=int, default=128)
    p.add_argument("--max-tokens", type=int, default=4096)
    p.add_argument("--max-docs", type=int, default=None)
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    import json
    targets = json.load(open(args.benoit_targets))
    requested = list(targets.keys())

    # Gold q-sentences per doc: gives us (a) the full doc text (join), (b) the gold
    # count = the target granularity for the LLM.
    grouped = load_manifesto_qsentences(args.corpus_csv, manifesto_ids=requested)
    doc_ids = [m for m in grouped if m in targets and targets[m]]
    if args.max_docs is not None:
        doc_ids = doc_ids[: int(args.max_docs)]
    LOGGER.info("segmenting %d docs (gemma, ~gold granularity)", len(doc_ids))

    import dspy
    from src.config.dspy_config import configure_dspy
    from src.core.dspy_batch_client import BatchedDSPyLM

    lm = BatchedDSPyLM(
        model=str(args.model), api_base=str(args.api_base), api_key=str(args.api_key),
        temperature=0.0, max_tokens=int(args.max_tokens), cache=False,
        max_concurrent=int(args.max_concurrent), batch_size=int(args.batch_size),
        batch_timeout=0.02,
    )
    configure_dspy(lm=lm)

    class SegmentSignature(dspy.Signature):
        """Split a political manifesto document into quasi-sentences: minimal text
        units each expressing ONE policy statement or claim, in the document's
        original language. Preserve the original wording (do not paraphrase or
        translate). Aim for about ``target_count`` units. Output ONE quasi-sentence
        per line, in document order, with no numbering or bullets."""

        document: str = dspy.InputField(desc="Full manifesto document text.")
        target_count: str = dspy.InputField(desc="Approximate number of quasi-sentences to produce.")
        quasi_sentences: str = dspy.OutputField(desc="One quasi-sentence per line, document order.")

    predictor = dspy.Predict(SegmentSignature)

    # FLAT (doc, window) work queue. Earlier this parallelized over DOCS and looped
    # a long doc's windows SERIALLY inside one thread, so the long-doc tail (a few
    # 2000+ q-sentence docs) pinned each doc to one serial chain and STARVED the
    # fleet (KV cache <13%, ~2 reqs/replica, ~70 tok/s on the tail). Flattening every
    # window into one queue keeps hundreds of windows in flight -> the fleet saturates
    # regardless of doc-length distribution. Results reassembled per-doc by window idx.
    window_chars = 12000
    gold_n_by_doc: Dict[str, int] = {}
    tasks: List[tuple] = []  # (doc_id, win_idx, win_text, win_target_n)
    for doc_id in doc_ids:
        qs = grouped[doc_id]
        full_text = " ".join(s.text for s in qs)
        gold_n = len(qs)
        gold_n_by_doc[doc_id] = gold_n
        windows = _char_windows(full_text, window_chars)
        total_chars = max(1, len(full_text))
        for wi, w in enumerate(windows):
            wn = gold_n if len(windows) == 1 else max(1, round(gold_n * len(w) / total_chars))
            tasks.append((doc_id, wi, w, wn))
    LOGGER.info("flattened to %d window-tasks across %d docs", len(tasks), len(doc_ids))

    def _seg_window_task(task):
        doc_id, wi, w, wn = task
        return doc_id, wi, _segment_window(w, wn, predictor, lm)

    # doc_id -> {win_idx: [parts]}, reassembled in window order after all complete
    parts_by_doc: Dict[str, Dict[int, List[str]]] = {d: {} for d in doc_ids}
    done = 0
    with ThreadPoolExecutor(max_workers=int(args.num_threads)) as ex:
        futs = [ex.submit(_seg_window_task, t) for t in tasks]
        for fut in as_completed(futs):
            doc_id, wi, parts = fut.result()
            parts_by_doc[doc_id][wi] = parts
            done += 1
            if done % 200 == 0:
                LOGGER.info("  %d/%d window-tasks done", done, len(tasks))

    rows: List[Dict[str, object]] = []
    n_fail = 0
    for i, doc_id in enumerate(doc_ids):
        gold_n = gold_n_by_doc[doc_id]
        wd = parts_by_doc[doc_id]
        parts = [p for wi in sorted(wd) for p in wd[wi]]  # concat windows in order
        if not parts:
            n_fail += 1
            continue
        # one CSV row per LLM quasi-sentence. cmp_code='000' = the CMP "non-policy"
        # label (valid NON-NULL; load_manifesto_qsentences drops null-code rows) —
        # per-leaf CMP codes are unused here (doc-level scores broadcast / LLM-scored).
        # annotations='True': the loader defaults to require_annotations=True and
        # DROPS falsey-annotation rows, so every kept segment is marked annotated.
        for pos, text in enumerate(parts):
            rows.append({
                "text": text, "cmp_code": "000", "eu_code": "", "pos": pos,
                "manifesto_id": doc_id, "party": "", "date": "",
                "language": "", "annotations": "True", "translation_en": "",
            })
        if (i + 1) % 25 == 0:
            LOGGER.info("  %d/%d docs reassembled (last %s: gold=%d llm=%d)",
                        i + 1, len(doc_ids), doc_id, gold_n, len(parts))

    out = Path(args.output_csv)
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out, index=False)
    LOGGER.info("wrote %d llm quasi-sentences across %d docs -> %s (%d docs failed)",
                len(rows), len(doc_ids) - n_fail, out, n_fail)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
