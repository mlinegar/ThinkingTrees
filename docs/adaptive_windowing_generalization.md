# Adaptive Windowing: Generalization Plan

This note records how adaptive embedding windowing should generalize beyond text
without changing the core policy.

## Core Idea

Treat every source as a 1D axis:

- text: character index
- video: time (ms/sec)
- PDF: page index
- social feeds: item index (comment/post order)

Then reuse the same coarse-to-fine refinement policy:

1. score coarse windows with a cheap model (embedding proxy),
2. refine uncertain/high-gradient windows,
3. keep stable/low-variation windows coarse,
4. merge adjacent low-drift windows.

## Cold Start: Don’t Refine On Noise

Adaptive windowing assumes the “cheap model” is a *meaningful* proxy for a
trusted target (truth labels or a trusted oracle score). If you enable adaptive
windowing before that proxy is trained/calibrated, span scores behave like noise
and boundary refinement becomes arbitrary.

See `docs/pipeline_ordering.md` for the intended bootstrap order.

## Current Code Contract

Implemented in `src/preprocessing/adaptive_windows.py`:

- `AxisWindow(start, end, unit)`
- `uniform_axis_windows(...)`
- `adaptive_refine_windows(...)`

Adapter contract implemented in `src/preprocessing/window_adapters.py`:

- `AxisWindowAdapter` protocol
- `TextCharWindowAdapter` (current text adapter)
- `TextPageWindowAdapter` (PDF/page-aligned text)
- `SequenceItemWindowAdapter` (social/feed item streams)
- `TimeSegmentWindowAdapter` (timeline segments for video/audio)
- `build_window_adapter(...)` (named adapter factory)
- `build_adaptive_windows_for_sample(...)` (adapter-driven entry point)

Current training path (`src/training/run_pipeline.py`) uses
`TextCharWindowAdapter` with `unit="char"` and maps each window to
`text[start:end]`.

Low-drift merge utility is also implemented in
`src/preprocessing/adaptive_windows.py`:

- `merge_adjacent_windows_by_embedding_drift(...)`

## Why This Is Modality-Agnostic

The adaptive policy only needs:

- an axis interval (`AxisWindow`),
- a `score_windows(windows) -> score[]` callback.

It does **not** assume sentence boundaries, pages, or any text-specific parser.

## Recommended Adapters (Future)

Define one adapter per modality with this interface:

- `total_extent(sample) -> int`
- `materialize(sample, window) -> embedding_input`
- `score_windows(sample, windows) -> relevance_scores`

Examples:

- Video adapter:
  - axis unit: `"ms"`
  - window materialization: transcript snippets, frame captions, or fused multimodal embeddings
- PDF adapter:
  - axis unit: `"page"`
  - window materialization: per-page text or OCR output
- Feed adapter:
  - axis unit: `"item"`
  - window materialization: concatenated item texts for contiguous ranges

## Cost Tradeoff Guidance

Practical cost has two terms:

- request overhead (HTTP/scheduling/batching),
- model compute (sequence-length dependent, often superlinear for longer windows).

So best practice is usually:

- avoid extremely tiny windows (too much overhead),
- avoid always-max windows (compute blow-up + weak localization),
- use coarse-to-fine adaptive windows (current default behavior).

## What To Keep Fixed For Honest Evaluation

Continue the same honesty discipline:

- use boundary split for adaptation,
- use evaluation split for reporting,
- log boundary/eval and cross-fit gaps.

This remains valid regardless of axis unit.
