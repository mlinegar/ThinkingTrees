# ThinkingTrees (OPS) Implementation Plan

Plan for building the ThinkingTrees package that implements Oracle-Preserving Summarization (OPS) on top of a fork of OmniThink. Sources: `doc/main.pdf` (OPS paper), `doc/OPS_OmniThink_Conversion_Chat.txt` (implementation discussion), and the `OmniThink/` codebase.

## Objectives and Success Criteria
- Produce a bottom-up summarization system that preserves oracle task values in expectation (sufficiency, merge consistency, idempotence) per the OPS paper.
- Turn OmniThink’s tree and DSPy scaffolding from a top-down expansion engine into a contraction/reduction engine that can ingest long documents, summarize hierarchically, and audit for OPS violation rates.
- Provide human+LLM oracle support: `f*` (human or high-grade model) and `f_hat` (surrogate DSPy program) with routing for flagged nodes.
- Deliver usable interfaces: CLI/SDK for batch pipelines and a lightweight UI to visualize trees, audits, and human feedback.
- Ship measurable quality gates: sampled violation rates below target thresholds, reproducible audits, and regression tests for core invariants.

## Scope and Non-Goals
- In scope: tree construction and contraction, OPS checks, DSPy programs for `g` and `f_hat`, audit-driven optimization loop, UI/CLI, packaging and deployment.
- Out of scope for v0: fine-tuning proprietary models, building full RAG search (keep hooks), large-scale distributed training (design for later).

## Architecture Overview
- **Data flow**: Raw doc → chunker → leaf nodes → parallel leaf summaries (`g`) → iterative layer-wise merges (same `g`) → optional stabilization passes (idempotence) → root summary.
- **Audit loop**: sample leaves (sufficiency), internal merges (two-route merge consistency), and on-range re-summarizations (idempotence). Record violation rates with confidence intervals.
- **Learning loop**: convert violations into DSPy training examples; optimize `g` (Leaf/Merge programs) and `f_hat` prompts with MIPRO/BootstrapFewShot. Use oracle-preserving metric as objective.
- **Interfaces**: Python package `thinkingtrees` with composable modules; CLI for batch runs; Streamlit (or minimal web) for inspection and human audits.
- **Storage**: tree state (raw spans at leaves, summaries everywhere), audit logs, and exemplar bank for training.

## Codebase Mapping (OmniThink → ThinkingTrees)
- Reuse: `Node`/`InformationTree` data structures, DSPy integration patterns, Streamlit UI shell, and utilities.
- Replace/extend:
  - Switch from expansion to contraction: add `layer` (height from leaves) and derived `depth` for UI; keep `.children` traversal intact.
  - New tree constructor that ingests chunked text instead of search results/mindmaps.
  - New DSPy signatures for `LeafSummarize`, `MergeSummarize`, `OracleApproximator`, `OracleComparator`.
  - Parallel layer processing (ThreadPoolExecutor) for large leaf sets with batch sizing to respect rate limits.
  - Audit module implementing OPS checks and logging.
  - Configuration for rubric/task schema (Y, d_Y) and chunking policy.
- Deprecate in core path: web search, outline/article writing agents (keep as optional tools/hints for future multimodal tasks).

## Data Structures
- `Tree`: holds nodes, `frontier` (current layer), root pointer, and max layer.
- `Node`:
  - `id`, `layer`, `children`, `raw_span` (only leaves store text; internal nodes can store byte offsets), `summary`, `metadata` (token counts, audit flags).
  - `virtual_depth` property maps `layer` to UI depth (root depth 0).
- `Rubric`: task definition (fields/types, instructions, metric); stored in conceptual pool/config file.
- `AuditRecord`: node id, check type, oracle_raw, oracle_summary, distance, pass/fail, prompt/version info.
- `Exemplar`: input, rubric, desired summary or oracle value, violation type (used for DSPy optimization).

## Core Modules
- `chunking.py`: deterministic splitter (by tokens/chars) producing ordered blocks with offsets; supports overlap for boundary checks.
- `summarizers.py`:
  - `LeafSummarize` (DSPy Signature) – g(b).
  - `MergeSummarize` (DSPy Signature) – g(g(u) ⊕ g(v)).
  - Optional `Stabilize` call for idempotence passes g(s).
- `oracle.py`:
  - `OracleApproximator` (`f_hat`) DSPy program; configurable models.
  - `OracleRouter` to decide when to escalate to `f*` (human/expensive model) based on uncertainty/flags.
  - `OracleComparator` to compute d_Y (boolean + distance).
- `tree.py`: constructors for leaf forest from chunks; `contract_layer` (serial) and `contract_layer_parallel` (thread pool, wave/batch support).
- `audit.py`: sampling policies (uniform leaves/edges/boundaries), execution of three OPS checks, CI calculation, and logging to disk.
- `optimize.py`: build exemplars from violations; run DSPy MIPRO/BootstrapFewShot with oracle-preserving metric; maintain prompt/version registry.
- `cli.py`: commands `ingest`, `summarize`, `audit`, `optimize`, `export` (JSON/CSV for summaries/audits).
- `ui/`: Streamlit pane to visualize tree, highlight failed nodes, allow human edits/labels; reuse OmniThink renderer with depth mapping.

## Pipeline (End-to-End)
1. **Ingest**: load doc(s), normalize encoding, compute token counts.
2. **Chunk**: split into contiguous blocks fitting context (configurable size/overlap); attach offsets.
3. **Build leaves**: instantiate nodes (layer=0, raw_span text, summary=None).
4. **Leaf summarization**: parallel calls to `LeafSummarize`; store summaries + token counts.
5. **Iterative merges**:
   - Pair frontier nodes; process in parallel via `MergeSummarize`.
   - Carry odd node upward unchanged or merge with pad policy.
   - Update `layer`, `summary`, and parent links; repeat until root.
6. **Stabilization**: optional additional passes g(summary) until idempotence budget exhausted or change < ε.
7. **Audit** (sampled):
   - **Sufficiency**: sample leaves; compare f*(b) vs f*(g(b)).
   - **Merge consistency**: sample internal nodes; compare f*(stored_parent) vs f*(g(u_L ⊕ u_R)) when raw fits context; if too long, rely on inductive checks.
   - **Idempotence**: sample summaries; compare f*(s) vs f*(g(s)).
   - Track per-edge failure rate p; report N_edges × p bound vs tolerance.
8. **Optimization loop**: convert failures to exemplars; run DSPy optimizer; redeploy updated programs; re-audit.
9. **Outputs**: root summary, tree JSON, audit report (rates, examples), exemplar bank.

## OPS-Specific Policies (from paper)
- Metrics: configurable d_Y (e.g., exact match for Boolean labels, weighted Hamming for tuples, numeric absolute error). Canonicalize summaries to fixed schema before oracle calls.
- Sampling budgets: fixed counts per audit run for leaves/merges/boundaries; adjust to control N × p as tree size grows.
- Boundary checks: optional overlap chunks to test cross-block evidence; include in merge sampling.
- Confidence: compute Wilson/Clopper-Pearson intervals on violation rates; gate releases on upper bound < threshold.
- Training equivalence: when downstream supervision depends only on oracle, training on audited summaries is valid; expose flag to export audited dataset for DPO/other objectives.

## Parallelization and Performance
- Layer-wise threading for leaf and merge calls; configurable `max_workers` and batch size to respect rate limits.
- Caching: memoize oracle calls on identical strings + rubric; store token counts to cap context usage.
- Streaming/logging: progress bars and structured logs for large corpora.

## Configuration
- YAML/JSON config for: model endpoints, chunk sizes/overlaps, rubric definition, sampling budgets, rate limits, stabilization budget, paths for logs/artifacts.
- Environment variables for API keys; allow local model endpoints (HF/llama.cpp) as drop-in LM providers.

## Testing and Evaluation
- Unit tests: chunking determinism, tree layer/depth mapping, serialization round-trips, comparator correctness for d_Y.
- Property tests: idempotence on canonicalizer, merge associativity on commutative tasks where applicable.
- Integration tests: small fixture document with known oracle; verify audit rates and pipeline outputs are stable.
- Performance tests: benchmark parallel contraction on synthetic large leaf sets.

## UI/UX Plan
- Reuse Streamlit shell: show tree with summaries, audit status badges, and rubric view.
- Node panel: raw span (or child summaries), current summary, oracle outputs, pass/fail reason, edit box for human correction (writes exemplar).
- Controls: run audit, re-run with new prompts, export reports.
- Depth mapping: `virtual_depth = max_layer - layer` to keep root at depth 0 for renderer.

## Deployment and Packaging
- Package as `thinkingtrees` Python module; supply `requirements.txt`/`environment.yml`.
- CLI entry points via `console_scripts`.
- Dockerfile for reproducible runs (CPU baseline; GPU optional).
- Optional Gradio/Streamlit image for demo.

## Milestones
1. **Repo scaffold (v0.1)**: create package layout, configs, chunker, tree/parallel contraction, DSPy signatures stubs, CLI skeleton.
2. **Baseline pipeline**: leaf/merge summarization end-to-end on sample doc; serialization and logging; parallel execution stable.
3. **Audit MVP**: implement OPS checks, sampling, comparator, basic report; manual oracle mode.
4. **Optimization loop**: exemplar bank, DSPy teleprompt integration, re-run audits with improved prompts.
5. **UI pass**: tree viewer with audit overlays, human feedback capture.
6. **Evaluation suite**: fixtures, unit/integration tests, performance benchmarks; CI wiring.
7. **Hardening**: config polish, caching, retry/backoff, documentation; package publish/Docker image.

## Risks and Open Questions
- Oracle cost/latency: need caching and human-in-loop thresholds; may require batching oracle calls.
- Context limits for merge consistency on high layers: rely on inductive audits plus occasional truncation/overlap checks; document limitations.
- Rate limits under parallelism: must tune batch sizes per provider; add semaphore-based dispatcher if DSPy lacks throttling.
- Task schemas: need a rubric library per task (binary event, tuples, counts, guided summaries) with canonicalization to ensure idempotence.
- UI effort vs. core: keep UI minimal to avoid blocking releases; CLI + JSON exports must stand alone.
