# ThinkingTrees Implementation Plan (v2)

Merged plan combining the concise OPS-first view from `cdx_implementation_plan.md` with the detailed build breakdown from `cld_implementation_plan.md`.

## Objectives and Success Criteria
- Build a bottom-up, oracle-preserving summarization engine (OPS) on top of an OmniThink fork.
- Enforce C1/C2/C3 (sufficiency, idempotence, merge consistency) with probabilistic audits and measurable violation rates.
- Support dual oracles: expensive `f*` (human/strong model) and surrogate `f_hat` (DSPy program) with routing and caching.
- Deliver usable artifacts: Python package + CLI, audit reports, exemplar bank, and a minimal UI for inspection and human edits.
- Quality gates: sampled violation rate below target (ε, δ) with confidence intervals; reproducible tests for core invariants.

## Theoretical Guardrails (OPS)
- C1: f*(b) = f*(g(b)) on leaves (sufficiency).
- C2: f*(s) = f*(g(s)) on any summary (idempotence).
- C3: f*(g(s_L ⊕ s_R)) = f*(s_L ⊕ s_R) on merges (merge consistency).
- Audit sizing (from paper): m = ⌈(2/ε²) ln(2/δ)⌉ uniform samples; target per-edge failure p so N_edges × p is acceptable.

## Architecture Overview
- Data flow: raw doc → chunker → leaf nodes → parallel leaf summaries (g) → layer-wise merges (g) → optional stabilization passes → root summary.
- Audit loop: sampled C1 (leaves), C3 (merges via two-route check when context fits), C2 (re-summarize on-range summaries). Logs distances and pass/fail.
- Optimization loop: violations → exemplars → DSPy teleprompt (BootstrapFewShot/MIPRO) → redeploy programs → re-audit.
- Interfaces: package `thinkingtrees`, CLI commands (`summarize`, `audit`, `optimize`, `export`), minimal Streamlit/Gradio pane for tree + audit + human edits.
- Storage: tree JSON (raw spans at leaves, summaries everywhere), audit logs, exemplar bank, prompt/version registry.

## Data Structures (best-of)
- `TreeNode`: id, layer (height from leaves), virtual_depth (for UI), children, parent, raw_span (leaves), summary, token counts, audit status/violation type, oracle cache.
- `ReductionTree`: root, node index, traversal helpers, serialization; frontier tracking for contraction.
- `Rubric`: task definition (name/description, d_Y comparator, oracle_function, surrogate_function, examples, schema/canonicalizer).
- `AuditRecord`: node id, check type (C1/C2/C3), oracle_raw/oracle_summary, distance, pass/fail, prompt versions.
- `Exemplar`: inputs + expected oracle/summary + violation type for DSPy training.

## Core Modules (combined set)
- `chunking.py`: deterministic splitter (size/overlap, optional power-of-two balancing), stores offsets for boundary tests.
- `summarizers.py`: DSPy signatures/modules for LeafSummarize, MergeSummarize, optional Stabilize (idempotence pass).
- `oracle.py`: OracleApproximator (`f_hat`), OracleComparator (d_Y), OracleRouter (cache, escalate to `f*`).
- `tree.py`: build leaves, contract layers (serial + parallel ThreadPoolExecutor with batch sizing), carry odd nodes, track layers.
- `audit.py`: sample policies, C1/C2/C3 checks, confidence intervals, reports.
- `optimize.py`: violation→exemplar, DSPy teleprompt wrappers, metrics keyed to oracle preservation.
- `cli.py`: entrypoints for summarize/audit/optimize/export; config-driven.
- `ui/`: Streamlit pane reusing OmniThink rendering; show tree with audit badges and human edit box.

## Pipeline (end-to-end)
1. Ingest/normalize doc; compute token lengths.
2. Chunk into spans (size/overlap) with offsets; create leaf nodes (layer=0).
3. Parallel leaf summarization via LeafSummarize; store summaries + token counts.
4. Layer-wise contraction:
   - Pair frontier nodes; batch ThreadPoolExecutor calls to MergeSummarize.
   - Carry odd node upward if needed; increment layer; repeat to root.
5. Optional stabilization (idempotence passes) on selected summaries.
6. Audit (sampled):
   - C1: f*(b) vs f*(g(b)) on sampled leaves.
   - C3: f*(stored_parent) vs f*(g(raw_L ⊕ raw_R)) when raw fits context; otherwise rely on inductive checks + overlaps.
   - C2: f*(s) vs f*(g(s)) on sampled summaries.
   - Report violation counts, rates, Wilson/Clopper-Pearson CI, and N_edges × p bound.
7. Optimization: turn violations into exemplars; run DSPy optimizer; redeploy prompts/programs; re-audit.
8. Outputs: root summary, tree JSON, audit report, exemplar bank, oracle usage stats.

## Mapping to OmniThink
- Reuse: Node/InformationTree traversal pattern, DSPy integration, Streamlit shell, LM wrappers and text utils.
- Extend: contraction engine (parallel layers), layer→depth mapping for UI, new DSPy signatures, audit/optimizer modules, rubric/config handling.
- De-scope: search/mindmap, outline/article generation for core path (keep as optional future bridge).

## Parallelization and Performance
- Parallel leaf/merge layers with configurable `max_workers` and batch size to respect rate limits; optional wave processing.
- Caching of oracle calls (keyed by text + rubric); token counting to avoid context overruns.
- Progress/logging hooks for large corpora; optional semantic chunking later.

## Configuration
- YAML/JSON: model endpoints, chunk size/overlap, rubric/schema, sampling budgets, stabilization budget, rate limits, paths for logs/artifacts.
- Env vars for API keys; allow local HF/llama.cpp endpoints.

## Testing and Metrics
- Unit: chunking determinism, TreeNode/ReductionTree, rubric comparator, oracle router caching, DSPy signature I/O.
- Property: idempotence on canonicalized outputs, merge associativity for commutative tasks.
- Integration: small fixture doc with planted C1/C3/C2 violations; audit detects them; pipeline serializes/deserializes correctly.
- Metrics: oracle-preservation rate, compression ratio, audit pass rate (ε, δ), bootstrap convergence (iterations), throughput under parallelism.

## Implementation Phases (condensed from both plans)
1. **Scaffold & Data Structures (Week 1)**: package layout, TreeNode/ReductionTree/Rubric, chunker; basic tests.
2. **DSPy Summarizers & Oracle (Week 2)**: Leaf/Merge signatures, OracleApproximator/Comparator/Router; initial prompts.
3. **Tree Builder & Parallel Contraction (Week 3)**: layer-wise contraction with batching; serialization; smoke tests.
4. **Audit Engine (Week 4)**: C1/C3/C2 checks, sampling math, reports; planted-violation tests.
5. **Optimization Loop (Week 5)**: violation→exemplar, DSPy teleprompt integration, oracle-preserving metric; re-audit loop.
6. **Interfaces (Week 6)**: CLI commands; minimal UI for tree + audits + human edits; JSON/CSV export.
7. **Testing/Benchmarks (Week 7)**: coverage push, perf/parallel benchmarks, audit CI validation.
8. **Hardening & Docs (Week 8)**: configs, retries/backoff, caching polish, docs/notebooks, Dockerfile.

## Risks and Mitigations
- Oracle cost/latency → caching, surrogate routing, batching; human escalation thresholds.
- Context blow-up for high-level C3 checks → rely on inductive audits + overlaps; document limitations.
- Rate limits under parallelism → batch size/semaphore control; provider-specific tunables.
- Rubric schema drift → canonicalization and fixed output schemas to stabilize idempotence; rubric library for common tasks.

## Deliverables
- `thinkingtrees` package with summarization, audit, and optimization modules.
- CLI + minimal UI; tree/audit JSON exports.
- Example rubrics and quickstart scripts; exemplar bank format.
- Test suite with planted-violation fixtures; performance benchmark scripts.
