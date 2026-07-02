# LM Interface Unification Plan (shared coordination doc, 2026-06-26)

> **Two agents are working this.** This is the shared surface — record progress and
> ownership here. Companion audit: [`lm_interface_audit_2026-06-26.md`](lm_interface_audit_2026-06-26.md).
> Done already: **R1** (canonical `EmbeddingClient` protocol in `treepo.llm.embedding`),
> **R2** (retired `OpenAIChatDiffusionBackend`; dgemma `engine="openai"` routes through
> canonical `LLMClient`/CHAT_OPENAI with fleet round-robin; `build_diffusion_backend`
> refuses OpenAI engines). This plan delivers **R3 + R4 + the surface collapse**.

## Goal

Treat **text generation** as one capability independent of wire protocol. For this
implementation pass, keep `CHAT_OPENAI` as the canonical text-generation surface
to avoid enum churn. `DIFFUSION_GENERATE` is now retired as a public construction
surface and retained only as archived compatibility data. `/v1/chat/completions`
and `/generate` are *transport adapters*, not model-family concepts. One
batch-client factory underneath. Covers both `ThinkingTrees` and `~/treepo`.

## Concept model (conflated today — separate them)

- **family** — learning/scoring method: `dspy`, `fno`, `trl`, `zero_shot_llm`.
- **engine** — provider/server: `vllm`, `sglang`, `openai`, `custom_http`.
- **transport** — wire protocol: `openai_chat`, `http_generate`.
- **model profile** — launch/config: e.g. `diffusiongemma-26b-a4b-it-nvfp4`.

"Diffusion" was being used as all four at once. After this pass it is at most a *transport*
(`http_generate`) and a *model profile* — never a surface or a family.

## Target

```
InferenceEngine (Protocol, unchanged)
  ├─ CHAT_OPENAI  (v1 canonical text surface; optional later rename to TEXT_GENERATION)
  │     request: chat-message batch OR prompt-string batch + max_tokens/temp/stop/extra
  │     output:  ordered text list + usage/latency/telemetry
  │     transports: openai_chat (default) | http_generate (genuine /generate engines)
  │     ChatInferenceEngine + BatchedDSPyLM ──both build client via──> build_batch_client()
  │     DIFFUSION_GENERATE = archived compatibility enum/payload → deleted after
  │       direct modules/tests migrate
  ├─ EMBEDDING   → EmbeddingInferenceEngine → one embedding client stack
  ├─ OPERATOR / SYMBOLIC_EXACT  (unchanged)
```

## Approach: one coordinated pass via additive shims

Big-bang *outcome*, incremental-green *mechanism*. Land the canonical layer +
aliases, migrate real consumers in the same pass, then delete archived modules
once `rg "DIFFUSION_GENERATE"` shows only enum/default-url/docs/test references.

## Workstreams & ownership

| WS | Scope | Owner |
|----|-------|-------|
| A | Canonical `TEXT_GENERATION` surface + transport adapters (`core/engines.py`, `runtime/contracts.py`, `core/inference_engine.py`) | agent-1 (this) |
| B | `core/batch_client_factory.py` (new); reroute `ChatInferenceEngine._ensure_batch_client`, `config/dspy_config.create_local_engine_lm`, the 4 sites in `pipelines/batched.py` | agent-1 |
| C | Collapse tree/runtime chat-vs-diffusion branching (`tree/async_operator.py`, `tree/treepo_stack.py`, `tree/generate_prompting.py`, `core/runtime_capabilities.py`); `/generate` aliases in `diffusion/` | agent-1 |
| D | `~/treepo` method registry: `diffusion`/`dgemma`/`diffusiongemma` default to chat/DSPy scorer, not regex zero-shot (`methods/families.py`, `methods/diffusion.py`) | Codex (done 2026-06-26) |
| E | Embedding consolidation into composable layers; fold `_research` mirror copies | agent-1 |
| F | Dual-repo sync (`ThinkingTrees/src` ↔ `~/treepo/_research`); docs/naming; remove aliases | both |

**Constraint:** DSPy stays a thin `dspy.LM` adapter — do **not** route it through
`ChatInferenceEngine` (preserve fleet round-robin + GIL/tokenizer tuning;
`routing_policy="affinity_load_aware"`). It shares the *factory*, not the *engine*.

## Execution order

0. **(done)** This doc + audit R3/R4 repointed here.
1. Canonical `CHAT_OPENAI` text surface + generate transport adapter; keep only
   `DIFFUSION_GENERATE` compatibility enum/payload data until final deletion (A).
2. `build_batch_client` factory; reroute engine/DSPy/pipeline construction (B).
3. Migrate tree/runtime/treepo-registry consumers off direct diffusion/generate surface decisions (C, D).
4. Embedding consolidation (E); dual-repo sync (F).
5. Delete archived compatibility modules/imports after grep proves no active
   family path depends on them; update docs.

## Verification

1. Routing: dgemma/vLLM `/v1` → `CHAT_OPENAI`/`openai_chat`; genuine `/generate`
   works when explicitly requested as `transport="generate"`; public
   `DIFFUSION_GENERATE` construction fails with replacement guidance.
2. Adapters: `openai_chat` drops `response_format` for dgemma (`TT_DSPY_DROP_RESPONSE_FORMAT=1`);
   fleet endpoints fan across all base URLs; `/generate` payload/parse unchanged.
3. `build_batch_client`: right class for 1 vs >1 URLs, kwargs threaded.
4. Surface parity: identical response shape from chat engine vs `/generate` engine.
5. Suites: `pytest tests -k "inference or engine or embedding or operator or diffusion or batch or runtime_context or treepo_stack or dspy_batch"`; `~/treepo` registry tests.
6. Live smoke (env-gated): re-run the DSPy-on-dgemma scoring job (`n=18, pearson=0.885`,
   4-GPU fleet ports 8004-8007) through `CHAT_OPENAI`; pearson + throughput unchanged.
7. Grep gate: after step 5, `rg "DIFFUSION_GENERATE|OpenAIChatDiffusionBackend"` → only
   enum/default-url/docs/test references, then delete the compatibility enum/payload
   if it is no longer needed.

## Progress log

- 2026-06-26 — agent-2: R1, R2 landed (see audit doc).
- 2026-06-26 — agent-1: merged plan written; audit R3/R4 repointed here. Starting WS A.
- 2026-06-26 — Codex: implemented the v1 consolidation path:
  - Added `build_batch_client` and routed `ChatInferenceEngine`, `BatchedDSPyLM`,
    and batched pipeline construction through it.
  - Added `GenerateChatClient`: genuine `/generate` engines now adapt under
    `CHAT_OPENAI` and return the same `TextOutput` response shape.
  - Changed tree text generation to emit `ChatInput`; requested `surface="generate"`
    keeps generate prompt templates but resolves to `CHAT_OPENAI`.
  - Folded DLLM alias options into `SGLangDiffusionBackend`; `SGLangDiffusionClient`
    remains a deprecated alias.
  - Synced the same core/tree behavior into `~/treepo/src/treepo/_research`.
- 2026-06-26 — agent-1: VERIFY pass on Codex's A/B/C (user chose "verify + harden,
  no overlapping edits"). Confirmed `_force_sync_client` correctly diverts the
  generate transport away from the OpenAI batch client (`inference_engine.py:279`).
  Green: 91 targeted (factory/generate-transport/dspy/operator/embedding/sglang/
  tree_engine/treepo_stack_generate_first) + 43 runtime/config + 259 in
  `tests/core`+`tests/diffusion`. Added `tests/core/test_text_surface_parity.py`
  (3 tests) pinning the cross-transport response-shape invariant + transport
  routing (omni→/generate auto; default chat never hits /generate). **Pre-existing,
  out-of-scope breakage (NOT from this work):** `tests/tree/test_identifiable_zero_
  learnability.py` fails to import `_fit_leaf_theta_mlp` from
  `segmented_lda_ctreepo_simulation.py` (last touched commit f38c8100e9, LDA-sim).
- 2026-06-26 — Codex: retired the direct public diffusion surface and hid old
  direct APIs:
  - `build_inference_engine(..., surface=DIFFUSION_GENERATE)` now raises and
    points to `surface=CHAT_OPENAI, transport="generate"`.
  - Engine registry/runtime capability surfaces no longer advertise
    `DIFFUSION_GENERATE`; `/generate` default URL logic remains only for the
    transport adapter.
  - `GenerateChatClient.chat_many` preserves the old batched `/generate` backend
    call shape under the chat surface.
  - Tree-stack `diffusion_backend` / `generate_backend` specs now build
    `ChatInferenceEngine + GenerateChatClient` instead of
    `AsyncFromDiffusionBackend` / `DiffusionInferenceEngine`.
  - Deleted the dead core `DiffusionInferenceEngine` wrapper and
    `AsyncFromDiffusionBackend` operator; the deprecated
    `FixedBinaryDiffusionTreeEngine.run_fixed_tree` also routes through
    `ChatInferenceEngine + GenerateChatClient`.
  - Package `__all__` exports hide direct diffusion tree/backend APIs; archived
    lazy imports warn and remain only for compatibility.
  - Green: 46 focused ThinkingTrees LM/tree tests, 15 standalone `~/treepo`
    parity/transport tests, and a compile pass over touched main/mirror modules.
- 2026-06-26 — Codex: WS E embedding consolidation landed:
  - `treepo.llm.embedding` now owns the protocol plus canonical clients:
    `HashingEmbeddingClient`, dense hash/`HashEmbeddingClient`,
    `OpenAICompatibleEmbeddingClient` (`vllm`/`sglang`/OpenAI/custom HTTP),
    optional `TransformersEmbeddingClient`, `DiskCachedEmbeddingClient`, and
    `build_embedding_client(...)`.
  - `EmbeddingInferenceEngine` builds through `treepo.llm.build_embedding_client`;
    mock embeddings use the canonical hash client.
  - `src.training.embedding_proxy.VLLMEmbeddingClient` and the `_research` copy
    are compatibility subclasses of the canonical HTTP client.
  - `_research` local embedding protocol copies were folded to canonical imports.
  - `SGLang` now advertises the `EMBEDDING` surface for deployments exposing an
    OpenAI-compatible `/v1/embeddings` route.
  - Green: 23 focused ThinkingTrees embedding/runtime/config tests, 12 standalone
    engine-parity tests, 7 standalone method embedding/qs tests, and py_compile
    over touched modules. One broader unrelated test in
    `tests/tasks/test_manifesto_dimension_fit_existing_results.py` still fails
    on missing `expert_target_scale`.
- 2026-06-26 — Codex: WS D landed in `~/treepo`:
  - `family="dgemma"` and `family="diffusiongemma"` now default to the DSPy/chat
    scorer path. Supplying `dspy_config` is forwarded unchanged; old OpenAI-chat
    diffusion backend config can synthesize a `DSPyFamilyConfig`.
  - Scorer choice remains model-orthogonal: synthesized dgemma configs now accept
    explicit DSPy/scorer fields via `scorer_config`, `dspy_overrides`, or direct
    fields such as `dimension` / `problem_id`; all six manifesto dimension
    scorers are covered by registry tests.
  - `family="diffusion"` remains the explicit zero-shot regex/generate family.
    The dgemma aliases can still opt into that legacy path only with
    `scorer="regex_zero_shot"` / `"zero_shot"` / `"regex"`, which warns.
  - Green: standalone registry, LLM batching, and diffusion OpenAI transport tests.
- **Still open:** WS F (delete archived `DIFFUSION_GENERATE` compatibility data
  and direct backend modules after the grep gate).
