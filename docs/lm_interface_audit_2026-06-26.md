# LM Interface Audit — how many harnesses, and is there one common API? (2026-06-26)

Scope: every way the codebase talks to a language/embedding model. Canonical
source = `ThinkingTrees/src`; `treepo/_research/*` is a path-for-path mirror (not
counted twice unless it genuinely diverges); `treepo/src/treepo/llm` +
`treepo/src/treepo/methods` are canonical-treepo.

## TL;DR

- **There IS one intended common interface:** `src/core/inference_engine.py`
  `InferenceEngine` (`execute`/`aexecute`/`asubmit`/`aexecute_many` over
  `InferenceRequest → InferenceResponse`). Active public construction now has
  **4 modes**: `CHAT_OPENAI`, `EMBEDDING`, `OPERATOR`, `SYMBOLIC_EXACT`.
  `DIFFUSION_GENERATE` remains in the enum/contracts only as archived
  compatibility data; `build_inference_engine(..., surface=DIFFUSION_GENERATE)`
  now raises and points callers to `surface=CHAT_OPENAI, transport="generate"`.
- **But adoption is partial.** Most concrete transports are *layered
  implementations* under a surface (fine), yet several **bypass** the common
  interface or **duplicate** it. Net distinct things below.

## Count

### Common interface (the target)
- **1** protocol: `InferenceEngine` + **4 active public** surface modes.
- Active surface engines: `ChatInferenceEngine`, `EmbeddingInferenceEngine`,
  `OperatorInferenceEngine`, `SymbolicExactInferenceEngine`.
- Archived/internal compatibility: `DiffusionInput` still exists in runtime
  contracts for old payload decoding; the old `DiffusionInferenceEngine` wrapper
  has been removed.

### Chat / text-generation transports — **6** (one active text path, two wire protocols)
Default wire protocol is OpenAI `/v1/chat/completions`; genuine `/generate`
servers are adapted underneath the same text surface:
1. `LLMClient` (`src/core/llm_client.py`) — base sync client (cache/retry).
2. `AsyncBatchLLMClient` (`src/core/batch_processor.py`) — async pooling over (1)'s endpoint.
3. `MultiServerBatchClient` (`src/core/batch_processor.py`) — multi-server routing over (2).
4. `BatchedDSPyLM` (`src/core/dspy_batch_client.py`) — `dspy.LM` adapter → (2)/(3).
5. `ChatInferenceEngine` (`src/core/inference_engine.py`) — the **common-interface** wrapper → (1) or (2)/(3).
6. `GenerateChatClient` (`src/core/inference_engine.py`) — `/generate` transport
   adapter under `CHAT_OPENAI`, including batched backend calls.
   → Really ONE active text-generation surface. DSPy stays a thin adapter that
   shares the batch factory rather than routing through `ChatInferenceEngine`.

### Archived diffusion `/generate` helpers — **5** (transport compatibility only)
1. `HTTPGenerateDiffusionBackend` (`src/diffusion/backends.py`) — base `/generate`.
2. `SGLangDiffusionBackend` — subclass.
3. `VLLMOmniDiffusionBackend` — subclass.
4. `SGLangDiffusionClient` (`src/diffusion/sglang_client.py`) — legacy DLLM-name wrapper over (2).
5. `InferenceDiffusionBackendAdapter` (`src/diffusion/backends.py`) — compatibility
   adapter that now calls `CHAT_OPENAI` requests internally.

### dgemma-via-OpenAI — **0 remaining bespoke classes**
The temporary `OpenAIChatDiffusionBackend` has been retired. dgemma now uses the
standard CHAT/OpenAI path. In standalone `treepo.methods`, the `dgemma` and
`diffusiongemma` family aliases now default to the DSPy/chat scorer; the
zero-shot regex/generate scorer is explicit `family="diffusion"` compatibility.

### Embedding interface — **1 protocol / 1 canonical client layer**
- Protocol definition: `treepo.llm.EmbeddingClient` is the single source of
  truth. ThinkingTrees re-exports it from the old protocol sites, and the
  `_research` mirror protocol copies were folded to imports.
- Canonical concrete clients now live in `treepo.llm.embedding`:
  `HashingEmbeddingClient`, `DenseHashEmbeddingClient` / `HashEmbeddingClient`,
  `OpenAICompatibleEmbeddingClient` (`VLLMEmbeddingClient` / `OpenAIEmbeddingClient`
  aliases), `TransformersEmbeddingClient`, `DiskCachedEmbeddingClient`, plus
  `build_embedding_client(...)`.
- Compatibility shims remain where old callers expect names:
  `src.training.embedding_proxy.VLLMEmbeddingClient` and the `_research` copy are
  thin subclasses of `OpenAICompatibleEmbeddingClient`;
  `src.ctreepo.embedding_cache.DiskCachedEmbeddingClient` re-exports the
  canonical wrapper; `SurfaceEmbeddingClient` remains the runtime-context shim
  over the `EMBEDDING` surface.

### Grand total distinct surfaces vs implementations
- **1** common interface, **4 active public modes** + archived
  `DIFFUSION_GENERATE` compatibility data.
- **~11** text-generation transport/helper classes still in tree for compatibility, but
  active family code now collapses to **1 text surface** with two transports:
  `openai_chat` and `http_generate`.
- Embeddings now use **1** protocol and **1** canonical client layer, with old
  import paths retained as compatibility wrappers.

## Conformance / fragmentation findings

1. **InferenceEngine is the right hub but still has compatibility bypasses**:
   `LLMClient` direct callers remain in older paths, DSPy shares the batch-client
   factory rather than routing through `ChatInferenceEngine`, and explicit
   `/generate` backend modules remain for archived/direct backend workflows.
   Tree-family paths now wrap `/generate` through `CHAT_OPENAI`.
2. **dgemma double-path is resolved**: it now serves through standard OpenAI chat
   (`vllm-openai:gemma`) and belongs to the **CHAT_OPENAI** surface, not a
   diffusion backend.
3. **EmbeddingClient protocol triplication is resolved**. Remaining embedding
   cleanup is mostly call-site migration away from old import names, not
   interface design.
4. **Two parallel batch paths** to the same `/v1/chat/completions`: DSPy bridge vs
   `ChatInferenceEngine`. Consolidatable but functionally fine.

## Recommendations (consolidation pass)

- **R1 (DONE 2026-06-26):** canonical `EmbeddingClient` Protocol now in
  `treepo/src/treepo/llm/embedding.py` (`@runtime_checkable`), exported from
  `treepo.llm`. The 3 ThinkingTrees defs (`embeddings/document_embedder.py`,
  `runtime/methods.py`, `tree/unified_artifacts.py`) now re-export it; `_research`
  mirrors now import it too.
- **R2 (DONE 2026-06-26):** dgemma is now a CHAT model. Retired the bespoke
  `OpenAIChatDiffusionBackend` (raw `/v1/chat/completions` HTTP); the diffusion
  family (`treepo.methods.diffusion`) routes `engine="openai"` through the
  canonical `LLMClient` (CHAT_OPENAI), round-robin + concurrent across the fleet.
  `build_diffusion_backend` now refuses OpenAI engines (chat surface owns them)
  and builds only genuine `/generate` engines. 166 methods tests green. → one
  fewer chat-transport reimplementation; dgemma == any chat LLM.
- **WS D (DONE 2026-06-26):** standalone `treepo.methods` now treats
  `family="dgemma"` / `"diffusiongemma"` as DSPy/chat scorer aliases by default.
  `family="diffusion"` remains the explicit zero-shot regex/generate baseline;
  dgemma aliases reach it only through deprecated `scorer="regex_zero_shot"`
  opt-in.
- **R3 + R4 + surface collapse → see the live plan:**
  [`docs/lm_interface_unification_plan.md`](lm_interface_unification_plan.md). That doc is
  the shared two-agent coordination surface. In brief:
  - **R3** — one `build_batch_client` factory shared by `ChatInferenceEngine` and the DSPy
    adapter (`BatchedDSPyLM` stays a thin `dspy.LM` adapter; not rewritten through the
    engine, to preserve fleet round-robin + GIL tuning).
  - **R4** — `/generate` demoted to a *transport* (`http_generate`) under canonical
    `CHAT_OPENAI`; `SGLangDiffusionClient` → one-line legacy alias;
    `DIFFUSION_GENERATE` retired as public construction surface and retained only
    as archived compatibility data until direct modules/tests can be deleted.

## Consolidation update (Codex, 2026-06-26)

- Added `src/core/batch_client_factory.py` and routed `ChatInferenceEngine`,
  `BatchedDSPyLM`, and batched pipeline construction through it. The factory
  selects `AsyncBatchLLMClient` vs `MultiServerBatchClient` and propagates model,
  API key, timeout, recovery, and routing settings.
- Added a generate transport adapter under the canonical text surface:
  `build_inference_engine(..., surface=CHAT_OPENAI, transport="generate")` wraps
  genuine `/generate` backends and returns normal `TextOutput` responses.
- Tree text generation now emits `ChatInput` requests. A requested
  `surface="generate"` preserves generate prompt templates but resolves to
  `CHAT_OPENAI`; `DIFFUSION_GENERATE` remains only for enum/default-URL and
  direct legacy backend compatibility.
- `SGLangDiffusionBackend` now owns DLLM alias options; `SGLangDiffusionClient`
  is a deprecated alias. The same behavior was mirrored into
  `~/treepo/src/treepo/_research`.

## Surface retirement update (Codex, 2026-06-26)

- `EngineSurface.DIFFUSION_GENERATE` is no longer advertised by the engine
  registry/runtime capability maps, and `build_inference_engine(...,
  surface=DIFFUSION_GENERATE)` now raises with the replacement call:
  `surface=CHAT_OPENAI, transport="generate"`.
- Explicit tree-stack `diffusion_backend` / `generate_backend` specs are wrapped
  as `ChatInferenceEngine + GenerateChatClient`; they still batch prompts through
  the underlying backend but expose normal `TextOutput` responses.
- The old core `DiffusionInferenceEngine` wrapper and `AsyncFromDiffusionBackend`
  operator were deleted. The deprecated `FixedBinaryDiffusionTreeEngine.run_fixed_tree`
  now also routes through `ChatInferenceEngine + GenerateChatClient`.
- Package-level direct diffusion APIs were hidden from `__all__`; archived lazy
  exports remain with deprecation warnings for old backend/tree-engine imports.
- The same behavior was mirrored into `~/treepo/src/treepo/_research`, while the
  standalone `treepo.methods.diffusion` path now builds `/generate` engines via
  the text surface instead of a public diffusion surface.

## Embedding consolidation update (Codex, 2026-06-26)

- Expanded `treepo.llm.embedding` from protocol-only into the canonical embedding
  client layer: deterministic hash, dense hash, OpenAI-compatible HTTP
  (`vllm`/`sglang`/OpenAI/custom), optional native `transformers`, disk cache,
  and `build_embedding_client(...)`.
- `EmbeddingInferenceEngine` now builds clients through `treepo.llm.build_embedding_client`;
  its mock path uses the canonical hash client.
- `SGLang` is now eligible for the `EMBEDDING` surface in the registry, matching
  any deployment that exposes OpenAI-compatible `/v1/embeddings`.
- Old imports are compatibility wrappers: `VLLMEmbeddingClient` subclasses the
  canonical HTTP client, and `src.ctreepo.embedding_cache` re-exports the
  canonical disk cache.
- `_research/tree/unified_artifacts.py` and
  `_research/unified_g_v1/realdoc/embedding.py` no longer define local embedding
  protocols.
