# Fixed-Binary Tree Diffusion Design

This note maps the new Lean surface in `FormalProofs.OPT.FixedBinaryTreeDiffusion` to the standalone runtime prototype.
The runtime is backend-agnostic: SGLang and vLLM-Omni are treated as particular engines behind the same fixed-tree interface.

## Lean Objects

- `TextCheckpoint(g, x, T, r) := ZR g x r T`
  Runtime meaning: the round-indexed text checkpoint induced by fixed-tree leaf denoise, merge denoise, and optional root refinement rounds.

- `LatentCheckpoint(encode, merge, T) := mergeFold encode merge T`
  Runtime meaning: the exact theorem-domain latent state obtained by composing leaf encoders and binary merges over the same fixed tree.

- `FixedBinaryTreeDiffusionSpec`
  Runtime meaning: one immutable fixed-tree schedule with deterministic soundness `S T = x`, a text-side summarizer `g`, and a latent theorem feature with exact mergeable structure.

## Runtime Mapping

- Backend adapter
  Implemented by `src.diffusion.backends.DiffusionBackend`.
  The fixed-tree engine depends only on the backend protocol. `SGLangDiffusionBackend` and `VLLMOmniDiffusionBackend` are concrete adapters.

- Leaf denoise
  Implemented by `FixedBinaryDiffusionTreeEngine.summarize_leaves(...)`.
  Each raw chunk is sent to the selected backend `/generate`-style surface with an explicit diffusion prompt and optional `dllm_algorithm`.

- Merge denoise
  Implemented by `FixedBinaryDiffusionTreeEngine.merge_level(...)`.
  Each binary pair is merged with `format_merge_input(left, right)` so the runtime monoid surface still matches the theorem-facing concatenation story.

- Resummary / refine round
  Implemented by `FixedBinaryDiffusionTreeEngine.refine_rounds(...)`.
  The first prototype applies refinement to the current root checkpoint only. This is enough to exercise the `R`-indexed checkpoint interface without mutating the existing AR stack.

- Latent exact state
  Implemented by `src.diffusion.markov_toy`.
  The exact Markov sketch is the first theorem-valid lane; it preserves the fixed-tree latent checkpoint exactly and reproduces the count-only insufficiency counterexample.

- Theorem-facing readout / certificate
  Lean exact transport is exposed through the new packaged theorems for factored readouts.
  Runtime-side certificates are emitted as structured JSON: per-level outputs, carried nodes, refinement rounds, and Markov toy exact-vs-count-only comparisons.

## Out Of Scope In V1

- Adaptive or stochastic trees as first-class objects.
- Gaussian / SDE diffusion foundations.
- A new joint stochastic state over `(text, latent)`.
- Wiring diffusion mode into `LLMClient`, `AsyncBatchLLMClient`, `SummarizationStrategy`, or `TreeAudit`.
- Claiming backend parity beyond the shared adapter contract. SGLang and vLLM-Omni can share the same fixed-tree engine, but their actual serving behavior still needs backend-specific validation.
