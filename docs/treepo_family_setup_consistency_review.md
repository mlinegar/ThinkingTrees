# treepo Family Setup Consistency Review (2026-06-26)

Review of how each major model family is constructed + how it talks to its
backend, via `treepo.methods`. Goal: one consistent setup; in particular,
**dgemma/diffusion should be served + consumed like any other vLLM/LLM, not via a
separate stack.**

## Per-family transport (how each reaches its backend)

| family | factory | backend / transport | server need |
|---|---|---|---|
| `oracle` | `methods.dispatch._method_oracle` | pure-Python oracle scoring | none (offline) |
| `learnable_constant` | `methods.families._make_learnable_constant` | torch scalar fit | none (offline) |
| `fno` | `methods.fno.build_fno_family` | **embedding client** (`embed_texts`) → vectors → FNO | embedding server (or offline hashing client) |
| `dspy` | `methods.dspy.build_dspy_family` | **OpenAI `/v1/chat/completions`** via DSPy (`lm_transport=batch/litellm`, `lm_config={model, api_base, api_key}`) | vLLM OpenAI server (:8000) |
| `trl` | `methods.trl.build_trl_family` | local HF model + GPU (SFT/GRPO subprocess) | GPU + HF model |
| **`diffusion`/`dgemma`** | `methods.diffusion.build_diffusion_family` | **bespoke `/generate`** (`treepo.llm.diffusion.build_diffusion_backend` → `_research.diffusion.backends`: HTTP/SGLang/`vllm_omni`) | a `/generate`-style server |

## The inconsistency

`dspy` consumes the model through the **standard OpenAI-compatible API**
(`/v1/chat/completions`). `diffusion`/`dgemma` instead uses a **separate
`/generate` transport** (`engine="vllm_omni"` by default). That is the "dgemma
has its own setup" problem:

- It needs a different server surface (`/generate`) than every other LLM family.
- The only installed vLLM (`/home/mlinegar/vllm-dgemma`, editable, the dgemma
  fork) already ships the **standard OpenAI `api_server`**
  (`vllm/entrypoints/openai/api_server.py`) and `start_vllm.sh` launches *that*.
  So a dgemma server brought up the normal way exposes `/v1/chat/completions`,
  **not** `/generate` — and the diffusion family's `/generate` backend can't talk
  to it.
- dgemma is known to work over the OpenAI API already: the only quirk is
  `response_format` crashing its engine (`TT_DSPY_DROP_RESPONSE_FORMAT=1`), which
  is an OpenAI-API-path detail — direct evidence dgemma is "just an OpenAI LLM."

**Conclusion (matches the user's call):** diffusion/dgemma should consume the
model via the standard OpenAI-compatible transport, same as `dspy`. The bespoke
`/generate` backend stays available for genuine `/generate`-only engines but is
**not** the dgemma path.

## Fix landed this pass

Added an **`engine="openai"`** option to the diffusion backend
(`treepo.llm.diffusion`) — a public, import-light OpenAI-compatible chat backend
that POSTs to `/v1/chat/completions` (reusing `treepo.llm.openai_compatible`),
never sends `response_format`, and returns `.texts` like the `/generate`
backends. `DiffusionTextFamily` is unchanged (it already calls
`backend.generate(prompts)`), so:

```python
treepo.methods.run("fit", {
    "family": "diffusion",
    "eval_data": trees,
    "backend_config": {"diffusion_config": {
        "backend": {"engine": "openai",
                    "base_url": "http://localhost:8004/v1",
                    "model": "google/diffusiongemma-..."}}},
})
```

now drives dgemma through the **same vLLM OpenAI server** as `dspy`. No
`/generate` server, no separate stack.

## Serving root cause (separate from the code fix)

The only vLLM is the dgemma fork; `start_vllm.sh gemma-4-31b-it` failed at
`import vllm._C` with `ImportError: libcudart.so.13`. Diagnosis:

- No system `nvcc`; cu13 ships `bin/nvcc` + `libcudart.so.13` under
  `vllm-env/.../nvidia/cu13/`.
- `start_vllm.sh`'s cu13 `LD_LIBRARY_PATH` block depends on a `site_packages`
  shell-derivation; in the long_job/non-interactive environment it did not put
  `cu13/lib` on `LD_LIBRARY_PATH`, so the precompiled extension couldn't load.
- **Verified fix:** `LD_LIBRARY_PATH=<vllm-env>/.../nvidia/cu13/lib python -c
  "import vllm._C"` → OK. The overnight runner now exports this before launching
  the server (non-invasive; does not modify `start_vllm.sh`).

This is environmental (CUDA lib path), not a model-code issue. A durable fix is to
make `start_vllm.sh` always prepend the cu13 lib dir (independent of `nvcc`
detection); deferred to the server-infra owner.

## Canonical dgemma serving (vLLM recipe — confirms the OpenAI path)

Per the vLLM Gemma4 recipe
(`docs.vllm.ai/projects/recipes/.../Google/Gemma4.html`, Block-Diffusion section),
dgemma (`google/diffusiongemma-26B-A4B-it`) is served by the **standard
`vllm/vllm-openai:gemma` image over the standard OpenAI API** — no `/generate`,
no separate stack:

```bash
vllm serve google/diffusiongemma-26B-A4B-it \
    --max-model-len 262144 --max-num-seqs 4 \
    --gpu-memory-utilization 0.85 --generation-config vllm \
    --enable-chunked-prefill --host 0.0.0.0 --port 8000
```

Required, dgemma-specific bits are all **serve flags**, not a different transport:
- `--generation-config vllm` (override the checkpoint's 256-token cap),
- `--max-num-seqs ≤ 4` (diffusion state buffers; OOM otherwise),
- `--enable-chunked-prefill`.

`config/settings.yaml` already encodes these for the
`diffusiongemma-26b-a4b-it-nvfp4` profile (`--max-num-seqs 4`,
`--generation-config vllm`, `--diffusion-config`, `--hf-overrides
diffusion_sampler`). **So dgemma serving needs no separate setup — it's a normal
vLLM OpenAI server with a few flags, consumed via the new `engine="openai"`
diffusion backend (or directly by the dspy family).** This is the single
consistent path the user asked for.

## Recommended follow-ups (not yet done)

- Consider registering `dgemma`/`diffusiongemma` to default `engine="openai"` so
  they are OpenAI-transport by default (one-liner in `build_diffusion_family`).
- Longer term: fold the zero-shot diffusion scorer and the dspy scorer behind one
  "LLM family" that differs only by optimizer (none vs GEPA), since both now share
  the OpenAI transport.
