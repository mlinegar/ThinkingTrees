# DiffusionGemma vLLM Runbook

This is the local recipe for serving DiffusionGemma on this host. It adapts the
Google/vLLM recipe to the local NVFP4 checkpoint and CUDA environment.

References:

- Google developer guide: https://developers.googleblog.com/en/diffusiongemma-the-developer-guide/
- vLLM DiffusionGemma recipe: https://recipes.vllm.ai/Google/diffusiongemma-26B-A4B-it

## Local Model

Profile:

```text
diffusiongemma-26b-a4b-it-nvfp4
```

Checkpoint:

```text
/mnt/data/models/RedHatAI/diffusiongemma-26B-A4B-it-NVFP4
```

Served model name:

```text
RedHatAI/diffusiongemma-26B-A4B-it-NVFP4
```

The profile is registered in `config/settings.yaml` with `max_model_len=262144`
and DiffusionGemma-specific vLLM flags:

- `--max-num-seqs 4`
- `--attention-backend TRITON_ATTN`
- `--generation-config vllm`
- `--hf-overrides '{"diffusion_sampler":"entropy_bound","diffusion_entropy_bound":0.1}'`
- `--diffusion-config '{"canvas_length":256}'`
- `--default-chat-template-kwargs '{"enable_thinking":true}'`

## vLLM Requirement

Stock vLLM did not expose DiffusionGemma support on this host during the
2026-06-11 smoke test. The working environment is:

```text
/home/mlinegar/vllm-env
```

with vLLM installed editable from:

```text
/home/mlinegar/vllm-dgemma
```

at commit:

```text
5d304b53b
```

That branch provides `--diffusion-config` and
`vllm.model_executor.models.diffusion_gemma`.

The repo launcher already configures the needed NVFP4 CUDA runtime paths,
including the pip-installed CUDA 13 toolkit and `curand_kernel.h` include path.
If launching manually, set these before `vllm serve`:

```bash
source /home/mlinegar/vllm-env/bin/activate
export VLLM_USE_V2_MODEL_RUNNER=1
export CUDA_HOME=/home/mlinegar/vllm-env/lib/python3.12/site-packages/nvidia/cu13
export LD_LIBRARY_PATH="$CUDA_HOME/lib:$CUDA_HOME/lib64:/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH:-}"
export CPATH="/home/mlinegar/vllm-env/lib/python3.12/site-packages/nvidia/curand/include:${CPATH:-}"
```

## Launch

Preferred profile launch:

```bash
./scripts/start_vllm.sh diffusiongemma-26b-a4b-it-nvfp4 \
  --port 8004 \
  --cuda-devices 0 \
  --gpu-mem 0.85
```

Detached launch through the repo job wrapper:

```bash
./venv/bin/python scripts/long_job.py launch \
  --name diffusiongemma_vllm \
  --job-root outputs/diffusiongemma_vllm_launcher \
  --cwd /home/mlinegar/ThinkingTrees \
  -- ./scripts/start_vllm.sh diffusiongemma-26b-a4b-it-nvfp4 \
    --port 8004 \
    --cuda-devices 0 \
    --gpu-mem 0.85
```

Manual equivalent:

```bash
source /home/mlinegar/vllm-env/bin/activate
export CUDA_VISIBLE_DEVICES=0
export VLLM_USE_V2_MODEL_RUNNER=1
export CUDA_HOME=/home/mlinegar/vllm-env/lib/python3.12/site-packages/nvidia/cu13
export LD_LIBRARY_PATH="$CUDA_HOME/lib:$CUDA_HOME/lib64:/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH:-}"
export CPATH="/home/mlinegar/vllm-env/lib/python3.12/site-packages/nvidia/curand/include:${CPATH:-}"

vllm serve /mnt/data/models/RedHatAI/diffusiongemma-26B-A4B-it-NVFP4 \
  --host 0.0.0.0 \
  --port 8004 \
  --served-model-name RedHatAI/diffusiongemma-26B-A4B-it-NVFP4 \
  --trust-remote-code \
  --max-model-len 262144 \
  --max-num-seqs 4 \
  --gpu-memory-utilization 0.85 \
  --attention-backend TRITON_ATTN \
  --generation-config vllm \
  --hf-overrides '{"diffusion_sampler":"entropy_bound","diffusion_entropy_bound":0.1}' \
  --diffusion-config '{"canvas_length":256}' \
  --default-chat-template-kwargs '{"enable_thinking":true}' \
  --enable-chunked-prefill
```

First startup on Blackwell may spend several minutes compiling and autotuning
FlashInfer SM120 kernels. The successful 2026-06-11 smoke run took about 277s
for profile, KV-cache setup, and warmup.

At full 262K context and `gpu_memory_utilization=0.85`, the running server used
about 87.6 GiB on one RTX PRO 6000 Blackwell GPU. Lower `--max-model-len` if a
smaller context is enough.

## Health Checks

```bash
curl http://localhost:8004/v1/models
curl http://localhost:8004/health
nvidia-smi
```

Expected model entry:

```text
RedHatAI/diffusiongemma-26B-A4B-it-NVFP4
```

## Smoke Request

Disable thinking per request when you want a clean final answer instead of the
model's thinking prefix:

```bash
curl -sS --max-time 180 http://localhost:8004/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "RedHatAI/diffusiongemma-26B-A4B-it-NVFP4",
    "messages": [
      {
        "role": "user",
        "content": "In exactly two short sentences, explain why diffusion language models generate text differently from autoregressive language models."
      }
    ],
    "max_tokens": 160,
    "temperature": 0.2,
    "chat_template_kwargs": {"enable_thinking": false}
  }'
```

The 2026-06-11 smoke test returned:

```text
Autoregressive models generate text sequentially by predicting the next token based on all previously generated ones. In contrast, diffusion models refine the entire sequence simultaneously by iteratively denoising noise into a coherent structure.
```

## Stop

For the detached launch above:

```bash
./venv/bin/python scripts/long_job.py stop \
  --job-root outputs/diffusiongemma_vllm_launcher
```

For a foreground launch, stop the shell process directly.

## Troubleshooting

If vLLM says `--diffusion-config` is unknown or falls back to a generic
Transformers model, check that `/home/mlinegar/vllm-env` is using the editable
`/home/mlinegar/vllm-dgemma` install:

```bash
source /home/mlinegar/vllm-env/bin/activate
export CUDA_HOME=/home/mlinegar/vllm-env/lib/python3.12/site-packages/nvidia/cu13
export LD_LIBRARY_PATH="$CUDA_HOME/lib:$CUDA_HOME/lib64:/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH:-}"
python - <<'PY'
import importlib.util
import vllm

print(vllm.__version__)
print(vllm.__file__)
print(importlib.util.find_spec("vllm.model_executor.models.diffusion_gemma"))
PY
vllm serve --help=diffusion-config
```

If FlashInfer compilation fails with `curand_kernel.h: No such file or
directory`, ensure `CPATH` includes:

```text
/home/mlinegar/vllm-env/lib/python3.12/site-packages/nvidia/curand/include
```

`scripts/start_vllm.sh` does this automatically for NVFP4 model profiles.
