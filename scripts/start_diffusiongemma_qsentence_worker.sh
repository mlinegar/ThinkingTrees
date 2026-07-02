#!/usr/bin/env bash
# Launch one DiffusionGemma worker for the manifesto quasi-sentence DSPy ladder.
#
# Differs from the original port-8004 launch (outputs/diffusiongemma_vllm_launcher):
#   - max-model-len 32768 (qsentence nodes are tiny; frees KV cache)
#   - max-num-seqs configurable, default 64 (the 262K server needed 4)
#   - enable_thinking false (thought text lands inline in message.content and
#     contaminates DSPy field parsing; scorer outputs are compact JSON)
#
# Memory note: the diffusion sampler materializes fp32 logits for EVERY
# canvas position: num_reqs x (canvas+1=257) x vocab(~254K) x 4B, where
# num_reqs = min(max_num_seqs, max_num_batched_tokens // 257) — see
# vllm-dgemma/vllm/v1/worker/gpu/warmup.py:59. At 31-32 reqs that is
# ~7.75 GiB and OOMs next to the KV pool on a 96 GiB card; 16 seqs (~4 GiB)
# fits at gpu-mem 0.75 and is near the compute saturation point anyway.
#
# Usage: ./scripts/start_diffusiongemma_qsentence_worker.sh <gpu> <port> [max_num_seqs] [gpu_mem] [max_batched_tokens] [max_model_len]
# Stop:  ./venv/bin/python scripts/long_job.py stop \
#          --job-root outputs/diffusiongemma_qsentence_worker_gpu<gpu>
#
# Long-doc preset (Benoit full-document scoring, 150K-token inputs):
#   ./scripts/start_diffusiongemma_qsentence_worker.sh <gpu> <port> 4 0.85 16384 262144
set -euo pipefail

GPU="${1:?usage: $0 <gpu> <port> [max_num_seqs] [gpu_mem] [max_batched_tokens] [max_model_len]}"
PORT="${2:?usage: $0 <gpu> <port> [max_num_seqs] [gpu_mem] [max_batched_tokens] [max_model_len]}"
MAX_NUM_SEQS="${3:-16}"
GPU_MEM="${4:-0.75}"
MAX_BATCHED_TOKENS="${5:-8192}"
MAX_MODEL_LEN="${6:-32768}"

REPO=/home/mlinegar/ThinkingTrees

read -r -d '' SERVE_CMD <<EOF || true
source /home/mlinegar/vllm-env/bin/activate
export CUDA_VISIBLE_DEVICES=${GPU}
export VLLM_USE_V2_MODEL_RUNNER=1
export CUDA_HOME=/home/mlinegar/vllm-env/lib/python3.12/site-packages/nvidia/cu13
export LD_LIBRARY_PATH="\$CUDA_HOME/lib:\$CUDA_HOME/lib64:/lib/x86_64-linux-gnu:\${LD_LIBRARY_PATH:-}"
export CPATH="/home/mlinegar/vllm-env/lib/python3.12/site-packages/nvidia/curand/include:\${CPATH:-}"
vllm serve /mnt/data/models/RedHatAI/diffusiongemma-26B-A4B-it-NVFP4 \\
  --host 0.0.0.0 \\
  --port ${PORT} \\
  --served-model-name RedHatAI/diffusiongemma-26B-A4B-it-NVFP4 \\
  --trust-remote-code \\
  --max-model-len ${MAX_MODEL_LEN} \\
  --max-num-seqs ${MAX_NUM_SEQS} \\
  --max-num-batched-tokens ${MAX_BATCHED_TOKENS} \\
  --gpu-memory-utilization ${GPU_MEM} \\
  --attention-backend TRITON_ATTN \\
  --generation-config vllm \\
  --hf-overrides '{"diffusion_sampler":"entropy_bound","diffusion_entropy_bound":0.1}' \\
  --diffusion-config '{"canvas_length":256}' \\
  --default-chat-template-kwargs '{"enable_thinking":false}' \\
  --enable-chunked-prefill
EOF

"${REPO}/venv/bin/python" "${REPO}/scripts/long_job.py" launch \
  --name "diffusiongemma_qsentence_worker_gpu${GPU}" \
  --description "DiffusionGemma qsentence worker GPU ${GPU} port ${PORT} (32K ctx, seqs ${MAX_NUM_SEQS}, no thinking)" \
  --job-root "${REPO}/outputs/diffusiongemma_qsentence_worker_gpu${GPU}" \
  --cwd "${REPO}" \
  --replace-existing \
  -- bash -lc "${SERVE_CMD}"
