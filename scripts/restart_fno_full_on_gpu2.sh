#!/usr/bin/env bash
# One-shot: move the FNO+EmbeddingGemma full-grid job from CPU to GPU 2.
set -euo pipefail
cd /home/mlinegar/ThinkingTrees
./venv/bin/python scripts/long_job.py stop \
  --job-root outputs/manifesto_qsentence_fno_embeddinggemma_full_launcher || true
sleep 5
./venv/bin/python scripts/long_job.py launch \
  --name manifesto_qsentence_fno_embeddinggemma_full \
  --job-root outputs/manifesto_qsentence_fno_embeddinggemma_full_launcher \
  --cwd /home/mlinegar/ThinkingTrees \
  --replace-existing \
  -- ./venv/bin/python scripts/run_manifesto_qsentence_dspy_ladder.py \
  --family fno \
  --embedding-backend local-hf \
  --embedding-model /mnt/data/models/google/embeddinggemma-300m \
  --embedding-device cuda:2 \
  --embedding-batch-size 128 \
  --fno-device cuda:2 \
  --fg-grid-dir outputs/manifesto_qsentence_dspy_labeled_grid \
  --leaf-qsentences "1" \
  --max-iterations 2 \
  --fno-epochs 40 \
  --fno-batch-size 4 \
  --fno-learning-rate 3e-3 \
  --output-dir outputs/manifesto_qsentence_fno_embeddinggemma_full \
  --verbose
