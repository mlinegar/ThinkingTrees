# Unsloth Qwen3.5 + GRPO Notes (2026-03-04)

## Sources
- https://unsloth.ai/docs/models/qwen3.5/fine-tune
- https://unsloth.ai/docs/reinforcement-learning-rl-guide
- https://huggingface.co/docs/trl/grpo_trainer

## Key points pulled from docs
- Unsloth Qwen3.5 fine-tuning path uses `FastLanguageModel.from_pretrained(...)`, with common defaults like `load_in_4bit=True` and LoRA via `FastLanguageModel.get_peft_model(...)`.
- Unsloth guidance notes vLLM integration for generation speedups, but RL optimization remains trainer-side (vLLM is not the optimizer).
- TRL GRPO is online RL: reward functions score sampled completions; additional dataset columns can be passed into reward functions via kwargs.

## Implication for ThinkingTrees
- A cleaner replacement for tournament-style loops is:
  1. Keep teacher-first data generation and local-law scoring.
  2. Train prompt/program policy with proxy+GEPA (fast, cheap signal shaping).
  3. Add optional GRPO adapter training where rewards come from local-law metrics (C1/C2/C3), not GenRM.
- For synthetic LawStress, rewards can be computed from known oracle fields directly (no external judge call).
- For real-anchor traces, rewards should call the large teacher scorer and apply split-filtered test gating.

## Current code gap to close
- `--generator-method grpo` exists, but Phase 3.25 currently creates a GRPO trainer without reward function wiring in the default path.
- We need a scorer-backed (large-model) reward function factory and explicit plumbing so GRPO works without GenRM.
