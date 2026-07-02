# CLAUDE.md - Claude Working Notes

## Start Here

- Read `AGENTS.md` first for environment setup, server commands, workflows, and
  current model/runtime assumptions.
- Read `docs/ctreepo_python_code_map_for_llms.md` before changing Python
  pipeline, optimizer, token-budget, C-TreePO, Semantic Forests, or manifesto
  workflow code.
- Keep `AGENTS.md` compact. The code map is the detailed handoff reference for
  repo topology, optimizer/settings behavior, backend differences, and known
  audit findings.

## Working Rules

- Use `rg`, `rg --files`, AST parsing, and direct source inspection before
  editing. Older README-style file maps may be stale.
- Preserve the dirty worktree. Do not revert or overwrite local changes unless
  explicitly asked.
- Keep C-TreePO and Semantic Forests scope distinct: C-TreePO covers theory,
  certification, f/g alternation, manifesto/Markov examples, and neural/tree
  method stacks; Semantic Forests covers systems, batching, runtime evaluation,
  feedback, and broader training infrastructure.
- Avoid committing generated outputs, logs, model/checkpoint artifacts, local
  environment files, and large data/media artifacts unless explicitly requested.
- For long-running jobs, prefer the repo launcher documented in `AGENTS.md`
  instead of ad hoc background processes.
- `FNOCountSketch.forward_doc_unified` defaults to `collect_full_trace=False`
  (training/eval fast path; see code-map "Performance" subsection under
  "Markov Publication And Tradeoff Pipelines"). Telemetry consumers must
  pass `collect_full_trace=True` explicitly. Do not introduce per-node
  `.cpu()` / `.item()` calls into the per-doc forward path - they re-serialize
  GPU work and tank throughput on long merge chains.
- **Head capacity (2026-05-03):** the unified-g default is
  `state_dim=128, hidden_dim=512`. Briefly tried bumping to
  `state_dim=2048, hidden_dim=2048, tree_merge_hidden_dim=4096` to crack
  the zero-merge root_mae ~2.14 floor on `recoverable_v5_t2048`; this was
  empirically refuted - the floor only moved to ~2.13 (noise) and several
  composition cells got worse (full100 @ leaf=256 went 1.06 -> 3.72,
  converging to a bad local min by best_epoch=10). So head capacity is NOT
  the bottleneck for that floor; the real limit is elsewhere (FNO encoder
  width, leaf-pooling info loss, or DGP irreducible noise). Don't bump
  state_dim/hidden_dim above 128/512 without a fresh experiment justifying
  it. See `feedback_head_capacity_was_not_the_bottleneck.md`.
- **Local laws on the PyTorch FNO surface (2026-05-05):** the JAX
  `learned_local_laws` result (literal-zero sufficiency recovery across all
  four `--law-architecture` cells) has not yet been replicated on
  `CleanUnifiedNO` or `fno_family.py`. The probe already implements both
  matched cells: `markov_node_witness` (↔ JAX `c2_theta`) and
  `markov_local_laws_fno` (↔ JAX `c2_self_consistency`); see
  `scripts/probe_clean_unified_no.py` lines 1805 (loss) and 2728-2742
  (CLI). The bridge experiment (`docs/markov_fno_local_law_bridge.md`)
  runs them at scale with matched bundle/seeds and a unified metric schema.
  Until that experiment lands, treat any "FNO + laws" claim as
  empirically unverified.

## Code Map Maintenance

- Do not copy the full Python code map into this file or `AGENTS.md`.
- Update `docs/ctreepo_python_code_map_for_llms.md` only after a fresh source
  inventory, AST parse sweep, and targeted searches over optimizers, token
  budgets, backend paths, TODOs, and fallback behavior.
- When changing optimizer or token-budget behavior, update the code map's
  matrix/flagged-issues sections in the same documentation pass.
