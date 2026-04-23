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

## Code Map Maintenance

- Do not copy the full Python code map into this file or `AGENTS.md`.
- Update `docs/ctreepo_python_code_map_for_llms.md` only after a fresh source
  inventory, AST parse sweep, and targeted searches over optimizers, token
  budgets, backend paths, TODOs, and fallback behavior.
- When changing optimizer or token-budget behavior, update the code map's
  matrix/flagged-issues sections in the same documentation pass.
