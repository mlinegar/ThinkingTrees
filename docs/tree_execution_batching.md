# Tree Execution Batching Invariants

Tree workloads should batch work across leaves, merge nodes, and documents
whenever the backing runtime exposes a batch API.

## Runtime Rules

- LLM-backed tree building should use the global batching path in
  `src/core/batch_orchestrator.py`, which batches leaf prompts and ready merge
  prompts across documents instead of processing one tree at a time.
- Embedding/FNO/neural-operator tree execution should use
  `src/tree/packed_execution.py` and `forward_packed_tree_batch(...)`, which
  encodes all leaves in chunks and merges all ready parents level-by-level.
- Learned numeric sketch workloads should batch rectangular leaf grids across
  examples and leaves, then run merge schedules with batched tensor operations.
- Per-node Python loops are acceptable only as a ragged fallback or for
  non-batchable external APIs; new fast paths should keep the loop outside the
  model/API call whenever possible.

## Current Implementations

- `src/core/batch_orchestrator.py`: global LLM leaf and merge batching.
- `src/tree/embedding_tree.py`: public `forward_ctreepo_batch(...)` wrapper.
- `src/tree/packed_execution.py`: packed, level-wise neural tree execution.
- `parallel/unified_g_v1/src/unified_g_v1/sketch/learned_scalar_sketch.py`:
  batched learned-sketch leaf encoding, merge schedules, and evaluation.
