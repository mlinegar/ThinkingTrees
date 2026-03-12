# ThinkingTrees Architecture Master Plan

## Goal
Converge Engram-style conditional memory and DualPath throughput work into one production architecture with backend abstraction and a phased SGLang migration, while preserving quality on every rollout gate.

## Locked Decisions
- Migration target: `sglang` default backend by final phase, with controlled `vllm` fallback.
- Rollout policy: safe throughput wins first, then abstraction/migration, then memory-system unification.
- Quality policy: no-regression gates on both Manifesto and RULER for every phase.

## Ground Truth
- `vllm==0.12.0` is current baseline runtime.
- Engram static-memory extraction/injection hooks are already in code and opt-in.
- SGLang infrastructure exists in codebase; package/environment wiring was incomplete at plan start.
- Existing targeted baseline tests were green before convergence work (`39 passed` in targeted suite).

## Phase Plan
| Phase | Objective | Primary Deliverables | Exit Gate |
|---|---|---|---|
| 0 | Baseline + traceability | This master plan, traceability CSV, gate script + baseline artifact folder | No behavior changes; baseline commands recorded |
| 1 | Safe throughput wins | Explicit routing policies + document affinity enforcement + pending-depth telemetry | Manifesto/RULER no regression; throughput uplift target met |
| 2 | Observability + scheduling | Backend-neutral metrics collector + load-aware scheduling hooks + token-aware queue policy | Telemetry visible per server; quality unchanged |
| 3 | Backend abstraction | Backend config/CLI + server-manager protocol + dual manager adapters | Both backends smoke-pass via one interface |
| 4 | Runtime/harness migration | Runtime eval + harness on backend abstraction with fallback toggles | RULER parity on SGLang; fallback verified |
| 5 | Training migration + memory v1 | Orchestrator capability model + conditional memory L1/L2 + canonical hash adoption | Training stable on SGLang primary; quality gates pass |
| 6 | Cutover + cleanup | SGLang default in settings + sensitivity suite report + rollback documentation | 3 consecutive gate passes; cutover accepted |

## Metrics Required Per Phase
- `docs/sec`
- `tokens/sec`
- App-cache hit/miss
- Queue depth per server
- Backend transition time
- Backend failure/retry counts
- Manifesto MAE and RULER primary score

## Rollback Rule
On any gate-quality regression beyond tolerance:
1. set `inference.backend.task_backend` and `inference.backend.genrm_backend` back to `vllm`
2. rerun failed gate
3. keep artifact + incident note under `outputs/arch_reports/phase_<n>/`

## Artifacts
- Baselines: `outputs/arch_baseline_<timestamp>/`
- Phase reports: `outputs/arch_reports/phase_<n>/report.md`
- Traceability map: `docs/architecture/workstream_traceability.csv`
