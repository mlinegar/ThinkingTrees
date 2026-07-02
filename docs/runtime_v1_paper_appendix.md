# Runtime v1 Paper Appendix Note

This appendix note records the launch-ready runtime method surface used for
LongBench v2 and related long-context experiments.

## Method Matrix

| Method | Paper role path | Final answer role |
|---|---|---|
| `full_context` | Full LongBench context is placed in the official-style prompt. | `scorer` |
| `retrieval` | `embedder` retrieves top evidence chunks from the context. | `scorer` |
| `summary_tree` | `summarizer` builds leaf and merge summaries over context chunks. | `scorer` |
| `state_tree` | `summarizer` or state-surrogate path renders compressed tree evidence. | `scorer` |
| `neural_operator` | `state_model` selects or renders evidence when configured; otherwise the method records an embedder fallback. | `scorer` |

In all v1 methods, `scorer` is the practical task scorer \(f\),
`summarizer` is \(g\), `oracle` is trusted evaluation \(f^*\), and
`state_model` is state-realization machinery rather than a direct answerer.

## Reproducibility Commands

Offline local gate:

```bash
python scripts/run_experiment.py check --suite v1 --report
```

Live endpoint gate:

```bash
python scripts/run_experiment.py check --suite v1 --live --check-endpoints --report
```

Full-stack LongBench run:

```bash
python scripts/run_runtime_eval.py init \
  --config config/runtime_eval/longbench_v2_full_stack.yaml \
  --output-dir outputs/runtime_eval \
  --experiment-id longbench_v2_full_stack

python scripts/run_runtime_eval.py run \
  --experiment-dir outputs/runtime_eval/longbench_v2_full_stack

python scripts/run_runtime_eval.py aggregate \
  --experiment-dir outputs/runtime_eval/longbench_v2_full_stack

python scripts/run_experiment.py report \
  --profile runtime_v1 \
  --output-root outputs/runtime_eval/longbench_v2_full_stack
```

## Reported Artifacts

The paper-facing method matrix is generated from:

- `predictions.jsonl`
- `calls.jsonl`
- `metrics.json`
- `results.jsonl`
- `paper_summary/runtime_v1_summary.json`
- `paper_summary/runtime_v1_summary.md`

The canonical provenance files are:

- `experiment_manifest.json`
- `experiment_status.json`
- `artifacts.json`

Prompt text and full contexts are intentionally excluded from canonical
sidecars.
