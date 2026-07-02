# Manifesto Full-Document Gemma-4 Handoff, 2026-04-29

## Current Setup

The current full-document Manifesto experiment uses raw manifesto text directly:

- Tree/input contract: `raw_manifesto_single_leaf_document`
- `g`: identity, one leaf/root per document
- `f`: one shared DSPy scorer over all six dimensions
- Model: `nvidia/Gemma-4-31B-IT-NVFP4`
- Server used: `http://localhost:8010/v1`
- Server context window observed: `max_model_len=262144`
- Input cap used: `150000` tokens per document for train/val/test
- Split: `50` train docs, `30` validation docs, `30` test docs
- Examples: `300` train, `180` validation, `180` test
- Dimensions: `economic`, `social`, `decentralization`, `environment`, `eu`, `immigration`
- Token cache: `outputs/manifesto_full_doc_gemma4_256k_20260428_225823/token_cache_gemma4_150k`

Coverage split:

```text
outputs/manifesto_full_doc_gemma4_256k_20260428_225823/coverage_split_benoit_full_docs_20260428_232048
```

Split digest:

```text
273ca80365ee3403e0bebeef27c905542339973bc7e8971735eae5f240739be8
```

## Finished Runs

### MIPRO Instruction-Only Run

Run root:

```text
outputs/manifesto_full_doc_gemma4_256k_20260428_225823/global_f_mipro_instruction_only_plainlm_doc150k_maxtok32_20260429_002437
```

Final metrics:

```text
macro_external_expert_pearson: 0.8411125759040282
economic:        0.967681857066395
social:          0.8840662799957995
decentralization:0.630114064111168
environment:     0.8737384101173143
eu:              0.7791186995821106
immigration:     0.9119561445513814
na_count:        0
prediction_rows: 180
```

Important artifacts:

```text
summary.json
predictions.jsonl
calls.jsonl
results.jsonl
run_manifest.json
experiment_status.json
launcher/job.log
```

Caveat: the original DSPy program save failed because the script attempted `save_program=True` to a `.json` path. A recovered program artifact was created afterward:

```text
program/dspy_program/
program/program_state.json
program/recovery_manifest.json
```

That recovered artifact is loadable, but it was reconstructed after the process exited from the logged best instruction. Treat the MIPRO run as reliable for outputs/metrics, not as a fully reliable reusable optimized program checkpoint.

### Baseline Default Global `f`

Run root:

```text
outputs/manifesto_full_doc_gemma4_256k_20260428_225823/global_f_default_plainlm_doc150k_maxtok32_live_20260429_051700
```

Final metrics:

```text
macro_external_expert_pearson: 0.8273584974611398
economic:        0.9631928685987109
social:          0.8855161306982141
decentralization:0.5792622001995943
environment:     0.8817912333499667
eu:              0.7491465161515662
immigration:     0.9052420357687868
na_count:        0
prediction_rows: 180
```

Important artifacts:

```text
summary.json
predictions.jsonl
predictions.live.jsonl
prediction_progress.json
calls.jsonl
results.jsonl
run_manifest.json
experiment_status.json
launcher/job.log
```

Reusable starting program:

```text
outputs/manifesto_full_doc_gemma4_256k_20260428_225823/global_f_default_plainlm_doc150k_maxtok32_live_20260429_051700/program/dspy_program
```

This directory contains `program.pkl` and `metadata.json`, and was verified with `dspy.load(...)`.

State-only artifact:

```text
outputs/manifesto_full_doc_gemma4_256k_20260428_225823/global_f_default_plainlm_doc150k_maxtok32_live_20260429_051700/program/program_state.json
```

## Code Changes Now Present

Script:

```text
scripts/run_manifesto_full_doc_dspy_global_f.py
```

Relevant behavior:

- Saves a loadable DSPy program directory at `program/dspy_program/`.
- Saves a lightweight state JSON at `program/program_state.json`.
- Streams live output during evaluation:
  - `predictions.live.jsonl`
  - `calls.jsonl`
  - `prediction_progress.json`
- Writes final sorted `predictions.jsonl` at completion.
- Supports `--initial-program-dir` to load a prior DSPy program as the starting global `f`.

Regression test:

```text
tests/tasks/test_manifesto_coverage_split_full_doc.py
```

The test now checks that full-doc global `f` writes a loadable program artifact and that `--initial-program-dir` is recorded in `run_manifest.json`.

Validated commands:

```bash
./venv/bin/python -m py_compile scripts/run_manifesto_full_doc_dspy_global_f.py
./venv/bin/pytest tests/tasks/test_manifesto_coverage_split_full_doc.py -q
```

## Starting Future Runs From The Saved Baseline Program

Use the baseline program as the reliable starting point:

```bash
--initial-program-dir \
outputs/manifesto_full_doc_gemma4_256k_20260428_225823/global_f_default_plainlm_doc150k_maxtok32_live_20260429_051700/program/dspy_program
```

Template:

```bash
./venv/bin/python scripts/run_manifesto_full_doc_dspy_global_f.py \
  --split-dir outputs/manifesto_full_doc_gemma4_256k_20260428_225823/coverage_split_benoit_full_docs_20260428_232048 \
  --output-dir outputs/manifesto_full_doc_gemma4_256k_20260428_225823/<new_run_name> \
  --base-url http://localhost:8010/v1 \
  --model nvidia/Gemma-4-31B-IT-NVFP4 \
  --train-docs 50 \
  --val-docs 30 \
  --test-docs 30 \
  --optimizer mipro \
  --dspy-budget light \
  --initial-program-dir outputs/manifesto_full_doc_gemma4_256k_20260428_225823/global_f_default_plainlm_doc150k_maxtok32_live_20260429_051700/program/dspy_program \
  --mipro-num-trials 8 \
  --mipro-minibatch-size 24 \
  --mipro-minibatch-full-eval-steps 2 \
  --max-bootstrapped-demos 0 \
  --max-labeled-demos 0 \
  --mipro-skip-bootstrap \
  --no-mipro-fewshot-aware-proposer \
  --no-mipro-data-aware-proposer \
  --mipro-view-data-batch-size 0 \
  --mipro-prompt-max-tokens 2048 \
  --mipro-prompt-temperature 0.7 \
  --dspy-num-threads 4 \
  --eval-num-threads 4 \
  --no-use-batched-lm \
  --train-max-input-tokens 150000 \
  --val-max-input-tokens 150000 \
  --test-max-input-tokens 150000 \
  --tokenizer-model /mnt/data/models/nvidia/Gemma-4-31B-IT-NVFP4 \
  --token-cache-dir outputs/manifesto_full_doc_gemma4_256k_20260428_225823/token_cache_gemma4_150k \
  --max-tokens 32 \
  --timeout-seconds 900 \
  --min-doc-chars 2000 \
  --disable-dspy-cache
```

For long runs, prefer `scripts/long_job.py launch` around that command.

## Interpretation

On the same split, MIPRO instruction-only improved macro Pearson from `0.8274` to `0.8411`, mostly from decentralization and EU. The improvement is modest but real on this test split.

The old MIPRO run is usable as a result artifact. The baseline run is the reliable saved starting program for subsequent optimization runs unless another run is launched with the patched saver and produces a stronger saved `program/dspy_program`.
