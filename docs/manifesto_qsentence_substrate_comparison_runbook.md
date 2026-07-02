# Manifesto q-sentence substrate comparison — runbook

Goal: directly comparable runs of three substrates on the SAME data:

| leg | substrate | family | leaf cells | server |
|---|---|---|---|---|
| A | Gemma-4-31B (medium LLM) | dspy | 1, 2, 4, 8, 16 | vLLM :8010 (1 GPU) |
| B | DiffusionGemma-26B-A4B | dspy | 1 (done first), 2, 4, 8, 16 | fleet :8004-:8007 |
| C | EmbeddingGemma-300m + FNO | fno | 1 | none (local-hf, CPU) |

**Comparability contract** (enforced/warned by the comparator):
- One bundle: `outputs/manifesto_qsentence_dspy_labeled_grid_smoke` — 8 train / 4 val / 8 test docs, seed-42 split (`split_ids.json`), cells leafq001–016 built 2026-06-11 on the same split.
- Same eval split (`test`), same metric schema (`grid_summary.json` rows), metrics on the normalized [0,1] CMP scale.
- Shared headline metric: external expert Pearson / MAE on RILE. DSPy legs also carry domain_1..7; the FNO leg is scalar-RILE until FNOFamily grows a vector head.
- Existing full-doc runs (Benoit dims, e.g. `outputs/manifesto_full_doc_gemma4_256k_*`, macro Pearson 0.84) target different labels — cite as context, do NOT put in the same table.

## Commands

Leg B1 — DiffusionGemma leaf=1 (IN FLIGHT 2026-06-11): output
`outputs/manifesto_qsentence_diffusiongemma_small`.

Leg B2 — DiffusionGemma leaves 2–16 (run after B1 frees the fleet):

```bash
./venv/bin/python scripts/long_job.py launch \
  --name manifesto_qsentence_diffusiongemma_small_leafgrid \
  --job-root outputs/manifesto_qsentence_diffusiongemma_small_leafgrid_launcher \
  --cwd /home/mlinegar/ThinkingTrees --replace-existing \
  --env TT_DSPY_DROP_RESPONSE_FORMAT=1 \
  -- ./venv/bin/python scripts/run_manifesto_qsentence_dspy_ladder.py \
  --fg-grid-dir outputs/manifesto_qsentence_dspy_labeled_grid_smoke \
  --leaf-qsentences "2,4,8,16" --max-iterations 2 --target-dimensions all \
  --dspy-optimizer gepa --dspy-budget light \
  --dspy-model "openai/RedHatAI/diffusiongemma-26B-A4B-it-NVFP4" \
  --dspy-api-base "http://localhost:8004/v1,http://localhost:8005/v1,http://localhost:8006/v1,http://localhost:8007/v1" \
  --dspy-lm-context-tokens 32768 \
  --output-dir outputs/manifesto_qsentence_diffusiongemma_small_leafgrid --verbose
```

Leg C — FNO + EmbeddingGemma (IN FLIGHT 2026-06-11): output
`outputs/manifesto_qsentence_fno_embeddinggemma_small`. CPU-only; relaunch via
the same command with `--family fno --embedding-backend local-hf
--embedding-model /mnt/data/models/google/embeddinggemma-300m --embedding-device cpu`.

Leg A — Gemma-4-31B (needs a free GPU; do AFTER B2):

```bash
# 1. Free one GPU from the DiffusionGemma fleet (gpu3 here):
./venv/bin/python scripts/long_job.py stop --job-root outputs/diffusiongemma_qsentence_worker_gpu3
# 2. Start Gemma-4 where the qsentence ladder expects it:
./scripts/start_vllm.sh gemma-4-31b-it-nvfp4 --port 8010 --cuda-devices 3
# 3. Run the same ladder (note: NO TT_DSPY_DROP_RESPONSE_FORMAT — mainline vLLM handles structured outputs):
./venv/bin/python scripts/long_job.py launch \
  --name manifesto_qsentence_gemma4_small \
  --job-root outputs/manifesto_qsentence_gemma4_small_launcher \
  --cwd /home/mlinegar/ThinkingTrees --replace-existing \
  -- ./venv/bin/python scripts/run_manifesto_qsentence_dspy_ladder.py \
  --fg-grid-dir outputs/manifesto_qsentence_dspy_labeled_grid_smoke \
  --leaf-qsentences "1,2,4,8,16" --max-iterations 2 --target-dimensions all \
  --dspy-optimizer gepa --dspy-budget light \
  --dspy-model "openai/nvidia/Gemma-4-31B-IT-NVFP4" \
  --dspy-api-base "http://localhost:8010/v1" \
  --dspy-lm-context-tokens 32768 \
  --output-dir outputs/manifesto_qsentence_gemma4_small --verbose
# 4. Restore the fleet worker afterwards:
./scripts/start_diffusiongemma_qsentence_worker.sh 3 8007
```

Caution: Gemma-4 at leaf=1 is the slow cell (one 31B call per node, ~13.5K
eval nodes). If it drags, drop leaf=1 from Leg A and compare the LLMs at
2–16 plus DiffusionGemma-only at 1 — note the asymmetry in the report.

## Final comparison

```bash
./venv/bin/python scripts/compare_manifesto_qsentence_substrates.py \
  gemma4=outputs/manifesto_qsentence_gemma4_small \
  dgemma_leaf1=outputs/manifesto_qsentence_diffusiongemma_small \
  dgemma_leafgrid=outputs/manifesto_qsentence_diffusiongemma_small_leafgrid \
  fno_embeddinggemma=outputs/manifesto_qsentence_fno_embeddinggemma_small \
  --output-dir outputs/manifesto_qsentence_substrate_comparison
```

Scale-up: repeat any leg on the full bundle
(`outputs/manifesto_qsentence_dspy_labeled_grid`, 140/30/48, leaf=1 built;
extend cells with the builder using its `split_ids.json`). Only DiffusionGemma
and FNO are realistic at full-grid leaf=1.

## Overnight queue (launched 2026-06-11 23:44 UTC)

`scripts/run_overnight_substrate_comparison.sh` (job root
`outputs/overnight_substrate_comparison_launcher`) serializes the fleet legs:
waits for the small leaf=1 run → DiffusionGemma FULL grid leaf=1 →
DiffusionGemma FULL grid leaves 2–16 → swaps GPU3 to Gemma-4 on :8010 for the
smoke-grid leaves 1–16 leg → restores the worker → runs the comparator into
`outputs/manifesto_qsentence_substrate_comparison_overnight`. A parallel CPU
job runs FNO+EmbeddingGemma on the FULL grid leaf=1
(`outputs/manifesto_qsentence_fno_embeddinggemma_full`). All DSPy legs use
`--dspy-max-train-records 2048` to bound GEPA's auto budget at 140 train docs.
Full-grid leaf cells 2–16 built 2026-06-11 on the locked split (218925 /
109776 / 55143 / 27776 nodes).

## Day-2 queue (launched 2026-06-12 07:07 UTC)

Two self-waiting jobs complete the matrix after the overnight v2 chain:
- `scripts/run_day2_substrate_comparison.sh` (`outputs/day2_substrate_comparison_launcher`):
  waits for v2 → swaps fleet to 4x Gemma-4 → **Gemma-4 FULL-grid coarse cells
  16,8,4,2** (`outputs/manifesto_qsentence_gemma4_full_coarse`; leaf=1 stays
  smoke-only for 31B) → restores fleet → day-2 comparator into
  `outputs/manifesto_qsentence_substrate_comparison_day2`.
- `scripts/run_day2_fno_leafgrid.sh` (`outputs/day2_fno_leafgrid_launcher`):
  waits for the FNO leaf=1 CPU run → **FNO FULL-grid leaves 16,8,4,2**
  (`outputs/manifesto_qsentence_fno_embeddinggemma_full_leafgrid`).
Cheap cells run first in both so results land incrementally.

Day-2b (launched 07:14): `scripts/run_day2b_dgemma_smoke_leafgrid.sh` waits for
day-2 → DiffusionGemma SMOKE-grid leaves 16,8,4,2
(`outputs/manifesto_qsentence_diffusiongemma_small_leafgrid`) so every Gemma-4
cell has a DiffusionGemma twin on the same bundle (user hypothesis: diffusion
slightly worse at matched cells). Final matched-pairs comparator →
`outputs/manifesto_qsentence_substrate_comparison_matched`.

## Phase 2 — recreate Benoit expert means per dimension (queued)

Target: the six Benoit dimensions, per-dimension expert Pearson, against the
Gemma-4 full-doc baseline (macro 0.8411 MIPRO / 0.8274 default; handoff
`docs/manifesto_full_doc_gemma4_handoff_20260429.md`; decentralization 0.579
is a known data anomaly — do not over-index). NOTE: no per-sentence gold here
— supervision is doc-level expert means only.

Leg D — DiffusionGemma full-doc global f (apples-to-apples with the 0.84 run):
`scripts/run_manifesto_full_doc_dspy_global_f.py` already trains ONE shared
`f(document, dimension, rubric)` over all six dims and reports per-dimension
Pearson. Needs long-context workers — relaunch (some of) the fleet with the
long-doc preset:
```bash
./scripts/start_diffusiongemma_qsentence_worker.sh <gpu> <port> 4 0.85 16384 262144
```
then point the runner's `--model/--api-base` at them with
`TT_DSPY_DROP_RESPONSE_FORMAT=1`, 150K-token input caps as in the handoff
command. Token-cache caveat: reuse the Gemma-4 cache only if the tokenizer
matches; otherwise re-tokenize to a new cache dir.

Leg E — embeddings (+FNO) per dimension: requires a small new builder that
emits qsentence-leaf trees for the Benoit doc set with ROOT target = expert
mean for one dimension (six bundles or a `dimension` metadata axis), expert
metadata in tree.metadata (`expert_score_native`) so the family-generic
external metrics populate. Then `--family fno` per dimension (root-weight
only; leaf/merge weights 0 since there is no local gold). Reuse
`_load_rows_by_dimension` from the full-doc runner for targets/splits.
