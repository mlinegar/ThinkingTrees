# Paper tables

All data-driven tables that appear in the C-TreePO paper are regenerated
from the raw run outputs by one driver script. Tables are version-controlled
so reviewers/collaborators can see them without re-running anything, but
they are *rebuilt* every time the driver runs so the paper's numbers
never drift from the source-of-truth `outputs/` JSONs.

## Reproducing the tables

```
bash paper/ctreepo/tables/make_tables.sh
```

This reads from:

- `outputs/overnight_benoit/scorer_only/{dim}/report.json`
- `outputs/overnight_benoit/full_pipeline/{dim}/report.json`
- `outputs/overnight_benoit/optimizer_bootstrap/{dim}/report.json`
- `outputs/phase2/{joint_optimize,joint_gepa,combined_pipeline}/report.json`
- `outputs/phase3/combined_c{chunk}/report.json`
- `outputs/chunk_sweep/{dim}_c{chunk}/report.json`
- `outputs/gemma3/scorer_only/{dim}/report.json` (when the Gemma-3 replication lands)
- `outputs/gemma3/full_pipeline/{dim}_c{chunk}/report.json`

…and writes the following to this directory:

| File | What it is |
|---|---|
| `benoit_comparison_pearson.md` | Main sectioned comparison table — Benoit references vs our per-dim pipeline × leaf size vs combined × leaf size vs scorer ablations vs Gemma-3 replication. Metric: Pearson r. |
| `benoit_comparison_pearson.tex` | Same table as booktabs LaTeX, drop-in includable from a paper section. |
| `benoit_comparison_mae.md` / `.tex` | Same comparison in MAE (where applicable; literals are em-dashed). |
| `chunk_sweep_per_dim.md` | r-vs-leaf-size table for our per-dim pipeline (6 dims × 4 leaf sizes). |
| `chunk_sweep_combined.md` | r-vs-leaf-size table for the combined pipeline (one shared summarizer + 6 scorers). |
| `overnight_roundup.md` | The broader roundup from `scripts/roundup_overnight.py` (includes every row, one place). |

## Regenerating the *data* (not just the tables)

Tables aggregate numbers from existing `outputs/` JSONs. To regenerate the
underlying numbers, re-run the experiment scripts. Minimum set to reproduce
every row currently in `benoit_comparison_pearson.md`:

```bash
# 1. Per-dim scorer-only on Benoit's summaries (all 6 dims)
for d in economic social immigration eu environment decentralization; do
  python scripts/phase0_score_benoit_summaries.py \
    --ports 8010 8011 8012 8013 --dimension $d \
    --output-dir outputs/overnight_benoit/scorer_only/$d
done

# 2. Per-dim full pipeline (summarize + merge + score), chunk_chars=24K
for d in economic social immigration eu environment decentralization; do
  python scripts/phase0_economic_pilot.py \
    --ports 8010 8011 8012 8013 --dimension $d \
    --mp-data-dir data/raw/manifesto_corpus_benoit \
    --chunk-chars 24000 --max-manifestos 1000 \
    --output-dir outputs/overnight_benoit/full_pipeline/$d
done

# 3. Per-dim chunk-size sweep (4 leaf sizes × 6 dims, n=40-50 each)
for d in economic social immigration eu environment decentralization; do
  for c in 64000 32000 16000 8000; do
    python scripts/phase0_economic_pilot.py \
      --ports 8010 8011 8012 8013 --dimension $d \
      --mp-data-dir data/raw/manifesto_corpus_benoit \
      --chunk-chars $c --max-manifestos 50 \
      --output-dir outputs/chunk_sweep/${d}_c${c}
  done
done

# 4. Combined pipeline × 4 leaf sizes
for c in 64000 32000 16000 8000; do
  python scripts/phase2_combined_pipeline.py \
    --ports 8010 8011 8012 8013 \
    --mp-data-dir data/raw/manifesto_corpus_benoit \
    --chunk-chars $c --max-manifestos 50 \
    --output-dir outputs/phase3/combined_c${c}
done

# 5. Joint scorer optimization (BFS + GEPA)
python scripts/phase2_joint_optimize.py --ports 8010 8011 8012 8013 \
  --optimizer bootstrap --output-dir outputs/phase2/joint_optimize
python scripts/phase2_joint_optimize.py --ports 8010 8011 8012 8013 \
  --optimizer gepa --gepa-auto light --output-dir outputs/phase2/joint_gepa

# 6. Combined pipeline on the full 229-manifesto test (for the main row)
python scripts/phase2_combined_pipeline.py --ports 8010 8011 8012 8013 \
  --mp-data-dir data/raw/manifesto_corpus_benoit \
  --chunk-chars 24000 --max-manifestos 1000 \
  --output-dir outputs/phase2/combined_pipeline

# 7. Regenerate tables
bash paper/ctreepo/tables/make_tables.sh
```

Prerequisites (server-side):

- 4× Gemma-4-31B-IT-NVFP4 on ports 8010–8013 (one GPU each). See
  `scripts/start_vllm.sh gemma-4-31b-it-nvfp4 --port 80NN --cuda-devices N`.
- Set `export MANIFESTO_MAX_TOKENS=8192` to cap summary length (keeps
  throughput stable — otherwise generation balloons to 5K+ tokens).
- Benoit's AJPS replication archive at `data/examples/benoit_dataverse/`
  (for expert means and anonymized GPT-4o summaries).
- MP corpus text at `data/raw/manifesto_corpus_benoit/` (pulled via
  `scripts/fetch_mp_text.py`).

Rough wall time on a 4× A100-96GB setup: per-dim runs land in ~45 min each
at leaf=24K; the full sweep overnight (all tables populate in ~6–8 h
wall-clock).
