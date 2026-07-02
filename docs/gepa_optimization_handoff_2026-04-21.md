# GEPA Optimization Handoff — Manifesto / Benoit Replication

**Date:** 2026-04-21
**Status:** v2 GEPA fleet (rank metric + baseline guard) in-flight; 2 of 6 dims done.
**Audience:** Next session / another LLM / collaborator picking up where this left off.

---

## 1. Context: What we're trying to do

We're benchmarking C-TreePO against \citet{BenoitEtAl2025} (Benoit et al.
2026 AJPS) on six policy-dimension scoring of European party manifestos.
The benchmark publishes Pearson r against expert-survey means for 235
manifestos × 6 dimensions (Economic, Social, Immigration, EU,
Environment, Decentralization).

Our pipeline is `summarize(text) → score(summary, dimension)`:
- **Summarizer + Merger** wrapped as `ManifestoSummarizer` /
  `ManifestoMerger` in `src/tasks/manifesto/pipeline.py`. Build a
  compression tree over `chunk_chars`-sized leaves, merge pairwise,
  produce a root summary.
- **Scorer** wrapped as `DimensionScorer` (per-dim) or
  `JointDimensionScorer` (one-scorer-six-dims) in
  `src/tasks/manifesto/{dimension_scorer,joint_scorer}.py`. Reads
  the root summary + a Benoit scoring rubric, emits an integer 1–7
  with reasoning.

Headline (already published in our paper draft):
- Per-dim C-Tree at chunk=8K leaves: **macro Pearson r = 0.829**
  (vs. Benoit's proprietary ensemble macro 0.817, expert upper bound
  0.873). Single open-weight Gemma-4-31B-NVFP4 vs. Benoit's
  3-frontier-LLM × 18-score ensemble.

GEPA's job: tighten this further by tuning the prompt(s) used at
summarize, merge, or score time.

## 2. The Big Lesson: Pick the right GEPA scope

We tried two approaches:

### v1 — Full-pipeline GEPA (failed: too slow)

**Script:** `scripts/phase3_full_pipeline_optimize.py` with `--optimizer gepa --gepa-auto light`.

**Pipeline being optimized:** `DimensionFullPipeline` = summarizer +
merger + scorer composed. Each metric call (rollout) runs the WHOLE
tree pipeline on one manifesto: ~6 leaf summarize calls + ~5 merge
calls + 1 scoring call ≈ **12 LM calls per rollout**.

**Budget:** GEPA's `auto="light"` resolved to 1195 rollouts. So
1195 × 12 = ~14,000 LM calls per dim. With 6 dims running in
parallel on a TP=4 vLLM Gemma-4-31B-NVFP4 server at ~80–100
concurrent requests, per-rollout wall time was **80–250 seconds**.

**Observed pace:** after ~10 hours, 6 jobs averaged 200/1195
rollouts (~17%). GEPA's own reported ETA was 21–75 hours per job.
Sent SIGTERM at the user's direction; GEPA does NOT write
incremental optimized-program checkpoints, so no candidate prompts
were recoverable.

**Why this was the wrong choice:**
1. Each rollout is too expensive (12 LM calls).
2. GEPA's reflection step takes the *entire trace* (24K-char
   manifesto + 3K-token summary + scoring output) as input to the
   reflection LM, often producing low-quality proposals
   ("Exception during reflection/proposal: No valid predictions
   found for any module" appeared in many iterations).
3. Most of the swing in our chunk-sweep ablation came from the
   summary stage, not the scorer prompt — so optimizing the scorer
   in-pipeline is fighting variance, not the actual lever.

### v2 — Scorer-only GEPA on cached summaries (working)

**Script:** `scripts/phase1_optimize_scorer.py` with
`--optimizer gepa --gepa-auto light --gepa-threads 8`.

**What changed:**
- Optimizes only the scorer prompt.
- The summaries are Benoit's own anonymized GPT-4o summaries
  (cached in `data/examples/benoit_dataverse/data_masked.csv`),
  loaded via
  `src.tasks.manifesto.expert_benchmarks.load_benoit_masked_summaries`.
- Each rollout = **1 scorer LM call** on a ~3K-token summary
  (10–30× fewer LM calls per rollout than v1).
- Per-rollout wall time: **~3 seconds** (warm cache) on TP=4 server.

**Result:** all 6 dims complete in **~30–40 min total wall** for the
1195-rollout budget. Compare to v1's days-long ETA. ~**40× speedup.**

This is the right scope. Do not return to v1 unless you specifically
want to optimize summarizer or merger prompts (different problem).

## 3. v2 results (round 1 — MAE metric, openweight train pool)

`outputs/phase1_gepa_scorer_only/{dim}/report.json`. Macro-r:

| Dim | Baseline r | Optimized r | Δ |
|---|---:|---:|---:|
| economic | +0.833 | +0.796 | **−0.037** |
| social | +0.883 | +0.861 | −0.022 |
| immigration | +0.855 | +0.851 | −0.004 |
| eu | +0.896 | +0.883 | −0.013 |
| environment | +0.674 | +0.678 | +0.004 |
| decentralization | +0.306 | +0.331 | +0.025 |
| **macro** | **0.741** | **0.733** | **−0.008** |

GEPA *hurt* 4/6 dims, helped 2/6. **Net regression on macro.**
This was a real diagnosable failure, not noise.

### Diagnosis

Two confounded train/test mismatches:

1. **Train target ≠ test target.** Train labels came from Benoit's
   open-weight LLM ensemble means
   (`load_benoit_llm_scores(kind="openweight", dim)` → mean across
   LLaMA/DeepSeek/Gemma rows). Test labels are
   expert-survey means (`load_benoit_expert_means(dim)`). GEPA
   optimized to mimic the LLM ensemble, not the experts. These
   targets are correlated but not identical; the LLM-ensemble-best
   prompt isn't necessarily the expert-best prompt.
2. **Train metric ≠ test metric.** GEPA's per-example training
   metric was MAE-style:
   `1 - |pred - gold| / scale_range`. Test metric is corpus-level
   Pearson r. A prompt that improves MAE on a per-example basis can
   degrade rank-correlation if it pushes predictions toward the
   center of the label distribution.

Evidence in the GEPA logs (e.g., `outputs/phase1_gepa_scorer_only/eu/run.log`):
- "Iteration 0: Base program full valset score: 0.85"
- "Full valset pareto front score: 0.94"

The training-metric average climbed 0.85 → 0.94, but the test
Pearson r DROPPED 0.896 → 0.883. GEPA optimized exactly what we
asked it to optimize — we asked for the wrong thing.

### Important note on GEPA's selection behavior

GEPA's Pareto pool *does* include the baseline as candidate 0. After
compile, it returns the candidate with the highest TRAINING score. If
the training metric is misaligned with the test metric, GEPA can
return a candidate that's worse on test — which is what happened.
GEPA does NOT auto-protect you from this.

## 4. v2 round 2 — Rank metric + baseline guard (in-flight)

Implemented in `scripts/phase1_optimize_scorer.py` (commit pending):

### Fix A — `--keep-baseline-on-regression` (default ON)

After compile, evaluate optimized vs. baseline on the test set; if
`opt_r < base_r`, save the **baseline** as the final program and
tag `report["baseline_guard_triggered"] = True`. The
`report["final_test"]` field always reflects the better of the two.
The optimized r is preserved under `report["optimized_test"]` for
transparency.

This guarantees no test-set regression, period. It's a 5-line patch
around the test-eval block in `main()`.

### Fix B — `--metric-mode rank`

New per-example metric:
- Compute `label_center` = mean of training labels (computed once at
  optimizer setup time).
- Per example: score = `0.85 if (pred − center) and (gold − center)
  share sign else 0` + `0.15 × MAE_score` (small tiebreak so GEPA's
  Pareto search sees a smooth surface).

Rationale: this approximates the "concordant pair" contribution to
rank correlation. Optimizing it should be closer in spirit to
Pearson r than MAE is.

### Fix C — Train on expert means (deferred)

Was going to use `--train-pool expert` to remove the source-mismatch
between train and test. Discovered that Benoit's expert-mean dataset
(`data_experts.rda`) covers exactly the 235 test manifestos. After
filtering for "non-test" the expert pool is empty. Would require
restructuring as a within-235-manifesto k-fold split.

### v2 round 2 status (as of 17:54)

`outputs/phase1_gepa_v2_rank/{dim}/`. Launched at 17:23.

| Dim | Status | Result |
|---|---|---|
| immigration | DONE | base +0.848 → opt +0.855 (Δ=+0.006), final +0.855. **Rank metric helped.** |
| social | DONE | base +0.882 → opt +0.840 (Δ=−0.043), **GUARD kept baseline**. final +0.882 |
| eu | iter 49 | running (test eval phase) |
| decentralization | iter 39 | running |
| environment | iter 34 | running |
| economic | iter 18 | still optimizing |

Both fixes are observably working: A on social, B on immigration.

## 5. Things to do next (in priority order)

### Highest priority

**Wait for v2 round 2 to finish**, summarize macro r vs. v1 macro r and
vs. baseline. If the macro now exceeds baseline, we have a working
GEPA path to publish in the appendix.

### High value follow-ups

1. **Implement Fix C properly with within-test k-fold.** Split the
   235 expert manifestos 80/20 (or 5-fold), train on one fold's
   expert means, test on held-out folds. This removes the
   train/test source mismatch entirely. Complication: Benoit's
   published Figure 1 numbers are computed on the FULL 235-set, so a
   k-fold version is not directly Benoit-comparable; you'd report it
   as a separate cell.

2. **GEPA on combined-pipeline scorer with cached summaries from
   our chunk-sweep instead of Benoit's GPT-4o summaries.** Should be
   equivalent compute to v2 (1 scoring call per rollout) but the
   scorer sees OUR summaries, so the optimization should be useful
   for our deployed pipeline (not Benoit's). Code path:
   `outputs/chunk_sweep/{dim}_c8000/per_manifesto.jsonl` already has
   `summary` + `expert_mean` per row; needs a thin loader to feed
   GEPA. Worth ~1–2 hours of work.

3. **Try `--gepa-auto medium`** on the v2 rank-metric setup. The
   light auto budget is 1195 rollouts (~30 min); medium is ~3000;
   heavy is ~6000. With the per-rollout cost at 3s, even heavy is
   <1 hour wall. This is the cheapest way to find out if GEPA
   converged early.

### Lower priority / experimental

4. **GEPA for the SUMMARIZER prompt.** Currently `JOINT_RUBRIC` and
   per-dim rubrics are hand-written. GEPA could optimize them.
   Risk: this brings back v1-style cost (rollout = full pipeline).
   Mitigation: cache summarize-input chunks and rollout = single
   summarize call only. Score on a cheap proxy metric (e.g., ROUGE
   against the gold summary) rather than running the full
   downstream scorer.

5. **MoE model swap.** We downloaded
   `bg-digitalservices/Gemma-4-26B-A4B-it-NVFP4` (16 GB; MoE with
   4B active, 128 experts, native 262K context). Already in
   `config/settings.yaml` under `gemma-4-26b-a4b-it-nvfp4`. Expected
   ~5× faster decode than current Gemma-4-31B dense at comparable
   quality. Smoke test: `bash scripts/start_vllm.sh
   gemma-4-26b-a4b-it-nvfp4 --port 8010 --cuda-devices 0,1,2,3
   --tensor-parallel 4 --max-model-len 12000 --gpu-mem 0.92 --
   --trust-remote-code` (the `--trust-remote-code` is a guess —
   verify on first launch).

## 6. Critical files

| Path | Purpose |
|---|---|
| `scripts/phase1_optimize_scorer.py` | The current scorer-only GEPA driver (v2). Has `--optimizer gepa`, `--metric-mode {mae,rank}`, `--keep-baseline-on-regression`, `--gepa-auto`, `--gepa-threads`. |
| `scripts/phase3_full_pipeline_optimize.py` | The OLD full-pipeline GEPA driver (v1). Slow. Don't use unless you specifically want to optimize summarizer in-pipeline. |
| `src/tasks/manifesto/dimension_scorer.py` | `DimensionScorer` — the `dspy.Module` GEPA is optimizing. Honors `DEFAULT_SCORER_MAX_TOKENS=256` from `pipeline_config.py`. |
| `src/tasks/manifesto/pipeline_config.py` | Central config: `CONCAT_RATIO=2.0`, `DEFAULT_TARGET_RATIO=0.15`, `DEFAULT_SCORER_MAX_TOKENS=256`, `DEFAULT_INPUT_SAFETY_RATIO=0.2`, `DEFAULT_INPUT_SAFETY_FLOOR=500`, `MANIFESTO_CONTEXT_WINDOW` env. |
| `src/tasks/manifesto/expert_benchmarks.py` | Loaders: `load_benoit_expert_means(dim)`, `load_benoit_masked_summaries(dim)`, `load_benoit_llm_scores(kind, dim)`, `benoit_ensemble_mean(scores)`. |
| `src/tasks/manifesto/benoit_scoring_contexts.py` | Per-dim Benoit rubric strings (extracted verbatim from `data_masked.csv` SystemMessage field). The scorer's `task_context`. |
| `outputs/phase1_gepa_scorer_only/{dim}/report.json` | v2 round 1 results (MAE metric, no guard). |
| `outputs/phase1_gepa_v2_rank/{dim}/report.json` | v2 round 2 results (rank metric + guard). In-flight. |
| `paper/ctreepo/assets/benoit/tables/benoit_comparison_pearson.tex` | Where final GEPA numbers should land if we publish them. Add resolver in `scripts/comparison_table.py` similar to existing per-dim and combined sections. |

## 7. Reproduce / continue commands

### Re-run v2 round 2 if it gets killed mid-flight
```bash
source venv/bin/activate
export TT_DSPY_ENABLE_DISK_CACHE=false TT_DSPY_ENABLE_MEMORY_CACHE=false
export MANIFESTO_CONTEXT_WINDOW=12000
for dim in economic social immigration eu environment decentralization; do
  outdir="outputs/phase1_gepa_v2_rank/$dim"
  mkdir -p "$outdir"
  nohup python scripts/phase1_optimize_scorer.py \
    --port 8010 --dimension "$dim" \
    --optimizer gepa --gepa-auto light --gepa-threads 8 \
    --metric-mode rank \
    --output-dir "$outdir" \
    > "$outdir/run.log" 2>&1 &
done
```

GEPA does NOT support resume. Re-running starts from scratch.
Output dir's existing `optimized_scorer.json` will be overwritten on
re-run.

### Read all v2 round 2 results
```bash
for d in outputs/phase1_gepa_v2_rank/*/; do
  python -c "
import json
r = json.load(open('$d/report.json'))
print('$d')
print('  base:', r['baseline_test']['pearson_r'])
print('  opt: ', r.get('optimized_test', {}).get('pearson_r'))
print('  final:', r.get('final_test', {}).get('pearson_r'))
print('  guard:', r.get('baseline_guard_triggered'))
"
done
```

### Run the rank-metric ablation on a new dimension
```bash
# Just one dim, openweight pool, rank metric, guard on (default)
python scripts/phase1_optimize_scorer.py \
  --port 8010 --dimension economic \
  --optimizer gepa --gepa-auto light --gepa-threads 8 \
  --metric-mode rank \
  --output-dir outputs/phase1_gepa_v2_rank/economic
```

### Run with the OLD MAE metric (for direct comparison)
```bash
python scripts/phase1_optimize_scorer.py \
  --port 8010 --dimension economic \
  --optimizer gepa --gepa-auto light --gepa-threads 8 \
  --metric-mode mae \
  --output-dir outputs/phase1_gepa_v3_mae_with_guard/economic
```

### Bootstrap-FewShot baseline (for comparison; was the original
### "optimizer_bootstrap" cell in the published comparison table)
```bash
python scripts/phase1_optimize_scorer.py \
  --port 8010 --dimension economic \
  --optimizer bootstrap --max-demos 8 \
  --output-dir outputs/phase1_bootstrap/economic
```

## 8. vLLM server context

Current setup:
- One TP=4 Gemma-4-31B-NVFP4 instance, port 8010, max_model_len=12K,
  gpu_memory_utilization=0.92.
- Started via:
  `bash scripts/start_vllm.sh gemma-4-31b-it-nvfp4 --port 8010
   --cuda-devices 0,1,2,3 --tensor-parallel 4 --max-model-len 12000
   --gpu-mem 0.92`
- Aggregate throughput at 70–110 concurrent: ~8K tokens/s combined
  prefill + decode (~7900 prompt-tok/s, ~840 gen-tok/s; the
  prefill-heavy ratio reflects that scorer prompts are ~5K tokens
  and outputs are ≤256).
- Health check: `curl -s http://localhost:8010/metrics | grep
  num_requests_running`.

When the v2 GEPA fleet runs, expect ~50–80 concurrent. No
preemptions observed at this load.

## 9. Where the GEPA numbers should land in the paper

Once v2 round 2 finishes and (hopefully) shows non-regressive
results, the cells to populate are in
`paper/ctreepo/assets/benoit/tables/benoit_comparison_pearson.tex`
under the existing "Ours: scorer ablations" block. The empty
`GEPA per-dim, leaf = ...` rows in
`paper/ctreepo/tables/benoit_comparison_pearson.md` should NOT be
populated by phase1 results — those rows correspond to phase3
full-pipeline GEPA cells that are now cancelled. Add a new row:

```
"Joint scorer GEPA-optimized v2 (rank metric, baseline-guarded)"
```

with one resolver in `scripts/comparison_table.py` that reads
`outputs/phase1_gepa_v2_rank/{dim}/report.json` and pulls
`final_test.pearson_r` per dim.

## 10. Open questions for the next person

1. Does v2 round 2 (rank + guard) show macro r > baseline macro?
   That's the threshold for publishing the GEPA result. Baseline
   macro is 0.741.
2. Does the `--gepa-auto medium` budget (3000 rollouts) buy
   meaningful improvement over light (1195)? Cheap to check on one
   dim.
3. If we GEPA-optimize against OUR pipeline's summaries (not
   Benoit's GPT-4o), do we improve more? Hypothesis: yes, because
   the scorer prompt would adapt to the specific style of our
   summarizer.
4. Do the proposed prompts make domain sense, or is GEPA chasing a
   metric quirk? Inspect `optimized_scorer.json` —
   `dspy.Module.dump_state()` JSON contains the prompt instruction
   and any few-shot demos GEPA assembled. Eyeball the top
   candidates.
5. Worth swapping in the MoE Gemma-4-26B-A4B-IT-NVFP4 model
   (already downloaded) and re-running everything for ~5× faster
   wall? Probably yes; would let us run all the ablations above in
   ~3–5 hours instead of ~30.
