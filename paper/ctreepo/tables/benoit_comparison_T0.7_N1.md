# Benoit-vs-ours comparison — Pearson r (higher better)

Columns: 6 policy dimensions + Macro (unweighted mean of available cells).

## Benoit (2026 AJPS) reference

|Method|Economic|Social|Immigration|EU|Environment|Decentral.|Macro|coverage|
|---|---:|---:|---:|---:|---:|---:|---:|---:|
|Proprietary ensemble, 18 scores (Fig 1)|+0.870|+0.920|+0.890|+0.910|+0.820|+0.490|+0.817|6/6|
|Expert upper bound (Table 3)|+0.880|+0.910|+0.880|+0.950|+0.840|+0.780|+0.873|6/6|
|LLaMA-3.3-70B (Table 6)|+0.840|+0.870|+0.860|+0.860|+0.680|+0.400|+0.752|6/6|
|DeepSeek-V3 (Table 6)|+0.840|+0.870|+0.890|+0.860|+0.790|+0.450|+0.783|6/6|
|Gemma-3-27B-IT (Table 6)|+0.860|+0.860|+0.890|+0.840|+0.860|+0.450|+0.793|6/6|

## Ours: per-dim pipeline × leaf size (Gemma-4-31B-NVFP4, 1 summarizer + 1 scorer per dim)

|Method|Economic|Social|Immigration|EU|Environment|Decentral.|Macro|coverage|
|---|---:|---:|---:|---:|---:|---:|---:|---:|
|leaf = 64 K chars (≈16K tokens)|+0.915|+0.803|+0.899|+0.920|+0.864|+0.459|+0.810|6/6|
|leaf = 32 K chars (≈8K tokens)|+0.934|+0.858|+0.901|+0.914|+0.884|+0.465|+0.826|6/6|
|leaf = 24 K chars (≈6K tokens) — full test n≈215|+0.896|+0.838|+0.859|+0.899|+0.789|+0.458|+0.790|6/6|
|leaf = 16 K chars (≈4K tokens)|+0.917|+0.845|+0.883|+0.882|+0.848|+0.546|+0.820|6/6|
|leaf =  8 K chars (≈2K tokens)|+0.938|+0.861|+0.880|+0.908|+0.859|+0.545|+0.832|6/6|

## Ours: combined pipeline × leaf size (one shared summarizer w/ JOINT_RUBRIC → 6 scores)

|Method|Economic|Social|Immigration|EU|Environment|Decentral.|Macro|coverage|
|---|---:|---:|---:|---:|---:|---:|---:|---:|
|leaf = 64 K chars|+0.914|+0.866|+0.901|+0.890|+0.773|+0.458|+0.800|6/6|
|leaf = 32 K chars|+0.928|+0.855|+0.901|+0.880|+0.782|+0.454|+0.800|6/6|
|leaf = 24 K chars (full test n=229)|+0.870|+0.855|+0.876|+0.905|+0.808|+0.396|+0.785|6/6|
|leaf = 16 K chars|+0.920|+0.835|+0.936|+0.885|+0.821|+0.388|+0.798|6/6|
|leaf =  8 K chars|+0.922|+0.887|+0.892|+0.915|+0.777|+0.407|+0.800|6/6|

## Ours: tiny-leaf extensions of the chunk sweep (Gemma-4)

|Method|Economic|Social|Immigration|EU|Environment|Decentral.|Macro|coverage|
|---|---:|---:|---:|---:|---:|---:|---:|---:|

## Ours: concat-no-merge (chunks summarized independently, joined, scored — tests whether the merge step carries signal)

|Method|Economic|Social|Immigration|EU|Environment|Decentral.|Macro|coverage|
|---|---:|---:|---:|---:|---:|---:|---:|---:|

## Ours: flat baseline (no chunk, no summary; truncate text and score) — tests whether tree is needed at all

|Method|Economic|Social|Immigration|EU|Environment|Decentral.|Macro|coverage|
|---|---:|---:|---:|---:|---:|---:|---:|---:|

## Ours: full-pipeline GEPA per-dim (both g and f optimized on pooled train set)

|Method|Economic|Social|Immigration|EU|Environment|Decentral.|Macro|coverage|
|---|---:|---:|---:|---:|---:|---:|---:|---:|

## Ours: full-pipeline GEPA combined (one shared g+f across 6 dims, JOINT_RUBRIC)

|Method|Economic|Social|Immigration|EU|Environment|Decentral.|Macro|coverage|
|---|---:|---:|---:|---:|---:|---:|---:|---:|

## Ours: scorer ablations (Gemma-4, Benoit's GPT-4o summaries held fixed)

|Method|Economic|Social|Immigration|EU|Environment|Decentral.|Macro|coverage|
|---|---:|---:|---:|---:|---:|---:|---:|---:|

## Ours: exact-model replication (Benoit's Gemma-3-27B-IT BF16)

|Method|Economic|Social|Immigration|EU|Environment|Decentral.|Macro|coverage|
|---|---:|---:|---:|---:|---:|---:|---:|---:|

