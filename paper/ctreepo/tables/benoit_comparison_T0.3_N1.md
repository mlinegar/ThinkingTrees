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
|leaf = 64 K chars (≈16K tokens)|+0.912|+0.821|+0.892|+0.909|+0.875|+0.508|+0.820|6/6|
|leaf = 32 K chars (≈8K tokens)|+0.927|+0.840|+0.886|+0.912|+0.871|+0.458|+0.816|6/6|
|leaf = 24 K chars (≈6K tokens) — full test n≈215|+0.885|+0.841|+0.857|+0.900|+0.811|+0.460|+0.792|6/6|
|leaf = 16 K chars (≈4K tokens)|+0.918|+0.843|+0.898|+0.903|+0.859|+0.517|+0.823|6/6|
|leaf =  8 K chars (≈2K tokens)|+0.940|+0.867|+0.883|+0.919|+0.851|+0.583|+0.841|6/6|

## Ours: combined pipeline × leaf size (one shared summarizer w/ JOINT_RUBRIC → 6 scores)

|Method|Economic|Social|Immigration|EU|Environment|Decentral.|Macro|coverage|
|---|---:|---:|---:|---:|---:|---:|---:|---:|
|leaf = 64 K chars|+0.910|+0.857|+0.903|+0.903|+0.763|+0.457|+0.799|6/6|
|leaf = 32 K chars|+0.922|+0.855|+0.905|+0.867|+0.808|+0.400|+0.793|6/6|
|leaf = 24 K chars (full test n=229)|+0.876|+0.857|+0.886|+0.901|+0.807|+0.417|+0.791|6/6|
|leaf = 16 K chars|+0.918|+0.859|+0.927|+0.885|+0.814|+0.422|+0.804|6/6|
|leaf =  8 K chars|+0.916|+0.888|+0.902|+0.916|+0.809|+0.416|+0.808|6/6|

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

