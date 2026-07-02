# Overnight Benoit comparison roundup

Pearson r per dimension. Benoit reference is from Figure 1 (proprietary 18-score ensemble), Table 3 (expert upper bound), and Table 6 (open-weight per-LLM).

## Per-dimension phase 0/1 results (per-dim summarizer + scorer)

|Dimension|Scorer-only<br>(ours, on Benoit GPT-4o)|Full pipeline<br>(per-dim)|Optimizer baseline|Optimizer optimized|Benoit Fig 1|Benoit Table 3|Benoit Table 6 open-weight|
|---|---|---|---|---|---|---|---|
|Economic|+0.832<br>[+0.79, +0.87], n=235|+0.892<br>[+0.86, +0.92], n=218|+0.830<br>[+0.79, +0.87], n=235|+0.822<br>[+0.78, +0.86], n=235|0.87|0.88|0.84 / 0.84 / 0.86|
|Social|+0.886<br>[+0.85, +0.91], n=207|+0.840<br>[+0.80, +0.88], n=217|+0.883<br>[+0.85, +0.91], n=207|+0.892<br>[+0.86, +0.92], n=209|0.92|0.91|0.87 / 0.87 / 0.86|
|Immigration|+0.858<br>[+0.81, +0.89], n=170|+0.867<br>[+0.82, +0.90], n=163|+0.852<br>[+0.80, +0.89], n=169|+0.853<br>[+0.81, +0.89], n=168|0.89|0.88|0.86 / 0.89 / 0.89|
|European Union|+0.905<br>[+0.88, +0.93], n=191|+0.896<br>[+0.86, +0.92], n=180|+0.909<br>[+0.88, +0.93], n=192|+0.912<br>[+0.88, +0.93], n=189|0.91|0.95|0.86 / 0.86 / 0.84|
|Environment|+0.668<br>[+0.58, +0.74], n=199|+0.814<br>[+0.76, +0.86], n=184|+0.676<br>[+0.59, +0.74], n=199|+0.627<br>[+0.53, +0.70], n=199|0.82|0.84|0.68 / 0.79 / 0.86|
|Decentralization|+0.311<br>[+0.19, +0.42], n=235|+0.464<br>[+0.35, +0.56], n=215|+0.307<br>[+0.19, +0.42], n=234|+0.297<br>[+0.18, +0.41], n=235|0.49|0.78|0.40 / 0.45 / 0.45|

## Phase 2 joint / combined results (shared g and f across dims)

|Dimension|Joint baseline<br>(shared scorer, unoptimized)|Joint optimized<br>(BootstrapFewShot on pooled train)|Combined pipeline<br>(one summary w/ JOINT_RUBRIC, all 6 scored)|Benoit Fig 1|
|---|---|---|---|---|
|Economic|+0.837<br>[+0.79, +0.87], n=234|+0.817<br>[+0.77, +0.86], n=235|+0.868<br>[+0.83, +0.90], n=229|0.87|
|Social|+0.881<br>[+0.85, +0.91], n=209|+0.891<br>[+0.86, +0.92], n=208|+0.850<br>[+0.81, +0.88], n=221|0.92|
|Immigration|+0.852<br>[+0.80, +0.89], n=169|+0.848<br>[+0.80, +0.89], n=168|+0.880<br>[+0.84, +0.91], n=162|0.89|
|European Union|+0.904<br>[+0.87, +0.93], n=192|+0.908<br>[+0.88, +0.93], n=191|+0.902<br>[+0.87, +0.93], n=186|0.91|
|Environment|+0.658<br>[+0.57, +0.73], n=199|+0.700<br>[+0.62, +0.76], n=199|+0.809<br>[+0.75, +0.85], n=189|0.82|
|Decentralization|+0.314<br>[+0.19, +0.42], n=235|+0.313<br>[+0.19, +0.42], n=235|+0.396<br>[+0.28, +0.51], n=208|0.49|

**Macro avg Pearson r across 6 dims:**
- joint baseline: +0.741
- joint optimized: +0.746
- combined pipeline: +0.784
