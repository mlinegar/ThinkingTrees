# Pearson r vs MP-derived logit scores (Benoit's secondary benchmark)

Compares our single-shot predictions against MP hand-coded quasi-sentence counts transformed via Lowe et al. 2011 logit. This is the alternate ground truth Benoit reports alongside expert surveys — shows what our method captures vs what the MP coders saw in the same text.

|Source|economic|social|immigration|eu|environment|decentralization|macro|
|---|---:|---:|---:|---:|---:|---:|---:|
|per-dim tree (24K)|+0.68 (n=218)|+0.49 (n=217)|+0.90 (n=74)|-0.82 (n=169)|+0.48 (n=183)|+0.50 (n=185)|+0.373|
|flat (24K trunc)|+0.77 (n=50)|+0.25 (n=48)|+0.92 (n=21)|-0.82 (n=30)|+0.47 (n=46)|+0.61 (n=26)|+0.367|
|concat (16K)|+0.76 (n=50)|+0.42 (n=49)|+0.92 (n=26)|-0.83 (n=39)|+0.43 (n=49)|+0.67 (n=38)|+0.397|
|combined (24K)|+0.67 (n=229)|+0.40 (n=221)|+0.88 (n=72)|-0.85 (n=204)|+0.38 (n=220)|+0.54 (n=182)|+0.337|

Benoit Table 7 comparison context: MP hand-coded logit scores place only 38% of coalition positions inside member-party ranges (vs 64% for their LLM ensemble) — MP and experts disagree for structural reasons, esp. on Decentralization. Our r values here show what *of* the MP signal we capture.
