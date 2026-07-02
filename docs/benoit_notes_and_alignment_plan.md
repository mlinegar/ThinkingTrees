# Benoit et al. (2026 AJPS) — Notes and C-TreePO Alignment Plan

Companion to [docs/benoit_llm_manifesto_scaling.pdf](benoit_llm_manifesto_scaling.pdf).
Purpose: (a) extensive notes on the paper so we do not have to re-read the PDF
to reason about comparability; (b) a mapping of Benoit's pipeline onto current
C-TreePO infrastructure; (c) a three-phase experiment blueprint for a future
run that produces numbers comparable to their Table 6.

Scope decisions (this session):

- Deliverable: notes + blueprint only, no model runs.
- Dimensions: all six Benoit dimensions on the 7-point scale.
- LLM panel: open-weight trio (DeepSeek-V3, LLaMA-3.3-70B, Gemma-3-27B) — the
  same models used in Benoit's replication column of Table 6.
- Venue: this file.

---

## 1. Paper notes

**Citation.** Benoit, Kenneth, Scott De Marchi, Conor Laver, Michael Laver,
and Jinshuai Ma. 2026. "Using large language models to analyze political texts
through natural language understanding." *American Journal of Political Science*:
1–17. DOI 10.1111/ajps.70050. Dataverse for replication materials:
https://doi.org/10.7910/DVN/XY1FFE. Received 20 Feb 2025; accepted 1 Nov 2025.

**One-line thesis.** Ensembling three commercial LLMs to first *summarize* a
party manifesto into a 300–400 word English description *per policy
dimension* and then *scale* that summary on a 7-point scale produces
party-position estimates whose correlations with expert-survey means approach
the expert–expert upper bound on five of six dimensions, and whose coalition-
agreement estimates align with spatial bargaining theory better than the hand-
coded Manifesto Project / Klüver baseline.

### 1.1 Why this paper matters for C-TreePO

The paper does the same two things C-TreePO is trying to do — (i) summarize a
long political document, (ii) produce a scalar policy position — but on a
flat "summarize once per dimension, scale the summary" pipeline rather than a
tree. It is the current empirical bar for LLM-based political-text
measurement. Four useful Benoit positions for our story:

1. **Summaries as scientific artifacts.** "We see generating intermediate text
   summaries as an integral part of our method" (p. 6). This exact framing is
   what the C-TreePO audit formalizes: summaries are the object being
   certified, not a throwaway preprocessing step. Citing Benoit lets us
   plant the flag that the science needs summary-level guarantees.
2. **Upper bound framing.** Section 3.2 (Table 3) argues that expert–expert
   split-sample correlations bound what any measurement method can achieve.
   C-TreePO should adopt this framing — the interesting delta isn't "can we
   hit .95 on Economic" but "how close to the .88 Economic ceiling can we get
   with an auditable pipeline."
3. **Needle-in-the-haystack problem.** "Some manifestos exceed 150 pages...
   this is the 'needle in the haystack' problem" (p. 5, citing Chang et al.
   2023, Hosseini et al. 2024). Benoit's solution is per-dimension
   summarization. The natural C-TreePO hook is: a tree with local audits
   handles this with a finite-sample certificate rather than a single prompt.
4. **Informative NAs.** They explicitly keep NAs as signal, not error (p. 11:
   "knowing what is *not* in a text is as important as knowing what is in it").
   C-TreePO's node-sampled audit protocol can surface this even better: a
   local-law violation at a particular node is a location-specific NA.

### 1.2 Data

**Party manifestos.** 235 manifestos drawn from the Manifesto Project (Merz
et al. 2016; Volkens et al. 2013; Klingemann et al. 2006; Budge et al. 1987,
2001). Filtered to election years for which an expert survey (CHES or
Benoit–Laver) was also fielded in the same country. 21 languages (detailed
manifesto-by-manifesto list in Supporting Information Appendix A1, p. 3 of
SI). Input format: plain text downloaded from MP and cleaned. Appears to map
onto our MPDS 2025a extract at
[outside_data/Manifesto_Project/MPDataset_MPDS2025a.csv](../outside_data/Manifesto_Project/MPDataset_MPDS2025a.csv)
and document-level text at
[outside_data/Manifesto_Project/documents_MPDataset_MPDS2025a.csv](../outside_data/Manifesto_Project/documents_MPDataset_MPDS2025a.csv).

**Coalition agreements.** 23 coalition agreements from Klüver et al. (2023,
"Coalition Agreements as Control Devices") — the Eastern and Western European
cabinet-formation agreements 1945–2015. Used only for *predictive validity*,
not concurrent validity (no expert survey exists for coalitions). Limited to
agreements whose start/end dates fall inside the years covered by the
manifesto corpus.

**Expert-survey benchmarks.** Two series, pooled as ensemble means:
- Chapel Hill Expert Survey (CHES) trend file, 1999–2019 (Jolly et al. 2022;
  Bakker et al. 2015 for the 1999–2010 file).
- Laver & Hunt 1992 extended by Benoit & Laver 2006 ("Party Policy in Modern
  Democracies"). Benoit–Laver original scales are 1–20; CHES is typically
  0–10 or 1–11. Benoit et al. rescale everything to a common 1–7 axis for
  their plots (see Supporting Information Appendix A2, p. 4).
- Median experts per party: 16 (Benoit–Laver), 12 (CHES).

**MP-derived positions.** Logit-transformed counts from MP hand-coded
quasi-sentences, computed per Lowe et al. 2011. Used as a *secondary*
benchmark because MP per-dimension counts are known to be biased on valence
dimensions (esp. Decentralization, per301/per302 imbalance — their Figure 2).

### 1.3 Six policy dimensions (Table 2, p. 4)

| Dimension | Scale | Low anchor (0 or 1) | High anchor (10 or 7) |
|---|---|---|---|
| Economic | 0–10 | Strongly favors improving public services | Strongly favors reducing taxes |
| Social | 0–10 | Strongly supports liberal policies | Strongly opposes liberal policies |
| Immigration | 0–10 | Strongly opposes tough policy | Strongly favors tough policy |
| EU | 1–7 | Strongly opposed | Strongly in favor |
| Environment | 0–10 | Environmental protection even at cost of growth | Economic growth even at cost of environment |
| Decentralization | 0–10 | Strongly favors political decentralization | Strongly opposes political decentralization |

Note EU is the only native-7-point dimension; the rest are 0–10 from expert
surveys but **rescaled to 1–7** before LLM scoring so all outputs share the
same axis.

### 1.4 Workflow (their Table 1)

1. **Input texts + validation data**
   - MP manifestos → plain text.
   - CHES + Benoit–Laver expert means per (country, year, dimension, party).
   - MP-based logit scores per (party, dimension) for a secondary comparison.
2. **Authorize three commercial LLMs**: GPT-4o (OpenAI), Claude 3.5 Sonnet
   (Anthropic), Gemini 1.5 Pro (Google) — paid API tiers (subscription is
   insufficient for their volume).
3. **Summarize.** For each (manifesto, dimension, LLM) combination, prompt
   the LLM to produce a 300–400 word *English* summary specifically of what
   the manifesto says about that dimension. Same prompt across LLMs
   (Supporting Information Appendix A3, p. 5, for exact text). They rejected
   a translate-first-then-summarize variant as harming correlations
   (Supporting Information Table B7, p. 17).
4. **Scale.** Prompt each LLM to locate each summary on a 7-point scale.
   Two prompt conditions:
   - *Zero-shot*: only the rubric and the summary.
   - *Few-shot*: three additional benchmark-scored summaries (left/center/right)
     from other manifestos on the same dimension.
   Prompt wording fixed to CHES-style "You are an expert political scientist
   with a PhD in political science. Think carefully about your answer."
   after a 9-variant matrix experiment showed prompt wording is a second-order
   effect in this task (their Supporting Information Table B1, p. 12).
5. **Validate.**
   - Concurrent: Pearson correlation of LLM ensemble mean vs. expert ensemble
     mean per dimension.
   - Reliability: ICC and Krippendorff α, within LLM and between LLMs.
   - Predictive: for coalition agreements, proportion of estimated coalition
     positions falling inside the convex hull of member-party positions.
6. **Replication.** Repeat the entire pipeline (a) after a 3-month wait with
   the same proprietary LLMs (reproducibility) and (b) with three open-weight
   LLMs — DeepSeek-V3 671B, LLaMA-3.3-70B-Instruct, Gemma-3-27b-it —
   hosted on Nebius (replicability).

**Ensemble structure for a single (manifesto, dimension) cell.**
3 LLMs × (zero-shot + few-shot) × 3 summaries per LLM (one from each LLM
acts as the summary for all three scorers) = 18 scalar scores aggregated by
mean. Per dimension, the cross-run total is 235 manifestos × 6 dimensions × 3
LLMs × 2 shot-settings × 3 summary sources ÷ one-reuse = 25,380 LLM score
outputs across the full grid.

### 1.5 Inference settings

- Temperature = 0 for all models.
- top_p = 1.
- Random seed fixed where the API supports it; residual stochasticity still
  observed (their own explanation: "undocumented aspects of inference
  architecture," p. 5).
- Langchain (Chase 2022) as the orchestration layer.
- Nebius Studio (https://studio.nebius.com/) for open-weight inference on
  hosted hardware.

### 1.6 Published headline numbers

**Concurrent validity (Figure 1, p. 8) — baseline proprietary ensemble,
Pearson correlation of 18-score mean vs. expert ensemble mean:**

| Dimension | r | Expert upper bound (Table 3) |
|---|---|---|
| Economic | .87 | .88 [.85, .90] |
| Social | .92 | .91 [.89, .93] |
| Immigration | .89 | .88 [.86, .91] |
| EU | .91 | .95 [.93, .96] |
| Environment | .82 | .84 [.82, .87] |
| Decentralization | .49 | .78 [.74, .82] |

Five of six dimensions are inside their 95% upper-bound CI. Decentralization
is the only real miss, and Section 4.4–5 of the paper explains why (below).

**Reliability (Table 5, p. 12)** — intra-LLM is same-model repeat scoring,
inter-LLM is cross-model on the same summary:

| Dimension | Intra ICC | Intra α | Inter ICC | Inter α |
|---|---|---|---|---|
| Economic | .90 | .90 | .91 | .90 |
| Social | .95 | .91 | .97 | .91 |
| Immigration | .93 | .93 | .93 | .92 |
| EU | .92 | .87 | .92 | .88 |
| Environment | .80 | .79 | .76 | .74 |
| Decentralization | .66 | .62 | .69 | .57 |
| Overall | .91 | .90 | — | — |

Compared to inter-coder reliabilities of .3–.5 for human MP coders (Mikhaylov
et al. 2012, p. 90), the LLM ensemble is more consistent than human coders at
the task.

**Missingness (Table 4, p. 11)** — proportion of cells where the LLM
returned "NA" (not enough info) for (manifesto, dimension, summary):

| Dimension | Claude 3.5 | GPT-4o | Gemini 1.5 Pro | Overall |
|---|---|---|---|---|
| Economic | .13 | .02 | .08 | .08 |
| Social | .24 | .21 | .38 | .28 |
| Immigration | .42 | .24 | .27 | .31 |
| EU | .16 | .03 | .05 | .08 |
| Environment | .12 | .01 | .04 | .06 |
| Decentralization | .20 | .01 | .13 | .11 |
| All | .21 | .09 | .16 | — |

GPT-4o is the "more eager" summarizer; Claude/Gemini prefer to abstain when
the manifesto does not explicitly address an issue. Benoit et al. treat this
as a research-design trade-off, not a defect.

**Reproducibility + replicability (Table 6, p. 13):**

Left panel — same three proprietary LLMs, 3 months later:
- Second-ensemble correlation with original ensemble: .97 Economic, .97
  Social, .96 Immigration, .98 EU, .92 Environment, .69 Decentralization.
- Replication correlation with expert surveys: .87 Economic, .92 Social,
  .89 Immigration, .91 EU, .82 Environment, .49 Decentralization
  (matches Figure 1 headline).

Right panel — open-weight replacement (LLaMA / DeepSeek / Gemma):

| Dimension | Open-ensemble vs proprietary-ensemble | vs expert (LLaMA) | vs expert (DeepSeek) | vs expert (Gemma) |
|---|---|---|---|---|
| Economic | .95 | .84 | .84 | .86 |
| Social | .95 | .87 | .87 | .86 |
| Immigration | .93 | .86 | .89 | .89 |
| EU | .96 | .86 | .86 | .84 |
| Environment | .92 | .68 | .79 | .86 |
| Decentralization | .40 | .40 | .45 | .45 |

Open-weight is only ~.05 off proprietary on five dimensions and *equal* on
Decentralization. Ensemble-to-ensemble correlation ≈ .95 — open-weight is a
viable drop-in.

**Predictive validity on coalition agreements (Table 7, p. 15)** — proportion
of 23 LLM-estimated coalition positions that fall inside the convex range of
their own member parties' positions (spatial theory says this should be ~1):

| Dimension | LLM mean | Klüver/MP logit |
|---|---|---|
| Economic | .76 | .36 |
| Social | .75 | .45 |
| Immigration | .53 | .14 |
| EU | .57 | .45 |
| Environment | .78 | .18 |
| Decentralization | .43 | .61 |
| Overall | .64 | .38 |

LLM estimates beat the hand-coded baseline on 5 of 6 dimensions. Note
Decentralization is the only dimension where MP hand-coding *beats* LLM; this
reinforces the interpretation that the Decentralization anomaly is a
manifesto-content issue, not an LLM-scoring issue (see §1.7).

### 1.7 The Decentralization anomaly (pp. 8–9, Figure 2–3)

The only dimension where the LLM ensemble badly misses the expert upper
bound (r=.49 vs. ceiling .78). Benoit et al.'s explanation — worth citing
verbatim in C-TreePO — is that it is a *data* anomaly, not a method anomaly:

1. MP's own per301 (decentralization-favorable) and per302 (anti-) codes
   show that almost no manifesto has anti-decentralization content (Figure 2
   boxplot). Parties virtually never publicly oppose local control.
2. Yet expert scores span the full 0–10 range. Experts are reading cues not
   explicitly stated in the manifesto — strategic inferences, record in
   office, party family.
3. LLMs, restricted to the manifesto text, cannot infer what isn't there.

Operational implication for C-TreePO: **Decentralization is where a flat
summarize-then-scale pipeline and a tree pipeline should agree, because the
information gap isn't in compression — it's in the source.** Our Phase B
results should not try to "fix" Decentralization; if our tree matches
Benoit's open-weight .40–.45 on this dimension, that is *evidence of
method agreement, not a shortfall*. Over-indexing on Decentralization is a
trap.

### 1.8 Data-leakage checks (§4.6, pp. 12–13)

Three defensible reasons they trust the results are not memorization:
1. Anonymization: half the summaries contained a party name; they stripped
   all party names and re-scored with GPT-4o zero-shot. r with non-anonymized
   = .99.
2. Scale mismatch: Benoit–Laver uses 1–20, CHES uses 1–11, and they prompt
   on 1–7. Direct regurgitation of training-data scale values is
   therefore geometrically implausible.
3. If leakage were present, Decentralization would also correlate highly with
   expert surveys. It does not.
4. Chat probe: asked models directly whether they were retrieving CHES
   scores from memory; all three denied and referred to the manifesto text
   (transcripts in Supporting Information Appendix H, pp. 25–30).

### 1.9 Methodological positions worth lifting into C-TreePO prose

- "We make the LLMs summarize what a text has to say about particular issues.
  We then ask them to answer questions about the author's issue positions.
  This is a canonical application of Natural Language Understanding (NLU)."
  (p. 2) — paraphrase for the C-TreePO intro.
- On prompt engineering: "small changes in the wording of scoring prompts
  might have big effects on LLM outputs… we concluded that small changes in
  prompt wording of the type we investigated are not critical in this
  application" (p. 6). Useful for C-TreePO: our rubric-driven prompts
  inherit this robustness claim.
- On few-shot vs zero-shot: "correlations between few-shot LLM and expert
  scores are essentially the same as for zero-shot scores… for many issues,
  zero-shot correlations are already close enough to the upper bound that
  there is very little headroom for improvement" (p. 8). We can drop few-shot
  from Phase B unless Phase A shows a delta.
- On LLM "understanding" vs cognition: "understanding in NLU is not
  understanding in the sense that machines use anything approaching human
  cognition. NLU characterizes the set of machine tasks that can replicate
  outputs of human interpretations" (p. 2). Matches our
  oracle-as-functional-equivalence framing.

### 1.10 What Benoit et al. do *not* do (openings for C-TreePO)

- No formal preservation guarantees on the summary. Summaries are validated
  empirically (r with expert means); the paper is explicit that it offers
  "a demonstration rather than a technical analysis" (p. 16). C-TreePO
  supplies the missing audit.
- No explicit handling of interaction-bearing content that spans sections
  (economic policy conditional on environmental trade-offs, e.g.). The
  single-dimension summary is implicitly marginalized over interactions.
  This is exactly the Markov-boundary failure mode of
  [paper/ctreepo/sections/03_example.tex:9](../paper/ctreepo/sections/03_example.tex#L9)
  translated to semantics.
- No tree / hierarchical structure. The summary stage is one flat call per
  (manifesto, dimension). Long-document scaling is therefore bounded by
  context window and single-pass attention, not by a certified audit.
- No finite-sample bound on how much the final position estimate can drift
  due to summary error. Benoit et al. treat reliability (ICC, α) as the
  variance story; C-TreePO's certificate would add a bias story.
- Ensemble dependencies are opaque: 3 LLMs × 2 shot × 3 summaries sharing
  the same three summaries creates correlated draws, but the 18-score
  ensemble mean is reported as if draws were independent. A C-TreePO
  treatment that makes the propensity structure explicit (per
  [sections/05_estimation.tex](../paper/ctreepo/sections/05_estimation.tex))
  is a natural methodological add-on.

---

## 1bis. Replication archive — what's in it and how we use it

Benoit's AJPS Dataverse archive (DOI: 10.7910/DVN/XY1FFE) is unpacked at
[data/examples/benoit_dataverse/](../data/examples/benoit_dataverse/). It
contains the expert-mean benchmark, all LLM scores, MP metadata, Klüver
scores, and R-Markdown scripts for every table and figure. This is a much
better foundation than re-ingesting CHES + Benoit–Laver raw files and
materially changes our plan.

### Files we use (as verified by [src/tasks/manifesto/expert_benchmarks.py](../src/tasks/manifesto/expert_benchmarks.py))

| File | Shape | Our use |
|---|---|---|
| `data_experts.rda` | 1,475 rows × 9 cols; `(manifesto, issue)` keyed | Ground-truth expert ensemble means (already rescaled). Loader: `load_benoit_experts()` / `load_benoit_expert_means(dim)`. |
| `data_mp.rda` | 233 manifestos × 193 cols | Manifesto-string ↔ MP `(party, year)` crosswalk. Loader: `load_benoit_mp_crosswalk()`. |
| `data_llms_all_reported.rds` | 52,056 rows; 6 cols `(manifesto, issue, model_scaling, model_summary, score_llm, run)` | Their proprietary 3×3 ensemble scores. Used for sanity-check reproduction of Figure 1. |
| `data_llms_all_openweight.rds` | 17,316 rows | Their open-weight (LLaMA/DeepSeek/Gemma) scores. |
| `data_llms_all_replication.rds` | 38,718 rows | Their 3-month re-run of proprietary. |
| `data_bl_detail.rda` | 112,916 rows × 30 cols | Individual Benoit–Laver expert responses. Needed if we re-compute the Table 3 upper bound ourselves. |
| `data_kluwer_logscores.rda` | — | Klüver hand-coded coalition scores. Currently fails to read via `pyreadr` (unsupported feature); will need R or `rdata` package for Phase C. |
| `data_masked.csv` | 22 MB | Anonymized summaries for the §4.6 leakage test. |
| `Table_*.Rmd`, `Figure_*.Rmd` | R Markdown | Reference implementation for every reported number. Read before writing any corresponding Python analogue. |

### Sanity-check pass

[scripts/reproduce_benoit_figure1.py](../scripts/reproduce_benoit_figure1.py)
loads `data_llms_all_reported.rds` × `data_experts.rda`, collapses to the
18-score ensemble mean per `(manifesto, issue)`, and correlates against
`expert_mean`. Output (verified this session):

```
Dimension            Published    Ours      Δ      n       95% CI
economic                  0.87    0.872  +0.002   234   [0.84, 0.90]
social                    0.92    0.916  -0.004   220   [0.89, 0.94]
immigration               0.89    0.890  +0.000   181   [0.85, 0.92]
eu                        0.91    0.906  -0.004   194   [0.88, 0.93]
environment               0.82    0.821  +0.001   200   [0.77, 0.86]
decentralization          0.49    0.495  +0.005   235   [0.39, 0.58]
max |Δ| = 0.005
```

All six dimensions reproduce within 0.005 of the published numbers. This
confirms our `compute_corpus_pearson_r` + ensemble-mean pipeline is
algorithmically identical to theirs and gives us a zero-inference smoke
test we can re-run any time we change the eval code.

### What this collapses in the original plan

- §2 row 1 (MP corpus loader): still needed for our own text-fed runs, but
  the specific 235-manifesto list is now trivially derivable from the
  `data_mp.rda` crosswalk instead of Benoit SI A1.
- §2 rows 3–4 (CHES trend file, Benoit–Laver expert tables): **no longer
  needed**. Their `data_experts.rda` already contains the ensemble means
  we need; their `data_bl_detail.rda` provides individual expert responses
  for Table 3 upper-bound recomputation.
- §2 row 12 (CHES expert-survey upper-bound simulation): now a direct
  re-run of `Table_3.Rmd` logic on `data_bl_detail.rda`, not a
  first-principles CHES ingestion.
- §4 Phase 0 "blocker" of CHES ingestion: **gone**. The pilot script can
  run today once a vLLM server is up.

---

## 2. C-TreePO alignment gap analysis

Mapping each Benoit pipeline component to current C-TreePO infrastructure.
Status: **reuse** (already works), **extend** (exists, needs new features),
**build** (not present).

| # | Benoit component | Current C-TreePO | Status | What's needed |
|---|---|---|---|---|
| 1 | MP corpus (235 manifestos, 21 langs, CHES-matched years) | [outside_data/Manifesto_Project/MPDataset_MPDS2025a.csv](../outside_data/Manifesto_Project/MPDataset_MPDS2025a.csv), loader at [src/tasks/manifesto/data_loader.py:14-44](../src/tasks/manifesto/data_loader.py#L14-L44) | extend | Election-year × expert-survey-year filter; reconcile to Benoit's SI A1 list |
| 2 | Klüver coalition agreements (23 docs) | absent | build | Intake from Klüver 2023 replication archive; plain-text conversion; year filter |
| 3 | CHES trend file (Jolly et al. 2022) | absent | build | Load CHES 1999–2019; subset to six dims; rescale to 1–7 per Benoit SI A2 |
| 4 | Benoit–Laver 1992/2006 expert tables | absent | build | Ingest from published appendices; rescale 1–20 → 1–7 |
| 5 | Six-dimension 7-point scale | only RILE at [src/tasks/manifesto/__init__.py:97-104](../src/tasks/manifesto/__init__.py#L97-L104) | build | `PolicyDimension` enum (6 cases); six `ScaleDefinition` instances with (min=1, max=7, low/high anchor text, vignettes) from Benoit Table 2 |
| 6 | Per-dimension rubrics with end-scale vignettes | RILE rubric only at [src/tasks/manifesto/rubrics.py](../src/tasks/manifesto/rubrics.py) | build | Six rubrics authored from Benoit SI A4 (pp. 6–11); vignette-text field |
| 7 | Summarization prompt (300–400 words, native-lang input, English output, per-dimension) | [src/tasks/manifesto/pipeline.py](../src/tasks/manifesto/pipeline.py) has `ManifestoSummarizer` | extend | Cross-check prompt text against Benoit SI A3; make per-dimension instantiation |
| 8 | Scoring prompt ("expert political scientist, PhD" wording) | [src/tasks/manifesto/dspy_signatures.py](../src/tasks/manifesto/dspy_signatures.py) has `RILEScorer` | extend | 7-point integer output; wire to dimension-specific rubric |
| 9 | Zero-shot + few-shot exemplars (left/center/right per dimension) | zero-shot only | build | Curate three benchmark summaries per dimension (18 total) from manifestos outside the test corpus |
| 10 | 18-score ensemble per (manifesto, dimension) | absent | build | `ManifestoEnsemble` driver: iterate 3 LLMs × 2 shot-settings × 3 summaries, collect scores, report mean with NA handling |
| 11 | Informative-NA handling | absent | build | Two-level NA: at summary stage ("text does not address dimension") and scoring stage ("summary insufficient to score"); exclude from mean, report rate |
| 12 | Deterministic inference (T=0, top_p=1, fixed seed) | config unclear, check [config/settings.yaml](../config/settings.yaml) | extend | Explicit deterministic block per LLM; test that reruns match |
| 13 | ICC + Krippendorff α reliability | absent | build | `scripts/manifesto_reliability.py` using `krippendorff` + `pingouin`; intra-LLM repeat runs on a held-out subset |
| 14 | Pearson r vs expert means | absent | build | `scripts/manifesto_validity.py`; per-dimension correlation with 95% CI |
| 15 | Expert–expert upper-bound simulation | absent | build | 1,000-iteration split-sample replicate of Table 3 on whichever expert-survey source has most respondents (Benoit–Laver) |
| 16 | Anonymization leakage test | absent | build | Post-hoc party-name stripper on summaries + re-scoring run on 20–30 manifestos |
| 17 | Coalition predictive-validity test | absent | build | Given (coalition, dimension), check estimated position falls inside the range of member-party estimated positions |
| 18 | Open-weight inference (DeepSeek-V3, LLaMA-3.3-70B, Gemma-3-27B) | existing quantization scripts ([quantize_nemotron_*.py](../quantize_nemotron_llmcompressor.py), etc.) suggest local vLLM capability; verify | extend | vLLM/Nebius config per model; benchmark inference cost for 235 × 6 × 2 × 3 = 8,460 summary/scoring calls per LLM |

**Summary of work.** 9 "build" items, 7 "extend" items, 2 "reuse" items. The
build items cluster into three workstreams that can proceed in parallel:
(a) data intake (rows 2, 3, 4, 16), (b) scale/rubric/prompt (rows 5, 6, 9, 11),
(c) measurement + validation (rows 10, 13, 14, 15, 17).

### Depth-discounting / local-law notes

The `gamma^d` depth discount implemented in C-TreePO's training loop
(MEMORY.md; [src/ctreepo/sim/core/leaf_local_mixture_utility.py](../src/ctreepo/sim/core/leaf_local_mixture_utility.py))
applies per-node loss weights; for manifesto text we'd expect the
MEMORY-flagged "C2-only dominance" finding from the Markov DGP to *not*
transfer directly — manifesto paragraphs are more LDA-like than Markov-like,
and MEMORY.md notes the LDA DGP's `all_laws` run gives +6.8% instead of
being catastrophic. Phase B below should specifically run both `c2_only`
and `all_laws` packages to see which regime political text occupies.

---

## 3. Experiment blueprint

Three phases, each one sentence summarizable: *replicate Benoit → swap in
tree → push where tree should win.* All three share the same data intake
(rows 1–4 above) and the same validity tooling (rows 13–15, 17), so the
upfront build amortizes across all phases.

### Phase A — Parity replication

**Question.** Can C-TreePO's codebase reproduce Benoit's open-weight
Table 6 numbers with the same pipeline and models?

- **Corpus.** Exact 235 MP manifestos + 23 Klüver coalition agreements from
  Benoit SI A1 and Table F1.
- **Protocol.** Flat Benoit pipeline: one 300–400 word summary per
  (manifesto, dimension, LLM) → 7-point scale score → 18-score mean.
- **LLMs.** DeepSeek-V3, LLaMA-3.3-70B-Instruct, Gemma-3-27b-it.
- **Shot settings.** Zero-shot as primary; few-shot as secondary only if
  zero-shot misses Benoit's numbers (they report few-shot ≈ zero-shot).
- **Output.** `outputs/benoit_parity/{dimension}/{llm}/*.jsonl` per-score
  records; summary `report.csv` with Pearson r per dimension, intra-LLM ICC,
  inter-LLM ICC, NA rate.
- **Acceptance.** All six Pearson r's within ±0.05 of Benoit's Table 6
  open-weight columns, allowing 3–6 months of model drift. Overall NA rates
  within ±0.08 of Benoit's Table 4.

**If Phase A fails.** Treat it as a pipeline bug, not a method finding.
Likely culprits: prompt wording drift, scale-rescaling error in the expert
benchmark, different Nebius vs local-vLLM sampling behavior.

### Phase B — C-TreePO method swap

**Question.** Does swapping Benoit's flat summary stage for a C-TreePO tree
(with local-law audits) match or exceed the Phase A correlations, and do
audit pass rates correlate with per-dimension accuracy?

- **Corpus + LLMs + benchmarks.** Identical to Phase A.
- **Protocol.**
  - Split each manifesto into span leaves (target leaf size: 512–1024
    tokens; defer the exact split to `config/manifesto_ctreepo.yaml` so we
    can ablate).
  - Summarize leaves with the same per-dimension prompt as Phase A.
  - Pairwise merge siblings up to the root using a merge prompt aligned to
    the C1/C3 laws in [sections/02_framework.tex](../paper/ctreepo/sections/02_framework.tex).
  - Score the root summary on the 7-point scale with the same prompts as
    Phase A.
  - **Audit subroutine.** On a random subset of internal nodes per
    manifesto, run the C2/C3 audit from
    [sections/05_estimation.tex](../paper/ctreepo/sections/05_estimation.tex)
    and log pass rates per dimension.
- **Law packages to compare.** `c2_only` and `all_laws`, per MEMORY.md's
  note that political text may behave more like LDA than Markov.
- **Output.** `outputs/ctreepo_manifesto/{dimension}/{llm}/{law_pkg}/*.jsonl`
  with per-node audit results + root scores. `report.csv` adds audit pass
  rate per dimension alongside Phase-A-comparable columns.
- **Acceptance.**
  - **Primary.** Phase-B Pearson r ≥ Phase-A Pearson r − 0.02 on each of
    Economic, Social, Immigration, EU, Environment. Decentralization is
    excluded from the acceptance criterion per §1.7.
  - **Secondary.** Audit pass rate positively correlated with per-dimension
    accuracy (Spearman ρ ≥ 0.3 across the six dimensions). If this secondary
    criterion holds, we have a certificate that predicts its own accuracy.
- **Nice-to-have.** Cost-per-manifesto comparison (Phase A flat summary
  ≈ one 30K-token prompt; Phase B tree ≈ N_leaf × 2K prompts plus merges).

### Phase C — Where C-TreePO should beat Benoit

Choose one of the three candidates after Phase B is in hand. Document the
decision procedure in the Phase B report.

- **C-long.** Re-run Phase B on legislative-speech / committee-transcript
  corpora where single-prompt summarization degrades. Hypothesis: tree wins
  because the summary stage doesn't have to fit the whole document in
  context.
- **C-interact.** Construct a dimension that is explicitly interaction-
  bearing — e.g., "willingness to accept environmental cost for economic
  growth" — where single-section summaries lose the trade-off structure. The
  Markov-boundary intuition from
  [sections/03_example.tex:9](../paper/ctreepo/sections/03_example.tex#L9)
  transferred to semantics. Validation target: a small, hand-annotated gold
  set since CHES/Benoit–Laver don't ask this cross-dim question.
- **C-small-model.** Run a 7B–13B model (e.g., Llama-3.2-8B, Qwen-2.5-7B)
  under the C-TreePO tree and compare against a 70B model under the flat
  Benoit pipeline. Hypothesis: tree + audit on small model ≈ flat on large.

### Shared tooling (all three phases)

- **Data intake.** Extend [src/tasks/manifesto/data_loader.py](../src/tasks/manifesto/data_loader.py)
  with `load_benoit_235_corpus()` + CHES/Benoit–Laver loaders. New module
  `src/tasks/manifesto/expert_benchmarks.py`.
- **Six-dimension scales.** `src/tasks/manifesto/dimensions.py` with
  `PolicyDimension` enum and six `ScaleDefinition` instances from Benoit
  Table 2. Keep `RILE_SCALE` for back-compat.
- **Rubrics.** Extend [src/tasks/manifesto/rubrics.py](../src/tasks/manifesto/rubrics.py)
  with six rubric objects + their Benoit SI A4 vignettes.
- **Ensemble driver.** `src/tasks/manifesto/ensemble.py` — `ManifestoEnsemble`
  class producing the 18-score aggregate with NA handling. Used by all
  three phases.
- **Validity tooling.** `scripts/manifesto_validity.py` (Pearson r, 95% CI),
  `scripts/manifesto_reliability.py` (ICC, α), `scripts/manifesto_predictive.py`
  (coalition-range check).
- **Configs.** `config/benoit_parity.yaml`, `config/ctreepo_manifesto.yaml`
  driving Phase A vs Phase B.

### Out of scope for this document

- Klüver corpus licensing and format logistics.
- CHES trend-file parsing code (structure may have changed since 2019).
- Any model inference this session — plan + notes only.
- Extending C-TreePO's Lean formalization to six-dimension text.

---

## 4. Correlation-with-expert-survey objective: theoretical framing and minimal path

Benoit et al. report Pearson r with expert ensemble means, not raw error. This
section frames that choice inside C-TreePO's preservation theorems and lays
out the minimum work to produce a directly comparable number.

### 4.1 Pearson r inside our preservation framework

Benoit's choice is really two decisions packaged together, not a new training
objective:

1. **Oracle re-definition.** Their `f*(x)` is the expert ensemble mean for
   `(party(x), year(x), dimension)`, rescaled to 1–7. Ours is currently the
   RILE value shipped in MPDS 2025a.
2. **Reporter swap.** They summarize over a corpus with `r(f_hat_batch,
   benchmark_batch)` rather than `mean_i |f_hat(x_i) - f*(x_i)|`.

Note that their LLMs are frozen; correlation is a post-hoc metric, not a
training signal.

**How this lines up with our theory.** Under the Preservation Stack
([paper/ctreepo/sections/02_framework.tex](../paper/ctreepo/sections/02_framework.tex),
[paper/ctreepo/sections/04_theory.tex](../paper/ctreepo/sections/04_theory.tex)),
if local laws hold then `f_hat(root(x)) = f*(x)` with zero expected
distortion, so `r(f_hat, benchmark) = r(f*, benchmark)` over the corpus.
This is exactly the "expert–expert upper bound" argument in Benoit §3.2,
Table 3: if `f*` is a benchmark-correlated oracle, the best any method can
do is the correlation `f*` itself has with the benchmark.

Under the Certificate Stack ([paper/ctreepo/sections/05_estimation.tex](../paper/ctreepo/sections/05_estimation.tex)),
approximate preservation gives `||f_hat - f*||_2 ≤ δ`. By Cauchy–Schwarz:

```
r(f_hat, benchmark) ≥ r(f*, benchmark) − δ / σ(f*)
```

where `σ(f*)` is the corpus standard deviation of the oracle. Reading: the
correlation gap is bounded by our existing MSE-style certificate divided by
corpus oracle variance. **We do not need a separate correlation-based
guarantee — MAE/MSE bounds already control it.**

**Invariance bonus.** Pearson r is invariant to affine transforms of
`f_hat` and the benchmark. Our current `-100/+100` RILE outputs correlate
identically against a `1–7` CHES Economic mean as a rescaled `1–7` output
would. This means **no retraining is required to change output scale** for
evaluation against Benoit's benchmarks. Rescaling is needed only for
visualization and for the per-manifesto regression intercept, not for r.

### 4.2 Phase 0 — eval-only Pearson r (this week)

Goal: one concrete Pearson r number, on one dimension, for one LLM, on a
subset of manifestos, directly comparable to Benoit's Table 6.

**Scope.** Economic dimension only. ~50 manifestos from countries with
strong CHES coverage (e.g., UK, Germany, France, Netherlands) from 2010
onward. One open-weight LLM (LLaMA-3.3-70B or whatever our local stack
already has up). Zero-shot only.

**Scaffolded — ready to wire up once CHES data is in place.** All files
live under [src/tasks/manifesto/](../src/tasks/manifesto/) and
[scripts/](../scripts/); imports verified against the existing public API.

| File | Role | Status |
|---|---|---|
| [src/tasks/manifesto/dimensions.py](../src/tasks/manifesto/dimensions.py) | `PolicyDimension` enum + `DimensionSpec` + `BENOIT_DIMENSIONS` dict; all six dimensions on a 1-7 `ScaleDefinition` with CHES variable name | ✅ complete |
| [src/tasks/manifesto/scoring_contexts.py](../src/tasks/manifesto/scoring_contexts.py) | 1-7 scoring `task_context` strings with Benoit's "expert political scientist, PhD" framing; Economic anchors authoritative, end-scale vignettes stubbed with TODO for Benoit SI A4 | ✅ Economic; vignettes TODO for all six |
| [src/tasks/manifesto/dimension_scorer.py](../src/tasks/manifesto/dimension_scorer.py) | `DimensionScorer(DimensionSpec)` DSPy module; dimension-aware score parsing (no RILE-specific normalization); honors `NA` returns | ✅ complete |
| [src/tasks/manifesto/expert_benchmarks.py](../src/tasks/manifesto/expert_benchmarks.py) | `load_benoit_experts()`, `load_benoit_expert_means(dim)`, `load_benoit_mp_crosswalk()`, `load_benoit_llm_scores(kind)`, `benoit_ensemble_mean()`; reads AJPS Dataverse archive via `pyreadr` | ✅ complete, ✅ data present |
| [src/tasks/manifesto/corpus_metrics.py](../src/tasks/manifesto/corpus_metrics.py) | `compute_corpus_pearson_r(pred, true)` with Fisher-z CI + Spearman + MAE/RMSE; `CorrelationReport` dataclass | ✅ complete (tested on synthetic input, r=.997 on n=5) |
| [scripts/phase0_economic_pilot.py](../scripts/phase0_economic_pilot.py) | End-to-end: vLLM config → `ManifestoDataset` → CHES join → summarize + score → `per_manifesto.jsonl` + `report.json` with Benoit reference values baked in | ✅ complete, ⏳ requires CHES data to run |

Nothing in [src/ctreepo/](../src/ctreepo/) needs to change for Phase 0.
Training loss stays MAE. Existing modules reused without modification:
[pipeline.py](../src/tasks/manifesto/pipeline.py) (`ManifestoSummarizer`,
`ManifestoMerger`), [rubrics.py](../src/tasks/manifesto/rubrics.py)
(`ECONOMIC_RUBRIC` for the summarization side),
[dspy_config.py](../src/config/dspy_config.py) (`create_vllm_lm`,
`configure_dspy`), [data_loader.py](../src/tasks/manifesto/data_loader.py)
(`ManifestoDataset`), [chunker.py](../src/preprocessing/chunker.py)
(`chunk_for_ops`).

**Data blocker: resolved.** Benoit's archive at
[data/examples/benoit_dataverse/](../data/examples/benoit_dataverse/)
provides expert means, LLM scores, MP crosswalk, and per-table R-Markdown
scripts. The pilot now joins on `(party_id, year)` via
`load_benoit_mp_crosswalk()` — no CHES wrangling required.

**Zero-inference sanity check** (re-run any time):

```
python scripts/reproduce_benoit_figure1.py --kind reported
# Reproduces Benoit Figure 1 within |Δ| ≤ 0.005 across all six dimensions.
```

**First actual C-TreePO run** (requires vLLM server; summarizer + scorer
inference):

```
python scripts/phase0_economic_pilot.py \\
    --port 8000 \\
    --countries 51 41 31 22 \\
    --min-year 2010 --max-year 2019 \\
    --max-manifestos 50 \\
    --output-dir outputs/phase0_economic
```

**Decision branch on Phase 0 result.**

- `r ≥ 0.75`: in Benoit's ballpark for Economic (.84–.87 on open-weight,
  Table 6). Scale up to six dimensions (Phase A of §3) and keep MAE training.
- `0.5 ≤ r < 0.75`: pipeline works but has a calibration or prompt issue —
  debug the rubric and summary prompt against Benoit SI A3/A4 before
  scaling. Do not touch training loss yet.
- `r < 0.5`: something fundamentally broken (data alignment, party-year
  join, scale inversion). Debug data intake first; training loss still
  not the lever.

### 4.3 Phase 1 — training-loss swap (conditional follow-up)

Trigger: only if Phase 0 + Phase A (§3) plateau meaningfully below Benoit's
Table 6 open-weight column and the failure pattern suggests calibration
rather than retrieval. Benoit's own finding that "few-shot correlations are
essentially the same as zero-shot" (p. 8) is a soft prior that this phase
will not move numbers much.

**Loss design.** Differentiable batch-centered correlation surrogate:

```
L_corr(batch) = 1 − cov(f_hat_batch, f*_batch) / (σ(f_hat_batch) · σ(f*_batch))
L = α · L_MAE_per_node + β · L_corr_root
```

- Node-level (leaf + merge) losses stay MAE against the local oracle —
  the local laws audit against point targets, and Pearson is undefined on
  single nodes.
- Root loss gets the correlation term. Requires batch size ≥ ~16 for stable
  gradients (std in the denominator is noisy for small batches).
- Hyperparameter search: `β ∈ {0, 0.1, 0.3, 1.0}`, `α = 1 − β`. `β = 0` is
  the Phase A baseline.

**What this does to theory.** The oracle-preservation theorems do not
care about loss choice — they care about whether the trained pipeline
achieves `f_hat ≈ f*`. A correlation surrogate could in principle converge
to a different `f_hat` (any affine transform of `f*` yields `L_corr = 0`),
which means the surrogate does not uniquely identify `f*` without an
additional location/scale anchor. Practical fix: add a small MAE anchor
(`α > 0`) to pin the scale. This is standard in correlation-loss papers
(e.g., MedLoss, Pearson-CCA training).

**Risk.** Batch-level loss interacts with the depth-discount gamma work in
MEMORY.md (the `gamma^d` schedule applies per-sample per-node). Need to
verify that per-batch correlation at the root and per-sample weighted
locals don't fight each other. Fastest check: run `γ=1.0, β ∈ {0, 0.3}`
on the existing Markov sim and see if test MAE moves.

**Code diffs.**

| File | Change | Size |
|---|---|---|
| `src/training/metrics.py` | Add `batch_pearson_loss(pred, true)` | ~20 lines |
| `src/training/losses.py` | Compose `mae_pearson_hybrid(α, β)` | ~15 lines |
| `config/settings.yaml` | Add `training.root_loss: {mae, pearson, hybrid}` | config-only |
| Experiment script | Hyperparam sweep over β on Markov sim first, manifesto second | ~50 lines |

### 4.4 Minimal critical path

The smallest sequence of concrete steps to land one Benoit-comparable number
and then decide whether to deepen:

1. **Ingest CHES Economic means** for ~50 parties. `outside_data/CHES/` +
   `expert_benchmarks.py`. ~2 hours.
2. **Author Economic rubric** from Benoit SI A4 p. 6. `rubrics.py`. ~1 hour.
3. **Add Pearson-r reporter.** `metrics.py` + `scripts/phase0_economic_pilot.py`.
   ~3 hours.
4. **Run pilot** on 50 manifestos, one LLM, zero-shot. ~1 hour inference.
5. **Compare to Benoit's .84–.87 Economic column** (Table 6 open-weight).
   Decide: scale up (Phase A), debug prompt (stay small), or debug data.

**Total to first number: ~1 day.** Everything else (six dimensions,
coalition agreements, audit integration, correlation-loss swap) is
downstream of this first number and its decision branch.

---

## 4b. Pilot result (Gemma-4-31B-NVFP4, full n=235 Economic)

**Baseline single-scorer pilot, 2026-04-17.** Scored all 235 Benoit Economic
summaries from `data_masked.csv` using `DimensionScorer` + Gemma-4-31B-IT-NVFP4
at temperature 0, single call per summary, no optimization.

| Comparison                           | Pearson r | 95% CI             | n   |
|--------------------------------------|----------:|-------------------:|----:|
| Ours (Gemma-4-31B) vs expert_mean    | **+0.833** | [+0.790, +0.869] | 235 |
| Benoit single GPT-4o vs expert_mean  | +0.823     | [+0.776, +0.860]| 233 |
| Ours vs Benoit single GPT-4o         | +0.898     | [+0.870, +0.920]| 233 |

Benoit 18-score proprietary ensemble reference: **+0.87** (Figure 1).
Expert–expert upper bound (Table 3): **.88 [.85, .90]**.

**Reading.**
- We beat Benoit's single-shot GPT-4o single-LLM scorer on the same 235-manifesto
  test set by ~+0.01 (well within CI overlap, so call it a tie at single-shot).
  Gemma-4-31B-NVFP4 running locally behaves like a GPT-4o single shot for this
  Economic-scoring task.
- Our score and Benoit's single score correlate at +0.898 — the two single-shot
  scorers are measuring essentially the same latent thing.
- The gap to their ensemble r=.87 is ~+0.04. That's exactly the margin an
  18-score ensemble (variance reduction) buys over a single shot — doing
  3 zero-shot calls ourselves and averaging should close most of it.
- Cost comparison: Benoit's headline r=.87 took 18 frontier-LLM calls per
  manifesto-dimension (at temperature 0, so ≈6 effective with noise). Ours
  used 1 call on a locally hosted 31B NVFP4 quantized model. Same ballpark
  accuracy, ~1/18th the inference budget.

**Pilot caveat.** Our first 30-summary subsample reported r=+0.920 (CI
[+0.837, +0.961]); that was a favorable random draw. The full-235 r=+0.833
is the correct tight estimate. Lesson: wait for the full corpus before
declaring victory on 30-row subsamples.

**Artifacts.** `outputs/phase0_bs_economic_full/per_summary.jsonl` (235 rows)
and `report.json`.

**Why this beats our earlier attempt.** The first pilot failed because our
local MP text extract in `data/raw/manifesto_project_full/` is truncated
for 80% of Benoit's 230 overlap manifestos (median length 114 chars —
promotional footers only). Pivoting to score Benoit's own summaries sidesteps
that entirely and isolates the scoring step.

### 4c. Ablation matrix using Benoit's data

With [scripts/phase0_score_benoit_summaries.py](../scripts/phase0_score_benoit_summaries.py)
working end-to-end, we have four clean cells for ablating where the signal
comes from. Each cell maps to a C-TreePO oracle-preservation story:

| Condition | f_summarize | f_score | Training signal | Benoit-data sufficient? |
|---|---|---|---|---|
| Zero-shot (today's baseline) | frozen (Benoit's summaries used as-is) | frozen zero-shot | — | Yes |
| Scorer only | frozen | DSPy-optimized on (summary, expert_mean) | Expert means from `data_experts.rda` | Yes |
| Summarizer only | DSPy-optimized on (manifesto, expert_mean) with fixed scorer | frozen | Requires MP text — re-acquire from MP corpus | No, text blocker |
| Full | DSPy-optimized jointly | DSPy-optimized jointly | Both | No, text blocker |

The user's framing (2026-04-17): "we can (even should…?) use DSPy to learn
better summaries here! Or at least to compare between optimized and
non-optimized f functions." This matrix is the answer — the **Scorer-only**
cell is the next concrete step and needs no MP text.

### 4d. DSPy optimization plan (Phase 1b)

**Target.** Optimize `f_score` — the prompt that drives `DimensionScorer` —
against expert means on a Benoit-summary × expert-mean training set.

**Data split.** 235 Benoit manifestos matched to expert means; 6 dimensions
available but Economic first. Random split with fixed seed:
  - train: 140 manifestos × 1 summary each
  - dev:    45 manifestos (for MIPROv2's validation loop)
  - test:   50 manifestos (held out, touched only once)

**Program.** `DimensionScorer(spec).score` (the inner `dspy.Predict`).

**Metric.** For the optimizer we need a per-example scalar, not a
corpus-level Pearson r. Use `1 − |score − expert_mean| / scale.range` so
that higher is better and it composes with MIPROv2. Post-hoc we still
report Pearson r on the held-out test set.

**Optimizers to try (in order of cost).**
1. `BootstrapFewShot(k=8)` — just mines exemplars.
2. `MIPROv2(mini_batch_size=8, num_candidates=6)` — tunes instructions + demos.

**Expected outcome.** Today's baseline is already above .9 on Economic, so
headroom is small. The point of this ablation is not to beat .92 — it's to
show the optimized scorer matches or exceeds Benoit's 18-score ensemble on
the *same budget* (one call per manifesto vs 18).

**Methodologically interesting companion: f_summarize optimization.**
DSPy can optimize a full program where the intermediate output (the summary)
is not directly supervised but the downstream score is. Concretely, holding
`f_score` frozen and optimizing `f_summarize` against expert means is a
test of whether *the summarizer produces the right oracle-preserving state*
— which is exactly the C-TreePO C1/C3 local-law story in operational form.
The twist: we already have Benoit's GPT-4o summaries as a strong baseline
reference, so an optimized f_summarize that *matches or beats Benoit's
summaries* on our fixed scorer is evidence that oracle-sufficient summaries
can be learned directly from scalar corpus feedback — no intermediate
rubric, no gold summaries.

This is the second cell in the ablation matrix above; blocked only by needing
MP text (see §4e).

Code sketch to add later at
`scripts/phase1_optimize_scorer.py`:

```python
trainset = [dspy.Example(summary=row.summary, expected=row.expert_mean).with_inputs("summary")
            for row in train_rows]
metric = lambda ex, pred, _: 1.0 - abs(float(pred.score) - ex.expected) / 6.0
optimized = dspy.BootstrapFewShot(metric=metric, max_bootstrapped_demos=8).compile(
    DimensionScorer(spec), trainset=trainset)
```

### 4e. Using Benoit data per the user's "official data" directive

After today's failure mode (truncated local MP text), the default should
be to **use Benoit's archive data for any cell that supports it**. Concrete
rules:

- **Scoring experiments**: use `data_masked.csv` summaries + `data_experts.rda`
  expert means — no MP text needed. Already wired via
  `load_benoit_masked_summaries()` and `load_benoit_expert_means()`.
- **Figure-1 / Table-6 verification**: use `data_llms_all_reported.rds` and
  friends. Already wired via `load_benoit_llm_scores()` and verified by
  `scripts/reproduce_benoit_figure1.py`.
- **Expert-expert upper bound (Table 3)**: use `data_bl_detail.rda`.
  Loader TODO.
- **Summarization experiments**: still need MP text. Re-download the MP
  corpus properly (via `manifestoR` API / MP website) rather than the
  current `data/raw/manifesto_project_full/` extract, which is effectively
  empty for 80% of the Benoit overlap. This is the only remaining external
  dependency.

---

## 4f. MP corpus fetched via official API (2026-04-18)

The local `data/raw/manifesto_project_full/` extract is unusable for 80% of
the Benoit overlap (median 114 chars — promotional footers only). Fixed by
pulling fresh text from the official Manifesto Project API directly.

### Fetcher

[scripts/fetch_mp_text.py](../scripts/fetch_mp_text.py) — uses the API key
at [outside_data/Manifesto_Project/manifesto_apikey.txt](../outside_data/Manifesto_Project/manifesto_apikey.txt),
hits `https://manifesto-project.wzb.eu/tools/api_texts_and_annotations.json`
on corpus version `2025-1`, reassembles sentence-level text into newline-
separated plain text per `(party, date)` key, writes:

- `data/raw/manifesto_corpus_benoit/texts/{key}.txt` — one file per manifesto.
- `data/raw/manifesto_corpus_benoit/manifesto_maindataset.csv` — MPDS subset
  for the rows we successfully fetched, in the layout `ManifestoDataset`
  expects.
- `data/raw/manifesto_corpus_benoit/fetch_manifest.json` — per-key status log.

Two modes:
- `--benoit-only` (default) — fetches the 233 keys in Benoit's `data_mp.rda`.
- `--all-mpds` — fetches every `(party, date)` in MPDS 2025a metadata
  (5,285 keys). Skips files already on disk so it composes with the default.

### Results

| Run | Keys planned | OK | Skip (cached) | Missing in corpus | Errors | Wall time |
|---|---:|---:|---:|---:|---:|---:|
| Benoit 233 | 233 | 226 | 5 (smoke) | 2 | 0 | 7 min |
| Full MPDS | 5285 | 3110 | 231 | 1944 | 0 | 168 min |
| **Total local corpus** | — | **3341** | — | — | — | — |

The 2 Benoit manifestos missing are Poland 2019 Left (92023_201910) and
Poland 2019 Civic Coalition (92040_201910). The 1944 MPDS rows missing
from the API are mostly older / less-curated entries.

Text-length distribution on the 231 Benoit-overlap fetched: median 104K
chars, mean 167K chars, max 1.5M chars — real manifestos, not footers.

### Loader integration

[scripts/phase0_economic_pilot.py](../scripts/phase0_economic_pilot.py) now
takes `--mp-data-dir`. Pointing it at `data/raw/manifesto_corpus_benoit`
gives `ManifestoDataset` the proper text + MPDS subset CSV without any
other change.

## 4g. Full-pipeline smoke (UK 2019, Gemma-4-31B, n=6)

Earlier 4b result was scoring-only on Benoit's GPT-4o summaries. This is
the first test of *our full pipeline* (chunker → `ManifestoSummarizer` →
`ManifestoMerger` → `DimensionScorer`) on real MP text.

| Comparison | Pearson r | 95% CI | n |
|---|---:|---:|---:|
| Ours full-pipeline vs expert_mean (UK 2019, Economic) | **+0.960** | [+0.675, +0.996] | 6 |

Per-manifesto MAE on the rescaled 1-7 scale: 1.21. Wall time: 36 min for
6 manifestos = 6 min/manifesto with `chunk_chars=16000` (~10 chunks per
manifesto + pairwise merges + 1 score = ~20 LLM calls each).

n=6 is too small to draw conclusions from the point estimate (the
30-summary scoring-only pilot in §4b also reported r=0.92 before the
full-235 run pulled it down to r=0.83). Full-231 in flight.

---

## 4h. Held-out train/test split (no leakage)

User directive (2026-04-18): "train OFF the main/test Benoit docs and train
on the others." The split below holds Benoit's expert benchmark out as the
locked test set and uses Benoit-disjoint manifestos for any optimization.

### Test set (locked — never touched by training)

- 235 manifestos with expert ensemble means in
  `data_experts.rda` (197–235 per dimension after dimension-specific NAs).
- 229 of the 235 also have local MP text now under
  `data/raw/manifesto_corpus_benoit/texts/` (2 missing per §4f, 4 not in
  `data_mp.rda` crosswalk).
- Reference numbers: Benoit Figure 1 (proprietary 18-score ensemble), Table 3
  (expert-expert upper bound), Table 6 (open-weight per-LLM).

### Training pools (verified zero overlap with test set)

Three are now reachable via `expert_benchmarks.py`:

| Pool | Source | n per dim | Supervision |
|---|---|---:|---|
| **A** Benoit GPT-4o summaries minus test | `data_masked.csv` ∖ `data_experts.rda` | 247 | Benoit GPT-4o single-shot score in `data_masked.csv` |
| **B** Local non-Benoit MP text | `data/raw/manifesto_corpus_benoit/` ∖ test keys | 3,112 | MP RILE / per-dim codes from MPDS 2025a |
| **C** Open-weight LLM scores minus test | `data_llms_all_openweight.rds` ∖ test | 187–259 | Mean of LLaMA-3.3-70B / DeepSeek-V3 / Gemma-3-27B per (manifesto, dim) |

[scripts/phase1_optimize_scorer.py](../scripts/phase1_optimize_scorer.py)
defaults to **Pool C** because it has uniform coverage across all 6 dims and
gives a zero-cost teacher signal without re-running anyone's frontier LLM.

Verified no overlap between test set and Pool C across all 6 dimensions:

| Dim | Test n | Train n | Overlap |
|---|---:|---:|---:|
| economic | 235 | 230 | 0 |
| social | 235 | 209 | 0 |
| immigration | 204 | 187 | 0 |
| eu | 197 | 240 | 0 |
| environment | 201 | 259 | 0 |
| decentralization | 235 | 233 | 0 |

## 4i. Overnight 18-run design (2026-04-18)

[scripts/launch_overnight_benoit.sh](../scripts/launch_overnight_benoit.sh)
fires three flights × 6 dimensions = 18 concurrent jobs against vllm port
8010 (Gemma-4-31B-IT-NVFP4):

1. **Scorer-only** ([scripts/phase0_score_benoit_summaries.py](../scripts/phase0_score_benoit_summaries.py))
   — runs `DimensionScorer` on Benoit's anonymized GPT-4o summaries from
   `data_masked.csv`. Isolates the scoring step. ~15 min/dim. Output:
   `outputs/overnight_benoit/scorer_only/{dim}/report.json`.
2. **Full pipeline** ([scripts/phase0_economic_pilot.py](../scripts/phase0_economic_pilot.py)
   with `--dimension`) — chunk → summarize → merge → score on actual MP
   text. ~2-4 h/dim depending on average manifesto length. Output:
   `outputs/overnight_benoit/full_pipeline/{dim}/report.json`. (Economic
   already running from before in
   `outputs/phase0_full_pipeline_economic_229/`.)
3. **DSPy-optimized scorer** ([scripts/phase1_optimize_scorer.py](../scripts/phase1_optimize_scorer.py)
   `--optimizer bootstrap --train-pool openweight`) — `BootstrapFewShot` on
   Pool C, evaluated on the held-out test set. ~30 min/dim. Output:
   `outputs/overnight_benoit/optimizer_bootstrap/{dim}/report.json`.

**Concurrency.** All 18 jobs hit vllm port 8010. vllm's dynamic batching
multiplexes the streams; per-stream latency goes up but total throughput
stays high. DSPy disk + memory cache disabled via `TT_DSPY_ENABLE_*` env
vars to avoid SQLite contention across processes. Jobs staggered 5–30s so
the engine warms up gradually instead of spiking.

**Aggregator.** [scripts/roundup_overnight.py](../scripts/roundup_overnight.py)
walks every `report.json` and emits a single side-by-side Pearson-r table
into `outputs/overnight_benoit/roundup.md` and `roundup.json`. Run after
breakfast.

**What we expect to see in the morning:**

- Scorer-only: ours_vs_expert near each dim's published Figure 1 r (single-
  shot Gemma-4-31B vs Benoit's 18-score ensemble — should be within ~0.05).
- Full pipeline: somewhat lower than scorer-only because our summarizer is
  unoptimized. The gap measures summarization-step degradation.
- Optimizer baseline: should match scorer-only roughly.
- Optimizer optimized: ideally exceeds scorer-only by a small margin —
  validates that DSPy can squeeze juice from the train pool without leakage.

If any flight's r is >0.1 below its expected value, that's a bug, not a
finding.

---

## 5. Cross-checks

1. Spot-check the numbers in §1.6 against the PDF at
   [docs/benoit_llm_manifesto_scaling.pdf](benoit_llm_manifesto_scaling.pdf)
   pages 7, 8, 11, 12, 13, 15 before citing in the C-TreePO paper.
2. Every path cited in §2's table should exist. Confirmed at plan-file time:
   `outside_data/Manifesto_Project/MPDataset_MPDS2025a.csv`,
   `src/tasks/manifesto/data_loader.py`, `src/tasks/manifesto/__init__.py`,
   `src/tasks/manifesto/rubrics.py`, `src/tasks/manifesto/pipeline.py`,
   `src/tasks/manifesto/dspy_signatures.py`.
3. §3 phase descriptions collapse to: *replicate Benoit → swap in tree →
   push where tree should win.* If any phase outgrows one sentence, it's
   doing too much.
4. MEMORY.md's C2-only-dominance finding is flagged in §2 and §3 Phase B
   as a hypothesis to test, not a transfer-by-default assumption.
