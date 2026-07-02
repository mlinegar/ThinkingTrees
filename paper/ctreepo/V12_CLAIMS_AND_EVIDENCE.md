# C-TreePO v12/v13 — Claims, Evidence, and Remaining Work

**Date:** 2026-07-02. **Deadline context:** ~36 hours to next-version submission.
**CURRENT paper version: `main_v14_polmeth.tex`** + `sections/v14_polmeth/`
+ `appendix/v14_polmeth/` (Polmeth reorganization, 2026-07-02). v14 merges the two
v13 lineages (theory files from `v13_triangle`: 03/08/10/B/E/G; manifesto files and
everything else from `v13_reconstruction`) and restructures: main text is 38 pp
(intro → sketches → framework → 4-theorem Main Results → audit → **§6 manifesto
deep dive, 14 pp, §6.1–6.9** → discussion+related → conclusion); controlled
examples (Markov interlude/FNO primer/Markov walkthrough/HLL parity) are now
Appendices G–I/K; new **Appendix B** (`AA_theory_details.tex`, full result ladder +
merge triangle + verification + Lean literature inventory) and new **Appendix R**
(`Z_treepo_software.tex`, treepo/fit/FamilyRuntime/vignette/CLI/artifact schema/
treepo-bench). All 4 remaining `\fixme`s converted to scope sentences per the
presentation rule (§3 items 3–6 remain the pending runs; their numbers slot into
the existing K/L tables when they land). New §6.4 prose: joint-training
decentralization oscillation (0.361→0.461→0.343…0.413 at leaf 8096, verified
against `combined_benoit_joint_teacher_all6_dspy_fixed_20260425_000122/
plots_by_dimension/manifesto_fg_ladder_dimension_rows.csv`) + two-track recipe
keyed on the f*-gap (well-served dims <0.23, dec ≥0.33). New §6.9 synthesis
("What the Corpus Teaches the Framework"). Builds clean: 129 pages, zero
unresolved refs/citations, zero duplicate labels, zero fixmes.
Frozen predecessors: `main_v13_reconstruction.tex` (126 pp) and
`main_v13_triangle.tex` (125 pp); before them `main_v12_external.tex` (116 pp),
itself a non-destructive copy of the dissertation baseline `main_new.tex` /
`sections/v2/`.

**Tier 0 status (all DONE in v13, 2026-07-02):** multi-dim f/g ladder integrated into
§9.5 + App H (new `tab:fg-ladder-perdim`, figures `fig:fg-ladder-perdim-live`/`-joint`
wired; stale fixme deleted; fixed-prompt fixme → scope sentence); App K fixmes rewritten
to verified state (HPO studies cited; law-weight run pending); App L open arms corrected
(treatment arm untrained; transfer eval runnable at leaves 1/8/16); companion naming
unified to "Semantic Forests companion". Remaining fixmes = exactly 4, all mapping to
Tier 1/2 runs below.

This document is the single ledger for: what the paper claims, what artifact backs each
claim, what code produced it, where it is written up, and what remains to be done.
All paths are relative to the repo root unless noted. All fixme statuses below were
**verified against disk on 2026-07-02**, not taken from memory.

---

## 1. What v12 added (relative to main_new)

| Change | File(s) |
|---|---|
| New Section 9b: oracle-grounded quasi-sentence training (labels, `treepo` vignette, results) | `sections/v12_external/09b_manifesto_qsentence.tex` |
| New Appendix K: quasi-sentence leaf-size ladder full grids | `appendix/v12_external/K_qsentence_leaf_ladder.tex` |
| New Appendix L: gold-seeded summarizer + leak diagnosis | `appendix/v12_external/L_gold_seeding.tex` |
| Abstract expanded to both manifesto supervision levels | `main_v12_external.tex` |
| Intro roadmap, §9.6, App H, conclusion re-pointed from "planned" to realized results | `01_introduction.tex`, `09_manifesto_llm.tex`, `H_benoit_replication.tex`, `14_conclusion.tex` |
| Repairs: 5 missing classical-sketches PNGs restored to `assets/sketches/figures/`; 4 dangling refs + 2 duplicate labels fixed; `LaverBudge1992` added to `paper/refs.bib` | `F_classical_parity.tex`, `G_mechanism_checks.tex`, `04_markov_interlude.tex` |
| STYLE.md prose sweep (em-dashes, "not X but Y", meta-references) applied across main text | concurrent editing session; spot-checked, no conflicts |

---

## 2. Claims ledger

### 2.1 Controlled sections (Markov, HLL) — inherited from main_new, not re-verified this session

| Claim | Written up | Evidence / assets |
|---|---|---|
| Tree matches/improves on flat FNO across supervision budgets; ~half of root labels replaceable by local labels | §6 (`06_markov_walkthrough.tex`), App G | `assets/markov/figures/markov_{simple,hard}_leaf_mass.png`, `markov_budget_split.png`; App G tables |
| HLL recovered byte-for-byte under deterministic merge; learned f+g tracks RSE floor | §7 (`07_hll_parity.tex`), App F | `hll_parity_curves.pdf`; `outputs/classical_sketches_paper_gonly_vectorized_all_cpu_20260421_190158/` (source of restored PNGs) |

These carried through from the polished dissertation baseline. No open fixmes.

### 2.2 Benoit replication, prompt-only pipeline (§9, App H)

| Claim | Number | Evidence | Code | Status |
|---|---|---|---|---|
| Single open-weight C-Tree matches frontier ensemble | macro r 0.829 (c=8K) vs 0.817 ensemble, 0.793 Gemma-3 repl.; 0.842 at c=4K | `assets/benoit/tables/benoit_comparison_pearson.tex` | Benoit replication pipeline (App H protocol) | in paper, stable |
| Chunk-invariance across 16× leaf sweep | ±0.027 macro | same table; App H ablations | same | in paper, stable |
| Compute gap ~2 orders of magnitude | ~17 vs 108 calls/manifesto | App H cost accounting | same | in paper, stable |

### 2.3 Alternating f/g prompt ladder (§9.5, App H ladder subsection)

| Claim | Number | Evidence | Code path | Status |
|---|---|---|---|---|
| Leaf-size robustness under iterative prompt updates (economic) | best-per-leaf r in [0.879, 0.886] for leaves ≥1024 tok; raw-init leaf-256 f¹g¹ r=0.909 | `assets/benoit/tables/manifesto_fg_ladder.tex`; `assets/benoit/figures/manifesto_fg_ladder_*.{pdf,png}` | `run_alternating_ladder.py` → `src/ctreepo/alternating.run_alternating_family` (centralized path) | in paper, stable |
| **Multi-dimension extension** | economic 0.897, social 0.928, immigration 0.894, EU 0.971, environment 0.779, decentralization 0.540 (best test ext-r per dim; spot-verified 2026-07-02 from `iteration_history.json`) | `outputs/manifesto_fg_alternating/scalar_dims_benoit_all6_fresh8192_dspy_20260427_015845/` (36/36 cells: 6 dims × leaves 256–8192 tok); joint variant `outputs/manifesto_fg_alternating/combined_benoit_joint_teacher_all6_dspy_fixed_20260425_000122/` with tidy CSV `plots_by_dimension/manifesto_fg_ladder_dimension_rows.csv` | `scripts/run_benoit_single_dimension_ladder_context_groups.sh` → `run_benoit_supervised_dspy_ladder.sh` → `run_alternating_ladder.py`; joint: `run_benoit_combined_joint_teacher_dspy_ladder.sh` | **DONE but NOT in paper.** Fixme at `09_manifesto_llm.tex:359` is STALE. Figures already rendered: `assets/benoit/figures/manifesto_singledim_per_dim_live.{pdf,png}` (+`_audit_gap`), `combined_perdim_leaf_invariance.pdf`, `{economic,decentralization}_singledim_leaf_invariance.pdf`, `manifesto_fg_combined_ladder_f1g0_f1g1.*`. Plot scripts: `scripts/plot_manifesto_singledim_per_dim_live.py`, `plot_manifesto_combined_ladder.py`, `plot_manifesto_single_dim_ladders.py`. Docs: `docs/manifesto_optimization_writeup.md`. Honest framing: five of six near/above split-expert; decentralization the hard case (joint-g degrades it; single-dim better). **TODO(writing): fold into §9.5 + App H, delete fixme.** |
| Matched-leaf fixed-prompt baseline | — | only single-point ablations exist, e.g. `outputs/manifesto_fg_alternating/economic_fg_baselines_20260423_165446/f0g0/report.json` (test r=0.826); no token-leaf sweep anywhere in `outputs/` | would be `run_alternating_ladder.py` eval-only at token leaves | **GENUINELY PENDING.** Fixme at `09_manifesto_llm.tex:360`. Recommendation: convert to plain scope sentence, do not spend fleet time (comparison already properly declined in text). |

### 2.4 Quasi-sentence label system (§9b.1)

| Claim | Number | Evidence | Code | Status |
|---|---|---|---|---|
| Doc-level published label = additive rollup of qsentence gold | RILE r=0.9975; domains 0.924–0.999 (n=2,157) | `outputs/qsentence_doc_label_correlation/correlation_report.json` | `scripts/check_qsentence_doc_label_correlation.py` (no model calls) | in paper (§9b, footnoted) |
| CMP-code construct ceiling vs Benoit expert means | CV R² 0.03 (decentralization) – 0.47 (immigration) | same report, `dimensions.*.ols_cv5_r2` | same script | in paper (§9b) |
| RILE formula / code sets | 13 right + 13 left codes, `total_non_header` denominator, norm (r+100)/200 | `src/tasks/manifesto/rile_codes.py:50-88`, `span_targets.py:113-162` | loaders `src/tasks/manifesto/{data_loader,span_annotations,expert_benchmarks}.py` | in paper (§9b Eq. rile) |
| Bundles: 218 W-Eur platforms, split 140/30/48, leaves ℓ∈{1,2,4,8,16}, ~437K nodes at ℓ=1 | — | `outputs/manifesto_qsentence_dspy_labeled_grid/{manifest.json,split_ids.json,leafq*/labeled_trees.jsonl}` | `scripts/build_manifesto_qsentence_dspy_labeled_grid.py` (no LLM labeling) | in paper (§9b.2) |

### 2.5 Quasi-sentence trained substrates (§9b.3, App K)

| Claim | Number | Evidence | Code | Status |
|---|---|---|---|---|
| FNO leaf ladder (RILE): peak at ℓ=8 with f→g lift | f 0.664 → g **0.718** ext-r; ladder 0.479/0.537/0.538/0.718/0.672 at ℓ=1/2/4/8/16 | `outputs/manifesto_qsentence_fno_embeddinggemma_full_leafgrid/grid_summary.md`; ℓ=1 in `..._full/grid_summary.json` | `scripts/run_manifesto_qsentence_dspy_ladder.py --family fno` (embeddinggemma-300m, 8 epochs) | in paper (Tab. qsentence-fno-ladder, App K) |
| Per-dim substrate comparison at ℓ=8: FNO wins all dense dims; sparse dims unlearnable for all | FNO g: rile 0.72, dom5 0.72, dom4 0.62, dom6 0.63; LLM g collapse (dgemma 0.12 rile) | `outputs/manifesto_qsentence_perdim_comparison/{FINAL_SUMMARY.md,per_dimension.csv,LLM_MERGE_DIAGNOSIS.md}` | `scripts/summarize_manifesto_qsentence_per_dimension.py`; runs via `run_fno_perdim_leaf8.sh` | in paper (Tab. qsentence-perdim, App K) |
| LLM f-stage = gold-label echo (int r=1.000 exactly); do not read LLM cells as reading ability | — | `outputs/manifesto_qsentence_diffusiongemma_full_leafgrid/grid_summary.md` (f-stage rows) | — | in paper (App K caveat) |
| LLM g-merge collapse cause = mean-over-dims reward (degenerate predict-the-mean optimum ≈0.90 reward) | per-dim controls: rile −0.283/−0.165/−0.184 at ℓ=2/4/8 | `LLM_MERGE_DIAGNOSIS.md`; `docs/qsentence_control_vs_test_comparison.md`; canonical controls `outputs/manifesto_qsentence_diffusiongemma_FULL218_leafgrid/dspy/leafq*/prediction_records/iter_02_post_eval.jsonl` | reward at `src/ctreepo/manifesto_qsentence_dspy_family.py:786-807` (`_score_vector_reward`, still mean-MAE) | in paper. **The remediation run is PENDING — see §3 item 3.** |
| Benoit 6-dim expert reconstruction (econ-transfer recipe) | social 0.715±0.024, econ 0.661±0.045, env 0.642±0.069, imm 0.438±0.094, dec 0.244, eu 0.232 (3 seeds, test) | `outputs/benoit_6dim_fno/FINAL_SUMMARY.md`; seed dirs `outputs/benoit_6dim_fno/<dim>/seed_*/`, `outputs/econ_seedcheck/` | econ HPO winner `outputs/hpo_econ_chunkfno/best.json` (modes 384, hidden 32, head 256, ep 12, rw 3.08) transferred; node supervision = LLM-span; grids `outputs/benoit_chunkgrid_forced_<dim>_llmspan/` | in paper (Tab. app-benoit-6dim-fno). **Per-dim-tuned re-run PENDING — see §3 item 4.** |

### 2.6 Gold seeding & related (§9b.4, App L)

| Claim | Number | Evidence | Code | Status |
|---|---|---|---|---|
| Seeded g reconstructs full rollup from 16 sampled leaf windows | all-dims 0.953 (g_direct) vs 0.966 sample baseline; RILE 0.955 vs 0.949; alignment-to-baseline 0.984; f_on_g ≈ g_direct ±0.002 | `outputs/manifesto_qsentence_sampled_supervision_full_leaf16_s16_align_20260623_0740/metrics.json` (full per-dim table verified) | `scripts/run_manifesto_qsentence_sampled_supervision.py --sample-state-source f_states` | in paper (Tab. app-gold-seeding) |
| Gold-state leak: `gold_stats` source = regressing gold on gold; fixed by `f_states` default | — | memory + flag semantics at `run_manifesto_qsentence_sampled_supervision.py:262-280,528` | same | in paper (App L narrative) |
| Exposure-bias signature: gold-children-trained merge collapses on own children | control arm int r=0.166 vs ext 0.916 | `outputs/sched_sampling_ab_leaf8_20260624_040326/control_rate0/grid_summary.md` | `scripts/run_scheduled_sampling_ab_leaf8.sh` | in paper (App L). **Treatment arm PENDING — see §3 item 5.** |

---

## 3. Open items — verified status and resolution paths

Six `\fixme` markers + two unmarked deferrals. Verified against disk 2026-07-02.

| # | Item | Location | Verified status | Resolution | Cost |
|---|---|---|---|---|---|
| 1 | Multi-dim f/g ladder "pending" | `09_manifesto_llm.tex:359` | **STALE — work done Apr 25–27** (see §2.3) | Writing only: fold results + wire existing figures, delete fixme | 0 compute |
| 2 | Matched-leaf fixed-prompt baseline | `09_manifesto_llm.tex:360`; App H "What remains" | **Genuinely pending** (only single-point f0g0 ablations exist) | Recommend: convert to scope sentence; run only if spare fleet time | ~1 day fleet if run |
| 3 | LLM merge under better objective | `K_qsentence_leaf_ladder.tex:138` | **Pending.** Within-dim Pearson reward never implemented (`_score_vector_reward` still mean-MAE). BUT local-law reward flags `--dspy-g-law-{c1,c3a,c3b}-reward-weight` wired Jun 24 (`manifesto_qsentence_dspy_family.py:842-849`, ladder flags `:1045-1059`), **never run with nonzero weights** (no hit in any `outputs/*/job.log`) | Run leaf-8 dspy g-stage with law weights on — code-free, and tests the paper's own thesis (objective must price the laws) | 1 GEPA g-stage on fleet (~hours) |
| 4 | Per-dim HPO for weak dims | `K_qsentence_leaf_ladder.tex:185` | **Half stale.** HPO studies exist (Jun 16): `outputs/hpo_{immigration,environment,eu,decentralization}_chunkfno/best.json` = 0.554 / 0.615 / 0.540 / 0.679 (~50 trials each; `outputs/perdim_hpo_chain/`). `hpo_social_chunkfno/best.json` empty (social used transfer, already best). **Missing: per-dim-tuned 3-seed TEST reconstruction** — `benoit_6dim_fno/FINAL_SUMMARY.md` still econ-transfer | Re-run 6-dim reconstruction with per-dim winners, 3 seeds/dim. No LLM needed (span labels cached in `benoit_chunkgrid_forced_<dim>_llmspan/`); FNO fits are seconds. Caution: HPO best values are val-objective; test may not reproduce (dec 0.679 val vs 0.244 transfer-test is a big gap) | hours, GPU-light, can coexist with Julia job |
| 5 | Scheduled-sampling rate-1 arm | `L_gold_seeding.tex:118` | **Pending — worse than "eval missing": treatment g-stage never TRAINED** (no `g_qsentence_dspy_iter_02.json` in either `sched_sampling_ab_leaf8_2026062{3_222439,4_040326}/sched_rate1/`; `..._leaf2_.../` sched arm never started). Control arm complete in all runs | Run sched_rate1 arm only at leaf 8 (`run_scheduled_sampling_ab_leaf8.sh` internals, `--dspy-g-scheduled-sampling-rate 1.0`), reuse saved control (`FULL218_leafgrid` prediction records) for the comparator | 1 g-training + eval on fleet (~hours) |
| 6 | Gold→dgemma transfer | `L_gold_seeding.tex:125` | **Runnable now.** Complete gemma4 gold f+g artifacts exist at leaf 8 & 16: `outputs/manifesto_parallel_llm_qsentence_20260621_075132/gemma4_fixed_leafgrid/dspy/leafq{008,016}/`; leaf 1: `outputs/manifesto_qsentence_followon_all_missing_20260623_182238/gemma4_full_leafgrid/dspy/leafq001/`. The empty-report runs (`gold_to_dgemma_20260622_*`) died because their *source* run (gemma4safe) FAILFAST-crashed (f only, no g). Script `scripts/run_manifesto_qsentence_gemma4_gold_to_dgemma.sh` supports eval-from-artifacts: env `SOURCE_RUN_ROOT=outputs/manifesto_parallel_llm_qsentence_20260621_075132 LEAF_QS=8,16` (leaf 1 via `SOURCE_GRID` pointed at the followon); artifact templates at lines 166-167; leaf 2/4 gold g artifacts do not exist | Eval-only run on dgemma fleet at leaves {1, 8, 16} | fast (~1-2 h fleet) |
| 7 | Honesty protocol "deferred to companion paper" | `10_estimation.tex:153` | Deliberate scope statement | Keep; unify naming with conclusion's "Semantic Forests companion" | 0 |
| 8 | Complexity-ladder plots "deferred to companion paper" | `G_mechanism_checks.tex:221` | Deliberate scope statement | Keep; same naming note | 0 |

**Presentation rule for the submission:** any item we decide *not* to run gets its red
`\fixme` converted to a normal scope sentence during the reconstruction pass. Red boxes
in an external-audience PDF invite questions we have already answered.

---

## 4. 36-hour run plan (priority order)

**Tier 0 — writing, zero compute, do regardless:**
1. Fold multi-dim ladder (§2.3) into §9.5 + App H; wire `manifesto_singledim_per_dim_live` / `combined_perdim_leaf_invariance` figures; delete fixme #1.
2. Rewrite App K HPO fixme to cite the existing studies (#4, first half).
3. Convert fixme #2 to a scope sentence.

**Tier 1 — cheap GPU, no LLM fleet, start immediately:**
4. Per-dim-tuned 6-dim FNO reconstruction (item #4, second half): 3 seeds × 4 dims with
   `outputs/hpo_<dim>_chunkfno/best.json` params against cached llmspan grids. Only run
   that can *raise headline numbers* before submission. Update
   `Tab. app-benoit-6dim-fno` if test confirms; keep transfer row for comparison.

**Tier 2 — requires the dgemma/Gemma-4 fleet (currently DOWN; all 4 GPUs hold Julia PID
1689275 at ~41 GB each, ~57 GB free per GPU → reduced `gpu-memory-utilization` fleet
likely coexists). In value-per-hour order:**
5. Gold→dgemma transfer eval, leaves {1, 8, 16} (item #6) — eval-only, fills App L placeholder.
6. LLM g-stage with law-reward weights on, leaf 8 (item #3) — the paper's thesis on the
   LLM substrate; directly answers the collapse table.
7. Sched-sampling rate-1 g-stage, leaf 8 (item #5) — completes the A/B.

**Explicitly deferred past submission:** fixed-prompt token-leaf sweep (#2);
within-dim-Pearson reward implementation (subsumed by the law-weight run for the
paper's narrative).

**Operational notes:** launch long runs via `scripts/long_job.py launch` (see
`AGENTS.md`); dgemma runs need `TT_DSPY_DROP_RESPONSE_FORMAT=1`; use the 4-GPU fleet
with round-robin routing, not affinity; `skip_lm_input_budget_check=True` for audits
(GIL guard starves the fleet).

---

## 5. After the runs: the larger reconstruction pass

Queued decisions (not blocking the runs):
- Whether §9 (prompt-only Benoit) and §9b (oracle-grounded qsentence) stay separate or
  merge into one manifesto mega-section with a supervision-level arc.
- Whether the multi-dim ladder promotes to a headline figure (it is currently strong
  enough: five of six dims at/above split-expert under iterative prompt training).
- `\fixme` → scope-sentence conversion for anything not resolved by the runs.
- Naming unification: "companion paper" → "Semantic Forests companion".
- Final STYLE.md sweep + rebuild after all table/figure updates.

## 6. Provenance quick reference

- **Corpora:** `data/raw/manifesto_project_full/manifesto_corpus_df.csv` (per-qsentence CMP codes, 541 MB); `manifesto_maindataset.csv` (doc-level MPDS); `data/raw/manifesto_corpus_benoit/` (full texts + Benoit `.rda` replication archive, loaded via `src/tasks/manifesto/expert_benchmarks.py`).
- **Label code:** `src/tasks/manifesto/{rile_codes,span_targets,span_annotations,dimensions,expert_scale}.py`.
- **Harness:** `scripts/run_manifesto_qsentence_dspy_ladder.py` (`--family {dspy,fno}`); bundle-config launcher `scripts/run_manifesto_qsentence_bundle.py`; comparators `scripts/compare_manifesto_qsentence_substrates.py`, `summarize_manifesto_qsentence_per_dimension.py`, `compare_qsentence_per_dim_pearson.py`.
- **Package:** official `~/treepo` (`treepo.fit`, `treepo.methods.contracts.FamilyRuntime`); TT bridge `src/ctreepo/treepo_bridge/fno.py`; vignette example `~/treepo/examples/methods/run_manifesto_end_to_end.py`.
