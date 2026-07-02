# `main_minimal` Manifesto-First Writing Blueprint

This blueprint is the implementation map for the minimal C-TreePO manuscript.
It is deliberately paragraph-level: each unit below corresponds to one main
paragraph, theorem block, figure/table callout, or appendix subsection. The
new draft should read from one motivating example outward: a single Manifesto
economic-policy dimension, an assumed task-relative state, local laws that
preserve that state, then guarantees and evidence.

## Global Commitments

- Running example: Manifesto economic policy, public services versus taxation.
- Conceptual state: `S_{\mathrm{econ}}(x)`, the economic evidence a trusted
  rubric oracle needs for span `x`.
- Main claim: C-TreePO is useful because local state preservation lets the
  researcher work at smaller granularity: paragraphs, sections, or chapters
  instead of whole manifestos.
- Discipline: present two claim tiers. The Manifesto/Benoit headline is a
  valid root-observed corpus evaluation because the sampled documents have
  full-document expert targets; it is not also a node-level C1/C2/C3 certificate.
  The quasi-sentence path is the planned oracle-grounded local-law route.
- Structure: keep algebra, neural-operator detail, Markov mechanisms,
  classical-sketch protocol grids, HLL parity, and extended related work in
  appendices unless one sentence is needed to orient the main text.
- Validation analogies: use only two main validation anchors. Markov is a
  play/Hamlet-style emotional-register arc for ordered boundary state. HLL is
  the primary classical mergeable-sketch example for unordered noisy state.
  Count-Min and Frequent Items are appendix-side query-taxonomy notes for
  frequency and heavy-hitter tasks, not main examples. Keep exact synthetic and
  sketch details in appendices.

## Editing Contract

Every main-text edit should preserve these invariants.

- The first screen of the paper should be about a measurement problem, not a
  formalism. Introduce a manifesto, an economic score, and the question of
  whether smaller units can carry the evidence needed for that score.
- The state is always task-relative. Do not write as though the summary must
  preserve all semantic content, all linguistic style, or all possible future
  tasks.
- The minimal paper should expose only the public theorem stack:
  preservation, schedule invariance, preference alignment, approximate gap, and
  finite-sample audit certificate. Long proofs and alternate formulations stay
  in Appendix C and the Lean map.
- The Manifesto/Benoit result is an empirical root-score result with observed
  full-document labels. It should not be described as "only prompt evidence" or
  as missing a prerequisite audit.
- The node-level audit path is a stronger claim: it certifies local laws inside
  realized trees and supports label-budget substitution. It should not be used
  to weaken the root-observed benchmark.
- Applications instantiate theorem premises. Avoid wording that says C-TreePO
  proves manifestos, legal reports, clinical notes, or policy documents are
  compressible without an application-specific state and oracle assumption.
- The algebraic material should support the story: states compose, scalar
  values usually do not. If a paragraph feels like it is proving a generic
  theorem about text, move it to the appendix or rewrite it as an assumption.

## Claim Tiers

Use this table to keep the paper's claims separated.

| Tier | Observation Unit | What Is Observed | Claim It Supports | Where It Lives |
| --- | --- | --- | --- | --- |
| Root-observed corpus evaluation | sampled documents | full-document expert/rubric targets | root prediction agrees with external labels on the sampled corpus | Manifesto/Benoit result, Appendix G |
| Node-level local-law certificate | sampled tree units | leaf, summary, and merge preservation judgments with logged propensities | C1/C2/C3 distortion is estimated for a realized tree | audit section, Appendix H |
| Local supervision / label-budget substitution | paragraphs, sections, merge spans | application-aligned local oracle labels | smaller units can train or audit the same preservation property | audit section, Markov validation, future Manifesto quasi-sentence path |
| Formal theorem stack | Lean objects and theorem hypotheses | assumptions, not empirical labels | conditional preservation, preference alignment, and finite-sample certificates | Appendix C, Lean map |
| Application narrative | domain examples | task descriptions and plausibility arguments | shows how a domain could instantiate the stack | applications/scope section |

Writing rules:

- For Tier 1, say "root-observed", "full-document labels", "external
  validation", "document-sampling unit", or "corpus-level evaluation".
- For Tier 2, say "node-level", "tree-unit", "local-law audit", "logged
  propensities", or "realized-tree certificate".
- For Tier 3, say "local labels" only when the sentence also identifies their
  alignment target: the same economic oracle/readout as the root task.
- Do not collapse Tier 1 into Tier 2. The benchmark can be complete as a
  root-observed evaluation while not yet being a local-law certificate.
- Do not promote Tier 5 examples to theorem statements. Phrase them as
  "backed by the theorem under assumptions..." or "an instance when...".

## Public Formal Anchors

These are the names the minimal paper should cite when it wants a clean Lean
surface.

- Preservation and schedules:
  - Main statement: C1 + C3 + context compatibility imply root preservation.
  - Repeated rounds: add C2/recompression stability.
  - Public wording: "structural preservation" and "schedule invariance".
  - Lean map source: `lean3/docs/PAPER_TO_LEAN_MAP.md`, preservation rows.
- Preference alignment:
  - Public surface: `PaperPreferenceStack`.
  - Exact case: residual `0` gives the same full and summary argmin sets.
  - Approximate case: exact summary minimizers are `2 * residual`
    epsilon-optimal for the full objective.
  - Public anchors: `paper_preference_stack_same_argmin` and
    `paper_preference_stack_summary_argmin_full_epsilon`.
- Error/certificate stack:
  - Public surface: `PaperErrorStack`.
  - Formula in prose: `C_meth * hatDelta_R + B_cal + B_est + B_clip`.
  - Public anchor: `paper_error_stack_high_prob`.
  - Do not revert to a bare `L * hatDelta_R` formula unless the method constant
    has explicitly been specialized to `L`.
- Applications:
  - DPO/GRPO are premise packages that instantiate the preference stack.
  - Manifesto/RILE is an empirical application that supplies root labels now
    and can supply local-law labels through quasi-sentence aggregation.
  - Neural operators certify deterministic realizers; stochastic summarizers are
    covered by the broader PMF theorem stack.

## Empirical Number Source Map

Keep all numeric claims tied to generated tables or source appendices.

- Headline Manifesto table:
  - Include file:
    `assets/benoit/tables/benoit_comparison_pearson.tex`.
  - Source markdown:
    `assets/benoit/tables/benoit_comparison_pearson.md`.
  - Required numbers: `8K` character tree macro `0.829`; proprietary ensemble
    `0.817`; matched open-weight `0.793`; economic `8K` row `0.939`;
    split-expert economic reference `0.880`.
- Economic prompt ladder:
  - Include/source file: `assets/benoit/tables/manifesto_fg_ladder.tex`.
  - Required numbers: `0.885`, `0.886`, `0.886`, `0.879` for
    `1024`--`8192` token leaves; drops to `0.861` at `512` and `0.830` at
    `256`.
- Chunk robustness:
  - Source files: `benoit_comparison_pearson.*` and
    `chunk_sweep_per_dim.md`.
  - Required distinction: character chunk sweep and token prompt ladder are
    different axes. Do not mix their leaf-size units.
- Manifesto quasi-sentence path:
  - Current blueprint numbers: `2,157` platforms and about `2.27M` coded spans.
  - These support the planned local-law route; they are not the source of the
    current root-observed Benoit correlations.
- Classical sketches:
  - HLL is the main classical mergeable-state example in the minimal paper.
    It demonstrates the key correction that scalar values need not compose
    even when bounded sketch states do.
  - RSE for `p=14` is about `0.81%`; write "under 1%" in main prose and keep
    HLL estimator detail in Appendix F.
  - Main source figure/table:
    `assets/hll/figures/hll_merge_learning_memory_median.pdf`, with
    `assets/hll/figures/hll_parity_curves.png` retained for the full grid.
  - Count-Min and Frequent Items are appendix-side related cases only:
    Count-Min for point frequency, Frequent Items for heavy hitters. They are
    associative and commutative but not idempotent because repeated evidence
    must count.
  - Current generated frequency rows use IID Zipf tokens over a 4096-token
    universe, with 128--512 tokens per document. Leaves are artificial
    partitions of an exchangeable stream, so these rows should be presented
    only as state-composition/parity checks if mentioned at all.
- Markov:
  - Main text uses emotional-register language. Appendix E may use the neutral
    synthetic color/register terminology.

## Wording Guardrails

Prefer these formulations:

- "The Manifesto/Benoit benchmark is root-observed: each sampled document has a
  full-document expert target, and the tree root is evaluated directly against
  that target."
- "The node-level local-law audit is stronger and different: it samples tree
  units with logged propensities to estimate realized preservation distortion."
- "Local labels help only when they measure the same oracle-preservation
  property as the root task."
- "The theorem is conditional on the state, oracle, and local-law assumptions."
- "Neural-operator approximation supplies a deterministic realizer route; the
  stochastic summarizer theorem stack remains the general formal surface."

Avoid these formulations:

- "The Manifesto result is incomplete because it lacks a node-level audit."
- "Prompt-only evidence" as a label for the root-observed corpus result.
- "The summary preserves meaning" without naming the task oracle.
- "Child scores merge into the root score" for Manifesto or any scalar-valued
  non-homomorphic readout.
- "DPO/GRPO are proved for manifesto data" rather than "DPO/GRPO objectives
  align under the application bundle assumptions."

## Running-Example Glossary

- `S_{\mathrm{econ}}(x)`: conceptual economic-policy state of span `x`.
  Contains rubric-relevant evidence, not style or all semantics.
- `r_{\mathrm{econ}}`: readout from state to seven-point economic scale.
- `\fstar_{\mathrm{econ}}(x)`: trusted economic oracle,
  `r_{\mathrm{econ}}(S_{\mathrm{econ}}(x))`.
- `g(x)`: produced summary/state surrogate for span `x`.
- `f_{\mathrm{econ}}(g(x))`: practical scorer applied to a produced summary.
- Leaf audit unit: compare `g(b)` to raw span `b` under the economic oracle.
- Merge audit unit: compare `g(g(u) concat g(v))` to the raw union span.
- C2 audit unit: re-summarize an already stored summary and check drift.
- Global label: full-manifesto/root economic judgment.
- Local label: paragraph/section/merge-level economic preservation judgment.
- Current Manifesto/Benoit setting: sampled documents have full-document labels
  observed at the root; node-level observations belong to the stronger audit
  and quasi-sentence supervision path.

## Main Figures, Tables, and Equations

- Main figure: `fig:min-plain-tree`, asset `01_base.pdf`.
  Purpose: show the compression tree as the object being audited.
- Main table: `tab:min-benoit-headline`, source
  `assets/benoit/tables/benoit_comparison_pearson.tex`.
  Purpose: headline Manifesto comparison plus economic running-example numbers.
- Alignment table: `tab:min-state-alignment`.
  Purpose: show what Markov, HLL, and Manifesto each teach about state.
- Main equations:
  - `\fstar_{\mathrm{econ}}(x)=r_{\mathrm{econ}}(S_{\mathrm{econ}}(x))`.
  - C1/C2/C3 equations under `\sim_{\mathrm{econ}}`.
  - Preservation root equation from Theorem 1.
  - `|G_{\mathrm{meth}}| <= L \Delta_R`.
  - Audit certificate:
    `|G| <= L \hat{\Delta}_R + B_cal + B_est + B_clip`.
  - Markov boundary correction:
    `c_L+c_R+1{L.last != R.first}`.
  - Ordered homomorphism:
    `h(u concat v)=h(u) odot h(v)`.
- Main-deferred objects:
  - `fig:local-laws-full`: appendix or later expansion only.
  - Markov result figures: Appendix E.
  - HLL parity table/curves: Appendix F.
  - Count-Min/Frequent Items frequency material: Appendix F taxonomy/provenance
    note only; do not promote a frequency figure into the main text.
  - Manifesto prompt-ladder heatmaps and per-cell table: Appendix G.
  - Neural-operator notation/projection diagrams: Appendix D.
  - Audit reporting template: Appendix H.

## Main Text Blueprint

### 1. Introduction

#### 1.1 Opening Paragraph: One Measurement Problem

- Purpose: make a first-time political scientist understand the target before
  any algebra appears.
- Main points:
  1. The object is a party manifesto and one economic-policy score.
  2. Evidence for that score is spread across the document.
  3. The practical question is whether we can work below whole-document
     granularity.
- Source material: social-science measurement setup; Manifesto/Benoit
  benchmark description.
- Figure/equation: none.
- Treatment: main, compressed.

#### 1.2 C-TreePO as State Composition

- Purpose: introduce the assumed state in plain language.
- Main points:
  1. `S_{\mathrm{econ}}(x)` is conceptual, not observed.
  2. Summaries are surrogates for that state.
  3. Valid compression means preserving oracle-visible state through the tree.
- Source material: framework local-law sections; sufficient-statistic language.
- Equation: optional inline mention only; full equation appears in Section 2.
- Treatment: main.

#### 1.3 Granularity Reduction

- Purpose: state the reader-facing payoff.
- Main points:
  1. The reported Manifesto/Benoit evidence uses root-level full-document labels.
  2. Local preservation makes paragraph/section/chapter labels useful when the
     claim is node-level certification or label-budget substitution.
  3. Failed local checks become measured distortion.
- Source material: audit and discussion sections.
- Figure/equation: none.
- Treatment: main.

#### 1.4 Roadmap

- Purpose: orient without repeating a full paper outline.
- Main points:
  1. Manifesto state first.
  2. Local laws and guarantees next.
  3. Manifesto results lead; Markov/HLL validate; appendices carry
     detail.
- Source material: current `main_minimal` order.
- Treatment: main.

### 2. Running Example: Economic-Policy State

#### 2.1 Define the State

- Purpose: make the latent-state assumption explicit.
- Main points:
  1. `S_{\mathrm{econ}}(x)` contains fiscal/welfare/market-intervention
     evidence.
  2. It may discard style and slogans.
  3. It fails if discarded evidence changes the economic score.
- Source material: Manifesto dimension description and social-science
  measurement section.
- Equation: none yet.
- Treatment: main.

#### 2.2 Define the Oracle Readout

- Purpose: connect state to the seven-point benchmark.
- Main points:
  1. `\fstar_{\mathrm{econ}}(x)=r_{\mathrm{econ}}(S_{\mathrm{econ}}(x))`.
  2. Prompt-only experiments approximate the readout with a rubric scorer.
  3. The reported Manifesto setting observes the expert/rubric target at the
     full-document root.
- Source material: Benoit target discussion and Manifesto pipeline.
- Equation: state/readout equation.
- Treatment: main.

#### 2.3 Span Oracles from Quasi-Sentences

- Purpose: explain how local oracle values can exist in this application.
- Main points:
  1. Quasi-sentence codings are available for `2,157` platforms and about
     `2.27M` coded spans.
  2. Aggregating codes inside a leaf or internal span gives span targets.
  3. This supports C1/C3 supervision for the stronger local-law certificate, but
     those results are separate from the root-observed headline.
- Source material: Manifesto oracle-grounded supervision subsection and
  Benoit appendix.
- Treatment: main, careful.

#### 2.4 Summary as Surrogate State

- Purpose: transition from social science to tree laws.
- Main points:
  1. Leaf summaries should preserve span state.
  2. Merge summaries should preserve union state.
  3. Re-summarization should not shift readout.
- Source material: framework C1/C2/C3.
- Treatment: main.

### 3. Compression Trees and Local Laws

#### 3.1 Objects

- Purpose: introduce notation only after the running example is clear.
- Main points:
  1. Strings, concatenation, oracle, metric.
  2. Summarizer `g` and economic readout `f_{\mathrm{econ}}`.
  3. Closed-form settings use `f=fstar`; Manifesto uses a scorer.
- Source material: framework definitions.
- Treatment: main.

#### 3.2 Tree Construction

- Purpose: define the executable object.
- Main points:
  1. Fixed contiguous leaves.
  2. Leaves summarize raw spans; internal nodes merge child summaries.
  3. Observations may be root-level labels or node-level local-law labels; raw
     spans are retained conceptually for the latter.
- Figure: include `fig:min-plain-tree`.
- Treatment: main.

#### 3.3 Oracle Equivalence

- Purpose: define what "same enough" means.
- Main points:
  1. `u \sim_{\mathrm{econ}} v` iff economic oracle distance is zero.
  2. All preservation is modulo the target, not literal string equality.
  3. This prevents generic semantic-similarity overclaims.
- Source material: framework equivalence section.
- Treatment: main.

#### 3.4 C1/C2/C3 Equations

- Purpose: make the local laws exact.
- Main points:
  1. C1: leaf summary preserves economic evidence.
  2. C2: stored summary is stable under re-summarization.
  3. C3: split-then-merge preserves raw union span.
- Equation: print C1/C2/C3.
- Figure: do not include `fig:local-laws-full` unless expanding main.
- Treatment: main.

#### 3.5 Context Compatibility

- Purpose: state the key induction side condition.
- Main points:
  1. Local equivalence must survive insertion into the same context.
  2. This is plausible for content rubrics, not universal.
  3. Discourse-sensitive tasks should be audited or excluded.
- Source material: framework assumption.
- Treatment: main.

### 4. What the Guarantees Say

#### 4.1 Theorem Bundle Setup

- Purpose: avoid a wall of unexplained theorems.
- Main points:
  1. Preservation is structural.
  2. Optimization equivalence follows from oracle-measurability.
  3. Audits give approximate certificates.
- Treatment: main.

#### 4.2 Inductive Preservation Theorem

- Purpose: preserve main result.
- Main points:
  1. C1+C3+context compatibility imply root preservation.
  2. Adding C2 gives repeated-round preservation.
  3. Expectation covers stochastic summarizers; node-level certificates
     complement root-observed corpus evaluation.
- Equation: root preservation equation.
- Treatment: main theorem statement; proof in Appendix C.

#### 4.3 Schedule Invariance Corollary

- Purpose: connect associativity/local laws to chunk schedule.
- Main points:
  1. Valid schedules agree in oracle value.
  2. This is not byte/string equality.
  3. It underwrites leaf-size robustness diagnostics.
- Treatment: main short corollary.

#### 4.4 Preference Equivalence

- Purpose: keep DPO/GRPO result without application sprawl.
- Main points:
  1. DPO/GRPO application bundles supply oracle-measurability and generator
     indexing premises.
  2. `PaperPreferenceStack` residual `0` gives identical summary/full argmins.
  3. Residual `epsilon` gives full-objective `2 * epsilon` optimality for exact
     summary minimizers.
- Treatment: main theorem statement; proof in Appendix C.

#### 4.5 Approximate Gap and Audit Certificate

- Purpose: turn violations into a reported bound.
- Main points:
  1. `|G_meth| <= C_meth Delta_R`.
  2. Audited estimate plus calibration/estimation/clipping gives the deployed
     `PaperErrorStack` certificate.
  3. This is the finite-sample version of "compression risk."
- Equation: both gap equations.
- Treatment: main theorem statements.

#### 4.6 Lean Crosswalk Note

- Purpose: keep formalization visible but not intrusive.
- Main points:
  1. Appendix C maps C1/C2/C3 to Lean L1/L3/L2.
  2. The state existence is an application assumption, not a theorem about
     political text.
  3. Lean proves conditional preservation, `PaperPreferenceStack`, and
     `PaperErrorStack` results.
- Treatment: one paragraph.

### 5. Manifesto Results

#### 5.1 Pipeline Paragraph

- Purpose: connect the running state to the actual experiment.
- Main points:
  1. Character chunks, dimension-specific rubric summaries, recursive merges.
  2. Root scored on seven-point economic/rubric scale.
  3. Correlation is against full-document Benoit expert-survey targets observed
     for the sampled corpus.
- Source material: Manifesto pipeline subsection.
- Treatment: main.

#### 5.2 Headline Six-Dimension Table

- Purpose: preserve the main empirical benchmark.
- Main points:
  1. `8K` per-dim tree macro `0.829`.
  2. Proprietary ensemble `0.817`; matched open-weight `0.793`.
  3. Economic row at `8K` is `0.939`; split-expert economic reference is
     `0.880`.
- Table: `tab:min-benoit-headline`.
- Treatment: main table.

#### 5.3 Granularity / Leaf-Size Robustness

- Purpose: make the result about local structure, not only correlation.
- Main points:
  1. Macro spread is `0.027` across `4K`--`64K` character leaves.
  2. Economic remains high across the same sweep.
  3. Stable root scores as leaves shrink make local auditing feasible.
- Treatment: main.

#### 5.4 Economic Prompt-Ladder Plateau

- Purpose: add the single-dimension story the user wants centered.
- Main points:
  1. Best economic external Pearson is `0.885`, `0.886`, `0.886`, `0.879`
     for `1024`--`8192` token leaves.
  2. The band straddles the `0.880` split-expert reference.
  3. Below `1024` tokens, the drop to `0.861` and `0.830` marks the
     small-leaf failure mode.
- Source material: prompt-ladder subsection and Benoit appendix.
- Treatment: main, but caveat token axis versus character axis.

#### 5.5 Claim Boundary

- Purpose: avoid overclaiming.
- Main points:
  1. Prompt-only headline is complete for root-level external validation.
  2. Node-level C1/C2/C3 auditing is a stronger certificate tier, not a
     prerequisite for the root-observed benchmark claim.
  3. Quasi-sentence local-law supervision is the separate certificate path.
- Treatment: main.

### 6. Validation Anchors

#### 6.1 Markov Mechanism

- Purpose: show why local state must carry boundary information.
- Main points:
  1. Use a play analogy: passages have hidden emotional registers such as
     calm, threat, grief, and resolve.
  2. The oracle counts register shifts across adjacent passages.
  3. Sufficient state is `(internal shifts, first register, last register)`.
  4. A shift at an act/scene boundary is invisible if either child drops its
     boundary register.
  5. This controlled setting observes node-level labels to test whether local
     labels can substitute for some root labels when the state is right.
- Equation: boundary correction.
- Treatment: main short; details Appendix E.

#### 6.2 HLL Cardinality Backstop

- Purpose: use HLL as the main classical mergeable-state example and make
  estimator noise central.
- Main points:
  1. HLL answers narrow distinct-presence/breadth queries, not line counts,
     top speakers, or narrative importance. In the play analogy, the
     load-bearing query is distinct character co-appearance pairs across
     scenes, not the tiny distinct-cast count alone.
  2. The scalar value does not compose: `|A|` and `|B|` do not determine
     `|A union B|`.
  3. HLL state is registers; each co-appearance pair is one stream item and
     merge is pointwise max.
  4. RSE for `p=14` is about `0.81%`, so say under 1%.
  5. Valid compression preserves noisy sketch/readout behavior rather than
     removing estimator noise.
  6. Count-Min/Frequent Items appear only as appendix-side frequency-task
     relatives.
- Treatment: main short; details Appendix F.

#### 6.3 Alignment Table and Bridge Back to Manifesto

- Purpose: prevent examples from feeling disconnected.
- Main points:
  1. Markov/emotion arc has known ordered boundary state.
  2. HLL/co-appearance breadth has known unordered noisy cardinality state.
  3. Manifesto economics has ordered semantic evidence plus measurement noise.
  4. The table should name state, merge target, and failure mode for each row.
- Treatment: main.

### 7. Algebraic and Representation Backstop

#### 7.1 State Versus Value

- Purpose: retain the Section 2.2 correction without leading with algebra.
- Main points:
  1. States compose; scalar values often do not.
  2. Distinct count warning.
  3. Manifesto child scores should not be expected to merge into root score.
- Treatment: main short, Appendix B full.

#### 7.2 Ordered Homomorphism

- Purpose: cite the ordered generalization.
- Main points:
  1. `h(u concat v)=h(u) odot h(v)`.
  2. Associative is required; commutative is not.
  3. Ordered text behaves like Markov, not HLL, unless the task is symmetric.
- Citation: Gibbons 1996.
- Treatment: main short.

#### 7.3 Mergeable Reduction Proposition

- Purpose: preserve the formal bridge.
- Main points:
  1. Strict oracle-value homomorphism is special.
  2. Classical state-level mergeability is broader.
  3. C-TreePO reduces to classical sketches when `g` serializes the state.
- Treatment: main proposition; proof Appendix B/C.

#### 7.4 Learned Representation Note

- Purpose: keep FNO/neural-operator story available without distracting.
- Main points:
  1. Representation proposes a state.
  2. Local laws define what it must preserve.
  3. Audit estimates realized violations.
- Treatment: main one paragraph; details Appendix D.

### 8. Audit and Label Budget

#### 8.1 Finite Population of Local Checks

- Purpose: operationalize the certificate.
- Main points:
  1. Root-observed corpus evaluation samples documents and observes root targets.
  2. Local-law certification samples leaves, summaries, and merge nodes with
     logged propensities.
  3. HT-style estimation recovers realized-tree distortion.
- Treatment: main.

#### 8.2 Paragraph/Section Labels

- Purpose: directly state the granularity benefit.
- Main points:
  1. Node-level local labels ask economic preservation questions.
  2. They are useful for certificates and label-budget substitution when aligned
     with the root oracle.
  3. They do not replace calibration to the full-document expert target.
- Treatment: main.

#### 8.3 Certificate Decomposition

- Purpose: explain terms.
- Main points:
  1. Empirical distortion term.
  2. Judge calibration term.
  3. Sampling and clipping terms.
- Equation: audit bound.
- Treatment: main.

### 9. Applications and Scope

#### 9.1 Application Class

- Purpose: generalize cautiously.
- Main points:
  1. Long, sectioned documents with rubric targets.
  2. Some datasets observe root labels; others require node-level audit labels.
  3. Examples: manifestos, legal, clinical, policy reports.
- Treatment: main.

#### 9.2 Surrogate-Label Risk

- Purpose: connect social-science caution.
- Main points:
  1. Correlation alone is not enough for downstream inference.
  2. C-TreePO separates compression error from judge calibration error.
  3. Cite Egami et al.
- Treatment: main.

#### 9.3 Neighboring Methods

- Purpose: shrink related work.
- Main points:
  1. Long context reduces truncation but not certification.
  2. RAG retrieves but does not certify omitted evidence.
  3. Tree compression lacks task-oracle audit.
- Treatment: main short; Appendix I full.

#### 9.4 Failure Modes

- Purpose: make assumptions legible.
- Main points:
  1. C1 fails when leaves miss evidence.
  2. C3 fails when child summaries drop cross-span tension.
  3. Oracle-measurability fails when annotators use style/tone outside target.
- Treatment: main.

### 10. Conclusion

#### 10.1 Restate Minimal Claim

- Purpose: close cleanly.
- Main points:
  1. Compression should preserve task-relative state.
  2. Manifesto economic state is the guiding root-observed example.
  3. Local laws make the stronger node-level certificate auditable.
- Treatment: main.

#### 10.2 Granularity Close

- Purpose: end on why this matters.
- Main points:
  1. Valid local structure moves work from whole documents to smaller units.
  2. Approximate validity gives a measured gap.
  3. Markov/HLL/Manifesto mark ordered known-state, unordered noisy sketch
     state, and learned semantic-state cases.
- Treatment: main.

## Appendix Blueprint

Appendices should be modular. A reader should be able to skip any one appendix
without losing the main Manifesto story, while a formal/proof reader should be
able to find every assumption and Lean-facing statement without hunting through
empirical prose.

### Appendix A: Notation and Assumption Bundles

- A.1 Manifesto running example: define economic state/readout again for
  appendix readers.
  - Include the root-observed versus node-level distinction.
  - State that `S_{\mathrm{econ}}` is assumed, not learned or proved.
- A.2 Tree objects: strings, leaves, summaries, raw spans, readout.
  - Keep notation compatible with main text.
  - Say raw spans are conceptually available for audit comparisons even when the
    deployed pipeline passes stored states.
- A.3 Bundles: Preservation, Recompression, Optimization, Certificate.
  - Each bundle should correspond to a public theorem stack.
  - Do not introduce a new assumption name unless it is used later.
- A.4 Granularity vocabulary: paragraph/section/chapter/book as partitions.
  - Distinguish document-sampling units from tree-audit units.

### Appendix B: Algebraic Background

- B.1 State-level mergeability: encode/merge/query discipline and citations.
  - Use this to explain why state mergeability is broader than value
    homomorphism.
- B.2 Ordered text: Gibbons list homomorphism and noncommutative merge.
  - Emphasize order-sensitive text tasks.
- B.3 State versus value: distinct-count warning, HLL state explanation.
  - Keep this as the canonical place for "do not merge scalar child scores".
- B.4 Reduction proof sketch: strict oracle homomorphism and state-level sketch
  reduction.
  - Make clear this is a sufficient special case, not the general Manifesto
    setting.

### Appendix C: Proofs and Formalization

- C.1 Inductive preservation proof.
  - Match the public theorem wording in Section 4.
  - Note stochastic summarizers are modeled through PMFs; deterministic
    realizers are a subcase.
- C.2 Schedule invariance proof.
  - State equality is oracle equality, not literal summary equality.
- C.3 Preference equivalence and gap proof.
  - Cite `PaperPreferenceStack` first, then method bundles.
- C.4 C2 independence counterexample.
  - Keep it as a warning about repeated recompression.
- C.5 Lean crosswalk: C1/C2/C3 to L1/L3/L2; `PaperPreferenceStack`,
  `PaperErrorStack`, and application-bundle names.
  - Ensure names match `lean3/docs/PAPER_TO_LEAN_MAP.md`.

### Appendix D: Neural-Operator Realization

- D.1 Why neural operators are appendix-level in the minimal draft.
  - They are realization/approximation machinery, not the first statement of
    the preservation theorem.
- D.2 Realized-call approximation.
  - Phrase as deterministic neural-operator realizer certification.
- D.3 Projection objective.
  - Tie projection error to the public error stack only through stated
    envelopes.
- D.4 Overlap between classical sketches and learned operators.
  - Avoid claiming a randomized neural-operator theorem unless one is added.

### Appendix E: Markov Details

- E.1 DGP, noting that the main text uses emotional-register language while
  the formal benchmark uses neutral color/register labels.
- E.2 Sufficient state and ordered merge.
  - Include `(internal shifts, first register, last register)`.
- E.3 Relation to the play analogy.
  - Keep the analogy in prose; keep exact data-generating process here.
- E.4 Empirical role of local supervision.
  - Say Markov deliberately observes node-level labels to test the
    label-budget/certificate path.
- E.5 Failure mode: dropped boundary state.
  - This is the canonical example of C3 failure.

### Appendix F: HLL and Classical Sketch Details

- F.1 HLL as mergeable state.
  - Explain the state-versus-value distinction: distinct-count values do not
    compose, but HLL register states do.
  - Include the Hamlet co-appearance-pair query, register update,
    pointwise-max merge, estimator, and RSE.
  - Use `assets/hll/figures/hll_merge_learning_memory_median.pdf`; mention
    `hll_parity_curves.png` as the full grid asset.
- F.2 Implementation parity.
  - Native register reductions should be byte-identical.
  - DataSketches may serialize equivalent sparse/dense sketches differently, so
    estimate-level agreement inside the HLL floor is the right invariant.
- F.3 Learned variants and broader sketch family.
  - State learned sketches are evidence for representation learning, not proof
    that arbitrary learned summaries are valid.
  - Keep Count-Min/Frequent Items as related frequency-task cases only.
- F.4 Query taxonomy.
  - HLL for cardinality/distinct presence.
  - Count-Min for point frequency; Frequent Items for heavy hitters.
  - Explicitly say line count is not narrative importance: "how many lines does
    Hamlet speak?" and "which characters account for most lines?" are sketch
    queries; "which characters matter to the play's meaning?" requires a
    task-specific oracle.

### Appendix G: Manifesto Details

- G.1 Corpus and target.
  - State document-level target construction and sampled-corpus evaluation.
- G.2 Root-observed pipeline.
  - Prefer "root-observed pipeline" over prompting-mechanics shorthand unless
    explicitly referring to prompting mechanics.
- G.3 Main numbers and chunk sweep.
  - Cite generated table names for each number.
- G.4 Economic prompt ladder.
  - Keep token leaves separate from character leaves.
- G.5 Oracle-grounded quasi-sentence path.
  - Present as the node-level local-law route, not as a missing prerequisite for
    G.2--G.4.

### Appendix H: Audit Details

- H.1 Document-sampling units versus tree-audit units and propensities.
  - This is the definitive distinction for root labels versus node labels.
- H.2 HT estimator.
  - Logged marginal propensities and positivity are required.
- H.3 Calibration and local labels.
  - Local labels must be aligned to the same oracle.
- H.4 `PaperErrorStack` reporting template.
  - Include `C_meth * hatDelta_R + B_cal + B_est + B_clip`.

### Appendix I: Related Work and Scope

- I.1 Mergeable summaries and data systems.
- I.2 Sufficient statistics.
- I.3 Social-science measurement and surrogate labels.
- I.4 Long-context/RAG/tree-compression neighbors.
- I.5 Scope boundary and expected failures.

## Revision Checklist

Run this checklist after any nontrivial edit to the minimal manuscript.

- Root/node language:
  - Search for stale wording that implies the root-observed benchmark is
    incomplete because a node-level audit is absent.
  - Search for prompting-mechanics shorthand that obscures the full-document-label
    evaluation design.
  - Search for global-label phrases that should instead say root or
    full-document labels.
  - Search positive anchors:
    `root-observed`, `full-document expert target`, `tree-unit`,
    `logged propensities`, and `local-law certificate`.
- Certificate language:
  - Search for `L\\hat{\\Delta}_R` and check whether it should be
    `C_meth * hatDelta_R + B_cal + B_est + B_clip`.
  - Confirm `PaperErrorStack` appears in Appendix C/H-facing text if the
    certificate is discussed.
- Preference language:
  - Confirm residual wording cites `PaperPreferenceStack`.
  - Confirm DPO/GRPO are described as application bundles or premise packages,
    not as manifesto-specific theorems.
- Empirical numbers:
  - Check all Manifesto numbers against generated tables under
    `assets/benoit/tables/`.
  - Keep character chunk sizes and token leaf sizes separate.
- Shakespeare/sketch framing:
  - Confirm HLL is the primary main-text classical sketch example.
  - Confirm Count-Min/Frequent Items appear only in Appendix F or another
    explicit frequency-task aside.
  - Confirm frequency sketches are never called idempotent.
  - Confirm any cited HLL figure exists under `assets/hll/figures/`.
  - For implementation provenance, the relevant smoke test is
    `pytest treepo/tests/sketches/test_broad_classical_sketches.py -q`.
- Build:
  - Run `cd paper/ctreepo && latexmk -pdf -interaction=nonstopmode main_minimal.tex`.
  - Check the log for undefined references/citations:
    `rg -n "Undefined|undefined|Citation.*undefined|Reference.*undefined|Label\\(s\\) may have changed" main_minimal.log`.
- Lean/doc crosswalk:
  - If theorem names or public stacks change, update
    `lean3/docs/PAPER_TO_LEAN_MAP.md` and
    `docs/ctreepo_appendix_proof_audit.md`.
  - Do not edit `docs/ctreepo_python_code_map_for_llms.md` for paper wording
    passes.
