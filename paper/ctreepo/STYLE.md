# C-TreePO Writing Style Guide

Match `main_new`'s voice. The rules below describe what `main_new`
looks like after several polish passes; `main_minimal` is the
cautionary example for what was rejected. When in doubt, open a
clean section that `main_new.tex` inputs (the Discussion and Theory
sections are good cadence references) and read it before editing.

**Naming note.** `main_new.tex` is the polished baseline. It happens
to input from a directory called `sections/v2/`, but "v2" here is
just a directory name; do not confuse it with the unrelated legacy
file `main_v2.tex` (flagged "reference only" in HANDOFF) or treat
"v2" as a meaningful version label. Always say "main_new" when
referring to the style baseline.

## Hard rules (do not violate)

1. **No "not X but Y."** Eliminate "not X, but Y", "not only X but
   also Y", "not just X but Y", "not merely X but Y." Replace with:
   assert Y directly. If X needs to be ruled out, rule it out in a
   separate sentence.
   - Bad: *"This is not a hand-designed sketch but a learned operator."*
   - Good: *"The operator is learned. A hand-designed sketch would
     solve a different problem."*
   - Acceptable trailing-clarification variant: *"is X, not Y"* at
     the end of a sentence after X is already stated. This is
     clarifying-by-contrast, not replacing-by-correction.

2. **Prefer parentheticals or footnotes over em-dashes.** Budget: at
   most one or two em-dashes per section, reserved for genuinely
   interruptive asides where commas would over-pause and parens would
   feel too quiet.
   - Bad: *"The summarizer $g$ --- a learned neural network --- replaces the hand-designed merge."*
   - Good (parens): *"The summarizer $g$ (a learned neural network)
     replaces the hand-designed merge."*
   - Good (split sentence): *"The summarizer $g$ is a learned neural
     network. It replaces the hand-designed merge."*
   - Good (footnote): *"The summarizer $g$ replaces the hand-designed
     merge.\footnote{$g$ is a learned neural network; architecture in §X.}"*
   - When an em-dash is genuinely the right tool, single em-dashes
     read better than wrapping a phrase in two em-dashes.

3. **No "rather than" stacking.** At most one "rather than" or
   "instead of" per paragraph. If a sentence wants two, split.

4. **No throat-clearing transitions.** Cut "It is worth noting that,"
   "Importantly,", "Crucially,", "It should be emphasized that," "We
   note that," "We remark that." Just say the thing.

5. **No hedge-then-claim.** Cut "It is reasonable to suggest that X"
   → "X." Cut "One might think that X" → "X." Reserve hedges for
   genuine epistemic uncertainty.

6. **No meta-references to the paper itself.** A published paper
   does not refer to itself as "this paper", "this draft", "this
   version", "this section", or "the reader". Cut all of:
   - *"In this paper we do X"* → *"We do X"*
   - *"The experiments in this paper..."* → *"The experiments..."*
   - *"Throughout the paper we use Y"* → *"Y is the running
     reference"*  / *"We use Y throughout"*
   - *"The rest of this paper develops X"* → name the section that
     develops X: *"Section~Y develops X"*
   - *"The rest of this section shows X"* → drop entirely; lead the
     next paragraph with the X claim
   - *"The role of this section is to X"* → drop entirely; X is
     what the section does, no need to announce it
   - *"What this section contributes is X"* → *"The contribution
     here is X"* or *"We extract X"*
   - *"Stated as the paper's thesis"* → drop; the proposition's
     prominence is conveyed by where it sits, not by labeling
   - *"Imagine a reader who..."* / *"the reader can..."* → use
     direct prose: *"Consider..."* or just state the claim
   - *"An object a reader can hold in mind"* → *"a concrete
     reference"* or drop
   - *"As shown later in this section"* / *"introduced later in
     this section"* → *"as shown below"* / *"introduced below"*
   - *"Used throughout the paper"* in figure/table captions →
     drop or replace with the specific scope (*"used in
     Sections~3--7"*).
   The pattern: a published paper makes claims about its subject,
   not about its own structure. Cross-references to specific
   sections by name (*"Section~5 introduces X"*) are fine; meta-
   commentary about the paper as an artifact is not.

7. **No defensive disclaimers.** Don't include sentences that
   explain what the paper is *not* doing or *not* claiming, when the
   thing was never claimed in the first place. Cut "we do not X
   because [Y]"; just leave X out and let its absence speak.
   Prefer the positive form: name the claim, its scope, and the
   evidence. Use contrast only when it prevents a concrete
   misreading.
   - Bad: *"We do not put per-call dollar costs side by side because
     the comparison mixes paradigms."* (The paper wasn't doing this
     comparison anyway; the disclaimer announces and defends a
     non-action.)
   - Good: just measure what you measure (call counts, supervision
     efficiency); silently omit the comparison you decided not to
     make.

8. **Every sentence does work.** Cut sentences that hedge, qualify,
   restate, or set up without adding a new claim, a new piece of
   evidence, or a new connective the reader needs. If a sentence can
   be deleted without weakening the argument, delete it. The test:
   read the paragraph with the sentence removed; if the argument
   still lands, the sentence was fluff.
   - Common fluff patterns: throat-clearing transitions ("In what
     follows..."), restated obvious-from-context implications
     ("This means that..."), reader-handholding ("As we saw
     above..."), and pre-emptive concessions to imagined critics
     ("Of course, one might object that...").

9. **Definitions, propositions, theorems, and assumptions contain
   only formal content.** The Definition / Proposition / Theorem /
   Assumption environments are reserved for the mathematical or
   structural statement. Interpretive prose, motivation, "plain-
   language readings," "this is why X exists" sentences, and design
   rationale belong in the surrounding text — before or after the
   box. Rule of thumb: if you can imagine the box copy-pasted into a
   textbook entry without alteration, it passes. After every theorem,
   proposition, corollary, or lemma, add one short prose unpacking
   outside the box: translate the statement into ordinary English and
   name the consequence it gives the paper. The unpacking must not add
   assumptions or proof content.
   - Bad:
     ```
     \begin{definition}[Oracle]
     An oracle is a function f*: X → Y mapping objects to task values.
     The oracle is what an expert rubric would produce; running it on
     every long input is too expensive, which is why this framework
     exists.
     \end{definition}
     ```
   - Good:
     ```
     \begin{definition}[Oracle]
     An oracle is a function f*: X → Y mapping objects to task values.
     \end{definition}

     The oracle is what an expert rubric would produce on the full
     input. Running it on every long input during training is too
     expensive, which is why a tree exists.
     ```

10. **Plain-English unpacking around math-heavy displays.** After (or
    before) a display equation that introduces a new symbol, relation,
    or claim, include one short prose sentence stating the content in
    words. The display says what; the prose says what it means.
    Particularly load-bearing for lifts of existing notation,
    equivalence relations, and theorems that read as one big symbolic
    statement.
    - Bad: *"Mergeability lifts coordinatewise:* \[…display…\] *Table 1
      records the correspondence."*
    - Good: *"Mergeability lifts in the natural way: if two states are
      valid summaries of two represented objects, their merge is a
      valid summary of the union. Formally,* \[…display…\] *."*

11. **No straw positions.** Don't justify a claim by ruling out an
    alternative no one proposed. State the positive content directly,
    with the affordance it gives the reader. The pattern "X because Y
    does not Z" reads as suspect when nothing in the surrounding text
    suggested Y. Replace with "X is W: it Z's because…".
    - Bad: *"The merge acts on states because the scalar answers do
      not contain enough information to combine."*
    - Good: *"Each shard's state has to carry whatever is needed to
      combine with another shard's state and still answer Q
      correctly. Distinct count makes this concrete: a distinct-count
      summary stores hash-based per-shard fingerprints so the merge
      can detect overlap."*

12. **Connect each technical property to its consequence.** When
    listing algebraic properties (associativity, idempotence),
    structural conditions (validity clauses), or assumption
    components, give each entry both its formal content and what it
    enables. The reader should see the property *and* what it buys.
    Don't list properties as a row of bare definitions.
    - Bad: *"Associativity of ⊕ is the schedule-invariance condition.
      Commutativity is specific to unordered or symmetric inputs.
      Idempotence is specific to set- or semilattice-like unions."*
    - Good: *"Associativity means (a ⊕ b) ⊕ c = a ⊕ (b ⊕ c). The
      schedule of merges does not matter; this is what makes parallel
      computation possible."* (and analogous treatments for the
      others)

13. **Distinguish framework names from object names.** A method,
    pipeline, or paper has a name (e.g. "C-TreePO"). The objects the
    method produces or operates on have separate names (e.g.
    "C-Tree"). Don't write "a [framework] [object]" when the object
    name alone is correct. Reserve the framework name for
    framework-level claims ("C-TreePO learns…", "the C-TreePO
    problem"); use the object name for the object ("a C-Tree fixes a
    tree T…").
    - Bad: *"A C-TreePO tree's job is to produce a root prediction
      that tracks the oracle."*
    - Good: *"A C-Tree's job is to produce a root prediction that
      tracks the oracle."*

14. **One vocabulary per concept.** Within a single discussion, pick
    one set of names for the same object and stay there. Introducing
    parallel aliases (e.g. "sketch" alongside "(encode, merge,
    query)") forces the reader to track redundant names.
    - Bad: *"a sketch of size k for a query Q… mergeable if
      sketch(u·v) and sketch(u) ⊕ sketch(v)…"* (using "sketch" as
      both the data structure and the operation that builds it)
    - Good: *"a summary scheme for a query family Q has three
      components — encode, merge, query… mergeable when
      encode(u·v) and encode(u) ⊕ encode(v)…"*

## Soft preferences (apply unless the cost is high)

6. **Lead with the claim, then unpack.** Sentences open with the
   conclusion or the load-bearing object, then qualify. The §13
   opener is the model: *"The theorems in Section~\ref{sec:theorems}
   preserve the oracle when the local laws hold, and the experiments
   in Sections~\ref{sec:markov-walkthrough}, \ref{sec:hll-parity},
   and~\ref{sec:manifesto-llm} exercise that preservation on three
   progressively richer oracles."*

7. **Use connectives that signal logical role.** *Concretely*,
   *equivalently*, *accordingly*, *that is*, *in particular* are
   preferred over generic *furthermore*, *moreover*, *additionally*.
   The connective should tell the reader what kind of move is
   happening (specialization, restatement, consequence).

8. **Active voice unless passive carries weight.** *We trained the
   tree on the recoverable DGP* over *the tree was trained on the
   recoverable DGP*. Passive is fine when the actor is the framework
   itself or the mechanism.

9. **Trim throat-clearers in subordinate clauses.** Cut "in order
   to," "as a means of," "with respect to," "in the context of," "in
   terms of." Each can almost always be replaced with one word or
   deleted.
   - Bad: *"We use Hamlet in order to motivate the boundary state argument."*
   - Good: *"We use Hamlet to motivate the boundary state argument."*

10. **Numbers and symbols read as objects, not topics.** Prefer *"The
    headline number is 0.829"* over *"In terms of the headline
    metric, we observe a value of 0.829."*

11. **Footnotes for engineering, not for asides.** Footnotes are
    good places to put architecture specs, training details, version
    numbers, dataset specifics, "see App.~X" pointers. They are a
    bad place to put intuition, qualifications, or scope statements;
    those belong in the main flow.

12. **One idea per sentence.** When a sentence has more than one
    connective ("and ... and ... while ... but ..."), split. Long
    sentences are fine if they have one clear spine.

13. **State the general before the specific.** Frame a section by
    introducing the abstract concept before picking out a specific
    instance as a worked example. Leading with one canonical example
    (e.g. HyperLogLog before mergeable summaries in general) anchors
    the reader's mental model to that instance and forces unwinding
    it later when generalizing.

14. **One allusion per setup paragraph.** Don't reference the same
    lineage or motivation twice in a preamble or section opening. If
    two opening paragraphs both gesture at "the literature does X, we
    do Y," consolidate into one.

## Intuition-forward bias

13. **Name the object before naming the symbol.** State what
    something *is* in plain language, then introduce the symbol:
    *"The summary $s_u$ at node $u$ is a compact text string
    representing the span under that node."* Not: *"$s_u$ is defined
    as $g(s_{u_L} \concat s_{u_R})$, where $g$ is the summarizer."*

14. **Display equations follow the sentence that motivates them.**
    Never open a paragraph with a display equation. The reader
    should know what they're about to see before they see it.

15. **Avoid recapitulating definitions in cross-references.**
    Reference Section X, do not re-define the symbol. The reader
    can flip back. Exception: in section openings, one short
    sentence reminding the reader of the previous section's
    punchline is good.

## Trim list (search-and-destroy)

Search for these and reconsider every hit:

- ` --- ` (em-dash) — replace with parens or split
- ` not only ` — replace with direct assertion
- ` we do not ` / ` does not ` / ` do not ` — rewrite as a positive
  scope statement where possible
- ` not a ` / ` not enough ` / ` not automatically ` — usually a
  defensive contrast; state the active claim instead
- ` rather than ` — keep at most one per paragraph
- ` instead of ` — usually deletable
- ` it is worth noting ` / ` it should be emphasized ` — delete
- ` in order to ` — replace with `to`
- ` in terms of ` — replace with `for`, `on`, or delete
- ` with respect to ` — replace with `for` or `on`
- ` it is the case that ` — delete
- ` we note that ` / ` we observe that ` — delete
- ` indeed ` — usually deletable
- ` quite ` / ` very ` / ` rather ` (as adverbs) — delete
- `this paper` / `this draft` / `this version` — replace with direct
  claim or delete (see hard rule 6)
- `this section` (as a meta-noun) / `the rest of this section` /
  `the role of this section` / `what this section contributes` —
  replace or delete (see hard rule 6)
- `the reader` / `a reader` / `imagine a reader` — replace with
  direct prose (see hard rule 6)
- `throughout the paper` / `used throughout the paper` — drop or
  replace with specific scope (see hard rule 6)
- ` overloaded ` (used adjectivally about notation) — judgmental.
  Replace with neutral "mapping," "translation," "presentation."
- ` event-level `, ` seedwise `, ` event-by-event ` — abstract
  framings that obscure plain probability content. Restate in
  concrete language ("over the random choices the sketch uses").
- ` is a state statement `, ` is a property statement `, ` is a
  category-theoretic statement ` — usually jargon for "this is a
  claim about Z." Just say what's claimed.
- ` our bet is to `, ` we wager that `, ` the gambit is `, ` the
  trick is to ` — colloquial framings for methodological choice. Use
  direct claim (`X learns…`).
- ` cannot be proved by inspection `, ` cannot be derived by
  inspection ` — usually overstated. Be precise about *what* can't
  be derived and what verification path the framework actually uses.
- ` walks through three moves `, ` covers four steps `, ` proceeds
  in N stages ` — meta-roadmap. Replace with the moves themselves,
  or just dive in.

## How to use this guide during a writing pass

For each paragraph in scope:

1. Read the paragraph aloud. Note any sentence that requires more
   than one breath.
2. Run the trim list grep against the paragraph. Each hit gets
   reconsidered.
3. Check em-dash count — over two, and at least one should become
   parens or a split.
4. Find the spine claim of the paragraph. Make sure sentence 1
   carries it.
5. Find any "not X but Y" or "not only X" constructions. Rewrite as
   direct assertion.
6. Read the paragraph aloud again. The second pass should be shorter
   and clearer than the first.
