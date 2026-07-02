# C-TreePO v10 Evidence Polish — Math Cohesion Draft Note

This is a handoff note for a new draft pass. It is not TeX source and should
not be treated as an implemented paper edit. The goal is to give another LLM a
concrete proposal for polishing the math discussion in Sections 3--5.

## Goal

Make Sections 3--5 read as one chain:

1. **State contract:** C1/C2/C3 define what a C-Tree state must preserve.
2. **Objective and transport:** the root/local objective trains toward that
   state contract, and the theorem ladder says what preservation buys.
3. **Audit and certificate:** sampled node labels estimate the remaining
   local-law distortion and turn it into a downstream error certificate.

The current structure is basically right. The recommended pass is moderate:
keep Sections 3, 4, and 5 separate, but make the handoffs between them more
explicit and shorten the estimator-heavy part of Section 5.3.

## Recommendation

Use **moderate cohesion**:

- Keep Section 3 as the state/local-law section.
- Keep Section 4 as the objective/theorem-ladder section.
- Keep Section 5 as the audit/certificate section.
- Move derivational HT details and assumption bookkeeping to Appendix A/F
  where those details already live.
- Keep the main text focused on what each object means and why the next object
  is needed.

The reader should not feel that Section 5.3 is a survey-statistics proof. They
only need the estimator, the finite-population target, and the four certificate
terms. The algebra can stay in Appendix A/F.

## Candidate Section 3 Closing Rewrite

This could replace or tighten the current closing of Section 3.

```tex
\subsection{From Exact Laws to Learning}

The exact theorem describes the endpoint: if every realized leaf,
state re-entry, and merge call stays in the right oracle fiber, the
root state can stand in for the full manifesto. Learning starts from
the fact that this endpoint is only partially observable. Root labels
are available for the document-level target; local oracle labels are
scarce; proxy scores are cheap.

The next two sections turn that gap into an objective and an audit.
Section~\ref{sec:v8-objective} writes the root/local objective: one
channel fits the document-level target, and one channel pushes the
state map toward C1/C2/C3 validity. Section~\ref{sec:v8-audit-certificate}
then treats the realized tree as a finite population of audit units,
so sampled oracle calls can estimate the remaining local-law
distortion.
```

## Candidate Section 4 Theorem-Ladder Transition

This could replace the heavier setup around the theorem ladder. If the
assumption-to-claim table feels too proof-inventory-like in the main text,
move it to Appendix F and point to it from a short sentence.

```tex
The theorem ladder has three levels. Preservation is the structural
level: exact C1/C2/C3 laws let node and root states replace the raw
spans they represent. Optimization adds a downstream compatibility
premise: the loss or preference objective must depend on the input
only through the preserved oracle interface. Certification relaxes
exact validity to an estimated distortion budget on the realized tree.

The rest of the section states the two main consequences needed in the
body. Appendix~\ref{app:v8-full-proofs} records the detailed assumption
crosswalk and proof bookkeeping.
```

## Candidate Shorter Section 5.3 Rewrite

This is the main compression candidate. It keeps the estimator visible but
moves the proof-as-you-go material to the appendices.

```tex
\subsection{Distortion Certificate}

The corrected local-law loss is the training object. The certificate
uses the same sampled nodes to estimate the realized tree's remaining
representation distortion. Let \(\mathcal U\) be the finite population
of sample-eligible tree units, \(N=|\mathcal U|\). Unit \(i\) has
produced state \(s_i\), represented span \(S(i)\), logged inclusion
probability \(\pi_i>0\), and inclusion indicator \(Z_i\).

For an accessible audit scorer \(J\), define the observed distortion
\[
  D_i^J=d_Y(J(s_i),J(S(i))).
\]
The Horvitz--Thompson estimate of mean node distortion is
\[
  \widehat{\Delta}_{\mathrm{HT}}^{J}
  =
  \frac{1}{N}
  \sum_{i\in\mathcal U}
  \frac{Z_i}{\pi_i}D_i^J .
\]
Correct logged marginal propensities make this an unbiased estimate
of the finite-population mean distortion measured by \(J\). Unequal
propensities change variance and effective sample size; they do not
change the target. Appendix~\ref{app:v8-objective-ipw} gives the
design-based derivation and variance envelope.

The reported certificate transports this estimated distortion through
the downstream method and then adds the three residual terms the audit
cannot hide:
\[
  |G_{\mathrm{meth}}|
  \le
  C_{\mathrm{meth}}\widehat{\Delta}_{\mathrm{HT}}
  +
  B_{\mathrm{cal}}
  +
  B_{\mathrm{est}}
  +
  B_{\mathrm{clip}} .
\]
Here \(C_{\mathrm{meth}}\widehat{\Delta}_{\mathrm{HT}}\) is sampled
representation distortion expressed in units of the downstream
quantity. \(B_{\mathrm{cal}}\) is accessible-scorer mismatch relative
to \(f^\ast\). \(B_{\mathrm{est}}\) is sampling uncertainty.
\(B_{\mathrm{clip}}\) is the price of clipping large weights or
residuals for variance control. Appendix~\ref{app:v8-full-proofs}
derives the four-term inequality.

A completed manifesto node audit would instantiate \(D_i^J\) with
span-level expert or quasi-sentence aggregates, log the node
propensities, and report the displayed certificate. The current
Benoit replication is the root-observed tier; the teacher-trace
diagnostics identify where such a sampled node audit should spend its
labels.
```

## Implementation Notes For The Next Draft

- Do not merge Sections 3--5 unless the whole theory block is being rewritten.
  The current section boundaries are useful.
- Consider moving the assumption-to-claim table from Section 4 to Appendix F.
  It is helpful proof bookkeeping, but it slows the main narrative.
- Preserve theorem/proposition labels where practical:
  `thm:v8-root-preservation`, `cor:v8-preferences-pass-to-root`,
  `prop:v8-manifesto-preference-equivalence`,
  `prop:v10-population-gap`, `prop:v10-corrected-loss-unbiased`,
  `prop:v10-ht-unbiased`, and `prop:v10-finite-sample-certificate`.
- If the HT unbiasedness proposition moves fully to Appendix A/F, update any
  main-text references and the appendix proof map/crosswalk tables.
- Keep Lean/proof-assistant references only in the dedicated crosswalk appendix.
  Main text and proof appendices should remain narratively self-contained.
- After any TeX implementation, run:

```bash
cd paper/ctreepo
latexmk -pdf -interaction=nonstopmode main_v10_evidence_polish.tex
```

Then check:

```bash
rg -n "Lean|FormalProofs|\\.lean|proof assistant|proof-assistant|machine-checked|machine checked|unified_preference_gap|MainTheorems|PreferenceBounds|formalization" \
  sections/v10_evidence_polish appendix/v10_evidence_polish/*.tex \
  | rg -v "E_machine_checked_crosswalk|PROOF_AUDIT"
```

The grep should return no user-facing main/proof-appendix hits.
