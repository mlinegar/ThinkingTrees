# C-TreePO v10_evidence_polish — Proof Audit

Working list for the proof readability pass. Every formal claim in the body or
appendix gets a row here, paired with where its proof lives. We work through
each row and check it against three criteria:

1. **Readable.** Plain English, no notation beyond what the body has already
   established. No Lean theorem names in the body. No "machine-checked as
   foo_bar_baz in path/to/file" dumps.
2. **Assumptions stated.** Each proof opens with the active assumptions made
   crisp, even when the parent statement already lists them.
3. **Linked.** Every body claim has a clear pointer to where its full proof
   lives (App F or App A); every body proof sketch terminates with that
   pointer.

**Lean is mentioned in exactly one place** — the consolidated entry in
Appendix~\ref{app:v8-lean-crosswalk}, with a tongue-in-cheek note that we
have, regrettably, not yet applied C-TreePO to audit our own proofs. Inside
that appendix the full repository surface is listed; outside it, no
\path|FormalProofs/...| or \path|machine_checked_name| appears in body or
other appendices.

## Inventory

### §3 C-Tree Math

| ID | Kind | Statement | Proof location | Status |
|---|---|---|---|---|
| `ass:v8-fixed-tree` | Assumption | Fixed realized tree | (assumption, no proof) | [x] cleared |
| `def:v8-local-laws` | Definition | C1 / C2 / C3 | (definition) | [x] cleared |
| `thm:v8-root-preservation` | Theorem | Root state preserves the document oracle under exact local laws | §3 inline sketch + App F line 60 | [x] |
| `cor:v8-preferences-pass-to-root` | Corollary | Oracle-compatible readouts pass through every node | §3 inline sketch + App F (subtree case of preservation) | [x] |

### §4 Objective and Theorem Ladder

| ID | Kind | Statement | Proof location | Status |
|---|---|---|---|---|
| `prop:v8-manifesto-preference-equivalence` | Proposition | Replacing manifesto by C-Tree root leaves population preference objective unchanged | §4 inline sketch + App F line 150 | [x] |
| `prop:v10-population-gap` | Proposition | Population gap bound \(|G_{\mathrm{meth}}|\le C_{\mathrm{meth}}\Delta_R\) | §4 inline sketch + App F line 249 (Population Transport) | [x] |

### §5 Audit and Proxy-Corrected Estimation

| ID | Kind | Statement | Proof location | Status |
|---|---|---|---|---|
| `prop:v10-corrected-loss-unbiased` | Proposition | Corrected node loss is unbiased for the oracle loss | §5 inline sketch + App A | [x] |
| `prop:v10-ht-unbiased` | Proposition | HT distortion estimator is unbiased | §5 inline sketch + App A | [x] |
| `prop:v10-finite-sample-certificate` | Proposition | Drift admits the four-term decomposition | §5 inline sketch + App F (sketch) + App A (component bounds) | [x] |

### Appendix F (theorem certificate details)

| ID | Kind | Statement | Proof location | Status |
|---|---|---|---|---|
| `prop:v8-all-node-preservation` | Proposition | Exact local laws preserve every represented span (every node, not only the root) | App F line 42 | [x] |
| `prop:v8-fixed-partition` | Proposition | Extension to any deterministic fixed partition | App F line 81 | [x] |
| `ex:v8-c2-independent` | Example | C1 and C3 do not imply C2 | App F line 92 (worked example) | [x] kept as example |
| `lem:v8-dpo-transport` | Lemma | DPO pointwise transport | App F line 204 | [x] |
| `prop:v8-population-transport` | Proposition | Population transport gives the population gap bound | App F line 249 | [x] |

## Lean cross-reference inventory (to remove from body)

These lines mention Lean theorem names or files outside Appendix E and need to
be stripped:

- `01_introduction.tex:119` — `machine-checked Lean counterparts` (drop adjective)
- `03_ctree_math.tex:190-194` — proof block for `thm:v8-root-preservation`
- `03_ctree_math.tex:230-234` — proof block for `cor:v8-preferences-pass-to-root`
- `04_objective_theorem_ladder.tex:46` — sentence about Lean carrying \(\gamma\)
- `04_objective_theorem_ladder.tex:175-180` — proof block for `prop:v8-manifesto-preference-equivalence`
- `04_objective_theorem_ladder.tex:231-234` — proof block for `prop:v10-population-gap`
- `05_audit_certificate.tex:91` — proof block for `prop:v10-corrected-loss-unbiased`
- `05_audit_certificate.tex:165-171` — proof block for `prop:v10-ht-unbiased`
- `05_audit_certificate.tex:211-218` — proof block for `prop:v10-finite-sample-certificate`
- `08_conclusion.tex:28` — `Lean-backed theorem stack`
- `appendix/v10_evidence_polish/F_theorem_certificate_details.tex:6-7` — pointer paragraph (keep ref to App E)

After cleanup, only `appendix/v10_evidence_polish/E_machine_checked_crosswalk.tex` mentions Lean.

## Verification (after the pass)

```text
$ grep -liE "lean|formalproofs|machine.checked" sections/v10_evidence_polish/*.tex \
    appendix/v10_evidence_polish/*.tex | grep -v E_machine_checked_crosswalk
sections/v10_evidence_polish/01_introduction.tex   # only \ref{app:v8-lean-crosswalk}
sections/v10_evidence_polish/03_ctree_math.tex     # "cleanup", "cleanest" — false positives
sections/v10_evidence_polish/04_objective_theorem_ladder.tex  # same
appendix/v10_evidence_polish/F_theorem_certificate_details.tex  # only \ref to App E
```

User-facing "Lean" text appears only inside App E. Build is clean: 52
pages, 0 `\fixme`, 7 theorem-likes paired with 7 proof envs in §3-§5,
0 undefined references.

