# Information Sufficiency Bridge

This note records the exact scope of the repository's information-theoretic
surface for C-TreePO.

## What Lean backs

- Oracle sufficiency in the Doob-Dynkin / Blackwell sense.
- Representation-level target sufficiency: representation fibers refine target
  fibers, and targets/readouts factor through the representation.
- Likelihood-family sufficiency as contextual sufficiency over explicit
  parameters.
- Likelihood-free response sufficiency as contextual sufficiency over simulator
  probes, response signatures, or selected slice targets.
- Almost-sure oracle equality under the fixed-partition raw/summary joint law.
- Almost-sure oracle factorization through the realized summary.
- A.e. score-transport statements in `ScoreTransport.lean`.
- Zero task-relevant KLIC when supervision densities are oracle-indexed.
- Deterministic collision impossibility: if a summary merges oracle-distinct
  inputs, no decoder can recover the oracle from that summary.

Main entry points:

- `FormalProofs/OPT/ScoreTransport.lean`
- `FormalProofs/OPT/InformationRepresentationSufficiency.lean`
- `FormalProofs/OPT/InformationSufficiency.lean`
- `FormalProofs/OPT/MainTheorems.lean`

## What Lean does not back here

- Full Shannon source-coding claims.
- General mutual-information chain-rule arguments for the document hierarchy.
- Variational MI-estimator guarantees for NASS/SSS objectives.
- Random-slice probability guarantees for SSS/NASSS.
- SSNL/SNLE likelihood estimation or posterior consistency.
- Surface-text reconstruction guarantees.
- Preservation of downstream objectives that distinguish points inside one
  oracle fiber.

## Intended interpretation

C-TreePO is formalized here as a task-relevant compression scheme. The summary
only needs to preserve distinctions visible to the chosen oracle `f*` and to
downstream objectives that factor through that oracle. Any stronger claim should
be marked as optional future work rather than as part of the current theorem
surface.
