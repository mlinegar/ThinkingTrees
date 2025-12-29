# Plan

Establish a GenRM-backed preference-judging and synthetic-data pipeline that encodes the leaf-law compatibility conditions from main.pdf, while keeping a path to a DSPy-learned judge for cost/latency tradeoffs.

## Requirements
- Ensure the GenRM model is present at `/mnt/data/models/nvidia/Qwen3-Nemotron-235B-A22B-GenRM` and register a `qwen3-nemotron-genrm` profile in `config/settings.yaml` for the server scripts.
- Use GenRM as the primary preference judge, with an optional DSPy approximation path for learned judging.
- Encode the local compatibility laws (L1-L3) and optional edge-wise probe (E1) as a machine-readable spec and wire them into audits and training data generation.
- Preserve provenance: model id, rubric/law id, generation settings, raw judge outputs, and derived scores.

## Leaf-law reference (from main.pdf)
- Realized string at node u: S(u) = b_i for leaf i, else S(u) = S(u_L) concat S(u_R).
- One-pass reduction: reduce_g(u) = g(S(u)) for leaf, else reduce_g(u) = g(reduce_g(u_L) concat reduce_g(u_R)).
- Root output: Z^(1)(x) = reduce_g(root(T)); normalization rounds: Z^(R)(x) = g(Z^(R-1)(x)) for R >= 2.
- Realized internal edges: (v, w) = (S(u_L), S(u_R)) at each internal node u.
- L1 (Leaf Sufficiency): E_g[ metric( f_star(g(b)), f_star(b) ) ] = 0 for every realized leaf b.
- L2 (Parentwise Compatibility): E_g^{downarrow u}[ metric( f_star( g(reduce_g(u_L) concat reduce_g(u_R)) ), f_star(S(u)) ) ] = 0 for every realized internal node u.
- L3 (On-Range Idempotence): E_g[ metric( f_star(g(Z)), f_star(Z) ) | Z ] = 0 for every Z in range(g).
- E1 (Edge-wise probe, optional): metric( f_star(v concat w), f_star( g(g(v) concat g(w)) ) ) = 0 at each realized internal edge (v, w).
- Note: If g is deterministic, expectations are pointwise; these local laws imply one-pass preservation in expectation.

## Scope
- In: model registration, GenRM judge adapter, preference data integration, leaf-law spec + checks, synthetic-data generation script, documentation updates.
- Out: full finetuning infrastructure, UI work, distributed orchestration.

## Files and entry points
- Model profiles and server scripts: `config/settings.yaml`, `scripts/start_oracle_server.sh`, `scripts/start_vllm.sh`.
- GenRM judge adapter: `src/ops_engine/training_framework/judges/genrm.py` (new) or `src/core/llm_client.py` extension.
- Preference pipeline: `src/ops_engine/training_framework/preference.py`.
- Synthetic data: `src/ops_engine/training_framework/synthetic_data.py`, `experiments/manifesto_rile/generate_synthetic_data.py`.
- Law checks + audits: `src/ops_engine/checks.py`, `src/ops_engine/auditor.py`, `src/ops_engine/training_framework/verification.py`.

## Data model / API changes
- Add `LeafLawSpec` (YAML + loader) to map law ids -> rubric text, checks to run, severity, and audit sampling policy.
- Add `GenRMResult` metadata (helpfulness score, ranking score, reasoning, raw text) to `PreferencePair` and dataset exports.
- Add a `RewardJudge` protocol (or extend `PairwiseJudge`) for GenRM-style scoring with fallbacks if OpenAI role validation blocks custom roles.
- Optional: add `edgewise_compatibility` check type to represent E1 explicitly.

## Action items
[ ] Add `qwen3-nemotron-genrm` to `config/settings.yaml` and update server docs to point at `/mnt/data/models/nvidia/Qwen3-Nemotron-235B-A22B-GenRM`.
[ ] Implement GenRM judge adapter with role formatting (`response_1`, `response_2`) and robust parsing of helpfulness/ranking outputs.
[ ] Wire GenRM into preference collection and exports (DPO/SFT + judge metadata) in `preference.py`.
[ ] Create `config/leaf_laws.yaml` and loader; map L1/L2/L3/E1 to checks in `checks.py`/`auditor.py`.
[ ] Add a synthetic-data pipeline script that generates candidate summaries, collects GenRM judgments, and emits law-aligned datasets.
[ ] Update docs to describe GenRM usage, law specs, and how to run the pipeline.
[ ] Add tests for GenRM parsing, law spec loading, and dataset serialization.

## Testing and validation
- Unit tests for GenRM parsing and LeafLawSpec loading.
- Smoke test: start GenRM vLLM server and run a single pairwise judgment.
- Small batch run of synthetic-data generation with law auditing enabled.

## Risks and edge cases
- OpenAI-compatible clients may reject nonstandard roles; may need raw HTTP fallback.
- Long context limits and model size may require smaller batch sizes or lower max_model_len.
- Expectations in L1-L3 are in terms of stochastic g; audits may need multiple samples or deterministic assumptions.

## Open questions
- Should we approximate E_g by multi-sampling g per node, or assume deterministic g for audits?
- How tightly should the learned DSPy judge be coupled to GenRM (distill-then-replace vs hybrid voting)?
- Do we want separate datasets for law audits vs preference learning, or a unified schema?
