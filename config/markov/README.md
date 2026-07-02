# Markov Run Configs

These TOML files are the versioned, human-edited entrypoints for the current
Markov reporting stack.

Canonical configs:

- `tradeoff_pipeline.v3.toml`: current v3-ready comparison-grid tradeoff lane
- `publication_bundle.v3.toml`: current v3-ready bundle lane
- `publication_bundle.iteration.toml`: faster daytime iteration lane
- `publication_bundle.publication.toml`: full overnight publication lane
- `publication_bundle.no10240.toml`: reuse-friendly publication lane capped at 4096 docs for downstream reruns
- `tradeoff_pipeline.iteration.toml`: faster tradeoff-only iteration lane
- `tradeoff_pipeline.publication.toml`: full tradeoff-only publication lane
- `tradeoff_pipeline.no10240.toml`: tradeoff-only lane capped at 4096 docs

Compatibility configs:

- `publication_bundle.standard.toml`
- `tradeoff_pipeline.standard.toml`

Those `standard` files remain valid, but new runs should prefer the explicit
`iteration`, `publication`, or `v3` variants.

The `v3` configs are the current comparison-grid surface:

- explicit family-aware run-intent hashing
- canonical full-doc `official_fno` / `official_fno_sumlen`
- multileaf `t128` supervision-recovery ladder
- superset supervision packages as the default comparison arm
- stable `comparison_grid_v3` tree-reference alias

Public alias layer:

- tree presets can use short names like `comparison_grid_v3`, `standard_tree`, `half_c1`, and `fno_parity_canary`
- supervision-recovery packages can use short names like `root100`, `root100_extra_local10`, `root100_mass_local10`, and package-group aliases like `comparison_grid_v3` or `mass_r100`
- law packages can use `root_only`, `c2_only`, and `all_laws`

The current publication and tradeoff configs include the oracle-budget frontier
phase, so the report now covers partial-review / effective-full-label
efficiency in addition to throughput, learnability, law ablations, and full-doc
FNO comparisons.

The checked-in `v3` configs keep the comparison-grid tree surface explicit via
the stable `comparison_grid_v3` preset alias. If you want a
capacity-locked winner instead, set `tree_reference.mode = "capacity_locked"`
explicitly in the TOML after a capacity sweep has produced the locked summary.

The `publication_bundle.no10240.toml` reuse lane expects an existing
`--capacity-root` when you want to preserve that shared comparable tree
reference while skipping a fresh capacity sweep.

The configs also carry the shared GPU-resident runtime surface. The intended
fast path on CUDA is:

- `data_mode = "resident"`
- `bucket_mode = "exact_then_bucketed"`
- `preload_splits = ["train", "val", "test"]`
- `preload_targets = true`
- `workers_per_mig = 1` for training/parity/tradeoff lanes
- `allow_multi_worker_screen = true` with `capacity_workers_per_mig = 2` for
  small capacity-screen jobs that underfill a MIG slice

`cpu_debug` is still available, but it is a compatibility/debug mode rather
than the documented fast path.

Recommended workflow:

```bash
source venv/bin/activate

python3 scripts/run_markov_publication_bundle.py \
  --config config/markov/publication_bundle.v3.toml \
  --plan-only

python3 scripts/run_markov_publication_bundle.py \
  --config config/markov/publication_bundle.iteration.toml \
  --plan-only

python3 scripts/run_markov_publication_bundle.py \
  --config config/markov/publication_bundle.publication.toml \
  --detach \
  --output-root outputs/markov_publication_bundle_$(date +%Y%m%d_%H%M%S)
```

Tradeoff-only workflow:

```bash
python3 scripts/run_markov_optimization_tradeoff_pipeline.py \
  --config config/markov/tradeoff_pipeline.v3.toml \
  --plan-only
python3 scripts/run_markov_optimization_tradeoff_pipeline.py \
  --config config/markov/tradeoff_pipeline.iteration.toml \
  --plan-only
```

Use `--write-config-template` when you want to generate a new custom starting
point, but prefer keeping important overnight study configs under
`config/markov/` rather than relying on ad hoc CLI flag bundles.
