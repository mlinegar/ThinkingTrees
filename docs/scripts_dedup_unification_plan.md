# scripts/ and treepo/src Duplication Audit — Unification Plan

**Coverage note (read this first):** This audit was run as a parallel multi-agent
exhaustive-read sweep. Three attempts hit persistent API rate-limiting; the
`run_*` cluster (245 of 466 scripts/ files — the largest cluster by far) got
only an 8-file sample instead of a full read, and `plot_*` (52 files),
`tutorial_*` (20), `small_misc_prefixed`/`misc_unprefixed` (~155 files
combined) got **zero** real coverage across all three attempts. Fully covered:
`report_*` (5 of 59 files, deep-read), `build_*` (18 of 35), `audit_*` (7 of
11), and all of `treepo/src/treepo` (68 files, by subpackage). Every pattern
below is either directly confirmed by an agent reading the actual file, or
confirmed repo-wide by `grep` (noted explicitly where that's the evidence).
Patterns marked "likely recurs in X" are inference from the pattern's
ubiquity in covered clusters, not a confirmed read of X — treat those as
hypotheses to verify with a quick grep before investing extraction effort.

## Summary

The dominant duplication theme in `scripts/` is **infrastructure-layer
copy-paste, not domain-logic duplication**: the same repo-root `sys.path`
bootstrap block is byte-identical in **197 of 466 files** (grep-confirmed —
the single largest finding in this audit by file count), the same
argparse/grid-parsing/CLI-report skeletons recur across `build_*`, `audit_*`,
and the `run_*` sample, and PDF/plotting page-builders are reinvented in most
of `report_*` despite a working canonical `src/ctreepo/sim/report/pdf_utils.py`
already existing (only 7 of 59 `report_*.py` files import it).

`treepo/src/treepo` is **partially, not fully, the "clean canonical package"**
per prior project history. It's architecturally sound — real Protocols, one
grid/runner core, zero argparse/plotting bleed into the package — but has its
own dense pocket of small-helper duplication: `_jsonable`, `_optional_int`,
`_safe_float`, `MIN_PROPENSITY`, and tree-metadata-accessor helpers are each
independently reimplemented 2–6 times across `core_and_top`, `methods/`, and
`bench/sketches/adapters/`, including one genuine **semantic bug** (two
different `_optional_int` contracts — one raises, one coalesces to `None`).
The single largest concentration of copy-paste on the treepo side is the
KLL/Quantiles/REQ/t-digest adapter classes in `bench/sketches/adapters/datasketches_quantiles.py`
— roughly 175 of that file's 248 lines are a copy-pasted template.

Net: `scripts/` needs a new shared library layer for CLI/orchestration glue
that doesn't exist yet. `treepo/src` needs internal consolidation of helpers
it already almost has (e.g. `treepo.state.jsonable` should be the one
`_jsonable`, not one of five).

## Cross-cluster duplication patterns (highest priority first)

### 1. `REPO_ROOT` / `sys.path` bootstrap boilerplate
- **What:** identical 3–5 line block resolving the repo root
  (`Path(__file__).resolve().parents[1]` or `.parent.parent`) and
  conditionally `sys.path.insert(0, ...)` before `from src.ctreepo...`
  imports.
- **Confirmed:** 197 of 466 files, grep-verified. Directly read and confirmed
  in all 5 sampled `report_*` files, 7 sampled `audit_*` files, and all 8
  sampled `run_*` files — i.e. every single file actually read in this audit
  had it. Given that hit rate, treat it as present in effectively the entire
  directory, including the unread `plot_*`/`build_*`/`tutorial_*`/misc
  clusters.
- **Fix:** the *real* fix is a proper editable install (`pip install -e .`)
  so no script needs manual `sys.path` surgery — this is the same fix
  already applied on the treepo side per project memory (`treepo` dep +
  `[tool.uv.sources] path=../treepo editable`). If that's out of scope for
  `scripts/` right now, collapse to one `scripts/_lib/bootstrap.py::ensure_repo_root_on_syspath() -> Path`
  called as a single line per script, so future changes touch one file
  instead of 197.
- **Effort:** small per call site (mechanical codemod), large in aggregate
  (197+ files to touch). Do via scripted regex replacement + spot-check, not
  by hand. **Do the editable-install fix first if at all possible** — it
  makes the shim unnecessary.

### 2. Grid/list/bool-flag CLI-arg parsing helpers
- **What:** two parallel, functionally-equivalent tokenizer families for
  comma/whitespace-separated int/float lists and bool→flag-string
  conversion: `_parse_int_grid`/`_parse_float_grid`/`_bool_flag` (returns
  `Tuple`, has an empty-list guard) vs `_parse_items`/`_parse_ints`/`_parse_floats`
  (returns `List`, no guard). Pasted verbatim with only naming/return-type
  differences.
- **Confirmed:** 13 files in `build_*` (`build_lda_tree_recovery_learned_cmds.py`,
  `build_lda_tree_recovery_learned_world_batch_cmds.py`,
  `build_lda_tree_recovery_cmds.py`, `build_lda_tree_utility_vector_cmds.py`,
  `build_leaf_local_mixture_utility_cmds.py`, `build_markov_capability_suite_cmds.py`,
  `build_markov_changepoint_ops_count_cmds.py`, `build_markov_narrative_suite_cmds.py`,
  `build_markov_supervision_narrative_cmds.py`, `build_segmented_lda_ctreepo_cmds.py`,
  `build_segment_lda_ops_weight_recovery_cmds.py`, `build_tree_relevant_lda_followup_cmds.py`,
  `build_tree_relevant_lda_local_law_cmds.py`).
- **Likely recurs in:** `run_*` (unaudited at scale — only 8/245 files read,
  none showed this exact pattern, but `run_*` is structurally similar to
  `build_*` for sweep-grid generation). Worth a follow-up
  `rg '_parse_int_grid|_parse_items|_bool_flag' scripts/run_*.py` before
  finalizing scope.
- **Fix:** `scripts/_lib/grid_parsing.py`:
  `parse_int_list(s: str, *, allow_empty: bool = False) -> list[int]`,
  `parse_float_list(s: str, *, allow_empty: bool = False) -> list[float]`,
  `bool_flag(flag: str, value: bool) -> str`. One API subsuming both existing
  variants.
- **Effort:** medium (13+ confirmed call sites; mechanical per-site but high
  volume).

### 3. Audit report-writing + exit-code CLI skeleton
- **What:** `records -> JSON/Markdown report` with
  `--json-out`/`--markdown-out`/`--check` flags,
  `json.dumps(payload, indent=2, sort_keys=True) + "\n"`, mkdir-then-write
  for markdown, and the literal `return 2 if args.check else 0` idiom.
- **Confirmed:** `audit_run_targets.py`, `audit_publication_entrypoints.py`
  (near-identical — both wrap `src.ctreepo.run_registry.audit_target_records`/
  `iter_run_targets`), plus `audit_unified_g_usage.py`, `audit_v7_paper_assets.py`
  follow the same shape.
- **Fix:** `scripts/_lib/audit_report.py`:
  `write_audit_report(records, *, markdown_fn, json_out, markdown_out) -> None`
  and `audit_exit_code(errors, check) -> int`. Callers keep their own
  record-fetching/markdown-table logic; only the write/exit plumbing moves.
- **Effort:** medium (4 confirmed files, each keeps custom logic around the
  shared plumbing). Highest-confidence single pattern in the whole audit —
  one line (`return 2 if args.check else 0`) is literally identical across
  files.

### 4. Manifest/contract file-discovery scanner
- **What:** `_iter_candidate_files(paths)` walking directories via `rglob`
  against a hardcoded manifest-filename set, paired with
  `paths: nargs="+"` + `--expected-*`/`--require-*`/`--allow-legacy` flags.
- **Confirmed:** `audit_run_manifests.py`, `audit_tree_bundle_contracts.py`
  (near-identical scanner, different validated schema — `RunManifest` vs
  `TreeBundle`), plus a related but distinct `directory.glob("*manifest*.json")`
  scan in `audit_v7_paper_assets.py`.
- **Fix:** `scripts/_lib/manifest_scan.py`: shared `KNOWN_MANIFEST_FILENAMES`
  constant + `iter_candidate_manifest_files(paths, names=None, extra_patterns=())`.
  Keep schema-specific `normalize_*`/`validate_*` calls per script — those
  genuinely differ.
- **Effort:** medium.

### 5. "Load config → run → dump JSON/CSV" CLI template
- **What:** `run_treepo_lda_benchmark.py` and `run_treepo_markov_benchmark.py`
  are ~92% identical (verified via diff, 47 lines each): same argparse
  (`--config`, `--json-out`, `--csv-out`, `--print-json`), same
  `load_yaml_or_json` + validation, same call-and-dump-json body. Only 5
  lines differ (docstring, imported benchmark function, description string,
  call site).
- **Fix:** `scripts/_lib/cli_common.py::run_treepo_benchmark_cli(benchmark_fn, description) -> int`,
  or collapse both scripts into one `run_treepo_benchmark.py --family {lda,markov}`
  and delete the redundant file.
- **Effort:** small (helper is ~30 lines; exactly 2 call sites confirmed).
  Good first PR — cheap, fully scoped, proves the pattern before wider
  rollout.

### 6. PDF page-rendering helpers reinvented per file (vs `pdf_utils.py`)
- **What:** `src/ctreepo/sim/report/pdf_utils.py` already centralizes
  `write_text_page`/`write_image_page`/`page_header` for `PdfPages`-based
  reports. Only 7 of 59 `report_*.py` files import it.
  `report_tree_relevant_lda_proportion_extension_publication.py` (`_text_page`,
  `_paragraph_page`, `_image_page`) and `report_tree_relevant_lda_stage3.py`
  (`_page_header`, `_paragraph`, `_caption`, `_textbox`, `_save_page`) each
  independently reinvent the same concept with different signatures.
  `report_tree_root_only_parity_pdf.py` is the one file that does it right.
- **Confirmed:** 3 files deep-read; the "only 7/59 import pdf_utils" fact is
  grep-confirmed, implying most of the other 52 `report_*.py` files likely
  reimplement this independently (unread — hypothesis, not confirmed per-file).
- **Fix:** expand `pdf_utils.py` with `write_paragraph_page(pdf, *, title, paragraphs, width=108)`
  and `write_textbox(ax, x, y, w, h, title, body, ...)`, reconciling the two
  divergent naming schemes into one API. Migrate the 2 confirmed files first,
  then grep the other 52 for the same shape.
- **Effort:** medium (API design needs to reconcile 2+ divergent signatures
  before any file can migrate; migration itself is mechanical per file but
  there may be dozens of them).

### 7. `safe_mean`/`safe_sem`/`_is_finite` reimplementation instead of importing existing helpers
- **What:** `src/ctreepo/sim/report/pdf_utils.py` and `src/ctreepo/sim/util.py`
  already export `safe_float`, and `pdf_utils.py` has (or should gain)
  `safe_mean`/`safe_sem`. Two LDA report files hand-roll byte-identical local
  `_safe_mean`/`_safe_sem` instead of importing. `_is_finite` (thin
  `math.isfinite` wrapper) has never been centralized.
- **Confirmed:** `report_tree_relevant_lda_proportion_extension_publication.py`,
  `report_tree_relevant_lda_stage3.py` (exact duplicates of each other);
  grep suggests 10 files repo-wide reimplement `_safe_mean`/`_safe_sem`.
- **Fix:** pure deletion + import — no new abstraction needed, the target
  already exists. Add `is_finite(x) -> bool` to the same module while there.
- **Effort:** small. **Do this first** — zero design work, proves the
  import path works (as `report_tree_root_only_parity_pdf.py` already does
  for other symbols in the same module).

## Cluster-specific patterns

**`build_*`**
- `CommandSpec`/`TargetedCommand`/`FollowupCommand` dataclasses + manifest/
  matrix-md writer boilerplate repeated across the 3
  `build_exact_utility_transport_*` scripts — extract to
  `scripts/_build_common/cmd_manifest_io.py` (medium effort, ~3 files).
- Same 3 scripts also duplicate a ~25-parameter shell-command-string builder
  closure (`_cmd(...)`) — extract to
  `scripts/_build_common/treepo_preference_cmd.py::build_treepo_preference_cmd(runner, json_summary, **fields)`
  (high priority given the ~200 duplicated lines, 3 confirmed files).
- `_fmt_float`/`_safe_float_tag` (float→filesystem-safe-string, 3 independent
  variants) — extract to `scripts/_build_common/formatting.py::slug_float()`
  (low priority, 4 files).
- `legacy_entrypoint` shim boilerplate (7 files) is **already** mostly solved
  by the existing `fail_legacy_entrypoint()` helper — only the ~6-line
  per-file `sys.path`/import preamble remains; low priority, don't force
  further consolidation since each shim's filename is intentionally
  discoverable.

**`audit_*`**
- All 11 `audit_*.py` scripts share the standard
  `build_parser()`/`main(argv=None) -> int`/`if __name__ == "__main__"`
  skeleton. Low priority in isolation (the actual flags differ per script)
  but fold it into the `scripts/_lib/audit_report.py` helper (pattern 3
  above) if a `main_with_exit_code(build_parser, run)` wrapper turns out
  useful once patterns 3–4 are extracted.

**`run_*` (only 8/245 files sampled — treat as illustrative, not exhaustive)**
- `run_treepo_stack_generate_demo.py` / `run_treepo_stack_markov_demo.py`
  share a ~50–55 line canonical-sidecar-writing epilogue
  (`benchmark_ref_from_parts`/`method_ref_from_parts`/`write_canonical_sidecars`/`ResultRow`)
  despite genuinely different core demo logic. Extract just the epilogue to
  `scripts/_lib/treepo_stack_demo_sidecars.py::write_treepo_stack_demo_sidecars(...)`.
- **Architectural smell, not literal duplication:** `run_tree_neural_teacher_first_scaling_push.py`
  and `run_tree_neural_unifiedf_push.py` import private (`_`-prefixed)
  functions from sibling `run_*` scripts as if they were library modules
  (e.g. `tfpush._direct_metric`, `lp._make_slot_config`). This makes those
  sibling scripts de facto shared libraries whose refactors can silently
  break the importers. Fix is a real refactor — promote the reused helpers
  to a public `scripts/tree_neural_push_lib.py` — not a quick extraction;
  treat as its own project (large effort).
- Given every one of the 8 sampled files had the bootstrap pattern (item 1)
  and 2 of 8 were near-duplicate pairs, it's reasonable to expect `run_*` —
  at 245 files, over half the directory — to be where the bulk of
  unaudited duplication lives. **This cluster should be the first thing a
  follow-up audit covers**, not treated as low-risk because this pass
  under-sampled it.

**`report_*` (LDA sub-family specifically)**
- Beyond the cross-cutting patterns above, `report_tree_relevant_lda_proportion_extension_publication.py`
  and `report_tree_relevant_lda_stage3.py` also duplicate: a `seed_*.json`
  result-tree loader (`_load_runs`), a groupby-bucket-aggregate loop
  (`_agg`/`_aggregate`), and `tau_diversity_index`/`tau_label` formatting
  (also present in `report_tree_relevant_lda_combined.py`,
  `report_tree_relevant_lda_proportion_extension.py`,
  `report_tree_relevant_lda_followup.py`,
  `report_lda_tree_methods_paper.py`,
  `report_lda_tree_methods_best_of.py` per grep — 7 files total). These are
  LDA-report-domain logic, not generic script glue — scope to
  `src/ctreepo/sim/report/lda_report_labels.py` and
  `src/ctreepo/sim/report/run_loading.py`, not `scripts/_lib`.

**`plot_*`, `tutorial_*`, misc buckets** — **no findings; not audited.** Given
the bootstrap pattern's 197-file grep-confirmed footprint and the fact that
`plot_*` (52 files) almost certainly shares the same "reinvent matplotlib
styling per file" shape that `report_*` showed for PDF page-builders, treat
these as high-probability-but-unverified. A follow-up pass should target
these before doing large-scale extraction work, since they may contain the
single biggest win in the whole repo and it would be a mistake to write them
off as "already checked."

## Near-duplicate / mergeable files

| Files | Similarity | Recommendation |
|---|---|---|
| `run_treepo_lda_benchmark.py`, `run_treepo_markov_benchmark.py` | ~92% identical text | Merge into `run_treepo_benchmark.py --family {lda,markov}`, delete one |
| `run_treepo_stack_generate_demo.py`, `run_treepo_stack_markov_demo.py` | Shared sidecar epilogue only; core logic differs | Keep separate, extract shared epilogue only |
| `report_tree_relevant_lda_proportion_extension_publication.py`, `report_tree_relevant_lda_stage3.py` | ~25–30% overlapping infrastructure, 0% overlapping narrative/figures | Keep separate, extract shared scaffolding to `src/ctreepo/sim/report/` |
| `build_lda_tree_recovery_learned_cmds.py`, `build_lda_tree_recovery_learned_world_batch_cmds.py` | First ~30 lines byte-identical; differ in granularity (per-cell vs cached-world bundle) | Keep separate, extract shared argparse+grid-parsing boilerplate |
| `build_lda_tree_recovery_cmds.py`, `build_lda_tree_utility_vector_cmds.py`, `build_leaf_local_mixture_utility_cmds.py` | 60–70% structural overlap (same sweep-builder skeleton), different runner targets | Keep separate, share a generic `build_grid_commands()` helper |

**Flagged dead/archived code (found incidentally, worth a maintainer
decision, not part of the dedup effort):**
- `run_tree_root_only_parity_diagnosis.py` — `main()` immediately calls
  `scripts._markov_report_archive.archived_report_exit(...)` and returns
  before reaching ~230 lines of now-unreachable `parse_args()`/execution
  code. Not `OLD_`-prefixed per repo convention despite being fully archived.
- `report_tree_root_only_parity_pdf.py` — same short-circuit pattern via
  `archived_report_exit(...)`, ~35 lines of dead code after the return. This
  file otherwise correctly uses `pdf_utils.py` and is the best reference
  example for migrating the other report scripts (pattern 6).

Per repo convention (`feedback_old_prefix_for_legacy.md`), these should
either be renamed with the `OLD_` prefix or have their dead tails deleted —
recommend a human decision rather than doing it as part of this dedup pass.

## treepo/src findings

Confirms the "clean canonical package" framing only partially:

- **`bench/sketches/adapters/`** — the single densest duplication pocket in
  `treepo/src`. `KLLFloatsDatasketchesAdapter`/`QuantilesFloatsDatasketchesAdapter`/
  `REQFloatsDatasketchesAdapter` in `datasketches_quantiles.py` are the same
  ~45-line class body three times (175 of 248 file lines); a `_require_datasketches`
  guard is pasted in all 5 adapter files; a
  `serialize`/`serialized_size_bytes`/`memory_bytes` boilerplate trio repeats
  across 9 of 11 adapter classes. This subpackage is otherwise well-factored
  (shared `Protocol` in `protocol.py`, one `tree_reducer.py`) — the
  duplication is a consistent "same OOP template copy-pasted per sketch
  family" style, ripe for a factory/mixin extraction.
- **`core_and_top`** — `_jsonable` reimplemented independently 5 times
  (`manifest.py`, `objective.py`, `sampling.py`, `evidence.py` each
  reinvent it instead of importing `treepo.state.jsonable`, which is already
  the most complete version and already correctly imported by `tree.py`/
  `artifacts.py`). `MIN_PROPENSITY` constant redeclared in 3 files.
  `stable_digest`/`_stable_digest` (sha256 of sorted-key JSON) duplicated in
  `manifest.py`/`objective.py`. `_optional_int`/`_optional_str` byte-identical
  in `tree.py`/`state.py`. A law-kind alias map is independently maintained
  in both `local_law.py` (authoritative) and `objective.py` — real drift
  risk if one is updated without the other.
- **`methods/`** — lighter duplication, mostly small helper functions
  repeated across family implementations that each independently implement
  the `FamilyRuntime` Protocol (`dspy.py`/`llm.py`/`fno.py`/`sketch.py`/
  `learnable.py`/`oracles.py`). Notable: `_safe_float` in `runtime.py` and
  `fno.py` are near line-for-line identical; `getattr(tree, 'metadata', None) or {}`
  appears 13+ times across 6 files with no shared helper; `fno.py` has its
  own parallel, less-general reimplementation of the dataclass-from-mapping
  override-merging logic that `canonical_defaults.load_dataclass` already
  does more generally.
- **`tasks/manifesto/`** — one genuine **correctness** finding, not just
  duplication: `replication.py:_root_label` and `state.py:_root_label` pull
  the same metadata field with *different* fallback-key priority and
  different None-handling contracts. This should be fixed as a bug, not
  just deduplicated.

Package-side extraction targets (add to existing modules, not new
`scripts/_lib`-style layer, since this is all already-importable library
code):
- `treepo/state.py` — extend `jsonable()` to be a strict superset (add
  bare-`Enum` and `Path` handling), delete the 4 other `_jsonable` copies.
- `treepo/common.py` (already hosts `finite_float`) — add canonical
  `MIN_PROPENSITY`, `validate_propensity()`, `optional_int`/`optional_float`/
  `optional_str`, `safe_float(require_finite=False)`.
- New `treepo/tree_utils.py` — `resolve_object_id()`, `tree_metadata()`,
  `leaf_token_groups()`/`leaf_texts()` for the `methods/` duplication.
- `tasks/manifesto/rile.py` (or new `_metadata.py`) — single
  `manifesto_root_label(tree_or_metadata, *, clamp=True, default=None)`
  fixing the `replication.py`/`state.py` divergence.

## Suggested package structure

Two separate layers, matching the two separate problems:

1. **`scripts/_lib/`** (new) — pure CLI/orchestration glue that has no
   business being a package export: `bootstrap.py` (repo-root sys.path),
   `grid_parsing.py`, `audit_report.py`, `manifest_scan.py`, `cli_common.py`.
   This is script-only infrastructure; it should never be imported by
   `src/ctreepo` or `treepo/src`, only by files in `scripts/`.
2. **`src/ctreepo/sim/report/`** (existing, expand) — domain-report logic
   that's genuinely reusable across report scripts but is specific to the
   simulation/report domain, not generic CLI glue: expanded `pdf_utils.py`
   (paragraph/textbox primitives), new `lda_report_labels.py`,
   `run_loading.py`, `pandoc.py`. This already exists and is the right home
   — the problem is under-adoption, not absence.
3. **`treepo/src/treepo/`** (existing package) — the coercion/metadata
   helpers listed above belong directly in existing modules
   (`state.py`, `common.py`) or one new `tree_utils.py`, since they're
   genuinely part of the installed package's internal API, not
   script-only glue.

Rule of thumb for judgment calls: if the logic only exists to make a
standalone script runnable/parseable/reportable (argparse, sys.path, JSON
report writing), it goes in `scripts/_lib/`. If it's domain math or
data-shape logic that both scripts and the package might need, it belongs
in `src/ctreepo/sim/report/` or `treepo/src/treepo/`, not `scripts/_lib/`.

## Suggested rollout order

1. **Warm-up (small, zero-risk):** `safe_mean`/`safe_sem` import fix in the
   2 confirmed `report_*` files (pattern 7) and `_drop_empty` /
   `_optional_int`/`_optional_str` consolidation in `treepo/src/treepo/core_and_top`
   — both are pure deletion + import, no API design needed, prove the
   pattern works before bigger investment.
2. **`treepo/state.jsonable` consolidation** — fix the 5-way `_jsonable`
   duplication and the `replication.py`/`state.py` `_root_label` correctness
   bug together, since both are in the same small `core_and_top`/`tasks/manifesto`
   surface and the bug fix has real value independent of dedup.
3. **`scripts/_lib/bootstrap.py`** — highest file-count win (197+ files) but
   decide editable-install vs. shim first; this blocks or unblocks a lot of
   downstream mechanical work depending on the choice.
4. **`scripts/_lib/audit_report.py` + `manifest_scan.py`** — small, fully
   scoped cluster (11 files total), good second real extraction after the
   warm-up.
5. **`build_*` grid-parsing + `_build_common/`** — 13+ confirmed files,
   highest-value `build_*` win.
6. **Follow-up audit of `plot_*`, `tutorial_*`, `run_*` (full 245 files), and
   the misc buckets** — this pass never got real coverage here. Given `run_*`
   is over half the directory and `plot_*` likely mirrors `report_*`'s PDF
   duplication pattern, this is probably where the largest untapped win is.
   Recommend running the same exhaustive-scan approach again, but with
   smaller/more serial batches to survive rate limits, before committing to
   the `pdf_utils.py` expansion (pattern 6) at full scope.
7. **`treepo/src/treepo/bench/sketches/adapters/` quantile-adapter factory**
   — largest single-file duplication concentration found (175/248 lines),
   but isolated and low-urgency since it's inside an already-working,
   well-tested subpackage.
8. **Architectural refactor: `run_tree_neural_*_push.py` private
   cross-script imports** — largest-effort item, treat as its own project
   once the mechanical extractions above are done.
