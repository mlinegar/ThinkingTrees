# Manifesto DSPy Parity Example

This example uses the current DSPy/chat scorer path and compares predictions
to the expert/teacher manifesto labels already carried by the qsentence grids.

Run a small existing grid check:

```bash
./venv/bin/python scripts/run_manifesto_qsentence_dspy_ladder.py --help
```

Inspect the completed diffusiongemma DSPy grid:

```bash
./venv/bin/python - <<'PY'
import json
from pathlib import Path

path = Path("outputs/manifesto_qsentence_diffusiongemma_full_leaf1/grid_summary.json")
payload = json.loads(path.read_text())
for row in payload["rows"]:
    print({
        "iteration": row["iteration"],
        "n_eval": row["n_eval"],
        "external_expert_pearson": row["external_expert_pearson"],
        "external_expert_mae_1_7": row["external_expert_mae_1_7"],
        "internal_f_pearson": row["internal_f_pearson"],
    })
PY
```

The first two rows are the transport/scorer parity check. They preserve the
expert labels with Pearson about `0.999` and MAE about `0.00245` on the
1-7 scale. The third row is a trained merge stage and is not the parity row.

To export the same qsentence grid supervision through the shared treepo
preference boundary:

```bash
./venv/bin/python scripts/export_manifesto_qsentence_preferences.py \
  --labeled-trees outputs/manifesto_qsentence_dspy_labeled_grid/leafq001/labeled_trees.jsonl \
  --output-dir outputs/manifesto_qsentence_preferences/leafq001 \
  --mode ranked
```

The exporter writes treepo's canonical `TreeRecord` JSONL, `PreferenceDataset`, Hugging Face
`DatasetDict`, supervised/DPO/reward/GRPO projections, and fine-tuning adapter
bundles under `finetune_adapters/`. Each qsentence/merge/root node becomes a
`target="g"` preference unit with the same canonical fields as treepo: `unit_id`,
`tree_id`, `doc_id`, `node_id`, `level`, `position`, `parent_id`,
`left_child_id`, and `right_child_id`. The context is the same DSPy qsentence
prompt and the top candidate is the compact CMP `TaskState(kind="manifesto_policy")`
target from the grid.

The default adapter exports include embedding pairs/triplets/ranked rows, TRL
SFT/DPO/reward/scalar-reward/GRPO JSONL, and `dspy_examples` rows. The result
JSON also contains a `thinkingtrees_dspy` dry-run under
`finetune_adapters.learning_adapters`; it points at the existing qsentence DSPy
family runtime and writes the prepared rows without starting a model service.

The qsentence ladder and qsentence labeled-grid builders now write the same bundle
automatically for every leaf row under `<output>/<family>/leafqNNN/treepo_finetune/`
or `<grid>/leafqNNN/treepo_finetune/`. This includes the CMP qsentence grid and
the Benoit qsentence grid. The scalar and joint Manifesto teacher-grid
generators, existing-results dimension replay, and embedding-FNO leaf grid write
the generic labeled-tree bundle under each `treepo_finetune/` directory, with root
`target="f"` rows plus node/root `target="g"` rows. Use
`--no-export-finetune-views` on those runners to suppress the extra JSONL files.

For any older Manifesto `labeled_trees.jsonl` artifact, use the generic exporter:

```bash
./venv/bin/python scripts/export_manifesto_labeled_tree_preferences.py \
  --labeled-trees outputs/some_manifesto_run/leaf_002/labeled_trees.jsonl \
  --output-dir outputs/some_manifesto_run/leaf_002/treepo_finetune \
  --kind auto \
  --mode ranked
```

For actual DSPy learning, keep using the qsentence ladder or pass the labeled
trees to the existing family runtime. The adapter boundary is now the data
handoff, not a second trainer stack:

```python
from pathlib import Path
from src.ctreepo.distillation import load_labeled_trees
from src.ctreepo.manifesto_qsentence_dspy_family import (
    ManifestoQSentenceDSPyFamily, ManifestoQSentenceDSPyFamilyConfig,
)
from src.ctreepo.treepo_bridge.manifesto_preferences import build_manifesto_qsentence_preferences
from src.training.finetune_adapters import train_finetune_adapter

trees = load_labeled_trees("outputs/manifesto_qsentence_dspy_labeled_grid/leafq001/labeled_trees.jsonl")
preferences = build_manifesto_qsentence_preferences(trees, mode="ranked")
family = ManifestoQSentenceDSPyFamily(config=ManifestoQSentenceDSPyFamilyConfig(lm_config={...}))

train_finetune_adapter(
    "thinkingtrees_dspy",
    preferences,
    Path("outputs/manifesto_qsentence_preferences/leafq001/dspy_train"),
    dry_run=False,
    family_runtime=family,
    kind="g",
    traces=trees,
    g_init="raw_concat",
    f="teacher_passthrough",
)
```
