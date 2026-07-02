# Unified g Audit (2026-04-21)

This artifact records manifesto/Benoit code paths that still mention split leaf and merge summarizers before the unified-g migration.

## Bucket Counts

| Bucket | Count |
|---|---:|
| `active_split_paths` | 62 |
| `active_unified_paths` | 0 |
| `legacy_compatible_split_classes` | 16 |
| `false_positives` | 31 |

## Active Split Paths

| Path | Line | Symbol | Reason | Snippet |
|---|---:|---|---|---|
| `scripts/phase0_economic_pilot.py` | 56 | `ManifestoMerger` | active runtime path references split leaf/merge g | `from src.tasks.manifesto.pipeline import ManifestoMerger, ManifestoSummarizer` |
| `scripts/phase0_economic_pilot.py` | 56 | `ManifestoSummarizer` | active runtime path references split leaf/merge g | `from src.tasks.manifesto.pipeline import ManifestoMerger, ManifestoSummarizer` |
| `scripts/phase0_economic_pilot.py` | 136 | `ManifestoSummarizer` | active runtime path references split leaf/merge g | `summarizer: ManifestoSummarizer,` |
| `scripts/phase0_economic_pilot.py` | 137 | `ManifestoMerger` | active runtime path references split leaf/merge g | `merger: ManifestoMerger,` |
| `scripts/phase0_economic_pilot.py` | 187 | `summary1/summary2 merge signature` | active runtime path references split leaf/merge g | `lambda p: merger(summary1=p[0], summary2=p[1], rubric=rubric), pairs` |
| `scripts/phase0_economic_pilot.py` | 245 | `ManifestoSummarizer` | active runtime path references split leaf/merge g | `summarizer = ManifestoSummarizer(use_cot=False)` |
| `scripts/phase0_economic_pilot.py` | 246 | `ManifestoMerger` | active runtime path references split leaf/merge g | `merger = ManifestoMerger(use_cot=False)` |
| `scripts/phase0_economic_pilot.py` | 246 | `merger attribute` | active runtime path references split leaf/merge g | `merger = ManifestoMerger(use_cot=False)` |
| `scripts/phase2_combined_pipeline.py` | 5 | `ManifestoSummarizer` | active runtime path references split leaf/merge g | `One `ManifestoSummarizer` with `JOINT_RUBRIC` produces a single summary per` |
| `scripts/phase2_combined_pipeline.py` | 67 | `ManifestoMerger` | active runtime path references split leaf/merge g | `from src.tasks.manifesto.pipeline import ManifestoMerger, ManifestoSummarizer` |
| `scripts/phase2_combined_pipeline.py` | 67 | `ManifestoSummarizer` | active runtime path references split leaf/merge g | `from src.tasks.manifesto.pipeline import ManifestoMerger, ManifestoSummarizer` |
| `scripts/phase2_combined_pipeline.py` | 131 | `ManifestoSummarizer` | active runtime path references split leaf/merge g | `summarizer: ManifestoSummarizer,` |
| `scripts/phase2_combined_pipeline.py` | 132 | `ManifestoMerger` | active runtime path references split leaf/merge g | `merger: ManifestoMerger,` |
| `scripts/phase2_combined_pipeline.py` | 152 | `summary1/summary2 merge signature` | active runtime path references split leaf/merge g | `lambda p: merger(summary1=p[0], summary2=p[1], rubric=rubric),` |
| `scripts/phase2_combined_pipeline.py` | 270 | `ManifestoSummarizer` | active runtime path references split leaf/merge g | `summarizer = ManifestoSummarizer(use_cot=False)` |
| `scripts/phase2_combined_pipeline.py` | 271 | `ManifestoMerger` | active runtime path references split leaf/merge g | `merger = ManifestoMerger(use_cot=False)` |
| `scripts/phase2_combined_pipeline.py` | 271 | `merger attribute` | active runtime path references split leaf/merge g | `merger = ManifestoMerger(use_cot=False)` |
| `scripts/phase3_combined_optimize.py` | 51 | `ManifestoMerger` | active runtime path references split leaf/merge g | `from src.tasks.manifesto.pipeline import ManifestoMerger, ManifestoSummarizer` |
| `scripts/phase3_combined_optimize.py` | 51 | `ManifestoSummarizer` | active runtime path references split leaf/merge g | `from src.tasks.manifesto.pipeline import ManifestoMerger, ManifestoSummarizer` |
| `scripts/phase3_combined_optimize.py` | 71 | `ManifestoSummarizer` | active runtime path references split leaf/merge g | `self.summarizer = ManifestoSummarizer(use_cot=False)` |
| `scripts/phase3_combined_optimize.py` | 72 | `ManifestoMerger` | active runtime path references split leaf/merge g | `self.merger = ManifestoMerger(use_cot=False)` |
| `scripts/phase3_combined_optimize.py` | 72 | `merger attribute` | active runtime path references split leaf/merge g | `self.merger = ManifestoMerger(use_cot=False)` |
| `scripts/phase3_combined_optimize.py` | 92 | `merger attribute` | active runtime path references split leaf/merge g | `lambda p: self.merger(summary1=p[0], summary2=p[1], rubric=self.rubric), pairs` |
| `scripts/phase3_combined_optimize.py` | 92 | `summary1/summary2 merge signature` | active runtime path references split leaf/merge g | `lambda p: self.merger(summary1=p[0], summary2=p[1], rubric=self.rubric), pairs` |
| `scripts/phase3_full_pipeline_optimize.py` | 62 | `ManifestoMerger` | active runtime path references split leaf/merge g | `from src.tasks.manifesto.pipeline import ManifestoMerger, ManifestoSummarizer` |
| `scripts/phase3_full_pipeline_optimize.py` | 62 | `ManifestoSummarizer` | active runtime path references split leaf/merge g | `from src.tasks.manifesto.pipeline import ManifestoMerger, ManifestoSummarizer` |
| `scripts/phase3_full_pipeline_optimize.py` | 86 | `ManifestoSummarizer` | active runtime path references split leaf/merge g | `self.summarizer = ManifestoSummarizer(use_cot=False)` |
| `scripts/phase3_full_pipeline_optimize.py` | 87 | `ManifestoMerger` | active runtime path references split leaf/merge g | `self.merger = ManifestoMerger(use_cot=False)` |
| `scripts/phase3_full_pipeline_optimize.py` | 87 | `merger attribute` | active runtime path references split leaf/merge g | `self.merger = ManifestoMerger(use_cot=False)` |
| `scripts/phase3_full_pipeline_optimize.py` | 108 | `merger attribute` | active runtime path references split leaf/merge g | `lambda p: self.merger(summary1=p[0], summary2=p[1], rubric=self.rubric), pairs` |
| `scripts/phase3_full_pipeline_optimize.py` | 108 | `summary1/summary2 merge signature` | active runtime path references split leaf/merge g | `lambda p: self.merger(summary1=p[0], summary2=p[1], rubric=self.rubric), pairs` |
| `scripts/run_manifesto_batched_example.py` | 427 | `create_merge_summarizer` | active runtime path references split leaf/merge g | `merge_module = task.create_merge_summarizer()` |
| `scripts/run_manifesto_batched_example.py` | 427 | `merge_module` | active runtime path references split leaf/merge g | `merge_module = task.create_merge_summarizer()` |
| `scripts/run_manifesto_batched_example.py` | 445 | `merge_module` | active runtime path references split leaf/merge g | `merge_module=merge_module,` |
| `src/tasks/manifesto/pipeline.py` | 202 | `ManifestoSummarizer` | active runtime path references split leaf/merge g | `class ManifestoSummarizer(dspy.Module):` |
| `src/tasks/manifesto/pipeline.py` | 273 | `ManifestoSummarizer` | active runtime path references split leaf/merge g | `Two-tier output control (see ``ManifestoSummarizer``). Inputs here are` |
| `src/tasks/manifesto/pipeline.py` | 300 | `RILEMerge` | active runtime path references split leaf/merge g | `self.merge = dspy.ChainOfThought(RILEMerge)` |
| `src/tasks/manifesto/pipeline.py` | 302 | `RILEMerge` | active runtime path references split leaf/merge g | `self.merge = dspy.Predict(RILEMerge)` |
| `src/tasks/manifesto/pipeline.py` | 304 | `summary1/summary2 merge signature` | active runtime path references split leaf/merge g | `def forward(self, summary1: str, summary2: str, rubric: str = RILE_PRESERVATION_RUBRIC) -> str:` |
| `src/tasks/manifesto/pipeline.py` | 307 | `summary1/summary2 merge signature` | active runtime path references split leaf/merge g | `summary1 + summary2,` |
| `src/tasks/manifesto/pipeline.py` | 318 | `summary1/summary2 merge signature` | active runtime path references split leaf/merge g | `result = self.merge(rubric=effective_rubric, summary1=summary1, summary2=summary2, config=cfg)` |
| `src/tasks/manifesto/pipeline.py` | 323 | `summary1/summary2 merge signature` | active runtime path references split leaf/merge g | `result = self.merge(rubric=effective_rubric, summary1=summary1, summary2=summary2, config=cfg)` |
| `src/tasks/manifesto/pipeline.py` | 326 | `summary1/summary2 merge signature` | active runtime path references split leaf/merge g | `merged = f"{summary1}\n\n{summary2}"` |
| `src/tasks/manifesto/pipeline.py` | 385 | `ManifestoSummarizer` | active runtime path references split leaf/merge g | `Wraps ManifestoSummarizer and translates parameter names:` |
| `src/tasks/manifesto/pipeline.py` | 392 | `ManifestoSummarizer` | active runtime path references split leaf/merge g | `self._inner = ManifestoSummarizer(use_cot=use_cot)` |
| `src/tasks/manifesto/pipeline.py` | 403 | `ManifestoMerger` | active runtime path references split leaf/merge g | `Wraps ManifestoMerger and translates parameter names:` |
| `src/tasks/manifesto/pipeline.py` | 411 | `ManifestoMerger` | active runtime path references split leaf/merge g | `self._inner = ManifestoMerger(use_cot=use_cot)` |
| `src/tasks/manifesto/pipeline.py` | 415 | `summary1/summary2 merge signature` | active runtime path references split leaf/merge g | `return self._inner(summary1=left_summary, summary2=right_summary, rubric=rubric)` |
| `src/tasks/manifesto/pipeline.py` | 422 | `ManifestoPipeline` | required target symbol audit marker | `class ManifestoPipeline(dspy.Module):` |
| `src/tasks/manifesto/pipeline.py` | 439 | `ManifestoSummarizer` | active runtime path references split leaf/merge g | `self.summarizer = ManifestoSummarizer(use_cot=use_cot)` |
| `src/tasks/manifesto/pipeline.py` | 440 | `ManifestoMerger` | active runtime path references split leaf/merge g | `self.merger = ManifestoMerger(use_cot=use_cot)` |
| `src/tasks/manifesto/pipeline.py` | 440 | `merger attribute` | active runtime path references split leaf/merge g | `self.merger = ManifestoMerger(use_cot=use_cot)` |
| `src/tasks/manifesto/pipeline.py` | 493 | `merger attribute` | active runtime path references split leaf/merge g | `return self.merger(summary1=s1, summary2=s2, rubric=rubric)` |
| `src/tasks/manifesto/pipeline.py` | 493 | `summary1/summary2 merge signature` | active runtime path references split leaf/merge g | `return self.merger(summary1=s1, summary2=s2, rubric=rubric)` |
| `src/tasks/manifesto/pipeline.py` | 523 | `ManifestoPipelineWithStrategy` | required target symbol audit marker | `class ManifestoPipelineWithStrategy(dspy.Module):` |
| `src/tasks/manifesto/pipeline.py` | 539 | `ManifestoPipelineWithStrategy` | required target symbol audit marker | `pipeline = ManifestoPipelineWithStrategy()` |
| `src/tasks/manifesto/pipeline.py` | 543 | `ManifestoPipelineWithStrategy` | required target symbol audit marker | `pipeline = ManifestoPipelineWithStrategy(judge=genrm_judge)` |
| `src/tasks/manifesto/pipeline.py` | 577 | `merge_module` | active runtime path references split leaf/merge g | `self.merge_module = merge_module or StrategyCompatibleMerger(use_cot=use_cot)` |
| `src/tasks/manifesto/pipeline.py` | 608 | `merge_module` | active runtime path references split leaf/merge g | `merge_module=self.merge_module,` |
| `src/tasks/manifesto_task.py` | 22 | `MergeSummarizer` | active runtime path references split leaf/merge g | `MergeSummarizer,` |
| `src/tasks/manifesto_task.py` | 107 | `manifesto_task.create_merge_summarizer` | required target symbol audit marker | `def create_merge_summarizer(self):` |
| `src/tasks/manifesto_task.py` | 109 | `MergeSummarizer` | active runtime path references split leaf/merge g | `return MergeSummarizer(use_cot=self._use_cot_merge)` |

## Active Unified Paths

_None detected._

## Legacy-Compatible Split Classes

| Path | Line | Symbol | Reason | Snippet |
|---|---:|---|---|---|
| `src/tasks/manifesto/pipeline.py` | 58 | `RILEMerge` | split compatibility definition | `class RILEMerge(dspy.Signature):` |
| `src/tasks/manifesto/pipeline.py` | 58 | `RILEMerge` | split compatibility definition | `class RILEMerge(dspy.Signature):` |
| `src/tasks/manifesto/pipeline.py` | 270 | `ManifestoMerger` | split compatibility definition | `class ManifestoMerger(dspy.Module):` |
| `src/tasks/manifesto/pipeline.py` | 270 | `ManifestoMerger` | split compatibility definition | `class ManifestoMerger(dspy.Module):` |
| `src/tasks/manifesto/pipeline.py` | 399 | `StrategyCompatibleMerger` | split compatibility definition | `class StrategyCompatibleMerger(dspy.Module):` |
| `src/tasks/manifesto/summarizer.py` | 13 | `MergeSummarizer` | legacy split compatibility surface | `from src.tasks.manifesto import LeafSummarizer, MergeSummarizer` |
| `src/tasks/manifesto/summarizer.py` | 17 | `MergeSummarizer` | legacy split compatibility surface | `merge_summarizer = MergeSummarizer()` |
| `src/tasks/manifesto/summarizer.py` | 32 | `GenericMerger` | legacy split compatibility surface | `GenericMerger,` |
| `src/tasks/manifesto/summarizer.py` | 90 | `ManifestoSummarizer` | legacy split compatibility surface | ```pipeline.ManifestoSummarizer`` docstring): prompt carries a soft` |
| `src/tasks/manifesto/summarizer.py` | 129 | `MergeSummarizer` | split compatibility definition | `class MergeSummarizer(dspy.Module):` |
| `src/tasks/manifesto/summarizer.py` | 129 | `MergeSummarizer` | split compatibility definition | `class MergeSummarizer(dspy.Module):` |
| `src/tasks/manifesto/summarizer.py` | 180 | `create_merge_summarizer` | split compatibility definition | `def create_summarizers(` |
| `src/tasks/manifesto/summarizer.py` | 195 | `MergeSummarizer` | legacy split compatibility surface | `return LeafSummarizer(use_cot=use_cot), MergeSummarizer(use_cot=use_cot)` |
| `src/tasks/manifesto/summarizer.py` | 197 | `GenericMerger` | legacy split compatibility surface | `return GenericSummarizer(use_cot=use_cot), GenericMerger(use_cot=use_cot)` |
| `src/tasks/manifesto_task.py` | 107 | `create_merge_summarizer` | split compatibility definition | `def create_merge_summarizer(self):` |
| `src/tasks/manifesto_task.py` | 107 | `create_merge_summarizer` | split compatibility definition | `def create_merge_summarizer(self):` |

## False Positives / Non-Runtime Mentions

| Path | Line | Symbol | Reason | Snippet |
|---|---:|---|---|---|
| `src/pipelines/batched.py` | 705 | `GenericMerger` | non-manifesto or non-runtime split reference | `from src.core.summarization import GenericSummarizer, GenericMerger` |
| `src/pipelines/batched.py` | 716 | `GenericMerger` | non-manifesto or non-runtime split reference | `merge_module=GenericMerger(),` |
| `src/pipelines/batched.py` | 716 | `merge_module` | non-manifesto or non-runtime split reference | `merge_module=GenericMerger(),` |
| `src/tasks/manifesto/__init__.py` | 53 | `MergeSummarizer` | non-manifesto or non-runtime split reference | `MergeSummarizer,` |
| `src/tasks/manifesto/__init__.py` | 71 | `RILEMerge` | non-manifesto or non-runtime split reference | `RILEMerge,` |
| `src/tasks/manifesto/__init__.py` | 74 | `ManifestoSummarizer` | non-manifesto or non-runtime split reference | `ManifestoSummarizer,` |
| `src/tasks/manifesto/__init__.py` | 75 | `ManifestoMerger` | non-manifesto or non-runtime split reference | `ManifestoMerger,` |
| `src/tasks/manifesto/__init__.py` | 80 | `ManifestoPipeline` | required target symbol audit marker | `ManifestoPipeline,` |
| `src/tasks/manifesto/__init__.py` | 81 | `ManifestoPipelineWithStrategy` | required target symbol audit marker | `ManifestoPipelineWithStrategy,` |
| `src/tasks/manifesto/__init__.py` | 130 | `MergeSummarizer` | non-manifesto or non-runtime split reference | `"MergeSummarizer",` |
| `src/tasks/manifesto/__init__.py` | 143 | `RILEMerge` | non-manifesto or non-runtime split reference | `"RILEMerge",` |
| `src/tasks/manifesto/__init__.py` | 147 | `ManifestoSummarizer` | non-manifesto or non-runtime split reference | `"ManifestoSummarizer",` |
| `src/tasks/manifesto/__init__.py` | 148 | `ManifestoMerger` | non-manifesto or non-runtime split reference | `"ManifestoMerger",` |
| `src/tasks/manifesto/__init__.py` | 154 | `ManifestoPipeline` | required target symbol audit marker | `"ManifestoPipeline",` |
| `src/tasks/manifesto/__init__.py` | 155 | `ManifestoPipelineWithStrategy` | required target symbol audit marker | `"ManifestoPipelineWithStrategy",` |
| `src/tasks/manifesto/lawstress_bootstrap_metric.py` | 213 | `summary1/summary2 merge signature` | non-manifesto or non-runtime split reference | `score_s1, score_s2 = _proxy_score_texts(proxy_model, embedding_client, [summary1, summary2])` |
| `src/tasks/manifesto/lawstress_bootstrap_metric.py` | 229 | `summary1/summary2 merge signature` | non-manifesto or non-runtime split reference | `pen2, notes2 = _length_penalty(summary2, summary1)` |
| `src/tasks/manifesto/lawstress_bootstrap_program.py` | 4 | `UnifiedG` | unified g marker | `- `UnifiedG`: a single summarizer g used for both leaf + merge inputs.` |
| `src/tasks/manifesto/lawstress_bootstrap_program.py` | 15 | `format_merge_input` | unified g marker | `from src.core.protocols import format_merge_input` |
| `src/tasks/manifesto/lawstress_bootstrap_program.py` | 20 | `UnifiedG` | unified g marker | `class UnifiedG(dspy.Module):` |
| `src/tasks/manifesto/lawstress_bootstrap_program.py` | 36 | `UnifiedG` | unified g marker | `def __init__(self, g: Optional[UnifiedG] = None) -> None:` |
| `src/tasks/manifesto/lawstress_bootstrap_program.py` | 38 | `UnifiedG` | unified g marker | `self.g = g or UnifiedG()` |
| `src/tasks/manifesto/lawstress_bootstrap_program.py` | 56 | `summary1/summary2 merge signature` | non-manifesto or non-runtime split reference | `summary2 = self.g(content=summary1, rubric=rubric)` |
| `src/tasks/manifesto/lawstress_bootstrap_program.py` | 57 | `summary1/summary2 merge signature` | non-manifesto or non-runtime split reference | `return dspy.Prediction(summary1=summary1, summary2=summary2)` |
| `src/tasks/manifesto/lawstress_bootstrap_program.py` | 63 | `format_merge_input` | unified g marker | `disjoint = self.g(content=format_merge_input(summary_a, summary_b), rubric=rubric)` |
| `src/tasks/manifesto/lawstress_bootstrap_program.py` | 64 | `format_merge_input` | unified g marker | `joint = self.g(content=format_merge_input(segment_a, segment_b), rubric=rubric)` |
| `src/tasks/manifesto/lawstress_bootstrap_program.py` | 76 | `UnifiedG` | unified g marker | `"UnifiedG",` |
| `src/tasks/manifesto/lawstress_eval.py` | 282 | `summary1/summary2 merge signature` | non-manifesto or non-runtime split reference | `summary2 = summary1` |
| `src/tasks/manifesto/teacher_trace_eval.py` | 231 | `summary1/summary2 merge signature` | non-manifesto or non-runtime split reference | `summary2 = summary1` |
| `src/tasks/manifesto/train_oracle.py` | 23 | `ManifestoPipeline` | required target symbol audit marker | `ManifestoPipeline,` |
| `src/tasks/manifesto/train_oracle.py` | 117 | `ManifestoPipeline` | required target symbol audit marker | `pipeline = ManifestoPipeline(chunk_size=2000)` |
