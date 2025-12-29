# ThinkingTrees Test Suite Documentation

This document provides comprehensive documentation for all tests in the ThinkingTrees project. The test suite consists of **34 test files** across two major test suites with approximately **13,600+ lines of test code**.

---

## Table of Contents

1. [Overview](#overview)
2. [Main Project Tests](#main-project-tests)
   - [Configuration & Fixtures](#configuration--fixtures)
   - [Core Tests](#core-tests)
   - [OPS Engine Tests](#ops-engine-tests)
   - [Preprocessing Tests](#preprocessing-tests)
3. [Langextract Tests](#langextract-tests)
   - [Core Language Extraction](#core-language-extraction)
   - [Data & Format Handling](#data--format-handling)
   - [Inference & Extraction](#inference--extraction)
   - [Provider & Plugin Tests](#provider--plugin-tests)
   - [API & Integration Tests](#api--integration-tests)
   - [Utility Tests](#utility-tests)
4. [Running Tests](#running-tests)
5. [Test Configuration](#test-configuration)

---

## Overview

### Test Frameworks Used

| Suite | Framework | Style |
|-------|-----------|-------|
| Main Project | pytest | Fixture-based, property testing |
| Langextract | absl.testing (absltest + parameterized) | Google-style unittest |

### Test Categories

- **Unit Tests**: Individual component testing in isolation
- **Integration Tests**: Component interaction testing
- **Property-Based Tests**: Invariant testing using hypothesis
- **Mock Tests**: LLM-dependent code with mocked responses
- **Live API Tests**: Optional tests against real API endpoints

---

## Main Project Tests

Location: `tests/`

### Configuration & Fixtures

#### `tests/conftest.py`
**Purpose**: Shared pytest fixtures used across all test modules.

| Fixture | Description |
|---------|-------------|
| `short_text` | Simple single-sentence text for basic testing |
| `medium_text` | Multi-paragraph text for chunking tests |
| `long_text` | Extended text for stress testing |
| `sample_rubric` | Default rubric template |
| `simple_rubric` | Minimal rubric for quick tests |
| `detailed_rubric` | Comprehensive rubric with multiple criteria |
| `mock_llm_client` | Mock LLM returning predictable responses |
| `identity_summarizer` | Summarizer that returns input unchanged |
| `truncating_summarizer` | Summarizer that truncates to N characters |
| `concatenating_summarizer` | Summarizer that joins child summaries |
| `simple_oracle` | Basic oracle for audit testing |
| `always_pass_oracle` | Oracle that always passes |
| `always_fail_oracle` | Oracle that always fails |
| `sample_tree` | Pre-built 3-level tree with 4 leaves |
| `sample_chunks` | Pre-made TextChunk list |
| `default_chunker` | DocumentChunker with default settings |
| `small_chunker` | DocumentChunker for creating many small chunks |

#### `tests/test_config.py`
**Purpose**: Configuration loader for test settings.

**Classes**:
- `TestConfig`: Loads and provides access to test configuration profiles
- `ConfigAccessor`: Utility for dot-notation access to nested config

**Test Profiles Available**:
- Chunking: `default`, `small`, `with_overlap`
- Builder: `default`, `many_chunks`
- Auditor: `default`, `exhaustive`, `minimal`

#### `tests/config/test_settings.yaml`
**Purpose**: YAML-based test configuration defining profiles and sample data.

```yaml
chunking:
  default: {max_chars: 500, min_chars: 50, overlap: 0}
  small: {max_chars: 100, min_chars: 20, overlap: 0}
  with_overlap: {max_chars: 100, min_chars: 20, overlap: 20}

builder:
  default: {max_chunk_chars: 200, min_chunk_chars: 20, chunk_strategy: sentence}
  many_chunks: {max_chunk_chars: 100, min_chunk_chars: 10, chunk_strategy: sentence}

auditor:
  default: {sample_budget: 10, discrepancy_threshold: 0.1}
  exhaustive: {sample_budget: 1000, discrepancy_threshold: 0.1}
  minimal: {sample_budget: 2, discrepancy_threshold: 0.5}
```

---

### Core Tests

#### `tests/core/test_data_models.py`
**Purpose**: Tests for the fundamental data structures (OPSNode and OPSTree).

**Test Classes**:

##### `TestOPSNode`
Tests for individual node operations and properties.

| Test | Description |
|------|-------------|
| `test_create_leaf_node` | Verify leaf nodes have raw_text_span and level=0 |
| `test_create_internal_node` | Verify internal nodes have children and level > 0 |
| `test_is_leaf_property` | `is_leaf` returns True iff no children |
| `test_node_id_unique` | Each node gets a unique ID |
| `test_leaf_invariants` | Leaves: raw_text_span set, no children, level=0 |
| `test_internal_invariants` | Internal: children set, level > 0 |
| `test_node_string_representation` | String repr includes id, level, and leaf status |
| `test_node_equality` | Nodes with same id are equal |
| `test_node_hash` | Nodes can be used in sets/dicts |

##### `TestOPSTree`
Tests for tree structure and traversal.

| Test | Description |
|------|-------------|
| `test_create_single_node_tree` | Tree with one leaf (root = leaf) |
| `test_create_binary_tree` | Standard binary tree structure |
| `test_height_calculation` | Tree height is max depth |
| `test_node_count` | Correct total node count |
| `test_leaf_count` | Correct leaf count |
| `test_traverse_preorder` | Preorder traversal visits nodes correctly |
| `test_traverse_postorder` | Postorder traversal visits nodes correctly |
| `test_traverse_level_order` | Level order traversal (BFS) |
| `test_find_node_by_id` | Locate node by ID |
| `test_get_path_to_root` | Path from leaf to root |
| `test_get_leaves` | Retrieve all leaf nodes |
| `test_get_internal_nodes` | Retrieve all internal nodes |

##### `TestBuildTreeFromLeaves`
Tests for the factory function that constructs trees from leaf nodes.

| Test | Description |
|------|-------------|
| `test_build_from_single_leaf` | Single leaf becomes root |
| `test_build_from_two_leaves` | Two leaves create height-1 tree |
| `test_build_from_multiple_leaves` | Correct tree structure for various leaf counts |
| `test_build_preserves_leaf_order` | Leaf order matches input order |

---

#### `tests/core/test_signatures.py`
**Purpose**: Tests for DSPy signature definitions.

**Test Classes**:

##### `TestSignatureImports`
| Test | Description |
|------|-------------|
| `test_recursive_summary_import` | RecursiveSummary signature imports correctly |
| `test_oracle_judge_import` | OracleJudge signature imports correctly |
| `test_sufficiency_check_import` | SufficiencyCheck signature imports correctly |
| `test_merge_consistency_check_import` | MergeConsistencyCheck signature imports correctly |

##### `TestSignatureFields`
| Test | Description |
|------|-------------|
| `test_recursive_summary_fields` | RecursiveSummary has expected input/output fields |
| `test_oracle_judge_fields` | OracleJudge has expected fields |
| `test_sufficiency_check_fields` | SufficiencyCheck has expected fields |
| `test_merge_consistency_check_fields` | MergeConsistencyCheck has expected fields |

---

#### `tests/core/test_llm_client.py`
**Purpose**: Tests for LLM client configuration and interaction.

**Test Classes**:

##### `TestLLMConfig`
| Test | Description |
|------|-------------|
| `test_create_vllm_config` | Factory method for vLLM configuration |
| `test_create_sglang_config` | Factory method for SGLang configuration |
| `test_create_openai_config` | Factory method for OpenAI configuration |
| `test_config_validation` | Invalid configurations raise errors |
| `test_config_from_env` | Configuration from environment variables |

##### `TestLLMResponse`
| Test | Description |
|------|-------------|
| `test_response_text_property` | Access response text |
| `test_response_metadata` | Response includes metadata (tokens, latency) |

##### `TestLLMClient`
| Test | Description |
|------|-------------|
| `test_client_creation` | Client instantiation with config |
| `test_client_complete` | Basic completion request |
| `test_client_error_handling` | Graceful error handling |

##### `TestMockLLMClient`
| Test | Description |
|------|-------------|
| `test_mock_returns_predictable` | Mock returns configured responses |
| `test_mock_tracks_calls` | Mock records all calls for inspection |
| `test_mock_configurable_responses` | Mock can be configured with specific responses |

---

### OPS Engine Tests

#### `tests/ops_engine/test_builder.py`
**Purpose**: Tests for tree construction from text chunks.

**Test Classes**:

##### `TestIdentitySummarizer`
| Test | Description |
|------|-------------|
| `test_returns_input_unchanged` | Summarizer returns exact input |
| `test_handles_empty_input` | Empty string returns empty string |

##### `TestTruncatingSummarizer`
| Test | Description |
|------|-------------|
| `test_truncates_long_text` | Text exceeding limit is truncated |
| `test_preserves_short_text` | Text under limit unchanged |
| `test_configurable_limit` | Truncation limit is configurable |

##### `TestConcatenatingSummarizer`
| Test | Description |
|------|-------------|
| `test_concatenates_children` | Joins child summaries with separator |
| `test_configurable_separator` | Separator is configurable |

##### `TestBuildConfig`
| Test | Description |
|------|-------------|
| `test_default_config` | Default configuration values |
| `test_custom_config` | Custom configuration values |
| `test_config_validation` | Invalid configs raise errors |

##### `TestOPSTreeBuilder`
| Test | Description |
|------|-------------|
| `test_build_single_chunk` | One chunk creates single-node tree |
| `test_build_two_chunks` | Two chunks create height-1 tree |
| `test_build_four_chunks` | Four chunks create height-2 balanced tree |
| `test_build_odd_chunks` | Odd chunk counts handled correctly |
| `test_build_from_text` | Build directly from text string |
| `test_build_from_file` | Build from file path |
| `test_build_from_chunks` | Build from pre-made TextChunk list |
| `test_leaf_summaries_set` | Leaf nodes have summaries from raw text |
| `test_internal_summaries_set` | Internal nodes have summaries from children |
| `test_tree_structure_valid` | All tree invariants hold |
| `test_rubric_passed_to_summarizer` | Rubric used in summarization calls |
| `test_build_statistics` | Builder returns build statistics |

##### `TestTreeBuilderPropertyTests`
Property-based tests using hypothesis.

| Test | Description |
|------|-------------|
| `test_leaf_count_matches_input` | Tree has same leaf count as input chunks |
| `test_height_bounded` | Tree height is log2(n) bounded |
| `test_parent_child_consistency` | All parent-child relationships valid |
| `test_all_leaves_at_level_zero` | All leaves have level=0 |

---

#### `tests/ops_engine/test_auditor.py`
**Purpose**: Tests for the audit system that validates tree quality.

**Test Classes**:

##### `TestSimpleScorer`
| Test | Description |
|------|-------------|
| `test_scorer_passing_summary` | Good summaries get high scores |
| `test_scorer_failing_summary` | Poor summaries get low scores |
| `test_scorer_returns_oracle_score` | Scorer returns OracleScore object |
| `test_scorer_with_rubric` | Rubric is accepted (used by LLM-based scorers) |

##### `TestAlwaysPassOracle`
| Test | Description |
|------|-------------|
| `test_always_passes` | Oracle always returns pass |
| `test_returns_zero_score` | Discrepancy score is 0 |

##### `TestAlwaysFailOracle`
| Test | Description |
|------|-------------|
| `test_always_fails` | Oracle always returns fail |
| `test_returns_high_score` | Discrepancy score is 1.0 |

##### `TestAuditConfig`
| Test | Description |
|------|-------------|
| `test_default_config` | Default audit configuration |
| `test_custom_budget` | Custom sample budget |
| `test_custom_threshold` | Custom discrepancy threshold |
| `test_sampling_strategy` | Different sampling strategies |

##### `TestAuditCheckResult`
| Test | Description |
|------|-------------|
| `test_result_creation` | Create audit check result |
| `test_result_passed_property` | Check if result passed |
| `test_result_score` | Access discrepancy score |
| `test_result_node_reference` | Result references checked node |

##### `TestAuditReport`
| Test | Description |
|------|-------------|
| `test_report_creation` | Create audit report |
| `test_report_summary` | Report provides summary statistics |
| `test_report_failed_checks` | Access failed checks |
| `test_report_passed_checks` | Access passed checks |
| `test_report_pass_rate` | Calculate pass rate |

##### `TestOPSAuditor`
| Test | Description |
|------|-------------|
| `test_audit_samples_within_budget` | Number of samples <= budget |
| `test_audit_checks_leaves` | Sufficiency check on leaf nodes |
| `test_audit_checks_internal` | Merge consistency on internal nodes |
| `test_audit_flags_failures` | Failed audits marked on nodes |
| `test_audit_records_scores` | Discrepancy scores recorded |
| `test_get_failed_nodes` | Retrieve all failed nodes |
| `test_audit_with_different_strategies` | Test various sampling strategies |
| `test_audit_returns_report` | Audit returns AuditReport |
| `test_audit_exhaustive` | Exhaustive audit (all nodes) |
| `test_audit_minimal` | Minimal audit (budget=2) |

##### `TestReviewQueue`
| Test | Description |
|------|-------------|
| `test_queue_creation` | Create empty review queue |
| `test_add_flagged_item` | Add item to queue |
| `test_queue_ordering` | Items ordered by priority |
| `test_get_next_item` | Get highest priority item |
| `test_mark_reviewed` | Mark item as reviewed |
| `test_queue_persistence` | Queue can be saved/loaded |

##### `TestFlaggedItem`
| Test | Description |
|------|-------------|
| `test_item_creation` | Create flagged item |
| `test_item_node_reference` | Item references flagged node |
| `test_item_reason` | Item has failure reason |
| `test_item_score` | Item has discrepancy score |
| `test_item_priority` | Priority calculated from score |

##### `TestAuditorReviewQueueIntegration`
| Test | Description |
|------|-------------|
| `test_audit_populates_queue` | Failed audits added to queue |
| `test_queue_has_correct_items` | Queue contains expected failures |
| `test_queue_priority_order` | Worst failures first |

---

#### `tests/ops_engine/test_optimizer.py`
**Purpose**: Tests for the optimization/fine-tuning system.

**Test Classes**:

##### `TestTrainingExample`
| Test | Description |
|------|-------------|
| `test_example_creation` | Create training example |
| `test_example_input_output` | Access input/output fields |
| `test_example_metadata` | Example has metadata |
| `test_example_serialization` | Serialize to dict |
| `test_example_deserialization` | Deserialize from dict |

##### `TestTrainingDataCollector`
| Test | Description |
|------|-------------|
| `test_collector_creation` | Create empty collector |
| `test_add_example` | Add example to collector |
| `test_get_examples` | Retrieve all examples |
| `test_filter_by_type` | Filter examples by type |
| `test_export_to_jsonl` | Export to JSONL format |

##### `TestOptimizationConfig`
| Test | Description |
|------|-------------|
| `test_default_config` | Default optimization settings |
| `test_custom_config` | Custom optimization settings |

##### `TestOPSOptimizer`
| Test | Description |
|------|-------------|
| `test_optimizer_creation` | Create optimizer |
| `test_collect_from_review_queue` | Collect examples from queue |
| `test_generate_training_data` | Generate training dataset |
| `test_optimization_workflow` | Full optimization workflow |

---

### Preprocessing Tests

#### `tests/preprocessing/test_chunker.py`
**Purpose**: Tests for text chunking functionality.

**Test Classes**:

##### `TestTextChunk`
| Test | Description |
|------|-------------|
| `test_chunk_creation` | Create TextChunk instance |
| `test_chunk_text_property` | Access chunk text |
| `test_chunk_char_interval` | Character positions in original doc |
| `test_chunk_start_end` | Start and end positions |
| `test_chunk_length` | Chunk length property |

##### `TestSentenceSplitter`
| Test | Description |
|------|-------------|
| `test_split_simple_sentences` | Basic period-separated sentences |
| `test_split_with_abbreviations` | Handle Dr., Mr., etc. |
| `test_split_with_ellipsis` | Handle ... correctly |
| `test_split_with_question_marks` | Split on ? |
| `test_split_with_exclamation` | Split on ! |
| `test_split_mixed_punctuation` | Mixed punctuation types |
| `test_split_empty_text` | Empty input returns empty list |
| `test_split_no_punctuation` | Text without punctuation |

##### `TestDocumentChunker`
| Test | Description |
|------|-------------|
| `test_chunk_short_text` | Text under limit returns single chunk |
| `test_chunk_at_sentence_boundary` | Chunks break at sentence ends |
| `test_chunk_respects_max_chars` | No chunk exceeds max_chars |
| `test_chunk_respects_min_chars` | Chunks meet minimum size |
| `test_chunk_preserves_all_text` | Joining chunks reproduces original |
| `test_chunk_with_overlap` | Overlap between adjacent chunks |
| `test_chunk_handles_long_word` | Word longer than max_chars handled |
| `test_chunk_empty_text` | Empty text returns empty list |
| `test_chunk_whitespace_only` | Whitespace-only returns empty list |
| `test_chunk_unicode` | Handle unicode correctly |
| `test_chunk_with_newlines` | Handle newlines in text |
| `test_chunk_configurable` | Chunker is configurable |

##### `TestParagraphChunker`
| Test | Description |
|------|-------------|
| `test_split_paragraphs` | Split on double newlines |
| `test_preserve_paragraph_structure` | Paragraph boundaries respected |
| `test_handle_single_paragraph` | Single paragraph handled |

##### `TestChunkForOps`
| Test | Description |
|------|-------------|
| `test_convenience_function` | `chunk_for_ops()` works correctly |
| `test_returns_text_chunks` | Returns list of TextChunk |
| `test_respects_parameters` | Parameters passed through |

##### `TestFileChunking`
| Test | Description |
|------|-------------|
| `test_chunk_from_file` | Chunk text from file path |
| `test_file_not_found` | Handle missing file |
| `test_file_encoding` | Handle different encodings |

---

## Langextract Tests

Location: `langextract/tests/`

The langextract test suite uses the `absl.testing` framework (absltest + parameterized) following Google-style testing patterns.

### Core Language Extraction

#### `annotation_test.py` (~41KB)
**Purpose**: Tests for the Annotator class and language model integration.

| Test Area | Description |
|-----------|-------------|
| Annotator Creation | Test annotator instantiation with various configs |
| Char Interval Matching | Verify character interval alignment with source |
| Source Text Alignment | Test alignment of extracted text with original |
| Extraction Data Structures | Test annotation data structure creation |
| Language Model Integration | Test LM calls during annotation |

#### `schema_test.py` (~12KB)
**Purpose**: Tests for schema definitions and validation.

| Test Area | Description |
|-----------|-------------|
| BaseSchema | Abstract class implementation tests |
| Schema Validation | Schema constraint validation |
| Schema Methods | Language model schema method tests |
| Field Definitions | Test field type definitions |

#### `resolver_test.py` (~83KB - largest test file)
**Purpose**: Comprehensive tests for text resolution and extraction matching.

| Test Area | Description |
|-----------|-------------|
| Text Resolution | Resolve extracted text to source positions |
| Char Interval Alignment | Complex interval alignment scenarios |
| Edge Cases | Boundary conditions and special cases |
| Performance | Large-scale resolution tests |

---

### Data & Format Handling

#### `chunking_test.py` (~19KB)
**Purpose**: Tests for sentence iteration and chunk management.

| Test Area | Description |
|-----------|-------------|
| Sentence Iteration | Test sentence boundary detection |
| Chunk Management | Chunk creation and positioning |
| Character Intervals | Track character positions |
| Chunk Splitting | Test various splitting strategies |

#### `data_lib_test.py` (~8.8KB)
**Purpose**: Tests for data structures and utilities.

| Test Area | Description |
|-----------|-------------|
| Data Structures | Core data class tests |
| Utility Functions | Helper function tests |
| Serialization | Data serialization/deserialization |

#### `format_handler_test.py` (~7.6KB)
**Purpose**: Tests for format handling capabilities.

| Test Area | Description |
|-----------|-------------|
| Format Detection | Detect input format types |
| Format Conversion | Convert between formats |
| Handler Registration | Test handler registration system |

#### `tokenizer_test.py` (~36KB)
**Purpose**: Comprehensive tokenization tests.

| Test Area | Description |
|-----------|-------------|
| Token Creation | Test token object creation |
| Tokenization | Text to token conversion |
| Token Operations | Token-level operations |
| Special Tokens | Handle special token types |

---

### Inference & Extraction

#### `inference_test.py` (~25KB)
**Purpose**: Tests for the inference module.

| Test Area | Description |
|-----------|-------------|
| Inference Workflows | End-to-end inference tests |
| Batch Processing | Batch inference handling |
| Model Integration | Language model integration |
| Result Processing | Inference result handling |

#### `extract_precedence_test.py` (~8.8KB)
**Purpose**: Tests for extraction precedence rules.

| Test Area | Description |
|-----------|-------------|
| Precedence Rules | Test extraction ordering rules |
| Conflict Resolution | Handle overlapping extractions |
| Priority Assignment | Test priority calculation |

#### `extract_schema_integration_test.py` (~11KB)
**Purpose**: Integration tests for extraction with schema.

| Test Area | Description |
|-----------|-------------|
| Schema Application | Apply schema to extraction |
| Validation | Validate extractions against schema |
| End-to-End | Full extraction pipeline tests |

#### `prompting_test.py` (~13KB)
**Purpose**: Tests for prompt generation.

| Test Area | Description |
|-----------|-------------|
| QAPromptGenerator | Question-answer prompt generation |
| PromptTemplateStructured | Structured prompt templates |
| Template Rendering | Test template variable substitution |
| Prompt Validation | Validate generated prompts |

#### `prompt_validation_test.py` (~13KB)
**Purpose**: Tests for prompt validation.

| Test Area | Description |
|-----------|-------------|
| Validation Rules | Test validation rule application |
| Error Detection | Detect invalid prompts |
| Correction Suggestions | Test correction recommendations |

---

### Provider & Plugin Tests

#### `factory_test.py` (~15KB)
**Purpose**: Tests for factory module and provider instantiation.

| Test Area | Description |
|-----------|-------------|
| Provider Factory | Test provider creation |
| FakeGeminiProvider | Mock Gemini provider for testing |
| FakeOpenAIProvider | Mock OpenAI provider for testing |
| Registry Integration | Test registry module (deprecated) |

#### `factory_schema_test.py` (~7.8KB)
**Purpose**: Tests for schema creation from examples.

| Test Area | Description |
|-----------|-------------|
| Schema From Examples | Create schema from example data |
| Schema Validation | Validate created schemas |
| Schema Application | Apply schemas to extraction |

#### `provider_schema_test.py` (~19KB)
**Purpose**: Tests for provider-specific schema implementations.

| Test Area | Description |
|-----------|-------------|
| Provider Schemas | Test per-provider schema handling |
| Schema Conversion | Convert between schema formats |
| Provider Compatibility | Test cross-provider compatibility |

#### `provider_plugin_test.py` (~22KB)
**Purpose**: Tests for the plugin architecture.

| Test Area | Description |
|-----------|-------------|
| Plugin Registration | Test plugin registration system |
| Plugin Loading | Test dynamic plugin loading |
| Plugin Lifecycle | Test plugin init/cleanup |
| Plugin Discovery | Test automatic plugin discovery |

#### `registry_test.py` (~7.6KB)
**Purpose**: Tests for registry module (now aliased to router).

| Test Area | Description |
|-----------|-------------|
| Registration | Test provider registration |
| Lookup | Test provider lookup |
| Aliasing | Test router aliasing |

#### `init_test.py` (~23KB)
**Purpose**: Tests for module initialization.

| Test Area | Description |
|-----------|-------------|
| Package Imports | Test package-level imports |
| Module Setup | Test module initialization |
| Public API | Test public API availability |

---

### API & Integration Tests

#### `test_live_api.py` (~30KB)
**Purpose**: Integration tests with live API endpoints.

| Test Area | Description |
|-----------|-------------|
| Live API Calls | Tests against real API (requires keys) |
| Provider Tests | Test each supported provider |
| Error Handling | Test API error handling |
| Rate Limiting | Test rate limit handling |

**Note**: These tests require API keys and are typically skipped in CI.

#### `test_gemini_batch_api.py` (~22KB)
**Purpose**: Tests for Google Gemini batch API.

| Test Area | Description |
|-----------|-------------|
| Batch Requests | Test batch request creation |
| Batch Processing | Test batch execution |
| Result Handling | Test batch result parsing |
| Error Recovery | Test batch error handling |

#### `test_ollama_integration.py` (~3.4KB)
**Purpose**: Integration tests with Ollama local LLM.

| Test Area | Description |
|-----------|-------------|
| Ollama Connection | Test connection to Ollama |
| Model Loading | Test model loading |
| Inference | Test local inference |

**Note**: Requires Ollama to be running locally.

#### `test_kwargs_passthrough.py` (~15KB)
**Purpose**: Tests for keyword argument passing through pipeline.

| Test Area | Description |
|-----------|-------------|
| Kwargs Propagation | Test argument passing through layers |
| Override Handling | Test argument overrides |
| Default Values | Test default argument handling |

---

### Utility Tests

#### `visualization_test.py` (~5.1KB)
**Purpose**: Tests for visualization utilities.

| Test Area | Description |
|-----------|-------------|
| Visualization Creation | Test visualization generation |
| Output Formats | Test different output formats |
| Rendering | Test rendering logic |

#### `progress_test.py` (~2.7KB)
**Purpose**: Tests for progress tracking.

| Test Area | Description |
|-----------|-------------|
| Progress Tracking | Test progress bar/reporting |
| Callbacks | Test progress callbacks |
| Completion | Test completion detection |

---

## Running Tests

### Main Project Tests (pytest)

```bash
# Run all tests
pytest tests/ -v

# Run specific module
pytest tests/core/test_data_models.py -v

# Run specific test class
pytest tests/ops_engine/test_builder.py::TestOPSTreeBuilder -v

# Run specific test
pytest tests/ops_engine/test_builder.py::TestOPSTreeBuilder::test_build_single_chunk -v

# Run with coverage
pytest tests/ --cov=src --cov-report=html

# Run property tests only
pytest tests/ -v -m property

# Run fast tests only (exclude slow integration tests)
pytest tests/ -v -m "not slow"

# Run with verbose output
pytest tests/ -v --tb=long
```

### Langextract Tests (absltest)

```bash
# Run all langextract tests
python -m pytest langextract/tests/ -v

# Or using absltest runner
python -m langextract.tests.annotation_test

# Run specific test file
python -m pytest langextract/tests/schema_test.py -v

# Skip live API tests
python -m pytest langextract/tests/ -v --ignore=langextract/tests/test_live_api.py
```

### Running Both Test Suites

```bash
# Run everything
pytest . -v

# Run with coverage for all
pytest . --cov=src --cov=langextract --cov-report=html
```

---

## Test Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `TEST_LLM_ENDPOINT` | LLM endpoint for integration tests | None |
| `TEST_API_KEY` | API key for live tests | None |
| `SKIP_SLOW_TESTS` | Skip slow integration tests | False |
| `OLLAMA_HOST` | Ollama host for local tests | localhost:11434 |

### pytest.ini / pyproject.toml Configuration

```ini
[pytest]
testpaths = tests langextract/tests
python_files = test_*.py *_test.py
python_classes = Test*
python_functions = test_*
markers =
    slow: marks tests as slow (deselect with '-m "not slow"')
    property: marks property-based tests
    integration: marks integration tests
    live_api: marks tests requiring live API access
```

### Test Data Location

- Main project: `tests/config/test_settings.yaml`
- Sample texts: Defined in `tests/conftest.py` fixtures
- Mock responses: Configured per-test or in fixtures

---

## Coverage Goals

| Module | Target | Notes |
|--------|--------|-------|
| `core/data_models.py` | 95% | Core data structures |
| `preprocessing/chunker.py` | 90% | Text processing |
| `ops_engine/builder.py` | 85% | Tree construction |
| `ops_engine/auditor.py` | 85% | Audit system |
| `ops_engine/optimizer.py` | 80% | Optimization |
| `core/llm_client.py` | 80% | Mock-heavy testing |
| `langextract/*` | 80% | Language extraction |

---

## Adding New Tests

### Main Project (pytest)

1. Create test file in appropriate directory (`tests/core/`, `tests/ops_engine/`, etc.)
2. Name file `test_<module>.py`
3. Create test classes prefixed with `Test`
4. Create test methods prefixed with `test_`
5. Use fixtures from `conftest.py` where appropriate
6. Add markers for slow/integration tests

### Langextract (absltest)

1. Create test file in `langextract/tests/`
2. Name file `<module>_test.py`
3. Import `from absl.testing import absltest`
4. Create test classes inheriting from `absltest.TestCase`
5. Use `@parameterized.parameters` for parameterized tests
