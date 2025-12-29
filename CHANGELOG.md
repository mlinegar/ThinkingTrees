# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2024-12-21

### Added
- Initial release
- OPS (Oracle-Preserving Summarization) tree-based document summarization
- Oracle-aligned training with DSPy optimizers (GEPA, MIPROv2, BootstrapRandomSearch)
- Batched pipeline for high-throughput document processing
- Manifesto RILE (Right-Left) prediction experiments
- Thread-safe oracle prediction caching for optimization efficiency
- Configuration profiles for different compute environments

### Performance
- Reduced default optimizer settings for faster training (~8 hours vs 30+ hours)
  - `num_candidates`: 16 → 6
  - `num_threads`: 128 → 64
  - `n_iterations`: 3 → 2
- Added `create_cached_oracle_metric()` to eliminate redundant oracle LLM calls

### Documentation
- Added pyproject.toml for modern Python packaging
- Added MIT LICENSE
- Added CHANGELOG.md
