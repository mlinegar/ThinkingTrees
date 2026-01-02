"""
Training pipeline module for OPS optimization.

This module provides CLI entry points for running training pipelines,
integrating the OPS training framework with task + dataset plugins.

Key Components:
    - run_training_pipeline: Main training pipeline
    - JudgeOptimizer: Single-pass judge optimization
    - TournamentOfTournamentsTrainer: Full iterative judge optimization loop
    - create_judge_trainset: Create training data for judge optimization
    - collect_preferences: Unified preference collection CLI

Preference Collection:
    - PreferenceDataSource: Protocol for data sources
    - DirectDocumentSource: Load from task data loader
    - LabeledTreeSource: Load from labeled trees (oracle-scored)
    - SyntheticDataSource: Load from JSONL/JSON files
    - PreferenceCollectionConfig: Configuration dataclasses
"""

# Pipeline entry points
from src.training.run_pipeline import run_training_pipeline, main

# Judge optimization (single pass)
from src.training.judge_optimization import (
    JudgeOptimizer,
    JudgeOptimizationConfig,
    create_judge_trainset,
    SkippedReasons,
    optimize_judge_from_preferences,
    load_optimized_judge,
)

# Tournament of Tournaments (full iterative loop)
from src.training.tournament_loop import (
    TournamentOfTournamentsTrainer,
    ToTConfig,
    ToTResult,
    run_tournament_of_tournaments,
)

# Unified preference collection
from src.training.collect_preferences import main as collect_preferences_main

# Alias for backward compatibility
collect_preferences = collect_preferences_main

# Data sources
from src.training.data_sources import (
    PreferenceDataSource,
    DirectDocumentSource,
    LabeledTreeSource,
    SyntheticDataSource,
    DataSourceExample,
    create_data_source,
)

# Preference config
from src.training.preference_config import (
    PreferenceCollectionConfig,
    JudgeType,
    DataSourceType,
    ServerConfig,
    GenerationSettings,
    JudgeSettings,
    DataSourceSettings,
)

__all__ = [
    'run_training_pipeline',
    'main',
    # Judge optimization (single pass)
    'JudgeOptimizer',
    'JudgeOptimizationConfig',
    'create_judge_trainset',
    'SkippedReasons',
    'optimize_judge_from_preferences',
    'load_optimized_judge',
    # Tournament of Tournaments (full iterative loop)
    'TournamentOfTournamentsTrainer',
    'ToTConfig',
    'ToTResult',
    'run_tournament_of_tournaments',
    # Unified preference collection
    'collect_preferences',
    'collect_preferences_main',
    # Data sources
    'PreferenceDataSource',
    'DirectDocumentSource',
    'LabeledTreeSource',
    'SyntheticDataSource',
    'DataSourceExample',
    'create_data_source',
    # Preference config
    'PreferenceCollectionConfig',
    'JudgeType',
    'DataSourceType',
    'ServerConfig',
    'GenerationSettings',
    'JudgeSettings',
    'DataSourceSettings',
]
