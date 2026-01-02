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


def __getattr__(name):
    """Lazy import to avoid circular import issues when running with -m flag."""
    if name == 'run_training_pipeline':
        from .run_pipeline import run_training_pipeline
        return run_training_pipeline
    elif name == 'main':
        from .run_pipeline import main
        return main
    elif name == 'JudgeOptimizer':
        from .judge_optimization import JudgeOptimizer
        return JudgeOptimizer
    elif name == 'JudgeOptimizationConfig':
        from .judge_optimization import JudgeOptimizationConfig
        return JudgeOptimizationConfig
    elif name == 'create_judge_trainset':
        from .judge_optimization import create_judge_trainset
        return create_judge_trainset
    elif name == 'SkippedReasons':
        from .judge_optimization import SkippedReasons
        return SkippedReasons
    elif name == 'optimize_judge_from_preferences':
        from .judge_optimization import optimize_judge_from_preferences
        return optimize_judge_from_preferences
    elif name == 'load_optimized_judge':
        from .judge_optimization import load_optimized_judge
        return load_optimized_judge
    # Tournament of Tournaments exports
    elif name == 'TournamentOfTournamentsTrainer':
        from .tournament_loop import TournamentOfTournamentsTrainer
        return TournamentOfTournamentsTrainer
    elif name == 'ToTConfig':
        from .tournament_loop import ToTConfig
        return ToTConfig
    elif name == 'ToTResult':
        from .tournament_loop import ToTResult
        return ToTResult
    elif name == 'run_tournament_of_tournaments':
        from .tournament_loop import run_tournament_of_tournaments
        return run_tournament_of_tournaments
    # Unified preference collection exports
    elif name == 'collect_preferences':
        from .collect_preferences import main
        return main
    elif name == 'collect_preferences_main':
        from .collect_preferences import main
        return main
    # Data source exports
    elif name == 'PreferenceDataSource':
        from .data_sources import PreferenceDataSource
        return PreferenceDataSource
    elif name == 'DirectDocumentSource':
        from .data_sources import DirectDocumentSource
        return DirectDocumentSource
    elif name == 'LabeledTreeSource':
        from .data_sources import LabeledTreeSource
        return LabeledTreeSource
    elif name == 'SyntheticDataSource':
        from .data_sources import SyntheticDataSource
        return SyntheticDataSource
    elif name == 'DataSourceExample':
        from .data_sources import DataSourceExample
        return DataSourceExample
    elif name == 'create_data_source':
        from .data_sources import create_data_source
        return create_data_source
    # Preference config exports
    elif name == 'PreferenceCollectionConfig':
        from .preference_config import PreferenceCollectionConfig
        return PreferenceCollectionConfig
    elif name == 'JudgeType':
        from .preference_config import JudgeType
        return JudgeType
    elif name == 'DataSourceType':
        from .preference_config import DataSourceType
        return DataSourceType
    elif name == 'ServerConfig':
        from .preference_config import ServerConfig
        return ServerConfig
    elif name == 'GenerationSettings':
        from .preference_config import GenerationSettings
        return GenerationSettings
    elif name == 'JudgeSettings':
        from .preference_config import JudgeSettings
        return JudgeSettings
    elif name == 'DataSourceSettings':
        from .preference_config import DataSourceSettings
        return DataSourceSettings
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


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
