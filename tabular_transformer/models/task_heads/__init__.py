"""
Task-specific heads for tabular transformer.

This module imports all task-specific head implementations
for use with the tabular transformer model.
"""

from tabular_transformer.models.task_heads.base import BaseTaskHead
from tabular_transformer.models.task_heads.classification import ClassificationHead
from tabular_transformer.models.task_heads.regression import RegressionHead
from tabular_transformer.models.task_heads.survival import SurvivalHead
from tabular_transformer.models.task_heads.competing_risks import CompetingRisksHead
from tabular_transformer.models.task_heads.count import CountHead
from tabular_transformer.models.task_heads.clustering import ClusteringHead
from tabular_transformer.models.task_heads.multi_task import MultiTaskHead

__all__ = [
    'BaseTaskHead',
    'ClassificationHead',
    'RegressionHead',
    'SurvivalHead',
    'CompetingRisksHead',
    'CountHead',
    'ClusteringHead',
    'MultiTaskHead',
]
