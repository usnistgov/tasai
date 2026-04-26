"""
TAS-AI Core Module

Contains:
- acquisition: Acquisition functions (HH, ANDiE, Uncertainty, Composite)
- entropy: Entropy estimation for information gain calculations
- forecast: Forecasting for selecting multiple future measurements
"""

from .acquisition import (
    AcquisitionFunction,
    AcquisitionResult,
    HHAcquisition,
    UncertaintyAcquisition,
    ANDiEAcquisition,
    CompositeAcquisition,
)

from .entropy import (
    differential_entropy,
    differential_entropy_knn,
    differential_entropy_kde,
    joint_entropy,
    conditional_entropy,
    mutual_information,
    expected_information_gain,
    information_rate,
    effective_sample_size,
    resample_posterior,
    gaussian_log_likelihood,
)

from .forecast import (
    Forecaster,
    ForecastPoint,
    ForecastResult,
    ExperimentRunner,
    grid_candidate_generator,
    random_candidate_generator,
)

__all__ = [
    # Acquisition
    'AcquisitionFunction',
    'AcquisitionResult',
    'HHAcquisition',
    'UncertaintyAcquisition',
    'ANDiEAcquisition',
    'CompositeAcquisition',
    # Entropy
    'differential_entropy',
    'differential_entropy_knn',
    'differential_entropy_kde',
    'joint_entropy',
    'conditional_entropy',
    'mutual_information',
    'expected_information_gain',
    'information_rate',
    'effective_sample_size',
    'resample_posterior',
    'gaussian_log_likelihood',
    # Forecasting
    'Forecaster',
    'ForecastPoint',
    'ForecastResult',
    'ExperimentRunner',
    'grid_candidate_generator',
    'random_candidate_generator',
]

# Gaussian Process components
from .gaussian_process import (
    LogGaussianProcess,
    AgnosticExplorer,
    HybridExplorer,
)

# MCTS components
from .mcts import (
    MCTSConfig,
    MCTSNode,
    MCTSPlanner,
    MCTSModelDiscrimination,
)

# Extend exports with GP + MCTS helpers
__all__.extend([
    'LogGaussianProcess',
    'AgnosticExplorer',
    'HybridExplorer',
    'MCTSConfig',
    'MCTSNode',
    'MCTSPlanner',
    'MCTSModelDiscrimination',
])
