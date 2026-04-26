"""
TAS-AI Physics Models

Pluggable physics models for autonomous experiments.
No external dependencies on refl1d or other reflectometry packages.
"""

from .base import (
    PhysicsModel,
    Parameter,
    CompositeModel,
)

from .order_parameter import (
    OrderParameterModel,
    IsingModel,
    WeissModel,
    FirstOrderModel,
    PowerLawModel,
    LandauModel,
    BKTModel,
    create_andie_ensemble,
    generate_synthetic_data,
)

from .square_lattice_afm import SquareLatticeAFM
from .bilayer_ferromagnet import SquareFMBilayer

# Spin wave models (optional - requires pyspinw or juliacall)
try:
    from .spinwave import (
        SpinWaveModel,
        SpinWaveConfig,
        TabularSpinWaveModel,
        get_available_backends,
        compare_backends,
    )
    _SPINWAVE_AVAILABLE = True
except ImportError:
    _SPINWAVE_AVAILABLE = False

__all__ = [
    # Base
    'PhysicsModel',
    'Parameter',
    'CompositeModel',
    # Order parameter models
    'OrderParameterModel',
    'IsingModel',
    'WeissModel',
    'FirstOrderModel',
    'PowerLawModel',
    'LandauModel',
    'BKTModel',
    'create_andie_ensemble',
    'generate_synthetic_data',
    # Analytic spin-wave backends used in the paper's closed-loop pilots
    'SquareLatticeAFM',
    'SquareFMBilayer',
]

# Add spin wave exports if available
if _SPINWAVE_AVAILABLE:
    __all__.extend([
        'SpinWaveModel',
        'SpinWaveConfig',
        'TabularSpinWaveModel',
        'get_available_backends',
        'compare_backends',
    ])
