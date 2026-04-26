"""
Instrument interfaces for TAS-AI.

Provides:
- InstrumentInterface: Abstract interface for TAS instruments
- TASSimulator: Simulated TAS instrument for testing
- InstrumentProxy: HTTP proxy for real instruments
- Motor models: TAS motor motion simulation
- Resolution: Cooper-Nathans TAS resolution via rescalculator
"""

from .base import InstrumentInterface, TASGeometry, MeasurementPoint, MeasurementResult
from .simulator import TASSimulator
from .proxy import NICEProxyClient
from .motors import (
    MotorConfig,
    TASMotorSystem,
    SimplifiedMotorModel,
    MotionAwareAcquisition,
    create_hhl_motor_system,
    create_simple_motor_model,
)
__all__ = [
    'InstrumentInterface',
    'TASGeometry',
    'MeasurementPoint',
    'MeasurementResult',
    'TASSimulator',
    'NICEProxyClient',
    'MotorConfig',
    'TASMotorSystem',
    'SimplifiedMotorModel',
    'MotionAwareAcquisition',
    'create_hhl_motor_system',
    'create_simple_motor_model',
]

# Optional resolution module (requires rescalculator)
try:
    from .resolution import (
        TASResolutionCalculator,
        create_default_tas_config,
    )
    __all__.extend(['TASResolutionCalculator', 'create_default_tas_config'])
except ImportError:
    pass  # rescalculator not installed
