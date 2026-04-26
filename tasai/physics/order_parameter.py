"""
Order Parameter Models for Phase Transitions

ANDiE-style physics models for autonomous diffraction experiments
studying phase transitions. These models predict the temperature dependence
of order parameters (e.g., magnetic order, structural distortion).

Reference: McDannald et al., Appl. Phys. Rev. 9, 021408 (2022)

Models included:
- IsingModel: M ∝ (1 - T/Tc)^β with β ≈ 0.325 (3D Ising)
- WeissModel: Mean-field with β = 0.5
- FirstOrderModel: Discontinuous transition
- PowerLawModel: General power law with tunable exponent
- LandauModel: Landau theory expansion

For TAS, the "E" coordinate represents temperature, and h,k,l select
which Bragg peak or Q-position to monitor.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple

from .base import PhysicsModel, Parameter


class OrderParameterModel(PhysicsModel):
    """
    Base class for order parameter models.
    
    These models predict intensity at a Bragg position as a function
    of temperature (passed as the E coordinate).
    
    I(T) ∝ M(T)² for magnetic Bragg peaks
    
    The h,k,l coordinates specify which reflection is being monitored,
    which can affect the overall intensity scale.
    """
    
    def __init__(self,
                 T_c: float = 100.0,
                 intensity_scale: float = 1000.0,
                 background: float = 1.0):
        """
        Initialize order parameter model.
        
        Parameters
        ----------
        T_c : float
            Critical temperature (K)
        intensity_scale : float
            Overall intensity scale factor
        background : float
            Temperature-independent background
        """
        super().__init__()
        
        # Common parameters
        self._parameters = [
            Parameter('T_c', T_c, bounds=(1.0, 1000.0), 
                     units='K', description='Critical temperature'),
            Parameter('scale', intensity_scale, bounds=(0.1, 1e6),
                     units='counts/s', description='Intensity scale'),
            Parameter('background', background, bounds=(0.0, 1e4),
                     units='counts/s', description='Background'),
        ]
        
        self._param_index = {p.name: i for i, p in enumerate(self._parameters)}
    
    def order_parameter(self, T: float) -> float:
        """
        Compute order parameter M(T).
        
        To be implemented by subclasses.
        Returns value between 0 (disordered) and 1 (fully ordered).
        """
        raise NotImplementedError
    
    def compute_intensity(self, h: float, k: float, l: float, E: float, **kwargs) -> float:
        """
        Compute intensity at temperature T (passed as E).
        
        I = scale * M(T)² + background
        """
        T = E  # Temperature passed as "energy" coordinate
        T_c = self.get_parameter('T_c').value
        scale = self.get_parameter('scale').value
        background = self.get_parameter('background').value
        
        M = self.order_parameter(T)
        
        # Intensity proportional to M² for magnetic peaks
        return scale * M**2 + background
    
    def compute_intensity_array(self, h: np.ndarray, k: np.ndarray,
                                 l: np.ndarray, E: np.ndarray, **kwargs) -> np.ndarray:
        """Vectorized intensity calculation."""
        T = E
        T_c = self.get_parameter('T_c').value
        scale = self.get_parameter('scale').value
        background = self.get_parameter('background').value
        
        M = np.array([self.order_parameter(t) for t in T])
        return scale * M**2 + background


class IsingModel(OrderParameterModel):
    """
    3D Ising model order parameter.
    
    M(T) = M₀ (1 - T/Tc)^β  for T < Tc
    M(T) = 0                 for T ≥ Tc
    
    The 3D Ising universality class has β ≈ 0.325.
    """
    
    def __init__(self,
                 T_c: float = 100.0,
                 beta: float = 0.325,
                 intensity_scale: float = 1000.0,
                 background: float = 1.0):
        super().__init__(T_c, intensity_scale, background)
        
        # Add Ising-specific parameter
        self._parameters.append(
            Parameter('beta', beta, bounds=(0.1, 0.5), fixed=True,
                     description='Critical exponent (Ising ≈ 0.325)')
        )
        self._param_index['beta'] = len(self._parameters) - 1
    
    def order_parameter(self, T: float) -> float:
        T_c = self.get_parameter('T_c').value
        beta = self.get_parameter('beta').value
        
        if T >= T_c:
            return 0.0
        
        # Avoid numerical issues near T_c
        reduced_T = max(1e-6, 1 - T / T_c)
        return reduced_T ** beta
    
    def description(self) -> str:
        return "3D Ising model: M = (1 - T/Tc)^β"


class WeissModel(OrderParameterModel):
    """
    Mean-field (Weiss) model order parameter.
    
    M(T) = M₀ (1 - T/Tc)^β  for T < Tc, with β = 0.5 (mean-field)
    
    Also known as Landau mean-field theory.
    """
    
    def __init__(self,
                 T_c: float = 100.0,
                 intensity_scale: float = 1000.0,
                 background: float = 1.0):
        super().__init__(T_c, intensity_scale, background)
        
        # Mean-field has fixed β = 0.5
        self._parameters.append(
            Parameter('beta', 0.5, bounds=(0.5, 0.5), fixed=True,
                     description='Mean-field exponent (fixed at 0.5)')
        )
        self._param_index['beta'] = len(self._parameters) - 1
    
    def order_parameter(self, T: float) -> float:
        T_c = self.get_parameter('T_c').value
        
        if T >= T_c:
            return 0.0
        
        reduced_T = max(1e-6, 1 - T / T_c)
        return np.sqrt(reduced_T)  # β = 0.5
    
    def description(self) -> str:
        return "Mean-field (Weiss) model: M = √(1 - T/Tc)"


class FirstOrderModel(OrderParameterModel):
    """
    First-order phase transition model.
    
    Order parameter has a discontinuous jump at T_c.
    Includes thermal hysteresis with separate heating/cooling temperatures.
    
    M(T) = M_high  for T < T_low
    M(T) = M_low   for T > T_high
    M(T) = interpolated for T_low < T < T_high (coexistence region)
    """
    
    def __init__(self,
                 T_c: float = 100.0,
                 T_width: float = 5.0,
                 M_high: float = 1.0,
                 M_low: float = 0.0,
                 intensity_scale: float = 1000.0,
                 background: float = 1.0):
        super().__init__(T_c, intensity_scale, background)
        
        # First-order specific parameters
        self._parameters.extend([
            Parameter('T_width', T_width, bounds=(0.1, 50.0),
                     units='K', description='Transition width (hysteresis)'),
            Parameter('M_high', M_high, bounds=(0.0, 1.0),
                     description='Order parameter below transition'),
            Parameter('M_low', M_low, bounds=(0.0, 1.0),
                     description='Order parameter above transition'),
        ])
        
        for p in self._parameters[-3:]:
            self._param_index[p.name] = len(self._parameters) - 1
        
        # Rebuild index
        self._param_index = {p.name: i for i, p in enumerate(self._parameters)}
    
    def order_parameter(self, T: float) -> float:
        T_c = self.get_parameter('T_c').value
        T_width = self.get_parameter('T_width').value
        M_high = self.get_parameter('M_high').value
        M_low = self.get_parameter('M_low').value
        
        T_low = T_c - T_width / 2
        T_high = T_c + T_width / 2
        
        if T <= T_low:
            return M_high
        elif T >= T_high:
            return M_low
        else:
            # Sigmoid interpolation in coexistence region
            x = (T - T_low) / T_width
            return M_high + (M_low - M_high) * (3*x**2 - 2*x**3)
    
    def description(self) -> str:
        return "First-order transition: discontinuous jump at Tc"


class PowerLawModel(OrderParameterModel):
    """
    General power-law order parameter model.
    
    M(T) = M₀ |1 - T/Tc|^β
    
    Allows fitting of the critical exponent β as a free parameter.
    Different universality classes have different β:
    - 3D Ising: β ≈ 0.325
    - 3D XY: β ≈ 0.345
    - 3D Heisenberg: β ≈ 0.365
    - Mean field: β = 0.5
    - 2D Ising: β = 0.125
    """
    
    def __init__(self,
                 T_c: float = 100.0,
                 beta: float = 0.33,
                 intensity_scale: float = 1000.0,
                 background: float = 1.0):
        super().__init__(T_c, intensity_scale, background)
        
        # β is a free parameter
        self._parameters.append(
            Parameter('beta', beta, bounds=(0.05, 0.6), fixed=False,
                     description='Critical exponent')
        )
        self._param_index['beta'] = len(self._parameters) - 1
    
    def order_parameter(self, T: float) -> float:
        T_c = self.get_parameter('T_c').value
        beta = self.get_parameter('beta').value
        
        if T >= T_c:
            return 0.0
        
        reduced_T = max(1e-6, 1 - T / T_c)
        return reduced_T ** beta
    
    def description(self) -> str:
        beta = self.get_parameter('beta').value
        return f"Power-law model: M = (1 - T/Tc)^{beta:.3f}"


class LandauModel(OrderParameterModel):
    """
    Landau free energy expansion model.
    
    F = a(T-Tc)M² + bM⁴ + cM⁶
    
    Minimizing F gives M(T). This model can describe both
    second-order (b > 0) and first-order (b < 0, c > 0) transitions.
    """
    
    def __init__(self,
                 T_c: float = 100.0,
                 a: float = 1.0,
                 b: float = 1.0,
                 c: float = 0.0,
                 intensity_scale: float = 1000.0,
                 background: float = 1.0):
        super().__init__(T_c, intensity_scale, background)
        
        # Landau coefficients
        self._parameters.extend([
            Parameter('a', a, bounds=(0.01, 10.0),
                     description='Quadratic coefficient'),
            Parameter('b', b, bounds=(-10.0, 10.0),
                     description='Quartic coefficient'),
            Parameter('c', c, bounds=(0.0, 10.0),
                     description='Sextic coefficient'),
        ])
        
        self._param_index = {p.name: i for i, p in enumerate(self._parameters)}
    
    def order_parameter(self, T: float) -> float:
        T_c = self.get_parameter('T_c').value
        a = self.get_parameter('a').value
        b = self.get_parameter('b').value
        c = self.get_parameter('c').value
        
        # Coefficient of M² term
        alpha = a * (T - T_c)
        
        if c == 0:
            # Quartic Landau (second-order transition)
            if alpha >= 0:
                return 0.0
            else:
                # M² = -α/(2b)
                return np.sqrt(-alpha / (2 * b)) if b > 0 else 0.0
        else:
            # Sextic Landau (can be first-order)
            # Solve: α + 2bM² + 3cM⁴ = 0 for M²
            # Let x = M², solve: 3cx² + 2bx + α = 0
            discriminant = 4*b**2 - 12*c*alpha
            
            if discriminant < 0:
                return 0.0
            
            x1 = (-2*b + np.sqrt(discriminant)) / (6*c)
            x2 = (-2*b - np.sqrt(discriminant)) / (6*c)
            
            # Take the positive solution that minimizes F
            candidates = [x for x in [x1, x2] if x > 0]
            if not candidates:
                return 0.0
            
            # Choose minimum F solution
            def free_energy(M2):
                return alpha * M2 + b * M2**2 + c * M2**3
            
            best_M2 = min(candidates, key=lambda x: free_energy(x))
            return np.sqrt(best_M2)
    
    def description(self) -> str:
        return "Landau model: F = a(T-Tc)M² + bM⁴ + cM⁶"


class BKTModel(OrderParameterModel):
    """
    Berezinskii-Kosterlitz-Thouless (BKT) transition model.
    
    For 2D XY systems. The transition is infinite-order, not power-law.
    
    Correlation length: ξ ∝ exp(b / √(T - T_BKT))
    Order parameter approximated from finite-size effects.
    """
    
    def __init__(self,
                 T_c: float = 100.0,
                 b: float = 1.5,
                 intensity_scale: float = 1000.0,
                 background: float = 1.0):
        super().__init__(T_c, intensity_scale, background)
        
        self._parameters.append(
            Parameter('b', b, bounds=(0.1, 5.0),
                     description='BKT correlation length coefficient')
        )
        self._param_index['b'] = len(self._parameters) - 1
    
    def order_parameter(self, T: float) -> float:
        T_c = self.get_parameter('T_c').value
        b = self.get_parameter('b').value
        
        if T <= T_c:
            return 1.0  # Quasi-long-range order
        
        # Exponential decay above T_BKT
        delta_T = T - T_c
        xi = np.exp(b / np.sqrt(delta_T))
        
        # Order parameter decays with system size / correlation length
        # Using a smooth approximation
        return np.exp(-1.0 / xi)
    
    def description(self) -> str:
        return "BKT transition: ξ ∝ exp(b/√(T-Tc))"


# =============================================================================
# Factory function for creating model ensembles
# =============================================================================

def create_andie_ensemble(T_c_estimate: float = 100.0,
                          T_c_range: Tuple[float, float] = None,
                          include_bkt: bool = False) -> List[PhysicsModel]:
    """
    Create an ensemble of competing models for ANDiE-style analysis.
    
    Parameters
    ----------
    T_c_estimate : float
        Initial estimate of critical temperature
    T_c_range : tuple, optional
        (min, max) bounds for T_c. Default: ±20% of estimate
    include_bkt : bool
        Include BKT model (for 2D systems)
    
    Returns
    -------
    list of PhysicsModel
        Ensemble of competing physics models
    """
    if T_c_range is None:
        T_c_range = (T_c_estimate * 0.8, T_c_estimate * 1.2)
    
    models = [
        IsingModel(T_c=T_c_estimate),
        WeissModel(T_c=T_c_estimate),
        FirstOrderModel(T_c=T_c_estimate),
        PowerLawModel(T_c=T_c_estimate),
    ]
    
    if include_bkt:
        models.append(BKTModel(T_c=T_c_estimate))
    
    # Set T_c bounds for all models
    for model in models:
        model.get_parameter('T_c').bounds = T_c_range
    
    return models


# =============================================================================
# Utility: Generate simulated data
# =============================================================================

def generate_synthetic_data(model: OrderParameterModel,
                            T_values: np.ndarray,
                            noise_level: float = 0.05,
                            seed: int = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate synthetic intensity data from an order parameter model.
    
    Parameters
    ----------
    model : OrderParameterModel
        Model to generate data from
    T_values : np.ndarray
        Temperatures at which to generate data
    noise_level : float
        Relative noise level (0.05 = 5% noise)
    seed : int, optional
        Random seed for reproducibility
    
    Returns
    -------
    T : np.ndarray
        Temperatures
    I : np.ndarray
        Intensities
    sigma : np.ndarray
        Uncertainties
    """
    if seed is not None:
        np.random.seed(seed)
    
    # Dummy h,k,l values (not used by order parameter models)
    h = np.zeros_like(T_values)
    k = np.zeros_like(T_values)
    l = np.zeros_like(T_values)
    
    # Get true intensities
    I_true = model.compute_intensity_array(h, k, l, T_values)
    
    # Add Poisson-like noise
    sigma = noise_level * np.sqrt(I_true + 1)
    I_noisy = I_true + np.random.normal(0, sigma)
    I_noisy = np.maximum(I_noisy, 0)  # Intensity can't be negative
    
    return T_values, I_noisy, sigma
