"""
TAS Simulator

Simulates a triple-axis spectrometer for development and testing.
Can use analytic functions or pre-computed S(Q,ω) data.

Supports realistic resolution convolution via rescalculator integration.
"""

import numpy as np
from typing import Callable, Optional, Dict, List, Union, TYPE_CHECKING
import time
import logging

from .base import (
    InstrumentInterface, MeasurementPoint, MeasurementResult,
    TASGeometry, estimate_move_time_from_angles
)

if TYPE_CHECKING:
    from .resolution import TASResolutionCalculator

logger = logging.getLogger(__name__)


class TASSimulator(InstrumentInterface):
    """
    Simulated TAS instrument for testing autonomous algorithms.

    Features:
    - Configurable S(Q,ω) function (analytic or tabulated)
    - Realistic noise model (Poisson counting statistics)
    - Timing simulation with optional speedup
    - Resolution convolution (simple Gaussian or full Cooper-Nathans via rescalculator)

    Example with realistic resolution (40' collimations, fixed Ef=14.7 meV):

        from tasai.instrument import TASResolutionCalculator, create_default_tas_config

        # Create resolution calculator with 40' collimations
        res_calc = TASResolutionCalculator(
            lattice_params=(5.0, 5.0, 5.0, 90, 90, 90),
            orient1=[1, 0, 0],
            orient2=[0, 1, 0],
            exp_config=create_default_tas_config(
                efixed=14.7,
                hcol=(40, 40, 40, 40)
            )
        )

        # Create simulator with realistic resolution
        sim = TASSimulator(
            geometry=geometry,
            sqw_function=my_sqw,
            resolution_calculator=res_calc
        )
    """

    def __init__(self,
                 geometry: TASGeometry,
                 sqw_function: Callable[[float, float, float, float], float],
                 background: float = 0.1,
                 intensity_scale: float = 1000.0,
                 resolution_fwhm: float = 1.0,  # meV (used if resolution_calculator is None)
                 resolution_calculator: Optional['TASResolutionCalculator'] = None,
                 noise_model: str = 'poisson',
                 time_scale: float = 1.0,
                 seed: Optional[int] = None):
        """
        Initialize TAS simulator.

        Parameters
        ----------
        geometry : TASGeometry
            TAS geometry calculator
        sqw_function : callable
            Function S(h, k, l, E) returning intensity (arbitrary units)
        background : float
            Background count rate (counts/sec)
        intensity_scale : float
            Scale factor from S(Q,ω) to count rate
        resolution_fwhm : float
            Energy resolution FWHM in meV (for simple Gaussian convolution).
            Ignored if resolution_calculator is provided.
        resolution_calculator : TASResolutionCalculator, optional
            Full resolution calculator for Cooper-Nathans convolution.
            If provided, uses realistic 4D resolution ellipsoid instead
            of simple Gaussian energy convolution.
            Default: 40' collimations, fixed Ef=14.7 meV.
        noise_model : str
            'poisson' for counting statistics, 'gaussian' for Gaussian, 'none' for no noise
        time_scale : float
            Simulation speedup factor (1.0 = real time, 100.0 = 100x faster)
        seed : int, optional
            Random seed for reproducibility
        """
        self.geometry = geometry
        self.sqw_function = sqw_function
        self.background = background
        self.intensity_scale = intensity_scale
        self.resolution_fwhm = resolution_fwhm
        self.resolution_calculator = resolution_calculator
        self.noise_model = noise_model
        self.time_scale = time_scale

        if seed is not None:
            np.random.seed(seed)

        # Current state
        self._current_position = MeasurementPoint(h=0, k=0, l=0, E=0)
        self._connected = True

        # History tracking
        self.measurement_history: List[MeasurementResult] = []
        self.total_simulated_time: float = 0.0
    
    @property
    def is_connected(self) -> bool:
        return self._connected
    
    def connect(self) -> bool:
        self._connected = True
        return True
    
    def disconnect(self):
        self._connected = False
    
    def get_current_position(self) -> MeasurementPoint:
        return MeasurementPoint(
            h=self._current_position.h,
            k=self._current_position.k,
            l=self._current_position.l,
            E=self._current_position.E,
            angles=self._current_position.angles
        )
    
    def calculate_angles(self, point: MeasurementPoint) -> Dict[str, float]:
        """Use geometry calculator to get angles."""
        return self.geometry.hkl_to_angles(point.h, point.k, point.l, point.E)
    
    def validate_point(self, point: MeasurementPoint) -> tuple:
        """Check if point is accessible."""
        if not self.geometry.is_accessible(point.h, point.k, point.l, point.E):
            return False, "Point not accessible with current geometry"
        return True, ""
    
    def estimate_move_time(self, from_point: MeasurementPoint,
                           to_point: MeasurementPoint) -> float:
        """Estimate time to move between points."""
        # Calculate angles if not present
        if from_point.angles is None:
            try:
                from_point.angles = self.calculate_angles(from_point)
            except ValueError:
                return 60.0  # Default if can't calculate
        
        if to_point.angles is None:
            try:
                to_point.angles = self.calculate_angles(to_point)
            except ValueError:
                return 60.0
        
        return estimate_move_time_from_angles(from_point.angles, to_point.angles)
    
    def measure(self, point: MeasurementPoint) -> MeasurementResult:
        """
        Simulate a measurement at the given point.
        
        Parameters
        ----------
        point : MeasurementPoint
            Point to measure (h, k, l, E, count_time)
        
        Returns
        -------
        MeasurementResult
            Simulated measurement result with realistic noise
        """
        # Validate
        valid, reason = self.validate_point(point)
        if not valid:
            raise ValueError(f"Invalid measurement point: {reason}")
        
        # Calculate angles
        try:
            point.angles = self.calculate_angles(point)
        except ValueError as e:
            raise ValueError(f"Cannot calculate angles: {e}")
        
        # Estimate move time
        move_time = self.estimate_move_time(self._current_position, point)
        
        # Get true S(Q,ω) value
        true_sqw = self._compute_sqw_with_resolution(point.h, point.k, point.l, point.E)
        
        # Calculate expected counts
        signal_rate = true_sqw * self.intensity_scale
        total_rate = signal_rate + self.background
        expected_counts = total_rate * point.count_time
        
        # Add noise
        counts = self._add_noise(expected_counts)
        
        # Calculate intensity and uncertainty
        intensity = counts / point.count_time
        uncertainty = np.sqrt(max(counts, 1)) / point.count_time
        
        # Simulate actual time passage (scaled)
        actual_wait = (move_time + point.count_time) / self.time_scale
        if actual_wait > 0.01:  # Don't sleep for tiny times
            time.sleep(actual_wait)
        
        # Update state
        self._current_position = MeasurementPoint(
            h=point.h, k=point.k, l=point.l, E=point.E,
            angles=point.angles
        )
        
        point.counts = int(counts)
        point.timestamp = time.time()
        
        elapsed = move_time + point.count_time
        self.total_simulated_time += elapsed
        
        result = MeasurementResult(
            point=point,
            intensity=intensity,
            uncertainty=uncertainty,
            elapsed_time=elapsed,
            metadata={
                'true_sqw': true_sqw,
                'expected_counts': expected_counts,
                'move_time': move_time,
                'simulated': True
            }
        )
        
        self.measurement_history.append(result)
        logger.debug(f"Simulated measurement: {point} -> I={intensity:.2f}±{uncertainty:.2f}")
        
        return result
    
    def _compute_sqw_with_resolution(self, h: float, k: float, l: float, E: float) -> float:
        """
        Compute S(Q,ω) with optional resolution convolution.

        If a TASResolutionCalculator is provided, uses full 4D Cooper-Nathans
        resolution convolution. Otherwise, uses simple 1D Gaussian energy convolution.

        Parameters
        ----------
        h, k, l : float
            Miller indices
        E : float
            Energy transfer in meV

        Returns
        -------
        float
            Resolution-convoluted S(Q,w) value
        """
        # If resolution calculator is available, use realistic convolution
        if self.resolution_calculator is not None:
            return self._compute_sqw_with_cooper_nathans(h, k, l, E)

        # Fallback: simple 1D Gaussian convolution in energy
        if self.resolution_fwhm <= 0:
            return self.sqw_function(h, k, l, E)

        sigma = self.resolution_fwhm / 2.355  # FWHM to sigma

        # Sample at ±3σ
        n_points = 7
        E_offsets = np.linspace(-3*sigma, 3*sigma, n_points)
        weights = np.exp(-E_offsets**2 / (2*sigma**2))
        weights /= weights.sum()

        result = 0.0
        for dE, w in zip(E_offsets, weights):
            result += w * self.sqw_function(h, k, l, E + dE)

        return result

    def _compute_sqw_with_cooper_nathans(self, h: float, k: float, l: float, E: float) -> float:
        """
        Compute S(Q,ω) with full Cooper-Nathans resolution convolution.

        Uses the TASResolutionCalculator to integrate over the 4D resolution
        ellipsoid (Qx, Qy, Qz, E).

        Parameters
        ----------
        h, k, l : float
            Miller indices
        E : float
            Energy transfer in meV

        Returns
        -------
        float
            Resolution-convoluted S(Q,w) value
        """
        # Get resolution matrix at this point
        fwhm, R0 = self.resolution_calculator.get_resolution_fwhm(h, k, l, E)

        # Convert FWHM to sigma for each direction
        sigma_Qx = fwhm['Qx'] / 2.355 if fwhm['Qx'] < np.inf else 0.1
        sigma_Qy = fwhm['Qy'] / 2.355 if fwhm['Qy'] < np.inf else 0.1
        sigma_E = fwhm['E'] / 2.355 if fwhm['E'] < np.inf else 0.5

        # Get coordinate system from resolution calculator
        res_calc = self.resolution_calculator
        xvec = np.array(res_calc.lattice.x[:, 0]) if hasattr(res_calc.lattice, 'x') else np.array([1, 0, 0])
        yvec = np.array(res_calc.lattice.y[:, 0]) if hasattr(res_calc.lattice, 'y') else np.array([0, 1, 0])

        # Simple Gaussian quadrature over 3 dimensions (Qx, Qy, E)
        # Using 5-point quadrature in each dimension
        n_quad = 5
        quad_points = np.linspace(-2, 2, n_quad)  # Sample at ±2σ
        weights = np.exp(-quad_points**2 / 2)
        weights /= weights.sum()

        result = 0.0
        total_weight = 0.0

        for i, (dx, wx) in enumerate(zip(quad_points * sigma_Qx, weights)):
            for j, (dy, wy) in enumerate(zip(quad_points * sigma_Qy, weights)):
                for k_idx, (dE, wE) in enumerate(zip(quad_points * sigma_E, weights)):
                    # Displaced Q point (in HKL space)
                    h1 = h + dx * xvec[0] + dy * yvec[0]
                    k1 = k + dx * xvec[1] + dy * yvec[1]
                    l1 = l + dx * xvec[2] + dy * yvec[2]
                    E1 = E + dE

                    # Weight for this sample point
                    w_total = wx * wy * wE

                    # Accumulate weighted S(Q,w)
                    try:
                        sqw_val = self.sqw_function(h1, k1, l1, E1)
                        result += w_total * sqw_val
                        total_weight += w_total
                    except (ValueError, RuntimeError):
                        # Skip points where S(Q,w) can't be evaluated
                        pass

        # Normalize
        if total_weight > 0:
            result /= total_weight

        return result
    
    def _add_noise(self, expected_counts: float) -> float:
        """Add noise according to noise model."""
        if self.noise_model == 'none':
            return expected_counts
        elif self.noise_model == 'poisson':
            return float(np.random.poisson(max(expected_counts, 0)))
        elif self.noise_model == 'gaussian':
            sigma = np.sqrt(max(expected_counts, 1))
            return max(0, np.random.normal(expected_counts, sigma))
        else:
            raise ValueError(f"Unknown noise model: {self.noise_model}")
    
    def reset(self):
        """Reset simulator state."""
        self._current_position = MeasurementPoint(h=0, k=0, l=0, E=0)
        self.measurement_history.clear()
        self.total_simulated_time = 0.0


# =============================================================================
# Example S(Q,ω) functions for testing
# =============================================================================

def ferromagnetic_magnon(h: float, k: float, l: float, E: float,
                         J: float = 10.0, S: float = 0.5,
                         D: float = 0.0,  # Anisotropy
                         gamma: float = 0.5,
                         T: float = 10.0) -> float:
    """
    Simple ferromagnetic magnon dispersion for 1D chain along h.
    
    ω(q) = 2JS(1 - cos(2πh)) + D
    
    Parameters
    ----------
    J : float
        Exchange constant in meV
    S : float
        Spin quantum number
    D : float
        Anisotropy gap in meV
    gamma : float
        Damping/linewidth in meV
    T : float
        Temperature in K (for Bose factor)
    """
    # Dispersion relation
    omega_q = 2 * J * S * (1 - np.cos(2 * np.pi * h)) + D
    
    # Lorentzian spectral function
    if gamma > 0:
        sqw = (gamma / np.pi) / ((E - omega_q)**2 + gamma**2)
    else:
        # Delta function approximation
        sqw = 1.0 if abs(E - omega_q) < 0.1 else 0.0
    
    # Bose factor for positive energy transfer
    if T > 0 and E > 0.1:
        kT = 0.0862 * T  # meV
        n_bose = 1.0 / (np.exp(E / kT) - 1)
        sqw *= (1 + n_bose)
    
    return max(sqw, 0)


def antiferromagnetic_2d(h: float, k: float, l: float, E: float,
                         J1: float = 5.0, J2: float = 0.0,
                         S: float = 0.5,
                         gamma: float = 1.0,
                         gap: float = 0.0) -> float:
    """
    2D antiferromagnet on square lattice with J1-J2 interactions.
    
    Linear spin wave dispersion (simplified).
    
    Parameters
    ----------
    J1 : float
        Nearest-neighbor exchange in meV
    J2 : float
        Next-nearest-neighbor exchange in meV
    S : float
        Spin quantum number
    gamma : float
        Linewidth in meV
    gap : float
        Spin gap in meV
    """
    # Structure factors
    gamma_1 = 0.5 * (np.cos(2*np.pi*h) + np.cos(2*np.pi*k))
    gamma_2 = np.cos(2*np.pi*h) * np.cos(2*np.pi*k)
    
    # Dispersion for Néel state (simplified)
    # Full expression involves sublattice structure
    A = 4 * J1 * S + 4 * J2 * S * (1 - gamma_2)
    B = 4 * J1 * S * gamma_1
    
    arg = A**2 - B**2
    if arg < 0:
        omega_q = 0
    else:
        omega_q = np.sqrt(arg) + gap
    
    # Spectral function
    if gamma > 0:
        sqw = (gamma / np.pi) / ((E - omega_q)**2 + gamma**2)
    else:
        sqw = 1.0 if abs(E - omega_q) < 0.1 else 0.0
    
    return max(sqw, 0)


def phonon_dispersion(h: float, k: float, l: float, E: float,
                      v_sound: float = 30.0,  # meV·Å
                      gamma: float = 0.5) -> float:
    """
    Simple acoustic phonon dispersion.
    
    ω(q) = v_s |q| near zone center
    
    Parameters
    ----------
    v_sound : float
        Sound velocity in meV·Å
    gamma : float
        Linewidth in meV
    """
    # Distance from zone center in r.l.u.
    q = np.sqrt(h**2 + k**2 + l**2)
    
    # Linear dispersion (acoustic)
    omega_q = v_sound * q
    
    # Lorentzian
    sqw = (gamma / np.pi) / ((E - omega_q)**2 + gamma**2)
    
    return max(sqw, 0)


def double_peak(h: float, k: float, l: float, E: float,
                E1: float = 5.0, E2: float = 15.0,
                I1: float = 1.0, I2: float = 0.5,
                gamma: float = 1.0) -> float:
    """
    Two Lorentzian peaks at fixed energies (for testing).
    
    Useful for testing ability to find and characterize multiple features.
    """
    peak1 = I1 * (gamma / np.pi) / ((E - E1)**2 + gamma**2)
    peak2 = I2 * (gamma / np.pi) / ((E - E2)**2 + gamma**2)
    return peak1 + peak2


class TabularSQW:
    """
    S(Q,ω) from pre-computed data (e.g., from Sunny.jl or experiment).
    
    Interpolates on a 4D grid.
    """
    
    def __init__(self, 
                 h_grid: np.ndarray,
                 k_grid: np.ndarray,
                 l_grid: np.ndarray,
                 E_grid: np.ndarray,
                 sqw_data: np.ndarray):
        """
        Initialize tabular S(Q,ω).
        
        Parameters
        ----------
        h_grid, k_grid, l_grid, E_grid : np.ndarray
            1D arrays defining the grid points
        sqw_data : np.ndarray
            4D array of S(Q,ω) values with shape (len(h), len(k), len(l), len(E))
        """
        from scipy.interpolate import RegularGridInterpolator
        
        self.h_grid = h_grid
        self.k_grid = k_grid
        self.l_grid = l_grid
        self.E_grid = E_grid
        self.sqw_data = sqw_data
        
        self._interpolator = RegularGridInterpolator(
            (h_grid, k_grid, l_grid, E_grid),
            sqw_data,
            method='linear',
            bounds_error=False,
            fill_value=0.0
        )
    
    def __call__(self, h: float, k: float, l: float, E: float) -> float:
        """Interpolate S(Q,ω) at given point."""
        return float(self._interpolator([[h, k, l, E]])[0])
