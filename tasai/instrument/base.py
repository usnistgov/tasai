"""
TAS-AI Instrument Abstraction Layer

Provides abstract interfaces for instrument control, enabling:
- Real instrument control via NICE proxy
- Simulation mode for testing
- Replay of recorded experiments
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict, Callable
from enum import Enum
import numpy as np


@dataclass
class MeasurementPoint:
    """A single measurement configuration in (h,k,l,E) space."""
    
    # Physics coordinates
    h: float
    k: float
    l: float
    E: float  # Energy transfer in meV
    
    # Instrument coordinates (filled by geometry calculator)
    angles: Optional[Dict[str, float]] = None
    
    # Measurement parameters
    count_time: float = 60.0  # seconds
    monitor: Optional[int] = None  # Monitor counts (alternative to time)
    
    # Results (filled after measurement)
    counts: Optional[int] = None
    monitor_counts: Optional[int] = None
    timestamp: Optional[float] = None
    
    def __repr__(self):
        return f"MeasurementPoint(h={self.h:.3f}, k={self.k:.3f}, l={self.l:.3f}, E={self.E:.2f})"
    
    def to_dict(self) -> dict:
        return {
            'h': self.h, 'k': self.k, 'l': self.l, 'E': self.E,
            'angles': self.angles,
            'count_time': self.count_time,
            'monitor': self.monitor,
            'counts': self.counts,
            'monitor_counts': self.monitor_counts,
            'timestamp': self.timestamp
        }
    
    @classmethod
    def from_dict(cls, d: dict) -> 'MeasurementPoint':
        return cls(**d)


@dataclass
class MeasurementResult:
    """Result from a measurement."""
    
    point: MeasurementPoint
    intensity: float  # counts per second (or per monitor)
    uncertainty: float  # statistical uncertainty
    elapsed_time: float  # Total time including movement
    metadata: Dict = field(default_factory=dict)
    
    @property
    def signal_to_noise(self) -> float:
        if self.uncertainty > 0:
            return self.intensity / self.uncertainty
        return 0.0


class InstrumentInterface(ABC):
    """
    Abstract interface for TAS instrument control.
    
    Implementations:
    - NICEProxy: Connects to NICE via proxy server
    - TASSimulator: Simulated instrument for testing
    - ReplayInstrument: Replays recorded experiments
    """
    
    @abstractmethod
    def get_current_position(self) -> MeasurementPoint:
        """Return current instrument position in physics coordinates."""
        pass
    
    @abstractmethod
    def calculate_angles(self, point: MeasurementPoint) -> Dict[str, float]:
        """
        Convert (h,k,l,E) to instrument angles.
        
        Returns dict with keys like 'A1', 'A2', 'A3', 'A4', 'A5', 'A6'
        depending on instrument configuration.
        """
        pass
    
    @abstractmethod
    def estimate_move_time(self, 
                           from_point: MeasurementPoint,
                           to_point: MeasurementPoint) -> float:
        """
        Estimate time to move between configurations.
        
        Returns time in seconds.
        """
        pass
    
    @abstractmethod
    def measure(self, point: MeasurementPoint) -> MeasurementResult:
        """
        Execute measurement and return result.
        
        This is the core method that interfaces with the instrument.
        For human-in-the-loop operation, implementations should support
        approval callbacks before execution.
        """
        pass
    
    @abstractmethod
    def validate_point(self, point: MeasurementPoint) -> Tuple[bool, str]:
        """
        Check if measurement point is achievable.
        
        Returns (is_valid, reason) tuple.
        """
        pass
    
    @property
    @abstractmethod
    def is_connected(self) -> bool:
        """Check if instrument connection is active."""
        pass
    
    def generate_trajectory(self, points: List[MeasurementPoint]) -> dict:
        """
        Generate a trajectory specification for multiple points.
        
        Can be used for dry-run validation before execution.
        """
        trajectory = {
            'type': 'TAS_SCAN',
            'points': []
        }
        
        for pt in points:
            if pt.angles is None:
                pt.angles = self.calculate_angles(pt)
            trajectory['points'].append(pt.to_dict())
        
        return trajectory


class TASGeometry:
    """
    Triple-axis spectrometer geometry calculations.
    
    Handles conversion between (h,k,l,E) and instrument angles,
    taking into account crystal orientation and instrument configuration.
    """
    
    def __init__(self,
                 lattice_params: Tuple[float, float, float, float, float, float],
                 orientation: Tuple[Tuple[float, float, float], Tuple[float, float, float]],
                 ei_fixed: bool = True,
                 fixed_energy: float = 14.7,  # meV
                 monochromator_d: float = 3.355,  # PG(002) in Å
                 analyzer_d: float = 3.355,
                 sense: Tuple[int, int, int] = (1, -1, 1),  # SM, SS, SA
                 scattering_plane_normal: Tuple[float, float, float] = (0, 0, 1)):
        """
        Initialize TAS geometry calculator.
        
        Parameters
        ----------
        lattice_params : tuple
            (a, b, c, alpha, beta, gamma) in Å and degrees
        orientation : tuple
            ((h1,k1,l1), (h2,k2,l2)) - two vectors defining scattering plane
        ei_fixed : bool
            If True, fixed incident energy; if False, fixed final energy
        fixed_energy : float
            The fixed energy (Ei or Ef) in meV
        monochromator_d : float
            Monochromator d-spacing in Å
        analyzer_d : float
            Analyzer d-spacing in Å
        sense : tuple
            Scattering senses: (monochromator, sample, analyzer)
            +1 = counter-clockwise, -1 = clockwise
        scattering_plane_normal : tuple
            Normal to scattering plane in crystal coordinates
        """
        self.a, self.b, self.c = lattice_params[:3]
        self.alpha, self.beta, self.gamma = np.radians(lattice_params[3:])
        self.orient1 = np.array(orientation[0])
        self.orient2 = np.array(orientation[1])
        self.ei_fixed = ei_fixed
        self.fixed_energy = fixed_energy
        self.mono_d = monochromator_d
        self.ana_d = analyzer_d
        self.sense = sense
        self.plane_normal = np.array(scattering_plane_normal)
        
        # Compute matrices
        self._compute_b_matrix()
        self._compute_ub_matrix()
    
    def _compute_b_matrix(self):
        """Compute B matrix (crystal to Cartesian)."""
        # Standard crystallographic B matrix
        ca, cb, cg = np.cos(self.alpha), np.cos(self.beta), np.cos(self.gamma)
        sa, sb, sg = np.sin(self.alpha), np.sin(self.beta), np.sin(self.gamma)
        
        # Volume factor
        val = 1 - ca**2 - cb**2 - cg**2 + 2*ca*cb*cg
        vol = self.a * self.b * self.c * np.sqrt(val)
        
        # Reciprocal lattice parameters
        astar = self.b * self.c * sa / vol
        bstar = self.a * self.c * sb / vol
        cstar = self.a * self.b * sg / vol
        
        # B matrix (Busing-Levy convention)
        self.B = np.array([
            [astar, bstar * cg, cstar * cb],
            [0, bstar * sg, -cstar * sb * ca],
            [0, 0, 1.0 / self.c]
        ])
    
    def _compute_ub_matrix(self):
        """Compute UB matrix from orientation vectors."""
        # Convert orientation vectors to Cartesian
        t1 = self.B @ self.orient1
        t2 = self.B @ self.orient2
        
        # Normalize
        t1 = t1 / np.linalg.norm(t1)
        
        # Gram-Schmidt to make orthogonal
        t2 = t2 - np.dot(t2, t1) * t1
        t2 = t2 / np.linalg.norm(t2)
        
        # Third vector
        t3 = np.cross(t1, t2)
        
        # U matrix
        self.U = np.array([t1, t2, t3]).T
        
        # UB matrix
        self.UB = self.U @ self.B
    
    def hkl_to_Q(self, h: float, k: float, l: float) -> np.ndarray:
        """Convert (h,k,l) to Q vector in lab frame (Å⁻¹)."""
        hkl = np.array([h, k, l])
        Q = 2 * np.pi * self.UB @ hkl
        return Q
    
    def Q_magnitude(self, h: float, k: float, l: float) -> float:
        """Calculate |Q| for given (h,k,l)."""
        Q = self.hkl_to_Q(h, k, l)
        return np.linalg.norm(Q)
    
    def energy_to_k(self, E: float) -> float:
        """Convert energy (meV) to wavevector (Å⁻¹)."""
        # E = ℏ²k²/2m = 2.072 k² meV (for k in Å⁻¹)
        return np.sqrt(E / 2.072)
    
    def k_to_energy(self, k: float) -> float:
        """Convert wavevector (Å⁻¹) to energy (meV)."""
        return 2.072 * k**2
    
    def hkl_to_angles(self, h: float, k: float, l: float, E: float) -> Dict[str, float]:
        """
        Calculate TAS angles for (h,k,l,E) point.
        
        Returns dict with A1-A6 angles in degrees.
        """
        # Get Q magnitude
        Q_mag = self.Q_magnitude(h, k, l)
        
        # Calculate ki and kf
        if self.ei_fixed:
            Ei = self.fixed_energy
            Ef = Ei - E
            if Ef <= 0:
                raise ValueError(f"Ef would be negative: Ei={Ei}, E={E}")
        else:
            Ef = self.fixed_energy
            Ei = Ef + E
        
        ki = self.energy_to_k(Ei)
        kf = self.energy_to_k(Ef)
        
        # Monochromator angle (A1 = θ_mono)
        # Bragg: 2d sin(θ) = λ = 2π/k
        lambda_i = 2 * np.pi / ki
        sin_theta_mono = lambda_i / (2 * self.mono_d)
        if abs(sin_theta_mono) > 1:
            raise ValueError(f"Monochromator angle not achievable for Ei={Ei} meV")
        theta_mono = np.arcsin(sin_theta_mono)
        A1 = np.degrees(theta_mono) * self.sense[0]
        A2 = 2 * A1  # Monochromator 2θ
        
        # Sample angle (A3) and scattering angle (A4)
        # From momentum conservation: Q = ki - kf
        # |Q|² = ki² + kf² - 2*ki*kf*cos(2θ_sample)
        cos_2theta = (ki**2 + kf**2 - (Q_mag/(2*np.pi))**2) / (2 * ki * kf)
        if abs(cos_2theta) > 1:
            raise ValueError(f"Scattering angle not achievable for Q={Q_mag}, ki={ki}, kf={kf}")
        two_theta = np.arccos(cos_2theta)
        A4 = np.degrees(two_theta) * self.sense[1]
        
        # A3 requires knowing the Q direction in the scattering plane
        # This is more complex and depends on the specific (h,k,l)
        Q_vec = self.hkl_to_Q(h, k, l)
        Q_angle = np.arctan2(Q_vec[1], Q_vec[0])  # Angle in scattering plane
        
        # A3 = angle to put Q along ki - kf direction
        # Simplified: assume Q is along the bisector
        A3 = np.degrees(Q_angle + two_theta/2)
        
        # Analyzer angles
        lambda_f = 2 * np.pi / kf
        sin_theta_ana = lambda_f / (2 * self.ana_d)
        if abs(sin_theta_ana) > 1:
            raise ValueError(f"Analyzer angle not achievable for Ef={Ef} meV")
        theta_ana = np.arcsin(sin_theta_ana)
        A5 = np.degrees(theta_ana) * self.sense[2]
        A6 = 2 * A5
        
        return {
            'A1': A1,
            'A2': A2,
            'A3': A3,
            'A4': A4,
            'A5': A5,
            'A6': A6,
            'Ei': Ei,
            'Ef': Ef,
            'ki': ki,
            'kf': kf,
            'Q': Q_mag
        }
    
    def is_accessible(self, h: float, k: float, l: float, E: float,
                      A3_limits: Tuple[float, float] = (-180, 180),
                      A4_limits: Tuple[float, float] = (-120, 120)) -> bool:
        """Check if (h,k,l,E) point is accessible."""
        try:
            angles = self.hkl_to_angles(h, k, l, E)
            
            # Check angle limits
            if not (A3_limits[0] <= angles['A3'] <= A3_limits[1]):
                return False
            if not (A4_limits[0] <= angles['A4'] <= A4_limits[1]):
                return False
            
            return True
        except ValueError:
            return False


# Motor speed configuration for timing estimates
@dataclass
class MotorSpeeds:
    """Motor speeds for timing estimates."""
    A1: float = 1.0  # deg/sec
    A2: float = 2.0
    A3: float = 2.0
    A4: float = 2.0
    A5: float = 1.0
    A6: float = 2.0
    overhead: float = 5.0  # Fixed overhead per move


def estimate_move_time_from_angles(from_angles: Dict[str, float],
                                    to_angles: Dict[str, float],
                                    speeds: MotorSpeeds = None) -> float:
    """
    Estimate movement time between two angle configurations.
    
    Assumes motors move in parallel, so time is max of individual times.
    """
    if speeds is None:
        speeds = MotorSpeeds()
    
    times = []
    for motor in ['A1', 'A2', 'A3', 'A4', 'A5', 'A6']:
        if motor in from_angles and motor in to_angles:
            delta = abs(to_angles[motor] - from_angles[motor])
            speed = getattr(speeds, motor)
            times.append(delta / speed)
    
    return max(times) + speeds.overhead if times else speeds.overhead
