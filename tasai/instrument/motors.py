"""
TAS Motor Motion Simulation

Simulates realistic motor motion for Triple-Axis Spectrometer experiments.
Based on AutoREFL's approach where move time affects acquisition optimization.

Key insight: The acquisition function should optimize INFORMATION RATE:
    score = expected_info_gain / (count_time + move_time)

This naturally prefers efficient measurement sequences that minimize
unnecessary motor movements while still gathering discriminating data.

Typical TAS motors:
    - A3 (sample rotation): ~1 deg/sec
    - A4 (scattering angle): ~1 deg/sec  
    - Monochromator/Analyzer: ~0.5 deg/sec
    - Sample translation: ~1 mm/sec
    - Temperature: ~1-10 K/min (depends on cooling power)

Reference: AutoREFL motion planning
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


@dataclass
class MotorConfig:
    """Configuration for a single motor axis."""
    name: str
    speed: float  # units per second
    acceleration: float = 0.0  # units per second^2 (0 = instant accel)
    backlash: float = 0.0  # extra time for direction change
    limits: Tuple[float, float] = (-np.inf, np.inf)
    current_position: float = 0.0
    
    def move_time(self, target: float) -> float:
        """Calculate time to move to target position."""
        distance = abs(target - self.current_position)
        
        if distance < 1e-6:
            return 0.0
        
        # Check limits
        if target < self.limits[0] or target > self.limits[1]:
            logger.warning(f"Motor {self.name} target {target} outside limits {self.limits}")
            return np.inf
        
        # Simple constant velocity model
        time = distance / self.speed
        
        # Add acceleration time if specified
        if self.acceleration > 0:
            # Trapezoidal velocity profile
            t_accel = self.speed / self.acceleration
            d_accel = 0.5 * self.acceleration * t_accel**2
            
            if distance < 2 * d_accel:
                # Short move: triangular profile
                time = 2 * np.sqrt(distance / self.acceleration)
            else:
                # Long move: trapezoidal profile
                time = 2 * t_accel + (distance - 2 * d_accel) / self.speed
        
        # Add backlash if changing direction
        # (simplified: just add backlash time for any move)
        if self.backlash > 0:
            time += self.backlash
        
        return time
    
    def move_to(self, target: float) -> float:
        """Move to target and return elapsed time."""
        time = self.move_time(target)
        if time < np.inf:
            self.current_position = target
        return time


@dataclass
class TASMotorSystem:
    """
    Complete TAS motor system for HHL measurements.
    
    Coordinates:
        - H, K, L: reciprocal lattice units
        - E: energy transfer (meV)
        - T: sample temperature (K)
    
    The system converts (H, K, L, E) to motor angles using
    the TAS geometry equations.
    """
    
    # Lattice parameters (Angstroms)
    a: float = 4.0
    b: float = 4.0
    c: float = 10.0
    
    # Fixed incident energy (for constant-ki mode)
    Ei: float = 14.7  # meV (typical thermal neutron)
    
    # Motor configurations
    motors: Dict[str, MotorConfig] = field(default_factory=dict)
    
    # Current position in (H, K, L, E, T) space
    current_H: float = 0.0
    current_K: float = 0.0
    current_L: float = 0.0
    current_E: float = 0.0
    current_T: float = 300.0  # K
    
    # Overhead times (seconds)
    count_overhead: float = 2.0  # Time to start/stop counting
    motor_overhead: float = 1.0  # Per-motor overhead
    
    def __post_init__(self):
        """Initialize default motors if not provided."""
        if not self.motors:
            self.motors = self._default_motors()
    
    def _default_motors(self) -> Dict[str, MotorConfig]:
        """Create default TAS motor configuration."""
        return {
            # Sample rotation (A3)
            'A3': MotorConfig(
                name='A3',
                speed=1.0,  # deg/sec
                acceleration=2.0,  # deg/sec^2
                backlash=0.5,  # seconds
                limits=(-180, 180),
                current_position=0.0
            ),
            # Scattering angle (A4 = 2θ)
            'A4': MotorConfig(
                name='A4',
                speed=1.0,
                acceleration=2.0,
                backlash=0.5,
                limits=(-120, 120),
                current_position=0.0
            ),
            # Monochromator angle
            'mono': MotorConfig(
                name='mono',
                speed=0.5,
                acceleration=1.0,
                backlash=1.0,
                limits=(10, 80),
                current_position=41.0  # For Ei=14.7 meV with PG
            ),
            # Analyzer angle
            'ana': MotorConfig(
                name='ana',
                speed=0.5,
                acceleration=1.0,
                backlash=1.0,
                limits=(10, 80),
                current_position=41.0
            ),
            # Sample temperature
            'temperature': MotorConfig(
                name='temperature',
                speed=2.0,  # K/sec (fast cryo)
                acceleration=0.0,
                backlash=0.0,
                limits=(1.5, 800),
                current_position=300.0
            ),
        }
    
    def hkl_to_angles(self, H: float, K: float, L: float, E: float) -> Dict[str, float]:
        """
        Convert (H, K, L, E) to motor angles.
        
        Uses standard TAS geometry for constant-ki mode.
        
        Returns dict with 'A3', 'A4', 'ana' angles.
        """
        # Wavevector calculations
        # ki = sqrt(Ei / 2.072)  # Angstrom^-1, for Ei in meV
        # kf = sqrt((Ei - E) / 2.072)
        
        ki = np.sqrt(self.Ei / 2.072)
        Ef = self.Ei - E
        
        if Ef <= 0:
            logger.warning(f"Invalid energy transfer E={E} meV (Ei={self.Ei})")
            return {'A3': np.nan, 'A4': np.nan, 'ana': np.nan}
        
        kf = np.sqrt(Ef / 2.072)
        
        # Q vector magnitude
        # For HHL: Q = (H, H, L) in r.l.u.
        # |Q| = 2π * sqrt((H/a)² + (K/b)² + (L/c)²)
        Qx = 2 * np.pi * H / self.a
        Qy = 2 * np.pi * K / self.b
        Qz = 2 * np.pi * L / self.c
        Q = np.sqrt(Qx**2 + Qy**2 + Qz**2)
        
        # Scattering angle from momentum conservation
        # Q² = ki² + kf² - 2*ki*kf*cos(2θ)
        cos_2theta = (ki**2 + kf**2 - Q**2) / (2 * ki * kf)
        
        if abs(cos_2theta) > 1:
            logger.warning(f"(H,K,L,E)=({H},{K},{L},{E}) not accessible")
            return {'A3': np.nan, 'A4': np.nan, 'ana': np.nan}
        
        two_theta = np.degrees(np.arccos(cos_2theta))
        
        # Sample angle (simplified - assumes Q along [HH0])
        # In reality this depends on crystal orientation
        A3 = np.degrees(np.arctan2(Qy, Qx)) + two_theta / 2
        
        # Analyzer angle (for PG002, d=3.355 Å)
        # λ = 2d sin(θ_ana)
        # λ = sqrt(81.81 / Ef)  # Angstroms
        lambda_f = np.sqrt(81.81 / Ef)
        d_ana = 3.355
        sin_theta_ana = lambda_f / (2 * d_ana)
        
        if abs(sin_theta_ana) > 1:
            logger.warning(f"Analyzer angle not accessible for E={E}")
            return {'A3': np.nan, 'A4': np.nan, 'ana': np.nan}
        
        theta_ana = np.degrees(np.arcsin(sin_theta_ana))
        
        return {
            'A3': A3,
            'A4': two_theta,
            'ana': theta_ana
        }
    
    def move_time(self, H: float, K: float, L: float, E: float, T: float = None) -> float:
        """
        Calculate total time to move to new (H, K, L, E, T) position.
        
        Motors move simultaneously, so total time = max of individual times.
        """
        if T is None:
            T = self.current_T
        
        # Get target angles
        angles = self.hkl_to_angles(H, K, L, E)
        
        if np.isnan(angles['A3']):
            return np.inf
        
        # Calculate time for each motor
        times = []
        
        # Goniometer motors
        times.append(self.motors['A3'].move_time(angles['A3']))
        times.append(self.motors['A4'].move_time(angles['A4']))
        times.append(self.motors['ana'].move_time(angles['ana']))
        
        # Temperature (if changing)
        if abs(T - self.current_T) > 0.1:
            times.append(self.motors['temperature'].move_time(T))
        
        # Total time = max (parallel motion) + overhead
        total_time = max(times) + self.motor_overhead * len([t for t in times if t > 0])
        
        return total_time
    
    def move_to(self, H: float, K: float, L: float, E: float, T: float = None) -> float:
        """
        Move to new position and return elapsed time.
        """
        if T is None:
            T = self.current_T
        
        move_time = self.move_time(H, K, L, E, T)
        
        if move_time < np.inf:
            # Update motor positions
            angles = self.hkl_to_angles(H, K, L, E)
            self.motors['A3'].move_to(angles['A3'])
            self.motors['A4'].move_to(angles['A4'])
            self.motors['ana'].move_to(angles['ana'])
            
            if abs(T - self.current_T) > 0.1:
                self.motors['temperature'].move_to(T)
            
            # Update current position
            self.current_H = H
            self.current_K = K
            self.current_L = L
            self.current_E = E
            self.current_T = T
        
        return move_time
    
    def measurement_time(self, H: float, K: float, L: float, E: float, 
                        count_time: float, T: float = None) -> float:
        """
        Calculate total time for a measurement including motion.
        
        Returns: move_time + count_time + overhead
        """
        move_time = self.move_time(H, K, L, E, T)
        return move_time + count_time + self.count_overhead
    
    def optimize_sequence(self, points: List[Tuple[float, float, float, float]], 
                         count_time: float = 60.0) -> List[int]:
        """
        Optimize measurement sequence to minimize total motion time.
        
        Uses nearest-neighbor heuristic (greedy TSP approximation).
        
        Parameters
        ----------
        points : list of (H, K, L, E) tuples
            Points to measure
        count_time : float
            Count time per point
        
        Returns
        -------
        list : Optimized order (indices into points list)
        """
        if len(points) <= 2:
            return list(range(len(points)))
        
        # Build distance matrix (move times)
        n = len(points)
        dist = np.zeros((n, n))
        
        # Save current position
        saved_pos = (self.current_H, self.current_K, self.current_L, self.current_E)
        
        for i in range(n):
            # Move to point i
            self.current_H, self.current_K, self.current_L, self.current_E = points[i]
            
            for j in range(n):
                if i != j:
                    dist[i, j] = self.move_time(*points[j])
        
        # Restore position
        self.current_H, self.current_K, self.current_L, self.current_E = saved_pos
        
        # Nearest neighbor heuristic
        # Start from point closest to current position
        start_times = [self.move_time(*p) for p in points]
        current = np.argmin(start_times)
        
        visited = [current]
        remaining = set(range(n)) - {current}
        
        while remaining:
            # Find nearest unvisited point
            nearest = min(remaining, key=lambda j: dist[visited[-1], j])
            visited.append(nearest)
            remaining.remove(nearest)
        
        return visited
    
    def estimate_experiment_time(self, points: List[Tuple[float, float, float, float]],
                                 count_time: float = 60.0,
                                 optimize: bool = True) -> Dict:
        """
        Estimate total experiment time.
        
        Returns dict with breakdown of time components.
        """
        if optimize:
            order = self.optimize_sequence(points, count_time)
            ordered_points = [points[i] for i in order]
        else:
            order = list(range(len(points)))
            ordered_points = points
        
        # Calculate times
        move_times = []
        
        # Save position
        saved_pos = (self.current_H, self.current_K, self.current_L, self.current_E)
        
        for p in ordered_points:
            mt = self.move_time(*p)
            move_times.append(mt)
            self.move_to(*p)
        
        # Restore position
        self.current_H, self.current_K, self.current_L, self.current_E = saved_pos
        
        total_move = sum(move_times)
        total_count = count_time * len(points)
        total_overhead = self.count_overhead * len(points)
        
        return {
            'n_points': len(points),
            'order': order,
            'total_time': total_move + total_count + total_overhead,
            'move_time': total_move,
            'count_time': total_count,
            'overhead': total_overhead,
            'efficiency': total_count / (total_move + total_count + total_overhead),
            'move_times': move_times
        }


@dataclass
class SimplifiedMotorModel:
    """
    Simplified motor model for fast calculations.
    
    Uses approximate move times based on coordinate changes
    without full angle calculations.
    """
    
    # Move speeds in coordinate units per second
    H_speed: float = 0.01  # r.l.u. per second
    K_speed: float = 0.01
    L_speed: float = 0.02  # Usually faster (less mechanics)
    E_speed: float = 0.5   # meV per second (analyzer motion)
    T_speed: float = 2.0   # K per second
    
    # Current position
    current_H: float = 0.0
    current_K: float = 0.0
    current_L: float = 0.0
    current_E: float = 5.0
    current_T: float = 5.0
    
    # Overhead
    overhead: float = 3.0  # seconds per move
    
    def move_time(self, H: float = None, K: float = None, L: float = None,
                  E: float = None, T: float = None) -> float:
        """Calculate move time to new position."""
        times = []
        
        if H is not None:
            times.append(abs(H - self.current_H) / self.H_speed)
        if K is not None:
            times.append(abs(K - self.current_K) / self.K_speed)
        if L is not None:
            times.append(abs(L - self.current_L) / self.L_speed)
        if E is not None:
            times.append(abs(E - self.current_E) / self.E_speed)
        if T is not None:
            times.append(abs(T - self.current_T) / self.T_speed)
        
        if not times:
            return 0.0
        
        # Parallel motion: max time + overhead
        return max(times) + self.overhead
    
    def move_to(self, H: float = None, K: float = None, L: float = None,
                E: float = None, T: float = None) -> float:
        """Move to position and return elapsed time."""
        time = self.move_time(H, K, L, E, T)
        
        if H is not None:
            self.current_H = H
        if K is not None:
            self.current_K = K
        if L is not None:
            self.current_L = L
        if E is not None:
            self.current_E = E
        if T is not None:
            self.current_T = T
        
        return time
    
    def measurement_time(self, H: float, K: float, L: float, E: float,
                        count_time: float, T: float = None) -> float:
        """Total time for measurement including motion."""
        return self.move_time(H, K, L, E, T) + count_time


def create_hhl_motor_system(a: float = 4.0, c: float = 10.0, 
                            Ei: float = 14.7) -> TASMotorSystem:
    """Create a motor system configured for HHL zone measurements."""
    return TASMotorSystem(a=a, b=a, c=c, Ei=Ei)


def create_simple_motor_model() -> SimplifiedMotorModel:
    """Create a simplified motor model for fast calculations."""
    return SimplifiedMotorModel()


# =============================================================================
# Acquisition function with motor motion
# =============================================================================

class MotionAwareAcquisition:
    """
    Acquisition function that accounts for motor motion time.
    
    Score = expected_info_gain^η / (count_time + move_time)
    
    This naturally balances:
    - High information points (near critical features)
    - Efficient motion (minimize unnecessary moves)
    """
    
    def __init__(self, motor_model: SimplifiedMotorModel = None, 
                 eta: float = 0.7, count_time: float = 60.0):
        self.motor = motor_model or SimplifiedMotorModel()
        self.eta = eta
        self.count_time = count_time
    
    def score(self, H: float, K: float, L: float, E: float,
              info_gain: float) -> float:
        """
        Score a candidate point.
        
        Parameters
        ----------
        H, K, L, E : float
            Candidate position
        info_gain : float
            Expected information gain at this point
        
        Returns
        -------
        float : Information rate (bits per second)
        """
        move_time = self.motor.move_time(H, K, L, E)
        total_time = self.count_time + move_time
        
        # Information rate
        score = (info_gain ** self.eta) / total_time
        
        return score
    
    def score_batch(self, candidates: np.ndarray, info_gains: np.ndarray) -> np.ndarray:
        """
        Score multiple candidates.
        
        Parameters
        ----------
        candidates : array of shape (n, 4)
            Candidate (H, K, L, E) positions
        info_gains : array of shape (n,)
            Expected information gains
        
        Returns
        -------
        array : Scores for each candidate
        """
        scores = np.zeros(len(candidates))
        
        for i, (cand, info) in enumerate(zip(candidates, info_gains)):
            scores[i] = self.score(*cand, info)
        
        return scores
    
    def select_best(self, candidates: np.ndarray, info_gains: np.ndarray,
                    n_select: int = 1) -> List[int]:
        """Select best candidates accounting for sequential motion."""
        if n_select == 1:
            scores = self.score_batch(candidates, info_gains)
            return [np.argmax(scores)]
        
        # For multiple selections, account for sequential motion
        selected = []
        remaining = list(range(len(candidates)))
        
        # Save motor position
        saved_pos = (self.motor.current_H, self.motor.current_K,
                     self.motor.current_L, self.motor.current_E)
        
        for _ in range(n_select):
            if not remaining:
                break
            
            # Score remaining candidates from current position
            scores = []
            for idx in remaining:
                cand = candidates[idx]
                score = self.score(*cand, info_gains[idx])
                scores.append(score)
            
            # Select best
            best_local = np.argmax(scores)
            best_global = remaining[best_local]
            
            selected.append(best_global)
            remaining.remove(best_global)
            
            # Update motor position
            self.motor.move_to(*candidates[best_global])
        
        # Restore motor position
        (self.motor.current_H, self.motor.current_K,
         self.motor.current_L, self.motor.current_E) = saved_pos
        
        return selected
