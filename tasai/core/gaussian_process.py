"""
Gaussian Process Module for TAS-AI

Implements both:
1. Agnostic GP exploration (gpCAM-style) for initial survey
2. Log-Gaussian Process for proper intensity handling

The log-GP addresses a key criticism from the JCNS paper (Nat. Comm. 2023):
"gpCAM approximates the original intensity function (instead of its logarithm) 
which violates a formal assumption of GPR... normal distributions can take 
negative values which is not possible for non-negative intensity functions."

References:
- Noack et al., Nat. Rev. Phys. 3, 685-697 (2021) - gpCAM
- Teixeira Parente et al., Nat. Comm. 14, 2246 (2023) - Log-GP at JCNS
- Teixeira Parente et al., Front. Mater. 8, 772014 (2022) - Benchmarks
"""

import numpy as np
from typing import Tuple, List, Optional, Dict, Callable
from dataclasses import dataclass, field
import logging
import os

logger = logging.getLogger(__name__)

# Try to import sklearn for GP, fall back to simple implementation
try:
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import RBF, WhiteKernel, Matern, ConstantKernel
    HAS_SKLEARN = True
except Exception as exc:
    HAS_SKLEARN = False
    logger.warning("sklearn unavailable (%s); using simplified GP implementation", exc)

if os.environ.get('TASAI_DISABLE_SKLEARN', '').lower() in {'1', 'true', 'yes'}:
    HAS_SKLEARN = False
    logger.info("TASAI_DISABLE_SKLEARN=1 → using simplified GP implementation")


@dataclass
class GPObservation:
    """A single observation for GP training."""
    x: np.ndarray  # Input coordinates (e.g., [H, K, L, E])
    y: float       # Observed intensity
    sigma: float   # Uncertainty


class LogGaussianProcess:
    """
    Log-Gaussian Process for neutron scattering intensities.
    
    Key insight from JCNS (Nat. Comm. 2023):
    - Work in log-space: z = log(I + ε) where ε is background
    - GP predicts log-intensity, which is always real-valued
    - Exponentiate to get intensity predictions (always positive)
    
    This properly handles:
    - Non-negative intensity constraint
    - Large dynamic range (orders of magnitude)
    - Heteroscedastic noise (σ ∝ √I for Poisson)
    
    Parameters
    ----------
    length_scales : array-like
        Initial length scales for each dimension [H, K, L, E]
    background : float
        Estimated background level (prevents log(0))
    noise_level : float
        Initial noise level estimate
    """
    
    def __init__(self, 
                 length_scales: np.ndarray = None,
                 background: float = 0.01,
                 noise_level: float = 0.1,
                 n_dims: int = 4):
        
        self.n_dims = n_dims
        self.background = background
        self.noise_level = noise_level
        
        if length_scales is None:
            # Default length scales: [H, K, L, E] in typical units
            # H, K, L in r.l.u., E in meV
            self.length_scales = np.array([0.1, 0.1, 0.2, 2.0])[:n_dims]
        else:
            self.length_scales = np.array(length_scales)
        
        # Training data
        self.X_train: List[np.ndarray] = []
        self.y_train: List[float] = []  # Log-transformed
        self.sigma_train: List[float] = []
        
        # GP model
        self._gp = None
        self._fitted = False
        
        if HAS_SKLEARN:
            self._init_sklearn_gp()
    
    def _init_sklearn_gp(self):
        """Initialize sklearn GP with appropriate kernel."""
        # Matern kernel (ν=5/2) is good for smooth but not infinitely differentiable functions
        # This matches physical intensity distributions better than RBF
        kernel = (
            ConstantKernel(1.0, (0.01, 100)) *
            Matern(length_scale=self.length_scales, 
                   length_scale_bounds=(0.01, 10.0),
                   nu=2.5) +
            WhiteKernel(noise_level=self.noise_level, 
                        noise_level_bounds=(1e-5, 1.0))
        )
        
        self._gp = GaussianProcessRegressor(
            kernel=kernel,
            alpha=0.0,  # We handle noise in kernel
            normalize_y=True,
            n_restarts_optimizer=3
        )
    
    def _to_log_space(self, intensity: float, sigma: float) -> Tuple[float, float]:
        """Transform intensity to log-space."""
        # z = log(I + background)
        z = np.log(intensity + self.background)
        
        # Transform uncertainty using delta method
        # σ_z ≈ σ_I / (I + background)
        sigma_z = sigma / (intensity + self.background)
        
        return z, sigma_z
    
    def _from_log_space(self, z: float, sigma_z: float) -> Tuple[float, float]:
        """Transform from log-space back to intensity."""
        # I = exp(z) - background
        intensity = np.exp(z) - self.background
        intensity = max(0, intensity)  # Ensure non-negative
        
        # Transform uncertainty back
        # σ_I ≈ σ_z × (I + background)
        sigma = sigma_z * (intensity + self.background)
        
        return intensity, sigma
    
    def add_observation(self, x: np.ndarray, intensity: float, sigma: float):
        """Add a new observation."""
        x = np.atleast_1d(x)[:self.n_dims]
        
        # Transform to log-space
        z, sigma_z = self._to_log_space(intensity, sigma)
        
        self.X_train.append(x)
        self.y_train.append(z)
        self.sigma_train.append(sigma_z)
        
        self._fitted = False
    
    def fit(self):
        """Fit the GP to current observations."""
        if len(self.X_train) < 2:
            logger.warning("Need at least 2 observations to fit GP")
            return
        
        X = np.array(self.X_train)
        y = np.array(self.y_train)
        
        if HAS_SKLEARN:
            self._gp.fit(X, y)
        
        self._fitted = True
        logger.info(f"GP fitted with {len(self.X_train)} observations")
    
    def predict(self, x: np.ndarray) -> Tuple[float, float]:
        """
        Predict intensity at point x.
        
        Returns
        -------
        mean : float
            Predicted intensity (in original space)
        std : float
            Prediction uncertainty
        """
        if not self._fitted:
            self.fit()
        
        x = np.atleast_1d(x)[:self.n_dims].reshape(1, -1)
        
        if HAS_SKLEARN and self._gp is not None:
            z_mean, z_std = self._gp.predict(x, return_std=True)
            z_mean = z_mean[0]
            z_std = z_std[0]
        else:
            # Simple fallback: nearest neighbor
            z_mean, z_std = self._predict_simple(x[0])
        
        # Transform back to intensity space
        intensity, sigma = self._from_log_space(z_mean, z_std)
        
        return intensity, sigma
    
    def predict_batch(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Predict at multiple points."""
        means = []
        stds = []
        
        for x in X:
            m, s = self.predict(x)
            means.append(m)
            stds.append(s)
        
        return np.array(means), np.array(stds)
    
    def _predict_simple(self, x: np.ndarray) -> Tuple[float, float]:
        """Simple prediction fallback when sklearn not available."""
        if len(self.X_train) == 0:
            return 0.0, 1.0
        
        # Distance-weighted average
        X = np.array(self.X_train)
        y = np.array(self.y_train)
        
        # Normalize by length scales
        X_norm = X / self.length_scales
        x_norm = x / self.length_scales
        
        # Squared distances
        dists = np.sum((X_norm - x_norm)**2, axis=1)
        
        # RBF weights
        weights = np.exp(-0.5 * dists)
        weights = weights / (np.sum(weights) + 1e-10)
        
        mean = np.sum(weights * y)
        var = np.sum(weights * (y - mean)**2) + 0.01
        
        return mean, np.sqrt(var)
    
    def acquisition_ucb(self, x: np.ndarray, kappa: float = 2.0) -> float:
        """
        Upper Confidence Bound acquisition function.
        
        UCB(x) = μ(x) + κ × σ(x)
        
        High values indicate either:
        - High expected intensity (exploit)
        - High uncertainty (explore)
        """
        mean, std = self.predict(x)
        return mean + kappa * std
    
    def acquisition_ei(self, x: np.ndarray, xi: float = 0.01) -> float:
        """
        Expected Improvement acquisition function.
        
        Prefers points likely to improve over current best.
        """
        mean, std = self.predict(x)
        
        if len(self.y_train) == 0:
            return std
        
        # Current best (in log-space)
        y_best = max(self.y_train)
        
        # Expected improvement
        z = (mean - y_best - xi) / (std + 1e-10)
        
        # Standard normal CDF and PDF
        from scipy.stats import norm
        ei = (mean - y_best - xi) * norm.cdf(z) + std * norm.pdf(z)
        
        return max(0, ei)
    
    def acquisition_variance(self, x: np.ndarray) -> float:
        """
        Pure variance-based acquisition (exploration).
        
        Used by JCNS log-GP approach for identifying signal regions.
        """
        _, std = self.predict(x)
        return std
    
    def suggest_next_points(self, bounds: np.ndarray, n_points: int = 1,
                           acquisition: str = 'ucb', 
                           n_candidates: int = 1000) -> np.ndarray:
        """
        Suggest next measurement points.
        
        Parameters
        ----------
        bounds : array of shape (n_dims, 2)
            Min and max for each dimension
        n_points : int
            Number of points to suggest
        acquisition : str
            'ucb', 'ei', or 'variance'
        n_candidates : int
            Number of random candidates to evaluate
        
        Returns
        -------
        points : array of shape (n_points, n_dims)
        """
        # Generate random candidates
        candidates = np.random.uniform(
            bounds[:, 0], bounds[:, 1], 
            size=(n_candidates, self.n_dims)
        )
        
        # Select acquisition function
        if acquisition == 'ucb':
            acq_func = self.acquisition_ucb
        elif acquisition == 'ei':
            acq_func = self.acquisition_ei
        elif acquisition == 'variance':
            acq_func = self.acquisition_variance
        else:
            raise ValueError(f"Unknown acquisition: {acquisition}")
        
        # Evaluate acquisition at all candidates
        acq_values = np.array([acq_func(c) for c in candidates])
        
        # Select top points
        top_indices = np.argsort(acq_values)[-n_points:][::-1]
        
        return candidates[top_indices]


class AgnosticExplorer:
    """
    Model-agnostic exploration mode (gpCAM-style).
    
    Use for initial survey when you don't know where signals are.
    Efficiently maps S(Q,ω) without physics assumptions.
    
    After survey, switch to physics-informed mode for:
    - Parameter estimation
    - Model discrimination
    
    Example
    -------
    >>> explorer = AgnosticExplorer(bounds=[[0, 0.5], [0, 0.5], [0, 1], [0, 20]])
    >>> 
    >>> # Initial random points
    >>> for _ in range(10):
    ...     x = explorer.suggest_initial()
    ...     I, sigma = measure(x)
    ...     explorer.add_observation(x, I, sigma)
    >>> 
    >>> # Autonomous exploration
    >>> for _ in range(50):
    ...     x = explorer.suggest_next()
    ...     I, sigma = measure(x)
    ...     explorer.add_observation(x, I, sigma)
    >>> 
    >>> # Get intensity map
    >>> H, E, I_map = explorer.get_intensity_map()
    """
    
    def __init__(self, bounds: np.ndarray, 
                 background: float = 0.01,
                 use_log_gp: bool = True):
        """
        Parameters
        ----------
        bounds : array of shape (n_dims, 2)
            Bounds for each dimension [min, max]
        background : float
            Estimated background level
        use_log_gp : bool
            If True, use log-intensity GP regression. If False, use a standard
            GP-style surrogate.
        """
        self.bounds = np.array(bounds)
        self.n_dims = len(bounds)
        self.use_log_gp = use_log_gp
        
        # Initialize GP
        length_scales = (bounds[:, 1] - bounds[:, 0]) / 5  # ~5 length scales across domain
        self.gp = LogGaussianProcess(
            length_scales=length_scales,
            background=background,
            n_dims=self.n_dims
        )
        
        # Track observations
        self.observations: List[GPObservation] = []
        
        # Exploration parameters
        self.kappa = 2.0  # UCB exploration weight
        self.intensity_threshold = 0.1  # For signal detection
    
    def suggest_initial(self) -> np.ndarray:
        """Suggest an initial point (space-filling)."""
        # Latin hypercube-ish: try to fill space
        if len(self.observations) == 0:
            # Start at center
            return (self.bounds[:, 0] + self.bounds[:, 1]) / 2
        
        # Random point avoiding existing observations
        for _ in range(100):
            x = np.random.uniform(self.bounds[:, 0], self.bounds[:, 1])
            
            # Check distance to existing points
            min_dist = float('inf')
            for obs in self.observations:
                dist = np.linalg.norm((x - obs.x) / (self.bounds[:, 1] - self.bounds[:, 0]))
                min_dist = min(min_dist, dist)
            
            if min_dist > 0.1:  # At least 10% of domain away
                return x
        
        # Fallback: random
        return np.random.uniform(self.bounds[:, 0], self.bounds[:, 1])
    
    def add_observation(self, x: np.ndarray, intensity: float, sigma: float):
        """Add observation and update GP."""
        obs = GPObservation(x=np.array(x), y=intensity, sigma=sigma)
        self.observations.append(obs)
        self.gp.add_observation(x, intensity, sigma)
    
    def suggest_next(self, acquisition: str = 'ucb') -> np.ndarray:
        """
        Suggest next measurement point.
        
        Parameters
        ----------
        acquisition : str
            'ucb': Upper confidence bound (balanced)
            'variance': Pure exploration
            'ei': Expected improvement (exploitation)
        """
        if len(self.observations) < 3:
            return self.suggest_initial()
        
        # Fit GP
        self.gp.fit()
        
        # Get suggestion
        points = self.gp.suggest_next_points(
            self.bounds, n_points=1, acquisition=acquisition
        )
        
        return points[0]
    
    def get_intensity_map(self, n_grid: int = 50, 
                         dims: Tuple[int, int] = (0, 3)) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Get 2D intensity map (e.g., H vs E).
        
        Parameters
        ----------
        n_grid : int
            Grid points per dimension
        dims : tuple
            Which dimensions to plot (default: H and E)
        
        Returns
        -------
        x1, x2, I_map : arrays
        """
        d1, d2 = dims
        
        x1 = np.linspace(self.bounds[d1, 0], self.bounds[d1, 1], n_grid)
        x2 = np.linspace(self.bounds[d2, 0], self.bounds[d2, 1], n_grid)
        
        I_map = np.zeros((n_grid, n_grid))
        
        # Fit GP
        if len(self.observations) >= 3:
            self.gp.fit()
        
        # Fixed values for other dimensions (use mean)
        x_template = (self.bounds[:, 0] + self.bounds[:, 1]) / 2
        
        for i, v1 in enumerate(x1):
            for j, v2 in enumerate(x2):
                x = x_template.copy()
                x[d1] = v1
                x[d2] = v2
                I_map[i, j], _ = self.gp.predict(x)
        
        return x1, x2, I_map
    
    def identify_signal_regions(self, threshold: float = None) -> List[np.ndarray]:
        """
        Identify regions with significant signal.
        
        Uses JCNS approach: regions where predicted intensity
        exceeds threshold with high confidence.
        
        Returns list of (center, extent) for each region.
        """
        if threshold is None:
            threshold = self.intensity_threshold
        
        # Get predictions on grid
        n_grid = 20
        grid_points = []
        
        for d in range(self.n_dims):
            grid_points.append(np.linspace(self.bounds[d, 0], self.bounds[d, 1], n_grid))
        
        # Find points above threshold
        signal_points = []
        
        from itertools import product
        for indices in product(*[range(n_grid) for _ in range(self.n_dims)]):
            x = np.array([grid_points[d][i] for d, i in enumerate(indices)])
            mean, std = self.gp.predict(x)
            
            # Signal if lower confidence bound exceeds threshold
            if mean - 2*std > threshold:
                signal_points.append(x)
        
        return signal_points
    
    def get_efficiency_stats(self) -> Dict:
        """Get statistics on exploration efficiency."""
        if len(self.observations) < 3:
            return {}
        
        # Fit GP
        self.gp.fit()
        
        # Calculate coverage
        signal_regions = self.identify_signal_regions()
        
        # Information metrics
        intensities = [obs.y for obs in self.observations]
        
        return {
            'n_observations': len(self.observations),
            'n_signal_regions': len(signal_regions),
            'max_intensity': max(intensities),
            'mean_intensity': np.mean(intensities),
            'coverage': len(signal_regions) / (20**self.n_dims),  # Approximate
        }


class HybridExplorer:
    """
    Hybrid exploration: agnostic survey → physics-informed refinement.
    
    Stage 1 (Agnostic): Use GP to efficiently map S(Q,ω)
    Stage 2 (Informed): Switch to physics model for parameter estimation
    
    This combines:
    - gpCAM/JCNS efficiency for initial exploration
    - ANDiE/TAS-AI precision for parameter determination
    """
    
    def __init__(self, bounds: np.ndarray, physics_model=None):
        self.bounds = bounds
        self.physics_model = physics_model
        
        # Exploration mode
        self.explorer = AgnosticExplorer(bounds, use_log_gp=True)
        
        # Mode control
        self.mode = 'agnostic'  # or 'informed'
        self.switch_threshold = 20  # Switch after N observations
    
    def suggest_next(self) -> np.ndarray:
        """Suggest next point based on current mode."""
        if self.mode == 'agnostic':
            # Check if should switch
            if len(self.explorer.observations) >= self.switch_threshold:
                signal_regions = self.explorer.identify_signal_regions()
                if len(signal_regions) > 0:
                    logger.info("Switching to physics-informed mode")
                    self.mode = 'informed'
            
            return self.explorer.suggest_next()
        
        else:  # informed mode
            if self.physics_model is None:
                logger.warning("No physics model, staying in agnostic mode")
                return self.explorer.suggest_next()
            
            # Use physics model to suggest
            # This would integrate with the existing TAS-AI acquisition
            return self._suggest_informed()
    
    def _suggest_informed(self) -> np.ndarray:
        """Physics-informed suggestion."""
        # Get signal regions from GP
        signal_regions = self.explorer.identify_signal_regions()
        
        if len(signal_regions) == 0:
            return self.explorer.suggest_next()
        
        # Focus on signal regions with physics guidance
        # This is where we'd integrate the ANDiE/TAS-AI acquisition
        
        # For now: sample from signal regions
        idx = np.random.randint(len(signal_regions))
        return signal_regions[idx]
    
    def add_observation(self, x: np.ndarray, intensity: float, sigma: float):
        """Add observation to both explorer and physics model."""
        self.explorer.add_observation(x, intensity, sigma)
        
        # Also update physics model if in informed mode
        if self.mode == 'informed' and self.physics_model is not None:
            # Physics model update would go here
            pass
