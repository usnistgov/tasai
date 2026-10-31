"""
Forecasting for Autonomous Experiments

Forecasting is a key feature of AutoREFL: selecting multiple future measurements
from a single MCMC inference step. This avoids the overhead of re-running MCMC
after each measurement while the instrument is moving/counting.

The algorithm:
1. Run MCMC to get posterior samples
2. Select best measurement point (highest information rate)
3. Simulate the expected outcome of that measurement
4. Update posterior weights using importance sampling (no new MCMC!)
5. Select next best point from updated posterior
6. Repeat for n_forecast points

This "forecasting" allows continuous measurement without waiting for MCMC.

Reference:
Hoogerheide & Heinrich, J. Appl. Cryst. 57, 1192–1204 (2024)
"Since continuous measurement is desirable for practical implementation,
AutoRefl features forecasting, in which the optimal positions of multiple
future measurements are predicted from existing measurements."
"""

import numpy as np
from typing import List, Tuple, Optional, Callable, Dict, Any, Protocol
from dataclasses import dataclass, field
from enum import Enum
import logging

logger = logging.getLogger(__name__)


@dataclass
class ForecastPoint:
    """A forecasted measurement point with metadata."""
    point: Any  # MeasurementPoint or similar
    score: float  # Acquisition function score
    expected_intensity: Any  # Scalar intensity or vector observation mean
    expected_uncertainty: Any  # Scalar error or vector marginal standard deviations
    count_time: float  # Optimal counting time
    cumulative_info_gain: float = 0.0  # Running total of information gain
    forecast_step: int = 0  # Which step in forecast sequence
    expected_covariance: Optional[np.ndarray] = None
    movement_time: float = 0.0
    effective_sample_size: Optional[float] = None


class ForecastStopReason(str, Enum):
    COMPLETED = "completed"
    NO_CANDIDATES = "no_candidates"
    MINIMUM_INFORMATION_GAIN = "minimum_information_gain"
    LOW_EFFECTIVE_SAMPLE_SIZE = "low_effective_sample_size"


class GaussianObservationAdapter(Protocol):
    """Vector observations for the existing mean-outcome forecasting policy.

    Predictions retain all N full parameter particles and have shape (N, k).
    Covariance is the full k-by-k Gaussian observation covariance, including
    fitted nuisance-parameter effects; it is not posterior prediction covariance.
    Count time is the whole observation's duration in seconds. Movement and
    settling remain in movement_time_func, avoiding double counting.
    """

    def predict(self, samples: np.ndarray, candidate: Any) -> np.ndarray: ...

    def covariance(self, candidate: Any, count_time: float,
                   predictions: np.ndarray, weights: np.ndarray) -> np.ndarray: ...

    def count_time(self, candidate: Any, predictions: np.ndarray,
                   weights: np.ndarray) -> float: ...


@dataclass
class ForecastResult:
    """Result of a forecasting operation."""
    points: List[ForecastPoint]
    initial_entropy: float
    final_entropy: float
    total_info_gain: float
    posterior_samples: np.ndarray  # Compatibility name: hypothetical, never measured evidence
    posterior_kind: str = "hypothetical_mean_outcome"
    stop_reason: ForecastStopReason = ForecastStopReason.COMPLETED
    requires_posterior_refresh: bool = False
    stop_details: Dict[str, Any] = field(default_factory=dict)

    @property
    def hypothetical_posterior_samples(self) -> np.ndarray:
        """Explicit name for the forecast-only resampled particles."""
        return self.posterior_samples


class Forecaster:
    """
    Forecasting engine for autonomous experiments.
    
    Selects multiple future measurements from a single posterior,
    updating predictions via importance sampling between selections.
    """
    
    def __init__(self,
                 physics_model: 'PhysicsModel',
                 candidate_generator: Callable[[], List[Any]],
                 poi_indices: List[int] = None,
                 eta: float = 0.7,
                 min_count_time: float = 10.0,
                 max_count_time: float = 600.0,
                 movement_time_func: Callable = None,
                 noise_model: str = 'poisson',
                 background_rate: float = 0.0,
                 observation_adapter: Optional[GaussianObservationAdapter] = None,
                 rng: Optional[np.random.Generator] = None,
                 min_effective_sample_size: Optional[float] = None):
        """
        Initialize forecaster.
        
        Parameters
        ----------
        physics_model : PhysicsModel
            Physics model for predictions
        candidate_generator : callable
            Function that returns list of candidate measurement points
        poi_indices : list of int, optional
            Parameters of interest (indices in posterior samples)
        eta : float
            Aggressiveness parameter for information rate (0 < η ≤ 1)
        min_count_time, max_count_time : float
            Bounds on counting time
        movement_time_func : callable, optional
            Function(from_point, to_point) -> time in seconds
        noise_model : str
            'poisson' or 'gaussian' for measurement noise model
        background_rate : float
            Background count rate (counts/second)
        observation_adapter : GaussianObservationAdapter, optional
            Supplies vector predictions, full observation covariance and duration.
            Uses the same posterior-mean pseudo-observation and entropy/time rule.
            Adapter durations are validated, not clipped to scalar counting bounds.
        rng : numpy.random.Generator, optional
            Seeded random generator for forecasting resampling. None uses np.random.
        min_effective_sample_size : float, optional
            Stop before accepting a winning update below this absolute particle ESS.
            Defaults to 10 for vector observations; disabled for the historical scalar path.
        """
        self.physics_model = physics_model
        self.candidate_generator = candidate_generator
        self.poi_indices = poi_indices
        self.eta = eta
        self.min_count_time = min_count_time
        self.max_count_time = max_count_time
        self.movement_time_func = movement_time_func or (lambda a, b: 30.0)
        self.noise_model = noise_model
        self.background_rate = background_rate
        self.observation_adapter = observation_adapter
        # None preserves np.random.seed compatibility for the historical path.
        self.rng = rng if rng is not None else np.random
        # Preserve historical scalar selection by default; vector observations
        # guard against particle collapse unless an explicit threshold is given.
        self.min_effective_sample_size = (
            10.0 if observation_adapter is not None and min_effective_sample_size is None
            else min_effective_sample_size
        )
        if self.min_effective_sample_size is not None:
            if not np.isfinite(self.min_effective_sample_size) or self.min_effective_sample_size <= 0:
                raise ValueError("min_effective_sample_size must be finite and positive")
    
    def forecast(self,
                 posterior_samples: np.ndarray,
                 n_forecast: int = 3,
                 current_position: Any = None,
                 min_info_gain: float = 0.01) -> ForecastResult:
        """
        Select n_forecast future measurement points.
        
        Parameters
        ----------
        posterior_samples : np.ndarray
            Current posterior samples, shape (n_samples, n_params)
        n_forecast : int
            Number of points to forecast (typically 1-5)
        current_position : MeasurementPoint, optional
            Current instrument position
        min_info_gain : float
            Stop forecasting if expected info gain drops below this
        
        Returns
        -------
        ForecastResult
            Forecasted points and associated metadata
        """
        from .entropy import differential_entropy, gaussian_log_likelihood
        
        # Working copies
        samples = np.array(posterior_samples, dtype=float, copy=True)
        if samples.ndim != 2 or len(samples) < 2 or samples.shape[1] == 0 or not np.all(np.isfinite(samples)):
            raise ValueError("posterior_samples must be a finite (N, parameters) array with N>=2")
        weights = np.ones(len(samples)) / len(samples)
        position = current_position
        
        # Track POI entropy
        if self.poi_indices is not None:
            poi_samples = samples[:, self.poi_indices]
        else:
            poi_samples = samples
        
        initial_entropy = differential_entropy(poi_samples)
        current_entropy = initial_entropy
        
        forecasted_points = []
        cumulative_info_gain = 0.0
        stop_reason = ForecastStopReason.COMPLETED
        stop_details = {}
        requires_posterior_refresh = False

        logger.info(f"Starting forecast: {n_forecast} points, H_initial={initial_entropy:.3f}")
        
        for step in range(n_forecast):
            # Generate candidates
            candidates = self.candidate_generator()
            
            if not candidates:
                logger.warning("No candidates generated, stopping forecast")
                stop_reason = ForecastStopReason.NO_CANDIDATES
                break
            
            # Score each candidate
            best_point = None
            best_score = -np.inf
            best_info = None
            
            for candidate in candidates:
                # Movement time
                if position is not None or self.observation_adapter is not None:
                    t_move = self.movement_time_func(position, candidate)
                else:
                    t_move = 30.0
                
                if self.observation_adapter is None:
                    # Historical scalar policy: preserve its mean-outcome noise,
                    # count-time estimate and likelihood without alteration.
                    predictions = self._predict_intensities(samples, candidate)
                    I_expected = np.average(predictions, weights=weights)
                    I_std = np.sqrt(np.average((predictions - I_expected)**2, weights=weights))
                    t_count = self._optimal_count_time(I_expected, I_std)
                    sigma_meas = self._measurement_uncertainty(I_expected, t_count)
                    log_likes = gaussian_log_likelihood(predictions, I_expected, sigma_meas)
                    observation_covariance = None
                else:
                    predictions, I_expected, observation_covariance, t_count, log_likes = (
                        self._vector_observation(samples, candidate, weights)
                    )
                    sigma_meas = np.sqrt(np.diag(observation_covariance))
                    if not np.isfinite(t_move) or t_move < 0:
                        raise ValueError("movement_time_func must return finite nonnegative seconds")

                # Compute new weights
                new_log_weights = np.log(weights + 1e-300) + log_likes
                new_log_weights -= np.max(new_log_weights)
                new_weights = np.exp(new_log_weights)
                new_weights /= np.sum(new_weights)
                ess = float(1.0 / np.sum(new_weights**2))

                # Entropy of reweighted posterior
                n_resample = min(len(samples), 500)
                indices = self.rng.choice(len(samples), n_resample, p=new_weights, replace=True)
                resampled = samples[indices]
                
                if self.poi_indices is not None:
                    resampled_poi = resampled[:, self.poi_indices]
                else:
                    resampled_poi = resampled
                
                H_after = differential_entropy(resampled_poi)
                info_gain = current_entropy - H_after
                
                # Information rate
                total_time = t_move + t_count
                if info_gain > 0 and total_time > 0:
                    score = (info_gain ** self.eta) / total_time
                else:
                    score = 0.0
                
                if score > best_score:
                    best_score = score
                    best_point = candidate
                    best_info = {
                        'I_expected': I_expected,
                        'sigma_meas': sigma_meas,
                        't_count': t_count,
                        't_move': t_move,
                        'info_gain': info_gain,
                        'H_after': H_after,
                        'new_weights': new_weights,
                        'predictions': predictions,
                        'log_likes': log_likes,
                        'covariance': observation_covariance,
                        'effective_sample_size': ess
                    }
            
            # Do not reset collapsed weights or claim an unreliable forecast.
            # Previously accepted steps remain explicitly hypothetical; actual
            # measurements must provide the posterior refresh.
            if (best_info is not None and self.min_effective_sample_size is not None
                    and best_info['effective_sample_size'] < self.min_effective_sample_size):
                stop_reason = ForecastStopReason.LOW_EFFECTIVE_SAMPLE_SIZE
                requires_posterior_refresh = True
                stop_details = {'forecast_step': step,
                                'effective_sample_size': best_info['effective_sample_size'],
                                'threshold': self.min_effective_sample_size}
                logger.warning("Stopping forecast before low-ESS update: %s", stop_details)
                break
            if best_point is None or best_info['info_gain'] < min_info_gain:
                stop_reason = ForecastStopReason.MINIMUM_INFORMATION_GAIN
                logger.info(f"Stopping forecast at step {step}: info_gain={best_info['info_gain'] if best_info else 0:.4f}")
                break
            
            # Record this forecast point
            cumulative_info_gain += best_info['info_gain']
            
            forecast_point = ForecastPoint(
                point=best_point,
                score=best_score,
                expected_intensity=best_info['I_expected'],
                expected_uncertainty=best_info['sigma_meas'],
                count_time=best_info['t_count'],
                cumulative_info_gain=cumulative_info_gain,
                forecast_step=step,
                expected_covariance=best_info['covariance'],
                movement_time=best_info['t_move'],
                effective_sample_size=best_info['effective_sample_size']
            )
            forecasted_points.append(forecast_point)
            
            logger.info(f"  Step {step}: score={best_score:.4f}, ΔH={best_info['info_gain']:.4f}, "
                       f"t_count={best_info['t_count']:.1f}s")
            
            # Update state for next iteration
            weights = best_info['new_weights']
            current_entropy = best_info['H_after']
            position = best_point
            
            # Set count time on the point
            if hasattr(best_point, 'count_time'):
                best_point.count_time = best_info['t_count']
        
        # Final resampled posterior
        n_final = min(len(samples), 1000)
        final_indices = self.rng.choice(len(samples), n_final, p=weights, replace=True)
        final_samples = samples[final_indices]
        if requires_posterior_refresh and not forecasted_points:
            # No hypothetical update was accepted; return the unchanged copy.
            final_samples = samples.copy()
        
        return ForecastResult(
            points=forecasted_points,
            initial_entropy=initial_entropy,
            final_entropy=current_entropy,
            total_info_gain=cumulative_info_gain,
            posterior_samples=final_samples,
            stop_reason=stop_reason,
            requires_posterior_refresh=requires_posterior_refresh,
            stop_details=stop_details
        )

    def _vector_observation(self, samples, candidate, weights):
        """Generalize the scalar Gaussian mean-outcome update without a new utility."""
        adapter = self.observation_adapter
        predictions = np.asarray(adapter.predict(samples, candidate), dtype=float)
        if (predictions.ndim != 2 or predictions.shape[0] != len(samples)
                or predictions.shape[1] == 0 or not np.all(np.isfinite(predictions))):
            raise ValueError("adapter.predict must return a finite (N, k) observation array")
        count_time = float(adapter.count_time(candidate, predictions, weights))
        if not np.isfinite(count_time) or count_time <= 0:
            raise ValueError("adapter.count_time must return finite positive seconds")
        covariance = np.array(adapter.covariance(candidate, count_time, predictions, weights),
                              dtype=float, copy=True)
        k = predictions.shape[1]
        if (covariance.shape != (k, k) or not np.all(np.isfinite(covariance))
                or not np.allclose(covariance, covariance.T, rtol=1e-10, atol=1e-12)):
            raise ValueError("adapter.covariance must return a finite symmetric (k, k) matrix")
        try:
            chol = np.linalg.cholesky(covariance)
        except np.linalg.LinAlgError as exc:
            raise ValueError("observation covariance must be positive definite") from exc
        expected = np.average(predictions, axis=0, weights=weights)
        whitened = np.linalg.solve(chol, (predictions - expected).T)
        # The Gaussian normalization is constant across particles and cancels
        # when weights are normalized. Retain off-diagonal observation terms.
        log_likes = -0.5 * np.sum(whitened**2, axis=0)
        return predictions, expected, covariance, count_time, log_likes

    def _predict_intensities(self, 
                              samples: np.ndarray,
                              point: Any) -> np.ndarray:
        """Predict intensity at point for each parameter sample."""
        predictions = np.zeros(len(samples))
        
        # Get point coordinates
        h = getattr(point, 'h', 0.0)
        k = getattr(point, 'k', 0.0)
        l = getattr(point, 'l', 0.0)
        E = getattr(point, 'E', getattr(point, 'T', 0.0))
        
        for i, params in enumerate(samples):
            # Set model parameters
            param_dict = self.physics_model.samples_to_params(params)
            self.physics_model.set_parameters(param_dict)
            
            # Predict
            predictions[i] = self.physics_model.compute_intensity(h, k, l, E)
        
        return predictions
    
    def _optimal_count_time(self, 
                             I_expected: float,
                             I_std: float) -> float:
        """
        Compute optimal counting time.
        
        Goal: measurement uncertainty ≈ posterior prediction uncertainty
        
        For Poisson: σ_meas = √(I*t)/t = √(I/t)
        Set σ_meas = I_std → t = I / I_std²
        """
        if I_std < 1e-10 or I_expected < 1e-10:
            return self.min_count_time
        
        # Target: σ_meas ≈ I_std
        # For Poisson: √((I+bkg)*t)/t ≈ I_std
        # Solve: (I+bkg)/t ≈ I_std² → t ≈ (I+bkg)/I_std²
        
        t_optimal = (I_expected + self.background_rate) / (I_std ** 2)
        
        return np.clip(t_optimal, self.min_count_time, self.max_count_time)
    
    def _measurement_uncertainty(self,
                                   I_expected: float,
                                   count_time: float) -> float:
        """
        Compute expected measurement uncertainty.
        
        For counting experiments: σ = √(N)/t where N = (I + bkg) * t
        """
        if count_time <= 0:
            return np.inf
        
        if self.noise_model == 'poisson':
            # Poisson: σ = √(counts) / t
            total_rate = I_expected + self.background_rate
            counts = total_rate * count_time
            sigma = np.sqrt(max(counts, 1.0)) / count_time
        else:
            # Gaussian approximation
            sigma = np.sqrt(I_expected + self.background_rate) / np.sqrt(count_time)
        
        return max(sigma, 1e-10)


class ExperimentRunner:
    """
    Main experiment loop with forecasting.
    
    Coordinates:
    - Physics model
    - MCMC inference
    - Forecasting
    - Instrument control (via interface)
    """
    
    def __init__(self,
                 physics_model: 'PhysicsModel',
                 instrument: 'InstrumentInterface',
                 candidate_generator: Callable,
                 poi_indices: List[int] = None,
                 mcmc_burn: int = 500,
                 mcmc_steps: int = 500,
                 mcmc_pop: int = 8,
                 n_forecast: int = 3,
                 eta: float = 0.7):
        """
        Initialize experiment runner.
        
        Parameters
        ----------
        physics_model : PhysicsModel
            Model for physics predictions
        instrument : InstrumentInterface
            Interface to instrument (or simulator)
        candidate_generator : callable
            Generates candidate measurement points
        poi_indices : list of int
            Parameters of interest
        mcmc_burn, mcmc_steps, mcmc_pop : int
            MCMC parameters
        n_forecast : int
            Number of points to forecast per MCMC run
        eta : float
            Information rate aggressiveness
        """
        self.physics_model = physics_model
        self.instrument = instrument
        self.candidate_generator = candidate_generator
        self.poi_indices = poi_indices
        
        # MCMC settings
        self.mcmc_burn = mcmc_burn
        self.mcmc_steps = mcmc_steps
        self.mcmc_pop = mcmc_pop
        
        # Forecasting settings
        self.n_forecast = n_forecast
        self.eta = eta
        
        # Data storage
        self.measurements: List[Dict] = []
        self.posterior_samples: Optional[np.ndarray] = None
        
        # Movement time estimation
        def estimate_move_time(from_pt, to_pt):
            if from_pt is None:
                return 30.0
            return self.instrument.estimate_move_time(from_pt, to_pt)
        
        # Create forecaster
        self.forecaster = Forecaster(
            physics_model=physics_model,
            candidate_generator=candidate_generator,
            poi_indices=poi_indices,
            eta=eta,
            movement_time_func=estimate_move_time
        )
    
    def add_measurement(self, point: Any, intensity: float, uncertainty: float):
        """Add a measurement to the dataset."""
        self.measurements.append({
            'point': point,
            'h': getattr(point, 'h', 0.0),
            'k': getattr(point, 'k', 0.0),
            'l': getattr(point, 'l', 0.0),
            'E': getattr(point, 'E', getattr(point, 'T', 0.0)),
            'intensity': intensity,
            'uncertainty': uncertainty
        })
    
    def get_data_arrays(self) -> Tuple[np.ndarray, ...]:
        """Get measurement data as arrays."""
        if not self.measurements:
            return tuple(np.array([]) for _ in range(6))
        
        h = np.array([m['h'] for m in self.measurements])
        k = np.array([m['k'] for m in self.measurements])
        l = np.array([m['l'] for m in self.measurements])
        E = np.array([m['E'] for m in self.measurements])
        I = np.array([m['intensity'] for m in self.measurements])
        sigma = np.array([m['uncertainty'] for m in self.measurements])
        
        return h, k, l, E, I, sigma
    
    def run_inference(self) -> np.ndarray:
        """Run MCMC inference on current data."""
        from ..inference.mcmc import MCMCRunner
        
        h, k, l, E, I, sigma = self.get_data_arrays()
        
        if len(I) == 0:
            # No data yet, sample from prior
            return self.physics_model.sample_prior(self.mcmc_steps * self.mcmc_pop)
        
        runner = MCMCRunner(
            self.physics_model,
            burn=self.mcmc_burn,
            steps=self.mcmc_steps,
            pop=self.mcmc_pop
        )
        
        self.posterior_samples = runner.run(h, k, l, E, I, sigma)
        return self.posterior_samples
    
    def run_iteration(self, 
                       current_position: Any = None,
                       execute_measurements: bool = True) -> List[ForecastPoint]:
        """
        Run one iteration: MCMC → Forecast → (optionally) Measure.
        
        Parameters
        ----------
        current_position : MeasurementPoint, optional
            Current instrument position
        execute_measurements : bool
            If True, execute the forecasted measurements
        
        Returns
        -------
        list of ForecastPoint
            Forecasted (and possibly executed) measurement points
        """
        # Run MCMC
        logger.info("Running MCMC inference...")
        posterior = self.run_inference()
        
        # Forecast
        logger.info(f"Forecasting {self.n_forecast} points...")
        forecast_result = self.forecaster.forecast(
            posterior,
            n_forecast=self.n_forecast,
            current_position=current_position
        )
        
        logger.info(f"Forecast complete: {len(forecast_result.points)} points, "
                   f"ΔH_total={forecast_result.total_info_gain:.4f}")
        
        # Execute measurements if requested
        if execute_measurements:
            for fp in forecast_result.points:
                # Measure
                result = self.instrument.measure(fp.point)
                
                # Record
                self.add_measurement(
                    fp.point,
                    result.intensity,
                    result.uncertainty
                )
                
                logger.info(f"Measured: I={result.intensity:.2f}±{result.uncertainty:.2f}")
        
        return forecast_result.points
    
    def run(self,
            max_time: float = None,
            max_iterations: int = None,
            max_measurements: int = None,
            initial_points: List[Any] = None,
            callback: Callable = None) -> Dict:
        """
        Run full autonomous experiment.
        
        Parameters
        ----------
        max_time : float, optional
            Maximum experiment time in seconds
        max_iterations : int, optional
            Maximum MCMC iterations
        max_measurements : int, optional
            Maximum number of measurements
        initial_points : list, optional
            Initial measurement points (if not provided, generates randomly)
        callback : callable, optional
            Called after each iteration with current state
        
        Returns
        -------
        dict
            Experiment results
        """
        import time
        
        start_time = time.time()
        iteration = 0
        current_position = None
        
        # Initial measurements
        if initial_points is None:
            initial_points = self.candidate_generator()[:5]
        
        logger.info(f"Taking {len(initial_points)} initial measurements...")
        for point in initial_points:
            result = self.instrument.measure(point)
            self.add_measurement(point, result.intensity, result.uncertainty)
            current_position = point
            
            if callback:
                callback(self, 'initial', point, result)
        
        # Main loop
        while True:
            # Check stopping conditions
            elapsed = time.time() - start_time
            
            if max_time and elapsed >= max_time:
                logger.info(f"Stopping: max_time reached ({elapsed:.1f}s)")
                break
            
            if max_iterations and iteration >= max_iterations:
                logger.info(f"Stopping: max_iterations reached ({iteration})")
                break
            
            if max_measurements and len(self.measurements) >= max_measurements:
                logger.info(f"Stopping: max_measurements reached ({len(self.measurements)})")
                break
            
            # Run iteration
            logger.info(f"\n=== Iteration {iteration + 1} ===")
            forecast_points = self.run_iteration(current_position, execute_measurements=True)
            
            if forecast_points:
                current_position = forecast_points[-1].point
            
            iteration += 1
            
            if callback:
                callback(self, 'iteration', iteration, forecast_points)
        
        # Final results
        final_posterior = self.run_inference()
        
        return {
            'n_measurements': len(self.measurements),
            'n_iterations': iteration,
            'elapsed_time': time.time() - start_time,
            'measurements': self.measurements,
            'posterior_samples': final_posterior
        }


# =============================================================================
# Utility: Simple candidate generators
# =============================================================================

def grid_candidate_generator(ranges: Dict[str, Tuple[float, float]],
                              n_per_dim: int = 10,
                              point_class: type = None):
    """
    Create a candidate generator for a grid of points.
    
    Parameters
    ----------
    ranges : dict
        {'h': (min, max), 'k': (min, max), ...}
    n_per_dim : int
        Number of points per dimension
    point_class : type
        Class to instantiate for points (must accept **kwargs)
    
    Returns
    -------
    callable
        Candidate generator function
    """
    def generator():
        from itertools import product
        
        dims = list(ranges.keys())
        grids = [np.linspace(ranges[d][0], ranges[d][1], n_per_dim) for d in dims]
        
        candidates = []
        for combo in product(*grids):
            kwargs = {d: v for d, v in zip(dims, combo)}
            if point_class:
                candidates.append(point_class(**kwargs))
            else:
                candidates.append(kwargs)
        
        return candidates
    
    return generator


def random_candidate_generator(ranges: Dict[str, Tuple[float, float]],
                                n_candidates: int = 50,
                                point_class: type = None):
    """
    Create a candidate generator for random points.
    
    Parameters
    ----------
    ranges : dict
        {'h': (min, max), 'E': (min, max), ...}
    n_candidates : int
        Number of candidates per call
    point_class : type
        Class to instantiate for points
    
    Returns
    -------
    callable
        Candidate generator function
    """
    def generator():
        candidates = []
        
        for _ in range(n_candidates):
            kwargs = {
                d: np.random.uniform(lo, hi)
                for d, (lo, hi) in ranges.items()
            }
            if point_class:
                candidates.append(point_class(**kwargs))
            else:
                candidates.append(kwargs)
        
        return candidates
    
    return generator
