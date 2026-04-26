"""
Acquisition Functions for Active Learning

This module provides swappable acquisition functions for autonomous experiments.
The key abstraction is that acquisition functions score candidate measurement points
based on expected information gain, uncertainty reduction, or other criteria.

Implementations:
- HHAcquisition: Hoogerheide-Heinrich information-rate maximization (from AutoREFL)
- UncertaintyAcquisition: Simple uncertainty sampling
- GPAcquisition: Gaussian process-based (gpCAM-style)
- ANDiEAcquisition: Physics-informed acquisition (from ANDiE)

References:
- Hoogerheide & Heinrich, J. Appl. Cryst. 57, 1192–1204 (2024)
- McDannald et al., Appl. Phys. Rev. 9, 021408 (2022)
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple, Callable
import numpy as np
from scipy import stats
import logging

logger = logging.getLogger(__name__)


@dataclass
class AcquisitionResult:
    """Result from acquisition function evaluation."""
    point: 'MeasurementPoint'  # Forward reference
    score: float  # Higher is better
    components: Dict[str, float] = None  # Breakdown of score
    metadata: Dict = None


class AcquisitionFunction(ABC):
    """
    Abstract base class for acquisition functions.
    
    Acquisition functions evaluate candidate measurement points and return
    scores indicating their expected value for the experiment goals.
    """
    
    @abstractmethod
    def evaluate(self, 
                 candidates: List['MeasurementPoint'],
                 posterior_samples: np.ndarray,
                 current_position: 'MeasurementPoint' = None,
                 **kwargs) -> List[AcquisitionResult]:
        """
        Evaluate acquisition function for candidate points.
        
        Parameters
        ----------
        candidates : list of MeasurementPoint
            Candidate measurement points to evaluate
        posterior_samples : np.ndarray
            MCMC samples from current posterior, shape (n_samples, n_params)
        current_position : MeasurementPoint, optional
            Current instrument position (for movement time calculation)
        
        Returns
        -------
        list of AcquisitionResult
            Scores for each candidate (higher is better)
        """
        pass
    
    @abstractmethod
    def select_next(self,
                    candidates: List['MeasurementPoint'],
                    posterior_samples: np.ndarray,
                    n_select: int = 1,
                    current_position: 'MeasurementPoint' = None,
                    **kwargs) -> List['MeasurementPoint']:
        """
        Select the best n_select points from candidates.
        
        Parameters
        ----------
        candidates : list of MeasurementPoint
            Candidate measurement points
        posterior_samples : np.ndarray
            MCMC samples from current posterior
        n_select : int
            Number of points to select
        current_position : MeasurementPoint, optional
            Current instrument position
        
        Returns
        -------
        list of MeasurementPoint
            Best points to measure next
        """
        pass
    
    def set_parameters_of_interest(self, param_indices: List[int]):
        """Set which parameters to optimize for (by index in posterior)."""
        self.poi_indices = param_indices
    
    def estimate_count_time(self, 
                            point: 'MeasurementPoint',
                            posterior_samples: np.ndarray,
                            physics_model: 'PhysicsModel' = None) -> float:
        """
        Estimate optimal count time for a measurement.
        
        Default implementation returns fixed time; override for adaptive counting.
        """
        return getattr(self, 'default_count_time', 60.0)


class HHAcquisition(AcquisitionFunction):
    """
    Hoogerheide-Heinrich acquisition function.
    
    Maximizes information acquisition RATE: ΔH / Δt
    where ΔH is the expected entropy reduction in parameters of interest
    and Δt includes both counting time and movement time.
    
    From: Hoogerheide & Heinrich, J. Appl. Cryst. 57, 1192–1204 (2024)
    
    Key insight: "Fast convergence of the global function does not necessarily 
    equate to fast convergence of any individual parameter describing that function."
    """
    
    def __init__(self,
                 physics_model: 'PhysicsModel',
                 poi_indices: List[int] = None,
                 eta: float = 0.7,
                 min_count_time: float = 10.0,
                 max_count_time: float = 600.0,
                 movement_time_func: Callable = None,
                 n_entropy_samples: int = 1000):
        """
        Initialize HH acquisition function.
        
        Parameters
        ----------
        physics_model : PhysicsModel
            Physics model for forward calculations
        poi_indices : list of int, optional
            Indices of parameters of interest in posterior
            If None, uses all parameters
        eta : float
            Aggressiveness parameter (0-1). Higher = more exploration
        min_count_time, max_count_time : float
            Bounds on counting time in seconds
        movement_time_func : callable, optional
            Function(from_point, to_point) -> time in seconds
        n_entropy_samples : int
            Number of samples for entropy estimation
        """
        self.physics_model = physics_model
        self.poi_indices = poi_indices
        self.eta = eta
        self.min_count_time = min_count_time
        self.max_count_time = max_count_time
        self.movement_time_func = movement_time_func or (lambda a, b: 30.0)
        self.n_entropy_samples = n_entropy_samples
        self.default_count_time = 60.0
    
    def evaluate(self,
                 candidates: List['MeasurementPoint'],
                 posterior_samples: np.ndarray,
                 current_position: 'MeasurementPoint' = None,
                 **kwargs) -> List[AcquisitionResult]:
        """Evaluate HH acquisition for each candidate."""
        from .entropy import differential_entropy, information_rate
        
        results = []
        
        # Current entropy of POI
        if self.poi_indices is not None:
            poi_samples = posterior_samples[:, self.poi_indices]
        else:
            poi_samples = posterior_samples
        
        H_current = differential_entropy(poi_samples)
        
        for candidate in candidates:
            # Estimate movement time
            if current_position is not None:
                t_move = self.movement_time_func(current_position, candidate)
            else:
                t_move = 30.0
            
            # Estimate optimal count time
            t_count = self.estimate_count_time(candidate, posterior_samples)
            candidate.count_time = t_count
            
            # Estimate expected entropy after measurement
            H_expected = self._estimate_posterior_entropy(
                candidate, posterior_samples
            )
            
            # Information gain
            delta_H = max(0, H_current - H_expected)
            
            # Information rate (the HH criterion)
            total_time = t_move + t_count
            info_rate_val = information_rate(delta_H, t_count, t_move, self.eta)
            
            results.append(AcquisitionResult(
                point=candidate,
                score=info_rate_val,
                components={
                    'delta_H': delta_H,
                    'H_current': H_current,
                    'H_expected': H_expected,
                    't_move': t_move,
                    't_count': t_count,
                    'info_rate': info_rate_val
                }
            ))
        
        return results
    
    def select_next(self,
                    candidates: List['MeasurementPoint'],
                    posterior_samples: np.ndarray,
                    n_select: int = 1,
                    current_position: 'MeasurementPoint' = None,
                    **kwargs) -> List['MeasurementPoint']:
        """Select best n_select points."""
        results = self.evaluate(candidates, posterior_samples, current_position)
        
        # Sort by score (descending)
        results.sort(key=lambda r: r.score, reverse=True)
        
        # For multiple selections, update current_position sequentially
        selected = []
        pos = current_position
        
        for i in range(min(n_select, len(results))):
            if i == 0:
                best = results[0]
            else:
                # Re-evaluate remaining candidates from new position
                remaining = [r.point for r in results if r.point not in selected]
                if not remaining:
                    break
                new_results = self.evaluate(remaining, posterior_samples, pos)
                new_results.sort(key=lambda r: r.score, reverse=True)
                best = new_results[0]
            
            selected.append(best.point)
            pos = best.point
        
        return selected
    
    def _estimate_posterior_entropy(self,
                                     candidate: 'MeasurementPoint',
                                     prior_samples: np.ndarray) -> float:
        """
        Estimate entropy of posterior after hypothetical measurement.
        
        Uses importance sampling to approximate the updated posterior.
        """
        from .entropy import (
            differential_entropy, gaussian_log_likelihood, 
            resample_posterior, effective_sample_size
        )
        
        # For each prior sample, compute expected measurement
        n_samples = min(len(prior_samples), self.n_entropy_samples)
        indices = np.random.choice(len(prior_samples), n_samples, replace=False)
        samples = prior_samples[indices]
        
        # Get point coordinates
        h = getattr(candidate, 'h', 0.0)
        k = getattr(candidate, 'k', 0.0)
        l = getattr(candidate, 'l', 0.0)
        E = getattr(candidate, 'E', getattr(candidate, 'T', 0.0))
        
        # Get predicted intensities for each parameter setting
        predictions = []
        for i, params in enumerate(samples):
            # Set model parameters
            param_dict = self.physics_model.samples_to_params(params)
            self.physics_model.set_parameters(param_dict)
            
            # Predict intensity at candidate point
            I_pred = self.physics_model.compute_intensity(h, k, l, E)
            predictions.append(I_pred)
        
        predictions = np.array(predictions)
        
        # Assume we'll measure something close to mean prediction
        I_measured = np.mean(predictions)
        count_time = getattr(candidate, 'count_time', self.default_count_time)
        sigma_measured = np.sqrt(max(I_measured, 1.0) * count_time) / count_time
        
        # Compute likelihood weights for importance sampling
        log_likes = gaussian_log_likelihood(predictions, I_measured, sigma_measured)
        
        # Resample
        resampled = resample_posterior(samples, log_likes)
        
        # Compute entropy of resampled distribution (POI only)
        if self.poi_indices is not None:
            resampled_poi = resampled[:, self.poi_indices]
        else:
            resampled_poi = resampled
        
        return differential_entropy(resampled_poi)
    
    def estimate_count_time(self,
                            point: 'MeasurementPoint',
                            posterior_samples: np.ndarray,
                            physics_model: 'PhysicsModel' = None) -> float:
        """
        Estimate optimal counting time.
        
        Balances measurement uncertainty against posterior uncertainty.
        """
        model = physics_model or self.physics_model
        
        # Get point coordinates
        h = getattr(point, 'h', 0.0)
        k = getattr(point, 'k', 0.0)
        l = getattr(point, 'l', 0.0)
        E = getattr(point, 'E', getattr(point, 'T', 0.0))
        
        # Get predicted intensity spread from posterior
        predictions = []
        n_samples = min(100, len(posterior_samples))
        for params in posterior_samples[:n_samples]:
            param_dict = model.samples_to_params(params)
            model.set_parameters(param_dict)
            I_pred = model.compute_intensity(h, k, l, E)
            predictions.append(I_pred)
        
        I_mean = np.mean(predictions)
        I_std = np.std(predictions)
        
        if I_std < 1e-10 or I_mean < 1e-10:
            return self.default_count_time
        
        # Target: measurement uncertainty ≈ posterior uncertainty
        # σ_meas = √(I*t) / t = √(I/t)
        # Set σ_meas = I_std → t = I_mean / I_std²
        t_optimal = I_mean / (I_std ** 2)
        
        return np.clip(t_optimal, self.min_count_time, self.max_count_time)


class UncertaintyAcquisition(AcquisitionFunction):
    """
    Simple uncertainty sampling acquisition.
    
    Selects points where the model prediction has highest uncertainty
    (variance across posterior samples).
    """
    
    def __init__(self,
                 physics_model: 'PhysicsModel',
                 movement_time_func: Callable = None):
        self.physics_model = physics_model
        self.movement_time_func = movement_time_func or (lambda a, b: 30.0)
        self.default_count_time = 60.0
    
    def evaluate(self,
                 candidates: List['MeasurementPoint'],
                 posterior_samples: np.ndarray,
                 current_position: 'MeasurementPoint' = None,
                 **kwargs) -> List[AcquisitionResult]:
        """Evaluate uncertainty at each candidate."""
        results = []
        
        for candidate in candidates:
            # Compute prediction variance
            predictions = []
            for params in posterior_samples[:100]:
                param_dict = self.physics_model.samples_to_params(params)
                self.physics_model.set_parameters(param_dict)
                I_pred = self.physics_model.compute_intensity(
                    candidate.h, candidate.k, candidate.l, candidate.E
                )
                predictions.append(I_pred)
            
            variance = np.var(predictions)
            
            results.append(AcquisitionResult(
                point=candidate,
                score=variance,
                components={'variance': variance, 'mean': np.mean(predictions)}
            ))
        
        return results
    
    def select_next(self,
                    candidates: List['MeasurementPoint'],
                    posterior_samples: np.ndarray,
                    n_select: int = 1,
                    current_position: 'MeasurementPoint' = None,
                    **kwargs) -> List['MeasurementPoint']:
        results = self.evaluate(candidates, posterior_samples, current_position)
        results.sort(key=lambda r: r.score, reverse=True)
        return [r.point for r in results[:n_select]]


class ANDiEAcquisition(AcquisitionFunction):
    """
    ANDiE-style physics-informed acquisition.
    
    Uses model comparison (Bayes factors) to select measurements that best
    discriminate between competing physics hypotheses.
    
    From: McDannald et al., Appl. Phys. Rev. 9, 021408 (2022)
    """
    
    def __init__(self,
                 hypothesis_models: List['PhysicsModel'],
                 prior_weights: List[float] = None,
                 movement_time_func: Callable = None):
        """
        Initialize ANDiE acquisition.
        
        Parameters
        ----------
        hypothesis_models : list of PhysicsModel
            Competing physics models (e.g., Ising, Weiss, first-order)
        prior_weights : list of float, optional
            Prior probability for each model
        """
        self.models = hypothesis_models
        self.n_models = len(hypothesis_models)
        
        if prior_weights is None:
            self.weights = np.ones(self.n_models) / self.n_models
        else:
            self.weights = np.array(prior_weights) / np.sum(prior_weights)
        
        self.movement_time_func = movement_time_func or (lambda a, b: 30.0)
        self.default_count_time = 60.0
    
    def update_weights(self, measurement: 'MeasurementResult'):
        """Update model weights based on new measurement (Bayesian update)."""
        likelihoods = []
        
        for model in self.models:
            # Compute likelihood of measurement under this model
            I_pred = model.compute_intensity(
                measurement.point.h, measurement.point.k,
                measurement.point.l, measurement.point.E
            )
            
            # Gaussian likelihood
            log_like = -0.5 * ((measurement.intensity - I_pred) / measurement.uncertainty) ** 2
            likelihoods.append(np.exp(log_like))
        
        likelihoods = np.array(likelihoods)
        
        # Bayesian update
        posterior = self.weights * likelihoods
        self.weights = posterior / np.sum(posterior)
        
        logger.info(f"Updated model weights: {self.weights}")
    
    def evaluate(self,
                 candidates: List['MeasurementPoint'],
                 posterior_samples: np.ndarray = None,
                 current_position: 'MeasurementPoint' = None,
                 **kwargs) -> List[AcquisitionResult]:
        """
        Evaluate model-discrimination potential.
        
        Score = expected KL divergence between model predictions.
        """
        results = []
        
        for candidate in candidates:
            # Get predictions from each model
            predictions = []
            for model in self.models:
                I_pred = model.compute_intensity(
                    candidate.h, candidate.k, candidate.l, candidate.E
                )
                predictions.append(I_pred)
            
            predictions = np.array(predictions)
            
            # Score: weighted variance of predictions
            # High variance = models disagree = informative measurement
            weighted_mean = np.sum(self.weights * predictions)
            weighted_var = np.sum(self.weights * (predictions - weighted_mean) ** 2)
            
            # Also consider: are any models already ruled out?
            # Don't waste time measuring where all remaining models agree
            active_models = self.weights > 0.01
            if np.sum(active_models) > 1:
                active_var = np.var(predictions[active_models])
            else:
                active_var = 0.0
            
            score = active_var
            
            results.append(AcquisitionResult(
                point=candidate,
                score=score,
                components={
                    'weighted_var': weighted_var,
                    'active_var': active_var,
                    'predictions': predictions.tolist(),
                    'n_active_models': np.sum(active_models)
                }
            ))
        
        return results
    
    def select_next(self,
                    candidates: List['MeasurementPoint'],
                    posterior_samples: np.ndarray = None,
                    n_select: int = 1,
                    current_position: 'MeasurementPoint' = None,
                    **kwargs) -> List['MeasurementPoint']:
        results = self.evaluate(candidates, posterior_samples, current_position)
        results.sort(key=lambda r: r.score, reverse=True)
        return [r.point for r in results[:n_select]]
    
    def get_best_model(self) -> Tuple['PhysicsModel', float]:
        """Return the model with highest posterior weight."""
        best_idx = np.argmax(self.weights)
        return self.models[best_idx], self.weights[best_idx]


class CompositeAcquisition(AcquisitionFunction):
    """
    Combines multiple acquisition functions with configurable weights.
    
    Useful for balancing exploration (uncertainty) vs exploitation (information rate)
    or combining physics-informed and model-free approaches.
    """
    
    def __init__(self,
                 acquisition_functions: List[AcquisitionFunction],
                 weights: List[float] = None):
        """
        Initialize composite acquisition.
        
        Parameters
        ----------
        acquisition_functions : list of AcquisitionFunction
            Component acquisition functions
        weights : list of float
            Relative weights (will be normalized)
        """
        self.functions = acquisition_functions
        
        if weights is None:
            self.weights = np.ones(len(acquisition_functions))
        else:
            self.weights = np.array(weights)
        
        self.weights /= np.sum(self.weights)
        self.default_count_time = 60.0
    
    def evaluate(self,
                 candidates: List['MeasurementPoint'],
                 posterior_samples: np.ndarray,
                 current_position: 'MeasurementPoint' = None,
                 **kwargs) -> List[AcquisitionResult]:
        """Combine scores from all component functions."""
        # Get results from each function
        all_results = []
        for func in self.functions:
            all_results.append(
                func.evaluate(candidates, posterior_samples, current_position, **kwargs)
            )
        
        # Normalize scores within each function
        normalized = []
        for results in all_results:
            scores = np.array([r.score for r in results])
            if np.std(scores) > 0:
                scores = (scores - np.mean(scores)) / np.std(scores)
            else:
                scores = np.zeros_like(scores)
            normalized.append(scores)
        
        normalized = np.array(normalized)  # (n_functions, n_candidates)
        
        # Weighted combination
        combined_scores = np.dot(self.weights, normalized)
        
        # Build results
        results = []
        for i, candidate in enumerate(candidates):
            components = {
                f'func_{j}': all_results[j][i].score 
                for j in range(len(self.functions))
            }
            
            results.append(AcquisitionResult(
                point=candidate,
                score=combined_scores[i],
                components=components
            ))
        
        return results
    
    def select_next(self,
                    candidates: List['MeasurementPoint'],
                    posterior_samples: np.ndarray,
                    n_select: int = 1,
                    current_position: 'MeasurementPoint' = None,
                    **kwargs) -> List['MeasurementPoint']:
        results = self.evaluate(candidates, posterior_samples, current_position)
        results.sort(key=lambda r: r.score, reverse=True)
        return [r.point for r in results[:n_select]]
