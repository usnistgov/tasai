"""
Physics Model Interface

Abstract interface for physics models used in autonomous experiments.
Models can represent:
- S(Q,ω) for inelastic scattering (TAS, TOF)
- Order parameters for phase transitions (ANDiE-style)
- Reflectivity profiles (AutoREFL)
- Any other parametric physical model

The key requirement is that models must:
1. Have tunable parameters with priors
2. Predict observables (intensity) at measurement points
3. Support parameter serialization for MCMC
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Union, Any
import numpy as np


@dataclass
class Parameter:
    """A model parameter with bounds, prior, and metadata."""
    name: str
    value: float
    bounds: Tuple[float, float] = (-np.inf, np.inf)
    fixed: bool = False
    
    # Prior specification
    prior_type: str = 'uniform'  # 'uniform', 'normal', 'log_uniform'
    prior_params: Dict = field(default_factory=dict)
    
    # Metadata
    units: str = ''
    description: str = ''
    
    def sample_prior(self, n: int = 1) -> np.ndarray:
        """Draw samples from prior distribution."""
        if self.prior_type == 'uniform':
            return np.random.uniform(self.bounds[0], self.bounds[1], n)
        elif self.prior_type == 'normal':
            mu = self.prior_params.get('mu', self.value)
            sigma = self.prior_params.get('sigma', 1.0)
            samples = np.random.normal(mu, sigma, n)
            return np.clip(samples, self.bounds[0], self.bounds[1])
        elif self.prior_type == 'log_uniform':
            log_low = np.log(max(self.bounds[0], 1e-10))
            log_high = np.log(self.bounds[1])
            return np.exp(np.random.uniform(log_low, log_high, n))
        else:
            raise ValueError(f"Unknown prior type: {self.prior_type}")
    
    def log_prior(self, value: float) -> float:
        """Compute log prior probability."""
        if value < self.bounds[0] or value > self.bounds[1]:
            return -np.inf
        
        if self.prior_type == 'uniform':
            return -np.log(self.bounds[1] - self.bounds[0])
        elif self.prior_type == 'normal':
            mu = self.prior_params.get('mu', self.value)
            sigma = self.prior_params.get('sigma', 1.0)
            return -0.5 * ((value - mu) / sigma) ** 2 - np.log(sigma * np.sqrt(2 * np.pi))
        elif self.prior_type == 'log_uniform':
            return -np.log(value) - np.log(np.log(self.bounds[1] / max(self.bounds[0], 1e-10)))
        else:
            return 0.0


class PhysicsModel(ABC):
    """
    Abstract interface for physics models.
    
    Subclasses implement specific physics:
    - OrderParameterModel: Phase transitions (ANDiE)
    - SpinWaveModel: Magnetic excitations (TAS-AI)
    - ReflectivityModel: Layered structures (AutoREFL)
    """
    
    def __init__(self):
        self._parameters: List[Parameter] = []
        self._param_index: Dict[str, int] = {}
    
    @property
    def parameters(self) -> List[Parameter]:
        """Return list of model parameters."""
        return self._parameters
    
    @property
    def free_parameters(self) -> List[Parameter]:
        """Return only non-fixed parameters."""
        return [p for p in self._parameters if not p.fixed]
    
    @property
    def n_free(self) -> int:
        """Number of free parameters."""
        return len(self.free_parameters)
    
    def get_parameter(self, name: str) -> Parameter:
        """Get parameter by name."""
        idx = self._param_index.get(name)
        if idx is None:
            raise KeyError(f"Unknown parameter: {name}")
        return self._parameters[idx]
    
    def set_parameter(self, name: str, value: float):
        """Set a single parameter value."""
        param = self.get_parameter(name)
        param.value = np.clip(value, param.bounds[0], param.bounds[1])
    
    def set_parameters(self, values: Dict[str, float]):
        """Set multiple parameter values."""
        for name, value in values.items():
            if name in self._param_index:
                self.set_parameter(name, value)
    
    def get_values(self) -> Dict[str, float]:
        """Get all parameter values as dict."""
        return {p.name: p.value for p in self._parameters}
    
    def get_free_values(self) -> np.ndarray:
        """Get free parameter values as array (for MCMC)."""
        return np.array([p.value for p in self.free_parameters])
    
    def set_free_values(self, values: np.ndarray):
        """Set free parameter values from array."""
        for param, value in zip(self.free_parameters, values):
            param.value = np.clip(value, param.bounds[0], param.bounds[1])
    
    def samples_to_params(self, samples: np.ndarray) -> Dict[str, float]:
        """Convert MCMC sample array to parameter dict."""
        free_params = self.free_parameters
        return {p.name: samples[i] for i, p in enumerate(free_params)}
    
    def params_to_samples(self) -> np.ndarray:
        """Convert current parameters to sample array."""
        return self.get_free_values()
    
    def log_prior(self) -> float:
        """Compute total log prior for current parameter values."""
        return sum(p.log_prior(p.value) for p in self.free_parameters)
    
    def sample_prior(self, n: int = 1) -> np.ndarray:
        """Draw n samples from joint prior."""
        samples = np.zeros((n, self.n_free))
        for i, param in enumerate(self.free_parameters):
            samples[:, i] = param.sample_prior(n)
        return samples
    
    @abstractmethod
    def compute_intensity(self, 
                          h: float, k: float, l: float, E: float,
                          **kwargs) -> float:
        """
        Compute predicted intensity at measurement point.
        
        For S(Q,ω) models, this is the scattering function.
        For order parameter models, h,k,l may be ignored and E=temperature.
        
        Parameters
        ----------
        h, k, l : float
            Reciprocal lattice coordinates
        E : float
            Energy transfer (meV) or temperature (K) depending on model
        
        Returns
        -------
        float
            Predicted intensity (arbitrary units)
        """
        pass
    
    def compute_intensity_array(self,
                                h: np.ndarray, k: np.ndarray, 
                                l: np.ndarray, E: np.ndarray,
                                **kwargs) -> np.ndarray:
        """
        Compute intensity for arrays of points.
        
        Default implementation loops; subclasses can vectorize.
        """
        result = np.zeros(len(h))
        for i in range(len(h)):
            result[i] = self.compute_intensity(h[i], k[i], l[i], E[i], **kwargs)
        return result
    
    def compute_jacobian(self,
                         h: float, k: float, l: float, E: float,
                         param_name: str,
                         delta: float = 0.01) -> float:
        """
        Compute sensitivity of intensity to parameter.
        
        Uses central difference by default.
        """
        param = self.get_parameter(param_name)
        original = param.value
        
        # Compute +/- delta
        param.value = original * (1 + delta)
        I_plus = self.compute_intensity(h, k, l, E)
        
        param.value = original * (1 - delta)
        I_minus = self.compute_intensity(h, k, l, E)
        
        param.value = original
        
        return (I_plus - I_minus) / (2 * delta * original)
    
    def log_likelihood(self,
                       h: np.ndarray, k: np.ndarray, l: np.ndarray, E: np.ndarray,
                       I_obs: np.ndarray, sigma: np.ndarray) -> float:
        """
        Compute log likelihood of observed data.
        
        Assumes Gaussian noise model.
        """
        I_pred = self.compute_intensity_array(h, k, l, E)
        
        chi2 = np.sum(((I_obs - I_pred) / sigma) ** 2)
        log_like = -0.5 * chi2 - np.sum(np.log(sigma * np.sqrt(2 * np.pi)))
        
        return log_like
    
    def log_posterior(self,
                      h: np.ndarray, k: np.ndarray, l: np.ndarray, E: np.ndarray,
                      I_obs: np.ndarray, sigma: np.ndarray) -> float:
        """Compute log posterior (log prior + log likelihood)."""
        lp = self.log_prior()
        if not np.isfinite(lp):
            return -np.inf
        return lp + self.log_likelihood(h, k, l, E, I_obs, sigma)
    
    @abstractmethod
    def description(self) -> str:
        """Return human-readable model description."""
        pass
    
    def __repr__(self):
        params_str = ", ".join(f"{p.name}={p.value:.4g}" for p in self._parameters)
        return f"{self.__class__.__name__}({params_str})"


class CompositeModel(PhysicsModel):
    """
    Combines multiple physics models.
    
    Useful for background + signal, multiple phases, etc.
    """
    
    def __init__(self, 
                 models: List[PhysicsModel],
                 combination: str = 'sum'):
        """
        Initialize composite model.
        
        Parameters
        ----------
        models : list of PhysicsModel
            Component models
        combination : str
            'sum' for additive, 'product' for multiplicative
        """
        super().__init__()
        self.models = models
        self.combination = combination
        
        # Collect all parameters with prefixed names
        for i, model in enumerate(models):
            prefix = f"m{i}_"
            for param in model.parameters:
                new_param = Parameter(
                    name=prefix + param.name,
                    value=param.value,
                    bounds=param.bounds,
                    fixed=param.fixed,
                    prior_type=param.prior_type,
                    prior_params=param.prior_params,
                    units=param.units,
                    description=f"[Model {i}] {param.description}"
                )
                self._parameters.append(new_param)
                self._param_index[new_param.name] = len(self._parameters) - 1
    
    def set_parameters(self, values: Dict[str, float]):
        """Set parameters, distributing to component models."""
        super().set_parameters(values)
        
        # Also update component models
        for i, model in enumerate(self.models):
            prefix = f"m{i}_"
            for param in model.parameters:
                full_name = prefix + param.name
                if full_name in values:
                    model.set_parameter(param.name, values[full_name])
    
    def compute_intensity(self, h: float, k: float, l: float, E: float, **kwargs) -> float:
        """Combine component model intensities."""
        if self.combination == 'sum':
            return sum(m.compute_intensity(h, k, l, E, **kwargs) for m in self.models)
        elif self.combination == 'product':
            result = 1.0
            for m in self.models:
                result *= m.compute_intensity(h, k, l, E, **kwargs)
            return result
        else:
            raise ValueError(f"Unknown combination: {self.combination}")
    
    def description(self) -> str:
        descs = [m.description() for m in self.models]
        op = " + " if self.combination == 'sum' else " × "
        return op.join(descs)
