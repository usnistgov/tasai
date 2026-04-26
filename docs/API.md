# TAS-AI API Reference

## Core Modules

### tasai.sunny

Spin wave calculation models.

#### SquareLatticeFM

```python
class SquareLatticeFM:
    """
    Square lattice ferromagnet with J1-J2 exchange.
    
    Parameters
    ----------
    J1 : float
        Nearest-neighbor exchange coupling (meV). Positive = FM.
    J2 : float
        Next-nearest-neighbor exchange coupling (meV). Positive = FM.
    D : float
        Single-ion anisotropy (meV). Positive = easy-axis.
    S : float, default=1.0
        Spin magnitude.
    
    Examples
    --------
    >>> model = SquareLatticeFM(J1=5.0, J2=0.5, D=0.1)
    >>> model.dispersion(0.25, 0.25, 0)
    24.5
    """
    
    def dispersion(self, H, K, L):
        """Calculate spin wave energy at (H,K,L)."""
        
    def intensity(self, H, K, L, E, eta=0.5):
        """Calculate S(Q,E) intensity with Lorentzian broadening."""
        
    def simulate_measurement(self, H, K, L, E, count_time=60, count_rate=100):
        """Simulate measurement with Poisson statistics."""
        
    def chi_squared(self, params=None):
        """Calculate chi-squared for current observations."""
        
    def log_likelihood(self, params=None):
        """Calculate log-likelihood for Bayesian inference."""
        
    def add_observation(self, H, K, L, E, I, sigma):
        """Add experimental data point."""
        
    def fit(self, method='grid'):
        """Fit parameters to observations."""
        
    def set_bounds(self, bounds_dict):
        """Set parameter bounds for fitting."""
```

#### NNOnlyModel

```python
class NNOnlyModel(SquareLatticeFM):
    """
    Nearest-neighbor only model (J2 fixed to 0).
    
    Use for model discrimination against J1-J2 model.
    """
```

### tasai.core

Core acquisition and inference components.

#### LogGaussianProcess

```python
class LogGaussianProcess:
    """
    Log-space Gaussian Process for intensity modeling.
    
    Addresses the JCNS criticism of standard GP: works in log-space
    to ensure non-negative intensity predictions.
    
    Parameters
    ----------
    length_scales : array-like, optional
        Initial length scales for each dimension.
    background : float, default=0.01
        Background level to prevent log(0).
    noise_level : float, default=0.1
        Initial noise estimate.
    n_dims : int, default=4
        Number of input dimensions.
    """
    
    def add_observation(self, x, intensity, sigma):
        """Add observation and update GP."""
        
    def predict(self, x):
        """Predict intensity at point x. Returns (mean, std)."""
        
    def acquisition_ucb(self, x, kappa=2.0):
        """Upper Confidence Bound acquisition function."""
        
    def acquisition_variance(self, x):
        """Pure variance-based acquisition (exploration)."""
        
    def suggest_next_points(self, bounds, n_points=1, acquisition='ucb'):
        """Suggest next measurement points."""
```

#### AgnosticExplorer

```python
class AgnosticExplorer:
    """
    Model-agnostic exploration (gpCAM-style).
    
    Use for initial survey when physics model is unknown.
    
    Parameters
    ----------
    bounds : array of shape (n_dims, 2)
        Min and max for each dimension.
    background : float, default=0.01
        Background estimate.
    use_log_gp : bool, default=True
        Use log-intensity GP regression (following the Teixeira-Parente / JCNS
        neutron active-learning approach).
    """
    
    def suggest_initial(self):
        """Suggest initial space-filling point."""
        
    def suggest_next(self, acquisition='ucb'):
        """Suggest next measurement point."""
        
    def add_observation(self, x, intensity, sigma):
        """Add observation."""
        
    def get_intensity_map(self, n_grid=50, dims=(0, 3)):
        """Get 2D intensity map."""
        
    def identify_signal_regions(self, threshold=None):
        """Find regions with significant signal."""
```

#### HybridExplorer

```python
class HybridExplorer:
    """
    Hybrid: agnostic survey → physics-informed refinement.
    
    Combines gpCAM efficiency with TAS-AI precision.
    """
    
    def suggest_next(self):
        """Suggest based on current mode (agnostic or informed)."""
```

### tasai.extensions

Reusable planning and hypothesis-generation helpers.

#### GoodenoughKanamoriAnalyzer

```python
class GoodenoughKanamoriAnalyzer:
    """
    Structure-aware exchange-path analyzer.

    Enumerates periodic exchange pathways between magnetic sites, labels
    bridge geometry, applies orbital-aware Goodenough-Kanamori sign rules,
    and returns ranked path hypotheses that can seed higher-level candidate
    Hamiltonian generation.
    """

    def find_exchange_paths(self, max_distance=8.0, max_bridging=2, bond_cutoff=2.8):
        """Enumerate magnetic exchange paths with periodic images."""

    def rank_paths(self, paths):
        """Rank paths by predicted strength and confidence."""

    def cluster_paths(self, paths, distance_tol=0.05, angle_tol=1.0):
        """Group symmetry-equivalent paths into candidate exchange families."""

    def generate_hamiltonians(self, paths, max_terms=3):
        """Build simple J1/J2/D candidate summaries from ranked path clusters."""
```

#### GNNHypothesisGenerator

```python
class GNNHypothesisGenerator:
    """
    High-level hypothesis generator for candidate spin Hamiltonians.

    In heuristic mode, this now delegates exchange-path analysis to
    GoodenoughKanamoriAnalyzer rather than using the older midpoint/angle-only
    approximation.
    """
```

### tasai.instrument

Instrument control and motor optimization.

#### SimplifiedMotorModel

```python
class SimplifiedMotorModel:
    """
    Simplified motor motion model for TAS.
    
    Parameters
    ----------
    H_speed : float, default=0.01
        H motion speed (r.l.u./sec).
    K_speed : float, default=0.01
        K motion speed (r.l.u./sec).
    L_speed : float, default=0.02
        L motion speed (r.l.u./sec).
    E_speed : float, default=0.5
        Energy motion speed (meV/sec).
    overhead : float, default=3.0
        Fixed overhead per move (sec).
    """
    
    def move_time(self, current, target):
        """Calculate time to move from current to target position."""
        
    def optimize_sequence(self, positions):
        """Optimize measurement sequence for minimum move time."""
```

#### TASMotorSystem

```python
class TASMotorSystem:
    """
    Full TAS motor geometry with scattering triangle.
    
    Converts (H,K,L,E) to motor angles (A3, A4, analyzer).
    """
    
    def to_angles(self, H, K, L, E):
        """Convert (H,K,L,E) to motor angles."""
        
    def move_time(self, current, target):
        """Calculate move time including all motors."""
```

#### MotionAwareAcquisition

```python
class MotionAwareAcquisition:
    """
    Acquisition function that accounts for motor motion.
    
    score = info_gain^η / (count_time + move_time)
    
    Parameters
    ----------
    motor_model : MotorModel
        Motor model for move time calculation.
    eta : float, default=0.7
        Information gain exponent.
    count_time : float, default=60.0
        Counting time per measurement.
    """
    
    def score(self, H, K, L, E, info_gain):
        """Calculate acquisition score for a candidate point."""
        
    def select_best(self, candidates, info_gains, n_select=1):
        """Select best candidates considering motion."""
```

### tasai.inference

Bayesian inference components.

#### MCMCSampler

```python
class MCMCSampler:
    """
    MCMC sampling with BUMPS/emcee backend.
    
    Parameters
    ----------
    model : PhysicsModel
        Model with log_likelihood method.
    n_walkers : int, default=32
        Number of MCMC walkers.
    n_steps : int, default=1000
        Number of MCMC steps.
    parallel : bool, default=True
        Use parallel evaluation.
    """
    
    def run(self, initial_params=None):
        """Run MCMC sampling."""
        
    def get_samples(self, burn=100, thin=1):
        """Get samples after burn-in."""
        
    def get_statistics(self):
        """Get parameter means and standard deviations."""
```

### tasai.physics

Physics models for order parameters.

#### OrderParameterModel

```python
class OrderParameterModel:
    """
    Temperature-dependent order parameter for phase transitions.
    
    I(T) = A × (1 - T/Tc)^(2β)  for T < Tc
    
    Parameters
    ----------
    Tc : float
        Transition temperature (K).
    beta : float, default=0.33
        Critical exponent.
    A : float, default=1.0
        Amplitude.
    """
    
    def intensity(self, T):
        """Calculate intensity at temperature T."""
        
    def fit(self, T_data, I_data, sigma_data):
        """Fit Tc and beta to data."""
```

---

## Example Workflows

### Parameter Determination

```python
from tasai.sunny import SquareLatticeFM
from tasai.instrument import SimplifiedMotorModel, MotionAwareAcquisition

# Setup
model = SquareLatticeFM(J1=5.0, J2=0.5, D=0.1)
motor = SimplifiedMotorModel()
acquisition = MotionAwareAcquisition(motor, eta=0.7)

# Autonomous loop
current_pos = (0, 0, 0, 0)
for i in range(50):
    # Generate candidates
    candidates = generate_candidates(model)
    info_gains = [model.expected_info_gain(c) for c in candidates]
    
    # Select best considering motion
    best_idx = acquisition.select_best(candidates, info_gains)
    target = candidates[best_idx]
    
    # Measure
    I, sigma = instrument.measure(*target)
    model.add_observation(*target, I, sigma)
    
    # Update
    model.fit()
    current_pos = target

# Results
print(model.get_parameters())
```

### Model Discrimination

```python
from tasai.sunny import SquareLatticeFM, NNOnlyModel

# Competing models
models = [NNOnlyModel(), SquareLatticeFM()]

for i in range(20):
    # Find maximum disagreement
    best_point, max_diff = find_discrimination_point(models)
    
    # Measure
    I, sigma = instrument.measure(*best_point)
    
    # Update both models
    for m in models:
        m.add_observation(*best_point, I, sigma)
        m.fit()
    
    # Calculate weights
    evidences = [m.evidence() for m in models]
    weights = np.exp(evidences) / np.sum(np.exp(evidences))
    
    print(f"Iteration {i}: weights = {weights}")
```

### Agnostic Survey + Physics Refinement

```python
from tasai.core import HybridExplorer
from tasai.sunny import SquareLatticeFM

# Start agnostic
bounds = np.array([[0, 0.5], [0, 0.5], [0, 1], [0, 30]])
explorer = HybridExplorer(bounds)

# Phase 1: Agnostic survey
for i in range(30):
    x = explorer.suggest_next()
    I, sigma = instrument.measure(*x)
    explorer.add_observation(x, I, sigma)

# Phase 2: Switch to physics model
model = SquareLatticeFM()
# ... transfer observations and continue with physics-informed acquisition
```

---

## Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `TASAI_LOG_LEVEL` | INFO | Logging level |
| `TASAI_PARALLEL` | 1 | Enable parallel MCMC |
| `TASAI_N_CORES` | auto | Number of CPU cores |
| `JULIA_NUM_THREADS` | auto | Julia threads for Sunny |

### Configuration File

```yaml
# tasai.yaml
model:
  type: j1j2
  parameters:
    J1: 5.0
    J2: 0.5
    D: 0.1
  bounds:
    J1: [0, 20]
    J2: [0, 5]
    D: [0, 1]

acquisition:
  type: motion_aware
  eta: 0.7
  count_time: 60
  
inference:
  method: mcmc
  n_walkers: 32
  n_steps: 1000
  
instrument:
  type: simulated
  # or: host: bt7.ncnr.nist.gov, port: 5000
```

---

*API version 1.0*
