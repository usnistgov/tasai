"""
Spin Wave Models for TAS-AI

Provides spin wave calculations using either:
- PySpinW: Pure Python implementation of SpinW
- Sunny.jl: Julia-based spin wave calculator (via juliacall)

Both backends implement linear spin wave theory (LSWT) and can compute
S(Q,ω) dynamical structure factors for magnetic materials.
"""

import numpy as np
from typing import Optional, List, Dict, Union, Callable, Tuple
from dataclasses import dataclass
import logging

from .base import PhysicsModel, Parameter

logger = logging.getLogger(__name__)


# Try to import backends
_PYSPINW_AVAILABLE = False
_SUNNY_AVAILABLE = False

try:
    from pyspinw import SpinW, sw_model, sw_egrid
    _PYSPINW_AVAILABLE = True
except ImportError:
    pass

try:
    import juliacall
    _SUNNY_AVAILABLE = True
except ImportError:
    pass


def get_available_backends() -> List[str]:
    """Return list of available spin wave backends."""
    backends = []
    if _PYSPINW_AVAILABLE:
        backends.append('pyspinw')
    if _SUNNY_AVAILABLE:
        backends.append('sunny')
    return backends


@dataclass
class SpinWaveConfig:
    """Configuration for spin wave model."""
    # Lattice parameters
    lat_const: Tuple[float, float, float] = (3.0, 3.0, 3.0)
    angles: Tuple[float, float, float] = (90.0, 90.0, 90.0)
    spacegroup: str = 'P 1'

    # Magnetic atoms: list of (position, spin, label)
    atoms: List[Tuple[List[float], float, str]] = None

    # Magnetic structure
    propagation_k: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    spin_directions: np.ndarray = None
    magstr_mode: str = 'direct'

    # Broadening and temperature
    energy_resolution: float = 0.1  # meV (Gaussian FWHM)
    temperature: float = 0.0  # K (for Bose factor)

    # Backend selection
    backend: str = 'auto'  # 'pyspinw', 'sunny', or 'auto'


class SpinWaveModel(PhysicsModel):
    """
    Spin wave model for TAS-AI simulations.

    Computes S(Q,ω) using linear spin wave theory via PySpinW or Sunny.jl.
    Parameters (exchange constants, anisotropies) can be fitted during MCMC.

    Examples
    --------
    >>> # Create FM chain model
    >>> model = SpinWaveModel.from_preset('chain', J=-1.0)
    >>> sqw = model.compute_intensity(0.5, 0, 0, 2.0)

    >>> # Create custom model
    >>> config = SpinWaveConfig(
    ...     lat_const=(3, 3, 6),
    ...     angles=(90, 90, 90),
    ...     atoms=[([0, 0, 0], 1.0, 'Cu')]
    ... )
    >>> model = SpinWaveModel(config, exchanges={'J1': 1.0})
    """

    def __init__(self,
                 config: SpinWaveConfig = None,
                 exchanges: Dict[str, float] = None,
                 anisotropies: Dict[str, float] = None,
                 bonds: Dict[str, int] = None,
                 exchange_bounds: Dict[str, Tuple[float, float]] = None,
                 backend: str = 'auto'):
        """
        Initialize spin wave model.

        Parameters
        ----------
        config : SpinWaveConfig
            Crystal and magnetic structure configuration
        exchanges : dict
            Exchange constants {label: value_in_meV}
        anisotropies : dict
            Single-ion anisotropies {label: value_in_meV}
        bonds : dict
            Bond assignments {matrix_label: bond_index}
        exchange_bounds : dict
            Parameter bounds {label: (min, max)}
        backend : str
            'pyspinw', 'sunny', or 'auto' (default: auto-select)
        """
        super().__init__()

        self.config = config or SpinWaveConfig()
        self.exchanges = exchanges or {}
        self.anisotropies = anisotropies or {}
        self.bonds = bonds or {}
        self.exchange_bounds = exchange_bounds or {}

        # Select backend
        self.backend_name = self._select_backend(backend)

        # Initialize backend-specific objects
        self._sw = None  # SpinW object (pyspinw)
        self._sunny_model = None  # Sunny model (juliacall)

        # Create parameters for exchange constants
        for name, value in self.exchanges.items():
            bounds = self.exchange_bounds.get(name, (-50.0, 50.0))
            param = Parameter(
                name=name,
                value=value,
                bounds=bounds,
                units='meV',
                description=f'Exchange constant {name}'
            )
            self._parameters.append(param)
            self._param_index[name] = len(self._parameters) - 1

        # Add anisotropy parameters
        for name, value in self.anisotropies.items():
            bounds = self.exchange_bounds.get(name, (-10.0, 10.0))
            param = Parameter(
                name=name,
                value=value,
                bounds=bounds,
                units='meV',
                description=f'Anisotropy {name}'
            )
            self._parameters.append(param)
            self._param_index[name] = len(self._parameters) - 1

        # Cached spectrum
        self._cached_spec = None
        self._cached_params = None

        # Build the model
        self._build_model()

    def _select_backend(self, requested: str) -> str:
        """Select available backend."""
        available = get_available_backends()

        if not available:
            raise ImportError(
                "No spin wave backend available. Install pyspinw:\n"
                "  pip install pyspinw\n"
                "Or for Sunny.jl support:\n"
                "  pip install juliacall"
            )

        if requested == 'auto':
            # Prefer pyspinw (pure Python, no Julia dependency)
            return 'pyspinw' if 'pyspinw' in available else available[0]

        if requested not in available:
            raise ValueError(
                f"Backend '{requested}' not available. "
                f"Available: {available}"
            )

        return requested

    def _build_model(self):
        """Build the spin wave model using selected backend."""
        if self.backend_name == 'pyspinw':
            self._build_pyspinw()
        elif self.backend_name == 'sunny':
            self._build_sunny()

    def _build_pyspinw(self):
        """Build model using PySpinW."""
        from pyspinw import SpinW

        sw = SpinW()

        # Set lattice
        sw.genlattice(
            lat_const=list(self.config.lat_const),
            angled=list(self.config.angles),
            spgr=self.config.spacegroup
        )

        # Add atoms
        if self.config.atoms:
            for pos, spin, label in self.config.atoms:
                sw.addatom(r=pos, S=spin, label=label)

        # Generate couplings
        sw.gencoupling(max_distance=10.0)

        # Add exchange matrices
        for name, value in self.exchanges.items():
            sw.addmatrix(label=name, value=value)

        # Add single-ion anisotropy matrices (easy-axis along z by default)
        for name, value in self.anisotropies.items():
            sw.addmatrix(label=name, value=np.diag([0.0, 0.0, float(value)]))
            sw.addaniso(mat=name)

        # Assign to bonds
        for name, bond_idx in self.bonds.items():
            sw.addcoupling(mat=name, bond=bond_idx)

        # Generate magnetic structure
        if self.config.spin_directions is not None:
            sw.genmagstr(
                mode=self.config.magstr_mode,
                k=list(self.config.propagation_k),
                S=self.config.spin_directions
            )
        else:
            # Default to z-polarized
            n_mag = sw.n_mag_atoms
            if n_mag > 0:
                S = np.zeros((n_mag, 3))
                S[:, 2] = 1.0
                sw.genmagstr(mode='direct', k=list(self.config.propagation_k), S=S)

        self._sw = sw

    def _build_sunny(self):
        """Build model using Sunny.jl (placeholder)."""
        raise NotImplementedError(
            "Sunny.jl backend not yet implemented. Use pyspinw."
        )

    def _update_exchanges(self):
        """Update exchange values in the model from current parameters."""
        if self._sw is None:
            return

        # Update exchange matrices
        for param in self._parameters:
            if param.name in self._sw._coupling.exchange_matrices:
                if param.name in self.anisotropies:
                    self._sw._coupling.exchange_matrices[param.name].matrix = np.diag(
                        [0.0, 0.0, param.value]
                    )
                else:
                    self._sw._coupling.exchange_matrices[param.name].matrix = (
                        param.value * np.eye(3)
                    )

        # Invalidate cache
        self._cached_spec = None
        self._cached_params = None

    def compute_sqw(self,
                    h_range: Tuple[float, float] = (-1, 1),
                    k_range: Tuple[float, float] = (-1, 1),
                    l_range: Tuple[float, float] = (-1, 1),
                    E_range: Tuple[float, float] = (0, 20),
                    n_h: int = 21, n_k: int = 21, n_l: int = 3, n_E: int = 100
                    ) -> Tuple[np.ndarray, ...]:
        """
        Compute S(Q,ω) on a 4D grid.

        Returns
        -------
        h_grid, k_grid, l_grid, E_grid, sqw_data : arrays
            Grids and 4D S(Q,ω) data
        """
        if self._sw is None:
            raise RuntimeError("Model not built. Call _build_model() first.")

        from pyspinw import sw_egrid

        # Create grids
        h_grid = np.linspace(*h_range, n_h)
        k_grid = np.linspace(*k_range, n_k)
        l_grid = np.linspace(*l_range, n_l)
        E_grid = np.linspace(*E_range, n_E)

        # Compute S(Q,ω) for each Q point
        sqw_data = np.zeros((n_h, n_k, n_l, n_E))

        for ih, h in enumerate(h_grid):
            for ik, k in enumerate(k_grid):
                for il, l in enumerate(l_grid):
                    # Compute dispersion at this Q
                    try:
                        # pyspinw expects explicit points as a 2D ndarray, not a
                        # single-point list path.
                        q_points = np.array([[h, k, l]], dtype=float)
                        spec = self._sw.spinwave(q_points, n_pts=1)
                        spec = sw_egrid(spec, Evect=E_grid,
                                       dE=self.config.energy_resolution,
                                       T=self.config.temperature)
                        sqw_data[ih, ik, il, :] = spec['swConv'][:, 0]
                    except Exception:
                        # Point may be inaccessible
                        sqw_data[ih, ik, il, :] = 0.0

        return h_grid, k_grid, l_grid, E_grid, sqw_data

    def compute_intensity(self,
                          h: float, k: float, l: float, E: float,
                          **kwargs) -> float:
        """
        Compute S(Q,ω) at a single point.

        Parameters
        ----------
        h, k, l : float
            Reciprocal lattice coordinates
        E : float
            Energy transfer in meV

        Returns
        -------
        float
            Scattering intensity (arbitrary units)
        """
        if self._sw is None:
            raise RuntimeError("Model not built")

        # Check if parameters changed
        current_params = tuple(p.value for p in self._parameters)
        if current_params != self._cached_params:
            self._update_exchanges()
            self._cached_params = current_params

        # Compute at this point
        try:
            from pyspinw import sw_egrid

            # pyspinw expects explicit points as a 2D ndarray, not a
            # single-point list path.
            q_points = np.array([[h, k, l]], dtype=float)
            spec = self._sw.spinwave(q_points, n_pts=1)
            spec = sw_egrid(spec,
                           Evect=np.array([E - 0.5, E, E + 0.5]),
                           dE=self.config.energy_resolution,
                           T=self.config.temperature)

            # Return intensity at central energy
            return float(spec['swConv'][1, 0])

        except Exception as e:
            logger.debug(f"SpinWave calculation failed at ({h},{k},{l},{E}): {e}")
            return 0.0

    def intensity(self, H: float, L: float, E: float) -> float:
        """Convenience wrapper for 1D cuts used in benchmarks."""
        return self.compute_intensity(H, 0.0, L, E)

    def dispersion(self, H: float, L: float = 0.0) -> float:
        """Return a representative mode energy at a single Q point."""
        spec = self.get_dispersion([[H, 0.0, L]], n_pts=1)
        omega = np.real(spec.get('omega', np.array([]))).ravel()
        omega = omega[np.isfinite(omega)]
        if omega.size == 0:
            return 0.0
        positive = omega[omega > 0]
        return float(positive.min() if positive.size else omega.min())

    def get_sqw_function(self) -> Callable[[float, float, float, float], float]:
        """
        Return a callable S(Q,ω) function for use with TASSimulator.

        Returns
        -------
        callable
            Function sqw(h, k, l, E) -> intensity
        """
        def sqw_func(h: float, k: float, l: float, E: float) -> float:
            return self.compute_intensity(h, k, l, E)

        return sqw_func

    def get_dispersion(self,
                       Q_path: List[List[float]],
                       n_pts: int = 100) -> Dict:
        """
        Calculate spin wave dispersion along a Q-path.

        Parameters
        ----------
        Q_path : list
            List of Q-points defining the path
        n_pts : int
            Number of points per segment

        Returns
        -------
        dict
            Spectrum dictionary with 'omega', 'hkl', etc.
        """
        if self._sw is None:
            raise RuntimeError("Model not built")

        # Update parameters if needed
        current_params = tuple(p.value for p in self._parameters)
        if current_params != self._cached_params:
            self._update_exchanges()
            self._cached_params = current_params

        return self._sw.spinwave(Q_path, n_pts=n_pts)

    def description(self) -> str:
        """Return model description."""
        params = ", ".join(f"{p.name}={p.value:.3f}" for p in self._parameters)
        return f"SpinWaveModel(backend={self.backend_name}, {params})"

    @classmethod
    def from_preset(cls,
                    name: str,
                    J: float = 1.0,
                    backend: str = 'auto',
                    **kwargs) -> 'SpinWaveModel':
        """
        Create spin wave model from a preset.

        Parameters
        ----------
        name : str
            Preset name: 'chain', 'square', 'triangular', 'kagome'
        J : float
            Exchange constant in meV
        backend : str
            Backend to use
        **kwargs
            Additional parameters

        Returns
        -------
        SpinWaveModel
            Configured model
        """
        presets = {
            'chain': {
                'lat_const': (3, 8, 8),
                'angles': (90, 90, 90),
                'atoms': [([0, 0, 0], 1.0, 'M1')],
                'propagation_k': (0, 0, 0),
                'exchanges': {'J1': J},
                'bonds': {'J1': 1},
            },
            'square': {
                'lat_const': (3, 3, 6),
                'angles': (90, 90, 90),
                'atoms': [([0, 0, 0], 1.0, 'Cu')],
                'propagation_k': (0.5, 0.5, 0),
                'exchanges': {'J1': J},
                'bonds': {'J1': 1},
            },
            'triangular': {
                'lat_const': (3, 3, 6),
                'angles': (90, 90, 120),
                'atoms': [([0, 0, 0], 1.0, 'Cu')],
                'propagation_k': (1/3, 1/3, 0),
                'magstr_mode': 'helical',
                'exchanges': {'J1': J},
                'bonds': {'J1': 1},
            },
        }

        if name not in presets:
            raise ValueError(f"Unknown preset: {name}. Available: {list(presets.keys())}")

        preset = presets[name]

        config = SpinWaveConfig(
            lat_const=preset['lat_const'],
            angles=preset['angles'],
            atoms=preset['atoms'],
            propagation_k=preset['propagation_k'],
            magstr_mode=preset.get('magstr_mode', 'direct'),
        )

        return cls(
            config=config,
            exchanges=preset['exchanges'],
            bonds=preset['bonds'],
            backend=backend,
            **kwargs
        )


class TabularSpinWaveModel(PhysicsModel):
    """
    Spin wave model using pre-computed tabular data.

    Useful when spin wave calculations are expensive and can be
    pre-computed on a grid (e.g., from Sunny.jl).
    """

    def __init__(self,
                 h_grid: np.ndarray,
                 k_grid: np.ndarray,
                 l_grid: np.ndarray,
                 E_grid: np.ndarray,
                 sqw_data: np.ndarray,
                 interpolation: str = 'linear'):
        """
        Initialize from pre-computed data.

        Parameters
        ----------
        h_grid, k_grid, l_grid, E_grid : np.ndarray
            1D arrays defining the grid
        sqw_data : np.ndarray
            4D array of S(Q,ω) values
        interpolation : str
            Interpolation method ('linear' or 'nearest')
        """
        super().__init__()

        from scipy.interpolate import RegularGridInterpolator

        self.h_grid = h_grid
        self.k_grid = k_grid
        self.l_grid = l_grid
        self.E_grid = E_grid
        self.sqw_data = sqw_data

        self._interpolator = RegularGridInterpolator(
            (h_grid, k_grid, l_grid, E_grid),
            sqw_data,
            method=interpolation,
            bounds_error=False,
            fill_value=0.0
        )

    def compute_intensity(self,
                          h: float, k: float, l: float, E: float,
                          **kwargs) -> float:
        """Interpolate S(Q,ω) at given point."""
        return float(self._interpolator([[h, k, l, E]])[0])

    def get_sqw_function(self) -> Callable[[float, float, float, float], float]:
        """Return callable for TASSimulator."""
        return lambda h, k, l, E: self.compute_intensity(h, k, l, E)

    def description(self) -> str:
        """Return model description."""
        shape = self.sqw_data.shape
        return f"TabularSpinWaveModel(shape={shape})"

    @classmethod
    def from_pyspinw(cls,
                     model: 'SpinWaveModel',
                     h_range: Tuple[float, float] = (-1, 1),
                     k_range: Tuple[float, float] = (-1, 1),
                     l_range: Tuple[float, float] = (-1, 1),
                     E_range: Tuple[float, float] = (0, 20),
                     n_h: int = 21, n_k: int = 21, n_l: int = 3, n_E: int = 100
                     ) -> 'TabularSpinWaveModel':
        """
        Create tabular model from a SpinWaveModel.

        Pre-computes S(Q,ω) on a grid for fast interpolation.
        """
        grids = model.compute_sqw(h_range, k_range, l_range, E_range,
                                   n_h, n_k, n_l, n_E)
        return cls(*grids)


def compare_backends(model_config: SpinWaveConfig,
                     exchanges: Dict[str, float],
                     bonds: Dict[str, int],
                     Q_path: List[List[float]],
                     n_pts: int = 100) -> Dict:
    """
    Compare spin wave calculations between available backends.

    Parameters
    ----------
    model_config : SpinWaveConfig
        Model configuration
    exchanges : dict
        Exchange constants
    bonds : dict
        Bond assignments
    Q_path : list
        Q-points for comparison
    n_pts : int
        Points per segment

    Returns
    -------
    dict
        Comparison results with dispersions from each backend
    """
    results = {}
    available = get_available_backends()

    for backend in available:
        try:
            model = SpinWaveModel(
                config=model_config,
                exchanges=exchanges,
                bonds=bonds,
                backend=backend
            )
            spec = model.get_dispersion(Q_path, n_pts=n_pts)
            results[backend] = {
                'omega': np.real(spec['omega']),
                'hkl': spec['hkl'],
                'success': True
            }
        except Exception as e:
            results[backend] = {
                'error': str(e),
                'success': False
            }

    # Compute differences if both available
    if len([r for r in results.values() if r.get('success')]) >= 2:
        backends = [k for k, v in results.items() if v.get('success')]
        if len(backends) >= 2:
            omega1 = results[backends[0]]['omega']
            omega2 = results[backends[1]]['omega']

            results['comparison'] = {
                'backends': backends,
                'max_diff': np.max(np.abs(omega1 - omega2)),
                'rms_diff': np.sqrt(np.mean((omega1 - omega2)**2)),
                'match': np.allclose(omega1, omega2, rtol=1e-3, atol=1e-6)
            }

    return results
