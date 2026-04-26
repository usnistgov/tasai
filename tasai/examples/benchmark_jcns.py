#!/usr/bin/env python3
"""
Benchmark Comparison: TAS-AI vs JCNS Log-GP vs gpCAM

Implements benchmark scenarios from:
- Teixeira Parente et al., Front. Mater. 8, 772014 (2022)
- "Benchmarking autonomous scattering experiments illustrated on TAS"

Test functions simulate different S(Q,ω) scenarios:
1. Single magnon branch
2. Multiple branches
3. Weak signal in noise
4. Sharp vs broad features

Metrics:
- Time to convergence (relative weighted error < threshold)
- Number of measurements needed
- Signal region identification accuracy
- Final reconstruction error

Run:
    python benchmark_jcns.py
    python benchmark_jcns.py --method tasai --scenario all
"""

import numpy as np
import matplotlib.pyplot as plt
import json
import os
import sys
import multiprocessing as mp
import logging
from typing import Dict, List, Tuple, Callable
from dataclasses import dataclass
import time
import argparse
from pathlib import Path

from tasai.core.gaussian_process import LogGaussianProcess, AgnosticExplorer

try:
    from tasai.instrument import TASResolutionCalculator, create_default_tas_config
    HAS_RESOLUTION = True
except Exception:
    HAS_RESOLUTION = False

METHOD_ORDER = ['grid', 'random', 'log_gp', 'tasai']
METHOD_COLORS = {
    'grid': '#4c566a',
    'random': '#f6c343',
    'log_gp': '#2d7dd2',
    'tasai': '#d1495b',
}

logger = logging.getLogger(__name__)


# =============================================================================
# Benchmark Test Functions (from JCNS paper)
# =============================================================================

def intensity_single_branch(H: float, E: float, 
                           J: float = 5.0, D: float = 0.1,
                           gamma: float = 0.5, A: float = 1.0,
                           background: float = 0.01) -> float:
    """
    Single magnon branch: I(H,E) = A × L(E - ω(H)) + bg
    
    Dispersion: ω(H) = 2J(1 - cos(2πH)) + D
    Lineshape: Lorentzian with width γ
    """
    # Dispersion
    omega = 2 * J * (1 - np.cos(2 * np.pi * H)) + D
    
    # Lorentzian intensity
    I = A * gamma / ((E - omega)**2 + gamma**2) / np.pi
    
    return I + background


def omega_single_branch(H: np.ndarray, J: float = 5.0, D: float = 0.1) -> np.ndarray:
    """Dispersion used by the single-branch, weak-signal, and sharp-feature cases."""
    return 2 * J * (1 - np.cos(2 * np.pi * H)) + D


def intensity_two_branches(H: float, E: float,
                          J1: float = 5.0, J2: float = 3.0,
                          D: float = 0.1, gamma: float = 0.5,
                          A1: float = 1.0, A2: float = 0.5,
                          background: float = 0.01) -> float:
    """
    Two magnon branches (e.g., acoustic + optic).
    """
    omega1 = 2 * J1 * (1 - np.cos(2 * np.pi * H)) + D
    omega2 = 2 * J2 * (1 - np.cos(2 * np.pi * H)) + D + 5.0  # Offset
    
    I1 = A1 * gamma / ((E - omega1)**2 + gamma**2) / np.pi
    I2 = A2 * gamma / ((E - omega2)**2 + gamma**2) / np.pi
    
    return I1 + I2 + background


def omega_two_branch_acoustic(H: np.ndarray, J1: float = 5.0, D: float = 0.1) -> np.ndarray:
    return 2 * J1 * (1 - np.cos(2 * np.pi * H)) + D


def omega_two_branch_optic(H: np.ndarray, J2: float = 3.0, D: float = 0.1, offset: float = 5.0) -> np.ndarray:
    return 2 * J2 * (1 - np.cos(2 * np.pi * H)) + D + offset


def intensity_weak_signal(H: float, E: float,
                         J: float = 5.0, gamma: float = 0.5,
                         A: float = 0.1, background: float = 0.05) -> float:
    """
    Weak signal buried in noise (signal-to-noise ~2).
    """
    omega = 2 * J * (1 - np.cos(2 * np.pi * H))
    I = A * gamma / ((E - omega)**2 + gamma**2) / np.pi
    return I + background


def intensity_sharp_feature(H: float, E: float,
                           J: float = 5.0, gamma: float = 0.1,
                           A: float = 1.0, background: float = 0.01) -> float:
    """
    Sharp feature (small γ) - harder to find.
    """
    omega = 2 * J * (1 - np.cos(2 * np.pi * H))
    I = A * gamma / ((E - omega)**2 + gamma**2) / np.pi
    return I + background


def intensity_gap_mode(H: float, E: float,
                      J: float = 5.0, Delta: float = 2.0,
                      gamma: float = 0.5, A: float = 1.0,
                      background: float = 0.01) -> float:
    """
    Gapped dispersion: ω(H) = sqrt(Δ² + (2J sin(πH))²)
    """
    omega = np.sqrt(Delta**2 + (2 * J * np.sin(np.pi * H))**2)
    I = A * gamma / ((E - omega)**2 + gamma**2) / np.pi
    return I + background


def omega_gap_mode(H: np.ndarray, J: float = 5.0, Delta: float = 2.0) -> np.ndarray:
    return np.sqrt(Delta**2 + (2 * J * np.sin(np.pi * H))**2)


# Collection of benchmark scenarios
BENCHMARK_SCENARIOS = {
    'single_branch': {
        'function': intensity_single_branch,
        'description': 'Single magnon dispersion',
        'bounds': np.array([[0, 0.5], [0, 40]]),  # H, E
        'difficulty': 'easy',
        'dispersions': [
            ('Mode', lambda H: omega_single_branch(H, J=5.0, D=0.1))
        ]
    },
    'two_branches': {
        'function': intensity_two_branches,
        'description': 'Two magnon branches',
        'bounds': np.array([[0, 0.5], [0, 40]]),
        'difficulty': 'medium',
        'dispersions': [
            ('Acoustic', lambda H: omega_two_branch_acoustic(H, J1=5.0, D=0.1)),
            ('Optic', lambda H: omega_two_branch_optic(H, J2=3.0, D=0.1, offset=5.0))
        ]
    },
    'weak_signal': {
        'function': intensity_weak_signal,
        'description': 'Weak signal in noise',
        'bounds': np.array([[0, 0.5], [0, 40]]),
        'difficulty': 'hard',
        'dispersions': [
            ('Mode', lambda H: omega_single_branch(H, J=5.0, D=0.0))
        ]
    },
    'sharp_feature': {
        'function': intensity_sharp_feature,
        'description': 'Sharp dispersion feature',
        'bounds': np.array([[0, 0.5], [0, 40]]),
        'difficulty': 'hard',
        'dispersions': [
            ('Mode', lambda H: omega_single_branch(H, J=5.0, D=0.0))
        ]
    },
    'gap_mode': {
        'function': intensity_gap_mode,
        'description': 'Gapped magnon mode',
        'bounds': np.array([[0, 0.5], [0, 30]]),
        'difficulty': 'medium',
        'dispersions': [
            ('Gap mode', lambda H: omega_gap_mode(H, J=5.0, Delta=2.0))
        ]
    }
}


def _make_pyspinw_intensity(model):
    """Build a vectorized S(Q,E) callable from a PySpinW model."""
    def intensity(H, E):
        H_arr = np.asarray(H)
        E_arr = np.asarray(E)
        HH, EE = np.broadcast_arrays(H_arr, E_arr)
        out = np.empty(HH.shape, dtype=float)
        it = np.nditer(HH, flags=['multi_index'])
        for _ in it:
            idx = it.multi_index
            h_val = float(HH[idx])
            e_val = float(EE[idx])
            out[idx] = model.compute_intensity(h_val, h_val, 0.0, e_val)
        return out
    return intensity


def _load_official_pyspinw():
    """Load the official pySpinW package from the active env or a local checkout."""
    try:
        import pyspinw  # type: ignore
        if hasattr(pyspinw, "sw_egrid"):
            return pyspinw
    except Exception:
        pyspinw = None

    official_path = os.environ.get("TASAI_OFFICIAL_PYSPINW_PATH")
    if not official_path:
        return None
    official_path = os.path.expanduser(official_path)
    if not os.path.isdir(official_path):
        raise RuntimeError(f"TASAI_OFFICIAL_PYSPINW_PATH not found: {official_path}")

    sys.path.insert(0, official_path)
    try:
        import pyspinw  # type: ignore
    except Exception as exc:
        raise RuntimeError(f"Failed to import official pySpinW from {official_path}: {exc}") from exc
    if not hasattr(pyspinw, "sw_egrid"):
        raise RuntimeError(
            f"Imported pySpinW from {official_path}, but it does not expose sw_egrid."
        )
    return pyspinw


def _official_pyspinw_intensity(spinw_model,
                                sw_egrid_fn,
                                energy_fwhm: float = 0.5,
                                q_scale: float = 1.0):
    """Build an S(Q,E) callable using official pySpinW SpinW API."""
    def intensity(H, E):
        H_arr = np.asarray(H)
        E_arr = np.asarray(E)
        HH, EE = np.broadcast_arrays(H_arr, E_arr)
        out = np.zeros(HH.shape, dtype=float)

        for idx in np.ndindex(HH.shape):
            h_val = float(HH[idx])
            e_val = float(EE[idx])
            if not np.isfinite(h_val) or not np.isfinite(e_val):
                continue

            q_vec = np.array([[q_scale * h_val, 0.0, 0.0]], dtype=float)
            try:
                spec = spinw_model.spinwave(q_vec, n_pts=1)
                sigma = max(energy_fwhm / 2.355, 1e-6)
                evec = np.array([e_val - 3 * sigma, e_val, e_val + 3 * sigma], dtype=float)
                conv = sw_egrid_fn(spec, Evect=evec, dE=energy_fwhm)
            except Exception:
                out[idx] = 0.0
                continue
            swconv = np.asarray(conv.get("swConv", np.array([0.0])), dtype=float).reshape(-1)
            out[idx] = float(swconv[1] if swconv.size >= 2 else swconv[0])

        return out

    return intensity


def _validate_pyspinw_scenarios(scenarios: Dict[str, Dict]) -> None:
    """Fail fast if the generated pySpinW intensity surfaces are degenerate."""
    H = np.linspace(0.0, 0.5, 9)
    E = np.linspace(0.0, 40.0, 41)
    grids = {}

    for name, cfg in scenarios.items():
        fn = cfg["function"]
        vals = np.array([fn(float(h), float(e)) for h in H for e in E], dtype=float)
        if np.count_nonzero(vals > 1e-12) < 3:
            raise RuntimeError(
                f"PySpinW scenario '{name}' is effectively zero everywhere; aborting benchmark."
            )
        grids[name] = vals

    names = list(grids.keys())
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a = grids[names[i]]
            b = grids[names[j]]
            if np.allclose(a, b, rtol=1e-6, atol=1e-10):
                raise RuntimeError(
                    f"PySpinW scenarios '{names[i]}' and '{names[j]}' are numerically identical; aborting benchmark."
                )


def _build_official_pyspinw_ground_truth_scenarios():
    """Create benchmark scenarios using the official pySpinW backend."""
    pyspinw = _load_official_pyspinw()
    if pyspinw is None:
        raise RuntimeError(
            "Official pySpinW requested but no usable pyspinw installation was found."
        )
    sw_egrid_fn = pyspinw.sw_egrid
    SpinW = pyspinw.SpinW

    base_bounds = np.array([[0, 0.5], [0, 40]])

    # Single-branch gapless model
    sw_single = SpinW()
    sw_single.genlattice(lat_const=[3, 3, 6], angled=[90, 90, 90], spgr='P 1')
    sw_single.addatom(r=[0, 0, 0], S=1, label='M1')
    sw_single.gencoupling(max_distance=4)
    sw_single.addmatrix(label='J1', value=2.5)
    sw_single.addcoupling(mat='J1', bond=1)
    sw_single.genmagstr(mode='direct', k=[0.5, 0, 0], S=np.array([[0, 0, 1]]))

    # Gapped model with stronger anisotropy
    sw_gapped = SpinW()
    sw_gapped.genlattice(lat_const=[3, 3, 6], angled=[90, 90, 90], spgr='P 1')
    sw_gapped.addatom(r=[0, 0, 0], S=1, label='M1')
    sw_gapped.gencoupling(max_distance=4)
    sw_gapped.addmatrix(label='J1', value=3.5)
    sw_gapped.addcoupling(mat='J1', bond=1)
    sw_gapped.addmatrix(label='A1', value=[0, 0, 0.3])
    sw_gapped.addaniso(mat='A1')
    sw_gapped.genmagstr(mode='direct', k=[0.5, 0, 0], S=np.array([[0, 0, 1]]))

    scenarios = {
        'pyspinw_single': {
            'function': _official_pyspinw_intensity(sw_single, sw_egrid_fn, energy_fwhm=0.5),
            'description': 'PySpinW (official) ground truth (J=2.5, gapless)',
            'bounds': base_bounds,
            'difficulty': 'medium',
            'dispersions': [],
        },
        'pyspinw_gapped': {
            'function': _official_pyspinw_intensity(sw_gapped, sw_egrid_fn, energy_fwhm=0.5),
            'description': 'PySpinW (official) ground truth (J=3.5, anisotropy=0.3)',
            'bounds': base_bounds,
            'difficulty': 'medium',
            'dispersions': [],
        }
    }
    _validate_pyspinw_scenarios(scenarios)
    return scenarios


def _build_pyspinw_ground_truth_scenarios():
    """Create benchmark scenarios using PySpinW as ground truth."""
    try:
        return _build_official_pyspinw_ground_truth_scenarios()
    except Exception as exc:
        logger.warning("Falling back to internal pyspinw backend: %s", exc)

    try:
        from tasai.physics.spinwave import SpinWaveModel, SpinWaveConfig
    except Exception as exc:
        raise RuntimeError(f"PySpinW ground truth requested but unavailable: {exc}") from exc

    config = SpinWaveConfig(
        lat_const=(3.0, 3.0, 6.0),
        angles=(90, 90, 90),
        atoms=[([0, 0, 0], 1.0, 'M1')],
        propagation_k=(0.0, 0.0, 0.0)
    )

    base_bounds = np.array([[0, 0.5], [0, 40]])

    model_a = SpinWaveModel(
        config=config,
        exchanges={'J1': 5.0},
        anisotropies={'D': 0.1},
        bonds={'J1': 1},
        backend='pyspinw'
    )
    model_b = SpinWaveModel(
        config=config,
        exchanges={'J1': 3.5},
        anisotropies={'D': 0.3},
        bonds={'J1': 1},
        backend='pyspinw'
    )

    scenarios = {
        'pyspinw_single': {
            'function': _make_pyspinw_intensity(model_a),
            'description': 'PySpinW ground truth (J1=5.0, D=0.1)',
            'bounds': base_bounds,
            'difficulty': 'medium',
            'dispersions': [],
        },
        'pyspinw_gapped': {
            'function': _make_pyspinw_intensity(model_b),
            'description': 'PySpinW ground truth (J1=3.5, D=0.3)',
            'bounds': base_bounds,
            'difficulty': 'medium',
            'dispersions': [],
        }
    }
    _validate_pyspinw_scenarios(scenarios)
    return scenarios


# =============================================================================
# Measurement Simulator
# =============================================================================

@dataclass
class SimulatedMeasurement:
    H: float
    E: float
    I: float
    sigma: float
    true_I: float


def simulate_measurement(H: float, E: float, 
                        intensity_func: Callable,
                        count_time: float = 60.0,
                        count_rate: float = 100.0) -> SimulatedMeasurement:
    """Simulate a neutron measurement with Poisson statistics."""
    true_I = intensity_func(H, E)
    
    # Poisson counts
    counts = np.random.poisson(max(1, int(true_I * count_rate * count_time)))
    
    I_measured = counts / (count_rate * count_time)
    sigma = np.sqrt(max(counts, 1)) / (count_rate * count_time)
    
    return SimulatedMeasurement(H=H, E=E, I=I_measured, sigma=sigma, true_I=true_I)


def _compute_sqw_with_gaussian_energy(h: float, k: float, l: float, E: float,
                                      sqw_function: Callable,
                                      fwhm: float) -> float:
    """Fallback: simple 1D Gaussian convolution in energy."""
    if fwhm <= 0:
        return sqw_function(h, k, l, E)
    sigma = fwhm / 2.355
    n_points = 7
    offsets = np.linspace(-3 * sigma, 3 * sigma, n_points)
    weights = np.exp(-offsets**2 / (2 * sigma**2))
    weights /= weights.sum()
    return float(np.sum([w * sqw_function(h, k, l, E + dE) for dE, w in zip(offsets, weights)]))


def _compute_sqw_with_cooper_nathans(h: float, k: float, l: float, E: float,
                                     sqw_function: Callable,
                                     res_calc: "TASResolutionCalculator",
                                     gaussian_fwhm: float) -> float:
    """Compute S(Q,ω) with full Cooper-Nathans resolution convolution."""
    if h == 0.0 and k == 0.0 and l == 0.0:
        return _compute_sqw_with_gaussian_energy(h, k, l, E, sqw_function, gaussian_fwhm)
    try:
        fwhm, _ = res_calc.get_resolution_fwhm(h, k, l, E)
    except Exception:
        return _compute_sqw_with_gaussian_energy(h, k, l, E, sqw_function, gaussian_fwhm)

    sigma_Qx = fwhm["Qx"] / 2.355 if fwhm["Qx"] < np.inf else 0.1
    sigma_Qy = fwhm["Qy"] / 2.355 if fwhm["Qy"] < np.inf else 0.1
    sigma_E = fwhm["E"] / 2.355 if fwhm["E"] < np.inf else 0.5

    xvec = np.array(res_calc.lattice.x[:, 0]) if hasattr(res_calc.lattice, "x") else np.array([1, 0, 0])
    yvec = np.array(res_calc.lattice.y[:, 0]) if hasattr(res_calc.lattice, "y") else np.array([0, 1, 0])

    if not np.all(np.isfinite([sigma_Qx, sigma_Qy, sigma_E])):
        return _compute_sqw_with_gaussian_energy(h, k, l, E, sqw_function, gaussian_fwhm)

    n_quad = 5
    quad_points = np.linspace(-2, 2, n_quad)
    weights = np.exp(-quad_points**2 / 2)
    weights /= weights.sum()

    result = 0.0
    total_weight = 0.0

    for dx, wx in zip(quad_points * sigma_Qx, weights):
        for dy, wy in zip(quad_points * sigma_Qy, weights):
            for dE, wE in zip(quad_points * sigma_E, weights):
                h1 = h + dx * xvec[0] + dy * yvec[0]
                k1 = k + dx * xvec[1] + dy * yvec[1]
                l1 = l + dx * xvec[2] + dy * yvec[2]
                E1 = E + dE

                w_total = wx * wy * wE
                try:
                    sqw_val = sqw_function(h1, k1, l1, E1)
                    result += w_total * sqw_val
                    total_weight += w_total
                except (ValueError, RuntimeError):
                    pass

    if total_weight > 0:
        result /= total_weight
    return result


def _wrap_intensity_with_resolution(intensity_func: Callable,
                                    res_calc: "TASResolutionCalculator",
                                    gaussian_fwhm: float) -> Callable:
    def sqw_func(h, k, l, E):
        return intensity_func(h, E)

    def wrapped(H: float, E: float) -> float:
        return _compute_sqw_with_cooper_nathans(H, 0.0, 0.0, E, sqw_func, res_calc, gaussian_fwhm)

    return wrapped


# =============================================================================
# Benchmark Methods
# =============================================================================

class GridMethod:
    """Baseline: regular grid scanning."""
    
    def __init__(self, bounds: np.ndarray, n_per_dim: int = 20):
        self.bounds = bounds
        self.n_per_dim = n_per_dim
        
        # Create grid
        H_grid = np.linspace(bounds[0, 0], bounds[0, 1], n_per_dim)
        E_grid = np.linspace(bounds[1, 0], bounds[1, 1], n_per_dim)
        
        self.grid_points = []
        for H in H_grid:
            for E in E_grid:
                self.grid_points.append(np.array([H, E]))
        
        self.current_idx = 0
        self._obs = []
    
    def suggest_next(self) -> np.ndarray:
        if self.current_idx >= len(self.grid_points):
            return None
        point = self.grid_points[self.current_idx]
        self.current_idx += 1
        return point
    
    def add_observation(self, x, I, sigma):
        self._obs.append((x, I, sigma))

    @property
    def observations(self):
        return self._obs


class RandomMethod:
    """Baseline: random sampling."""
    
    def __init__(self, bounds: np.ndarray):
        self.bounds = bounds
        self.observations = []
    
    def suggest_next(self) -> np.ndarray:
        return np.random.uniform(self.bounds[:, 0], self.bounds[:, 1])
    
    def add_observation(self, x, I, sigma):
        self.observations.append((x, I, sigma))


class LogGPMethod:
    """Enhanced Log-GP explorer: log-intensity GP regression (Teixeira-Parente / JCNS-inspired)
    with coarse-grid init, consumed-area exclusion, 1D cosine energy taper, and linear-space
    variance weighting. See SI Note S1 of the TAS-AI manuscript for the design rationale."""

    def __init__(self, bounds: np.ndarray, max_measurements: int = 200):
        self.bounds = bounds
        self.explorer = AgnosticExplorer(bounds, use_log_gp=True)
        self.max_measurements = max_measurements
        self._obs = []

        # Coarse initialization budget (intentionally much smaller than pure grid baseline).
        init_budget = max(12, min(30, int(0.2 * max_measurements)))
        h_range = bounds[0, 1] - bounds[0, 0]
        e_range = bounds[1, 1] - bounds[1, 0]
        self.consume_dh = max(0.01, 0.03 * h_range)
        self.consume_dE = max(0.25, 0.03 * e_range)
        n_h = max(3, int(np.sqrt(init_budget * 0.8)))
        n_e = max(3, int(np.ceil(init_budget / n_h)))
        h_vals = np.linspace(bounds[0, 0], bounds[0, 1], n_h)
        e_vals = np.linspace(bounds[1, 0], bounds[1, 1], n_e)
        self.init_points = [np.array([h, e], dtype=float) for h in h_vals for e in e_vals]
        self._init_idx = 0

        # JCNS-like intensity handling
        self.tau = None
        self.gamma = None

    def _estimate_tau_gamma(self):
        if len(self._obs) < len(self.init_points):
            return
        vals = np.array([o[1] for o in self._obs[:len(self.init_points)]], dtype=float)
        if vals.size < 6:
            return
        q10, q30, q50, q70, q90 = np.percentile(vals, [10, 30, 50, 70, 90])
        self.tau = max(q70, 1e-6)
        self.gamma = max(0.0, min(q30, q50, 0.5 * q90))

    def _transform_intensity(self, I: float) -> float:
        if self.tau is None or self.gamma is None:
            return max(I, 0.0)
        return min(max(I - self.gamma, 0.0), self.tau)

    def _is_consumed(self, x: np.ndarray) -> bool:
        if len(self._obs) == 0:
            return False
        for xo, _, _ in self._obs:
            dh = (x[0] - xo[0]) / self.consume_dh
            dE = (x[1] - xo[1]) / self.consume_dE
            if dh * dh + dE * dE <= 1.0:
                return True
        return False

    def _energy_window(self, E: float, taper: float = 0.1) -> float:
        emin, emax = self.bounds[1]
        if emax <= emin:
            return 1.0
        u = (E - emin) / (emax - emin)
        u = min(max(u, 0.0), 1.0)
        if u < taper:
            return 0.5 * (1.0 - np.cos(np.pi * u / taper))
        if u > 1.0 - taper:
            return 0.5 * (1.0 - np.cos(np.pi * (1.0 - u) / taper))
        return 1.0
    
    def suggest_next(self) -> np.ndarray:
        # Coarse grid init first
        while self._init_idx < len(self.init_points):
            x = self.init_points[self._init_idx]
            self._init_idx += 1
            if not self._is_consumed(x):
                return x

        if len(self.explorer.observations) < 5:
            return self.explorer.suggest_initial()

        # Active phase: linearized variance + 1D taper + consumed-area exclusion + motion-awareness
        self.explorer.gp.fit()
        n_candidates = 1500
        candidates = np.random.uniform(self.bounds[:, 0], self.bounds[:, 1], size=(n_candidates, 2))
        if len(self._obs) == 0:
            return candidates[0]
        last_x = self._obs[-1][0]

        best_score = -np.inf
        best_x = None
        for x in candidates:
            if self._is_consumed(x):
                continue
            mean, std = self.explorer.gp.predict(x)
            linear_var = max(std, 0.0) ** 2
            window = self._energy_window(float(x[1]))
            dh = abs(float(x[0] - last_x[0]))
            dE = abs(float(x[1] - last_x[1]))
            move_cost = 1.0 + dh + 0.05 * dE
            score = (linear_var * window) / move_cost
            if score > best_score:
                best_score = score
                best_x = x

        if best_x is None:
            return self.explorer.suggest_next(acquisition='variance')
        return best_x
    
    def add_observation(self, x, I, sigma):
        x = np.array(x, dtype=float)
        self._obs.append((x, float(I), float(sigma)))
        self._estimate_tau_gamma()
        I_eff = self._transform_intensity(float(I))
        self.explorer.add_observation(x, I_eff, sigma)
    
    @property
    def observations(self):
        return list(self._obs)


class TASAIMethod:
    """
    TAS-AI physics-informed method.
    
    Uses physics model to guide acquisition:
    - Focuses on dispersion curve
    - Optimizes for parameter estimation
    """
    
    def __init__(self, bounds: np.ndarray, 
                 J_prior: float = 5.0, gamma_prior: float = 0.5,
                 backend: str = 'heuristic'):
        self.bounds = bounds
        self.J_est = J_prior
        self.gamma_est = gamma_prior
        self.observations = []
        self.backend = backend
        
        # Track parameter uncertainty
        self.J_std = 2.0
        self._physics_model = None
    
    def _ensure_physics_model(self):
        if self.backend not in {'sunny', 'pyspinw'}:
            return
        if self._physics_model is None:
            if self.backend == 'sunny':
                try:
                    from tasai.sunny import SquareLatticeFM
                except ImportError:
                    return
                self._physics_model = SquareLatticeFM(J1=self.J_est, J2=0.0, D=0.1)
            elif self.backend == 'pyspinw':
                try:
                    from tasai.physics.spinwave import SpinWaveModel, SpinWaveConfig
                    config = SpinWaveConfig(
                        lat_const=(3.8, 3.8, 8.0),
                        angles=(90, 90, 90),
                        atoms=[([0, 0, 0], 1.0, 'M1')],
                        propagation_k=(0.0, 0.0, 0.0)
                    )
                    self._physics_model = SpinWaveModel(
                        config=config,
                        exchanges={'J1': self.J_est},
                        bonds={'J1': 1},
                        backend='pyspinw'
                    )
                    test_val = float(self._physics_model.compute_intensity(0.25, 0.25, 0.0, 10.0))
                    if not np.isfinite(test_val):
                        raise RuntimeError("PySpinW intensity invalid")
                except Exception as exc:
                    print(f"[WARN] PySpinW backend unavailable ({exc}); falling back to Sunny analytic model.")
                    self.backend = 'sunny'
                    self._physics_model = None
                    self._ensure_physics_model()
    
    def _fit_physics_model(self):
        if self.backend not in {'sunny', 'pyspinw'}:
            return
        if len(self.observations) < 5:
            return
        self._ensure_physics_model()
        if self._physics_model is None:
            return
        
        H_data = np.array([obs[0][0] for obs in self.observations])
        E_data = np.array([obs[0][1] for obs in self.observations])
        I_data = np.array([obs[1] for obs in self.observations])
        sigma_data = np.maximum(np.array([obs[2] for obs in self.observations]), 1e-2)

        if self.backend == 'pyspinw':
            param_vals = self._physics_model.get_values()
            if 'J1' not in param_vals:
                return
            J1_grid = np.linspace(1.0, 10.0, 15)
            best_chi2 = np.inf
            best_J1 = param_vals['J1']

            for J1 in J1_grid:
                self._physics_model.set_parameters({'J1': J1})
                pred = np.array([
                    self._physics_model.intensity(float(h), 0.0, float(e))
                    for h, e in zip(H_data, E_data)
                ])
                chi2 = np.sum(((I_data - pred) / sigma_data)**2)
                if chi2 < best_chi2:
                    best_chi2 = chi2
                    best_J1 = J1

            self._physics_model.set_parameters({'J1': best_J1})
            return
        
        J1_grid = np.linspace(1.0, 10.0, 15)
        J2_grid = np.linspace(-0.5, 2.0, 12)
        D_grid = np.linspace(0.01, 0.3, 8)
        
        best_chi2 = np.inf
        best_params = (self._physics_model.J1, self._physics_model.J2, self._physics_model.D)
        
        for J1 in J1_grid:
            for J2 in J2_grid:
                for D in D_grid:
                    self._physics_model.set_parameters(J1=J1, J2=J2, D=D)
                    pred = self._physics_model.intensity(H_data, 0, E_data)
                    chi2 = np.sum(((I_data - pred) / sigma_data)**2)
                    if chi2 < best_chi2:
                        best_chi2 = chi2
                        best_params = (J1, J2, D)
        
        if self.backend == 'sunny':
            self._physics_model.set_parameters(J1=best_params[0],
                                               J2=best_params[1],
                                               D=best_params[2])
        elif self.backend == 'pyspinw':
            try:
                self._physics_model.set_parameter('J1', best_params[0])
            except Exception:
                pass
    
    def _estimated_dispersion(self, H: float) -> float:
        """Current estimate of dispersion."""
        return 2 * self.J_est * (1 - np.cos(2 * np.pi * H))
    
    def suggest_next(self) -> np.ndarray:
        if self.backend == 'sunny':
            point = self._suggest_with_physics()
            if point is not None:
                return point
        if len(self.observations) < 5:
            # Initial: sample along estimated dispersion
            H = np.random.uniform(self.bounds[0, 0] + 0.05, self.bounds[0, 1])
            E = self._estimated_dispersion(H) + np.random.normal(0, 1)
            E = np.clip(E, self.bounds[1, 0], self.bounds[1, 1])
            return np.array([H, E])
        
        # Physics-informed: focus on dispersion curve
        # Score by information gain for J estimation
        
        best_score = -np.inf
        best_point = None
        
        for _ in range(50):
            H = np.random.uniform(self.bounds[0, 0] + 0.02, self.bounds[0, 1])
            E_disp = self._estimated_dispersion(H)
            
            # Sample near dispersion
            E = E_disp + np.random.uniform(-2, 2)
            E = np.clip(E, self.bounds[1, 0], self.bounds[1, 1])
            
            # Score: high info gain where dω/dJ is large
            # dω/dJ = 2(1 - cos(2πH))
            sensitivity = 2 * (1 - np.cos(2 * np.pi * H))
            
            # Novelty
            novelty = 1.0
            for (x, _, _) in self.observations:
                dist = np.sqrt((H - x[0])**2 + (E - x[1])**2 / 100)
                novelty *= (1 - 0.8 * np.exp(-dist**2 / 0.01))
            
            score = sensitivity * novelty
            
            if score > best_score:
                best_score = score
                best_point = np.array([H, E])
        
        return best_point
    
    def _suggest_with_physics(self) -> np.ndarray:
        if self._physics_model is None or len(self.observations) < 5:
            return None
        if not hasattr(self._physics_model, 'dispersion'):
            return None
        H_candidates = np.linspace(self.bounds[0, 0] + 0.02,
                                   self.bounds[0, 1] - 0.02, 25)
        best_score = -np.inf
        best_point = None
        
        for H in H_candidates:
            E_disp = float(self._physics_model.dispersion(H, 0))
            offsets = np.array([-1.0, -0.5, 0.0, 0.5, 1.0])
            E_candidates = E_disp + 0.5 * offsets
            E_candidates = E_candidates[(E_candidates > self.bounds[1, 0]) &
                                        (E_candidates < self.bounds[1, 1])]
            if len(E_candidates) == 0:
                continue
            for E in E_candidates:
                intensity = float(self._physics_model.intensity(H, 0, E))
                novelty = 1.0
                for (x, _, _) in self.observations:
                    dist = np.sqrt((H - x[0])**2 + (E - x[1])**2 / 25)
                    novelty *= (1 - 0.8 * np.exp(-dist**2 / 0.005))
                j2_sensitivity = 1 + 0.5 * np.sin(4 * np.pi * H)**2
                score = intensity * novelty * j2_sensitivity
                if score > best_score:
                    best_score = score
                    best_point = np.array([H, E])
        
        return best_point
    
    def add_observation(self, x, I, sigma):
        self.observations.append((x, I, sigma))
        
        # Simple J update based on peak positions
        if len(self.observations) >= 5:
            self._update_J_estimate()
            self._fit_physics_model()
    
    def _update_J_estimate(self):
        """Update J estimate from observations."""
        # Find observations with high intensity
        high_I = [(x, I) for (x, I, _) in self.observations if I > 0.1]
        
        if len(high_I) < 3:
            return
        
        # Fit: E_peak ≈ 2J(1 - cos(2πH))
        J_estimates = []
        for (x, I) in high_I:
            H, E = x
            denom = 2 * (1 - np.cos(2 * np.pi * H))
            if denom > 0.1:
                J_estimates.append(E / denom)
        
        if J_estimates:
            self.J_est = np.median(J_estimates)
            self.J_std = np.std(J_estimates) if len(J_estimates) > 2 else 1.0


# =============================================================================
# Benchmark Runner
# =============================================================================

def compute_reconstruction_error(method, intensity_func: Callable, 
                                bounds: np.ndarray, n_grid: int = 30) -> float:
    """
    Compute relative weighted reconstruction error.
    
    Error = Σ |I_pred - I_true| × I_true / Σ I_true²
    
    This weights errors by intensity (focuses on signal regions).
    Predictions are reconstructed from the raw measured intensities for every
    method so the benchmark reflects acquisition quality rather than
    method-specific surrogate transforms.
    """
    H_grid = np.linspace(bounds[0, 0], bounds[0, 1], n_grid)
    E_grid = np.linspace(bounds[1, 0], bounds[1, 1], n_grid)

    observations = list(method.observations) if hasattr(method, 'observations') else []
    
    error_sum = 0.0
    weight_sum = 0.0
    
    for H in H_grid:
        for E in E_grid:
            I_true = intensity_func(H, E)
            I_pred = _idw_predict(observations, H, E)
            
            I_pred = max(float(I_pred), 0.0)
            I_true = max(float(I_true), 0.0)

            error_sum += abs(I_pred - I_true) * I_true
            weight_sum += I_true**2
    
    return error_sum / (weight_sum + 1e-12)


def _idw_predict(observations: List, H: float, E: float, k: int = 8, power: float = 2.0) -> float:
    """Inverse-distance interpolation from raw observations."""
    if len(observations) == 0:
        return 0.0

    weighted = []
    for (x, I, _) in observations:
        dist2 = (H - x[0])**2 + (E - x[1])**2 / 100.0
        if dist2 <= 1e-12:
            return float(I)
        weighted.append((dist2, float(I)))

    weighted.sort(key=lambda item: item[0])
    top = weighted[: min(k, len(weighted))]
    weights = np.array([1.0 / (d2 ** (power / 2.0)) for d2, _ in top], dtype=float)
    values = np.array([val for _, val in top], dtype=float)
    return float(np.dot(weights, values) / np.sum(weights))


def _nearest_neighbor_predict(observations: List, H: float, E: float) -> float:
    """Simple nearest neighbor prediction."""
    if len(observations) == 0:
        return 0.0
    
    min_dist = float('inf')
    nearest_I = 0.0
    
    for (x, I, _) in observations:
        dist = (H - x[0])**2 + (E - x[1])**2 / 100
        if dist < min_dist:
            min_dist = dist
            nearest_I = I
    
    return nearest_I


def run_benchmark(scenario_name: str, method_name: str, 
                 max_measurements: int = 200,
                 error_threshold: float = 0.3,
                 seed: int = 42,
                 tasai_backend: str = 'heuristic',
                 resolution_calculator: "TASResolutionCalculator | None" = None,
                 gaussian_fwhm: float = 1.0,
                 checkpoint_dir: str | None = None,
                 checkpoint_prefix: str = "jcns_checkpoint",
                 checkpoint_interval: int | None = None) -> Dict:
    """
    Run a single benchmark.
    
    Returns dict with:
    - measurements_to_converge: N to reach error threshold
    - final_error: error after max_measurements
    - time_per_suggestion: computation time
    - errors: error history
    """
    np.random.seed(seed)
    
    scenario = BENCHMARK_SCENARIOS[scenario_name]
    intensity_func = scenario['function']
    if resolution_calculator is not None:
        intensity_func = _wrap_intensity_with_resolution(
            intensity_func,
            resolution_calculator,
            gaussian_fwhm,
        )
    bounds = scenario['bounds']
    
    # Create method
    if method_name == 'grid':
        n_per_dim = int(np.sqrt(max_measurements))
        method = GridMethod(bounds, n_per_dim=n_per_dim)
    elif method_name == 'random':
        method = RandomMethod(bounds)
    elif method_name == 'log_gp':
        method = LogGPMethod(bounds)
    elif method_name == 'tasai':
        method = TASAIMethod(bounds, backend=tasai_backend)
    else:
        raise ValueError(f"Unknown method: {method_name}")
    
    # Run experiment
    errors = []
    times = []
    measurements_to_converge = None
    
    for i in range(max_measurements):
        # Suggest next point
        t0 = time.time()
        x = method.suggest_next()
        t_suggest = time.time() - t0
        times.append(t_suggest)
        
        if x is None:
            break
        
        # Measure
        m = simulate_measurement(x[0], x[1], intensity_func)
        method.add_observation(x, m.I, m.sigma)
        
        # Compute error every 5 measurements (or on checkpoint interval)
        if (i + 1) % 5 == 0 or i == max_measurements - 1 or (
            checkpoint_interval and (i + 1) % checkpoint_interval == 0
        ):
            error = compute_reconstruction_error(method, intensity_func, bounds)
            errors.append((i + 1, error))
            
            if measurements_to_converge is None and error < error_threshold:
                measurements_to_converge = i + 1

            if checkpoint_dir and checkpoint_interval and (
                (i + 1) % checkpoint_interval == 0 or i == max_measurements - 1
            ):
                _write_run_checkpoint(
                    {
                        'scenario': scenario_name,
                        'method': method_name,
                        'measurements_to_converge': measurements_to_converge,
                        'final_error': error,
                        'mean_time_per_suggestion': np.mean(times),
                        'errors': errors,
                        'n_measurements': len(method.observations),
                    },
                    checkpoint_dir,
                    checkpoint_prefix,
                    scenario_name,
                    method_name,
                    seed,
                )
    
    return {
        'scenario': scenario_name,
        'method': method_name,
        'measurements_to_converge': measurements_to_converge,
        'final_error': errors[-1][1] if errors else float('inf'),
        'mean_time_per_suggestion': np.mean(times),
        'errors': errors,
        'n_measurements': len(method.observations)
    }


def _write_checkpoint(results: Dict, checkpoint_dir: str, checkpoint_prefix: str, scenario: str) -> None:
    path = Path(checkpoint_dir) / f"{checkpoint_prefix}_{scenario}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(results, f, indent=2, default=float)


def _write_run_checkpoint(run_result: Dict, checkpoint_dir: str, checkpoint_prefix: str,
                          scenario: str, method: str, seed: int) -> None:
    path = Path(checkpoint_dir) / f"{checkpoint_prefix}_{scenario}_{method}_seed{seed}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(run_result, f, indent=2, default=float)


def run_all_benchmarks(methods: List[str] = None,
                       scenarios: List[str] = None,
                       n_runs: int = 5,
                       max_measurements: int = 120,
                       error_threshold: float = 0.2,
                       tasai_backend: str = 'heuristic',
                       seed_start: int = 0,
                       resolution_calculator: "TASResolutionCalculator | None" = None,
                       gaussian_fwhm: float = 1.0,
                       checkpoint_dir: str | None = None,
                       checkpoint_prefix: str = "jcns_checkpoint",
                       checkpoint_interval: int | None = None) -> Dict:
    """Run all benchmarks with multiple seeds."""
    
    if methods is None:
        methods = ['grid', 'random', 'log_gp', 'tasai']
    
    if scenarios is None:
        scenarios = list(BENCHMARK_SCENARIOS.keys())
    
    results = {}
    
    for scenario in scenarios:
        print(f"\nScenario: {scenario}")
        print("-" * 50)
        results[scenario] = {}
        
        for method in methods:
            run_results = []
            
            for seed in range(seed_start, seed_start + n_runs):
                r = run_benchmark(scenario, method, 
                                 max_measurements=max_measurements,
                                 error_threshold=error_threshold,
                                 seed=seed,
                                 tasai_backend=tasai_backend,
                                 resolution_calculator=resolution_calculator,
                                 gaussian_fwhm=gaussian_fwhm,
                                 checkpoint_dir=checkpoint_dir,
                                 checkpoint_prefix=checkpoint_prefix,
                                 checkpoint_interval=checkpoint_interval)
                run_results.append(r)
            
            # Aggregate
            converge_times = [r['measurements_to_converge'] for r in run_results 
                            if r['measurements_to_converge'] is not None]
            final_errors = [r['final_error'] for r in run_results]
            
            results[scenario][method] = {
                'mean_converge': np.mean(converge_times) if converge_times else float('inf'),
                'std_converge': np.std(converge_times) if len(converge_times) > 1 else 0,
                'converge_rate': len(converge_times) / n_runs,
                'mean_final_error': np.mean(final_errors),
                'std_final_error': np.std(final_errors),
                'runs': run_results
            }
            
            conv_str = f"{results[scenario][method]['mean_converge']:.0f}" if converge_times else "N/A"
            print(f"  {method:10s}: converge={conv_str:>5s} pts, "
                  f"final_error={results[scenario][method]['mean_final_error']:.3f}")

        if checkpoint_dir:
            _write_checkpoint(results, checkpoint_dir, checkpoint_prefix, scenario)
    
    results['metadata'] = {
        'max_measurements': max_measurements,
        'error_threshold': error_threshold,
        'n_runs': n_runs,
        'seed_start': seed_start,
    }
    return results


def plot_benchmark_results(results: Dict, save_path: str = None):
    """Plot benchmark comparison."""
    scenarios = [s for s in results.keys() if s != 'metadata']
    methods = list(results[scenarios[0]].keys())
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    panel_labels = ['(a)', '(b)', '(c)', '(d)']
    metadata = results.get('metadata', {})
    threshold = metadata.get('error_threshold', 0.1)
    max_measurements = metadata.get('max_measurements', 0)
    for scenario in scenarios:
        for method in methods:
            runs = results[scenario][method]['runs']
            if runs:
                max_measurements = max(max_measurements, max(r['n_measurements'] for r in runs))
    max_measurements = max(max_measurements, 1)
    
    x = np.arange(len(scenarios))
    width = 0.18
    scenario_labels = [s.replace('_', '\n') for s in scenarios]
    
    # 1. Convergence comparison
    ax = axes[0, 0]
    ax.text(0.01, 0.95, panel_labels[0], transform=ax.transAxes,
            fontsize=12, fontweight='bold', va='top')
    for i, method in enumerate(methods):
        conv_times = np.array([results[s][method]['mean_converge'] for s in scenarios], dtype=float)
        conv_err = np.array([results[s][method]['std_converge'] for s in scenarios], dtype=float)
        conv_times[~np.isfinite(conv_times)] = np.nan
        ax.bar(
            x + i*width,
            conv_times,
            width,
            label=method,
            yerr=conv_err,
            capsize=3,
            color=METHOD_COLORS.get(method, None)
        )
    
    ax.set_ylabel('Measurements to converge')
    ax.set_xticks(x + width * (len(methods) - 1) / 2)
    ax.set_xticklabels(scenario_labels, fontsize=9)
    ax.set_ylim(0, max_measurements * 1.05)
    ax.legend()
    ax.set_title('Convergence speed (lower is better)')
    
    # 2. Final error comparison
    ax = axes[0, 1]
    ax.text(0.01, 0.95, panel_labels[1], transform=ax.transAxes,
            fontsize=12, fontweight='bold', va='top')
    for i, method in enumerate(methods):
        errors = np.array([results[s][method]['mean_final_error'] for s in scenarios], dtype=float)
        err_std = np.array([results[s][method]['std_final_error'] for s in scenarios], dtype=float)
        ax.bar(
            x + i*width,
            errors,
            width,
            label=method,
            yerr=err_std,
            capsize=3,
            color=METHOD_COLORS.get(method, None)
        )
    
    ax.set_ylabel('Final reconstruction error')
    ax.set_xticks(x + width * (len(methods) - 1) / 2)
    ax.set_xticklabels(scenario_labels, fontsize=9)
    ax.legend()
    ax.set_title('Final error (lower is better)')
    
    # 3. Error curves for single_branch
    ax = axes[1, 0]
    ax.text(0.01, 0.95, panel_labels[2], transform=ax.transAxes,
            fontsize=12, fontweight='bold', va='top')
    if 'single_branch' in results:
        for method in methods:
            runs = results['single_branch'][method]['runs']
            if runs and runs[0]['errors']:
                errors = runs[0]['errors']
                ns, errs = zip(*errors)
                ax.plot(
                    ns,
                    errs,
                    'o-',
                    label=method,
                    markersize=4,
                    color=METHOD_COLORS.get(method, None)
                )
    
    ax.set_xlabel('Number of measurements')
    ax.set_ylabel('Reconstruction error')
    ax.set_yscale('log')
    ax.axhline(threshold, color='k', ls='--', alpha=0.5,
               label=f'Target error {threshold}')
    ax.legend()
    ax.set_title('Convergence curves (single branch)')
    
    # 4. Relative speedup vs grid
    ax = axes[1, 1]
    ax.text(0.01, 0.95, panel_labels[3], transform=ax.transAxes,
            fontsize=12, fontweight='bold', va='top')
    baseline = {s: results[s]['grid']['mean_converge'] for s in scenarios}
    
    plotted_methods = [m for m in methods if m != 'grid']
    for i, method in enumerate(plotted_methods):
        speedups = []
        for s in scenarios:
            base = baseline[s]
            target = results[s][method]['mean_converge']
            if np.isfinite(base) and np.isfinite(target) and target > 0:
                speedups.append(base / target)
            else:
                speedups.append(np.nan)
        ax.bar(
            x + i*width,
            speedups,
            width,
            label=method,
            color=METHOD_COLORS.get(method, None)
        )
    
    ax.axhline(1.0, color='k', ls='--', alpha=0.5, label='Grid parity')
    ax.set_ylabel('Speedup vs grid (×)')
    ax.set_xticks(x + width * (len(plotted_methods) - 1) / 2)
    ax.set_xticklabels(scenario_labels, fontsize=9)
    ax.set_ylim(0, max(1.5, ax.get_ylim()[1]))
    ax.legend()
    ax.set_title('Relative efficiency')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"\nPlot saved to: {save_path}")
    
    plt.show()
    return fig


def plot_scenario_dispersions(save_path: str = None):
    """Visualize benchmark scenarios to show where modes live."""
    scenario_names = list(BENCHMARK_SCENARIOS.keys())
    n_scenarios = len(scenario_names)
    ncols = 3
    nrows = int(np.ceil(n_scenarios / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.2 * ncols, 3.6 * nrows), sharey=False)
    axes = np.atleast_2d(axes)
    panel_labels = [f"({chr(97 + i)})" for i in range(nrows * ncols)]
    
    for idx, name in enumerate(scenario_names):
        row, col = divmod(idx, ncols)
        ax = axes[row, col]
        ax.text(0.02, 0.95, panel_labels[idx], transform=ax.transAxes,
                fontsize=11, fontweight='bold', va='top')
        scenario = BENCHMARK_SCENARIOS[name]
        bounds = scenario['bounds']
        
        H = np.linspace(bounds[0, 0], bounds[0, 1], 160)
        E = np.linspace(bounds[1, 0], bounds[1, 1], 200)
        HH, EE = np.meshgrid(H, E, indexing='ij')
        intensity = scenario['function'](HH, EE)
        
        im = ax.pcolormesh(H, E, intensity.T, shading='auto', cmap='inferno')
        for label, func in scenario.get('dispersions', []):
            ax.plot(H, func(H), lw=1.5, label=label)
        
        ax.set_title(name.replace('_', ' ').title())
        ax.set_xlabel('[H H 0] (r.l.u.)')
        ax.set_ylabel('Energy (meV)')
        if scenario.get('dispersions'):
            ax.legend(fontsize=8, loc='upper right')
    
    # Hide unused axes
    for idx in range(n_scenarios, nrows * ncols):
        row, col = divmod(idx, ncols)
        axes[row, col].axis('off')
    
    cax = fig.add_axes([0.92, 0.15, 0.015, 0.7])
    fig.colorbar(im, cax=cax, label='Intensity (a.u.)')
    plt.subplots_adjust(right=0.9, hspace=0.4, wspace=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"\nScenario figure saved to: {save_path}")
    
    plt.show()
    return fig


def main():
    parser = argparse.ArgumentParser(description="JCNS Benchmark Comparison")
    parser.add_argument("--method", type=str, default=None,
                       help="Method to test (grid, random, log_gp, tasai)")
    parser.add_argument("--scenario", type=str, default=None,
                       help="Scenario to test")
    parser.add_argument("--max-measurements", type=int, default=120)
    parser.add_argument("--n-runs", type=int, default=5)
    parser.add_argument("--seed", type=int, default=None,
                       help="Seed for a single-run benchmark (overrides n-runs)")
    parser.add_argument("--error-threshold", type=float, default=0.2,
                       help="Relative error threshold for convergence")
    parser.add_argument("--save", type=str, help="Save benchmark summary figure")
    parser.add_argument("--save-scenarios", type=str, help="Save scenario dispersion figure")
    parser.add_argument("--summary-json", type=str, help="Write benchmark results to JSON")
    parser.add_argument("--tasai-backend", type=str, default='pyspinw',
                       choices=['heuristic', 'sunny', 'pyspinw'],
                       help="Acquisition backend for TAS-AI method")
    parser.add_argument("--ground-truth", type=str, default='analytic',
                       choices=['analytic', 'pyspinw'],
                       help="Ground truth intensity model")
    parser.add_argument("--cooper-nathans", action="store_true",
                       help="Apply full Cooper-Nathans resolution convolution")
    parser.add_argument("--resolution-backend", type=str, default="numpy",
                       choices=["auto", "numba", "numpy", "pytorch"],
                       help="Resolution backend for rescalculator")
    parser.add_argument("--hcol", type=float, nargs=4, default=(40, 40, 40, 40),
                       help="Horizontal collimations in arcmin (4 values)")
    parser.add_argument("--vcol", type=float, nargs=4, default=(120, 120, 120, 120),
                       help="Vertical collimations in arcmin (4 values)")
    parser.add_argument("--efixed", type=float, default=14.7,
                       help="Fixed final energy (meV) for resolution model")
    parser.add_argument("--gaussian-fwhm", type=float, default=1.0,
                       help="Fallback Gaussian energy FWHM (meV)")
    parser.add_argument("--no-plot", action="store_true")
    parser.add_argument("--checkpoint-dir", type=str, default=None,
                       help="Write partial JSON after each scenario")
    parser.add_argument("--checkpoint-prefix", type=str, default="jcns_checkpoint",
                       help="Prefix for checkpoint JSON files")
    parser.add_argument("--checkpoint-interval", type=int, default=0,
                       help="Write per-run checkpoints every N measurements (0 disables)")
    
    args = parser.parse_args()
    
    methods = [args.method] if args.method else None
    scenarios = [args.scenario] if args.scenario else None
    if args.ground_truth == 'pyspinw':
        global BENCHMARK_SCENARIOS
        BENCHMARK_SCENARIOS = _build_pyspinw_ground_truth_scenarios()
    
    print("=" * 60)
    print("JCNS Benchmark Comparison")
    print("Comparing: Grid, Random, Log-GP (JCNS), TAS-AI")
    print("=" * 60)
    
    n_runs = args.n_runs
    seed_start = 0
    if args.seed is not None:
        n_runs = 1
        seed_start = args.seed

    resolution_calculator = None
    if args.cooper_nathans:
        if not HAS_RESOLUTION:
            raise RuntimeError("Resolution convolution requested but rescalculator is not available.")
        resolution_calculator = TASResolutionCalculator(
            lattice_params=(4.0, 4.0, 10.0, 90, 90, 90),
            orient1=[1, 0, 0],
            orient2=[0, 1, 0],
            exp_config=create_default_tas_config(
                efixed=args.efixed,
                hcol=tuple(args.hcol),
                vcol=tuple(args.vcol),
            ),
            backend=args.resolution_backend,
        )

    checkpoint_interval = args.checkpoint_interval if args.checkpoint_interval > 0 else None
    results = run_all_benchmarks(
        methods=methods,
        scenarios=scenarios,
        n_runs=n_runs,
        max_measurements=args.max_measurements,
        error_threshold=args.error_threshold,
        tasai_backend=args.tasai_backend,
        seed_start=seed_start,
        resolution_calculator=resolution_calculator,
        gaussian_fwhm=args.gaussian_fwhm,
        checkpoint_dir=args.checkpoint_dir,
        checkpoint_prefix=args.checkpoint_prefix,
        checkpoint_interval=checkpoint_interval
    )
    results["metadata"]["ground_truth"] = args.ground_truth

    if resolution_calculator is not None:
        results["metadata"]["resolution_method"] = "cooper-nathans"
        results["metadata"]["resolution_hcol"] = list(args.hcol)
        results["metadata"]["resolution_vcol"] = list(args.vcol)
        results["metadata"]["resolution_efixed"] = args.efixed
        results["metadata"]["resolution_backend"] = args.resolution_backend
    
    if not args.no_plot or args.save:
        plot_benchmark_results(results, save_path=args.save)
    
    if args.save_scenarios:
        plot_scenario_dispersions(save_path=args.save_scenarios)
    
    if args.summary_json:
        import json
        out_path = Path(args.summary_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open('w') as f:
            json.dump(results, f, indent=2, default=float)
        print(f"\nBenchmark summary written to: {out_path}")
    
    # Print summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    all_scenarios = [s for s in results.keys() if s != 'metadata']
    if not all_scenarios:
        return
    all_methods = list(results[all_scenarios[0]].keys())
    
    # Average speedup vs grid (only when grid results are present)
    has_grid = all(
        'grid' in results[scenario]
        for scenario in all_scenarios
    )
    if has_grid:
        for method in all_methods:
            speedups = []
            for scenario in all_scenarios:
                grid_conv = results[scenario]['grid']['mean_converge']
                method_conv = results[scenario][method]['mean_converge']
                if grid_conv < float('inf') and method_conv < float('inf'):
                    speedups.append(grid_conv / method_conv)
            
            if speedups:
                print(f"{method:10s}: {np.mean(speedups):.1f}x average speedup vs grid")
    else:
        print("Grid baseline missing; skipping average speedup vs grid.")


if __name__ == "__main__":
    main()
