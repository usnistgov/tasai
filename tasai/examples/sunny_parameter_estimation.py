#!/usr/bin/env python3
"""
Parameter Estimation for Square Lattice FM

Autonomous experiment to determine exchange parameters (J1, J2, D)
from spin wave measurements in the HHL zone.

This example demonstrates:
1. Setting up the Sunny.jl model
2. Generating synthetic data with known parameters
3. Running TAS-AI autonomous loop to recover parameters
4. Using MCMC for posterior inference

True parameters:
    J1 = 1.5 meV   (nearest-neighbor, FM)
    J2 = 0.2 meV   (next-nearest-neighbor, weak FM)
    D  = 0.1 meV   (easy-axis anisotropy)

Measurement zone: (H, H, 0) with H ∈ [0, 0.5] r.l.u.
Energy range: 0-8 meV

Usage:
    python parameter_estimation.py
    python parameter_estimation.py --n-iterations 30 --parallel
"""

import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional
import time
import argparse

from tasai.sunny import (
    SquareLatticeFM,
    generate_HHL_path,
    simulate_measurement
)
from tasai.inference import MCMCRunner
from tasai.core import Forecaster


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class ExperimentConfig:
    """Configuration for parameter estimation experiment."""
    # True parameters (what we're trying to find)
    true_J1: float = 1.5   # meV
    true_J2: float = 0.2   # meV
    true_D: float = 0.1    # meV
    
    # Measurement range
    H_min: float = 0.0
    H_max: float = 0.5
    E_min: float = 0.5
    E_max: float = 8.0
    L: float = 0.0  # Fixed L value
    
    # Measurement parameters
    count_time: float = 60.0  # seconds
    count_rate: float = 100.0  # counts/sec at I=1
    
    # MCMC parameters
    mcmc_burn: int = 200
    mcmc_steps: int = 200
    mcmc_pop: int = 8
    parallel: bool = False
    
    # Autonomous loop
    n_iterations: int = 20
    n_forecast: int = 3
    eta: float = 0.7
    
    # Initial points
    n_initial: int = 10


# =============================================================================
# Data structures
# =============================================================================

@dataclass
class Measurement:
    """A single inelastic neutron scattering measurement."""
    H: float
    K: float
    L: float
    E: float
    I: float
    sigma: float
    source: str  # 'initial', 'ai', 'user'
    timestamp: float


# =============================================================================
# Simple MCMC (for demo without full BUMPS)
# =============================================================================

class SimpleMCMC:
    """
    Simple Metropolis-Hastings MCMC for parameter estimation.
    
    For demonstration - in production use tasai.inference.MCMCRunner.
    """
    
    def __init__(self, model: SquareLatticeFM, 
                 burn: int = 200, steps: int = 200):
        self.model = model
        self.burn = burn
        self.steps = steps
        
        # Data
        self.H_data: Optional[np.ndarray] = None
        self.K_data: Optional[np.ndarray] = None
        self.L_data: Optional[np.ndarray] = None
        self.E_data: Optional[np.ndarray] = None
        self.I_data: Optional[np.ndarray] = None
        self.sigma_data: Optional[np.ndarray] = None
        
        # Proposal widths
        self.proposal_width = {
            'J1': 0.1,
            'J2': 0.05,
            'D': 0.02
        }
    
    def set_data(self, measurements: List[Measurement]):
        """Set measurement data."""
        self.H_data = np.array([m.H for m in measurements])
        self.K_data = np.array([m.K for m in measurements])
        self.L_data = np.array([m.L for m in measurements])
        self.E_data = np.array([m.E for m in measurements])
        self.I_data = np.array([m.I for m in measurements])
        self.sigma_data = np.array([m.sigma for m in measurements])
    
    def log_likelihood(self, params: np.ndarray) -> float:
        """Compute log likelihood."""
        # Set model parameters
        self.model.parameter_vector = params
        
        # Compute predicted intensities
        I_pred = self.model.compute_intensity_array(
            self.H_data, self.K_data, self.L_data, self.E_data
        )
        
        # Gaussian likelihood
        chi2 = np.sum(((self.I_data - I_pred) / self.sigma_data)**2)
        return -0.5 * chi2
    
    def log_prior(self, params: np.ndarray) -> float:
        """Log prior (uniform in bounds)."""
        self.model.parameter_vector = params
        return self.model.log_prior()
    
    def log_posterior(self, params: np.ndarray) -> float:
        """Log posterior = log likelihood + log prior."""
        lp = self.log_prior(params)
        if not np.isfinite(lp):
            return -np.inf
        return lp + self.log_likelihood(params)
    
    def run(self) -> np.ndarray:
        """Run MCMC and return chain."""
        n_params = self.model.n_free
        
        # Initialize from current parameters
        current = self.model.parameter_vector.copy()
        current_log_prob = self.log_posterior(current)
        
        # Storage
        chain = np.zeros((self.burn + self.steps, n_params))
        
        # Run chain
        n_accept = 0
        for i in range(self.burn + self.steps):
            # Propose new parameters
            proposal = current.copy()
            for j, p in enumerate(self.model.free_params):
                proposal[j] += np.random.normal(0, self.proposal_width[p])
            
            # Accept/reject
            proposal_log_prob = self.log_posterior(proposal)
            
            if np.log(np.random.random()) < proposal_log_prob - current_log_prob:
                current = proposal
                current_log_prob = proposal_log_prob
                n_accept += 1
            
            chain[i] = current
            
            # Adaptive proposal (during burn-in)
            if i < self.burn and i > 50 and i % 50 == 0:
                accept_rate = n_accept / (i + 1)
                scale = 2.0 if accept_rate > 0.4 else 0.5 if accept_rate < 0.2 else 1.0
                for p in self.proposal_width:
                    self.proposal_width[p] *= scale
        
        # Return post-burn-in samples
        return chain[self.burn:]


# =============================================================================
# Acquisition Function
# =============================================================================

class SpinWaveAcquisition:
    """
    Acquisition function for spin wave measurements.
    
    Focuses measurements where:
    1. Model predictions are sensitive to parameter changes
    2. We haven't measured yet (exploration)
    3. Near dispersion curves (where signal is)
    """
    
    def __init__(self, model: SquareLatticeFM, eta: float = 0.7):
        self.model = model
        self.eta = eta
    
    def score(self, candidates: np.ndarray, 
              measurements: List[Measurement],
              param_uncertainty: Dict[str, float]) -> np.ndarray:
        """
        Score candidate measurement points.
        
        Parameters
        ----------
        candidates : ndarray
            Array of (H, K, L, E) points, shape (n_candidates, 4)
        measurements : list
            Existing measurements
        param_uncertainty : dict
            Current parameter uncertainties
        
        Returns
        -------
        scores : ndarray
            Score for each candidate (higher = more valuable)
        """
        n_candidates = len(candidates)
        scores = np.zeros(n_candidates)
        
        # Measured points for novelty calculation
        measured_HE = np.array([[m.H, m.E] for m in measurements]) if measurements else np.zeros((0, 2))
        
        for i, (H, K, L, E) in enumerate(candidates):
            # 1. Predicted intensity (want to measure where there's signal)
            I_pred = self.model.compute_intensity(H, K, L, E)
            signal_score = np.log1p(I_pred * 100)  # Log scale
            
            # 2. Parameter sensitivity (numerical gradient)
            sensitivity = 0.0
            for p in self.model.free_params:
                orig = getattr(self.model, p)
                delta = param_uncertainty.get(p, 0.1) * 0.1
                
                setattr(self.model, p, orig + delta)
                I_plus = self.model.compute_intensity(H, K, L, E)
                
                setattr(self.model, p, orig - delta)
                I_minus = self.model.compute_intensity(H, K, L, E)
                
                setattr(self.model, p, orig)
                
                dI_dp = (I_plus - I_minus) / (2 * delta)
                sensitivity += (dI_dp * param_uncertainty.get(p, 0.1))**2
            
            sensitivity_score = np.sqrt(sensitivity)
            
            # 3. Novelty (distance from existing measurements)
            novelty = 1.0
            for H_m, E_m in measured_HE:
                dist = np.sqrt((H - H_m)**2 + ((E - E_m)/5)**2)
                novelty *= (1 - 0.5 * np.exp(-dist**2 / 0.01))
            
            # Combined score
            scores[i] = (signal_score * sensitivity_score * novelty) ** self.eta
        
        return scores


# =============================================================================
# Main Experiment
# =============================================================================

class ParameterEstimationExperiment:
    """
    Autonomous experiment for determining J1, J2, D parameters.
    """
    
    def __init__(self, config: ExperimentConfig):
        self.config = config
        
        # True model (nature)
        self.true_model = SquareLatticeFM(
            J1=config.true_J1,
            J2=config.true_J2,
            D=config.true_D
        )
        self.true_model._use_mock = True  # Use analytical for speed
        
        # Estimated model (our belief)
        self.est_model = SquareLatticeFM(
            J1=1.0,   # Initial guess
            J2=0.0,
            D=0.05
        )
        self.est_model._use_mock = True
        
        # Uncertainties
        self.uncertainties = {
            'J1': 1.0,
            'J2': 0.5,
            'D': 0.2
        }
        
        # Acquisition function
        self.acquisition = SpinWaveAcquisition(self.est_model, eta=config.eta)
        
        # State
        self.measurements: List[Measurement] = []
        self.iteration = 0
        
        # History
        self.param_history = {p: [] for p in ['J1', 'J2', 'D']}
        self.uncertainty_history = {p: [] for p in ['J1', 'J2', 'D']}
    
    def measure(self, H: float, E: float, source: str = 'ai') -> Measurement:
        """Simulate a measurement at (H, H, L, E)."""
        K = H  # HHL zone
        L = self.config.L
        
        I, sigma = simulate_measurement(
            self.true_model, H, K, L, E,
            count_time=self.config.count_time,
            count_rate=self.config.count_rate
        )
        
        m = Measurement(
            H=H, K=K, L=L, E=E,
            I=I, sigma=sigma,
            source=source,
            timestamp=time.time()
        )
        self.measurements.append(m)
        return m
    
    def generate_candidates(self, n_candidates: int = 200) -> np.ndarray:
        """Generate candidate measurement points."""
        H = np.random.uniform(self.config.H_min, self.config.H_max, n_candidates)
        E = np.random.uniform(self.config.E_min, self.config.E_max, n_candidates)
        K = H  # HHL zone
        L = np.full(n_candidates, self.config.L)
        
        return np.column_stack([H, K, L, E])
    
    def fit_model(self):
        """Fit model to current data using MCMC."""
        if len(self.measurements) < 5:
            return
        
        # Run MCMC
        mcmc = SimpleMCMC(self.est_model, burn=self.config.mcmc_burn, steps=self.config.mcmc_steps)
        mcmc.set_data(self.measurements)
        
        chain = mcmc.run()
        
        # Update estimates from posterior
        for i, p in enumerate(self.est_model.free_params):
            samples = chain[:, i]
            setattr(self.est_model, p, np.median(samples))
            self.uncertainties[p] = np.std(samples)
        
        # Record history
        for p in ['J1', 'J2', 'D']:
            self.param_history[p].append(getattr(self.est_model, p))
            self.uncertainty_history[p].append(self.uncertainties[p])
    
    def select_next_points(self, n_points: int = 3) -> List[Tuple[float, float]]:
        """Select next measurement points using acquisition function."""
        candidates = self.generate_candidates(200)
        
        selected = []
        temp_measurements = self.measurements.copy()
        
        for _ in range(n_points):
            scores = self.acquisition.score(candidates, temp_measurements, self.uncertainties)
            best_idx = np.argmax(scores)
            
            H, K, L, E = candidates[best_idx]
            selected.append((H, E))
            
            # Add fake measurement to avoid selecting same point
            temp_measurements.append(Measurement(H=H, K=K, L=L, E=E, I=0, sigma=1, source='temp', timestamp=0))
        
        return selected
    
    def add_initial_points(self):
        """Add initial measurements on a grid."""
        H_init = np.linspace(self.config.H_min + 0.05, self.config.H_max - 0.05, 5)
        E_init = np.linspace(self.config.E_min + 0.5, self.config.E_max - 0.5, 2)
        
        print(f"Adding {len(H_init) * len(E_init)} initial measurements...")
        for H in H_init:
            for E in E_init:
                m = self.measure(H, E, source='initial')
                print(f"  (H,H,0)=({H:.2f},{H:.2f},0) E={E:.1f}meV: I={m.I:.4f}±{m.sigma:.4f}")
    
    def run_iteration(self) -> Dict:
        """Run one iteration of autonomous loop."""
        self.iteration += 1
        
        # Fit model
        self.fit_model()
        
        # Select next points
        next_points = self.select_next_points(self.config.n_forecast)
        
        # Measure
        for H, E in next_points:
            self.measure(H, E, source='ai')
        
        return {
            'iteration': self.iteration,
            'n_measurements': len(self.measurements),
            'J1': self.est_model.J1,
            'J1_std': self.uncertainties['J1'],
            'J2': self.est_model.J2,
            'J2_std': self.uncertainties['J2'],
            'D': self.est_model.D,
            'D_std': self.uncertainties['D'],
            'next_points': next_points
        }
    
    def run(self) -> Dict:
        """Run full experiment."""
        print("=" * 70)
        print("Parameter Estimation: Square Lattice FM")
        print("=" * 70)
        print(f"True parameters: J1={self.config.true_J1}, J2={self.config.true_J2}, D={self.config.true_D}")
        print(f"Measurement zone: (H,H,0) with H∈[{self.config.H_min}, {self.config.H_max}]")
        print(f"Energy range: [{self.config.E_min}, {self.config.E_max}] meV")
        print("=" * 70)
        
        # Initial measurements
        self.add_initial_points()
        self.fit_model()
        
        print(f"\nInitial estimates:")
        print(f"  J1 = {self.est_model.J1:.3f} ± {self.uncertainties['J1']:.3f} meV (true: {self.config.true_J1})")
        print(f"  J2 = {self.est_model.J2:.3f} ± {self.uncertainties['J2']:.3f} meV (true: {self.config.true_J2})")
        print(f"  D  = {self.est_model.D:.3f} ± {self.uncertainties['D']:.3f} meV (true: {self.config.true_D})")
        
        print(f"\nRunning {self.config.n_iterations} autonomous iterations...")
        print("-" * 70)
        
        for _ in range(self.config.n_iterations):
            results = self.run_iteration()
            
            print(f"Iter {results['iteration']:2d}: "
                  f"J1={results['J1']:.3f}±{results['J1_std']:.3f} | "
                  f"J2={results['J2']:.3f}±{results['J2_std']:.3f} | "
                  f"D={results['D']:.3f}±{results['D_std']:.3f} | "
                  f"{results['n_measurements']} pts")
        
        # Final results
        print("\n" + "=" * 70)
        print("FINAL RESULTS")
        print("=" * 70)
        print(f"  J1 = {self.est_model.J1:.4f} ± {self.uncertainties['J1']:.4f} meV (true: {self.config.true_J1}, error: {abs(self.est_model.J1 - self.config.true_J1):.4f})")
        print(f"  J2 = {self.est_model.J2:.4f} ± {self.uncertainties['J2']:.4f} meV (true: {self.config.true_J2}, error: {abs(self.est_model.J2 - self.config.true_J2):.4f})")
        print(f"  D  = {self.est_model.D:.4f} ± {self.uncertainties['D']:.4f} meV (true: {self.config.true_D}, error: {abs(self.est_model.D - self.config.true_D):.4f})")
        print(f"  Total measurements: {len(self.measurements)}")
        print("=" * 70)
        
        return {
            'J1': self.est_model.J1,
            'J2': self.est_model.J2,
            'D': self.est_model.D,
            'J1_std': self.uncertainties['J1'],
            'J2_std': self.uncertainties['J2'],
            'D_std': self.uncertainties['D'],
            'n_measurements': len(self.measurements),
            'measurements': self.measurements,
            'param_history': self.param_history,
            'uncertainty_history': self.uncertainty_history
        }
    
    def plot_results(self, save_path: str = None):
        """Plot experiment results."""
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        # 1. Data in H-E space
        ax = axes[0, 0]
        H_data = [m.H for m in self.measurements]
        E_data = [m.E for m in self.measurements]
        I_data = [m.I for m in self.measurements]
        colors = {'initial': 'purple', 'ai': 'blue'}
        for source, color in colors.items():
            mask = [m.source == source for m in self.measurements]
            H_s = [h for h, m in zip(H_data, mask) if m]
            E_s = [e for e, m in zip(E_data, mask) if m]
            I_s = [i for i, m in zip(I_data, mask) if m]
            sc = ax.scatter(H_s, E_s, c=I_s, cmap='viridis', s=50, 
                           vmin=0, vmax=max(I_data), label=source, alpha=0.7)
        
        # Overlay dispersion
        H_disp = np.linspace(0, 0.5, 100)
        E_disp = self.est_model.dispersion_HHL(H_disp)
        ax.plot(H_disp, E_disp, 'r-', lw=2, label='Fit dispersion')
        E_true = self.true_model.dispersion_HHL(H_disp)
        ax.plot(H_disp, E_true, 'k--', lw=2, alpha=0.5, label='True dispersion')
        
        ax.set_xlabel('[H H 0] (r.l.u.)')
        ax.set_ylabel('E (meV)')
        ax.set_title('Measurements in HHL Zone')
        ax.legend(loc='upper left')
        plt.colorbar(sc, ax=ax, label='Intensity')
        
        # 2. J1 convergence
        ax = axes[0, 1]
        iters = np.arange(1, len(self.param_history['J1']) + 1)
        ax.fill_between(iters, 
                        np.array(self.param_history['J1']) - np.array(self.uncertainty_history['J1']),
                        np.array(self.param_history['J1']) + np.array(self.uncertainty_history['J1']),
                        alpha=0.3, color='blue')
        ax.plot(iters, self.param_history['J1'], 'b-o', markersize=4)
        ax.axhline(self.config.true_J1, color='k', ls='--', label=f'True J1={self.config.true_J1}')
        ax.set_xlabel('Iteration')
        ax.set_ylabel('J1 (meV)')
        ax.set_title('J1 Convergence')
        ax.legend()
        
        # 3. J2 convergence
        ax = axes[0, 2]
        ax.fill_between(iters,
                        np.array(self.param_history['J2']) - np.array(self.uncertainty_history['J2']),
                        np.array(self.param_history['J2']) + np.array(self.uncertainty_history['J2']),
                        alpha=0.3, color='green')
        ax.plot(iters, self.param_history['J2'], 'g-o', markersize=4)
        ax.axhline(self.config.true_J2, color='k', ls='--', label=f'True J2={self.config.true_J2}')
        ax.set_xlabel('Iteration')
        ax.set_ylabel('J2 (meV)')
        ax.set_title('J2 Convergence')
        ax.legend()
        
        # 4. D convergence
        ax = axes[1, 0]
        ax.fill_between(iters,
                        np.array(self.param_history['D']) - np.array(self.uncertainty_history['D']),
                        np.array(self.param_history['D']) + np.array(self.uncertainty_history['D']),
                        alpha=0.3, color='red')
        ax.plot(iters, self.param_history['D'], 'r-o', markersize=4)
        ax.axhline(self.config.true_D, color='k', ls='--', label=f'True D={self.config.true_D}')
        ax.set_xlabel('Iteration')
        ax.set_ylabel('D (meV)')
        ax.set_title('D (Anisotropy) Convergence')
        ax.legend()
        
        # 5. Measurement histogram
        ax = axes[1, 1]
        ax.hist2d(H_data, E_data, bins=[20, 20], cmap='Blues')
        ax.set_xlabel('[H H 0] (r.l.u.)')
        ax.set_ylabel('E (meV)')
        ax.set_title('Measurement Distribution')
        
        # 6. Fit vs True dispersion
        ax = axes[1, 2]
        H_plot = np.linspace(0, 0.5, 100)
        ax.plot(H_plot, self.true_model.dispersion_HHL(H_plot), 'k-', lw=2, label='True')
        ax.plot(H_plot, self.est_model.dispersion_HHL(H_plot), 'b--', lw=2, label='Fit')
        ax.set_xlabel('[H H 0] (r.l.u.)')
        ax.set_ylabel('E (meV)')
        ax.set_title('Dispersion: True vs Fit')
        ax.legend()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"\nPlot saved to: {save_path}")
        
        plt.show()


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Parameter Estimation for Square Lattice FM")
    parser.add_argument("--J1", type=float, default=1.5, help="True J1 (meV)")
    parser.add_argument("--J2", type=float, default=0.2, help="True J2 (meV)")
    parser.add_argument("--D", type=float, default=0.1, help="True D (meV)")
    parser.add_argument("--n-iterations", type=int, default=20, help="Autonomous iterations")
    parser.add_argument("--n-forecast", type=int, default=3, help="Points per iteration")
    parser.add_argument("--parallel", action="store_true", help="Use parallel MCMC")
    parser.add_argument("--no-plot", action="store_true", help="Skip plotting")
    parser.add_argument("--save", type=str, help="Save plot to file")
    
    args = parser.parse_args()
    
    config = ExperimentConfig(
        true_J1=args.J1,
        true_J2=args.J2,
        true_D=args.D,
        n_iterations=args.n_iterations,
        n_forecast=args.n_forecast,
        parallel=args.parallel
    )
    
    np.random.seed(42)
    
    experiment = ParameterEstimationExperiment(config)
    results = experiment.run()
    
    if not args.no_plot:
        experiment.plot_results(save_path=args.save)
    
    return results


if __name__ == "__main__":
    main()
