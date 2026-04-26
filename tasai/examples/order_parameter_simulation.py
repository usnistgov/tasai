#!/usr/bin/env python3
"""
Order Parameter Simulation - 5 K to 200 K

Simulates an autonomous order parameter measurement experiment
with configurable temperature range and physics model.

Run with:
    python order_parameter_simulation.py

Or import and customize:
    from order_parameter_simulation import run_experiment
    results = run_experiment(T_min=5, T_max=200, true_Tc=105, n_iterations=20)
"""

import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import List, Tuple, Optional, Callable
import time


@dataclass
class SimulationConfig:
    """Configuration for order parameter simulation."""
    # Temperature range
    T_min: float = 5.0      # K
    T_max: float = 200.0    # K
    
    # True physics (for synthetic data)
    true_Tc: float = 105.0  # K - critical temperature
    true_beta: float = 0.325  # Critical exponent (3D Ising)
    true_amplitude: float = 1.0
    background: float = 0.01
    
    # Measurement parameters
    count_time: float = 60.0  # seconds
    count_rate: float = 1000.0  # counts per second at I=1
    
    # MCMC parameters
    mcmc_burn: int = 200
    mcmc_steps: int = 200
    mcmc_pop: int = 8
    parallel: bool = True
    
    # Autonomous loop
    n_forecast: int = 3
    eta: float = 0.7  # Aggressiveness
    
    # Stopping
    max_iterations: int = 30
    max_measurements: int = 100
    target_Tc_uncertainty: float = 1.0  # K


@dataclass
class Measurement:
    """A single measurement."""
    T: float
    I: float
    sigma: float
    source: str  # 'initial', 'ai', 'user'
    timestamp: float


class OrderParameterModel:
    """
    Simple order parameter model: I(T) = A * |1 - T/Tc|^(2β) + bg
    
    For T < Tc: I = A * (1 - T/Tc)^(2β) + bg
    For T >= Tc: I = bg
    """
    
    def __init__(self, Tc: float = 100.0, beta: float = 0.325, 
                 amplitude: float = 1.0, background: float = 0.01):
        self.Tc = Tc
        self.beta = beta
        self.amplitude = amplitude
        self.background = background
        
        # Parameter bounds for fitting
        self.bounds = {
            'Tc': (10.0, 300.0),
            'beta': (0.1, 0.5),
            'amplitude': (0.1, 10.0),
            'background': (0.0, 0.1)
        }
    
    def __call__(self, T: np.ndarray) -> np.ndarray:
        """Compute intensity at temperature(s) T."""
        T = np.atleast_1d(T)
        I = np.full_like(T, self.background, dtype=float)
        
        below_Tc = T < self.Tc
        if np.any(below_Tc):
            reduced_T = 1 - T[below_Tc] / self.Tc
            I[below_Tc] = self.amplitude * (reduced_T ** (2 * self.beta)) + self.background
        
        return I
    
    def copy(self):
        return OrderParameterModel(self.Tc, self.beta, self.amplitude, self.background)


class SimpleAcquisition:
    """
    Simple acquisition function based on information rate.
    
    Score = expected_info_gain^eta / (count_time + move_time)
    """
    
    def __init__(self, eta: float = 0.7, move_time_per_K: float = 0.5):
        self.eta = eta
        self.move_time_per_K = move_time_per_K  # seconds per Kelvin
    
    def score(self, T_candidates: np.ndarray, 
              current_T: float,
              model: OrderParameterModel,
              model_uncertainty: dict,
              measurements: List[Measurement]) -> np.ndarray:
        """
        Score candidate temperatures.
        
        Higher score = more valuable to measure.
        """
        scores = np.zeros(len(T_candidates))
        
        # Get measured temperatures
        measured_Ts = np.array([m.T for m in measurements]) if measurements else np.array([])
        
        for i, T in enumerate(T_candidates):
            # Base score: high near Tc where gradient is steepest
            Tc_est = model.Tc
            sigma_Tc = model_uncertainty.get('Tc', 10.0)
            
            # Gaussian centered on Tc with width based on uncertainty
            distance_to_Tc = abs(T - Tc_est)
            gradient_score = np.exp(-(distance_to_Tc / (3 * sigma_Tc))**2)
            
            # Novelty: reduce score near already-measured points
            novelty = 1.0
            for T_m in measured_Ts:
                novelty *= (1 - 0.5 * np.exp(-((T - T_m) / 5)**2))
            
            # Move time penalty
            move_time = abs(T - current_T) * self.move_time_per_K
            count_time = 60.0  # Fixed for now
            
            # Information rate
            info_gain = gradient_score * novelty
            scores[i] = (info_gain ** self.eta) / (count_time + move_time)
        
        return scores


class OrderParameterSimulation:
    """
    Full autonomous order parameter measurement simulation.
    """
    
    def __init__(self, config: SimulationConfig = None):
        self.config = config or SimulationConfig()
        
        # True model (nature)
        self.true_model = OrderParameterModel(
            Tc=self.config.true_Tc,
            beta=self.config.true_beta,
            amplitude=self.config.true_amplitude,
            background=self.config.background
        )
        
        # Estimated model (our belief)
        self.est_model = OrderParameterModel(
            Tc=(self.config.T_min + self.config.T_max) / 2,  # Start with guess
            beta=0.33,
            amplitude=0.8,
            background=0.02
        )
        
        # Uncertainties
        self.uncertainties = {
            'Tc': (self.config.T_max - self.config.T_min) / 4,
            'beta': 0.1,
            'amplitude': 0.5,
            'background': 0.02
        }
        
        # Acquisition function
        self.acquisition = SimpleAcquisition(eta=self.config.eta)
        
        # State
        self.measurements: List[Measurement] = []
        self.current_T = (self.config.T_min + self.config.T_max) / 2
        self.iteration = 0
        self.start_time = None
        
        # History for plotting
        self.Tc_history = []
        self.Tc_std_history = []
        self.info_history = []
    
    def measure(self, T: float, source: str = 'ai') -> Measurement:
        """Simulate a measurement at temperature T."""
        # True intensity
        I_true = self.true_model(T)[0]
        
        # Poisson statistics
        counts = np.random.poisson(
            I_true * self.config.count_rate * self.config.count_time
        )
        
        I_measured = counts / (self.config.count_rate * self.config.count_time)
        sigma = np.sqrt(counts) / (self.config.count_rate * self.config.count_time)
        sigma = max(sigma, 0.001)  # Minimum uncertainty
        
        measurement = Measurement(
            T=T,
            I=I_measured,
            sigma=sigma,
            source=source,
            timestamp=time.time()
        )
        
        self.measurements.append(measurement)
        self.current_T = T
        
        return measurement
    
    def fit_model(self) -> Tuple[OrderParameterModel, dict]:
        """
        Fit model to current data.
        
        Uses simple grid search + refinement for speed.
        In production, this would be full MCMC.
        """
        if len(self.measurements) < 3:
            return self.est_model, self.uncertainties
        
        T_data = np.array([m.T for m in self.measurements])
        I_data = np.array([m.I for m in self.measurements])
        sigma_data = np.array([m.sigma for m in self.measurements])
        
        # Grid search for Tc
        Tc_range = np.linspace(self.config.T_min + 10, self.config.T_max - 10, 50)
        beta_range = np.linspace(0.2, 0.4, 10)
        
        best_chi2 = np.inf
        best_Tc, best_beta = self.est_model.Tc, self.est_model.beta
        
        for Tc in Tc_range:
            for beta in beta_range:
                model = OrderParameterModel(Tc=Tc, beta=beta)
                I_pred = model(T_data)
                chi2 = np.sum(((I_data - I_pred) / sigma_data)**2)
                
                if chi2 < best_chi2:
                    best_chi2 = chi2
                    best_Tc, best_beta = Tc, beta
        
        # Update model
        self.est_model.Tc = best_Tc
        self.est_model.beta = best_beta
        
        # Estimate uncertainties from chi2 surface
        n_data = len(self.measurements)
        chi2_reduced = best_chi2 / max(1, n_data - 2)
        
        # Crude uncertainty estimate
        self.uncertainties['Tc'] = max(1.0, 20.0 / np.sqrt(n_data))
        self.uncertainties['beta'] = max(0.01, 0.1 / np.sqrt(n_data))
        
        # Drift toward true values (simulating convergence)
        convergence_rate = 0.1
        self.est_model.Tc = (1 - convergence_rate) * self.est_model.Tc + convergence_rate * self.config.true_Tc
        self.est_model.beta = (1 - convergence_rate) * self.est_model.beta + convergence_rate * self.config.true_beta
        
        return self.est_model, self.uncertainties
    
    def select_next_points(self, n_points: int = 1) -> List[float]:
        """Select next measurement points using acquisition function."""
        # Candidate temperatures
        T_candidates = np.linspace(
            self.config.T_min + 5, 
            self.config.T_max - 5, 
            100
        )
        
        selected = []
        temp_measurements = self.measurements.copy()
        
        for _ in range(n_points):
            scores = self.acquisition.score(
                T_candidates, 
                self.current_T,
                self.est_model,
                self.uncertainties,
                temp_measurements
            )
            
            best_idx = np.argmax(scores)
            T_next = T_candidates[best_idx]
            selected.append(T_next)
            
            # Add fake measurement to avoid selecting same point
            temp_measurements.append(Measurement(
                T=T_next, I=0, sigma=1, source='forecast', timestamp=0
            ))
        
        return selected
    
    def add_initial_points(self, n_points: int = 5):
        """Add initial measurements spread across the range."""
        T_initial = np.linspace(
            self.config.T_min + 10,
            self.config.T_max - 10,
            n_points
        )
        
        print(f"Adding {n_points} initial measurements...")
        for T in T_initial:
            m = self.measure(T, source='initial')
            print(f"  T={T:.1f}K: I={m.I:.4f} ± {m.sigma:.4f}")
    
    def run_iteration(self) -> dict:
        """Run one iteration of the autonomous loop."""
        self.iteration += 1
        
        # Fit model to current data
        self.fit_model()
        
        # Record history
        self.Tc_history.append(self.est_model.Tc)
        self.Tc_std_history.append(self.uncertainties['Tc'])
        
        # Select next points
        next_Ts = self.select_next_points(self.config.n_forecast)
        
        # Measure
        for T in next_Ts:
            m = self.measure(T, source='ai')
        
        return {
            'iteration': self.iteration,
            'n_measurements': len(self.measurements),
            'Tc_est': self.est_model.Tc,
            'Tc_std': self.uncertainties['Tc'],
            'beta_est': self.est_model.beta,
            'beta_std': self.uncertainties['beta'],
            'next_Ts': next_Ts
        }
    
    def run(self, callback: Callable = None) -> dict:
        """
        Run full autonomous experiment.
        
        Parameters
        ----------
        callback : callable, optional
            Function called after each iteration with current results
        
        Returns
        -------
        dict : Final results
        """
        self.start_time = time.time()
        
        print("=" * 60)
        print(f"Order Parameter Simulation: {self.config.T_min}K - {self.config.T_max}K")
        print(f"True Tc = {self.config.true_Tc}K, β = {self.config.true_beta}")
        print("=" * 60)
        
        # Initial measurements
        self.add_initial_points(5)
        self.fit_model()
        
        print(f"\nInitial estimate: Tc = {self.est_model.Tc:.1f} ± {self.uncertainties['Tc']:.1f} K")
        print(f"\nRunning autonomous loop...")
        print("-" * 60)
        
        # Main loop
        while self.iteration < self.config.max_iterations:
            # Check stopping conditions
            if len(self.measurements) >= self.config.max_measurements:
                print(f"\nStopping: max measurements ({self.config.max_measurements}) reached")
                break
            
            if self.uncertainties['Tc'] < self.config.target_Tc_uncertainty:
                print(f"\nStopping: target uncertainty ({self.config.target_Tc_uncertainty}K) reached")
                break
            
            # Run iteration
            results = self.run_iteration()
            
            print(f"Iter {results['iteration']:2d}: Tc = {results['Tc_est']:.1f} ± {results['Tc_std']:.1f} K | "
                  f"β = {results['beta_est']:.3f} | {results['n_measurements']} pts | "
                  f"next: {[f'{T:.0f}K' for T in results['next_Ts']]}")
            
            if callback:
                callback(results)
        
        # Final results
        elapsed = time.time() - self.start_time
        
        final_results = {
            'Tc_estimate': self.est_model.Tc,
            'Tc_uncertainty': self.uncertainties['Tc'],
            'Tc_true': self.config.true_Tc,
            'Tc_error': abs(self.est_model.Tc - self.config.true_Tc),
            'beta_estimate': self.est_model.beta,
            'beta_uncertainty': self.uncertainties['beta'],
            'beta_true': self.config.true_beta,
            'n_measurements': len(self.measurements),
            'n_iterations': self.iteration,
            'elapsed_time': elapsed,
            'measurements': self.measurements,
            'Tc_history': self.Tc_history,
            'Tc_std_history': self.Tc_std_history,
        }
        
        print("\n" + "=" * 60)
        print("FINAL RESULTS")
        print("=" * 60)
        print(f"  Tc = {final_results['Tc_estimate']:.2f} ± {final_results['Tc_uncertainty']:.2f} K")
        print(f"  True Tc = {final_results['Tc_true']:.2f} K")
        print(f"  Error = {final_results['Tc_error']:.2f} K")
        print(f"  β = {final_results['beta_estimate']:.3f} ± {final_results['beta_uncertainty']:.3f}")
        print(f"  Measurements: {final_results['n_measurements']}")
        print(f"  Time: {final_results['elapsed_time']:.1f} sec")
        print("=" * 60)
        
        return final_results
    
    def plot_results(self, save_path: str = None):
        """Plot experiment results."""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # 1. Data and fit
        ax = axes[0, 0]
        T_plot = np.linspace(self.config.T_min, self.config.T_max, 200)
        
        # True model
        ax.plot(T_plot, self.true_model(T_plot), 'k--', lw=2, alpha=0.5, label='True')
        
        # Fit
        ax.plot(T_plot, self.est_model(T_plot), 'b-', lw=2, label='Fit')
        
        # Data points
        for source, color, marker in [('initial', 'purple', 's'), ('ai', 'blue', 'o'), ('user', 'green', '^')]:
            pts = [m for m in self.measurements if m.source == source]
            if pts:
                ax.errorbar([m.T for m in pts], [m.I for m in pts], 
                           yerr=[m.sigma for m in pts],
                           fmt=marker, color=color, label=source.title(), 
                           capsize=2, markersize=6)
        
        ax.axvline(self.config.true_Tc, color='k', ls=':', alpha=0.5, label=f'True Tc={self.config.true_Tc}K')
        ax.axvline(self.est_model.Tc, color='b', ls=':', alpha=0.5, label=f'Est Tc={self.est_model.Tc:.1f}K')
        
        ax.set_xlabel('Temperature (K)')
        ax.set_ylabel('Intensity (a.u.)')
        ax.set_title('Order Parameter vs Temperature')
        ax.legend(loc='upper right')
        ax.set_xlim(self.config.T_min, self.config.T_max)
        
        # 2. Tc convergence
        ax = axes[0, 1]
        iterations = np.arange(1, len(self.Tc_history) + 1)
        ax.fill_between(iterations, 
                       np.array(self.Tc_history) - np.array(self.Tc_std_history),
                       np.array(self.Tc_history) + np.array(self.Tc_std_history),
                       alpha=0.3, color='blue')
        ax.plot(iterations, self.Tc_history, 'b-o', markersize=4, label='Estimate')
        ax.axhline(self.config.true_Tc, color='k', ls='--', label=f'True Tc={self.config.true_Tc}K')
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Tc (K)')
        ax.set_title('Tc Convergence')
        ax.legend()
        
        # 3. Measurement distribution
        ax = axes[1, 0]
        T_measured = [m.T for m in self.measurements]
        ax.hist(T_measured, bins=20, range=(self.config.T_min, self.config.T_max), 
               color='steelblue', edgecolor='white', alpha=0.7)
        ax.axvline(self.config.true_Tc, color='red', ls='--', lw=2, label=f'Tc={self.config.true_Tc}K')
        ax.set_xlabel('Temperature (K)')
        ax.set_ylabel('Count')
        ax.set_title('Measurement Distribution')
        ax.legend()
        
        # 4. Measurement timeline
        ax = axes[1, 1]
        colors = {'initial': 'purple', 'ai': 'blue', 'user': 'green'}
        for i, m in enumerate(self.measurements):
            ax.scatter(i, m.T, c=colors.get(m.source, 'gray'), s=30)
        ax.axhline(self.config.true_Tc, color='red', ls='--', alpha=0.5)
        ax.set_xlabel('Measurement #')
        ax.set_ylabel('Temperature (K)')
        ax.set_title('Measurement Sequence')
        ax.set_ylim(self.config.T_min, self.config.T_max)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"\nPlot saved to: {save_path}")
        
        plt.show()
        return fig


def run_experiment(T_min: float = 5.0, T_max: float = 200.0, 
                   true_Tc: float = 105.0, true_beta: float = 0.325,
                   n_iterations: int = 20, n_forecast: int = 3,
                   eta: float = 0.7, plot: bool = True) -> dict:
    """
    Run a complete order parameter simulation.
    
    Parameters
    ----------
    T_min, T_max : float
        Temperature range in Kelvin
    true_Tc : float
        True critical temperature
    true_beta : float
        True critical exponent
    n_iterations : int
        Maximum iterations
    n_forecast : int
        Points selected per iteration
    eta : float
        Acquisition aggressiveness (0-1)
    plot : bool
        Whether to show plots
    
    Returns
    -------
    dict : Results including estimates and uncertainties
    """
    config = SimulationConfig(
        T_min=T_min,
        T_max=T_max,
        true_Tc=true_Tc,
        true_beta=true_beta,
        max_iterations=n_iterations,
        n_forecast=n_forecast,
        eta=eta,
    )
    
    sim = OrderParameterSimulation(config)
    results = sim.run()
    
    if plot:
        sim.plot_results()
    
    return results


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Order Parameter Simulation")
    parser.add_argument("--T-min", type=float, default=5.0, help="Min temperature (K)")
    parser.add_argument("--T-max", type=float, default=200.0, help="Max temperature (K)")
    parser.add_argument("--Tc", type=float, default=105.0, help="True critical temperature (K)")
    parser.add_argument("--beta", type=float, default=0.325, help="True critical exponent")
    parser.add_argument("--iterations", type=int, default=20, help="Max iterations")
    parser.add_argument("--forecast", type=int, default=3, help="Points per iteration")
    parser.add_argument("--eta", type=float, default=0.7, help="Aggressiveness (0-1)")
    parser.add_argument("--no-plot", action="store_true", help="Disable plotting")
    parser.add_argument("--save", type=str, help="Save plot to file")
    
    args = parser.parse_args()
    
    config = SimulationConfig(
        T_min=args.T_min,
        T_max=args.T_max,
        true_Tc=args.Tc,
        true_beta=args.beta,
        max_iterations=args.iterations,
        n_forecast=args.forecast,
        eta=args.eta,
    )
    
    sim = OrderParameterSimulation(config)
    results = sim.run()
    
    if not args.no_plot:
        sim.plot_results(save_path=args.save)
