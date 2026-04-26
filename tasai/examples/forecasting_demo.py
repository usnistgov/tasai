#!/usr/bin/env python3
"""
Forecasting Demo - AutoREFL-Style Autonomous Experiment

This example demonstrates the key innovation from AutoREFL: forecasting.
Instead of running MCMC after every measurement, we select multiple
future measurements from a single posterior, updating via importance
sampling between selections.

Benefits:
- No MCMC downtime while instrument moves/counts
- Continuous measurement without waiting
- Maintains ~same performance for 1-5 forecasted points

Reference:
Hoogerheide & Heinrich, J. Appl. Cryst. 57, 1192–1204 (2024)
"""

import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import List, Tuple, Dict
import time

from tasai.physics.base import PhysicsModel, Parameter
from tasai.physics.order_parameter import IsingModel, PowerLawModel
from tasai.core.entropy import differential_entropy, information_rate
from tasai.core.forecast import (
    Forecaster, ForecastResult, ExperimentRunner,
    random_candidate_generator
)
from tasai.inference.mcmc import MCMCRunner


# =============================================================================
# Simple measurement point for order parameter experiments
# =============================================================================

@dataclass
class TemperaturePoint:
    """Measurement point for order parameter (temperature scan)."""
    T: float
    count_time: float = 60.0
    
    @property
    def h(self): return 0.0
    @property
    def k(self): return 0.0
    @property
    def l(self): return 0.0
    @property
    def E(self): return self.T  # Temperature as "E" coordinate


# =============================================================================
# Simple simulated instrument
# =============================================================================

class SimulatedOrderParameterInstrument:
    """Simulated instrument for order parameter measurements."""
    
    def __init__(self, 
                 true_model: PhysicsModel,
                 noise_level: float = 0.03,
                 count_rate: float = 1000.0):
        self.true_model = true_model
        self.noise_level = noise_level
        self.count_rate = count_rate
        self.position = None
    
    def measure(self, point: TemperaturePoint):
        """Simulate a measurement."""
        # Get true intensity
        I_true = self.true_model.compute_intensity(0, 0, 0, point.T)
        
        # Simulate counts
        counts = I_true * self.count_rate * point.count_time
        counts = np.random.poisson(max(1, int(counts)))
        
        # Convert back to rate
        I_measured = counts / (self.count_rate * point.count_time)
        uncertainty = np.sqrt(counts) / (self.count_rate * point.count_time)
        
        self.position = point
        
        # Return result-like object
        class Result:
            pass
        result = Result()
        result.intensity = I_measured
        result.uncertainty = max(uncertainty, 0.001)
        return result
    
    def estimate_move_time(self, from_pt, to_pt):
        """Estimate time to move between points."""
        if from_pt is None:
            return 30.0
        # Temperature ramp rate: ~2 K/minute
        delta_T = abs(to_pt.T - from_pt.T)
        return max(10.0, delta_T / 2.0 * 60.0)  # seconds


# =============================================================================
# Demo: Compare forecasting vs single-point selection
# =============================================================================

def demo_forecasting_comparison():
    """
    Compare forecasting (selecting multiple points) vs single-point selection.
    
    Shows that forecasting maintains similar performance while allowing
    continuous measurement without MCMC downtime.
    """
    print("=" * 70)
    print("Forecasting vs Single-Point Selection Comparison")
    print("=" * 70)
    
    # True physics: Ising model
    np.random.seed(42)
    true_model = IsingModel(T_c=150.0, beta=0.325, intensity_scale=1.0, background=0.01)
    
    # Model for inference (PowerLaw - more flexible)
    inference_model = PowerLawModel(T_c=145.0, beta=0.33, intensity_scale=1.0, background=0.01)
    
    # Temperature range
    T_range = (80.0, 220.0)
    
    # Run both strategies
    results = {}
    
    for strategy_name, n_forecast in [('Single', 1), ('Forecast-3', 3), ('Forecast-5', 5)]:
        print(f"\n--- Strategy: {strategy_name} (n_forecast={n_forecast}) ---")
        
        # Reset model
        inference_model.set_parameter('T_c', 145.0)
        inference_model.set_parameter('beta', 0.33)
        
        # Create fresh instrument
        instrument = SimulatedOrderParameterInstrument(true_model)
        
        # Candidate generator
        def generate_candidates():
            temps = np.random.uniform(T_range[0], T_range[1], 30)
            return [TemperaturePoint(T=t) for t in temps]
        
        # Create forecaster
        forecaster = Forecaster(
            physics_model=inference_model,
            candidate_generator=generate_candidates,
            poi_indices=[0, 3],  # T_c and beta
            eta=0.7,
            min_count_time=30.0,
            max_count_time=300.0,
            movement_time_func=instrument.estimate_move_time
        )
        
        # Measurement storage
        measurements = []
        T_measured = []
        I_measured = []
        sigma_measured = []
        
        # Initial measurements (random)
        print("  Initial measurements...")
        initial_temps = np.linspace(T_range[0], T_range[1], 5)
        for T in initial_temps:
            point = TemperaturePoint(T=T, count_time=60.0)
            result = instrument.measure(point)
            measurements.append({'T': T, 'I': result.intensity, 'sigma': result.uncertainty})
            T_measured.append(T)
            I_measured.append(result.intensity)
            sigma_measured.append(result.uncertainty)
        
        # Track information gain over time
        info_history = []
        time_history = []
        elapsed_time = 0.0
        
        # Run autonomous loop
        n_iterations = 5
        print(f"  Running {n_iterations} iterations with forecasting...")
        
        for iteration in range(n_iterations):
            # Run MCMC
            runner = MCMCRunner(inference_model, burn=200, steps=200, pop=4)
            h = np.zeros(len(T_measured))
            k = np.zeros(len(T_measured))
            l = np.zeros(len(T_measured))
            E = np.array(T_measured)
            I = np.array(I_measured)
            sigma = np.array(sigma_measured)
            
            posterior = runner.run(h, k, l, E, I, sigma)
            
            # Track entropy
            poi_samples = posterior[:, [0, 3]]  # T_c, beta
            current_entropy = differential_entropy(poi_samples)
            info_history.append(current_entropy)
            time_history.append(elapsed_time)
            
            # Forecast
            forecast_result = forecaster.forecast(
                posterior,
                n_forecast=n_forecast,
                current_position=instrument.position
            )
            
            # Execute forecasted measurements
            for fp in forecast_result.points:
                # Estimate move time
                if instrument.position:
                    move_time = instrument.estimate_move_time(instrument.position, fp.point)
                else:
                    move_time = 30.0
                elapsed_time += move_time + fp.count_time
                
                # Measure
                result = instrument.measure(fp.point)
                
                # Record
                T_measured.append(fp.point.T)
                I_measured.append(result.intensity)
                sigma_measured.append(result.uncertainty)
                measurements.append({
                    'T': fp.point.T, 
                    'I': result.intensity, 
                    'sigma': result.uncertainty
                })
            
            print(f"    Iter {iteration+1}: {len(forecast_result.points)} points, "
                  f"H={current_entropy:.3f}, elapsed={elapsed_time/60:.1f}min")
        
        # Final inference
        runner = MCMCRunner(inference_model, burn=300, steps=300, pop=6)
        posterior = runner.run(
            np.zeros(len(T_measured)), np.zeros(len(T_measured)),
            np.zeros(len(T_measured)), np.array(T_measured),
            np.array(I_measured), np.array(sigma_measured)
        )
        
        # Get parameter estimates
        T_c_samples = posterior[:, 0]
        beta_samples = posterior[:, 3]
        
        results[strategy_name] = {
            'T_measured': np.array(T_measured),
            'I_measured': np.array(I_measured),
            'n_measurements': len(T_measured),
            'elapsed_time': elapsed_time,
            'T_c_mean': np.mean(T_c_samples),
            'T_c_std': np.std(T_c_samples),
            'beta_mean': np.mean(beta_samples),
            'beta_std': np.std(beta_samples),
            'info_history': info_history,
            'time_history': time_history,
            'posterior': posterior
        }
        
        print(f"  Results: T_c = {np.mean(T_c_samples):.2f} ± {np.std(T_c_samples):.2f} K "
              f"(true: 150.0)")
        print(f"           β = {np.mean(beta_samples):.3f} ± {np.std(beta_samples):.3f} "
              f"(true: 0.325)")
    
    # Plot comparison
    plot_forecasting_comparison(results, true_model, T_range)
    
    return results


def plot_forecasting_comparison(results: Dict, true_model: PhysicsModel, T_range: Tuple):
    """Plot comparison of forecasting strategies."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    T_plot = np.linspace(T_range[0], T_range[1], 200)
    I_true = [true_model.compute_intensity(0, 0, 0, T) for T in T_plot]
    
    colors = {'Single': 'blue', 'Forecast-3': 'green', 'Forecast-5': 'red'}
    
    # Panel 1: Data and fits
    ax1 = axes[0, 0]
    ax1.plot(T_plot, I_true, 'k-', linewidth=2, label='True', alpha=0.7)
    
    for name, res in results.items():
        ax1.scatter(res['T_measured'], res['I_measured'], 
                   c=colors[name], alpha=0.6, s=30, label=f'{name} ({res["n_measurements"]} pts)')
    
    ax1.axvline(150.0, color='k', linestyle='--', alpha=0.5, label='True Tc')
    ax1.set_xlabel('Temperature (K)')
    ax1.set_ylabel('Intensity')
    ax1.set_title('Measurement Distribution')
    ax1.legend()
    
    # Panel 2: T_c estimates
    ax2 = axes[0, 1]
    names = list(results.keys())
    T_c_means = [results[n]['T_c_mean'] for n in names]
    T_c_stds = [results[n]['T_c_std'] for n in names]
    
    x = np.arange(len(names))
    ax2.bar(x, T_c_means, yerr=T_c_stds, color=[colors[n] for n in names], alpha=0.7, capsize=5)
    ax2.axhline(150.0, color='k', linestyle='--', label='True Tc')
    ax2.set_xticks(x)
    ax2.set_xticklabels(names)
    ax2.set_ylabel('Tc (K)')
    ax2.set_title('Critical Temperature Estimates')
    ax2.legend()
    
    # Panel 3: β estimates
    ax3 = axes[1, 0]
    beta_means = [results[n]['beta_mean'] for n in names]
    beta_stds = [results[n]['beta_std'] for n in names]
    
    ax3.bar(x, beta_means, yerr=beta_stds, color=[colors[n] for n in names], alpha=0.7, capsize=5)
    ax3.axhline(0.325, color='k', linestyle='--', label='True β')
    ax3.set_xticks(x)
    ax3.set_xticklabels(names)
    ax3.set_ylabel('β')
    ax3.set_title('Critical Exponent Estimates')
    ax3.legend()
    
    # Panel 4: Efficiency comparison
    ax4 = axes[1, 1]
    
    # Compute efficiency: error reduction per unit time
    for name in names:
        res = results[name]
        # Lower is better for uncertainty
        efficiency = 1.0 / (res['T_c_std'] * res['elapsed_time'] / 60)  # per minute
        ax4.bar(name, efficiency, color=colors[name], alpha=0.7)
    
    ax4.set_ylabel('Efficiency (1 / (σ × time))')
    ax4.set_title('Measurement Efficiency')
    
    plt.tight_layout()
    plt.savefig('forecasting_comparison.png', dpi=150)
    print("\nSaved: forecasting_comparison.png")
    plt.show()


# =============================================================================
# Demo: Full experiment with ExperimentRunner
# =============================================================================

def demo_experiment_runner():
    """
    Demonstrate the full ExperimentRunner with forecasting.
    """
    print("\n" + "=" * 70)
    print("Full Experiment Runner Demo")
    print("=" * 70)
    
    np.random.seed(123)
    
    # True physics
    true_model = IsingModel(T_c=150.0, beta=0.325, intensity_scale=1.0, background=0.01)
    
    # Model for inference
    inference_model = PowerLawModel(T_c=140.0, beta=0.35)
    
    # Instrument
    instrument = SimulatedOrderParameterInstrument(true_model)
    
    # Candidate generator
    def generate_candidates():
        temps = np.random.uniform(80.0, 220.0, 40)
        return [TemperaturePoint(T=t) for t in temps]
    
    # Create experiment runner
    runner = ExperimentRunner(
        physics_model=inference_model,
        instrument=instrument,
        candidate_generator=generate_candidates,
        poi_indices=[0, 3],  # T_c and beta
        mcmc_burn=200,
        mcmc_steps=200,
        mcmc_pop=4,
        n_forecast=3,  # Select 3 points per MCMC run
        eta=0.7
    )
    
    # Callback to track progress
    def progress_callback(runner, event, *args):
        if event == 'iteration':
            iteration, forecast_points = args
            print(f"  Completed iteration {iteration}")
    
    # Run experiment
    print("\nRunning autonomous experiment...")
    print("  - n_forecast = 3 (select 3 points per MCMC run)")
    print("  - poi = [T_c, β]")
    
    results = runner.run(
        max_iterations=4,
        initial_points=[TemperaturePoint(T=t) for t in [90, 120, 150, 180, 210]],
        callback=progress_callback
    )
    
    # Print results
    print(f"\nExperiment complete:")
    print(f"  Total measurements: {results['n_measurements']}")
    print(f"  Total iterations: {results['n_iterations']}")
    print(f"  Elapsed time: {results['elapsed_time']:.1f}s")
    
    # Get final parameter estimates
    posterior = results['posterior_samples']
    T_c_est = np.mean(posterior[:, 0])
    T_c_std = np.std(posterior[:, 0])
    beta_est = np.mean(posterior[:, 3])
    beta_std = np.std(posterior[:, 3])
    
    print(f"\nParameter estimates:")
    print(f"  T_c = {T_c_est:.2f} ± {T_c_std:.2f} K (true: 150.0)")
    print(f"  β = {beta_est:.3f} ± {beta_std:.3f} (true: 0.325)")
    
    return results


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Forecasting demonstration")
    parser.add_argument("--demo", choices=['comparison', 'runner', 'all'],
                       default='all', help="Which demo to run")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    args = parser.parse_args()
    
    np.random.seed(args.seed)
    
    if args.demo == 'comparison':
        demo_forecasting_comparison()
    elif args.demo == 'runner':
        demo_experiment_runner()
    elif args.demo == 'all':
        demo_forecasting_comparison()
        demo_experiment_runner()
