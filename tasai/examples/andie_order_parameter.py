#!/usr/bin/env python3
"""
ANDiE-Style Autonomous Phase Transition Analysis

This example demonstrates:
1. Physics-informed acquisition (discriminating between Ising, Weiss, first-order models)
2. Swappable physics models
3. Swappable acquisition functions
4. Full autonomous measurement loop

The scenario: We're measuring a magnetic Bragg peak intensity as a function
of temperature to determine the nature of a phase transition.

True physics: 3D Ising transition at Tc = 150 K with β = 0.325
Goal: Autonomously discover Tc and determine universality class
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple

from tasai.physics.base import PhysicsModel, Parameter
from tasai.physics.order_parameter import (
    IsingModel, WeissModel, FirstOrderModel, PowerLawModel,
    create_andie_ensemble, generate_synthetic_data
)
from tasai.core.acquisition import (
    AcquisitionFunction, HHAcquisition, UncertaintyAcquisition,
    ANDiEAcquisition, CompositeAcquisition
)
from tasai.inference.mcmc import MCMCRunner, compute_bayes_factor


# =============================================================================
# Simulated measurement point (for order parameter, T is the variable)
# =============================================================================

class TemperaturePoint:
    """A temperature measurement point."""
    def __init__(self, T: float, count_time: float = 60.0):
        self.h = 0.0  # Dummy for order parameter models
        self.k = 0.0
        self.l = 0.0
        self.E = T  # Temperature stored in E coordinate
        self.count_time = count_time
        
    @property
    def T(self):
        return self.E
    
    def __repr__(self):
        return f"T={self.T:.1f}K"


class SimulatedMeasurement:
    """Simulated measurement result."""
    def __init__(self, point: TemperaturePoint, intensity: float, uncertainty: float):
        self.point = point
        self.intensity = intensity
        self.uncertainty = uncertainty


# =============================================================================
# Main autonomous experiment class
# =============================================================================

class AutonomousOrderParameter:
    """
    Autonomous order parameter measurement experiment.
    
    Demonstrates swappable components:
    - Physics models: IsingModel, WeissModel, etc.
    - Acquisition functions: ANDiE, HH, Uncertainty, Composite
    """
    
    def __init__(self,
                 true_model: PhysicsModel,
                 hypothesis_models: List[PhysicsModel],
                 acquisition: AcquisitionFunction,
                 T_range: Tuple[float, float] = (50, 200),
                 noise_level: float = 0.05):
        """
        Initialize experiment.
        
        Parameters
        ----------
        true_model : PhysicsModel
            The "true" physics (used to generate simulated data)
        hypothesis_models : list of PhysicsModel
            Competing models to discriminate between
        acquisition : AcquisitionFunction
            Acquisition function to select next measurements
        T_range : tuple
            Temperature range to explore
        noise_level : float
            Relative noise level for simulated measurements
        """
        self.true_model = true_model
        self.hypothesis_models = hypothesis_models
        self.acquisition = acquisition
        self.T_range = T_range
        self.noise_level = noise_level
        
        # Measurement history
        self.measurements: List[SimulatedMeasurement] = []
        self.T_measured: List[float] = []
        self.I_measured: List[float] = []
        self.sigma_measured: List[float] = []
        
        # Model weights (for ANDiE-style)
        self.model_weights = np.ones(len(hypothesis_models)) / len(hypothesis_models)
    
    def measure(self, T: float, count_time: float = 60.0) -> SimulatedMeasurement:
        """Simulate a measurement at temperature T."""
        point = TemperaturePoint(T, count_time)
        
        # Get true intensity
        I_true = self.true_model.compute_intensity(0, 0, 0, T)
        
        # Add noise (Poisson-like)
        sigma = self.noise_level * np.sqrt(I_true + 1)
        I_obs = max(0, I_true + np.random.normal(0, sigma))
        
        result = SimulatedMeasurement(point, I_obs, sigma)
        
        # Record
        self.measurements.append(result)
        self.T_measured.append(T)
        self.I_measured.append(I_obs)
        self.sigma_measured.append(sigma)
        
        return result
    
    def get_candidates(self, n_candidates: int = 50) -> List[TemperaturePoint]:
        """Generate candidate measurement points."""
        # Uniform grid with some randomness
        T_candidates = np.linspace(self.T_range[0], self.T_range[1], n_candidates)
        T_candidates += np.random.randn(n_candidates) * 2  # Add jitter
        T_candidates = np.clip(T_candidates, self.T_range[0], self.T_range[1])
        
        return [TemperaturePoint(T) for T in T_candidates]
    
    def run_inference(self, model: PhysicsModel) -> np.ndarray:
        """Run MCMC inference for a single model."""
        runner = MCMCRunner(model, burn=300, steps=200, walkers=8)
        
        # Convert data to arrays
        h = np.zeros(len(self.T_measured))
        k = np.zeros(len(self.T_measured))
        l = np.zeros(len(self.T_measured))
        E = np.array(self.T_measured)
        I = np.array(self.I_measured)
        sigma = np.array(self.sigma_measured)
        
        chain = runner.run(h, k, l, E, I, sigma)
        return chain
    
    def update_model_weights(self):
        """Update model weights based on current data (Bayesian model comparison)."""
        if len(self.measurements) < 3:
            return
        
        # Convert data
        T = np.array(self.T_measured)
        I = np.array(self.I_measured)
        sigma = np.array(self.sigma_measured)
        h = k = l = np.zeros_like(T)
        
        # Compute log likelihood for each model at MAP parameters
        log_likes = []
        
        for model in self.hypothesis_models:
            # Quick inference
            runner = MCMCRunner(model, burn=100, steps=50, walkers=4)
            chain = runner.run(h, k, l, T, I, sigma)
            
            # Use mean parameters
            mean_params = np.mean(chain, axis=0)
            model.set_free_values(mean_params)
            
            ll = model.log_likelihood(h, k, l, T, I, sigma)
            log_likes.append(ll)
        
        # Convert to weights (softmax-like)
        log_likes = np.array(log_likes)
        log_likes -= np.max(log_likes)  # Normalize
        weights = np.exp(log_likes)
        self.model_weights = weights / np.sum(weights)
        
        # Update ANDiE acquisition if applicable
        if isinstance(self.acquisition, ANDiEAcquisition):
            self.acquisition.weights = self.model_weights
    
    def select_next_temperature(self) -> float:
        """Use acquisition function to select next measurement temperature."""
        candidates = self.get_candidates()
        
        # Get posterior samples from best current model
        best_model_idx = np.argmax(self.model_weights)
        best_model = self.hypothesis_models[best_model_idx]
        
        if len(self.measurements) > 0:
            chain = self.run_inference(best_model)
        else:
            chain = best_model.sample_prior(100)
        
        # Evaluate acquisition function
        if isinstance(self.acquisition, ANDiEAcquisition):
            # ANDiE doesn't need posterior samples
            selected = self.acquisition.select_next(candidates, None, n_select=1)
        else:
            selected = self.acquisition.select_next(candidates, chain, n_select=1)
        
        return selected[0].T
    
    def run(self, 
            n_iterations: int = 20,
            n_initial: int = 5,
            verbose: bool = True) -> dict:
        """
        Run autonomous experiment.
        
        Parameters
        ----------
        n_iterations : int
            Number of autonomous iterations
        n_initial : int
            Number of initial random measurements
        verbose : bool
            Print progress
        
        Returns
        -------
        dict
            Results including final model weights, parameter estimates, etc.
        """
        if verbose:
            print("=" * 60)
            print("Autonomous Order Parameter Measurement")
            print("=" * 60)
            print(f"True model: {self.true_model}")
            print(f"Hypothesis models: {[type(m).__name__ for m in self.hypothesis_models]}")
            print(f"Acquisition function: {type(self.acquisition).__name__}")
            print()
        
        # Initial measurements (random)
        if verbose:
            print(f"Taking {n_initial} initial measurements...")
        
        initial_Ts = np.random.uniform(self.T_range[0], self.T_range[1], n_initial)
        for T in initial_Ts:
            result = self.measure(T)
            if verbose:
                print(f"  T={T:.1f}K: I={result.intensity:.1f}±{result.uncertainty:.1f}")
        
        # Autonomous loop
        if verbose:
            print(f"\nStarting autonomous loop ({n_iterations} iterations)...")
        
        for i in range(n_iterations):
            # Update model weights
            self.update_model_weights()
            
            # Select next temperature
            T_next = self.select_next_temperature()
            
            # Measure
            result = self.measure(T_next)
            
            if verbose:
                weights_str = ", ".join([f"{type(m).__name__}:{w:.2f}" 
                                        for m, w in zip(self.hypothesis_models, self.model_weights)])
                print(f"  [{i+1}/{n_iterations}] T={T_next:.1f}K: I={result.intensity:.1f} | {weights_str}")
        
        # Final analysis
        if verbose:
            print("\n" + "=" * 60)
            print("Final Results")
            print("=" * 60)
        
        # Get best model
        best_idx = np.argmax(self.model_weights)
        best_model = self.hypothesis_models[best_idx]
        
        # Run final inference
        final_chain = self.run_inference(best_model)
        
        # Get parameter estimates
        runner = MCMCRunner(best_model)
        runner.chain = final_chain
        summary = runner.get_summary()
        
        if verbose:
            print(f"\nBest model: {type(best_model).__name__} (weight: {self.model_weights[best_idx]:.3f})")
            print("\nModel weights:")
            for m, w in zip(self.hypothesis_models, self.model_weights):
                print(f"  {type(m).__name__}: {w:.3f}")
            print("\nParameter estimates:")
            for name, stats in summary.items():
                print(f"  {name}: {stats['mean']:.3f} ± {stats['std']:.3f}")
            print(f"\nTrue T_c: {self.true_model.get_parameter('T_c').value:.1f} K")
        
        return {
            'best_model': best_model,
            'best_model_name': type(best_model).__name__,
            'model_weights': self.model_weights,
            'parameter_summary': summary,
            'n_measurements': len(self.measurements),
            'final_chain': final_chain
        }
    
    def plot_results(self, save_path: str = None):
        """Visualize experiment results."""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Plot 1: Data and model fits
        ax1 = axes[0, 0]
        T_plot = np.linspace(self.T_range[0], self.T_range[1], 200)
        
        # True model
        I_true = [self.true_model.compute_intensity(0, 0, 0, T) for T in T_plot]
        ax1.plot(T_plot, I_true, 'k-', linewidth=2, label='True', alpha=0.7)
        
        # Measured data
        ax1.errorbar(self.T_measured, self.I_measured, yerr=self.sigma_measured,
                    fmt='o', markersize=8, capsize=3, label='Measured')
        
        # Best fit models
        for i, (model, weight) in enumerate(zip(self.hypothesis_models, self.model_weights)):
            if weight > 0.01:  # Only show significant models
                I_model = [model.compute_intensity(0, 0, 0, T) for T in T_plot]
                ax1.plot(T_plot, I_model, '--', linewidth=1.5, alpha=0.7,
                        label=f'{type(model).__name__} ({weight:.2f})')
        
        ax1.set_xlabel('Temperature (K)')
        ax1.set_ylabel('Intensity')
        ax1.set_title('Order Parameter Measurement')
        ax1.legend()
        
        # Plot 2: Model weights over time
        ax2 = axes[0, 1]
        ax2.bar(range(len(self.hypothesis_models)), self.model_weights)
        ax2.set_xticks(range(len(self.hypothesis_models)))
        ax2.set_xticklabels([type(m).__name__ for m in self.hypothesis_models], rotation=45)
        ax2.set_ylabel('Model Weight')
        ax2.set_title('Final Model Weights')
        
        # Plot 3: Measurement sequence
        ax3 = axes[1, 0]
        ax3.scatter(range(len(self.T_measured)), self.T_measured, 
                   c=range(len(self.T_measured)), cmap='viridis', s=50)
        ax3.axhline(self.true_model.get_parameter('T_c').value, 
                   color='r', linestyle='--', label='True Tc')
        ax3.set_xlabel('Measurement Number')
        ax3.set_ylabel('Temperature (K)')
        ax3.set_title('Measurement Sequence')
        ax3.legend()
        
        # Plot 4: Histogram of temperatures
        ax4 = axes[1, 1]
        ax4.hist(self.T_measured, bins=20, edgecolor='black', alpha=0.7)
        ax4.axvline(self.true_model.get_parameter('T_c').value,
                   color='r', linestyle='--', linewidth=2, label='True Tc')
        ax4.set_xlabel('Temperature (K)')
        ax4.set_ylabel('Count')
        ax4.set_title('Temperature Distribution')
        ax4.legend()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150)
            print(f"Saved: {save_path}")
        
        plt.show()


# =============================================================================
# Demonstration functions
# =============================================================================

def demo_andie_acquisition():
    """Demonstrate ANDiE-style physics-informed acquisition."""
    print("\n" + "=" * 60)
    print("Demo 1: ANDiE-style Acquisition")
    print("=" * 60)
    
    # True physics: Ising model
    true_model = IsingModel(T_c=150.0, beta=0.325, intensity_scale=1000, background=10)
    
    # Competing hypotheses
    hypotheses = [
        IsingModel(T_c=145.0),  # Start with slightly wrong Tc
        WeissModel(T_c=145.0),
        FirstOrderModel(T_c=145.0),
    ]
    
    # ANDiE acquisition (physics-informed)
    acquisition = ANDiEAcquisition(hypotheses)
    
    # Run experiment
    experiment = AutonomousOrderParameter(
        true_model=true_model,
        hypothesis_models=hypotheses,
        acquisition=acquisition,
        T_range=(80, 220),
        noise_level=0.03
    )
    
    results = experiment.run(n_iterations=15, n_initial=5, verbose=True)
    experiment.plot_results('andie_demo.png')
    
    return results


def demo_hh_acquisition():
    """Demonstrate HH (information-rate) acquisition."""
    print("\n" + "=" * 60)
    print("Demo 2: HH Information-Rate Acquisition")
    print("=" * 60)
    
    # True physics: Ising model
    true_model = IsingModel(T_c=150.0, beta=0.325, intensity_scale=1000, background=10)
    
    # Use PowerLaw model for inference (more flexible)
    inference_model = PowerLawModel(T_c=145.0, beta=0.33)
    
    # HH acquisition targeting T_c and beta
    acquisition = HHAcquisition(
        physics_model=inference_model,
        poi_indices=[0, 3],  # T_c and beta
        eta=0.7
    )
    
    # Run experiment
    experiment = AutonomousOrderParameter(
        true_model=true_model,
        hypothesis_models=[inference_model],
        acquisition=acquisition,
        T_range=(80, 220),
        noise_level=0.03
    )
    
    results = experiment.run(n_iterations=15, n_initial=5, verbose=True)
    experiment.plot_results('hh_demo.png')
    
    return results


def demo_composite_acquisition():
    """Demonstrate composite acquisition combining multiple strategies."""
    print("\n" + "=" * 60)
    print("Demo 3: Composite Acquisition (ANDiE + Uncertainty)")
    print("=" * 60)
    
    # True physics
    true_model = IsingModel(T_c=150.0, beta=0.325, intensity_scale=1000, background=10)
    
    # Hypotheses
    hypotheses = [
        IsingModel(T_c=145.0),
        WeissModel(T_c=145.0),
        FirstOrderModel(T_c=145.0),
    ]
    
    # Composite: 70% ANDiE (model discrimination) + 30% uncertainty sampling
    andie_acq = ANDiEAcquisition(hypotheses)
    uncertainty_acq = UncertaintyAcquisition(physics_model=hypotheses[0])
    
    composite_acq = CompositeAcquisition(
        [andie_acq, uncertainty_acq],
        weights=[0.7, 0.3]
    )
    
    # Run
    experiment = AutonomousOrderParameter(
        true_model=true_model,
        hypothesis_models=hypotheses,
        acquisition=composite_acq,
        T_range=(80, 220),
        noise_level=0.03
    )
    
    results = experiment.run(n_iterations=15, n_initial=5, verbose=True)
    experiment.plot_results('composite_demo.png')
    
    return results


def compare_acquisition_functions():
    """Compare different acquisition functions on the same problem."""
    print("\n" + "=" * 60)
    print("Comparison: Different Acquisition Functions")
    print("=" * 60)
    
    np.random.seed(42)
    
    # True physics
    true_model = IsingModel(T_c=150.0, beta=0.325, intensity_scale=1000, background=10)
    
    results = {}
    
    # 1. Random acquisition (baseline)
    print("\n--- Random Acquisition ---")
    class RandomAcquisition(AcquisitionFunction):
        def evaluate(self, candidates, posterior, current_pos=None, **kwargs):
            from tasai.core.acquisition import AcquisitionResult
            return [AcquisitionResult(c, np.random.random()) for c in candidates]
        def select_next(self, candidates, posterior, n_select=1, current_pos=None, **kwargs):
            import random
            return random.sample(candidates, min(n_select, len(candidates)))
    
    hypotheses = [IsingModel(T_c=145.0), WeissModel(T_c=145.0)]
    exp = AutonomousOrderParameter(true_model, hypotheses, RandomAcquisition(), noise_level=0.03)
    results['Random'] = exp.run(n_iterations=15, n_initial=5, verbose=False)
    print(f"  Best: {results['Random']['best_model_name']}, weight: {max(results['Random']['model_weights']):.3f}")
    
    # 2. Uncertainty acquisition
    print("\n--- Uncertainty Acquisition ---")
    np.random.seed(42)
    hypotheses = [IsingModel(T_c=145.0), WeissModel(T_c=145.0)]
    exp = AutonomousOrderParameter(true_model, hypotheses, 
                                   UncertaintyAcquisition(hypotheses[0]), noise_level=0.03)
    results['Uncertainty'] = exp.run(n_iterations=15, n_initial=5, verbose=False)
    print(f"  Best: {results['Uncertainty']['best_model_name']}, weight: {max(results['Uncertainty']['model_weights']):.3f}")
    
    # 3. ANDiE acquisition
    print("\n--- ANDiE Acquisition ---")
    np.random.seed(42)
    hypotheses = [IsingModel(T_c=145.0), WeissModel(T_c=145.0)]
    exp = AutonomousOrderParameter(true_model, hypotheses,
                                   ANDiEAcquisition(hypotheses), noise_level=0.03)
    results['ANDiE'] = exp.run(n_iterations=15, n_initial=5, verbose=False)
    print(f"  Best: {results['ANDiE']['best_model_name']}, weight: {max(results['ANDiE']['model_weights']):.3f}")
    
    # Summary
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"{'Method':<15} {'Best Model':<15} {'Confidence':<12} {'Tc Estimate':<12}")
    print("-" * 54)
    
    for name, res in results.items():
        tc_est = res['parameter_summary'].get('T_c', {}).get('mean', np.nan)
        print(f"{name:<15} {res['best_model_name']:<15} {max(res['model_weights']):<12.3f} {tc_est:<12.1f}")
    
    print(f"\nTrue: IsingModel, Tc = 150.0 K")


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="ANDiE-style autonomous experiment demo")
    parser.add_argument("--demo", choices=['andie', 'hh', 'composite', 'compare', 'all'],
                       default='andie', help="Which demo to run")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    args = parser.parse_args()
    
    np.random.seed(args.seed)
    
    if args.demo == 'andie':
        demo_andie_acquisition()
    elif args.demo == 'hh':
        demo_hh_acquisition()
    elif args.demo == 'composite':
        demo_composite_acquisition()
    elif args.demo == 'compare':
        compare_acquisition_functions()
    elif args.demo == 'all':
        demo_andie_acquisition()
        demo_hh_acquisition()
        demo_composite_acquisition()
        compare_acquisition_functions()
