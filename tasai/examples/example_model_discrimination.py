#!/usr/bin/env python3
"""
Example: Model Discrimination - Does J2 matter?

This example demonstrates ANDiE-style model discrimination to determine
whether next-nearest-neighbor (J2) interactions are present in the data.

Competing models:
    Model A: NN only    H = -J1 Σ_<i,j> Si·Sj - D Σ_i (Sz)²
    Model B: J1-J2      H = -J1 Σ_<i,j> Si·Sj - J2 Σ_<<i,j>> Si·Sj - D Σ_i (Sz)²

The key insight: J2 affects dispersion curvature near zone boundary.
At H=0.25 (quarter BZ), NN and NNN models predict different energies.

Dispersion:
    NN:    ω(H) ∝ 2J1(1 - cos(2πH))
    J1-J2: ω(H) ∝ 2J1(1 - cos(2πH)) + 2J2(1 - cos(4πH))

At H=0.25:
    NN:    cos(2π·0.25) = 0,    cos(4π·0.25) = -1
    J1-J2: extra J2 contribution!

Run with:
    python example_model_discrimination.py

Or customize:
    python example_model_discrimination.py --J2 1.0 --iterations 25
"""

import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import List, Tuple, Dict
import time
import argparse

from tasai.sunny import SquareLatticeFM, NNOnlyModel


@dataclass
class Measurement:
    H: float
    L: float
    E: float
    I: float
    sigma: float


class ModelDiscriminator:
    """
    Bayesian model discrimination between NN-only and J1-J2 models.
    
    Uses evidence approximation (Laplace) to compute model weights.
    """
    
    def __init__(self, true_J1: float = 5.0, true_J2: float = 0.5, true_D: float = 0.1,
                 include_j2: bool = True):
        """
        Initialize discriminator.
        
        Parameters
        ----------
        true_J1, true_J2, true_D : float
            True model parameters
        include_j2 : bool
            If True, true model includes J2. If False, true model is NN-only.
        """
        self.include_j2 = include_j2
        
        # True model (nature)
        if include_j2:
            self.true_model = SquareLatticeFM(J1=true_J1, J2=true_J2, D=true_D)
        else:
            self.true_model = NNOnlyModel(J1=true_J1, D=true_D)
        
        # Competing models
        self.model_nn = NNOnlyModel(J1=4.0, D=0.1)  # 2 free params
        self.model_j1j2 = SquareLatticeFM(J1=4.0, J2=0.0, D=0.1)  # 3 free params
        
        # Model weights (prior: equal)
        self.weights = {'NN': 0.5, 'J1J2': 0.5}
        
        # Best-fit parameters for each model
        self.best_params = {
            'NN': {'J1': 4.0, 'D': 0.1},
            'J1J2': {'J1': 4.0, 'J2': 0.0, 'D': 0.1}
        }
        
        # Log-evidence for each model
        self.log_evidence = {'NN': 0.0, 'J1J2': 0.0}
        
        # Data
        self.measurements: List[Measurement] = []
        self.count_time = 30.0
        self.count_rate = 20.0
        
        # History
        self.weight_history = []
        self.evidence_history = []
    
    def measure(self, H: float, E: float) -> Measurement:
        """Take a measurement."""
        I, sigma = self.true_model.simulate_measurement(
            H,
            0,
            E,
            count_time=self.count_time,
            count_rate=self.count_rate
        )
        m = Measurement(H=H, L=0, E=E, I=I, sigma=sigma)
        self.measurements.append(m)
        return m
    
    def fit_models(self):
        """Fit both models to current data."""
        if len(self.measurements) < 3:
            return
        
        H_data = np.array([m.H for m in self.measurements])
        L_data = np.array([m.L for m in self.measurements])
        E_data = np.array([m.E for m in self.measurements])
        I_data = np.array([m.I for m in self.measurements])
        sigma_data = np.array([m.sigma for m in self.measurements])
        
        # Fit NN model (grid search over J1, D)
        best_chi2_nn = np.inf
        for J1 in np.linspace(2.0, 10.0, 20):
            for D in np.linspace(0.01, 0.3, 10):
                model = NNOnlyModel(J1=J1, D=D)
                chi2 = model.chi_squared(H_data, L_data, E_data, I_data, sigma_data)
                if chi2 < best_chi2_nn:
                    best_chi2_nn = chi2
                    self.best_params['NN'] = {'J1': J1, 'D': D}
        
        self.model_nn = NNOnlyModel(**self.best_params['NN'])
        
        # Fit J1-J2 model (grid search over J1, J2, D)
        best_chi2_j1j2 = np.inf
        for J1 in np.linspace(2.0, 10.0, 15):
            for J2 in np.linspace(-0.5, 2.0, 12):
                for D in np.linspace(0.01, 0.3, 8):
                    model = SquareLatticeFM(J1=J1, J2=J2, D=D)
                    chi2 = model.chi_squared(H_data, L_data, E_data, I_data, sigma_data)
                    if chi2 < best_chi2_j1j2:
                        best_chi2_j1j2 = chi2
                        self.best_params['J1J2'] = {'J1': J1, 'J2': J2, 'D': D}
        
        self.model_j1j2 = SquareLatticeFM(**self.best_params['J1J2'])
        
        n = len(self.measurements)

        # Compute log-evidence using the Akaike Information Criterion (AIC)
        # log(evidence) ≈ -0.5 * (χ² + 2k), where k is the parameter count.
        k_nn = 2  # J1, D
        k_j1j2 = 3  # J1, J2, D

        self.log_evidence['NN'] = -0.5 * (best_chi2_nn + 2 * k_nn)
        self.log_evidence['J1J2'] = -0.5 * (best_chi2_j1j2 + 2 * k_j1j2)
        
        # Update weights via Bayes factor
        log_bf = self.log_evidence['J1J2'] - self.log_evidence['NN']
        
        # Softmax to get weights
        max_log = max(self.log_evidence.values())
        total = np.exp(self.log_evidence['NN'] - max_log) + np.exp(self.log_evidence['J1J2'] - max_log)
        
        self.weights['NN'] = np.exp(self.log_evidence['NN'] - max_log) / total
        self.weights['J1J2'] = np.exp(self.log_evidence['J1J2'] - max_log) / total
        
        # Record history
        self.weight_history.append(self.weights.copy())
        self.evidence_history.append({
            'NN': best_chi2_nn / max(1, n - k_nn),
            'J1J2': best_chi2_j1j2 / max(1, n - k_j1j2)
        })
    
    def suggest_next_point(self) -> Tuple[float, float]:
        """
        Select next measurement to maximize model discrimination.
        
        Strategy: Measure where the two models disagree most.
        """
        H_candidates = np.linspace(0.05, 0.45, 20)
        
        best_score = -np.inf
        best_point = (0.25, 5.0)
        
        for H in H_candidates:
            # Get dispersion from both models
            E_nn = self.model_nn.dispersion(H, 0)
            E_j1j2 = self.model_j1j2.dispersion(H, 0)
            
            # Energy difference between models
            E_diff = abs(E_nn - E_j1j2)
            
            # Candidate energies: at both dispersions and in between
            E_candidates = np.array([E_nn, E_j1j2, (E_nn + E_j1j2) / 2])
            E_candidates = E_candidates[(E_candidates > 0.5) & (E_candidates < 15)]
            
            for E in E_candidates:
                # Predicted intensities
                I_nn = self.model_nn.intensity(H, 0, E)
                I_j1j2 = self.model_j1j2.intensity(H, 0, E)
                
                # Score: maximize intensity difference
                intensity_diff = abs(I_nn - I_j1j2)
                
                # Novelty factor
                novelty = 1.0
                for m in self.measurements:
                    dist = np.sqrt((H - m.H)**2 + (E - m.E)**2 / 10)
                    novelty *= (1 - 0.7 * np.exp(-dist**2 / 0.01))
                
                # Combined score
                score = intensity_diff * novelty * (1 + E_diff)
                
                if score > best_score:
                    best_score = score
                    best_point = (H, E)
        
        return best_point
    
    def add_initial_points(self, n_points: int = 6):
        """Add initial measurements."""
        print(f"Adding {n_points} initial measurements...")
        
        # Measure at strategic H values
        H_init = [0.1, 0.2, 0.25, 0.3, 0.35, 0.4][:n_points]
        
        for H in H_init:
            E = self.true_model.dispersion(H, 0)
            m = self.measure(H, E)
            print(f"  H={H:.2f}, E={E:.2f}: I={m.I:.4f}")
    
    def run(self, n_iterations: int = 20, points_per_iter: int = 2):
        """Run model discrimination experiment."""
        true_model_name = "J1-J2" if self.include_j2 else "NN-only"
        
        print("=" * 70)
        print("Model Discrimination: Does J2 matter?")
        print(f"True model: {true_model_name}")
        if self.include_j2:
            print(f"True params: J1={self.true_model.J1}, J2={self.true_model.J2}, D={self.true_model.D}")
        else:
            print(f"True params: J1={self.true_model.J1}, D={self.true_model.D}")
        print("=" * 70)
        
        # Initial measurements
        self.add_initial_points(6)
        self.fit_models()
        
        print(f"\nInitial weights: NN={self.weights['NN']:.1%}, J1-J2={self.weights['J1J2']:.1%}")
        print("\nRunning discrimination loop...")
        print("-" * 70)
        
        for iteration in range(n_iterations):
            # Select and measure discriminating points
            for _ in range(points_per_iter):
                H, E = self.suggest_next_point()
                self.measure(H, E)
            
            # Fit models and update weights
            self.fit_models()
            
            # Determine leading model
            if self.weights['J1J2'] > self.weights['NN']:
                leader = "J1-J2"
                confidence = self.weights['J1J2']
            else:
                leader = "NN"
                confidence = self.weights['NN']
            
            # Print progress
            print(f"Iter {iteration+1:2d}: "
                  f"NN={self.weights['NN']:.1%} J1-J2={self.weights['J1J2']:.1%} | "
                  f"χ²: {self.evidence_history[-1]['NN']:.2f} vs {self.evidence_history[-1]['J1J2']:.2f} | "
                  f"Leader: {leader} ({confidence:.1%})")
            
            # Check for decisive evidence (Kass & Raftery: >95% is "decisive")
            if max(self.weights.values()) > 0.95:
                print(f"\nDecisive evidence for {leader}!")
                break
        
        # Final results
        winner = "J1-J2" if self.weights['J1J2'] > self.weights['NN'] else "NN"
        correct = (winner == "J1-J2" and self.include_j2) or (winner == "NN" and not self.include_j2)
        
        print("\n" + "=" * 70)
        print("FINAL RESULTS")
        print("=" * 70)
        print(f"  Model weights:")
        print(f"    NN-only:  {self.weights['NN']:.1%}")
        print(f"    J1-J2:    {self.weights['J1J2']:.1%}")
        print(f"  Winner: {winner}")
        print(f"  Correct: {'✓ YES' if correct else '✗ NO'}")
        print()
        print(f"  Best NN params:    J1={self.best_params['NN']['J1']:.2f}, D={self.best_params['NN']['D']:.3f}")
        print(f"  Best J1-J2 params: J1={self.best_params['J1J2']['J1']:.2f}, J2={self.best_params['J1J2']['J2']:.2f}, D={self.best_params['J1J2']['D']:.3f}")
        print(f"  Measurements: {len(self.measurements)}")
        
        # Bayes factor interpretation
        bf = self.weights['J1J2'] / max(self.weights['NN'], 1e-10)
        if bf > 100:
            strength = "Decisive evidence for J2"
        elif bf > 10:
            strength = "Strong evidence for J2"
        elif bf > 3:
            strength = "Moderate evidence for J2"
        elif bf > 1:
            strength = "Weak evidence for J2"
        elif bf > 0.33:
            strength = "Inconclusive"
        elif bf > 0.1:
            strength = "Weak evidence against J2"
        elif bf > 0.01:
            strength = "Strong evidence against J2"
        else:
            strength = "Decisive evidence against J2"
        
        print(f"  Bayes factor (J1-J2 / NN): {bf:.2f}")
        print(f"  Interpretation: {strength}")
        print("=" * 70)
        
        return {
            'winner': winner,
            'correct': correct,
            'weights': self.weights,
            'bayes_factor': bf,
            'n_measurements': len(self.measurements)
        }
    
    def plot_results(self, save_path: str = None, show: bool = True):
        """Plot discrimination results."""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        panel_labels = ['(a)', '(b)', '(c)', '(d)']
        
        # 1. Dispersion comparison
        ax = axes[0, 0]
        ax.text(0.01, 0.95, panel_labels[0], transform=ax.transAxes,
                fontsize=12, fontweight='bold', va='top')
        H_plot = np.linspace(0, 0.5, 100)
        
        E_true = self.true_model.dispersion(H_plot, 0)
        E_nn = self.model_nn.dispersion(H_plot, 0)
        E_j1j2 = self.model_j1j2.dispersion(H_plot, 0)
        
        ax.plot(H_plot, E_true, 'k-', lw=3, label='True', alpha=0.8)
        ax.plot(H_plot, E_nn, 'b--', lw=2, label=f'NN (J1={self.best_params["NN"]["J1"]:.2f})')
        ax.plot(H_plot, E_j1j2, 'r:', lw=2, label=f'J1-J2 (J2={self.best_params["J1J2"]["J2"]:.2f})')
        
        # Data points
        ax.scatter([m.H for m in self.measurements], [m.E for m in self.measurements],
                  c='green', s=30, alpha=0.6, zorder=5, label='Data')
        
        ax.set_xlabel('[H H 0] (r.l.u.)')
        ax.set_ylabel('Energy (meV)')
        ax.set_title('Dispersion Comparison')
        ax.legend()
        ax.set_xlim(0, 0.5)
        
        # 2. Model weights evolution
        ax = axes[0, 1]
        ax.text(0.01, 0.95, panel_labels[1], transform=ax.transAxes,
                fontsize=12, fontweight='bold', va='top')
        iterations = np.arange(1, len(self.weight_history) + 1)
        
        nn_weights = [w['NN'] for w in self.weight_history]
        j1j2_weights = [w['J1J2'] for w in self.weight_history]
        
        ax.fill_between(iterations, 0, nn_weights, alpha=0.3, color='blue', label='NN')
        ax.fill_between(iterations, nn_weights, 1, alpha=0.3, color='red', label='J1-J2')
        ax.plot(iterations, nn_weights, 'b-', lw=2)
        ax.plot(iterations, j1j2_weights, 'r-', lw=2)
        ax.axhline(0.5, color='k', ls='--', alpha=0.3)
        ax.axhline(0.95, color='g', ls=':', alpha=0.5, label='95% threshold')
        
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Model Weight')
        ax.set_title('Model Weight Evolution')
        ax.legend(loc='center right')
        ax.set_ylim(0, 1)
        
        # 3. Chi-squared comparison
        ax = axes[1, 0]
        ax.text(0.01, 0.95, panel_labels[2], transform=ax.transAxes,
                fontsize=12, fontweight='bold', va='top')
        chi2_nn = [e['NN'] for e in self.evidence_history]
        chi2_j1j2 = [e['J1J2'] for e in self.evidence_history]
        
        ax.plot(iterations, chi2_nn, 'b-o', label='NN χ²/N', markersize=4)
        ax.plot(iterations, chi2_j1j2, 'r-s', label='J1-J2 χ²/N', markersize=4)
        
        ax.set_xlabel('Iteration')
        ax.set_ylabel('χ²/N')
        ax.set_title('Reduced Chi-Squared')
        ax.set_yscale('log')
        if chi2_nn and chi2_j1j2:
            min_val = min(min(chi2_nn), min(chi2_j1j2))
            max_val = max(max(chi2_nn), max(chi2_j1j2))
            ax.set_ylim(min_val * 0.8, max_val * 1.2)
        ax.legend()
        
        # 4. Where models differ
        ax = axes[1, 1]
        ax.text(0.01, 0.95, panel_labels[3], transform=ax.transAxes,
                fontsize=12, fontweight='bold', va='top')
        
        # Difference between models
        E_diff = np.abs(E_j1j2 - E_nn)
        ax.plot(H_plot, E_diff, 'purple', lw=2)
        ax.fill_between(H_plot, 0, E_diff, alpha=0.3, color='purple')
        
        # Mark measurement locations
        for m in self.measurements:
            ax.axvline(m.H, color='green', alpha=0.2, lw=0.5)
        
        ax.set_xlabel('[H H 0] (r.l.u.)')
        ax.set_ylabel('|E_J1J2 - E_NN| (meV)')
        ax.set_title('Model Discrimination Power\n(Purple = where models differ most)')
        ax.set_xlim(0, 0.5)
        
        # Add annotation about J2-sensitive shoulder (avoid implying global maximum)
        ax.annotate('J2-sensitive\nregion', xy=(0.25, E_diff[50]), 
                   xytext=(0.35, max(E_diff) * 0.7),
                   arrowprops=dict(arrowstyle='->', color='black'),
                   fontsize=10)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"\nPlot saved to: {save_path}")
        
        if show:
            plt.show()
        else:
            plt.close(fig)
        return fig


def main():
    parser = argparse.ArgumentParser(description="Model discrimination: NN vs J1-J2")
    parser.add_argument("--J1", type=float, default=5.0, help="True J1 (meV)")
    parser.add_argument("--J2", type=float, default=0.5, help="True J2 (meV), 0 for NN-only")
    parser.add_argument("--D", type=float, default=0.1, help="True D (meV)")
    parser.add_argument("--iterations", type=int, default=20, help="Max iterations")
    parser.add_argument("--no-j2", action="store_true", help="True model is NN-only")
    parser.add_argument("--no-plot", action="store_true", help="Disable plotting")
    parser.add_argument("--save", type=str, help="Save plot to file")
    
    args = parser.parse_args()
    
    np.random.seed(42)
    
    include_j2 = not args.no_j2 and args.J2 != 0
    
    discriminator = ModelDiscriminator(
        true_J1=args.J1,
        true_J2=args.J2 if include_j2 else 0,
        true_D=args.D,
        include_j2=include_j2
    )
    
    results = discriminator.run(n_iterations=args.iterations)
    
    discriminator.plot_results(save_path=args.save, show=not args.no_plot)


if __name__ == "__main__":
    main()
