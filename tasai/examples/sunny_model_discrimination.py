#!/usr/bin/env python3
"""
Model Discrimination: J1-Only vs J1+J2 Exchange

Autonomous experiment to determine whether next-nearest-neighbor exchange (J2)
is needed to explain spin wave data, using ANDiE-style model comparison.

This example demonstrates:
1. Setting up two competing models (J1-only vs J1+J2)
2. Calculating Bayesian evidence for each model
3. Using acquisition function that maximizes model discrimination
4. Determining which model best describes the data

Competing models:
    Model A (J1-only):  H = -J1 Σ_<i,j> Si·Sj - D Σ_i (Siz)²
    Model B (J1+J2):    H = -J1 Σ_<i,j> Si·Sj - J2 Σ_<<i,j>> Si·Sj - D Σ_i (Siz)²

The key difference is in the dispersion relation:
    J1-only: ω(H,H,0) = 4SJ1 * sqrt(sin²(πH) * (sin²(πH) + D/J1))
    J1+J2:   Modified by J2 terms that add curvature

Usage:
    python model_discrimination.py
    python model_discrimination.py --truth J1+J2 --true-J2 0.3
    python model_discrimination.py --truth J1-only
"""

import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional
import time
import argparse
from scipy.special import logsumexp

from tasai.sunny import (
    SquareLatticeFM,
    SquareLatticeFM_J1Only,
    simulate_measurement
)


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class DiscriminationConfig:
    """Configuration for model discrimination experiment."""
    # True model: 'J1-only' or 'J1+J2'
    true_model_type: str = 'J1+J2'
    
    # True parameters
    true_J1: float = 1.5   # meV
    true_J2: float = 0.25  # meV (only used if true_model_type == 'J1+J2')
    true_D: float = 0.1    # meV
    
    # Measurement range
    H_min: float = 0.0
    H_max: float = 0.5
    E_min: float = 0.5
    E_max: float = 10.0
    L: float = 0.0
    
    # Measurement parameters
    count_time: float = 60.0
    count_rate: float = 100.0
    
    # MCMC parameters
    mcmc_burn: int = 150
    mcmc_steps: int = 150
    
    # Autonomous loop
    n_iterations: int = 25
    n_forecast: int = 3
    
    # Model discrimination threshold
    decisive_bayes_factor: float = 10.0  # ln(BF) > ln(10) is "strong evidence"


@dataclass
class Measurement:
    """A single measurement."""
    H: float
    K: float
    L: float
    E: float
    I: float
    sigma: float
    source: str
    timestamp: float


# =============================================================================
# Simple MCMC with Evidence Estimation
# =============================================================================

class MCMCWithEvidence:
    """
    MCMC sampler that also estimates marginal likelihood (evidence).
    
    Uses harmonic mean estimator (biased but simple).
    For production, use nested sampling or thermodynamic integration.
    """
    
    def __init__(self, model, burn: int = 150, steps: int = 150):
        self.model = model
        self.burn = burn
        self.steps = steps
        
        self.H_data = None
        self.K_data = None
        self.L_data = None
        self.E_data = None
        self.I_data = None
        self.sigma_data = None
        
        self.proposal_width = {p: 0.1 for p in model.free_params}
    
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
        self.model.parameter_vector = params
        I_pred = self.model.compute_intensity_array(
            self.H_data, self.K_data, self.L_data, self.E_data
        )
        chi2 = np.sum(((self.I_data - I_pred) / self.sigma_data)**2)
        return -0.5 * chi2
    
    def log_prior(self, params: np.ndarray) -> float:
        """Log prior (uniform)."""
        self.model.parameter_vector = params
        return self.model.log_prior()
    
    def run(self) -> Tuple[np.ndarray, float]:
        """
        Run MCMC and estimate evidence.
        
        Returns
        -------
        chain : ndarray
            MCMC samples
        log_evidence : float
            Estimated log marginal likelihood
        """
        n_params = self.model.n_free
        
        # Initialize
        current = self.model.parameter_vector.copy()
        current_log_like = self.log_likelihood(current)
        
        chain = np.zeros((self.burn + self.steps, n_params))
        log_likes = np.zeros(self.burn + self.steps)
        
        # Run MCMC
        for i in range(self.burn + self.steps):
            proposal = current.copy()
            for j, p in enumerate(self.model.free_params):
                proposal[j] += np.random.normal(0, self.proposal_width.get(p, 0.1))
            
            if self.log_prior(proposal) > -np.inf:
                proposal_log_like = self.log_likelihood(proposal)
                
                if np.log(np.random.random()) < proposal_log_like - current_log_like:
                    current = proposal
                    current_log_like = proposal_log_like
            
            chain[i] = current
            log_likes[i] = current_log_like
        
        # Estimate evidence using harmonic mean (biased but simple)
        # log Z ≈ -log(mean(1/L)) where L is likelihood
        post_burn_log_likes = log_likes[self.burn:]
        
        # More stable: use importance sampling estimate
        # Approximate with max likelihood + volume correction
        max_log_like = np.max(post_burn_log_likes)
        n_params = self.model.n_free
        
        # BIC-style approximation: log Z ≈ max_log_like - (k/2)*log(n)
        n_data = len(self.I_data)
        log_evidence = max_log_like - 0.5 * n_params * np.log(n_data)
        
        return chain[self.burn:], log_evidence


# =============================================================================
# Model Discrimination Acquisition
# =============================================================================

class DiscriminationAcquisition:
    """
    Acquisition function for model discrimination.
    
    Selects points where the two models make different predictions.
    """
    
    def __init__(self, model_A, model_B):
        self.model_A = model_A
        self.model_B = model_B
    
    def score(self, candidates: np.ndarray, 
              measurements: List[Measurement]) -> np.ndarray:
        """
        Score candidates by how discriminating they are.
        
        High score = models predict very different intensities.
        """
        n_candidates = len(candidates)
        scores = np.zeros(n_candidates)
        
        measured_HE = np.array([[m.H, m.E] for m in measurements]) if measurements else np.zeros((0, 2))
        
        for i, (H, K, L, E) in enumerate(candidates):
            # Predictions from each model
            I_A = self.model_A.compute_intensity(H, K, L, E)
            I_B = self.model_B.compute_intensity(H, K, L, E)
            
            # Discrimination score: how different are predictions?
            # Use relative difference, weighted by signal strength
            avg_I = (I_A + I_B) / 2 + 0.01  # Avoid division by zero
            discrimination = abs(I_A - I_B) / avg_I
            
            # Signal score: want to measure where there's signal
            signal = np.log1p(avg_I * 100)
            
            # Novelty: avoid re-measuring
            novelty = 1.0
            for H_m, E_m in measured_HE:
                dist = np.sqrt((H - H_m)**2 + ((E - E_m)/5)**2)
                novelty *= (1 - 0.5 * np.exp(-dist**2 / 0.01))
            
            scores[i] = discrimination * signal * novelty
        
        return scores


# =============================================================================
# Main Experiment
# =============================================================================

class ModelDiscriminationExperiment:
    """
    Autonomous experiment to discriminate between J1-only and J1+J2 models.
    """
    
    def __init__(self, config: DiscriminationConfig):
        self.config = config
        
        # Create true model
        if config.true_model_type == 'J1-only':
            self.true_model = SquareLatticeFM_J1Only(J1=config.true_J1, D=config.true_D)
        else:
            self.true_model = SquareLatticeFM(J1=config.true_J1, J2=config.true_J2, D=config.true_D)
        self.true_model._use_mock = True
        
        # Competing models
        self.model_A = SquareLatticeFM_J1Only(J1=1.0, D=0.05)  # J1-only
        self.model_B = SquareLatticeFM(J1=1.0, J2=0.0, D=0.05)  # J1+J2
        self.model_A._use_mock = True
        self.model_B._use_mock = True
        
        # Model weights (prior: 50-50)
        self.weight_A = 0.5  # P(Model A | data)
        self.weight_B = 0.5  # P(Model B | data)
        
        # Evidence estimates
        self.log_evidence_A = 0.0
        self.log_evidence_B = 0.0
        
        # Acquisition function
        self.acquisition = DiscriminationAcquisition(self.model_A, self.model_B)
        
        # State
        self.measurements: List[Measurement] = []
        self.iteration = 0
        
        # History
        self.weight_history_A = []
        self.weight_history_B = []
        self.bayes_factor_history = []
    
    def measure(self, H: float, E: float, source: str = 'ai') -> Measurement:
        """Simulate measurement."""
        K = H
        L = self.config.L
        
        I, sigma = simulate_measurement(
            self.true_model, H, K, L, E,
            count_time=self.config.count_time,
            count_rate=self.config.count_rate
        )
        
        m = Measurement(H=H, K=K, L=L, E=E, I=I, sigma=sigma, source=source, timestamp=time.time())
        self.measurements.append(m)
        return m
    
    def update_model_weights(self):
        """Update model weights using Bayesian evidence."""
        if len(self.measurements) < 5:
            return
        
        # Fit model A (J1-only)
        mcmc_A = MCMCWithEvidence(self.model_A, burn=self.config.mcmc_burn, steps=self.config.mcmc_steps)
        mcmc_A.set_data(self.measurements)
        chain_A, self.log_evidence_A = mcmc_A.run()
        
        # Update model A parameters from posterior
        for i, p in enumerate(self.model_A.free_params):
            setattr(self.model_A, p, np.median(chain_A[:, i]))
        
        # Fit model B (J1+J2)
        mcmc_B = MCMCWithEvidence(self.model_B, burn=self.config.mcmc_burn, steps=self.config.mcmc_steps)
        mcmc_B.set_data(self.measurements)
        chain_B, self.log_evidence_B = mcmc_B.run()
        
        # Update model B parameters
        for i, p in enumerate(self.model_B.free_params):
            setattr(self.model_B, p, np.median(chain_B[:, i]))
        
        # Update weights using Bayes' theorem
        # P(M|D) ∝ P(D|M) * P(M)
        # With equal priors: P(M|D) ∝ P(D|M) = Evidence
        
        # Use log-sum-exp for numerical stability
        log_total = logsumexp([self.log_evidence_A, self.log_evidence_B])
        self.weight_A = np.exp(self.log_evidence_A - log_total)
        self.weight_B = np.exp(self.log_evidence_B - log_total)
        
        # Bayes factor: evidence_B / evidence_A
        # log(BF) = log_evidence_B - log_evidence_A
        bayes_factor = self.log_evidence_B - self.log_evidence_A
        
        self.weight_history_A.append(self.weight_A)
        self.weight_history_B.append(self.weight_B)
        self.bayes_factor_history.append(bayes_factor)
    
    def generate_candidates(self, n: int = 200) -> np.ndarray:
        """Generate candidate measurement points."""
        H = np.random.uniform(self.config.H_min, self.config.H_max, n)
        E = np.random.uniform(self.config.E_min, self.config.E_max, n)
        K = H
        L = np.full(n, self.config.L)
        return np.column_stack([H, K, L, E])
    
    def select_next_points(self, n_points: int = 3) -> List[Tuple[float, float]]:
        """Select discriminating measurement points."""
        candidates = self.generate_candidates(200)
        
        selected = []
        temp_measurements = self.measurements.copy()
        
        for _ in range(n_points):
            scores = self.acquisition.score(candidates, temp_measurements)
            best_idx = np.argmax(scores)
            
            H, K, L, E = candidates[best_idx]
            selected.append((H, E))
            
            temp_measurements.append(Measurement(H=H, K=K, L=L, E=E, I=0, sigma=1, source='temp', timestamp=0))
        
        return selected
    
    def add_initial_points(self):
        """Add initial measurements."""
        # Grid of initial points
        H_init = np.linspace(0.05, 0.45, 4)
        E_init = np.linspace(1.0, 8.0, 3)
        
        print(f"Adding {len(H_init) * len(E_init)} initial measurements...")
        for H in H_init:
            for E in E_init:
                m = self.measure(H, E, source='initial')
    
    def get_leading_model(self) -> Tuple[str, float]:
        """Return leading model and confidence."""
        if self.weight_A > self.weight_B:
            return "J1-only", self.weight_A
        else:
            return "J1+J2", self.weight_B
    
    def is_decisive(self) -> bool:
        """Check if we have decisive evidence for one model."""
        bf = abs(self.log_evidence_B - self.log_evidence_A)
        return bf > np.log(self.config.decisive_bayes_factor)
    
    def run(self) -> Dict:
        """Run model discrimination experiment."""
        print("=" * 70)
        print("Model Discrimination: J1-Only vs J1+J2")
        print("=" * 70)
        print(f"True model: {self.config.true_model_type}")
        if self.config.true_model_type == 'J1+J2':
            print(f"True parameters: J1={self.config.true_J1}, J2={self.config.true_J2}, D={self.config.true_D}")
        else:
            print(f"True parameters: J1={self.config.true_J1}, D={self.config.true_D}")
        print(f"Decisive threshold: Bayes factor > {self.config.decisive_bayes_factor}")
        print("=" * 70)
        
        # Initial measurements
        self.add_initial_points()
        self.update_model_weights()
        
        leader, confidence = self.get_leading_model()
        print(f"\nInitial: Leading model = {leader} ({confidence*100:.1f}%)")
        
        print(f"\nRunning autonomous discrimination...")
        print("-" * 70)
        print(f"{'Iter':>4} | {'J1-only':>8} | {'J1+J2':>8} | {'log(BF)':>8} | {'Leading':>10} | {'Status':>10}")
        print("-" * 70)
        
        for _ in range(self.config.n_iterations):
            self.iteration += 1
            
            # Select and measure
            next_points = self.select_next_points(self.config.n_forecast)
            for H, E in next_points:
                self.measure(H, E, source='ai')
            
            # Update weights
            self.update_model_weights()
            
            leader, confidence = self.get_leading_model()
            bf = self.bayes_factor_history[-1] if self.bayes_factor_history else 0
            
            status = "DECISIVE" if self.is_decisive() else "uncertain"
            
            print(f"{self.iteration:4d} | {self.weight_A*100:7.1f}% | {self.weight_B*100:7.1f}% | "
                  f"{bf:8.2f} | {leader:>10} | {status:>10}")
            
            if self.is_decisive():
                print(f"\n*** Decisive evidence for {leader}! ***")
                break
        
        # Final results
        leader, confidence = self.get_leading_model()
        correct = (leader == self.config.true_model_type)
        
        print("\n" + "=" * 70)
        print("FINAL RESULTS")
        print("=" * 70)
        print(f"  Selected model: {leader} ({confidence*100:.1f}% confidence)")
        print(f"  True model: {self.config.true_model_type}")
        print(f"  Correct: {'✓ YES' if correct else '✗ NO'}")
        print(f"  log(Bayes Factor): {self.bayes_factor_history[-1]:.3f}")
        print(f"  Total measurements: {len(self.measurements)}")
        
        if leader == 'J1+J2':
            print(f"\n  Best-fit J1+J2 parameters:")
            print(f"    J1 = {self.model_B.J1:.4f} meV")
            print(f"    J2 = {self.model_B.J2:.4f} meV")
            print(f"    D  = {self.model_B.D:.4f} meV")
        else:
            print(f"\n  Best-fit J1-only parameters:")
            print(f"    J1 = {self.model_A.J1:.4f} meV")
            print(f"    D  = {self.model_A.D:.4f} meV")
        
        print("=" * 70)
        
        return {
            'selected_model': leader,
            'confidence': confidence,
            'correct': correct,
            'log_bayes_factor': self.bayes_factor_history[-1] if self.bayes_factor_history else 0,
            'n_measurements': len(self.measurements),
            'weight_history_A': self.weight_history_A,
            'weight_history_B': self.weight_history_B,
            'bayes_factor_history': self.bayes_factor_history,
            'model_A_params': {'J1': self.model_A.J1, 'D': self.model_A.D},
            'model_B_params': {'J1': self.model_B.J1, 'J2': self.model_B.J2, 'D': self.model_B.D}
        }
    
    def plot_results(self, save_path: str = None):
        """Plot discrimination results."""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # 1. Model weights over time
        ax = axes[0, 0]
        iters = np.arange(1, len(self.weight_history_A) + 1)
        ax.fill_between(iters, 0, self.weight_history_A, alpha=0.5, color='blue', label='J1-only')
        ax.fill_between(iters, self.weight_history_A, 1, alpha=0.5, color='green', label='J1+J2')
        ax.plot(iters, self.weight_history_A, 'b-', lw=2)
        ax.plot(iters, self.weight_history_B, 'g-', lw=2)
        ax.axhline(0.5, color='k', ls='--', alpha=0.5)
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Model Probability')
        ax.set_title('Model Weights Evolution')
        ax.legend(loc='upper right')
        ax.set_ylim(0, 1)
        
        # 2. Bayes factor
        ax = axes[0, 1]
        ax.plot(iters, self.bayes_factor_history, 'k-o', markersize=4)
        ax.axhline(np.log(10), color='r', ls='--', label='Strong evidence threshold')
        ax.axhline(-np.log(10), color='r', ls='--')
        ax.axhline(0, color='gray', ls='-', alpha=0.3)
        ax.fill_between(iters, -np.log(10), np.log(10), alpha=0.1, color='gray', label='Inconclusive')
        ax.set_xlabel('Iteration')
        ax.set_ylabel('log(Bayes Factor)')
        ax.set_title('Bayes Factor: J1+J2 vs J1-only')
        ax.legend()
        
        # 3. Measurements in H-E space
        ax = axes[1, 0]
        H_data = [m.H for m in self.measurements]
        E_data = [m.E for m in self.measurements]
        I_data = [m.I for m in self.measurements]
        
        sc = ax.scatter(H_data, E_data, c=I_data, cmap='viridis', s=40, alpha=0.7)
        plt.colorbar(sc, ax=ax, label='Intensity')
        
        # Overlay dispersions
        H_plot = np.linspace(0, 0.5, 100)
        ax.plot(H_plot, self.model_A.dispersion_HHL(H_plot), 'b--', lw=2, label='J1-only fit')
        ax.plot(H_plot, self.model_B.dispersion_HHL(H_plot), 'g--', lw=2, label='J1+J2 fit')
        ax.plot(H_plot, self.true_model.dispersion_HHL(H_plot), 'k-', lw=2, alpha=0.5, label='True')
        
        ax.set_xlabel('[H H 0] (r.l.u.)')
        ax.set_ylabel('E (meV)')
        ax.set_title('Measurements and Model Dispersions')
        ax.legend(loc='upper left')
        
        # 4. Dispersion comparison
        ax = axes[1, 1]
        H_plot = np.linspace(0, 0.5, 100)
        
        E_true = self.true_model.dispersion_HHL(H_plot)
        E_A = self.model_A.dispersion_HHL(H_plot)
        E_B = self.model_B.dispersion_HHL(H_plot)
        
        ax.plot(H_plot, E_true, 'k-', lw=3, label=f'True ({self.config.true_model_type})')
        ax.plot(H_plot, E_A, 'b--', lw=2, label=f'J1-only (J1={self.model_A.J1:.2f})')
        ax.plot(H_plot, E_B, 'g--', lw=2, label=f'J1+J2 (J1={self.model_B.J1:.2f}, J2={self.model_B.J2:.2f})')
        
        ax.set_xlabel('[H H 0] (r.l.u.)')
        ax.set_ylabel('E (meV)')
        ax.set_title('Dispersion Comparison')
        ax.legend()
        
        # Add text with results
        leader, conf = self.get_leading_model()
        fig.suptitle(f"Model Discrimination Result: {leader} ({conf*100:.0f}% confidence)", 
                    fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"\nPlot saved to: {save_path}")
        
        plt.show()


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Model Discrimination: J1-only vs J1+J2")
    parser.add_argument("--truth", type=str, default="J1+J2", choices=["J1-only", "J1+J2"],
                       help="True model type")
    parser.add_argument("--true-J1", type=float, default=1.5, help="True J1 (meV)")
    parser.add_argument("--true-J2", type=float, default=0.25, help="True J2 (meV, for J1+J2 model)")
    parser.add_argument("--true-D", type=float, default=0.1, help="True D (meV)")
    parser.add_argument("--n-iterations", type=int, default=25, help="Max iterations")
    parser.add_argument("--no-plot", action="store_true", help="Skip plotting")
    parser.add_argument("--save", type=str, help="Save plot to file")
    
    args = parser.parse_args()
    
    config = DiscriminationConfig(
        true_model_type=args.truth,
        true_J1=args.true_J1,
        true_J2=args.true_J2,
        true_D=args.true_D,
        n_iterations=args.n_iterations
    )
    
    np.random.seed(42)
    
    experiment = ModelDiscriminationExperiment(config)
    results = experiment.run()
    
    if not args.no_plot:
        experiment.plot_results(save_path=args.save)
    
    return results


if __name__ == "__main__":
    main()
