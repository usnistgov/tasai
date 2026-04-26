#!/usr/bin/env python3
"""
Compare Laplace/LM covariance vs full MCMC posterior on the single-branch scenario.

This script generates synthetic single-branch data, fits parameters with:
- Laplace/LM approximation (grid-search MAP + Jacobian covariance)
- MCMC posterior sampling (emcee/DREAM via MCMCRunner)

Outputs timing and parameter estimates for a one-scenario comparison.
"""

import argparse
import time
import numpy as np
from scipy.optimize import least_squares

from tasai.physics.base import PhysicsModel, Parameter
from tasai.inference import MCMCRunner


class SingleBranchModel(PhysicsModel):
    """Analytic single-branch dispersion model with J1 and D free parameters."""

    def __init__(self, J1: float = 5.0, D: float = 0.1,
                 gamma: float = 0.5, A: float = 1.0, background: float = 0.01):
        super().__init__()
        self.gamma = gamma
        self.A = A
        self.background = background

        self._parameters.append(Parameter(
            name='J1', value=J1, bounds=(1.0, 12.0), units='meV',
            description='Nearest-neighbor exchange'
        ))
        self._param_index['J1'] = 0

        self._parameters.append(Parameter(
            name='D', value=D, bounds=(0.0, 0.5), units='meV',
            description='Single-ion anisotropy'
        ))
        self._param_index['D'] = 1

    def compute_intensity(self, h: float, k: float, l: float, E: float, **kwargs) -> float:
        J1 = self.get_parameter('J1').value
        D = self.get_parameter('D').value
        omega = 2 * J1 * (1 - np.cos(2 * np.pi * h)) + D
        I = self.A * self.gamma / ((E - omega) ** 2 + self.gamma ** 2) / np.pi
        return I + self.background

    def description(self) -> str:
        return "Single-branch analytic dispersion (J1, D)"


def generate_data(n_points: int, seed: int = 0, sigma: float = 0.02):
    rng = np.random.default_rng(seed)
    H = rng.uniform(0.0, 0.5, size=n_points)
    true_model = SingleBranchModel(J1=5.0, D=0.1)
    E_true = 2 * 5.0 * (1 - np.cos(2 * np.pi * H)) + 0.1
    E = E_true + rng.normal(0.0, 0.5, size=n_points)

    I_true = np.array([true_model.compute_intensity(h, 0.0, 0.0, e) for h, e in zip(H, E)])
    I_obs = I_true + rng.normal(0.0, sigma, size=n_points)
    sigma_arr = np.full_like(I_obs, sigma)

    return H, np.zeros_like(H), np.zeros_like(H), E, I_obs, sigma_arr


def laplace_fit(model: SingleBranchModel, h, k, l, E, I, sigma,
                n_starts: int = 10, seed: int = 0):
    rng = np.random.default_rng(seed)
    params = ['J1', 'D']
    bounds = np.array([model.get_parameter(p).bounds for p in params], dtype=float)

    def residuals(values: np.ndarray) -> np.ndarray:
        model.set_parameters({'J1': values[0], 'D': values[1]})
        I_pred = model.compute_intensity_array(h, k, l, E)
        return (I_pred - I) / sigma

    # Build start points: current value + random draws from bounds
    starts = [model.get_free_values()]
    for _ in range(max(n_starts - 1, 0)):
        starts.append(rng.uniform(bounds[:, 0], bounds[:, 1]))

    best_chi2 = np.inf
    best = None
    best_jac = None
    for x0 in starts:
        result = least_squares(residuals, x0=x0, bounds=bounds.T, method='trf')
        chi2 = np.sum(result.fun ** 2)
        if chi2 < best_chi2:
            best_chi2 = chi2
            best = result.x
            best_jac = result.jac

    model.set_parameters({'J1': best[0], 'D': best[1]})

    # Approximate covariance from Jacobian at best fit
    try:
        W = np.diag(1.0 / (sigma ** 2))
        cov = np.linalg.inv(best_jac.T @ W @ best_jac)
    except np.linalg.LinAlgError:
        cov = np.full((2, 2), np.nan)

    return best, cov, best_chi2


def mcmc_fit(h, k, l, E, I, sigma, burn=200, steps=200, pop=6, backend="emcee"):
    model = SingleBranchModel(J1=4.0, D=0.1)
    runner = MCMCRunner(model, burn=burn, steps=steps, pop=pop, backend=backend, parallel=False)
    runner.set_data(h=h, k=k, l=l, E=E, I=I, sigma=sigma)
    t0 = time.perf_counter()
    if backend == "bumps_simple":
        samples = runner._run_bumps_simple()  # Force DreamFit path for BUMPS
    else:
        samples = runner.run()
    t1 = time.perf_counter()
    mean = np.mean(samples, axis=0)
    std = np.std(samples, axis=0)
    return mean, std, t1 - t0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-points", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--sigma", type=float, default=0.02)
    parser.add_argument("--mcmc-burn", type=int, default=200)
    parser.add_argument("--mcmc-steps", type=int, default=200)
    parser.add_argument("--mcmc-pop", type=int, default=6)
    parser.add_argument("--mcmc-backend", type=str, default="emcee",
                        choices=["emcee", "bumps", "bumps_simple", "metropolis", "auto"])
    parser.add_argument("--laplace-starts", type=int, default=10)
    parser.add_argument("--tripwire-chi2", type=float, default=5.0,
                        help="Run MCMC if reduced chi2 exceeds this threshold")
    parser.add_argument("--mcmc-only-on-tripwire", action="store_true",
                        help="Skip MCMC unless tripwire is triggered")
    args = parser.parse_args()

    h, k, l, E, I, sigma = generate_data(args.n_points, args.seed, args.sigma)

    # Laplace/LM (grid MAP + Jacobian covariance)
    laplace_model = SingleBranchModel(J1=4.0, D=0.1)
    t0 = time.perf_counter()
    best, cov, chi2 = laplace_fit(laplace_model, h, k, l, E, I, sigma,
                                  n_starts=args.laplace_starts, seed=args.seed)
    t1 = time.perf_counter()
    laplace_time = t1 - t0
    laplace_std = np.sqrt(np.diag(cov)) if np.all(np.isfinite(cov)) else [np.nan, np.nan]
    dof = max(len(I) - 2, 1)
    chi2_red = chi2 / dof

    tripwire = chi2_red > args.tripwire_chi2

    mean = std = None
    mcmc_time = None
    if (not args.mcmc_only_on_tripwire) or tripwire:
        mean, std, mcmc_time = mcmc_fit(h, k, l, E, I, sigma,
                                       burn=args.mcmc_burn, steps=args.mcmc_steps,
                                       pop=args.mcmc_pop, backend=args.mcmc_backend)

    print("=== Single-branch posterior comparison ===")
    print(f"N points: {args.n_points}, sigma: {args.sigma}")
    print("Laplace/LM (grid MAP + Jacobian covariance):")
    print(f"  J1={best[0]:.3f} ± {laplace_std[0]:.3f}, D={best[1]:.3f} ± {laplace_std[1]:.3f}")
    print(f"  chi2={chi2:.2f} (red={chi2_red:.2f}), time={laplace_time:.3f}s")
    if tripwire:
        print(f"  Tripwire: reduced chi2 > {args.tripwire_chi2} → running MCMC")
    if mean is not None:
        print(f"MCMC ({args.mcmc_backend}):")
        print(f"  J1={mean[0]:.3f} ± {std[0]:.3f}, D={mean[1]:.3f} ± {std[1]:.3f}")
        print(f"  time={mcmc_time:.3f}s")
    else:
        print(f"MCMC skipped (tripwire not triggered)")


if __name__ == "__main__":
    main()
