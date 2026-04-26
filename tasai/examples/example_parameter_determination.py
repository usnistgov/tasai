#!/usr/bin/env python3
"""
Figure 1 / Table 1 reproduction script.

This script compares multiple measurement policies (Grid, Random, library
Log-GP, TAS-AI) on the square-lattice spin-wave toy model and produces:

1. A four-panel figure showing RMS parameter error vs. measurement count,
   the TAS-AI measurement path, parameter convergence, and χ²/N.
2. A JSON summary containing convergence counts and final RMS errors for each
   policy which is ingested when building Table 1 in the manuscript.

Important: this is a controlled parameter-refinement study. The task assumes
the Hamiltonian family is already known, and the comparison policies are set up
to compare parameter-contraction behavior after discovery/localization rather
than to benchmark blind end-to-end autonomy.

Run:
    python example_parameter_determination.py \
        --max-measurements 80 \
        --threshold 0.2 \
        --figure tasai_paper_clean/paper/figure1_convergence.png \
        --summary-json tasai_review/logs/figure1_summary.json
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize

from tasai.core.gaussian_process import AgnosticExplorer
from tasai.instrument.motors import MotionAwareAcquisition, SimplifiedMotorModel
from tasai.sunny import SquareLatticeFM


def _set_plot_style() -> None:
    for style in ("seaborn-v0_8-whitegrid", "seaborn-whitegrid", "default"):
        try:
            plt.style.use(style)
            return
        except OSError:
            continue


# ---------------------------------------------------------------------------
# Measurement container
# ---------------------------------------------------------------------------

@dataclass
class Measurement:
    H: float
    E: float
    I: float
    sigma: float
    source: str


@dataclass(frozen=True)
class BenchmarkConfig:
    true_J1: float = 5.0
    true_J2: float = 0.8
    true_D: float = 0.15
    prior_J1: float = 4.4
    prior_J2: float = 0.25
    prior_D: float = 0.08
    init_points: int = 4
    init_mode: str = "prior_triplet"
    count_time: float = 30.0
    use_motion: bool = False


# ---------------------------------------------------------------------------
# Parameter estimator
# ---------------------------------------------------------------------------

class ParameterEstimator:
    """Light-weight estimator used for all policies."""

    def __init__(self, config: BenchmarkConfig | None = None):
        self.config = config or BenchmarkConfig()
        self.true_params = np.array([self.config.true_J1, self.config.true_J2, self.config.true_D])
        self.true_model = SquareLatticeFM(
            J1=self.config.true_J1,
            J2=self.config.true_J2,
            D=self.config.true_D,
        )
        self.prior_params = np.array([self.config.prior_J1, self.config.prior_J2, self.config.prior_D])
        self.prior_model = SquareLatticeFM(
            J1=self.config.prior_J1,
            J2=self.config.prior_J2,
            D=self.config.prior_D,
        )
        self.est_model = SquareLatticeFM(
            J1=self.config.prior_J1,
            J2=self.config.prior_J2,
            D=self.config.prior_D,
        )
        self.count_time = 30.0
        self.count_rate = 20.0
        self.count_time = float(self.config.count_time)
        self.measurements: List[Measurement] = []
        self.motor = SimplifiedMotorModel(current_E=5.0)
        self.total_time_s = 0.0
        self.time_history: List[float] = []
        self.J1_history: List[float] = []
        self.J2_history: List[float] = []
        self.D_history: List[float] = []
        self.chi2_history: List[float] = []
        self.rms_history: List[float] = []
        self.H_range = (0.02, 0.48)
        self.E_range = (0.5, 50.0)
        self.param_bounds = np.array([
            [3.0, 7.0],
            [0.0, 1.2],
            [0.02, 0.3],
        ], dtype=float)

    # ------------------------------------------------------------------
    # Measurement / modeling helpers
    # ------------------------------------------------------------------
    def measure(self, H: float, E: float, source: str) -> Measurement:
        move_time = self.motor.move_time(H, H, 0.0, E)
        self.motor.move_to(H, H, 0.0, E)
        self.total_time_s += move_time + self.count_time
        I, sigma = self.true_model.simulate_measurement(
            H, 0.0, E, count_time=self.count_time, count_rate=self.count_rate
        )
        m = Measurement(H=H, E=E, I=I, sigma=sigma, source=source)
        self.measurements.append(m)
        return m

    def add_initial_points(self, n_points: int | None = None, mode: str | None = None):
        n_points = n_points or self.config.init_points
        mode = mode or self.config.init_mode

        if mode == "prior_branch":
            Hs = np.linspace(self.H_range[0] + 0.03, self.H_range[1] - 0.03, n_points)
            seeds = [(float(H), float(self.prior_model.dispersion(H, 0.0))) for H in Hs]
        elif mode == "prior_triplet":
            base_H = np.array([0.07, 0.18, 0.30, 0.42, 0.46], dtype=float)
            offsets = np.array([0.0, 0.35, -0.35, 0.55, -0.55], dtype=float)
            seeds = []
            for idx, H in enumerate(base_H[:n_points]):
                E0 = float(self.prior_model.dispersion(H, 0.0))
                seeds.append((float(H), float(np.clip(E0 + offsets[idx], self.E_range[0], self.E_range[1]))))
        else:
            raise ValueError(f"Unknown init mode: {mode}")

        for H, E in seeds:
            self.measure(H, E, source='initial')
        self.fit_model()

    def fit_model(self):
        if len(self.measurements) < 4:
            return

        H = np.array([m.H for m in self.measurements])
        E = np.array([m.E for m in self.measurements])
        I = np.array([m.I for m in self.measurements])
        sigma = np.array([m.sigma for m in self.measurements])

        def chi2_for(params: np.ndarray) -> float:
            clipped = np.clip(params, self.param_bounds[:, 0], self.param_bounds[:, 1])
            model = SquareLatticeFM(J1=float(clipped[0]), J2=float(clipped[1]), D=float(clipped[2]))
            return float(model.chi_squared(H, np.zeros_like(H), E, I, sigma))

        coarse_ranked: List[Tuple[float, Tuple[float, float, float]]] = []
        for J1 in np.linspace(3.0, 7.0, 9):
            for J2 in np.linspace(0.0, 1.2, 8):
                for D in np.linspace(0.02, 0.3, 6):
                    params = (float(J1), float(J2), float(D))
                    coarse_ranked.append((chi2_for(np.array(params)), params))

        coarse_ranked.sort(key=lambda item: item[0])
        best = float(coarse_ranked[0][0])
        best_params = np.array(coarse_ranked[0][1], dtype=float)

        starts = [best_params, np.array([self.est_model.J1, self.est_model.J2, self.est_model.D], dtype=float)]
        starts.extend(np.array(params, dtype=float) for _, params in coarse_ranked[1:4])

        seen = set()
        for start in starts:
            key = tuple(np.round(start, 8))
            if key in seen:
                continue
            seen.add(key)
            try:
                result = minimize(
                    chi2_for,
                    x0=start,
                    method='L-BFGS-B',
                    bounds=[tuple(row) for row in self.param_bounds],
                )
            except Exception:
                continue
            if result.success and float(result.fun) < best:
                best = float(result.fun)
                best_params = np.array(result.x, dtype=float)

        self.est_model.J1, self.est_model.J2, self.est_model.D = map(float, best_params)
        self.J1_history.append(float(best_params[0]))
        self.J2_history.append(float(best_params[1]))
        self.D_history.append(float(best_params[2]))
        self.chi2_history.append(best / max(len(self.measurements) - 3, 1))

        self.rms_history.append(self.current_rms_error())
        self.time_history.append(float(self.total_time_s))

    def current_rms_error(self) -> float:
        params = np.array([self.est_model.J1, self.est_model.J2, self.est_model.D])
        return float(np.sqrt(np.mean((params - self.true_params) ** 2)))

    def suggest_tas_point(self) -> Tuple[float, float]:
        H_candidates = np.linspace(self.H_range[0], self.H_range[1], 25)
        candidates: List[Tuple[float, float]] = []
        infos: List[float] = []
        for H in H_candidates:
            E0 = self.est_model.dispersion(H, 0.0)
            for dE in (-0.8, -0.3, 0, 0.3, 0.8):
                E = E0 + dE
                if not (self.E_range[0] < E < self.E_range[1]):
                    continue
                info = self._info_gain_heuristic(H, E) * self._novelty(H, E)
                candidates.append((float(H), float(E)))
                infos.append(float(info))

        if not candidates:
            return 0.25, 15.0

        if self.config.use_motion:
            acquisition = MotionAwareAcquisition(
                motor_model=self.motor,
                eta=0.7,
                count_time=self.count_time,
            )
            cand_array = np.array([[H, H, 0.0, E] for H, E in candidates], dtype=float)
            scores = acquisition.score_batch(cand_array, np.array(infos, dtype=float))
            return candidates[int(np.argmax(scores))]

        return candidates[int(np.argmax(np.array(infos, dtype=float)))]

    def _info_gain_heuristic(self, H: float, E: float) -> float:
        dispersion = self.est_model.dispersion(H, 0.0)
        dist = abs(E - dispersion)
        intensity = np.exp(-dist ** 2 / 0.8)
        j2 = 1 + 0.5 * np.sin(4 * np.pi * H) ** 2
        return intensity * j2

    def _novelty(self, H: float, E: float) -> float:
        novelty = 1.0
        for m in self.measurements:
            dh = H - m.H
            de = (E - m.E) / 5.0
            novelty *= (1 - 0.8 * np.exp(-(dh ** 2 + de ** 2) / 0.01))
        return novelty


# ---------------------------------------------------------------------------
# Measurement policies
# ---------------------------------------------------------------------------

class MeasurementPolicy:
    def __init__(self, name: str):
        self.name = name

    def reset(self, estimator: ParameterEstimator):
        pass

    def select(self, estimator: ParameterEstimator, iteration: int) -> Tuple[float, float]:
        raise NotImplementedError


class GridPolicy(MeasurementPolicy):
    def __init__(self):
        super().__init__('grid')
        self.Hs: np.ndarray | None = None
        self.idx = 0

    def reset(self, estimator: ParameterEstimator):
        self.Hs = np.linspace(estimator.H_range[0], estimator.H_range[1], 60)
        self.idx = 0

    def select(self, estimator: ParameterEstimator, iteration: int):
        assert self.Hs is not None
        if self.idx >= len(self.Hs):
            self.idx = self.idx % len(self.Hs)
        H = float(self.Hs[self.idx])
        self.idx += 1
        E = float(np.clip(estimator.est_model.dispersion(H, 0.0), estimator.E_range[0], estimator.E_range[1]))
        return H, E


class RandomPolicy(MeasurementPolicy):
    def __init__(self):
        super().__init__('random')

    def select(self, estimator: ParameterEstimator, iteration: int):
        H = np.random.uniform(*estimator.H_range)
        E = estimator.est_model.dispersion(H, 0.0) + np.random.uniform(-2, 2)
        return H, np.clip(E, estimator.E_range[0], estimator.E_range[1])


class LogGPPolicy(MeasurementPolicy):
    """Observation-driven Log-GP exploration via the library GP stack."""

    def __init__(self):
        super().__init__('log_gp')
        self.explorer: AgnosticExplorer | None = None
        self._synced_measurements = 0

    def reset(self, estimator: ParameterEstimator):
        bounds = np.array([
            [estimator.H_range[0], estimator.H_range[1]],
            [estimator.E_range[0], estimator.E_range[1]],
        ], dtype=float)
        self.explorer = AgnosticExplorer(bounds, use_log_gp=True)
        self._synced_measurements = 0
        self._sync_observations(estimator)

    def _sync_observations(self, estimator: ParameterEstimator) -> None:
        if self.explorer is None:
            return
        while self._synced_measurements < len(estimator.measurements):
            meas = estimator.measurements[self._synced_measurements]
            self.explorer.add_observation(
                np.array([meas.H, meas.E], dtype=float),
                float(meas.I),
                float(meas.sigma),
            )
            self._synced_measurements += 1

    def _is_near_existing(self, estimator: ParameterEstimator, H: float, E: float) -> bool:
        for meas in estimator.measurements:
            dh = abs(H - meas.H)
            dE = abs(E - meas.E)
            if dh <= 0.015 and dE <= 1.0:
                return True
        return False

    def select(self, estimator: ParameterEstimator, iteration: int):
        self._sync_observations(estimator)
        assert self.explorer is not None

        for _ in range(25):
            point = self.explorer.suggest_next(acquisition='variance')
            H = float(np.clip(point[0], estimator.H_range[0], estimator.H_range[1]))
            E = float(np.clip(point[1], estimator.E_range[0], estimator.E_range[1]))
            if not self._is_near_existing(estimator, H, E):
                return H, E

        return tuple(map(float, self.explorer.suggest_initial()))


class TASPolicy(MeasurementPolicy):
    def __init__(self):
        super().__init__('tas_ai')

    def select(self, estimator: ParameterEstimator, iteration: int):
        return estimator.suggest_tas_point()


def create_policies() -> Dict[str, MeasurementPolicy]:
    return {
        'grid': GridPolicy(),
        'random': RandomPolicy(),
        'log_gp': LogGPPolicy(),
        'tas_ai': TASPolicy(),
    }


# ---------------------------------------------------------------------------
# Runner utilities
# ---------------------------------------------------------------------------

def run_policy(policy: MeasurementPolicy,
               max_measurements: int,
               threshold: float,
               config: BenchmarkConfig) -> Dict:
    estimator = ParameterEstimator(config=config)
    estimator.add_initial_points()
    policy.reset(estimator)

    rms_history = estimator.rms_history.copy()
    time_history = estimator.time_history.copy()
    measurement_count = len(estimator.measurements)
    converge_iter = None
    converge_time_s = None

    while measurement_count < max_measurements:
        H, E = policy.select(estimator, measurement_count)
        estimator.measure(H, E, source=policy.name)
        estimator.fit_model()
        rms_history.append(estimator.current_rms_error())
        time_history.append(float(estimator.total_time_s))
        measurement_count += 1
        if rms_history[-1] <= threshold and converge_iter is None:
            converge_iter = measurement_count
            converge_time_s = float(estimator.total_time_s)

    summary = {
        'policy': policy.name,
        'history': rms_history,
        'time_history_s': time_history,
        'converged_at': converge_iter,
        'converged_time_s': converge_time_s,
        'final_rms': rms_history[-1],
        'final_time_s': float(estimator.total_time_s),
        'estimator': estimator,
    }
    return summary


def plot_summary(results: Dict[str, Dict], figure_path: Path):
    _set_plot_style()
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))

    # RMS Panel
    ax0 = axes[0, 0]
    for name, result in results.items():
        history = result['history']
        x = np.arange(len(history)) + 1  # start at measurement 1
        ax0.plot(x, history, label=name)
    ax0.set_xlabel('Measurement #')
    ax0.set_ylabel('RMS parameter error (meV)')
    ax0.set_title('(a) RMS parameter error vs measurement count', loc='left', fontweight='bold')
    ax0.legend()

    # Measurement path for TAS-AI
    tas_est = results['tas_ai']['estimator']
    ax1 = axes[0, 1]
    for m in tas_est.measurements:
        color = '#1f77b4' if m.source == 'tas_ai' else '#ff7f0e'
        ax1.scatter(m.H, m.E, c=color, s=35, edgecolors='black', linewidths=0.4)
    ax1.set_xlabel('[H H 0] (r.l.u.)')
    ax1.set_ylabel('E (meV)')
    ax1.set_title('(b) TAS-AI measurement path', loc='left', fontweight='bold')

    # Parameter traces
    ax2 = axes[1, 0]
    ax2.plot(tas_est.J1_history, label='J1')
    ax2.plot(tas_est.J2_history, label='J2')
    ax2.plot(tas_est.D_history, label='D')
    ax2.axhline(tas_est.true_params[0], linestyle='--', color='C0')
    ax2.axhline(tas_est.true_params[1], linestyle='--', color='C1')
    ax2.axhline(tas_est.true_params[2], linestyle='--', color='C2')
    ax2.set_xlabel('Iteration')
    ax2.set_ylabel('Parameter value (meV)')
    ax2.set_title('(c) TAS-AI parameter convergence', loc='left', fontweight='bold')
    ax2.legend()

    # Chi^2 panel
    ax3 = axes[1, 1]
    ax3.plot(tas_est.chi2_history, color='C3')
    ax3.set_xlabel('Iteration')
    ax3.set_ylabel('χ² / N')
    ax3.set_yscale('log')
    ax3.set_title('(d) Reduced χ² for TAS-AI fits', loc='left', fontweight='bold')

    plt.tight_layout()
    fig.savefig(figure_path, dpi=300)
    plt.close(fig)
    print(f"Wrote {figure_path}")


# ---------------------------------------------------------------------------
# Main CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description='Parameter determination benchmark')
    parser.add_argument('--max-measurements', type=int, default=80)
    parser.add_argument('--threshold', type=float, default=0.2)
    parser.add_argument('--true-j1', type=float, default=5.0)
    parser.add_argument('--true-j2', type=float, default=0.8)
    parser.add_argument('--true-d', type=float, default=0.15)
    parser.add_argument('--prior-j1', type=float, default=4.4)
    parser.add_argument('--prior-j2', type=float, default=0.25)
    parser.add_argument('--prior-d', type=float, default=0.08)
    parser.add_argument('--init-points', type=int, default=4)
    parser.add_argument('--init-mode', choices=['prior_branch', 'prior_triplet'], default='prior_triplet')
    parser.add_argument('--count-time', type=float, default=30.0)
    parser.add_argument('--use-motion', action='store_true')
    parser.add_argument('--figure', type=Path, required=True,
                        help='Path for the multi-panel figure output')
    parser.add_argument('--summary-json', type=Path,
                        help='Optional path to write JSON summary for Table 1')
    args = parser.parse_args()
    np.random.seed(0)

    config = BenchmarkConfig(
        true_J1=args.true_j1,
        true_J2=args.true_j2,
        true_D=args.true_d,
        prior_J1=args.prior_j1,
        prior_J2=args.prior_j2,
        prior_D=args.prior_d,
        init_points=args.init_points,
        init_mode=args.init_mode,
        count_time=args.count_time,
        use_motion=args.use_motion,
    )

    results: Dict[str, Dict] = {}
    for name, policy in create_policies().items():
        print(f"Running policy: {name}")
        results[name] = run_policy(policy, args.max_measurements, args.threshold, config=config)

    plot_summary(results, args.figure)

    summary = {}
    for name, result in results.items():
        summary[name] = {
            'converged_at': result['converged_at'],
            'converged_time_s': result['converged_time_s'],
            'final_rms': result['final_rms'],
            'final_time_s': result['final_time_s'],
            'history': result['history'],
            'time_history_s': result['time_history_s'],
        }

    if args.summary_json:
        args.summary_json.parent.mkdir(parents=True, exist_ok=True)
        with open(args.summary_json, 'w', encoding='utf-8') as fh:
            json.dump({
                'threshold': args.threshold,
                'max_measurements': args.max_measurements,
                'config': config.__dict__,
                'policies': summary
            }, fh, indent=2)
        print(f"Wrote summary to {args.summary_json}")


if __name__ == '__main__':
    main()
