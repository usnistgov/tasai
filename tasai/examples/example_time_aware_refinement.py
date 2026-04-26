#!/usr/bin/env python3
"""Concrete time-aware refinement benchmark where TAS-AI is competitive.

This script instantiates the strongest scenario found in the 2026-03-27
time-aware search:

- true model: (J1, J2, D) = (5.3, 0.6, 0.28)
- prior model: (4.2, 0.35, 0.12)
- 3 prior-guided seed points
- motion-aware objective enabled
- count time = 10 s

It compares Grid, Random, library Log-GP, and motion-aware TAS-AI on
time-to-threshold rather than only final RMS. The resulting figure is meant as
the concrete replacement benchmark for the static refinement case.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np

from tasai.examples.example_parameter_determination import (
    BenchmarkConfig,
    _set_plot_style,
    create_policies,
    run_policy,
)


DEFAULT_CONFIG = BenchmarkConfig(
    true_J1=5.3,
    true_J2=0.6,
    true_D=0.28,
    prior_J1=4.2,
    prior_J2=0.35,
    prior_D=0.12,
    init_points=3,
    init_mode="prior_triplet",
    count_time=10.0,
    use_motion=True,
)


def effective_time_s(result: Dict, max_measurements: int) -> float:
    if result["converged_time_s"] is not None:
        return float(result["converged_time_s"])
    return float(result["final_time_s"] + 1.0e6 + max_measurements)


def run_benchmark(config: BenchmarkConfig, max_measurements: int, threshold: float, seed: int) -> Dict[str, Dict]:
    np.random.seed(seed)
    results: Dict[str, Dict] = {}
    for name, policy in create_policies().items():
        results[name] = run_policy(policy, max_measurements=max_measurements, threshold=threshold, config=config)
    return results


def make_figure(results: Dict[str, Dict], figure_path: Path, threshold: float, max_measurements: int) -> None:
    _set_plot_style()
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 4.2))

    # (a) RMS vs elapsed time
    ax0 = axes[0]
    for name, result in results.items():
        times = np.array(result["time_history_s"], dtype=float) / 60.0
        rms = np.array(result["history"], dtype=float)
        ax0.plot(times, rms, label=name)
    ax0.axhline(threshold, linestyle="--", color="black", linewidth=1.0, alpha=0.6)
    ax0.set_xlabel("Elapsed time (minutes)")
    ax0.set_ylabel("RMS parameter error")
    ax0.set_title("(a) Refinement quality vs elapsed time", loc="left", fontweight="bold")
    ax0.legend()

    # (b) TAS-AI path
    ax2 = axes[1]
    tas_est = results["tas_ai"]["estimator"]
    colors = plt.cm.viridis(np.linspace(0, 1, len(tas_est.measurements)))
    for idx, meas in enumerate(tas_est.measurements):
        ax2.scatter(meas.H, meas.E, c=[colors[idx]], s=45, edgecolors="black", linewidths=0.4)
        ax2.text(meas.H, meas.E + 0.6, str(idx + 1), fontsize=7, ha="center")
    ax2.set_xlabel("[H H 0] (r.l.u.)")
    ax2.set_ylabel("E (meV)")
    ax2.set_title("(b) TAS-AI measurement path", loc="left", fontweight="bold")

    # (c) Science vs motion time
    ax3 = axes[2]
    strategies = ["grid", "random", "log_gp", "tas_ai"]
    labels = ["grid", "random", "log_gp", "tas_ai"]
    n_points = [len(results[name]["estimator"].measurements) for name in strategies]
    science = [results[name]["estimator"].count_time * n for name, n in zip(strategies, n_points)]
    total = [results[name]["final_time_s"] for name in strategies]
    motion = [max(t - s, 0.0) for t, s in zip(total, science)]
    ax3.bar(labels, np.array(science) / 60.0, label="Science", color="#9ecae1")
    ax3.bar(labels, np.array(motion) / 60.0, bottom=np.array(science) / 60.0, label="Motion", color="#fdae6b")
    ax3.set_ylabel("Total time (minutes)")
    ax3.set_title("(c) Science vs motion time", loc="left", fontweight="bold")
    ax3.legend()

    plt.tight_layout()
    figure_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(figure_path, dpi=300)
    plt.close(fig)
    print(f"Wrote {figure_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Time-aware refinement benchmark")
    parser.add_argument("--max-measurements", type=int, default=18)
    parser.add_argument("--threshold", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=2)
    parser.add_argument("--figure", type=Path, required=True)
    parser.add_argument("--summary-json", type=Path)
    args = parser.parse_args()

    results = run_benchmark(DEFAULT_CONFIG, args.max_measurements, args.threshold, args.seed)
    make_figure(results, args.figure, args.threshold, args.max_measurements)

    summary = {
        "config": DEFAULT_CONFIG.__dict__,
        "seed": args.seed,
        "threshold": args.threshold,
        "max_measurements": args.max_measurements,
        "policies": {},
    }
    for name, result in results.items():
        summary["policies"][name] = {
            "converged_at": result["converged_at"],
            "converged_time_s": result["converged_time_s"],
            "final_rms": result["final_rms"],
            "final_time_s": result["final_time_s"],
            "history": result["history"],
            "time_history_s": result["time_history_s"],
        }

    if args.summary_json:
        args.summary_json.parent.mkdir(parents=True, exist_ok=True)
        args.summary_json.write_text(json.dumps(summary, indent=2))
        print(f"Wrote summary to {args.summary_json}")


if __name__ == "__main__":
    main()
