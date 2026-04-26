#!/usr/bin/env python3
"""
Controlled motor-motion scheduling study used for Figure 3 / Table 2.

This script creates a fixed list of measurement candidates along the HHL
dispersion and evaluates three scheduling strategies:

1. Random ordering
2. Nearest-neighbour ordering
3. Motion-aware TAS-AI ordering (info gain divided by count+move time)

For each strategy it records the move time, science time, and cumulative
wall-clock time. A four-panel figure is produced showing the TAS-AI sequence,
move-time comparison, stacked total time, and cumulative time traces. The JSON
summary is ingested when building Table 2 in the manuscript.

Important: this is a fixed-candidate ordering study, not a fair benchmark of
full adaptive discovery/planning policies. The candidate set is precomputed on
the model dispersion so the comparison isolates scheduling effects after
scientifically relevant points have already been identified.

Run:
    python example_with_motor_motion.py \
        --figure tasai_paper_clean/paper/figure3_motion.png \
        --summary-json tasai_review/logs/motion_summary.json
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence

import matplotlib.pyplot as plt
import numpy as np

from tasai.sunny import SquareLatticeFM
from tasai.instrument.motors import SimplifiedMotorModel, MotionAwareAcquisition


COUNT_TIME = 60.0  # seconds
SEED = 0


def _set_plot_style() -> None:
    for style in ("seaborn-v0_8-whitegrid", "seaborn-whitegrid", "default"):
        try:
            plt.style.use(style)
            return
        except OSError:
            continue


@dataclass
class Candidate:
    H: float
    E: float
    label: str


def create_candidates(model: SquareLatticeFM, n_points: int = 30) -> List[Candidate]:
    Hs = np.linspace(0.05, 0.45, n_points)
    candidates = []
    for H in Hs:
        E = model.dispersion(H, 0.0)
        candidates.append(Candidate(H=H, E=E, label=f"H={H:.2f}"))
    return candidates


def info_gain(model: SquareLatticeFM, candidate: Candidate) -> float:
    disp = model.dispersion(candidate.H, 0.0)
    dist = abs(candidate.E - disp)
    intensity = np.exp(-dist ** 2 / 1.0)
    j2 = 1 + 0.5 * np.sin(4 * np.pi * candidate.H) ** 2
    return intensity * j2


def simulate_order(order: Sequence[Candidate], motor: SimplifiedMotorModel) -> Dict:
    move_times = []
    for cand in order:
        move = motor.move_time(cand.H, cand.H, 0.0, cand.E)
        motor.move_to(cand.H, cand.H, 0.0, cand.E)
        move_times.append(move)
    move_times = np.array(move_times)
    count_times = np.full(len(order), COUNT_TIME)
    cumulative = np.cumsum(move_times + count_times)
    return {
        'order': order,
        'move_times': move_times,
        'count_times': count_times,
        'cumulative_time': cumulative,
        'total_move': float(move_times.sum()),
        'total_science': float(count_times.sum()),
    }


def random_strategy(candidates: List[Candidate], motor: SimplifiedMotorModel) -> Dict:
    rng = np.random.default_rng(SEED)
    order = list(candidates)
    rng.shuffle(order)
    return simulate_order(order, motor)


def nearest_strategy(candidates: List[Candidate], motor: SimplifiedMotorModel) -> Dict:
    remaining = list(candidates)
    order = []
    current = remaining.pop(0)
    order.append(current)
    while remaining:
        next_idx = min(
            range(len(remaining)),
            key=lambda i: np.hypot(current.H - remaining[i].H, current.E - remaining[i].E),
        )
        current = remaining.pop(next_idx)
        order.append(current)
    return simulate_order(order, motor)


def tas_strategy(candidates: List[Candidate], motor: SimplifiedMotorModel,
                 model: SquareLatticeFM) -> Dict:
    remaining = list(candidates)
    acquisition = MotionAwareAcquisition(motor_model=motor, eta=0.7, count_time=COUNT_TIME)
    order = []
    while remaining:
        infos = np.array([info_gain(model, c) for c in remaining])
        cand_array = np.array([[c.H, c.H, 0.0, c.E] for c in remaining])
        scores = acquisition.score_batch(cand_array, infos)
        best_idx = int(np.argmax(scores))
        best = remaining.pop(best_idx)
        order.append(best)
        # Update the planning motor so subsequent acquisition scores see the
        # route state accumulated so far.
        motor.move_time(best.H, best.H, 0.0, best.E)
        motor.move_to(best.H, best.H, 0.0, best.E)
    # Evaluate the planned route from a fresh initial motor state, just like
    # the random and nearest-neighbour baselines.
    return simulate_order(order, SimplifiedMotorModel())


def build_orders(model: SquareLatticeFM, candidates: List[Candidate]) -> Dict[str, Dict]:
    orders = {}
    # Each strategy needs its own motor instance because move_time depends on state.
    orders['random'] = random_strategy(candidates, SimplifiedMotorModel())
    orders['nearest'] = nearest_strategy(candidates, SimplifiedMotorModel())
    orders['tas_ai'] = tas_strategy(candidates, SimplifiedMotorModel(), model)
    return orders


def make_figure(orders: Dict[str, Dict], figure_path: Path):
    _set_plot_style()
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))

    # Panel (a): TAS-AI measurement sequence
    tas_order = orders['tas_ai']['order']
    ax0 = axes[0, 0]
    colors = plt.cm.viridis(np.linspace(0, 1, len(tas_order)))
    for idx, cand in enumerate(tas_order):
        ax0.scatter(cand.H, cand.E, c=[colors[idx]], s=50, edgecolors='black', linewidths=0.4)
        ax0.text(cand.H, cand.E + 0.8, str(idx + 1), fontsize=7, ha='center')
    ax0.set_xlabel('[H H 0] (r.l.u.)')
    ax0.set_ylabel('E (meV)')
    ax0.set_title('(a) TAS-AI motion-aware sequence', loc='left', fontweight='bold')

    # Panel (b): Move-time comparison
    ax1 = axes[0, 1]
    strategies = ['random', 'nearest', 'tas_ai']
    move_totals = [orders[s]['total_move'] / 60 for s in strategies]  # minutes
    ax1.bar(strategies, move_totals, color=['#ff7f0e', '#1f77b4', '#2ca02c'])
    ax1.set_ylabel('Total move time (minutes)')
    ax1.set_title('(b) Move time per strategy', loc='left', fontweight='bold')

    # Panel (c): Stacked science + motion
    ax2 = axes[1, 0]
    science = [orders[s]['total_science'] / 60 for s in strategies]
    ax2.bar(strategies, science, label='Science', color='#9edae5')
    ax2.bar(strategies, move_totals, bottom=science, label='Motion', color='#ffbb78')
    ax2.set_ylabel('Total time (minutes)')
    ax2.set_title('(c) Science vs motion time', loc='left', fontweight='bold')
    ax2.legend()

    # Panel (d): Cumulative wall-clock time
    ax3 = axes[1, 1]
    for s, label in zip(strategies, ['Random', 'Nearest', 'TAS-AI']):
        cumulative = orders[s]['cumulative_time'] / 60
        ax3.plot(range(1, len(cumulative) + 1), cumulative, label=label)
    ax3.set_xlabel('Measurement #')
    ax3.set_ylabel('Cumulative time (minutes)')
    ax3.set_title('(d) Cumulative experiment duration', loc='left', fontweight='bold')
    ax3.legend()

    plt.tight_layout()
    fig.savefig(figure_path, dpi=300)
    plt.close(fig)
    print(f"Wrote {figure_path}")


def main():
    parser = argparse.ArgumentParser(description='Motor-aware scheduling benchmark')
    parser.add_argument('--figure', type=Path, required=True,
                        help='Output path for the multi-panel figure')
    parser.add_argument('--summary-json', type=Path,
                        help='Optional JSON path for Table 2 ingestion')
    args = parser.parse_args()

    np.random.seed(SEED)
    model = SquareLatticeFM(J1=5.0, J2=0.8, D=0.1)
    candidates = create_candidates(model)
    orders = build_orders(model, candidates)

    make_figure(orders, args.figure)

    summary = {}
    for name, result in orders.items():
        summary[name] = {
            'total_move_s': result['total_move'],
            'total_science_s': result['total_science'],
            'total_time_s': result['total_move'] + result['total_science'],
        }
    if args.summary_json:
        args.summary_json.parent.mkdir(parents=True, exist_ok=True)
        with open(args.summary_json, 'w', encoding='utf-8') as fh:
            json.dump(summary, fh, indent=2)
        print(f"Wrote summary to {args.summary_json}")


if __name__ == '__main__':
    main()
