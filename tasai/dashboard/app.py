#!/usr/bin/env python3
"""
TAS-AI Dashboard - Full-Featured Web Interface

A comprehensive real-time dashboard for monitoring and controlling autonomous
neutron scattering experiments with all advanced features.

Features:
- Live data visualization with updating plots
- Acquisition function selection (swappable during pause)
- Model comparison panel (ANDiE weights)
- Information gain tracking
- Measurement queue with batch operations
- 2D acquisition heatmap (click to add)
- Residuals and fit quality
- Stopping conditions with progress
- Keyboard shortcuts
- Toast notifications
- Pause/Resume controls
- Export capabilities

Run with:
    python app.py --port 8050 --debug

Dependencies:
    pip install dash dash-bootstrap-components plotly pandas numpy
"""

import dash
from dash import dcc, html, Input, Output, State, callback, ctx, ALL, MATCH
import dash_bootstrap_components as dbc
from dash.exceptions import PreventUpdate
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import pandas as pd
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional, Any, Tuple
from datetime import datetime
import json
import threading
import time
from enum import Enum
import logging
import uuid

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# =============================================================================
# Data Models
# =============================================================================

class ExperimentState(Enum):
    IDLE = "idle"
    RUNNING = "running"
    PAUSED = "paused"
    MEASURING = "measuring"
    COMPUTING = "computing"


@dataclass
class MeasurementPoint:
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    h: float = 0.0
    k: float = 0.0
    l: float = 0.0
    E: float = 0.0
    count_time: float = 60.0
    source: str = "ai"
    priority: int = 0
    selected: bool = False
    
    def to_dict(self):
        return asdict(self)


@dataclass 
class Measurement:
    point: MeasurementPoint
    intensity: float
    uncertainty: float
    timestamp: str
    duration: float
    fit_intensity: float = 0.0


@dataclass
class ModelWeight:
    name: str
    weight: float
    color: str


@dataclass
class Notification:
    id: str
    message: str
    level: str
    timestamp: float
    duration: int = 5000


@dataclass
class ExperimentConfig:
    acquisition_function: str = "HH"
    eta: float = 0.7
    n_forecast: int = 3
    min_count_time: float = 10.0
    max_count_time: float = 300.0
    poi_indices: List[int] = field(default_factory=lambda: [0, 1])
    max_time: float = 3600.0
    max_measurements: int = 100
    target_entropy: float = 0.5
    mcmc_burn: int = 500
    mcmc_steps: int = 500
    mcmc_pop: int = 8


# =============================================================================
# Experiment Backend
# =============================================================================

class ExperimentBackend:
    def __init__(self):
        self.state = ExperimentState.IDLE
        self.config = ExperimentConfig()
        self.measurements: List[Measurement] = []
        self.queue: List[MeasurementPoint] = []
        self.notifications: List[Notification] = []
        self.log_messages: List[Dict] = []
        self.start_time: Optional[float] = None
        
        # True physics
        self.true_Tc = 150.0
        self.true_beta = 0.325
        
        # Estimates
        self.estimated_Tc = 145.0
        self.estimated_Tc_std = 10.0
        self.estimated_beta = 0.35
        self.estimated_beta_std = 0.05
        
        # Model weights
        self.model_weights = [
            ModelWeight("Ising", 0.33, "#00b4d8"),
            ModelWeight("Weiss", 0.33, "#00f5a0"),
            ModelWeight("FirstOrder", 0.34, "#a855f7"),
        ]
        
        # Info tracking
        self.info_history: List[Tuple[float, float]] = []
        self.current_entropy = 5.0
        self.total_info_gain = 0.0
        
        # Acquisition data
        self.acquisition_scores_1d: Dict[str, np.ndarray] = {}
        self.acquisition_scores_2d: Optional[np.ndarray] = None
        self.q_range = np.linspace(0, 1.5, 50)
        self.e_range = np.linspace(0, 40, 50)
        
        self.lock = threading.Lock()
        self.running = False
        self.thread: Optional[threading.Thread] = None
    
    def start(self):
        with self.lock:
            if self.state not in [ExperimentState.IDLE, ExperimentState.PAUSED]:
                return
            self.state = ExperimentState.RUNNING
            self.start_time = self.start_time or time.time()
            self._log("Experiment started", "info")
            self._notify("Experiment started", "success")
        
        self.running = True
        self.thread = threading.Thread(target=self._run_loop, daemon=True)
        self.thread.start()
    
    def pause(self):
        with self.lock:
            if self.state == ExperimentState.RUNNING:
                self.running = False
                self.state = ExperimentState.PAUSED
                self._log("Experiment paused", "warning")
                self._notify("Paused - controls unlocked", "warning")
    
    def stop(self):
        with self.lock:
            self.running = False
            self.state = ExperimentState.IDLE
            self._log("Experiment stopped", "error")
            self._notify("Experiment stopped", "error")
    
    def reset(self):
        with self.lock:
            self.running = False
            self.state = ExperimentState.IDLE
            self.measurements = []
            self.queue = []
            self.info_history = []
            self.total_info_gain = 0.0
            self.current_entropy = 5.0
            self.start_time = None
            self.estimated_Tc = 145.0
            self.estimated_Tc_std = 10.0
            self.estimated_beta = 0.35
            self.estimated_beta_std = 0.05
            self.model_weights = [
                ModelWeight("Ising", 0.33, "#00b4d8"),
                ModelWeight("Weiss", 0.33, "#00f5a0"),
                ModelWeight("FirstOrder", 0.34, "#a855f7"),
            ]
            self._log("Experiment reset", "info")
            self._notify("Experiment reset", "info")
    
    def add_to_queue(self, point: MeasurementPoint):
        with self.lock:
            self.queue.append(point)
            self._log(f"Added T={point.E:.1f}K", "info")
            self._notify(f"Added T={point.E:.1f}K to queue", "info")
    
    def add_batch_to_queue(self, temps: List[float], count_time: float = 60.0):
        with self.lock:
            for T in temps:
                self.queue.append(MeasurementPoint(E=T, count_time=count_time, source="user"))
            self._log(f"Batch added {len(temps)} points", "info")
            self._notify(f"Added {len(temps)} points", "success")
    
    def clear_queue(self):
        with self.lock:
            count = len(self.queue)
            self.queue = []
            self._log("Queue cleared", "warning")
            self._notify(f"Cleared {count} points", "warning")
    
    def remove_selected(self):
        with self.lock:
            removed = sum(1 for p in self.queue if p.selected)
            self.queue = [p for p in self.queue if not p.selected]
            if removed:
                self._notify(f"Removed {removed} points", "info")
    
    def toggle_selection(self, point_id: str):
        with self.lock:
            for p in self.queue:
                if p.id == point_id:
                    p.selected = not p.selected
                    break
    
    def move_item(self, point_id: str, direction: int):
        with self.lock:
            for i, p in enumerate(self.queue):
                if p.id == point_id:
                    new_idx = max(0, min(len(self.queue) - 1, i + direction))
                    if new_idx != i:
                        self.queue.pop(i)
                        self.queue.insert(new_idx, p)
                    break
    
    def add_from_heatmap(self, q: float, e: float):
        point = MeasurementPoint(h=q, k=0, l=0, E=e, count_time=60.0, source="user", priority=1)
        self.add_to_queue(point)
    
    def update_config(self, **kwargs):
        with self.lock:
            for key, value in kwargs.items():
                if hasattr(self.config, key):
                    setattr(self.config, key, value)
    
    def get_notifications(self) -> List[Dict]:
        with self.lock:
            now = time.time()
            active = [n for n in self.notifications if now - n.timestamp < n.duration / 1000]
            self.notifications = []
            return [{"id": n.id, "message": n.message, "level": n.level, "duration": n.duration} for n in active]
    
    def get_state(self) -> Dict:
        with self.lock:
            elapsed = time.time() - self.start_time if self.start_time else 0
            info_rate = self.total_info_gain / (elapsed / 60) if elapsed > 0 else 0
            chi_sq = self._calc_chi_squared()
            
            return {
                "state": self.state.value,
                "n_measurements": len(self.measurements),
                "elapsed_time": elapsed,
                "queue_length": len(self.queue),
                "estimated_Tc": self.estimated_Tc,
                "estimated_Tc_std": self.estimated_Tc_std,
                "estimated_beta": self.estimated_beta,
                "estimated_beta_std": self.estimated_beta_std,
                "acquisition_function": self.config.acquisition_function,
                "eta": self.config.eta,
                "n_forecast": self.config.n_forecast,
                "total_info_gain": self.total_info_gain,
                "current_entropy": self.current_entropy,
                "info_rate": info_rate,
                "chi_squared": chi_sq,
                "max_time": self.config.max_time,
                "max_measurements": self.config.max_measurements,
                "target_entropy": self.config.target_entropy,
            }
    
    def get_model_weights(self) -> List[Dict]:
        with self.lock:
            return [{"name": m.name, "weight": m.weight, "color": m.color} for m in self.model_weights]
    
    def get_measurements_df(self) -> pd.DataFrame:
        with self.lock:
            if not self.measurements:
                return pd.DataFrame(columns=["E", "I", "sigma", "source", "timestamp", "fit_I", "residual"])
            data = []
            for m in self.measurements:
                residual = (m.intensity - m.fit_intensity) / m.uncertainty if m.uncertainty > 0 else 0
                data.append({"E": m.point.E, "I": m.intensity, "sigma": m.uncertainty,
                           "source": m.point.source, "timestamp": m.timestamp,
                           "fit_I": m.fit_intensity, "residual": residual})
            return pd.DataFrame(data)
    
    def get_queue_list(self) -> List[Dict]:
        with self.lock:
            return [{"id": p.id, "E": p.E, "count_time": p.count_time, "source": p.source,
                    "priority": p.priority, "selected": p.selected} for p in self.queue]
    
    def get_info_history(self) -> Tuple[List[float], List[float]]:
        with self.lock:
            if not self.info_history:
                return [], []
            return [t for t, _ in self.info_history], [g for _, g in self.info_history]
    
    def get_log_messages(self, n: int = 50) -> List[Dict]:
        with self.lock:
            return self.log_messages[-n:]
    
    def get_acquisition_2d(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        with self.lock:
            if self.acquisition_scores_2d is None:
                self._update_acquisition_2d()
            return self.q_range, self.e_range, self.acquisition_scores_2d
    
    def _notify(self, message: str, level: str = "info", duration: int = 5000):
        self.notifications.append(Notification(str(uuid.uuid4())[:8], message, level, time.time(), duration))
    
    def _log(self, message: str, level: str = "info"):
        self.log_messages.append({"time": datetime.now().strftime("%H:%M:%S"), "message": message, "level": level})
    
    def _calc_chi_squared(self) -> float:
        if len(self.measurements) < 2:
            return 0.0
        chi_sq = sum(((m.intensity - m.fit_intensity) / m.uncertainty) ** 2 
                    for m in self.measurements if m.uncertainty > 0)
        return chi_sq / max(1, len(self.measurements) - 2)
    
    def _update_acquisition_2d(self):
        Q, E = np.meshgrid(self.q_range, self.e_range)
        dispersion = 10 * (1 - np.cos(2 * np.pi * Q))
        scores = np.exp(-((E - dispersion) / 5) ** 2)
        for m in self.measurements:
            scores *= (1 - 0.5 * np.exp(-((Q - m.point.h) ** 2 + (E - m.point.E) ** 2) / 25))
        self.acquisition_scores_2d = scores
    
    def _run_loop(self):
        while self.running:
            try:
                with self.lock:
                    elapsed = time.time() - self.start_time if self.start_time else 0
                    if elapsed > self.config.max_time:
                        self._log("Max time reached", "warning")
                        self._notify("Max time reached", "success")
                        self.state = ExperimentState.IDLE
                        self.running = False
                        break
                    if len(self.measurements) >= self.config.max_measurements:
                        self._log("Max measurements reached", "warning")
                        self._notify("Max measurements reached", "success")
                        self.state = ExperimentState.IDLE
                        self.running = False
                        break
                    if self.current_entropy < self.config.target_entropy:
                        self._log("Target entropy reached", "success")
                        self._notify("Target entropy reached!", "success")
                        self.state = ExperimentState.IDLE
                        self.running = False
                        break
                
                point = self._get_next_point()
                if point is None:
                    time.sleep(0.5)
                    continue
                
                with self.lock:
                    self.state = ExperimentState.MEASURING
                    self._log(f"Measuring T={point.E:.1f}K", "info")
                
                time.sleep(min(point.count_time / 30, 1.5))
                
                intensity, uncertainty = self._simulate_measurement(point)
                fit_intensity = self._compute_fit(point.E)
                
                measurement = Measurement(point, intensity, uncertainty,
                                         datetime.now().strftime("%H:%M:%S"), point.count_time, fit_intensity)
                
                with self.lock:
                    self.measurements.append(measurement)
                    self._log(f"I={intensity:.3f}±{uncertainty:.3f}", "info")
                    self._notify(f"T={point.E:.1f}K: I={intensity:.3f}", "success", 3000)
                
                with self.lock:
                    self.state = ExperimentState.COMPUTING
                
                time.sleep(0.3)
                self._update_estimates()
                self._update_model_weights()
                self._update_acquisition_2d()
                
                with self.lock:
                    self.state = ExperimentState.RUNNING
                    
            except Exception as e:
                logger.exception("Error in experiment loop")
                self._log(f"Error: {str(e)}", "error")
                time.sleep(1)
    
    def _get_next_point(self) -> Optional[MeasurementPoint]:
        with self.lock:
            if self.queue:
                self.queue.sort(key=lambda p: -p.priority)
                return self.queue.pop(0)
            return self._suggest_point()
    
    def _suggest_point(self) -> MeasurementPoint:
        T_range = np.linspace(80, 220, 100)
        if self.config.acquisition_function == "HH":
            scores = np.exp(-((T_range - self.estimated_Tc) / 30) ** 2) * (1 + 0.3 * np.random.rand(len(T_range)))
        elif self.config.acquisition_function == "ANDiE":
            scores = np.exp(-((T_range - self.estimated_Tc) / 20) ** 2) + 0.4 * np.exp(-((T_range - self.true_Tc + 15) / 15) ** 2)
        elif self.config.acquisition_function == "Uncertainty":
            scores = np.ones_like(T_range)
            for m in self.measurements:
                scores *= (1 - 0.4 * np.exp(-((T_range - m.point.E) / 8) ** 2))
        else:
            scores = np.exp(-((T_range - self.estimated_Tc) / 25) ** 2)
        
        self.acquisition_scores_1d = {"T": T_range, "scores": scores}
        scores = scores ** self.config.eta
        best_idx = np.argmax(scores)
        return MeasurementPoint(E=T_range[best_idx] + np.random.uniform(-2, 2), source="ai", count_time=60.0)
    
    def _simulate_measurement(self, point: MeasurementPoint) -> Tuple[float, float]:
        T = point.E
        M = (1 - T / self.true_Tc) ** self.true_beta if T < self.true_Tc else 0
        I_true = M ** 2 + 0.01
        counts = np.random.poisson(max(1, int(I_true * 1000 * point.count_time)))
        return counts / (1000 * point.count_time), max(np.sqrt(counts) / (1000 * point.count_time), 0.001)
    
    def _compute_fit(self, T: float) -> float:
        M = (1 - T / self.estimated_Tc) ** self.estimated_beta if T < self.estimated_Tc else 0
        return M ** 2 + 0.01
    
    def _update_estimates(self):
        with self.lock:
            if len(self.measurements) < 3:
                return
            n = len(self.measurements)
            self.estimated_Tc_std = max(0.5, 8.0 / np.sqrt(n))
            self.estimated_beta_std = max(0.005, 0.04 / np.sqrt(n))
            self.estimated_Tc = 0.92 * self.estimated_Tc + 0.08 * self.true_Tc
            self.estimated_beta = 0.92 * self.estimated_beta + 0.08 * self.true_beta
            for m in self.measurements:
                m.fit_intensity = self._compute_fit(m.point.E)
            old_entropy = self.current_entropy
            self.current_entropy = max(0.1, self.estimated_Tc_std * self.estimated_beta_std * 10)
            self.total_info_gain += max(0, old_entropy - self.current_entropy)
            elapsed = time.time() - self.start_time if self.start_time else 0
            self.info_history.append((elapsed / 60, self.total_info_gain))
    
    def _update_model_weights(self):
        with self.lock:
            if len(self.measurements) < 5:
                return
            n = len(self.measurements)
            ising_w = 0.33 + 0.5 * (1 - np.exp(-n / 20))
            remaining = 1 - ising_w
            self.model_weights[0].weight = ising_w
            self.model_weights[1].weight = remaining * 0.6
            self.model_weights[2].weight = remaining * 0.4


backend = ExperimentBackend()


# =============================================================================
# Dash App
# =============================================================================

CUSTOM_CSS = """
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;600&family=IBM+Plex+Sans:wght@400;500;600&display=swap');

:root {
    --bg-primary: #0a0e14;
    --bg-secondary: #12161c;
    --bg-card: #151a22;
    --accent-blue: #00b4d8;
    --accent-green: #00f5a0;
    --accent-orange: #ff9f43;
    --accent-red: #ff6b6b;
    --accent-purple: #a855f7;
    --text-primary: #e6edf3;
    --text-secondary: #8b949e;
    --text-muted: #6e7681;
    --border-color: #2d3748;
}

body { font-family: 'IBM Plex Sans', sans-serif; background: var(--bg-primary); color: var(--text-primary); }

.dashboard-header {
    background: linear-gradient(180deg, var(--bg-secondary) 0%, var(--bg-primary) 100%);
    border-bottom: 1px solid var(--border-color);
    padding: 0.75rem 1.5rem;
    position: sticky; top: 0; z-index: 100;
}

.logo-text {
    font-family: 'JetBrains Mono', monospace;
    font-weight: 600; font-size: 1.25rem;
    background: linear-gradient(135deg, var(--accent-blue), #00d4ff);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
}

.status-indicator {
    display: inline-flex; align-items: center; gap: 0.5rem;
    padding: 0.25rem 0.75rem; border-radius: 9999px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.7rem; font-weight: 500;
    text-transform: uppercase; letter-spacing: 0.05em;
}

.status-idle { background: rgba(110, 118, 129, 0.15); color: var(--text-muted); border: 1px solid var(--text-muted); }
.status-running { background: rgba(0, 245, 160, 0.1); color: var(--accent-green); border: 1px solid var(--accent-green); animation: pulse 2s infinite; }
.status-paused { background: rgba(255, 159, 67, 0.1); color: var(--accent-orange); border: 1px solid var(--accent-orange); }
.status-measuring { background: rgba(0, 180, 216, 0.1); color: var(--accent-blue); border: 1px solid var(--accent-blue); animation: pulse 1s infinite; }
.status-computing { background: rgba(168, 85, 247, 0.1); color: var(--accent-purple); border: 1px solid var(--accent-purple); }

@keyframes pulse { 0%, 100% { opacity: 1; } 50% { opacity: 0.6; } }

.card-panel {
    background: var(--bg-card); border: 1px solid var(--border-color);
    border-radius: 8px; padding: 1rem; height: 100%;
}

.card-title {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.65rem; font-weight: 600;
    text-transform: uppercase; letter-spacing: 0.12em;
    color: var(--text-muted);
    margin-bottom: 0.75rem; padding-bottom: 0.5rem;
    border-bottom: 1px solid var(--border-color);
}

.metric-value { font-family: 'JetBrains Mono', monospace; font-size: 1.4rem; font-weight: 600; color: var(--accent-blue); }
.metric-value-sm { font-size: 1rem; }
.metric-label { font-size: 0.65rem; color: var(--text-muted); text-transform: uppercase; letter-spacing: 0.08em; margin-top: 0.25rem; }
.metric-uncertainty { font-family: 'JetBrains Mono', monospace; font-size: 0.85rem; color: var(--text-secondary); }

.queue-item {
    display: flex; justify-content: space-between; align-items: center;
    padding: 0.5rem 0.75rem; background: rgba(26, 32, 41, 0.8);
    border-radius: 6px; margin-bottom: 0.35rem;
    font-family: 'JetBrains Mono', monospace; font-size: 0.75rem;
    cursor: pointer; transition: all 0.15s; border: 1px solid transparent;
}
.queue-item:hover { background: rgba(0, 180, 216, 0.1); }
.queue-item-selected { background: rgba(0, 180, 216, 0.15) !important; border-color: var(--accent-blue) !important; }
.queue-item-ai { border-left: 3px solid var(--accent-blue); }
.queue-item-user { border-left: 3px solid var(--accent-green); }

.priority-badge { font-size: 0.6rem; padding: 0.1rem 0.4rem; border-radius: 4px; margin-left: 0.5rem; }
.priority-high { background: rgba(255, 159, 67, 0.2); color: var(--accent-orange); }
.priority-urgent { background: rgba(255, 107, 107, 0.2); color: var(--accent-red); }

.log-entry { font-family: 'JetBrains Mono', monospace; font-size: 0.7rem; padding: 0.2rem 0; border-bottom: 1px solid rgba(45, 55, 72, 0.5); display: flex; gap: 0.5rem; }
.log-time { color: var(--text-muted); min-width: 55px; }
.log-info { color: var(--text-primary); }
.log-warning { color: var(--accent-orange); }
.log-error { color: var(--accent-red); }
.log-success { color: var(--accent-green); }

.control-btn { font-family: 'JetBrains Mono', monospace; font-weight: 500; font-size: 0.75rem; text-transform: uppercase; letter-spacing: 0.05em; border-radius: 6px; }
.control-btn:disabled { opacity: 0.4; }

.input-dark { background: rgba(26, 32, 41, 0.8) !important; border-color: var(--border-color) !important; color: var(--text-primary) !important; font-family: 'JetBrains Mono', monospace; font-size: 0.85rem; }
.input-dark:focus { border-color: var(--accent-blue) !important; box-shadow: 0 0 0 2px rgba(0, 180, 216, 0.15) !important; }

.notification-container { position: fixed; top: 70px; right: 20px; z-index: 1050; max-width: 320px; }
.notification-toast { background: var(--bg-card) !important; border: 1px solid var(--border-color); border-radius: 8px; margin-bottom: 0.5rem; animation: slideIn 0.3s ease-out; }
.notification-info { border-left: 4px solid var(--accent-blue); }
.notification-success { border-left: 4px solid var(--accent-green); }
.notification-warning { border-left: 4px solid var(--accent-orange); }
.notification-error { border-left: 4px solid var(--accent-red); }
@keyframes slideIn { from { transform: translateX(100%); opacity: 0; } to { transform: translateX(0); opacity: 1; } }

.model-bar { height: 24px; border-radius: 4px; margin-bottom: 0.35rem; display: flex; align-items: center; padding: 0 0.5rem; font-family: 'JetBrains Mono', monospace; font-size: 0.7rem; transition: width 0.5s ease; }

kbd { background: rgba(26, 32, 41, 0.8); border: 1px solid var(--border-color); border-radius: 4px; padding: 0.1rem 0.4rem; font-family: 'JetBrains Mono', monospace; font-size: 0.65rem; color: var(--text-secondary); }

.help-btn { width: 28px; height: 28px; border-radius: 50%; display: flex; align-items: center; justify-content: center; font-size: 0.85rem; background: rgba(26, 32, 41, 0.8); border: 1px solid var(--border-color); color: var(--text-secondary); cursor: pointer; }
.help-btn:hover { background: var(--accent-blue); color: white; border-color: var(--accent-blue); }

::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: rgba(26, 32, 41, 0.8); }
::-webkit-scrollbar-thumb { background: var(--border-color); border-radius: 3px; }
"""

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.DARKLY], suppress_callback_exceptions=True)

app.index_string = '''<!DOCTYPE html><html><head>{%metas%}<title>TAS-AI Dashboard</title>{%favicon%}{%css%}<style>''' + CUSTOM_CSS + '''</style></head><body>{%app_entry%}<footer>{%config%}{%scripts%}{%renderer%}</footer></body></html>'''

def create_header():
    return html.Div([
        dbc.Row([
            dbc.Col([html.Span("TAS-AI", className="logo-text"), html.Span(" Dashboard", className="text-muted ms-1", style={"fontSize": "0.9rem"})], width="auto"),
            dbc.Col([html.Div(id="status-indicator", className="status-indicator status-idle")], width="auto", className="d-flex align-items-center"),
            dbc.Col([html.Span(id="elapsed-time", className="me-3", style={"fontFamily": "'JetBrains Mono', monospace", "fontSize": "0.85rem"}),
                    html.Span(id="measurement-count", style={"fontFamily": "'JetBrains Mono', monospace", "fontSize": "0.85rem"})], className="d-flex align-items-center"),
            dbc.Col([html.Button("?", id="btn-help", className="help-btn me-2", title="Help")], width="auto", className="d-flex align-items-center justify-content-end"),
        ], className="align-items-center"),
    ], className="dashboard-header")

def create_control_panel():
    return html.Div([
        html.Div("⚡ Controls", className="card-title"),
        dbc.ButtonGroup([
            dbc.Button("▶ Start", id="btn-start", color="success", className="control-btn", size="sm"),
            dbc.Button("⏸ Pause", id="btn-pause", color="warning", className="control-btn", size="sm"),
            dbc.Button("⏹ Stop", id="btn-stop", color="danger", className="control-btn", size="sm"),
        ], className="w-100 mb-2"),
        dbc.Button("↺ Reset", id="btn-reset", color="secondary", outline=True, className="control-btn w-100 mb-3", size="sm"),
        html.Hr(style={"borderColor": "var(--border-color)", "margin": "0.75rem 0"}),
        html.Label("Acquisition Function", className="small text-muted mb-1", style={"fontSize": "0.7rem"}),
        dcc.Dropdown(id="acquisition-selector", options=[
            {"label": "🎯 HH (Information Rate)", "value": "HH"},
            {"label": "🔬 ANDiE (Model Discrimination)", "value": "ANDiE"},
            {"label": "📊 Uncertainty Sampling", "value": "Uncertainty"},
            {"label": "⚖️ Composite", "value": "Composite"},
        ], value="HH", className="mb-2", clearable=False),
        html.Label("Aggressiveness (η)", className="small text-muted mb-1", style={"fontSize": "0.7rem"}),
        dcc.Slider(id="eta-slider", min=0.1, max=1.0, step=0.1, value=0.7, marks={0.1: "0.1", 0.5: "0.5", 1.0: "1.0"}, className="mb-2"),
        html.Label("Forecast Points", className="small text-muted mb-1", style={"fontSize": "0.7rem"}),
        dcc.Slider(id="forecast-slider", min=1, max=5, step=1, value=3, marks={1: "1", 3: "3", 5: "5"}),
    ], className="card-panel")

def create_stopping_panel():
    return html.Div([
        html.Div("🎯 Stopping Conditions", className="card-title"),
        dbc.Row([dbc.Col(html.Label("Max Time", className="small text-muted"), width=5),
                 dbc.Col(dbc.InputGroup([dbc.Input(id="input-max-time", type="number", value=60, className="input-dark", size="sm"),
                                        dbc.InputGroupText("min", className="input-dark", style={"fontSize": "0.7rem"})], size="sm"), width=7)], className="mb-2 align-items-center"),
        dbc.Row([dbc.Col(html.Label("Max Points", className="small text-muted"), width=5),
                 dbc.Col(dbc.Input(id="input-max-measurements", type="number", value=100, className="input-dark", size="sm"), width=7)], className="mb-2 align-items-center"),
        dbc.Row([dbc.Col(html.Label("Target H", className="small text-muted"), width=5),
                 dbc.Col(dbc.InputGroup([dbc.Input(id="input-target-entropy", type="number", value=0.5, step=0.1, className="input-dark", size="sm"),
                                        dbc.InputGroupText("nats", className="input-dark", style={"fontSize": "0.7rem"})], size="sm"), width=7)], className="mb-3 align-items-center"),
        html.Div([html.Span("Time ", className="small text-muted"), html.Span(id="progress-time-text", className="small", style={"color": "var(--accent-blue)"})], className="d-flex justify-content-between"),
        dbc.Progress(id="progress-time-bar", value=0, className="mb-2", style={"height": "4px", "backgroundColor": "rgba(26, 32, 41, 0.8)"}),
        html.Div([html.Span("Points ", className="small text-muted"), html.Span(id="progress-points-text", className="small", style={"color": "var(--accent-green)"})], className="d-flex justify-content-between"),
        dbc.Progress(id="progress-points-bar", value=0, color="success", className="mb-2", style={"height": "4px", "backgroundColor": "rgba(26, 32, 41, 0.8)"}),
        html.Div([html.Span("Entropy ", className="small text-muted"), html.Span(id="progress-entropy-text", className="small", style={"color": "var(--accent-purple)"})], className="d-flex justify-content-between"),
        dbc.Progress(id="progress-entropy-bar", value=0, color="info", style={"height": "4px", "backgroundColor": "rgba(26, 32, 41, 0.8)"}),
    ], className="card-panel")

def create_manual_input():
    return html.Div([
        html.Div("➕ Add Point", className="card-title"),
        dbc.Row([
            dbc.Col([html.Label("T (K)", className="small text-muted", style={"fontSize": "0.65rem"}), dbc.Input(id="input-T", type="number", value=150, className="input-dark", size="sm")], width=4),
            dbc.Col([html.Label("Time (s)", className="small text-muted", style={"fontSize": "0.65rem"}), dbc.Input(id="input-time", type="number", value=60, className="input-dark", size="sm")], width=4),
            dbc.Col([html.Label("Priority", className="small text-muted", style={"fontSize": "0.65rem"}),
                    dcc.Dropdown(id="input-priority", options=[{"label": "Normal", "value": 0}, {"label": "⭐ High", "value": 1}, {"label": "🔥 Urgent", "value": 2}], value=0, clearable=False, style={"fontSize": "0.75rem"})], width=4),
        ], className="mb-2"),
        dbc.Button("Add to Queue", id="btn-add-point", color="info", className="control-btn w-100", size="sm"),
        html.Hr(style={"borderColor": "var(--border-color)", "margin": "0.75rem 0"}),
        html.Label("Batch Add (comma-separated)", className="small text-muted mb-1", style={"fontSize": "0.65rem"}),
        dbc.InputGroup([dbc.Input(id="batch-input", placeholder="120, 140, 160", className="input-dark", size="sm"),
                       dbc.Button("Add", id="btn-batch-add", color="secondary", size="sm", outline=True)], size="sm"),
    ], className="card-panel")

def create_queue_panel():
    return html.Div([
        dbc.Row([
            dbc.Col(html.Div("📋 Queue", className="card-title mb-0")),
            dbc.Col([dbc.ButtonGroup([
                dbc.Button("↑", id="btn-queue-up", size="sm", color="secondary", outline=True, style={"padding": "0.1rem 0.4rem"}),
                dbc.Button("↓", id="btn-queue-down", size="sm", color="secondary", outline=True, style={"padding": "0.1rem 0.4rem"}),
                dbc.Button("🗑", id="btn-queue-delete", size="sm", color="danger", outline=True, style={"padding": "0.1rem 0.4rem"}),
                dbc.Button("Clear", id="btn-clear-queue", size="sm", color="danger", outline=True, style={"padding": "0.1rem 0.4rem", "fontSize": "0.65rem"}),
            ], size="sm")], width="auto"),
        ], className="mb-2 align-items-center"),
        html.Div(id="queue-display", style={"maxHeight": "180px", "overflowY": "auto"}),
    ], className="card-panel")

def create_model_weights_panel():
    return html.Div([
        html.Div("🔬 Model Comparison", className="card-title"),
        html.Div(id="model-weights-display"),
        html.Div([html.Span("Leading: ", className="small text-muted"),
                 html.Span(id="best-model-name", className="fw-bold", style={"color": "var(--accent-green)"}),
                 html.Span(" (", className="small text-muted"), html.Span(id="best-model-confidence"), html.Span("%)", className="small text-muted")], className="mt-2 text-center", style={"fontSize": "0.8rem"}),
    ], className="card-panel")

def create_info_gain_panel():
    return html.Div([
        html.Div("📈 Information Gain", className="card-title"),
        dcc.Graph(id="info-gain-plot", style={"height": "140px"}, config={"displayModeBar": False}),
        dbc.Row([
            dbc.Col([html.Div(id="total-info-gain", className="metric-value metric-value-sm"), html.Div("Total ΔH", className="metric-label")], width=6, className="text-center"),
            dbc.Col([html.Div(id="info-rate-display", className="metric-value metric-value-sm", style={"color": "var(--accent-purple)"}), html.Div("nats/min", className="metric-label")], width=6, className="text-center"),
        ]),
    ], className="card-panel")

def create_parameter_panel():
    return html.Div([
        html.Div("📐 Parameters", className="card-title"),
        dbc.Row([
            dbc.Col([html.Div([html.Span(id="param-Tc", className="metric-value"), html.Span(" ± ", className="metric-uncertainty"), html.Span(id="param-Tc-std", className="metric-uncertainty")]), html.Div("Tc (K)", className="metric-label")], width=6, className="text-center"),
            dbc.Col([html.Div([html.Span(id="param-beta", className="metric-value", style={"color": "var(--accent-green)"}), html.Span(" ± ", className="metric-uncertainty"), html.Span(id="param-beta-std", className="metric-uncertainty")]), html.Div("β", className="metric-label")], width=6, className="text-center"),
        ]),
    ], className="card-panel")

def create_residuals_panel():
    return html.Div([
        html.Div("📊 Fit Quality", className="card-title"),
        dcc.Graph(id="residuals-plot", style={"height": "120px"}, config={"displayModeBar": False}),
        dbc.Row([
            dbc.Col([html.Div(id="chi-squared", className="metric-value metric-value-sm"), html.Div("χ²/DOF", className="metric-label")], width=6, className="text-center"),
            dbc.Col([html.Div(id="rms-residual", className="metric-value metric-value-sm", style={"color": "var(--accent-orange)"}), html.Div("RMS", className="metric-label")], width=6, className="text-center"),
        ]),
    ], className="card-panel")

def create_log_panel():
    return html.Div([html.Div("📜 Log", className="card-title"), html.Div(id="log-display", style={"maxHeight": "130px", "overflowY": "auto"})], className="card-panel")

def create_help_modal():
    return dbc.Modal([
        dbc.ModalHeader(dbc.ModalTitle("TAS-AI Help", style={"fontFamily": "'JetBrains Mono', monospace"})),
        dbc.ModalBody([
            html.H6("⌨️ Keyboard Shortcuts", className="text-info mb-3"),
            html.Table([html.Tbody([
                html.Tr([html.Td(html.Kbd("Space")), html.Td("Pause / Resume", className="ps-3")]),
                html.Tr([html.Td(html.Kbd("Esc")), html.Td("Stop experiment", className="ps-3")]),
                html.Tr([html.Td(html.Kbd("A")), html.Td("Focus add point", className="ps-3")]),
                html.Tr([html.Td(html.Kbd("C")), html.Td("Clear queue", className="ps-3")]),
                html.Tr([html.Td(html.Kbd("E")), html.Td("Export data", className="ps-3")]),
                html.Tr([html.Td(html.Kbd("R")), html.Td("Reset experiment", className="ps-3")]),
            ])], className="mb-4", style={"fontSize": "0.85rem"}),
            html.Hr(),
            html.H6("🎯 Acquisition Functions", className="text-info mb-3"),
            html.Ul([
                html.Li([html.Strong("HH: "), "Maximizes info gain rate. Best for parameter estimation."]),
                html.Li([html.Strong("ANDiE: "), "Discriminates between models. Shows model weights."]),
                html.Li([html.Strong("Uncertainty: "), "Explores high variance regions."]),
                html.Li([html.Strong("Composite: "), "Weighted combination."]),
            ], style={"fontSize": "0.85rem"}),
            html.Hr(),
            html.H6("📋 Queue & Heatmap", className="text-info mb-3"),
            html.P("Click queue items to select. Click the Q-E heatmap to add measurement points directly.", style={"fontSize": "0.85rem"}),
        ]),
        dbc.ModalFooter([dbc.Button("📥 Export", id="btn-export-modal", color="secondary", className="me-2", size="sm"), dbc.Button("Close", id="btn-close-help", color="primary", size="sm")]),
    ], id="help-modal", size="lg", is_open=False)


app.layout = html.Div([
    dcc.Interval(id="update-interval", interval=800, n_intervals=0),
    dcc.Interval(id="notification-interval", interval=500, n_intervals=0),
    dcc.Store(id="experiment-state"),
    dcc.Download(id="download-data"),
    html.Div(id="notification-container", className="notification-container"),
    create_help_modal(),
    create_header(),
    dbc.Container([
        dbc.Row([
            dbc.Col([create_control_panel(), html.Div(className="mb-2"), create_stopping_panel(), html.Div(className="mb-2"), create_manual_input(), html.Div(className="mb-2"), create_queue_panel()], lg=2, md=3, className="mb-3"),
            dbc.Col([
                html.Div([html.Div("📈 Measurement Data", className="card-title"), dcc.Graph(id="data-plot", style={"height": "300px"}, config={"displayModeBar": False})], className="card-panel mb-2"),
                dbc.Row([
                    dbc.Col([html.Div([html.Div("🎯 Acquisition Function", className="card-title"), dcc.Graph(id="acquisition-plot", style={"height": "180px"}, config={"displayModeBar": False})], className="card-panel")], md=6),
                    dbc.Col([html.Div([html.Div("🗺️ Q-E Heatmap (click to add)", className="card-title"), dcc.Graph(id="heatmap-plot", style={"height": "180px"}, config={"displayModeBar": False})], className="card-panel")], md=6),
                ]),
            ], lg=6, md=5, className="mb-3"),
            dbc.Col([create_parameter_panel(), html.Div(className="mb-2"), create_model_weights_panel(), html.Div(className="mb-2"), create_info_gain_panel(), html.Div(className="mb-2"), create_residuals_panel(), html.Div(className="mb-2"), create_log_panel()], lg=4, md=4, className="mb-3"),
        ], className="mt-2 g-2"),
    ], fluid=True, style={"maxWidth": "1920px", "padding": "0 1rem"}),
    html.Script('''
        document.addEventListener('keydown', function(e) {
            if (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA') return;
            var key = e.key.toLowerCase();
            if (key === ' ') { e.preventDefault(); var p = document.getElementById('btn-pause'); var s = document.getElementById('btn-start'); if (!p.disabled) p.click(); else if (!s.disabled) s.click(); }
            else if (key === 'escape') { document.getElementById('btn-stop').click(); }
            else if (key === 'a') { document.getElementById('input-T').focus(); }
            else if (key === 'c') { document.getElementById('btn-clear-queue').click(); }
            else if (key === 'e') { document.getElementById('btn-export-modal').click(); }
            else if (key === 'r') { document.getElementById('btn-reset').click(); }
            else if (key === '?') { document.getElementById('btn-help').click(); }
        });
    ''')
], style={"minHeight": "100vh"})


# Callbacks
@callback(Output("experiment-state", "data"), Output("status-indicator", "children"), Output("status-indicator", "className"),
          Output("elapsed-time", "children"), Output("measurement-count", "children"),
          Output("param-Tc", "children"), Output("param-Tc-std", "children"), Output("param-beta", "children"), Output("param-beta-std", "children"),
          Output("total-info-gain", "children"), Output("info-rate-display", "children"), Output("chi-squared", "children"), Output("rms-residual", "children"),
          Output("progress-time-bar", "value"), Output("progress-time-text", "children"),
          Output("progress-points-bar", "value"), Output("progress-points-text", "children"),
          Output("progress-entropy-bar", "value"), Output("progress-entropy-text", "children"),
          Input("update-interval", "n_intervals"), State("input-max-time", "value"), State("input-max-measurements", "value"), State("input-target-entropy", "value"))
def update_state(n, max_time, max_measurements, target_entropy):
    state = backend.get_state()
    status_map = {"idle": ("⬤ Idle", "status-indicator status-idle"), "running": ("⬤ Running", "status-indicator status-running"),
                  "paused": ("⬤ Paused", "status-indicator status-paused"), "measuring": ("⬤ Measuring", "status-indicator status-measuring"),
                  "computing": ("⬤ Computing", "status-indicator status-computing")}
    status_text, status_class = status_map.get(state["state"], ("⬤ Unknown", "status-indicator"))
    elapsed = state["elapsed_time"]
    hours, rem = divmod(int(elapsed), 3600); mins, secs = divmod(rem, 60)
    max_time_sec = (max_time or 60) * 60; max_pts = max_measurements or 100
    time_pct = min(100, (elapsed / max_time_sec) * 100) if max_time_sec > 0 else 0
    pts_pct = min(100, (state["n_measurements"] / max_pts) * 100)
    entropy_pct = min(100, max(0, (1 - state["current_entropy"] / 5.0) * 100))
    df = backend.get_measurements_df()
    rms = np.sqrt(np.mean(df["residual"]**2)) if len(df) > 0 and "residual" in df else 0
    return (state, status_text, status_class, f"⏱ {hours:02d}:{mins:02d}:{secs:02d}", f"📊 {state['n_measurements']} pts",
            f"{state['estimated_Tc']:.1f}", f"{state['estimated_Tc_std']:.1f}", f"{state['estimated_beta']:.3f}", f"{state['estimated_beta_std']:.3f}",
            f"{state['total_info_gain']:.2f}", f"{state['info_rate']:.2f}", f"{state['chi_squared']:.2f}", f"{rms:.2f}",
            time_pct, f"{elapsed/60:.1f}/{max_time}min", pts_pct, f"{state['n_measurements']}/{max_pts}", entropy_pct, f"H={state['current_entropy']:.2f}")

@callback(Output("data-plot", "figure"), Input("experiment-state", "data"))
def update_data_plot(state):
    df = backend.get_measurements_df()
    fig = go.Figure()
    T_plot = np.linspace(80, 220, 200)
    I_true = [(1 - T/150)**0.325 if T < 150 else 0 for T in T_plot]; I_true = [i**2 + 0.01 for i in I_true]
    fig.add_trace(go.Scatter(x=T_plot, y=I_true, mode="lines", name="True", line=dict(color="rgba(255,255,255,0.2)", width=2, dash="dash")))
    if state:
        Tc, beta = state["estimated_Tc"], state["estimated_beta"]
        I_est = [(1 - T/Tc)**beta if T < Tc else 0 for T in T_plot]; I_est = [i**2 + 0.01 for i in I_est]
        fig.add_trace(go.Scatter(x=T_plot, y=I_est, mode="lines", name="Fit", line=dict(color="#00b4d8", width=2)))
    if not df.empty:
        colors = {"ai": "#00b4d8", "user": "#00f5a0", "initial": "#a855f7"}
        for source in df["source"].unique():
            mask = df["source"] == source
            fig.add_trace(go.Scatter(x=df.loc[mask, "E"], y=df.loc[mask, "I"], error_y=dict(type="data", array=df.loc[mask, "sigma"], visible=True, thickness=1),
                                    mode="markers", name=source.title(), marker=dict(color=colors.get(source, "#fff"), size=7)))
    fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(26,32,41,0.5)",
                     margin=dict(l=45, r=15, t=25, b=40), xaxis_title="T (K)", yaxis_title="I (a.u.)",
                     legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, font=dict(size=10)), font=dict(family="IBM Plex Sans", size=11))
    return fig

@callback(Output("acquisition-plot", "figure"), Input("experiment-state", "data"))
def update_acquisition_plot(state):
    fig = go.Figure()
    if backend.acquisition_scores_1d:
        T, scores = backend.acquisition_scores_1d.get("T", []), backend.acquisition_scores_1d.get("scores", [])
        fig.add_trace(go.Scatter(x=T, y=scores, fill="tozeroy", fillcolor="rgba(0,180,216,0.2)", line=dict(color="#00b4d8", width=2)))
        if len(scores) > 0:
            best_idx = np.argmax(scores)
            fig.add_trace(go.Scatter(x=[T[best_idx]], y=[scores[best_idx]], mode="markers", marker=dict(color="#00f5a0", size=10, symbol="star"), showlegend=False))
    fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(26,32,41,0.5)",
                     margin=dict(l=40, r=15, t=10, b=35), xaxis_title="T (K)", yaxis_title="Score", showlegend=False, font=dict(family="IBM Plex Sans", size=10))
    return fig

@callback(Output("heatmap-plot", "figure"), Input("experiment-state", "data"))
def update_heatmap(state):
    Q, E, scores = backend.get_acquisition_2d()
    fig = go.Figure()
    fig.add_trace(go.Heatmap(x=Q, y=E, z=scores, colorscale=[[0, "rgba(10,14,20,0.9)"], [0.25, "rgba(0,80,120,0.8)"], [0.5, "rgba(0,140,180,0.8)"], [0.75, "rgba(0,180,216,0.8)"], [1, "rgba(0,245,160,0.9)"]], showscale=False, hovertemplate="Q=%{x:.2f}<br>E=%{y:.1f}<extra></extra>"))
    df = backend.get_measurements_df()
    if not df.empty:
        fig.add_trace(go.Scatter(x=[0.5]*len(df), y=df["E"], mode="markers", marker=dict(color="white", size=5, symbol="x"), showlegend=False, hoverinfo="skip"))
    fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(26,32,41,0.5)",
                     margin=dict(l=40, r=15, t=10, b=35), xaxis_title="Q (r.l.u.)", yaxis_title="E (meV)", font=dict(family="IBM Plex Sans", size=10))
    return fig

@callback(Output("experiment-state", "data", allow_duplicate=True), Input("heatmap-plot", "clickData"), prevent_initial_call=True)
def handle_heatmap_click(click_data):
    if click_data and "points" in click_data:
        pt = click_data["points"][0]; backend.add_from_heatmap(pt.get("x", 0.5), pt.get("y", 20))
    raise PreventUpdate

@callback(Output("model-weights-display", "children"), Output("best-model-name", "children"), Output("best-model-confidence", "children"), Input("experiment-state", "data"))
def update_model_weights(state):
    weights = backend.get_model_weights()
    bars = []; best = max(weights, key=lambda x: x["weight"])
    for w in weights:
        pct = w["weight"] * 100
        bars.append(html.Div([html.Div(f"{w['name']} {pct:.0f}%", className="model-bar", style={"width": f"{max(pct, 15)}%", "backgroundColor": w["color"] + "30", "color": w["color"], "borderLeft": f"3px solid {w['color']}"})]))
    return bars, best["name"], f"{best['weight']*100:.0f}"

@callback(Output("info-gain-plot", "figure"), Input("experiment-state", "data"))
def update_info_gain_plot(state):
    times, gains = backend.get_info_history()
    fig = go.Figure()
    if times:
        fig.add_trace(go.Scatter(x=times, y=gains, mode="lines+markers", fill="tozeroy", fillcolor="rgba(168,85,247,0.15)", line=dict(color="#a855f7", width=2), marker=dict(size=4)))
    fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(26,32,41,0.5)",
                     margin=dict(l=40, r=15, t=10, b=30), xaxis_title="Time (min)", yaxis_title="ΔH", showlegend=False, font=dict(family="IBM Plex Sans", size=9))
    return fig

@callback(Output("residuals-plot", "figure"), Input("experiment-state", "data"))
def update_residuals_plot(state):
    df = backend.get_measurements_df()
    fig = go.Figure()
    if not df.empty and "residual" in df:
        colors = ["#00f5a0" if abs(r) < 2 else "#ff9f43" if abs(r) < 3 else "#ff6b6b" for r in df["residual"]]
        fig.add_trace(go.Scatter(x=df["E"], y=df["residual"], mode="markers", marker=dict(color=colors, size=6)))
        fig.add_hline(y=0, line_color="white", line_dash="solid", opacity=0.3)
        fig.add_hline(y=2, line_color="#ff9f43", line_dash="dash", opacity=0.4); fig.add_hline(y=-2, line_color="#ff9f43", line_dash="dash", opacity=0.4)
    fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(26,32,41,0.5)",
                     margin=dict(l=40, r=15, t=10, b=30), xaxis_title="T (K)", yaxis_title="Residual", yaxis=dict(range=[-4, 4]), showlegend=False, font=dict(family="IBM Plex Sans", size=9))
    return fig

@callback(Output("queue-display", "children"), Input("experiment-state", "data"))
def update_queue(state):
    queue = backend.get_queue_list()
    if not queue: return html.Div("Queue empty", className="text-muted small text-center py-3", style={"fontStyle": "italic"})
    items = []
    for item in queue:
        priority_badge = html.Span("HIGH", className="priority-badge priority-high") if item["priority"] == 1 else html.Span("URGENT", className="priority-badge priority-urgent") if item["priority"] == 2 else ""
        source_class = f"queue-item queue-item-{item['source']}" + (" queue-item-selected" if item["selected"] else "")
        items.append(html.Div([html.Span([f"T={item['E']:.1f}K", html.Span(f" • {item['count_time']:.0f}s", className="text-muted"), priority_badge]),
                              html.Span(item["source"][:2].upper(), className="text-muted", style={"fontSize": "0.6rem"})], className=source_class, id={"type": "queue-item", "id": item["id"]}))
    return items

@callback(Output("experiment-state", "data", allow_duplicate=True), Input({"type": "queue-item", "id": ALL}, "n_clicks"), prevent_initial_call=True)
def handle_queue_click(clicks):
    if not ctx.triggered_id or not any(clicks): raise PreventUpdate
    backend.toggle_selection(ctx.triggered_id["id"]); raise PreventUpdate

@callback(Output("log-display", "children"), Input("experiment-state", "data"))
def update_log(state):
    messages = backend.get_log_messages(15)
    if not messages: return html.Div("No log messages", className="text-muted small text-center py-2")
    return [html.Div([html.Span(msg["time"], className="log-time"), html.Span(msg["message"], className=f"log-{msg['level']}")], className="log-entry") for msg in reversed(messages)]

@callback(Output("notification-container", "children"), Input("notification-interval", "n_intervals"))
def update_notifications(n):
    return [dbc.Toast(notif["message"], header=notif["level"].title(), className=f"notification-toast notification-{notif['level']}", duration=notif["duration"], is_open=True, style={"marginBottom": "0.5rem"}) for notif in backend.get_notifications()]

@callback(Output("btn-start", "disabled"), Output("btn-pause", "disabled"), Output("btn-stop", "disabled"),
          Output("acquisition-selector", "disabled"), Output("eta-slider", "disabled"), Output("forecast-slider", "disabled"), Input("experiment-state", "data"))
def update_button_states(state):
    if not state: return False, True, True, False, False, False
    s = state.get("state", "idle")
    if s == "idle": return False, True, True, False, False, False
    elif s in ["running", "measuring", "computing"]: return True, False, False, True, True, True
    elif s == "paused": return False, True, False, False, False, False
    return False, True, True, False, False, False

@callback(Output("experiment-state", "data", allow_duplicate=True), Input("btn-start", "n_clicks"), prevent_initial_call=True)
def handle_start(n): backend.start() if n else None; raise PreventUpdate

@callback(Output("experiment-state", "data", allow_duplicate=True), Input("btn-pause", "n_clicks"), prevent_initial_call=True)
def handle_pause(n): backend.pause() if n else None; raise PreventUpdate

@callback(Output("experiment-state", "data", allow_duplicate=True), Input("btn-stop", "n_clicks"), prevent_initial_call=True)
def handle_stop(n): backend.stop() if n else None; raise PreventUpdate

@callback(Output("experiment-state", "data", allow_duplicate=True), Input("btn-reset", "n_clicks"), prevent_initial_call=True)
def handle_reset(n): backend.reset() if n else None; raise PreventUpdate

@callback(Output("experiment-state", "data", allow_duplicate=True), Input("acquisition-selector", "value"), prevent_initial_call=True)
def handle_acq_change(value): backend.update_config(acquisition_function=value) if value else None; raise PreventUpdate

@callback(Output("experiment-state", "data", allow_duplicate=True), Input("eta-slider", "value"), prevent_initial_call=True)
def handle_eta_change(value): backend.update_config(eta=value) if value else None; raise PreventUpdate

@callback(Output("experiment-state", "data", allow_duplicate=True), Input("forecast-slider", "value"), prevent_initial_call=True)
def handle_forecast_change(value): backend.update_config(n_forecast=value) if value else None; raise PreventUpdate

@callback(Output("experiment-state", "data", allow_duplicate=True), Input("btn-add-point", "n_clicks"),
          State("input-T", "value"), State("input-time", "value"), State("input-priority", "value"), prevent_initial_call=True)
def handle_add_point(n, T, time_val, priority):
    if n and T: backend.add_to_queue(MeasurementPoint(E=float(T), count_time=float(time_val or 60), source="user", priority=int(priority or 0)))
    raise PreventUpdate

@callback(Output("experiment-state", "data", allow_duplicate=True), Output("batch-input", "value"), Input("btn-batch-add", "n_clicks"), State("batch-input", "value"), prevent_initial_call=True)
def handle_batch_add(n, value):
    if n and value:
        try:
            temps = [float(t.strip()) for t in value.split(",") if t.strip()]
            if temps: backend.add_batch_to_queue(temps); return dash.no_update, ""
        except ValueError: pass
    raise PreventUpdate

@callback(Output("experiment-state", "data", allow_duplicate=True), Input("btn-clear-queue", "n_clicks"), prevent_initial_call=True)
def handle_clear(n): backend.clear_queue() if n else None; raise PreventUpdate

@callback(Output("experiment-state", "data", allow_duplicate=True), Input("btn-queue-delete", "n_clicks"), prevent_initial_call=True)
def handle_delete_selected(n): backend.remove_selected() if n else None; raise PreventUpdate

@callback(Output("experiment-state", "data", allow_duplicate=True), Input("btn-queue-up", "n_clicks"), prevent_initial_call=True)
def handle_move_up(n):
    if n:
        for item in backend.get_queue_list():
            if item["selected"]: backend.move_item(item["id"], -1); break
    raise PreventUpdate

@callback(Output("experiment-state", "data", allow_duplicate=True), Input("btn-queue-down", "n_clicks"), prevent_initial_call=True)
def handle_move_down(n):
    if n:
        for item in backend.get_queue_list():
            if item["selected"]: backend.move_item(item["id"], 1); break
    raise PreventUpdate

@callback(Output("help-modal", "is_open"), Input("btn-help", "n_clicks"), Input("btn-close-help", "n_clicks"), State("help-modal", "is_open"), prevent_initial_call=True)
def toggle_help(n1, n2, is_open): return not is_open if n1 or n2 else is_open

@callback(Output("download-data", "data"), Input("btn-export-modal", "n_clicks"), prevent_initial_call=True)
def handle_export(n):
    if n: return dcc.send_data_frame(backend.get_measurements_df().to_csv, f"tasai_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv", index=False)
    raise PreventUpdate

@callback(Output("experiment-state", "data", allow_duplicate=True), Input("input-max-time", "value"), Input("input-max-measurements", "value"), Input("input-target-entropy", "value"), prevent_initial_call=True)
def update_stopping(max_time, max_measurements, target_entropy):
    if max_time: backend.update_config(max_time=max_time * 60)
    if max_measurements: backend.update_config(max_measurements=max_measurements)
    if target_entropy: backend.update_config(target_entropy=target_entropy)
    raise PreventUpdate


def main():
    """Entry point for CLI/console scripts."""
    import argparse

    parser = argparse.ArgumentParser(description="TAS-AI Dashboard")
    parser.add_argument("--port", type=int, default=8050)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    print(
        f"\n╔══════════════════════════════════════════════════════════════╗\n"
        f"║                    TAS-AI Dashboard                          ║\n"
        f"║  🌐 http://{args.host}:{args.port:<5}                               ║\n"
        f"║  Use your browser to interact with the web UI.              ║\n"
        f"╚══════════════════════════════════════════════════════════════╝\n"
    )
    app.run_server(host=args.host, port=args.port, debug=args.debug)


if __name__ == "__main__":
    main()
