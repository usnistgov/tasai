"""
Additional Dashboard Components

Extended components for the TAS-AI dashboard including:
- Model comparison panel (for ANDiE)
- Information gain tracker
- Stopping conditions panel
- Advanced queue management
- Experiment summary
"""

import dash
from dash import dcc, html, Input, Output, State, callback
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from datetime import datetime


def create_model_comparison_panel():
    """
    Panel showing model weights and discrimination progress.
    Used with ANDiE acquisition function.
    """
    return html.Div([
        html.Div("Model Comparison (ANDiE)", className="card-title"),
        
        # Model weights bar chart
        dcc.Graph(id="model-weights-plot", style={"height": "150px"},
                 config={"displayModeBar": False}),
        
        # Best model indicator
        html.Div([
            html.Span("Leading Model: ", className="text-muted small"),
            html.Span(id="best-model-name", className="fw-bold"),
            html.Span(" (", className="text-muted small"),
            html.Span(id="best-model-confidence", style={"color": "#00f5a0"}),
            html.Span("% confidence)", className="text-muted small"),
        ], className="mt-2 text-center"),
        
    ], className="card-panel", id="model-comparison-panel")


def create_info_gain_tracker():
    """
    Panel showing cumulative information gain over time.
    """
    return html.Div([
        html.Div("Information Gain", className="card-title"),
        
        dcc.Graph(id="info-gain-plot", style={"height": "180px"},
                 config={"displayModeBar": False}),
        
        dbc.Row([
            dbc.Col([
                html.Div(id="total-info-gain", className="metric-value", 
                        style={"fontSize": "1.2rem"}),
                html.Div("Total ΔH (nats)", className="metric-label"),
            ], width=6),
            dbc.Col([
                html.Div(id="info-rate", className="metric-value",
                        style={"fontSize": "1.2rem", "color": "#a855f7"}),
                html.Div("Rate (nats/min)", className="metric-label"),
            ], width=6),
        ], className="mt-2 text-center"),
        
    ], className="card-panel")


def create_stopping_conditions_panel():
    """
    Panel for configuring stopping conditions.
    """
    return html.Div([
        html.Div("Stopping Conditions", className="card-title"),
        
        # Max time
        dbc.Row([
            dbc.Col([
                html.Label("Max Time", className="small text-muted"),
            ], width=6),
            dbc.Col([
                dbc.InputGroup([
                    dbc.Input(id="input-max-time", type="number", value=60,
                             className="input-dark", size="sm"),
                    dbc.InputGroupText("min", style={
                        "backgroundColor": "#1a2029",
                        "borderColor": "#2d3748",
                        "color": "#8b949e",
                        "fontSize": "0.8rem"
                    }),
                ], size="sm"),
            ], width=6),
        ], className="mb-2 align-items-center"),
        
        # Max measurements
        dbc.Row([
            dbc.Col([
                html.Label("Max Points", className="small text-muted"),
            ], width=6),
            dbc.Col([
                dbc.Input(id="input-max-measurements", type="number", value=100,
                         className="input-dark", size="sm"),
            ], width=6),
        ], className="mb-2 align-items-center"),
        
        # Target entropy
        dbc.Row([
            dbc.Col([
                html.Label("Target H", className="small text-muted"),
            ], width=6),
            dbc.Col([
                dbc.InputGroup([
                    dbc.Input(id="input-target-entropy", type="number", value=0.1,
                             step=0.01, className="input-dark", size="sm"),
                    dbc.InputGroupText("nats", style={
                        "backgroundColor": "#1a2029",
                        "borderColor": "#2d3748",
                        "color": "#8b949e",
                        "fontSize": "0.8rem"
                    }),
                ], size="sm"),
            ], width=6),
        ], className="mb-2 align-items-center"),
        
        # Progress indicators
        html.Hr(style={"borderColor": "#2d3748"}),
        
        html.Div([
            html.Div([
                html.Span("Time: ", className="small text-muted"),
                html.Span(id="progress-time", className="small"),
            ]),
            dbc.Progress(id="progress-time-bar", value=0, className="mb-2",
                        style={"height": "4px", "backgroundColor": "#1a2029"}),
        ]),
        
        html.Div([
            html.Div([
                html.Span("Measurements: ", className="small text-muted"),
                html.Span(id="progress-measurements", className="small"),
            ]),
            dbc.Progress(id="progress-measurements-bar", value=0, className="mb-2",
                        style={"height": "4px", "backgroundColor": "#1a2029"}),
        ]),
        
        html.Div([
            html.Div([
                html.Span("Entropy: ", className="small text-muted"),
                html.Span(id="progress-entropy", className="small"),
            ]),
            dbc.Progress(id="progress-entropy-bar", value=0,
                        style={"height": "4px", "backgroundColor": "#1a2029"}),
        ]),
        
    ], className="card-panel")


def create_experiment_summary():
    """
    Panel showing experiment summary statistics.
    """
    return html.Div([
        html.Div("Experiment Summary", className="card-title"),
        
        dbc.Row([
            dbc.Col([
                html.Div([
                    html.I(className="fas fa-clock me-2", style={"color": "#00b4d8"}),
                    html.Span(id="summary-duration", className="fw-bold"),
                ]),
                html.Div("Duration", className="metric-label"),
            ], width=6, className="text-center mb-2"),
            
            dbc.Col([
                html.Div([
                    html.I(className="fas fa-chart-line me-2", style={"color": "#00f5a0"}),
                    html.Span(id="summary-measurements", className="fw-bold"),
                ]),
                html.Div("Measurements", className="metric-label"),
            ], width=6, className="text-center mb-2"),
        ]),
        
        dbc.Row([
            dbc.Col([
                html.Div([
                    html.I(className="fas fa-robot me-2", style={"color": "#a855f7"}),
                    html.Span(id="summary-ai-points", className="fw-bold"),
                ]),
                html.Div("AI Points", className="metric-label"),
            ], width=6, className="text-center mb-2"),
            
            dbc.Col([
                html.Div([
                    html.I(className="fas fa-user me-2", style={"color": "#ff9f43"}),
                    html.Span(id="summary-user-points", className="fw-bold"),
                ]),
                html.Div("User Points", className="metric-label"),
            ], width=6, className="text-center mb-2"),
        ]),
        
        html.Hr(style={"borderColor": "#2d3748"}),
        
        # Efficiency metrics
        html.Div([
            html.Span("Avg. efficiency: ", className="small text-muted"),
            html.Span(id="summary-efficiency", className="small fw-bold",
                     style={"color": "#00b4d8"}),
            html.Span(" nats/min", className="small text-muted"),
        ], className="text-center"),
        
    ], className="card-panel")


def create_advanced_queue_panel():
    """
    Advanced queue panel with reordering and batch operations.
    """
    return html.Div([
        dbc.Row([
            dbc.Col([
                html.Div("Measurement Queue", className="card-title mb-0")
            ]),
            dbc.Col([
                dbc.ButtonGroup([
                    dbc.Button("↑", id="btn-queue-up", size="sm", color="secondary", 
                              outline=True, title="Move selected up"),
                    dbc.Button("↓", id="btn-queue-down", size="sm", color="secondary",
                              outline=True, title="Move selected down"),
                    dbc.Button("🗑", id="btn-queue-delete", size="sm", color="danger",
                              outline=True, title="Delete selected"),
                    dbc.Button("Clear", id="btn-clear-queue", size="sm", color="danger",
                              outline=True),
                ], size="sm")
            ], width="auto"),
        ], className="mb-2"),
        
        # Queue list with selection
        html.Div(id="advanced-queue-display", style={
            "maxHeight": "250px",
            "overflowY": "auto"
        }),
        
        html.Hr(style={"borderColor": "#2d3748"}),
        
        # Batch add
        html.Div([
            html.Label("Quick Add (T values, comma-separated)", 
                      className="small text-muted mb-1"),
            dbc.InputGroup([
                dbc.Input(id="batch-add-input", placeholder="120, 140, 160, 180",
                         className="input-dark", size="sm"),
                dbc.Button("Add All", id="btn-batch-add", color="info", size="sm"),
            ], size="sm"),
        ]),
        
    ], className="card-panel")


def create_2d_heatmap_panel():
    """
    2D heatmap for Q-E space visualization (TAS mode).
    """
    return html.Div([
        html.Div("Acquisition Heatmap (Q-E)", className="card-title"),
        
        dcc.Graph(id="acquisition-heatmap", style={"height": "300px"},
                 config={"displayModeBar": False}),
        
        dbc.Row([
            dbc.Col([
                html.Label("Q range", className="small text-muted"),
                dcc.RangeSlider(id="q-range-slider", min=0, max=2, step=0.1,
                               value=[0, 1], marks={0: "0", 1: "1", 2: "2"}),
            ], width=6),
            dbc.Col([
                html.Label("E range", className="small text-muted"),
                dcc.RangeSlider(id="e-range-slider", min=0, max=50, step=5,
                               value=[0, 30], marks={0: "0", 25: "25", 50: "50"}),
            ], width=6),
        ], className="mt-2"),
        
    ], className="card-panel")


def create_residuals_panel():
    """
    Panel showing fit residuals.
    """
    return html.Div([
        html.Div("Fit Residuals", className="card-title"),
        
        dcc.Graph(id="residuals-plot", style={"height": "150px"},
                 config={"displayModeBar": False}),
        
        dbc.Row([
            dbc.Col([
                html.Div(id="chi-squared", className="metric-value",
                        style={"fontSize": "1rem"}),
                html.Div("χ²/DOF", className="metric-label"),
            ], width=6, className="text-center"),
            dbc.Col([
                html.Div(id="rms-residual", className="metric-value",
                        style={"fontSize": "1rem", "color": "#a855f7"}),
                html.Div("RMS", className="metric-label"),
            ], width=6, className="text-center"),
        ]),
        
    ], className="card-panel")


# =============================================================================
# Plot Generation Functions
# =============================================================================

def make_model_weights_figure(model_names, weights):
    """Create model weights bar chart for ANDiE."""
    fig = go.Figure()
    
    colors = ["#00b4d8", "#00f5a0", "#a855f7", "#ff9f43", "#ff6b6b"]
    
    fig.add_trace(go.Bar(
        x=model_names,
        y=weights,
        marker_color=colors[:len(model_names)],
        text=[f"{w*100:.1f}%" for w in weights],
        textposition="outside",
        textfont=dict(size=10)
    ))
    
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(26, 32, 41, 0.8)",
        margin=dict(l=20, r=20, t=10, b=30),
        yaxis=dict(range=[0, 1], showgrid=False),
        xaxis=dict(showgrid=False),
        font=dict(family="IBM Plex Sans", size=10)
    )
    
    return fig


def make_info_gain_figure(times, info_gains):
    """Create information gain over time plot."""
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=times, y=info_gains,
        mode="lines+markers",
        fill="tozeroy",
        fillcolor="rgba(168, 85, 247, 0.2)",
        line=dict(color="#a855f7", width=2),
        marker=dict(size=4)
    ))
    
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(26, 32, 41, 0.8)",
        margin=dict(l=40, r=20, t=10, b=30),
        xaxis_title="Time (min)",
        yaxis_title="ΔH (nats)",
        font=dict(family="IBM Plex Sans", size=10)
    )
    
    return fig


def make_acquisition_heatmap(Q_range, E_range, scores):
    """Create 2D acquisition function heatmap."""
    fig = go.Figure()
    
    fig.add_trace(go.Heatmap(
        x=Q_range,
        y=E_range,
        z=scores,
        colorscale=[
            [0, "rgba(10, 14, 20, 0.8)"],
            [0.3, "rgba(0, 100, 150, 0.8)"],
            [0.6, "rgba(0, 180, 216, 0.8)"],
            [1, "rgba(0, 245, 160, 0.8)"]
        ],
        showscale=False
    ))
    
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(26, 32, 41, 0.8)",
        margin=dict(l=40, r=20, t=10, b=40),
        xaxis_title="Q (r.l.u.)",
        yaxis_title="E (meV)",
        font=dict(family="IBM Plex Sans", size=10)
    )
    
    return fig


def make_residuals_figure(x_data, residuals, uncertainties):
    """Create residuals plot."""
    fig = go.Figure()
    
    # Normalized residuals
    norm_residuals = residuals / uncertainties
    
    # Color by magnitude
    colors = ["#00f5a0" if abs(r) < 2 else "#ff9f43" if abs(r) < 3 else "#ff6b6b" 
              for r in norm_residuals]
    
    fig.add_trace(go.Scatter(
        x=x_data,
        y=norm_residuals,
        mode="markers",
        marker=dict(color=colors, size=6)
    ))
    
    # Reference lines
    fig.add_hline(y=0, line_color="white", line_dash="solid", opacity=0.3)
    fig.add_hline(y=2, line_color="#ff9f43", line_dash="dash", opacity=0.5)
    fig.add_hline(y=-2, line_color="#ff9f43", line_dash="dash", opacity=0.5)
    
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(26, 32, 41, 0.8)",
        margin=dict(l=40, r=20, t=10, b=30),
        xaxis_title="Temperature (K)",
        yaxis_title="(I - Ifit)/σ",
        yaxis=dict(range=[-5, 5]),
        font=dict(family="IBM Plex Sans", size=10)
    )
    
    return fig


# =============================================================================
# Notification System
# =============================================================================

def create_notification_container():
    """Container for toast notifications."""
    return html.Div(id="notification-container", style={
        "position": "fixed",
        "top": "80px",
        "right": "20px",
        "zIndex": "1000",
        "maxWidth": "350px"
    })


def create_notification(message, level="info", duration=5000):
    """
    Create a notification toast.
    
    level: "info", "success", "warning", "error"
    """
    colors = {
        "info": "#00b4d8",
        "success": "#00f5a0",
        "warning": "#ff9f43",
        "error": "#ff6b6b"
    }
    
    icons = {
        "info": "ℹ️",
        "success": "✓",
        "warning": "⚠️",
        "error": "✕"
    }
    
    return dbc.Toast(
        [html.P(message, className="mb-0")],
        header=f"{icons.get(level, 'ℹ️')} {level.title()}",
        icon=level,
        duration=duration,
        style={
            "backgroundColor": "#1a2029",
            "borderLeft": f"4px solid {colors.get(level, '#00b4d8')}",
            "color": "#e6edf3"
        }
    )


# =============================================================================
# Help / Documentation Modal
# =============================================================================

def create_help_modal():
    """Create help modal with documentation."""
    return dbc.Modal([
        dbc.ModalHeader(dbc.ModalTitle("TAS-AI Dashboard Help")),
        dbc.ModalBody([
            html.H5("Acquisition Functions", className="text-info"),
            html.Ul([
                html.Li([
                    html.Strong("HH (Information Rate): "),
                    "Maximizes information gain per unit time. Best for parameter estimation."
                ]),
                html.Li([
                    html.Strong("ANDiE (Model Discrimination): "),
                    "Selects points that best distinguish between competing physics models."
                ]),
                html.Li([
                    html.Strong("Uncertainty Sampling: "),
                    "Measures where model predictions are most uncertain. Good for exploration."
                ]),
                html.Li([
                    html.Strong("Composite: "),
                    "Weighted combination of strategies."
                ]),
            ]),
            
            html.Hr(),
            
            html.H5("Queue Management", className="text-info"),
            html.P([
                "Add custom measurement points to override AI suggestions. ",
                "Points are executed in priority order (Urgent > High > Normal), ",
                "then in the order added."
            ]),
            
            html.Hr(),
            
            html.H5("Keyboard Shortcuts", className="text-info"),
            html.Ul([
                html.Li([html.Kbd("Space"), " - Pause/Resume"]),
                html.Li([html.Kbd("Esc"), " - Stop experiment"]),
                html.Li([html.Kbd("A"), " - Add point to queue"]),
                html.Li([html.Kbd("C"), " - Clear queue"]),
                html.Li([html.Kbd("E"), " - Export data"]),
            ]),
            
            html.Hr(),
            
            html.H5("Parameters", className="text-info"),
            html.Ul([
                html.Li([
                    html.Strong("η (Eta): "),
                    "Aggressiveness (0-1). Higher values favor exploration."
                ]),
                html.Li([
                    html.Strong("Forecast Points: "),
                    "Number of points selected per MCMC run. Higher = fewer MCMC calls."
                ]),
            ]),
        ]),
        dbc.ModalFooter(
            dbc.Button("Close", id="close-help", className="ms-auto")
        ),
    ], id="help-modal", size="lg")


# =============================================================================
# Settings Panel
# =============================================================================

def create_settings_panel():
    """Create advanced settings panel."""
    return dbc.Offcanvas([
        html.H5("Advanced Settings", className="mb-4"),
        
        # MCMC settings
        html.H6("MCMC Configuration", className="text-info"),
        dbc.Row([
            dbc.Col([
                html.Label("Burn-in", className="small"),
                dbc.Input(id="mcmc-burn", type="number", value=500, size="sm",
                         className="input-dark"),
            ], width=4),
            dbc.Col([
                html.Label("Steps", className="small"),
                dbc.Input(id="mcmc-steps", type="number", value=500, size="sm",
                         className="input-dark"),
            ], width=4),
            dbc.Col([
                html.Label("Pop", className="small"),
                dbc.Input(id="mcmc-pop", type="number", value=8, size="sm",
                         className="input-dark"),
            ], width=4),
        ], className="mb-3"),
        
        html.Hr(),
        
        # Count time settings
        html.H6("Count Time Bounds", className="text-info"),
        dbc.Row([
            dbc.Col([
                html.Label("Min (s)", className="small"),
                dbc.Input(id="min-count-time", type="number", value=10, size="sm",
                         className="input-dark"),
            ], width=6),
            dbc.Col([
                html.Label("Max (s)", className="small"),
                dbc.Input(id="max-count-time", type="number", value=300, size="sm",
                         className="input-dark"),
            ], width=6),
        ], className="mb-3"),
        
        html.Hr(),
        
        # Display settings
        html.H6("Display", className="text-info"),
        dbc.Checklist(
            id="display-settings",
            options=[
                {"label": "Show true model", "value": "show_true"},
                {"label": "Show confidence bands", "value": "show_bands"},
                {"label": "Show acquisition heatmap", "value": "show_heatmap"},
                {"label": "Dark mode", "value": "dark_mode"},
            ],
            value=["show_true", "dark_mode"],
            className="mb-3"
        ),
        
        html.Hr(),
        
        # Update interval
        html.H6("Update Interval", className="text-info"),
        dcc.Slider(
            id="update-interval-slider",
            min=500, max=5000, step=500, value=1000,
            marks={500: "0.5s", 1000: "1s", 2000: "2s", 5000: "5s"}
        ),
        
    ], id="settings-offcanvas", title="Settings", placement="end")


# =============================================================================
# CSS for additional components
# =============================================================================

ADDITIONAL_CSS = """
.queue-item-selected {
    background: rgba(0, 180, 216, 0.2) !important;
    border: 1px solid var(--accent-blue);
}

.notification-enter {
    animation: slideIn 0.3s ease-out;
}

@keyframes slideIn {
    from {
        transform: translateX(100%);
        opacity: 0;
    }
    to {
        transform: translateX(0);
        opacity: 1;
    }
}

.pulse-dot {
    animation: pulse 2s infinite;
}

@keyframes pulse {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.5; }
}

.metric-card {
    transition: transform 0.2s, box-shadow 0.2s;
}

.metric-card:hover {
    transform: translateY(-2px);
    box-shadow: 0 4px 12px rgba(0, 180, 216, 0.2);
}

kbd {
    background: var(--bg-tertiary);
    border: 1px solid var(--border-color);
    border-radius: 4px;
    padding: 2px 6px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.75rem;
}
"""
