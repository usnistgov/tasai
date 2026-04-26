# TAS-AI Dashboard Guide

The TAS-AI Dashboard provides real-time visualization and control of autonomous neutron scattering experiments. It is implemented with [Dash](https://dash.plotly.com/) and runs as a lightweight web app (served locally, viewable in any browser).

## Quick Start

```bash
# Start the dashboard (serves http://127.0.0.1:8050/)
python -m tasai.dashboard.app --port 8050
```

Then open `http://127.0.0.1:8050/` in your browser (or use the `tasai-dashboard` console script which wraps the same entry point).

## Dashboard Layout

The Dash application renders a single-page control room with the following regions:

- **Control bar** – Start/Pause/Stop/Reset buttons, export button, queue utilities, and the help shortcut
- **Dispersion + Parameters** – Streaming plot of ω(H,H,0) overlaid with measurements plus a parameter card with J₁, J₂, D (or Sunny order parameters) and χ²/N
- **Model weights & information gain** – Side-by-side cards showing posterior probabilities, Bayes factors, and cumulative/instantaneous information gain
- **Acquisition heatmap** – 2D (H,E) intensity/score map; click anywhere to create a candidate measurement
- **Measurement queue** – Table of pending measurements with acquisition scores and statuses
- **Event log & toasts** – Time-stamped log plus dismissible notifications for warnings/errors

### Controls

| Control | Description |
|---------|-------------|
| ▶ **Start**, ⏸ **Pause**, ⏹ **Stop** | Manage the autonomous loop. Pause keeps the queue intact; Stop clears the active measurement. |
| ↺ **Reset** | Clears measurements and queue, returning the planner to its initial state. |
| **Add to Queue** | Push the current H, K, L, E, and count-time values (taken from the sliders/inputs) onto the queue. |
| **Batch Add** | Paste multiple whitespace-separated lines (`H K L E count`) to enqueue several points at once. |
| **Queue ↑/↓/🗑/Clear** | Reorder or remove queued measurements directly in the browser. |
| **Export** | Download the current measurement log as CSV (includes timestamps, intensities, and fitted values). |
| **? Help** | Opens a modal with tips and shortcut reminders. |

### Dispersion & Parameter Cards

- The **Dispersion** plot overlays the current fit (solid line) with the latest measurements. Point colors encode the source (AI-selected, forced coverage, manual). Error bars show propagated uncertainty.  
- The **Parameter** card lists the current MAP estimates, 1σ uncertainties, χ²/N, and the total measurement count. Colors shift from amber to green as uncertainties shrink.

### Model Weights & Evidence

- Horizontal bars show the posterior probability of each candidate Hamiltonian.
- A Bayes factor badge summarizes the evidence ratio vs. the nearest competitor (same interpretation as in the manuscript).
- Once the winning model exceeds 95 % probability, the card highlights that the discrimination goal has been satisfied.

### Information Gain

- **Cumulative bits** – Running total of the Shannon information gathered.
- **Last step** – How much the most recent point contributed.
- **Rate** – Bits per minute to track efficiency.  
The progress bar turns green when the target entropy (set in the settings panel) is met.

### Measurement Queue

Columns: queue index, `(H,K,L)`, energy transfer, count time, acquisition score, and status.

- 🟢 **Measuring** – the spectrometer is currently at this point.
- 🟡 **Queued** – waiting its turn.
- ✅ **Complete** – data has been folded into the fit.
- 🟠 **Paused** – pending user confirmation (e.g., coverage seeds).

You can drag rows to reorder them or use the arrow buttons for precise moves. Clicking a row also surfaces its metadata in the sidebar.

### Acquisition Heatmap

When the Sunny/analytic backend provides precomputed scores, the central heatmap displays score × intensity across H and energy. Clicking a pixel fills the Add-to-Queue inputs, allowing you to seed tactical points with a single tap.

### Event Log & Notifications

A scrollable log at the bottom lists each measurement, posterior update, or warning with timestamps. Toast notifications (top-right) mirror major events so you can step away from the keyboard and still see when TAS-AI pauses or finishes.

## Keyboard Shortcuts

The web app installs a small key listener that forwards a few essential shortcuts to the corresponding buttons:

| Key | Action |
|-----|--------|
| `Space` | Toggle Pause/Resume (prefers Pause if available, otherwise Start) |
| `Esc` | Stop immediately |
| `A` | Focus the temperature/energy input so you can type a new value |
| `C` | Clear the queue |
| `E` | Open the export modal |
| `R` | Reset the experiment |
| `?` | Open the help modal |

All other interactions (adding/removing points, editing settings, switching acquisition functions) are performed through the on-screen controls and are touch/mouse friendly.

## Command-Line Options

```bash
python -m tasai.dashboard.app [OPTIONS]
# or
tasai-dashboard [OPTIONS]
```

| Option | Default | Description |
|--------|---------|-------------|
| `--host` | `127.0.0.1` | Interface to bind (use `0.0.0.0` inside Docker or on remote machines) |
| `--port` | `8050` | TCP port for the HTTP server |
| `--debug` | `False` | Enable Dash hot reload / debugger (developer use only) |

## Troubleshooting

- **Dashboard does not start** – Ensure Dash dependencies are installed (`pip install dash dash-bootstrap-components plotly pandas`).  
- **Blank page** – Confirm you’re visiting the URL printed to the terminal (default `http://127.0.0.1:8050/`). Refresh or try a different browser if the assets fail to load.  
- **Slow performance** – Close duplicate dashboard tabs, lower the queue size, or run the CLI without `--debug`.  
- **Cannot reach the instrument** – Verify the proxy server is running (`python -m tasai.proxy_server.server`) and that your firewall allows the configured host/port.  

For additional questions open an issue on GitHub or run `tasai-dashboard --help` for the latest CLI options.

---

*Dashboard version 1.0 | Built with Dash*
