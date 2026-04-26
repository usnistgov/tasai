# TAS-AI: Autonomous Triple-Axis Spectrometer Control

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![Public Domain (U.S.)](https://img.shields.io/badge/License-Public%20Domain%20(U.S.)-blue.svg)](https://github.com/williamratcliff-nist/tasai/blob/main/LICENSE)

**TAS-AI** is a Python framework for autonomous neutron scattering experiments on triple-axis spectrometers. It combines Bayesian inference, physics-informed models, and active learning to optimize measurement strategies in real-time.

Developed at the NIST Center for Neutron Research (NCNR).

## Why TAS-AI?

Traditional neutron scattering experiments require scientists to manually decide each measurement point—a slow, labor-intensive process. TAS-AI automates this by:

- **Learning from each measurement** to decide what to measure next
- **Using physics models** to focus on informative regions
- **Accounting for motor motion** to minimize dead time
- **Providing real-time visualization** of the experiment

Result: **2-5× faster** experiments with **fewer measurements**.

## Features

- 🎯 **Physics-Informed Acquisition**: Uses spin wave models to guide measurements
- 🔄 **Model Discrimination**: Bayesian comparison of competing Hamiltonians (e.g., NN vs J1-J2)
- 📊 **Real-Time Dashboard**: Live web visualization built with Dash
- ⚡ **Motor Motion Optimization**: Minimizes dead time with intelligent path planning
- 🧮 **MCMC Inference**: Full Bayesian parameter estimation with BUMPS/emcee
- 🔬 **Sunny-Inspired Models**: Fast Python spin wave calculations
- 📈 **Benchmark Suite**: Compare against gpCAM, Log-GP, and grid methods

## Installation

### Option 1: Mamba/Conda (Recommended)

Mamba is significantly faster than conda for dependency resolution:

```bash
# Clone repository
git clone https://github.com/williamratcliff-nist/tasai.git
cd tasai

# Create environment with mamba (fast)
mamba env create -f environment.yml
mamba activate tasai

# Or with conda (slower)
conda env create -f environment.yml
conda activate tasai
```

### Option 2: Pip

```bash
git clone https://github.com/williamratcliff-nist/tasai.git
cd tasai
pip install -e ".[all]"
```

> **Spin-wave dependency note:** The optional `pyspinw` backend depends on
> SciPy's legacy Fortran wrappers that were removed in version 1.23.  We
> therefore pin `scipy>=1.8,<1.23` in the core requirements.  If your
> existing environment already pulled in a newer SciPy, create a fresh
> virtual environment before installing the `spinwave` extra so the TAS-AI
> benchmark can exercise the SpinW backend instead of falling back to the
> analytic Sunny model.

### Option 3: Docker

```bash
# Build image
docker build -t tasai .

# Run dashboard (accessible at http://localhost:8050)
docker run -p 8050:8050 tasai micromamba run -n tasai python -m tasai.dashboard.app --host 0.0.0.0

# Run examples
docker run tasai micromamba run -n tasai python -m tasai.examples.example_parameter_determination
```

### Verify Installation

```bash
python -c "import tasai; print('TAS-AI installed successfully!')"
```

## Quick Start

```bash
# Run dashboard demo (or use `tasai-dashboard`)
python -m tasai.dashboard.app --port 8050 --debug

# Run parameter estimation example
python -m tasai.examples.example_parameter_determination

# Run model discrimination example
python -m tasai.examples.example_model_discrimination

# Run order parameter example
python -m tasai.examples.order_parameter_simulation
```

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    TAS-AI Framework                         │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐ │
│  │  Dashboard  │  │   Physics   │  │     Inference       │ │
│  │  (Dash UI)  │  │   Models    │  │   (MCMC/Bayes)      │ │
│  └──────┬──────┘  └──────┬──────┘  └──────────┬──────────┘ │
│         │                │                     │            │
│  ┌──────▼────────────────▼─────────────────────▼──────────┐│
│  │                  Acquisition Function                   ││
│  │         score = info_gain^η / (count + move)           ││
│  └──────────────────────────┬─────────────────────────────┘│
│                             │                               │
│  ┌──────────────────────────▼─────────────────────────────┐│
│  │              Instrument Interface                       ││
│  │    (Simulator / HTTP Proxy / Direct Control)           ││
│  └─────────────────────────────────────────────────────────┘│
└─────────────────────────────────────────────────────────────┘
```

## Documentation

- [User Manual](docs/MANUAL.md) - Complete guide for users
- [Dashboard Guide](docs/DASHBOARD.md) - Dashboard features and controls
- [Benchmarks](docs/BENCHMARKS.md) - Performance comparisons
- [Sunny Integration](docs/SUNNY.md) - Spin wave calculations
- [API Reference](docs/API.md) - Developer documentation

## Examples

| Example | Description |
|---------|-------------|
| `tasai/examples/example_parameter_determination.py` | Determine J1, J2, D from spin wave data |
| `tasai/examples/example_model_discrimination.py` | Test if J2 interactions are needed |
| `tasai/examples/example_with_motor_motion.py` | Include motor motion in optimization |
| `tasai/examples/order_parameter_simulation.py` | Phase transition (Tc, β) measurement |
| `tasai/examples/benchmark_jcns.py` | Compare against JCNS Log-GP method |

## Reproducing the Closed-Loop Results

The closed-loop drivers that produce the manuscript figures (Figure 9 hybrid handoff, Figure 10 LLM-audited run, and the Section 5 audit ablations) live in the companion `paper-tasai` repository, not in this library. The library provides the physics backends (`tasai.physics.SquareLatticeAFM`, `tasai.physics.SquareFMBilayer`), acquisition, resolution, and MCTS modules that those drivers import. Follow the reproducibility guide in the paper repository for the exact invocations used to generate each figure.


## Comparison with Other Approaches

| Feature | TAS-AI | gpCAM (ILL) | Log-GP (JCNS) | ANDiE (ORNL) |
|---------|--------|-------------|---------------|--------------|
| Physics-informed | ✅ | ❌ | ❌ | ✅ |
| Model discrimination | ✅ | ❌ | ❌ | ✅ |
| Motor motion | ✅ | ❌ | ❌ | ❌ |
| Log-space GP | ✅ | ❌ | ✅ | ❌ |
| Real-time dashboard | ✅ | ❌ | ❌ | ❌ |

## Requirements

- Python 3.9+
- NumPy, SciPy, Matplotlib
- Dash, dash-bootstrap-components, plotly, pandas (for dashboard)
- Optional: scikit-learn (for GP), emcee/bumps (for MCMC)
- Optional: Julia + Sunny.jl (for advanced spin wave calculations)

## Citation

If you use TAS-AI in your research, please cite:

```bibtex
@software{tasai2024,
  title = {TAS-AI: Autonomous Triple-Axis Spectrometer Control},
  author = {NIST Center for Neutron Research},
  year = {2024},
  url = {https://github.com/williamratcliff-nist/tasai}
}
```

## Related Work

- [ANDiE](https://doi.org/10.1063/5.0082956) - Autonomous Neutron Diffraction Explorer
- [gpCAM](https://github.com/lbl-camera/gpCAM) - Gaussian Process Campaign
- [AutoREFL](https://github.com/ncnr/autorefl) - Autonomous Reflectometry
- [SpinW](https://spinw.org) - Spin Wave Calculation Library
- [Sunny.jl](https://github.com/SunnySuite/Sunny.jl) - Modern Spin Dynamics

## License

This repository uses the NIST software statement in [LICENSE](LICENSE). In the
United States, software developed by NIST employees is not subject to copyright
protection under 17 U.S.C. 105; foreign rights may still apply.

## Contributing

Contributions welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.
