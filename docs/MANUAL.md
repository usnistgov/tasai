# TAS-AI User Manual

## Table of Contents

1. [Introduction](#introduction)
2. [Installation](#installation)
3. [Getting Started](#getting-started)
4. [Core Concepts](#core-concepts)
5. [Running Experiments](#running-experiments)
6. [The Dashboard](#the-dashboard)
7. [Physics Models](#physics-models)
8. [Troubleshooting](#troubleshooting)

---

## Introduction

### What is TAS-AI?

TAS-AI is software that helps you run neutron scattering experiments more efficiently on triple-axis spectrometers (TAS). Instead of manually deciding where to measure next, TAS-AI uses machine learning and physics models to automatically select the most informative measurement points.

### Why Use TAS-AI?

**Traditional approach:**
1. Scientist guesses where to measure
2. Waits for measurement to complete
3. Looks at data, decides next point
4. Repeat hundreds of times

**With TAS-AI:**
1. Define your physics model and goals
2. TAS-AI automatically selects optimal points
3. Get results 2-5× faster with fewer measurements

### Key Capabilities

| What You Want | TAS-AI Feature |
|---------------|----------------|
| Find exchange parameters (J1, J2, D) | Parameter Determination mode |
| Test if a model is correct | Model Discrimination mode |
| Find transition temperature | Order Parameter mode |
| Efficient Q-E mapping | Agnostic GP mode |

For structure-driven model proposal, TAS-AI also includes a hypothesis
generation stack in `tasai.extensions`. The reusable
`GoodenoughKanamoriAnalyzer` performs periodic exchange-path enumeration and
orbital-aware Goodenough-Kanamori sign analysis, while
`GNNHypothesisGenerator` wraps that path analysis into candidate Hamiltonians
and can optionally swap in learned GNN backends.

---

## Installation

### Requirements

- Python 3.9 - 3.13 (3.14 not yet fully supported)
- 4 GB RAM minimum (8 GB recommended)
- Works on Windows, Mac, and Linux

### Option 1: Mamba/Conda (Recommended)

Mamba provides much faster dependency resolution than conda:

```bash
# Clone the repository
git clone https://github.com/williamratcliff-nist/tasai.git
cd tasai

# Create environment with mamba (recommended - much faster)
mamba env create -f environment.yml
mamba activate tasai

# Or with conda (slower, but works)
conda env create -f environment.yml
conda activate tasai
```

**Note**: Conda's classic solver can take several minutes. Mamba typically completes in under 30 seconds.

### Option 2: Pip Installation

```bash
# Clone the repository
git clone https://github.com/williamratcliff-nist/tasai.git
cd tasai

# Install with all dependencies
pip install -e ".[all]"

# Or minimal install + specific extras
pip install -e .
pip install dash dash-bootstrap-components plotly bumps scikit-learn
```

### Option 3: Docker

For reproducible environments without local installation:

```bash
# Build the image
docker build -t tasai .

# Run the dashboard (accessible at http://localhost:8050)
docker run -p 8050:8050 tasai micromamba run -n tasai python dashboard/app.py --host 0.0.0.0

# Run an example
docker run tasai micromamba run -n tasai python examples/example_parameter_determination.py

# Interactive shell
docker run -it tasai micromamba run -n tasai bash
```

### Verify Installation

```bash
python -c "import tasai; print('TAS-AI installed successfully!')"
```

### Optional: Julia/Sunny.jl

For advanced spin wave calculations:

```bash
# Install Julia from https://julialang.org
# Then in Julia:
using Pkg
Pkg.add("Sunny")
```

---

## Getting Started

### Your First Simulation

Let's run a simple example that determines exchange parameters from spin wave data:

```bash
cd tasai/examples
python example_parameter_determination.py
```

This will:
1. Simulate a magnetic material with known J1, J2, D values
2. Run TAS-AI to autonomously determine these parameters
3. Show you how well it recovered the true values

### Understanding the Output

```
=== Autonomous Parameter Determination ===
True parameters: J1=5.00, J2=0.80, D=0.150 meV

Iteration 1/8:
  Measuring at H=0.125, E=12.5 meV
  Intensity: 0.234 ± 0.015
  
Iteration 2/8:
  Measuring at H=0.250, E=24.0 meV  ← High J2 sensitivity!
  Intensity: 0.567 ± 0.024

...

Final estimates:
  J1 = 5.26 ± 0.41 meV (true: 5.00)
  J2 = 0.71 ± 0.10 meV (true: 0.80)
  D  = 0.01 ± 0.02 meV (true: 0.15)
```

---

## Core Concepts

### The Acquisition Function

TAS-AI decides where to measure using an "acquisition function" that balances:

```
Score = (Information Gain)^η / (Count Time + Move Time)
```

- **Information Gain**: How much will this measurement reduce uncertainty?
- **Count Time**: How long to count neutrons at this point?
- **Move Time**: How long to physically move motors to this position?
- **η (eta)**: Tuning parameter (0.5-1.0, default 0.7)

Higher η → Prioritize information gain (explore more)
Lower η → Prioritize efficiency (exploit known features)

### Physics Models

TAS-AI uses physics models to predict what you'll measure. Current models:

| Model | Use Case | Parameters |
|-------|----------|------------|
| `SquareLatticeFM` | 2D ferromagnet | J1, J2, D |
| `NNOnlyModel` | Simple NN exchange | J1, D |
| `OrderParameter` | Phase transitions | Tc, β, A |

### Model Discrimination

When you're not sure which model is correct, TAS-AI can help you decide:

```python
# Is J2 exchange needed, or is NN-only sufficient?
models = [NNOnlyModel(), SquareLatticeFM()]
weights = run_discrimination(models, data)

# Result: weights = [0.02, 0.98]
# → 98% probability that J2 is needed!
```

---

## Running Experiments

### Mode 1: Parameter Determination

**Goal:** Find the values of parameters in a known model.

```python
from tasai.sunny import SquareLatticeFM

# Define your model
model = SquareLatticeFM(J1=5.0, J2=0.5, D=0.1)  # Initial guesses

# Run autonomous loop
for iteration in range(20):
    # TAS-AI suggests next point
    H, E = acquisition.suggest_next()
    
    # Measure (or simulate)
    I, sigma = instrument.measure(H, 0, 0, E)
    
    # Update model
    model.add_observation(H, E, I, sigma)
    model.fit()
    
print(f"J1 = {model.J1} ± {model.J1_err}")
```

### Mode 2: Model Discrimination

**Goal:** Decide which physics model best describes your data.

```python
from tasai.sunny import SquareLatticeFM, NNOnlyModel

# Competing models
model_j1j2 = SquareLatticeFM()  # Has J2
model_nn = NNOnlyModel()         # J2 fixed to 0

# Run discrimination
for iteration in range(10):
    # Measure where models disagree most
    H, E = find_maximum_disagreement(model_j1j2, model_nn)
    I, sigma = instrument.measure(H, 0, 0, E)
    
    # Update both models
    model_j1j2.add_observation(H, E, I, sigma)
    model_nn.add_observation(H, E, I, sigma)
    
    # Calculate Bayes factor
    evidence_ratio = model_j1j2.evidence() / model_nn.evidence()
    print(f"Bayes factor: {evidence_ratio:.1f}")
```

### Mode 3: Order Parameter (Phase Transitions)

**Goal:** Find transition temperature Tc and critical exponent β.

```python
from tasai.physics import OrderParameterModel

model = OrderParameterModel()

# Measure at Bragg peak vs temperature
for iteration in range(30):
    T = acquisition.suggest_temperature()
    I, sigma = instrument.measure_bragg(T)
    
    model.add_observation(T, I, sigma)
    model.fit()

print(f"Tc = {model.Tc} ± {model.Tc_err} K")
print(f"β = {model.beta} ± {model.beta_err}")
```

### Mode 4: Agnostic Exploration

**Goal:** Map S(Q,ω) without prior physics knowledge.

```python
from tasai.core import AgnosticExplorer

# Define measurement bounds
bounds = [
    [0, 0.5],   # H range (r.l.u.)
    [0, 30],    # E range (meV)
]

explorer = AgnosticExplorer(bounds, use_log_gp=True)

# Autonomous mapping
for iteration in range(100):
    H, E = explorer.suggest_next()
    I, sigma = instrument.measure(H, 0, 0, E)
    explorer.add_observation([H, E], I, sigma)

# Get intensity map
H_grid, E_grid, I_map = explorer.get_intensity_map()
```

---

## The Dashboard

See [DASHBOARD.md](DASHBOARD.md) for the complete dashboard guide.

### Quick Start

```bash
python -m tasai.dashboard.app
```

### Key Panels

| Panel | What It Shows |
|-------|---------------|
| **Dispersion** | Current spin wave dispersion fit |
| **Parameters** | J1, J2, D values with uncertainties |
| **Model Weights** | Which model is winning (ANDiE-style) |
| **Info Gain** | Information gained per measurement |
| **Queue** | Upcoming measurements |

### Keyboard Shortcuts

| Key | Action |
|-----|--------|
| `Space` | Pause/Resume |
| `N` | Force next measurement |
| `R` | Reset experiment |
| `Q` | Quit |
| `?` | Show help |

---

## Physics Models

### Square Lattice Ferromagnet

The `SquareLatticeFM` model describes spin waves in a 2D square lattice:

**Hamiltonian:**
```
H = -J1 Σ Si·Sj (nearest neighbor)
    -J2 Σ Si·Sj (next-nearest neighbor)  
    -D  Σ (Sz)² (single-ion anisotropy)
```

**Dispersion relation:**
```
ω(H) = 2S [2J1(1 - cos(2πH)) + 2J2(1 - cos²(2πH))] + D(2S - 1)

with J1, J2 > 0 corresponding to ferromagnetic exchange because the
Hamiltonian is written as

H = -J1 Σ_NN Si·Sj - J2 Σ_NNN Si·Sj - D Σ_i (S_i^z)^2
```

**Parameters:**

| Parameter | Meaning | Typical Range |
|-----------|---------|---------------|
| J1 | NN exchange | 1-20 meV |
| J2 | NNN exchange | 0-5 meV |
| D | Anisotropy | 0-1 meV |
| S | Spin | 0.5-2.5 |

### Key Physics Insight

**Where does J2 matter most?**

At H = 0.25 (quarter of Brillouin zone):
- cos(2πH) = 0 → J1 term vanishes
- cos²(2πH) = 0 → the J2 contribution is maximal along the H=K cut

This is why TAS-AI often suggests measuring near H = 0.25 when discriminating J1-J2 models!

---

## Troubleshooting

### Common Issues

**Q: The fit isn't converging**

A: Try these steps:
1. Check that your initial parameter guesses are reasonable
2. Increase the number of measurements
3. Make sure you're measuring near the dispersion (not in background)

**Q: Model discrimination gives 50/50 weights**

A: The models may be too similar at the measured points. Try:
1. Running more iterations
2. Checking if you're measuring where models differ

**Q: Dashboard won't start**

A: Install the Dash dependencies: `pip install dash dash-bootstrap-components plotly pandas`

**Q: "No module named tasai"**

A: Make sure you installed with: `pip install -e .`

### Getting Help

- Check the [examples/](../examples/) directory
- Open an issue on GitHub
- Email: neutronscattering@nist.gov

---

## Glossary

| Term | Definition |
|------|------------|
| **TAS** | Triple-Axis Spectrometer |
| **r.l.u.** | Reciprocal Lattice Units |
| **meV** | Milli-electron-volt (energy unit) |
| **Acquisition function** | Formula that scores candidate measurements |
| **Bayes factor** | Ratio of model probabilities |
| **GP** | Gaussian Process (statistical model) |
| **MCMC** | Markov Chain Monte Carlo (sampling method) |
| **BZ** | Brillouin Zone |

---

*Last updated: January 2025*
