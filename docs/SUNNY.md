# Sunny.jl Integration Guide

TAS-AI includes spin wave calculation capabilities inspired by [Sunny.jl](https://github.com/SunnySuite/Sunny.jl), a modern Julia library for spin dynamics simulations.

## Overview

### What is Sunny.jl?

Sunny.jl is a Julia package for simulating spin systems, developed at ORNL. It can:
- Calculate spin wave dispersions using linear spin wave theory
- Perform classical Monte Carlo simulations
- Handle complex magnetic structures (spirals, multi-k, etc.)
- Compute S(Q,ω) for comparison with neutron data

### TAS-AI's Approach

TAS-AI includes **pure Python implementations** of common spin wave models, providing:
- Fast analytical calculations without Julia dependency
- Direct integration with Bayesian inference
- Automatic differentiation compatibility (JAX/PyTorch)

For complex materials, you can optionally use the full Sunny.jl via Julia interop.

## Built-in Models

### SquareLatticeFM

A 2D square lattice ferromagnet with nearest-neighbor (J1) and next-nearest-neighbor (J2) exchange.

```python
from tasai.sunny import SquareLatticeFM

# Create model with parameters
model = SquareLatticeFM(
    J1=5.0,   # Nearest-neighbor exchange (meV), positive = FM
    J2=0.5,   # Next-nearest-neighbor exchange (meV), positive = FM
    D=0.1,    # Single-ion anisotropy (meV)
    S=1.0     # Spin magnitude
)

# Calculate dispersion at specific Q-point
H, K, L = 0.25, 0.25, 0
omega = model.dispersion(H, K, L)
print(f"Energy at (0.25, 0.25, 0): {omega:.2f} meV")

# Calculate S(Q,E) intensity
E = 15.0  # Energy transfer
I = model.intensity(H, K, L, E, eta=0.5)
print(f"Intensity: {I:.4f}")
```

**Hamiltonian:**
```
H = -J1 Σ<i,j> Si·Sj - J2 Σ<<i,j>> Si·Sj - D Σi (Szi)²
```

**Dispersion along the HHL cut (H=K):**
```
ω(H) = 2S [2J1(1 - cos(2πH)) + 2J2(1 - cos²(2πH))] + D(2S - 1)

For S = 1/2, the single-ion shift vanishes exactly.
```

### NNOnlyModel

Simplified model with only nearest-neighbor exchange (J2 = 0).

```python
from tasai.sunny import NNOnlyModel

model = NNOnlyModel(J1=5.0, D=0.1, S=1.0)

# This is equivalent to:
# SquareLatticeFM(J1=5.0, J2=0.0, D=0.1, S=1.0)
```

Use this for model discrimination to test whether J2 is needed.

### OrderParameterModel

For phase transition measurements (temperature-dependent order parameter).

```python
from tasai.physics import OrderParameterModel

model = OrderParameterModel(
    Tc=150.0,   # Transition temperature (K)
    beta=0.33,  # Critical exponent
    A=1.0       # Amplitude
)

# Calculate intensity vs temperature
T = 100.0
I = model.intensity(T)
```

**Functional form:**
```
I(T) = A × (1 - T/Tc)^(2β)  for T < Tc
I(T) = 0                     for T ≥ Tc
```

## Using the Models

### Parameter Estimation

```python
from tasai.sunny import SquareLatticeFM
import numpy as np

# Create model with initial guesses
model = SquareLatticeFM(J1=5.0, J2=0.5, D=0.1)

# Define parameter bounds
model.set_bounds({
    'J1': (0, 20),
    'J2': (0, 5),
    'D': (0, 1)
})

# Add experimental data
data = [
    # (H, K, L, E, I_measured, sigma)
    (0.1, 0.1, 0, 8.5, 0.25, 0.02),
    (0.2, 0.2, 0, 16.2, 0.42, 0.03),
    (0.3, 0.3, 0, 22.1, 0.31, 0.02),
]

for H, K, L, E, I, sigma in data:
    model.add_observation(H, K, L, E, I, sigma)

# Fit parameters
result = model.fit()
print(f"J1 = {result['J1']:.2f} ± {result['J1_err']:.2f} meV")
print(f"J2 = {result['J2']:.2f} ± {result['J2_err']:.2f} meV")
```

### Simulating Measurements

```python
# Simulate a measurement with Poisson statistics
H, K, L, E = 0.25, 0.25, 0, 20.0
I_true = model.intensity(H, K, L, E, eta=0.5)

# Add realistic noise
I_measured, sigma = model.simulate_measurement(
    H, K, L, E,
    count_time=60.0,
    count_rate=100.0
)

print(f"True intensity: {I_true:.4f}")
print(f"Measured: {I_measured:.4f} ± {sigma:.4f}")
```

### Model Discrimination

```python
from tasai.sunny import SquareLatticeFM, NNOnlyModel

# Two competing models
model_j1j2 = SquareLatticeFM(J1=5.0, J2=0.5, D=0.1)
model_nn = NNOnlyModel(J1=5.0, D=0.1)

# Find where they differ most
H_values = np.linspace(0, 0.5, 100)
E_disp_j1j2 = [model_j1j2.dispersion(H, H, 0) for H in H_values]
E_disp_nn = [model_nn.dispersion(H, H, 0) for H in H_values]

difference = np.abs(np.array(E_disp_j1j2) - np.array(E_disp_nn))
H_best = H_values[np.argmax(difference)]
print(f"Maximum discrimination at H = {H_best:.3f}")
# → H = 0.25 (quarter BZ, where J2 matters most)
```

## HHL Zone Physics

### Why H = K (HHL zone)?

For square lattice systems, the HHL high-symmetry direction (H = K) is often measured because:
1. Simplifies the dispersion to 1D
2. Contains zone center (0,0,0) and zone boundary (0.5,0.5,0)
3. J2 effects are maximal at H = 0.25

### Dispersion Along HHL

```python
# Calculate dispersion along (H,H,0)
H_values = np.linspace(0, 0.5, 100)
E_values = [model.dispersion(H, H, 0) for H in H_values]

import matplotlib.pyplot as plt
plt.plot(H_values, E_values)
plt.xlabel('H = K (r.l.u.)')
plt.ylabel('Energy (meV)')
plt.title('Spin wave dispersion along (H,H,0)')
```

### Key Q-points

| Q-point | Name | J1 sensitivity | J2 sensitivity |
|---------|------|----------------|----------------|
| (0,0,0) | Γ | Low | Low |
| (0.25,0.25,0) | M/2 | Medium | **High** |
| (0.5,0.5,0) | M | High | Medium |
| (0.5,0,0) | X | High | Low |

## Advanced: Julia Interop

For complex magnetic structures not covered by built-in models, you can call Sunny.jl directly.

### Setup

```bash
# Install Julia
# Download from https://julialang.org/downloads/

# Install Sunny.jl
julia -e 'using Pkg; Pkg.add("Sunny")'

# Install PyJulia
pip install julia
python -c "import julia; julia.install()"
```

### Basic Usage

```python
from julia import Main

# Load Sunny
Main.eval('using Sunny')

# Create a crystal
Main.eval('''
crystal = Crystal(
    latvecs = [4.0 0 0; 0 4.0 0; 0 0 10.0],
    positions = [[0, 0, 0]],
    spacegroup = 1
)
''')

# Create spin system
Main.eval('''
sys = System(crystal, (1,1,1), [SpinInfo(1, S=1)])
set_exchange!(sys, 1.0, Bond(1, 1, [1, 0, 0]))
''')

# Calculate spin waves
Main.eval('''
swt = SpinWaveTheory(sys)
q = [0.25, 0.25, 0]
disp = dispersion(swt, [q])
''')

omega = Main.disp[0]
print(f"Energy: {omega} meV")
```

### Using the SunnyInterface

TAS-AI provides a convenience wrapper:

```python
from tasai.sunny import SunnyInterface

# Initialize (starts Julia if needed)
sunny = SunnyInterface()

# Define crystal and Hamiltonian
sunny.set_crystal(
    latvecs=[[4, 0, 0], [0, 4, 0], [0, 0, 10]],
    positions=[[0, 0, 0]],
    spins=[1.0]
)

sunny.add_exchange(J=5.0, bond=[1, 0, 0])  # NN
sunny.add_exchange(J=0.5, bond=[1, 1, 0])  # NNN

# Calculate dispersion
q_points = [[h, h, 0] for h in np.linspace(0, 0.5, 50)]
energies = sunny.dispersion(q_points)
```

## Performance

### Built-in Python Models

| Operation | Time |
|-----------|------|
| Single dispersion | 0.01 ms |
| 100-point dispersion | 0.5 ms |
| S(Q,E) intensity | 0.05 ms |
| 50×100 intensity map | 10 ms |
| Grid search fit | 100 ms |

### With Julia/Sunny.jl

| Operation | Time |
|-----------|------|
| Julia startup | 2-5 s (once) |
| Crystal setup | 10 ms |
| Single dispersion | 1 ms |
| 100-point dispersion | 20 ms |

**Recommendation:** Use built-in Python models for autonomous loops (faster). Use Sunny.jl for complex structures or validation.

## Extending the Models

### Custom Dispersion

```python
from tasai.sunny import SpinWaveModel

class MyCustomModel(SpinWaveModel):
    """Custom model with additional interactions."""
    
    def __init__(self, J1, J2, J3, D):
        super().__init__()
        self.J1 = J1
        self.J2 = J2
        self.J3 = J3  # Third-neighbor
        self.D = D
    
    def dispersion(self, H, K, L):
        # Your dispersion relation here
        A = (2 * self.J1 * (np.cos(2*np.pi*H) + np.cos(2*np.pi*K) - 2)
             + 2 * self.J2 * (np.cos(2*np.pi*(H+K)) + np.cos(2*np.pi*(H-K)) - 2)
             + 2 * self.J3 * (np.cos(4*np.pi*H) + np.cos(4*np.pi*K) - 2))
        
        omega = 2 * self.S * np.sqrt(np.abs(A + self.D) * np.abs(A + self.D + 2*self.D*self.S))
        return omega
    
    @property
    def free_parameters(self):
        return ['J1', 'J2', 'J3', 'D']
```

### Adding to TAS-AI

```python
# Register your model
from tasai.sunny import register_model

register_model('my_custom', MyCustomModel)

# Use in experiments
model = MyCustomModel(J1=5.0, J2=0.5, J3=0.1, D=0.1)
```

## Comparison with SpinW

| Feature | TAS-AI/Sunny | SpinW |
|---------|--------------|-------|
| Language | Python/Julia | MATLAB |
| License | Open source | Open source |
| Speed | Fast | Fast |
| Complexity | Simple-moderate | Any |
| Fitting | Built-in | External |
| Autonomous | Yes | No |

**When to use SpinW:**
- Complex multi-sublattice structures
- Need symbolic calculations
- Existing SpinW scripts

**When to use TAS-AI:**
- Autonomous experiments
- Parameter fitting
- Model discrimination

## Troubleshooting

### Julia won't start

```bash
# Check Julia is installed
julia --version

# Reinstall PyJulia
pip uninstall julia
pip install julia
python -c "import julia; julia.install()"
```

### Import errors

```python
# Make sure Sunny is installed
from julia import Main
Main.eval('using Pkg; Pkg.add("Sunny")')
```

### Slow Julia startup

Julia has ~2-5 second startup time. For interactive use, keep the session alive:

```python
# Initialize once
from tasai.sunny import SunnyInterface
sunny = SunnyInterface(keep_alive=True)

# Reuse for multiple calculations
sunny.dispersion(q1)
sunny.dispersion(q2)  # Fast - no restart
```

---

## References

1. Sunny.jl documentation: https://sunnysuite.github.io/Sunny.jl/

2. Linear spin wave theory: S. Toth and B. Lake, J. Phys.: Condens. Matter 27, 166002 (2015)

3. SpinW: https://spinw.org/
