#!/usr/bin/env python3
"""
Example: Running TAS-AI in simulation mode

This demonstrates the complete workflow:
1. Initialize geometry and simulator
2. Run autonomous experiment loop
3. Visualize results

No real instrument required - uses internal simulation.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import List

from tasai.instrument.base import TASGeometry, MeasurementPoint
from tasai.instrument.simulator import TASSimulator, antiferromagnetic_2d
try:
    from tasai.instrument import TASResolutionCalculator, create_default_tas_config
except ImportError:
    TASResolutionCalculator = None
    create_default_tas_config = None


def main():
    print("=" * 60)
    print("TAS-AI Simulation Example")
    print("=" * 60)
    
    # =========================================================================
    # 1. Define crystal structure and geometry
    # =========================================================================
    print("\n1. Setting up TAS geometry...")
    
    geometry = TASGeometry(
        lattice_params=(3.9, 3.9, 13.0, 90, 90, 90),  # Cuprate-like
        orientation=((1, 0, 0), (0, 1, 0)),  # (H, K, 0) scattering plane
        ei_fixed=True,
        fixed_energy=14.7,  # meV
    )
    res_calc = None
    if TASResolutionCalculator is not None and create_default_tas_config is not None:
        res_calc = TASResolutionCalculator(
            lattice_params=(3.9, 3.9, 13.0, 90, 90, 90),
            orient1=[1, 0, 0],
            orient2=[0, 1, 0],
            exp_config=create_default_tas_config(
                efixed=14.7,
                hcol=(40, 40, 40, 40)
            )
        )
        print("   Using Cooper-Nathans resolution via rescalculator (40' collimations)")
    else:
        print("   [WARN] rescalculator not available; using simple Gaussian resolution")
    
    # =========================================================================
    # 2. Set up simulator with known S(Q,ω)
    # =========================================================================
    print("2. Initializing simulator...")
    
    # True parameters (what we're trying to discover)
    TRUE_J1 = 5.0  # meV
    TRUE_J2 = 0.5  # meV
    
    def sqw_function(h, k, l, E):
        return antiferromagnetic_2d(h, k, l, E, J1=TRUE_J1, J2=TRUE_J2, gamma=1.0)
    
    simulator = TASSimulator(
        geometry=geometry,
        sqw_function=sqw_function,
        background=0.1,
        intensity_scale=1000.0,
        time_scale=1000.0,  # 1000x speedup for demo
        seed=42,  # Reproducible
        resolution_calculator=res_calc
    )
    
    print(f"   True parameters: J1={TRUE_J1} meV, J2={TRUE_J2} meV")
    
    # =========================================================================
    # 3. Define scan region
    # =========================================================================
    print("3. Defining measurement region...")
    
    # We'll scan along (h, 0, 0) in energy
    h_values = np.linspace(0.1, 1.0, 10)
    E_values = np.linspace(1.0, 20.0, 20)
    
    # =========================================================================
    # 4. Run measurements (simplified - not using full autonomous loop)
    # =========================================================================
    print("4. Running simulated measurements...")
    
    measurements = []
    
    for h in h_values:
        for E in E_values:
            point = MeasurementPoint(
                h=h, k=0.0, l=0.0, E=E,
                count_time=30.0  # 30 seconds per point
            )
            
            # Check if accessible
            valid, reason = simulator.validate_point(point)
            if not valid:
                continue
            
            try:
                result = simulator.measure(point)
                measurements.append(result)
            except ValueError as e:
                print(f"   Skipping ({h:.2f}, 0, 0, {E:.1f}): {e}")
    
    print(f"   Completed {len(measurements)} measurements")
    print(f"   Total simulated time: {simulator.total_simulated_time:.1f} s")
    
    # =========================================================================
    # 5. Visualize results
    # =========================================================================
    print("5. Generating visualization...")
    
    # Extract data for plotting
    h_data = np.array([m.point.h for m in measurements])
    E_data = np.array([m.point.E for m in measurements])
    I_data = np.array([m.intensity for m in measurements])
    
    # Create figure
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Left: Intensity map
    ax1 = axes[0]
    scatter = ax1.scatter(h_data, E_data, c=I_data, cmap='hot', s=50)
    ax1.set_xlabel('[H H 0] (r.l.u.)')
    ax1.set_ylabel('E (meV)')
    ax1.set_title('Measured Intensity')
    plt.colorbar(scatter, ax=ax1, label='Intensity (counts/s)')
    
    # Add theoretical dispersion line
    h_theory = np.linspace(0.1, 1.0, 100)
    gamma_q = 0.5 * (np.cos(2*np.pi*h_theory) + 1)  # k=0
    omega_theory = 2 * TRUE_J1 * 0.5 * np.sqrt(np.maximum(1 - gamma_q**2, 0))
    ax1.plot(h_theory, omega_theory, 'w--', linewidth=2, label='Theory')
    ax1.legend()
    
    # Right: Constant-Q cut
    ax2 = axes[1]
    h_target = 0.5
    mask = np.abs(h_data - h_target) < 0.1
    if np.any(mask):
        ax2.errorbar(
            E_data[mask], I_data[mask],
            yerr=[m.uncertainty for m in np.array(measurements)[mask]],
            fmt='o', capsize=3
        )
    ax2.set_xlabel('E (meV)')
    ax2.set_ylabel('Intensity (counts/s)')
    ax2.set_title(f'Constant-Q cut at h={h_target}')
    
    plt.tight_layout()
    plt.savefig('tasai_simulation_example.png', dpi=150)
    print("   Saved: tasai_simulation_example.png")
    
    # =========================================================================
    # 6. Summary
    # =========================================================================
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"Total measurements: {len(measurements)}")
    print(f"Simulated experiment time: {simulator.total_simulated_time:.1f} s "
          f"({simulator.total_simulated_time/3600:.2f} hours)")
    print(f"Actual wall time: ~{len(measurements) * 0.001:.1f} s (with 1000x speedup)")
    print("\nTrue parameters:")
    print(f"  J1 = {TRUE_J1} meV")
    print(f"  J2 = {TRUE_J2} meV")
    print("\nNext steps:")
    print("  1. Implement acquisition function to select optimal next points")
    print("  2. Add MCMC fitting to extract parameters from data")
    print("  3. Close the loop for fully autonomous operation")
    
    return measurements


def demo_proxy_mode():
    """
    Demonstrate using the proxy server.
    
    Run this in a separate terminal first:
        python -m tasai.proxy_server.server --mode simulation
    """
    print("\n" + "=" * 60)
    print("Proxy Mode Demo")
    print("=" * 60)
    
    from tasai.instrument.proxy import NICEProxyClient
    from tasai.instrument.base import MeasurementPoint
    
    # Connect to proxy
    client = NICEProxyClient(
        proxy_url="http://localhost:8080",
        dry_run=False
    )
    
    # Set up callbacks
    def on_status(msg):
        print(f"  [STATUS] {msg}")
    
    def on_approval(point):
        print(f"  [APPROVAL] ({point.h:.3f}, {point.k:.3f}, {point.l:.3f}, {point.E:.2f})")
        # In real use, this would prompt the user
        return True  # Auto-approve for demo
    
    client.set_status_callback(on_status)
    client.set_approval_callback(on_approval)
    
    # Connect
    if not client.connect():
        print("Failed to connect to proxy. Is the server running?")
        print("Start it with: python -m tasai.proxy_server.server --mode simulation")
        return
    
    # Take a measurement
    print("\nTaking measurement...")
    point = MeasurementPoint(h=0.5, k=0.0, l=0.0, E=10.0, count_time=60.0)
    
    try:
        result = client.measure(point)
        print(f"\nResult: I = {result.intensity:.2f} ± {result.uncertainty:.2f}")
    except Exception as e:
        print(f"Measurement failed: {e}")


if __name__ == "__main__":
    # Run main simulation demo
    measurements = main()
    
    # Uncomment to test proxy mode (requires running server first)
    # demo_proxy_mode()
