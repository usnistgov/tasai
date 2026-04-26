#!/usr/bin/env python3
"""
TAS-AI MCMC Benchmarking Script

Test MCMC timing on your machine with different configurations.

Run with:
    python benchmark_mcmc.py

Typical results on M1 MacBook Pro:
    Single-thread:  ~1.8 sec/run
    8-core parallel: ~0.35 sec/run (5x speedup)
"""

import time
import numpy as np

def benchmark():
    print("=" * 60)
    print("TAS-AI MCMC Benchmark")
    print("=" * 60)
    
    # Check CPU info
    import os
    n_cores = os.cpu_count() or 1
    print(f"\nDetected {n_cores} CPU cores")
    
    # Try to import components
    print("\nChecking dependencies...")
    
    has_bumps = False
    has_emcee = False
    
    try:
        import bumps
        has_bumps = True
        print("  ✓ BUMPS available")
    except ImportError:
        print("  ✗ BUMPS not installed (pip install bumps)")
    
    try:
        import emcee
        has_emcee = True
        print("  ✓ emcee available")
    except ImportError:
        print("  ✗ emcee not installed (pip install emcee)")
    
    if not has_emcee:
        print("\nNeed at least emcee for benchmarking. Install with: pip install emcee")
        return
    
    # Create test model
    print("\n" + "-" * 60)
    print("Setting up test...")
    
    from tasai.physics import IsingModel
    from tasai.inference import MCMCRunner
    
    # Create Ising model with 2 free parameters
    model = IsingModel(T_c=150.0, beta=0.325)
    print(f"  Model: IsingModel with {model.n_free} free parameters")
    
    # Generate synthetic data
    np.random.seed(42)
    T_data = np.linspace(80, 200, 20)
    I_true = np.array([model.compute_intensity(0, 0, 0, T) for T in T_data])
    I_noisy = I_true + 0.02 * np.random.randn(len(T_data))
    sigma = np.full_like(T_data, 0.02)
    
    print(f"  Data: {len(T_data)} points")
    
    # Benchmark configurations
    configs = [
        {"name": "Single-thread", "burn": 200, "steps": 200, "pop": 8, "parallel": False},
        {"name": "Single-thread (short)", "burn": 100, "steps": 100, "pop": 8, "parallel": False},
    ]
    
    if n_cores > 1:
        configs.extend([
            {"name": f"{n_cores}-core parallel", "burn": 200, "steps": 200, "pop": 8, "parallel": True},
            {"name": f"{n_cores}-core (short)", "burn": 100, "steps": 100, "pop": 8, "parallel": True},
        ])
    
    print("\n" + "-" * 60)
    print("Running benchmarks (3 runs each)...")
    print("-" * 60)
    
    results = []
    
    for config in configs:
        name = config.pop("name")
        
        # Create runner
        runner = MCMCRunner(model, backend='emcee', **config)
        runner.set_data(
            h=np.zeros_like(T_data), k=np.zeros_like(T_data),
            l=np.zeros_like(T_data), E=T_data,
            I=I_noisy, sigma=sigma
        )
        
        # Warm up
        _ = runner.run()
        
        # Time 3 runs
        times = []
        for i in range(3):
            t0 = time.perf_counter()
            chain = runner.run()
            t1 = time.perf_counter()
            times.append(t1 - t0)
        
        avg_time = np.mean(times)
        std_time = np.std(times)
        
        n_evals = config["pop"] * model.n_free * (config["burn"] + config["steps"])
        
        print(f"\n{name}:")
        print(f"  Time: {avg_time:.3f} ± {std_time:.3f} sec")
        print(f"  Evaluations: {n_evals}")
        print(f"  Rate: {n_evals/avg_time:.0f} evals/sec")
        print(f"  Chain shape: {chain.shape}")
        
        results.append({
            "name": name,
            "time": avg_time,
            "evals": n_evals,
            "rate": n_evals / avg_time
        })
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    baseline = results[0]["time"]
    for r in results:
        speedup = baseline / r["time"]
        print(f"  {r['name']:25s}: {r['time']:.3f}s ({speedup:.1f}x)")
    
    # Recommendations
    print("\n" + "-" * 60)
    print("RECOMMENDATIONS FOR YOUR MACHINE:")
    print("-" * 60)
    
    if n_cores >= 8:
        print("  ✓ You have 8+ cores - use parallel=True for best performance")
        print("  ✓ Expected speedup: 4-6x")
    elif n_cores >= 4:
        print("  ✓ You have 4+ cores - parallel=True will help")
        print("  ✓ Expected speedup: 2-4x")
    else:
        print("  • Few cores detected - single-thread may be sufficient")
    
    print("\n  For autonomous experiments:")
    print("    - Use burn=200, steps=200 for fast iterations")
    print("    - Use burn=500, steps=500 for final analysis")
    print("    - Set parallel=True in MCMCRunner for multi-core")
    
    print("\nExample usage:")
    print("  runner = MCMCRunner(model, burn=200, steps=200, parallel=True)")
    print("  chain = runner.run()")
    
    return results


if __name__ == "__main__":
    try:
        benchmark()
    except KeyboardInterrupt:
        print("\nBenchmark interrupted")
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
