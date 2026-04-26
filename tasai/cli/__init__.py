"""
TAS-AI Command Line Interface
"""

import argparse
import sys
from importlib.metadata import version as pkg_version, PackageNotFoundError


def main():
    """Main entry point for TAS-AI CLI."""
    parser = argparse.ArgumentParser(
        description="TAS-AI: Autonomous Triple-Axis Spectrometer Control",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  tasai dashboard            Start the dashboard
  tasai simulate             Run simulation demo
  tasai benchmark            Run benchmarks
  
For more information, see: https://github.com/williamratcliff-nist/tasai
        """
    )
    
    parser.add_argument(
        'command',
        nargs='?',
        choices=['dashboard', 'simulate', 'benchmark', 'version'],
        default='version',
        help='Command to run'
    )
    
    parser.add_argument(
        '--simulation', '-s',
        action='store_true',
        help='Run in simulation mode'
    )
    
    args = parser.parse_args()
    
    if args.command == 'version':
        try:
            ver = pkg_version("tasai")
        except PackageNotFoundError:
            ver = "unknown"
        print(f"TAS-AI version {ver}")
        print("NIST Center for Neutron Research")
        
    elif args.command == 'dashboard':
        try:
            from tasai.dashboard.app import main as dashboard_main
            dashboard_main()
        except ImportError:
            print("Dashboard requires Dash. Install with:")
            print("  pip install tasai[dashboard]")
            sys.exit(1)
            
    elif args.command == 'simulate':
        print("Running simulation demo...")
        from tasai.examples.simulation_demo import main as sim_main
        sim_main()
        
    elif args.command == 'benchmark':
        print("Running benchmarks...")
        from tasai.examples.benchmark_jcns import main as bench_main
        bench_main()


if __name__ == "__main__":
    main()
