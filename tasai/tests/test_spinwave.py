"""
Spin Wave Model Tests for TAS-AI

Tests the SpinWaveModel integration with pyspinw and compares results
against analytical expectations and Sunny.jl (when available).

These tests verify:
1. PySpinW backend produces correct dispersions
2. SpinWaveModel integrates properly with TASSimulator
3. Results match between backends (if both available)
4. Analytical results are reproduced

References:
- SpinW tutorials: https://spinw.org/tutorials/
- Sunny.jl SpinW ports: https://sunnysuite.github.io/Sunny.jl/stable/examples/spinw/
"""

import numpy as np
import pytest
import sys
import os

# Add paths for local development
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

pyspinw_path = os.environ.get("PYSPINW_PATH")
if pyspinw_path and os.path.exists(pyspinw_path) and pyspinw_path not in sys.path:
    sys.path.insert(0, pyspinw_path)


# Check what's available
PYSPINW_AVAILABLE = False
SUNNY_AVAILABLE = False

try:
    from pyspinw import SpinW, sw_model
    PYSPINW_AVAILABLE = True
except ImportError:
    pass

try:
    import juliacall
    SUNNY_AVAILABLE = True
except ImportError:
    pass


# Import TAS-AI modules
def import_spinwave_model():
    """Import SpinWaveModel."""
    from tasai.physics.spinwave import SpinWaveModel, SpinWaveConfig
    return SpinWaveModel, SpinWaveConfig


def import_simulator():
    """Import TASSimulator. Returns None tuple if unavailable."""
    try:
        from tasai.instrument.simulator import TASSimulator
        from tasai.instrument.base import TASGeometry, MeasurementPoint
        return TASSimulator, TASGeometry, MeasurementPoint
    except ImportError:
        return None, None, None


def import_compare_backends():
    """Import compare_backends."""
    from tasai.physics.spinwave import compare_backends
    return compare_backends


class TestPySpinWBasics:
    """Test basic PySpinW functionality."""

    @pytest.mark.skipif(not PYSPINW_AVAILABLE, reason="PySpinW not available")
    def test_fm_chain_dispersion(self):
        """
        Test FM Heisenberg chain dispersion.

        Analytical result: ω(q) = 2|J|S(1 - cos(2πq))
        For J=-1, S=1: ω(0.5) = 4 meV
        """
        sw = SpinW()
        sw.genlattice(lat_const=[3, 8, 8], angled=[90, 90, 90])
        sw.addatom(r=[0, 0, 0], S=1, label='Fe')
        sw.gencoupling(max_distance=4)
        sw.addmatrix(label='J', value=-1.0)  # FM
        sw.addcoupling(mat='J', bond=1)
        sw.genmagstr(mode='direct', k=[0, 0, 0], S=np.array([[0, 0, 1]]))

        spec = sw.spinwave([[0, 0, 0], [1, 0, 0]], n_pts=201)
        omega = np.real(spec['omega'][:, 0])

        # Analytical
        q = np.linspace(0, 1, 201)
        omega_analytical = 2 * 1.0 * 1.0 * (1 - np.cos(2 * np.pi * q))

        # Check RMS error
        rms = np.sqrt(np.mean((omega - omega_analytical)**2))
        assert rms < 0.01, f"RMS error {rms:.4f} exceeds tolerance"

        # Check key points
        assert abs(omega[0]) < 0.01, f"ω(0) = {omega[0]:.4f}, expected ~0"
        assert abs(omega[100] - 4.0) < 0.1, f"ω(0.5) = {omega[100]:.4f}, expected ~4"
        assert abs(omega[-1]) < 0.01, f"ω(1) = {omega[-1]:.4f}, expected ~0"

    @pytest.mark.skipif(not PYSPINW_AVAILABLE, reason="PySpinW not available")
    def test_afm_chain_dispersion(self):
        """Test AFM chain - should have correct energy range."""
        sw = sw_model('chain', 1.0)  # J = 1 meV AFM
        spec = sw.spinwave([[0, 0, 0], [1, 0, 0]], n_pts=101)
        omega = np.real(spec['omega'][:, 0])

        # AFM chain should have max ~4JS
        omega_max = np.max(omega)
        assert 3.0 < omega_max < 5.0, f"Max ω = {omega_max:.2f}, expected ~4"
        assert np.all(omega >= -1e-6), "Found negative frequencies"

    @pytest.mark.skipif(not PYSPINW_AVAILABLE, reason="PySpinW not available")
    def test_square_afm_goldstone(self):
        """Test square AFM has Goldstone mode at M point."""
        sw = sw_model('squareAF', 1.0)
        # Use a path ending at M point
        spec = sw.spinwave([[0, 0, 0], [0.5, 0.5, 0]], n_pts=11)
        omega = np.real(spec['omega'][-1, 0])  # Last point is M

        # M point should have Goldstone mode (ω ≈ 0)
        assert omega < 0.5, f"ω at M = {omega:.4f}, expected ~0 (Goldstone)"

    @pytest.mark.skipif(not PYSPINW_AVAILABLE, reason="PySpinW not available")
    def test_triangular_120_structure(self):
        """Test triangular AFM with 120° structure."""
        sw = sw_model('triAF', 1.0)
        spec = sw.spinwave([[0, 0, 0], [1/3, 1/3, 0]], n_pts=50)
        omega = np.real(spec['omega'])

        # All frequencies should be non-negative
        assert np.all(omega >= -1e-6), "Found negative frequencies"
        # Should have non-zero dispersion
        assert np.max(omega) > 0, "No dispersion found"


class TestSpinWaveModel:
    """Test SpinWaveModel integration with TAS-AI."""

    @pytest.mark.skipif(not PYSPINW_AVAILABLE, reason="PySpinW not available")
    def test_from_preset_chain(self):
        """Test creating model from preset."""
        SpinWaveModel, _ = import_spinwave_model()

        model = SpinWaveModel.from_preset('chain', J=-1.0)

        # Check it was created
        assert model is not None
        assert model.backend_name == 'pyspinw'
        assert len(model.parameters) >= 1

    @pytest.mark.skipif(not PYSPINW_AVAILABLE, reason="PySpinW not available")
    def test_compute_intensity(self):
        """Test computing S(Q,ω) at a point."""
        SpinWaveModel, _ = import_spinwave_model()

        model = SpinWaveModel.from_preset('chain', J=-1.0)

        # At zone center, zero energy transfer should be low
        I_low = model.compute_intensity(0, 0, 0, 0.5)

        # At zone boundary, near dispersion should be higher
        I_high = model.compute_intensity(0.5, 0, 0, 4.0)

        # Intensity at dispersion should be higher
        assert I_high >= I_low, f"Expected I(q=0.5,E=4) >= I(q=0,E=0.5)"

    @pytest.mark.skipif(not PYSPINW_AVAILABLE, reason="PySpinW not available")
    def test_get_sqw_function(self):
        """Test getting callable S(Q,ω) function."""
        SpinWaveModel, _ = import_spinwave_model()

        model = SpinWaveModel.from_preset('chain', J=-1.0)
        sqw = model.get_sqw_function()

        # Should be callable
        assert callable(sqw)

        # Should return a value
        result = sqw(0.5, 0, 0, 4.0)
        assert isinstance(result, (int, float))
        assert result >= 0

    @pytest.mark.skipif(not PYSPINW_AVAILABLE, reason="PySpinW not available")
    def test_get_dispersion(self):
        """Test getting full dispersion."""
        SpinWaveModel, _ = import_spinwave_model()

        model = SpinWaveModel.from_preset('chain', J=-1.0)
        spec = model.get_dispersion([[0, 0, 0], [1, 0, 0]], n_pts=50)

        assert 'omega' in spec
        assert 'hkl' in spec
        assert spec['omega'].shape[0] == 50

    @pytest.mark.skipif(not PYSPINW_AVAILABLE, reason="PySpinW not available")
    def test_parameter_update(self):
        """Test that updating parameters changes the dispersion."""
        SpinWaveModel, _ = import_spinwave_model()

        model = SpinWaveModel.from_preset('chain', J=-1.0)

        # Get dispersion with J=-1
        spec1 = model.get_dispersion([[0, 0, 0], [0.5, 0, 0]], n_pts=10)
        omega1 = np.real(spec1['omega'][5, 0])  # Middle point

        # Change J to -2 (doubles the bandwidth)
        model.set_parameter('J1', -2.0)
        spec2 = model.get_dispersion([[0, 0, 0], [0.5, 0, 0]], n_pts=10)
        omega2 = np.real(spec2['omega'][5, 0])

        # Energy should roughly double
        ratio = omega2 / omega1
        assert 1.5 < ratio < 2.5, f"Energy ratio {ratio:.2f}, expected ~2"


class TestTASSimulatorIntegration:
    """Test SpinWaveModel with TASSimulator."""

    @pytest.mark.skipif(not PYSPINW_AVAILABLE, reason="PySpinW not available")
    def test_simulator_with_spinwave(self):
        """Test using SpinWaveModel with TASSimulator."""
        SpinWaveModel, _ = import_spinwave_model()
        TASSimulator, TASGeometry, MeasurementPoint = import_simulator()

        if TASSimulator is None:
            pytest.skip("TASSimulator not available")

        # Create spin wave model
        model = SpinWaveModel.from_preset('chain', J=-1.0)
        sqw_func = model.get_sqw_function()

        # Create geometry with correct signature
        geometry = TASGeometry(
            lattice_params=(3.0, 3.0, 3.0, 90.0, 90.0, 90.0),
            orientation=((1, 0, 0), (0, 1, 0)),
            fixed_energy=14.7
        )

        # Create simulator
        simulator = TASSimulator(
            geometry=geometry,
            sqw_function=sqw_func,
            time_scale=1000.0,  # Fast simulation
            noise_model='none'  # No noise for reproducibility
        )

        # Use a point that should be accessible
        # Low Q, moderate energy transfer
        point = MeasurementPoint(h=0.25, k=0, l=0, E=2.0, count_time=1.0)

        # First check if point is valid, skip if geometry doesn't support it
        valid, reason = simulator.validate_point(point)
        if not valid:
            pytest.skip(f"Test point not accessible in this geometry: {reason}")

        result = simulator.measure(point)

        assert result is not None
        assert result.intensity >= 0


class TestBackendComparison:
    """Compare results between backends."""

    @pytest.mark.skipif(not PYSPINW_AVAILABLE, reason="PySpinW not available")
    def test_pyspinw_vs_analytical_fm(self):
        """Compare PySpinW FM chain with analytical result."""
        # Use PySpinW directly for this comparison
        sw = SpinW()
        sw.genlattice(lat_const=[3, 8, 8], angled=[90, 90, 90])
        sw.addatom(r=[0, 0, 0], S=1, label='Fe')
        sw.gencoupling(max_distance=4)
        sw.addmatrix(label='J', value=-1.0)  # FM
        sw.addcoupling(mat='J', bond=1)
        sw.genmagstr(mode='direct', k=[0, 0, 0], S=np.array([[0, 0, 1]]))

        spec = sw.spinwave([[0, 0, 0], [1, 0, 0]], n_pts=101)
        omega = np.real(spec['omega'][:, 0])

        # Analytical: ω = 2|J|S(1-cos(2πq))
        q = np.linspace(0, 1, 101)
        omega_analytical = 2 * 1.0 * 1.0 * (1 - np.cos(2 * np.pi * q))

        rms = np.sqrt(np.mean((omega - omega_analytical)**2))
        assert rms < 0.1, f"RMS error {rms:.4f} exceeds tolerance"

    @pytest.mark.skipif(not (PYSPINW_AVAILABLE and SUNNY_AVAILABLE),
                        reason="Both backends required")
    def test_pyspinw_vs_sunny(self):
        """Compare PySpinW and Sunny.jl for FM chain."""
        from tasai.physics.spinwave import compare_backends, SpinWaveConfig

        config = SpinWaveConfig(
            lat_const=(3, 8, 8),
            angles=(90, 90, 90),
            atoms=[([0, 0, 0], 1.0, 'Fe')],
            propagation_k=(0, 0, 0),
        )

        results = compare_backends(
            model_config=config,
            exchanges={'J1': -1.0},
            bonds={'J1': 1},
            Q_path=[[0, 0, 0], [1, 0, 0]],
            n_pts=50
        )

        if 'comparison' in results:
            assert results['comparison']['match'], (
                f"Backends disagree: max diff = {results['comparison']['max_diff']:.4e}"
            )


class TestKnownSystems:
    """Test against known magnetic systems from literature."""

    @pytest.mark.skipif(not PYSPINW_AVAILABLE, reason="PySpinW not available")
    def test_sw01_fm_chain(self):
        """
        SW01 - FM Heisenberg chain (Sunny tutorial port)

        Expected: ω(q) = 2|J|S(1 - cos(2πq))
        """
        sw = SpinW()
        sw.genlattice(lat_const=[3, 8, 8], angled=[90, 90, 90])
        sw.addatom(r=[0, 0, 0], S=1, label='MCu1')
        sw.gencoupling(max_distance=4)
        sw.addmatrix(label='Ja', value=-1.0)
        sw.addcoupling(mat='Ja', bond=1)
        sw.genmagstr(mode='direct', k=[0, 0, 0], S=np.array([[0, 0, 1]]))

        spec = sw.spinwave([[0, 0, 0], [1, 0, 0]], n_pts=401)
        omega = np.real(spec['omega'][:, 0])

        # Check zone boundary
        assert abs(omega[200] - 4.0) < 0.1, f"ω(0.5) = {omega[200]:.4f}, expected 4.0"
        # Check zone center
        assert abs(omega[0]) < 0.05, f"ω(0) = {omega[0]:.4f}, expected 0"

    @pytest.mark.skipif(not PYSPINW_AVAILABLE, reason="PySpinW not available")
    def test_sw04_square_afm(self):
        """
        SW04 - Square lattice AFM

        Expected: Goldstone mode at M=(0.5,0.5,0)
        """
        sw = sw_model('squareAF', 1.0)
        spec = sw.spinwave([[0, 0, 0], [0.5, 0.5, 0]], n_pts=51)
        omega = np.real(spec['omega'])

        # M point (end of path) should have Goldstone
        omega_M = omega[-1, 0]
        assert omega_M < 0.5, f"ω at M = {omega_M:.4f}, expected Goldstone (~0)"

        # All positive
        assert np.all(omega >= -1e-6), "Found negative frequencies"

    @pytest.mark.skipif(not PYSPINW_AVAILABLE, reason="PySpinW not available")
    def test_kagome_120(self):
        """
        Kagome lattice with 120° order.

        Classical ground state energy: E = -J per site for 120° structure
        """
        sw = SpinW()
        sw.genlattice(lat_const=[6, 6, 10], angled=[90, 90, 120])

        sw.addatom(r=[0.5, 0, 0], S=1, label='Fe1')
        sw.addatom(r=[0, 0.5, 0], S=1, label='Fe2')
        sw.addatom(r=[0.5, 0.5, 0], S=1, label='Fe3')

        sw.gencoupling(max_distance=4)
        sw.addmatrix(label='J1', value=1.0)
        sw.addcoupling(mat='J1', bond=1)
        sw.addcoupling(mat='J1', bond=2)
        sw.addcoupling(mat='J1', bond=3)

        # 120° structure
        phi = np.array([0, 2*np.pi/3, 4*np.pi/3])
        S = np.array([[np.cos(p), np.sin(p), 0] for p in phi])
        sw.genmagstr(mode='direct', k=[0, 0, 0], S=S)

        spec = sw.spinwave([[0, 0, 0], [0.5, 0, 0]], n_pts=50)
        omega = np.real(spec['omega'])

        # All non-negative (ground state should be stable)
        assert np.all(omega >= -0.1), "Found unstable modes (negative ω)"


def run_all_tests():
    """Run all tests and report results."""
    print("\n" + "=" * 60)
    print("TAS-AI Spin Wave Model Tests")
    print("=" * 60)

    print(f"\nBackend availability:")
    print(f"  PySpinW: {'Available' if PYSPINW_AVAILABLE else 'Not installed'}")
    print(f"  Sunny.jl: {'Available' if SUNNY_AVAILABLE else 'Not installed'}")

    if not PYSPINW_AVAILABLE:
        print("\nWARNING: PySpinW not available. Install with:")
        print("  pip install pyspinw")
        print("  Or: pip install -e /path/to/pyspinw")
        return False

    # Run pytest
    import subprocess
    result = subprocess.run(
        ['python', '-m', 'pytest', __file__, '-v', '--tb=short'],
        capture_output=False
    )
    return result.returncode == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
