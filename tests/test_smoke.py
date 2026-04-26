import numpy as np

from tasai.physics.analytical import SquareLatticeFM


def test_analytical_backend_runs():
    """Ensure the analytical backend executes without optional dependencies."""
    model = SquareLatticeFM(J1=5.0, J2=0.0)
    energy = float(model.dispersion(0.5))
    assert np.isfinite(energy)
    assert energy > 0
