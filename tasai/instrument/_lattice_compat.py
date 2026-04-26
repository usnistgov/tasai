"""
Minimal Python 3 compatible lattice calculator for TAS resolution.

This provides the core functionality needed by rescalculator:
- Lattice class for crystal structure
- modvec function for Q magnitude calculation
- scalar function for dot products in reciprocal space
"""

import numpy as np
from typing import Union, Tuple

# Physical constants
eps = 1e-3


class Lattice:
    """
    Crystal lattice calculator for TAS resolution.

    Provides transformation matrices and orientation for resolution calculations.

    Parameters
    ----------
    a, b, c : float
        Lattice constants in Angstroms
    alpha, beta, gamma : float
        Lattice angles in degrees
    orient1 : array_like
        First orientation vector [h, k, l] shape (1, 3) or (3,)
    orient2 : array_like
        Second orientation vector [h, k, l] shape (1, 3) or (3,)
    """

    def __init__(
        self,
        a: float = 2 * np.pi,
        b: float = 2 * np.pi,
        c: float = 2 * np.pi,
        alpha: float = 90.0,
        beta: float = 90.0,
        gamma: float = 90.0,
        orient1: np.ndarray = None,
        orient2: np.ndarray = None,
    ):
        # Lattice parameters (can be arrays for multiple points)
        self.a = np.atleast_1d(np.asarray(a, dtype=np.float64))
        self.b = np.atleast_1d(np.asarray(b, dtype=np.float64))
        self.c = np.atleast_1d(np.asarray(c, dtype=np.float64))

        self.alphad = np.atleast_1d(np.asarray(alpha, dtype=np.float64))
        self.betad = np.atleast_1d(np.asarray(beta, dtype=np.float64))
        self.gammad = np.atleast_1d(np.asarray(gamma, dtype=np.float64))

        self.alpha = np.radians(self.alphad)
        self.beta = np.radians(self.betad)
        self.gamma = np.radians(self.gammad)

        # Number of points
        self.npts = len(self.a)

        # Default orientations
        if orient1 is None:
            orient1 = np.array([[1, 0, 0]])
        if orient2 is None:
            orient2 = np.array([[0, 1, 0]])

        self.orient1 = np.atleast_2d(orient1).T  # Shape (3, npts)
        self.orient2 = np.atleast_2d(orient2).T

        # Compute reciprocal lattice
        self._compute_reciprocal()

        # Compute metric tensors
        self._compute_metric_tensors()

        # Compute standard coordinate system
        self._compute_standard_system()

    def _compute_reciprocal(self):
        """
        Compute reciprocal lattice parameters.

        For reciprocal lattice:
        a* = 2π * (b × c) / V
        |a*| = 2π * b*c*sin(α) / V

        The metric tensor gstar is in units of (2π/Å)² = Å⁻².
        """
        cos_alpha = np.cos(self.alpha)
        cos_beta = np.cos(self.beta)
        cos_gamma = np.cos(self.gamma)
        sin_alpha = np.sin(self.alpha)
        sin_beta = np.sin(self.beta)
        sin_gamma = np.sin(self.gamma)

        # Volume (real space)
        V = self.a * self.b * self.c * np.sqrt(
            1 - cos_alpha**2 - cos_beta**2 - cos_gamma**2
            + 2 * cos_alpha * cos_beta * cos_gamma
        )

        # Reciprocal lattice parameters (include 2π factor)
        # |a*| = 2π * b*c*sin(α) / V
        self.astar = 2 * np.pi * self.b * self.c * sin_alpha / V
        self.bstar = 2 * np.pi * self.a * self.c * sin_beta / V
        self.cstar = 2 * np.pi * self.a * self.b * sin_gamma / V

        # Reciprocal lattice angles
        # For orthogonal systems, these equal 90 degrees
        # General formula: cos(α*) = (cos β cos γ - cos α) / (sin β sin γ)
        arg_alpha = (cos_beta * cos_gamma - cos_alpha) / (sin_beta * sin_gamma + eps)
        arg_beta = (cos_alpha * cos_gamma - cos_beta) / (sin_alpha * sin_gamma + eps)
        arg_gamma = (cos_alpha * cos_beta - cos_gamma) / (sin_alpha * sin_beta + eps)

        # Clamp to [-1, 1] to avoid numerical issues with arccos
        self.alphastar = np.arccos(np.clip(arg_alpha, -1, 1))
        self.betastar = np.arccos(np.clip(arg_beta, -1, 1))
        self.gammastar = np.arccos(np.clip(arg_gamma, -1, 1))

    def _compute_metric_tensors(self):
        """Compute metric tensors for real and reciprocal space."""
        # Real space metric tensor g
        cos_alpha = np.cos(self.alpha)
        cos_beta = np.cos(self.beta)
        cos_gamma = np.cos(self.gamma)

        self.g = np.zeros((3, 3, self.npts))
        self.g[0, 0, :] = self.a**2
        self.g[1, 1, :] = self.b**2
        self.g[2, 2, :] = self.c**2
        self.g[0, 1, :] = self.g[1, 0, :] = self.a * self.b * cos_gamma
        self.g[0, 2, :] = self.g[2, 0, :] = self.a * self.c * cos_beta
        self.g[1, 2, :] = self.g[2, 1, :] = self.b * self.c * cos_alpha

        # Reciprocal space metric tensor gstar
        cos_alphastar = np.cos(self.alphastar)
        cos_betastar = np.cos(self.betastar)
        cos_gammastar = np.cos(self.gammastar)

        self.gstar = np.zeros((3, 3, self.npts))
        self.gstar[0, 0, :] = self.astar**2
        self.gstar[1, 1, :] = self.bstar**2
        self.gstar[2, 2, :] = self.cstar**2
        self.gstar[0, 1, :] = self.gstar[1, 0, :] = self.astar * self.bstar * cos_gammastar
        self.gstar[0, 2, :] = self.gstar[2, 0, :] = self.astar * self.cstar * cos_betastar
        self.gstar[1, 2, :] = self.gstar[2, 1, :] = self.bstar * self.cstar * cos_alphastar

    def _compute_standard_system(self):
        """
        Compute standard right-handed coordinate system (x, y, z) in Q-space.

        The coordinate system is defined such that:
        - x is along the first orientation vector (orient1)
        - z is perpendicular to the scattering plane (orient1 x orient2)
        - y is perpendicular to x in the scattering plane (z x x)

        All vectors are stored in reciprocal lattice units (h, k, l).
        """
        # x along orient1 direction (unit vector in reciprocal lattice units)
        q1 = self._modvec_internal(
            self.orient1[0, :], self.orient1[1, :], self.orient1[2, :], 'latticestar'
        )

        self.x = np.zeros((3, self.npts))
        # Handle potential division by zero
        for i in range(self.npts):
            if q1[i] > eps:
                self.x[0, i] = self.orient1[0, i] / q1[i]
                self.x[1, i] = self.orient1[1, i] / q1[i]
                self.x[2, i] = self.orient1[2, i] / q1[i]
            else:
                self.x[0, i] = 1.0
                self.x[1, i] = 0.0
                self.x[2, i] = 0.0

        # z = orient1 x orient2 (perpendicular to scattering plane)
        self.z = np.zeros((3, self.npts))
        cross = np.cross(self.orient1.T, self.orient2.T).T
        qz = self._modvec_internal(cross[0, :], cross[1, :], cross[2, :], 'latticestar')

        # Check for collinear vectors
        for i in range(self.npts):
            if qz[i] > eps:
                self.z[0, i] = cross[0, i] / qz[i]
                self.z[1, i] = cross[1, i] / qz[i]
                self.z[2, i] = cross[2, i] / qz[i]
            else:
                print('WARNING: Orientation vectors are collinear')
                self.z[0, i] = 0.0
                self.z[1, i] = 0.0
                self.z[2, i] = 1.0

        # y = z x x (perpendicular to x in scattering plane)
        self.y = np.zeros((3, self.npts))
        cross_y = np.cross(self.z.T, self.x.T).T
        qy = self._modvec_internal(cross_y[0, :], cross_y[1, :], cross_y[2, :], 'latticestar')

        for i in range(self.npts):
            if qy[i] > eps:
                self.y[0, i] = cross_y[0, i] / qy[i]
                self.y[1, i] = cross_y[1, i] / qy[i]
                self.y[2, i] = cross_y[2, i] / qy[i]
            else:
                self.y[0, i] = 0.0
                self.y[1, i] = 1.0
                self.y[2, i] = 0.0

    def _modvec_internal(
        self, x: np.ndarray, y: np.ndarray, z: np.ndarray, latticetype: str
    ) -> np.ndarray:
        """Internal modvec calculation."""
        if latticetype == 'latticestar':
            g = self.gstar
        else:
            g = self.g

        result = np.zeros(self.npts)
        for i in range(self.npts):
            vec = np.array([x[i] if hasattr(x, '__getitem__') else x,
                           y[i] if hasattr(y, '__getitem__') else y,
                           z[i] if hasattr(z, '__getitem__') else z])
            result[i] = np.sqrt(vec @ g[:, :, i] @ vec)

        return result


def modvec(
    x: Union[float, np.ndarray],
    y: Union[float, np.ndarray],
    z: Union[float, np.ndarray],
    latticetype: str,
    lattice: Lattice,
) -> np.ndarray:
    """
    Compute the magnitude of a vector in real or reciprocal space.

    Parameters
    ----------
    x, y, z : float or array
        Vector components in (h, k, l) coordinates
    latticetype : str
        'lattice' for real space, 'latticestar' for reciprocal space
    lattice : Lattice
        Lattice object with metric tensor

    Returns
    -------
    magnitude : array
        Vector magnitude in inverse Angstroms (reciprocal) or Angstroms (real)
    """
    x = np.atleast_1d(np.asarray(x, dtype=np.float64))
    y = np.atleast_1d(np.asarray(y, dtype=np.float64))
    z = np.atleast_1d(np.asarray(z, dtype=np.float64))

    npts = len(x)

    if latticetype == 'latticestar':
        g = lattice.gstar
    else:
        g = lattice.g

    result = np.zeros(npts)
    for i in range(npts):
        gi = i if i < g.shape[2] else 0
        vec = np.array([x[i], y[i], z[i]])
        result[i] = np.sqrt(vec @ g[:, :, gi] @ vec)

    return result


def scalar(
    x1: Union[float, np.ndarray],
    y1: Union[float, np.ndarray],
    z1: Union[float, np.ndarray],
    x2: Union[float, np.ndarray],
    y2: Union[float, np.ndarray],
    z2: Union[float, np.ndarray],
    latticetype: str,
    lattice: Lattice,
) -> np.ndarray:
    """
    Compute the scalar product of two vectors in real or reciprocal space.

    Parameters
    ----------
    x1, y1, z1 : float or array
        First vector components
    x2, y2, z2 : float or array
        Second vector components
    latticetype : str
        'lattice' for real space, 'latticestar' for reciprocal space
    lattice : Lattice
        Lattice object with metric tensor

    Returns
    -------
    product : array
        Scalar product
    """
    x1 = np.atleast_1d(np.asarray(x1, dtype=np.float64))
    y1 = np.atleast_1d(np.asarray(y1, dtype=np.float64))
    z1 = np.atleast_1d(np.asarray(z1, dtype=np.float64))
    x2 = np.atleast_1d(np.asarray(x2, dtype=np.float64))
    y2 = np.atleast_1d(np.asarray(y2, dtype=np.float64))
    z2 = np.atleast_1d(np.asarray(z2, dtype=np.float64))

    npts = len(x1)

    if latticetype == 'latticestar':
        g = lattice.gstar
    else:
        g = lattice.g

    result = np.zeros(npts)
    for i in range(npts):
        gi = i if i < g.shape[2] else 0
        vec1 = np.array([x1[i], y1[i], z1[i]])
        vec2 = np.array([x2[i], y2[i], z2[i]])
        result[i] = vec1 @ g[:, :, gi] @ vec2

    return result


class Orientation:
    """Orientation matrix helper (placeholder for compatibility)."""

    def __init__(self, orient1, orient2):
        self.orient1 = np.atleast_2d(orient1)
        self.orient2 = np.atleast_2d(orient2)
