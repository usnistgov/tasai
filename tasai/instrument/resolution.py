"""
TAS Resolution Calculator Integration

Wraps the rescalculator library to provide realistic resolution calculations
for triple-axis spectrometer simulations.

Default configuration: 40' collimations, fixed Ef=14.7 meV, Cooper-Nathans method.

This module requires:
- rescalculator (pip install rescalculator or from github.com/scattering/rescalculator)
- icp-lattice-calculator (pip install icp-lattice-calculator)
"""

import numpy as np
from typing import Optional, Dict, Tuple, Callable
import sys

# Try to import from installed packages first
try:
    # Use installed lattice_calculator (icp-lattice-calculator)
    from lattice_calculator import Lattice, modvec, scalar, Orientation
except ImportError:
    # Fall back to local Python 3 compatible lattice calculator
    from ._lattice_compat import Lattice, modvec, scalar, Orientation
    # Make lattice_calculator module available for rescalculator import
    import types
    lattice_calculator = types.ModuleType('lattice_calculator')
    lattice_calculator.Lattice = Lattice
    lattice_calculator.modvec = modvec
    lattice_calculator.scalar = scalar
    lattice_calculator.Orientation = Orientation
    sys.modules['lattice_calculator'] = lattice_calculator

# Import from rescalculator package
from rescalculator import TASResolution, ConvolutionCalculator


def create_default_tas_config(
    efixed: float = 14.7,
    hcol: Tuple[float, float, float, float] = (40, 40, 40, 40),
    vcol: Tuple[float, float, float, float] = (120, 120, 120, 120),
    mono_mosaic: float = 30,
    ana_mosaic: float = 30,
    sample_mosaic: float = 30,
    method: int = 0  # 0 = Cooper-Nathans, 1 = Popovici
) -> Dict:
    """
    Create a default TAS experiment configuration.

    Parameters
    ----------
    efixed : float
        Fixed final energy in meV (default 14.7)
    hcol : tuple
        Horizontal collimations in arc-minutes: (pre-mono, pre-sample, pre-ana, pre-det)
        Default: 40' throughout
    vcol : tuple
        Vertical collimations in arc-minutes
        Default: 120' throughout
    mono_mosaic : float
        Monochromator mosaic spread in arc-minutes (default 30)
    ana_mosaic : float
        Analyzer mosaic spread in arc-minutes (default 30)
    sample_mosaic : float
        Sample mosaic spread in arc-minutes (default 30)
    method : int
        Resolution method: 0 = Cooper-Nathans (default), 1 = Popovici

    Returns
    -------
    EXP : dict
        Experiment configuration dictionary for rescalculator
    """
    EXP = {
        'efixed': efixed,
        'infin': -1,  # Fixed Ef mode (-1 = fixed Ef, +1 = fixed Ei)
        'dir1': 1,    # Monochromator scattering sense (+1 = CCW)
        'dir2': 1,    # Analyzer scattering sense (+1 = CCW)
        'hcol': np.array(hcol, dtype=np.float64),  # Horizontal collimations (arcmin)
        'vcol': np.array(vcol, dtype=np.float64),  # Vertical collimations (arcmin)
        'mono': {
            'tau': 'pg(002)',      # PG(002) monochromator
            'mosaic': mono_mosaic,  # arc-minutes
            'vmosaic': mono_mosaic,
            'rh': 1e6,             # Horizontal radius (cm), 1e6 = flat
            'rv': 1e6,             # Vertical radius (cm)
        },
        'ana': {
            'tau': 'pg(002)',      # PG(002) analyzer
            'mosaic': ana_mosaic,   # arc-minutes
            'vmosaic': ana_mosaic,
            'rh': 1e6,
            'rv': 1e6,
        },
        'sample': {
            'mosaic': sample_mosaic,   # arc-minutes
            'vmosaic': sample_mosaic,
        },
        'arms': [200, 200, 150, 150, 100],  # Flight path lengths (cm): L0, L1p, L1, L2, L3
        'horifoc': -1,   # Horizontal focusing (-1 = off)
        'method': method,  # 0 = Cooper-Nathans
        'moncor': 1,     # Monitor correction (1 = on)
    }
    return EXP


class TASResolutionCalculator:
    """
    Resolution calculator for triple-axis spectrometer simulations.

    Wraps rescalculator to provide:
    - Resolution matrix calculations in (Qx, Qy, Qz, E) space
    - Resolution ellipse parameters (FWHM, orientation)
    - S(Q,w) convolution with resolution function

    Default configuration: 40' collimations, fixed Ef=14.7 meV, Cooper-Nathans.

    Parameters
    ----------
    lattice_params : tuple
        (a, b, c, alpha, beta, gamma) in Angstroms and degrees
    orient1 : array_like
        First orientation vector [h, k, l]
    orient2 : array_like
        Second orientation vector [h, k, l]
    exp_config : dict, optional
        Experiment configuration. If None, uses 40'/14.7 meV defaults.
    backend : str
        Computational backend: 'auto', 'numba', 'numpy', or 'pytorch'

    Example
    -------
    >>> res = TASResolutionCalculator(
    ...     lattice_params=(5.0, 5.0, 5.0, 90, 90, 90),
    ...     orient1=[1, 0, 0],
    ...     orient2=[0, 1, 0]
    ... )
    >>> # Get resolution at a single Q,E point
    >>> fwhm, R0 = res.get_resolution_fwhm(h=1.0, k=0.0, l=0.0, E=5.0)
    >>> print(f"Energy FWHM: {fwhm['E']:.3f} meV")
    """

    def __init__(
        self,
        lattice_params: Tuple[float, float, float, float, float, float],
        orient1: np.ndarray,
        orient2: np.ndarray,
        exp_config: Optional[Dict] = None,
        backend: str = 'numpy'
    ):
        a, b, c, alpha, beta, gamma = lattice_params

        # Create lattice calculator
        self.lattice = Lattice(
            a=a, b=b, c=c,
            alpha=alpha, beta=beta, gamma=gamma,
            orient1=np.atleast_2d(orient1),
            orient2=np.atleast_2d(orient2)
        )

        # Store experiment configuration
        self.exp_config = exp_config if exp_config is not None else create_default_tas_config()

        # Create resolution calculator
        self._res_calc = TASResolution(self.lattice, backend=backend)

    @property
    def efixed(self) -> float:
        """Fixed energy in meV."""
        return self.exp_config['efixed']

    @property
    def collimations(self) -> Dict[str, np.ndarray]:
        """Collimation settings in arc-minutes."""
        return {
            'horizontal': self.exp_config['hcol'],
            'vertical': self.exp_config['vcol']
        }

    def calculate_resolution_matrix(
        self,
        H: np.ndarray,
        K: np.ndarray,
        L: np.ndarray,
        W: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate resolution matrices at specified (H, K, L, W) points.

        Parameters
        ----------
        H, K, L : array_like
            Miller indices of Q points
        W : array_like
            Energy transfer values in meV

        Returns
        -------
        R0 : array
            Resolution prefactors, shape (npts,)
        RMS : array
            Resolution matrices in sample coordinates, shape (4, 4, npts)
            Matrix indices: [Qx, Qy, Qz, E]
        """
        H = np.atleast_1d(H)
        K = np.atleast_1d(K)
        L = np.atleast_1d(L)
        W = np.atleast_1d(W)

        npts = len(H)
        self.lattice.npts = npts

        # Expand EXP to match number of points
        EXP_list = [self.exp_config] * npts

        R0, RMS = self._res_calc.ResMatS(H, K, L, W, EXP_list)

        # Convert to numpy if needed
        if hasattr(RMS, 'cpu'):
            RMS = RMS.cpu().numpy()

        return np.asarray(R0).flatten(), np.asarray(RMS)

    def get_resolution_fwhm(
        self,
        h: float,
        k: float,
        l: float,
        E: float
    ) -> Tuple[Dict[str, float], float]:
        """
        Get resolution FWHM along principal axes at a single point.

        Parameters
        ----------
        h, k, l : float
            Miller indices
        E : float
            Energy transfer in meV

        Returns
        -------
        fwhm : dict
            FWHM values in each direction:
            - 'Qx': along Q direction (inverse Angstroms)
            - 'Qy': perpendicular to Q in scattering plane (inverse Angstroms)
            - 'Qz': out of scattering plane (inverse Angstroms)
            - 'E': energy (meV)
        R0 : float
            Resolution prefactor (normalization)
        """
        R0, RMS = self.calculate_resolution_matrix(
            np.array([h]), np.array([k]), np.array([l]), np.array([E])
        )

        # Extract matrix
        M = RMS[:, :, 0]

        # FWHM = 2*sqrt(2*ln(2)) / sqrt(M_ii) for Gaussian
        # Factor: 2*sqrt(2*ln(2)) = 2.355
        fwhm_factor = 2.355

        fwhm = {
            'Qx': fwhm_factor / np.sqrt(np.abs(M[0, 0])) if M[0, 0] > 0 else np.inf,
            'Qy': fwhm_factor / np.sqrt(np.abs(M[1, 1])) if M[1, 1] > 0 else np.inf,
            'Qz': fwhm_factor / np.sqrt(np.abs(M[3, 3])) if M[3, 3] > 0 else np.inf,
            'E': fwhm_factor / np.sqrt(np.abs(M[2, 2])) if M[2, 2] > 0 else np.inf,
        }

        return fwhm, float(R0[0])

    def get_resolution_ellipse(
        self,
        h: float,
        k: float,
        l: float,
        E: float,
        projection: str = 'QxE'
    ) -> Dict:
        """
        Get resolution ellipse parameters for a 2D projection.

        Parameters
        ----------
        h, k, l : float
            Miller indices
        E : float
            Energy transfer in meV
        projection : str
            Which 2D projection: 'QxE', 'QxQy', 'QyE'

        Returns
        -------
        ellipse : dict
            - 'center': (x, y) center coordinates
            - 'width': FWHM along major axis
            - 'height': FWHM along minor axis
            - 'angle': rotation angle in degrees
            - 'eigenvalues': eigenvalues of the 2x2 projected matrix
        """
        R0, RMS = self.calculate_resolution_matrix(
            np.array([h]), np.array([k]), np.array([l]), np.array([E])
        )

        M = RMS[:, :, 0]

        # Select indices for projection
        # Matrix ordering: [Qx, Qy, E, Qz]
        proj_map = {
            'QxE': (0, 2),
            'QxQy': (0, 1),
            'QyE': (1, 2),
            'QxQz': (0, 3),
            'QyQz': (1, 3),
            'EQz': (2, 3),
        }

        if projection not in proj_map:
            raise ValueError(f"Unknown projection '{projection}'. "
                           f"Use one of: {list(proj_map.keys())}")

        i, j = proj_map[projection]

        # Extract 2x2 submatrix
        M2 = np.array([[M[i, i], M[i, j]],
                       [M[j, i], M[j, j]]])

        # Eigenvalue decomposition
        eigenvalues, eigenvectors = np.linalg.eig(M2)
        eigenvalues = np.real(eigenvalues)
        eigenvectors = np.real(eigenvectors)

        # Sort by eigenvalue (largest first)
        idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]

        # FWHM along principal axes
        fwhm_factor = 2.355
        widths = fwhm_factor / np.sqrt(np.abs(eigenvalues))

        # Rotation angle (angle of first eigenvector)
        angle = np.degrees(np.arctan2(eigenvectors[1, 0], eigenvectors[0, 0]))

        return {
            'center': (h if i == 0 else (k if i == 1 else E),
                      E if j == 2 else (k if j == 1 else l)),
            'width': widths[0],
            'height': widths[1],
            'angle': angle,
            'eigenvalues': eigenvalues
        }

    def convolve_sqw(
        self,
        sqw_function: Callable,
        H: np.ndarray,
        K: np.ndarray,
        L: np.ndarray,
        W: np.ndarray,
        params: Dict = None,
        accuracy: Tuple[int, int] = (7, 0)
    ) -> np.ndarray:
        """
        Convolve S(Q,w) function with resolution.

        Parameters
        ----------
        sqw_function : callable
            Function sqw(H, K, L, W, params) -> intensity array of shape (modes, npts)
        H, K, L : array_like
            Miller indices
        W : array_like
            Energy transfers in meV
        params : dict, optional
            Parameters passed to sqw_function
        accuracy : tuple
            (M0, M1) accuracy parameters for Gaussian quadrature.
            Higher values = more accurate but slower.

        Returns
        -------
        intensity : array
            Convoluted intensity at each point
        """
        H = np.atleast_1d(H)
        K = np.atleast_1d(K)
        L = np.atleast_1d(L)
        W = np.atleast_1d(W)

        npts = len(H)
        self.lattice.npts = npts

        if params is None:
            params = {}

        EXP_list = [self.exp_config] * npts

        conv_calc = ConvolutionCalculator(self._res_calc, backend='numpy')

        return conv_calc.convolve(
            sqw_function, None, H, K, L, W, EXP_list, params,
            method='fixed', accuracy=accuracy
        )

    def get_energy_resolution(self, E: float, h: float = 0, k: float = 0, l: float = 1) -> float:
        """
        Get energy resolution (FWHM) at a given energy transfer.

        Convenience method for getting just the energy resolution.

        Parameters
        ----------
        E : float
            Energy transfer in meV
        h, k, l : float
            Miller indices (default: zone center along L)

        Returns
        -------
        dE : float
            Energy resolution FWHM in meV
        """
        fwhm, _ = self.get_resolution_fwhm(h, k, l, E)
        return fwhm['E']


def create_resolution_convolver(
    lattice_params: Tuple[float, float, float, float, float, float],
    orient1: np.ndarray,
    orient2: np.ndarray,
    efixed: float = 14.7,
    hcol: Tuple[float, float, float, float] = (40, 40, 40, 40)
) -> TASResolutionCalculator:
    """
    Factory function to create a resolution calculator with common TAS settings.

    Parameters
    ----------
    lattice_params : tuple
        (a, b, c, alpha, beta, gamma)
    orient1, orient2 : array_like
        Orientation vectors
    efixed : float
        Fixed energy (default 14.7 meV)
    hcol : tuple
        Horizontal collimations (default 40' throughout)

    Returns
    -------
    TASResolutionCalculator
        Configured resolution calculator

    Example
    -------
    >>> res = create_resolution_convolver(
    ...     (5.0, 5.0, 5.0, 90, 90, 90),
    ...     [1, 0, 0], [0, 1, 0],
    ...     efixed=14.7, hcol=(40, 40, 40, 40)
    ... )
    """
    exp_config = create_default_tas_config(efixed=efixed, hcol=hcol)
    return TASResolutionCalculator(
        lattice_params=lattice_params,
        orient1=np.array(orient1),
        orient2=np.array(orient2),
        exp_config=exp_config
    )
