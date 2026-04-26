"""Analytic square-lattice bilayer ferromagnet spin-wave model.

Two ferromagnetic square-lattice sheets coupled by an interlayer exchange
J_perp. The in-plane dispersion is the standard square-lattice FM branch
with optional easy-axis gap D(2S-1); the bilayer coupling splits this into
acoustic and optic branches separated by

    Δ_optic = 2 z_perp S J_perp,

with L-dependent structure-factor weights

    w_ac(L) = cos²(π z_bi L),  w_op(L) = sin²(π z_bi L),

where z_bi is the interlayer spacing in r.l.u. along c*.

This module implements the bilayer backend used in the §5.3.2 audit
ablation of the TAS-AI manuscript. The single-branch comparator is the
same object constructed with J_perp = 0.
"""

from __future__ import annotations

import math
from typing import Tuple


class SquareFMBilayer:
    """Analytic bilayer ferromagnet with acoustic + optic branches.

    Parameters
    ----------
    S : float
        Spin quantum number.
    J_par : float
        In-plane ferromagnetic exchange (meV, positive convention).
    J_perp : float
        Interlayer exchange (meV). Set to 0 to obtain the monolayer
        comparator used in the §5.3.2 ablation.
    D : float
        Easy-axis anisotropy (meV). Contributes D(2S-1) to the LSWT gap.
    z_bi : float
        Interlayer spacing in r.l.u. along c*. Controls the L-dependent
        weight partition between acoustic and optic branches.
    z_perp : int
        Number of nearest interlayer neighbors (typically 1 for a simple
        bilayer).
    gamma : float
        Half-width (HWHM) of the Lorentzian energy line shape (meV).
    amp : float
        Peak intensity scale (arb. units).
    background : float
        Additive flat background.
    """

    def __init__(
        self,
        S: float,
        J_par: float,
        J_perp: float,
        D: float,
        z_bi: float,
        z_perp: int,
        gamma: float,
        amp: float,
        background: float,
    ) -> None:
        self.S = float(S)
        self.J_par = float(J_par)
        self.J_perp = float(J_perp)
        self.D = float(D)
        self.z_bi = float(z_bi)
        self.z_perp = int(z_perp)
        self.gamma = float(gamma)
        self.amp = float(amp)
        self.background = float(background)

    def anisotropy_gap(self) -> float:
        return float(self.D * (2.0 * self.S - 1.0))

    def omega_mono(self, H: float, K: float) -> float:
        """Monolayer / acoustic-branch energy (meV) at in-plane (H, K)."""
        return float(
            2.0
            * self.J_par
            * self.S
            * (2.0 - math.cos(2.0 * math.pi * H) - math.cos(2.0 * math.pi * K))
            + self.anisotropy_gap()
        )

    def optic_shift(self) -> float:
        """Interlayer-coupling-induced acoustic-to-optic gap (meV)."""
        return float(2.0 * self.z_perp * self.S * self.J_perp)

    def omega_ac(self, H: float, K: float) -> float:
        return self.omega_mono(H, K)

    def omega_op(self, H: float, K: float) -> float:
        return float(self.omega_mono(H, K) + self.optic_shift())

    def bilayer_weights(self, L: float) -> Tuple[float, float]:
        """Return (acoustic, optic) spectral weights as a function of L (r.l.u.)."""
        phi = math.pi * self.z_bi * float(L)
        return float(math.cos(phi) ** 2), float(math.sin(phi) ** 2)

    def lorentzian(self, E: float, E0: float) -> float:
        return float(self.gamma / (((float(E) - float(E0)) ** 2) + self.gamma ** 2))

    def intensity_bilayer(self, H: float, K: float, L: float, E: float) -> float:
        """Total intensity (acoustic + optic) at (H, K, L, E)."""
        w_ac, w_op = self.bilayer_weights(L)
        return float(
            self.background
            + self.amp
            * (
                w_ac * self.lorentzian(E, self.omega_ac(H, K))
                + w_op * self.lorentzian(E, self.omega_op(H, K))
            )
        )

    def intensity_monolayer(self, H: float, K: float, L: float, E: float) -> float:
        """Comparator intensity: only the acoustic branch, L-weighted identically."""
        w_ac, _ = self.bilayer_weights(L)
        return float(self.background + self.amp * w_ac * self.lorentzian(E, self.omega_mono(H, K)))


__all__ = ["SquareFMBilayer"]
