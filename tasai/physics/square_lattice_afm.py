"""Square-lattice antiferromagnet J1-J2-D spin-wave model.

Néel-phase linear spin-wave theory for a 2D square-lattice antiferromagnet,

    H = J1 Σ_<ij>  S_i · S_j
      + J2 Σ_<<ik>> S_i · S_k
      - D  Σ_i    (S_i^z)^2      (easy-axis; sign follows the manuscript convention)

The dispersion is computed in the reduced momentum q = Q - Q_AF relative to the
AFM ordering vector. For the Néel ground state,

    ω(q) = √(A_q^2 - B_q^2),
    A_q  = 4 S J1 - 4 S J2 (1 - γ2) + D (2S - 1),
    B_q  = 4 S J1 γ1,
    γ1   = (cos(2π q_h) + cos(2π q_k)) / 2,
    γ2   =  cos(2π q_h) · cos(2π q_k).

The one-magnon transverse structure factor weight is (A_q + B_q) / ω.

Resolution is not hard-coded: callers pass an explicit sigma_E (or an optional
resolution callable) so this module can be used standalone without the rest of
the TAS-AI instrument stack.

This module implements the analytic backend used in the closed-loop audit
pilots of the TAS-AI manuscript (Section 3.6 / Figure 10).
"""

from __future__ import annotations

from typing import Callable, Optional, Tuple

import numpy as np


class SquareLatticeAFM:
    """Analytic Néel-phase spin-wave intensity for a square-lattice AFM.

    Parameters
    ----------
    J1 : float
        Nearest-neighbor exchange (meV). Must be positive for the Néel phase.
    J2 : float
        Next-nearest-neighbor exchange (meV). Must satisfy J2 < J1/2 for the
        Néel phase to be stable; out-of-range values produce NaN intensities.
    D : float
        Single-ion easy-axis anisotropy (meV). Enters ω as the LSWT shift
        D (2S-1), which vanishes for S = 1/2.
    S : float
        Spin quantum number.
    background : float
        Additive flat background in intensity units.
    ordering_vector : tuple of float
        The magnetic ordering vector (q_h, q_k) in r.l.u. that defines the
        reduced momentum origin. Default (0.5, 0.5) corresponds to the
        square-lattice AFM zone-corner order.
    lattice_a : float
        In-plane lattice constant in Ångstroms; only used for the Fe3+ magnetic
        form factor when it is not overridden.
    form_factor : callable, optional
        f(|q|) returning the magnetic form factor at momentum magnitude |q|
        (in inverse Ångstroms). Defaults to the Fe3+ j0 form factor, which is
        appropriate for the closed-loop pilot in the manuscript and harmless
        as a neutral default elsewhere.
    """

    def __init__(
        self,
        J1: float = 1.25,
        J2: float = 0.20,
        D: float = 0.02,
        S: float = 2.5,
        background: float = 0.5,
        ordering_vector: Tuple[float, float] = (0.5, 0.5),
        lattice_a: float = 4.0,
        form_factor: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    ) -> None:
        self.J1 = float(J1)
        self.J2 = float(J2)
        self.D = float(D)
        self.S = float(S)
        self.background = float(background)
        self.ordering_vector = tuple(float(x) for x in ordering_vector)
        self.lattice_a = float(lattice_a)
        self._form_factor = form_factor

    # ------------------------------------------------------------------
    # Core dispersion
    # ------------------------------------------------------------------
    def _kernel(
        self, h: np.ndarray, k: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Return broadcast (H, K, A_q, B_q, omega) arrays.

        Outside the Néel stability window (J1 <= 0 or J2 >= J1/2) the function
        returns NaN arrays rather than raising; this is deliberately quiet so
        the class can be used inside vectorized acquisition loops that may
        probe unphysical parameter combinations during fitting.
        """
        H, K = np.broadcast_arrays(
            np.asarray(h, dtype=float), np.asarray(k, dtype=float)
        )

        if self.J1 <= 0.0 or self.J2 >= 0.5 * self.J1:
            nan = np.full(H.shape, np.nan, dtype=float)
            return H, K, nan, nan, nan

        d_eff = 0.0 if self.S <= 0.5 else self.D * (1.0 - 1.0 / (2.0 * self.S))

        qh = H - self.ordering_vector[0]
        qk = K - self.ordering_vector[1]
        qx = 2.0 * np.pi * qh
        qy = 2.0 * np.pi * qk

        gamma1 = 0.5 * (np.cos(qx) + np.cos(qy))
        gamma2 = np.cos(qx) * np.cos(qy)

        A = (
            4.0 * self.S * self.J1
            - 4.0 * self.S * self.J2 * (1.0 - gamma2)
            + 2.0 * self.S * d_eff
        )
        B = 4.0 * self.S * self.J1 * gamma1

        arg = A * A - B * B
        tol = 1e-12 * np.maximum(1.0, np.maximum(np.abs(A * A), np.abs(B * B)))

        omega2 = np.full_like(arg, np.nan, dtype=float)
        valid = arg >= -tol
        omega2[valid] = np.maximum(arg[valid], 0.0)
        omega = np.sqrt(omega2)
        return H, K, A, B, omega

    def omega(self, h, k):
        """Spin-wave energy at (h, k) in r.l.u. (absolute Q, not reduced)."""
        _, _, _, _, w = self._kernel(h, k)
        return _as_scalar(w)

    def A_q_B_q(self, h, k):
        """Return (A_q, B_q) linear-spin-wave coefficients at (h, k)."""
        _, _, A, B, _ = self._kernel(h, k)
        return _as_scalar(A), _as_scalar(B)

    # ------------------------------------------------------------------
    # Intensity
    # ------------------------------------------------------------------
    def intensity(
        self,
        h,
        k,
        E,
        sigma_E,
        I0: float = 100.0,
        accessible: Optional[np.ndarray] = None,
    ):
        """Predicted neutron intensity at (h, k, E) with Gaussian energy line shape.

        Parameters
        ----------
        h, k, E : array-like
            Broadcastable arrays of momentum (r.l.u.) and energy transfer (meV).
        sigma_E : array-like
            Gaussian energy-resolution sigma (meV) at each (h, k, E). Required;
            the caller is responsible for applying any instrument-specific
            resolution model (e.g. Cooper-Nathans via tasai.instrument).
        I0 : float
            Peak intensity scale. The default matches the pilot closed-loop
            configuration in the manuscript.
        accessible : bool array, optional
            Kinematic accessibility mask. Unspecified ⇒ everywhere accessible.

        Returns
        -------
        float or ndarray
            Intensity = background + I0 · |f(q)|² · (A+B)/ω · Gaussian(E; ω, σ_E).
        """
        H, K, EE = np.broadcast_arrays(
            np.asarray(h, dtype=float),
            np.asarray(k, dtype=float),
            np.asarray(E, dtype=float),
        )
        sig = np.broadcast_to(np.asarray(sigma_E, dtype=float), EE.shape).copy()

        out = np.full(EE.shape, float(self.background), dtype=float)
        _, _, A_q, B_q, omega = self._kernel(H, K)

        if accessible is None:
            acc_mask = np.ones(EE.shape, dtype=bool)
        else:
            acc_mask = np.broadcast_to(np.asarray(accessible, dtype=bool), EE.shape)

        if np.any(acc_mask):
            with np.errstate(divide="ignore", invalid="ignore"):
                sw_weight = (A_q + B_q) / omega
                q_mag = (2.0 * np.pi / self.lattice_a) * np.sqrt(H * H + K * K)
                ff = self._form_factor_fn()(q_mag)
                x = (EE - omega) / sig
                profile = np.exp(-0.5 * x * x) / (np.sqrt(2.0 * np.pi) * sig)

            bad = acc_mask & (
                ~np.isfinite(omega)
                | ~np.isfinite(sw_weight)
                | ~np.isfinite(profile)
                | (sig <= 0.0)
            )
            out[bad] = np.nan
            good = acc_mask & ~bad
            out[good] += I0 * (ff[good] ** 2) * sw_weight[good] * profile[good]

        return _as_scalar(out)

    # ------------------------------------------------------------------
    # Form factor
    # ------------------------------------------------------------------
    def _form_factor_fn(self) -> Callable[[np.ndarray], np.ndarray]:
        if self._form_factor is not None:
            return self._form_factor
        return _fe3_form_factor


def _fe3_form_factor(q_mag: np.ndarray) -> np.ndarray:
    """Fe3+ j0 magnetic form factor from International Tables coefficients."""
    s = np.asarray(q_mag, dtype=float) / (4.0 * np.pi)
    s2 = s * s
    return (
        0.396 * np.exp(-13.244 * s2)
        + 0.629 * np.exp(-4.903 * s2)
        + -0.0314 * np.exp(-0.35 * s2)
        + 0.0044
    )


def _as_scalar(arr: np.ndarray):
    arr = np.asarray(arr)
    return arr.item() if arr.ndim == 0 or arr.size == 1 else arr


__all__ = ["SquareLatticeAFM"]
