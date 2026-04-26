"""Smoke tests for the analytic paper-pilot spin-wave backends."""

import math

import numpy as np
import pytest

from tasai.physics import SquareFMBilayer, SquareLatticeAFM


class TestSquareLatticeAFM:
    def test_dispersion_is_zero_at_ordering_vector(self):
        afm = SquareLatticeAFM(J1=1.0, J2=0.0, D=0.0, S=0.5, ordering_vector=(0.5, 0.5))
        # At Q = Q_AF, γ1 = 1 and γ2 = 1 so A = B = 4SJ1 ⇒ ω = 0.
        assert afm.omega(0.5, 0.5) == pytest.approx(0.0, abs=1e-9)

    def test_dispersion_positive_away_from_ordering(self):
        afm = SquareLatticeAFM(J1=1.25, J2=0.20, D=0.02, S=2.5)
        w = afm.omega(0.75, 0.75)
        assert np.isfinite(w) and w > 0.0

    def test_anisotropy_gap_vanishes_for_spin_half(self):
        """D(2S-1) = 0 when S = 1/2, matching the manuscript's exact LSWT form."""
        a = SquareLatticeAFM(J1=1.0, J2=0.0, D=10.0, S=0.5)
        b = SquareLatticeAFM(J1=1.0, J2=0.0, D=0.0, S=0.5)
        # same dispersion at a non-trivial point
        assert a.omega(0.6, 0.6) == pytest.approx(b.omega(0.6, 0.6))

    def test_intensity_peaks_near_dispersion(self):
        afm = SquareLatticeAFM(J1=1.25, J2=0.20, D=0.02, S=2.5)
        w = afm.omega(0.65, 0.65)
        I_peak = afm.intensity(0.65, 0.65, w, sigma_E=0.3)
        I_off = afm.intensity(0.65, 0.65, w + 5.0, sigma_E=0.3)
        assert I_peak > 10.0 * I_off

    def test_neel_stability_window_produces_nan_outside(self):
        # J2 >= J1/2 leaves the Néel phase → class returns NaN.
        bad = SquareLatticeAFM(J1=1.0, J2=0.6, D=0.0, S=0.5)
        assert np.isnan(bad.omega(0.7, 0.7))


class TestSquareFMBilayer:
    def _make(self, J_perp=0.45):
        return SquareFMBilayer(
            S=1.0, J_par=2.0, J_perp=J_perp, D=0.10, z_bi=0.35,
            z_perp=1, gamma=0.60, amp=100.0, background=0.10,
        )

    def test_optic_shift_matches_analytic_form(self):
        bi = self._make()
        assert bi.optic_shift() == pytest.approx(2.0 * 1 * 1.0 * 0.45)

    def test_weights_sum_to_one(self):
        bi = self._make()
        for L in (0.0, 0.16, 0.3, 0.5):
            w_ac, w_op = bi.bilayer_weights(L)
            assert w_ac + w_op == pytest.approx(1.0, abs=1e-12)

    def test_monolayer_recovered_when_optic_weight_vanishes(self):
        """At integer L the optic weight vanishes and the two backends agree.

        The manuscript's comparator is specifically the no-optic model, so
        these only coincide when either w_op = 0 (integer L) or the acoustic
        and optic branches are otherwise made equivalent by construction."""
        bi = self._make()
        H, K, L = 0.24, 0.24, 0.0  # w_op(L=0) = sin(0)^2 = 0
        E = bi.omega_ac(H, K)
        I_bi = bi.intensity_bilayer(H, K, L, E)
        I_mono = bi.intensity_monolayer(H, K, L, E)
        assert I_bi == pytest.approx(I_mono, rel=1e-9)

    def test_optic_weight_small_at_paper_L(self):
        """At the paper's L_fixed = 0.16 the optic branch is suppressed."""
        bi = self._make()
        _, w_op = bi.bilayer_weights(0.16)
        assert 0.01 < w_op < 0.1
