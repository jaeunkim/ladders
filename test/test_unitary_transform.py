"""
Tests for Expression.unitary_transform() (unified Bogoliubov / squeezing transformation).

Validates against closed-form results and reproduces the central quantum result
of Crimin, Garraway, Verdu 2021, "Quantisation of the elliptical Penning trap"
(J. Phys. B 54, 115501): eqs (22) -> (27), (28).

Run directly (python test_transform.py) or with pytest.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import warnings

import numpy as np
import sympy as sp

from ladders import Expression, scalar_multiply, squeeze_generator


def test_single_mode_squeeze_closed_form():
    """S a S+ = cosh(r) a + e^{i phi} sinh(r) a+  (Crimin 2021, eq 24)."""
    r, phi = 0.35, 0.7
    zeta = r * np.exp(1j * phi)
    out = Expression("a").unitary_transform(squeeze_generator("a", zeta), order=25)
    assert np.isclose(out.expr_dict["a"], np.cosh(r))
    assert np.isclose(out.expr_dict["a+"], np.sinh(r) * np.exp(1j * phi))


def test_dagger_flag():
    """S+ a S = cosh(r) a - sinh(r) a+ for real zeta."""
    r = 0.4
    out = Expression("a").unitary_transform(squeeze_generator("a", r), order=25, dagger=True)
    assert np.isclose(out.expr_dict["a"], np.cosh(r))
    assert np.isclose(out.expr_dict["a+"], -np.sinh(r))


def test_squeezed_vacuum_photon_number():
    """Constant term of S+ (a+ a) S is sinh^2(r): vacuum photon number."""
    r = 0.4
    n_out = Expression("a+_a").unitary_transform(squeeze_generator("a", r), order=25, dagger=True)
    assert np.isclose(n_out.expr_dict[""], np.sinh(r) ** 2)


def test_two_mode_rotation():
    """Beamsplitter-type Bogoliubov: e^S a e^-S = cos(t) a - sin(t) b for S = t(a+ b - a b+)."""
    theta = 0.3
    S2 = scalar_multiply(Expression("a+_b(+)-1a_b+"), theta)
    out = Expression("a").unitary_transform(S2, order=20)
    assert np.isclose(out.expr_dict["a"], np.cos(theta))
    assert np.isclose(out.expr_dict["b"], -np.sin(theta))


def test_commuting_generator_is_identity():
    """[S, A] = 0 => transform returns A unchanged (early termination path)."""
    out = Expression("a+_a").unitary_transform(Expression("a+_a"), order=50)
    assert out.expr_dict["a+_a"] == 1
    assert all(abs(v) < 1e-12 for k, v in out.expr_dict.items() if k != "a+_a")


def test_unitarity_preserved():
    """[S a S+, S a+ S+] = 1: the transformed operators are still bosonic."""
    G = squeeze_generator("a", 0.4)
    b = Expression("a").unitary_transform(G, order=25)
    b_dag = Expression("a+").unitary_transform(G, order=25)
    comm = b * b_dag - b_dag * b
    assert np.isclose(comm.expr_dict[""], 1)
    assert all(abs(v) < 1e-9 for k, v in comm.expr_dict.items() if k != "")


def test_convergence_warning_fires_when_undertruncated():
    """Large squeezing at low order is inaccurate, and must not fail silently."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        Expression("a").unitary_transform(squeeze_generator("a", 5.0), order=6)
    assert len(caught) == 1, f"expected one warning, got {len(caught)}"
    assert "may not have converged" in str(caught[0].message)


def test_no_warning_when_converged():
    """A well-converged series must stay quiet."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        Expression("a").unitary_transform(squeeze_generator("a", 0.4))  # default order
    assert caught == [], f"unexpected warning: {[str(c.message) for c in caught]}"


def test_no_warning_when_series_terminates_exactly():
    """[S, A] = 0 gives an exact answer, not a truncated one: no warning."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        Expression("a+_a").unitary_transform(Expression("a+_a"))
    assert caught == [], f"unexpected warning: {[str(c.message) for c in caught]}"


def test_convergence_check_skipped_for_symbolic_coefficients():
    """Symbolic coefficients cannot be compared numerically; must not raise."""
    r = sp.Symbol("r", real=True)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        out = Expression("a").unitary_transform(squeeze_generator("a", r), order=4)
    assert caught == [], f"unexpected warning: {[str(c.message) for c in caught]}"
    assert sp.simplify(out.expr_dict["a"] - (1 + r**2 / 2 + r**4 / 24)) == 0


def test_convergence_tol_none_silences_check():
    """convergence_tol=None is the documented escape hatch."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        Expression("a").unitary_transform(
            squeeze_generator("a", 5.0), order=6, convergence_tol=None
        )
    assert caught == [], f"unexpected warning: {[str(c.message) for c in caught]}"


def test_default_order_is_accurate_at_moderate_squeezing():
    """The default order must give machine precision across the useful r range."""
    for r in [0.1, 0.5, 1.0, 2.0, 3.0]:
        out = Expression("a").unitary_transform(squeeze_generator("a", r))
        assert np.isclose(out.expr_dict["a"], np.cosh(r), rtol=1e-12), r
        assert np.isclose(out.expr_dict["a+"], np.sinh(r), rtol=1e-12), r


def test_crimin_2021_elliptical_trap_diagonalization():
    """
    Reproduce Crimin et al. 2021, section 3.2: squeezing the elliptical-trap
    Hamiltonian (eq 22) with the parameters of eqs (25)-(26) must yield the
    diagonal Hamiltonian of eq (27) with the mode frequencies of eq (28).
    Modes: p = cyclotron (+), m = magnetron (-), z = axial. hbar = 1.
    """
    wc, wz, eps = 5.0, 2.0, 0.3
    w1 = np.sqrt(wc**2 - 2 * wz**2)
    wp, wm = (wc + w1) / 2, (wc - w1) / 2                    # eq (5)
    gam = eps * wz**2 / (w1 * wc)                            # eq (21)

    kappa = eps * wz**2 / (2 * w1)                           # eq (25)
    K = (wc / (2 * kappa)) * (np.sqrt(1 + 4 * kappa**2 / wc**2) - 1)
    zeta_p = 0.5 * np.arctanh(-kappa / (wp + kappa * K))     # eq (26)
    zeta_m = 0.5 * np.arctanh(-kappa / (wm + kappa * K))

    # eq (22): H' after the two-mode rotation U of eq (20)
    H = scalar_multiply(Expression("p+_p(+)m+_m(+)1"), w1 / 2) + \
        scalar_multiply(Expression("p+_p(+)-1m+_m"), wc * np.sqrt(1 + gam**2) / 2) + \
        scalar_multiply(Expression("z+_z(+)0.5"), wz) + \
        scalar_multiply(Expression("p+_p+(+)p_p(+)-1m+_m+(+)-1m_m"),
                        eps * wz**2 / (4 * w1))

    # eq (27): S(zeta_p) S(zeta_m) H S(zeta_p)+ S(zeta_m)+
    Hz = H.unitary_transform(squeeze_generator("p", zeta_p), order=30) \
          .unitary_transform(squeeze_generator("m", zeta_m), order=30)

    wt_p = np.sqrt(wp**2 + w1 * kappa * K)                   # eq (28)
    wt_m = np.sqrt(wm**2 - w1 * kappa * K)

    # off-diagonal terms vanish
    for key in ["p+_p+", "p_p", "m+_m+", "m_m"]:
        assert abs(Hz.expr_dict.get(key, 0)) < 1e-10, (key, Hz.expr_dict[key])
    # diagonal matches eq (27) with the frequencies of eq (28)
    assert np.isclose(Hz.expr_dict["p+_p"], wt_p)
    assert np.isclose(Hz.expr_dict["m+_m"], -wt_m)
    assert np.isclose(Hz.expr_dict["z+_z"], wz)
    assert np.isclose(Hz.expr_dict[""], (wt_p - wt_m + wz) / 2)


if __name__ == "__main__":
    test_single_mode_squeeze_closed_form()
    test_dagger_flag()
    test_squeezed_vacuum_photon_number()
    test_two_mode_rotation()
    test_commuting_generator_is_identity()
    test_unitarity_preserved()
    test_convergence_warning_fires_when_undertruncated()
    test_no_warning_when_converged()
    test_no_warning_when_series_terminates_exactly()
    test_convergence_check_skipped_for_symbolic_coefficients()
    test_convergence_tol_none_silences_check()
    test_default_order_is_accurate_at_moderate_squeezing()
    test_crimin_2021_elliptical_trap_diagonalization()
    print("ALL TESTS PASSED")
