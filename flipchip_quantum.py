"""
flipchip_quantum.py
Energy spectrum of the anharmonic (elliptical + quartic + sextic) Penning trap.

Pipeline (Crimin, Garraway, Verdu 2021, extended by the anharmonic terms):
  1. quadratic part  -> omega_z, ellipticity eps, omega_1, omega_+/-  (their eqs 4-10)
  2. quantise with a_x, a_y, a_z (eqs 14-16) and a_+/- (eq 17)
  3. U (eq 20-21) then S(zeta_+) S(zeta_-) (eqs 23-26): H_quad -> diagonal, eq 27-28
     -- all done symbolically on operator strings with ladders.py --
  4. apply the same unitaries to x, y, z  ->  x~, y~, z~ linear in the new ladders
  5. V_anh = sum C_ijk x~^i y~^j z~^k, normal ordered by ladders.py
       * number-conserving part -> Kerr coefficients (1st-order perturbation theory)
       * full operator in a truncated Fock basis -> non-perturbative spectrum
Mode letters: p = cyclotron (+), m = magnetron (-), z = axial.
Energies are handled as H/hbar in rad/s.
"""
import numpy as np
from ladders import Expression, scalar_multiply, squeeze_generator

e_ch = 1.602176634e-19
m_e = 9.1093837015e-31
hbar = 1.054571817e-34
q_e = -e_ch


# ----------------------------------------------------------------------------
# 1-2. quadratic parameters
# ----------------------------------------------------------------------------
def quad_params(C, B, q=q_e, m=m_e):
    """From Taylor coefficients (V/m^2) and B (T) -> dict of trap frequencies.
    Raises ValueError if the trap is unstable."""
    C002, C200, C020 = C[(0, 0, 2)], C[(2, 0, 0)], C[(0, 2, 0)]
    wz2 = 2 * q * C002 / m
    if wz2 <= 0:
        raise ValueError("axial anti-confinement (q*C002 <= 0)")
    wz = np.sqrt(wz2)
    eps = (C200 - C020) / C002                                   # eq 8
    wc = abs(q) * B / m
    w1_2 = wc ** 2 - 2 * wz2
    if w1_2 <= 0:
        raise ValueError("omega_1 imaginary: B too weak for this axial curvature")
    w1 = np.sqrt(w1_2)
    wp, wm = (wc + w1) / 2, (wc - w1) / 2                        # eq 5
    # elliptical frequencies, eq 28
    root = np.sqrt(wc ** 2 * w1 ** 2 + eps ** 2 * wz ** 4)
    wt_p2 = 0.5 * (wc ** 2 - wz2) + 0.5 * root
    wt_m2 = 0.5 * (wc ** 2 - wz2) - 0.5 * root
    if wt_m2 <= 0:
        raise ValueError("magnetron unstable (elliptical): |eps| too large for this B")
    return dict(wz=wz, eps=eps, wc=wc, w1=w1, wp=wp, wm=wm,
                wt_p=np.sqrt(wt_p2), wt_m=np.sqrt(wt_m2),
                x_zpf=np.sqrt(hbar / (m * w1)),                  # from eq 14: x = x_zpf (a_x + a_x+)
                z_zpf=np.sqrt(hbar / (2 * m * wz)))              # from eq 16


# ----------------------------------------------------------------------------
# 3. the two-step Crimin transformation as ladders.py generators
# ----------------------------------------------------------------------------
def crimin_generators(P):
    """Generators (exponents) of U (eq 20-21) and S(zeta_+), S(zeta_-) (eqs 23-26)."""
    wz, eps, wc, w1, wp, wm = (P[k] for k in ("wz", "eps", "wc", "w1", "wp", "wm"))
    gam = eps * wz ** 2 / (w1 * wc)                               # eq 21
    theta = np.arctan(gam)
    G_U = scalar_multiply(Expression("p+_m(+)m+_p"), 0.5j * theta)
    kappa = eps * wz ** 2 / (2 * w1)                              # eq 25
    if kappa == 0:
        K = 1.0                                                   # limit of eq 25, kappa -> 0
    else:
        K = (wc / (2 * kappa)) * (np.sqrt(1 + 4 * kappa ** 2 / wc ** 2) - 1)
    zeta_p = 0.5 * np.arctanh(-kappa / (wp + kappa * K))          # eq 26
    zeta_m = 0.5 * np.arctanh(-kappa / (wm + kappa * K))
    return dict(G_U=G_U, G_Sp=squeeze_generator("p", zeta_p), G_Sm=squeeze_generator("m", zeta_m),
                theta=theta, zeta_p=zeta_p, zeta_m=zeta_m)


def to_diag_frame(A, G, order=30, **kw):
    """S(zeta_+) S(zeta_-) U  A  U+ S+(zeta_-) S+(zeta_+)  for any Expression A."""
    return (A.unitary_transform(G["G_U"], order=order, **kw)
             .unitary_transform(G["G_Sp"], order=order, **kw)
             .unitary_transform(G["G_Sm"], order=order, **kw))


def lab_operators(P):
    """x, y, z, px, py in the lab (a_+, a_-, a_z) basis, eqs 14-17.  Units: m and kg m/s (hbar kept)."""
    X0, Z0 = P["x_zpf"], P["z_zpf"]
    P0 = hbar / (2 * X0)                                          # px = -i P0 (a_x - a_x+)
    s2 = 1 / np.sqrt(2)
    # Inverse of eq 17 *in the phase convention that reproduces eq 19*:
    # eq 17 taken literally gives a_x=(a_+ + a_-)/sqrt2, a_y=i(a_+ - a_-)/sqrt2, whose
    # (x^2-y^2) has +a_-a_- and real cross terms, i.e. a_- differs from eq 19 by a
    # factor -i.  We use a_- (eq 19) = -i * a_- (eq 17), which is a harmless relabelling:
    #   a_x = (a_+ + i a_-)/sqrt2 ,  a_y = (i a_+ + a_-)/sqrt2
    ax = scalar_multiply(Expression("p(+)1jm"), s2)
    ax_d = scalar_multiply(Expression("p+(+)-1jm+"), s2)
    ay = scalar_multiply(Expression("1jp(+)m"), s2)
    ay_d = scalar_multiply(Expression("-1jp+(+)m+"), s2)
    x = scalar_multiply(ax + ax_d, X0)
    y = scalar_multiply(ay + ay_d, X0)
    px = scalar_multiply(ax - ax_d, -1j * P0)
    py = scalar_multiply(ay - ay_d, -1j * P0)
    z = scalar_multiply(Expression("z(+)z+"), Z0)
    return x, y, z, px, py


def quadratic_hamiltonian_lab(P, m=m_e):
    """H_eps / hbar built directly from eq 10 with the lab operators (validation of eq 19)."""
    x, y, z, px, py = lab_operators(P)
    wc, w1, wz, eps = P["wc"], P["w1"], P["wz"], P["eps"]
    H = scalar_multiply(px * px + py * py, 1 / (2 * m))
    H = H + scalar_multiply(x * py - y * px, wc / 2)
    H = H + scalar_multiply(x * x + y * y, 0.5 * m * (w1 / 2) ** 2)
    H = H + scalar_multiply(x * x - y * y, 0.25 * m * eps * wz ** 2)
    # axial part: + hbar wz (n_z + 1/2), added directly (it is already diagonal)
    H = scalar_multiply(H, 1 / hbar) + scalar_multiply(Expression("z+_z(+)0.5"), wz)
    return H


def diag_frame_operators(P, order=30):
    """x~, y~, z~ (lab position operators expressed in the ladders of the diagonal frame)."""
    G = crimin_generators(P)
    x, y, z, _, _ = lab_operators(P)
    xt = to_diag_frame(x, G, order)
    yt = to_diag_frame(y, G, order)
    return xt, yt, z, G                                           # z untouched by U, S


def ladder_amplitudes(expr):
    """{('p'|'m'|'z'): (c_a, c_adag)} of a linear Expression  sum c_a a + c_adag a+."""
    out = {}
    for mode in ("p", "m", "z"):
        out[mode] = (complex(expr.expr_dict.get(mode, 0)), complex(expr.expr_dict.get(mode + "+", 0)))
    return out


# ----------------------------------------------------------------------------
# 5a. anharmonic potential as a normal-ordered ladders.py Expression
# ----------------------------------------------------------------------------
def anharmonic_operator(C, xt, yt, z, degrees=(4,), q=q_e):
    """q * sum_{deg in degrees} sum_{i+j+k=deg} C_ijk x~^i y~^j z~^k  / hbar   (rad/s)."""
    powers = {}

    def pw(op, n):
        key = (id(op), n)
        if key not in powers:
            if n == 0:
                powers[key] = Expression("1")
            else:
                r = op
                for _ in range(n - 1):
                    r = r * op
                powers[key] = r
        return powers[key]

    V = Expression("")
    for (i, j, k), c in C.items():
        if (i + j + k) not in degrees or c == 0:
            continue
        term = pw(xt, i) * pw(yt, j) * pw(z, k)
        V = V + scalar_multiply(term, q * c / hbar)
    return V


def kerr_table(V):
    """Number-conserving coefficients of a normal-ordered Expression (rad/s).
    Self-Kerr K_ii multiplies a_i+ a_i+ a_i a_i, cross-Kerr multiplies a_i+ a_i a_j+ a_j."""
    d = V.expr_dict
    g = lambda k: complex(d.get(k, 0)).real
    return dict(
        shift_p=g("p+_p"), shift_m=g("m+_m"), shift_z=g("z+_z"),
        K_pp=g("p+_p+_p_p"), K_mm=g("m+_m+_m_m"), K_zz=g("z+_z+_z_z"),
        K_pm=g("p+_p_m+_m"), K_mz=g("m+_m_z+_z"), K_pz=g("p+_p_z+_z"),
        K_mmm=g("m+_m+_m+_m_m_m"),
    )


def alpha_from_kerr(kt):
    """First-order magnetron anharmonicity  alpha = E(0,2,0) - 2E(0,1,0) + E(0,0,0)  (rad/s)."""
    return 2 * kt["K_mm"]          # <n|a+a+aa|n> = n(n-1): 0,0,2


# ----------------------------------------------------------------------------
# 5b. Fock-space numerics (fast path for optimisation, and non-perturbative check)
# ----------------------------------------------------------------------------
def _ladder(dim):
    return np.diag(np.sqrt(np.arange(1, dim)), 1)


class FockEngine:
    """Truncated Fock representation of the diagonal-frame Hamiltonian
    H/hbar = wt_p n_p - wt_m n_m + wz n_z + V(x~, y~, z~)."""

    def __init__(self, dims=(4, 8, 4), pad=(4, 4, 4)):
        if np.isscalar(pad):
            pad = (pad,) * 3
        self.dims = dims
        self.pad = pad
        D = [d + p for d, p in zip(dims, pad)]
        I = [np.eye(d) for d in D]
        a = [_ladder(d) for d in D]
        self.a = [np.kron(np.kron(a[0], I[1]), I[2]),
                  np.kron(np.kron(I[0], a[1]), I[2]),
                  np.kron(np.kron(I[0], I[1]), a[2])]
        self.n = [A.conj().T @ A for A in self.a]
        self.D = D
        # indices that survive truncation to dims
        keep = np.zeros(D, bool)
        keep[:dims[0], :dims[1], :dims[2]] = True
        self.keep = np.where(keep.ravel())[0]
        self.labels = np.array(np.unravel_index(self.keep, D)).T

    def position(self, amps):
        """amps: {mode: (c_a, c_adag)} -> matrix of  sum c_a a + c_adag a+."""
        M = 0
        for i, mode in enumerate("pmz"):
            ca, cd = amps[mode]
            M = M + ca * self.a[i] + cd * self.a[i].conj().T
        return M

    def hamiltonian(self, P, C, amps_x, amps_y, amps_z, degrees=(4, 6), q=q_e):
        X, Y, Z = (self.position(a) for a in (amps_x, amps_y, amps_z))
        H = P["wt_p"] * self.n[0] - P["wt_m"] * self.n[1] + P["wz"] * self.n[2]
        H = H.astype(complex)
        pw = {}

        def power(M, k, tag):
            if (tag, k) not in pw:
                pw[(tag, k)] = np.linalg.matrix_power(M, k) if k else np.eye(len(M))
            return pw[(tag, k)]

        for (i, j, k), c in C.items():
            if (i + j + k) in degrees and c != 0:
                H += (q * c / hbar) * power(X, i, "x") @ power(Y, j, "y") @ power(Z, k, "z")
        # truncate to the un-padded space (powers were computed in the padded one)
        return H[np.ix_(self.keep, self.keep)]

    def first_order_alpha(self, H):
        """diag elements -> alpha_- = E(0,2,0)-2E(0,1,0)+E(0,0,0) (first-order PT)."""
        d = np.real(np.diag(H))
        idx = {tuple(l): i for i, l in enumerate(self.labels)}
        E = lambda n: d[idx[(0, n, 0)]]
        return E(2) - 2 * E(1) + E(0)

    def magnetron_ladder(self, H, nmax=6):
        """Non-perturbative: diagonalise, label eigenstates by the dominant Fock
        component, return E_n for |0,n,0>, n=0..nmax (rad/s), and their purity."""
        w, v = np.linalg.eigh((H + H.conj().T) / 2)
        idx = {tuple(l): i for i, l in enumerate(self.labels)}
        E, purity = [], []
        for n in range(nmax + 1):
            row = idx[(0, n, 0)]
            k = np.argmax(np.abs(v[row, :]) ** 2)
            E.append(w[k]); purity.append(np.abs(v[row, k]) ** 2)
        return np.array(E), np.array(purity)


def magnetron_projected(H, engine, nmax):
    """Project the truncated-Fock Hamiltonian on |n_+=0, n_z=0>: single-mode magnetron
    Hamiltonian (dim nmax+1) including the zero-point cross-Kerr shifts of the other modes."""
    idx = {tuple(l): i for i, l in enumerate(engine.labels)}
    rows = [idx[(0, n, 0)] for n in range(nmax + 1)]
    return H[np.ix_(rows, rows)]


def ladder_by_overlap(H, nmax):
    """Eigen-decompose a single-mode Hamiltonian and label states by dominant Fock component."""
    w, v = np.linalg.eigh((H + H.conj().T) / 2)
    E, pur = [], []
    for n in range(nmax + 1):
        k = np.argmax(np.abs(v[n, :]) ** 2)
        E.append(w[k]); pur.append(np.abs(v[n, k]) ** 2)
    return np.array(E), np.array(pur)


def second_order_energies(V, P, nlist=(0, 1, 2)):
    """
    Second-order perturbative energy shift of |0, n, 0> (cyclotron & axial in vacuum)
    from the normal-ordered anharmonic operator V, summing over ALL intermediate
    3-mode Fock states:  E2(n) = sum_{f != (0,n,0)} |<f|V|0,n,0>|^2 / (E_n - E_f).
    The amplitudes follow analytically from the term strings, so this costs ~ms.
    """
    from math import factorial, sqrt
    wp, wm, wz = P["wt_p"], P["wt_m"], P["wz"]
    E0 = lambda a, n, e: wp * a - wm * n + wz * e
    terms = []
    for key, c in V.expr_dict.items():
        ops = key.split("_") if key else []
        a = sum(o == "p+" for o in ops); b = sum(o == "p" for o in ops)
        cc = sum(o == "m+" for o in ops); d = sum(o == "m" for o in ops)
        e = sum(o == "z+" for o in ops); f = sum(o == "z" for o in ops)
        if b == 0 and f == 0 and c != 0:
            terms.append((a, cc, d, e, complex(c)))
    out = {}
    for n in nlist:
        amp = {}
        for a, cc, d, e, c in terms:
            if d > n:
                continue
            A = c * sqrt(factorial(a)) * sqrt(factorial(e)) \
                * sqrt(factorial(n) / factorial(n - d)) * sqrt(factorial(n - d + cc) / factorial(n - d))
            fin = (a, n - d + cc, e)
            amp[fin] = amp.get(fin, 0) + A
        s = 0.0
        for fin, A in amp.items():
            if fin == (0, n, 0):
                continue
            s += abs(A) ** 2 / (E0(0, n, 0) - E0(*fin))
        out[n] = s
    return out


def alpha_second_order(V, P):
    """Second-order contribution to alpha_- = E(2) - 2E(1) + E(0)."""
    E2 = second_order_energies(V, P)
    return E2[2] - 2 * E2[1] + E2[0]


def magnetron_1mode_matrix(V, P, dim=40):
    """
    Project H = -w~_- n_- + V onto the cyclotron and axial vacuum.  Because V is normal
    ordered, <0_+ 0_z| V |0_+ 0_z> keeps exactly the terms that contain magnetron
    operators only, so the projection is a filter on the term strings of the
    ladders.py Expression.  Matrix elements of normal-ordered a+^k a^l are exact in a
    truncated basis (no padding needed).  Returns the (dim x dim) matrix in rad/s.
    """
    a = _ladder(dim)
    ad = a.T
    H = -P["wt_m"] * (ad @ a)
    H = H.astype(complex)
    pw = {}
    for key, c in V.expr_dict.items():
        ops = key.split("_") if key else []
        if any(not o.startswith("m") for o in ops):
            continue
        k = sum(o == "m+" for o in ops)
        l = sum(o == "m" for o in ops)
        if (k, l) not in pw:
            pw[(k, l)] = np.linalg.matrix_power(ad, k) @ np.linalg.matrix_power(a, l)
        H += c * pw[(k, l)]
    return H


def magnetron_spectrum(V, P, dim=40, nmax=6):
    """Non-perturbative quasi-bound magnetron ladder: E_n (rad/s) and purities."""
    return ladder_by_overlap(magnetron_1mode_matrix(V, P, dim), nmax)


def fast_alpha(C, B, engine, order=30):
    """One-shot: Taylor dict + B -> (alpha_first_order [rad/s], params, amplitudes).
    Uses ladders.py for the Bogoliubov transform of x, y and the Fock engine for
    the expectation values (identical to the symbolic route, but ~50x faster)."""
    P = quad_params(C, B)
    xt, yt, z, G = diag_frame_operators(P, order)
    ax, ay, az = ladder_amplitudes(xt), ladder_amplitudes(yt), ladder_amplitudes(z)
    H = engine.hamiltonian(P, C, ax, ay, az)
    return engine.first_order_alpha(H), P, (ax, ay, az), G
