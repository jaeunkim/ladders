"""
flipchip_optimize.py
Maximise the magnetron self-anharmonicity |alpha_-| of the flip-chip Penning trap
over electrode geometry, electrode voltages and magnetic field.

Design vector (10 numbers) for a fixed chip separation d = 2h:
    cx/h, cz/h, w/h, hh/h        compensation electrode centre and size, in units of h
    rw/h, ah/h                   radial-electrode width, axial-electrode height, in units of h
    Vc, Vr, Va                   voltages [V]  (same pattern on both chips)
    log10(mu)                    B = (1+mu) * B_crit,  B_crit = sqrt(2) omega_z m/|q|
                                 (omega_1 = omega_c sqrt(1 - 1/(1+mu)^2): always stable)
Constraints (all adjustable through CONSTRAINTS):
    trap stability: q C002 > 0, omega_c^2 > 2 omega_z^2 (built in), |eps| < 1
    |alpha_-| <= R_MAX * omega~_-     (Kerr ratio: ladder folds at n ~ 1/R_MAX; a transmon has ~0.04),
                                      alpha_- = first + second order perturbation theory
    at least N_LEVELS consecutive magnetron levels |0>,|1>,... with Fock purity >= PURITY_MIN
                                      (the anharmonic magnetron potential is unbounded, so the
                                      ladder is quasi-bound and terminates)
    omega~_- <= S_MAX * omega~_+      (magnetron and cyclotron spectrally distinct)
    omega~_- >= WM_MIN
    d(omega~_-) <= |alpha_-| / NOISE_RATIO   where d(omega~_-) is the rms jitter of the
                                      magnetron frequency from voltage noise SIGMA_V (relative)
                                      + V_FLOOR (absolute) on each channel and SIGMA_B on B.
                                      This is what really forbids eps -> 1 and omega_1 -> 0.
    |V| <= V_MAX,  B_MIN <= B <= B_MAX,  fabrication bounds from geometry_ok()
"""
import warnings
import numpy as np
from scipy.optimize import minimize

from flipchip_potential import (DEFAULT_GEOM, GAP, um, taylor_all_groups, combine,
                                geometry_ok, laplace_residual, GROUPS)
from flipchip_quantum import (quad_params, diag_frame_operators, anharmonic_operator,
                              kerr_table, alpha_from_kerr, alpha_second_order, magnetron_spectrum, m_e, e_ch)

CONSTRAINTS = dict(R_MAX=0.10, S_MAX=0.5, WM_MIN=2 * np.pi * 1e6,
                   N_LEVELS=4, PURITY_MIN=0.9, FOCK_DIM=16,
                   NOISE_RATIO=10.0, SIGMA_V=1e-6, V_FLOOR=1e-6, SIGMA_B=1e-8,
                   V_MAX=14.0, B_MIN=0.05, B_MAX=0.3)

SHAPE_NAMES = ["cx/h", "cz/h", "w/h", "hh/h", "rw/h", "ah/h"]
SHAPE_LO = np.array([0.15, 0.15, 0.05, 0.05, 0.3, 0.3])
SHAPE_HI = np.array([3.0, 3.0, 3.0, 3.0, 12.0, 12.0])
LOGMU_LO, LOGMU_HI = -3.0, 1.5


def b_crit(C):
    """B at which omega_1 = 0 for the axial curvature in C (needs q C002 > 0)."""
    wz2 = 2 * (-e_ch) * C[(0, 0, 2)] / m_e
    return np.sqrt(2 * wz2) * m_e / e_ch if wz2 > 0 else np.nan


def unpack(p, sep, Cg=None):
    h = sep / 2
    g = dict(comp_cx=p[0] * h, comp_cz=p[1] * h, comp_w=p[2] * h, comp_h=p[3] * h,
             radial_w=p[4] * h, axial_h=p[5] * h, sep=sep)
    volts = dict(comp=p[6], radial=p[7], axial=p[8])
    B = None
    if Cg is not None:
        B = (1 + 10.0 ** p[9]) * b_crit(combine(Cg, volts))
    return g, volts, B


def pack(g, volts, B, Cg):
    h = g["sep"] / 2
    mu = B / b_crit(combine(Cg, volts)) - 1
    return np.array([g["comp_cx"] / h, g["comp_cz"] / h, g["comp_w"] / h, g["comp_h"] / h,
                     g["radial_w"] / h, g["axial_h"] / h, volts["comp"], volts["radial"],
                     volts["axial"], np.log10(mu)])


class TaylorCache:
    """Geometry -> per-group Taylor coefficients (the expensive step), memoised."""
    def __init__(self, **fit_kw):
        self.store = {}
        self.fit_kw = dict(npts=800, nmax=30, max_deg=10)
        self.fit_kw.update(fit_kw)

    def __call__(self, g):
        key = tuple(round(g[k] / um, 3) for k in ("comp_cx", "comp_cz", "comp_w", "comp_h", "radial_w", "axial_h", "sep"))
        if key not in self.store:
            self.store[key] = taylor_all_groups(g, **self.fit_kw)[0]
        return self.store[key]


def magnetron_freq_jitter(Cg, volts, B, cons):
    """rms jitter of omega~_- from independent noise on each voltage channel and on B."""
    P0 = quad_params(combine(Cg, volts), B)
    tot = 0.0
    contrib = {}
    for k in GROUPS:
        dV = cons["SIGMA_V"] * abs(volts[k]) + cons["V_FLOOR"]
        v2 = dict(volts); v2[k] += dV
        try:
            d = quad_params(combine(Cg, v2), B)["wt_m"] - P0["wt_m"]
        except ValueError:
            d = np.inf
        contrib[k] = abs(d); tot += d ** 2
    try:
        dB = quad_params(combine(Cg, volts), B * (1 + cons["SIGMA_B"]))["wt_m"] - P0["wt_m"]
    except ValueError:
        dB = np.inf
    contrib["B"] = abs(dB); tot += dB ** 2
    return np.sqrt(tot), contrib


def quad_check(p, sep, Cg, cons):
    """Cheap feasibility (no ladders): returns (C, B, P) or None."""
    g, volts, B = unpack(p, sep, Cg)
    if not geometry_ok(g) or max(abs(v) for v in volts.values()) > cons["V_MAX"]:
        return None
    if not (np.isfinite(B) and cons["B_MIN"] <= B <= cons["B_MAX"]):
        return None
    C = combine(Cg, volts)
    try:
        P = quad_params(C, B)
    except ValueError:
        return None
    if P["wt_m"] < cons["WM_MIN"] or P["wt_m"] / P["wt_p"] > cons["S_MAX"]:
        return None
    return C, B, P


def evaluate(p, sep, cache, cons=CONSTRAINTS, degrees=(4, 6), order=30):
    """Full physics evaluation of one design.  Returns a dict (never raises)."""
    g, volts, _ = unpack(p, sep)
    out = dict(g=g, volts=volts, ok=False, reason="")
    if not geometry_ok(g):
        out["reason"] = "geometry"; return out
    Cg = cache(g)
    chk = quad_check(p, sep, Cg, cons)
    if chk is None:
        out["reason"] = "quadratic-level constraint"; return out
    C, B, P = chk
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        try:
            xt, yt, z, G = diag_frame_operators(P, order)
        except Warning:
            try:
                xt, yt, z, G = diag_frame_operators(P, 3 * order)
            except Warning:
                out["reason"] = "BCH not converged (|zeta| too large)"; return out
    V = anharmonic_operator(C, xt, yt, z, degrees=degrees)
    kt = kerr_table(V)
    alpha1 = alpha_from_kerr(kt)                                   # first-order PT
    alpha2 = alpha_second_order(V, P)                              # second-order PT
    alpha = alpha1 + alpha2                                        # smooth objective
    E, pur = magnetron_spectrum(V, P, dim=cons["FOCK_DIM"], nmax=6)  # non-perturbative check
    alpha_np = E[2] - 2 * E[1] + E[0]
    n_bound = 0
    for q in pur:
        if q < cons["PURITY_MIN"]:
            break
        n_bound += 1
    jitter, contrib = magnetron_freq_jitter(Cg, volts, B, cons)
    out.update(ok=True, B=B, C=C, Cg=Cg, P=P, G=G, V=V, kerr=kt, alpha1=alpha1, alpha2=alpha2, alpha=alpha,
               alpha_np=alpha_np, E=E, purity=pur, n_bound=n_bound,
               ratio=abs(alpha) / P["wt_m"], mode_sep=P["wt_m"] / P["wt_p"],
               jitter=jitter, jitter_contrib=contrib,
               noise_ratio=abs(alpha) / jitter if jitter > 0 else np.inf)
    return out


def score(p, sep, cache, cons=CONSTRAINTS, **kw):
    """log10|alpha_-/2pi| minus quadratic hinge penalties; -50 if infeasible."""
    r = evaluate(p, sep, cache, cons, **kw)
    if not r["ok"] or r["alpha"] == 0:
        return -50.0
    s = np.log10(abs(r["alpha"]) / (2 * np.pi))
    s -= 50 * max(0.0, r["ratio"] / cons["R_MAX"] - 1) ** 2
    s -= 50 * max(0.0, cons["NOISE_RATIO"] / r["noise_ratio"] - 1) ** 2
    if r["n_bound"] < cons["N_LEVELS"]:
        s -= 2.0 * (cons["N_LEVELS"] - r["n_bound"])
    return s


def random_design(rng, cons=CONSTRAINTS):
    shape = SHAPE_LO * (SHAPE_HI / SHAPE_LO) ** rng.random(6)          # log-uniform
    volts = rng.uniform(-cons["V_MAX"], cons["V_MAX"], 3)
    logmu = rng.uniform(LOGMU_LO, LOGMU_HI)
    return np.concatenate([shape, volts, [logmu]])


def optimize_sep(sep, n_geom=40, n_volt=40, n_polish=3, nm_iter=300, seed=0,
                 cons=CONSTRAINTS, cache=None, verbose=True, x0=None):
    """
    Stage 1: n_geom random geometries x n_volt random (V, mu) each, filtered by the
             cheap quadratic check; ladders evaluation of the survivors.
    Stage 2: Nelder-Mead polish (all 10 parameters) from the n_polish best seeds.
    """
    rng = np.random.default_rng(seed)
    cache = cache or TaylorCache()
    f = lambda p: -score(p, sep, cache, cons)
    seeds = []
    for _ in range(n_geom):
        for _ in range(200):                                   # resample until fabricable
            base = random_design(rng, cons)
            g, _, _ = unpack(base, sep)
            if geometry_ok(g):
                break
        else:
            continue
        Cg = cache(g)
        cands = []
        for _ in range(n_volt):
            p = base.copy()
            p[6:9] = rng.uniform(-cons["V_MAX"], cons["V_MAX"], 3)
            p[9] = rng.uniform(LOGMU_LO, LOGMU_HI)
            Bc = b_crit(combine(Cg, dict(comp=p[6], radial=p[7], axial=p[8])))
            if not np.isfinite(Bc):
                continue
            B = (1 + 10.0 ** p[9]) * Bc
            if B > cons["B_MAX"]:                              # B_crit ~ sqrt(V): scale voltages down
                p[6:9] *= 0.98 * (cons["B_MAX"] / B) ** 2
            if quad_check(p, sep, Cg, cons) is not None:
                cands.append(p)
        for p in cands[:8]:
            seeds.append((f(p), p))
    if x0 is not None:
        seeds.append((f(np.array(x0, float)), np.array(x0, float)))
    seeds.sort(key=lambda t: t[0])
    best = None
    for fval, p in seeds[:n_polish]:
        if fval >= 50:
            continue
        res = minimize(f, p, method="Nelder-Mead",
                       options=dict(maxiter=nm_iter, maxfev=nm_iter, xatol=1e-4, fatol=1e-4, adaptive=True))
        if best is None or res.fun < best.fun:
            best = res
    if best is None:
        if verbose:
            print(f"sep = {sep/um:.1f} um: no feasible design found")
        return None
    r = evaluate(best.x, sep, cache, cons)
    r["score"] = -best.fun
    r["x"] = best.x
    r["n_seeds"] = len(seeds)
    if verbose:
        report(r)
    return r


def report(r):
    P, k = r["P"], r["kerr"]
    g, v = r["g"], r["volts"]
    jc = r["jitter_contrib"]
    print(f"sep = {g['sep']/um:7.1f} um   B = {r['B']:.3f} T (= {r['B']/b_crit(r['C']):.3f} x B_crit)   "
          f"V(comp,rad,ax) = ({v['comp']:+.2f}, {v['radial']:+.2f}, {v['axial']:+.2f}) V")
    print(f"  comp: centre ({g['comp_cx']/um:.1f}, {g['comp_cz']/um:.1f}) um, size {g['comp_w']/um:.1f} x {g['comp_h']/um:.1f} um;"
          f"  radial width {g['radial_w']/um:.0f} um, axial height {g['axial_h']/um:.0f} um")
    print(f"  wz/2pi = {P['wz']/2/np.pi/1e6:9.2f} MHz   eps = {P['eps']:+.5f}   w1/wc = {P['w1']/P['wc']:.4f}   zeta_- = {r['G']['zeta_m']:+.3f}")
    print(f"  w~+/2pi = {P['wt_p']/2/np.pi/1e9:7.3f} GHz   w~-/2pi = {P['wt_m']/2/np.pi/1e6:9.3f} MHz")
    print(f"  alpha_-/2pi = {r['alpha']/2/np.pi/1e6:+9.4f} MHz  (1st order {r['alpha1']/2/np.pi/1e6:+.4f}, 2nd order {r['alpha2']/2/np.pi/1e6:+.4f};"
          f" non-perturbative {r['alpha_np']/2/np.pi/1e6:+.4f})")
    print(f"  |alpha|/w~- = {r['ratio']:.3f}   tau_gate = 5/|alpha| = {5/abs(r['alpha'])*1e6:.3f} us")
    print(f"  magnetron ladder: transitions/2pi {np.round(np.diff(r['E'])/2/np.pi/1e6, 2)} MHz;"
          f"  purity {np.round(r['purity'], 3)}  -> {r['n_bound']} clean levels")
    print(f"  w~- jitter/2pi = {r['jitter']/2/np.pi/1e3:.2f} kHz  (comp {jc['comp']/2/np.pi/1e3:.2f}, rad {jc['radial']/2/np.pi/1e3:.2f}, "
          f"ax {jc['axial']/2/np.pi/1e3:.2f}, B {jc['B']/2/np.pi/1e3:.2f} kHz)   |alpha|/jitter = {r['noise_ratio']:.1f}")
    print(f"  cross-Kerr/2pi: mag-ax {k['K_mz']/2/np.pi/1e6:+.4f} MHz, mag-cyc {k['K_pm']/2/np.pi/1e6:+.4f} MHz;"
          f"  axial self-Kerr 2K_zz/2pi = {2*k['K_zz']/2/np.pi/1e6:+.4f} MHz")
