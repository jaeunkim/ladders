"""
flipchip_optimize.py
Maximise the magnetron self-anharmonicity |alpha_-| of the flip-chip Penning trap
over electrode geometry, electrode voltages and magnetic field.

Design vector (11 numbers) for a fixed chip separation d = 2h:
    cx/h, cz/h, w/h, hh/h        compensation electrode centre and size, in units of h
    rw/h, ah/h, aw/h             radial-electrode width, axial-electrode height and axial-electrode
                                 width, in units of h  (the axial width is a free parameter)
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
    |V| <= V_MAX,  B_MIN <= B <= B_MAX,  fabrication bounds from geometry_ok():
                                      every electrode dimension >= fp.MIN_FEATURE (10 um by
                                      default) and an inter-electrode gap of CONSTRAINTS["GAP"]
"""
import warnings
import numpy as np
from scipy.optimize import minimize

import flipchip_potential as fp          # read fp.MIN_FEATURE live, so the notebook can change it
from flipchip_potential import (DEFAULT_GEOM, um, taylor_all_groups, combine,
                                geometry_ok, laplace_residual, GROUPS, GEOM_KEYS,
                                axial_width, gap_of)
from flipchip_quantum import (quad_params, diag_frame_operators, anharmonic_operator,
                              kerr_table, alpha_from_kerr, alpha_second_order, magnetron_spectrum, m_e, e_ch)

CONSTRAINTS = dict(R_MAX=0.10, S_MAX=0.5, WM_MIN=2 * np.pi * 1e6,
                   N_LEVELS=4, PURITY_MIN=0.9, FOCK_DIM=16,
                   NOISE_RATIO=10.0, SIGMA_V=1e-6, V_FLOOR=1e-6, SIGMA_B=1e-8,
                   V_MAX=14.0, B_MIN=0.05, B_MAX=0.3,
                   GAP=5 * um)          # inter-electrode gap stamped into every optimised geometry

SHAPE_NAMES = ["cx/h", "cz/h", "w/h", "hh/h", "rw/h", "ah/h", "aw/h"]
SHAPE_LO = np.array([0.15, 0.15, 0.05, 0.05, 0.3, 0.3, 0.05])
SHAPE_HI = np.array([3.0, 3.0, 3.0, 3.0, 12.0, 12.0, 6.0])
N_SHAPE = len(SHAPE_NAMES)          # 7
IV = slice(N_SHAPE, N_SHAPE + 3)    # slice of the design vector holding (Vc, Vr, Va)
ILOGMU = N_SHAPE + 3                # index of log10(mu)
NDIM = N_SHAPE + 4                  # 11
SIZE_IDX = [2, 3, 4, 5, 6]          # entries that are electrode dimensions (subject to MIN_FEATURE)
LOGMU_LO, LOGMU_HI = -3.0, 1.5


def shape_bounds(sep, cons=CONSTRAINTS):
    """(lo, hi) for the shape part of the design vector, in units of h, with the lower bounds
    lifted so that a sampled geometry can satisfy the minimum feature size fp.MIN_FEATURE and
    the gap.  For very small separations this window can be narrow, but it is never empty."""
    h = sep / 2
    mf, gap = fp.MIN_FEATURE, cons["GAP"]
    lo, hi = SHAPE_LO.copy(), SHAPE_HI.copy()
    lo[SIZE_IDX] = np.maximum(lo[SIZE_IDX], mf / h)          # every electrode dimension >= mf
    lo[0] = max(lo[0], (gap + mf / 2) / h)                   # cx: comp must clear the axis by a gap
    lo[1] = max(lo[1], (gap + mf) / 2 / h)                   # cz: mirrored comp pair separated by a gap
    return lo, np.maximum(hi, lo * 1.5)


def b_crit(C):
    """B at which omega_1 = 0 for the axial curvature in C (needs q C002 > 0)."""
    wz2 = 2 * (-e_ch) * C[(0, 0, 2)] / m_e
    return np.sqrt(2 * wz2) * m_e / e_ch if wz2 > 0 else np.nan


def unpack(p, sep, Cg=None, cons=CONSTRAINTS):
    h = sep / 2
    g = dict(comp_cx=p[0] * h, comp_cz=p[1] * h, comp_w=p[2] * h, comp_h=p[3] * h,
             radial_w=p[4] * h, axial_h=p[5] * h, axial_w=p[6] * h,
             gap=cons["GAP"], sep=sep)
    volts = dict(comp=p[7], radial=p[8], axial=p[9])
    B = None
    if Cg is not None:
        B = (1 + 10.0 ** p[ILOGMU]) * b_crit(combine(Cg, volts))
    return g, volts, B


def pack(g, volts, B, Cg):
    h = g["sep"] / 2
    mu = B / b_crit(combine(Cg, volts)) - 1
    return np.array([g["comp_cx"] / h, g["comp_cz"] / h, g["comp_w"] / h, g["comp_h"] / h,
                     g["radial_w"] / h, g["axial_h"] / h, axial_width(g) / h,
                     volts["comp"], volts["radial"], volts["axial"], np.log10(mu)])


class TaylorCache:
    """Geometry -> per-group Taylor coefficients (the expensive step), memoised."""
    def __init__(self, **fit_kw):
        self.store = {}
        self.fit_kw = dict(npts=800, nmax=30, max_deg=10)
        self.fit_kw.update(fit_kw)

    def __call__(self, g):
        key = tuple(round(v / um, 3) for v in
                    [g[k] for k in GEOM_KEYS[:6]] + [axial_width(g), gap_of(g), g["sep"]])
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
    g, volts, B = unpack(p, sep, Cg, cons)
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
    g, volts, _ = unpack(p, sep, None, cons)
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


def random_design(rng, sep, cons=CONSTRAINTS):
    lo, hi = shape_bounds(sep, cons)
    shape = lo * (hi / lo) ** rng.random(N_SHAPE)                      # log-uniform
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
            base = random_design(rng, sep, cons)
            g, _, _ = unpack(base, sep, None, cons)
            if geometry_ok(g):
                break
        else:
            continue
        Cg = cache(g)
        cands = []
        for _ in range(n_volt):
            p = base.copy()
            p[IV] = rng.uniform(-cons["V_MAX"], cons["V_MAX"], 3)
            p[ILOGMU] = rng.uniform(LOGMU_LO, LOGMU_HI)
            Bc = b_crit(combine(Cg, dict(zip(GROUPS, p[IV]))))
            if not np.isfinite(Bc):
                continue
            B = (1 + 10.0 ** p[ILOGMU]) * Bc
            if B > cons["B_MAX"]:                              # B_crit ~ sqrt(V): scale voltages down
                p[IV] *= 0.98 * (cons["B_MAX"] / B) ** 2
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
          f"  radial width {g['radial_w']/um:.1f} um, axial {axial_width(g)/um:.1f} (w) x {g['axial_h']/um:.1f} (h) um;"
          f"  gap {gap_of(g)/um:.1f} um, min feature {min(g['comp_w'], g['comp_h'], g['radial_w'], g['axial_h'], axial_width(g))/um:.1f} um")
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
