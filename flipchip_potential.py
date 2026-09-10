"""
flipchip_potential.py
Python port of 20260531_flip_chip_potential_analytic.m (FlipChipBoth branch).

Coordinates (Verdu 2011 convention, as in the Mathematica file):
    y : chip-normal (the two chips sit at y = +h and y = -h, separation d = 2h)
    z : trap axis (B || z)
    x : in-plane transverse
Electrode rectangles live in the x-z plane. Both chips carry the same pattern
and the same voltages (potFlipBoth), so the potential is exactly even in y.

Potential of a rectangular patch at unit voltage on an otherwise grounded plane:
House 2008, eq. 22.  Two parallel grounded planes -> alternating image series.
"""
import numpy as np

um = 1e-6
GAP = 5 * um           # fallback inter-electrode gap, used only when a geometry dict has no "gap" key
# 20260910 Jaeun: GAP tp 5um, as discussed in the lab meeting. (TODO: try 3, 5, 10, 20 um and see how the coefficients change)
# The gap is now a per-geometry parameter (g["gap"]): DEFAULT_GEOM (the GDS design) uses 20 um,
# the optimisation loop uses CONSTRAINTS["GAP"] = 5 um.

MIN_FEATURE = 10 * um  # smallest fabricable electrode dimension (comp width/height, radial width,
                       # axial height, axial width).  Adjust with fp.MIN_FEATURE = ... , or pass
                       # min_feature=... to geometry_ok().


# ----------------------------------------------------------------------------
# 1. House 2008 single-plane rectangle + flip-chip image series
# ----------------------------------------------------------------------------
def pot0(x, x0, y, z, z0):
    """Recurring term of House 2008 eq. 22 (unit voltage)."""
    return np.arctan((x - x0) * (z - z0) / (y * np.sqrt((x - x0) ** 2 + y ** 2 + (z - z0) ** 2))) / (2 * np.pi)


def pot_house(x, xup, xlow, y, z, zup, zlow):
    """Unit-voltage potential above a rectangle [xlow,xup]x[zlow,zup] in the plane y=0."""
    return (pot0(x, xup, y, z, zup) - pot0(x, xlow, y, z, zup)
            - (pot0(x, xup, y, z, zlow) - pot0(x, xlow, y, z, zlow)))


def pot_flip_both(x, xup, xlow, y, z, zup, zlow, h, nmax=30):
    """Both chips (at y=+h and y=-h) carry the same rectangle at unit voltage.
    Alternating image series, truncated at nmax (converges like 1/n^2, and the
    contribution to derivatives at the centre converges much faster)."""
    out = 0.0
    for n in range(nmax + 1):
        s = (-1) ** n
        out = out + s * (pot_house(x, xup, xlow, (2 * n + 1) * h - y, z, zup, zlow)
                         + pot_house(x, xup, xlow, (2 * n + 1) * h + y, z, zup, zlow))
    return out


# ----------------------------------------------------------------------------
# 2. Electrode layout (sensei's 20260525_penning_test_v2.gds, parameterised)
# ----------------------------------------------------------------------------
DEFAULT_GEOM = dict(
    comp_cx=300 * um,      # compCenterX : x-centre of the upper-right compensation electrode
    comp_cz=300 * um,      # compCenterY (Mathematica) : z-centre of the same electrode
    comp_w=260 * um,       # compWidth   : x-extent
    comp_h=260 * um,       # compHeight  : z-extent
    radial_w=2550 * um,    # radialWidth : x-extent of the radial electrodes
    axial_h=4570 * um,     # axialHeight : z-extent of the axial electrodes
    axial_w=300 * um,      # axialWidth  : x-extent of the axial electrodes (free parameter; 300 um
                           #               = 2*comp_cx - comp_w - 2*gap, the GDS value at gap = 20 um)
    gap=20 * um,           # inter-electrode gap of the GDS design
    sep=1000 * um,         # flipSeparation d = 2h
)

GEOM_KEYS = ("comp_cx", "comp_cz", "comp_w", "comp_h", "radial_w", "axial_h", "axial_w", "gap", "sep")


def gap_of(g):
    """Inter-electrode gap of a geometry (falls back to the module default GAP)."""
    return g.get("gap", GAP)


def axial_width(g):
    """x-extent of the axial electrodes.  Free parameter g["axial_w"]; geometry dicts written
    before it existed fall back to the old rule (axial electrode exactly spanning the space
    between the two mirrored compensation/radial columns)."""
    aw = g.get("axial_w")
    return 2 * g["comp_cx"] - g["comp_w"] - 2 * gap_of(g) if aw is None else aw


def electrode_rects(g):
    """Return dict group -> list of rectangles (xlow, xup, zlow, zup).
    Mirror copies as in the Mathematica file (ReflectionTransform)."""
    cx, cz, w, hh = g["comp_cx"], g["comp_cz"], g["comp_w"], g["comp_h"]
    gap = gap_of(g)
    aw = axial_width(g)

    comp_ur = (cx - w / 2, cx + w / 2, cz - hh / 2, cz + hh / 2)
    rad_ur = (cx + w / 2 + gap, cx + w / 2 + gap + g["radial_w"], cz - hh / 2, cz + hh / 2)
    ax_u = (-aw / 2, aw / 2, cz + hh / 2 + gap, cz + hh / 2 + gap + g["axial_h"])

    def mirror_x(r):
        return (-r[1], -r[0], r[2], r[3])

    def mirror_z(r):
        return (r[0], r[1], -r[3], -r[2])

    four = lambda r: [r, mirror_x(r), mirror_z(r), mirror_x(mirror_z(r))]
    return {"comp": four(comp_ur), "radial": four(rad_ur), "axial": [ax_u, mirror_z(ax_u)]}


def geometry_ok(g, min_feature=None):
    """Fabrication / topology sanity: every electrode dimension >= the minimum feature size
    (MIN_FEATURE, or the min_feature argument), no overlaps, sep >= 20 um."""
    mf = MIN_FEATURE if min_feature is None else min_feature
    cx, cz, w, hh = g["comp_cx"], g["comp_cz"], g["comp_w"], g["comp_h"]
    gap = gap_of(g)
    return (gap > 0
            and min(w, hh, g["radial_w"], g["axial_h"], axial_width(g)) >= mf
            and (cx - w / 2) > gap and (cz - hh / 2) >= gap / 2
            and g["sep"] >= 20 * um)


def group_potential(group, g, x, y, z, nmax=30):
    """Unit-voltage potential of one electrode group at points (x,y,z)."""
    h = g["sep"] / 2
    tot = 0.0
    for (xl, xu, zl, zu) in electrode_rects(g)[group]:
        tot = tot + pot_flip_both(x, xu, xl, y, z, zu, zl, h, nmax)
    return tot


GROUPS = ("comp", "radial", "axial")


def total_potential(g, volts, x, y, z, nmax=30):
    """volts = dict(comp=Vc, radial=Vr, axial=Va) [V]."""
    return sum(volts[k] * group_potential(k, g, x, y, z, nmax) for k in GROUPS)


# ----------------------------------------------------------------------------
# 3. Taylor coefficients C_ijk at the trap centre by harmonic-polynomial fit
# ----------------------------------------------------------------------------
def even_monomials(max_deg):
    """All (i,j,k) with i,j,k even and i+j+k <= max_deg (the only ones that
    survive the three mirror symmetries x->-x, y->-y, z->-z)."""
    out = []
    for i in range(0, max_deg + 1, 2):
        for j in range(0, max_deg + 1 - i, 2):
            for k in range(0, max_deg + 1 - i - j, 2):
                out.append((i, j, k))
    return out


def fit_taylor(g, group, max_deg=10, rel_radius=0.08, npts=1500, nmax=30, seed=0):
    """
    Least-squares fit of the unit-voltage potential of one group to an even
    polynomial of degree max_deg, sampled at random points filling the ball of
    radius R = rel_radius * h around the origin (points must fill the ball, not
    lie on shells: on a shell r^2 is constant and x^6 r^2 is indistinguishable
    from x^8).  Returns {(i,j,k): C_ijk} with
    C_ijk = (1/i!j!k!) d^{i+j+k} Phi / dx^i dy^j dz^k  [V/m^(i+j+k) per volt].

    Because the flip-chip symmetry nearly cancels the curvature of the
    compensation electrodes, the quartic term overtakes the quadratic one only
    ~30 um from the centre at d = 1 mm, so the fit radius must be small; the
    degree-2/4/6 coefficients are then validated against the Laplace equation
    by laplace_residual() and against high-precision finite differences.
    """
    rng = np.random.default_rng(seed)
    h = g["sep"] / 2
    R = rel_radius * h
    mons = even_monomials(max_deg)
    v = rng.normal(size=(npts, 3))
    v /= np.linalg.norm(v, axis=1)[:, None]
    r = R * (0.15 + 0.85 * rng.random(npts) ** (1 / 3))       # fill the ball, avoid r~0
    X, Y, Z = (v * r[:, None]).T
    A = np.stack([(X / R) ** i * (Y / R) ** j * (Z / R) ** k for (i, j, k) in mons], axis=1)
    phi = group_potential(group, g, X, Y, Z, nmax)
    phi0 = group_potential(group, g, 0.0, 0.0, 0.0, nmax)
    coef, *_ = np.linalg.lstsq(A[:, 1:], phi - phi0, rcond=None)   # constant term pinned to phi(0)
    coef = np.concatenate([[phi0], coef])
    resid = np.max(np.abs(A @ coef - phi))
    C = {m: c / R ** sum(m) for m, c in zip(mons, coef)}
    return C, resid


def laplace_residual(C):
    """Relative violation of the Laplace equation by the degree-2 and degree-4
    parts of a coefficient dict.  Should be ~1e-6 or smaller."""
    def lap(deg):
        # nabla^2 of sum C_ijk x^i y^j z^k restricted to total degree 'deg',
        # evaluated coefficient-wise: term x^i -> i(i-1) x^{i-2}
        out = {}
        for (i, j, k), c in C.items():
            if i + j + k != deg:
                continue
            for (di, dj, dk), f in (((2, 0, 0), i * (i - 1)), ((0, 2, 0), j * (j - 1)), ((0, 0, 2), k * (k - 1))):
                if f:
                    key = (i - di, j - dj, k - dk)
                    out[key] = out.get(key, 0.0) + f * c
        return out
    res = {}
    for deg in (2, 4, 6):
        l = lap(deg)
        scale = max(abs(c) * (max(1, i) * max(1, i - 1)) for (i, j, k), c in C.items() if i + j + k == deg)
        res[deg] = max(abs(v) for v in l.values()) / scale if l else 0.0
    return res


def taylor_all_groups(g, **kw):
    """{group: C_dict}; also returns the max fit residual."""
    out, worst = {}, 0.0
    for grp in GROUPS:
        C, r = fit_taylor(g, grp, **kw)
        out[grp] = C
        worst = max(worst, r)
    return out, worst


def combine(Cgroups, volts):
    """Total C_ijk for a voltage set."""
    keys = Cgroups["comp"].keys()
    return {m: sum(volts[k] * Cgroups[k][m] for k in GROUPS) for m in keys}
