"""N-body simulation helpers."""
import astropy.units as u
import numpy as np
import matplotlib.pyplot as plt
import agama
import pyfalcon
import scipy

agama.setUnits(length=1, velocity=1, mass=1)  # 1 kpc, 1 km/s, 1 Msun

def dynfricAccel(pot_host, sigma, pos, vel, mass):
    """Chandrasekhar dynamical friction acceleration for a point mass in host."""
    r = np.sqrt(np.sum(pos ** 2))
    v = np.sqrt(np.sum(vel ** 2))
    if v == 0:
        return np.zeros(3)

    rho = pot_host.density(pos)
    coulombLog = 3.0
    X = v / (np.sqrt(2) * sigma(r))

    return -vel / v * (
        4 * np.pi * agama.G ** 2 * mass * rho * coulombLog *
        (scipy.special.erf(X) - 2 / np.sqrt(np.pi) * X * np.exp(-X * X)) / v ** 2
    )

def king_rt_over_scaleRadius(W0=3.0, trunc=1.0):
    pot = agama.Potential(type="King", mass=1.0, scaleRadius=1.0, W0=float(W0), trunc=float(trunc))
    def _rho(r): return pot.density(r, 0.0, 0.0)
    r_hi = 1.0
    while _rho(r_hi) > 0 and r_hi < 1e6:
        r_hi *= 2.0
    r_lo = r_hi / 2.0

    for _ in range(80):
        r_mid = 0.5 * (r_lo + r_hi)
        if _rho(r_mid) > 0:
            r_lo = r_mid
        else:
            r_hi = r_mid
    return r_lo

def tidal_radius(pot_host, r, msat, dr_frac=1e-3):
    """
    Instantaneous tidal radius along +x-axis at Galactocentric radius r (axis proxy).
    """
    r = float(r)
    msat = float(msat)

    Fx = pot_host.force(r, 0.0, 0.0)[0]
    Vc2 = -r * Fx
    if Vc2 <= 0:
        raise ValueError(f"Non-positive Vc^2 at r={r}: Vc2={Vc2}")

    Omega2 = Vc2 / (r * r)

    dr = max(dr_frac * r, 1e-6)
    Phi_p = pot_host.potential(r + dr, 0.0, 0.0)
    Phi_0 = pot_host.potential(r,      0.0, 0.0)
    Phi_m = pot_host.potential(r - dr, 0.0, 0.0)
    d2Phi = (Phi_p - 2.0 * Phi_0 + Phi_m) / (dr * dr)

    f = 1.0 - d2Phi / Omega2
    if f <= 0:
        raise ValueError(f"Non-positive f at r={r}: f={f}")

    Menc = Vc2 * r / agama.G
    rtidal = (msat / (f * Menc)) ** (1.0 / 3.0) * r
    return rtidal

def make_satellite_ics(ft, seed, M_SAT, Nbody, pot_host, r_center0, KING_W0, KING_TRUNC, RT_OVER_R0):
    np.random.seed(int(seed))
    M = float(M_SAT)
    ft = float(ft)

    ra = float(np.linalg.norm(r_center0[0:3]))
    r_tidal_a = tidal_radius(pot_host, ra, M)
    r_out = ft * r_tidal_a
    r0 = r_out / RT_OVER_R0  # King scaleRadius

    pot_sat = agama.Potential(
        type="King",
        mass=M,
        scaleRadius=float(r0),
        W0=float(KING_W0),
        trunc=float(KING_TRUNC),
    )
    initmass = float(pot_sat.totalMass())

    df_sat = agama.DistributionFunction(type="quasispherical", potential=pot_sat)
    f_xv, mass = agama.GalaxyModel(pot_sat, df_sat).sample(int(Nbody))

    f_xv = np.array(f_xv, copy=True)
    mass = np.array(mass, copy=True)

    # Shift to Galactocentric initial center
    f_xv += r_center0
    return f_xv, mass, initmass, float(r_out), float(r_tidal_a), float(r0)