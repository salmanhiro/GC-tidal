"""Globular cluster (GC) model and core physical calculations."""

import numpy as np

# Gravitational constant in units of kpc, km/s, Msun
_G = 4.3009e-6  # kpc (km/s)^2 Msun^-1

# Unit conversion: 1 kpc/(km/s) in Gyr
# 1 kpc = 3.085677581e16 km; 1 Gyr = 3.15576e16 s
_kpc_km_per_s_to_Gyr = 3.085677581e16 / 3.15576e16  # ≈ 0.9778


class GlobularCluster:
    """A simple globular cluster model.

    Parameters
    ----------
    mass : float
        Total mass of the cluster in solar masses (Msun).
    r_half : float
        Half-mass radius of the cluster in kpc.
    n_stars : int, optional
        Number of stars in the cluster (used for relaxation time).
        Defaults to ``int(mass)`` (i.e., 1 Msun per star).
    """

    def __init__(self, mass, r_half, n_stars=None):
        if mass <= 0:
            raise ValueError("mass must be positive")
        if r_half <= 0:
            raise ValueError("r_half must be positive")
        self.mass = float(mass)
        self.r_half = float(r_half)
        self.n_stars = int(n_stars) if n_stars is not None else int(mass)

    def tidal_radius(self, r_orb, m_enc):
        """Compute the Jacobi (tidal) radius of the cluster.

        For a cluster on a circular orbit at galactocentric distance *r_orb*
        inside an enclosed host mass *m_enc*, the tidal radius is

        .. math::

            r_t = r_{\\rm orb} \\left(\\frac{M_{\\rm GC}}{3\\,M_{\\rm enc}}\\right)^{1/3}

        Parameters
        ----------
        r_orb : float
            Galactocentric orbital radius in kpc.
        m_enc : float
            Host galaxy mass enclosed within *r_orb* in Msun.

        Returns
        -------
        float
            Tidal radius in kpc.
        """
        if r_orb <= 0:
            raise ValueError("r_orb must be positive")
        if m_enc <= 0:
            raise ValueError("m_enc must be positive")
        return r_orb * (self.mass / (3.0 * m_enc)) ** (1.0 / 3.0)

    def velocity_dispersion(self):
        """Compute the 1-D line-of-sight velocity dispersion.

        Uses the Plummer-sphere approximation:

        .. math::

            \\sigma_{\\rm 1D} = \\sqrt{\\frac{G\\,M}{6\\,r_{1/2}}}

        Returns
        -------
        float
            1-D velocity dispersion in km/s.
        """
        return np.sqrt(_G * self.mass / (6.0 * self.r_half))

    def half_mass_relaxation_time(self):
        """Compute the half-mass relaxation time.

        Uses the Spitzer (1987) formula:

        .. math::

            t_{\\rm rh} = \\frac{0.138\\,N^{1/2}\\,r_{1/2}^{3/2}}{
                          (G\\,m_\\star)^{1/2}\\,\\ln(0.11\\,N)}

        where :math:`m_\\star = M / N` is the mean stellar mass.

        Returns
        -------
        float
            Half-mass relaxation time in Gyr (assuming kpc, km/s, Msun units,
            the raw result is converted from kpc/km s to Gyr).
        """
        N = self.n_stars
        m_star = self.mass / N
        # t_rh in kpc / (km/s) = kpc s / km; convert to Gyr
        t_rh_raw = (
            0.138
            * np.sqrt(N)
            * self.r_half ** 1.5
            / (np.sqrt(_G * m_star) * np.log(0.11 * N))
        )
        return t_rh_raw * _kpc_km_per_s_to_Gyr
