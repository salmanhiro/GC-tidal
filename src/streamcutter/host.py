"""Galactic host potential utilities."""

import numpy as np
import agama


def build_host(pot_ini):
    pot_host = agama.Potential(pot_ini)

    def sigma(r):
        r = float(r)
        Fx = pot_host.force(r, 0.0, 0.0)[0]
        Vc2 = -r * Fx
        if Vc2 <= 0:
            raise ValueError(f"Non-positive Vc^2 at r={r}: {Vc2}")
        return np.sqrt(Vc2) / np.sqrt(2.0)

    return pot_host, sigma
