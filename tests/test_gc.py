"""Tests for the streamcutter.gc module."""

import math
import pytest
from streamcutter.gc import GlobularCluster, _G, _kpc_km_per_s_to_Gyr


class TestGlobularClusterInit:
    def test_basic_creation(self):
        gc = GlobularCluster(mass=1e5, r_half=0.005)
        assert gc.mass == 1e5
        assert gc.r_half == 0.005

    def test_default_n_stars(self):
        gc = GlobularCluster(mass=1e5, r_half=0.005)
        assert gc.n_stars == int(1e5)

    def test_custom_n_stars(self):
        gc = GlobularCluster(mass=1e5, r_half=0.005, n_stars=50000)
        assert gc.n_stars == 50000

    def test_invalid_mass_raises(self):
        with pytest.raises(ValueError):
            GlobularCluster(mass=0, r_half=0.005)
        with pytest.raises(ValueError):
            GlobularCluster(mass=-1e5, r_half=0.005)

    def test_invalid_r_half_raises(self):
        with pytest.raises(ValueError):
            GlobularCluster(mass=1e5, r_half=0)
        with pytest.raises(ValueError):
            GlobularCluster(mass=1e5, r_half=-0.005)


class TestTidalRadius:
    def setup_method(self):
        self.gc = GlobularCluster(mass=1e5, r_half=0.005)

    def test_known_value(self):
        # r_t = r_orb * (M_gc / (3 * M_enc))^(1/3)
        r_orb = 8.0          # kpc
        m_enc = 6e10         # Msun
        expected = r_orb * (1e5 / (3.0 * 6e10)) ** (1.0 / 3.0)
        assert math.isclose(self.gc.tidal_radius(r_orb, m_enc), expected, rel_tol=1e-9)

    def test_scales_with_r_orb(self):
        m_enc = 6e10
        rt1 = self.gc.tidal_radius(8.0, m_enc)
        rt2 = self.gc.tidal_radius(16.0, m_enc)
        assert math.isclose(rt2 / rt1, 2.0, rel_tol=1e-9)

    def test_scales_with_mass(self):
        r_orb, m_enc = 8.0, 6e10
        gc2 = GlobularCluster(mass=8e5, r_half=0.005)
        rt1 = self.gc.tidal_radius(r_orb, m_enc)
        rt2 = gc2.tidal_radius(r_orb, m_enc)
        # mass ratio = 8 => r_t ratio = 8^(1/3) = 2
        assert math.isclose(rt2 / rt1, 2.0, rel_tol=1e-9)

    def test_invalid_r_orb_raises(self):
        with pytest.raises(ValueError):
            self.gc.tidal_radius(0, 6e10)

    def test_invalid_m_enc_raises(self):
        with pytest.raises(ValueError):
            self.gc.tidal_radius(8.0, 0)


class TestVelocityDispersion:
    def test_known_value(self):
        mass = 1e5
        r_half = 0.005
        gc = GlobularCluster(mass=mass, r_half=r_half)
        expected = math.sqrt(_G * mass / (6.0 * r_half))
        assert math.isclose(gc.velocity_dispersion(), expected, rel_tol=1e-9)

    def test_positive(self):
        gc = GlobularCluster(mass=1e5, r_half=0.005)
        assert gc.velocity_dispersion() > 0

    def test_increases_with_mass(self):
        gc1 = GlobularCluster(mass=1e5, r_half=0.005)
        gc2 = GlobularCluster(mass=2e5, r_half=0.005)
        assert gc2.velocity_dispersion() > gc1.velocity_dispersion()

    def test_decreases_with_radius(self):
        gc1 = GlobularCluster(mass=1e5, r_half=0.005)
        gc2 = GlobularCluster(mass=1e5, r_half=0.010)
        assert gc2.velocity_dispersion() < gc1.velocity_dispersion()


class TestHalfMassRelaxationTime:
    def test_positive(self):
        gc = GlobularCluster(mass=1e5, r_half=0.005, n_stars=100000)
        assert gc.half_mass_relaxation_time() > 0

    def test_known_value(self):
        N = 100000
        mass = 1e5
        r_half = 0.005
        gc = GlobularCluster(mass=mass, r_half=r_half, n_stars=N)
        m_star = mass / N
        expected = (
            0.138
            * math.sqrt(N)
            * r_half ** 1.5
            / (math.sqrt(_G * m_star) * math.log(0.11 * N))
        ) * _kpc_km_per_s_to_Gyr
        assert math.isclose(gc.half_mass_relaxation_time(), expected, rel_tol=1e-9)

    def test_increases_with_n_stars(self):
        gc1 = GlobularCluster(mass=1e5, r_half=0.005, n_stars=10000)
        gc2 = GlobularCluster(mass=1e5, r_half=0.005, n_stars=100000)
        assert gc2.half_mass_relaxation_time() > gc1.half_mass_relaxation_time()
