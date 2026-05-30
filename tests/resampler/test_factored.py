import numpy as np
import pytest
import scipy.sparse as sp

from geohalo.resampler import FactoredResampler, Resampler


@pytest.mark.parametrize("iterations", [1, 2, 3])
def test_factored_matches_materialized_refine(iterations: int) -> None:
    s_lat = np.array([0.0, 1.0, 2.0, 3.0])
    s_lon = np.array([0.0, 1.0, 2.0])
    t_lat = np.linspace(0.0, 3.0, 12)
    t_lon = np.linspace(0.0, 2.0, 9)
    r = Resampler.compute(s_lat, s_lon, t_lat, t_lon, iterations=iterations)
    f = FactoredResampler.compute(s_lat, s_lon, t_lat, t_lon, iterations=iterations)
    rng = np.random.default_rng(0)
    flat = rng.uniform(0, 10, size=(7, s_lat.size * s_lon.size))
    expected = (r.transform_matrix @ flat.T).T
    np.testing.assert_allclose(f.apply_flat(flat), expected, atol=1e-9)


@pytest.mark.parametrize("iterations", [1, 2, 3])
def test_factored_matches_materialized_coarsen(iterations: int) -> None:
    s_lat = np.linspace(0.0, 4.0, 9)
    s_lon = np.linspace(0.0, 4.0, 9)
    t_lat = np.linspace(0.0, 4.0, 3)
    t_lon = np.linspace(0.0, 4.0, 3)
    r = Resampler.compute(s_lat, s_lon, t_lat, t_lon, iterations=iterations)
    f = FactoredResampler.compute(s_lat, s_lon, t_lat, t_lon, iterations=iterations)
    rng = np.random.default_rng(1)
    flat = rng.uniform(0, 10, size=(4, s_lat.size * s_lon.size))
    expected = (r.transform_matrix @ flat.T).T
    np.testing.assert_allclose(f.apply_flat(flat), expected, atol=1e-9)


@pytest.mark.parametrize("iterations", [1, 2, 3])
def test_fuse_left_matches_w_times_t_refine(iterations: int) -> None:
    s_lat = np.array([0.0, 1.0, 2.0, 3.0])
    s_lon = np.array([0.0, 1.0, 2.0])
    t_lat = np.linspace(0.0, 3.0, 12)
    t_lon = np.linspace(0.0, 2.0, 9)
    r = Resampler.compute(s_lat, s_lon, t_lat, t_lon, iterations=iterations)
    f = FactoredResampler.compute(s_lat, s_lon, t_lat, t_lon, iterations=iterations)
    n_t = t_lat.size * t_lon.size
    w = sp.random(6, n_t, density=0.3, format="csr", random_state=0)
    expected = (w @ r.transform_matrix).toarray()
    np.testing.assert_allclose(f.fuse_left(w).toarray(), expected, atol=1e-9)


@pytest.mark.parametrize("iterations", [1, 2, 3])
def test_fuse_left_matches_w_times_t_coarsen(iterations: int) -> None:
    s_lat = np.linspace(0.0, 4.0, 9)
    s_lon = np.linspace(0.0, 4.0, 9)
    t_lat = np.linspace(0.0, 4.0, 3)
    t_lon = np.linspace(0.0, 4.0, 3)
    r = Resampler.compute(s_lat, s_lon, t_lat, t_lon, iterations=iterations)
    f = FactoredResampler.compute(s_lat, s_lon, t_lat, t_lon, iterations=iterations)
    n_t = t_lat.size * t_lon.size
    w = sp.random(4, n_t, density=0.5, format="csr", random_state=1)
    expected = (w @ r.transform_matrix).toarray()
    np.testing.assert_allclose(f.fuse_left(w).toarray(), expected, atol=1e-9)


def test_factored_digest_matches_resampler() -> None:
    s_lat = np.array([0.0, 1.0, 2.0])
    s_lon = np.array([0.0, 1.0])
    t_lat = np.linspace(0.0, 2.0, 5)
    t_lon = np.linspace(0.0, 1.0, 3)
    r = Resampler.compute(s_lat, s_lon, t_lat, t_lon, iterations=2)
    f = FactoredResampler.compute(s_lat, s_lon, t_lat, t_lon, iterations=2)
    assert f.digest == r.digest
