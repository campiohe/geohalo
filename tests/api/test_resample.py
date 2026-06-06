import numpy as np
import xarray as xr

from geohalo.api import resample_grid, resample_grid_with_matrix
from geohalo.resampler import Resampler


def _da(values, lats, lons, extra_dims=()):
    dims = (*extra_dims, "latitude", "longitude")
    return xr.DataArray(values, dims=dims, coords={"latitude": lats, "longitude": lons})


def test_with_matrix_shapes() -> None:
    s_lat = np.array([0.0, 1.0, 2.0])
    s_lon = np.array([0.0, 1.0, 2.0])
    t_lat = np.linspace(0.0, 2.0, 6)
    t_lon = np.linspace(0.0, 2.0, 6)
    r = Resampler.compute(s_lat, s_lon, t_lat, t_lon, iterations=1)
    out = resample_grid_with_matrix(_da(np.arange(9.0).reshape(3, 3), s_lat, s_lon), r)
    assert out.sizes == {"latitude": 6, "longitude": 6}


def test_with_matrix_preserves_batch() -> None:
    s_lat = np.array([0.0, 1.0, 2.0])
    s_lon = np.array([0.0, 1.0, 2.0])
    t_lat = np.linspace(0.0, 2.0, 6)
    t_lon = np.linspace(0.0, 2.0, 6)
    r = Resampler.compute(s_lat, s_lon, t_lat, t_lon, iterations=1)
    da = _da(np.zeros((4, 3, 3)), s_lat, s_lon, extra_dims=("member",))
    out = resample_grid_with_matrix(da, r)
    assert out.dims == ("member", "latitude", "longitude")
    assert out.sizes["member"] == 4


def test_resample_grid_mean_preserved_constant() -> None:
    lats = np.array([0.0, 1.0, 2.0])
    lons = np.array([0.0, 1.0, 2.0])
    out = resample_grid(_da(np.full((3, 3), 5.0), lats, lons), target_resolution=0.5, iterations=3)
    np.testing.assert_allclose(out.values, 5.0, atol=1e-9)


def test_resample_grid_descending_lats_match_ascending() -> None:
    # CLAUDE.md convention: a descending grid and its flipped twin give the same results.
    rng = np.random.default_rng(7)
    vals = rng.random((4, 5))
    lats = np.linspace(0.0, 3.0, 4)
    lons = np.linspace(0.0, 4.0, 5)
    da = _da(vals, lats, lons)
    out_asc = resample_grid(da, target_resolution=0.5, iterations=3)
    out_desc = resample_grid(da.sortby("latitude", ascending=False), target_resolution=0.5, iterations=3)
    np.testing.assert_allclose(out_desc.sortby("latitude").values, out_asc.values, atol=1e-12)


def test_resample_grid_dataset() -> None:
    lats = np.array([0.0, 1.0, 2.0])
    lons = np.array([0.0, 1.0, 2.0])
    ds = xr.Dataset(
        {"t": (("latitude", "longitude"), np.full((3, 3), 2.0)),
         "u": (("latitude", "longitude"), np.full((3, 3), 9.0))},
        coords={"latitude": lats, "longitude": lons},
    )
    out = resample_grid(ds, target_resolution=0.5)
    assert isinstance(out, xr.Dataset)
    assert set(out.data_vars) == {"t", "u"}
