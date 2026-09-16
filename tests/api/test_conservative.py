import numpy as np
import pytest
import xarray as xr

import geohalo as ghl
from geohalo.geometry import target_coords_from_resolution
from tests.resampler.test_conservative import nested_grids


@pytest.mark.parametrize("descending", [False, True])
@pytest.mark.parametrize("dataset", [False, True])
@pytest.mark.parametrize("skipna", [False, True])
def test_conservative_xarray_metadata_custom_dims_and_missing(descending, dataset, skipna):
    lat, lon, tlat, tlon = nested_grids()
    values = np.random.default_rng(0).random((2, 3, lat.size, lon.size), dtype=np.float32)
    values[..., 5, 5] = np.nan
    grid = xr.DataArray(values, dims=("step", "member", "lat", "lon"),
                        coords={"step": [1, 2], "member": [3, 4, 5], "lat": lat, "lon": lon, "model": "test"},
                        attrs={"units": "mm"}, name="rain")
    operator = ghl.Resampler.compute(lat, lon, tlat[::-1], tlon, method="conservative")
    expected_values = operator.apply_grid(values, skipna=skipna).reshape(2, 3, 10, 10)
    if descending:
        grid = grid.isel(lat=slice(None, None, -1))
    grid = grid.transpose("lon", "step", "lat", "member")
    if dataset:
        grid = xr.Dataset({"rain": grid, "scalar": xr.DataArray(7)}, attrs={"source": "test"})
    actual = ghl.resample_grid_with_matrix(grid, operator, lat_dim="lat", lon_dim="lon", skipna=skipna)
    variable = actual.rain if dataset else actual
    assert variable.dims == ("step", "member", "lat", "lon")
    np.testing.assert_allclose(variable.to_numpy(), expected_values, equal_nan=True)
    np.testing.assert_array_equal(variable.lat, tlat[::-1])
    assert variable.attrs == {"units": "mm"}
    assert variable.model == "test"
    assert actual.attrs == grid.attrs
    if dataset:
        xr.testing.assert_identical(actual.scalar, grid.scalar)


@pytest.mark.parametrize("period", [None, 360])
@pytest.mark.parametrize("normalization", ["destination", "covered"])
@pytest.mark.parametrize("resolution", [.5, 7., 400.])
def test_conservative_resolution_api_and_singleton_targets(period, normalization, resolution):
    lat, lon = np.array([0., 1., 2.]), np.arange(-180., 180., 30.)
    grid = xr.DataArray(np.ones((3, lon.size)), dims=("latitude", "longitude"),
                        coords={"latitude": lat, "longitude": lon})
    tlat, tlon = target_coords_from_resolution(lat, lon, resolution, period=period)
    actual = ghl.resample_grid(grid, resolution, method="conservative", period=period,
                               normalization=normalization)
    np.testing.assert_array_equal(actual.latitude, tlat)
    np.testing.assert_array_equal(actual.longitude, tlon)
    assert np.isfinite(actual).all()
    if normalization == "covered":
        np.testing.assert_allclose(actual, 1.)
    else:
        assert float(actual.min()) > 0
        assert float(actual.max()) <= 1. + 1e-14


def test_convenience_explicit_singleton_source_bounds_and_skipna():
    grid = xr.DataArray([[2., np.nan]], dims=("latitude", "longitude"),
                        coords={"latitude": [0.], "longitude": [0., 1.]})
    bounds = (np.array([-.5, .5]), np.array([-.5, .5, 1.5]))
    result = ghl.resample_grid(grid, 4., method="conservative", source_bounds=bounds, skipna=True)
    np.testing.assert_allclose(result, [[2.]])


@pytest.mark.parametrize("options", [
    {"method": "bad"}, {"method": "conservative", "iterations": 2},
    {"method": "conservative", "normalization": "bad"},
    {"normalization": "covered"}, {"source_bounds": (np.arange(3.), np.arange(3.))},
])
def test_invalid_method_options_across_entry_points(options, tmp_path):
    coords = np.arange(2.)
    grid = xr.DataArray(np.ones((2, 2)), dims=("latitude", "longitude"),
                        coords={"latitude": coords, "longitude": coords})
    for call in (
        lambda: ghl.Resampler.compute(coords, coords, coords, coords, **options),
        lambda: ghl.resample_grid(grid, .5, **options),
        lambda: ghl.LocalCache(tmp_path).get_or_compute_resampler(coords, coords, coords, coords, **options),
    ):
        with pytest.raises(ValueError, match=r"method|normalization|iterations"):
            call()


def test_skipna_requires_conservative_before_reading(monkeypatch):
    coords = np.arange(2.)
    grid = xr.DataArray(np.ones((2, 2)), dims=("latitude", "longitude"),
                        coords={"latitude": coords, "longitude": coords})
    operator = ghl.Resampler.compute(coords, coords, coords, coords)

    def forbidden(*_args, **_kwargs):
        pytest.fail("invalid skipna must fail before loading values")

    monkeypatch.setattr(xr.DataArray, "to_numpy", forbidden)
    with pytest.raises(ValueError, match="requires method"):
        ghl.resample_grid_with_matrix(grid, operator, skipna=True)
    with pytest.raises(ValueError, match="requires method"):
        ghl.resample_grid(grid, .5, skipna=True)
