from collections import Counter

import geopandas as gpd
import numpy as np
import pytest
import shapely
import xarray as xr
from zarr.storage import MemoryStore

import geohalo as ghl
from geohalo.geometry import target_coords_from_resolution
from geohalo.resampler import FactoredResampler
from tests.restricted_operator.test_zarr import CountingStore


def seam_inputs():
    lat, lon = np.arange(3.), np.arange(-180., 180., 2.)
    target_lat, target_lon = np.arange(0., 2.01, .5), np.arange(176., 184.01, .5)
    geoms = gpd.GeoSeries([
        shapely.box(178.6, .1, 179.6, .7), shapely.box(180.3, 1.1, 181.4, 1.8),
    ], index=["zulu", "alpha"])
    stencil = ghl.Stencil.compute(target_lat, target_lon, geoms)
    grid = xr.DataArray(
        np.random.default_rng(0).normal(size=(2, lat.size, lon.size)).astype(np.float32),
        dims=("time", "latitude", "longitude"),
        coords={"time": [0, 1], "latitude": lat, "longitude": lon, "model": "test"},
        name="value", attrs={"units": "K"},
    )
    return grid, stencil, geoms


@pytest.mark.parametrize("descending", [False, True])
@pytest.mark.parametrize("how", ["mean", "sum"])
@pytest.mark.parametrize("iterations", [1, 3])
def test_periodic_fused_restricted_and_stencil_agree(descending, how, iterations):
    grid, stencil, _ = seam_inputs()
    if descending:
        grid = grid.isel(latitude=slice(None, None, -1), longitude=slice(None, None, -1))
    lat, lon = grid.latitude.values, grid.longitude.values
    resampler = ghl.Resampler.compute(lat, lon, stencil.lats, stencil.lons, iterations=iterations, period=360)
    resampled = ghl.resample_grid_with_matrix(grid, resampler)
    expected = ghl.reduce_with_stencil(resampled, stencil, how=how)
    operator = ghl.ReduceOperator.compute(stencil, lat, lon, iterations=iterations, period=360)
    plan = ghl.RestrictedOperator.compute(operator, lat, 1, 30)
    full = ghl.reduce_with_operator(grid, operator, how=how)
    xr.testing.assert_allclose(full, expected, rtol=1e-12, atol=1e-12)
    xr.testing.assert_identical(ghl.reduce_with_restricted_operator(grid, plan, how=how), full)
    xr.testing.assert_identical(
        ghl.reduce_with_stencil(grid, stencil, resample_iterations=iterations, period=360, how=how), full,
    )
    preserved = ghl.reduce_with_stencil(grid, stencil, period=360, resample_iterations=iterations,
                                      how=how, preserve_dtype=True)
    xr.testing.assert_identical(preserved, full.astype(np.float32))
    np.testing.assert_array_equal(full.geom, ["zulu", "alpha"])
    assert full.attrs == grid.attrs
    columns = np.unique(operator.matrix.indices % lon.size)
    assert columns.min() == 0
    assert columns.max() == lon.size - 1
    # The seam halo reaches both ends, but never pulls in the middle of the globe.
    assert np.all((columns < 10) | (columns >= lon.size - 10))


@pytest.mark.parametrize("missing", [False, True])
@pytest.mark.parametrize("weighted", [False, True])
@pytest.mark.parametrize("how", ["mean", "sum"])
def test_periodic_masked_and_weighted_paths(missing, weighted, how):
    grid, stencil, _ = seam_inputs()
    if missing:
        grid.values[0, 0, 0] = np.nan
    ds = xr.Dataset({"value": grid, "scalar": xr.DataArray(7)}, attrs={"source": "test"})
    if weighted:
        ds["weight"] = abs(grid.fillna(2.)) + 1.
    kwargs = {"how": how, "weight_key": "weight" if weighted else None, "preserve_dtype": True}
    resampler = FactoredResampler.compute(grid.latitude.values, grid.longitude.values,
                                          stencil.lats, stencil.lons, iterations=2, period=360)
    # The stencil path runs the factored recurrence before masking. With NaNs,
    # algebraic cancellation in a materialized transform changes propagation,
    # so keep that existing masking convention in the reference.
    resampled = ds.drop_dims(["latitude", "longitude"])
    for name in ds.data_vars:
        if "latitude" in ds[name].dims:
            values = resampler.apply_flat(ds[name].to_numpy().reshape(2, -1))
            resampled[name] = xr.DataArray(
                values.reshape(2, stencil.lats.size, stencil.lons.size), dims=grid.dims,
                coords={"time": grid.time, "latitude": stencil.lats, "longitude": stencil.lons},
                attrs=ds[name].attrs,
            )
    # Cast after normalization, matching preserve_dtype's result-only contract.
    expected = ghl.reduce_with_stencil(resampled, stencil, **{**kwargs, "preserve_dtype": False})
    for name in ds.data_vars:
        if "latitude" in ds[name].dims:
            expected[name] = expected[name].astype(np.float32)
    actual = ghl.reduce_with_stencil(ds, stencil, period=360, resample_iterations=2, **kwargs)
    xr.testing.assert_allclose(actual, expected, rtol=2e-6, atol=1e-7)
    assert actual.value.dtype == np.float32
    assert actual.attrs == ds.attrs
    xr.testing.assert_identical(actual.scalar, ds.scalar)


@pytest.mark.parametrize("as_dataset", [False, True])
def test_periodic_convenience_full_cycle_and_custom_dims(as_dataset):
    grid, _, _ = seam_inputs()
    grid = grid.rename(latitude="lat", longitude="lon").isel(lat=slice(None, None, -1))
    grid = grid.transpose("lon", "time", "lat")
    if as_dataset:
        grid = xr.Dataset({"value": grid, "scalar": xr.DataArray(7)}, attrs={"source": "test"})
    tlat, tlon = target_coords_from_resolution(grid.lat.values, grid.lon.values, .5, period=360)
    resampler = ghl.Resampler.compute(grid.lat.values, grid.lon.values, tlat, tlon, period=360, iterations=2)
    expected = ghl.resample_grid_with_matrix(grid, resampler, lat_dim="lat", lon_dim="lon")
    actual = ghl.resample_grid(grid, .5, period=360, iterations=2, lat_dim="lat", lon_dim="lon")
    xr.testing.assert_identical(actual, expected)
    assert actual.lon[-1] == 179.5
    assert actual.sizes["lon"] == 720
    geoms = gpd.GeoSeries([shapely.box(179.3, .1, 179.7, .7)], index=["seam"])
    stencil = ghl.Stencil.compute(tlat, tlon, geoms)
    expected_reduce = ghl.reduce_with_stencil(grid, stencil, period=360, resample_iterations=2,
                                            lat_dim="lat", lon_dim="lon")
    actual_reduce = ghl.reduce(grid, geoms, period=360, target_resolution=.5, resample_iterations=2,
                               lat_dim="lat", lon_dim="lon")
    xr.testing.assert_identical(actual_reduce, expected_reduce)


@pytest.mark.parametrize("dask", [False, True])
@pytest.mark.parametrize("descending", [False, True])
def test_periodic_zarr_reads_only_seam_chunks(dask, descending):
    grid, stencil, _ = seam_inputs()
    grid = grid.drop_vars("model")
    if descending:
        grid = grid.isel(latitude=slice(None, None, -1))
    operator = ghl.ReduceOperator.compute(stencil, grid.latitude.values, grid.longitude.values,
                                         iterations=3, period=360)
    store = CountingStore(MemoryStore())
    grid.to_dataset().to_zarr(store, encoding={"value": {"chunks": (1, 1, 30)}},
                             zarr_format=3, consolidated=False)
    with xr.open_zarr(store, chunks={} if dask else None, consolidated=False) as ds:
        store.reads.clear()
        plan = ghl.RestrictedOperator.from_grid(operator, ds.value)
        assert not any(key.startswith("value/c/") for key in store.reads)
        actual = ghl.reduce_with_restricted_operator(ds.value, plan)
        xr.testing.assert_identical(actual, ghl.reduce_with_operator(grid, operator))
        rows, cols = np.divmod(operator.matrix.indices, grid.sizes["longitude"])
        if descending:
            rows = grid.sizes["latitude"] - 1 - rows
        touched = set(zip(rows, cols // 30, strict=True))
        assert {col for _, col in touched} == {0, 5}
        expected_reads = Counter(f"value/c/{t}/{r}/{c}" for t in range(2) for r, c in touched)
        assert Counter(key for key in store.reads if key.startswith("value/c/")) == expected_reads


@pytest.mark.parametrize("missing", [False, True])
def test_invalid_period_rejected_even_when_grids_match(missing, tmp_path):
    grid, _, _ = seam_inputs()
    if missing:
        grid.values[0, 0, 0] = np.nan
    geoms = gpd.GeoSeries([shapely.box(0., 0., 2., 1.)])
    lat, lon = grid.latitude.values, grid.longitude.values
    stencil = ghl.Stencil.compute(lat, lon, geoms)
    cache = ghl.LocalCache(tmp_path)
    calls = [
        lambda period: ghl.ReduceOperator.compute(stencil, lat, lon, period=period),
        lambda period: ghl.reduce_with_stencil(grid, stencil, period=period),
        lambda period: ghl.reduce(grid, geoms, period=period),
        lambda period: ghl.resample_grid(grid, .5, period=period),
        lambda period: cache.get_or_compute_resampler(lat, lon, lat, lon, period=period),
        lambda period: cache.get_or_compute_reduce_operator(stencil, lat, lon, period=period),
    ]
    for call in calls:
        for period in (0, np.nan, 358):  # 358 repeats the first source centre at the last one
            with pytest.raises(ValueError, match="period"):
                call(period)
