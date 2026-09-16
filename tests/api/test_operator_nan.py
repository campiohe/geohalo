"""Source-cell NaN masking for reusable full-grid and restricted operators."""

import tracemalloc
from dataclasses import replace

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp
import shapely
import xarray as xr

from geohalo import (
    LocalCache,
    ReduceOperator,
    RestrictedOperator,
    Stencil,
    reduce_with_operator,
    reduce_with_restricted_operator,
    reduce_with_stencil,
)
from tests.restricted_operator._helpers import make_grid, make_operator


def _array_apply(plan, values, how="mean"):
    gathered = plan.gather(values[..., rows, cols] for rows, cols in plan.windows)
    return plan.apply(gathered, how=how, skipna=True)


def _reference(operator, canonical, how):
    """Explicit per-zone weighted sums, independent of the projection helpers."""
    out = np.empty((*canonical.shape[:-2], len(operator.keys)))
    for batch in np.ndindex(canonical.shape[:-2]):
        flat = canonical[batch].ravel()
        for row in range(operator.matrix.shape[0]):
            weights = operator.matrix.getrow(row)
            values = flat[weights.indices]
            valid = ~np.isnan(values)
            total = np.sum(weights.data[valid] * values[valid])
            found = np.sum(weights.data[valid])
            out[(*batch, row)] = total if how == "sum" else total / found if found > 0 else np.nan
    return out


def test_issue_16_constant_field_reproducer_and_cache_reuse(tmp_path, monkeypatch):
    lat, lon = np.linspace(10, -10, 21), np.linspace(-10, 10, 21)
    zones = gpd.GeoSeries([shapely.box(-6, -6, -2, -2)], index=["one"], crs="EPSG:4326")
    stencil = Stencil.compute(lat, lon, zones)
    cache = LocalCache(tmp_path)
    operator = cache.get_or_compute_reduce_operator(stencil, lat, lon)
    plan = cache.get_or_compute_restricted_operator(operator, lat, 8, 8)
    values = np.full((1, 21, 21), 4.0, dtype=np.float32)
    values[:, 13:15, 5:7] = np.nan
    grid = xr.DataArray(values, dims=("step", "latitude", "longitude"),
                        coords={"latitude": lat, "longitude": lon})

    def no_build(*_args, **_kwargs):
        pytest.fail("applying NaN-aware cached operators must not rebuild them")

    monkeypatch.setattr(ReduceOperator, "compute", no_build)
    monkeypatch.setattr(RestrictedOperator, "compute", no_build)
    assert cache.get_or_compute_reduce_operator(stencil, lat, lon).digest == operator.digest
    assert cache.get_or_compute_restricted_operator(operator, lat, 8, 8).digest == plan.digest
    assert np.isnan(reduce_with_operator(grid, operator)).all()
    assert np.isnan(reduce_with_restricted_operator(grid, plan)).all()
    for how in ("mean", "sum"):
        expected = reduce_with_stencil(grid, stencil, how=how)
        actual = reduce_with_operator(grid, operator, how=how, skipna=True)
        xr.testing.assert_allclose(actual, expected)
        xr.testing.assert_identical(reduce_with_restricted_operator(grid, plan, how=how, skipna=True), actual)
        np.testing.assert_array_equal(_array_apply(plan, values, how), actual.values)
    np.testing.assert_allclose(reduce_with_operator(grid, operator, skipna=True).values, [[4.0]])


@pytest.mark.parametrize("shape", [(8, 10), (101, 103)])
@pytest.mark.parametrize("descending", [False, True])
@pytest.mark.parametrize("layout", ["contiguous", "strided", "fortran"])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("how", ["mean", "sum"])
def test_source_mask_reference_and_layouts(shape, descending, layout, dtype, how):
    operator = make_operator(n_lat=shape[0], n_lon=shape[1])
    operator = replace(operator, row_sums=np.asarray(operator.matrix.sum(axis=1)).ravel())
    values = np.arange(6 * np.prod(shape), dtype=dtype).reshape(2, 3, *shape) + 1
    values[0, 0].flat[1] = np.nan
    values[0, 1].flat[65] = np.nan
    values[0, 2] = np.nan
    values[1, 0].flat[19] = np.nan  # Only a negative coefficient survives for zone2.
    values[1, 2].flat[79] = np.nan  # Outside every polygon.
    expected = _reference(operator, values, how)
    lat = operator.source_lat
    if descending:
        values, lat = values[..., ::-1, :], lat[::-1]
    if layout == "contiguous":
        values = np.ascontiguousarray(values)
    elif layout == "strided":
        values = np.ascontiguousarray(values.transpose(2, 0, 3, 1)).transpose(1, 3, 0, 2)
    else:
        values = np.asfortranarray(values)
    grid = xr.DataArray(values, dims=("member", "step", "latitude", "longitude"),
                        coords={"latitude": lat, "longitude": operator.source_lon})
    plan = RestrictedOperator.compute(operator, lat, 3, 4)
    snapshot = values.copy()
    for actual in (
        reduce_with_operator(grid, operator, how=how, skipna=True).to_numpy(),
        reduce_with_restricted_operator(grid, plan, how=how, skipna=True).to_numpy(),
        _array_apply(plan, values, how),
        operator.apply_grid(values, descending=descending, how=how, skipna=True),
    ):
        np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)
        assert actual.dtype == np.float64
    np.testing.assert_array_equal(values, snapshot)


@pytest.mark.parametrize("dtype", [np.float32, np.float64, np.int16])
@pytest.mark.parametrize("batch_shape", [(), (2, 3), (0, 2)])
def test_clean_slices_are_identical_and_empty_batches_work(dtype, batch_shape):
    operator = make_operator()
    grid = make_grid(operator, descending=True, batch_shape=batch_shape, dtype=dtype)
    plan = RestrictedOperator.compute(operator, grid.latitude.values, 2, 2)
    for how in ("mean", "sum"):
        expected = reduce_with_operator(grid, operator, how=how)
        xr.testing.assert_identical(reduce_with_operator(grid, operator, how=how, skipna=True), expected)
        xr.testing.assert_identical(reduce_with_restricted_operator(grid, plan, how=how, skipna=True), expected)
        np.testing.assert_array_equal(_array_apply(plan, grid.values, how), expected.values, strict=True)


@pytest.mark.parametrize("how", ["mean", "sum"])
def test_dataset_custom_dims_and_metadata(how):
    operator = make_operator(multiindex=True)
    operator = replace(operator, row_sums=np.asarray(operator.matrix.sum(axis=1)).ravel())
    grid = make_grid(operator, descending=True).rename(latitude="lat", longitude="lon")
    grid.to_numpy()[..., 7, 1] = np.nan
    ds = xr.Dataset({"dirty": grid, "clean": grid.fillna(42), "scalar": 9, "lat_only": ("lat", np.arange(8))},
                    attrs={"model": "test"})
    plan = RestrictedOperator.compute(operator, grid.lat.values, 2, 2)
    options = {"how": how, "skipna": True, "lat_dim": "lat", "lon_dim": "lon", "geom_dim": "polygon"}
    actual = reduce_with_operator(ds, operator, **options)
    xr.testing.assert_identical(reduce_with_restricted_operator(ds, plan, **options), actual)
    xr.testing.assert_identical(actual.dirty, reduce_with_operator(ds.dirty, operator, **options))
    assert actual.attrs == ds.attrs
    assert actual.dirty.attrs == grid.attrs
    xr.testing.assert_identical(actual.scalar, ds.scalar)
    xr.testing.assert_identical(actual.lat_only, ds.lat_only)
    assert isinstance(actual.indexes["polygon"], pd.MultiIndex)
    xr.testing.assert_identical(actual.valid_time, grid.valid_time)
    xr.testing.assert_identical(actual.model, grid.model)


@pytest.mark.parametrize("iterations", [1, 2, 3])
def test_fused_source_mask_is_explicitly_not_target_mask(iterations):
    lat, lon = np.arange(8.), np.arange(10.)
    geoms = gpd.GeoSeries([shapely.box(1.7, 1.6, 2.2, 2.3), shapely.box(7.8, 5.2, 8.2, 5.8)], index=["a", "b"])
    stencil = Stencil.compute(np.linspace(0, 7, 16), np.linspace(0, 9, 20), geoms)
    operator = ReduceOperator.compute(stencil, lat, lon, iterations=iterations)
    assert np.any(operator.matrix.data < 0)
    values = np.arange(80.).reshape(8, 10)
    values[1, 1] = np.nan
    grid = xr.DataArray(values, dims=("latitude", "longitude"), coords={"latitude": lat, "longitude": lon})
    plan = RestrictedOperator.compute(operator, lat, 2, 2)
    for how in ("mean", "sum"):
        expected = _reference(operator, values, how)
        actual = reduce_with_operator(grid, operator, how=how, skipna=True)
        np.testing.assert_allclose(actual.values, expected, rtol=1e-12)
        xr.testing.assert_identical(reduce_with_restricted_operator(grid, plan, how=how, skipna=True), actual)
    assert np.isnan(reduce_with_stencil(grid, stencil, resample_iterations=iterations).to_numpy()[0])
    assert np.isfinite(reduce_with_operator(grid, operator, skipna=True).to_numpy()[0])


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_signed_cancellation_nonpositive_denominators_and_coefficient_order(dtype):
    matrix = sp.csr_matrix(
        (np.array([1e16, -1e16, 1, 2, 1, -1, 2, -1, 2], dtype=dtype),
         [65, 1, 3, 4, 1, 3, 4, 1, 4], [0, 4, 7, 9]), shape=(3, 80),
    )
    operator = replace(make_operator(matrix), row_sums=np.array([3., 2., 1.]))
    grid = xr.ones_like(make_grid(operator, dtype=dtype))
    grid.to_numpy()[..., 0, 4] = np.nan
    plan = RestrictedOperator.compute(operator, operator.source_lat, 2, 2)
    for how, expected in (("mean", [1., np.nan, np.nan]), ("sum", [1., 0., -1.])):
        actual = reduce_with_operator(grid, operator, how=how, skipna=True)
        np.testing.assert_array_equal(actual.values, np.broadcast_to(expected, actual.shape))
        xr.testing.assert_identical(reduce_with_restricted_operator(grid, plan, how=how, skipna=True), actual)


def test_only_nan_is_missing_not_infinity():
    operator = make_operator(sp.csr_matrix(([1., 1.], ([0, 0], [0, 1])), shape=(1, 80)))
    grid = make_grid(operator, batch_shape=())
    grid.to_numpy()[0, :2] = [np.inf, np.nan]
    plan = RestrictedOperator.compute(operator, operator.source_lat, 2, 2)
    for how in ("mean", "sum"):
        assert np.isposinf(reduce_with_operator(grid, operator, how=how, skipna=True).item())
        assert np.isposinf(_array_apply(plan, grid.values, how).item())


@pytest.mark.parametrize("dtype", [np.float32, np.float64, np.int32])
def test_custom_operator_normalization_keeps_numpy_dtype_promotion(dtype):
    operator = make_operator(sp.csr_matrix(np.ones((2, 80), dtype=dtype)))
    operator = replace(operator, row_sums=np.full(2, 80, dtype=dtype))
    grid = make_grid(operator, dtype=dtype)
    plan = RestrictedOperator.compute(operator, operator.source_lat, 2, 2)
    expected = reduce_with_operator(grid, operator)
    xr.testing.assert_identical(reduce_with_operator(grid, operator, skipna=True), expected)
    xr.testing.assert_identical(reduce_with_restricted_operator(grid, plan, skipna=True), expected)
    if dtype != np.int32:
        grid.to_numpy()[..., 0, 0] = np.nan
        expected_values = grid.to_numpy()[..., 0, 1:].sum(axis=-1) + grid.to_numpy()[..., 1:, :].sum(axis=(-2, -1))
        expected_values = expected_values / 79
        actual = reduce_with_operator(grid, operator, skipna=True)
        assert actual.dtype == expected.dtype
        np.testing.assert_array_equal(actual.values, np.repeat(expected_values[..., None], 2, axis=-1))


@pytest.mark.parametrize("batch_shape", [(), (3,), (0, 2)])
def test_no_support_means_nan_sums_zero_without_reading(batch_shape, monkeypatch):
    operator = replace(make_operator(sp.csr_matrix((3, 80))), row_sums=np.zeros(3))
    grid = make_grid(operator, batch_shape=batch_shape)
    plan = RestrictedOperator.compute(operator, operator.source_lat, 2, 2)
    original = xr.DataArray.to_numpy

    def no_values(array):
        if "latitude" in array.dims and "longitude" in array.dims:
            pytest.fail("empty restricted plan must not read source values")
        return original(array)

    monkeypatch.setattr(xr.DataArray, "to_numpy", no_values)
    for how in ("mean", "sum"):
        expected = np.full((*batch_shape, 3), np.nan if how == "mean" else 0.)
        actual = reduce_with_restricted_operator(grid, plan, how=how, skipna=True)
        np.testing.assert_array_equal(actual.values, expected)
        np.testing.assert_array_equal(plan.apply(np.empty((*batch_shape, 0)), how=how, skipna=True), expected)


@pytest.mark.parametrize("how", ["mean", "sum"])
def test_only_dirty_means_need_a_second_product(how, monkeypatch):
    operator = make_operator()
    grid = make_grid(operator, batch_shape=(3,))
    grid.to_numpy()[1, 0, 1] = np.nan
    plan = RestrictedOperator.compute(operator, operator.source_lat, 2, 2)
    gathered = plan.gather(grid.to_numpy()[..., r, c] for r, c in plan.windows)
    matmul = sp.csr_matrix.__matmul__
    calls = []

    def record(matrix, values):
        calls.append(values.shape)
        return matmul(matrix, values)

    monkeypatch.setattr(sp.csr_matrix, "__matmul__", record)
    plan.apply(gathered, how=how, skipna=True)
    assert len(calls) == (4 if how == "mean" else 3)


@pytest.mark.parametrize("descending", [False, True])
def test_nan_masks_do_not_allocate_a_full_batch_or_scan_unrelated_large_cells(descending):
    operator = make_operator(n_lat=721, n_lon=1440)
    operator = replace(operator, row_sums=np.asarray(operator.matrix.sum(axis=1)).ravel())
    values = np.full((24, 721, 1440), 4., dtype=np.float32)
    values[:, 0, 1] = np.nan
    if descending:
        values = values[:, ::-1]
    grid = xr.DataArray(values, dims=("step", "latitude", "longitude"),
                        coords={"latitude": operator.source_lat[::-1] if descending else operator.source_lat,
                                "longitude": operator.source_lon})
    for _ in range(2):
        tracemalloc.start()
        try:
            actual = reduce_with_operator(grid, operator, skipna=True)
            peak = tracemalloc.get_traced_memory()[1]
        finally:
            tracemalloc.stop()
        np.testing.assert_allclose(actual.values, 4., rtol=1e-12)
        assert peak < values[0].nbytes


def test_apply_grid_rejects_invalid_reduction():
    operator = make_operator()
    with pytest.raises(ValueError, match="how must"):
        operator.apply_grid(np.ones((8, 10)), how="median", skipna=True)


@pytest.mark.parametrize("descending", [False, True])
def test_masked_large_full_coverage_operator_uses_bounded_slices(descending):
    shape = (101, 103)
    operator = make_operator(sp.csr_matrix(np.ones((2, np.prod(shape)))), n_lat=shape[0], n_lon=shape[1])
    operator = replace(operator, row_sums=np.full(2, np.prod(shape), dtype=float))
    values = np.ones((100, *shape), dtype=np.float32)
    values[:, :5] = np.nan
    if descending:
        values = values[:, ::-1]
    for _ in range(2):
        tracemalloc.start()
        try:
            actual = operator.apply_grid(values, descending=descending, how="mean", skipna=True)
            peak = tracemalloc.get_traced_memory()[1]
        finally:
            tracemalloc.stop()
        np.testing.assert_array_equal(actual, np.ones((100, 2)))
        assert peak < values.nbytes
