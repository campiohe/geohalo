"""Opt-in coefficient storage and result dtypes, without changing default precision."""

import hashlib
import tracemalloc
from dataclasses import replace

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp
import shapely
import xarray as xr

import geohalo as ghl
from geohalo.geometry import geom_digest, grid_digest
from geohalo.reduce_operator import reduce_operator_digest
from geohalo.stencil import stencil_digest
from tests.restricted_operator._helpers import make_grid, make_operator


def _geometry():
    lat, lon = np.arange(8.), np.arange(10.)
    geoms = gpd.GeoSeries(
        [shapely.box(1.7, 1.6, 2.2, 2.3), shapely.box(5.8, 4.2, 8.2, 5.8)], index=["zulu", "alpha"],
    )
    return lat, lon, geoms


@pytest.mark.parametrize("dtype", [np.float32, "float32", np.dtype(">f4"), np.float64, float, "float64", None])
@pytest.mark.parametrize("refined", [False, True])
def test_builder_dtype_aliases_coefficients_and_digest(dtype, refined):
    lat, lon, geoms = _geometry()
    target_lat, target_lon = (np.linspace(0, 7, 16), np.linspace(0, 9, 20)) if refined else (lat, lon)
    expected_dtype = np.dtype(dtype).newbyteorder("=")
    reference = ghl.Stencil.compute(target_lat, target_lon, geoms)
    stencil = ghl.Stencil.compute(target_lat[::-1], target_lon, geoms, dtype=dtype)
    assert stencil.occupancy_matrix.dtype == expected_dtype
    assert stencil.row_sums.dtype == np.float64
    assert stencil.lats.dtype == stencil.lons.dtype == np.float64
    pd.testing.assert_index_equal(stencil.keys, geoms.index)
    np.testing.assert_array_equal(stencil.occupancy_matrix.data, reference.occupancy_matrix.data.astype(expected_dtype))
    assert stencil.digest == stencil_digest(target_lat, target_lon, geoms, dtype=dtype)
    assert stencil.digest == stencil_digest(target_lat, target_lon, geoms.iloc[::-1], dtype=expected_dtype)
    assert (stencil.digest == reference.digest) == (expected_dtype == np.float64)

    # Use the same stencil to isolate the operator's coefficient cast.
    baseline = ghl.ReduceOperator.compute(stencil, lat, lon, iterations=2)
    operator = ghl.ReduceOperator.compute(stencil, lat[::-1], lon, iterations=2, dtype=dtype)
    assert operator.matrix.dtype == expected_dtype
    assert operator.row_sums.dtype == np.float64
    np.testing.assert_array_equal(operator.matrix.data, baseline.matrix.data.astype(expected_dtype))
    np.testing.assert_array_equal(operator.matrix.indices, baseline.matrix.indices)
    np.testing.assert_array_equal(operator.row_sums, stencil.row_sums)
    assert operator.digest == reduce_operator_digest(stencil.digest, lat, lon, 2, dtype=expected_dtype)
    assert (operator.digest == baseline.digest) == (expected_dtype == np.float64)
    plan = ghl.RestrictedOperator.compute(operator, lat[::-1], 3, 4)
    assert plan.matrix.dtype == expected_dtype
    np.testing.assert_array_equal(plan.row_sums, operator.row_sums)
    if expected_dtype == np.float32:
        assert operator.matrix.data.nbytes * 2 == baseline.matrix.data.nbytes


def test_default_digest_bytes_are_unchanged():
    lat, lon, geoms = _geometry()
    old_stencil = hashlib.sha256(grid_digest(lat, lon) + b"sph" + geom_digest(geoms)).digest()
    old_operator = hashlib.sha256(old_stencil + grid_digest(lat, lon) + b"2").digest()
    assert stencil_digest(lat, lon, geoms) == old_stencil
    assert stencil_digest(lat, lon, geoms, dtype="float64") == old_stencil
    assert reduce_operator_digest(old_stencil, lat, lon, 2) == old_operator
    assert reduce_operator_digest(old_stencil, lat, lon, 2, dtype="float64") == old_operator


@pytest.mark.parametrize("dtype", [np.float16, np.int64, np.complex64, bool, object, "datetime64[ns]", "garbage"])
def test_unsupported_coefficient_dtypes_rejected(dtype, tmp_path):
    lat, lon, geoms = _geometry()
    stencil = ghl.Stencil.compute(lat, lon, geoms)
    cache = ghl.LocalCache(tmp_path)
    for build in (
        lambda: ghl.Stencil.compute(lat, lon, geoms, dtype=dtype),
        lambda: ghl.ReduceOperator.compute(stencil, lat, lon, dtype=dtype),
        lambda: cache.get_or_compute_stencil(lat, lon, geoms, dtype=dtype),
        lambda: cache.get_or_compute_reduce_operator(stencil, lat, lon, dtype=dtype),
    ):
        with pytest.raises((ValueError, TypeError), match=r"dtype|data type"):
            build()


@pytest.mark.parametrize("empty", [False, True])
def test_float32_stencil_sums_accumulate_in_float64(empty):
    matrix = sp.csr_matrix((3, 3), dtype=np.float32) if empty else sp.csr_matrix(
        (np.array([1e8, 1, 1], dtype=np.float32), [2, 0, 1], [0, 0, 3, 3]), shape=(3, 3),
    )
    stencil = ghl.Stencil(matrix, pd.Index(["empty", "values", "empty2"]), np.array([0.]), np.arange(3.), b"test")
    np.testing.assert_array_equal(stencil.row_sums, [0, 0 if empty else 100000002., 0])
    assert stencil.row_sums.dtype == np.float64


@pytest.fixture(params=["local", pytest.param("redis", marks=pytest.mark.redis)])
def cache(request, tmp_path):
    if request.param == "local":
        return ghl.LocalCache(tmp_path)
    return ghl.RedisCache(request.getfixturevalue("redis_client"))


def test_cache_dtype_isolation_aliases_row_order_and_skipped_builds(cache, monkeypatch):
    lat, lon, geoms = _geometry()
    built = {}
    for stencil_dtype in (np.float32, np.float64):
        stencil = cache.get_or_compute_stencil(lat, lon, geoms, dtype=stencil_dtype)
        for operator_dtype in (np.float32, np.float64):
            operator = cache.get_or_compute_reduce_operator(stencil, lat, lon, dtype=operator_dtype)
            plan = cache.get_or_compute_restricted_operator(operator, lat[::-1], 3, 4)
            built[stencil_dtype, operator_dtype] = stencil, operator, plan
    assert len({objects[1].digest for objects in built.values()}) == 4
    assert len({objects[2].digest for objects in built.values()}) == 4

    def no_build(*_args, **_kwargs):
        pytest.fail("dtype aliases and row-order changes must reuse cached builds")

    for cls in (ghl.Stencil, ghl.ReduceOperator, ghl.RestrictedOperator):
        monkeypatch.setattr(cls, "compute", no_build)
    for (stencil_dtype, operator_dtype), expected in built.items():
        stencil = cache.get_or_compute_stencil(lat, lon, geoms.iloc[::-1], dtype=np.dtype(stencil_dtype).name)
        operator = cache.get_or_compute_reduce_operator(stencil, lat, lon, dtype=np.dtype(operator_dtype).name)
        plan = cache.get_or_compute_restricted_operator(operator, lat[::-1], 3, 4)
        for actual, reference in zip((stencil, operator, plan), expected, strict=True):
            pd.testing.assert_index_equal(actual.keys, geoms.iloc[::-1].index)
            assert actual.digest == reference.digest
            actual_matrix = actual.occupancy_matrix if isinstance(actual, ghl.Stencil) else actual.matrix
            reference_matrix = reference.occupancy_matrix if isinstance(reference, ghl.Stencil) else reference.matrix
            assert actual_matrix.dtype == reference_matrix.dtype
            np.testing.assert_array_equal(actual_matrix.toarray(), reference_matrix.toarray()[::-1])
            np.testing.assert_array_equal(actual.row_sums, reference.row_sums[::-1])
        grid = make_grid(operator, descending=True, dtype=np.float32)
        expected_result = ghl.reduce_with_operator(grid, operator, preserve_dtype=True)
        assert expected_result.dtype == np.float32
        xr.testing.assert_identical(
            ghl.reduce_with_restricted_operator(grid, plan, preserve_dtype=True), expected_result,
        )


@pytest.mark.parametrize("matrix_dtype", [np.float32, np.float64])
@pytest.mark.parametrize("value_dtype", [np.float16, np.float32, np.float64, np.int16, np.bool_])
@pytest.mark.parametrize("how", ["mean", "sum"])
@pytest.mark.parametrize("skipna", [False, True])
@pytest.mark.parametrize("batch_shape", [(), (2, 3), (0, 2)])
def test_preserve_result_dtype_across_prebuilt_apis(matrix_dtype, value_dtype, how, skipna, batch_shape):
    # Synthetic small weights also exercise float16 results without area-sum overflow.
    operator = make_operator()
    operator = replace(operator, matrix=operator.matrix.astype(matrix_dtype))
    grid = make_grid(operator, descending=True, batch_shape=batch_shape, dtype=value_dtype)
    if skipna and grid.size and grid.dtype.kind == "f":
        grid.to_numpy()[..., 7, 1] = np.nan
    plan = ghl.RestrictedOperator.compute(operator, grid.latitude.to_numpy(), 3, 4)
    reference = ghl.reduce_with_operator(grid, operator, how=how, skipna=skipna)
    dtype = grid.dtype if grid.dtype.kind == "f" else reference.dtype
    expected = reference.astype(dtype)
    options = {"how": how, "skipna": skipna, "preserve_dtype": True}
    xr.testing.assert_identical(ghl.reduce_with_operator(grid, operator, **options), expected)
    xr.testing.assert_identical(ghl.reduce_with_restricted_operator(grid, plan, **options), expected)
    gathered = plan.gather(grid.to_numpy()[..., r, c] for r, c in plan.windows)
    for actual in (
        plan.apply(gathered, **options),
        operator.apply_grid(grid.to_numpy(), descending=True, **options),
    ):
        np.testing.assert_array_equal(actual, expected.to_numpy(), strict=True)


@pytest.mark.parametrize("refined", [False, True])
@pytest.mark.parametrize("how", ["mean", "sum"])
@pytest.mark.parametrize("masked", [False, True])
@pytest.mark.parametrize("weighted", [False, True])
def test_preserve_dtype_convenience_and_stencil_paths(refined, how, masked, weighted):
    lat, lon, geoms = _geometry()
    base = ghl.ReduceOperator.compute(ghl.Stencil.compute(lat, lon, geoms), lat, lon)
    grid = make_grid(base, descending=True, dtype=np.float32)
    if masked:
        grid.to_numpy()[..., 6, 1] = np.nan
    if weighted:
        grid = grid.assign_coords(weight=(grid.dims, np.ones(grid.shape, dtype=np.float32)))
    options = {"how": how, "weight_key": "weight" if weighted else None}
    target_lat, target_lon = (np.linspace(0, 7, 15), np.linspace(0, 9, 19)) if refined else (lat, lon)
    for dtype in (np.float32, np.float64):
        stencil = ghl.Stencil.compute(target_lat, target_lon, geoms, dtype=dtype)
        reference = ghl.reduce_with_stencil(grid, stencil, **options)
        actual = ghl.reduce_with_stencil(grid, stencil, preserve_dtype=True, **options)
        xr.testing.assert_allclose(actual, reference.astype(np.float32))
        assert actual.dtype == np.float32
    options["target_resolution"] = 0.5 if refined else None
    reference = ghl.reduce(grid, geoms, **options)
    actual = ghl.reduce(grid, geoms, preserve_dtype=True, **options)
    xr.testing.assert_allclose(actual, reference.astype(np.float32))
    assert actual.dtype == np.float32


@pytest.mark.parametrize("missing", [False, True])
def test_dataset_preserves_each_variable_dtype_and_metadata(missing):
    lat, lon, geoms = _geometry()
    stencil = ghl.Stencil.compute(lat, lon, geoms, dtype=np.float32)
    operator = ghl.ReduceOperator.compute(stencil, lat, lon, dtype=np.float32)
    grid = make_grid(operator, descending=True, dtype=np.float32)
    if missing:
        grid.to_numpy()[..., 6, 1] = np.nan
    ds = xr.Dataset({"single": grid, "double": grid.astype(np.float64), "integer": grid.fillna(0).astype(np.int16),
                     "unrelated": ("dim0", np.arange(3, dtype=np.int8))}, attrs={"test": "dtype"})
    plan = ghl.RestrictedOperator.compute(operator, grid.latitude.to_numpy(), 3, 4)
    for result in (
        ghl.reduce_with_operator(ds, operator, skipna=True, preserve_dtype=True),
        ghl.reduce_with_restricted_operator(ds, plan, skipna=True, preserve_dtype=True),
        ghl.reduce_with_stencil(ds, stencil, preserve_dtype=True),
        ghl.reduce(ds, geoms, preserve_dtype=True),
    ):
        assert result.single.dtype == np.float32
        assert result.double.dtype == result.integer.dtype == np.float64
        assert result.attrs == ds.attrs
        assert result.single.attrs == ds.single.attrs
        xr.testing.assert_identical(result.unrelated, ds.unrelated)
        xr.testing.assert_identical(result.valid_time, ds.valid_time)


def test_build_cast_keeps_coefficient_order_and_duplicate_entries():
    matrix = sp.csr_matrix(([1e16, -1e16, 1., 2.], [2, 0, 1, 1], [0, 4]), shape=(1, 4))
    stencil = ghl.Stencil(matrix, pd.Index(["zone"]), np.arange(2.), np.arange(2.), b"custom")
    operator = ghl.ReduceOperator.compute(stencil, stencil.lats, stencil.lons, dtype=np.float32)
    np.testing.assert_array_equal(operator.matrix.indices, matrix.indices)
    np.testing.assert_array_equal(operator.matrix.data, matrix.data.astype(np.float32))
    np.testing.assert_array_equal(operator.apply_grid(np.ones((2, 2), dtype=np.float32)), [3.])


@pytest.mark.parametrize("refined", [False, True])
@pytest.mark.parametrize("skipna", [False, True])
def test_float32_accuracy_and_output_size_on_real_geometry(refined, skipna):
    lat, lon, geoms = _geometry()
    target_lat, target_lon = (np.linspace(0, 7, 16), np.linspace(0, 9, 20)) if refined else (lat, lon)
    stencil = ghl.Stencil.compute(target_lat, target_lon, geoms)
    stencil32 = ghl.Stencil.compute(target_lat, target_lon, geoms, dtype=np.float32)
    reference_operator = ghl.ReduceOperator.compute(stencil, lat, lon, iterations=3)
    operator = ghl.ReduceOperator.compute(stencil32, lat, lon, iterations=3, dtype=np.float32)
    grid = make_grid(operator, dtype=np.float32)
    grid.to_numpy()[...] = np.random.default_rng(123).uniform(250, 320, grid.shape).astype(np.float32)
    if skipna:
        grid.to_numpy()[..., 1, 1] = np.nan
    for how in ("mean", "sum"):
        expected = ghl.reduce_with_operator(grid, reference_operator, how=how, skipna=skipna)
        actual = ghl.reduce_with_operator(grid, operator, how=how, skipna=skipna, preserve_dtype=True)
        assert actual.dtype == np.float32
        assert actual.nbytes * 2 == expected.nbytes
        np.testing.assert_allclose(actual.to_numpy(), expected.to_numpy(), rtol=1e-5)


@pytest.mark.parametrize("batch_shape", [(), (3,), (0, 2)])
@pytest.mark.parametrize("how", ["mean", "sum"])
def test_empty_support_preserves_dtype_without_reads(batch_shape, how, monkeypatch):
    operator = replace(make_operator(sp.csr_matrix((3, 80), dtype=np.float32)), row_sums=np.zeros(3))
    grid = make_grid(operator, batch_shape=batch_shape, dtype=np.float32)
    plan = ghl.RestrictedOperator.compute(operator, operator.source_lat, 2, 2)
    to_numpy = xr.DataArray.to_numpy

    def no_data(array):
        if "latitude" in array.dims and "longitude" in array.dims:
            pytest.fail("an empty plan must not read grid data")
        return to_numpy(array)

    monkeypatch.setattr(xr.DataArray, "to_numpy", no_data)
    expected = np.full((*batch_shape, 3), np.nan if how == "mean" else 0, dtype=np.float32)
    actual = ghl.reduce_with_restricted_operator(grid, plan, how=how, skipna=True, preserve_dtype=True)
    np.testing.assert_array_equal(actual.to_numpy(), expected, strict=True)
    np.testing.assert_array_equal(
        plan.apply(np.empty((*batch_shape, 0), dtype=np.float32), how=how, skipna=True, preserve_dtype=True),
        expected, strict=True,
    )


@pytest.mark.parametrize("skipna", [False, True])
def test_large_strided_float32_result_has_bounded_temporaries(skipna):
    operator = make_operator(n_lat=721, n_lon=1440)
    grid = make_grid(operator, descending=True, batch_shape=(24,), dtype=np.float32)
    if skipna:
        grid.to_numpy()[..., -1, 1] = np.nan
    # Cold and warm paths must not cast the full global batch to float64.
    for _ in range(2):
        tracemalloc.start()
        try:
            result = ghl.reduce_with_operator(grid, operator, skipna=skipna, preserve_dtype=True)
            peak = tracemalloc.get_traced_memory()[1]
        finally:
            tracemalloc.stop()
        assert result.dtype == np.float32
        assert peak < grid.to_numpy()[0].nbytes
