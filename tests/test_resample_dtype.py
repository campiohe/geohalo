"""Result dtype controls keep resampling arithmetic and memory bounds intact."""

import tracemalloc
from dataclasses import replace
from functools import partial

import numpy as np
import pytest
import scipy.sparse as sp
import xarray as xr

import geohalo as ghl
from tests.resampler.test_conservative import partial_operator


@pytest.fixture(params=[
    ("meanpreserving", "destination", False),
    ("conservative", "destination", False),
    ("conservative", "covered", False),
    ("conservative", "destination", True),
])
def resampling(request):
    method, normalization, skipna = request.param
    lat, lon = np.arange(-30., 40., 20.), np.arange(-135., 180., 90.)
    operator = ghl.Resampler.compute(
        lat, lon, np.linspace(70, -70, 5), np.arange(-180., 180., 45.),
        method=method, normalization=normalization, period=360,
        iterations=3 if method == "meanpreserving" else 1,
    )
    return operator, skipna


@pytest.mark.parametrize("value_dtype", [
    np.float16, np.float32, np.dtype(">f4"), np.float64, np.int16, np.bool_, np.complex64,
])
@pytest.mark.parametrize(("batch_shape", "layout", "descending"), [
    ((), "contiguous", False), ((2, 3), "strided", True),
    ((2,), "fortran", False), ((0, 2), "contiguous", True),
])
def test_result_dtype_matches_cast_after_resampling(resampling, value_dtype, batch_shape, layout, descending):
    operator, skipna = resampling
    values = np.random.default_rng(0).uniform(-3, 7, (*batch_shape, 4, 4)).astype(value_dtype)
    if values.dtype.kind == "c":
        values += 1j * values
    if values.dtype.kind in "fc":
        values[..., 1, 1] = np.nan
    if layout == "strided":
        values = values[..., ::-1]
    elif layout == "fortran":
        values = np.asfortranarray(values)
    reference = operator.apply_grid(values, descending=descending, skipna=skipna)
    assert reference.dtype == np.result_type(values.dtype, np.float64)
    dtype = values.dtype if values.dtype.kind == "f" else reference.dtype
    actual = operator.apply_grid(values, descending=descending, skipna=skipna, preserve_dtype=True)
    np.testing.assert_array_equal(actual, reference.astype(dtype), strict=True)


@pytest.mark.parametrize("builder", ["matrix", "resolution"])
def test_xarray_each_variable_dtype_metadata_and_orientation(resampling, builder):
    operator, skipna = resampling
    values = np.random.default_rng(0).uniform(250, 300, (3, 4, 4)).astype(np.float32)
    values[1, 1, 1] = np.nan
    values[2] = np.nan
    grid = xr.DataArray(
        values, dims=("step", "lat", "lon"),
        coords={"step": [1, 2, 3], "lat": operator.source_lat, "lon": operator.source_lon,
                "forecast": ("step", [4, 5, 6]), "model": "test"},
        name="single", attrs={"units": "K"},
    ).isel(lat=slice(None, None, -1)).transpose("lon", "step", "lat")
    source = xr.Dataset({
        "single": grid, "double": grid.astype(np.float64),
        "integer": grid.fillna(0).astype(np.int16),
        "unrelated": ("step", np.arange(3, dtype=np.int8)),
    }, attrs={"source": "dtype test"})
    options = {"lat_dim": "lat", "lon_dim": "lon", "skipna": skipna}
    if builder == "matrix":
        apply = partial(ghl.resample_grid_with_matrix, resampler=operator)
    else:
        apply = partial(
            ghl.resample_grid, target_resolution=10., method=operator.method,
            normalization=operator.normalization, period=360,
        )
    reference = apply(source, **options)
    expected = reference.assign(single=reference.single.astype(np.float32))
    actual = apply(source, preserve_dtype=True, **options)
    xr.testing.assert_identical(actual, expected)
    assert actual.single.dtype == np.float32
    assert actual.double.dtype == actual.integer.dtype == np.float64
    assert actual.attrs == source.attrs
    xr.testing.assert_identical(actual.unrelated, source.unrelated)
    xr.testing.assert_identical(actual.forecast, source.forecast)
    xr.testing.assert_identical(apply(grid, preserve_dtype=True, **options), expected.single)


def test_meanpreserving_keeps_float64_accumulation_before_result_cast():
    operator = ghl.Resampler(
        sp.csr_matrix([[1., 1., 1.]]), np.array([0.]), np.arange(3.),
        np.array([0.]), np.array([1.]), b"cancellation",
    )
    values = np.array([[1e8, 1., -1e8]], dtype=np.float32)
    # A float32 accumulation loses the middle term; the final cast must keep it.
    actual = operator.apply_grid(values, preserve_dtype=True)
    np.testing.assert_array_equal(actual, np.array([1.], dtype=np.float32), strict=True)
    assert operator.transform_matrix.dtype == np.float64


@pytest.mark.parametrize("skipna", [False, True])
def test_conservative_normalizes_tiny_coverage_before_casting(skipna):
    operator = partial_operator(normalization="covered")
    latitude, longitude = operator.axis_weights
    operator = replace(operator, axis_weights=(latitude, longitude * 1e-50))
    values = np.array([[[2., 4.]], [[2., np.nan]], [[np.nan, np.nan]]], dtype=np.float32)
    # Coverage would underflow in float32, although the normalized values fit.
    reference = operator.apply_grid(values, skipna=skipna).astype(np.float32)
    actual = operator.apply_grid(values, skipna=skipna, preserve_dtype=True)
    np.testing.assert_array_equal(actual, reference, strict=True)
    assert np.isfinite(actual).any()
    assert all(axis.dtype == np.float64 for axis in operator.axis_weights)


@pytest.mark.parametrize("method", ["meanpreserving", "conservative"])
def test_cached_and_serialized_resamplers_support_both_output_dtypes(method, tmp_path, monkeypatch):
    cache = ghl.LocalCache(tmp_path)
    coords, target = np.arange(3.), np.linspace(0, 2, 5)
    operator = cache.get_or_compute_resampler(coords, coords, target, target, method=method)
    values = np.arange(9, dtype=np.float32).reshape(3, 3)
    reference = operator.apply_grid(values)
    restored = ghl.Resampler.from_npz(operator.to_npz())

    def no_build(*_args, **_kwargs):
        pytest.fail("changing result dtype must reuse the cached resampler")

    monkeypatch.setattr(ghl.Resampler, "compute", no_build)
    cached = cache.get_or_compute_resampler(coords, coords, target, target, method=method)
    for resampler in (operator, restored, cached):
        assert resampler.digest == operator.digest
        for preserve_dtype in (True, False):
            actual = resampler.apply_grid(values, preserve_dtype=preserve_dtype)
            dtype = np.float32 if preserve_dtype else np.float64
            np.testing.assert_array_equal(actual, reference.astype(dtype), strict=True)


@pytest.mark.parametrize(("method", "skipna"), [
    ("meanpreserving", False), ("conservative", False), ("conservative", True),
])
def test_float32_output_does_not_allocate_a_full_float64_batch(method, skipna):
    lat, lon = np.linspace(-40, 40, 101), np.linspace(-80, 80, 103)
    operator = ghl.Resampler.compute(
        lat, lon, np.linspace(-40, 40, 201), np.linspace(-80, 80, 205), method=method,
    )
    values = np.ones((24, lat.size, lon.size), dtype=np.float32)[..., ::-1]
    if skipna:
        values[:, 50, 50] = np.nan
    source = xr.DataArray(values, dims=("step", "latitude", "longitude"),
                          coords={"latitude": lat, "longitude": lon})
    for _ in range(2):
        tracemalloc.start()
        try:
            actual = ghl.resample_grid_with_matrix(source, operator, skipna=skipna, preserve_dtype=True)
            peak = tracemalloc.get_traced_memory()[1]
        finally:
            tracemalloc.stop()
        assert actual.dtype == np.float32
        # Allow per-slice work arrays, but not a full float64 output then astype.
        assert peak < actual.nbytes + 12 * (values[0].nbytes + actual.to_numpy()[0].nbytes)
        np.testing.assert_allclose(actual.to_numpy()[np.isfinite(actual)], 1., rtol=1e-6)
