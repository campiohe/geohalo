from collections import Counter
from itertools import product

import dask.array as da
import geopandas as gpd
import numpy as np
import pytest
import scipy.sparse as sp
import shapely
import xarray as xr
from dask import delayed

from geohalo import (
    ReduceOperator,
    RestrictedOperator,
    Stencil,
    api,
    reduce_with_operator,
    reduce_with_restricted_operator,
)
from tests.restricted_operator._helpers import make_grid, make_operator


@pytest.mark.parametrize("descending", [False, True])
@pytest.mark.parametrize("how", ["mean", "sum"])
@pytest.mark.parametrize("batch_shape", [(), (5,), (3, 2), (0, 2)])
@pytest.mark.parametrize("multiindex", [False, True])
def test_application_matches_full_grid_and_metadata(descending, how, batch_shape, multiindex):
    operator = make_operator(multiindex=multiindex)
    grid = make_grid(operator, descending=descending, batch_shape=batch_shape)
    # Reorder all dims to exercise noncontiguous input and nonleading batch dims.
    grid = grid.transpose(*reversed(grid.dims))
    plan = RestrictedOperator.compute(operator, grid.latitude.values, (3, 1, 4), (2, 3, 1, 4))
    got = reduce_with_restricted_operator(grid, plan, how=how)
    expected = reduce_with_operator(grid, operator, how=how)
    xr.testing.assert_identical(got, expected)


@pytest.mark.parametrize("dtype", [np.float32, np.float64, np.int16])
def test_custom_dims_and_dtype(dtype):
    operator = make_operator()
    grid = make_grid(operator, dtype=dtype).rename(latitude="lat", longitude="lon")
    grid = grid.assign_coords(lat_label=("lat", np.arange(8)))
    plan = RestrictedOperator.compute(operator, operator.source_lat, 2, 2)
    got = reduce_with_restricted_operator(grid, plan, lat_dim="lat", lon_dim="lon", geom_dim="polygon")
    expected = reduce_with_operator(grid, operator, lat_dim="lat", lon_dim="lon", geom_dim="polygon")
    xr.testing.assert_identical(got, expected)
    assert "lat_label" not in got.coords


def test_dataset_preserves_passthrough_variables_and_attrs():
    operator = make_operator(multiindex=True)
    grid = make_grid(operator)
    ds = xr.Dataset(
        {"t2m": grid, "tp": grid + 1, "scalar": xr.DataArray(7), "lat_var": ("latitude", np.arange(8))},
        attrs={"source": "test"},
    )
    ds.tp.attrs = {"units": "mm"}
    plan = RestrictedOperator.compute(operator, operator.source_lat, 2, 2)
    xr.testing.assert_identical(
        reduce_with_restricted_operator(ds, plan), reduce_with_operator(ds, operator),
    )


@pytest.mark.parametrize("iterations", [1, 2, 3])
@pytest.mark.parametrize("how", ["mean", "sum"])
def test_real_refined_operator_includes_resampling_halo(iterations, how):
    lat, lon = np.arange(8.0), np.arange(10.0)
    geoms = gpd.GeoSeries([shapely.box(1.7, 1.6, 2.2, 2.3), shapely.box(7.8, 5.2, 8.2, 5.8)], index=["a", "b"])
    stencil = Stencil.compute(np.linspace(0, 7, 16), np.linspace(0, 9, 20), geoms)
    operator = ReduceOperator.compute(stencil, lat, lon, iterations=iterations)
    grid = make_grid(operator, descending=True)
    plan = RestrictedOperator.compute(operator, grid.latitude.values, 2, 2)
    xr.testing.assert_allclose(
        reduce_with_restricted_operator(grid, plan, how=how), reduce_with_operator(grid, operator, how=how),
        rtol=1e-13, atol=1e-13,
    )


def _lazy_grid(operator, descending, *, missing=False):
    eager = make_grid(operator, descending=descending, batch_shape=(5,))
    if missing:
        eager.values[..., 7 if descending else 0, 1] = np.nan
    chunks = ((2, 2, 1), (3, 1, 4), (2, 3, 1, 4))
    edges = [np.cumsum((0, *sizes)) for sizes in chunks]
    reads = []

    @delayed(pure=False)
    def read(index):
        reads.append(index)
        slices = tuple(slice(axis[i], axis[i + 1]) for axis, i in zip(edges, index, strict=True))
        return eager.values[slices]

    blocks = {}
    for index in product(*(range(len(sizes)) for sizes in chunks)):
        shape = tuple(sizes[i] for sizes, i in zip(chunks, index, strict=True))
        blocks[index] = da.from_delayed(read(index), shape=shape, dtype=eager.dtype)
    lazy = da.concatenate([
        da.concatenate([
            da.concatenate([blocks[t, r, c] for c in range(4)], axis=2) for r in range(3)
        ], axis=1) for t in range(3)
    ], axis=0)
    return eager, eager.copy(data=lazy), reads


@pytest.mark.parametrize("descending", [False, True])
@pytest.mark.parametrize("skipna", [False, True])
def test_dask_reads_each_touched_chunk_once_and_no_others(descending, skipna):
    operator = make_operator()
    eager, lazy, reads = _lazy_grid(operator, descending, missing=skipna)
    plan = RestrictedOperator.from_grid(operator, lazy)
    assert reads == []
    got = reduce_with_restricted_operator(lazy, plan, skipna=skipna)
    xr.testing.assert_identical(got, reduce_with_operator(eager, operator, skipna=skipna))
    row, col = np.divmod(operator.matrix.indices, 10)
    if descending:
        row = 7 - row
    touched = set(zip(
        np.searchsorted([3, 4, 8], row, side="right"), np.searchsorted([2, 5, 6, 10], col, side="right"), strict=True,
    ))
    expected = Counter((t, r, c) for t in range(3) for r, c in touched)
    assert Counter(reads) == expected
    assert len(reads) < 3 * 3 * 4
    assert isinstance(got.data, np.ndarray)


def test_invalid_inputs_fail_before_reading_data():
    operator = make_operator()
    _, lazy, reads = _lazy_grid(operator, False)
    plan = RestrictedOperator.from_grid(operator, lazy)
    for changed in (lazy.assign_coords(latitude=lazy.latitude + 1), lazy.isel(latitude=slice(None, None, -1))):
        with pytest.raises(ValueError, match="stored source grid"):
            reduce_with_restricted_operator(changed, plan)
    with pytest.raises(ValueError, match="chunks do not match"):
        reduce_with_restricted_operator(lazy.chunk({"latitude": 2}), plan)
    with pytest.raises(ValueError, match="how must"):
        reduce_with_restricted_operator(lazy, plan, how="median")
    with pytest.raises(ValueError, match="missing required dims"):
        reduce_with_restricted_operator(lazy.isel(latitude=0), plan)
    assert reads == []


def test_unchunked_batches_are_bounded(monkeypatch):
    operator = make_operator()
    grid = make_grid(operator, batch_shape=(100,))
    plan = RestrictedOperator.compute(operator, operator.source_lat, 2, 2)
    sizes = []
    to_numpy = xr.DataArray.to_numpy
    monkeypatch.setattr(api, "_RESTRICTED_BATCH_BYTES", 1024)

    def record(array):
        if "latitude" in array.dims and "longitude" in array.dims:
            sizes.append(array.sizes["dim0"])
        return to_numpy(array)

    monkeypatch.setattr(xr.DataArray, "to_numpy", record)
    got = reduce_with_restricted_operator(grid, plan)
    assert max(sizes) < grid.sizes["dim0"]
    xr.testing.assert_identical(got, reduce_with_operator(grid, operator))


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_unsorted_coefficients_keep_accumulation_order(dtype):
    matrix = sp.csr_matrix((np.array([1e16, -1e16, 1], dtype=dtype), [65, 1, 3], [0, 3]), shape=(1, 80))
    operator = make_operator(matrix)
    grid = xr.ones_like(make_grid(operator, descending=True))
    plan = RestrictedOperator.compute(operator, grid.latitude.values, 2, 2)
    for how in ("mean", "sum"):
        expected = reduce_with_operator(grid, operator, how=how)
        got = reduce_with_restricted_operator(grid, plan, how=how)
        xr.testing.assert_identical(got, expected)
        np.testing.assert_array_equal(got.values, np.ones(got.shape))


def test_zero_operator_does_not_read_data():
    operator = make_operator(sp.csr_matrix((3, 80)))
    _, lazy, reads = _lazy_grid(operator, False)
    plan = RestrictedOperator.from_grid(operator, lazy)
    got = reduce_with_restricted_operator(lazy, plan)
    np.testing.assert_array_equal(got.values, np.zeros((5, 3)))
    assert reads == []


def test_full_coverage_operator():
    operator = make_operator(sp.csr_matrix(np.ones((2, 80))))
    grid = make_grid(operator, descending=True)
    plan = RestrictedOperator.compute(operator, grid.latitude.values, 2, 2)
    assert plan.windows == ((slice(0, 8), slice(0, 10)),)
    xr.testing.assert_identical(reduce_with_restricted_operator(grid, plan), reduce_with_operator(grid, operator))
