"""Caller row order must survive every reduction and canonical cache boundary."""

from dataclasses import replace

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp
import shapely
import xarray as xr

import geohalo as ghl
from geohalo.cache import _deser_reduce_op, _deser_restricted_op, _deser_stencil


def _geoms(keys=None):
    if keys is None:
        keys = pd.Index(["zulu", "alpha", "mike"], name="zone")
    boxes = [
        shapely.box(-2 + i % 5 - 0.3, -2 + i // 5 - 0.2, -2 + i % 5 + 0.2 + i * 0.01, -2 + i // 5 + 0.3)
        for i in range(len(keys))
    ]
    return gpd.GeoSeries(boxes, index=keys, crs="EPSG:4326")


def _grid(descending=False, missing=False):
    lat = np.arange(-2.0, 3.0)
    values = np.arange(1.0, 51.0).reshape(2, 5, 5)
    if missing:
        values[1, 0, 0] = np.nan
    grid = xr.DataArray(
        values, dims=("step", "latitude", "longitude"),
        coords={"step": [0, 1], "latitude": lat, "longitude": lat}, name="weather", attrs={"units": "K"},
    )
    return grid.isel(latitude=slice(None, None, -1)) if descending else grid


def _build(geoms, *, refined=False, cache=None, force=False):
    source = np.arange(-2.0, 3.0)
    target = np.linspace(-2, 2, 9) if refined else source
    if cache is None:
        stencil = ghl.Stencil.compute(target, target, geoms)
        operator = ghl.ReduceOperator.compute(stencil, source, source, iterations=2)
        restricted = ghl.RestrictedOperator.compute(operator, source[::-1], 2, 3)
    else:
        stencil = cache.get_or_compute_stencil(target, target, geoms, force_recompute=force)
        operator = cache.get_or_compute_reduce_operator(
            stencil, source, source, iterations=2, force_recompute=force,
        )
        restricted = cache.get_or_compute_restricted_operator(
            operator, source[::-1], 2, 3, force_recompute=force,
        )
    return stencil, operator, restricted


def _assert_equal(actual, expected):
    for a, b in zip(actual, expected, strict=True):
        pd.testing.assert_index_equal(a.keys, b.keys)
        assert a.digest == b.digest
        am = a.occupancy_matrix if isinstance(a, ghl.Stencil) else a.matrix
        bm = b.occupancy_matrix if isinstance(b, ghl.Stencil) else b.matrix
        assert am.shape == bm.shape
        for field in ("indptr", "indices", "data"):
            np.testing.assert_array_equal(getattr(am, field), getattr(bm, field))
        np.testing.assert_array_equal(a.row_sums, b.row_sums)
    assert actual[2].windows == expected[2].windows
    for a, b in zip(actual[2].gathers, expected[2].gathers, strict=True):
        np.testing.assert_array_equal(a, b)


def _no_build(*_args, **_kwargs):
    pytest.fail("a row-order change must reuse the cached build")


@pytest.mark.parametrize("how", ["mean", "sum"])
@pytest.mark.parametrize("descending", [False, True])
@pytest.mark.parametrize("refined", [False, True])
@pytest.mark.parametrize("missing", [False, True])
def test_reductions_follow_input_rows(how, descending, refined, missing):
    geoms = _geoms()
    grid = _grid(descending, missing)
    stencil, operator, _ = _build(geoms, refined=refined)
    restricted = ghl.RestrictedOperator.compute(operator, grid.latitude.values, 2, 3)
    expected = []
    expected_masked = []
    for i in range(len(geoms)):
        single_stencil, single, _ = _build(geoms.iloc[[i]], refined=refined)
        expected.append(ghl.reduce_with_operator(grid, single, how=how, skipna=missing).values)
        expected_masked.append(ghl.reduce_with_stencil(grid, single_stencil, how=how, resample_iterations=2).values)
    expected = np.concatenate(expected, axis=-1)
    expected_masked = np.concatenate(expected_masked, axis=-1)
    values = grid.to_numpy()
    arrays = [values[..., rows, cols] for rows, cols in restricted.windows]
    np.testing.assert_allclose(restricted.apply(restricted.gather(arrays), how=how, skipna=missing), expected)
    np.testing.assert_allclose(operator.apply_grid(values, descending=descending, how=how, skipna=missing), expected)
    direct = ghl.reduce(grid, geoms, how=how, target_resolution=0.5 if refined else None, resample_iterations=2)
    for result, reference in (
        (ghl.reduce_with_operator(grid, operator, how=how, skipna=missing), expected),
        (ghl.reduce_with_restricted_operator(grid, restricted, how=how, skipna=missing), expected),
        (ghl.reduce_with_stencil(grid, stencil, how=how, resample_iterations=2), expected_masked),
        (direct, expected_masked),
    ):
        assert list(result.geom.values) == list(geoms.index)
        assert result.dims == ("step", "geom")
        assert result.attrs == grid.attrs
        np.testing.assert_allclose(result.values, reference)


_KEYS = [
    pytest.param(pd.Index(["zulu", "alpha", "mike"], name="zone"), id="strings"),
    pytest.param(pd.Index([20, 3, 100], name="zone"), id="integers"),
    pytest.param(pd.Index(["zulu", 7, ("region", "alpha")], tupleize_cols=False, name="zone"), id="mixed"),
    pytest.param(pd.Index([("z", 1), ("a", 2), ("m", 3)], tupleize_cols=False, name="zone"), id="tuples"),
    pytest.param(pd.MultiIndex.from_tuples([("z", 1), ("a", 2), ("m", 3)], names=["region", "zone"]), id="multiindex"),
    pytest.param(pd.Index([np.nan, "zulu", "alpha"], name="zone"), id="null-label"),
    pytest.param(pd.CategoricalIndex(
        ["alpha", "mike", "zulu"], categories=["mike", "zulu", "alpha"], ordered=True, name="zone",
    ), id="categorical"),
    pytest.param(pd.DatetimeIndex(["2020-03-01", "2020-01-01", "2020-02-01"], tz="UTC", name="zone"), id="datetime"),
]


@pytest.fixture(params=["local", pytest.param("redis", marks=pytest.mark.redis)])
def cache(request, tmp_path):
    if request.param == "redis":
        return ghl.RedisCache(request.getfixturevalue("redis_client"))
    return ghl.LocalCache(tmp_path)


@pytest.mark.parametrize("keys", _KEYS)
@pytest.mark.parametrize("refined", [False, True])
def test_cache_reuses_all_operators_across_row_orders(cache, monkeypatch, keys, refined):
    geoms = _geoms(keys)
    requests = [geoms, geoms.iloc[[2, 0, 1]], geoms.iloc[::-1]]
    references = [_build(request, refined=refined) for request in requests]
    first = _build(geoms, refined=refined, cache=cache)
    _assert_equal(first, references[0])
    # A forced build must also return caller order and store canonical rows.
    _assert_equal(_build(requests[1], refined=refined, cache=cache, force=True), references[1])
    for cls in (ghl.Stencil, ghl.ReduceOperator, ghl.RestrictedOperator):
        monkeypatch.setattr(cls, "compute", _no_build)
    for request, reference in zip(requests, references, strict=True):
        cached = _build(request, refined=refined, cache=cache)
        _assert_equal(cached, reference)
        assert [obj.digest for obj in cached] == [obj.digest for obj in first]
    canonical = np.argsort([repr(key) for key in geoms.index])
    for namespace, deserialize, original in zip(
        ("stencil", "reduceop", "restrictedop"),
        (_deser_stencil, _deser_reduce_op, _deser_restricted_op), first, strict=True,
    ):
        stored = deserialize(cache._load(namespace, original.digest.hex()[:16]))
        assert stored.keys.tolist() == keys.take(canonical).tolist()
    _assert_equal(first, references[0])  # no mutation when another caller requests a different order


@pytest.mark.parametrize("multiindex", [False, True])
def test_duplicate_keys_roundtrip_is_positional(cache, monkeypatch, multiindex):
    # Enough repeated labels to exercise nontrivial argsort tie permutations.
    labels = ["zulu", "alpha", "zulu", "alpha", "mike"] * 4
    keys = (
        pd.MultiIndex.from_tuples([("region", label) for label in labels], names=["region", "zone"])
        if multiindex else pd.Index(labels, name="zone")
    )
    geoms = _geoms(keys)
    expected = _build(geoms)
    _assert_equal(_build(geoms, cache=cache), expected)
    for cls in (ghl.Stencil, ghl.ReduceOperator, ghl.RestrictedOperator):
        monkeypatch.setattr(cls, "compute", _no_build)
    _assert_equal(_build(geoms, cache=cache), expected)
    # Rows with the same label must still represent their distinct geometries.
    assert not np.array_equal(expected[0].occupancy_matrix[0].toarray(), expected[0].occupancy_matrix[2].toarray())


def test_preexisting_canonical_operator_payloads(cache, monkeypatch):
    geoms = _geoms()
    canonical = geoms.iloc[np.argsort([repr(key) for key in geoms.index])]
    # These sorted builds have exactly the row layout/payload used before #18.
    _build(canonical, refined=True, cache=cache)
    expected = _build(geoms, refined=True)
    for cls in (ghl.Stencil, ghl.ReduceOperator, ghl.RestrictedOperator):
        monkeypatch.setattr(cls, "compute", _no_build)
    _assert_equal(_build(geoms, refined=True, cache=cache), expected)


def test_dataset_multiindex_and_tree_alignment():
    keys = pd.MultiIndex.from_tuples([("region", k) for k in ("zulu", "alpha", "mike")], names=["region", "zone"])
    geoms = _geoms(keys)
    stencil, operator, restricted = _build(geoms)
    grid = _grid(descending=True)
    dataset = xr.Dataset({"a": grid, "b": grid * 2, "other": ("step", [7, 8])}, attrs={"source": "test"})
    expected = ghl.reduce(dataset, geoms)
    for actual in (
        ghl.reduce_with_stencil(dataset, stencil),
        ghl.reduce_with_operator(dataset, operator),
        ghl.reduce_with_restricted_operator(dataset, restricted),
    ):
        pd.testing.assert_index_equal(actual.indexes["geom"], keys)
        xr.testing.assert_identical(actual, expected)
    edges = pd.DataFrame({"parent": [("region", "root")] * 3}, index=keys)
    tree = ghl.BiasTree.compute(edges)
    rolled = ghl.aggregate_bias_with_tree(expected, tree)
    for key in keys:
        xr.testing.assert_equal(rolled.a.sel(geom=key), expected.a.sel(geom=key))
    np.testing.assert_allclose(rolled.a.sel(geom=("region", "root")), expected.a.mean("geom"))


def test_cache_reordering_keeps_sparse_coefficient_order(tmp_path):
    stencil, operator, _ = _build(_geoms())
    # Normalizers are not necessarily fused-matrix row sums (signed weights).
    matrix = sp.csr_matrix(
        ([1e16, 1, -1e16, 7, 2, 3, 5], [2, 0, 1, 9, 5, 24, 12], [0, 3, 5, 7]), shape=(3, 25),
    )
    operator = replace(operator, matrix=matrix, row_sums=np.array([17.0, 29.0, 31.0]))
    cache = ghl.LocalCache(tmp_path)
    original = cache.get_or_compute_restricted_operator(operator, operator.source_lat, 2, 3)
    loaded = cache.get_or_compute_restricted_operator(operator, operator.source_lat, 2, 3)
    _assert_equal((stencil, operator, loaded), (stencil, operator, original))
    np.testing.assert_array_equal(loaded.row_sums, [17, 29, 31])
    ones = np.ones(loaded.matrix.shape[1])
    np.testing.assert_array_equal(loaded.apply(ones), original.apply(ones))
