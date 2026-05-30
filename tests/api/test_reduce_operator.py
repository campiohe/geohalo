import geopandas as gpd
import numpy as np
import pytest
import shapely
import xarray as xr

import geohalo.reduce_operator as ro
from geohalo.api import reduce_with_operator, reduce_with_stencil
from geohalo.cache import LocalCache
from geohalo.reduce_operator import ReduceOperator, reduce_operator_digest
from geohalo.resampler import Resampler
from geohalo.stencil import Stencil


def _da(values, lats, lons, extra=None):
    dims = ("latitude", "longitude")
    coords = {"latitude": lats, "longitude": lons}
    if extra is not None:
        name, vals = extra
        dims = (name, *dims)
        coords[name] = vals
    return xr.DataArray(values, dims=dims, coords=coords)


def _setup():
    coarse_lat = np.array([0.0, 1.0, 2.0, 3.0])
    coarse_lon = np.array([0.0, 1.0, 2.0])
    fine_lat = np.linspace(0.0, 3.0, 8)
    fine_lon = np.linspace(0.0, 2.0, 6)
    geoms = gpd.GeoSeries(
        [shapely.box(0.2, 0.2, 1.8, 2.8), shapely.box(1.0, 0.0, 2.0, 1.5)],
        index=["a", "b"],
    )
    stencil = Stencil.compute(fine_lat, fine_lon, geoms)
    return coarse_lat, coarse_lon, stencil


@pytest.mark.parametrize("iterations", [1, 2, 3])
def test_matrix_matches_occ_at_t(iterations: int) -> None:
    coarse_lat, coarse_lon, stencil = _setup()
    op = ReduceOperator.compute(stencil, coarse_lat, coarse_lon, iterations=iterations)
    t = Resampler.compute(coarse_lat, coarse_lon, stencil.lats, stencil.lons, iterations=iterations)
    expected = (stencil.occupancy_matrix @ t.transform_matrix).toarray()
    np.testing.assert_allclose(op.matrix.toarray(), expected, atol=1e-9)


@pytest.mark.parametrize("iterations", [1, 3])
@pytest.mark.parametrize("how", ["mean", "sum"])
def test_apply_matches_reduce_with_stencil(iterations: int, how: str) -> None:
    coarse_lat, coarse_lon, stencil = _setup()
    rng = np.random.default_rng(0)
    da = _da(rng.uniform(0, 10, size=(5, 4, 3)), coarse_lat, coarse_lon, extra=("member", np.arange(5)))
    op = ReduceOperator.compute(stencil, coarse_lat, coarse_lon, iterations=iterations)
    got = reduce_with_operator(da, op, how=how)
    expected = reduce_with_stencil(da, stencil, resample_iterations=iterations, how=how)
    np.testing.assert_allclose(got.transpose(*expected.dims).values, expected.values, atol=1e-9)
    assert list(got["geom"].values) == list(expected["geom"].values)


def test_apply_two_batch_dims_reshapes() -> None:
    # Two leading batch dims -> 3-D output. Guards the matmul result staying a plain ndarray
    # (a scipy np.matrix cannot reshape to >2-D) and the (batch..., geom) reshape being correct.
    coarse_lat, coarse_lon, stencil = _setup()
    rng = np.random.default_rng(11)
    da = xr.DataArray(
        rng.uniform(0, 10, size=(3, 2, 4, 3)),
        dims=("member", "step", "latitude", "longitude"),
        coords={
            "member": np.arange(3), "step": np.arange(2),
            "latitude": coarse_lat, "longitude": coarse_lon,
        },
    )
    op = ReduceOperator.compute(stencil, coarse_lat, coarse_lon, iterations=2)
    got = reduce_with_operator(da, op)
    assert got.dims == ("member", "step", "geom")
    assert got.shape == (3, 2, 2)
    expected = reduce_with_stencil(da, stencil, resample_iterations=2)
    np.testing.assert_allclose(got.transpose(*expected.dims).values, expected.values, atol=1e-9)


def test_apply_same_grid_no_resample() -> None:
    lat = np.array([0.0, 1.0, 2.0, 3.0])
    lon = np.array([0.0, 1.0, 2.0])
    geoms = gpd.GeoSeries([shapely.box(0.2, 0.2, 1.8, 2.8)], index=["a"])
    stencil = Stencil.compute(lat, lon, geoms)
    da = _da(np.random.default_rng(1).uniform(0, 10, size=(4, 3)), lat, lon)
    op = ReduceOperator.compute(stencil, lat, lon, iterations=2)
    got = reduce_with_operator(da, op)
    expected = reduce_with_stencil(da, stencil)
    np.testing.assert_allclose(got.values, expected.values, atol=1e-9)


def test_apply_descending_source_lat() -> None:
    coarse_lat, coarse_lon, stencil = _setup()
    rng = np.random.default_rng(2)
    vals = rng.uniform(0, 10, size=(4, 3))
    da_asc = _da(vals, coarse_lat, coarse_lon)
    da_desc = _da(vals[::-1], coarse_lat[::-1], coarse_lon)
    op = ReduceOperator.compute(stencil, coarse_lat, coarse_lon, iterations=2)
    np.testing.assert_allclose(
        reduce_with_operator(da_desc, op).values,
        reduce_with_operator(da_asc, op).values,
        atol=1e-9,
    )


def test_reduce_with_operator_grid_mismatch_raises() -> None:
    coarse_lat, coarse_lon, stencil = _setup()
    op = ReduceOperator.compute(stencil, coarse_lat, coarse_lon, iterations=2)
    wrong = _da(np.zeros((5, 3)), np.linspace(0.0, 4.0, 5), coarse_lon)  # 5 lats, op expects 4
    with pytest.raises(ValueError, match="does not match the operator's source grid"):
        reduce_with_operator(wrong, op)


def test_digest_stable_and_iteration_sensitive() -> None:
    coarse_lat, coarse_lon, stencil = _setup()
    d1 = reduce_operator_digest(stencil.digest, coarse_lat, coarse_lon, 1)
    d1b = reduce_operator_digest(stencil.digest, coarse_lat, coarse_lon, 1)
    d3 = reduce_operator_digest(stencil.digest, coarse_lat, coarse_lon, 3)
    assert d1 == d1b
    assert d1 != d3


def test_apply_dataset() -> None:
    coarse_lat, coarse_lon, stencil = _setup()
    rng = np.random.default_rng(3)
    ds = xr.Dataset(
        {
            "t2m": _da(rng.uniform(250, 300, size=(4, 3)), coarse_lat, coarse_lon),
            "tp": _da(rng.uniform(0, 5, size=(4, 3)), coarse_lat, coarse_lon),
            "scalar": xr.DataArray(1.0),
        },
    )
    op = ReduceOperator.compute(stencil, coarse_lat, coarse_lon, iterations=2)
    got = reduce_with_operator(ds, op)
    expected = reduce_with_stencil(ds, stencil, resample_iterations=2)
    for var in ("t2m", "tp"):
        np.testing.assert_allclose(got[var].values, expected[var].values, atol=1e-9)
    assert "scalar" in got


def test_clean_dataset_builds_resampler_once(monkeypatch) -> None:
    coarse_lat, coarse_lon, stencil = _setup()  # coarse data grid, fine stencil -> resample happens
    rng = np.random.default_rng(7)
    ds = xr.Dataset(
        {
            "t2m": _da(rng.uniform(250, 300, size=(4, 3)), coarse_lat, coarse_lon),
            "tp": _da(rng.uniform(0, 5, size=(4, 3)), coarse_lat, coarse_lon),
        },
    )
    calls = {"n": 0}
    real = ro.FactoredResampler.compute

    def counting(*args, **kwargs):
        calls["n"] += 1
        return real(*args, **kwargs)

    monkeypatch.setattr(ro.FactoredResampler, "compute", counting)
    reduce_with_stencil(ds, stencil, resample_iterations=2)
    assert calls["n"] == 1  # one resampler for the whole Dataset, not one per variable


def test_mixed_clean_and_nan_dataset_matches_per_var() -> None:
    coarse_lat, coarse_lon, stencil = _setup()
    rng = np.random.default_rng(8)
    clean = rng.uniform(250, 300, size=(4, 3))
    dirty = rng.uniform(0, 5, size=(4, 3))
    dirty[0, 0] = np.nan
    ds = xr.Dataset(
        {"t2m": _da(clean, coarse_lat, coarse_lon), "tp": _da(dirty, coarse_lat, coarse_lon)},
    )
    out = reduce_with_stencil(ds, stencil, resample_iterations=2)
    exp_clean = reduce_with_stencil(
        _da(clean, coarse_lat, coarse_lon).rename("t2m").to_dataset(), stencil, resample_iterations=2,
    )
    exp_dirty = reduce_with_stencil(
        _da(dirty, coarse_lat, coarse_lon).rename("tp").to_dataset(), stencil, resample_iterations=2,
    )
    np.testing.assert_allclose(out["t2m"].values, exp_clean["t2m"].values, atol=1e-9)
    np.testing.assert_allclose(out["tp"].values, exp_dirty["tp"].values, atol=1e-9, equal_nan=True)


def test_cache_roundtrip(tmp_path) -> None:
    coarse_lat, coarse_lon, stencil = _setup()
    cache = LocalCache(tmp_path)
    op1 = cache.get_or_compute_reduce_operator(stencil, coarse_lat, coarse_lon, iterations=2)
    op2 = cache.get_or_compute_reduce_operator(stencil, coarse_lat, coarse_lon, iterations=2)  # cache hit
    assert op1.digest == op2.digest
    np.testing.assert_array_equal(op1.matrix.toarray(), op2.matrix.toarray())
    np.testing.assert_array_equal(op1.row_sums, op2.row_sums)
    direct = ReduceOperator.compute(stencil, coarse_lat, coarse_lon, iterations=2)
    np.testing.assert_allclose(op2.matrix.toarray(), direct.matrix.toarray(), atol=1e-12)
