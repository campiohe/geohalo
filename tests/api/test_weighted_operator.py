"""Operators weighted by a fine lattice of counts, reweighted by cell, and stacked by rows."""

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
import shapely
import xarray as xr

import geohalo as ghl


def _grid(south, north, west, east, step):
    """Cell centres from cell edges, latitude north to south as the store keeps it."""
    lat = np.arange(south + step / 2, north, step)[::-1].copy()
    lon = np.arange(west + step / 2, east, step)
    return lat, lon


def _zones():
    return gpd.GeoSeries(
        [shapely.Point(5.0, 45.0).buffer(1.3), shapely.Polygon([(3.1, 43.2), (6.9, 44.1), (4.6, 46.8)])],
        index=pd.Index(["round", "triangle"], name="zone"), crs="EPSG:4326",
    )


GRID = _grid(40.0, 50.0, 0.0, 10.0, 0.25)


def test_concat_of_one_operator_is_that_operator():
    op = ghl.ReduceOperator.compute(ghl.Stencil.compute(*GRID, _zones()), *GRID)
    assert ghl.ReduceOperator.concat([op]) is op


def test_concat_stacks_rows_in_order():
    first = ghl.ReduceOperator.compute(ghl.Stencil.compute(*GRID, _zones()), *GRID, dtype=np.float32)
    other = _zones().set_axis(pd.Index(["x", "y"], name="zone"))
    second = ghl.ReduceOperator.compute(ghl.Stencil.compute(*GRID, other), *GRID, dtype=np.float32)
    stacked = ghl.ReduceOperator.concat([first, second])
    assert list(stacked.keys) == ["round", "triangle", "x", "y"]
    assert stacked.keys.name == "zone"
    assert stacked.matrix.dtype == np.float32
    np.testing.assert_array_equal(stacked.matrix.toarray(), np.vstack([first.matrix.toarray(),
                                                                       second.matrix.toarray()]))
    np.testing.assert_array_equal(stacked.row_sums, np.r_[first.row_sums, second.row_sums])
    assert stacked.digest not in (first.digest, second.digest)


def test_concat_refuses_what_does_not_stack():
    stencil = ghl.Stencil.compute(*GRID, _zones())
    op = ghl.ReduceOperator.compute(stencil, *GRID)
    with pytest.raises(ValueError, match="at least one"):
        ghl.ReduceOperator.concat([])
    with pytest.raises(ValueError, match="repeat"):
        ghl.ReduceOperator.concat([op, op])
    other = _zones().set_axis(pd.Index(["a", "b"], name="zone"))
    with pytest.raises(ValueError, match="dtype"):
        ghl.ReduceOperator.concat([op, ghl.ReduceOperator.compute(
            ghl.Stencil.compute(*GRID, other), *GRID, dtype=np.float32)])
    elsewhere = _grid(40.0, 50.0, 0.0, 10.0, 0.5)
    with pytest.raises(ValueError, match="source grid"):
        ghl.ReduceOperator.concat([op, ghl.ReduceOperator.compute(ghl.Stencil.compute(*GRID, other), *elsewhere)])
    with pytest.raises(ValueError, match="iteration"):
        ghl.ReduceOperator.concat([op, ghl.ReduceOperator.compute(
            ghl.Stencil.compute(*GRID, other), *GRID, iterations=2)])
    unnamed = _zones().set_axis(pd.Index(["c", "d"]))
    with pytest.raises(ValueError, match="key names"):
        ghl.ReduceOperator.concat([op, ghl.ReduceOperator.compute(ghl.Stencil.compute(*GRID, unnamed), *GRID)])


def test_reweight_scales_columns_and_drops_cells_weighing_nothing():
    op = ghl.ReduceOperator.compute(ghl.Stencil.compute(*GRID, _zones()), *GRID)
    weights = np.random.default_rng(3).random((GRID[0].size, GRID[1].size))
    weights[10:20] = np.nan
    scaled = op.reweight(weights[::-1])
    stored = op.reweight(weights, descending=True)
    np.testing.assert_array_equal(scaled.matrix.toarray(), stored.matrix.toarray())
    assert scaled.digest == stored.digest
    expected = op.matrix.toarray() * np.nan_to_num(weights[::-1]).ravel()
    np.testing.assert_allclose(scaled.matrix.toarray(), expected, rtol=1e-15)
    assert not np.any(scaled.matrix.data == 0)
    np.testing.assert_allclose(scaled.row_sums, expected.sum(axis=1), rtol=1e-12)
    assert scaled.keys.equals(op.keys)
    assert scaled.iterations == op.iterations


def test_reweight_keeps_float32_coefficients_with_float64_row_sums():
    op = ghl.ReduceOperator.compute(ghl.Stencil.compute(*GRID, _zones()), *GRID, dtype=np.float32)
    scaled = op.reweight(np.full((GRID[0].size, GRID[1].size), 3.0))
    assert scaled.matrix.dtype == np.float32
    assert scaled.row_sums.dtype == np.float64
    np.testing.assert_array_equal(scaled.row_sums, scaled.matrix.astype(np.float64).sum(axis=1).A1)


def test_reweight_digest_ignores_cells_the_operator_never_reads():
    op = ghl.ReduceOperator.compute(ghl.Stencil.compute(*GRID, _zones()), *GRID)
    weights = np.ones((GRID[0].size, GRID[1].size))
    elsewhere = weights.copy()
    elsewhere[0, 0] = 5.0
    read = weights.copy()
    read[GRID[0].size // 2, GRID[1].size // 2] = 5.0
    assert op.reweight(weights).digest == op.reweight(elsewhere).digest
    assert op.reweight(weights).digest != op.reweight(read).digest
    assert op.reweight(weights).digest != op.digest


def test_reweight_refuses_a_wrong_shape_and_negative_weights():
    op = ghl.ReduceOperator.compute(ghl.Stencil.compute(*GRID, _zones()), *GRID)
    with pytest.raises(ValueError, match="shape"):
        op.reweight(np.ones((3, 3)))
    with pytest.raises(ValueError, match="negative or infinite"):
        op.reweight(-np.ones((GRID[0].size, GRID[1].size)))
    with pytest.raises(TypeError, match="real numbers"):
        op.reweight(np.ones((GRID[0].size, GRID[1].size), dtype=bool))


def test_reweight_by_density_is_the_weight_key_mean():
    stencil = ghl.Stencil.compute(*GRID, _zones())
    rng = np.random.default_rng(4)
    density = rng.random((GRID[0].size, GRID[1].size))
    field = rng.normal(size=(3, GRID[0].size, GRID[1].size))
    grid = xr.Dataset(
        {"t": (("step", "latitude", "longitude"), field), "w": (("latitude", "longitude"), density)},
        coords={"latitude": GRID[0], "longitude": GRID[1], "step": [0, 1, 2]},
    )
    expected = ghl.reduce_with_stencil(grid[["t", "w"]], stencil, weight_key="w")["t"]
    op = ghl.ReduceOperator.compute(stencil, *GRID).reweight(density, descending=True)
    got = op.apply_grid(field, descending=True, how="mean")
    np.testing.assert_allclose(got, expected.transpose("step", "geom").to_numpy(), rtol=1e-12)
