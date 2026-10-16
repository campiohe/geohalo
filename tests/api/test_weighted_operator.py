"""Operators weighted by a fine lattice of counts, reweighted by cell, and stacked by rows."""

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
import shapely

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
