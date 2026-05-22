"""Row-sum-to-1: the area-normalization invariant (CLAUDE.md #3)."""

import numpy as np

from geohalo.geometry import PolygonSet
from geohalo.grid import GridSpec
from geohalo.weights import compute_weights


def test_each_row_sums_to_one_simple(
    simple_grid: GridSpec,
    simple_polygons: PolygonSet,
) -> None:
    weights = compute_weights(simple_polygons, simple_grid)
    row_sums = np.asarray(weights.matrix.sum(axis=1)).ravel()
    np.testing.assert_allclose(row_sums, 1.0, rtol=1e-10, atol=1e-10)


def test_each_row_sums_to_one_high_latitude(
    high_latitude_grid: GridSpec,
    high_latitude_polygons: PolygonSet,
) -> None:
    weights = compute_weights(high_latitude_polygons, high_latitude_grid)
    row_sums = np.asarray(weights.matrix.sum(axis=1)).ravel()
    np.testing.assert_allclose(row_sums, 1.0, rtol=1e-10, atol=1e-10)
