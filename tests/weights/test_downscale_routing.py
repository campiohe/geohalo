"""Downscale routing: matrix shape and metadata at factor 1 vs factor > 1."""

import numpy as np

from geohalo.geometry import PolygonSet
from geohalo.grid import GridSpec
from geohalo.weights import compute_weights


def test_factor_1_matrix_shape_native(
    simple_grid: GridSpec,
    simple_polygons: PolygonSet,
) -> None:
    weights = compute_weights(simple_polygons, simple_grid)
    n_lat, n_lon = simple_grid.shape
    assert weights.matrix.shape == (len(simple_polygons.keys), n_lat * n_lon)


def test_factor_2_matrix_folded_to_native_shape(
    simple_grid: GridSpec,
    simple_polygons: PolygonSet,
) -> None:
    weights = compute_weights(
        simple_polygons, simple_grid, target_resolution=0.5,
    )
    n_lat, n_lon = simple_grid.shape
    assert weights.downscale_factor == 2
    assert weights.matrix.shape == (len(simple_polygons.keys), n_lat * n_lon)
    row_sums = np.asarray(weights.matrix.sum(axis=1)).ravel()
    np.testing.assert_allclose(row_sums, 1.0, rtol=1e-10, atol=1e-10)


def test_target_resolution_stored(
    simple_grid: GridSpec,
    simple_polygons: PolygonSet,
) -> None:
    weights = compute_weights(
        simple_polygons, simple_grid, target_resolution=0.5,
    )
    assert weights.target_resolution == 0.5
    np.testing.assert_allclose(weights.achieved_resolution, 0.5)
