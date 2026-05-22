"""Weights metadata propagation."""

import numpy as np

from geohalo.geometry import PolygonSet
from geohalo.grid import GridSpec
from geohalo.weights import compute_weights


def test_polygon_keys_match_sorted_polygonset(
    simple_grid: GridSpec,
    simple_polygons: PolygonSet,
) -> None:
    weights = compute_weights(simple_polygons, simple_grid)
    assert weights.polygon_keys == simple_polygons.keys


def test_key_names_propagated(
    simple_grid: GridSpec,
    simple_polygons: PolygonSet,
) -> None:
    weights = compute_weights(simple_polygons, simple_grid)
    assert weights.key_names == simple_polygons.key_names


def test_native_shape_matches_grid(
    simple_grid: GridSpec,
    simple_polygons: PolygonSet,
) -> None:
    weights = compute_weights(simple_polygons, simple_grid)
    assert weights.native_shape == simple_grid.shape


def test_digests_stored(
    simple_grid: GridSpec,
    simple_polygons: PolygonSet,
) -> None:
    weights = compute_weights(simple_polygons, simple_grid)
    assert weights.grid_digest == simple_grid.digest
    assert weights.polyset_digest == simple_polygons.digest


def test_factor_1_default_metadata(
    simple_grid: GridSpec,
    simple_polygons: PolygonSet,
) -> None:
    weights = compute_weights(simple_polygons, simple_grid)
    assert weights.downscale_factor == 1
    assert weights.target_resolution is None
    np.testing.assert_allclose(weights.achieved_resolution, 1.0)
