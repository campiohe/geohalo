"""EmptyCoverageError: fail-fast on degenerate or outside polygons."""

import pytest

from geohalo.geometry import PolygonSet
from geohalo.grid import GridSpec
from geohalo.weights import EmptyCoverageError, compute_weights


def test_polygon_outside_grid_raises(
    simple_grid: GridSpec,
    outside_polygons: PolygonSet,
) -> None:
    with pytest.raises(EmptyCoverageError, match="away"):
        compute_weights(outside_polygons, simple_grid)
