"""aggregate(constant_field, W) returns that constant for every polygon."""

from collections.abc import Callable

import numpy as np
import xarray as xr

from geohalo.aggregate import aggregate
from geohalo.geometry import PolygonSet
from geohalo.grid import GridSpec
from geohalo.weights import compute_weights


def test_constant_field_yields_that_constant(
    simple_grid: GridSpec,
    simple_polygons: PolygonSet,
    make_field: Callable[..., xr.DataArray],
) -> None:
    weights = compute_weights(simple_polygons, simple_grid)
    out = aggregate(make_field(simple_grid, value=3.14), weights)
    np.testing.assert_allclose(out.values, 3.14, rtol=1e-12)
