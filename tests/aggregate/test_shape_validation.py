"""aggregate rejects DataArrays that don't match weights.native_shape."""

import numpy as np
import pytest
import xarray as xr

from geohalo.aggregate import aggregate
from geohalo.geometry import PolygonSet
from geohalo.grid import GridSpec
from geohalo.weights import compute_weights


def test_shape_mismatch_raises(
    simple_grid: GridSpec,
    simple_polygons: PolygonSet,
) -> None:
    weights = compute_weights(simple_polygons, simple_grid)
    wrong_shape = xr.DataArray(
        np.zeros((3, 3)),
        dims=("latitude", "longitude"),
        coords={"latitude": np.arange(3.0), "longitude": np.arange(3.0)},
    )
    with pytest.raises(ValueError, match="does not match"):
        aggregate(wrong_shape, weights)
