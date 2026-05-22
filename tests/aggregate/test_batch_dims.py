"""Batch dims are preserved and ordered correctly in the output."""

import numpy as np
import xarray as xr

from geohalo.aggregate import aggregate
from geohalo.geometry import PolygonSet
from geohalo.grid import GridSpec
from geohalo.weights import compute_weights


def test_batch_dims_preserved(
    simple_grid: GridSpec,
    simple_polygons: PolygonSet,
) -> None:
    weights = compute_weights(simple_polygons, simple_grid)
    n_lat, n_lon = simple_grid.shape
    rng = np.random.default_rng(seed=3)
    values = rng.standard_normal((4, 2, n_lat, n_lon))
    da = xr.DataArray(
        values,
        dims=("time", "number", "latitude", "longitude"),
        coords={
            "time": np.arange(4),
            "number": np.arange(2),
            "latitude": simple_grid.lats,
            "longitude": simple_grid.lons,
        },
    )
    out = aggregate(da, weights)
    assert out.dims == ("time", "number", "polygon")
    assert out.sizes == {
        "time": 4,
        "number": 2,
        "polygon": len(simple_polygons.keys),
    }
