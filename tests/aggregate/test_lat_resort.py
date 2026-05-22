"""aggregate internally re-sorts a descending-lat DataArray (CLAUDE.md #1)."""

import numpy as np
import xarray as xr

from geohalo.aggregate import aggregate
from geohalo.geometry import PolygonSet
from geohalo.grid import GridSpec
from geohalo.weights import compute_weights


def test_descending_lat_dataarray_resorted_internally(
    simple_grid: GridSpec,
    simple_polygons: PolygonSet,
) -> None:
    weights = compute_weights(simple_polygons, simple_grid)
    n_lat, n_lon = simple_grid.shape
    values = np.arange(n_lat * n_lon, dtype=np.float64).reshape(n_lat, n_lon)

    ascending = xr.DataArray(
        values, dims=("latitude", "longitude"),
        coords={"latitude": simple_grid.lats, "longitude": simple_grid.lons},
    )
    descending = xr.DataArray(
        values[::-1, :], dims=("latitude", "longitude"),
        coords={"latitude": simple_grid.lats[::-1], "longitude": simple_grid.lons},
    )
    out_a = aggregate(ascending, weights)
    out_d = aggregate(descending, weights)
    np.testing.assert_allclose(out_a.values, out_d.values, rtol=1e-12)
