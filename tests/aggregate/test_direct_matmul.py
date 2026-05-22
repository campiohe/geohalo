"""aggregate agrees with the direct W @ flat matmul on random fields."""

import numpy as np
import xarray as xr

from geohalo.aggregate import aggregate
from geohalo.geometry import PolygonSet
from geohalo.grid import GridSpec
from geohalo.weights import compute_weights


def test_aggregate_matches_direct_matmul(
    simple_grid: GridSpec,
    simple_polygons: PolygonSet,
) -> None:
    weights = compute_weights(simple_polygons, simple_grid)
    n_lat, n_lon = simple_grid.shape
    rng = np.random.default_rng(seed=7)
    values = rng.standard_normal((n_lat, n_lon))
    da = xr.DataArray(
        values, dims=("latitude", "longitude"),
        coords={"latitude": simple_grid.lats, "longitude": simple_grid.lons},
    )
    out = aggregate(da, weights)
    expected = (weights.matrix @ values.ravel())
    np.testing.assert_allclose(out.values, expected, rtol=1e-12)
