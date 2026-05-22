"""Row-sum-to-1 across hypothesis-generated grids."""

import numpy as np
import shapely
from hypothesis import HealthCheck, given, settings

from geohalo.geometry import PolygonSet
from geohalo.grid import GridSpec
from geohalo.weights import compute_weights
from tests._strategies import regular_grid_st


@given(grid=regular_grid_st())
@settings(
    max_examples=30,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow, HealthCheck.data_too_large],
)
def test_row_sum_to_one_random_grid(grid: GridSpec) -> None:
    """Row-sum-to-1 should hold for any grid + simple-rectangle polygon."""
    dlat = float(abs(grid.lats[1] - grid.lats[0]))
    dlon = float(abs(grid.lons[1] - grid.lons[0]))
    lat_lo = float(grid.lats[0]) + dlat * 0.6
    lat_hi = float(grid.lats[-1]) - dlat * 0.6
    lon_lo = float(grid.lons[0]) + dlon * 0.6
    lon_hi = float(grid.lons[-1]) - dlon * 0.6
    ps = PolygonSet.build(
        geoms=[shapely.box(lon_lo, lat_lo, lon_hi, lat_hi)],
        keys=[(0,)],
    )
    weights = compute_weights(ps, grid)
    row_sums = np.asarray(weights.matrix.sum(axis=1)).ravel()
    np.testing.assert_allclose(row_sums, 1.0, rtol=1e-9, atol=1e-9)
