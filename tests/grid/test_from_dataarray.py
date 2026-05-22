"""GridSpec.from_dataarray constructor."""

import numpy as np
import xarray as xr

from geohalo.grid import GridSpec


def test_round_trip_with_named_dims() -> None:
    lats = np.array([0.0, 1.0, 2.0])
    lons = np.array([10.0, 11.0])
    da = xr.DataArray(
        np.zeros((3, 2)),
        dims=("latitude", "longitude"),
        coords={"latitude": lats, "longitude": lons},
    )
    grid = GridSpec.from_dataarray(da)
    np.testing.assert_array_equal(grid.lats, lats)
    np.testing.assert_array_equal(grid.lons, lons)


def test_custom_dim_names() -> None:
    da = xr.DataArray(
        np.zeros((2, 2)),
        dims=("y", "x"),
        coords={"y": [0.0, 1.0], "x": [0.0, 1.0]},
    )
    grid = GridSpec.from_dataarray(da, lat_dim="y", lon_dim="x")
    np.testing.assert_array_equal(grid.lats, [0.0, 1.0])
