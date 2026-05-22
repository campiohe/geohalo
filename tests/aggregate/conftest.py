from collections.abc import Callable

import numpy as np
import pytest
import xarray as xr

from geohalo.grid import GridSpec


@pytest.fixture
def make_field() -> Callable[..., xr.DataArray]:
    """Factory: `make_field(grid, value=1.0)` -> constant-`value` DataArray."""
    def _make(grid: GridSpec, value: float = 1.0) -> xr.DataArray:
        n_lat, n_lon = grid.shape
        return xr.DataArray(
            np.full((n_lat, n_lon), value, dtype=np.float64),
            dims=("latitude", "longitude"),
            coords={"latitude": grid.lats, "longitude": grid.lons},
        )
    return _make
