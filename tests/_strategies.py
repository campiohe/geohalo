"""Hypothesis strategies for geohalo tests.

Strategies are intentionally bounded — lat ranges stay inside ±80° so we
don't generate degenerate cells near the poles, grids stay small so any
shrunk failing example is human-readable.
"""

import numpy as np
import shapely
import xarray as xr
from hypothesis import strategies as st

from geohalo.geometry import PolygonSet
from geohalo.grid import GridSpec


@st.composite
def regular_grid_st(draw: st.DrawFn) -> GridSpec:
    """3-12 lats x 3-12 lons, dlat ∈ [0.25°, 5°], lats inside [-80°, 80°]."""
    n_lat = draw(st.integers(min_value=3, max_value=12))
    n_lon = draw(st.integers(min_value=3, max_value=12))
    dlat = draw(st.floats(min_value=0.25, max_value=5.0))
    half_span = dlat * (n_lat - 1) / 2.0
    lat_center = draw(
        st.floats(
            min_value=-80.0 + half_span + 1.0,
            max_value=80.0 - half_span - 1.0,
        ),
    )
    lats = lat_center + (np.arange(n_lat) - (n_lat - 1) / 2.0) * dlat
    lon_center = draw(st.floats(min_value=-170.0, max_value=170.0))
    lons = lon_center + (np.arange(n_lon) - (n_lon - 1) / 2.0) * dlat
    return GridSpec(lats=lats, lons=lons)


@st.composite
def polygon_in_grid_st(draw: st.DrawFn, grid: GridSpec) -> shapely.Polygon:
    """Axis-aligned rectangle clamped to the grid interior."""
    dlat = float(abs(grid.lats[1] - grid.lats[0]))
    dlon = float(abs(grid.lons[1] - grid.lons[0]))
    lat_lo = float(grid.lats[0]) + dlat * 0.6
    lat_hi = float(grid.lats[-1]) - dlat * 0.6
    lon_lo = float(grid.lons[0]) + dlon * 0.6
    lon_hi = float(grid.lons[-1]) - dlon * 0.6

    x0 = draw(st.floats(min_value=lon_lo, max_value=lon_hi - dlon * 0.2))
    x1 = draw(st.floats(min_value=x0 + dlon * 0.1, max_value=lon_hi))
    y0 = draw(st.floats(min_value=lat_lo, max_value=lat_hi - dlat * 0.2))
    y1 = draw(st.floats(min_value=y0 + dlat * 0.1, max_value=lat_hi))
    return shapely.box(x0, y0, x1, y1)


@st.composite
def polygonset_in_grid_st(
    draw: st.DrawFn,
    grid: GridSpec,
    *,
    min_size: int = 1,
    max_size: int = 4,
) -> PolygonSet:
    n = draw(st.integers(min_value=min_size, max_value=max_size))
    geoms = [draw(polygon_in_grid_st(grid)) for _ in range(n)]
    keys = [(i,) for i in range(n)]
    return PolygonSet.build(geoms=geoms, keys=keys, key_names=("polygon_id",))


@st.composite
def dataarray_for_grid_st(
    draw: st.DrawFn,
    grid: GridSpec,
    *,
    batch_dims: tuple[tuple[str, int], ...] = (),
    allow_nan: bool = False,
) -> xr.DataArray:
    """Build an `xr.DataArray` shaped to `grid`, with optional batch dims."""
    n_lat, n_lon = grid.shape
    shape = (*tuple(s for _, s in batch_dims), n_lat, n_lon)
    if allow_nan:
        data = draw(
            st.lists(
                st.floats(min_value=-1e3, max_value=1e3, allow_nan=False, allow_infinity=False)
                | st.just(np.nan),
                min_size=int(np.prod(shape)),
                max_size=int(np.prod(shape)),
            ),
        )
    else:
        data = draw(
            st.lists(
                st.floats(min_value=-1e3, max_value=1e3, allow_nan=False, allow_infinity=False),
                min_size=int(np.prod(shape)),
                max_size=int(np.prod(shape)),
            ),
        )
    arr = np.asarray(data, dtype=np.float64).reshape(shape)
    dims = (*tuple(name for name, _ in batch_dims), "latitude", "longitude")
    coords = {
        **{name: np.arange(size) for name, size in batch_dims},
        "latitude": grid.lats,
        "longitude": grid.lons,
    }
    return xr.DataArray(arr, dims=dims, coords=coords)
