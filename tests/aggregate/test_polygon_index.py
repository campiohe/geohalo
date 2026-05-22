"""Output polygon dim is a MultiIndex named after weights.key_names."""

from collections.abc import Callable

import xarray as xr

from geohalo.aggregate import aggregate
from geohalo.geometry import PolygonSet
from geohalo.grid import GridSpec
from geohalo.weights import compute_weights


def test_output_polygon_dim_is_multiindex_with_keynames(
    simple_grid: GridSpec,
    simple_polygons: PolygonSet,
    make_field: Callable[..., xr.DataArray],
) -> None:
    weights = compute_weights(simple_polygons, simple_grid)
    out = aggregate(make_field(simple_grid, value=0.0), weights)
    idx = out["polygon"].to_index()
    assert list(idx.names) == list(simple_polygons.key_names)
