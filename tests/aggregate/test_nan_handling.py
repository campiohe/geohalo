"""aggregate NaN-aware renormalization paths."""

import numpy as np
import xarray as xr

from geohalo.aggregate import aggregate
from geohalo.geometry import PolygonSet
from geohalo.grid import GridSpec
from geohalo.weights import compute_weights


def test_no_nans_fast_path_matches_manual_masked_computation(
    simple_grid: GridSpec,
    simple_polygons: PolygonSet,
) -> None:
    """No-NaN input: fast path == numer/denom computed by hand."""
    weights = compute_weights(simple_polygons, simple_grid)
    n_lat, n_lon = simple_grid.shape
    rng = np.random.default_rng(seed=11)
    values = rng.standard_normal((n_lat, n_lon))
    da = xr.DataArray(
        values, dims=("latitude", "longitude"),
        coords={"latitude": simple_grid.lats, "longitude": simple_grid.lons},
    )
    out_fast = aggregate(da, weights).values

    flat = values.ravel()
    valid = np.ones_like(flat)
    numer = weights.matrix @ flat
    denom = weights.matrix @ valid
    out_masked_manual = numer / denom
    np.testing.assert_allclose(out_fast, out_masked_manual, rtol=1e-12, atol=1e-12)


def test_single_nan_renormalizes_correctly_for_constant_field(
    simple_grid: GridSpec,
    simple_polygons: PolygonSet,
) -> None:
    """Constant-1 field with one NaN cell: polygons that touch other cells still yield 1.0."""
    weights = compute_weights(simple_polygons, simple_grid)
    n_lat, n_lon = simple_grid.shape
    values = np.ones((n_lat, n_lon), dtype=np.float64)
    values[0, 0] = np.nan
    da = xr.DataArray(
        values, dims=("latitude", "longitude"),
        coords={"latitude": simple_grid.lats, "longitude": simple_grid.lons},
    )
    out = aggregate(da, weights)
    for i, value in enumerate(out.values):
        row = weights.matrix.getrow(i).toarray().reshape(n_lat, n_lon)
        row_other = row.copy()
        row_other[0, 0] = 0.0
        if row_other.sum() > 0:
            np.testing.assert_allclose(value, 1.0, rtol=1e-12)


def test_all_nan_field_yields_all_nan_output(
    simple_grid: GridSpec,
    simple_polygons: PolygonSet,
) -> None:
    weights = compute_weights(simple_polygons, simple_grid)
    n_lat, n_lon = simple_grid.shape
    da = xr.DataArray(
        np.full((n_lat, n_lon), np.nan, dtype=np.float64),
        dims=("latitude", "longitude"),
        coords={"latitude": simple_grid.lats, "longitude": simple_grid.lons},
    )
    out = aggregate(da, weights)
    assert np.isnan(out.values).all()
