import geopandas as gpd
import numpy as np
import shapely
import xarray as xr

from geohalo.api import reduce_with_stencil
from geohalo.stencil import Stencil


def _da(values, lats, lons):
    return xr.DataArray(values, dims=("latitude", "longitude"),
                        coords={"latitude": lats, "longitude": lons})


def test_mean_excludes_nan() -> None:
    lats = np.array([0.0, 1.0, 2.0])
    lons = np.array([0.0, 1.0, 2.0])
    values = np.array([[1.0, 1.0, 1.0], [np.nan, np.nan, np.nan], [3.0, 3.0, 3.0]])
    da = _da(values, lats, lons)
    geoms = gpd.GeoSeries([shapely.box(-0.4, -0.4, 2.4, 2.4)], index=["box"])
    out = reduce_with_stencil(da, Stencil.compute(lats, lons, geoms))
    np.testing.assert_allclose(out.values, [2.0], rtol=1e-2)


def test_mean_all_nan_is_nan() -> None:
    lats = np.array([0.0, 1.0])
    lons = np.array([0.0, 1.0])
    da = _da(np.full((2, 2), np.nan), lats, lons)
    geoms = gpd.GeoSeries([shapely.box(-0.4, -0.4, 1.4, 1.4)], index=["box"])
    out = reduce_with_stencil(da, Stencil.compute(lats, lons, geoms))
    assert np.isnan(out.values[0])


def test_sum_drops_nan() -> None:
    lats = np.array([0.0, 1.0])
    lons = np.array([0.0, 1.0])
    da = _da(np.array([[1.0, np.nan], [1.0, 1.0]]), lats, lons)
    geoms = gpd.GeoSeries([shapely.box(-0.4, -0.4, 1.4, 1.4)], index=["box"])
    out = reduce_with_stencil(da, Stencil.compute(lats, lons, geoms), how="sum")
    assert np.isfinite(out.values[0])
