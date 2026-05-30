import geopandas as gpd
import numpy as np
import shapely
import xarray as xr

from geohalo.api import reduce_with_stencil
from geohalo.stencil import Stencil


def test_constant_weight_equals_no_weight() -> None:
    lats = np.array([0.0, 1.0])
    lons = np.array([0.0, 1.0])
    ds = xr.Dataset(
        {"t": (("latitude", "longitude"), np.array([[1.0, 2.0], [3.0, 4.0]])),
         "pop": (("latitude", "longitude"), np.full((2, 2), 7.0))},
        coords={"latitude": lats, "longitude": lons},
    )
    geoms = gpd.GeoSeries([shapely.box(-0.4, -0.4, 1.4, 1.4)], index=["box"])
    stencil = Stencil.compute(lats, lons, geoms)
    a = reduce_with_stencil(ds, stencil)
    b = reduce_with_stencil(ds, stencil, weight_key="pop")
    np.testing.assert_allclose(a["t"].values, b["t"].values, rtol=1e-9)


def test_pop_weighted_mean() -> None:
    lats = np.array([0.0, 1.0])
    lons = np.array([0.0, 1.0])
    ds = xr.Dataset(
        {"t": (("latitude", "longitude"), np.array([[10.0, 10.0], [20.0, 20.0]])),
         "pop": (("latitude", "longitude"), np.array([[1.0, 1.0], [9.0, 9.0]]))},
        coords={"latitude": lats, "longitude": lons},
    )
    geoms = gpd.GeoSeries([shapely.box(-0.4, -0.4, 1.4, 1.4)], index=["box"])
    out = reduce_with_stencil(ds, Stencil.compute(lats, lons, geoms), weight_key="pop")
    np.testing.assert_allclose(out["t"].values, [19.0], rtol=1e-2)


def test_weight_key_coord_on_dataarray() -> None:
    lats = np.array([0.0, 60.0])
    lons = np.array([0.0, 1.0])
    da = xr.DataArray(
        np.array([[10.0, 10.0], [20.0, 20.0]]),
        dims=("latitude", "longitude"), coords={"latitude": lats, "longitude": lons},
    )
    geoms = gpd.GeoSeries([shapely.box(-0.4, -0.4, 1.4, 60.4)], index=["box"])
    out = reduce_with_stencil(da, Stencil.compute(lats, lons, geoms), weight_key="latitude")
    np.testing.assert_allclose(out.values, [20.0], rtol=1e-2)
