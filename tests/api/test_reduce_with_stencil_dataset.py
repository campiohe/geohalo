import geopandas as gpd
import numpy as np
import shapely
import xarray as xr

from geohalo.api import reduce_with_stencil
from geohalo.stencil import Stencil


def test_dataset_two_vars() -> None:
    lats = np.array([0.0, 1.0])
    lons = np.array([0.0, 1.0])
    ds = xr.Dataset(
        {"t": (("latitude", "longitude"), np.array([[1.0, 1.0], [2.0, 2.0]])),
         "u": (("latitude", "longitude"), np.array([[10.0, 10.0], [20.0, 20.0]]))},
        coords={"latitude": lats, "longitude": lons},
    )
    geoms = gpd.GeoSeries([shapely.box(-0.4, -0.4, 1.4, 1.4)], index=["box"])
    out = reduce_with_stencil(ds, Stencil.compute(lats, lons, geoms))
    assert isinstance(out, xr.Dataset)
    assert set(out.data_vars) == {"t", "u"}
    np.testing.assert_allclose(out["t"].values, [1.5], rtol=1e-3)
    np.testing.assert_allclose(out["u"].values, [15.0], rtol=1e-3)


def test_dataset_passthrough_nonspatial() -> None:
    lats = np.array([0.0, 1.0])
    lons = np.array([0.0, 1.0])
    ds = xr.Dataset(
        {"t": (("latitude", "longitude"), np.array([[1.0, 1.0], [2.0, 2.0]])),
         "meta": ((), 99.0)},
        coords={"latitude": lats, "longitude": lons},
    )
    geoms = gpd.GeoSeries([shapely.box(-0.4, -0.4, 1.4, 1.4)], index=["box"])
    out = reduce_with_stencil(ds, Stencil.compute(lats, lons, geoms))
    assert float(out["meta"]) == 99.0
