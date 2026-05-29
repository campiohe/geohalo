import geopandas as gpd
import numpy as np
import shapely
import xarray as xr

from geohalo.api import reduce, reduce_with_stencil
from geohalo.geometry import target_coords_from_resolution
from geohalo.stencil import Stencil


def _da(values, lats, lons):
    return xr.DataArray(values, dims=("latitude", "longitude"),
                        coords={"latitude": lats, "longitude": lons})


def test_reduce_no_resample() -> None:
    lats = np.array([0.0, 1.0])
    lons = np.array([0.0, 1.0])
    da = _da(np.array([[1.0, 2.0], [3.0, 4.0]]), lats, lons)
    geoms = gpd.GeoSeries([shapely.box(-0.4, -0.4, 1.4, 1.4)], index=["box"])
    out = reduce(da, geoms)
    assert out.dims == ("geom",)


def test_reduce_with_target_resolution_matches_manual() -> None:
    lats = np.array([0.0, 1.0, 2.0, 3.0])
    lons = np.array([0.0, 1.0, 2.0])
    rng = np.random.default_rng(0)
    da = _da(rng.uniform(0, 10, size=(4, 3)), lats, lons)
    geoms = gpd.GeoSeries([shapely.box(0.2, 0.2, 1.8, 2.8)], index=["box"])

    out = reduce(da, geoms, target_resolution=0.5, resample_iterations=2)

    tlat, tlon = target_coords_from_resolution(lats, lons, 0.5)
    stencil = Stencil.compute(tlat, tlon, geoms)
    manual = reduce_with_stencil(da, stencil, resample_iterations=2)
    np.testing.assert_allclose(out.values, manual.values, rtol=1e-9)
