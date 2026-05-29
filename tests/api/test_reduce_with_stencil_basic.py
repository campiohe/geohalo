import geopandas as gpd
import numpy as np
import pytest
import shapely
import xarray as xr

from geohalo.api import reduce_with_stencil
from geohalo.stencil import Stencil


def _da(values, lats, lons, extra_dims=(), extra_coords=None):
    dims = (*extra_dims, "latitude", "longitude")
    coords = {"latitude": lats, "longitude": lons, **(extra_coords or {})}
    return xr.DataArray(values, dims=dims, coords=coords)


def test_uniform_constant() -> None:
    lats = np.array([0.0, 1.0, 2.0])
    lons = np.array([0.0, 1.0, 2.0])
    da = _da(np.full((3, 3), 5.0), lats, lons)
    geoms = gpd.GeoSeries([shapely.box(-0.4, -0.4, 1.4, 1.4)], index=["box"])
    out = reduce_with_stencil(da, Stencil.compute(lats, lons, geoms))
    assert out.dims == ("geom",)
    np.testing.assert_allclose(out.values, [5.0])


def test_batch_preserved() -> None:
    lats = np.array([0.0, 1.0])
    lons = np.array([0.0, 1.0])
    da = _da(np.arange(8.0).reshape(2, 2, 2), lats, lons,
             extra_dims=("member",), extra_coords={"member": [0, 1]})
    geoms = gpd.GeoSeries([shapely.box(-0.4, -0.4, 1.4, 1.4)], index=["box"])
    out = reduce_with_stencil(da, Stencil.compute(lats, lons, geoms))
    assert out.dims == ("member", "geom")
    assert out.shape == (2, 1)


def test_missing_dim_raises() -> None:
    lats = np.array([0.0, 1.0])
    lons = np.array([0.0, 1.0])
    da = xr.DataArray(np.zeros((2, 2)), dims=("y", "x"), coords={"y": lats, "x": lons})
    geoms = gpd.GeoSeries([shapely.box(-0.4, -0.4, 1.4, 1.4)], index=["box"])
    stencil = Stencil.compute(lats, lons, geoms)
    with pytest.raises(ValueError, match="missing"):
        reduce_with_stencil(da, stencil)
