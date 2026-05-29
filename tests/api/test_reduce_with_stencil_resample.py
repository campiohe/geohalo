import geopandas as gpd
import numpy as np
import shapely
import xarray as xr

from geohalo.api import reduce_with_stencil, resample_grid_with_matrix
from geohalo.resampler import Resampler
from geohalo.stencil import Stencil


def _da(values, lats, lons):
    return xr.DataArray(values, dims=("latitude", "longitude"),
                        coords={"latitude": lats, "longitude": lons})


def test_stencil_on_finer_grid_triggers_resample() -> None:
    coarse_lat = np.array([0.0, 1.0, 2.0])
    coarse_lon = np.array([0.0, 1.0, 2.0])
    da = _da(np.full((3, 3), 5.0), coarse_lat, coarse_lon)
    fine_lat = np.linspace(0.0, 2.0, 6)
    fine_lon = np.linspace(0.0, 2.0, 6)
    geoms = gpd.GeoSeries([shapely.box(-0.4, -0.4, 1.4, 1.4)], index=["box"])
    stencil = Stencil.compute(fine_lat, fine_lon, geoms)
    out = reduce_with_stencil(da, stencil, resample_iterations=3)
    np.testing.assert_allclose(out.values, [5.0], atol=1e-6)


def test_fused_equals_two_step() -> None:
    coarse_lat = np.array([0.0, 1.0, 2.0, 3.0])
    coarse_lon = np.array([0.0, 1.0, 2.0])
    rng = np.random.default_rng(0)
    da = _da(rng.uniform(0, 10, size=(4, 3)), coarse_lat, coarse_lon)
    fine_lat = np.linspace(0.0, 3.0, 8)
    fine_lon = np.linspace(0.0, 2.0, 6)
    geoms = gpd.GeoSeries([shapely.box(0.2, 0.2, 1.8, 2.8)], index=["box"])
    stencil = Stencil.compute(fine_lat, fine_lon, geoms)

    fused = reduce_with_stencil(da, stencil, resample_iterations=2)

    r = Resampler.compute(coarse_lat, coarse_lon, fine_lat, fine_lon, iterations=2)
    resampled = resample_grid_with_matrix(da, r)
    two_step = reduce_with_stencil(resampled, stencil)
    np.testing.assert_allclose(fused.values, two_step.values, rtol=1e-9)
