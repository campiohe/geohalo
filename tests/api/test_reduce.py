import geopandas as gpd
import numpy as np
import pytest
import shapely
import xarray as xr

from geohalo.api import reduce, reduce_with_stencil
from geohalo.geometry import EARTH_RADIUS_M, target_coords_from_resolution
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


@pytest.mark.parametrize("how", ["mean", "sum"])
@pytest.mark.parametrize("masked", [False, True])
@pytest.mark.parametrize("descending", [False, True])
def test_exact_reduce_matches_analytical_partial_areas(how, masked, descending):
    lats, lons = np.array([60., 70.]), np.array([0., 10.])
    values = np.array([[10., np.nan if masked else 20.], [30., 40.]])
    data = _da(values, lats, lons)
    if descending:
        data = data.isel(latitude=slice(None, None, -1))
    geoms = gpd.GeoSeries([shapely.box(-5, 60, 15, 70)], index=["zone"])
    row_area = EARTH_RADIUS_M**2 * np.deg2rad(10) * np.diff(np.sin(np.deg2rad([60, 65, 70])))
    weights = np.repeat(row_area[:, None], 2, axis=1)
    expected = np.nansum(values * weights)
    if how == "mean":
        expected /= weights[~np.isnan(values)].sum()
    actual = reduce(data, geoms, how=how, partial_cell_weighting="exact")
    np.testing.assert_allclose(actual.values, [expected], rtol=1e-14)
    assert actual.geom.values.tolist() == ["zone"]
    assert not np.allclose(actual.values, reduce(data, geoms, how=how).values, rtol=1e-4)


def test_exact_refined_reduce_matches_explicit_resampling():
    from geohalo import (  # noqa: PLC0415
        ReduceOperator,
        RestrictedOperator,
        reduce_with_restricted_operator,
        resample_grid,
    )

    lats, lons = np.arange(50., 71., 10), np.arange(0., 21., 10)
    data = _da(np.arange(9.).reshape(3, 3), lats, lons)
    geoms = gpd.GeoSeries([shapely.Polygon([(2, 52), (17, 60), (8, 68)])], index=["zone"])
    fine = resample_grid(data, target_resolution=5)
    expected = reduce(fine, geoms, partial_cell_weighting="exact")
    actual = reduce(data, geoms, target_resolution=5, partial_cell_weighting="exact")
    np.testing.assert_allclose(actual.values, expected.values, rtol=1e-14)
    stencil = Stencil.compute(fine.latitude.values, fine.longitude.values, geoms, partial_cell_weighting="exact")
    operator = ReduceOperator.compute(stencil, lats, lons)
    plan = RestrictedOperator.compute(operator, lats, 1, 1)
    np.testing.assert_allclose(reduce_with_restricted_operator(data, plan).values, expected.values, rtol=1e-14)
