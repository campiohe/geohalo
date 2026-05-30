import geopandas as gpd
import numpy as np
import pytest
import shapely

from geohalo.stencil import EmptyOverlapError, Stencil


@pytest.fixture
def grid_lats() -> np.ndarray:
    return np.array([-2.0, -1.0, 0.0, 1.0, 2.0])


@pytest.fixture
def grid_lons() -> np.ndarray:
    return np.array([-2.5, -1.5, -0.5, 0.5, 1.5, 2.5])


@pytest.fixture
def simple_geoms() -> gpd.GeoSeries:
    return gpd.GeoSeries(
        [shapely.box(-0.9, -0.4, -0.1, 0.4), shapely.box(-0.3, -0.3, 0.7, 0.7)],
        index=["sub", "multi"],
    )


def test_compute_shape(grid_lats, grid_lons, simple_geoms) -> None:
    s = Stencil.compute(grid_lats, grid_lons, simple_geoms)
    assert s.occupancy_matrix.shape == (2, 30)
    assert list(s.keys) == ["multi", "sub"]


def test_row_sums_positive(grid_lats, grid_lons, simple_geoms) -> None:
    s = Stencil.compute(grid_lats, grid_lons, simple_geoms)
    assert np.all(s.row_sums > 0)


def test_descending_lats_stored_ascending(grid_lons, simple_geoms) -> None:
    s = Stencil.compute(np.array([2.0, 1.0, 0.0, -1.0, -2.0]), grid_lons, simple_geoms)
    assert s.lats[0] < s.lats[-1]


def test_empty_overlap_raises(grid_lats, grid_lons) -> None:
    away = gpd.GeoSeries([shapely.box(100, 100, 101, 101)], index=["away"])
    with pytest.raises(EmptyOverlapError, match="away"):
        Stencil.compute(grid_lats, grid_lons, away)


def test_digest_includes_geoms(grid_lats, grid_lons, simple_geoms) -> None:
    s1 = Stencil.compute(grid_lats, grid_lons, simple_geoms)
    s2 = Stencil.compute(grid_lats, grid_lons, gpd.GeoSeries([shapely.box(0, 0, 1, 1)], index=["x"]))
    assert s1.digest != s2.digest


def test_input_order_invariant(grid_lats, grid_lons) -> None:
    a = shapely.box(-1, -1, 0, 0)
    b = shapely.box(0, 0, 1, 1)
    s1 = Stencil.compute(grid_lats, grid_lons, gpd.GeoSeries([a, b], index=["a", "b"]))
    s2 = Stencil.compute(grid_lats, grid_lons, gpd.GeoSeries([b, a], index=["b", "a"]))
    assert s1.digest == s2.digest


def test_not_geoseries_raises(grid_lats, grid_lons) -> None:
    with pytest.raises(TypeError, match="GeoSeries"):
        Stencil.compute(grid_lats, grid_lons, [shapely.box(0, 0, 1, 1)])


def test_empty_geoms_raises(grid_lats, grid_lons) -> None:
    with pytest.raises(ValueError, match="empty"):
        Stencil.compute(grid_lats, grid_lons, gpd.GeoSeries([]))


def test_irregular_grid_raises(grid_lons, simple_geoms) -> None:
    irregular_lats = np.array([-2.0, -1.0, 0.0, 1.0, 5.0])
    with pytest.raises(ValueError, match="regularly spaced"):
        Stencil.compute(irregular_lats, grid_lons, simple_geoms)
