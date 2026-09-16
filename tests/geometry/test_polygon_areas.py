from itertools import pairwise

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
import shapely
from scipy.integrate import quad

from geohalo import Stencil
from geohalo.geometry import EARTH_RADIUS_M, cell_areas, midpoint_edges, polygon_areas


def rectangle_area(west, south, east, north):
    # Stable difference-of-sines form, including narrow polar rectangles.
    band = 2 * np.cos(np.deg2rad((north + south) / 2)) * np.sin(np.deg2rad((north - south) / 2))
    return EARTH_RADIUS_M**2 * np.deg2rad(east - west) * band


@pytest.mark.parametrize("reverse", [False, True])
def test_every_cell_matches_spherical_cell_areas(reverse):
    lat = np.array([-70.0, -30.0, 0.0, 20.0, 60.0, 80.0])
    lon = np.array([-170.0, -50.0, 60.0, 150.0])
    y, x = midpoint_edges(lat), midpoint_edges(lon)
    boxes = [shapely.box(a, b, c, d) for b, d in pairwise(y) for a, c in pairwise(x)]
    if reverse:
        boxes = shapely.reverse(boxes)
    np.testing.assert_allclose(polygon_areas(boxes).reshape(len(lat), len(lon)), cell_areas(lat, lon), rtol=2e-14)


@pytest.mark.parametrize("width", [0.1, 20, 180, 300, 360])
@pytest.mark.parametrize(("bottom", "top"), [(-90, 90), (-90, -70), (-5, 5), (60, 90), (89.99999, 90)])
@pytest.mark.parametrize("shift", [-720, 0, 720])
def test_rectangles_do_not_wrap_or_choose_smaller_complement(width, bottom, top, shift):
    west = shift - width / 2
    east = shift + width / 2
    geom = shapely.box(west, bottom, east, top)
    np.testing.assert_allclose(polygon_areas([geom]), [rectangle_area(west, bottom, east, top)], rtol=3e-14)


def test_full_globe_and_hemisphere():
    geoms = [shapely.box(-180, -90, 180, 90), shapely.box(0, -90, 180, 90), shapely.box(0, 0, 360, 90)]
    np.testing.assert_allclose(polygon_areas(geoms), np.array([4, 2, 2]) * np.pi * EARTH_RADIUS_M**2, rtol=1e-15)


@pytest.mark.parametrize(("bottom", "height"), [(0, 60), (-60, 120), (45, 0.1), (80, 10), (89.99999, 0.00001)])
@pytest.mark.parametrize("width", [0.001, 60, 300])
def test_sloped_edges_match_independent_surface_quadrature(bottom, height, width):
    top = bottom + height
    # Use the represented height so rounding of bottom + height isn't part of the comparison.
    height = top - bottom
    triangle = shapely.Polygon([(0, bottom), (width, bottom), (width, top)])
    expected = EARTH_RADIUS_M**2 * np.deg2rad(width) * np.deg2rad(height) * quad(
        lambda t: (1 - t) * np.cos(np.deg2rad(bottom + t * height)), 0, 1,
    )[0]
    np.testing.assert_allclose(polygon_areas([triangle, shapely.reverse(triangle)]), expected, rtol=3e-9)


def test_analytic_triangle_is_not_endpoint_trapezoid():
    geom = shapely.Polygon([(0, 0), (60, 0), (60, 60)])
    np.testing.assert_allclose(polygon_areas([geom]), EARTH_RADIUS_M**2 / 2, rtol=1e-15)


@pytest.mark.parametrize("step", [0.1, 1, 10])
@pytest.mark.parametrize("reverse", [False, True])
def test_collinear_vertices_do_not_change_area(step, reverse):
    geom = shapely.Polygon([(-170, -80), (140, -50), (100, 70), (-60, 60)])
    subdivided = shapely.segmentize(geom, step)
    if reverse:
        subdivided = shapely.reverse(subdivided)
    np.testing.assert_allclose(polygon_areas([subdivided]), polygon_areas([geom]), rtol=3e-14)


@pytest.mark.parametrize("seed", range(5))
def test_general_polygon_matches_surface_integral(seed):
    rng = np.random.default_rng(seed)
    geom = shapely.MultiPoint(rng.uniform([-150, -70], [150, 70], size=(20, 2))).convex_hull
    ys = np.unique(shapely.get_coordinates(geom)[:, 1])

    def cross_section(y):
        line = shapely.LineString([(-180, y), (180, y)])
        return geom.intersection(line).length * np.cos(np.deg2rad(y))

    expected = quad(cross_section, ys[0], ys[-1], points=ys[1:-1], epsabs=1e-9)[0]
    expected *= EARTH_RADIUS_M**2 * np.deg2rad(1)**2
    np.testing.assert_allclose(polygon_areas([geom]), expected, rtol=2e-14)


@pytest.mark.parametrize("reverse_shell", [False, True])
@pytest.mark.parametrize("reverse_hole", [False, True])
def test_holes_and_nested_collections(reverse_shell, reverse_hole):
    shell = shapely.box(-20, -30, 30, 70)
    hole = shapely.Polygon([(0, 0), (10, 0), (10, 60)])
    shell_ring = list(shell.exterior.coords)[::-1 if reverse_shell else 1]
    hole_ring = list(hole.exterior.coords)[::-1 if reverse_hole else 1]
    polygon = shapely.Polygon(shell_ring, [hole_ring])
    separate = shapely.box(90, 0, 100, 10)
    multi = shapely.MultiPolygon([polygon, separate])
    nested = shapely.GeometryCollection([
        shapely.GeometryCollection([multi, shapely.Point(0, 0)]), shapely.LineString([(0, 0), (1, 1)]),
    ])
    a, b, c = polygon_areas([shell, hole, separate])
    np.testing.assert_allclose(polygon_areas([polygon, multi, nested]), [a - b, a - b + c, a - b + c], rtol=1e-14)


def test_collection_areas_are_additive_not_unioned():
    box = shapely.box(0, 0, 1, 1)
    np.testing.assert_allclose(polygon_areas([shapely.GeometryCollection([box, box])]), 2 * polygon_areas([box]))


@pytest.mark.parametrize("geom", [
    shapely.Polygon(), shapely.MultiPolygon(), shapely.GeometryCollection(), shapely.Point(),
    shapely.Point(1, 2), shapely.LineString([(0, 0), (1, 1)]), shapely.MultiPoint([(1, 2), (3, 4)]),
    shapely.MultiLineString([[(0, 0), (1, 1)]]), shapely.LinearRing([(0, 0), (1, 0), (1, 1), (0, 0)]),
])
def test_empty_and_zero_area_geometries(geom):
    np.testing.assert_array_equal(polygon_areas([geom]), [0.0])


def test_missing_is_nan_and_z_is_ignored():
    xyz = shapely.Polygon([(0, 0, 5), (60, 0, 10), (60, 60, -9)])
    out = polygon_areas([None, xyz])
    assert np.isnan(out[0])
    np.testing.assert_allclose(out[1], EARTH_RADIUS_M**2 / 2)


@pytest.mark.parametrize("crs", [None, "EPSG:4326", "OGC:CRS84"])
def test_geoseries_order_and_duplicate_labels(crs):
    geoms = [shapely.box(0, 0, 3, 3), None, shapely.box(0, 0, 1, 1)]
    series = gpd.GeoSeries(geoms, index=pd.Index(["z", "a", "z"], name="zone"), crs=crs)
    out = polygon_areas(series)
    assert isinstance(out, np.ndarray)
    assert out.dtype == np.float64
    assert out.shape == (3,)
    np.testing.assert_array_equal(out, polygon_areas(geoms))
    np.testing.assert_array_equal(polygon_areas(tuple(geoms)), out)
    np.testing.assert_array_equal(polygon_areas(np.asarray(geoms, dtype=object)), out)


@pytest.mark.parametrize("geoms", [[], (), np.array([], dtype=object), gpd.GeoSeries([], crs=4326)])
def test_empty_input(geoms):
    out = polygon_areas(geoms)
    assert out.shape == (0,)
    assert out.dtype == np.float64


@pytest.mark.parametrize("crs", ["EPSG:3857", "EPSG:4269"])
def test_no_implicit_crs_conversion(crs):
    with pytest.raises(ValueError, match="EPSG:4326"):
        polygon_areas(gpd.GeoSeries([shapely.box(0, 0, 1, 1)], crs=crs))


@pytest.mark.parametrize("value", [None, shapely.box(0, 0, 1, 1), [[shapely.Point(0, 0)]], np.empty((2, 0))])
def test_requires_one_dimensional_sequence(value):
    with pytest.raises(ValueError, match="one-dimensional"):
        polygon_areas(value)


@pytest.mark.parametrize("value", [1, "POLYGON EMPTY", np.nan, pd.NA, [0, 1]])
def test_rejects_non_geometries(value):
    arr = np.empty(2, dtype=object)
    arr[:] = [shapely.Point(0, 0), value]
    with pytest.raises(TypeError, match=r"geoms\[1\]"):
        polygon_areas(arr)


@pytest.mark.parametrize(("geom", "message"), [
    (shapely.box(0, -91, 1, 0), "latitudes"),
    (shapely.box(0, 0, 1, 91), "latitudes"),
    (shapely.Point(float("inf"), 0), "finite coordinates"),
    (shapely.Point(float("nan"), 0), "finite coordinates"),
    (shapely.box(0, -90, 361, 90), "360 degrees"),
    (shapely.Polygon([(0, 0), (1, 1), (1, 0), (0, 1)]), "invalid"),
    (shapely.Polygon([(0, 0), (1, 0), (2, 0)]), "invalid"),
    (shapely.MultiPolygon([shapely.box(0, 0, 2, 2), shapely.box(1, 1, 3, 3)]), "invalid"),
])
def test_invalid_geometries_fail_explicitly(geom, message):
    with pytest.raises(ValueError, match=message):
        polygon_areas([geom])


def test_hole_outside_shell_is_not_silently_clamped():
    geom = shapely.Polygon(shapely.box(0, 0, 1, 1).exterior.coords, [shapely.box(2, 2, 3, 3).exterior.coords])
    with pytest.raises(ValueError, match="invalid"):
        polygon_areas([geom])


def test_antimeridian_requires_explicit_unwrapping_or_splitting():
    unwrapped = shapely.box(170, -10, 190, 10)
    split = shapely.MultiPolygon([shapely.box(170, -10, 180, 10), shapely.box(-180, -10, -170, 10)])
    wide = shapely.box(-170, -10, 170, 10)
    narrow_area = polygon_areas([unwrapped])[0]
    np.testing.assert_allclose(polygon_areas([split, wide]), [narrow_area, 17 * narrow_area], rtol=1e-15)


def test_grid_aligned_stencil_area_agrees():
    lat, lon = np.arange(30, 71, 10), np.arange(0, 41, 10)
    geoms = gpd.GeoSeries([shapely.box(-5, 25, 45, 75), shapely.box(5, 45, 25, 65)])
    stencil = Stencil.compute(lat, lon, geoms)
    np.testing.assert_allclose(stencil.row_sums, polygon_areas(geoms), rtol=1e-14)


def test_partial_cell_stencil_area_is_only_an_approximation():
    lat, lon = np.array([30, 60]), np.array([0, 30])
    geoms = gpd.GeoSeries([shapely.box(-15, 15, 15, 30)])
    stencil = Stencil.compute(lat, lon, geoms)
    expected_stencil = cell_areas(lat, lon)[0, 0] / 2
    np.testing.assert_allclose(stencil.row_sums, [expected_stencil])
    assert polygon_areas(geoms)[0] > expected_stencil * 1.05
