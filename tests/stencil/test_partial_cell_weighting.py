"""Opt-in spherical intersections preserve area across cell boundaries."""

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
import shapely
from shapely.affinity import translate

from geohalo import EmptyOverlapError, Stencil
from geohalo.geometry import EARTH_RADIUS_M, cell_areas, polygon_areas
from geohalo.stencil import stencil_digest


def _rectangle_area(west, south, east, north):
    return EARTH_RADIUS_M**2 * np.deg2rad(east - west) * (
        np.sin(np.deg2rad(north)) - np.sin(np.deg2rad(south))
    )


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("descending", [False, True])
def test_rectangular_partial_weights_match_spherical_integrals(dtype, descending):
    lats, lons = np.array([60., 70.]), np.array([0., 10.])
    geoms = gpd.GeoSeries([shapely.box(-3, 60, 12, 72)], index=["zone"])
    stencil = Stencil.compute(lats[::-1] if descending else lats, lons, geoms,
                              partial_cell_weighting="exact", dtype=dtype)
    expected = np.array([_rectangle_area(w, s, e, n)
                         for s, n in [(60, 65), (65, 72)] for w, e in [(-3, 5), (5, 12)]])
    tolerance = 1e-7 if dtype == np.float32 else 1e-14
    np.testing.assert_allclose(stencil.occupancy_matrix.toarray()[0], expected, rtol=tolerance)
    np.testing.assert_allclose(stencil.row_sums, [_rectangle_area(-3, 60, 12, 72)], rtol=tolerance)
    assert stencil.occupancy_matrix.dtype == dtype
    assert stencil.row_sums.dtype == np.float64
    assert stencil.partial_cell_weighting == "exact"
    assert stencil.digest == stencil_digest(lats, lons, geoms, partial_cell_weighting="exact", dtype=dtype)


@pytest.mark.parametrize("shift", [0, 180])
def test_complex_polygon_area_is_preserved_with_clipping_and_row_order(shift):
    lats, lons = np.arange(50., 71., 10), np.arange(0., 21., 10) + shift
    geometries = [
        shapely.Polygon([(-8, 44), (24, 49), (18, 78), (-8, 44)]),
        shapely.box(-3, 47, 24, 74).difference(shapely.box(2, 52, 18, 68)),
        shapely.MultiPolygon([shapely.box(-8, 44, 2, 53), shapely.box(12, 63, 28, 78)]),
        shapely.Polygon([(0, 50, 3), (12, 54, 4), (7, 68, 5)]),
    ]
    geometries = [translate(geom, xoff=shift) for geom in geometries]
    keys = pd.MultiIndex.from_tuples([("z", 2), ("a", 1), ("z", 2), ("b", 0)], names=["zone", "part"])
    geoms = gpd.GeoSeries(geometries, index=keys, crs="EPSG:4326")
    footprint = shapely.box(-5 + shift, 45, 25 + shift, 75)
    expected = polygon_areas(geoms.intersection(footprint))
    stencil = Stencil.compute(lats, lons, geoms, partial_cell_weighting="exact")
    pd.testing.assert_index_equal(stencil.keys, keys)
    np.testing.assert_allclose(stencil.row_sums, expected, rtol=2e-14)
    assert np.all(stencil.occupancy_matrix.data >= 0)
    assert np.all(stencil.occupancy_matrix.toarray() <= cell_areas(lats, lons).ravel() * (1 + 1e-14))


def test_aligned_refinement_does_not_change_exact_area():
    geoms = gpd.GeoSeries([shapely.Polygon([(-4, 56), (12, 62), (8, 74), (-4, 56)])])
    coarse = Stencil.compute(np.array([60., 70.]), np.array([0., 10.]), geoms, partial_cell_weighting="exact")
    fine = Stencil.compute(np.arange(56.25, 75, 2.5), np.arange(-3.75, 15, 2.5), geoms,
                          partial_cell_weighting="exact")
    # Sum each 4x4 group of children back into its coarse parent.
    regrouped = fine.occupancy_matrix.toarray().reshape(2, 4, 2, 4).sum(axis=(1, 3))
    np.testing.assert_allclose(regrouped.ravel(), coarse.occupancy_matrix.toarray()[0], rtol=1e-14)


def test_full_cells_skip_intersection_but_tiny_holes_are_not_rounded_away(monkeypatch):
    lats, lons = np.array([60., 70.]), np.array([0., 10.])
    whole = shapely.box(-5, 55, 15, 75)
    hole = shapely.box(0, 60, 0.001, 60.001)
    geoms = gpd.GeoSeries([whole, whole.difference(hole)])
    approximate = Stencil.compute(lats, lons, geoms)
    # exactextract's coverage fractions round this nearly-full cell to 1.
    np.testing.assert_array_equal(approximate.occupancy_matrix.toarray()[0],
                                  approximate.occupancy_matrix.toarray()[1])
    intersect = shapely.intersection
    sizes = []

    def count_intersections(geom, cells):
        sizes.append(len(cells))
        return intersect(geom, cells)

    monkeypatch.setattr(shapely, "intersection", count_intersections)
    exact = Stencil.compute(lats, lons, geoms, partial_cell_weighting="exact")
    assert sum(sizes) == 1
    removed = exact.row_sums[0] - exact.row_sums[1]
    np.testing.assert_allclose(removed, polygon_areas([hole])[0], rtol=2e-7)


def test_pole_centered_cells_integrate_only_physical_intersection():
    lats, lons = np.array([88., 90.]), np.array([0., 2.])
    geoms = gpd.GeoSeries([shapely.box(-1, 89, 3, 90)])
    stencil = Stencil.compute(lats, lons, geoms, partial_cell_weighting="exact")
    np.testing.assert_allclose(stencil.row_sums, [_rectangle_area(-1, 89, 3, 90)], rtol=1e-12)


@pytest.mark.parametrize("mode", ["approximate", "exact"])
def test_outside_and_touching_only_polygons_raise(mode):
    for geom in (shapely.box(15, 55, 20, 75), shapely.box(20, 55, 30, 75)):
        with pytest.raises(EmptyOverlapError) as caught:
            Stencil.compute(np.array([60., 70.]), np.array([0., 10.]),
                            gpd.GeoSeries([geom], index=["outside"]), partial_cell_weighting=mode)
        assert caught.value.geom_key == "outside"


@pytest.mark.parametrize("geom", [shapely.Polygon(), shapely.Point(0, 60), shapely.LineString([(0, 60), (1, 61)])])
def test_exact_zero_area_geometry_raises(geom):
    with pytest.raises(EmptyOverlapError) as caught:
        Stencil.compute(np.array([60., 70.]), np.array([0., 10.]), gpd.GeoSeries([geom], index=["empty"]),
                        partial_cell_weighting="exact")
    assert caught.value.geom_key == "empty"


@pytest.mark.parametrize("kwargs", [
    {"partial_cell_weighting": "unknown"}, {"partial_cell_weighting": None},
    {"partial_cell_weighting": "exact", "spherical_correction": False},
])
def test_invalid_weighting_is_rejected_by_compute_and_digest(kwargs):
    lats = np.array([0., 1.])
    geoms = gpd.GeoSeries([shapely.box(0, 0, 1, 1)])
    for build in (Stencil.compute, stencil_digest):
        with pytest.raises(ValueError, match="partial_cell_weighting"):
            build(lats, lats, geoms, **kwargs)


@pytest.mark.parametrize(("geom", "crs", "message"), [
    (shapely.box(0, 0, 1, 1), "EPSG:3857", "EPSG:4326"),
    (shapely.box(0, 89, 1, 91), None, "latitudes"),
    (shapely.box(-181, 0, 181, 1), None, "360 degrees"),
    (shapely.Polygon([(0, 0), (1, 1), (1, 0), (0, 1)]), None, "invalid"),
])
def test_exact_validates_geometry_before_extraction(geom, crs, message, monkeypatch):
    def no_extract(*_args, **_kwargs):
        pytest.fail("invalid geometries must fail before native extraction")

    monkeypatch.setattr("geohalo.stencil.exact_extract", no_extract)
    with pytest.raises(ValueError, match=message):
        Stencil.compute(np.array([0., 1.]), np.array([0., 1.]), gpd.GeoSeries([geom], crs=crs),
                        partial_cell_weighting="exact")
