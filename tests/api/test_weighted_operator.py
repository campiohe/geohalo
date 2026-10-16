"""Operators weighted by a fine lattice of counts, reweighted by cell, and stacked by rows."""

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
import shapely
import xarray as xr

import geohalo as ghl
from geohalo.geometry import cell_areas

PIXEL = 1 / 120


def _grid(south, north, west, east, step):
    """Cell centres from cell edges, latitude north to south as the store keeps it."""
    lat = np.arange(south + step / 2, north, step)[::-1].copy()
    lon = np.arange(west + step / 2, east, step)
    return lat, lon


def _zones():
    return gpd.GeoSeries(
        [shapely.Point(5.0, 45.0).buffer(1.3), shapely.Polygon([(3.1, 43.2), (6.9, 44.1), (4.6, 46.8)])],
        index=pd.Index(["round", "triangle"], name="zone"), crs="EPSG:4326",
    )


def _window(south, west, rows, cols, *, descending=True):
    lat = south + (np.arange(rows) + 0.5) * PIXEL
    lon = west + (np.arange(cols) + 0.5) * PIXEL
    return (lat[::-1].copy() if descending else lat), lon


def _pixel_areas(lat, lon):
    ascending = lat[0] < lat[-1]
    areas = cell_areas(lat if ascending else lat[::-1], lon)
    return areas if ascending else areas[::-1].copy()


GRID = _grid(40.0, 50.0, 0.0, 10.0, 0.25)
WINDOW = _window(42.0, 2.0, 720, 720)  # 42..48 N, 2..8 E


def test_concat_of_one_operator_is_that_operator():
    op = ghl.ReduceOperator.compute(ghl.Stencil.compute(*GRID, _zones()), *GRID)
    assert ghl.ReduceOperator.concat([op]) is op


def test_concat_stacks_rows_in_order():
    first = ghl.ReduceOperator.compute(ghl.Stencil.compute(*GRID, _zones()), *GRID, dtype=np.float32)
    other = _zones().set_axis(pd.Index(["x", "y"], name="zone"))
    second = ghl.ReduceOperator.compute(ghl.Stencil.compute(*GRID, other), *GRID, dtype=np.float32)
    stacked = ghl.ReduceOperator.concat([first, second])
    assert list(stacked.keys) == ["round", "triangle", "x", "y"]
    assert stacked.keys.name == "zone"
    assert stacked.matrix.dtype == np.float32
    np.testing.assert_array_equal(stacked.matrix.toarray(), np.vstack([first.matrix.toarray(),
                                                                       second.matrix.toarray()]))
    np.testing.assert_array_equal(stacked.row_sums, np.r_[first.row_sums, second.row_sums])
    assert stacked.digest not in (first.digest, second.digest)


def test_concat_refuses_what_does_not_stack():
    stencil = ghl.Stencil.compute(*GRID, _zones())
    op = ghl.ReduceOperator.compute(stencil, *GRID)
    with pytest.raises(ValueError, match="at least one"):
        ghl.ReduceOperator.concat([])
    with pytest.raises(ValueError, match="repeat"):
        ghl.ReduceOperator.concat([op, op])
    other = _zones().set_axis(pd.Index(["a", "b"], name="zone"))
    with pytest.raises(ValueError, match="dtype"):
        ghl.ReduceOperator.concat([op, ghl.ReduceOperator.compute(
            ghl.Stencil.compute(*GRID, other), *GRID, dtype=np.float32)])
    elsewhere = _grid(40.0, 50.0, 0.0, 10.0, 0.5)
    with pytest.raises(ValueError, match="source grid"):
        ghl.ReduceOperator.concat([op, ghl.ReduceOperator.compute(ghl.Stencil.compute(*GRID, other), *elsewhere)])
    with pytest.raises(ValueError, match="iteration"):
        ghl.ReduceOperator.concat([op, ghl.ReduceOperator.compute(
            ghl.Stencil.compute(*GRID, other), *GRID, iterations=2)])
    unnamed = _zones().set_axis(pd.Index(["c", "d"]))
    with pytest.raises(ValueError, match="key names"):
        ghl.ReduceOperator.concat([op, ghl.ReduceOperator.compute(ghl.Stencil.compute(*GRID, unnamed), *GRID)])


def test_reweight_scales_columns_and_drops_cells_weighing_nothing():
    op = ghl.ReduceOperator.compute(ghl.Stencil.compute(*GRID, _zones()), *GRID)
    weights = np.random.default_rng(3).random((GRID[0].size, GRID[1].size))
    weights[10:20] = np.nan
    scaled = op.reweight(weights[::-1])
    stored = op.reweight(weights, descending=True)
    np.testing.assert_array_equal(scaled.matrix.toarray(), stored.matrix.toarray())
    assert scaled.digest == stored.digest
    expected = op.matrix.toarray() * np.nan_to_num(weights[::-1]).ravel()
    np.testing.assert_allclose(scaled.matrix.toarray(), expected, rtol=1e-15)
    assert not np.any(scaled.matrix.data == 0)
    np.testing.assert_allclose(scaled.row_sums, expected.sum(axis=1), rtol=1e-12)
    assert scaled.keys.equals(op.keys)
    assert scaled.iterations == op.iterations


def test_reweight_keeps_float32_coefficients_with_float64_row_sums():
    op = ghl.ReduceOperator.compute(ghl.Stencil.compute(*GRID, _zones()), *GRID, dtype=np.float32)
    scaled = op.reweight(np.full((GRID[0].size, GRID[1].size), 3.0))
    assert scaled.matrix.dtype == np.float32
    assert scaled.row_sums.dtype == np.float64
    np.testing.assert_array_equal(scaled.row_sums, scaled.matrix.astype(np.float64).sum(axis=1).A1)


def test_reweight_digest_ignores_cells_the_operator_never_reads():
    op = ghl.ReduceOperator.compute(ghl.Stencil.compute(*GRID, _zones()), *GRID)
    weights = np.ones((GRID[0].size, GRID[1].size))
    elsewhere = weights.copy()
    elsewhere[0, 0] = 5.0
    read = weights.copy()
    read[GRID[0].size // 2, GRID[1].size // 2] = 5.0
    assert op.reweight(weights).digest == op.reweight(elsewhere).digest
    assert op.reweight(weights).digest != op.reweight(read).digest
    assert op.reweight(weights).digest != op.digest


def test_reweight_refuses_a_wrong_shape_and_negative_weights():
    op = ghl.ReduceOperator.compute(ghl.Stencil.compute(*GRID, _zones()), *GRID)
    with pytest.raises(ValueError, match="shape"):
        op.reweight(np.ones((3, 3)))
    with pytest.raises(ValueError, match="negative or infinite"):
        op.reweight(-np.ones((GRID[0].size, GRID[1].size)))
    with pytest.raises(TypeError, match="real numbers"):
        op.reweight(np.ones((GRID[0].size, GRID[1].size), dtype=bool))


def test_reweight_by_density_is_the_weight_key_mean():
    stencil = ghl.Stencil.compute(*GRID, _zones())
    rng = np.random.default_rng(4)
    density = rng.random((GRID[0].size, GRID[1].size))
    field = rng.normal(size=(3, GRID[0].size, GRID[1].size))
    grid = xr.Dataset(
        {"t": (("step", "latitude", "longitude"), field), "w": (("latitude", "longitude"), density)},
        coords={"latitude": GRID[0], "longitude": GRID[1], "step": [0, 1, 2]},
    )
    expected = ghl.reduce_with_stencil(grid[["t", "w"]], stencil, weight_key="w")["t"]
    op = ghl.ReduceOperator.compute(stencil, *GRID).reweight(density, descending=True)
    got = op.apply_grid(field, descending=True, how="mean")
    np.testing.assert_allclose(got, expected.transpose("step", "geom").to_numpy(), rtol=1e-12)


def test_uniform_counts_reproduce_the_planar_coverage_of_every_cell():
    lat, lon = WINDOW
    op = ghl.ReduceOperator.compute_weighted(_zones(), lat, lon, np.ones((lat.size, lon.size)), *GRID)
    planar = ghl.Stencil.compute(GRID[0], GRID[1], _zones(), spherical_correction=False)
    np.testing.assert_allclose(
        (op.matrix / op.row_sums[:, None]).toarray(),
        (planar.occupancy_matrix / planar.row_sums[:, None]).toarray(), atol=1e-12,
    )


def test_counts_proportional_to_pixel_area_reproduce_compute():
    lat, lon = WINDOW
    op = ghl.ReduceOperator.compute_weighted(_zones(), lat, lon, _pixel_areas(lat, lon), *GRID)
    stencil = ghl.Stencil.compute(GRID[0], GRID[1], _zones())
    reference = ghl.ReduceOperator.compute(stencil, *GRID)
    weighted, unweighted = op.matrix.toarray(), reference.matrix.toarray()
    np.testing.assert_array_equal(weighted > 0, unweighted > 0)
    full = np.isclose(ghl.Stencil.compute(GRID[0], GRID[1], _zones(), spherical_correction=False)
                      .occupancy_matrix.toarray(), 1.0)
    np.testing.assert_allclose(weighted[full], unweighted[full], rtol=1e-12)
    partial = (unweighted > 0) & ~full
    bound = np.deg2rad(0.25) / 2 * np.tan(np.deg2rad(48.0))
    assert np.max(np.abs(weighted[partial] / unweighted[partial] - 1)) < bound
    np.testing.assert_allclose(op.row_sums, reference.row_sums, rtol=1e-5)


def _split():
    """The south-western 0.25 degree cell of four, split by two zones at 5.125 E, everyone in its western half."""
    lat, lon = _window(45.0, 5.0, 60, 60)
    people = np.zeros((lat.size, lon.size))
    people[np.ix_(lat < 45.25, lon < 5.125)] = 10.0
    halves = gpd.GeoSeries([shapely.box(5.0, 45.0, 5.125, 45.25), shapely.box(5.125, 45.0, 5.25, 45.25)],
                           index=["west", "east"])
    return halves, lat, lon, people, _grid(45.0, 45.5, 5.0, 5.5, 0.25)


def test_a_split_cell_counts_only_the_people_on_each_side():
    halves, lat, lon, people, grid = _split()
    values = np.array([[1.0, 2.0], [7.0, 3.0]])  # north to south; the split cell holds 7

    exact = ghl.ReduceOperator.compute_weighted(halves, lat, lon, people, *grid)
    np.testing.assert_allclose(exact.row_sums, [4500.0, 0.0])
    np.testing.assert_allclose(exact.apply_grid(values, descending=True, how="mean", skipna=True), [7.0, np.nan])

    unweighted = ghl.ReduceOperator.compute(ghl.Stencil.compute(*grid, halves), *grid)
    totals = np.array([[0.0, 0.0], [people.sum(), 0.0]])
    areas = cell_areas(grid[0][::-1], grid[1])[::-1]
    by_cell = unweighted.reweight(totals / areas, descending=True)
    np.testing.assert_allclose(by_cell.row_sums, [2250.0, 2250.0], rtol=1e-6)
    np.testing.assert_allclose(by_cell.apply_grid(values, descending=True, how="mean", skipna=True), [7.0, 7.0])


def test_cell_and_exact_agree_where_every_border_cell_is_empty():
    lat, lon = WINDOW
    zone = gpd.GeoSeries([shapely.box(4.1, 44.1, 5.9, 45.9)], index=["inner"])
    people = np.random.default_rng(1).gamma(0.5, 20, size=(lat.size, lon.size))
    inside = np.add.outer((lat > 44.25) & (lat < 45.75), (lon > 4.25) & (lon < 5.75)) == 2
    people[~inside] = 0.0
    exact = ghl.ReduceOperator.compute_weighted(zone, lat, lon, people, *GRID, dtype=np.float32)
    unweighted = ghl.ReduceOperator.compute(ghl.Stencil.compute(*GRID, zone), *GRID, dtype=np.float32)
    rows = np.searchsorted(-GRID[0], -lat)  # stored cell row of each pixel
    cols = np.searchsorted(GRID[1] + 0.125, lon)
    per_cell = np.zeros((GRID[0].size, GRID[1].size))
    np.add.at(per_cell, (rows[:, None], cols[None, :]), people)
    areas = cell_areas(GRID[0][::-1], GRID[1])[::-1]
    by_cell = unweighted.reweight(per_cell / areas, descending=True)
    np.testing.assert_allclose(by_cell.matrix.toarray(), exact.matrix.toarray(), rtol=2e-7)


def test_missing_weights_count_as_nobody():
    lat, lon = WINDOW
    people = np.random.default_rng(2).random((lat.size, lon.size))
    people[::7, ::5] = 0.0
    holes = people.copy()
    holes[::7, ::5] = np.nan
    one = ghl.ReduceOperator.compute_weighted(_zones(), lat, lon, people, *GRID)
    other = ghl.ReduceOperator.compute_weighted(_zones(), lat, lon, holes, *GRID)
    np.testing.assert_array_equal(one.matrix.toarray(), other.matrix.toarray())
    assert one.digest == other.digest


@pytest.mark.parametrize("bad", [-1.0, np.inf])
def test_negative_or_infinite_weights_are_refused(bad):
    lat, lon = WINDOW
    people = np.ones((lat.size, lon.size))
    people[3, 4] = bad
    with pytest.raises(ValueError, match="negative or infinite"):
        ghl.ReduceOperator.compute_weighted(_zones(), lat, lon, people, *GRID)


def test_a_source_edge_between_pixel_edges_is_refused():
    lat, lon = WINDOW
    with pytest.raises(ValueError, match="not a pixel edge"):
        ghl.ReduceOperator.compute_weighted(_zones(), lat, lon, np.ones((lat.size, lon.size)),
                                            GRID[0] + 0.003, GRID[1])


def test_a_source_grid_finer_than_the_pixels_is_refused():
    lat, lon = WINDOW
    fine = _grid(40.0, 50.0, 0.0, 10.0, 0.005)
    with pytest.raises(ValueError, match="not a pixel edge"):
        ghl.ReduceOperator.compute_weighted(_zones(), lat, lon, np.ones((lat.size, lon.size)), *fine)


def test_a_geometry_leaving_the_window_is_refused():
    lat, lon = WINDOW
    wide = gpd.GeoSeries([shapely.box(1.5, 44.0, 3.0, 45.0)], index=["wide"])
    with pytest.raises(ValueError, match="'wide' leaves the weight window"):
        ghl.ReduceOperator.compute_weighted(wide, lat, lon, np.ones((lat.size, lon.size)), *GRID)


def test_a_geometry_off_the_source_grid_does_not_overlap_it():
    lat, lon = WINDOW
    grid = _grid(40.0, 50.0, 6.0, 10.0, 0.25)
    west = gpd.GeoSeries([shapely.box(2.5, 44.0, 3.5, 45.0)], index=["west"])
    with pytest.raises(ghl.EmptyOverlapError, match="'west'"):
        ghl.ReduceOperator.compute_weighted(west, lat, lon, np.ones((lat.size, lon.size)), *grid)
    beyond = _grid(40.0, 50.0, 8.0, 10.0, 0.25)
    with pytest.raises(ghl.EmptyOverlapError, match="'west'"):
        ghl.ReduceOperator.compute_weighted(west, lat, lon, np.ones((lat.size, lon.size)), *beyond)


def test_a_geometry_over_nobody_has_a_row_of_zeros():
    lat, lon = WINDOW
    people = np.ones((lat.size, lon.size))
    people[:, lon < 5.0] = 0.0
    zones = gpd.GeoSeries([shapely.box(2.5, 44.0, 4.5, 45.0), shapely.box(5.5, 44.0, 6.5, 45.0)],
                          index=["empty", "lived"])
    op = ghl.ReduceOperator.compute_weighted(zones, lat, lon, people, *GRID)
    assert op.matrix[0].nnz == 0
    assert op.row_sums[0] == 0.0
    assert op.row_sums[1] > 0.0
    values = np.ones((GRID[0].size, GRID[1].size))
    out = op.apply_grid(values, descending=True, how="mean", skipna=True)
    assert np.isnan(out[0])
    assert out[1] == pytest.approx(1.0)
    restricted = ghl.RestrictedOperator.compute(op, GRID[0], 8, 8)
    assert restricted.matrix[0].nnz == 0


def test_a_window_and_a_grid_read_in_either_latitude_order_give_one_operator():
    north_first = ghl.ReduceOperator.compute_weighted(
        _zones(), *WINDOW, _pixel_areas(*WINDOW), *GRID)
    lat, lon = _window(42.0, 2.0, 720, 720, descending=False)
    south_first = ghl.ReduceOperator.compute_weighted(
        _zones(), lat, lon, _pixel_areas(lat, lon), GRID[0][::-1], GRID[1])
    np.testing.assert_array_equal(north_first.matrix.toarray(), south_first.matrix.toarray())
    np.testing.assert_array_equal(north_first.source_lat, south_first.source_lat)
    assert north_first.digest == south_first.digest


def test_float32_coefficients_keep_float64_row_sums_of_what_is_stored():
    op = ghl.ReduceOperator.compute_weighted(
        _zones(), *WINDOW, _pixel_areas(*WINDOW), *GRID, dtype=np.float32)
    assert op.matrix.dtype == np.float32
    assert op.row_sums.dtype == np.float64
    np.testing.assert_array_equal(op.row_sums, op.matrix.astype(np.float64).sum(axis=1).A1)


def test_the_digest_covers_the_weights_and_the_dtype():
    people = _pixel_areas(*WINDOW)
    base = ghl.ReduceOperator.compute_weighted(_zones(), *WINDOW, people, *GRID)
    more = people.copy()
    more[100, 100] *= 2
    assert ghl.ReduceOperator.compute_weighted(_zones(), *WINDOW, more, *GRID).digest != base.digest
    assert ghl.ReduceOperator.compute_weighted(
        _zones(), *WINDOW, people, *GRID, dtype=np.float32).digest != base.digest
    stencil = ghl.Stencil.compute(*GRID, _zones())
    assert base.digest != ghl.ReduceOperator.compute(stencil, *GRID).digest


def test_batches_stacked_in_order_are_the_operator_built_at_once():
    lat, lon = WINDOW
    people = _pixel_areas(lat, lon) * np.random.default_rng(5).random((lat.size, lon.size))
    zones = _zones()
    whole = ghl.ReduceOperator.compute_weighted(zones, lat, lon, people, *GRID, dtype=np.float32)
    parts = []
    for key in zones.index:
        one = zones.loc[[key]]
        minx, miny, maxx, maxy = one.total_bounds
        rows = (lat > miny - 0.25) & (lat < maxy + 0.25)
        cols = (lon > minx - 0.25) & (lon < maxx + 0.25)
        parts.append(ghl.ReduceOperator.compute_weighted(
            one, lat[rows], lon[cols], people[np.ix_(rows, cols)], *GRID, dtype=np.float32))
    stacked = ghl.ReduceOperator.concat(parts)
    np.testing.assert_array_equal(stacked.matrix.toarray(), whole.matrix.toarray())
    np.testing.assert_array_equal(stacked.row_sums, whole.row_sums)
    pd.testing.assert_index_equal(stacked.keys, whole.keys)
    assert stacked.matrix.dtype == np.float32


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_weighted_reweighted_and_stacked_operators_round_trip_through_npz(dtype):
    weighted = ghl.ReduceOperator.compute_weighted(
        _zones(), *WINDOW, _pixel_areas(*WINDOW), *GRID, dtype=dtype)
    reweighted = weighted.reweight(np.full((GRID[0].size, GRID[1].size), 2.0))
    other = ghl.ReduceOperator.compute_weighted(
        _zones().set_axis(pd.Index(["x", "y"], name="zone")), *WINDOW, _pixel_areas(*WINDOW), *GRID, dtype=dtype)
    for op in (weighted, reweighted, ghl.ReduceOperator.concat([weighted, other])):
        again = ghl.ReduceOperator.from_npz(op.to_npz())
        assert again.matrix.dtype == op.matrix.dtype
        for name in ("data", "indices", "indptr"):
            np.testing.assert_array_equal(getattr(again.matrix, name), getattr(op.matrix, name))
        np.testing.assert_array_equal(again.row_sums, op.row_sums)
        pd.testing.assert_index_equal(again.keys, op.keys, exact=True)
        assert again.digest == op.digest
        assert again.iterations == op.iterations


def test_weights_of_another_kind_or_a_westward_window_are_refused():
    lat, lon = WINDOW
    people = np.ones((lat.size, lon.size))
    with pytest.raises(TypeError, match="real numbers"):
        ghl.ReduceOperator.compute_weighted(_zones(), lat, lon, people.astype(bool), *GRID)
    with pytest.raises(ValueError, match="shape"):
        ghl.ReduceOperator.compute_weighted(_zones(), lat, lon, people[1:], *GRID)
    with pytest.raises(ValueError, match="at least two pixels"):
        ghl.ReduceOperator.compute_weighted(_zones(), lat[:1], lon, people[:1], *GRID)
    with pytest.raises(ValueError, match="weight_lon must ascend"):
        ghl.ReduceOperator.compute_weighted(_zones(), lat, lon[::-1], people[:, ::-1], *GRID)
    with pytest.raises(ValueError, match="source_lon must ascend"):
        ghl.ReduceOperator.compute_weighted(_zones(), lat, lon, people, GRID[0], GRID[1][::-1])


def test_integer_counts_are_read_as_float64():
    lat, lon = WINDOW
    people = np.random.default_rng(6).integers(0, 50, size=(lat.size, lon.size))
    as_int = ghl.ReduceOperator.compute_weighted(_zones(), lat, lon, people, *GRID)
    as_float = ghl.ReduceOperator.compute_weighted(_zones(), lat, lon, people.astype(np.float64), *GRID)
    np.testing.assert_array_equal(as_int.matrix.toarray(), as_float.matrix.toarray())
    assert as_int.digest == as_float.digest


def test_geometries_that_are_not_polygons_in_degrees_are_refused():
    lat, lon = WINDOW
    people = np.ones((lat.size, lon.size))
    with pytest.raises(TypeError, match="GeoSeries"):
        ghl.ReduceOperator.compute_weighted(list(_zones()), lat, lon, people, *GRID)
    with pytest.raises(ValueError, match="empty"):
        ghl.ReduceOperator.compute_weighted(_zones().iloc[:0], lat, lon, people, *GRID)
    with pytest.raises(ValueError, match="missing"):
        ghl.ReduceOperator.compute_weighted(gpd.GeoSeries([None], index=["gap"]), lat, lon, people, *GRID)
    with pytest.raises(ValueError, match="EPSG:4326"):
        ghl.ReduceOperator.compute_weighted(_zones().set_crs("EPSG:3857", allow_override=True),
                                            lat, lon, people, *GRID)
