"""A weighted build over a ten-million-pixel window holds the window and little else."""

import tracemalloc

import geopandas as gpd
import numpy as np
import shapely

import geohalo as ghl

PIXEL = 1 / 120


def _europe():
    """2400 x 4200 pixels (35-55 N, 10 W-25 E), 60 % empty, and 200 zones tiling it."""
    rng = np.random.default_rng(0)
    lat = 55.0 - (np.arange(2400) + 0.5) * PIXEL
    lon = -10.0 + (np.arange(4200) + 0.5) * PIXEL
    people = rng.gamma(0.3, 50.0, size=(lat.size, lon.size)).astype(np.float32)
    people[rng.random(people.shape) < 0.6] = 0.0
    frame = shapely.box(-9.9, 35.1, 24.9, 54.9)
    seeds = shapely.MultiPoint(np.c_[rng.uniform(-9.5, 24.5, 200), rng.uniform(35.5, 54.5, 200)])
    zones = gpd.GeoSeries([shapely.intersection(cell, frame)
                           for cell in shapely.voronoi_polygons(seeds, extend_to=frame).geoms])
    return zones, lat, lon, people


def _traced(build):
    tracemalloc.start()
    try:
        op = build()
        return op, tracemalloc.get_traced_memory()[1]
    finally:
        tracemalloc.stop()


def _row_oracle(zone, lat, lon, people, grid):
    """People per ascending cell of ``grid`` in ``zone``, from shapely's coverage of every pixel."""
    minx, miny, maxx, maxy = zone.bounds
    rows = np.flatnonzero((lat > miny - PIXEL) & (lat < maxy + PIXEL))
    cols = np.flatnonzero((lon > minx - PIXEL) & (lon < maxx + PIXEL))
    r, c = np.meshgrid(rows, cols, indexing="ij")
    pixels = shapely.box(lon[c] - PIXEL / 2, lat[r] - PIXEL / 2, lon[c] + PIXEL / 2, lat[r] + PIXEL / 2)
    coverage = shapely.area(shapely.intersection(pixels, zone)) / PIXEL**2
    lat_asc = grid[0][::-1]
    step = lat_asc[1] - lat_asc[0]
    cell_row = np.floor((lat[r] - (lat_asc[0] - step / 2)) / step).astype(np.int64)
    cell_col = np.floor((lon[c] - (grid[1][0] - step / 2)) / step).astype(np.int64)
    out = np.zeros(grid[0].size * grid[1].size)
    np.add.at(out, (cell_row * grid[1].size + cell_col).ravel(), (coverage * people[r, c]).ravel())
    return out


def test_a_ten_million_pixel_window_builds_without_copying_it():
    zones, lat, lon, people = _europe()
    grid = (np.arange(89.975, -90.0, -0.05), np.arange(-179.975, 180.0, 0.05))
    op, peak = _traced(lambda: ghl.ReduceOperator.compute_weighted(zones, lat, lon, people, *grid, dtype=np.float32))
    assert peak < people.nbytes  # 38 MiB: no copy of the window, no dense per-zone array
    total = people.astype(np.float64)[:, 12:-12][12:-12].sum()  # the frame trims 0.1 degree on every side
    np.testing.assert_allclose(op.row_sums.sum(), total, rtol=1e-6)
    for row in (0, 77, 199):
        np.testing.assert_allclose(op.matrix[row].toarray().ravel(), _row_oracle(zones[row], lat, lon, people, grid),
                                   rtol=1e-5, atol=1e-3)


def test_a_sparsely_inhabited_zone_holds_nothing_the_size_of_its_bounding_box():
    _, lat, lon, _ = _europe()
    people = np.zeros((lat.size, lon.size), dtype=np.float32)
    rng = np.random.default_rng(1)
    people[rng.integers(0, lat.size, 1000), rng.integers(0, lon.size, 1000)] = 7.0
    zone = gpd.GeoSeries([shapely.box(-9.9, 35.1, 24.9, 54.9)])
    op, peak = _traced(lambda: ghl.ReduceOperator.compute_weighted(zone, lat, lon, people, lat, lon))
    # exactextract copies the zone's box of counts (38 MiB); a float64 per cell of that box would be 77 MiB more.
    assert peak < 1.5 * people.nbytes
    inside = np.logical_and.outer((lat > 35.1) & (lat < 54.9), (lon > -9.9) & (lon < 24.9))
    np.testing.assert_allclose(op.row_sums, [people[inside].sum()], rtol=1e-12)


def test_a_strided_or_south_to_north_window_is_read_in_place():
    zones, lat, lon, people = _europe()
    zones = zones.iloc[:40]
    grid = (np.arange(89.875, -90.0, -0.25), np.arange(-179.875, 180.0, 0.25))
    whole = ghl.ReduceOperator.compute_weighted(zones, lat, lon, people, *grid, dtype=np.float32)
    wider = np.zeros((lat.size, lon.size + 240), dtype=np.float32)
    wider[:, 120:-120] = people
    strided = wider[:, 120:-120]
    assert not strided.flags.c_contiguous
    south_first = people[::-1].copy()
    for window_lat, counts in ((lat, strided), (lat[::-1], south_first)):
        op, peak = _traced(lambda window_lat=window_lat, counts=counts: ghl.ReduceOperator.compute_weighted(
            zones, window_lat, lon, counts, *grid, dtype=np.float32))
        assert peak < counts.nbytes
        np.testing.assert_array_equal(op.matrix.toarray(), whole.matrix.toarray())
        assert op.digest == whole.digest
