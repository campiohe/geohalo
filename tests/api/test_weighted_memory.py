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


def test_a_ten_million_pixel_window_builds_without_copying_it():
    zones, lat, lon, people = _europe()
    grid = (np.arange(89.975, -90.0, -0.05), np.arange(-179.975, 180.0, 0.05))
    tracemalloc.start()
    try:
        op = ghl.ReduceOperator.compute_weighted(zones, lat, lon, people, *grid, dtype=np.float32)
        peak = tracemalloc.get_traced_memory()[1]
    finally:
        tracemalloc.stop()
    assert peak < people.nbytes  # 38 MiB: no copy of the window, no dense per-zone array
    total = people.astype(np.float64)[:, 12:-12][12:-12].sum()  # the frame trims 0.1 degree on every side
    np.testing.assert_allclose(op.row_sums.sum(), total, rtol=1e-6)
