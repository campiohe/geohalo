"""Build cost of ``ReduceOperator.compute_weighted`` over a ten-million-pixel window of 30 arc-second counts.

Run: ``uv run python -m benchmarks.weighted_build``.
The window is 2400 x 4200 float32 pixels (35-55 N, 10 W-25 E, 38 MiB), 60 % of them empty. Peak is tracemalloc's,
so it counts NumPy and pandas allocations and not exactextract's own coverage grid (at most
``max_cells_in_memory`` cells per polygon). The window is allocated before tracing starts.
"""

import argparse
import time
import tracemalloc

import geopandas as gpd
import numpy as np
import shapely

from geohalo import ReduceOperator

PIXEL = 1 / 120
GRIDS = {
    "0.25": (np.arange(90.0, -90.001, -0.25), np.arange(-180.0, 180.0, 0.25)),
    "0.05": (np.arange(89.975, -90.0, -0.05), np.arange(-179.975, 180.0, 0.05)),
}


def _window(rng):
    lat = 55.0 - (np.arange(2400) + 0.5) * PIXEL
    lon = -10.0 + (np.arange(4200) + 0.5) * PIXEL
    people = rng.gamma(0.3, 50.0, size=(lat.size, lon.size)).astype(np.float32)
    people[rng.random(people.shape) < 0.6] = 0.0
    return lat, lon, people


def _zones(rng, count):
    frame = shapely.box(-9.9, 35.1, 24.9, 54.9)
    if count == 1:
        return gpd.GeoSeries([shapely.Point(7.5, 45.0).buffer(9.4, quad_segs=256)])
    seeds = shapely.MultiPoint(np.c_[rng.uniform(-9.5, 24.5, count), rng.uniform(35.5, 54.5, count)])
    return gpd.GeoSeries([shapely.intersection(cell, frame)
                          for cell in shapely.voronoi_polygons(seeds, extend_to=frame).geoms])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    rng = np.random.default_rng(0)
    lat, lon, people = _window(rng)
    print(f"Window: {people.shape[0]}x{people.shape[1]} = {people.size / 1e6:.1f}M pixels, "
          f"{people.nbytes / 2**20:.0f} MiB float32, {np.count_nonzero(people) / 1e6:.1f}M inhabited")
    print("| Grid | Zones | Median s | Traced peak MiB | nnz | NPZ KiB |")
    print("| --- | ---: | ---: | ---: | ---: | ---: |")
    for count in (1, 200, 3000):
        zones = _zones(rng, count)
        for name, grid in GRIDS.items():
            times = []
            for _ in range(args.repeats):
                start = time.perf_counter()
                ReduceOperator.compute_weighted(zones, lat, lon, people, *grid, dtype=np.float32)
                times.append(time.perf_counter() - start)
            tracemalloc.start()
            op = ReduceOperator.compute_weighted(zones, lat, lon, people, *grid, dtype=np.float32)
            peak = tracemalloc.get_traced_memory()[1]
            tracemalloc.stop()
            print(f"| {name} | {len(zones)} | {np.median(times):.2f} | {peak / 2**20:.0f} | {op.matrix.nnz:,} | "
                  f"{len(op.to_npz()) / 1024:,.0f} |")


if __name__ == "__main__":
    main()
