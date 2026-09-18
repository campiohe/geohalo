"""Compare approximate and exact stencil weights using synthetic high-latitude zones.

Run: ``uv run python -m benchmarks.partial_cell_weighting``. No downloads required.
Geometry generation and reference areas are outside the timed region.
"""

import argparse
import time

import geopandas as gpd
import numpy as np
import shapely

from geohalo import ReduceOperator, Stencil
from geohalo.geometry import polygon_areas


def _measure(fn, repeats):
    times = []
    for _ in range(repeats):
        start = time.perf_counter()
        result = fn()
        times.append(time.perf_counter() - start)
    return result, float(np.median(times))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--polygons", type=int, default=250)
    parser.add_argument("--quad-segs", type=int, default=32)
    parser.add_argument("--resolution", type=float, default=0.25)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    if min(args.polygons, args.quad_segs, args.repeats) < 1 or not 0 < args.resolution <= 5:
        parser.error("counts must be positive and resolution must be in (0, 5]")
    rng = np.random.default_rng(0)
    geoms = gpd.GeoSeries([
        shapely.Point(lon, lat).buffer(radius, quad_segs=args.quad_segs)
        for lon, lat, radius in zip(rng.uniform(-20, 20, args.polygons),
                                    rng.uniform(45, 75, args.polygons),
                                    rng.uniform(.2, 1, args.polygons), strict=True)
    ], crs="EPSG:4326")
    lats, lons = np.arange(40, 80, args.resolution), np.arange(-25, 25, args.resolution)
    expected_areas = polygon_areas(geoms)
    values = rng.uniform(0, 50, (25, len(lats), len(lons)))
    print(f"{len(geoms)} polygons; {shapely.get_num_coordinates(geoms.to_numpy()).sum()} vertices; "
          f"{len(lats)}x{len(lons)} cells; median of {args.repeats} builds", flush=True)
    timings = {}
    for mode in ("approximate", "exact"):
        stencil, elapsed = _measure(
            lambda mode=mode: Stencil.compute(lats, lons, geoms, partial_cell_weighting=mode), args.repeats,
        )
        operator = ReduceOperator.compute(stencil, lats, lons)
        operator.apply_grid(values)  # warm the application path
        _, apply_time = _measure(lambda operator=operator: operator.apply_grid(values), max(10, args.repeats))
        relative_error = np.max(np.abs(stencil.row_sums / expected_areas - 1))
        print(f"{mode:11s}: build {elapsed:.4f} s; apply 25 slices {apply_time * 1000:.3f} ms; "
              f"max relative area error {relative_error:.3g}; nnz {stencil.occupancy_matrix.nnz}", flush=True)
        timings[mode] = elapsed
        if mode == "exact":
            np.testing.assert_allclose(stencil.row_sums, expected_areas, rtol=1e-12)
    print(f"Exact / approximate build time: {timings['exact'] / timings['approximate']:.2f}x")


if __name__ == "__main__":
    main()
