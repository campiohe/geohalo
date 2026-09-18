"""Compare conservative contraction orders on synthetic global grids.

Run: ``uv run python -m benchmarks.conservative_refinement``. No downloads needed.
Times exclude construction and validation; variants are warmed and alternate
which runs first. Peaks, when requested, are measured separately from timings.
Use ``--no-missing`` for clean inputs, ``--no-skipna`` for NaN propagation,
and ``--normalization covered`` for covered-area normalization.
"""

import argparse
import gc
import platform
import time
import tracemalloc
from unittest.mock import patch

import numpy as np
import scipy

from geohalo import Resampler
from geohalo._conservative import ConservativeGrid

CASES = (
    ("refine", "0.25 -> 0.10, 4 slices", (720, 1440), (1800, 3600), 4),
    ("refine", "0.25 -> 0.05, 1 slice", (720, 1440), (3600, 7200), 1),
    ("coarsen", "0.25 -> 0.50, 24 slices", (720, 1440), (360, 720), 24),
    ("coarsen", "0.10 -> 0.25, 4 slices", (1800, 3600), (720, 1440), 4),
    ("anisotropic", "2000x200 -> 100x8000", (2000, 200), (100, 8000), 1),
    ("anisotropic", "100x8000 -> 2000x200", (100, 8000), (2000, 200), 1),
)


def _previous_project(self, values):
    """The geohalo 2.0.0 rule, including latitude-first ties."""
    if self.latitude.shape[0] * self.longitude.shape[1] <= self.latitude.shape[1] * self.longitude.shape[0]:
        return (self.longitude @ (self.latitude @ values).T).T
    return self.latitude @ (self.longitude @ values.T).T


def _grid(shape):
    lat_edges = np.linspace(-90, 90, shape[0] + 1)
    lon_edges = np.linspace(-180, 180, shape[1] + 1)
    lat = ((lat_edges[:-1] + lat_edges[1:]) / 2)[::-1]
    lon = (lon_edges[:-1] + lon_edges[1:]) / 2
    return lat, lon


def _measure(operator, values, args):
    variants = (("previous", _previous_project), ("current", ConservativeGrid._project))  # noqa: SLF001
    # Warm both variants and compare through the public API before timing.
    expected = operator.apply_grid(values, descending=True, skipna=args.skipna)
    with patch.object(ConservativeGrid, "_project", _previous_project):
        previous = operator.apply_grid(values, descending=True, skipna=args.skipna)
    np.testing.assert_allclose(expected, previous, rtol=1e-12, atol=1e-12, equal_nan=True)
    del expected, previous
    timings = {label: [] for label, _ in variants}
    for repeat in range(args.repeats):
        for label, project in variants[::1 if repeat % 2 == 0 else -1]:
            with patch.object(ConservativeGrid, "_project", project):
                gc.collect()
                start = time.perf_counter()
                operator.apply_grid(values, descending=True, skipna=args.skipna)
                timings[label].append((time.perf_counter() - start) * 1000)
    peaks = {}
    if args.memory:
        for label, project in variants:
            with patch.object(ConservativeGrid, "_project", project):
                tracemalloc.start()
                try:
                    operator.apply_grid(values, descending=True, skipna=args.skipna)
                    peaks[label] = tracemalloc.get_traced_memory()[1] / 2**20
                finally:
                    tracemalloc.stop()
    return timings, peaks


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", choices=("all", "refine", "coarsen", "anisotropic"), default="all")
    parser.add_argument("--repeats", type=int, default=4)
    parser.add_argument("--skipna", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--missing", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--normalization", choices=("destination", "covered"), default="destination")
    parser.add_argument("--memory", action="store_true")
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("repeats must be positive")
    print(f"Python {platform.python_version()}; NumPy {np.__version__}; SciPy {scipy.__version__}")
    print(f"{args.repeats} interleaved repeats; float32 inputs; descending latitude; "
          f"skipna={args.skipna}; missing={args.missing}; normalization={args.normalization}", flush=True)
    rng = np.random.default_rng(0)
    for group, name, source_shape, target_shape, steps in CASES:
        if args.case not in ("all", group):
            continue
        operator = Resampler.compute(*_grid(source_shape), *_grid(target_shape),
                                     method="conservative", period=360, normalization=args.normalization)
        values = rng.uniform(270, 290, (steps, *source_shape)).astype(np.float32)
        if args.missing:
            values[:, ::37, ::53] = np.nan
        timings, peaks = _measure(operator, values, args)
        before, after = (float(np.median(timings[label])) for label in ("previous", "current"))
        print(f"{name}: median {before:.1f} -> {after:.1f} ms ({before / after:.2f}x); "
              f"min {min(timings['previous']):.1f} -> {min(timings['current']):.1f} ms; results match", flush=True)
        if peaks:
            print(f"  peak allocations: {peaks['previous']:.1f} -> {peaks['current']:.1f} MiB", flush=True)


if __name__ == "__main__":
    main()
