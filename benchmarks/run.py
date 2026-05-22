"""Benchmark suite. Prints three Markdown tables to stdout.

Run: `uv run python -m benchmarks.run` (writes to stdout)
     `uv run python -m benchmarks.run --out perf.md` (writes to file)

Inputs: GADM Brazil L2 polygons + synthetic 0.25° Brazil-bbox grid.
First run downloads ~6 MB to ~/.cache/geohalo/ and is cached thereafter.
"""

import argparse
import io
import platform
import resource
import subprocess
import sys
import time
from collections.abc import Callable
from importlib import metadata
from pathlib import Path

DEFAULT_CACHE_DIR = Path.home() / ".cache" / "geohalo"

# resource.getrusage(...).ru_maxrss is in kilobytes on Linux, bytes on macOS/BSD.
_RU_MAXRSS_TO_BYTES = 1024 if sys.platform.startswith("linux") else 1


def _peak_rss_bytes() -> int:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * _RU_MAXRSS_TO_BYTES


def _time(
    fn: Callable[[], object], *, warmup: int = 2, iters: int = 7,
) -> tuple[dict[str, float], int]:
    """Run `fn` warmup + iters times.

    Returns (timing dict in ms with median/p10/p90, peak RSS delta in bytes
    observed during the call sequence). The RSS delta is "max RSS after" minus
    "max RSS before" — capturing new high-water-marks only, so steady-state
    calls show 0.
    """
    rss_before = _peak_rss_bytes()
    for _ in range(warmup):
        fn()
    samples_ms: list[float] = []
    for _ in range(iters):
        t = time.perf_counter()
        fn()
        samples_ms.append((time.perf_counter() - t) * 1000)
    samples_ms.sort()
    rss_after = _peak_rss_bytes()
    timing = {
        "median": samples_ms[iters // 2],
        "p10": samples_ms[iters // 10],
        "p90": samples_ms[(iters * 9) // 10],
    }
    return timing, max(0, rss_after - rss_before)


def _format_ms(ms: float) -> str:
    """Pretty-format a millisecond value."""
    if ms >= 1000:
        return f"{ms / 1000:.2f} s"
    if ms >= 10:
        return f"{ms:.0f} ms"
    return f"{ms:.1f} ms"


def _format_bytes(b: int) -> str:
    """Pretty-format a byte count as B / KB / MB."""
    if b >= 1024 * 1024:
        return f"{b / (1024 * 1024):.1f} MB"
    if b >= 1024:
        return f"{b / 1024:.0f} KB"
    return f"{b} B"


def _format_row_timing(t: dict[str, float]) -> str:
    """Right-aligned 'median  (p10 - p90)' cell."""
    return f"{_format_ms(t['median'])}  ({_format_ms(t['p10'])} – {_format_ms(t['p90'])})"


def _git_sha() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            check=True,
            timeout=2.0,
            text=True,
        )
        return out.stdout.strip()
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, FileNotFoundError):
        return "unknown"


def _env_block() -> str:
    py = sys.version_info
    py_str = f"{py.major}.{py.minor}.{py.micro}"
    deps: list[str] = []
    for pkg in ("scipy", "numpy", "shapely", "xarray", "exactextract"):
        try:
            deps.append(f"{pkg} {metadata.version(pkg)}")
        except metadata.PackageNotFoundError:
            deps.append(f"{pkg} ?")
    return (
        f"Environment: Python {py_str} on {platform.system()} {platform.machine()}, "
        f"geohalo @ {_git_sha()}, {', '.join(deps)}.\n"
        f"Grid: 0.25° over Brazil bbox (-74, -34, -34, 6) — 160×160 = 25,600 cells.\n"
        f"Polygons: GADM Brazil L2 (~5570 total).\n"
        f"Timing: 2 warmup + 7 iterations per row, reporting median (p10 – p90)."
    )


def _print_table(rows: list[list[str]], headers: list[str]) -> str:
    """Render a left-aligned Markdown table."""
    widths = [
        max(len(headers[i]), max(len(r[i]) for r in rows))
        for i in range(len(headers))
    ]
    sep = "| " + " | ".join("-" * w for w in widths) + " |"
    head = "| " + " | ".join(h.ljust(w) for h, w in zip(headers, widths, strict=True)) + " |"
    body_lines = [
        "| " + " | ".join(c.ljust(w) for c, w in zip(row, widths, strict=True)) + " |"
        for row in rows
    ]
    return "\n".join([head, sep, *body_lines])


def _make_da(grid, *, batch_shape: dict[str, int], nan_density: float = 0.0):
    """Build an xr.DataArray over `grid` shaped (*batch_shape, lat, lon)."""
    import numpy as np  # noqa: PLC0415
    import xarray as xr  # noqa: PLC0415

    rng = np.random.default_rng(seed=0)
    n_lat, n_lon = grid.shape
    shape = (*tuple(batch_shape.values()), n_lat, n_lon)
    arr = rng.standard_normal(shape)
    if nan_density > 0:
        mask = rng.random(shape) < nan_density
        arr = np.where(mask, np.nan, arr)
    dims = (*tuple(batch_shape.keys()), "latitude", "longitude")
    coords = {
        **{name: np.arange(size) for name, size in batch_shape.items()},
        "latitude": grid.lats,
        "longitude": grid.lons,
    }
    return xr.DataArray(arr, dims=dims, coords=coords)


def _bench_aggregate(polygons_full, grid) -> list[list[str]]:
    """Seven rows. See spec table 2."""
    from benchmarks._data import sample_polygons  # noqa: PLC0415
    from geohalo.aggregate import aggregate  # noqa: PLC0415
    from geohalo.weights import compute_weights  # noqa: PLC0415

    small = sample_polygons(polygons_full, "small")
    medium = sample_polygons(polygons_full, "medium")
    large = polygons_full

    w_small_f1 = compute_weights(small, grid)
    w_medium_f1 = compute_weights(medium, grid)
    w_large_f1 = compute_weights(large, grid)
    target_res_f4 = float(abs(grid.lats[1] - grid.lats[0])) / 4
    w_large_f4 = compute_weights(large, grid, target_resolution=target_res_f4)

    spec = [
        (len(small.keys),  w_small_f1,  {"member": 50},               "1",          0.00, "50"),
        (len(medium.keys), w_medium_f1, {"member": 50},               "1",          0.00, "50"),
        (len(large.keys),  w_large_f1,  {"member": 50},               "1",          0.00, "50"),
        (len(large.keys),  w_large_f1,  {"member": 50, "step": 10},   "1",          0.00, "500"),
        (len(large.keys),  w_large_f1,  {"member": 50, "step": 40},   "1",          0.00, "2 000"),
        (len(large.keys),  w_large_f4,  {"member": 50},               "4",          0.00, "50"),
        (len(large.keys),  w_large_f1,  {"member": 50},               "1 (1% NaN)", 0.01, "50"),
    ]

    rows: list[list[str]] = []
    for n, weights, batch_shape, factor_label, nan_density, slices_label in spec:
        da = _make_da(grid, batch_shape=batch_shape, nan_density=nan_density)
        t, rss_delta = _time(lambda d=da, w=weights: aggregate(d, w))
        items = list(batch_shape.items())
        # Python-tuple-style repr: trailing comma only when single-element.
        tail = "," if len(items) == 1 else ""
        batch_repr = "(" + ", ".join(f"{k}={v}" for k, v in items) + tail + ")"
        rows.append([
            str(n), batch_repr, slices_label, factor_label,
            _format_row_timing(t),
            _format_bytes(rss_delta),
        ])
    return rows


def _make_leaves_da(leaf_keys, key_names, *, batch_shape: dict[str, int]):
    """Build a DataArray over (`*batch_shape`, polygon=leaf MultiIndex)."""
    import numpy as np  # noqa: PLC0415
    import pandas as pd  # noqa: PLC0415
    import xarray as xr  # noqa: PLC0415

    rng = np.random.default_rng(seed=0)
    shape = (*tuple(batch_shape.values()), len(leaf_keys))
    arr = rng.standard_normal(shape)
    dims = (*tuple(batch_shape.keys()), "polygon")
    polygon_index = pd.MultiIndex.from_tuples(leaf_keys, names=list(key_names))
    coords = {
        **{name: np.arange(size) for name, size in batch_shape.items()},
        "polygon": polygon_index,
    }
    return xr.DataArray(arr, dims=dims, coords=coords)


def _bench_compute_bias(polygons_full) -> list[list[str]]:
    """Four rows. See spec table 3."""
    from benchmarks._data import (  # noqa: PLC0415
        build_deep_hierarchy,
        build_gadm_hierarchy,
        sample_polygons,
    )
    from geohalo.bias import compute_bias  # noqa: PLC0415

    medium = sample_polygons(polygons_full, "medium")
    h_gadm_medium = build_gadm_hierarchy(medium)
    h_gadm_large = build_gadm_hierarchy(polygons_full)
    h_deep = build_deep_hierarchy(len(polygons_full.keys))

    spec = [
        (
            len(medium.keys), 2, "medium GADM (state -> muni)",
            h_gadm_medium, {"member": 50},
        ),
        (
            len(polygons_full.keys), 2, "full GADM (state -> muni)",
            h_gadm_large, {"member": 50},
        ),
        (
            len(polygons_full.keys), 2, "full GADM (state -> muni)",
            h_gadm_large, {"member": 50, "step": 10},
        ),
        (
            len(polygons_full.keys), 4, "synthetic deep",
            h_deep, {"member": 50},
        ),
    ]

    rows: list[list[str]] = []
    for n_leaves, depth, label, hierarchy, batch_shape in spec:
        da = _make_leaves_da(hierarchy.leaf_keys, hierarchy.key_names, batch_shape=batch_shape)
        t, rss_delta = _time(lambda d=da, h=hierarchy: compute_bias(d, h))
        items = list(batch_shape.items())
        tail = "," if len(items) == 1 else ""
        batch_repr = "(" + ", ".join(f"{k}={v}" for k, v in items) + tail + ")"
        rows.append([
            str(n_leaves), str(depth), label, batch_repr,
            _format_row_timing(t),
            _format_bytes(rss_delta),
        ])
    return rows


def _bench_compute_weights(polygons_full, grid) -> list[list[str]]:
    """Six rows: (n_polygons in {50, 500, 5570}) x (factor in {1, 4})."""
    from benchmarks._data import sample_polygons  # noqa: PLC0415
    from geohalo.cache import _serialize  # noqa: PLC0415
    from geohalo.weights import compute_weights  # noqa: PLC0415

    rows: list[list[str]] = []
    for label in ("small", "medium", "large"):
        subset = sample_polygons(polygons_full, label)
        n = len(subset.keys)
        for factor in (1, 4):
            target_res = None if factor == 1 else (
                float(abs(grid.lats[1] - grid.lats[0])) / factor
            )
            t, rss_delta = _time(
                lambda s=subset, g=grid, tr=target_res: compute_weights(
                    s, g, target_resolution=tr,
                ),
            )
            w = compute_weights(subset, grid, target_resolution=target_res)
            in_mem = (
                w.matrix.data.nbytes
                + w.matrix.indices.nbytes
                + w.matrix.indptr.nbytes
            )
            cache_size = len(_serialize(w))
            rows.append([
                str(n), str(factor),
                _format_row_timing(t),
                _format_bytes(in_mem),
                _format_bytes(cache_size),
                _format_bytes(rss_delta),
            ])
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Run geohalo benchmarks.")
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="path to write the Markdown report (default: stdout only)",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=DEFAULT_CACHE_DIR,
        help=f"directory for the cached GADM download (default: {DEFAULT_CACHE_DIR})",
    )
    args = parser.parse_args()

    from benchmarks._data import (  # noqa: PLC0415
        BRAZIL_BBOX,
        build_polygon_set,
        fetch_gadm_brazil_l2,
        make_brazil_grid,
    )

    grid = make_brazil_grid()
    print("loading GADM polygons ...", file=sys.stderr)
    geojson = fetch_gadm_brazil_l2(args.cache_dir)
    polygons_full, _ = build_polygon_set(geojson, BRAZIL_BBOX)
    print(f"  loaded {len(polygons_full.keys)} polygons", file=sys.stderr)

    buf = io.StringIO()
    print("# geohalo benchmarks\n", file=buf)
    print(_env_block(), file=buf)
    print(file=buf)

    print("## `compute_weights` (one-time per (grid, polygons))\n", file=buf)
    print(_print_table(
        _bench_compute_weights(polygons_full, grid),
        headers=[
            "n_polygons", "factor", "median  (p10 – p90)",
            "CSR mem", "blob size", "ΔRSS",
        ],
    ), file=buf)
    print(file=buf)

    print("## `aggregate` (hot path)\n", file=buf)
    print(_print_table(
        _bench_aggregate(polygons_full, grid),
        headers=[
            "n_polygons", "batch", "slices", "factor",
            "median  (p10 – p90)", "ΔRSS",
        ],
    ), file=buf)
    print(file=buf)

    print("## `compute_bias` (DAG rollup)\n", file=buf)
    print(_print_table(
        _bench_compute_bias(polygons_full),
        headers=[
            "n_leaves", "depth", "hierarchy", "batch",
            "median  (p10 – p90)", "ΔRSS",
        ],
    ), file=buf)
    print(file=buf)

    out_text = buf.getvalue()
    print(out_text)
    if args.out is not None:
        args.out.write_text(out_text)


if __name__ == "__main__":
    main()
