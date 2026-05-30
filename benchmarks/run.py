"""Benchmark suite. Prints Markdown tables to stdout.

Run: `uv run python -m benchmarks.run` (writes to stdout)
     `uv run python -m benchmarks.run --out perf.md` (writes to file)

Inputs: real ECMWF IFS ENS 0.25 degree global fields (fetched from S3 on first run,
cached as netCDF thereafter) and GADM polygons for multiple regions.
First run downloads ~1 GB; cached runs are network-free.
"""

import argparse
import io
import platform
import subprocess
import sys
import threading
import time
from collections.abc import Callable
from importlib import metadata
from pathlib import Path

import psutil

DEFAULT_CACHE_DIR = Path.home() / ".cache" / "geohalo"

_RSS_SAMPLE_INTERVAL_S = 0.005


def _time(
    fn: Callable[[], object], *, warmup: int = 2, iters: int = 7,
) -> tuple[dict[str, float], int, object]:
    """Run `fn` warmup + iters times; return (timing ms dict, peak RSS delta, last fn() result).

    ΔRSS is the peak *current* resident set during the timed region minus the baseline right
    before it, sampled by a background thread. This is a true per-row figure — unlike
    ``resource.getrusage().ru_maxrss``, which is a monotonic all-time high-water mark and would
    credit the whole jump to whichever row happened to allocate first. The last result is
    returned so callers (e.g. _bench_cache) can capture the built object without an extra build.
    """
    proc = psutil.Process()
    last: object = None
    for _ in range(warmup):
        last = fn()

    base = proc.memory_info().rss
    peak = base
    stop = threading.Event()

    def _sample() -> None:
        nonlocal peak
        while not stop.is_set():
            rss = proc.memory_info().rss
            peak = max(peak, rss)
            time.sleep(_RSS_SAMPLE_INTERVAL_S)

    sampler = threading.Thread(target=_sample, daemon=True)
    sampler.start()
    samples_ms: list[float] = []
    for _ in range(iters):
        t = time.perf_counter()
        last = fn()
        samples_ms.append((time.perf_counter() - t) * 1000)
    stop.set()
    sampler.join()
    peak = max(peak, proc.memory_info().rss)

    samples_ms.sort()
    timing = {
        "median": samples_ms[iters // 2],
        "p10": samples_ms[iters // 10],
        "p90": samples_ms[(iters * 9) // 10],
    }
    return timing, max(0, peak - base), last


def _format_ms(ms: float) -> str:
    if ms >= 1000:
        return f"{ms / 1000:.2f} s"
    if ms >= 10:
        return f"{ms:.0f} ms"
    return f"{ms:.1f} ms"


def _format_bytes(b: int) -> str:
    if b >= 1024 * 1024:
        return f"{b / (1024 * 1024):.1f} MB"
    if b >= 1024:
        return f"{b / 1024:.0f} KB"
    return f"{b} B"


def _format_row_timing(t: dict[str, float]) -> str:
    return f"{_format_ms(t['median'])}  ({_format_ms(t['p10'])} - {_format_ms(t['p90'])})"


def _format_speedup(ratio: float) -> str:
    if ratio == float("inf"):
        return "inf"
    if ratio >= 100:
        return f"{ratio:.0f}x"
    return f"{ratio:.1f}x"


def _batch_repr(batch_shape: dict[str, int]) -> str:
    items = list(batch_shape.items())
    tail = "," if len(items) == 1 else ""
    return "(" + ", ".join(f"{k}={v}" for k, v in items) + tail + ")"


def _git_sha() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, check=True, timeout=2.0, text=True,
        )
        return out.stdout.strip()
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, FileNotFoundError):
        return "unknown"


def _cpu_model() -> str:
    """Best-effort CPU model name; falls back to platform.processor() then 'unknown'."""
    try:
        with open("/proc/cpuinfo") as fh:
            for line in fh:
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or "unknown"


def _hardware_str() -> str:
    physical = psutil.cpu_count(logical=False)
    logical = psutil.cpu_count(logical=True)
    cores = f"{physical}C/{logical}T" if physical else f"{logical}T"
    total_gb = psutil.virtual_memory().total / (1024 ** 3)
    return f"Hardware: {_cpu_model()} ({cores}), {total_gb:.1f} GB RAM"


def _print_table(rows: list[list[str]], headers: list[str]) -> str:
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


def _refined_coords(lats, lons, factor: int):
    from geohalo.geometry import target_coords_from_resolution  # noqa: PLC0415

    res = float(abs(lats[1] - lats[0])) / factor
    return target_coords_from_resolution(lats, lons, res)


def _make_leaves_da(leaf_keys, *, batch_shape: dict[str, int]):
    """Build a DataArray over (`*batch_shape`, geom=leaf keys)."""
    import numpy as np  # noqa: PLC0415
    import xarray as xr  # noqa: PLC0415

    rng = np.random.default_rng(seed=0)
    shape = (*tuple(batch_shape.values()), len(leaf_keys))
    arr = rng.standard_normal(shape)
    dims = (*tuple(batch_shape.keys()), "geom")
    geom_vals = np.empty(len(leaf_keys), dtype=object)
    geom_vals[:] = list(leaf_keys)
    coords = {
        **{name: np.arange(size) for name, size in batch_shape.items()},
        "geom": ("geom", geom_vals),
    }
    return xr.DataArray(arr, dims=dims, coords=coords)


# Cold builds are timed with fewer samples than the hot path: heavy global 0.05 degree
# builds can take minutes, so warmup+7 iters would be infeasible. Hot path keeps 2+7.
COLD_TIMING = {"warmup": 1, "iters": 3}
COLD_TIMING_HEAVY = {"warmup": 0, "iters": 1}  # global 0.05 degree resample builds


def _bench_cache(miss_call, hit_call, *, miss_cfg) -> dict:
    """Measure a cache miss (rebuild + serialize + store) and a hit (load + deserialize).

    `miss_call` / `hit_call` are zero-arg thunks calling the same `get_or_compute_*`
    with `force_recompute=True` / `False`. The build is timed via `miss_call` (last
    result captured, so the heavy global build runs only once); the hit is then
    timed on the populated cache. Asserts the loaded object's digest matches the
    build's, so a broken load cannot be reported as a fast hit.

    Returns {miss_timing, miss_rss, hit_timing, hit_rss, speedup, obj}.
    """
    miss_timing, miss_rss, built = _time(miss_call, **miss_cfg)
    hit_timing, hit_rss, loaded = _time(hit_call)
    if loaded.digest != built.digest:
        msg = "cache hit returned a different object than the build"
        raise RuntimeError(msg)
    hit_median = hit_timing["median"]
    speedup = miss_timing["median"] / hit_median if hit_median > 0 else float("inf")
    return {
        "miss_timing": miss_timing,
        "miss_rss": miss_rss,
        "hit_timing": hit_timing,
        "hit_rss": hit_rss,
        "speedup": speedup,
        "obj": built,
    }


def _csr_bytes(m) -> int:
    return m.data.nbytes + m.indices.nbytes + m.indptr.nbytes


def _env_block(
    da, cycle_label: str, polygon_counts: dict[str, int], *, cache_repr: str, mask_fraction: float,
) -> str:
    py = sys.version_info
    py_str = f"{py.major}.{py.minor}.{py.micro}"
    deps: list[str] = []
    for pkg in ("scipy", "numpy", "shapely", "xarray", "exactextract", "geopandas", "cfgrib"):
        try:
            deps.append(f"{pkg} {metadata.version(pkg)}")
        except metadata.PackageNotFoundError:
            deps.append(f"{pkg} ?")
    counts = ", ".join(f"{k}={v}" for k, v in polygon_counts.items())
    return (
        f"Environment: Python {py_str} on {platform.system()} {platform.machine()}, "
        f"geohalo @ {_git_sha()}, {', '.join(deps)}.\n"
        f"{_hardware_str()}.\n"
        f"ECMWF: IFS ENS 0.25 degree, cycle {cycle_label}, "
        f"global {da.sizes['latitude']}x{da.sizes['longitude']} grid.\n"
        f"Polygons: {counts}.\n"
        f"Cache: LocalCache @ {cache_repr} (fresh temp dir per run; RedisCache not measured).\n"
        f"Masked path: NaN mask = {mask_fraction:g} of cells (seed 0), synthetic path-coverage.\n"
        f"Timing: hot/hit rows 2 warmup + 7 iters; miss rows 1 warmup + 3 iters "
        f"(global 0.05 degree resample builds: 1 timed build). Median (p10 - p90). "
        f"miss = build + serialize + store; hit = load + deserialize; speedup = miss/hit. "
        f"Hot rows operate on ascending-latitude data. dRSS = peak current-RSS over baseline."
    )


def _bench_ecmwf_inputs(da) -> list[list[str]]:
    """Characterize the three source tensors (no compute; report shape/size/load)."""
    configs = [
        ("(member=1, step=1)", da.isel(member=0, step=0)),
        ("(member=50, step=1)", da.isel(step=0)),
        ("(member=50, step=4)", da),
    ]
    rows: list[list[str]] = []
    for label, sub in configs:
        arr = sub.to_numpy()
        rows.append([
            label,
            "x".join(str(s) for s in arr.shape),
            f"{arr.size:,}",
            _format_bytes(arr.nbytes),
        ])
    return rows


def _region_specs(cache_dir):
    from benchmarks._data import (  # noqa: PLC0415
        americas_countries,
        americas_europe_countries,
        brazil_country,
        brazil_municipalities,
        us_counties,
    )
    return [
        ("Brazil (country)", lambda: brazil_country(cache_dir), None),
        ("Americas (countries)", lambda: americas_countries(cache_dir), None),
        ("Americas & Europe (countries)", lambda: americas_europe_countries(cache_dir), None),
        ("Brazil (municipalities)", lambda: brazil_municipalities(cache_dir), 0.05),
        ("United States (counties)", lambda: us_counties(cache_dir), 0.05),
    ]


def _bench_reduce_cold(da, cache_dir, cache):
    """Per-artifact cache miss/hit rows; returns (rows, operators, stencils).

    operators[label] = [(iters_label, ReduceOperator | None)] for the clean hot table.
    stencils[label]  = (Stencil, target_label, n_polys) for the masked hot table.
    """
    import numpy as np  # noqa: PLC0415

    from benchmarks._mem import run_guarded  # noqa: PLC0415
    from geohalo.cache import _ser_reduce_op, _ser_stencil  # noqa: PLC0415
    from geohalo.geometry import ensure_ascending_lats  # noqa: PLC0415

    src_lat, _ = ensure_ascending_lats(da["latitude"].to_numpy())
    src_lon = np.asarray(da["longitude"].to_numpy(), dtype=np.float64)

    rows: list[list[str]] = []
    operators: dict[str, list[tuple[str, object]]] = {}
    stencils: dict[str, tuple[object, str, int]] = {}

    for label, builder, target_res in _region_specs(cache_dir):
        geoms = builder()
        n = len(geoms)
        if target_res is None:
            t_lat, t_lon = src_lat, src_lon
            iter_list = [None]
            target_label = "native"
        else:
            t_lat, t_lon = _refined_coords(
                src_lat, src_lon,
                factor=round(float(abs(src_lat[1] - src_lat[0])) / target_res),
            )
            iter_list = [1, 3]
            target_label = f"{target_res} deg"

        # Stencil is sparse and bounded by polygon coverage (not the full target grid),
        # so it is not memory-guarded even on the global 0.05 deg grid.
        st_cfg = COLD_TIMING if target_res is None else COLD_TIMING_HEAVY
        st = _bench_cache(
            lambda g=geoms, la=t_lat, lo=t_lon: cache.get_or_compute_stencil(la, lo, g, force_recompute=True),
            lambda g=geoms, la=t_lat, lo=t_lon: cache.get_or_compute_stencil(la, lo, g),
            miss_cfg=st_cfg,
        )
        stencil = st["obj"]
        stencils[label] = (stencil, target_label, n)
        rows.append([
            label, "Stencil", str(n), target_label, "n/a",
            _format_row_timing(st["miss_timing"]), _format_row_timing(st["hit_timing"]),
            _format_speedup(st["speedup"]),
            _format_bytes(_csr_bytes(stencil.occupancy_matrix)),
            _format_bytes(len(_ser_stencil(stencil))),
            _format_bytes(st["miss_rss"]),
        ])

        for iters in iter_list:
            it = 1 if iters is None else iters
            it_label = "n/a" if iters is None else str(iters)
            cfg = COLD_TIMING if target_res is None else COLD_TIMING_HEAVY

            # Guard the whole miss+hit measurement: the heavy fused-operator build can trip the
            # memory cap, and an unguarded MemoryError would crash the suite instead of skipping.
            def _measure(s=stencil, i=it, c=cfg):
                return _bench_cache(
                    lambda ss=s, ii=i: cache.get_or_compute_reduce_operator(
                        ss, src_lat, src_lon, iterations=ii, force_recompute=True),
                    lambda ss=s, ii=i: cache.get_or_compute_reduce_operator(
                        ss, src_lat, src_lon, iterations=ii),
                    miss_cfg=c,
                )

            metrics, err = run_guarded(_measure)
            if err is not None:
                rows.append([
                    label, "ReduceOperator", str(n), target_label, it_label,
                    "skipped (>mem)", "skipped", "-", "-", "-", "skipped",
                ])
                operators.setdefault(label, []).append((it_label, None))
                continue
            op = metrics["obj"]
            operators.setdefault(label, []).append((it_label, op))
            rows.append([
                label, "ReduceOperator", str(n), target_label, it_label,
                _format_row_timing(metrics["miss_timing"]), _format_row_timing(metrics["hit_timing"]),
                _format_speedup(metrics["speedup"]),
                _format_bytes(_csr_bytes(op.matrix)),
                _format_bytes(len(_ser_reduce_op(op))),
                _format_bytes(metrics["miss_rss"]),
            ])
    return rows, operators, stencils


def _bench_reduce_hot(da, operators, stencils, *, mask_fraction=0.3):
    from benchmarks._data import apply_nan_mask  # noqa: PLC0415
    from geohalo.api import reduce_with_operator, reduce_with_stencil  # noqa: PLC0415

    # Sort to ascending latitude once: ECMWF grids are stored 90->-90, and the reduce paths
    # re-sort descending input on *every* call. The operator's source grid is canonically
    # ascending, so pre-sorting measures the apply rather than a per-call re-orientation a
    # caller would do once. .load() materialises each slice so timed iters don't re-read disk.
    da = da.sortby("latitude")
    batch_configs = [
        ("(member=1, step=1)", da.isel(member=0, step=0).load()),
        ("(member=50, step=1)", da.isel(step=0).load()),
        ("(member=50, step=4)", da.load()),
    ]
    masked_lookup = {lbl: apply_nan_mask(sub, fraction=mask_fraction) for lbl, sub in batch_configs}

    rows: list[list[str]] = []
    for label, entries in operators.items():
        stencil, target_label, _ = stencils[label]
        is_native = target_label == "native"
        for iters_label, op in entries:
            n_polys = "-" if op is None else str(len(op.keys))
            for batch_label, sub in batch_configs:
                if op is None:
                    rows.append([label, n_polys, iters_label, batch_label, "clean", "skipped (>mem)", "-"])
                    continue
                t, rss, _ = _time(lambda d=sub, o=op: reduce_with_operator(d, o))
                rows.append([label, n_polys, iters_label, batch_label, "clean",
                             _format_row_timing(t), _format_bytes(rss)])

        # Masked (NaN) path: native-grid regions only. There the masked path is a pure
        # occupancy matmul with per-cell renormalisation, directly comparable to the clean
        # operator. Resample regions re-run the full un-fused resampler per call (seconds),
        # so they are deliberately out of this table.
        if not is_native:
            continue
        for batch_label, _ in batch_configs:
            da_nan = masked_lookup[batch_label]
            t, rss, _ = _time(lambda d=da_nan, s=stencil: reduce_with_stencil(d, s))
            rows.append([label, str(len(stencil.keys)), "n/a", batch_label, "masked",
                         _format_row_timing(t), _format_bytes(rss)])
    return rows


def _bench_bias_tree(cache_dir, cache):
    from benchmarks._data import brazil_municipalities, build_brazil_hierarchy  # noqa: PLC0415
    from geohalo.api import aggregate_bias_with_tree  # noqa: PLC0415
    from geohalo.cache import _ser_tree  # noqa: PLC0415

    geoms = brazil_municipalities(cache_dir)
    specs = [
        ("muni -> state", 2, build_brazil_hierarchy(geoms, depth=2)),
        ("muni -> state -> country", 3, build_brazil_hierarchy(geoms, depth=3)),
    ]
    rows: list[list[str]] = []
    for label, depth, edges in specs:
        metrics = _bench_cache(
            lambda e=edges: cache.get_or_compute_tree(e, force_recompute=True),
            lambda e=edges: cache.get_or_compute_tree(e),
            miss_cfg=COLD_TIMING,
        )
        tree = metrics["obj"]
        leaf_da = _make_leaves_da(list(tree.leaf_keys), batch_shape={"member": 50})
        ht, h_rss, _ = _time(lambda d=leaf_da, tr=tree: aggregate_bias_with_tree(d, tr))
        rows.append([
            label, str(len(tree.leaf_keys)), str(depth),
            _format_row_timing(metrics["miss_timing"]), _format_row_timing(metrics["hit_timing"]),
            _format_speedup(metrics["speedup"]),
            _format_bytes(_csr_bytes(tree.rollup_matrix)), _format_bytes(len(_ser_tree(tree))),
            _format_row_timing(ht), _format_bytes(max(metrics["miss_rss"], h_rss)),
        ])
    return rows


def main() -> None:
    import shutil  # noqa: PLC0415
    import tempfile  # noqa: PLC0415

    parser = argparse.ArgumentParser(description="Run geohalo full benchmarks.")
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--mem-limit-gb", type=float, default=10.0)
    parser.add_argument("--mask-fraction", type=float, default=0.3)
    args = parser.parse_args()

    from benchmarks._data import fetch_ecmwf_global  # noqa: PLC0415
    from benchmarks._mem import set_address_space_limit  # noqa: PLC0415
    from geohalo.cache import LocalCache  # noqa: PLC0415

    set_address_space_limit(args.mem_limit_gb)
    print(f"address-space limit set to {args.mem_limit_gb} GiB", file=sys.stderr)

    print("loading ECMWF global field ...", file=sys.stderr)
    da = fetch_ecmwf_global(args.cache_dir)
    cycle_label = da.attrs.get("ecmwf_cycle", "unknown")  # the cycle actually loaded (matches the data)
    print(f"  cycle {cycle_label}", file=sys.stderr)

    artifact_cache_dir = Path(tempfile.mkdtemp(prefix="geohalo-bench-"))
    cache = LocalCache(artifact_cache_dir)
    try:
        cold_rows, operators, stencils = _bench_reduce_cold(da, args.cache_dir, cache)
        polygon_counts = {
            label: len(next(op for _, op in entries if op is not None).keys)
            for label, entries in operators.items()
            if any(op is not None for _, op in entries)
        }

        buf = io.StringIO()
        print("# geohalo benchmarks\n", file=buf)
        print(_env_block(da, cycle_label, polygon_counts,
                         cache_repr=str(artifact_cache_dir), mask_fraction=args.mask_fraction), file=buf)
        print(file=buf)

        print("## ECMWF inputs (whole world)\n", file=buf)
        print(_print_table(_bench_ecmwf_inputs(da),
            headers=["batch", "shape", "cells", "in-mem"]), file=buf)
        print(file=buf)

        print("## Reduce - precompute (cache miss vs hit)\n", file=buf)
        print(_print_table(cold_rows,
            headers=["region", "artifact", "n_polys", "target", "iters",
                     "miss (build+store)", "hit (load+deser)", "speedup",
                     "CSR", "blob", "dRSS"]), file=buf)
        print(file=buf)

        print("## Reduce - hot (apply)\n", file=buf)
        print(_print_table(_bench_reduce_hot(da, operators, stencils, mask_fraction=args.mask_fraction),
            headers=["region", "n_polys", "iters", "batch", "path", "median  (p10 - p90)", "dRSS"]), file=buf)
        print(file=buf)

        print("## Bias tree (cache miss/hit + hot)\n", file=buf)
        print(_print_table(_bench_bias_tree(args.cache_dir, cache),
            headers=["hierarchy", "n_leaves", "depth", "miss", "hit", "speedup",
                     "rollup CSR", "blob", "apply  (p10 - p90)", "dRSS"]), file=buf)
        print(file=buf)

        out_text = buf.getvalue()
        print(out_text)
        if args.out is not None:
            args.out.write_text(out_text)
    finally:
        shutil.rmtree(artifact_cache_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
