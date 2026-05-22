<p align="center">
  <img src="docs/figures/logo.svg" alt="geohalo" width="180"/>
</p>

<h1 align="center">geohalo</h1>

<p align="center">
  <em>Exact-fractional-area zonal statistics over weather grids.</em>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/python-3.12%2B-3776ab.svg" alt="Python 3.12+"/>
  <img src="https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json" alt="Ruff"/>
  <img src="https://img.shields.io/badge/managed%20by-uv-de5fe9.svg" alt="Managed by uv"/>
  <img src="https://img.shields.io/badge/PRs-welcome-brightgreen.svg" alt="PRs welcome"/>
</p>

---

Given a regular lat/lon mesh of weather values (loaded with `xarray`
from GRIB, NetCDF, Zarr, …) and an arbitrary set of polygons,
`geohalo` reduces the spatial dimensions of the mesh to one value per
polygon with **sub-cell precision** and **millisecond-scale aggregation**
in the hot path.

The expensive geometric work happens once; every subsequent forecast
collapses to a single sparse · dense matmul.

## How it works

Aggregation is a linear operator:

```
aggregates = W @ flat_grid_values
```

where `W ∈ ℝ^(N_polygons × N_cells)` is a sparse matrix whose entries are
the **exact fractional area** of `cell ∩ polygon` weighted by each cell's
true surface area on a sphere, then row-normalised so each polygon's
weights sum to 1.

`W` depends only on the grid topology and the polygon set — not on the
forecast data. The library splits the work into two phases:

1. **Precompute** (`compute_weights`, one-time per `(grid, polygons)` pair):
   call [exactextract](https://github.com/isciences/exactextract) for
   exact fractional cell coverage, multiply by per-cell spherical area
   `R² · Δlon · (sin(lat_top) − sin(lat_bot))` to correct for latitude,
   row-normalise into a `scipy.sparse.csr_matrix`.

2. **Aggregate** (`aggregate`, hot path):
   one sparse · dense matmul, broadcast over every non-spatial dim
   (ensemble, lead time, level, …). NaN-aware: a second matmul against a
   validity mask renormalises per slice without rebuilding `W`.

## Install

`geohalo` targets **Python ≥ 3.12**.

```bash
uv add geohalo            # or: pip install geohalo
```

Optional extras:

- `redis` — for the `RedisWeightCache` backend
- `matplotlib` — for the helpers in `geohalo.plot`

## Quickstart

```python
import xarray as xr
from geohalo import GridSpec, PolygonSet, compute_weights, aggregate

da = xr.open_dataset("forecast.grib", engine="cfgrib")["t2m"]
grid = GridSpec.from_dataarray(da)

polygons = PolygonSet.build(
    geoms=[poly_a, poly_b, poly_c],          # list of shapely geometries
    keys=[("BR", "SP"), ("BR", "RJ"), ("BR", "MG")],
    key_names=("country", "state"),
)

weights = compute_weights(polygons, grid)    # one-time
out = aggregate(da, weights)                 # hot path; ms-scale
# out: xr.DataArray over (..., polygon), polygon a MultiIndex on key_names
```

The output preserves every non-spatial dim of `da` (ensemble member,
lead time, vertical level, …) and replaces `(latitude, longitude)` with
a single `polygon` dim indexed by the polygon keys.

### Caching

`compute_weights` is the only expensive step. Wrap it in a cache so it
runs once per `(grid, polygons, downscale_factor)`:

```python
from geohalo import LocalWeightCache       # or RedisWeightCache

cache = LocalWeightCache("./.geohalo-cache")
weights = cache.get_or_compute(polygons, grid)
```

Both caches share the same key schema — the cache key embeds the grid's
SHA-256 digest, the polygon set's SHA-256 digest, and the downscale factor —
so any change to the grid, the polygons, or the downscaling settings
invalidates the cache implicitly.

## Performance

The headline claim — *millisecond-scale aggregation in the hot path* — is
backed by the suite under [`benchmarks/`](benchmarks/run.py). All rows time
operations against the GADM Brazil L2 polygons (~5570 munis) on a synthetic
0.25° grid covering Brazil (160×160 = 25,600 cells). Re-run with
`uv run python -m benchmarks.run`.

<!-- BENCHMARK START -->

Environment: Python 3.14.4 on Linux x86_64, geohalo @ 90c432b, scipy 1.17.1, numpy 2.4.6, shapely 2.1.2, xarray 2026.4.0, exactextract 0.3.0.
Grid: 0.25° over Brazil bbox (-74, -34, -34, 6) — 160×160 = 25,600 cells.
Polygons: GADM Brazil L2 (~5570 total).
Timing: 2 warmup + 7 iterations per row, reporting median (p10 – p90).
Memory: `CSR mem` is the in-RAM size of the sparse weight matrix; `blob size` is the serialized cache payload (what `LocalWeightCache`/`RedisWeightCache` writes); `ΔRSS` is the process-RSS high-water-mark delta observed during that row.

Batch shapes follow ECMWF forecast conventions: `member=50` is a 50-perturbed-member ensemble (one slice per member), and `step` is forecast lead time (e.g., `step=40` is 40 lead times — a 10-day forecast sampled at 6 h). A `(member=50, step=10)` DataArray therefore contains 500 forecast slices stacked along two batch dims; `aggregate` flattens them, runs one sparse · dense matmul, and reshapes the result.

### `compute_weights` (one-time per (grid, polygons))

| n_polygons | factor | median  (p10 – p90)       | CSR mem | blob size | ΔRSS     |
| ---------- | ------ | ------------------------- | ------- | --------- | -------- |
| 50         | 1      | 9.1 ms  (8.7 ms – 9.7 ms) | 4 KB    | 6 KB      | 4.0 MB   |
| 50         | 4      | 170 ms  (165 ms – 180 ms) | 12 KB   | 14 KB     | 185.7 MB |
| 507        | 1      | 119 ms  (115 ms – 122 ms) | 39 KB   | 53 KB     | 0 B      |
| 507        | 4      | 297 ms  (287 ms – 344 ms) | 122 KB  | 136 KB    | 14.6 MB  |
| 5571       | 1      | 1.36 s  (1.31 s – 1.40 s) | 430 KB  | 579 KB    | 0 B      |
| 5571       | 4      | 1.62 s  (1.60 s – 1.78 s) | 1.3 MB  | 1.5 MB    | 110.5 MB |

### `aggregate` (hot path)

| n_polygons | batch                | slices | factor     | median  (p10 – p90)       | ΔRSS     |
| ---------- | -------------------- | ------ | ---------- | ------------------------- | -------- |
| 50         | (member=50,)         | 50     | 1          | 5.3 ms  (4.7 ms – 5.5 ms) | 0 B      |
| 507        | (member=50,)         | 50     | 1          | 5.0 ms  (4.8 ms – 5.2 ms) | 0 B      |
| 5571       | (member=50,)         | 50     | 1          | 11 ms  (9.8 ms – 13 ms)   | 0 B      |
| 5571       | (member=50, step=10) | 500    | 1          | 108 ms  (102 ms – 112 ms) | 41.2 MB  |
| 5571       | (member=50, step=40) | 2 000  | 1          | 522 ms  (502 ms – 680 ms) | 426.8 MB |
| 5571       | (member=50,)         | 50     | 4          | 11 ms  (11 ms – 14 ms)    | 0 B      |
| 5571       | (member=50,)         | 50     | 1 (1% NaN) | 16 ms  (15 ms – 18 ms)    | 0 B      |

### `compute_bias` (DAG rollup)

| n_leaves | depth | hierarchy                   | batch                | median  (p10 – p90)       | ΔRSS |
| -------- | ----- | --------------------------- | -------------------- | ------------------------- | ---- |
| 507      | 2     | medium GADM (state -> muni) | (member=50,)         | 3.8 ms  (2.4 ms – 4.6 ms) | 0 B  |
| 5571     | 2     | full GADM (state -> muni)   | (member=50,)         | 15 ms  (13 ms – 19 ms)    | 0 B  |
| 5571     | 2     | full GADM (state -> muni)   | (member=50, step=10) | 19 ms  (18 ms – 20 ms)    | 0 B  |
| 5571     | 4     | synthetic deep              | (member=50,)         | 8.9 ms  (8.6 ms – 9.5 ms) | 0 B  |

<!-- BENCHMARK END -->

Numbers are point-in-time on the author's machine and may vary ±20% by
hardware. Cold-import overhead (~0.3 s for `import geohalo`) is excluded —
the suite measures steady-state cost. Re-generate after any perf-relevant
change.

## Why exact fractional coverage

The three common alternatives — *centroid / point-in-polygon*,
*all-touched / rasterise*, and *area-weighted intersection* — differ by
how they treat cells on the polygon boundary. The same polygon over a
5 × 5 mesh produces three very different weight vectors:

```
   Centroid                All-touched                Exact fractional
  (cell.center ∈ P)         (cell ∩ P ≠ ∅)            (area(cell ∩ P)/area(cell))

  +---+---+---+---+---+    +---+---+---+---+---+    +-----+-----+-----+-----+-----+
  | 0 | 0 | 0 | 0 | 0 |    | 0 | 1 | 1 | 1 | 0 |    | 0.0 | 0.1 | 0.4 | 0.2 | 0.0 |
  +---+---+---+---+---+    +---+---+---+---+---+    +-----+-----+-----+-----+-----+
  | 0 | 0 | 1 | 1 | 0 |    | 0 | 1 | 1 | 1 | 1 |    | 0.0 | 0.6 | 1.0 | 0.8 | 0.1 |
  +---+---+---+---+---+    +---+---+---+---+---+    +-----+-----+-----+-----+-----+
  | 0 | 1 | 1 | 0 | 0 |    | 1 | 1 | 1 | 1 | 0 |    | 0.3 | 1.0 | 1.0 | 0.4 | 0.0 |
  +---+---+---+---+---+    +---+---+---+---+---+    +-----+-----+-----+-----+-----+
  | 0 | 1 | 0 | 0 | 0 |    | 1 | 1 | 1 | 0 | 0 |    | 0.4 | 0.7 | 0.3 | 0.0 | 0.0 |
  +---+---+---+---+---+    +---+---+---+---+---+    +-----+-----+-----+-----+-----+
  | 0 | 0 | 0 | 0 | 0 |    | 0 | 0 | 0 | 0 | 0 |    | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
  +---+---+---+---+---+    +---+---+---+---+---+    +-----+-----+-----+-----+-----+

      lossy:                bloated:                  unbiased:
      misses ~½ of cells    counts every grazed       weight ∝ true overlap
      near the boundary     cell as if fully inside   area
```

| Method                | Bias                                             | Cost            |
| --------------------- | ------------------------------------------------ | --------------- |
| Centroid              | High; polygons smaller than a cell collapse to 0 | Cheap           |
| All-touched           | Overcounts by ~½-cell around the perimeter       | Cheap           |
| **Exact fractional**  | **Unbiased**                                     | One-time only   |

### Latitude correction

The coverage fraction is per-cell; the *physical* weight is per-cell-area.
On the sphere a cell's area shrinks with latitude:

```
area at  0° latitude  ████████████████████   100 %
area at 30° latitude  █████████████████       87 %
area at 60° latitude  ██████████              50 %
area at 80° latitude  ███                     17 %
```

`geohalo` multiplies each cell's fractional coverage by
`R² · Δlon · (sin(lat_top) − sin(lat_bot))` before row-normalising, so
high-latitude cells aren't over-weighted.

## Sub-cell precision via mean-preserving downscaling

Many polygons are smaller than a single grid cell. The cell mean is
their best estimate under a "uniform within a cell" assumption, but
`geohalo` ships an optional **mean-preserving downscaling** step that
refines the grid `factor×` per axis and uses neighbour information to
give sub-cell polygons a sharper answer — **without violating the
published cell-mean contract** (the average of the `factor²` children
of any parent cell equals the parent's original value exactly).

```python
weights = compute_weights(polygons, grid, target_resolution=0.05)
```

The downscale operator `M = B + P − P·A·B` (bilinear upsample plus
mean-correction) is built as a sparse matrix and **baked into the
cached `W`** at precompute time, so `aggregate()` has zero per-call
downscaling cost. See [`docs/downscaling.md`](docs/downscaling.md) for
the algorithm derivation and benchmarks.

## Rolling up a hierarchy: `compute_bias`

When leaf polygons (e.g. municipalities) belong to one or more parent
groupings (states, basins, custom zones with weighted membership),
`BiasHierarchy` precomposes the parent-child-weight DAG into a sparse
matrix so rollups also collapse to a matmul:

```python
from geohalo import BiasHierarchy, compute_bias

hierarchy = BiasHierarchy.build(
    edges=[
        (("BR", "SP"), ("BR", "SP", "muni_a"), 1.0),
        (("BR", "SP"), ("BR", "SP", "muni_b"), 1.0),
        (("BR", "RJ"), ("BR", "RJ", "muni_c"), 1.0),
    ],
    key_names=("country", "state", "muni"),
)

rolled = compute_bias(leaf_aggregates, hierarchy)
```

Each parent's value is the normalised weighted average of its
transitively-contributing leaves. NaN handling is symmetric to
`aggregate`: parents with no finite contributing leaf return `NaN`
(`on_nan_child="ignore"`, the default), or you can opt into a hard
failure (`on_nan_child="raise"`).

## Non-goals

- **No reprojection** — EPSG:4326 throughout (grids and polygons).
- **No per-variable cache** — `W` depends on grid + polygons only.
- **No WGS84-ellipsoidal cell areas** — spherical is within ~0.3 %.

## Development

```bash
uv sync                                          # install deps
uv run python examples/visualize.py              # end-to-end example
```
