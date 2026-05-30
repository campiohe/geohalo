<p align="center">
  <img src="docs/figures/logo.svg" alt="geohalo" width="180"/>
</p>

<h1 align="center">geohalo</h1>

<p align="center">
  <em>Exact-fractional-area zonal statistics over regular lat/lon grids.</em>
</p>

<p align="center">
  <a href="https://pypi.org/project/geohalo/"><img src="https://img.shields.io/pypi/v/geohalo.svg?color=3775a9&logo=pypi&logoColor=white" alt="PyPI"/></a>
  <a href="https://pypi.org/project/geohalo/"><img src="https://img.shields.io/pypi/pyversions/geohalo.svg?color=3776ab&logo=python&logoColor=white" alt="Python versions"/></a>
  <a href="LICENSE"><img src="https://img.shields.io/pypi/l/geohalo.svg?color=blue" alt="License: MIT"/></a>
  <a href="https://campiohe.github.io/geohalo/"><img src="https://img.shields.io/badge/docs-geohalo-f59e0b?logo=materialformkdocs&logoColor=white" alt="Documentation"/></a>
  <a href="https://github.com/campiohe/geohalo/actions/workflows/release.yml"><img src="https://img.shields.io/github/actions/workflow/status/campiohe/geohalo/release.yml?label=release&logo=github" alt="Release workflow"/></a>
  <a href="https://github.com/astral-sh/uv"><img src="https://img.shields.io/badge/managed%20by-uv-de5fe9.svg?logo=astral&logoColor=white" alt="Managed by uv"/></a>
</p>

---

Given a regular lat/lon mesh of gridded values — temperature, precipitation,
population density, a land-cover fraction, a satellite band, … (loaded with
`xarray` from GRIB, NetCDF, Zarr, …) — and an arbitrary set of polygons,
`geohalo` reduces the spatial dimensions of the mesh to one value per
polygon with **sub-cell precision** and **millisecond-scale aggregation**
in the hot path.

The expensive geometric work happens once; every subsequent grid
collapses to a single sparse · dense matmul.

## How it works

Aggregation is a linear operator:

```
aggregates = W @ flat_grid_values
```

where `W ∈ ℝ^(N_polygons × N_cells)` is a sparse matrix whose entries are
the **exact fractional area** of `cell ∩ polygon` weighted by each cell's
true surface area on a sphere; the `how="mean"` hot path divides by each
polygon's total overlap area.

`W` (the `Stencil`) depends only on the grid topology and the polygon set —
not on the grid values. The library splits the work into two phases:

1. **Precompute** (`Stencil.compute`, one-time per `(grid, polygons)` pair):
   call [exactextract](https://github.com/isciences/exactextract) for
   exact fractional cell coverage, multiply by per-cell spherical area
   `R² · Δlon · (sin(lat_top) − sin(lat_bot))` to correct for latitude,
   store as a `scipy.sparse.csr_matrix`, and optionally cache the result.

2. **Reduce** (`reduce_with_stencil`, hot path):
   one sparse · dense matmul, broadcast over every non-spatial dim
   (time, ensemble member, band, level, …). NaN-aware: a second matmul against a
   validity mask renormalises per slice without rebuilding `W`. When the
   input grid differs from the stencil's, a cached `Resampler` matrix is
   fused in (`occupancy @ transform`) so it stays a single matmul.

## Install

`geohalo` targets **Python ≥ 3.12**.

```bash
uv add geohalo            # or: pip install geohalo
```

Optional extras:

- `redis` — for the `RedisCache` backend
- `matplotlib` — for the helpers in `geohalo.plot`

## Quickstart

```python
import numpy as np
import geopandas as gpd
import xarray as xr
from shapely.geometry import box
import geohalo as ghl

# any regular lat/lon DataArray works; a synthetic field so this runs as-is
lats = np.arange(-25.0, -19.0, 0.25)
lons = np.arange(-50.0, -42.0, 0.25)
lon2d, lat2d = np.meshgrid(lons, lats)
field = 290.0 + 5.0 * np.cos(np.deg2rad(4 * lat2d)) + 0.1 * lon2d

da = xr.DataArray(
    field, dims=("latitude", "longitude"),
    coords={"latitude": lats, "longitude": lons}, name="value",
)

geoms = gpd.GeoSeries(
    [box(-49, -24, -47, -22), box(-47, -24, -45, -22), box(-46, -22, -44, -20)],
    index=["SP", "RJ", "MG"],            # the index holds the keys
)

out = ghl.reduce(da, geoms)                  # hot path; ms-scale
out_fine = ghl.reduce(da, geoms, target_resolution=0.05)   # refine the grid first
# out: xr.DataArray over (..., geom)
```

The output preserves every non-spatial dim of `da` (time, ensemble member,
band, vertical level, …) and replaces `(latitude, longitude)` with
a single `geom` dim indexed by the GeoSeries keys.

`reduce` also accepts an `xr.Dataset` (every spatial data var is reduced),
`how={"mean", "sum"}`, a `weight_key` naming a per-cell weight variable, and
`spherical_correction=False` to disable the latitude-area correction.

### Caching

The expensive precompute is the `Stencil` (and, when resampling, the
`Resampler`). Wrap them in a cache so they run once per `(grid, polygons)`:

```python
import geohalo as ghl   # ghl.RedisCache is also available

cache = ghl.LocalCache("./.geohalo-cache")
stencil = cache.get_or_compute_stencil(da.latitude.values, da.longitude.values, geoms)
out = ghl.reduce_with_stencil(da, stencil)
```

Each cached object's key is a SHA-256 digest of its inputs (grid coords +
spherical flag + polygons for a `Stencil`; source/target coords + iterations
for a `Resampler`), so any change to those inputs invalidates the cache
implicitly.

#### Fused reduce operator

When you reduce over a *resampled* grid, the resample (`Resampler` matrix `T`)
and the aggregation (stencil occupancy `W`) compose into one operator `M = W·T`
that acts directly on the source grid. `geohalo` builds `M` without ever
materialising `T` or the fine field — `W` has only `n_polygons` rows, so the
fusion stays thin. `ReduceOperator` is that fused operator, and it's by far the
most compact thing to cache (it does not grow with target resolution or
iteration count):

```python
import geohalo as ghl   # plus get_or_compute_reduce_operator on the cache

op = cache.get_or_compute_reduce_operator(
    stencil, da.latitude.values, da.longitude.values, iterations=3,
)
out = ghl.reduce_with_operator(da, op)     # (..., geom); also accepts how="sum"
```

For a 0.25° → 0.05° refine (~3.2M target cells) over 500 polygons, the
materialised resampler is a 358 MB cache blob and **cannot build at all** at
`iterations=3`; the fused `ReduceOperator` is a **0.40 MB** blob, builds in
~0.5 s, and loads in ~0.5 ms. The clean fast path of `reduce` /
`reduce_with_stencil` uses the same fusion internally; cache the operator when
you apply it repeatedly (many grid slices — time steps, members, bands — or
runs). See [the reduce-operator guide](https://campiohe.github.io/geohalo/concepts/reduce-operator/).

## Resampling grids

Resampling is a first-class, reusable operation. `resample_grid` builds a
value-independent sparse `Resampler` matrix (cacheable via
`LocalCache.get_or_compute_resampler`) and applies it:

```python
import geohalo as ghl

fine = ghl.resample_grid(da, target_resolution=0.05, iterations=3)
```

It works in either direction (refine or coarsen); mean-preservation is exact
wherever geometrically possible. See [the downscaling guide](https://campiohe.github.io/geohalo/concepts/downscaling/).

## Performance

The headline claim — *millisecond-scale aggregation in the hot path* — is
backed by the suite under [`benchmarks/`](benchmarks/run.py). All rows time
operations against the GADM Brazil L2 polygons (~5570 munis) on a synthetic
0.25° grid covering Brazil (160×160 = 25,600 cells). Re-run with
`uv run python -m benchmarks.run`.

<!-- BENCHMARK START -->

Environment: Python 3.14.4 on Linux x86_64, geohalo @ f13ac45, scipy 1.17.1, numpy 2.4.6, shapely 2.1.2, xarray 2026.4.0, exactextract 0.3.0, geopandas 1.1.3.
Grid: 0.25° over Brazil bbox (-74, -34, -34, 6) — 160×160 = 25,600 cells.
Polygons: GADM Brazil L2 (~5570 total).
Timing: 2 warmup + 7 iterations per row, reporting median (p10 – p90).
Memory: `CSR mem` is the in-RAM size of the sparse matrix; `blob size` is the serialized cache payload (what `LocalCache`/`RedisCache` writes); `ΔRSS` is the process-RSS high-water-mark delta observed during that row.

Batch shapes follow ECMWF forecast conventions: `member=50` is a 50-perturbed-member ensemble (one slice per member), and `step` is forecast lead time (e.g., `step=40` is 40 lead times — a 10-day forecast sampled at 6 h). A `(member=50, step=10)` DataArray therefore contains 500 forecast slices stacked along two batch dims; `reduce_with_stencil` flattens them, runs one sparse · dense matmul, and reshapes the result.

### `Stencil.compute` (one-time per (grid, polygons))

| n_polygons | factor | median  (p10 – p90)       | CSR mem | blob size | ΔRSS     |
| ---------- | ------ | ------------------------- | ------- | --------- | -------- |
| 50         | 1      | 21 ms  (19 ms – 28 ms)    | 6 KB    | 11 KB     | 6.2 MB   |
| 50         | 4      | 23 ms  (22 ms – 34 ms)    | 56 KB   | 67 KB     | 6.0 MB   |
| 507        | 1      | 170 ms  (161 ms – 181 ms) | 38 KB   | 55 KB     | 11.2 MB  |
| 507        | 4      | 187 ms  (182 ms – 196 ms) | 275 KB  | 299 KB    | 4.4 MB   |
| 5571       | 1      | 2.06 s  (1.96 s – 2.18 s) | 430 KB  | 581 KB    | 138.9 MB |
| 5571       | 4      | 2.27 s  (2.20 s – 2.42 s) | 3.1 MB  | 3.2 MB    | 24.4 MB  |

### `Resampler.compute` (one-time per (source grid, target grid))

`factor=4` here means refining the 160×160 grid to 637×637. The power series
`Σⱼ Gʲ·B` (`G = I − B·A`) is accumulated by applying `G` to `B` on the right, so
every intermediate stays `(n_target, n_source)` — the dense `(n_target,
n_target)` operator is never materialised. Higher iteration counts still cost
more (the transform fills in as its reach grows), but the build stays
sub-second and the default `resample_iterations=1` is cheapest:

| grid               | iterations | median  (p10 – p90)       | CSR mem  | ΔRSS     |
| ------------------ | ---------- | ------------------------- | -------- | -------- |
| 160x160 -> 637x637 | 1          | 95 ms  (93 ms – 130 ms)   | 43.1 MB  | 56.8 MB  |
| 160x160 -> 637x637 | 3          | 1.24 s  (939 ms – 1.28 s) | 224.0 MB | 770.7 MB |

### `reduce_with_stencil` (hot path)

| n_polygons | batch                | slices | factor     | median  (p10 – p90)       | ΔRSS    |
| ---------- | -------------------- | ------ | ---------- | ------------------------- | ------- |
| 50         | (member=50,)         | 50     | 1          | 3.6 ms  (3.2 ms – 6.2 ms) | 0 B     |
| 507        | (member=50,)         | 50     | 1          | 3.7 ms  (3.4 ms – 4.1 ms) | 0 B     |
| 5571       | (member=50,)         | 50     | 1          | 5.8 ms  (4.9 ms – 7.5 ms) | 0 B     |
| 5571       | (member=50, step=10) | 500    | 1          | 196 ms  (189 ms – 209 ms) | 0 B     |
| 5571       | (member=50, step=40) | 2 000  | 1          | 670 ms  (625 ms – 942 ms) | 30.6 MB |
| 5571       | (member=50,)         | 50     | 4          | 113 ms  (111 ms – 118 ms) | 0 B     |
| 5571       | (member=50,)         | 50     | 1 (1% NaN) | 14 ms  (14 ms – 15 ms)    | 0 B     |

### `reduce_with_stencil` on `xr.Dataset` (hot path)

| n_polygons | batch                | n_vars | slices | median  (p10 – p90)       | ΔRSS |
| ---------- | -------------------- | ------ | ------ | ------------------------- | ---- |
| 5571       | (member=50,)         | 3      | 50     | 16 ms  (14 ms – 21 ms)    | 0 B  |
| 5571       | (member=50, step=10) | 3      | 500    | 512 ms  (487 ms – 534 ms) | 0 B  |

### `resample_grid_with_matrix` (hot path)

Applying a prebuilt `Resampler` matrix to data — the per-call cost of
resampling (distinct from the one-time `Resampler.compute` above):

| grid               | batch        | slices | median  (p10 – p90)       | ΔRSS |
| ------------------ | ------------ | ------ | ------------------------- | ---- |
| 160x160 -> 319x319 | (member=50,) | 50     | 45 ms  (41 ms – 56 ms)    | 0 B  |
| 160x160 -> 637x637 | (member=50,) | 50     | 139 ms  (136 ms – 155 ms) | 0 B  |

### `aggregate_bias_with_tree` (tree rollup)

| n_leaves | depth | hierarchy                   | batch                | median  (p10 – p90)       | ΔRSS |
| -------- | ----- | --------------------------- | -------------------- | ------------------------- | ---- |
| 507      | 2     | medium GADM (muni -> state) | (member=50,)         | 1.1 ms  (0.9 ms – 1.2 ms) | 0 B  |
| 5571     | 2     | full GADM (muni -> state)   | (member=50,)         | 5.6 ms  (5.0 ms – 6.8 ms) | 0 B  |
| 5571     | 2     | full GADM (muni -> state)   | (member=50, step=10) | 19 ms  (17 ms – 23 ms)    | 0 B  |
| 5571     | 4     | synthetic deep              | (member=50,)         | 2.8 ms  (2.7 ms – 4.4 ms) | 0 B  |

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
out = ghl.reduce(da, geoms, target_resolution=0.05)
```

`reduce` builds a `Stencil` on the refined target grid and a `Resampler`
that maps the source grid onto it; the hot path fuses `occupancy @ transform`
into a single matmul. The resample matrix is the N-iteration generalization
of the classic `M = B + P − P·A·B` operator and preserves each source cell's
mean exactly. See [the downscaling guide](https://campiohe.github.io/geohalo/concepts/downscaling/).

## Rolling up a hierarchy: `aggregate_bias`

When leaf polygons (e.g. municipalities) belong to a parent grouping
(states, basins, custom zones), `aggregate_bias` precomposes the
parent-child tree into a sparse matrix so rollups also collapse to a matmul:

```python
import pandas as pd
import geohalo as ghl

edges = pd.DataFrame(
    {"parent": ["SP", "SP", "RJ"]},
    index=pd.Index(["muni_a", "muni_b", "muni_c"], name="child"),
)

rolled = ghl.aggregate_bias(leaf_aggregates, edges)
```

The DataFrame index is the child; the `parent` column is its parent. Each
child has at most one parent — `geohalo` enforces this (tree, not DAG). Pass
`how="sum"` for weighted sums, or a `weight_col` for non-uniform edge weights.
Each parent's value is the normalised weighted average (or sum) of its
transitively-contributing leaves; NaN leaves are dropped and the remaining
weights renormalised.

## Non-goals

- **No reprojection** — EPSG:4326 throughout (grids and polygons).
- **No per-variable cache** — the `Stencil` depends on grid + polygons only.
- **No WGS84-ellipsoidal cell areas** — spherical is within ~0.3 %
  (`spherical_correction=False` gives planar/equal-area weights).
- **No DAG hierarchies** — each child has exactly one parent (tree only).
- **No `how={"min", "max"}`** — `mean` and `sum` only.

## Development

```bash
uv sync                                          # install deps
uv run pytest                                    # tests
uv run ruff check .                              # lint
uv run --group docs mkdocs serve                 # preview the docs locally
uv run --group docs python docs/gen_figures.py   # regenerate the doc figures
```

Docs are built with [Material for MkDocs](https://squidfunk.github.io/mkdocs-material/)
and deployed to <https://campiohe.github.io/geohalo/> on every push to `main`.
