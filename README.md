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

## Non-goals

- **No reprojection** — EPSG:4326 throughout (grids and polygons).
- **No per-variable cache** — `W` depends on grid + polygons only.
- **No WGS84-ellipsoidal cell areas** — spherical is within ~0.3 %.

## Development

```bash
uv sync                                          # install deps
```
