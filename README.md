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

The expensive geometric work happens once; every subsequent grid collapses
to a single sparse · dense matmul.

📖 **Full documentation: <https://campiohe.github.io/geohalo/>**

## How it works

Aggregation is a linear operator:

```
aggregates = W @ flat_grid_values
```

where `W ∈ ℝ^(N_polygons × N_cells)` is a sparse matrix whose entries are the
**exact fractional area** of `cell ∩ polygon` weighted by each cell's true
surface area on a sphere. `W` (the `Stencil`) depends only on the grid topology
and the polygon set — not on the grid values — so it is built once (and
cacheable) and reused for every slice. See
[Aggregation as a linear operator](https://campiohe.github.io/geohalo/concepts/linear-operator/).

## Install

`geohalo` targets **Python ≥ 3.12**.

```bash
uv add geohalo            # or: pip install geohalo
```

Optional extras: `redis` (the `RedisCache` backend) and `matplotlib`
(the helpers in `geohalo.plot`).

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
band, vertical level, …) and replaces `(latitude, longitude)` with a single
`geom` dim indexed by the GeoSeries keys.

`reduce` also accepts an `xr.Dataset` (every spatial data var is reduced),
`how={"mean", "sum"}`, a `weight_key` naming a per-cell weight variable, and
`spherical_correction=False` to disable the latitude-area correction.

## Documentation

Everything is covered in depth at **<https://campiohe.github.io/geohalo/>**:

- **[Quickstart](https://campiohe.github.io/geohalo/quickstart/)** — runnable
  examples: batching, datasets, weights, NaN handling, refining, rollups.
- **Concepts** —
  [linear operator](https://campiohe.github.io/geohalo/concepts/linear-operator/) ·
  [the stencil](https://campiohe.github.io/geohalo/concepts/stencil/) ·
  [why exact fractional coverage](https://campiohe.github.io/geohalo/concepts/exact-coverage/) ·
  [latitude correction](https://campiohe.github.io/geohalo/concepts/latitude-correction/) ·
  [mean-preserving downscaling](https://campiohe.github.io/geohalo/concepts/downscaling/) ·
  [the fused reduce operator](https://campiohe.github.io/geohalo/concepts/reduce-operator/) ·
  [NaN-aware & weighted reduction](https://campiohe.github.io/geohalo/concepts/masked/) ·
  [hierarchical rollups](https://campiohe.github.io/geohalo/concepts/bias-tree/)
- **Guides** —
  [caching the precompute](https://campiohe.github.io/geohalo/guides/caching/) ·
  [resampling grids](https://campiohe.github.io/geohalo/guides/resampling/)
- **[API reference](https://campiohe.github.io/geohalo/api/)**

## Performance

The hot path is a single sparse · dense matmul: a 50-member batch over the
~5,570 GADM Brazil L2 municipalities reduces in **single-digit milliseconds**,
and the one-time `Stencil` precompute is seconds and cacheable. Methodology,
full tables, and the fused-operator size win are on the
[Performance](https://campiohe.github.io/geohalo/performance/) page; re-run the
suite with `uv run python -m benchmarks.run`.

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
