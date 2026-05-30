---
title: geohalo
---

# geohalo

This may not be the fastest way to put a gridded field onto polygons, but it is a
correct one — and after the first build, it is also the fast one.

geohalo computes **exact-fractional-area zonal statistics** over regular lat/lon grids.
Give it any gridded field — temperature, precipitation, population density, a land-cover
fraction, a satellite band — loaded with [xarray](https://docs.xarray.dev) from GRIB,
NetCDF, Zarr, … — and an arbitrary set of polygons, and it reduces the spatial dimensions
of the grid to *one value per polygon*, with sub-cell precision and millisecond-scale
aggregation in the hot path.

The expensive geometric work happens **once**. Every subsequent grid — every time step,
ensemble member, scenario, or band — collapses to a single sparse · dense matmul.

## Documentation

- **[Quickstart](quickstart.md)** — install, build a stencil, reduce a grid
- **Concepts**
    - [Aggregation as a linear operator](concepts/linear-operator.md) — the one idea
    - [The stencil](concepts/stencil.md) — the object that holds \(\mathbf{W}\)
    - [Why exact fractional coverage](concepts/exact-coverage.md) — unbiased boundaries
    - [Latitude correction](concepts/latitude-correction.md) — cells shrink toward the poles
    - [Mean-preserving downscaling](concepts/downscaling.md) — sub-cell precision
    - [The fused reduce operator](concepts/reduce-operator.md) — \(\mathbf{W}\mathbf{T}\) in one matmul
    - [NaN-aware & weighted reduction](concepts/masked.md) — renormalising per slice
    - [Hierarchical rollups](concepts/bias-tree.md) — leaves up the tree
- **Guides** — [Caching the precompute](guides/caching.md) ·
  [Resampling grids](guides/resampling.md)
- **[Performance](performance.md)** — the benchmark suite behind the millisecond claim
- **[API reference](api.md)** — the full public surface
