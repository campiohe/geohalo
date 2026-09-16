# Resampling grids

Resampling is a **first-class, reusable** operation in geohalo — not just a hidden step
inside `reduce`. If you want the refined (or coarsened) field itself, `resample_grid`
gives it to you, backed by the same [mean-preserving](../concepts/downscaling.md) sparse
transform.

## One call

```python
import geohalo as ghl

fine = ghl.resample_grid(da, target_resolution=0.05, iterations=3)
```

It works in **either direction** — a smaller `target_resolution` refines, a larger one
coarsens — and mean-preservation is exact wherever geometrically possible (always when
refining; coarsening can't preserve a source cell that has no target child).

`resample_grid` accepts an `xr.DataArray` or an `xr.Dataset` (every spatial data variable
is resampled, the rest pass through) and preserves all non-spatial dims.

## Separating build from apply

`resample_grid` builds a `Resampler` and applies it in one shot. To reuse the transform
across many slices, build it once and apply with `resample_grid_with_matrix`:

```python
import geohalo as ghl

cache = ghl.LocalCache("./.geohalo-cache")
t_lat, t_lon = ghl.geometry.target_coords_from_resolution(da.latitude.values, da.longitude.values, 0.05)

resampler = cache.get_or_compute_resampler(
    da.latitude.values, da.longitude.values, t_lat, t_lon, iterations=3,
)
fine = ghl.resample_grid_with_matrix(da, resampler)
```

The `Resampler` is value-independent and [cacheable](caching.md) — built once per
`(source grid, target grid, iterations, period)`.

## Periodic longitude

Global longitude has a seam, not an outer edge. Opt into wrapping with
`period=360`; the default `period=None` keeps the existing clamped behavior.
Latitude is never wrapped by the grid APIs.

```python
fine = ghl.resample_grid(global_da, target_resolution=0.5, period=360)
out = ghl.reduce(global_da, geoms, target_resolution=0.5, period=360)
```

With `period` set, `resample_grid`, `reduce(..., target_resolution=...)`, and
`geometry.target_coords_from_resolution` generate **one full longitude cycle**,
starting at the minimum source longitude and excluding the repeated endpoint.
For source centres `-180, -178, …, 178`, a 0.5° target includes
`-180, -179.5, …, 179.5`. Choose a resolution that divides the period if you need
the spacing across the seam to equal the interior spacing. Without `period`,
target generation continues to use the source min/max extent.

Explicit targets passed to `Resampler.compute`, `FactoredResampler.compute`, or
their cache methods are **not extended, sorted, or relabelled**. For example,
targets `179.5`, `-180.5`, and `539.5` all sample the same location. Sources may
use either the −180°/180° or 0°/360° convention, or another shifted cycle, and
may be ascending or descending. Source columns retain their supplied order.

```python
resampler = cache.get_or_compute_resampler(
    global_da.latitude.to_numpy(), global_da.longitude.to_numpy(),
    target_lat, target_lon, iterations=3, period=360,
)
fine = ghl.resample_grid_with_matrix(global_da, resampler)

op = cache.get_or_compute_reduce_operator(
    stencil, global_da.latitude.to_numpy(), global_da.longitude.to_numpy(),
    iterations=3, period=360,
)
out = ghl.reduce_with_operator(global_da, op)
```

The period is baked into the operators, so apply calls do not need it again.
`reduce_with_stencil(..., period=360)` also supports the option, including its
masked and weighted paths. Restricted read plans automatically include the
source chunks on both sides of the seam, without reading intervening chunks.

The underlying linear weights now give the seam blend from issue #22:

```python
import numpy as np

from geohalo.geometry import bilinear_matrix_1d

lon = np.arange(-180.0, 180.0, 2.0)
weights = bilinear_matrix_1d(lon, np.array([179.5]), period=360)
weights.toarray()[0, [0, -1]]  # [0.75, 0.25]: −180° and 178°
```

The full resampler additionally performs its mean-preserving correction; its
output is not just this bilinear blend. Both interpolation and nearest-parent
assignment wrap, preserving each occupied parent's **unweighted child mean**
across the seam. A nearest-parent tie chooses the lower *unwrapped* neighbour:
179° lies halfway between 178° and 180° (the wrapped −180° cell), so it belongs
to 178°. The existing overshoot and missing-data semantics are unchanged.

### Input contract and limits

The period must be positive and finite. Periodic source coordinates must be
finite, one-dimensional, strictly monotonic, and span **less than** the period.
Do not include both −180° and +180°, or both 0° and 360°: they repeat a cell.
A single source centre is constant everywhere. Periodic targets must be finite
and one-dimensional; their order is unrestricted. The 1-D helpers support
irregular spacing, while stencil construction still requires a regular raster.

Setting `period` explicitly declares that the source represents a cycle; there
is no automatic global-grid detection. Using it on a regional source connects
the last centre to the first across the remaining gap and makes resolution-based
calls generate a full cycle, so leave it unset for ordinary regional grids.

This option changes **sampling only**. Stencils still use ordinary EPSG:4326
polygon coordinates: it does not split, unwrap, or duplicate geometries across
the antimeridian. Supply geometries in the stencil's coordinate domain, splitting
or unwrapping them yourself as needed. A prebuilt stencil can use an unwrapped
regional target such as 175°–185° while sampling a −180°–180° source periodically.
Without resampling, `reduce(..., period=360)` does not add periodic polygon
coverage. This is also not spherical-area-conservative regridding.

## Choosing `iterations`

`iterations` controls how far the [mean-preserving correction](../concepts/downscaling.md)
reaches across the grid. Every value preserves each parent cell's mean exactly; higher
counts trade build cost for smoothness.

| `iterations` | Character                                    | Cost              |
| ------------ | -------------------------------------------- | ----------------- |
| `1` (default) | classic operator, blocky but exact          | cheapest          |
| `2`–`3`      | visibly smoother, still sub-second to build  | moderate          |
| higher       | smoother still; transform fills in further   | grows with reach  |

```python
coarse = ghl.resample_grid(da, target_resolution=0.5)          # coarsen, iters=1
smooth = ghl.resample_grid(da, target_resolution=0.05, iterations=4)
```

## Cost note: materialised vs fused

`resample_grid` materialises the full transform \(\mathbf{T}\), because you asked for the
**field**. That matrix can be large for a big refinement (hundreds of MB at fine target
resolutions and high iteration counts).

If the refined field is only a stepping stone to **per-polygon values**, you don't need
\(\mathbf{T}\) at all — go through `reduce(..., target_resolution=…)`, which
[fuses the resample into the stencil](../concepts/reduce-operator.md) and never builds
the fine grid:

```python
# materialises the fine field (you want the grid)
fine = ghl.resample_grid(da, target_resolution=0.05, iterations=3)

# never materialises the fine field (you want polygon values)
out = ghl.reduce(da, geoms, target_resolution=0.05, resample_iterations=3)
```

## Handling descending latitudes

ECMWF and many GRIB products ship latitudes **descending** (90 → −90). geohalo's
`Resampler.compute` and `FactoredResampler.compute` canonicalise source latitudes
to ascending order when building their matrices. The xarray apply helpers map
matrix columns to the input's stored latitude order, so you do not need to flip
anything yourself or copy a descending grid into ascending order.
Ascending and descending versions of the same source grid share one cache entry.

`resample_grid_with_matrix` checks both latitude and longitude coordinates against
the resampler's source grid and raises `ValueError` on a mismatch, even when the
grid shapes are identical. Target coordinates keep the order supplied when the
resampler was built.

If you multiply `Resampler.transform_matrix` directly, or call
`FactoredResampler.apply_flat`, flatten source values with **ascending latitude**
and longitude in `source_lon` order. This corrects the convention in 1.1.0, where
matrices built from descending latitudes expected descending values, causing the
xarray helpers to flip north and south.

## Application memory

For large source grids, the xarray helpers apply the transform one batch slice at
a time. Small contiguous grids use bounded batches. This avoids copying or
casting every source slice at once while preserving the matrix's arithmetic
precision. The dense target output still needs to fit in memory, alongside
temporary storage for one source/target slice and cached matrix indices for
descending latitudes. Lazy inputs are still loaded in full before application.
