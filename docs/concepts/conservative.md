# Conservative regridding

Use `method="conservative"` when grid values represent cell averages and you
need spherical area-overlap averages. It works for coarsening or refinement,
including non-nested grids and arbitrary resolution ratios. It does not smooth
inside a source cell: nested refinement replicates that cell's value.

The default [`method="meanpreserving"`](downscaling.md) is unchanged. That
method smooths during refinement and preserves each parent's unweighted mean
over its nearest-parent children. It can overshoot at discontinuities and is
not an area-weighted coarsener.

## Spherical overlaps, two 1-D factors

For an EPSG:4326 cell bounded by constant latitudes and longitudes, its area is

\[
A = R^2\,\Delta\lambda\,[\sin(\phi_N)-\sin(\phi_S)].
\]

Thus overlap fractions separate into a latitude factor measured in
`sin(latitude)` and a longitude factor measured in degrees. The radius and
degree-to-radian longitude conversion cancel in the ratios. The default weights
are overlap area divided by the **full target cell area**:

\[
y_t = \sum_s \frac{A_{t\cap s}}{A_t} x_s.
\]

The two sparse factors are stored as `resampler.axis_weights`, ordered
`(latitude, longitude)`. Applying them takes two sparse contractions per batch
slice, choosing the smaller intermediate. A full grid-to-grid Kronecker matrix
is never built: for this method, `resampler.transform_matrix is None`.
Use `resampler.apply_grid(values)` or `resample_grid_with_matrix` for application.
Mean-preserving resamplers still expose their existing CSR transform matrix.

For finite data and matching footprints, the default preserves
`sum(value * cell_area)` up to floating-point roundoff. For fully covered cells,
weights are nonnegative and sum to one, so values remain within the source
range. On partially covered cells, uncovered area contributes zero, which can
move a positive constant below its original value; see normalization below.

## Matched footprints

This example describes the same 40°–60° latitude, 0°–20° longitude footprint
on two nested grids. It returns spherical-area-weighted 4×4 block means:

```python
import numpy as np
import xarray as xr
import geohalo as ghl

fine_lat = np.arange(40.25, 60.0, 0.5)
fine_lon = np.arange(0.25, 20.0, 0.5)
coarse_lat = np.arange(41.0, 60.0, 2.0)
coarse_lon = np.arange(1.0, 20.0, 2.0)
source = xr.DataArray(
    np.random.default_rng(0).random((40, 40)),
    dims=("latitude", "longitude"),
    coords={"latitude": fine_lat, "longitude": fine_lon},
)
resampler = ghl.Resampler.compute(
    fine_lat, fine_lon, coarse_lat, coarse_lon, method="conservative",
)
coarse = ghl.resample_grid_with_matrix(source, resampler)
```

Conservation over the **entire source** requires that the target cells cover
its footprint. Otherwise only the overlapping portion contributes. The
convenience call `resample_grid(source, 2, method="conservative")` retains the
existing source-centre-based target generation, not edge-aligned generation:
its target centres would start at 40.25° and 0.25°, not at 41° and 1° as above.
Do not assume matching cell extents merely because centre ranges are similar.
Use explicit target coordinates or bounds when the footprint matters.

## Coordinates and cell bounds

Conservative coordinates must be finite, one-dimensional, and strictly
monotonic. Both ascending and descending axes work. Source latitude is stored
ascending, as for the existing resampler; the xarray adapter handles its stored
orientation. Source longitude and target row order are retained. The NumPy
`apply_grid` method accepts `descending=True` for descending source latitude.

Without explicit bounds, midpoint edges are inferred. Inferred latitude edges
are clipped at −90° and +90°; centres and explicit latitude bounds must already
be inside that range. Midpoint inference requires at least two coordinates,
except a singleton periodic source longitude, which covers a full cycle.

Pass `source_bounds=(lat_edges, lon_edges)` and/or
`target_bounds=(lat_edges, lon_edges)` to the builder or cache method when cell
edges differ from these assumptions. Each edge array has `N + 1` entries for
`N` centres, follows that axis's direction, and must contain each centre inside
its cell. Explicit bounds support single-cell source or target axes:

```python
one_cell = ghl.Resampler.compute(
    fine_lat, fine_lon, np.array([50.0]), np.array([10.0]),
    method="conservative",
    target_bounds=(np.array([40.0, 60.0]), np.array([0.0, 20.0])),
)
```

`resample_grid` accepts `source_bounds` and derives target bounds from the
requested resolution, so a coarsened singleton target also works. Its generated
target coordinates are unchanged from the existing helper. Irregular grids are
supported by conservative regridding, though polygon stencils still require
regular raster coordinates. Computing physical areas for explicit or clipped
bounds must use those bounds in the area formula above.

With `period=360`, longitude overlaps wrap across the seam. Inferred source
longitude bounds use cyclic midpoints and cover a full cycle; explicit source
longitude bounds must cover exactly one period. Source and target centre spans
must be less than a period, with no repeated endpoint. Target bounds may cover
at most one period. Explicit targets remain regional unless their supplied or
inferred bounds cover a full cycle; they may be shifted by whole periods.
The convenience API generates a full cycle and uses cyclic target midpoint
bounds. Latitude never wraps, and polygon geometries are not changed.

## Partial coverage and missing values

The builder's `normalization` accepts:

- `"destination"` (default): divide by full target area. A constant 10 covering
  half a target cell produces 5; this retains the integral over the overlap.
- `"covered"`: divide by source-covered target area. The same cell produces
  10. For an integral, multiply by the **covered** area, not the full target area.

These correspond to the destination-area and fraction-area conventions
described in the [ESMF normalization documentation](https://earthsystemmodeling.org/docs/release/ESMF_8_4_1/ESMF_refdoc/node3.html).
`resampler.coverage` returns a `(target_lat, target_lon)` array of covered-area
fractions, independent of data values. Cells with no overlap return NaN under
either normalization; the raw 1-D matrices contain zero rows for them.

NaNs propagate from overlapping source cells by default. Opt into valid-area
averaging at application time:

```python
coarse = ghl.resample_grid_with_matrix(source, resampler, skipna=True)
flat = resampler.apply_grid(source.to_numpy(), skipna=True)
```

`skipna=True` projects both zero-filled values and a validity mask, then divides
once after **both** axis contractions. This overrides destination normalization:
the denominator is valid covered area, even on clean partially covered cells.
All-missing and unmapped cells return NaN. Only NaNs are excluded; infinities
are not treated as missing. Valid-area renormalization is not a promise to
preserve the original full-area integral when coverage or data are missing.

Resampling `skipna=True` is currently supported only for conservative operators;
using it with the mean-preserving method raises `ValueError`. This does not
change the separate `skipna` behavior of prebuilt polygon reducers.

## Reuse, memory, and scope

`LocalCache.get_or_compute_resampler` and its Redis counterpart accept the same
method, normalization, period, and bounds options. Cache hits restore the two
axis factors without rebuilding; `skipna` is apply-time and does not change the
key. Both methods use the [portable NPZ format](serialization.md); input digests
are unchanged, while legacy pickle cache entries require a one-time rebuild.

Conservative coefficients and real-valued results use float64. Work is bounded
per batch slice, including missing-data masks and reversed/strided inputs; the
dense output still needs to fit in memory. As before, the xarray resampling
adapter eagerly loads lazy inputs before application. This is not chunk-aware
source reading or a new Dask execution mode.

Leave `iterations=1` with `method="conservative"`; other values are rejected
because iteration only controls the mean-preserving algorithm. `FactoredResampler`
and fused polygon reducers remain mean-preserving. To conservatively regrid
before polygon reduction, first resample the field, then reduce it on the
resulting grid. No reprojection or higher-order reconstruction is introduced.
