# Chunk-aware reduction

A [fused `ReduceOperator`](reduce-operator.md) avoids constructing a fine grid,
but `reduce_with_operator` still loads the full source array. For lazy Zarr or
Dask inputs, use `RestrictedOperator` to read only the spatial chunks that
contribute to the polygons.

```python
import xarray as xr
import geohalo as ghl

# chunks={} uses backend-preferred Dask chunks; chunks=None also works without Dask.
ds = xr.open_zarr("forecast.zarr", chunks={})
da = ds["t2m"]
op = cache.get_or_compute_reduce_operator(
    stencil, da.latitude.to_numpy(), da.longitude.to_numpy(), iterations=3,
)
restricted = ghl.RestrictedOperator.from_grid(op, da)
out = ghl.reduce_with_restricted_operator(da, restricted)  # (..., geom)
totals = ghl.reduce_with_restricted_operator(da, restricted, how="sum")
```

The existing entry points are unchanged. This is an explicit, opt-in path for
clean (non-NaN, unweighted) inputs, with an **eager** xarray result. Names, attrs,
scalar and batch coordinates, and polygon MultiIndex keys are preserved.

## What the plan contains

1. The nonzero columns of the fused operator, including any resampling halo.
2. Disjoint windows in the grid's **stored** latitude order, aligned to chunks.
3. Positions of contributing cells inside each window and a compact CSR matrix.

Adjacent chunks are grouped into filled rectangles. These rectangles do not
overlap or include untouched chunks: a diagonal or ring of touched chunks is
not replaced with its larger bounding box. Zero-valued stored coefficients do
not trigger reads. Remapping columns preserves coefficient accumulation order,
including negative coefficients introduced by resampling.

Application slices each window with `isel` **before** accessing values, gathers
the contributing cells, and multiplies by the compact matrix. Latitude sorting,
full-grid materialization, and conversion to a full-width CSC matrix are avoided.

## Bring your own reader

If your service already reads arrays directly (for example, through Zarr's async
API), use the same plan without constructing xarray objects or scheduling Dask
tasks:

```python
restricted = ghl.RestrictedOperator.compute(op, stored_latitudes, 64, 64)

# Each array has shape (..., rows, columns), in restricted.windows order.
# Here values is a NumPy array; an external reader can supply the same windows.
arrays = [values[..., rows, cols] for rows, cols in restricted.windows]
gathered = restricted.gather(arrays)       # (..., contributing_cells)
means = restricted.apply(gathered)        # (..., zones), in restricted.keys order
totals = restricted.apply(gathered, how="sum")
```

`gather` checks the window count, trailing spatial shapes, and matching leading
batch shapes. It selects cells in the compact matrix's column order. The result
keeps the input dtype; mixed window dtypes use NumPy's common dtype. `apply`
preserves the existing sparse arithmetic precision and divides by `row_sums`
for means, so float32 inputs still normally produce float64 results. Neither
method changes the input arrays or introduces missing-data renormalization.

You can read concurrently, provided the resulting arrays retain the plan's
window order. Alternatively, pass a generator to `gather` to read and release
one window at a time:

```python
gathered = restricted.gather(
    values[..., rows, cols] for rows, cols in restricted.windows
)
```

The NumPy methods do not validate coordinates or storage chunks: your reader is
responsible for the plan's stored orientation, axis order, and window ordering.
They are eager, do not perform I/O, and do not manage concurrency. Process large
batch dimensions in blocks to bound the gathered buffer. The xarray adapter
uses these same methods and retains its automatic batch/window reading policy.
Dask and Zarr remain optional; xarray is still a package dependency, but no
xarray objects are needed for these methods.

An all-zero operator has no windows. `gather([])` then returns a float64 vector
of shape `(0,)`; `apply` accepts it for an unbatched result. To retain a batch
shape or input dtype, call `apply(np.empty((*batch_shape, 0), dtype=...))`
directly. Empty batch dimensions are also supported for nonempty plans.

## Chunk layouts and coordinates

`from_grid` inspects the data variable's Dask chunks first, then
`encoding["preferred_chunks"]`, then `encoding["chunks"]`. It reads coordinate
arrays but does not load grid values. Dask and Zarr are optional: install the
backend you use; geohalo does not add them as runtime dependencies.

For a grid without chunk metadata, pass explicit sizes:

```python
# Integers mean regular chunks, with a shorter final chunk if necessary.
restricted = ghl.RestrictedOperator.compute(op, da.latitude.to_numpy(), 64, 64)

# Tuples specify every chunk length and must sum to the corresponding axis size.
# Example for a 721 x 1440 grid:
restricted = ghl.RestrictedOperator.compute(
    op, da.latitude.to_numpy(), (100, 164, 200, 257), (64,) * 22 + (32,),
)
```

A plan is specific to the stored latitude orientation, longitude coordinates,
and spatial chunk boundaries. Application rejects incompatible coordinates or
known chunk layouts before loading that variable. Rebuild after rechunking or
changing orientation. After slicing or rearranging backend arrays, ensure their
encoding metadata still describes the current view; stale encoding should be
corrected or cleared before supplying explicit chunk sizes. Dimension-keyed
`preferred_chunks` survives dimension transposes, unlike positional metadata.

For efficient physical I/O, align Dask chunks with storage chunks. A Dask graph
that rechunks or transforms data may need extra upstream reads; a storage backend
may also fetch larger units, such as Zarr shards. The plan controls requested
logical windows, not those underlying backend decisions. See xarray's
[`open_zarr` chunk options](https://docs.xarray.dev/en/stable/generated/xarray.open_zarr.html)
and [Dask slicing behavior](https://docs.dask.org/en/stable/array-slicing.html).

## Batch memory and Datasets

The reducer processes native batch chunks separately, reading one spatial window
at a time and keeping a compact gathered buffer for that batch. Without batch
chunk metadata, blocks target roughly 8 MiB of working data, with a minimum of
one slice. Native chunks can exceed this target. Memory also includes the eager
output, sparse operator, and backend decoding/scheduler allocations; this is not
a process-memory cap.

One plan serves every Dataset variable with the same spatial grid and layout:

```python
out = ghl.reduce_with_restricted_operator(ds, restricted)
```

Variables without both spatial dimensions pass through unchanged. Variables can
have different batch dimensions and batch chunks. For different spatial layouts,
create and apply a separate plan per layout. Use `lat_dim`, `lon_dim`, and
`geom_dim` for custom dimension names (`from_grid` accepts the spatial names too).

## Cache the plan

Both `LocalCache` and `RedisCache` support:

```python
restricted = cache.get_or_compute_restricted_operator(
    op, da.latitude.to_numpy(),
    da.variable.chunksizes["latitude"], da.variable.chunksizes["longitude"],
)
```

The digest includes the fused operator digest, **stored** coordinates, and
normalized spatial chunk sizes. Unlike the canonical fused operator, ascending
and descending read plans have different digests. Regular integer chunk sizes
and equivalent explicit tuples share a cache entry. Batch sizes/chunks and grid
values are not part of the key. Existing operator cache formats are unchanged.

## Limitations

NaN handling and per-cell weights still use
[`reduce_with_stencil`](masked.md). Renormalizing a fused matrix on source cells
is not generally equivalent to resampling first and masking target cells, so this
path does not silently introduce different missing-data semantics. It also does
not return a deferred Dask result or distribute reductions across workers.

For measured read counts, transferred payload, and memory, see
[performance](../performance.md#chunk-aware-reads).
