"""Public API: reduce(_with_stencil), resample_grid(_with_matrix), aggregate_bias(_with_tree)."""

from collections.abc import Callable, Hashable, Iterator
from itertools import pairwise, product
from typing import Literal

import geopandas as gpd
import numpy as np
import pandas as pd
import xarray as xr

from geohalo._conservative import GridBounds, Normalization, ResampleMethod, validate_method
from geohalo._sparse import projection_dtype
from geohalo.bias_tree import BiasTree
from geohalo.geometry import (
    _conservative_edges,
    _validate_period,
    ensure_ascending_lats,
    same_grid,
    target_coords_from_resolution,
)
from geohalo.reduce_operator import ReduceOperator
from geohalo.resampler import FactoredResampler, Resampler
from geohalo.restricted_operator import RestrictedOperator, _grid_chunks
from geohalo.stencil import PartialCellWeighting, Stencil

_RESTRICTED_BATCH_BYTES = 8 * 1024 * 1024


def _geom_coord(keys: pd.Index, geom_dim: str) -> pd.Index:
    """Name the keys Index after geom_dim so it aligns as a dim coordinate."""
    if isinstance(keys, pd.MultiIndex):
        return keys
    return keys.rename(geom_dim)


def _require_spatial_dims(obj: xr.DataArray | xr.Dataset, lat_dim: str, lon_dim: str) -> None:
    if lat_dim not in obj.dims or lon_dim not in obj.dims:
        raise ValueError(
            f"grid is missing required dims; got {tuple(obj.dims)}, need {lat_dim} and {lon_dim}",
        )


def _map_spatial_vars(
    ds: xr.Dataset,
    fn: Callable[[xr.DataArray], xr.DataArray],
    lat_dim: str,
    lon_dim: str,
) -> xr.Dataset:
    """Apply `fn` to each lat/lon-bearing data var; pass the rest through unchanged."""
    spatial = [n for n, v in ds.data_vars.items() if lat_dim in v.dims and lon_dim in v.dims]
    out = xr.Dataset({n: fn(ds[n]) for n in spatial})
    for n in ds.data_vars:
        if n not in spatial:
            out[n] = ds[n]
    out.attrs = dict(ds.attrs)  # the per-var fn carries var attrs; the dataset-level attrs need this
    return out


def _carryover_coords(src: xr.DataArray, keep_dims: list[str]) -> dict[Hashable, xr.DataArray]:
    """Coords of `src` whose dims all survive — scalar coords and batch-aligned aux coords.

    Subsumes the dimension coordinates of `keep_dims` and additionally preserves
    scalar coords (``dims == ()``) and auxiliary coords aligned to a retained dim
    (e.g. ``valid_time`` along ``time``). Coords touching the collapsed/replaced
    spatial dims are excluded — their dims are not in `keep_dims`.
    """
    keep = set(keep_dims)
    return {name: c for name, c in src.coords.items() if set(c.dims) <= keep}


def _apply_matrix_da(
    da: xr.DataArray,
    resampler: Resampler,
    lat_dim: str,
    lon_dim: str,
    skipna: bool,
) -> xr.DataArray:
    """Validate the source grid and apply the resampler over the spatial dims."""
    _require_spatial_dims(da, lat_dim, lon_dim)
    source_lat, descending = ensure_ascending_lats(da[lat_dim].to_numpy())
    source_lon = da[lon_dim].to_numpy()
    if not same_grid(source_lat, source_lon, resampler.source_lat, resampler.source_lon):
        raise ValueError(
            f"grid ({source_lat.size}, {source_lon.size}) does not match the resampler's source grid "
            f"({resampler.source_lat.size}, {resampler.source_lon.size})",
        )
    batch_dims = [d for d in da.dims if d not in (lat_dim, lon_dim)]
    arr = da.transpose(*batch_dims, lat_dim, lon_dim).to_numpy()
    out_flat = resampler.apply_grid(arr, descending=descending, skipna=skipna)
    out_lat, out_lon = resampler.target_lat, resampler.target_lon
    out = out_flat.reshape(*arr.shape[:-2], out_lat.size, out_lon.size)
    return xr.DataArray(
        out,
        dims=(*batch_dims, lat_dim, lon_dim),
        coords={**_carryover_coords(da, batch_dims), lat_dim: out_lat, lon_dim: out_lon},
        name=da.name,
        attrs=dict(da.attrs),
    )


def resample_grid_with_matrix[T: xr.DataArray | xr.Dataset](
    source: T,
    resampler: Resampler,
    *,
    lat_dim: str = "latitude",
    lon_dim: str = "longitude",
    skipna: bool = False,
) -> T:
    """Resample a matching source grid, accepting either latitude orientation.

    Raises ``ValueError`` if the source coordinates differ from the resampler's
    source grid. Output coordinates follow the resampler's target order.
    For conservative resamplers, ``skipna=True`` averages only valid covered
    area; by default contributing NaNs propagate. Unmapped cells return NaN.
    """
    if skipna and resampler.method != "conservative":
        raise ValueError("resampling skipna=True requires method='conservative'")
    if isinstance(source, xr.Dataset):
        return _map_spatial_vars(
            source,
            lambda da: resample_grid_with_matrix(da, resampler, lat_dim=lat_dim, lon_dim=lon_dim, skipna=skipna),
            lat_dim,
            lon_dim,
        )
    return _apply_matrix_da(source, resampler, lat_dim, lon_dim, skipna)


def resample_grid[T: xr.DataArray | xr.Dataset](
    source: T,
    target_resolution: float,
    *,
    lat_dim: str = "latitude",
    lon_dim: str = "longitude",
    iterations: int = 1,
    period: float | None = None,
    method: ResampleMethod = "meanpreserving",
    normalization: Normalization = "destination",
    source_bounds: GridBounds | None = None,
    skipna: bool = False,
) -> T:
    """Resample at a target spacing; ``period`` makes longitude a full cycle.

    ``period=360`` wraps interpolation and parent assignment at the seam. The
    generated longitude centres start at the source minimum and exclude its
    repeated endpoint. ``period=None`` retains clamped, min/max-based targets.
    ``method='conservative'`` uses spherical area overlaps, with optional
    ``normalization='covered'`` or apply-time ``skipna=True``. Target generation
    still uses source centres, not their outer cell bounds; use an explicitly
    built Resampler for matched source/target footprints. ``source_bounds`` is
    an optional pair of latitude/longitude edge arrays (conservative only).
    """
    validate_method(method, iterations, normalization, source_bounds, None)
    if skipna and method != "conservative":
        raise ValueError("resampling skipna=True requires method='conservative'")
    src_lat = source[lat_dim].to_numpy()
    src_lon = source[lon_dim].to_numpy()
    t_lat, t_lon = target_coords_from_resolution(src_lat, src_lon, target_resolution, period=period)
    target_bounds = None
    if method == "conservative":
        # Generated axes have a known spacing, even if coarsening leaves only
        # one centre. Cyclic longitude closes the seam at a cyclic midpoint.
        lat_edges = np.clip(np.r_[t_lat - target_resolution / 2, t_lat[-1] + target_resolution / 2], -90., 90.)
        lon_edges = (
            _conservative_edges(t_lon, None, latitude=False, period=period, cyclic=True)[0]
            if period is not None else np.r_[t_lon - target_resolution / 2, t_lon[-1] + target_resolution / 2]
        )
        target_bounds = lat_edges, lon_edges
    resampler = Resampler.compute(
        src_lat, src_lon, t_lat, t_lon, iterations=iterations, period=period,
        method=method, normalization=normalization, source_bounds=source_bounds, target_bounds=target_bounds,
    )
    return resample_grid_with_matrix(source, resampler, lat_dim=lat_dim, lon_dim=lon_dim, skipna=skipna)


def reduce_with_operator[T: xr.DataArray | xr.Dataset](
    grid: T,
    operator: ReduceOperator,
    *,
    how: Literal["mean", "sum"] = "mean",
    skipna: bool = False,
    preserve_dtype: bool = False,
    lat_dim: str = "latitude",
    lon_dim: str = "longitude",
    geom_dim: str = "geom",
) -> T:
    """Reduce ``grid`` to per-polygon values using the fused source-grid operator.

    By default, contributing NaNs propagate. ``skipna=True`` omits missing
    source cells and renormalizes means over their surviving signed weights;
    nonpositive denominators return NaN. Sums omit missing contributions,
    returning zero when all are missing. With fused resampling, source masking
    differs from resample-then-mask and negative weights can cause overshoots.
    For target-cell masking or per-cell weights, use ``reduce_with_stencil``.
    Large grids gather contributing cells per batch slice, retaining the
    operator's precision without copying or upcasting the entire batch.
    ``preserve_dtype=True`` retains each floating input variable's dtype in the
    result, without lowering matrix precision. Integer means remain floating.
    """
    if how not in ("mean", "sum"):
        raise ValueError(f"how must be 'mean' or 'sum', got {how!r}")
    if isinstance(grid, xr.Dataset):
        return _map_spatial_vars(
            grid,
            lambda da: reduce_with_operator(
                da, operator, how=how, skipna=skipna, preserve_dtype=preserve_dtype,
                lat_dim=lat_dim, lon_dim=lon_dim, geom_dim=geom_dim,
            ),
            lat_dim,
            lon_dim,
        )

    _require_spatial_dims(grid, lat_dim, lon_dim)
    da_lat, descending = ensure_ascending_lats(grid[lat_dim].to_numpy())
    da_lon = grid[lon_dim].to_numpy()
    if not same_grid(da_lat, da_lon, operator.source_lat, operator.source_lon):
        raise ValueError(
            f"grid ({da_lat.size}, {da_lon.size}) does not match the operator's source grid "
            f"({operator.source_lat.size}, {operator.source_lon.size})",
        )
    batch_dims = [d for d in grid.dims if d not in (lat_dim, lon_dim)]
    arr = grid.transpose(*batch_dims, lat_dim, lon_dim).to_numpy()
    proj = operator.apply_grid(arr, descending=descending, how=how, skipna=skipna, preserve_dtype=preserve_dtype)
    return xr.DataArray(
        proj,
        dims=(*batch_dims, geom_dim),
        coords={**_carryover_coords(grid, batch_dims), geom_dim: _geom_coord(operator.keys, geom_dim)},
        name=grid.name,
        attrs=dict(grid.attrs),
    )


def _restricted_batch_slices(
    grid: xr.DataArray, batch_dims: list[str], operator: RestrictedOperator,
) -> Iterator[tuple[slice, ...]]:
    """Use native batch chunks, or bounded blocks for unchunked batch dimensions."""
    chunks = {dim: _grid_chunks(grid, dim) for dim in batch_dims}
    largest_window = max(
        ((rows.stop - rows.start) * (cols.stop - cols.start) for rows, cols in operator.windows),
        default=0,
    )
    bytes_per_slice = (
        (largest_window + operator.matrix.shape[1]) * grid.dtype.itemsize
        + len(operator.keys) * np.result_type(grid.dtype, operator.matrix.dtype).itemsize
    )
    capacity = max(1, _RESTRICTED_BATCH_BYTES // max(1, bytes_per_slice))
    for sizes in chunks.values():
        if sizes:
            capacity = max(1, capacity // max(sizes))
    for dim in reversed(batch_dims):
        if chunks[dim] is None:
            size = grid.sizes[dim]
            step = min(max(1, size), capacity)
            count, tail = divmod(size, step)
            chunks[dim] = (step,) * count + ((tail,) if tail else ())
            capacity = max(1, capacity // step)
    slices = []
    for dim in batch_dims:
        edges = np.cumsum((0, *chunks[dim]))
        slices.append([slice(int(a), int(b)) for a, b in pairwise(edges)])
    return product(*slices)


def reduce_with_restricted_operator[T: xr.DataArray | xr.Dataset](
    grid: T,
    operator: RestrictedOperator,
    *,
    how: Literal["mean", "sum"] = "mean",
    skipna: bool = False,
    preserve_dtype: bool = False,
    lat_dim: str = "latitude",
    lon_dim: str = "longitude",
    geom_dim: str = "geom",
) -> T:
    """Reduce data, reading only the plan's contributing spatial chunks.

    Returns an eager result, preserving batch coordinates, names, and attrs.
    Each batch chunk is processed separately; only one spatial window and its
    gathered cells are held at a time. Without batch chunk metadata, blocks target
    8 MiB of working data (at least one slice). Native chunks may exceed this.

    Coordinates must match the plan's stored latitude order and longitude grid.
    Known spatial chunk layouts must also match; rebuild the plan after changing
    either. Dataset variables without both spatial dimensions pass through.
    NaNs propagate by default. ``skipna=True`` omits missing source cells,
    renormalizing means over surviving signed weights (NaN if nonpositive).
    Sums omit missing contributions, returning zero if all are missing. With
    fused resampling this is not resample-then-mask. Per-cell weights and
    target-cell masking still require ``reduce_with_stencil``.
    ``preserve_dtype=True`` retains each floating input variable's result dtype;
    integer inputs keep normal promotion, including floating-point means.
    """
    if how not in ("mean", "sum"):
        raise ValueError(f"how must be 'mean' or 'sum', got {how!r}")
    if isinstance(grid, xr.Dataset):
        return _map_spatial_vars(
            grid,
            lambda da: reduce_with_restricted_operator(
                da, operator, how=how, skipna=skipna, preserve_dtype=preserve_dtype,
                lat_dim=lat_dim, lon_dim=lon_dim, geom_dim=geom_dim,
            ),
            lat_dim, lon_dim,
        )
    _require_spatial_dims(grid, lat_dim, lon_dim)
    if not same_grid(grid[lat_dim].to_numpy(), grid[lon_dim].to_numpy(), operator.source_lat, operator.source_lon):
        raise ValueError("grid does not match the restricted operator's stored source grid (including latitude order)")
    for dim, expected in ((lat_dim, operator.lat_chunks), (lon_dim, operator.lon_chunks)):
        actual = _grid_chunks(grid, dim)
        if actual is not None and actual != expected:
            raise ValueError(f"{dim} chunks do not match the restricted operator; rebuild the plan for this layout")

    batch_dims = [dim for dim in grid.dims if dim not in (lat_dim, lon_dim)]
    batch_shape = tuple(grid.sizes[dim] for dim in batch_dims)
    row_sums = operator.row_sums if how == "mean" else None
    dtype = projection_dtype(grid.dtype, operator.matrix.dtype, row_sums, preserve_dtype=preserve_dtype)
    out = np.zeros((*batch_shape, len(operator.keys)), dtype=dtype)
    if operator.matrix.shape[1]:
        def read_windows(selection: dict[str, slice]) -> Iterator[np.ndarray]:
            for rows, cols in operator.windows:
                # Slice BEFORE accessing values: backend arrays and Dask cull
                # unrelated chunks. Do not retain the previous window.
                window = grid.isel({**selection, lat_dim: rows, lon_dim: cols})
                yield window.transpose(*batch_dims, lat_dim, lon_dim).to_numpy()
                del window

        for index in _restricted_batch_slices(grid, batch_dims, operator):
            selection = dict(zip(batch_dims, index, strict=True))
            gathered = operator.gather(read_windows(selection))
            out[index] = operator.apply(gathered, how=how, skipna=skipna, preserve_dtype=preserve_dtype)
            del gathered
    elif out.size:
        # No source cells are read, but preserve mean/empty-denominator semantics.
        out[...] = operator.apply(np.empty(0, dtype=grid.dtype), how=how, skipna=skipna, preserve_dtype=preserve_dtype)
    return xr.DataArray(
        out, dims=(*batch_dims, geom_dim),
        coords={**_carryover_coords(grid, batch_dims), geom_dim: _geom_coord(operator.keys, geom_dim)},
        name=grid.name, attrs=dict(grid.attrs),
    )


def reduce_with_stencil[T: xr.DataArray | xr.Dataset](
    grid: T,
    stencil: Stencil,
    *,
    resample_iterations: int = 1,
    lat_dim: str = "latitude",
    lon_dim: str = "longitude",
    geom_dim: str = "geom",
    weight_key: str | None = None,
    how: Literal["mean", "sum"] = "mean",
    preserve_dtype: bool = False,
    period: float | None = None,
) -> T:
    """Reduce in stencil row order, automatically handling NaNs and cell weights.

    ``preserve_dtype=True`` retains each floating input variable's result dtype;
    integer means remain floating. Clean fusion uses the stencil's coefficient
    dtype. Masked/weighted normalization retains its existing arithmetic precision.
    ``period=360`` wraps resampling in longitude; stencil coordinates and
    polygon geometries are not wrapped or changed.
    """
    if how not in ("mean", "sum"):
        raise ValueError(f"how must be 'mean' or 'sum', got {how!r}")
    _require_spatial_dims(grid, lat_dim, lon_dim)
    src_lat, _ = ensure_ascending_lats(grid[lat_dim].to_numpy())
    src_lon = np.asarray(grid[lon_dim].to_numpy(), dtype=np.float64)
    period = _validate_period(src_lon, period)
    grid = grid.compute()  # materialise once: avoids double-decoding lazy data

    # Clean path: build the fused operator once and delegate.
    if weight_key is None and not _any_spatial_nan(grid, lat_dim, lon_dim):
        operator = ReduceOperator.compute(
            stencil, src_lat, src_lon, iterations=resample_iterations, dtype=stencil.occupancy_matrix.dtype,
            period=period,
        )
        return reduce_with_operator(
            grid, operator, how=how, preserve_dtype=preserve_dtype, lat_dim=lat_dim, lon_dim=lon_dim, geom_dim=geom_dim,
        )

    # Masked path: build the resampler once, project per variable with renormalisation.
    if same_grid(src_lat, src_lon, stencil.lats, stencil.lons):
        resampler = None
    else:
        resampler = FactoredResampler.compute(
            src_lat, src_lon, stencil.lats, stencil.lons, iterations=resample_iterations, period=period,
        )
    if isinstance(grid, xr.Dataset):
        return _map_spatial_vars(
            grid,
            lambda da: _reduce_masked_da(
                da, stencil, resampler, weight_key, how, lat_dim, lon_dim, geom_dim, grid, preserve_dtype,
            ),
            lat_dim,
            lon_dim,
        )
    return _reduce_masked_da(
        grid, stencil, resampler, weight_key, how, lat_dim, lon_dim, geom_dim, grid, preserve_dtype,
    )


def _any_spatial_nan(grid: xr.DataArray | xr.Dataset, lat_dim: str, lon_dim: str) -> bool:
    if isinstance(grid, xr.Dataset):
        return any(
            bool(np.isnan(v.to_numpy()).any())
            for v in grid.data_vars.values()
            if lat_dim in v.dims and lon_dim in v.dims
        )
    return bool(np.isnan(grid.to_numpy()).any())


def _project_masked(
    flat: np.ndarray,
    weight_flat: np.ndarray | None,
    stencil: Stencil,
    resampler: FactoredResampler | None,
    how: Literal["mean", "sum"],
) -> np.ndarray:
    """Resample (if needed), then aggregate with per-cell NaN/weight renormalisation."""
    occ = stencil.occupancy_matrix
    resampled = flat if resampler is None else resampler.apply_flat(flat)
    valid = ~np.isnan(resampled)
    if weight_flat is None:
        # Unweighted: the per-cell weight is 1, so skip materialising a full ones array
        # and the two no-op multiplies — each a (batch, n_cells) allocation that on the
        # 50x4 ensemble runs to gigabytes (see benchmarks: masked dRSS vs clean).
        numer = (occ @ np.where(valid, resampled, 0.0).T).T
        if how == "sum":
            return numer
        denom = (occ @ valid.astype(np.float64).T).T
    else:
        valid = valid & ~np.isnan(weight_flat)
        numer = (occ @ (weight_flat * np.where(valid, resampled, 0.0)).T).T
        if how == "sum":
            return numer
        denom = (occ @ (weight_flat * valid.astype(np.float64)).T).T
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(denom > 0, numer / denom, np.nan)


def _reduce_masked_da(
    da: xr.DataArray,
    stencil: Stencil,
    resampler: FactoredResampler | None,
    weight_key: str | None,
    how: Literal["mean", "sum"],
    lat_dim: str,
    lon_dim: str,
    geom_dim: str,
    weight_source: xr.DataArray | xr.Dataset,
    preserve_dtype: bool,
) -> xr.DataArray:
    _require_spatial_dims(da, lat_dim, lon_dim)
    lat = da[lat_dim].to_numpy()
    if lat.size > 1 and lat[0] > lat[-1]:
        da = da.sortby(lat_dim)
    batch_dims = [d for d in da.dims if d not in (lat_dim, lon_dim)]
    arr = da.transpose(*batch_dims, lat_dim, lon_dim).to_numpy()
    flat = arr.reshape(-1, arr.shape[-2] * arr.shape[-1])
    weight_flat = _resolve_weight_flat(da, weight_source, weight_key, lat_dim, lon_dim, batch_dims, resampler)
    if preserve_dtype and arr.dtype.kind == "f":
        # Normalize into the requested dtype one slice at a time, without a
        # full-batch float64 result or resampled intermediate to cast afterward.
        out_flat = np.empty((flat.shape[0], len(stencil.keys)), dtype=arr.dtype)
        for i in range(flat.shape[0]):
            weights = None if weight_flat is None else weight_flat[i:i + 1]
            out_flat[i] = _project_masked(flat[i:i + 1], weights, stencil, resampler, how)[0]
    else:
        out_flat = _project_masked(flat, weight_flat, stencil, resampler, how)
    out = out_flat.reshape(*arr.shape[:-2], len(stencil.keys))
    return xr.DataArray(
        out,
        dims=(*batch_dims, geom_dim),
        coords={**_carryover_coords(da, batch_dims), geom_dim: _geom_coord(stencil.keys, geom_dim)},
        name=da.name,
        attrs=dict(da.attrs),
    )


def _resolve_weight_flat(
    da: xr.DataArray,
    source: xr.DataArray | xr.Dataset,
    weight_key: str | None,
    lat_dim: str,
    lon_dim: str,
    batch_dims: list[str],
    resampler: FactoredResampler | None,
) -> np.ndarray | None:
    """Resolve weight_key on `source` (lookup namespace), broadcast to `da`."""
    if weight_key is None:
        return None
    try:
        w = source[weight_key]
    except (KeyError, AttributeError) as exc:
        raise ValueError(f"weight_key {weight_key!r} not found in the grid being reduced") from exc
    full = (*batch_dims, lat_dim, lon_dim)
    w = w.broadcast_like(da).transpose(*full)
    w_arr = w.to_numpy()
    n_lat_src, n_lon_src = w_arr.shape[-2:]
    w_flat = w_arr.reshape(-1, n_lat_src * n_lon_src)
    if resampler is not None:
        w_flat = resampler.apply_flat(w_flat)
    return w_flat


def reduce[T: xr.DataArray | xr.Dataset](  # noqa: PLR0913 - preserve the public keyword API
    grid: T,
    geoms: gpd.GeoSeries,
    *,
    target_resolution: float | None = None,
    resample_iterations: int = 1,
    spherical_correction: bool = True,
    partial_cell_weighting: PartialCellWeighting = "approximate",
    lat_dim: str = "latitude",
    lon_dim: str = "longitude",
    geom_dim: str = "geom",
    weight_key: str | None = None,
    how: Literal["mean", "sum"] = "mean",
    preserve_dtype: bool = False,
    period: float | None = None,
) -> T:
    """Build a float64 stencil and reduce polygons in caller order.

    ``preserve_dtype=True`` retains each floating input variable's result dtype;
    integer means remain floating. For float32 coefficients, prebuild a stencil
    or operator with ``dtype=np.float32`` and use its corresponding reducer.
    ``period=360`` enables cyclic longitude resampling and generates a full
    cycle when ``target_resolution`` is set. Geometries themselves do not wrap.
    ``partial_cell_weighting="exact"`` opts into spherical polygon-cell
    intersection areas; see :meth:`Stencil.compute`. The default is approximate.
    """
    src_lat = grid[lat_dim].to_numpy()
    src_lon = grid[lon_dim].to_numpy()
    if target_resolution is None:
        stencil = Stencil.compute(
            src_lat, src_lon, geoms, spherical_correction=spherical_correction,
            partial_cell_weighting=partial_cell_weighting,
        )
    else:
        tlat, tlon = target_coords_from_resolution(src_lat, src_lon, target_resolution, period=period)
        stencil = Stencil.compute(
            tlat, tlon, geoms, spherical_correction=spherical_correction,
            partial_cell_weighting=partial_cell_weighting,
        )
    return reduce_with_stencil(
        grid, stencil, resample_iterations=resample_iterations,
        lat_dim=lat_dim, lon_dim=lon_dim, geom_dim=geom_dim,
        weight_key=weight_key, how=how, preserve_dtype=preserve_dtype,
        period=period,
    )


def aggregate_bias_with_tree[T: xr.DataArray | xr.Dataset](
    reduced: T,
    tree: BiasTree,
    *,
    geom_dim: str = "geom",
) -> T:
    if isinstance(reduced, xr.Dataset):
        return reduced.map(
            lambda da: aggregate_bias_with_tree(da, tree, geom_dim=geom_dim)
            if geom_dim in da.dims else da,
        )
    if geom_dim not in reduced.dims:
        raise ValueError(f"reduced has no dim named {geom_dim!r}; dims={reduced.dims}")
    available = set(reduced[geom_dim].to_index())
    missing = [k for k in tree.leaf_keys if k not in available]
    if missing:
        raise ValueError(f"missing leaf(s) in input: {missing[:5]!r}")

    leaves = reduced.sel({geom_dim: list(tree.leaf_keys)})
    batch_dims = [d for d in leaves.dims if d != geom_dim]
    arr = leaves.transpose(*batch_dims, geom_dim).to_numpy()
    flat = arr.reshape(-1, len(tree.leaf_keys))

    valid = ~np.isnan(flat)
    if valid.all():
        # flat @ R.T keeps the dense operand C-contiguous (see _apply_matrix_da).
        out_flat = np.asarray(flat @ tree.rollup_matrix.T)
    elif tree.how == "mean":
        filled = np.where(valid, flat, 0.0)
        numer = (tree.rollup_matrix @ filled.T).T
        denom = (tree.rollup_matrix @ valid.astype(np.float64).T).T
        with np.errstate(invalid="ignore", divide="ignore"):
            out_flat = np.where(denom > 0, numer / denom, np.nan)
    else:  # sum
        filled = np.where(valid, flat, 0.0)
        out_flat = (tree.rollup_matrix @ filled.T).T

    out = out_flat.reshape(*arr.shape[:-1], len(tree.keys))
    return xr.DataArray(
        out,
        dims=(*batch_dims, geom_dim),
        coords={**_carryover_coords(leaves, batch_dims), geom_dim: _geom_coord(tree.keys, geom_dim)},
        name=leaves.name,
        attrs=dict(leaves.attrs),
    )


def aggregate_bias[T: xr.DataArray | xr.Dataset](
    reduced: T,
    edges: pd.DataFrame,
    *,
    geom_dim: str = "geom",
    parent_col: str = "parent",
    weight_col: str | None = None,
    how: Literal["mean", "sum"] = "mean",
) -> T:
    tree = BiasTree.compute(edges, parent_col=parent_col, weight_col=weight_col, how=how)
    return aggregate_bias_with_tree(reduced, tree, geom_dim=geom_dim)
