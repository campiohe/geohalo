"""Public API: reduce(_with_stencil), resample_grid(_with_matrix), aggregate_bias(_with_tree)."""

from collections.abc import Callable
from typing import Literal

import geopandas as gpd
import numpy as np
import pandas as pd
import scipy.sparse as sp
import xarray as xr

from geohalo.bias_tree import BiasTree
from geohalo.geometry import ensure_ascending_lats, same_grid, target_coords_from_resolution
from geohalo.reduce_operator import ReduceOperator
from geohalo.resampler import FactoredResampler, Resampler
from geohalo.stencil import Stencil


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
    return out


def _apply_matrix_da(
    da: xr.DataArray,
    matrix: sp.csr_matrix,
    lat_dim: str,
    lon_dim: str,
    out_lat: np.ndarray,
    out_lon: np.ndarray,
) -> xr.DataArray:
    """Apply a (n_target, n_source) sparse matrix over the spatial dims of da."""
    lat_values = da[lat_dim].to_numpy()
    if lat_values.size > 1 and lat_values[0] > lat_values[-1]:
        da = da.sortby(lat_dim)
    batch_dims = [d for d in da.dims if d not in (lat_dim, lon_dim)]
    arr = da.transpose(*batch_dims, lat_dim, lon_dim).to_numpy()
    n_lat_src, n_lon_src = arr.shape[-2:]
    flat = arr.reshape(-1, n_lat_src * n_lon_src)
    # flat @ matrix.T keeps the dense operand C-contiguous; (matrix @ flat.T).T would hand
    # scipy an F-contiguous operand it may copy. Benchmarks show the two are close at the
    # shapes we hit, so this is a tidy-default rather than a hot-spot. np.asarray guards
    # against scipy returning an np.matrix.
    out_flat = np.asarray(flat @ matrix.T)
    out = out_flat.reshape(*arr.shape[:-2], out_lat.size, out_lon.size)
    return xr.DataArray(
        out,
        dims=(*batch_dims, lat_dim, lon_dim),
        coords={**{d: da[d] for d in batch_dims}, lat_dim: out_lat, lon_dim: out_lon},
    )


def resample_grid_with_matrix[T: xr.DataArray | xr.Dataset](
    source: T,
    resampler: Resampler,
    *,
    lat_dim: str = "latitude",
    lon_dim: str = "longitude",
) -> T:
    if isinstance(source, xr.Dataset):
        return _map_spatial_vars(
            source,
            lambda da: resample_grid_with_matrix(da, resampler, lat_dim=lat_dim, lon_dim=lon_dim),
            lat_dim,
            lon_dim,
        )
    return _apply_matrix_da(
        source, resampler.transform_matrix, lat_dim, lon_dim,
        resampler.target_lat, resampler.target_lon,
    )


def resample_grid[T: xr.DataArray | xr.Dataset](
    source: T,
    target_resolution: float,
    *,
    lat_dim: str = "latitude",
    lon_dim: str = "longitude",
    iterations: int = 1,
) -> T:
    src_lat = source[lat_dim].to_numpy()
    src_lon = source[lon_dim].to_numpy()
    t_lat, t_lon = target_coords_from_resolution(src_lat, src_lon, target_resolution)
    resampler = Resampler.compute(src_lat, src_lon, t_lat, t_lon, iterations=iterations)
    return resample_grid_with_matrix(source, resampler, lat_dim=lat_dim, lon_dim=lon_dim)


def reduce_with_operator[T: xr.DataArray | xr.Dataset](
    grid: T,
    operator: ReduceOperator,
    *,
    how: Literal["mean", "sum"] = "mean",
    lat_dim: str = "latitude",
    lon_dim: str = "longitude",
    geom_dim: str = "geom",
) -> T:
    """Reduce ``grid`` to per-polygon values with one fused matmul on the source grid.

    Assumes clean (non-NaN, unweighted) data — the fused operator cannot
    renormalise per cell. For NaN or per-cell weighting, use ``reduce_with_stencil``.
    """
    if how not in ("mean", "sum"):
        raise ValueError(f"how must be 'mean' or 'sum', got {how!r}")
    if isinstance(grid, xr.Dataset):
        return _map_spatial_vars(
            grid,
            lambda da: reduce_with_operator(da, operator, how=how, lat_dim=lat_dim, lon_dim=lon_dim, geom_dim=geom_dim),
            lat_dim,
            lon_dim,
        )

    _require_spatial_dims(grid, lat_dim, lon_dim)
    lat = grid[lat_dim].to_numpy()
    if lat.size > 1 and lat[0] > lat[-1]:
        grid = grid.sortby(lat_dim)
    da_lat = grid[lat_dim].to_numpy()
    da_lon = grid[lon_dim].to_numpy()
    if not same_grid(da_lat, da_lon, operator.source_lat, operator.source_lon):
        raise ValueError(
            f"grid ({da_lat.size}, {da_lon.size}) does not match the operator's source grid "
            f"({operator.source_lat.size}, {operator.source_lon.size})",
        )
    batch_dims = [d for d in grid.dims if d not in (lat_dim, lon_dim)]
    arr = grid.transpose(*batch_dims, lat_dim, lon_dim).to_numpy()
    flat = arr.reshape(-1, arr.shape[-2] * arr.shape[-1])
    # flat @ M.T keeps the dense operand C-contiguous (see _apply_matrix_da for the rationale).
    proj = np.asarray(flat @ operator.matrix.T)
    if how == "mean":
        proj = proj / operator.row_sums[None, :]
    out = proj.reshape(*arr.shape[:-2], len(operator.keys))
    return xr.DataArray(
        out,
        dims=(*batch_dims, geom_dim),
        coords={**{d: grid[d] for d in batch_dims}, geom_dim: _geom_coord(operator.keys, geom_dim)},
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
) -> T:
    if how not in ("mean", "sum"):
        raise ValueError(f"how must be 'mean' or 'sum', got {how!r}")
    _require_spatial_dims(grid, lat_dim, lon_dim)
    grid = grid.compute()  # materialise once: avoids double-decoding lazy data
    src_lat, _ = ensure_ascending_lats(grid[lat_dim].to_numpy())
    src_lon = np.asarray(grid[lon_dim].to_numpy(), dtype=np.float64)

    # Clean path: build the fused operator once and delegate.
    if weight_key is None and not _any_spatial_nan(grid, lat_dim, lon_dim):
        operator = ReduceOperator.compute(stencil, src_lat, src_lon, iterations=resample_iterations)
        return reduce_with_operator(grid, operator, how=how, lat_dim=lat_dim, lon_dim=lon_dim, geom_dim=geom_dim)

    # Masked path: build the resampler once, project per variable with renormalisation.
    if same_grid(src_lat, src_lon, stencil.lats, stencil.lons):
        resampler = None
    else:
        resampler = FactoredResampler.compute(
            src_lat, src_lon, stencil.lats, stencil.lons, iterations=resample_iterations,
        )
    if isinstance(grid, xr.Dataset):
        return _map_spatial_vars(
            grid,
            lambda da: _reduce_masked_da(da, stencil, resampler, weight_key, how, lat_dim, lon_dim, geom_dim, grid),
            lat_dim,
            lon_dim,
        )
    return _reduce_masked_da(grid, stencil, resampler, weight_key, how, lat_dim, lon_dim, geom_dim, grid)


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
) -> xr.DataArray:
    _require_spatial_dims(da, lat_dim, lon_dim)
    lat = da[lat_dim].to_numpy()
    if lat.size > 1 and lat[0] > lat[-1]:
        da = da.sortby(lat_dim)
    batch_dims = [d for d in da.dims if d not in (lat_dim, lon_dim)]
    arr = da.transpose(*batch_dims, lat_dim, lon_dim).to_numpy()
    flat = arr.reshape(-1, arr.shape[-2] * arr.shape[-1])
    weight_flat = _resolve_weight_flat(da, weight_source, weight_key, lat_dim, lon_dim, batch_dims, resampler)
    out_flat = _project_masked(flat, weight_flat, stencil, resampler, how)
    out = out_flat.reshape(*arr.shape[:-2], len(stencil.keys))
    return xr.DataArray(
        out,
        dims=(*batch_dims, geom_dim),
        coords={**{d: da[d] for d in batch_dims}, geom_dim: _geom_coord(stencil.keys, geom_dim)},
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


def reduce[T: xr.DataArray | xr.Dataset](
    grid: T,
    geoms: gpd.GeoSeries,
    *,
    target_resolution: float | None = None,
    resample_iterations: int = 1,
    spherical_correction: bool = True,
    lat_dim: str = "latitude",
    lon_dim: str = "longitude",
    geom_dim: str = "geom",
    weight_key: str | None = None,
    how: Literal["mean", "sum"] = "mean",
) -> T:
    src_lat = grid[lat_dim].to_numpy()
    src_lon = grid[lon_dim].to_numpy()
    if target_resolution is None:
        stencil = Stencil.compute(src_lat, src_lon, geoms, spherical_correction=spherical_correction)
    else:
        tlat, tlon = target_coords_from_resolution(src_lat, src_lon, target_resolution)
        stencil = Stencil.compute(tlat, tlon, geoms, spherical_correction=spherical_correction)
    return reduce_with_stencil(
        grid, stencil, resample_iterations=resample_iterations,
        lat_dim=lat_dim, lon_dim=lon_dim, geom_dim=geom_dim,
        weight_key=weight_key, how=how,
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
        coords={**{d: leaves[d] for d in batch_dims}, geom_dim: _geom_coord(tree.keys, geom_dim)},
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
