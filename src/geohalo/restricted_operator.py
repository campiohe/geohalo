"""A fused operator restricted to disjoint windows of contributing source chunks."""

import hashlib
import numbers
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd
import scipy.sparse as sp
import xarray as xr

from geohalo._sparse import project_values, projection_dtype
from geohalo.geometry import ensure_ascending_lats, grid_digest, same_grid
from geohalo.reduce_operator import ReduceOperator

type ChunkSizes = int | tuple[int, ...]


def _normalize_chunks(chunks: ChunkSizes, size: int, name: str) -> tuple[int, ...]:
    if isinstance(chunks, numbers.Integral) and not isinstance(chunks, (bool, np.bool_)):
        if chunks <= 0:
            raise ValueError(f"{name} chunks must be positive integers")
        count, tail = divmod(size, int(chunks))
        return (int(chunks),) * count + ((tail,) if tail else ())
    try:
        sizes = tuple(chunks)
    except TypeError as exc:
        raise ValueError(f"{name} chunks must be an integer or a tuple of integers") from exc
    if any(not isinstance(n, numbers.Integral) or isinstance(n, (bool, np.bool_)) or n <= 0 for n in sizes):
        raise ValueError(f"{name} chunks must be positive integers")
    if sum(sizes) != size:
        raise ValueError(f"{name} chunks must sum to {size}, got {sizes}")
    return tuple(int(n) for n in sizes)


def _grid_chunks(grid: xr.DataArray, dim: str) -> tuple[int, ...] | None:
    """Inspect chunk metadata without accessing the array's values/data property."""
    if grid.sizes[dim] == 0:
        return ()
    # DataArray.chunksizes also inspects auxiliary coordinates, whose independent
    # chunks need not agree with the data variable's read layout.
    chunks = grid.variable.chunksizes.get(dim)
    if chunks is None:
        chunks = grid.encoding.get("preferred_chunks", {}).get(dim)
    if chunks is None:
        encoded = grid.encoding.get("chunks")
        if encoded is not None:
            if len(encoded) != grid.ndim:
                raise ValueError("encoding['chunks'] must have one entry per dimension")
            chunks = encoded[grid.dims.index(dim)]
    return None if chunks is None else _normalize_chunks(chunks, grid.sizes[dim], dim)


def _restriction_inputs(
    operator: ReduceOperator, source_lat: np.ndarray, lat_chunks: ChunkSizes, lon_chunks: ChunkSizes,
) -> tuple[np.ndarray, tuple[int, ...], tuple[int, ...]]:
    lat = np.asarray(source_lat, dtype=np.float64)
    if lat.ndim != 1:
        raise ValueError("source_lat must be one-dimensional")
    canonical, _ = ensure_ascending_lats(lat)
    if not same_grid(canonical, operator.source_lon, operator.source_lat, operator.source_lon):
        raise ValueError("source_lat does not match the operator's source grid")
    return (
        lat,
        _normalize_chunks(lat_chunks, lat.size, "latitude"),
        _normalize_chunks(lon_chunks, operator.source_lon.size, "longitude"),
    )


def restricted_operator_digest(
    operator: ReduceOperator, source_lat: np.ndarray, lat_chunks: ChunkSizes, lon_chunks: ChunkSizes,
) -> bytes:
    """Hash the fused operator, stored latitude order, and normalized chunk layout."""
    lat, lat_chunks, lon_chunks = _restriction_inputs(operator, source_lat, lat_chunks, lon_chunks)
    digest = hashlib.sha256()
    digest.update(operator.digest)
    digest.update(grid_digest(lat, operator.source_lon))
    digest.update(repr((lat_chunks, lon_chunks)).encode())
    return digest.digest()


def _chunk_rectangles(touched: np.ndarray) -> tuple[list[list[int]], np.ndarray]:
    """Merge contiguous row runs with identical spans on adjacent chunk rows.

    Unlike connected-component bounding boxes, these rectangles have no holes or
    overlaps. No dense array of the entire storage chunk lattice is needed.
    ``touched`` contains unique (row, column) pairs in lexicographic order.
    """
    window_of = np.empty(len(touched), dtype=np.intp)
    bounds = []
    active = {}
    breaks = np.flatnonzero(
        (np.diff(touched[:, 0]) != 0) | (np.diff(touched[:, 1]) != 1),
    ) + 1
    for start, stop in zip(np.r_[0, breaks], np.r_[breaks, len(touched)], strict=True):
        if start == stop:
            continue
        row, col = map(int, touched[start])
        end = int(touched[stop - 1, 1]) + 1
        span = (col, end)
        index = active.get(span)
        if index is not None and bounds[index][1] == row:
            bounds[index][1] = row + 1
        else:
            index = len(bounds)
            bounds.append([row, row + 1, col, end])
            active[span] = index
        window_of[start:stop] = index
    return bounds, window_of


@dataclass(frozen=True)
class RestrictedOperator:
    """Reusable chunk-aligned read plan for a :class:`ReduceOperator`.

    Latitude coordinates and windows are in *stored* order, not canonical order.
    A plan is specific to that orientation and the spatial chunk layout. The
    compact matrix keeps the original coefficient accumulation order.
    """

    windows: tuple[tuple[slice, slice], ...]
    gathers: tuple[np.ndarray, ...]
    matrix: sp.csr_matrix
    row_sums: np.ndarray
    keys: pd.Index
    source_lat: np.ndarray
    source_lon: np.ndarray
    lat_chunks: tuple[int, ...]
    lon_chunks: tuple[int, ...]
    digest: bytes

    def gather(self, arrays: Iterable[np.ndarray]) -> np.ndarray:
        """Gather already-read windows into ``(..., contributing_cells)``.

        Supply exactly one NumPy array per entry in ``windows``, in that order.
        Each array must have trailing ``(rows, columns)`` dimensions matching its
        window and the same leading batch shape. Coordinates cannot be checked:
        the caller is responsible for reading in the plan's stored grid order.

        Iterators are consumed one window at a time, without retaining earlier
        arrays or flattening whole noncontiguous windows. A common input dtype
        is preserved; mixed dtypes are promoted with ``numpy.result_type``.
        No I/O, xarray objects, or Dask scheduling is involved.

        With no windows, ``gather([])`` returns an empty float64 vector. For a
        batched empty plan, pass ``np.empty((*batch_shape, 0), dtype=...)``
        directly to ``apply`` instead.
        """
        iterator = iter(arrays)
        gathered = None
        offset = 0
        for i, ((rows, cols), positions) in enumerate(zip(self.windows, self.gathers, strict=True)):
            try:
                values = np.asarray(next(iterator))
            except StopIteration as exc:
                raise ValueError(f"expected {len(self.windows)} window arrays, got {i}") from exc
            spatial_shape = (rows.stop - rows.start, cols.stop - cols.start)
            if values.ndim < 2 or values.shape[-2:] != spatial_shape:
                raise ValueError(f"window {i} expected trailing shape {spatial_shape}, got {values.shape}")
            if gathered is None:
                gathered = np.empty((*values.shape[:-2], self.matrix.shape[1]), dtype=values.dtype)
            elif values.shape[:-2] != gathered.shape[:-1]:
                raise ValueError(
                    f"window {i} expected batch shape {gathered.shape[:-1]}, got {values.shape[:-2]}",
                )
            dtype = np.result_type(gathered.dtype, values.dtype)
            if dtype != gathered.dtype:
                promoted = np.empty(gathered.shape, dtype=dtype)
                promoted[..., :offset] = gathered[..., :offset]
                gathered = promoted
            r, c = np.divmod(positions, spatial_shape[1])
            gathered[..., offset:offset + positions.size] = values[..., r, c]
            offset += positions.size
            del values
        sentinel = object()
        if next(iterator, sentinel) is not sentinel:
            raise ValueError(f"expected {len(self.windows)} window arrays, got more")
        return np.empty(0, dtype=np.float64) if gathered is None else gathered

    def apply(
        self, gathered: np.ndarray, *, how: Literal["mean", "sum"] = "mean", skipna: bool = False,
    ) -> np.ndarray:
        """Reduce ``(..., contributing_cells)`` to ``(..., zones)`` in ``keys`` order.

        ``gathered`` must be in the column order produced by ``gather``. Sum
        returns the sparse projection; mean additionally divides by ``row_sums``.
        Matrix precision and coefficient accumulation order are preserved, with
        per-batch-slice products to avoid upcasting or copying an entire batch.

        By default, contributing NaNs propagate. ``skipna=True`` omits missing
        source cells: sums return the remaining weighted contributions (zero if
        all are missing), and means divide by the surviving signed weight (NaN
        if nonpositive). With fused resampling this differs from masking after
        resampling; negative coefficients can still produce overshoots.
        This method does not read data or require xarray objects or Dask.
        """
        if how not in ("mean", "sum"):
            raise ValueError(f"how must be 'mean' or 'sum', got {how!r}")
        gathered = np.asarray(gathered)
        if gathered.ndim < 1 or gathered.shape[-1] != self.matrix.shape[1]:
            raise ValueError(f"expected trailing cell dimension {self.matrix.shape[1]}, got {gathered.shape}")
        batch_shape = gathered.shape[:-1]
        row_sums = self.row_sums if how == "mean" else None
        dtype = projection_dtype(gathered.dtype, self.matrix.dtype, row_sums)
        out = np.empty((*batch_shape, self.matrix.shape[0]), dtype=dtype)
        for index in np.ndindex(batch_shape):
            out[index] = project_values(self.matrix, gathered[index], row_sums=row_sums, skipna=skipna)
        return out

    @classmethod
    def compute(
        cls,
        operator: ReduceOperator,
        source_lat: np.ndarray,
        lat_chunks: ChunkSizes,
        lon_chunks: ChunkSizes,
    ) -> "RestrictedOperator":
        """Build from stored latitudes and explicit chunk sizes (regular or irregular)."""
        lat, lat_chunks, lon_chunks = _restriction_inputs(operator, source_lat, lat_chunks, lon_chunks)
        matrix = operator.matrix.copy()
        matrix.eliminate_zeros()
        columns, indices = np.unique(matrix.indices, return_inverse=True)
        row, col = np.divmod(columns, operator.source_lon.size)
        if lat.size > 1 and lat[0] > lat[-1]:
            row = lat.size - 1 - row
        lat_edges, lon_edges = np.cumsum((0, *lat_chunks)), np.cumsum((0, *lon_chunks))
        chunk_row = np.searchsorted(lat_edges, row, side="right") - 1
        chunk_col = np.searchsorted(lon_edges, col, side="right") - 1
        touched, chunk_of = np.unique(np.column_stack((chunk_row, chunk_col)), axis=0, return_inverse=True)
        bounds, window_of_chunk = _chunk_rectangles(touched)
        windows = tuple(
            (slice(int(lat_edges[r0]), int(lat_edges[r1])), slice(int(lon_edges[c0]), int(lon_edges[c1])))
            for r0, r1, c0, c1 in bounds
        )
        window_of = window_of_chunk[chunk_of]
        order = np.argsort(window_of, kind="stable")
        offsets = np.r_[0, np.cumsum(np.bincount(window_of, minlength=len(windows)))]
        gathers = []
        for i, (rows, cols) in enumerate(windows):
            selected = order[offsets[i]:offsets[i + 1]]
            gathers.append((row[selected] - rows.start) * (cols.stop - cols.start) + col[selected] - cols.start)
        inverse = np.empty_like(order)
        inverse[order] = np.arange(order.size)
        compact = sp.csr_matrix(
            (matrix.data, inverse[indices], matrix.indptr), shape=(matrix.shape[0], columns.size),
        )
        return cls(
            windows, tuple(gathers), compact, operator.row_sums, operator.keys,
            lat.copy(), operator.source_lon.copy(), lat_chunks, lon_chunks,
            restricted_operator_digest(operator, lat, lat_chunks, lon_chunks),
        )

    @classmethod
    def from_grid(
        cls,
        operator: ReduceOperator,
        grid: xr.DataArray,
        *,
        lat_dim: str = "latitude",
        lon_dim: str = "longitude",
    ) -> "RestrictedOperator":
        """Infer layout from Dask chunks or backend encoding, without reading values.

        For a Dataset, pass one spatial variable; reuse the plan for variables
        with the same layout. For missing/stale metadata, use ``compute`` with
        explicit chunk sizes instead. Dask and Zarr are not required dependencies.
        """
        if not isinstance(grid, xr.DataArray):
            raise TypeError("from_grid expects a DataArray; select a spatial Dataset variable")
        if lat_dim not in grid.dims or lon_dim not in grid.dims:
            raise ValueError(f"grid is missing required dims {lat_dim!r} and {lon_dim!r}")
        lat_chunks, lon_chunks = _grid_chunks(grid, lat_dim), _grid_chunks(grid, lon_dim)
        if lat_chunks is None or lon_chunks is None:
            raise ValueError("grid has no spatial chunk metadata; use RestrictedOperator.compute with explicit chunks")
        canonical, _ = ensure_ascending_lats(grid[lat_dim].to_numpy())
        if not same_grid(canonical, grid[lon_dim].to_numpy(), operator.source_lat, operator.source_lon):
            raise ValueError("grid does not match the operator's source grid")
        return cls.compute(operator, grid[lat_dim].to_numpy(), lat_chunks, lon_chunks)
