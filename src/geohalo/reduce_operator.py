"""ReduceOperator: the fused ``occ @ T`` operator for the reduce hot path.

Building a resampler ``T`` and a stencil ``occ`` separately and multiplying them
at every call is expensive when the target grid is fine: ``T`` spans the whole
fine grid even though only the cells under polygons matter. ``ReduceOperator``
precomputes the fused ``occ @ T`` once (thinly, via
:meth:`FactoredResampler.fuse_left`, so ``T`` is never materialised) into a tiny
``(n_polygons, n_source)`` matrix. Applying it is a single sparse matmul on the
*source* grid; the fine grid never appears at apply time.

This is the cacheable unit for the reduce path: the fused matrix is orders of
magnitude smaller than ``T`` and its size is independent of the iteration count.
"""

import hashlib
from collections.abc import Sequence
from dataclasses import dataclass
from functools import cached_property
from typing import Literal

import geopandas as gpd
import numpy as np
import pandas as pd
import scipy.sparse as sp
from numpy.typing import DTypeLike

from geohalo._serialization import NPZSerializable
from geohalo._sparse import GridMatrix, cast_matrix, operator_dtype, stored_row_sums
from geohalo._weighted import counts_digest, require_inside, require_nested, weight_window, weighted_matrix
from geohalo.geometry import (
    _validate_period,
    ensure_ascending_lats,
    geom_digest,
    grid_digest,
    midpoint_edges,
    require_regular_grid,
    same_grid,
)
from geohalo.resampler import FactoredResampler
from geohalo.stencil import Stencil


def reduce_operator_digest(
    stencil_digest: bytes,
    source_lat: np.ndarray,
    source_lon: np.ndarray,
    iterations: int,
    *,
    dtype: DTypeLike = np.float64,
    period: float | None = None,
) -> bytes:
    """Cache key for a fused operator, derivable from inputs without building it.

    Canonicalises the source latitudes to ascending so a descending grid and its
    flipped twin hash identically (the resampler treats them the same).
    Coefficient dtype distinguishes entries; float64 retains existing keys.
    """
    src_lat_asc, _ = ensure_ascending_lats(source_lat)
    period = _validate_period(source_lon, period)
    h = hashlib.sha256()
    h.update(stencil_digest)
    h.update(grid_digest(src_lat_asc, source_lon))
    h.update(str(iterations).encode())
    dtype = operator_dtype(dtype)
    if dtype != np.float64:
        h.update(b"dtype:" + dtype.name.encode())
    if period is not None:
        h.update(b"period:" + np.float64(period).tobytes())
    return h.digest()


@dataclass(frozen=True)
class ReduceOperator(NPZSerializable):
    matrix: sp.csr_matrix  # fused occ@T, (n_polygons, n_source); == occ if grids match
    row_sums: np.ndarray  # per-polygon occupancy total, for mean normalisation
    keys: pd.Index
    source_lat: np.ndarray  # ascending (canonical)
    source_lon: np.ndarray
    iterations: int
    digest: bytes

    @cached_property
    def _grid_matrix(self) -> GridMatrix:
        return GridMatrix.for_reduction(self.matrix, (self.source_lat.size, self.source_lon.size))

    def apply_grid(
        self, values: np.ndarray, *, descending: bool = False,
        how: Literal["mean", "sum"] = "sum", skipna: bool = False,
        preserve_dtype: bool = False,
    ) -> np.ndarray:
        """Project (..., latitude, longitude) values; default to an unnormalized sum.

        Large grids gather only referenced cells, one batch slice at a time.
        ``descending=True`` interprets source rows in descending latitude order.
        The canonical matrix and its arithmetic precision are unchanged.

        ``how="mean"`` divides by ``row_sums``. With ``skipna=True``, sums omit
        missing source cells and means divide by the surviving signed weight
        (NaN if nonpositive). This is source-cell masking, not resample-then-mask.

        ``preserve_dtype=True`` returns floating inputs in their own dtype.
        Multiplication and normalization keep their existing precision; integer
        inputs retain normal promotion, so means are never truncated to integers.
        """
        if how not in ("mean", "sum"):
            raise ValueError(f"how must be 'mean' or 'sum', got {how!r}")
        return self._grid_matrix.apply(
            values, descending=descending, row_sums=self.row_sums if how == "mean" else None,
            skipna=skipna, preserve_dtype=preserve_dtype,
        )

    def __repr__(self) -> str:
        return (
            f"ReduceOperator(polygons={len(self.keys)}, source={self.source_lat.size}x{self.source_lon.size}, "
            f"iterations={self.iterations}, nnz={self.matrix.nnz})"
        )

    @classmethod
    def compute(
        cls,
        stencil: Stencil,
        source_lat: np.ndarray,
        source_lon: np.ndarray,
        *,
        iterations: int = 1,
        dtype: DTypeLike = np.float64,
        period: float | None = None,
    ) -> "ReduceOperator":
        """Fuse with float64 (default) or float32 stored coefficients.

        Fusion uses the existing float64 resampling math before casting. The
        stencil's float64 row sums are retained; casting does not sort entries.
        Request float32 explicitly even if the stencil already uses float32.
        ``period=360`` wraps resampling in longitude, not polygon geometries.
        """
        dtype = operator_dtype(dtype)
        src_lat_asc, _ = ensure_ascending_lats(source_lat)
        src_lon = np.asarray(source_lon, dtype=np.float64)
        period = _validate_period(src_lon, period)
        occ = stencil.occupancy_matrix

        if same_grid(src_lat_asc, src_lon, stencil.lats, stencil.lons):
            matrix = occ.tocsr()
        else:
            resampler = FactoredResampler.compute(
                src_lat_asc, src_lon, stencil.lats, stencil.lons, iterations=iterations, period=period,
            )
            matrix = resampler.fuse_left(occ)

        digest = reduce_operator_digest(stencil.digest, src_lat_asc, src_lon, iterations, dtype=dtype, period=period)
        return cls(
            matrix=cast_matrix(matrix, dtype),
            row_sums=np.asarray(stencil.row_sums, dtype=np.float64),
            keys=stencil.keys,
            source_lat=src_lat_asc,
            source_lon=src_lon,
            iterations=iterations,
            digest=digest,
        )

    @classmethod
    def compute_weighted(
        cls,
        geometries: gpd.GeoSeries,
        weight_lat: np.ndarray,
        weight_lon: np.ndarray,
        weights: np.ndarray,
        source_lat: np.ndarray,
        source_lon: np.ndarray,
        *,
        dtype: DTypeLike = np.float64,
    ) -> "ReduceOperator":
        """Weight each geometry's cells by the counts of a fine lattice that nests in the source grid.

        ``matrix[i, j]`` is the sum over the pixels ``k`` of cell ``j`` of
        ``coverage(i, k) * weights[k]``, where the coverage is the planar
        fraction of the pixel inside the geometry (exactextract, as
        :meth:`Stencil.compute` takes it). ``row_sums`` are the float64 sums of
        the stored coefficients, so ``how="mean"`` answers
        ``sum_j M_ij x_j / sum_j M_ij``.

        ``weight_lat``/``weight_lon`` are the centres of a regular window of the
        lattice, latitude in either order and longitude ascending; ``weights``
        has shape ``(weight_lat.size, weight_lon.size)`` in that order and holds
        counts per pixel. NaN is 0; negative or infinite counts raise
        ValueError. ``source_lat`` may run either way and is stored ascending,
        as in :meth:`compute`. Longitudes are not wrapped: pixels outside the
        source grid, and the counts in them, are dropped.

        Raises ValueError when a source cell edge inside the window is not a
        pixel edge (1e-9 degrees), or when a geometry's bounds leave the window;
        EmptyOverlapError when a geometry covers no area of the source grid
        inside the window. A geometry whose pixels all weigh 0 has a row of
        zeros (no stored entries), which a mean with ``skipna=True`` answers as
        NaN.
        """
        dtype = operator_dtype(dtype)
        if not isinstance(geometries, gpd.GeoSeries):
            raise TypeError(f"geometries must be a gpd.GeoSeries, got {type(geometries).__name__}")
        if len(geometries) == 0:
            raise ValueError("geometries is empty; need at least one polygon")
        if geometries.isna().any():
            raise ValueError("geometries contains missing geometries; expected polygons")
        if geometries.crs is not None and not geometries.crs.equals("EPSG:4326", ignore_axis_order=True):
            raise ValueError("geometries must be in longitude/latitude degrees (EPSG:4326)")
        window = weight_window(weight_lat, weight_lon, weights)
        src_lat, _ = ensure_ascending_lats(source_lat)
        src_lon = np.asarray(source_lon, dtype=np.float64)
        require_regular_grid(src_lat, "latitude")
        require_regular_grid(src_lon, "longitude")
        if src_lon.size > 1 and src_lon[0] > src_lon[-1]:
            raise ValueError("source_lon must ascend")
        lat_edges, lon_edges = midpoint_edges(src_lat), midpoint_edges(src_lon)
        require_nested(lat_edges, window.south, window.lat_step, window.north, "latitude")
        require_nested(lon_edges, window.west, window.lon_step, window.east, "longitude")
        require_inside(geometries, window, lat_edges, lon_edges)

        matrix = cast_matrix(weighted_matrix(geometries, window, src_lat, src_lon), dtype)
        matrix.eliminate_zeros()
        h = hashlib.sha256(b"geohalo:weighted:1")
        h.update(geom_digest(geometries))
        h.update(grid_digest(window.lat[::-1], window.lon))
        h.update(counts_digest(window))
        h.update(grid_digest(src_lat, src_lon))
        h.update(dtype.name.encode())
        return cls(
            matrix=matrix,
            row_sums=stored_row_sums(matrix),
            keys=geometries.index,
            source_lat=src_lat,
            source_lon=src_lon,
            iterations=1,
            digest=h.digest(),
        )

    def reweight(self, cell_weights: np.ndarray, *, descending: bool = False) -> "ReduceOperator":
        """Scale each column by its cell's weight; ``cell_weights`` has the source grid's shape.

        ``descending=True`` reads its rows north to south, as :meth:`apply_grid`
        does. NaN is 0; a negative or infinite weight on a cell the operator
        reads raises ValueError. Entries that become 0 are dropped and
        ``row_sums`` are recomputed in float64 from the stored coefficients.
        Only the cells the operator reads are looked at, and the digest covers
        exactly those weights.
        """
        shape = (self.source_lat.size, self.source_lon.size)
        weights = np.asarray(cell_weights)
        if weights.shape != shape:
            raise ValueError(f"cell_weights must have shape {shape}, got {weights.shape}")
        if weights.dtype.kind not in "fiu":
            raise TypeError(f"cell_weights must be real numbers, got {weights.dtype}")
        rows, cols = np.divmod(self.matrix.indices, shape[1])
        if descending:
            rows = shape[0] - 1 - rows
        scale = weights[rows, cols].astype(np.float64)
        scale[np.isnan(scale)] = 0.0
        if np.isinf(scale).any() or (scale < 0).any():
            raise ValueError("cell_weights must not be negative or infinite; NaN is read as 0")
        scaled = (self.matrix.data.astype(np.float64) * scale).astype(self.matrix.dtype)
        matrix = sp.csr_matrix(
            (scaled, self.matrix.indices.copy(), self.matrix.indptr.copy()), shape=self.matrix.shape,
        )
        matrix.eliminate_zeros()
        columns, first = np.unique(self.matrix.indices, return_index=True)
        h = hashlib.sha256(b"geohalo:reweighted:1")
        h.update(self.digest)
        h.update(columns.astype(np.int64).tobytes())
        h.update((scale[first] + 0.0).tobytes())
        return type(self)(
            matrix=matrix,
            row_sums=stored_row_sums(matrix),
            keys=self.keys,
            source_lat=self.source_lat,
            source_lon=self.source_lon,
            iterations=self.iterations,
            digest=h.digest(),
        )

    @classmethod
    def concat(cls, operators: Sequence["ReduceOperator"]) -> "ReduceOperator":
        """Stack operators' rows in order.

        They share one source grid, coefficient dtype, iteration count and key
        names, and no key repeats. One operator is returned as it is; otherwise
        the digest covers the parts' digests in order.
        """
        operators = list(operators)
        if not operators:
            raise ValueError("concat needs at least one operator")
        first = operators[0]
        for other in operators[1:]:
            if not same_grid(other.source_lat, other.source_lon, first.source_lat, first.source_lon):
                raise ValueError("operators must share one source grid")
            if other.matrix.dtype != first.matrix.dtype:
                raise ValueError("operators must share one coefficient dtype")
            if other.iterations != first.iterations:
                raise ValueError("operators must share one iteration count")
            if list(other.keys.names) != list(first.keys.names):
                raise ValueError("operators must share key names")
        if len(operators) == 1:
            return first
        keys = first.keys.append([other.keys for other in operators[1:]])
        if keys.has_duplicates:
            raise ValueError(f"keys repeat across operators: {list(keys[keys.duplicated()].unique()[:10])!r}")
        h = hashlib.sha256(b"geohalo:concat:1")
        for other in operators:
            h.update(other.digest)
        return cls(
            matrix=sp.vstack([other.matrix for other in operators], format="csr"),
            row_sums=np.concatenate([np.asarray(other.row_sums, dtype=np.float64) for other in operators]),
            keys=keys,
            source_lat=first.source_lat,
            source_lon=first.source_lon,
            iterations=first.iterations,
            digest=h.digest(),
        )
