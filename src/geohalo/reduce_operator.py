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
from dataclasses import dataclass
from functools import cached_property
from typing import Literal

import numpy as np
import pandas as pd
import scipy.sparse as sp
from numpy.typing import DTypeLike

from geohalo._serialization import NPZSerializable
from geohalo._sparse import GridMatrix, cast_matrix, operator_dtype
from geohalo.geometry import _validate_period, ensure_ascending_lats, grid_digest, same_grid
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
