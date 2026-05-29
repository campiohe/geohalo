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

import numpy as np
import pandas as pd
import scipy.sparse as sp

from geohalo.geometry import ensure_ascending_lats, grid_digest, same_grid
from geohalo.resampler import FactoredResampler
from geohalo.stencil import Stencil


def reduce_operator_digest(
    stencil_digest: bytes,
    source_lat: np.ndarray,
    source_lon: np.ndarray,
    iterations: int,
) -> bytes:
    """Cache key for a fused operator, derivable from inputs without building it.

    Canonicalises the source latitudes to ascending so a descending grid and its
    flipped twin hash identically (the resampler treats them the same).
    """
    src_lat_asc, _ = ensure_ascending_lats(source_lat)
    h = hashlib.sha256()
    h.update(stencil_digest)
    h.update(grid_digest(src_lat_asc, source_lon))
    h.update(str(iterations).encode())
    return h.digest()


@dataclass(frozen=True)
class ReduceOperator:
    matrix: sp.csr_matrix  # fused occ@T, (n_polygons, n_source); == occ if grids match
    row_sums: np.ndarray  # per-polygon occupancy total, for mean normalisation
    keys: pd.Index
    source_lat: np.ndarray  # ascending (canonical)
    source_lon: np.ndarray
    iterations: int
    digest: bytes

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
    ) -> "ReduceOperator":
        src_lat_asc, _ = ensure_ascending_lats(source_lat)
        src_lon = np.asarray(source_lon, dtype=np.float64)
        occ = stencil.occupancy_matrix

        if same_grid(src_lat_asc, src_lon, stencil.lats, stencil.lons):
            matrix = occ.tocsr()
        else:
            resampler = FactoredResampler.compute(
                src_lat_asc, src_lon, stencil.lats, stencil.lons, iterations=iterations,
            )
            matrix = resampler.fuse_left(occ)

        digest = reduce_operator_digest(stencil.digest, src_lat_asc, src_lon, iterations)
        return cls(
            matrix=matrix,
            row_sums=np.asarray(stencil.row_sums, dtype=np.float64),
            keys=stencil.keys,
            source_lat=src_lat_asc,
            source_lon=src_lon,
            iterations=iterations,
            digest=digest,
        )
