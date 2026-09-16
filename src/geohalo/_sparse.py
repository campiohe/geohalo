"""Apply spatial sparse matrices without copying or upcasting a whole batch."""

import math
from dataclasses import dataclass
from functools import cached_property

import numpy as np
import scipy.sparse as sp

_SMALL_GRID_CELLS = 10_000
_BATCH_BYTES = 1024 * 1024


def projection_dtype(
    value_dtype: np.dtype, matrix_dtype: np.dtype, row_sums: np.ndarray | None,
) -> np.dtype:
    """Match sparse multiplication followed by NumPy's true-division promotion."""
    dtype = np.result_type(value_dtype, matrix_dtype)
    if row_sums is not None:
        dtype = np.result_type(dtype, row_sums.dtype)
        if dtype.kind in "biu":
            dtype = np.dtype(np.float64)
    return dtype


def project_values(
    matrix: sp.csr_matrix,
    values: np.ndarray,
    *,
    row_sums: np.ndarray | None = None,
    skipna: bool = False,
) -> np.ndarray:
    """Project a source vector (or a bounded clean batch), optionally normalizing.

    NaN-aware callers pass one slice at a time. Only dirty means need a second
    product; sums omit missing contributions without changing their weights.
    A nonpositive surviving signed weight is not a valid mean denominator.
    """
    valid = ~np.isnan(values) if skipna else None
    if valid is not None and not valid.all():
        total = matrix @ np.where(valid, values, 0)
        if row_sums is None:
            return total
        dtype = projection_dtype(values.dtype, matrix.dtype, row_sums)
        denominator = matrix @ valid.astype(dtype)
    else:
        total = matrix @ values if values.ndim == 1 else values @ matrix.T
        if row_sums is None:
            return total
        denominator = row_sums
    if not skipna:
        return total / denominator
    out = np.full(total.shape, np.nan, dtype=projection_dtype(values.dtype, matrix.dtype, row_sums))
    np.divide(total, denominator, out=out, where=denominator > 0)
    return out


@dataclass(frozen=True)
class GridMatrix:
    matrix: sp.csr_matrix
    source_shape: tuple[int, int]
    cells: tuple[np.ndarray, np.ndarray] | None = None

    @classmethod
    def for_reduction(cls, matrix: sp.csr_matrix, source_shape: tuple[int, int]) -> "GridMatrix":
        """Compact large operators to their referenced columns, keeping coefficient order."""
        if matrix.shape[1] <= _SMALL_GRID_CELLS:
            return cls(matrix, source_shape)
        columns, indices = np.unique(matrix.indices, return_inverse=True)
        # Gathering nearly the whole grid adds indexing work without a useful
        # memory saving. Keep those operators on the bounded full-slice path.
        if columns.size > matrix.shape[1] // 2:
            return cls(matrix, source_shape)
        compact = sp.csr_matrix(
            (matrix.data, indices, matrix.indptr), shape=(matrix.shape[0], columns.size),
        )
        return cls(compact, source_shape, np.divmod(columns, source_shape[1]))

    @cached_property
    def _descending_matrix(self) -> sp.csr_matrix:
        """Remap columns to stored latitude order without reordering coefficients."""
        n_lat, n_lon = self.source_shape
        row, col = np.divmod(self.matrix.indices, n_lon)
        indices = (n_lat - 1 - row) * n_lon + col
        # Share coefficients and row pointers; only column indices change.
        return sp.csr_matrix((self.matrix.data, indices, self.matrix.indptr), shape=self.matrix.shape)

    def apply(
        self, values: np.ndarray, *, descending: bool,
        row_sums: np.ndarray | None = None, skipna: bool = False,
    ) -> np.ndarray:
        """Apply to (..., latitude, longitude), returning (..., matrix rows)."""
        if values.shape[-2:] != self.source_shape:
            raise ValueError(f"expected trailing source shape {self.source_shape}, got {values.shape}")
        batch_shape = values.shape[:-2]
        dtype = projection_dtype(values.dtype, self.matrix.dtype, row_sums)
        out = np.empty((*batch_shape, self.matrix.shape[0]), dtype=dtype)
        matrix = self.matrix
        if self.cells is None and descending:
            matrix = self._descending_matrix

        # Small contiguous grids benefit from batched products. Bound each
        # block so SciPy's contiguous copy/upcast cannot scale with batch size.
        n_source = math.prod(self.source_shape)
        bytes_per_slice = (
            n_source * (values.dtype.itemsize + dtype.itemsize) + matrix.shape[0] * dtype.itemsize
        )
        block_size = max(1, _BATCH_BYTES // max(bytes_per_slice, 1))
        if not skipna and n_source <= _SMALL_GRID_CELLS and block_size > 1 and values.flags.c_contiguous:
            batch_size = math.prod(batch_shape)
            flat = values.reshape(batch_size, n_source)
            result = out.reshape(batch_size, matrix.shape[0])
            for start in range(0, batch_size, block_size):
                stop = start + block_size
                result[start:stop] = project_values(matrix, flat[start:stop], row_sums=row_sums)
            return out

        if self.cells is not None:
            rows, cols = self.cells
            if descending:
                rows = self.source_shape[0] - 1 - rows
        for index in np.ndindex(batch_shape):
            # Index the batch before flattening: a strided array must never
            # trigger a reshape copy of the entire input. Gather first when
            # only a subset of source cells contributes to the reduction.
            step = values[index]
            flat = step.ravel() if self.cells is None else step[rows, cols]
            out[index] = project_values(matrix, flat, row_sums=row_sums, skipna=skipna)
        return out
