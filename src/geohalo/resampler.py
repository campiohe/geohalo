"""Resampler: value-independent grid->grid transform matrix."""

import hashlib
from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp

from geohalo.geometry import bilinear_matrix_1d, nearest_index


@dataclass(frozen=True)
class Resampler:
    transform_matrix: sp.csr_matrix
    source_lat: np.ndarray
    source_lon: np.ndarray
    target_lat: np.ndarray
    target_lon: np.ndarray
    digest: bytes

    def __repr__(self) -> str:
        return (
            f"Resampler(source=({self.source_lat.size}, {self.source_lon.size}), "
            f"target=({self.target_lat.size}, {self.target_lon.size}), "
            f"nnz={self.transform_matrix.nnz})"
        )

    @classmethod
    def compute(
        cls,
        source_lat: np.ndarray,
        source_lon: np.ndarray,
        target_lat: np.ndarray,
        target_lon: np.ndarray,
        *,
        iterations: int = 1,
    ) -> "Resampler":
        if iterations < 1:
            raise ValueError(f"iterations must be >= 1, got {iterations}")
        source_lat = np.asarray(source_lat, dtype=np.float64)
        source_lon = np.asarray(source_lon, dtype=np.float64)
        target_lat = np.asarray(target_lat, dtype=np.float64)
        target_lon = np.asarray(target_lon, dtype=np.float64)

        transform = _build_transform(source_lat, source_lon, target_lat, target_lon, iterations)
        digest = resampler_digest(source_lat, source_lon, target_lat, target_lon, iterations)
        return cls(
            transform_matrix=transform,
            source_lat=source_lat,
            source_lon=source_lon,
            target_lat=target_lat,
            target_lon=target_lon,
            digest=digest,
        )


def _build_factors(
    source_lat: np.ndarray,
    source_lon: np.ndarray,
    target_lat: np.ndarray,
    target_lon: np.ndarray,
) -> tuple[sp.csr_matrix, sp.csr_matrix, sp.csr_matrix]:
    """Iteration-independent operators (B, A, P) shared by both resampler forms.

    B: bilinear interpolation (target <- source). A: source <- target mean.
    P: source -> target nearest broadcast.
    """
    n_s_lon = source_lon.size
    n_s = source_lat.size * n_s_lon
    n_t = target_lat.size * target_lon.size

    b = sp.kron(
        bilinear_matrix_1d(source_lat, target_lat),
        bilinear_matrix_1d(source_lon, target_lon),
        format="csr",
    )

    parent_lat = nearest_index(source_lat, target_lat)
    parent_lon = nearest_index(source_lon, target_lon)
    parent_flat = (parent_lat[:, None] * n_s_lon + parent_lon[None, :]).ravel()
    t_idx = np.arange(n_t)

    p = sp.csr_matrix((np.ones(n_t), (t_idx, parent_flat)), shape=(n_t, n_s))
    counts = np.bincount(parent_flat, minlength=n_s)
    inv = np.where(counts > 0, 1.0 / np.maximum(counts, 1), 0.0)
    a = sp.csr_matrix((inv[parent_flat], (parent_flat, t_idx)), shape=(n_s, n_t))
    return b, a, p


def _build_transform(
    source_lat: np.ndarray,
    source_lon: np.ndarray,
    target_lat: np.ndarray,
    target_lon: np.ndarray,
    iterations: int,
) -> sp.csr_matrix:
    n_s = source_lat.size * source_lon.size
    b, a, p = _build_factors(source_lat, source_lon, target_lat, target_lon)

    # y_op = (sum_{j=0}^{iterations-1} G^j) @ B,  G = I_T - B@A.
    # Push B inside the recurrence and apply G on the right so every
    # intermediate stays (n_t, n_s) instead of materialising the dense
    # (n_t, n_t) operator G and its powers.
    acc = b
    term = b
    for _ in range(iterations - 1):
        term = (term - b @ (a @ term)).tocsr()
        term.eliminate_zeros()
        acc = (acc + term).tocsr()
    y_op = acc.tocsr()

    # T = y_op + P @ (I_S - A @ y_op)
    correction = (p @ (sp.eye(n_s, format="csr") - a @ y_op)).tocsr()
    transform = (y_op + correction).tocsr()
    transform.eliminate_zeros()
    return transform


@dataclass(frozen=True)
class FactoredResampler:
    """Un-fused resampler: keeps (B, A, P) and runs the iteration at apply time.

    Unlike :class:`Resampler`, this builds none of the power series, so `compute`
    costs only the three base operators. Each `apply_flat` call then pays the
    per-iteration matmuls instead of amortising them into one materialised
    transform. This is the form the reduce path needs: `fuse_left` can compute
    ``w @ T`` for a thin ``w`` without ever materialising ``T`` (see
    :class:`~geohalo.reduce_operator.ReduceOperator`).
    """

    b: sp.csr_matrix
    a: sp.csr_matrix
    p: sp.csr_matrix
    iterations: int
    source_lat: np.ndarray
    source_lon: np.ndarray
    target_lat: np.ndarray
    target_lon: np.ndarray
    digest: bytes

    @classmethod
    def compute(
        cls,
        source_lat: np.ndarray,
        source_lon: np.ndarray,
        target_lat: np.ndarray,
        target_lon: np.ndarray,
        *,
        iterations: int = 1,
    ) -> "FactoredResampler":
        if iterations < 1:
            raise ValueError(f"iterations must be >= 1, got {iterations}")
        source_lat = np.asarray(source_lat, dtype=np.float64)
        source_lon = np.asarray(source_lon, dtype=np.float64)
        target_lat = np.asarray(target_lat, dtype=np.float64)
        target_lon = np.asarray(target_lon, dtype=np.float64)
        b, a, p = _build_factors(source_lat, source_lon, target_lat, target_lon)
        digest = resampler_digest(source_lat, source_lon, target_lat, target_lon, iterations)
        return cls(
            b=b,
            a=a,
            p=p,
            iterations=iterations,
            source_lat=source_lat,
            source_lon=source_lon,
            target_lat=target_lat,
            target_lon=target_lon,
            digest=digest,
        )

    def apply_flat(self, flat: np.ndarray) -> np.ndarray:
        """Apply the transform to data laid out as (batch, n_source)."""
        xt = np.asarray(flat, dtype=np.float64).T
        y_op = self.b @ xt
        term = y_op
        for _ in range(self.iterations - 1):
            term = term - self.b @ (self.a @ term)
            y_op = y_op + term
        resid = xt - self.a @ y_op
        return (y_op + self.p @ resid).T

    def fuse_left(self, w: sp.csr_matrix) -> sp.csr_matrix:
        """Return ``w @ T`` without ever materialising ``T``.

        ``w`` is an aggregation operator on the target grid (e.g. a stencil's
        occupancy matrix, ``n_rows x n_target``). The product passes through the
        target grid, but because ``w`` is thin every intermediate stays
        ``n_rows``-by-something, so this scales to high iteration counts and
        large targets where building ``T`` itself would not fit.

        ``T = y_op + P(I - A·y_op)``, ``y_op = (sum_j G^j)·B``, ``G = I - B·A``.
        Then ``w@T = w@P + (w - w@P@A)·(sum_j G^j)·B``, and the series is
        accumulated by right-applying ``G`` to the thin ``(w - w@P@A)``.
        """
        wp = (w @ self.p).tocsr()  # (n_rows, n_source)
        acc = (w - wp @ self.a).tocsr()  # (n_rows, n_target)
        term = acc
        for _ in range(self.iterations - 1):
            term = (term - (term @ self.b) @ self.a).tocsr()  # term @ G
            term.eliminate_zeros()
            acc = (acc + term).tocsr()
        return (wp + acc @ self.b).tocsr()


def resampler_digest(
    source_lat: np.ndarray,
    source_lon: np.ndarray,
    target_lat: np.ndarray,
    target_lon: np.ndarray,
    iterations: int,
) -> bytes:
    """Cache key for a resampler, derivable from inputs without building it."""
    h = hashlib.sha256()
    for arr in (source_lat, source_lon, target_lat, target_lon):
        h.update(np.asarray(arr, dtype=np.float64).tobytes())
    h.update(str(iterations).encode())
    return h.digest()
