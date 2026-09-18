"""Resampler: value-independent materialized or separable grid transforms."""

import hashlib
from dataclasses import dataclass
from functools import cached_property

import numpy as np
import scipy.sparse as sp

from geohalo._conservative import (
    ConservativeGrid,
    GridBounds,
    Normalization,
    ResampleMethod,
    canonical_bounds,
    validate_method,
)
from geohalo._serialization import NPZSerializable
from geohalo._sparse import GridMatrix
from geohalo.geometry import (
    _validate_period,
    bilinear_matrix_1d,
    conservative_matrix_1d,
    ensure_ascending_lats,
    nearest_index,
)


@dataclass(frozen=True)
class Resampler(NPZSerializable):
    """Grid transform whose source columns use ascending latitude, then longitude.

    Target rows retain the supplied coordinate order. Direct matrix callers must
    flatten source values in ``(source_lat, source_lon)`` order; the xarray API
    handles descending source latitudes automatically.

    The default mean-preserving method stores ``transform_matrix``. Conservative
    resampling instead stores ``axis_weights=(latitude, longitude)`` and leaves
    ``transform_matrix=None``; use ``apply_grid`` for either representation.
    """

    transform_matrix: sp.csr_matrix | None
    source_lat: np.ndarray
    source_lon: np.ndarray
    target_lat: np.ndarray
    target_lon: np.ndarray
    digest: bytes
    axis_weights: tuple[sp.csr_matrix, sp.csr_matrix] | None = None
    normalization: Normalization = "destination"

    @property
    def method(self) -> ResampleMethod:
        return "conservative" if self.axis_weights is not None else "meanpreserving"

    @cached_property
    def _conservative_grid(self) -> ConservativeGrid:
        if self.axis_weights is None:
            raise ValueError("area coverage is only available for conservative resamplers")
        return ConservativeGrid(*self.axis_weights, self.normalization)

    @property
    def coverage(self) -> np.ndarray:
        """Source-covered fraction of each target cell (conservative only)."""
        return self._conservative_grid.coverage

    @cached_property
    def _grid_matrix(self) -> GridMatrix:
        return GridMatrix(self.transform_matrix, (self.source_lat.size, self.source_lon.size))

    def apply_grid(
        self, values: np.ndarray, *, descending: bool = False, skipna: bool = False, preserve_dtype: bool = False,
    ) -> np.ndarray:
        """Transform (..., latitude, longitude) values into (..., n_target).

        ``descending=True`` interprets source rows in descending latitude order.
        Temporary dense allocations are bounded per slice or small batch block.
        Conservative resamplers use two 1-D contractions per slice. Uncovered
        cells return NaN. For this method only, ``skipna=True`` renormalizes over
        valid covered area, overriding destination-area normalization; all
        missing cells return NaN. The default propagates contributing NaNs.

        ``preserve_dtype=True`` returns real floating inputs in their own dtype
        for either method. Computation and normalization retain their existing
        precision; only result storage changes. Integer, boolean, and complex
        inputs retain normal promotion. Defaults are unchanged.
        """
        if self.axis_weights is not None:
            return self._conservative_grid.apply(
                values, descending=descending, skipna=skipna, preserve_dtype=preserve_dtype,
            )
        if skipna:
            raise ValueError("resampling skipna=True requires method='conservative'")
        return self._grid_matrix.apply(values, descending=descending, preserve_dtype=preserve_dtype)

    def __repr__(self) -> str:
        storage = (
            f"method='conservative', axis_nnz={sum(axis.nnz for axis in self.axis_weights)}"
            if self.axis_weights is not None else f"nnz={self.transform_matrix.nnz}"
        )
        return (
            f"Resampler(source=({self.source_lat.size}, {self.source_lon.size}), "
            f"target=({self.target_lat.size}, {self.target_lon.size}), "
            f"{storage})"
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
        period: float | None = None,
        method: ResampleMethod = "meanpreserving",
        normalization: Normalization = "destination",
        source_bounds: GridBounds | None = None,
        target_bounds: GridBounds | None = None,
    ) -> "Resampler":
        """Build a mean-preserving or separable conservative grid transform.

        The default clamps both axes. Set ``period=360`` for cyclic longitude,
        with no repeated source endpoint. Explicit target coordinates retain
        their values and order, including targets outside the source cycle.

        ``method='conservative'`` averages spherical cell overlaps using two
        1-D ``axis_weights``; ``transform_matrix`` is None. ``normalization`` is
        'destination' (full target area) or 'covered' (source-covered area).
        Optional ``source_bounds``/``target_bounds`` are (latitude, longitude)
        edge arrays, each one longer than its coordinate axis, in that axis's
        order. Otherwise midpoint edges are inferred (latitude clipped at the
        poles, cyclic source longitude when periodic). Singleton noncyclic axes
        require bounds. Conservative axes must be strictly monotonic, and
        ``iterations`` must remain 1. Geometries and existing reducers are not
        changed by this grid-resampling option.
        """
        validate_method(method, iterations, normalization, source_bounds, target_bounds)
        if iterations < 1:
            raise ValueError(f"iterations must be >= 1, got {iterations}")
        source_lat, descending = ensure_ascending_lats(source_lat)
        source_bounds = canonical_bounds(source_bounds, descending_lat=descending)
        target_bounds = canonical_bounds(target_bounds)
        source_lon = np.asarray(source_lon, dtype=np.float64)
        target_lat = np.asarray(target_lat, dtype=np.float64)
        target_lon = np.asarray(target_lon, dtype=np.float64)

        axis_weights = None
        if method == "conservative":
            axis_weights = (
                conservative_matrix_1d(
                    source_lat, target_lat, latitude=True,
                    source_bounds=None if source_bounds is None else source_bounds[0],
                    target_bounds=None if target_bounds is None else target_bounds[0],
                ),
                conservative_matrix_1d(
                    source_lon, target_lon, period=period,
                    source_bounds=None if source_bounds is None else source_bounds[1],
                    target_bounds=None if target_bounds is None else target_bounds[1],
                ),
            )
            transform = None
        else:
            transform = _build_transform(source_lat, source_lon, target_lat, target_lon, iterations, period=period)
        digest = resampler_digest(
            source_lat, source_lon, target_lat, target_lon, iterations, period=period,
            method=method, normalization=normalization, source_bounds=source_bounds, target_bounds=target_bounds,
        )
        return cls(
            transform_matrix=transform,
            source_lat=source_lat,
            source_lon=source_lon,
            target_lat=target_lat,
            target_lon=target_lon,
            digest=digest,
            axis_weights=axis_weights,
            normalization=normalization,
        )


def _build_factors(
    source_lat: np.ndarray,
    source_lon: np.ndarray,
    target_lat: np.ndarray,
    target_lon: np.ndarray,
    *,
    period: float | None = None,
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
        bilinear_matrix_1d(source_lon, target_lon, period=period),
        format="csr",
    )

    parent_lat = nearest_index(source_lat, target_lat)
    parent_lon = nearest_index(source_lon, target_lon, period=period)
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
    *,
    period: float | None = None,
) -> sp.csr_matrix:
    n_s = source_lat.size * source_lon.size
    b, a, p = _build_factors(source_lat, source_lon, target_lat, target_lon, period=period)

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

    Source cells use ascending latitude, then the supplied longitude order,
    matching :class:`Resampler`. Target cells retain the supplied order.
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
        period: float | None = None,
    ) -> "FactoredResampler":
        """Build factors with the same longitude-only ``period`` as Resampler."""
        if iterations < 1:
            raise ValueError(f"iterations must be >= 1, got {iterations}")
        source_lat, _ = ensure_ascending_lats(source_lat)
        source_lon = np.asarray(source_lon, dtype=np.float64)
        target_lat = np.asarray(target_lat, dtype=np.float64)
        target_lon = np.asarray(target_lon, dtype=np.float64)
        b, a, p = _build_factors(source_lat, source_lon, target_lat, target_lon, period=period)
        digest = resampler_digest(source_lat, source_lon, target_lat, target_lon, iterations, period=period)
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
        """Apply to (batch, n_source) values flattened in ascending latitude order."""
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
    *,
    period: float | None = None,
    method: ResampleMethod = "meanpreserving",
    normalization: Normalization = "destination",
    source_bounds: GridBounds | None = None,
    target_bounds: GridBounds | None = None,
) -> bytes:
    """Cache key with ascending source latitudes; target order is significant.

    Nonperiodic keys retain their original bytes; cyclic keys include the period.
    """
    validate_method(method, iterations, normalization, source_bounds, target_bounds)
    period = _validate_period(source_lon, period)
    source_lat, descending = ensure_ascending_lats(source_lat)
    source_bounds = canonical_bounds(source_bounds, descending_lat=descending)
    target_bounds = canonical_bounds(target_bounds)
    h = hashlib.sha256()
    for arr in (source_lat, source_lon, target_lat, target_lon):
        h.update(np.asarray(arr, dtype=np.float64).tobytes())
    h.update(str(iterations).encode())
    if period is not None:
        h.update(b"period:" + np.float64(period).tobytes())
    if method == "conservative":
        h.update(b"method:conservative;normalization:" + normalization.encode())
        for name, bounds in ((b"source_bounds:", source_bounds), (b"target_bounds:", target_bounds)):
            h.update(name)
            if bounds is None:
                h.update(b"inferred")
            else:
                for axis in bounds:
                    h.update(repr(axis.shape).encode())
                    h.update(axis.tobytes())
    return h.digest()
