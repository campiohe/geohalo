"""LocalCache / RedisCache for precomputed geometry and grid operators.

Every cache key is derived from the *inputs* (via the ``*_digest`` helpers) so a
hit returns the stored object without ever running the expensive build. The two
backends share all of that logic in :class:`_Cache`; they differ only in the
``_load``/``_store`` storage primitives.
"""

from collections.abc import Callable
from dataclasses import replace
from pathlib import Path
from typing import Literal

import geopandas as gpd
import numpy as np
import pandas as pd
from numpy.typing import DTypeLike

try:
    import redis
except ImportError as exc:
    redis = None
    _REDIS_IMPORT_ERROR: ImportError | None = exc
else:
    _REDIS_IMPORT_ERROR = None

from geohalo._conservative import GridBounds, Normalization, ResampleMethod
from geohalo.bias_tree import BiasTree, tree_digest
from geohalo.reduce_operator import ReduceOperator, reduce_operator_digest
from geohalo.resampler import Resampler, resampler_digest
from geohalo.restricted_operator import ChunkSizes, RestrictedOperator, restricted_operator_digest
from geohalo.stencil import Stencil, stencil_digest

STENCIL_PREFIX = "geohalo:stencil:npz:v1"
RESAMPLER_PREFIX = "geohalo:resampler:npz:v1"
TREE_PREFIX = "geohalo:tree:npz:v1"
REDUCE_OP_PREFIX = "geohalo:reduceop:npz:v1"
RESTRICTED_OP_PREFIX = "geohalo:restrictedop:npz:v1"

# namespace -> Redis key prefix; LocalCache uses the namespace directly as a subdir.
_REDIS_PREFIXES = {
    "stencil": STENCIL_PREFIX,
    "resampler": RESAMPLER_PREFIX,
    "tree": TREE_PREFIX,
    "reduceop": REDUCE_OP_PREFIX,
    "restrictedop": RESTRICTED_OP_PREFIX,
}


# Private aliases retained for benchmark and cache callers; there is one format.
_ser_stencil, _deser_stencil = Stencil.to_npz, Stencil.from_npz
_ser_resampler, _deser_resampler = Resampler.to_npz, Resampler.from_npz
_ser_tree, _deser_tree = BiasTree.to_npz, BiasTree.from_npz
_ser_reduce_op, _deser_reduce_op = ReduceOperator.to_npz, ReduceOperator.from_npz
_ser_restricted_op, _deser_restricted_op = RestrictedOperator.to_npz, RestrictedOperator.from_npz


def _take_rows[T: Stencil | ReduceOperator | RestrictedOperator](
    obj: T, order: np.ndarray, keys: pd.Index,
) -> T:
    """Permute every row-aligned field, without copying already-ordered matrices."""
    if np.array_equal(order, np.arange(len(order))):
        return obj if obj.keys.identical(keys) else replace(obj, keys=keys)
    if isinstance(obj, Stencil):
        return replace(obj, occupancy_matrix=obj.occupancy_matrix[order], keys=keys)
    return replace(obj, matrix=obj.matrix[order], row_sums=obj.row_sums[order], keys=keys)


class _Cache:
    """Get-or-compute logic shared by both backends.

    The key for each object is derived from its *inputs* via a ``*_digest``
    helper, so a hit short-circuits the (expensive) build entirely. Subclasses
    implement only ``_load`` and ``_store``.
    """

    def _load(self, namespace: str, key: str) -> bytes | None:
        raise NotImplementedError

    def _store(self, namespace: str, key: str, blob: bytes) -> None:
        raise NotImplementedError

    def _get_or_compute[T](
        self,
        namespace: str,
        digest: bytes,
        compute: Callable[[], T],
        serialize: Callable[[T], bytes],
        deserialize: Callable[[bytes], T],
        force: bool,
    ) -> T:
        key = digest.hex()[:16]
        if not force:
            blob = self._load(namespace, key)
            if blob is not None:
                obj = deserialize(blob)
                if obj.digest != digest:
                    raise ValueError("cached object digest does not match requested inputs")
                return obj
        obj = compute()
        self._store(namespace, key, serialize(obj))
        return obj

    def _get_or_compute_rows[T: Stencil | ReduceOperator | RestrictedOperator](
        self,
        namespace: str,
        digest: bytes,
        keys: pd.Index,
        compute: Callable[[], T],
        serialize: Callable[[T], bytes],
        deserialize: Callable[[bytes], T],
        force: bool,
    ) -> T:
        """Store canonical rows, but return the requested order on both misses and hits.

        Use the same positional sort as geometry hashing, not a key lookup:
        duplicate labels need distinct rows too. Standalone NPZ exports retain
        their object's row order; only cache storage uses canonical rows.
        """
        order = np.argsort([repr(key) for key in keys])
        inverse = np.empty_like(order)
        inverse[order] = np.arange(len(order))
        return self._get_or_compute(
            namespace, digest, compute,
            lambda obj: serialize(_take_rows(obj, order, keys.take(order))),
            lambda blob: _take_rows(deserialize(blob), inverse, keys),
            force,
        )

    def get_or_compute_stencil(
        self,
        lats: np.ndarray,
        lons: np.ndarray,
        geoms: gpd.GeoSeries,
        *,
        spherical_correction: bool = True,
        dtype: DTypeLike = np.float64,
        force_recompute: bool = False,
    ) -> Stencil:
        digest = stencil_digest(lats, lons, geoms, spherical_correction=spherical_correction, dtype=dtype)
        return self._get_or_compute_rows(
            "stencil",
            digest,
            geoms.index,
            lambda: Stencil.compute(lats, lons, geoms, spherical_correction=spherical_correction, dtype=dtype),
            _ser_stencil,
            _deser_stencil,
            force_recompute,
        )

    def get_or_compute_resampler(  # noqa: PLR0913 - preserve the public keyword API
        self,
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
        force_recompute: bool = False,
    ) -> Resampler:
        digest = resampler_digest(
            source_lat, source_lon, target_lat, target_lon, iterations, period=period,
            method=method, normalization=normalization, source_bounds=source_bounds, target_bounds=target_bounds,
        )
        return self._get_or_compute(
            "resampler",
            digest,
            lambda: Resampler.compute(
                source_lat, source_lon, target_lat, target_lon, iterations=iterations, period=period,
                method=method, normalization=normalization, source_bounds=source_bounds, target_bounds=target_bounds,
            ),
            _ser_resampler,
            _deser_resampler,
            force_recompute,
        )

    def get_or_compute_tree(
        self,
        edges: pd.DataFrame,
        *,
        parent_col: str = "parent",
        weight_col: str | None = None,
        how: Literal["mean", "sum"] = "mean",
        force_recompute: bool = False,
    ) -> BiasTree:
        digest = tree_digest(edges, parent_col=parent_col, weight_col=weight_col, how=how)
        return self._get_or_compute(
            "tree",
            digest,
            lambda: BiasTree.compute(edges, parent_col=parent_col, weight_col=weight_col, how=how),
            _ser_tree,
            _deser_tree,
            force_recompute,
        )

    def get_or_compute_reduce_operator(
        self,
        stencil: Stencil,
        source_lat: np.ndarray,
        source_lon: np.ndarray,
        *,
        iterations: int = 1,
        dtype: DTypeLike = np.float64,
        period: float | None = None,
        force_recompute: bool = False,
    ) -> ReduceOperator:
        digest = reduce_operator_digest(stencil.digest, source_lat, source_lon, iterations, dtype=dtype, period=period)
        return self._get_or_compute_rows(
            "reduceop",
            digest,
            stencil.keys,
            lambda: ReduceOperator.compute(
                stencil, source_lat, source_lon, iterations=iterations, dtype=dtype, period=period,
            ),
            _ser_reduce_op,
            _deser_reduce_op,
            force_recompute,
        )

    def get_or_compute_restricted_operator(
        self,
        operator: ReduceOperator,
        source_lat: np.ndarray,
        lat_chunks: ChunkSizes,
        lon_chunks: ChunkSizes,
        *,
        force_recompute: bool = False,
    ) -> RestrictedOperator:
        """Cache a read plan for a fused operator, stored latitude order, and layout."""
        digest = restricted_operator_digest(operator, source_lat, lat_chunks, lon_chunks)
        return self._get_or_compute_rows(
            "restrictedop", digest, operator.keys,
            lambda: RestrictedOperator.compute(operator, source_lat, lat_chunks, lon_chunks),
            _ser_restricted_op, _deser_restricted_op, force_recompute,
        )


class LocalCache(_Cache):
    """NPZ files under ``path/<namespace>/<key>.npz``; legacy entries are ignored."""

    def __init__(self, path: str | Path) -> None:
        self._root = Path(path)

    def _path(self, namespace: str, key: str) -> Path:
        return self._root / namespace / f"{key}.npz"

    def _load(self, namespace: str, key: str) -> bytes | None:
        path = self._path(namespace, key)
        return path.read_bytes() if path.exists() else None

    def _store(self, namespace: str, key: str, blob: bytes) -> None:
        path = self._path(namespace, key)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_bytes(blob)
        tmp.replace(path)  # atomic publish


class RedisCache(_Cache):
    """Values under ``<prefix>:<key>`` in Redis."""

    def __init__(self, client: "redis.Redis") -> None:
        if _REDIS_IMPORT_ERROR is not None:
            raise ImportError(
                "RedisCache requires the 'redis' package. Install via geohalo[redis].",
            ) from _REDIS_IMPORT_ERROR
        self._client = client

    def _load(self, namespace: str, key: str) -> bytes | None:
        return self._client.get(f"{_REDIS_PREFIXES[namespace]}:{key}")

    def _store(self, namespace: str, key: str, blob: bytes) -> None:
        self._client.set(f"{_REDIS_PREFIXES[namespace]}:{key}", blob)
