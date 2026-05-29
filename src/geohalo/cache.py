"""LocalCache / RedisCache for Stencil, Resampler, BiasTree, and ReduceOperator.

Every cache key is derived from the *inputs* (via the ``*_digest`` helpers) so a
hit returns the stored object without ever running the expensive build. The two
backends share all of that logic in :class:`_Cache`; they differ only in the
``_load``/``_store`` storage primitives.
"""

import pickle as pk
from collections.abc import Callable
from pathlib import Path
from typing import Literal

import geopandas as gpd
import numpy as np
import pandas as pd
import scipy.sparse as sp

try:
    import redis
except ImportError as exc:
    redis = None
    _REDIS_IMPORT_ERROR: ImportError | None = exc
else:
    _REDIS_IMPORT_ERROR = None

from geohalo.bias_tree import BiasTree, tree_digest
from geohalo.reduce_operator import ReduceOperator, reduce_operator_digest
from geohalo.resampler import Resampler, resampler_digest
from geohalo.stencil import Stencil, stencil_digest

STENCIL_PREFIX = "geohalo:stencil:v1"
RESAMPLER_PREFIX = "geohalo:resampler:v1"
TREE_PREFIX = "geohalo:tree:v1"
REDUCE_OP_PREFIX = "geohalo:reduceop:v1"
PAYLOAD_VERSION = 1

# namespace -> Redis key prefix; LocalCache uses the namespace directly as a subdir.
_REDIS_PREFIXES = {
    "stencil": STENCIL_PREFIX,
    "resampler": RESAMPLER_PREFIX,
    "tree": TREE_PREFIX,
    "reduceop": REDUCE_OP_PREFIX,
}


def _csr_payload(m: sp.csr_matrix) -> dict:
    return {"data": m.data, "indices": m.indices, "indptr": m.indptr, "shape": m.shape}


def _csr_from_payload(p: dict) -> sp.csr_matrix:
    return sp.csr_matrix(
        (np.asarray(p["data"]), np.asarray(p["indices"]), np.asarray(p["indptr"])),
        shape=tuple(p["shape"]),
    )


def _index_payload(idx: pd.Index) -> dict:
    return {"values": list(idx), "names": list(idx.names)}


def _index_from_payload(p: dict) -> pd.Index:
    names, values = p["names"], p["values"]
    if len(names) > 1:
        return pd.MultiIndex.from_tuples(values, names=names)
    return pd.Index(values, name=names[0] if names else None, tupleize_cols=False)


def _ser_stencil(s: Stencil) -> bytes:
    return pk.dumps(
        {
            "version": PAYLOAD_VERSION,
            "matrix": _csr_payload(s.occupancy_matrix),
            "keys": _index_payload(s.keys),
            "lats": s.lats,
            "lons": s.lons,
            "digest": s.digest,
            "spherical_correction": s.spherical_correction,
        },
        protocol=pk.HIGHEST_PROTOCOL,
    )


def _deser_stencil(blob: bytes) -> Stencil:
    p = pk.loads(blob)
    if p.get("version") != PAYLOAD_VERSION:
        raise ValueError(f"unsupported stencil payload version: {p.get('version')!r}")
    return Stencil(
        occupancy_matrix=_csr_from_payload(p["matrix"]),
        keys=_index_from_payload(p["keys"]),
        lats=np.asarray(p["lats"]),
        lons=np.asarray(p["lons"]),
        digest=p["digest"],
        spherical_correction=p["spherical_correction"],
    )


def _ser_resampler(r: Resampler) -> bytes:
    return pk.dumps(
        {
            "version": PAYLOAD_VERSION,
            "matrix": _csr_payload(r.transform_matrix),
            "source_lat": r.source_lat,
            "source_lon": r.source_lon,
            "target_lat": r.target_lat,
            "target_lon": r.target_lon,
            "digest": r.digest,
        },
        protocol=pk.HIGHEST_PROTOCOL,
    )


def _deser_resampler(blob: bytes) -> Resampler:
    p = pk.loads(blob)
    if p.get("version") != PAYLOAD_VERSION:
        raise ValueError(f"unsupported resampler payload version: {p.get('version')!r}")
    return Resampler(
        transform_matrix=_csr_from_payload(p["matrix"]),
        source_lat=np.asarray(p["source_lat"]),
        source_lon=np.asarray(p["source_lon"]),
        target_lat=np.asarray(p["target_lat"]),
        target_lon=np.asarray(p["target_lon"]),
        digest=p["digest"],
    )


def _ser_tree(t: BiasTree) -> bytes:
    return pk.dumps(
        {
            "version": PAYLOAD_VERSION,
            "matrix": _csr_payload(t.rollup_matrix),
            "keys": _index_payload(t.keys),
            "digest": t.digest,
            "how": t.how,
        },
        protocol=pk.HIGHEST_PROTOCOL,
    )


def _deser_tree(blob: bytes) -> BiasTree:
    p = pk.loads(blob)
    if p.get("version") != PAYLOAD_VERSION:
        raise ValueError(f"unsupported tree payload version: {p.get('version')!r}")
    return BiasTree(
        rollup_matrix=_csr_from_payload(p["matrix"]),
        keys=_index_from_payload(p["keys"]),
        digest=p["digest"],
        how=p["how"],
    )


def _ser_reduce_op(o: ReduceOperator) -> bytes:
    return pk.dumps(
        {
            "version": PAYLOAD_VERSION,
            "matrix": _csr_payload(o.matrix),
            "row_sums": o.row_sums,
            "keys": _index_payload(o.keys),
            "source_lat": o.source_lat,
            "source_lon": o.source_lon,
            "iterations": o.iterations,
            "digest": o.digest,
        },
        protocol=pk.HIGHEST_PROTOCOL,
    )


def _deser_reduce_op(blob: bytes) -> ReduceOperator:
    p = pk.loads(blob)
    if p.get("version") != PAYLOAD_VERSION:
        raise ValueError(f"unsupported reduce-operator payload version: {p.get('version')!r}")
    return ReduceOperator(
        matrix=_csr_from_payload(p["matrix"]),
        row_sums=np.asarray(p["row_sums"]),
        keys=_index_from_payload(p["keys"]),
        source_lat=np.asarray(p["source_lat"]),
        source_lon=np.asarray(p["source_lon"]),
        iterations=p["iterations"],
        digest=p["digest"],
    )


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
                return deserialize(blob)
        obj = compute()
        self._store(namespace, key, serialize(obj))
        return obj

    def get_or_compute_stencil(
        self,
        lats: np.ndarray,
        lons: np.ndarray,
        geoms: gpd.GeoSeries,
        *,
        spherical_correction: bool = True,
        force_recompute: bool = False,
    ) -> Stencil:
        digest = stencil_digest(lats, lons, geoms, spherical_correction=spherical_correction)
        return self._get_or_compute(
            "stencil",
            digest,
            lambda: Stencil.compute(lats, lons, geoms, spherical_correction=spherical_correction),
            _ser_stencil,
            _deser_stencil,
            force_recompute,
        )

    def get_or_compute_resampler(
        self,
        source_lat: np.ndarray,
        source_lon: np.ndarray,
        target_lat: np.ndarray,
        target_lon: np.ndarray,
        *,
        iterations: int = 1,
        force_recompute: bool = False,
    ) -> Resampler:
        digest = resampler_digest(source_lat, source_lon, target_lat, target_lon, iterations)
        return self._get_or_compute(
            "resampler",
            digest,
            lambda: Resampler.compute(source_lat, source_lon, target_lat, target_lon, iterations=iterations),
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
        force_recompute: bool = False,
    ) -> ReduceOperator:
        digest = reduce_operator_digest(stencil.digest, source_lat, source_lon, iterations)
        return self._get_or_compute(
            "reduceop",
            digest,
            lambda: ReduceOperator.compute(stencil, source_lat, source_lon, iterations=iterations),
            _ser_reduce_op,
            _deser_reduce_op,
            force_recompute,
        )


class LocalCache(_Cache):
    """Pickle files under ``path/<namespace>/<key>.pkl``."""

    def __init__(self, path: str | Path) -> None:
        self._root = Path(path)

    def _path(self, namespace: str, key: str) -> Path:
        return self._root / namespace / f"{key}.pkl"

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
