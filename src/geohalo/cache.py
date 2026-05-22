import dataclasses
import pickle as pk
from pathlib import Path

import numpy as np
import scipy.sparse as sp

try:
    import redis
except ImportError as exc:
    redis = None
    _REDIS_IMPORT_ERROR: ImportError | None = exc
else:
    _REDIS_IMPORT_ERROR = None

from geohalo.downscale import resolve_factor
from geohalo.geometry import PolygonSet
from geohalo.grid import GridSpec
from geohalo.weights import Weights, compute_weights

CACHE_KEY_PREFIX = "geohalo:weights:v4"


class RedisWeightCache:
    def __init__(self, client: "redis.Redis") -> None:
        if _REDIS_IMPORT_ERROR is not None:
            raise ImportError(
                "RedisWeightCache requires the 'redis' package. Install it with `pip install geohalo[redis]`.",
            ) from _REDIS_IMPORT_ERROR
        self._client = client

    def get_or_compute(
        self,
        polygons: PolygonSet,
        grid: GridSpec,
        *,
        target_resolution: float | None = None,
        force_recompute: bool = False,
    ) -> Weights:
        factor, achieved = resolve_factor(grid, target_resolution)
        key = _cache_key(grid.digest, polygons.digest, factor)
        if not force_recompute:
            blob = self._client.get(key)
            if blob is not None:
                cached = _deserialize(blob)
                return dataclasses.replace(
                    cached,
                    target_resolution=target_resolution,
                    achieved_resolution=achieved,
                )
        weights = compute_weights(
            polygons,
            grid,
            target_resolution=target_resolution,
        )
        self._client.set(key, _serialize(weights))
        return weights


def _cache_key(grid_digest: bytes, polyset_digest: bytes, factor: int) -> str:
    components = _key_components(grid_digest, polyset_digest, factor)
    return f"{CACHE_KEY_PREFIX}:{components[0]}:{components[1]}:{components[2]}"


def _key_components(
    grid_digest: bytes,
    polyset_digest: bytes,
    factor: int,
) -> tuple[str, str, str]:
    return (grid_digest.hex()[:16], polyset_digest.hex()[:16], str(factor))


def _local_key_filename(
    grid_digest: bytes,
    polyset_digest: bytes,
    factor: int,
) -> str:
    components = _key_components(grid_digest, polyset_digest, factor)
    return f"{components[0]}_{components[1]}_{components[2]}.pkl"


def _serialize(weights: Weights) -> bytes:
    payload = {
        "version": 6,
        "data": weights.matrix.data,
        "indices": weights.matrix.indices,
        "indptr": weights.matrix.indptr,
        "shape": weights.matrix.shape,
        "polygon_keys": weights.polygon_keys,
        "key_names": weights.key_names,
        "native_shape": weights.native_shape,
        "grid_digest": weights.grid_digest,
        "polyset_digest": weights.polyset_digest,
        "downscale_factor": weights.downscale_factor,
        "target_resolution": weights.target_resolution,
        "achieved_resolution": weights.achieved_resolution,
    }
    return pk.dumps(payload, protocol=pk.HIGHEST_PROTOCOL)


def _deserialize(blob: bytes) -> Weights:
    payload = pk.loads(blob)
    version = payload.get("version")
    if version != 6:
        raise ValueError(
            f"unsupported cache payload version: {version!r}. "
            f"Pre-v6 entries are from before the log-interpolation removal; clear the cache and recompute.",
        )
    matrix = sp.csr_matrix(
        (np.asarray(payload["data"]), np.asarray(payload["indices"]), np.asarray(payload["indptr"])),
        shape=tuple(payload["shape"]),
    )
    return Weights(
        matrix=matrix,
        polygon_keys=list(payload["polygon_keys"]),
        key_names=tuple(payload["key_names"]),
        native_shape=tuple(payload["native_shape"]),
        grid_digest=payload["grid_digest"],
        polyset_digest=payload["polyset_digest"],
        downscale_factor=int(payload["downscale_factor"]),
        target_resolution=payload["target_resolution"],
        achieved_resolution=float(payload["achieved_resolution"]),
    )


class LocalWeightCache:
    def __init__(self, path: str | Path) -> None:
        self._root = Path(path)

    def get_or_compute(
        self,
        polygons: PolygonSet,
        grid: GridSpec,
        *,
        target_resolution: float | None = None,
        force_recompute: bool = False,
    ) -> Weights:
        factor, achieved = resolve_factor(grid, target_resolution)
        filename = _local_key_filename(grid.digest, polygons.digest, factor)
        path = self._root / filename
        if not force_recompute and path.exists():
            cached = _deserialize(path.read_bytes())
            return dataclasses.replace(
                cached,
                target_resolution=target_resolution,
                achieved_resolution=achieved,
            )
        weights = compute_weights(
            polygons,
            grid,
            target_resolution=target_resolution,
        )
        blob = _serialize(weights)
        self._root.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        try:
            tmp.write_bytes(blob)
            tmp.replace(path)
        finally:
            if tmp.exists():
                tmp.unlink()
        return weights
