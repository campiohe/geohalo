import hashlib
from dataclasses import dataclass

import shapely


@dataclass(frozen=True, repr=False)
class PolygonSet:
    keys: list[tuple]
    geoms: list[shapely.Geometry]
    key_names: tuple[str, ...]
    digest: bytes

    def __repr__(self) -> str:
        return (
            f"PolygonSet(n={len(self.keys)}, key_names={self.key_names}, "
            f"digest={self.digest[:4].hex()}...)"
        )

    @classmethod
    def build(
        cls,
        *,
        geoms: list[shapely.Geometry],
        keys: list[tuple] | None = None,
        key_names: tuple[str, ...] = ("polygon_id",),
    ) -> "PolygonSet":
        if not geoms:
            raise ValueError("PolygonSet must be non-empty")
        if not key_names:
            raise ValueError("key_names must contain at least one name")
        if keys is None:
            keys = [(i,) for i in range(len(geoms))]
        if len(keys) != len(geoms):
            raise ValueError("keys and geoms must have equal length")
        arity = len(key_names)
        for k in keys:
            if not isinstance(k, tuple):
                raise TypeError(f"each key must be a tuple, got {type(k).__name__}: {k!r}")
            if len(k) != arity:
                raise ValueError(
                    f"key {k!r} has length {len(k)}, expected {arity} to match key_names={key_names}",
                )
        order = sorted(range(len(keys)), key=lambda i: keys[i])
        sorted_keys = [keys[i] for i in order]
        sorted_geoms = [geoms[i] for i in order]
        h = hashlib.sha256()
        h.update(repr(key_names).encode())
        for k, g in zip(sorted_keys, sorted_geoms, strict=True):
            h.update(repr(k).encode())
            h.update(shapely.to_wkb(g))
        return cls(
            keys=sorted_keys,
            geoms=sorted_geoms,
            key_names=key_names,
            digest=h.digest(),
        )
