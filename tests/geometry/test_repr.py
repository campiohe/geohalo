"""PolygonSet.__repr__."""

from collections.abc import Callable

import shapely

from geohalo.geometry import PolygonSet


def test_repr_contains_n_keynames_and_digest_prefix(
    box: Callable[[int], shapely.Polygon],
) -> None:
    ps = PolygonSet.build(geoms=[box(0), box(1)], keys=[(0,), (1,)])
    text = repr(ps)
    assert "n=2" in text
    assert "key_names=('polygon_id',)" in text
    assert ps.digest[:4].hex() in text
