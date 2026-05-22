"""PolygonSet sorts by key on construction."""

from collections.abc import Callable

import shapely

from geohalo.geometry import PolygonSet


def test_keys_sorted_by_default_tuple_ordering(box: Callable[[int], shapely.Polygon]) -> None:
    ps = PolygonSet.build(
        geoms=[box(0), box(1), box(2)],
        keys=[("c",), ("a",), ("b",)],
        key_names=("name",),
    )
    assert ps.keys == [("a",), ("b",), ("c",)]


def test_geoms_reordered_consistent_with_keys(box: Callable[[int], shapely.Polygon]) -> None:
    geoms = [box(0), box(1), box(2)]
    ps = PolygonSet.build(
        geoms=geoms,
        keys=[("c",), ("a",), ("b",)],
        key_names=("name",),
    )
    # geom for ("a",) was originally paired with index 1 -> box(1)
    assert ps.geoms[0] == geoms[1]
    assert ps.geoms[1] == geoms[2]
    assert ps.geoms[2] == geoms[0]
