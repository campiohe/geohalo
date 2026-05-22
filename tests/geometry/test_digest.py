"""PolygonSet.digest invariants."""

from collections.abc import Callable

import shapely

from geohalo.geometry import PolygonSet


def test_digest_invariant_to_input_order(box: Callable[[int], shapely.Polygon]) -> None:
    a = PolygonSet.build(
        geoms=[box(0), box(1), box(2)],
        keys=[("a",), ("b",), ("c",)],
        key_names=("name",),
    )
    b = PolygonSet.build(
        geoms=[box(2), box(0), box(1)],
        keys=[("c",), ("a",), ("b",)],
        key_names=("name",),
    )
    assert a.digest == b.digest


def test_digest_changes_with_key_names(box: Callable[[int], shapely.Polygon]) -> None:
    a = PolygonSet.build(geoms=[box(0)], keys=[(0,)], key_names=("a",))
    b = PolygonSet.build(geoms=[box(0)], keys=[(0,)], key_names=("b",))
    assert a.digest != b.digest


def test_digest_changes_with_geometry(box: Callable[[int], shapely.Polygon]) -> None:
    a = PolygonSet.build(geoms=[box(0)], keys=[(0,)])
    b = PolygonSet.build(geoms=[box(5)], keys=[(0,)])
    assert a.digest != b.digest


def test_digest_changes_with_key_value(box: Callable[[int], shapely.Polygon]) -> None:
    a = PolygonSet.build(geoms=[box(0)], keys=[(0,)])
    b = PolygonSet.build(geoms=[box(0)], keys=[(1,)])
    assert a.digest != b.digest


def test_digest_is_32_bytes(box: Callable[[int], shapely.Polygon]) -> None:
    ps = PolygonSet.build(geoms=[box(0)], keys=[(0,)])
    assert isinstance(ps.digest, bytes)
    assert len(ps.digest) == 32
