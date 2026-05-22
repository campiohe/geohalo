"""PolygonSet.build input validation."""

from collections.abc import Callable

import pytest
import shapely

from geohalo.geometry import PolygonSet


def test_empty_geoms_raises() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        PolygonSet.build(geoms=[])


def test_empty_key_names_raises(box: Callable[[int], shapely.Polygon]) -> None:
    with pytest.raises(ValueError, match="at least one name"):
        PolygonSet.build(geoms=[box(0)], keys=[(0,)], key_names=())


def test_keys_geoms_length_mismatch_raises(box: Callable[[int], shapely.Polygon]) -> None:
    with pytest.raises(ValueError, match="equal length"):
        PolygonSet.build(geoms=[box(0), box(1)], keys=[(0,)])


def test_non_tuple_key_raises(box: Callable[[int], shapely.Polygon]) -> None:
    with pytest.raises(TypeError, match="must be a tuple"):
        PolygonSet.build(geoms=[box(0)], keys=[0])  # type: ignore[list-item]


def test_arity_mismatch_raises(box: Callable[[int], shapely.Polygon]) -> None:
    with pytest.raises(ValueError, match="length 1, expected 2"):
        PolygonSet.build(
            geoms=[box(0)],
            keys=[(0,)],
            key_names=("country", "region"),
        )


def test_default_keys_when_none_provided(box: Callable[[int], shapely.Polygon]) -> None:
    ps = PolygonSet.build(geoms=[box(0), box(1), box(2)])
    assert ps.keys == [(0,), (1,), (2,)]
    assert ps.key_names == ("polygon_id",)
