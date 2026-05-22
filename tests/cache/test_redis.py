"""RedisWeightCache against a real Redis container."""

from typing import TYPE_CHECKING
from unittest import mock

import pytest

from geohalo import cache as cache_module
from geohalo.cache import RedisWeightCache
from geohalo.geometry import PolygonSet
from geohalo.grid import GridSpec

if TYPE_CHECKING:
    import redis


@pytest.mark.redis
def test_first_call_writes_second_call_reads(
    redis_client: "redis.Redis",
    simple_grid: GridSpec,
    simple_polygons: PolygonSet,
) -> None:
    cache = RedisWeightCache(client=redis_client)
    with mock.patch(
        "geohalo.cache.compute_weights",
        wraps=cache_module.compute_weights,
    ) as patched:
        cache.get_or_compute(simple_polygons, simple_grid)
        cache.get_or_compute(simple_polygons, simple_grid)
    assert patched.call_count == 1


@pytest.mark.redis
def test_target_resolution_reattached_on_hit(
    redis_client: "redis.Redis",
    simple_grid: GridSpec,
    simple_polygons: PolygonSet,
) -> None:
    cache = RedisWeightCache(client=redis_client)
    w1 = cache.get_or_compute(
        simple_polygons, simple_grid, target_resolution=0.5,
    )
    w2 = cache.get_or_compute(
        simple_polygons, simple_grid, target_resolution=0.51,
    )
    assert w1.target_resolution == 0.5
    assert w2.target_resolution == 0.51
    assert w1.downscale_factor == w2.downscale_factor == 2


@pytest.mark.redis
def test_force_recompute_bypasses_cache(
    redis_client: "redis.Redis",
    simple_grid: GridSpec,
    simple_polygons: PolygonSet,
) -> None:
    cache = RedisWeightCache(client=redis_client)
    with mock.patch(
        "geohalo.cache.compute_weights",
        wraps=cache_module.compute_weights,
    ) as patched:
        cache.get_or_compute(simple_polygons, simple_grid)
        cache.get_or_compute(
            simple_polygons, simple_grid, force_recompute=True,
        )
    assert patched.call_count == 2
