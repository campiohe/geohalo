"""LocalWeightCache: read/write, atomic writes, target_resolution re-attach."""

from pathlib import Path
from unittest import mock

from geohalo import cache as cache_module
from geohalo.cache import LocalWeightCache
from geohalo.geometry import PolygonSet
from geohalo.grid import GridSpec


def test_first_call_writes_second_call_reads(
    tmp_path: Path,
    simple_grid: GridSpec,
    simple_polygons: PolygonSet,
) -> None:
    cache = LocalWeightCache(tmp_path)
    with mock.patch(
        "geohalo.cache.compute_weights",
        wraps=cache_module.compute_weights,
    ) as patched:
        cache.get_or_compute(simple_polygons, simple_grid)
        cache.get_or_compute(simple_polygons, simple_grid)
    assert patched.call_count == 1


def test_force_recompute_bypasses_cache(
    tmp_path: Path,
    simple_grid: GridSpec,
    simple_polygons: PolygonSet,
) -> None:
    cache = LocalWeightCache(tmp_path)
    with mock.patch(
        "geohalo.cache.compute_weights",
        wraps=cache_module.compute_weights,
    ) as patched:
        cache.get_or_compute(simple_polygons, simple_grid)
        cache.get_or_compute(
            simple_polygons, simple_grid, force_recompute=True,
        )
    assert patched.call_count == 2


def test_atomic_write_leaves_no_tmp_files(
    tmp_path: Path,
    simple_grid: GridSpec,
    simple_polygons: PolygonSet,
) -> None:
    cache = LocalWeightCache(tmp_path)
    cache.get_or_compute(simple_polygons, simple_grid)
    leftover = list(tmp_path.glob("*.tmp"))
    assert leftover == []


def test_target_resolution_not_in_key_same_factor(
    tmp_path: Path,
    simple_grid: GridSpec,
    simple_polygons: PolygonSet,
) -> None:
    """target_resolution=0.5 and 0.51 resolve to the same factor.
    Second call hits the cache, but the returned weights carry the caller's
    target_resolution value.
    """
    cache = LocalWeightCache(tmp_path)
    with mock.patch(
        "geohalo.cache.compute_weights",
        wraps=cache_module.compute_weights,
    ) as patched:
        w1 = cache.get_or_compute(
            simple_polygons, simple_grid, target_resolution=0.5,
        )
        w2 = cache.get_or_compute(
            simple_polygons, simple_grid, target_resolution=0.51,
        )
    assert patched.call_count == 1
    assert w1.target_resolution == 0.5
    assert w2.target_resolution == 0.51
    assert w1.downscale_factor == w2.downscale_factor == 2


def test_different_factor_different_entry(
    tmp_path: Path,
    simple_grid: GridSpec,
    simple_polygons: PolygonSet,
) -> None:
    cache = LocalWeightCache(tmp_path)
    with mock.patch(
        "geohalo.cache.compute_weights",
        wraps=cache_module.compute_weights,
    ) as patched:
        cache.get_or_compute(
            simple_polygons, simple_grid, target_resolution=0.5,
        )
        cache.get_or_compute(
            simple_polygons, simple_grid, target_resolution=0.25,
        )
    assert patched.call_count == 2
