import geopandas as gpd
import numpy as np
import shapely

from benchmarks.run import _bench_cache, _format_speedup, _time
from geohalo.cache import LocalCache


def test_time_returns_timing_rss_and_last_result() -> None:
    calls = []
    timing, rss, result = _time(lambda: calls.append(1) or 7, warmup=0, iters=1)
    assert result == 7
    assert "median" in timing
    assert "p10" in timing
    assert "p90" in timing
    assert rss >= 0


def test_format_speedup() -> None:
    assert _format_speedup(182.4) == "182x"
    assert _format_speedup(3.14) == "3.1x"
    assert _format_speedup(float("inf")) == "inf"


def test_bench_cache_miss_then_hit(tmp_path) -> None:
    cache = LocalCache(tmp_path)
    lats = np.array([0.0, 1.0, 2.0])
    lons = np.array([0.0, 1.0, 2.0])
    geoms = gpd.GeoSeries([shapely.box(-0.4, -0.4, 2.4, 2.4)], index=["box"])

    metrics = _bench_cache(
        lambda: cache.get_or_compute_stencil(lats, lons, geoms, force_recompute=True),
        lambda: cache.get_or_compute_stencil(lats, lons, geoms),
        miss_cfg={"warmup": 0, "iters": 1},
    )
    assert metrics["obj"].digest == cache.get_or_compute_stencil(lats, lons, geoms).digest
    assert metrics["speedup"] > 0
    assert any(tmp_path.rglob("*.pkl"))
