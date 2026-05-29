"""A cache hit must skip the (expensive) build, not just the serialization."""

import geopandas as gpd
import numpy as np
import pandas as pd
import shapely

import geohalo.bias_tree as bias_tree_mod
import geohalo.reduce_operator as reduce_operator_mod
import geohalo.resampler as resampler_mod
import geohalo.stencil as stencil_mod
from geohalo.cache import LocalCache


class _CallCounter:
    """Wrap a classmethod's underlying function, counting invocations."""

    def __init__(self) -> None:
        self.calls = 0

    def wrap(self, func):
        def counted(*args, **kwargs):
            self.calls += 1
            return func(*args, **kwargs)

        return counted


def test_stencil_hit_skips_compute(tmp_path, monkeypatch) -> None:
    lats = np.array([0.0, 1.0])
    lons = np.array([0.0, 1.0])
    geoms = gpd.GeoSeries([shapely.box(-0.4, -0.4, 1.4, 1.4)], index=["box"])
    cache = LocalCache(tmp_path)

    counter = _CallCounter()
    monkeypatch.setattr(stencil_mod.Stencil, "compute", counter.wrap(stencil_mod.Stencil.compute))

    cache.get_or_compute_stencil(lats, lons, geoms)  # miss -> builds
    cache.get_or_compute_stencil(lats, lons, geoms)  # hit  -> must not build
    assert counter.calls == 1

    cache.get_or_compute_stencil(lats, lons, geoms, force_recompute=True)  # forced rebuild
    assert counter.calls == 2


def test_resampler_hit_skips_compute(tmp_path, monkeypatch) -> None:
    s_lat, s_lon = np.array([0.0, 1.0, 2.0]), np.array([0.0, 1.0, 2.0])
    t_lat, t_lon = np.linspace(0.0, 2.0, 5), np.linspace(0.0, 2.0, 5)
    cache = LocalCache(tmp_path)

    counter = _CallCounter()
    monkeypatch.setattr(resampler_mod.Resampler, "compute", counter.wrap(resampler_mod.Resampler.compute))

    cache.get_or_compute_resampler(s_lat, s_lon, t_lat, t_lon, iterations=2)
    cache.get_or_compute_resampler(s_lat, s_lon, t_lat, t_lon, iterations=2)
    assert counter.calls == 1


def test_tree_hit_skips_compute(tmp_path, monkeypatch) -> None:
    edges = pd.DataFrame({"parent": ["p", "p"]}, index=pd.Index(["a", "b"], name="child"))
    cache = LocalCache(tmp_path)

    counter = _CallCounter()
    monkeypatch.setattr(bias_tree_mod.BiasTree, "compute", counter.wrap(bias_tree_mod.BiasTree.compute))

    cache.get_or_compute_tree(edges)
    cache.get_or_compute_tree(edges)
    assert counter.calls == 1


def test_reduce_operator_hit_skips_compute(tmp_path, monkeypatch) -> None:
    lats = np.array([0.0, 1.0, 2.0])
    lons = np.array([0.0, 1.0, 2.0])
    geoms = gpd.GeoSeries([shapely.box(-0.4, -0.4, 2.4, 2.4)], index=["box"])
    cache = LocalCache(tmp_path)
    stencil = stencil_mod.Stencil.compute(lats, lons, geoms)

    counter = _CallCounter()
    monkeypatch.setattr(
        reduce_operator_mod.ReduceOperator,
        "compute",
        counter.wrap(reduce_operator_mod.ReduceOperator.compute),
    )

    cache.get_or_compute_reduce_operator(stencil, lats, lons)
    cache.get_or_compute_reduce_operator(stencil, lats, lons)
    assert counter.calls == 1
