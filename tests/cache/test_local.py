import geopandas as gpd
import numpy as np
import pandas as pd
import shapely

from geohalo.cache import LocalCache


def test_stencil_roundtrip(tmp_path) -> None:
    lats = np.array([0.0, 1.0])
    lons = np.array([0.0, 1.0])
    geoms = gpd.GeoSeries([shapely.box(-0.4, -0.4, 1.4, 1.4)], index=["box"])
    cache = LocalCache(tmp_path)
    s1 = cache.get_or_compute_stencil(lats, lons, geoms)
    s2 = cache.get_or_compute_stencil(lats, lons, geoms)
    np.testing.assert_array_equal(s1.occupancy_matrix.toarray(), s2.occupancy_matrix.toarray())
    assert s1.digest == s2.digest
    assert list(s1.keys) == list(s2.keys)


def test_stencil_spherical_separates(tmp_path) -> None:
    lats = np.array([60.0, 61.0])
    lons = np.array([0.0, 1.0])
    geoms = gpd.GeoSeries([shapely.box(0.3, 60.3, 0.7, 60.7)], index=["x"])
    cache = LocalCache(tmp_path)
    s1 = cache.get_or_compute_stencil(lats, lons, geoms, spherical_correction=True)
    s2 = cache.get_or_compute_stencil(lats, lons, geoms, spherical_correction=False)
    assert s1.digest != s2.digest


def test_resampler_roundtrip(tmp_path) -> None:
    s_lat = np.array([0.0, 1.0, 2.0])
    s_lon = np.array([0.0, 1.0, 2.0])
    t_lat = np.linspace(0.0, 2.0, 5)
    t_lon = np.linspace(0.0, 2.0, 5)
    cache = LocalCache(tmp_path)
    r1 = cache.get_or_compute_resampler(s_lat, s_lon, t_lat, t_lon, iterations=2)
    r2 = cache.get_or_compute_resampler(s_lat, s_lon, t_lat, t_lon, iterations=2)
    np.testing.assert_array_equal(r1.transform_matrix.toarray(), r2.transform_matrix.toarray())


def test_tree_roundtrip(tmp_path) -> None:
    edges = pd.DataFrame({"parent": ["p", "p"]}, index=pd.Index(["a", "b"], name="child"))
    cache = LocalCache(tmp_path)
    t1 = cache.get_or_compute_tree(edges)
    t2 = cache.get_or_compute_tree(edges)
    np.testing.assert_array_equal(t1.rollup_matrix.toarray(), t2.rollup_matrix.toarray())
    assert t1.how == "mean"
