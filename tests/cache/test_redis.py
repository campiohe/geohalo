import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
import shapely

from geohalo.cache import RedisCache


@pytest.mark.redis
def test_redis_stencil(redis_client) -> None:
    cache = RedisCache(redis_client)
    lats = np.array([0.0, 1.0])
    lons = np.array([0.0, 1.0])
    geoms = gpd.GeoSeries([shapely.box(-0.4, -0.4, 1.4, 1.4)], index=["box"])
    s1 = cache.get_or_compute_stencil(lats, lons, geoms)
    s2 = cache.get_or_compute_stencil(lats, lons, geoms)
    np.testing.assert_array_equal(s1.occupancy_matrix.toarray(), s2.occupancy_matrix.toarray())


@pytest.mark.redis
def test_redis_resampler(redis_client) -> None:
    cache = RedisCache(redis_client)
    s_lat = np.array([0.0, 1.0, 2.0])
    s_lon = np.array([0.0, 1.0, 2.0])
    t_lat = np.linspace(0.0, 2.0, 5)
    t_lon = np.linspace(0.0, 2.0, 5)
    r1 = cache.get_or_compute_resampler(s_lat, s_lon, t_lat, t_lon)
    r2 = cache.get_or_compute_resampler(s_lat, s_lon, t_lat, t_lon)
    np.testing.assert_array_equal(r1.transform_matrix.toarray(), r2.transform_matrix.toarray())


@pytest.mark.redis
def test_redis_tree(redis_client) -> None:
    cache = RedisCache(redis_client)
    edges = pd.DataFrame({"parent": ["p", "p"]}, index=pd.Index(["a", "b"], name="child"))
    t1 = cache.get_or_compute_tree(edges)
    t2 = cache.get_or_compute_tree(edges)
    np.testing.assert_array_equal(t1.rollup_matrix.toarray(), t2.rollup_matrix.toarray())
