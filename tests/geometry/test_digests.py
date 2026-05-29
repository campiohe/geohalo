import geopandas as gpd
import numpy as np
import pandas as pd
import shapely

from geohalo.geometry import geom_digest, grid_digest


def test_grid_digest_is_bytes() -> None:
    d = grid_digest(np.array([0.0, 1.0]), np.array([0.0, 1.0]))
    assert isinstance(d, bytes)
    assert len(d) == 32


def test_grid_digest_changes_with_coords() -> None:
    a = grid_digest(np.array([0.0, 1.0]), np.array([0.0, 1.0]))
    b = grid_digest(np.array([0.0, 2.0]), np.array([0.0, 1.0]))
    assert a != b


def test_grid_digest_dtype_stable() -> None:
    a = grid_digest(np.array([0.0, 1.0], dtype=np.float32), np.array([0.0, 1.0]))
    b = grid_digest(np.array([0.0, 1.0], dtype=np.float64), np.array([0.0, 1.0]))
    assert a == b


def test_geom_digest_order_invariant() -> None:
    a = shapely.box(0, 0, 1, 1)
    b = shapely.box(2, 2, 3, 3)
    d1 = geom_digest(gpd.GeoSeries([a, b], index=["x", "y"]))
    d2 = geom_digest(gpd.GeoSeries([b, a], index=["y", "x"]))
    assert d1 == d2


def test_geom_digest_differs_with_geometry() -> None:
    d1 = geom_digest(gpd.GeoSeries([shapely.box(0, 0, 1, 1)], index=["k"]))
    d2 = geom_digest(gpd.GeoSeries([shapely.box(0, 0, 2, 2)], index=["k"]))
    assert d1 != d2


def test_geom_digest_multiindex() -> None:
    idx = pd.MultiIndex.from_tuples([("BR", "SP"), ("BR", "RJ")], names=["c", "s"])
    d = geom_digest(gpd.GeoSeries([shapely.box(0, 0, 1, 1), shapely.box(2, 2, 3, 3)], index=idx))
    assert isinstance(d, bytes)
