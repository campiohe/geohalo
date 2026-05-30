import geopandas as gpd
import numpy as np
import shapely

from geohalo.stencil import Stencil


def test_spherical_flag_changes_digest() -> None:
    lats = np.array([60.0, 61.0, 62.0])
    lons = np.array([0.0, 1.0, 2.0])
    geoms = gpd.GeoSeries([shapely.box(0.3, 60.3, 1.7, 61.7)], index=["x"])
    s1 = Stencil.compute(lats, lons, geoms, spherical_correction=True)
    s2 = Stencil.compute(lats, lons, geoms, spherical_correction=False)
    assert s1.digest != s2.digest
    assert not np.isclose(s1.row_sums[0], s2.row_sums[0])
