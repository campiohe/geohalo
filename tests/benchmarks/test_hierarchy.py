import geopandas as gpd
import pandas as pd
import shapely

from benchmarks._data import build_brazil_hierarchy


def _munis():
    keys = [("BRA.1_1", "BRA.1.1_1"), ("BRA.1_1", "BRA.1.2_1"), ("BRA.2_1", "BRA.2.1_1")]
    geoms = [shapely.box(0, 0, 1, 1)] * len(keys)
    return gpd.GeoSeries(geoms, index=pd.Index(keys, name="gid", tupleize_cols=False))


def test_depth3_edges_muni_to_state_to_country():
    edges = build_brazil_hierarchy(_munis())
    # municipality children point at their state (GID_1)
    assert edges.loc[[("BRA.1_1", "BRA.1.1_1")], "parent"].iloc[0] == "BRA.1_1"
    # state children point at the country root
    assert edges.loc[["BRA.1_1"], "parent"].iloc[0] == "BRA"
    assert edges.loc[["BRA.2_1"], "parent"].iloc[0] == "BRA"
    # one row per muni (3) + one per distinct state (2)
    assert len(edges) == 5
    assert edges.index.is_unique


def test_flat_edges_muni_to_state_only():
    edges = build_brazil_hierarchy(_munis(), depth=2)
    assert len(edges) == 3
    assert set(edges["parent"]) == {"BRA.1_1", "BRA.2_1"}
