import geopandas as gpd

from benchmarks._data import AMERICAS_ISO3, EUROPE_ISO3, polygons_from_gadm


def _square(x0, y0):
    return {
        "type": "Polygon",
        "coordinates": [[[x0, y0], [x0 + 1, y0], [x0 + 1, y0 + 1], [x0, y0 + 1], [x0, y0]]],
    }


def _geojson_level2():
    return {
        "features": [
            {"geometry": _square(0, 0), "properties": {"GID_1": "A.1_1", "GID_2": "A.1.1_1"}},
            {"geometry": _square(2, 0), "properties": {"GID_1": "A.2_1", "GID_2": "A.2.1_1"}},
        ],
    }


def test_polygons_from_gadm_keys_by_requested_level_2():
    geoms = polygons_from_gadm(_geojson_level2(), level=2)
    assert isinstance(geoms, gpd.GeoSeries)
    assert list(geoms.index) == [("A.1_1", "A.1.1_1"), ("A.2_1", "A.2.1_1")]
    assert len(geoms) == 2


def test_polygons_from_gadm_level_0_keys_by_country():
    gj = {"features": [{"geometry": _square(0, 0), "properties": {"GID_0": "BRA"}}]}
    geoms = polygons_from_gadm(gj, level=0)
    assert list(geoms.index) == ["BRA"]


def test_iso3_lists_are_disjoint_and_sized():
    assert len(AMERICAS_ISO3) >= 35
    assert len(EUROPE_ISO3) >= 40
    assert set(AMERICAS_ISO3).isdisjoint(EUROPE_ISO3)
    assert "BRA" in AMERICAS_ISO3
    assert "DEU" in EUROPE_ISO3
