"""The WKB input path must preserve GeoJSON coverage and existing cache keys."""

import hashlib

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp
import shapely
from exactextract import exact_extract
from exactextract.raster import NumPyRasterSource

from geohalo import EmptyOverlapError, LocalCache, Stencil
from geohalo.cache import _ser_stencil
from geohalo.geometry import cell_areas, geom_digest, grid_digest
from geohalo.stencil import stencil_digest


def _geoms(multiindex=False, crs=None):
    geometries = [
        shapely.box(-1.6, -1.3, 0.2, 0.4),
        shapely.MultiPolygon([shapely.box(-2, 1, -1, 2), shapely.box(1, -2, 2, -1)]),
        shapely.Polygon(
            [(-2, -2), (2, -2), (2, 2), (-2, 2)],
            holes=[[(-0.7, -0.6), (0.8, -0.6), (0.8, 0.9), (-0.7, 0.9)]],
        ),
        shapely.box(1.7, 1.8, 4.0, 4.0),  # clipped by the grid edge
        shapely.Point(0.4, -0.3).buffer(1.3, quad_segs=200),
        shapely.Polygon([(0, 0, 5), (1, 0, 5), (1, 1, 5), (0, 0, 5)]),
    ]
    labels = ["polygon", "multi", "hole", "edge", "detailed", "3d"]
    keys = (
        pd.MultiIndex.from_tuples([("BR", key) for key in labels], names=["country", "zone"])
        if multiindex else pd.Index(labels, name="zone")
    )
    return gpd.GeoSeries(geometries, index=keys, crs=crs)


def _geojson_matrix(lats, lons, geoms, spherical):
    """Independent reference using the previous geometry serialization path."""
    ordered = geoms.iloc[np.argsort([repr(key) for key in geoms.index])]
    raster = NumPyRasterSource(np.zeros((5, 5)), xmin=-2.5, ymin=-2.5, xmax=2.5, ymax=2.5)
    features = [
        {"type": "Feature", "geometry": shapely.geometry.mapping(geom), "properties": {"i": i}}
        for i, geom in enumerate(ordered)
    ]
    frame = exact_extract(raster, features, ["cell_id", "coverage"], output="pandas", include_cols=[])
    areas = cell_areas(lats, lons, spherical=spherical)
    rows, cols, weights = [], [], []
    for i, record in frame.iterrows():
        ids = record["cell_id"].astype(np.int64)
        row, col = lats.size - 1 - ids // lons.size, ids % lons.size
        rows.extend([i] * ids.size)
        cols.extend(row * lons.size + col)
        weights.extend(record["coverage"] * areas[row, col])
    return ordered.index, sp.csr_matrix((weights, (rows, cols)), shape=(len(ordered), 25))


def _previous_geom_digest(geoms):
    ordered = geoms.iloc[np.argsort([repr(key) for key in geoms.index])]
    digest = hashlib.sha256()
    digest.update(repr(tuple(geoms.index.names)).encode())
    for key, geom in zip(ordered.index, ordered, strict=True):
        digest.update(repr(key).encode())
        digest.update(shapely.to_wkb(geom))
    return digest.digest()


def _previous_stencil_digest(lats, lons, geoms, spherical=True):
    digest = hashlib.sha256()
    digest.update(grid_digest(np.sort(lats), lons))
    digest.update(b"sph" if spherical else b"flat")
    digest.update(_previous_geom_digest(geoms))
    return digest.digest()


@pytest.mark.parametrize("spherical", [False, True])
@pytest.mark.parametrize("descending", [False, True])
@pytest.mark.parametrize("multiindex", [False, True])
@pytest.mark.parametrize("crs", [None, "EPSG:4326"])
def test_wkb_matches_geojson_matrix_and_digest(spherical, descending, multiindex, crs):
    lat, lon = np.arange(-2.0, 3.0), np.arange(-2.0, 3.0)
    geoms = _geoms(multiindex, crs)
    expected_keys, expected = _geojson_matrix(lat, lon, geoms, spherical)
    expected = expected[expected_keys.get_indexer(geoms.index)]
    source_lat = lat[::-1] if descending else lat
    stencil = Stencil.compute(source_lat, lon, geoms, spherical_correction=spherical)
    pd.testing.assert_index_equal(stencil.keys, geoms.index)
    np.testing.assert_array_equal(stencil.occupancy_matrix.indptr, expected.indptr)
    np.testing.assert_array_equal(stencil.occupancy_matrix.indices, expected.indices)
    np.testing.assert_array_equal(stencil.occupancy_matrix.data, expected.data)
    np.testing.assert_array_equal(stencil.row_sums, np.asarray(expected.sum(axis=1)).ravel())
    assert stencil.digest == _previous_stencil_digest(source_lat, lon, geoms, spherical)
    assert stencil.digest == stencil_digest(source_lat, lon, geoms, spherical_correction=spherical)


def test_stencil_serializes_geometries_once_as_wkb(monkeypatch):
    lat = np.arange(-2.0, 3.0)
    geoms = _geoms()
    expected_digest = _previous_stencil_digest(lat, lat, geoms)
    encode = shapely.to_wkb
    calls = []

    def vectorized(geometries, *args, **kwargs):
        assert isinstance(geometries, np.ndarray)
        calls.append(len(geometries))
        return encode(geometries, *args, **kwargs)

    def no_geojson(*_args, **_kwargs):
        pytest.fail("stencil construction must not serialize GeoJSON")

    monkeypatch.setattr(shapely, "to_wkb", vectorized)
    monkeypatch.setattr(shapely.geometry, "mapping", no_geojson)
    stencil = Stencil.compute(lat, lat, geoms)
    assert calls == [len(geoms)]
    assert stencil.digest == expected_digest


@pytest.mark.parametrize("multiindex", [False, True])
def test_vectorized_geom_digest_preserves_existing_bytes(monkeypatch, multiindex):
    geoms = _geoms(multiindex)
    expected = _previous_geom_digest(geoms)
    encode = shapely.to_wkb
    calls = []

    def vectorized(geometries):
        assert isinstance(geometries, np.ndarray)
        calls.append(len(geometries))
        return encode(geometries)

    monkeypatch.setattr(shapely, "to_wkb", vectorized)
    assert geom_digest(geoms.iloc[::-1]) == expected
    assert calls == [len(geoms)]


def test_preexisting_stencil_cache_entry_is_reused(tmp_path, monkeypatch):
    lat = np.arange(-2.0, 3.0)
    geoms = _geoms(multiindex=True)
    keys, matrix = _geojson_matrix(lat, lat, geoms, spherical=True)
    digest = _previous_stencil_digest(lat, lat, geoms)
    previous = Stencil(matrix, keys, lat, lat, digest)
    cache = LocalCache(tmp_path)
    cache._store("stencil", digest.hex()[:16], _ser_stencil(previous))

    def no_build(*_args, **_kwargs):
        pytest.fail("existing cache entry should skip stencil construction")

    monkeypatch.setattr(Stencil, "compute", no_build)
    cached = cache.get_or_compute_stencil(lat[::-1], lat, geoms.iloc[::-1])
    assert cached.digest == digest
    requested = geoms.iloc[::-1].index
    pd.testing.assert_index_equal(cached.keys, requested)
    np.testing.assert_array_equal(cached.occupancy_matrix.toarray(), matrix[keys.get_indexer(requested)].toarray())


def test_wkb_empty_overlap_reports_original_key():
    lat = np.arange(-2.0, 3.0)
    key = ("BR", "outside")
    geoms = gpd.GeoSeries(
        [shapely.box(20, 20, 21, 21)], index=pd.MultiIndex.from_tuples([key], names=["country", "zone"]),
    )
    with pytest.raises(EmptyOverlapError) as caught:
        Stencil.compute(lat, lat, geoms)
    assert caught.value.geom_key == key


def test_missing_geometry_is_rejected_before_native_extraction():
    lat = np.arange(-2.0, 3.0)
    geoms = gpd.GeoSeries([shapely.box(0, 0, 1, 1), None], index=["valid", "missing"])
    with pytest.raises(ValueError, match="missing geometries"):
        Stencil.compute(lat, lat, geoms)
