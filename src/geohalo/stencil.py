"""Stencil: per-geom x per-cell area coverage. EmptyOverlapError lives here."""

import hashlib
from collections.abc import Hashable, Iterator
from dataclasses import dataclass, field
from typing import Literal

import geopandas as gpd
import numpy as np
import pandas as pd
import scipy.sparse as sp
import shapely
from exactextract import exact_extract
from exactextract.feature import Feature, FeatureSource
from exactextract.raster import NumPyRasterSource
from numpy.typing import DTypeLike

from geohalo._serialization import NPZSerializable
from geohalo._sparse import cast_matrix, operator_dtype
from geohalo.geometry import (
    EARTH_RADIUS_M,
    _geom_digest_from_wkb,
    _polygon_area_steradians,
    cell_areas,
    ensure_ascending_lats,
    geom_digest,
    grid_digest,
    polygon_areas,
    require_regular_grid,
)

type PartialCellWeighting = Literal["approximate", "exact"]


def _validate_partial_cell_weighting(mode: PartialCellWeighting, *, spherical_correction: bool) -> None:
    if mode not in ("approximate", "exact"):
        raise ValueError("partial_cell_weighting must be 'approximate' or 'exact'")
    if mode == "exact" and not spherical_correction:
        raise ValueError("partial_cell_weighting='exact' requires spherical_correction=True")


class _WKBFeature(Feature):
    """Read-only geometry feature; stencil extraction copies no attributes."""

    def __init__(self, wkb: bytes) -> None:
        super().__init__()
        self._wkb = wkb

    def geometry(self) -> bytes:
        return self._wkb

    def fields(self) -> list[str]:
        return []

    def set_geometry_format(self) -> str:
        return "wkb"


class _WKBFeatureSource(FeatureSource):
    """Feed pre-encoded geometry bytes directly to exactextract."""

    def __init__(self, wkb: np.ndarray) -> None:
        super().__init__()
        self._wkb = wkb

    def count(self) -> int:
        return len(self._wkb)

    def __iter__(self) -> Iterator[_WKBFeature]:
        for wkb in self._wkb:
            yield _WKBFeature(wkb)

    def srs_wkt(self) -> None:
        # Match the previous GeoJSON input: the grid and polygons are EPSG:4326.
        return None


class EmptyOverlapError(Exception):
    def __init__(self, geom_key: Hashable) -> None:
        super().__init__(f"polygon {geom_key!r} does not intersect the grid")
        self.geom_key = geom_key


@dataclass(frozen=True)
class Stencil(NPZSerializable):
    occupancy_matrix: sp.csr_matrix
    keys: pd.Index
    lats: np.ndarray
    lons: np.ndarray
    digest: bytes
    spherical_correction: bool = True
    partial_cell_weighting: PartialCellWeighting = "approximate"
    row_sums: np.ndarray = field(init=False, repr=False)

    def __post_init__(self) -> None:
        _validate_partial_cell_weighting(self.partial_cell_weighting, spherical_correction=self.spherical_correction)
        matrix = self.occupancy_matrix
        if matrix.dtype == np.float32:
            # CSR.sum(dtype=float64) may cast only AFTER float32 accumulation.
            # Reduce stored coefficients in float64 without a float64 matrix copy.
            row_sums = np.zeros(matrix.shape[0], dtype=np.float64)
            nonempty = np.diff(matrix.indptr) > 0
            row_sums[nonempty] = np.add.reduceat(matrix.data, matrix.indptr[:-1][nonempty], dtype=np.float64)
        else:
            row_sums = np.asarray(matrix.sum(axis=1)).ravel()
        object.__setattr__(self, "row_sums", row_sums)

    def __repr__(self) -> str:
        return (
            f"Stencil(geoms={len(self.keys)}, "
            f"shape=({len(self.lats)}, {len(self.lons)}), "
            f"nnz={self.occupancy_matrix.nnz})"
        )

    @classmethod
    def compute(
        cls,
        lats: np.ndarray,
        lons: np.ndarray,
        geoms: gpd.GeoSeries,
        *,
        spherical_correction: bool = True,
        partial_cell_weighting: PartialCellWeighting = "approximate",
        dtype: DTypeLike = np.float64,
    ) -> "Stencil":
        """Build caller-ordered rows with float64 (default) or float32 coefficients.

        Geometry calculations use float64 before casting stored coefficients.
        Row sums accumulate those stored weights in float64. Coordinates remain
        float64; dtype is part of the canonical digest, with old float64 keys intact.

        ``partial_cell_weighting="approximate"`` (default) multiplies planar
        coverage fractions by whole-cell areas. ``"exact"`` integrates the
        spherical area of each polygon-cell intersection, with straight lon/lat
        edges as in :func:`geohalo.geometry.polygon_areas`. It requires
        ``spherical_correction=True``. Polygon and MultiPolygon inputs follow
        that helper's CRS, coordinate, topology, and longitude requirements.
        The extra work occurs only during stencil construction.
        """
        dtype = operator_dtype(dtype)
        _validate_partial_cell_weighting(partial_cell_weighting, spherical_correction=spherical_correction)
        if not isinstance(geoms, gpd.GeoSeries):
            raise TypeError(f"geoms must be a gpd.GeoSeries, got {type(geoms).__name__}")
        if len(geoms) == 0:
            raise ValueError("geoms is empty; need at least one polygon to build a stencil")
        lats_asc, _ = ensure_ascending_lats(lats)
        lons_arr = np.asarray(lons, dtype=np.float64)
        require_regular_grid(lats_asc, "latitude")
        require_regular_grid(lons_arr, "longitude")

        if geoms.isna().any():
            # exactextract's native WKB reader cannot safely handle a null geometry.
            raise ValueError("geoms contains missing geometries; expected polygons")
        exact_geoms = None
        if partial_cell_weighting == "exact":
            # Validate the complete inputs before native extraction, including
            # CRS, topology, coordinate range, and the longitude edge convention.
            zero_area = np.flatnonzero(polygon_areas(geoms) <= 0)
            if zero_area.size:
                raise EmptyOverlapError(geoms.index[zero_area[0]])
            exact_geoms = geoms.to_numpy()
        wkb = shapely.to_wkb(geoms.to_numpy())

        matrix = _build_occupancy_matrix(
            lats_asc, lons_arr, geoms.index, wkb, spherical_correction=spherical_correction,
            exact_geoms=exact_geoms,
        )
        order = np.argsort([repr(k) for k in geoms.index])
        digest = _stencil_digest_from_geometry_digest(
            lats_asc, lons_arr, _geom_digest_from_wkb(geoms.index.take(order), wkb[order]),
            spherical_correction=spherical_correction, dtype=dtype,
            partial_cell_weighting=partial_cell_weighting,
        )
        return cls(
            occupancy_matrix=cast_matrix(matrix, dtype),
            keys=geoms.index,
            lats=lats_asc,
            lons=lons_arr,
            digest=digest,
            spherical_correction=spherical_correction,
            partial_cell_weighting=partial_cell_weighting,
        )


def _build_occupancy_matrix(
    lats: np.ndarray,
    lons: np.ndarray,
    keys: pd.Index,
    wkb: np.ndarray,
    *,
    spherical_correction: bool,
    exact_geoms: np.ndarray | None = None,
) -> sp.csr_matrix:
    n_lat, n_lon = lats.size, lons.size
    template = np.zeros((n_lat, n_lon), dtype=np.float64)
    xmin = float(lons[0] - (lons[1] - lons[0]) / 2)
    xmax = float(lons[-1] + (lons[-1] - lons[-2]) / 2)
    ymin = float(lats[0] - (lats[1] - lats[0]) / 2)
    ymax = float(lats[-1] + (lats[-1] - lats[-2]) / 2)
    src = NumPyRasterSource(template, xmin=xmin, ymin=ymin, xmax=xmax, ymax=ymax)
    features = _WKBFeatureSource(wkb)
    df = exact_extract(src, features, ops=["cell_id", "coverage"], output="pandas", include_cols=[])

    if exact_geoms is None:
        areas = cell_areas(lats, lons, spherical=spherical_correction)
    else:
        # Use the uniform footprint seen by exactextract, even for coordinates
        # whose spacing differs slightly within require_regular_grid's tolerance.
        lat_edges = np.linspace(ymin, ymax, n_lat + 1)
        lon_edges = np.linspace(xmin, xmax, n_lon + 1)
    rows: list[np.ndarray] = []
    cols: list[np.ndarray] = []
    data: list[np.ndarray] = []
    for i, key in enumerate(keys):
        cell_ids = np.asarray(df.iloc[i]["cell_id"], dtype=np.int64)
        coverage = np.asarray(df.iloc[i]["coverage"], dtype=np.float64)
        if cell_ids.size == 0:
            raise EmptyOverlapError(key)
        row_top = cell_ids // n_lon
        col = cell_ids % n_lon
        row_asc = n_lat - 1 - row_top
        if exact_geoms is None:
            weight = coverage * areas[row_asc, col]
        else:
            weight = _exact_cell_weights(exact_geoms[i], row_asc, col, lat_edges, lon_edges)
        if weight.sum() <= 0:
            raise EmptyOverlapError(key)
        rows.append(np.full(cell_ids.size, i, dtype=np.int64))
        cols.append(row_asc * n_lon + col)
        data.append(weight)

    return sp.csr_matrix(
        (np.concatenate(data), (np.concatenate(rows), np.concatenate(cols))),
        shape=(len(keys), n_lat * n_lon),
    )


def _exact_cell_weights(
    geom: shapely.Geometry,
    rows: np.ndarray,
    cols: np.ndarray,
    lat_edges: np.ndarray,
    lon_edges: np.ndarray,
) -> np.ndarray:
    """Keep whole-cell areas and integrate clipped boundary cells on the sphere."""
    south, north = lat_edges[rows], lat_edges[rows + 1]
    west, east = lon_edges[cols], lon_edges[cols + 1]
    cells = shapely.box(west, south, east, north)
    # Coverage from exactextract can round nearly-full cells to 1. Check actual
    # containment so tiny holes or uncovered strips still get integrated.
    partial = ~shapely.covers(geom, cells)
    weights = EARTH_RADIUS_M**2 * np.deg2rad(east - west) * (
        np.sin(np.deg2rad(north)) - np.sin(np.deg2rad(south))
    )
    intersections = shapely.intersection(geom, cells[partial])
    # Input geometries were validated once before extraction. Integrate GEOS
    # intersections directly without repeating public-input validation per cell.
    weights[partial] = EARTH_RADIUS_M**2 * np.fromiter(
        (_polygon_area_steradians(part) for part in intersections), dtype=np.float64, count=len(intersections),
    )
    return weights


def stencil_digest(
    lats: np.ndarray,
    lons: np.ndarray,
    geoms: gpd.GeoSeries,
    *,
    spherical_correction: bool = True,
    partial_cell_weighting: PartialCellWeighting = "approximate",
    dtype: DTypeLike = np.float64,
) -> bytes:
    """Cache key for a stencil, derivable from inputs without building it.

    Canonicalises latitudes to ascending so a grid and its flipped twin hash
    identically; ``geom_digest`` is order-invariant, so geometry order does not
    matter either.
    Coefficient dtype and partial-cell weighting distinguish entries; approximate
    float64 stencils retain existing keys.
    """
    if partial_cell_weighting == "exact" and geoms.crs is not None and not geoms.crs.equals(
        "EPSG:4326", ignore_axis_order=True,
    ):
        # Geometry digests encode WKB, not CRS; reject a projected CRS even on
        # a cache hit for identical coordinate bytes previously labelled lon/lat.
        raise ValueError("exact partial-cell weighting requires longitude/latitude degrees (EPSG:4326)")
    lats_asc, _ = ensure_ascending_lats(lats)
    lons_arr = np.asarray(lons, dtype=np.float64)
    return _stencil_digest_from_geometry_digest(
        lats_asc, lons_arr, geom_digest(geoms), spherical_correction=spherical_correction, dtype=dtype,
        partial_cell_weighting=partial_cell_weighting,
    )


def _stencil_digest_from_geometry_digest(
    lats: np.ndarray,
    lons: np.ndarray,
    geometry_digest: bytes,
    *,
    spherical_correction: bool,
    partial_cell_weighting: PartialCellWeighting = "approximate",
    dtype: DTypeLike = np.float64,
) -> bytes:
    """Combine canonical grid coordinates with an already-computed geometry digest."""
    _validate_partial_cell_weighting(partial_cell_weighting, spherical_correction=spherical_correction)
    h = hashlib.sha256()
    h.update(grid_digest(lats, lons))
    h.update(b"sph" if spherical_correction else b"flat")
    h.update(geometry_digest)
    dtype = operator_dtype(dtype)
    if dtype != np.float64:
        h.update(b"dtype:" + dtype.name.encode())
    if partial_cell_weighting == "exact":
        h.update(b"partial_cell_weighting:exact")
    return h.digest()
