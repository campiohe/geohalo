"""Stencil: per-geom x per-cell area coverage. EmptyOverlapError lives here."""

import hashlib
from collections.abc import Hashable
from dataclasses import dataclass, field

import geopandas as gpd
import numpy as np
import pandas as pd
import scipy.sparse as sp
import shapely
from exactextract import exact_extract
from exactextract.raster import NumPyRasterSource

from geohalo.geometry import (
    cell_areas,
    ensure_ascending_lats,
    geom_digest,
    grid_digest,
    require_regular_grid,
)


class EmptyOverlapError(Exception):
    def __init__(self, geom_key: Hashable) -> None:
        super().__init__(f"polygon {geom_key!r} does not intersect the grid")
        self.geom_key = geom_key


@dataclass(frozen=True)
class Stencil:
    occupancy_matrix: sp.csr_matrix
    keys: pd.Index
    lats: np.ndarray
    lons: np.ndarray
    digest: bytes
    spherical_correction: bool = True
    row_sums: np.ndarray = field(init=False, repr=False)

    def __post_init__(self) -> None:
        row_sums = np.asarray(self.occupancy_matrix.sum(axis=1)).ravel()
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
    ) -> "Stencil":
        if not isinstance(geoms, gpd.GeoSeries):
            raise TypeError(f"geoms must be a gpd.GeoSeries, got {type(geoms).__name__}")
        if len(geoms) == 0:
            raise ValueError("geoms is empty; need at least one polygon to build a stencil")
        lats_asc, _ = ensure_ascending_lats(lats)
        lons_arr = np.asarray(lons, dtype=np.float64)
        require_regular_grid(lats_asc, "latitude")
        require_regular_grid(lons_arr, "longitude")

        order = np.argsort([repr(k) for k in geoms.index])
        sorted_geoms = geoms.iloc[order]

        matrix = _build_occupancy_matrix(
            lats_asc, lons_arr, sorted_geoms, spherical_correction=spherical_correction,
        )
        digest = stencil_digest(lats_asc, lons_arr, geoms, spherical_correction=spherical_correction)
        return cls(
            occupancy_matrix=matrix,
            keys=sorted_geoms.index,
            lats=lats_asc,
            lons=lons_arr,
            digest=digest,
            spherical_correction=spherical_correction,
        )


def _build_occupancy_matrix(
    lats: np.ndarray,
    lons: np.ndarray,
    geoms: gpd.GeoSeries,
    *,
    spherical_correction: bool,
) -> sp.csr_matrix:
    n_lat, n_lon = lats.size, lons.size
    template = np.zeros((n_lat, n_lon), dtype=np.float64)
    xmin = float(lons[0] - (lons[1] - lons[0]) / 2)
    xmax = float(lons[-1] + (lons[-1] - lons[-2]) / 2)
    ymin = float(lats[0] - (lats[1] - lats[0]) / 2)
    ymax = float(lats[-1] + (lats[-1] - lats[-2]) / 2)
    src = NumPyRasterSource(template, xmin=xmin, ymin=ymin, xmax=xmax, ymax=ymax)
    features = [
        {"type": "Feature", "geometry": shapely.geometry.mapping(g), "properties": {"i": i}}
        for i, g in enumerate(geoms.to_numpy())
    ]
    df = exact_extract(src, features, ops=["cell_id", "coverage"], output="pandas", include_cols=[])

    areas = cell_areas(lats, lons, spherical=spherical_correction)
    rows: list[np.ndarray] = []
    cols: list[np.ndarray] = []
    data: list[np.ndarray] = []
    for i, key in enumerate(geoms.index):
        cell_ids = np.asarray(df.iloc[i]["cell_id"], dtype=np.int64)
        coverage = np.asarray(df.iloc[i]["coverage"], dtype=np.float64)
        if cell_ids.size == 0:
            raise EmptyOverlapError(key)
        row_top = cell_ids // n_lon
        col = cell_ids % n_lon
        row_asc = n_lat - 1 - row_top
        weight = coverage * areas[row_asc, col]
        if weight.sum() <= 0:
            raise EmptyOverlapError(key)
        rows.append(np.full(cell_ids.size, i, dtype=np.int64))
        cols.append(row_asc * n_lon + col)
        data.append(weight)

    return sp.csr_matrix(
        (np.concatenate(data), (np.concatenate(rows), np.concatenate(cols))),
        shape=(len(geoms), n_lat * n_lon),
    )


def stencil_digest(
    lats: np.ndarray,
    lons: np.ndarray,
    geoms: gpd.GeoSeries,
    *,
    spherical_correction: bool = True,
) -> bytes:
    """Cache key for a stencil, derivable from inputs without building it.

    Canonicalises latitudes to ascending so a grid and its flipped twin hash
    identically; ``geom_digest`` is order-invariant, so geometry order does not
    matter either.
    """
    lats_asc, _ = ensure_ascending_lats(lats)
    lons_arr = np.asarray(lons, dtype=np.float64)
    h = hashlib.sha256()
    h.update(grid_digest(lats_asc, lons_arr))
    h.update(b"sph" if spherical_correction else b"flat")
    h.update(geom_digest(geoms))
    return h.digest()
