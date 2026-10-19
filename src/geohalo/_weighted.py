"""A reduce operator weighted by counts on a fine lattice whose pixels nest in the source cells."""

import hashlib
from dataclasses import dataclass

import geopandas as gpd
import numpy as np
import scipy.sparse as sp
import shapely
from exactextract import exact_extract
from exactextract.raster import NumPyRasterSource

from geohalo.geometry import COORD_TOL, midpoint_edges, require_regular_grid
from geohalo.stencil import EmptyOverlapError, _WKBFeatureSource

FEATURES_PER_EXTRACT = 16


@dataclass(frozen=True)
class Window:
    """Pixel centres north to south and west to east, their counts in that order, and the window's edges."""

    lat: np.ndarray
    lon: np.ndarray
    counts: np.ndarray
    north: float
    south: float
    west: float
    east: float

    @property
    def lat_step(self) -> float:
        return (self.north - self.south) / self.lat.size

    @property
    def lon_step(self) -> float:
        return (self.east - self.west) / self.lon.size


def real_array(values: np.ndarray, name: str) -> np.ndarray:
    """``values`` as an array of real numbers; a masked array's masked entries become NaN."""
    array = np.asarray(values)
    if array.dtype.kind not in "fiu":
        raise TypeError(f"{name} must be real numbers, got {array.dtype}")
    mask = np.ma.getmask(values)
    if mask is not np.ma.nomask and mask.any():
        array = np.where(mask, np.nan, array if array.dtype.kind == "f" else array.astype(np.float64))
    return array


def weight_window(weight_lat: np.ndarray, weight_lon: np.ndarray, weights: np.ndarray) -> Window:
    """Validate a window of counts and orient it north to south, as exactextract reads a raster.

    Float32 and float64 counts already north to south are not copied. A masked array's masked counts are NaN.
    """
    lat = np.asarray(weight_lat, dtype=np.float64)
    lon = np.asarray(weight_lon, dtype=np.float64)
    if lat.ndim != 1 or lon.ndim != 1 or lat.size < 2 or lon.size < 2:
        raise ValueError("weight_lat and weight_lon must be one-dimensional with at least two pixels each")
    counts = real_array(weights, "weights")
    if counts.shape != (lat.size, lon.size):
        raise ValueError(f"weights must have shape {(lat.size, lon.size)}, got {counts.shape}")
    if counts.dtype not in (np.dtype(np.float32), np.dtype(np.float64)):
        counts = counts.astype(np.float64)
    if np.isinf(counts).any() or (counts < 0).any():
        raise ValueError("weights must not be negative or infinite; NaN is read as 0")
    if lat[0] < lat[-1]:
        lat, counts = lat[::-1], counts[::-1]
    if lon[0] > lon[-1]:
        raise ValueError("weight_lon must ascend")
    require_regular_grid(lat, "weight latitude")
    require_regular_grid(lon, "weight longitude")
    dlat = (lat[0] - lat[-1]) / (lat.size - 1)
    dlon = (lon[-1] - lon[0]) / (lon.size - 1)
    return Window(
        lat=lat, lon=lon, counts=np.ascontiguousarray(counts),
        north=float(lat[0] + dlat / 2), south=float(lat[-1] - dlat / 2),
        west=float(lon[0] - dlon / 2), east=float(lon[-1] + dlon / 2),
    )


def periodic_shifts(edges: np.ndarray, start: float, stop: float, period: float | None) -> np.ndarray:
    """Multiples of ``period`` that move ``[edges[0], edges[-1]]`` onto ``[start, stop]``; ``[0]`` without one."""
    if period is None:
        return np.zeros(1)
    first = np.ceil((start - edges[-1]) / period)
    last = np.floor((stop - edges[0]) / period)
    return np.arange(first, last + 1) * period


NESTING_TOL = 1e-3


def require_nested(cell_edges: np.ndarray, start: float, step: float, stop: float, axis: str) -> None:
    """Every cell edge strictly inside ``(start, stop)`` must be a pixel edge ``start + n * step``.

    An edge may be off by ``NESTING_TOL * step``: a pixel is assigned by its centre, half a step from either edge.
    """
    tol = NESTING_TOL * step
    inner = cell_edges[(cell_edges > start + tol) & (cell_edges < stop - tol)]
    off = np.abs(inner - (start + np.round((inner - start) / step) * step))
    stray = inner[off > tol]
    if stray.size:
        raise ValueError(f"the source {axis} edge {stray[0]:.9f} is not a pixel edge of the weight window")


def require_inside(
    geometries: gpd.GeoSeries, window: Window, lat_edges: np.ndarray, lon_edges: np.ndarray, period: float | None,
) -> None:
    """Each geometry lies in the window and overlaps the source grid, or a copy of it ``period`` apart, there."""
    shapes = geometries.to_numpy()
    minx, miny, maxx, maxy = shapely.bounds(shapes).T
    leaves = (
        (minx < window.west - COORD_TOL) | (miny < window.south - COORD_TOL)
        | (maxx > window.east + COORD_TOL) | (maxy > window.north + COORD_TOL)
    )
    if leaves.any():
        raise ValueError(f"geometry {geometries.index[np.flatnonzero(leaves)[0]]!r} leaves the weight window")
    south, north = max(window.south, lat_edges[0]), min(window.north, lat_edges[-1])
    overlap = np.zeros(len(shapes))
    for shift in periodic_shifts(lon_edges, window.west, window.east, period):
        west, east = max(window.west, lon_edges[0] + shift), min(window.east, lon_edges[-1] + shift)
        if west < east and south < north:
            overlap += shapely.area(shapely.clip_by_rect(shapes, west, south, east, north))
    missed = np.flatnonzero(~(overlap > 0))
    if missed.size:
        raise EmptyOverlapError(geometries.index[missed[0]])


def weighted_matrix(
    geometries: gpd.GeoSeries, window: Window, source_lat: np.ndarray, source_lon: np.ndarray, period: float | None,
) -> sp.csr_matrix:
    """``(n_geometries, n_source)``: the coverage of each pixel times its count, summed into its source cell.

    ``source_lat`` ascends. A pixel belongs to the cell its centre is in, its longitude first moved by a multiple
    of ``period`` into ``[west edge, west edge + period)`` when one is given. Pixels with no count never leave
    exactextract (``nodata=0``; NaN is skipped too). Pixels outside the source grid are dropped. Rows keep the
    caller's order; columns ascend within a row.
    """
    lat_edges, lon_edges = midpoint_edges(source_lat), midpoint_edges(source_lon)
    lon = window.lon if period is None else lon_edges[0] + np.mod(window.lon - lon_edges[0], period)
    cell_row = np.searchsorted(lat_edges, window.lat, side="right") - 1
    cell_col = np.searchsorted(lon_edges, lon, side="right") - 1
    n_lat, n_lon = source_lat.size, source_lon.size
    raster = NumPyRasterSource(
        window.counts, xmin=window.west, ymin=window.south, xmax=window.east, ymax=window.north, nodata=0,
    )
    wkb = shapely.to_wkb(geometries.to_numpy())
    data: list[np.ndarray] = []
    indices: list[np.ndarray] = []
    counts = np.zeros(len(wkb), dtype=np.int64)
    for start in range(0, len(wkb), FEATURES_PER_EXTRACT):
        frame = exact_extract(
            raster, _WKBFeatureSource(wkb[start:start + FEATURES_PER_EXTRACT]),
            ops=["cell_id", "coverage", "values"], output="pandas", include_cols=[],
        )
        for offset, (pixel, coverage, count) in enumerate(
            zip(frame["cell_id"], frame["coverage"], frame["values"], strict=True),
        ):
            # exactextract numbers pixels row-major from the raster's top (northern) row.
            pixel_row, pixel_col = np.divmod(np.asarray(pixel, dtype=np.int64), window.lon.size)
            rows, cols = cell_row[pixel_row], cell_col[pixel_col]
            inside = (rows >= 0) & (rows < n_lat) & (cols >= 0) & (cols < n_lon)
            if not inside.any():
                continue
            rows, cols = rows[inside], cols[inside]
            people = np.asarray(coverage, dtype=np.float64)[inside] * np.asarray(count, dtype=np.float64)[inside]
            row0, col0 = rows.min(), cols.min()
            width = cols.max() - col0 + 1
            sums = np.bincount((rows - row0) * width + (cols - col0), weights=people)
            local = np.flatnonzero(sums)
            data.append(sums[local])
            indices.append((local // width + row0) * n_lon + local % width + col0)
            counts[start + offset] = local.size
        del frame
    return sp.csr_matrix(
        (
            np.concatenate(data) if data else np.zeros(0, dtype=np.float64),
            np.concatenate(indices) if indices else np.zeros(0, dtype=np.int64),
            np.r_[0, np.cumsum(counts)],
        ),
        shape=(len(wkb), n_lat * n_lon),
    )


def counts_digest(window: Window) -> bytes:
    """SHA-256 of the counts south to north as float64, NaN and -0.0 read as 0, one row at a time."""
    h = hashlib.sha256()
    for row in window.counts[::-1]:
        h.update((np.nan_to_num(row, nan=0.0).astype(np.float64) + 0.0).tobytes())
    return h.digest()
