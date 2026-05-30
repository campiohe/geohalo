"""Pure grid math: edges, cell areas, digests, coord generation, resample building blocks."""

import hashlib

import geopandas as gpd
import numpy as np
import scipy.sparse as sp
import shapely


def midpoint_edges(centres: np.ndarray) -> np.ndarray:
    """Cell edges from centres; size N+1 for N centres."""
    if centres.size < 2:
        raise ValueError("need >= 2 coordinates to derive edges")
    mids = (centres[:-1] + centres[1:]) / 2.0
    first = centres[0] - (mids[0] - centres[0])
    last = centres[-1] + (centres[-1] - mids[-1])
    return np.concatenate([[first], mids, [last]])


def ensure_ascending_lats(lats: np.ndarray) -> tuple[np.ndarray, bool]:
    """Return (ascending_lats, was_descending). Copies on flip."""
    lats = np.asarray(lats, dtype=np.float64)
    if lats.size > 1 and lats[0] > lats[-1]:
        return lats[::-1].copy(), True
    return lats, False


COORD_TOL = 1e-9


def same_grid(lat_a: np.ndarray, lon_a: np.ndarray, lat_b: np.ndarray, lon_b: np.ndarray) -> bool:
    """True if two lat/lon grids match in shape and coordinates (within COORD_TOL)."""
    return bool(
        lat_a.size == lat_b.size
        and lon_a.size == lon_b.size
        and np.allclose(lat_a, lat_b, atol=COORD_TOL)
        and np.allclose(lon_a, lon_b, atol=COORD_TOL),
    )


def require_regular_grid(coords: np.ndarray, name: str) -> None:
    """Raise if `coords` is not regularly spaced.

    exactextract's raster model (and the stencil's bounding-box math) assume a
    uniform grid; an irregular axis would silently misplace coverage fractions.
    """
    coords = np.asarray(coords, dtype=np.float64)
    if coords.size < 2:
        return
    diffs = np.diff(coords)
    step = float(diffs.mean())
    if step == 0 or not np.allclose(diffs, step, rtol=1e-4, atol=1e-9):
        raise ValueError(
            f"{name} grid must be regularly spaced (geohalo assumes a uniform EPSG:4326 raster); "
            "got irregular spacing",
        )


EARTH_RADIUS_M = 6_371_008.8


def cell_areas(lats: np.ndarray, lons: np.ndarray, *, spherical: bool = True) -> np.ndarray:
    """Per-cell area. spherical=True → m²; spherical=False → all 1.0."""
    lats = np.asarray(lats, dtype=np.float64)
    lons = np.asarray(lons, dtype=np.float64)
    if not spherical:
        return np.ones((lats.size, lons.size), dtype=np.float64)
    lat_edges = midpoint_edges(lats)
    lon_edges = midpoint_edges(lons)
    sin_top = np.sin(np.deg2rad(lat_edges[1:]))
    sin_bot = np.sin(np.deg2rad(lat_edges[:-1]))
    dlon_rad = np.deg2rad(np.diff(lon_edges))
    area_per_lat = (EARTH_RADIUS_M**2) * (sin_top - sin_bot)
    return area_per_lat[:, None] * dlon_rad[None, :]


def grid_digest(lats: np.ndarray, lons: np.ndarray) -> bytes:
    """SHA-256 over canonical lat/lon bytes + EPSG tag (no spherical flag)."""
    h = hashlib.sha256()
    h.update(np.asarray(lats, dtype=np.float64).tobytes())
    h.update(np.asarray(lons, dtype=np.float64).tobytes())
    h.update(b"epsg:4326")
    return h.digest()


def geom_digest(geoms: gpd.GeoSeries) -> bytes:
    """SHA-256 of sorted-key (repr(key), WKB(geom)) pairs. Order-invariant."""
    order = np.argsort([repr(k) for k in geoms.index])
    sorted_geoms = geoms.iloc[order]
    h = hashlib.sha256()
    h.update(repr(tuple(geoms.index.names)).encode())
    for key, geom in zip(sorted_geoms.index, sorted_geoms.to_numpy(), strict=True):
        h.update(repr(key).encode())
        h.update(shapely.to_wkb(geom))
    return h.digest()


def target_coords_from_resolution(
    source_lat: np.ndarray,
    source_lon: np.ndarray,
    target_resolution: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Target centres from source extent + a target step, via arange.

    Spans [min, max] of each source coord at the requested step. Works for
    refine (smaller step) or coarsen (larger step).
    """
    if target_resolution <= 0:
        raise ValueError(f"target_resolution must be > 0, got {target_resolution}")
    source_lat = np.asarray(source_lat, dtype=np.float64)
    source_lon = np.asarray(source_lon, dtype=np.float64)
    tlat = np.arange(source_lat.min(), source_lat.max() + target_resolution / 2, target_resolution)
    tlon = np.arange(source_lon.min(), source_lon.max() + target_resolution / 2, target_resolution)
    return tlat, tlon


def bilinear_matrix_1d(source: np.ndarray, target: np.ndarray) -> sp.csr_matrix:
    """(n_target, n_source) 1-D linear interpolation matrix with edge clamping.

    Handles a descending `source` array (e.g. ECMWF latitudes) by sorting it
    ascending for the lookup and mapping the column indices back to the
    caller's ordering. `target` may be in any order — each row is independent.
    """
    source = np.asarray(source, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    n_s, n_t = source.size, target.size
    if n_s == 1:
        # A single source cell is constant along this axis: every target reads it.
        return sp.csr_matrix((np.ones(n_t), (np.arange(n_t), np.zeros(n_t, dtype=np.int64))), shape=(n_t, 1))
    ascending = source[0] <= source[-1]
    src = source if ascending else source[::-1]
    idx = np.clip(np.searchsorted(src, target) - 1, 0, n_s - 2)
    x0 = src[idx]
    x1 = src[idx + 1]
    frac = np.clip((target - x0) / (x1 - x0), 0.0, 1.0)
    col_lo, col_hi = idx, idx + 1
    if not ascending:
        col_lo, col_hi = n_s - 1 - idx, n_s - 1 - (idx + 1)
    rows = np.concatenate([np.arange(n_t), np.arange(n_t)])
    cols = np.concatenate([col_lo, col_hi])
    data = np.concatenate([1.0 - frac, frac])
    m = sp.csr_matrix((data, (rows, cols)), shape=(n_t, n_s))
    m.sum_duplicates()
    return m


def nearest_index(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Index of the nearest source centre for each target centre (ties → lower).

    Handles a descending `source` array by sorting ascending for the lookup
    and mapping indices back to the caller's ordering.
    """
    source = np.asarray(source, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    n_s = source.size
    if n_s == 1:
        return np.zeros(target.size, dtype=np.int64)
    ascending = source[0] <= source[-1]
    src = source if ascending else source[::-1]
    pos = np.clip(np.searchsorted(src, target), 1, n_s - 1)
    left = src[pos - 1]
    right = src[pos]
    choose_left = (target - left) <= (right - target)
    idx = np.where(choose_left, pos - 1, pos)
    return idx if ascending else n_s - 1 - idx
