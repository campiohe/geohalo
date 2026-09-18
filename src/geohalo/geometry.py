"""Pure grid math: edges, cell areas, digests, coord generation, resample building blocks."""

import hashlib
from collections.abc import Sequence
from math import fsum

import geopandas as gpd
import numpy as np
import pandas as pd
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


def polygon_areas(geoms: gpd.GeoSeries | Sequence[shapely.Geometry | None] | np.ndarray) -> np.ndarray:
    """Spherical areas in m², using the same radius as :func:`cell_areas`.

    Accept a GeoSeries or a one-dimensional sequence of Shapely geometries in
    longitude/latitude degrees. Return a float64 array in input order, without
    index labels. Missing geometries return NaN; empty geometries and non-area
    geometries return zero. Polygon holes are subtracted and multipart or
    collection areas are added (not unioned). Z coordinates are ignored.

    Edges are straight segments in the supplied lon/lat coordinates, not
    geodesics. Integration is analytic, including sloped edges. Longitudes
    are NOT wrapped: a box from -170 to 170 spans 340 degrees, while a narrow
    seam-crossing box can use 170 to 190, or be split at the antimeridian.
    Each polygon must span at most 360 degrees. Full-globe boxes are supported.

    Coordinates must be finite, latitudes within [-90, 90], and geometries
    topologically valid. Invalid inputs raise ValueError; non-geometries raise
    TypeError. A GeoSeries CRS, when set, must be EPSG:4326 (or equivalent).
    No reprojection, repair, clipping, or longitude unwrapping is performed.

    Default stencil weights use planar cell-coverage fractions times spherical
    cell areas, approximating this area in partially covered cells. Stencils
    with ``partial_cell_weighting="exact"`` integrate their intersections with
    this same model. For planar areas, use Shapely's ``area``.
    """
    if isinstance(geoms, gpd.GeoSeries):
        if geoms.crs is not None and not geoms.crs.equals("EPSG:4326", ignore_axis_order=True):
            raise ValueError("polygon_areas requires longitude/latitude degrees (EPSG:4326)")
        geoms = geoms.to_numpy()
    geometries = np.asarray(geoms, dtype=object)
    if geometries.ndim != 1:
        raise ValueError("geoms must be a one-dimensional sequence; wrap a single geometry in a list")
    result = np.empty(len(geometries), dtype=np.float64)
    for i, geom in enumerate(geometries):
        if geom is None:
            result[i] = np.nan
            continue
        if not isinstance(geom, shapely.Geometry):
            raise TypeError(f"geoms[{i}] must be a Shapely geometry or None")
        coords = shapely.get_coordinates(geom)
        if not np.isfinite(coords).all() or np.any(np.abs(coords[:, 1]) > 90):
            raise ValueError(f"geoms[{i}] requires finite coordinates and latitudes within [-90, 90]")
        if not geom.is_valid:
            raise ValueError(f"geoms[{i}] is invalid: {shapely.is_valid_reason(geom)}")
        result[i] = EARTH_RADIUS_M**2 * _polygon_area_steradians(geom)
    return result


def _polygon_area_steradians(geometry: shapely.Geometry) -> float:
    """Accumulate polygon components without depending on ring orientation."""
    pending = [geometry]
    areas = []
    while pending:
        part = pending.pop()
        if part.is_empty:
            continue
        if isinstance(part, shapely.Polygon):
            if part.bounds[2] - part.bounds[0] > 360:
                raise ValueError("each polygon must span at most 360 degrees of longitude")
            outer = abs(_lonlat_ring_integral(np.asarray(part.exterior.coords)))
            holes = fsum(abs(_lonlat_ring_integral(np.asarray(ring.coords))) for ring in part.interiors)
            areas.append(max(0.0, outer - holes))
        elif isinstance(part, (shapely.MultiPolygon, shapely.GeometryCollection)):
            pending.extend(part.geoms)
    return fsum(areas)


def _lonlat_ring_integral(coords: np.ndarray) -> float:
    """Integrate sin(latitude) d(longitude) exactly along linear lon/lat edges.

    For an edge with midpoint m and half latitude difference h, its mean sine
    is sin(m) * sinc(h/pi). Subtracting a constant reference sine leaves the
    closed-ring integral unchanged and avoids cancellation for small polygons.
    """
    latitudes = coords[:, 1]
    reference = (latitudes.min() + latitudes.max()) / 2
    offsets = np.deg2rad(latitudes - reference)
    middle = (offsets[:-1] + offsets[1:]) / 2
    half_step = np.deg2rad(np.diff(latitudes) / 2)
    reference = np.deg2rad(reference)
    # sinc(h/pi) - 1 loses its quadratic term for short edges. A local Taylor
    # expansion retains it, including for tiny triangles near a pole.
    h2 = half_step**2
    correction = np.where(
        np.abs(half_step) < 1e-3,
        h2 * (-1 / 6 + h2 * (1 / 120 - h2 / 5040)),
        np.sinc(half_step / np.pi) - 1,
    )
    mean_sine = (
        2 * np.cos(reference + middle / 2) * np.sin(middle / 2)
        + np.sin(reference + middle) * correction
    )
    return fsum(np.deg2rad(np.diff(coords[:, 0])) * mean_sine)


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
    return _geom_digest_from_wkb(sorted_geoms.index, shapely.to_wkb(sorted_geoms.to_numpy()))


def _geom_digest_from_wkb(keys: pd.Index, wkb: np.ndarray) -> bytes:
    """Hash keys already sorted by repr and their matching WKB bytes."""
    h = hashlib.sha256()
    h.update(repr(tuple(keys.names)).encode())
    for key, encoded in zip(keys, wkb, strict=True):
        h.update(repr(key).encode())
        h.update(encoded)
    return h.digest()


def target_coords_from_resolution(
    source_lat: np.ndarray,
    source_lon: np.ndarray,
    target_resolution: float,
    *,
    period: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Target centres from source extent + a target step, via arange.

    Spans [min, max] of each source coord at the requested step. Works for
    refine (smaller step) or coarsen (larger step). With ``period``, longitude
    instead spans one full cycle starting at its minimum, excluding the repeated
    endpoint. Latitude is never periodic.
    """
    if target_resolution <= 0:
        raise ValueError(f"target_resolution must be > 0, got {target_resolution}")
    source_lat = np.asarray(source_lat, dtype=np.float64)
    source_lon = np.asarray(source_lon, dtype=np.float64)
    period = _validate_period(source_lon, period)
    tlat = np.arange(source_lat.min(), source_lat.max() + target_resolution / 2, target_resolution)
    tlon = (
        np.arange(source_lon.min(), source_lon.max() + target_resolution / 2, target_resolution)
        if period is None else source_lon.min() + np.arange(0.0, period, target_resolution)
    )
    return tlat, tlon


def _validate_period(source: np.ndarray, period: float | None) -> float | None:
    """Normalize an optional period and validate its non-repeated source cycle."""
    if period is None:
        return None
    period = float(period)
    if not np.isfinite(period) or period <= 0:
        raise ValueError("period must be finite and > 0")
    source = np.asarray(source, dtype=np.float64)
    if source.ndim != 1 or source.size == 0 or not np.isfinite(source).all():
        raise ValueError("periodic source coordinates must be a nonempty finite 1-D array")
    diffs = np.diff(source)
    if not (np.all(diffs > 0) or np.all(diffs < 0)):
        raise ValueError("periodic source coordinates must be strictly monotonic")
    if abs(source[-1] - source[0]) >= period:
        raise ValueError("periodic source coordinates must span less than period; omit the repeated endpoint")
    return period


def _periodic_brackets(
    source: np.ndarray, target: np.ndarray, period: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Locate cyclic neighbours without copying three cycles of the source axis."""
    period = _validate_period(source, period)
    if target.ndim != 1 or not np.isfinite(target).all():
        raise ValueError("periodic target coordinates must be a finite 1-D array")
    ascending = source[0] <= source[-1]
    src = source if ascending else source[::-1]
    # A single extra centre covers the seam. Modulo also handles targets many
    # periods away; only column indices are folded, never the caller's coords.
    extended = np.append(src, src[0] + period)
    wrapped = src[0] + np.remainder(target - src[0], period)
    lo = np.minimum(np.searchsorted(src, wrapped, side="right") - 1, src.size - 1)
    frac = (wrapped - extended[lo]) / (extended[lo + 1] - extended[lo])
    hi = (lo + 1) % src.size
    if not ascending:
        lo, hi = src.size - 1 - lo, src.size - 1 - hi
    return lo, hi, frac


def bilinear_matrix_1d(
    source: np.ndarray, target: np.ndarray, *, period: float | None = None,
) -> sp.csr_matrix:
    """(n_target, n_source) 1-D linear interpolation, clamped unless periodic.

    Handles a descending `source` array (e.g. ECMWF latitudes) by sorting it
    ascending for the lookup and mapping the column indices back to the
    caller's ordering. `target` may be in any order — each row is independent.
    ``period`` wraps across the seam (e.g. 360 for longitude). Periodic sources
    must be finite, strictly monotonic, and span less than a positive finite
    period: do not include both endpoints of the same cycle. Targets may lie
    any number of cycles away. A single source cell is constant everywhere.
    """
    source = np.asarray(source, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    n_s, n_t = source.size, target.size
    if period is not None:
        lo, hi, frac = _periodic_brackets(source, target, period)
        return sp.csr_matrix(
            (np.r_[1.0 - frac, frac], (np.tile(np.arange(n_t), 2), np.r_[lo, hi])), shape=(n_t, n_s),
        )
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


def nearest_index(source: np.ndarray, target: np.ndarray, *, period: float | None = None) -> np.ndarray:
    """Index of the nearest source centre for each target centre (ties → lower).

    Handles a descending `source` array by sorting ascending for the lookup
    and mapping indices back to the caller's ordering. ``period`` uses the same
    cyclic neighbours and validation as ``bilinear_matrix_1d``. Seam ties choose
    the lower *unwrapped* neighbour (the last centre of the ascending cycle).
    """
    source = np.asarray(source, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    if period is not None:
        lo, hi, frac = _periodic_brackets(source, target, period)
        return np.where(frac <= 0.5, lo, hi)
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


def _conservative_centres(
    centres: np.ndarray, *, latitude: bool, period: float | None,
) -> tuple[np.ndarray, bool]:
    """Validate centres and return them ascending, with their orientation flag."""
    centres = np.asarray(centres, dtype=np.float64)
    if centres.ndim != 1 or centres.size == 0 or not np.isfinite(centres).all():
        raise ValueError("conservative coordinates must be a nonempty finite 1-D array")
    diffs = np.diff(centres)
    if not (np.all(diffs > 0) or np.all(diffs < 0)):
        raise ValueError("conservative coordinates must be strictly monotonic")
    descending = centres.size > 1 and centres[0] > centres[-1]
    ascending = centres[::-1] if descending else centres
    if latitude and np.any(np.abs(ascending) > 90):
        raise ValueError("latitude centres must lie within [-90, 90]")
    if period is not None:
        _validate_period(ascending, period)
    return ascending, descending


def _conservative_edges(
    centres: np.ndarray, edges: np.ndarray | None, *, latitude: bool, period: float | None, cyclic: bool,
) -> tuple[np.ndarray, bool]:
    """Ascending cell edges and whether the caller's axis was descending."""
    ascending, descending = _conservative_centres(centres, latitude=latitude, period=period)
    if edges is None:
        if cyclic:
            # Midpoints across the seam close a full source cycle, including
            # irregular grids and a single cell covering the entire period.
            first = (ascending[-1] - period + ascending[0]) / 2
            result = np.r_[first, (ascending[:-1] + ascending[1:]) / 2, first + period]
        else:
            if ascending.size < 2:
                raise ValueError("single-cell conservative axes require explicit bounds")
            result = midpoint_edges(ascending)
        result = np.clip(result, -90., 90.) if latitude else result
    else:
        result = np.asarray(edges, dtype=np.float64)
        if result.shape != (ascending.size + 1,) or not np.isfinite(result).all():
            raise ValueError("cell bounds must be a finite 1-D array of length coordinates.size + 1")
        result = result[::-1] if descending else result
        if latitude and np.any(np.abs(result) > 90):
            raise ValueError("latitude bounds must lie within [-90, 90]")
    if not np.all(np.diff(result) > 0):
        raise ValueError("cell bounds must be strictly monotonic in the coordinate order")
    if np.any(ascending < result[:-1]) or np.any(ascending > result[1:]):
        raise ValueError("each coordinate must lie within its cell bounds")
    if period is not None:
        span = result[-1] - result[0]
        if cyclic and not np.isclose(span, period, rtol=1e-12, atol=0):
            raise ValueError("periodic source bounds must cover exactly one period")
        if not cyclic and span > period and not np.isclose(span, period, rtol=1e-12, atol=0):
            raise ValueError("periodic target bounds must span no more than one period")
    return result, descending


def conservative_matrix_1d(
    source: np.ndarray,
    target: np.ndarray,
    *,
    latitude: bool = False,
    period: float | None = None,
    source_bounds: np.ndarray | None = None,
    target_bounds: np.ndarray | None = None,
) -> sp.csr_matrix:
    """Cell overlaps divided by full target-cell widths, as a sparse 1-D matrix.

    ``latitude=True`` measures widths in sin(latitude), otherwise in coordinate
    units (longitude degrees). Latitude edges inferred from centres are clipped
    at the poles. Explicit bounds must follow their coordinate order and have
    one extra entry; they allow single-cell axes. Coordinates must be finite
    and strictly monotonic, in either direction. Without bounds, midpoint edges
    are inferred, requiring at least two centres on noncyclic axes.

    ``period`` wraps longitude overlaps only. Source bounds must cover one full
    cycle; inferred source bounds use cyclic midpoints. Target bounds remain
    regional midpoint edges unless supplied, and may span at most one period.
    Empty-overlap rows are zero, partial-overlap rows sum to their covered
    fraction, and only strictly positive overlaps are stored (no 0 * NaN).
    """
    if latitude and period is not None:
        raise ValueError("latitude cannot be periodic")
    period = _validate_period(source, period)
    src, reverse_source = _conservative_edges(
        source, source_bounds, latitude=latitude, period=period, cyclic=period is not None,
    )
    dst, reverse_target = _conservative_edges(
        target, target_bounds, latitude=latitude, period=period, cyclic=False,
    )
    if latitude:
        src, dst = np.sin(np.deg2rad(src)), np.sin(np.deg2rad(dst))
    if np.any(np.diff(src) <= 0) or np.any(np.diff(dst) <= 0):
        raise ValueError("cell bounds must have positive width in the overlap measure")
    widths = np.diff(dst)
    rows, columns, data = [], [], []
    n_source, n_target = src.size - 1, dst.size - 1
    for row, (lower, upper, width) in enumerate(zip(dst[:-1], dst[1:], widths, strict=True)):
        intervals = [(lower, upper)]
        if period is not None:
            left = src[0] + (lower - src[0]) % period
            right = left + width
            intervals = [(left, min(right, src[-1]))]
            if right > src[-1]:
                intervals.append((src[0], src[0] + right - src[-1]))
        for left, right in intervals:
            start = max(0, int(np.searchsorted(src, left, side="right")) - 1)
            stop = min(n_source, int(np.searchsorted(src, right, side="left")))
            col = np.arange(start, stop)
            overlap = np.minimum(src[col + 1], right) - np.maximum(src[col], left)
            keep = overlap > 0
            col, overlap = col[keep], overlap[keep]
            rows.extend([n_target - 1 - row if reverse_target else row] * col.size)
            columns.extend(n_source - 1 - col if reverse_source else col)
            data.extend(overlap / width)
    return sp.csr_matrix((data, (rows, columns)), shape=(n_target, n_source))
