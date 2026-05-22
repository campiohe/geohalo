import numpy as np
import scipy.sparse as sp
from scipy import ndimage

from geohalo.grid import GridSpec


def refine_grid(grid: GridSpec, factor: int) -> GridSpec:
    if factor < 1:
        raise ValueError(f"factor must be >= 1, got {factor}")
    if factor == 1:
        return grid
    refined_lats = _subdivide_centres(grid.lats, factor)
    refined_lons = _subdivide_centres(grid.lons, factor)
    return GridSpec(lats=refined_lats, lons=refined_lons)


def downscale_plane(data: np.ndarray, factor: int, *, iterations: int = 1) -> np.ndarray:
    # 2-D only by design. Calling scipy.ndimage.zoom on a 3-D array with
    # zoom=(1, k, k) is empirically ~25% slower per cell than calling it
    # on N independent 2-D slices, because scipy's N-D zoom path isn't
    # optimised for the leading zoom=1 case. _apply_downscale loops over
    # batch dims and calls this kernel; the per-iteration Python overhead
    # is negligible vs the in-kernel work.
    #
    # iterations=1 (default, backward-compatible): one bilinear upsample plus
    # one per-parent constant correction. Cheap, but the constant correction
    # produces a visible blocky pattern at parent boundaries when adjacent
    # parents drift in opposite directions.
    #
    # iterations>1: between the upsample and the final hard correction,
    # apply (iterations-1) passes of "compute per-parent drift, bilinearly
    # distribute it across children, add". Each pass smooths the correction
    # field across parent boundaries and shrinks residual drift; the final
    # constant-per-parent step then closes the remaining (small) gap to
    # guarantee exact mean preservation regardless of iteration count.
    if factor < 1:
        raise ValueError(f"factor must be >= 1, got {factor}")
    if iterations < 1:
        raise ValueError(f"iterations must be >= 1, got {iterations}")
    if factor == 1:
        return data
    if data.ndim != 2:
        raise ValueError(f"downscale_plane expects a 2-D array, got ndim={data.ndim}")
    n_lat, n_lon = data.shape
    upsampled = ndimage.zoom(data, factor, order=1, mode="nearest", grid_mode=True)
    for _ in range(iterations - 1):
        upsampled_5d = upsampled.reshape(n_lat, factor, n_lon, factor)
        parent_means = upsampled_5d.mean(axis=(1, 3))
        drift = data - parent_means
        upsampled = upsampled + ndimage.zoom(
            drift, factor, order=1, mode="nearest", grid_mode=True,
        )
    upsampled_5d = upsampled.reshape(n_lat, factor, n_lon, factor)
    parent_means = upsampled_5d.mean(axis=(1, 3))
    corrected_5d = upsampled_5d + (data - parent_means)[:, None, :, None]
    return corrected_5d.reshape(n_lat * factor, n_lon * factor)


def build_downscale_operator(grid: GridSpec, factor: int) -> sp.csr_matrix:
    if factor < 1:
        raise ValueError(f"factor must be >= 1, got {factor}")
    n_lat, n_lon = grid.shape
    n_native = n_lat * n_lon
    if factor == 1:
        return sp.eye(n_native, format="csr")
    b_lat = _build_bilinear_matrix_1d(n_lat, factor)
    b_lon = _build_bilinear_matrix_1d(n_lon, factor)
    b = sp.kron(b_lat, b_lon, format="csr")
    p = _build_parent_matrix(n_lat, n_lon, factor)
    a = (p.T / (factor * factor)).tocsr()
    return (b + p - p @ a @ b).tocsr()


def _build_bilinear_matrix_1d(n: int, factor: int) -> sp.csr_matrix:
    n_out = n * factor
    i = np.arange(n_out)
    pos = np.clip((i + 0.5) / factor - 0.5, 0.0, n - 1)
    lo = np.floor(pos).astype(np.int64)
    hi = np.minimum(lo + 1, n - 1)
    frac = pos - lo
    rows = np.concatenate([i, i])
    cols = np.concatenate([lo, hi])
    data = np.concatenate([1.0 - frac, frac])
    return sp.csr_matrix((data, (rows, cols)), shape=(n_out, n))


def _build_parent_matrix(n_lat: int, n_lon: int, factor: int) -> sp.csr_matrix:
    n_refined = n_lat * factor * n_lon * factor
    n_native = n_lat * n_lon
    rows = np.arange(n_refined)
    refined_lat = rows // (n_lon * factor)
    refined_lon = rows % (n_lon * factor)
    parent_lat = refined_lat // factor
    parent_lon = refined_lon // factor
    cols = parent_lat * n_lon + parent_lon
    data = np.ones(n_refined, dtype=np.float64)
    return sp.csr_matrix((data, (rows, cols)), shape=(n_refined, n_native))


def resolve_factor(grid: GridSpec, target_resolution: float | None) -> tuple[int, float]:
    dlat = float(abs(grid.lats[1] - grid.lats[0]))
    if target_resolution is None:
        return 1, dlat
    if target_resolution <= 0:
        raise ValueError(f"target_resolution must be > 0, got {target_resolution}")
    dlon = float(abs(grid.lons[1] - grid.lons[0]))
    if not np.isclose(dlat, dlon, rtol=1e-3):
        raise ValueError(f"non-square source grid: dlat={dlat}, dlon={dlon}")
    factor = max(1, round(dlat / target_resolution))
    achieved = dlat / factor
    return factor, achieved


def _subdivide_centres(centres: np.ndarray, factor: int) -> np.ndarray:
    step = float(centres[1] - centres[0]) if centres.size > 1 else 1.0
    new_step = step / factor
    offsets = (np.arange(factor) - (factor - 1) / 2) * new_step
    return (centres[:, None] + offsets[None, :]).ravel()
