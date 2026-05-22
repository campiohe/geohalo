from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp
import shapely
from exactextract import exact_extract
from exactextract.raster import NumPyRasterSource

from geohalo.downscale import build_downscale_operator, refine_grid, resolve_factor
from geohalo.geometry import PolygonSet
from geohalo.grid import GridSpec


class EmptyCoverageError(Exception):
    """A polygon produced no overlapping cells. Raised eagerly to fail fast."""


@dataclass(frozen=True)
class Weights:
    matrix: sp.csr_matrix
    polygon_keys: list[tuple]
    key_names: tuple[str, ...]
    native_shape: tuple[int, int]
    grid_digest: bytes
    polyset_digest: bytes
    downscale_factor: int = 1
    target_resolution: float | None = None
    achieved_resolution: float = 1.0

    def __repr__(self) -> str:
        return (
            f"Weights(polygons={len(self.polygon_keys)}, "
            f"key_names={self.key_names}, "
            f"native_shape={self.native_shape}, "
            f"factor={self.downscale_factor}, "
            f"achieved_res={self.achieved_resolution})"
        )


def compute_weights(
    polygons: PolygonSet,
    grid: GridSpec,
    *,
    target_resolution: float | None = None,
) -> Weights:
    factor, achieved_resolution = resolve_factor(grid, target_resolution)
    native_shape = grid.shape
    native_grid_digest = grid.digest

    effective_grid = grid if factor == 1 else refine_grid(grid, factor)
    matrix_refined = _exact_extract_matrix(polygons, effective_grid)

    if factor > 1:
        operator = build_downscale_operator(grid, factor)
        matrix = (matrix_refined @ operator).tocsr()
    else:
        matrix = matrix_refined

    return Weights(
        matrix=matrix,
        polygon_keys=list(polygons.keys),
        key_names=polygons.key_names,
        native_shape=native_shape,
        grid_digest=native_grid_digest,
        polyset_digest=polygons.digest,
        downscale_factor=factor,
        target_resolution=target_resolution,
        achieved_resolution=achieved_resolution,
    )


def _exact_extract_matrix(polygons: PolygonSet, grid: GridSpec) -> sp.csr_matrix:
    n_lat, n_lon = grid.shape
    template = np.zeros((n_lat, n_lon), dtype=np.float64)
    xmin = float(grid.lons[0] - (grid.lons[1] - grid.lons[0]) / 2)
    xmax = float(grid.lons[-1] + (grid.lons[-1] - grid.lons[-2]) / 2)
    ymin = float(grid.lats[0] - (grid.lats[1] - grid.lats[0]) / 2)
    ymax = float(grid.lats[-1] + (grid.lats[-1] - grid.lats[-2]) / 2)
    src = NumPyRasterSource(template, xmin=xmin, ymin=ymin, xmax=xmax, ymax=ymax)

    features = [
        {"type": "Feature", "geometry": shapely.geometry.mapping(g), "properties": {"i": i}}
        for i, g in enumerate(polygons.geoms)
    ]
    df = exact_extract(src, features, ops=["cell_id", "coverage"], output="pandas", include_cols=[])

    rows: list[np.ndarray] = []
    cols: list[np.ndarray] = []
    data: list[np.ndarray] = []
    for i, key in enumerate(polygons.keys):
        cell_ids = np.asarray(df.iloc[i]["cell_id"], dtype=np.int64)
        coverage = np.asarray(df.iloc[i]["coverage"], dtype=np.float64)
        if cell_ids.size == 0:
            raise EmptyCoverageError(f"polygon {key} does not intersect the grid")
        row_top = cell_ids // n_lon
        col = cell_ids % n_lon
        row_asc = n_lat - 1 - row_top
        cell_area = grid.cell_area[row_asc, col]
        physical_area = coverage * cell_area
        total = physical_area.sum()
        if total <= 0:
            raise EmptyCoverageError(f"polygon {key} produced zero total area")
        weight = physical_area / total
        rows.append(np.full(cell_ids.size, i, dtype=np.int64))
        cols.append(row_asc * n_lon + col)
        data.append(weight)

    return sp.csr_matrix(
        (np.concatenate(data), (np.concatenate(rows), np.concatenate(cols))),
        shape=(len(polygons.keys), n_lat * n_lon),
    )
