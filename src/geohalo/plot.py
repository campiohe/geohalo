from collections.abc import Hashable
from typing import TYPE_CHECKING

import geopandas as gpd
import numpy as np
import shapely
import xarray as xr
from scipy import ndimage

from geohalo.geometry import midpoint_edges
from geohalo.resampler import Resampler

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

    from geohalo.stencil import Stencil

try:
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection
    from matplotlib.patches import Polygon as MplPolygon
except ImportError as exc:
    plt = None  # type: ignore[assignment]
    LineCollection = None  # type: ignore[assignment]
    MplPolygon = None  # type: ignore[assignment]
    _MPL_IMPORT_ERROR: ImportError | None = exc
else:
    _MPL_IMPORT_ERROR = None


def _require_matplotlib() -> None:
    if _MPL_IMPORT_ERROR is not None:
        raise ImportError(
            "geohalo.plot requires matplotlib. Install it with `pip install geohalo[matplotlib]`.",
        ) from _MPL_IMPORT_ERROR


def _grid_extent(lats: np.ndarray, lons: np.ndarray) -> tuple[float, float, float, float]:
    dlon = float(lons[1] - lons[0]) if lons.size > 1 else 1.0
    dlat = float(lats[1] - lats[0]) if lats.size > 1 else 1.0
    return (
        float(lons[0] - dlon / 2),
        float(lons[-1] + dlon / 2),
        float(lats[0] - dlat / 2),
        float(lats[-1] + dlat / 2),
    )


def _subdivide(centres: np.ndarray, factor: int) -> np.ndarray:
    if factor == 1:
        return np.asarray(centres, dtype=np.float64)
    centres = np.asarray(centres, dtype=np.float64)
    step = float(centres[1] - centres[0]) if centres.size > 1 else 1.0
    offsets = (np.arange(factor) - (factor - 1) / 2) * (step / factor)
    return (centres[:, None] + offsets[None, :]).ravel()


def _draw_polygon(ax: "Axes", geom: shapely.Geometry, **kwargs: object) -> None:
    if geom.is_empty:
        return
    if isinstance(geom, shapely.MultiPolygon):
        for part in geom.geoms:
            _draw_polygon(ax, part, **kwargs)
        return
    if not isinstance(geom, shapely.Polygon):
        return
    xs, ys = geom.exterior.coords.xy
    ax.add_patch(MplPolygon(np.column_stack([xs, ys]), closed=True, **kwargs))
    for interior in geom.interiors:
        xs_i, ys_i = interior.coords.xy
        ax.add_patch(MplPolygon(np.column_stack([xs_i, ys_i]), closed=True, **kwargs))


def plot_grid(
    lats: np.ndarray,
    lons: np.ndarray,
    geoms: gpd.GeoSeries | None = None,
    *,
    ax: "Axes | None" = None,
    title: str | None = None,
) -> "Axes":
    """Native lat/lon mesh as a wireframe, with optional polygon outlines."""
    _require_matplotlib()
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 6))

    lon_edges = midpoint_edges(np.asarray(lons))
    lat_edges = midpoint_edges(np.asarray(lats))
    segments = [[(x, lat_edges[0]), (x, lat_edges[-1])] for x in lon_edges]
    segments.extend([(lon_edges[0], y), (lon_edges[-1], y)] for y in lat_edges)
    ax.add_collection(LineCollection(segments, colors="lightgray", linewidths=0.5))

    if geoms is not None:
        for geom in geoms.to_numpy():
            _draw_polygon(ax, geom, facecolor="none", edgecolor="C0", linewidth=1.0)

    xmin, xmax, ymin, ymax = _grid_extent(np.asarray(lats), np.asarray(lons))
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_aspect("equal")
    ax.set_xlabel("longitude")
    ax.set_ylabel("latitude")
    if title:
        ax.set_title(title)
    return ax


def plot_coverage(
    stencil: "Stencil",
    geoms: gpd.GeoSeries,
    key: Hashable,
    *,
    ax: "Axes | None" = None,
    cmap: str = "viridis",
    title: str | None = None,
    zoom: bool = True,
    padding_cells: float = 2.0,
) -> "Axes":
    """Heatmap of one polygon's row of the stencil (row-normalised for display)."""
    _require_matplotlib()
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 6))

    keys = list(stencil.keys)
    try:
        row_idx = keys.index(key)
    except ValueError as e:
        raise KeyError(f"polygon key {key!r} not found in stencil") from e

    lats, lons = stencil.lats, stencil.lons
    row = np.asarray(stencil.occupancy_matrix[row_idx].todense()).reshape(lats.size, lons.size)
    total = row.sum()
    if total > 0:
        row = row / total
    masked = np.ma.masked_where(row == 0, row)

    full_xmin, full_xmax, full_ymin, full_ymax = _grid_extent(lats, lons)
    im = ax.imshow(
        masked, origin="lower",
        extent=(full_xmin, full_xmax, full_ymin, full_ymax),
        cmap=cmap, aspect="equal",
    )
    plt.colorbar(im, ax=ax, label="weight (row-normalised)")

    lon_edges = midpoint_edges(lons)
    lat_edges = midpoint_edges(lats)
    segments = [[(x, lat_edges[0]), (x, lat_edges[-1])] for x in lon_edges]
    segments.extend([(lon_edges[0], y), (lon_edges[-1], y)] for y in lat_edges)
    ax.add_collection(LineCollection(segments, colors="lightgray", linewidths=0.4, zorder=1))

    poly_geom = dict(zip(geoms.index, geoms.to_numpy(), strict=True)).get(key)
    if poly_geom is not None:
        _draw_polygon(ax, poly_geom, facecolor="none", edgecolor="red", linewidth=1.5, zorder=3)

    if zoom and poly_geom is not None and not poly_geom.is_empty:
        minx, miny, maxx, maxy = poly_geom.bounds
        dlon = float(abs(lons[1] - lons[0])) if lons.size > 1 else 1.0
        dlat = float(abs(lats[1] - lats[0])) if lats.size > 1 else 1.0
        ax.set_xlim(minx - padding_cells * dlon, maxx + padding_cells * dlon)
        ax.set_ylim(miny - padding_cells * dlat, maxy + padding_cells * dlat)
    else:
        ax.set_xlim(full_xmin, full_xmax)
        ax.set_ylim(full_ymin, full_ymax)

    ax.set_xlabel("longitude")
    ax.set_ylabel("latitude")
    ax.set_title(title or f"coverage for polygon {key}")
    return ax


def plot_downscale_comparison(
    da_slice: np.ndarray | xr.DataArray,
    lats: np.ndarray,
    lons: np.ndarray,
    *,
    factor: int,
    iterations: tuple[int, int, int, int] = (1, 2, 3, 4),
    figsize: tuple[float, float] = (14, 8),
    cmap: str = "viridis",
    native_label: str = "native",
) -> "Figure":
    """2x3 comparison: native, raw bilinear, and four Resampler iteration counts.

    Each "linear iter=N" panel applies the value-independent Resampler matrix
    (N iterations) for a factor-N refinement; every refined panel preserves the
    parent cell mean exactly regardless of N.
    """
    _require_matplotlib()
    if factor < 1:
        raise ValueError(f"factor must be >= 1, got {factor}")
    if len(iterations) != 4 or any(it < 1 for it in iterations):
        raise ValueError(f"iterations must be a 4-tuple of positive ints, got {iterations!r}")

    arr = np.asarray(da_slice.values if isinstance(da_slice, xr.DataArray) else da_slice)
    if arr.ndim != 2:
        raise ValueError(f"da_slice must be 2-D, got ndim={arr.ndim}")
    lats = np.asarray(lats, dtype=np.float64)
    lons = np.asarray(lons, dtype=np.float64)

    if factor == 1:
        fig, ax = plt.subplots(1, 1, figsize=(figsize[0] / 3, figsize[1] / 2))
        xmin, xmax, ymin, ymax = _grid_extent(lats, lons)
        im = ax.imshow(arr, origin="lower", extent=(xmin, xmax, ymin, ymax), cmap=cmap, aspect="equal")
        ax.set_title(native_label)
        ax.set_xlabel("longitude")
        ax.set_ylabel("latitude")
        fig.colorbar(im, ax=ax)
        return fig

    r_lats = _subdivide(lats, factor)
    r_lons = _subdivide(lons, factor)
    bilinear_raw = ndimage.zoom(arr, factor, order=1, mode="nearest", grid_mode=True)

    def _linear(it: int) -> np.ndarray:
        transform = Resampler.compute(lats, lons, r_lats, r_lons, iterations=it).transform_matrix
        return (transform @ arr.ravel()).reshape(r_lats.size, r_lons.size)

    iter_outputs = [_linear(it) for it in iterations]

    panels = [
        (arr, lats, lons, native_label),
        (bilinear_raw, r_lats, r_lons, "bilinear (raw, not mean-preserving)"),
        (iter_outputs[0], r_lats, r_lons, f"linear iter={iterations[0]} (constant correction)"),
        (iter_outputs[1], r_lats, r_lons, f"linear iter={iterations[1]}"),
        (iter_outputs[2], r_lats, r_lons, f"linear iter={iterations[2]}"),
        (iter_outputs[3], r_lats, r_lons, f"linear iter={iterations[3]} (smoothed)"),
    ]
    vmin = min(float(p[0].min()) for p in panels)
    vmax = max(float(p[0].max()) for p in panels)

    fig, axes = plt.subplots(2, 3, figsize=figsize)
    im = None
    for ax, (data, plats, plons, title) in zip(axes.flat, panels, strict=True):
        xmin, xmax, ymin, ymax = _grid_extent(plats, plons)
        im = ax.imshow(
            data, origin="lower", extent=(xmin, xmax, ymin, ymax),
            vmin=vmin, vmax=vmax, cmap=cmap, aspect="equal",
        )
        ax.set_title(title)
        ax.set_xlabel("longitude")
    for row in axes:
        row[0].set_ylabel("latitude")
    fig.colorbar(im, ax=axes, shrink=0.7)
    return fig


def plot_aggregate_choropleth(
    aggregated: xr.DataArray,
    geoms: gpd.GeoSeries,
    *,
    ax: "Axes | None" = None,
    cmap: str = "viridis",
    title: str | None = None,
    geom_dim: str = "geom",
    edgecolor: str = "black",
    linewidth: float = 0.3,
) -> "Axes":
    """Polygons coloured by their final aggregated value (1-D over `geom_dim`)."""
    _require_matplotlib()
    if aggregated.ndim != 1:
        raise ValueError(
            f"aggregated must be 1-D over `{geom_dim}`; got dims={aggregated.dims}. "
            "Reduce or select batch dims before plotting.",
        )
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 6))

    values = aggregated.to_numpy()
    finite = np.isfinite(values)
    if not finite.any():
        raise ValueError("aggregated has no finite values to plot")
    norm = plt.Normalize(vmin=float(values[finite].min()), vmax=float(values[finite].max()))
    colormap = plt.get_cmap(cmap)

    poly_index = aggregated[geom_dim].to_index()
    lookup = dict(zip(geoms.index, geoms.to_numpy(), strict=True))

    for value, key in zip(values, poly_index, strict=True):
        geom = lookup.get(key)
        if geom is None:
            continue
        facecolor = "lightgray" if not np.isfinite(value) else colormap(norm(value))
        _draw_polygon(ax, geom, facecolor=facecolor, edgecolor=edgecolor, linewidth=linewidth)

    sm = plt.cm.ScalarMappable(norm=norm, cmap=colormap)
    plt.colorbar(sm, ax=ax, label=aggregated.name or "aggregated value")

    xs = [b for g in geoms.to_numpy() for b in g.bounds[::2]]
    ys = [b for g in geoms.to_numpy() for b in g.bounds[1::2]]
    if xs and ys:
        ax.set_xlim(min(xs), max(xs))
        ax.set_ylim(min(ys), max(ys))
    ax.set_aspect("equal")
    ax.set_xlabel("longitude")
    ax.set_ylabel("latitude")
    if title:
        ax.set_title(title)
    return ax
