from typing import TYPE_CHECKING

import numpy as np
import shapely
import xarray as xr
from scipy import ndimage

from geohalo.downscale import downscale_plane, refine_grid
from geohalo.geometry import PolygonSet
from geohalo.grid import GridSpec
from geohalo.weights import Weights

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

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
            "geohalo.plot requires matplotlib. Install it with `pip install geohalo[viz]`.",
        ) from _MPL_IMPORT_ERROR


def _grid_extent(grid: GridSpec) -> tuple[float, float, float, float]:
    dlon = float(grid.lons[1] - grid.lons[0]) if grid.lons.size > 1 else 1.0
    dlat = float(grid.lats[1] - grid.lats[0]) if grid.lats.size > 1 else 1.0
    return (
        float(grid.lons[0] - dlon / 2),
        float(grid.lons[-1] + dlon / 2),
        float(grid.lats[0] - dlat / 2),
        float(grid.lats[-1] + dlat / 2),
    )


def _draw_polygon(ax: "Axes", geom: shapely.Geometry, **kwargs) -> None:
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
    grid: GridSpec,
    polygons: PolygonSet | None = None,
    *,
    ax: "Axes | None" = None,
    polygon_subset: list[tuple] | None = None,
    title: str | None = None,
) -> "Axes":
    """Native lat/lon mesh as a wireframe, with optional polygon outlines.

    Useful first look at "which cells does each polygon touch."
    """
    _require_matplotlib()
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 6))

    lon_edges = _edges(grid.lons)
    lat_edges = _edges(grid.lats)
    segments = [[(x, lat_edges[0]), (x, lat_edges[-1])] for x in lon_edges]
    segments.extend([(lon_edges[0], y), (lon_edges[-1], y)] for y in lat_edges)
    ax.add_collection(LineCollection(segments, colors="lightgray", linewidths=0.5))

    if polygons is not None:
        keys_to_draw = set(polygon_subset) if polygon_subset is not None else None
        for key, geom in zip(polygons.keys, polygons.geoms, strict=True):
            if keys_to_draw is not None and key not in keys_to_draw:
                continue
            _draw_polygon(ax, geom, facecolor="none", edgecolor="C0", linewidth=1.0)

    xmin, xmax, ymin, ymax = _grid_extent(grid)
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_aspect("equal")
    ax.set_xlabel("longitude")
    ax.set_ylabel("latitude")
    if title:
        ax.set_title(title)
    return ax


def plot_coverage(
    grid: GridSpec,
    polygons: PolygonSet,
    weights: Weights,
    polygon_key: tuple,
    *,
    ax: "Axes | None" = None,
    cmap: str = "viridis",
    title: str | None = None,
    zoom: bool = True,
    padding_cells: float = 2.0,
) -> "Axes":
    """Heatmap of the row of `W` corresponding to one polygon.

    Each cell is coloured by its final (area-corrected, row-normalised) weight.
    Overlays the polygon outline (red) and a wireframe of nearby grid cells.
    By default zooms to the polygon's bbox + `padding_cells` cells of context
    so sub-cell polygons stay visible; pass `zoom=False` to show the full grid.
    """
    _require_matplotlib()
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 6))

    try:
        row_idx = weights.polygon_keys.index(polygon_key)
    except ValueError as e:
        raise KeyError(f"polygon key {polygon_key!r} not found in weights") from e

    row = np.asarray(weights.matrix[row_idx].todense()).reshape(weights.native_shape)
    masked = np.ma.masked_where(row == 0, row)

    full_xmin, full_xmax, full_ymin, full_ymax = _grid_extent(grid)
    im = ax.imshow(
        masked,
        origin="lower",
        extent=(full_xmin, full_xmax, full_ymin, full_ymax),
        cmap=cmap,
        aspect="equal",
    )
    plt.colorbar(im, ax=ax, label="weight (row-normalised)")

    lon_edges = _edges(grid.lons)
    lat_edges = _edges(grid.lats)
    segments = [[(x, lat_edges[0]), (x, lat_edges[-1])] for x in lon_edges]
    segments.extend([(lon_edges[0], y), (lon_edges[-1], y)] for y in lat_edges)
    ax.add_collection(
        LineCollection(segments, colors="lightgray", linewidths=0.4, zorder=1),
    )

    poly_geom: shapely.Geometry | None = None
    try:
        poly_idx = polygons.keys.index(polygon_key)
        poly_geom = polygons.geoms[poly_idx]
        _draw_polygon(
            ax, poly_geom,
            facecolor="none", edgecolor="red", linewidth=1.5, zorder=3,
        )
    except ValueError:
        pass

    if zoom and poly_geom is not None and not poly_geom.is_empty:
        minx, miny, maxx, maxy = poly_geom.bounds
        dlon = float(abs(grid.lons[1] - grid.lons[0])) if grid.lons.size > 1 else 1.0
        dlat = float(abs(grid.lats[1] - grid.lats[0])) if grid.lats.size > 1 else 1.0
        ax.set_xlim(minx - padding_cells * dlon, maxx + padding_cells * dlon)
        ax.set_ylim(miny - padding_cells * dlat, maxy + padding_cells * dlat)
    else:
        ax.set_xlim(full_xmin, full_xmax)
        ax.set_ylim(full_ymin, full_ymax)

    ax.set_xlabel("longitude")
    ax.set_ylabel("latitude")
    ax.set_title(title or f"coverage for polygon {polygon_key}")
    return ax


def plot_downscale_comparison(
    da_slice: np.ndarray | xr.DataArray,
    grid: GridSpec,
    *,
    factor: int,
    iterations: tuple[int, int, int, int] = (1, 2, 4, 10),
    figsize: tuple[float, float] = (14, 8),
    cmap: str = "viridis",
    native_label: str = "native",
) -> "Figure":
    """2x3 grid comparing native, raw bilinear, and four iteration counts of
    the linear mean-preserving downscale kernel.

    Layout:
      Row 1: native cell-mean field | raw bilinear upsample (drifts) | linear iter=N0
      Row 2: linear iter=N1 | linear iter=N2 | linear iter=N3

    Each "linear iter=N" panel is the output of `downscale_plane(arr, factor,
    iterations=N)`. With N=1 the per-parent additive correction is constant
    inside each parent block, which produces a visible blocky pattern at
    parent boundaries. Higher N distributes the correction bilinearly across
    children, smoothing the pattern; a final hard correction step still
    guarantees exact per-parent mean preservation regardless of N.

    At factor=1 there is nothing to downscale; the function returns a
    single-panel figure of the native field.
    """
    _require_matplotlib()
    if factor < 1:
        raise ValueError(f"factor must be >= 1, got {factor}")
    if len(iterations) != 4 or any(it < 1 for it in iterations):
        raise ValueError(
            f"iterations must be a 4-tuple of positive ints, got {iterations!r}",
        )

    arr = np.asarray(da_slice.values if isinstance(da_slice, xr.DataArray) else da_slice)
    if arr.ndim != 2:
        raise ValueError(f"da_slice must be 2-D, got ndim={arr.ndim}")

    if factor == 1:
        fig, ax = plt.subplots(1, 1, figsize=(figsize[0] / 3, figsize[1] / 2))
        xmin, xmax, ymin, ymax = _grid_extent(grid)
        im = ax.imshow(
            arr, origin="lower", extent=(xmin, xmax, ymin, ymax), cmap=cmap, aspect="equal",
        )
        ax.set_title(native_label)
        ax.set_xlabel("longitude")
        ax.set_ylabel("latitude")
        fig.colorbar(im, ax=ax)
        return fig

    refined_grid = refine_grid(grid, factor)
    bilinear_raw = ndimage.zoom(arr, factor, order=1, mode="nearest", grid_mode=True)
    iter_outputs = [downscale_plane(arr, factor, iterations=it) for it in iterations]

    panels = [
        (arr, grid, native_label),
        (bilinear_raw, refined_grid, "bilinear (raw, not mean-preserving)"),
        (iter_outputs[0], refined_grid, f"linear iter={iterations[0]} (constant correction)"),
        (iter_outputs[1], refined_grid, f"linear iter={iterations[1]}"),
        (iter_outputs[2], refined_grid, f"linear iter={iterations[2]}"),
        (iter_outputs[3], refined_grid, f"linear iter={iterations[3]} (smoothed)"),
    ]
    vmin = min(float(p[0].min()) for p in panels)
    vmax = max(float(p[0].max()) for p in panels)

    fig, axes = plt.subplots(2, 3, figsize=figsize)
    im = None
    for ax, (data, g, title) in zip(axes.flat, panels, strict=True):
        xmin, xmax, ymin, ymax = _grid_extent(g)
        im = ax.imshow(
            data, origin="lower",
            extent=(xmin, xmax, ymin, ymax),
            vmin=vmin, vmax=vmax,
            cmap=cmap, aspect="equal",
        )
        ax.set_title(title)
        ax.set_xlabel("longitude")
    for row in axes:
        row[0].set_ylabel("latitude")
    fig.colorbar(im, ax=axes, shrink=0.7)
    return fig


def plot_aggregate_choropleth(
    aggregated: xr.DataArray,
    polygons: PolygonSet,
    *,
    ax: "Axes | None" = None,
    cmap: str = "viridis",
    title: str | None = None,
    polygon_dim: str = "polygon",
    edgecolor: str = "black",
    linewidth: float = 0.3,
) -> "Axes":
    """Polygons coloured by their final aggregated value.

    `aggregated` must be 1-D after reducing batch dims (e.g.,
    `aggregate(da, w).isel(number=0, step=0)` for a single member/lead time).
    """
    _require_matplotlib()
    if aggregated.ndim != 1:
        raise ValueError(
            f"aggregated must be 1-D over `{polygon_dim}`; got dims={aggregated.dims}. "
            "Reduce or select batch dims before plotting.",
        )
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 6))

    values = aggregated.to_numpy()
    finite = np.isfinite(values)
    if not finite.any():
        raise ValueError("aggregated has no finite values to plot")
    vmin = float(values[finite].min())
    vmax = float(values[finite].max())
    norm = plt.Normalize(vmin=vmin, vmax=vmax)
    colormap = plt.get_cmap(cmap)

    poly_index = aggregated[polygon_dim].to_index()
    polygons_lookup = dict(zip(polygons.keys, polygons.geoms, strict=True))

    for value, key in zip(values, poly_index, strict=True):
        lookup_key = key if isinstance(key, tuple) else (key,)
        geom = polygons_lookup.get(lookup_key)
        if geom is None:
            continue
        if not np.isfinite(value):
            _draw_polygon(ax, geom, facecolor="lightgray", edgecolor=edgecolor, linewidth=linewidth)
        else:
            _draw_polygon(
                ax, geom,
                facecolor=colormap(norm(value)),
                edgecolor=edgecolor,
                linewidth=linewidth,
            )

    sm = plt.cm.ScalarMappable(norm=norm, cmap=colormap)
    plt.colorbar(sm, ax=ax, label=aggregated.name or "aggregated value")

    xs = [b for g in polygons.geoms for b in g.bounds[::2]]
    ys = [b for g in polygons.geoms for b in g.bounds[1::2]]
    if xs and ys:
        ax.set_xlim(min(xs), max(xs))
        ax.set_ylim(min(ys), max(ys))
    ax.set_aspect("equal")
    ax.set_xlabel("longitude")
    ax.set_ylabel("latitude")
    if title:
        ax.set_title(title)
    return ax


def _edges(centres: np.ndarray) -> np.ndarray:
    if centres.size < 2:
        raise ValueError("need >= 2 coordinates to derive edges")
    mids = (centres[:-1] + centres[1:]) / 2.0
    first = centres[0] - (mids[0] - centres[0])
    last = centres[-1] + (centres[-1] - mids[-1])
    return np.concatenate([[first], mids, [last]])
