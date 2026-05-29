"""End-to-end visualization script using public weather + polygon data.

Produces four PNGs under docs/figures/ that double as visual debugging
artefacts and embedded README documentation:

  grid_overlay.png             — native lat/lon mesh with polygons overlaid
  coverage_polygon_<key>.png   — exact-fractional weights for one polygon
  downscale_linear.png         — native vs raw bilinear vs linear refined field (tp)
  aggregate_choropleth.png     — polygons coloured by final t2m aggregate

Data is fetched (and cached locally) on first run:

  * GADM 4.1 Brazil level-2 municipalities (~5570 polygons, ~6 MB zip)
    from https://geodata.ucdavis.edu/gadm/gadm4.1/json/

  * The latest available ECMWF IFS ENS forecast at 0.25° resolution from
    s3://ecmwf-forecasts/. The full step file is ~6 GB; we use the
    accompanying .index file to byte-range fetch only the messages we
    need (2t + tp for a few members at step=6h), pulling ~5-10 MB.

By default, cached downloads live under ~/.cache/geohalo/. Run:

    uv run python examples/visualize.py
    uv run python examples/visualize.py --polygon-key BRA.25_1,BRA.25.580_1
"""

import argparse
import textwrap
import warnings
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import shapely
import xarray as xr
from matplotlib.figure import Figure

from benchmarks._data import brazil_municipalities, fetch_enfo_step_grib, latest_enfo_cycle
from geohalo import LocalCache, reduce_with_stencil
from geohalo.plot import (
    _draw_polygon,
    plot_aggregate_choropleth,
    plot_coverage,
    plot_downscale_comparison,
    plot_grid,
)

ENFO_PARAMS = ("2t", "tp")
ENFO_MEMBERS = (1, 2, 3, 4, 5)
ENFO_STEP_HOURS = 6
# Eastern Brazil — matches the bbox used by docs/downscaling.md so the
# example produces a comparable polygon count (~hundreds after filtering).
DEFAULT_BBOX = (-54.0, -35.0, -33.2, 8.0)  # xmin, ymin, xmax, ymax
DEFAULT_CACHE_DIR = Path.home() / ".cache" / "geohalo"

_FULL_BBOX_CAPTION = (
    "Six panels of the same forecast slice. native: the published cell-mean field. "
    "bilinear (raw): naive 4x bilinear upsample — smooth but drifts off each parent's mean. "
    "linear iter=N: N passes of mean-preserving correction (default N=1, the cached operator); "
    "every panel preserves the parent cell mean exactly regardless of N."
)

_ZOOM_SUBTITLE = (
    "Each panel is a 4x refinement of the same slice — "
    "native cell-mean, raw bilinear (not mean-preserving), "
    "and 1 / 2 / 3 / 4 iterations of the linear mean-preserving kernel"
)


def _crop_da_to_bbox(
    da: xr.DataArray,
    bbox: tuple[float, float, float, float],
    *,
    lon_dim: str = "longitude",
    lat_dim: str = "latitude",
) -> xr.DataArray:
    """Crop a global ECMWF DataArray (lon in [0, 360)) to a lon/lat bbox.

    ECMWF GRIB longitudes are 0..360; the bbox uses -180..180.
    """
    xmin, ymin, xmax, ymax = bbox
    lons = da[lon_dim].values
    if lons.max() > 180:
        new_lons = np.where(lons > 180, lons - 360, lons)
        da = da.assign_coords({lon_dim: new_lons}).sortby(lon_dim)
    return da.sel(
        {lon_dim: slice(xmin, xmax), lat_dim: slice(ymax, ymin)},
    )


def _smallest_polygon_key(geoms: gpd.GeoSeries) -> tuple:
    areas = geoms.area
    return areas.index[int(np.argmin(areas.to_numpy()))]


def _select_first(da: xr.DataArray, *, candidates: tuple[str, ...]) -> xr.DataArray:
    """Reduce every batch dim present in `candidates` down to its first index."""
    sel = {dim: 0 for dim in candidates if dim in da.dims}
    return da.isel(sel) if sel else da


def _crop_grid_and_da(
    da_slice: xr.DataArray,
    lats: np.ndarray,
    lons: np.ndarray,
    *,
    bounds: tuple[float, float, float, float],
) -> tuple[xr.DataArray, np.ndarray, np.ndarray]:
    """Crop a 2-D DataArray + lat/lon arrays to a lat/lon bbox (xmin, ymin, xmax, ymax)."""
    xmin, ymin, xmax, ymax = bounds
    lat_mask = (lats >= ymin) & (lats <= ymax)
    lon_mask = (lons >= xmin) & (lons <= xmax)
    if not lat_mask.any() or not lon_mask.any():
        raise ValueError(f"bbox {bounds!r} does not intersect grid")
    new_lats = np.asarray(lats[lat_mask])
    new_lons = np.asarray(lons[lon_mask])
    new_da = da_slice.sel(latitude=new_lats, longitude=new_lons)
    return new_da, new_lats, new_lons


def _finalise_downscale_figure(
    fig: Figure, *, title: str, caption: str, subtitle: str | None = None,
) -> None:
    """Add a bold title, optional subtitle, mm colorbar label, and a wrapped
    methodology caption to a `plot_downscale_comparison` figure.

    Mutates `fig` in place. Assumes the colorbar axis is `fig.axes[-1]` (which
    is how `plot_downscale_comparison` constructs it).

    Title/subtitle/caption are placed *outside* the [0, 1] axes bounds so they
    don't disturb the colorbar's position; `bbox_inches="tight"` at save time
    expands the canvas to include them.
    """
    cbar_ax = fig.axes[-1]
    cbar_ax.set_ylabel("tp (mm) — accumulated precipitation", fontsize=10)
    # Header: title on its own line, subtitle below in a lighter weight.
    fig.text(
        0.5, 1.06, title,
        ha="center", va="bottom", fontsize=15, fontweight="bold",
    )
    if subtitle is not None:
        fig.text(
            0.5, 1.02, subtitle,
            ha="center", va="bottom", fontsize=10, color="#555",
        )
    # Footer caption, wrapped to a width matching the figure.
    wrapped = "\n".join(textwrap.wrap(caption, width=140))
    fig.text(
        0.5, -0.02, wrapped,
        ha="center", va="top", fontsize=9.5, color="#333",
    )


def _parse_polygon_key(s: str | None) -> tuple | None:
    if s is None:
        return None
    parts = [p.strip() for p in s.split(",") if p.strip()]
    return tuple(int(p) if p.lstrip("-").isdigit() else p for p in parts)


def main() -> None:  # noqa: PLR0915
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--polygon-key",
        type=str,
        default=None,
        help="comma-separated key for the coverage heatmap (default: smallest polygon)",
    )
    parser.add_argument(
        "--downscale-polygon-key",
        type=str,
        default=None,
        help="comma-separated key for the downscale zoom (default: smallest polygon)",
    )
    parser.add_argument(
        "--downscale-padding-cells",
        type=float,
        default=4.0,
        help="extra cells of context to include around the downscale polygon (default: 4)",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "docs" / "figures",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=DEFAULT_CACHE_DIR,
        help=f"directory for downloaded GADM + GRIB + computed weights (default: {DEFAULT_CACHE_DIR})",
    )
    parser.add_argument(
        "--date",
        type=str,
        default=None,
        help="ECMWF forecast date YYYYMMDD (default: latest available)",
    )
    parser.add_argument(
        "--cycle",
        type=str,
        default=None,
        help="ECMWF cycle 00z/06z/12z/18z (default: latest available)",
    )
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    warnings.filterwarnings("ignore", category=UserWarning, module="gribapi")

    # ---- fetch data ----
    if args.date and args.cycle:
        date_str, cycle = args.date, args.cycle
    else:
        date_str, cycle = latest_enfo_cycle((ENFO_STEP_HOURS,))
        print(f"using latest available enfo cycle: {date_str} {cycle}")  # noqa: T201

    grib_path = fetch_enfo_step_grib(
        date_str, cycle, ENFO_STEP_HOURS, ENFO_PARAMS, ENFO_MEMBERS, args.cache_dir,
    )

    # ---- build polygons + grid (t2m and tp share the 0.25° global grid) ----
    da_t2m_global = xr.open_dataset(
        grib_path, engine="cfgrib",
        backend_kwargs={"filter_by_keys": {"shortName": "2t"}, "indexpath": ""},
    )["t2m"]
    da_tp_global = xr.open_dataset(
        grib_path, engine="cfgrib",
        backend_kwargs={"filter_by_keys": {"shortName": "tp"}, "indexpath": ""},
    )["tp"]

    da_t2m = _crop_da_to_bbox(da_t2m_global, DEFAULT_BBOX)
    da_tp = _crop_da_to_bbox(da_tp_global, DEFAULT_BBOX)
    lats = da_t2m["latitude"].to_numpy()
    lons = da_t2m["longitude"].to_numpy()

    geoms_full = brazil_municipalities(args.cache_dir)
    bbox_geom = shapely.box(*DEFAULT_BBOX)
    geoms = geoms_full[geoms_full.intersects(bbox_geom)]
    print(f"grid: ({lats.size}, {lons.size}), {len(geoms)} polygons in bbox")  # noqa: T201

    cache = LocalCache(args.cache_dir / "weights")
    stencil = cache.get_or_compute_stencil(lats, lons, geoms)

    # ---- grid overlay ----
    ax = plot_grid(
        lats, lons, geoms,
        title=f"ECMWF ENS t2m grid ({lats.size}x{lons.size}) "
              f"with {len(geoms)} GADM polygons",
    )
    ax.figure.savefig(args.out_dir / "grid_overlay.png", dpi=120, bbox_inches="tight")
    plt.close(ax.figure)
    print(f"wrote {args.out_dir / 'grid_overlay.png'}")  # noqa: T201

    # ---- coverage heatmap for one polygon ----
    polygon_key = _parse_polygon_key(args.polygon_key) or _smallest_polygon_key(geoms)
    ax = plot_coverage(
        stencil, geoms, polygon_key,
        title=f"exact-fractional weights for polygon {polygon_key}",
    )
    safe_key = "_".join(str(p) for p in polygon_key).replace(".", "-")
    cov_path = args.out_dir / f"coverage_polygon_{safe_key}.png"
    ax.figure.savefig(cov_path, dpi=120, bbox_inches="tight")
    plt.close(ax.figure)
    print(f"wrote {cov_path}")  # noqa: T201

    # ---- t2m choropleth ----
    da_t2m_slice = _select_first(da_t2m, candidates=("number", "step", "time"))
    out_t2m = reduce_with_stencil(da_t2m_slice, stencil)
    ax = plot_aggregate_choropleth(
        out_t2m, geoms,
        title="t2m aggregate per polygon (member 1, step 6h)",
    )
    ax.figure.savefig(args.out_dir / "aggregate_choropleth.png", dpi=120, bbox_inches="tight")
    plt.close(ax.figure)
    print(f"wrote {args.out_dir / 'aggregate_choropleth.png'}")  # noqa: T201

    # ---- tp downscale comparison (full bbox) ----
    tp_slice = _select_first(da_tp, candidates=("number", "step", "time"))
    # Visualise in mm. The native GRIB is in meters of accumulated tp; mm is
    # the conventional precipitation unit and reads more naturally on a colorbar.
    tp_slice_mm = tp_slice * 1000
    # Eastern Brazil is ~2:1 tall; use a taller+wider canvas so panels don't
    # stretch and inter-row spacing survives the natural aspect ratio. Do NOT
    # call subplots_adjust afterwards — it displaces the already-laid-out colorbar.
    fig = plot_downscale_comparison(tp_slice_mm, lats, lons, factor=4, figsize=(16, 11))
    _finalise_downscale_figure(
        fig,
        title=(
            f"Mean-preserving 4x downscaling of ECMWF ENS tp over eastern Brazil "
            f"({date_str} {cycle}, step {ENFO_STEP_HOURS}h, member 1)"
        ),
        caption=_FULL_BBOX_CAPTION,
    )
    fig.savefig(args.out_dir / "downscale_linear.png", dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {args.out_dir / 'downscale_linear.png'}")  # noqa: T201

    # ---- tp downscale zoomed into one polygon's bbox ----
    if args.downscale_polygon_key is not None:
        ds_key = _parse_polygon_key(args.downscale_polygon_key)
    else:
        ds_key = _smallest_polygon_key(geoms)
    _geom_lookup = dict(zip(geoms.index, geoms.to_numpy(), strict=True))
    zoom_targets: list[tuple[tuple, shapely.Geometry]] = [
        (ds_key, _geom_lookup[ds_key]),
    ]

    dlon = float(abs(lons[1] - lons[0]))
    dlat = float(abs(lats[1] - lats[0]))
    pad_lon = args.downscale_padding_cells * dlon
    pad_lat = args.downscale_padding_cells * dlat

    for ds_key, ds_geom in zoom_targets:
        minx, miny, maxx, maxy = ds_geom.bounds
        bounds = (minx - pad_lon, miny - pad_lat, maxx + pad_lon, maxy + pad_lat)
        tp_cropped, crop_lats, crop_lons = _crop_grid_and_da(tp_slice_mm, lats, lons, bounds=bounds)
        if min(crop_lats.size, crop_lons.size) < 2:
            print(  # noqa: T201
                f"  skip zoom: cropped grid too small ({crop_lats.size}x{crop_lons.size}) "
                f"for polygon {ds_key}",
            )
            continue
        fig = plot_downscale_comparison(
            tp_cropped, crop_lats, crop_lons, factor=4, figsize=(14, 8),
            native_label=f"native ({crop_lats.size}x{crop_lons.size} cells)",
        )
        for ax in fig.axes:
            if ax.get_xlabel() == "longitude":
                _draw_polygon(
                    ax, ds_geom,
                    facecolor="none", edgecolor="red", linewidth=1.5, zorder=3,
                )

        location = " / ".join(str(p) for p in ds_key)
        title = f"Mean-preserving 4x downscaling near {location}"
        caption = (
            f"ECMWF ENS tp, {date_str} {cycle} +{ENFO_STEP_HOURS}h, member 1. "
            f"Red outline: polygon GID_2 = {ds_key[1]}. Every panel applies the "
            f"linear mean-preserving kernel at the indicated iteration count."
        )
        _finalise_downscale_figure(
            fig, title=title, subtitle=_ZOOM_SUBTITLE, caption=caption,
        )

        safe_ds_key = "_".join(str(p) for p in ds_key).replace(".", "-")
        zoom_path = args.out_dir / f"downscale_zoom_polygon_{safe_ds_key}.png"
        fig.savefig(zoom_path, dpi=120, bbox_inches="tight")
        plt.close(fig)
        print(f"wrote {zoom_path}")  # noqa: T201


if __name__ == "__main__":
    main()
