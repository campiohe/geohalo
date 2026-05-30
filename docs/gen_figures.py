"""Generate the documentation figures for geohalo.

Every figure is self-contained (synthetic data, deterministic) and uses the
project's logo palette: deep navy ink on white cards, with an amber "halo"
sequential colormap for coverage/weight fields.

    uv run python docs/gen_figures.py

Writes PNGs into docs/figures/. The hand-kept `mean-preserving-downscale.png`
and `logo.svg` are never touched.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import shapely
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import FancyArrowPatch, Polygon as MplPolygon

# ---------------------------------------------------------------- palette ----
NAVY = "#0b1220"      # ink / logo background
SLATE = "#1e293b"     # secondary ink
SLATE_SOFT = "#64748b"  # gridlines / muted
AMBER = "#f59e0b"     # primary accent (logo cells)
AMBER_DARK = "#b45309"
CREAM = "#fef3c7"     # logo polygon stroke
RUST = "#7c2d12"

# Sequential "halo" map: faint cream -> amber -> rust. White-ish at zero so
# empty cells melt into the white card.
HALO = LinearSegmentedColormap.from_list(
    "halo", ["#fffdf7", CREAM, "#fcd34d", AMBER, AMBER_DARK, RUST],
)

OUT = Path(__file__).resolve().parent / "figures"
OUT.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.size": 11,
    "font.family": "sans-serif",
    "axes.edgecolor": SLATE,
    "axes.labelcolor": NAVY,
    "text.color": NAVY,
    "xtick.color": SLATE,
    "ytick.color": SLATE,
    "figure.facecolor": "white",
    "savefig.facecolor": "white",
    "savefig.dpi": 150,
})


def _save(fig: plt.Figure, name: str) -> None:
    path = OUT / name
    fig.savefig(path, bbox_inches="tight", pad_inches=0.18)
    plt.close(fig)
    print(f"wrote {path}")


def _draw_cells(ax, lats, lons, values, *, vmax=None, edge=SLATE_SOFT, lw=0.6):
    """Draw a regular cell mesh shaded by `values` (n_lat x n_lon)."""
    dlon = lons[1] - lons[0]
    dlat = lats[1] - lats[0]
    vmax = vmax if vmax is not None else float(np.nanmax(values)) or 1.0
    for i, y in enumerate(lats):
        for j, x in enumerate(lons):
            v = values[i, j]
            ax.add_patch(
                plt.Rectangle(
                    (x - dlon / 2, y - dlat / 2), dlon, dlat,
                    facecolor=HALO(v / vmax) if v > 0 else "white",
                    edgecolor=edge, linewidth=lw,
                ),
            )
    ax.set_xlim(lons[0] - dlon / 2, lons[-1] + dlon / 2)
    ax.set_ylim(lats[0] - dlat / 2, lats[-1] + dlat / 2)
    ax.set_aspect("equal")


def _outline(ax, geom, color=NAVY, lw=2.2):
    polys = geom.geoms if isinstance(geom, shapely.MultiPolygon) else [geom]
    for p in polys:
        xs, ys = p.exterior.coords.xy
        ax.add_patch(MplPolygon(np.column_stack([xs, ys]), closed=True,
                                facecolor="none", edgecolor=color, linewidth=lw,
                                joinstyle="round"))


# --------------------------------------------------- fig 1: coverage methods ----
def fig_coverage_methods() -> None:
    """Centroid vs all-touched vs exact-fractional on one polygon over a 5x5 mesh."""
    n = 5
    lons = np.arange(n, dtype=float)
    lats = np.arange(n, dtype=float)
    # A blobby polygon crossing several cell boundaries.
    poly = shapely.Polygon([
        (0.7, 0.6), (2.6, 0.3), (3.9, 1.6), (3.5, 3.2),
        (2.2, 3.8), (0.9, 3.1), (0.3, 1.7),
    ])

    centroid = np.zeros((n, n))
    touched = np.zeros((n, n))
    exact = np.zeros((n, n))
    for i, y in enumerate(lats):
        for j, x in enumerate(lons):
            cell = shapely.box(x - 0.5, y - 0.5, x + 0.5, y + 0.5)
            inter = cell.intersection(poly).area
            exact[i, j] = inter  # cell area = 1, so fraction == area
            touched[i, j] = 1.0 if inter > 1e-9 else 0.0
            centroid[i, j] = 1.0 if poly.contains(shapely.Point(x, y)) else 0.0

    panels = [
        ("Centroid", centroid, r"cell centre $\in P$", "lossy: misses half-cells"),
        ("All-touched", touched, r"cell $\cap\,P \neq \varnothing$", "bloated: grazed cells count fully"),
        ("Exact fractional", exact, r"area$(\,$cell$\,\cap\,P)\,/\,$area$($cell$)$", "unbiased: weight = true overlap"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.6))
    for ax, (title, vals, formula, caption) in zip(axes, panels, strict=True):
        _draw_cells(ax, lats, lons, vals, vmax=1.0)
        _outline(ax, poly, color=NAVY, lw=2.4)
        for i, y in enumerate(lats):
            for j, x in enumerate(lons):
                v = vals[i, j]
                if v > 1e-9:
                    ax.text(x, y, f"{v:.2f}".lstrip("0") if v < 1 else "1",
                            ha="center", va="center", fontsize=8.5,
                            color="white" if v > 0.55 else NAVY, fontweight="bold")
        ax.set_title(title, fontsize=13, fontweight="bold", color=NAVY, pad=8)
        ax.text(0.5, -0.07, formula, transform=ax.transAxes, ha="center", va="top", fontsize=10.5)
        ax.text(0.5, -0.17, caption, transform=ax.transAxes, ha="center", va="top",
                fontsize=9.5, color=SLATE_SOFT, style="italic")
        ax.set_xticks([]); ax.set_yticks([])
    _save(fig, "coverage-methods.png")


# --------------------------------------------------- fig 2: latitude area ----
def fig_latitude_area() -> None:
    """How a constant-degree cell's true area collapses toward the pole."""
    R = 6_371_008.8
    dlat = dlon = 1.0  # 1-degree cell
    lat = np.linspace(0, 85, 400)
    sin_top = np.sin(np.deg2rad(lat + dlat / 2))
    sin_bot = np.sin(np.deg2rad(lat - dlat / 2))
    area = R**2 * np.deg2rad(dlon) * (sin_top - sin_bot)
    rel = area / area[0]

    fig, (axc, axg) = plt.subplots(1, 2, figsize=(12, 4.7),
                                   gridspec_kw={"width_ratios": [1.25, 1]})

    # left: relative area curve
    axc.fill_between(lat, rel, color=AMBER, alpha=0.25)
    axc.plot(lat, rel, color=AMBER_DARK, lw=2.4)
    for lt in (0, 30, 60, 80):
        r = float(rel[np.argmin(np.abs(lat - lt))])
        axc.plot([lt], [r], "o", color=NAVY, ms=6)
        axc.annotate(f"{r*100:.0f}%", (lt, r), textcoords="offset points",
                     xytext=(6, 8), fontsize=10, color=NAVY, fontweight="bold")
    axc.set_xlabel("latitude (°)")
    axc.set_ylabel("cell area  /  equatorial cell area")
    axc.set_title("A 1° cell shrinks with latitude", fontsize=12.5, fontweight="bold")
    axc.set_xlim(0, 85); axc.set_ylim(0, 1.05)
    axc.grid(True, color="#e2e8f0", lw=0.8)
    axc.spines[["top", "right"]].set_visible(False)

    # right: stacked cells of proportional width
    bands = [0, 30, 60, 80]
    axg.set_title(r"physical weight $\propto R^2\,\Delta\lambda\,(\sin\varphi_\mathrm{top}-\sin\varphi_\mathrm{bot})$",
                  fontsize=11, fontweight="bold")
    for k, lt in enumerate(bands):
        r = float(rel[np.argmin(np.abs(lat - lt))])
        y = len(bands) - 1 - k
        axg.add_patch(plt.Rectangle((0, y + 0.12), r, 0.76, facecolor=HALO(r),
                                    edgecolor=SLATE, lw=1.0))
        axg.text(-0.04, y + 0.5, f"{lt}°", ha="right", va="center", fontsize=11, fontweight="bold")
        axg.text(r + 0.02, y + 0.5, f"{r*100:.0f}%", ha="left", va="center", fontsize=10, color=SLATE)
    axg.set_xlim(-0.18, 1.18); axg.set_ylim(-0.1, len(bands))
    axg.axis("off")
    _save(fig, "latitude-area.png")


# ------------------------------------------------- fig 3: stencil halo --------
def fig_stencil_halo() -> None:
    """One polygon's row of the stencil: exact fractional coverage x cell area."""
    n_lat, n_lon = 13, 14
    lons = np.arange(n_lon, dtype=float)
    lats = np.arange(n_lat, dtype=float)
    poly = shapely.Polygon([
        (2.3, 2.1), (6.0, 1.2), (9.4, 2.6), (11.2, 5.5), (10.3, 8.7),
        (7.1, 10.4), (3.8, 9.6), (1.6, 7.0), (1.1, 4.2),
    ]).difference(shapely.Point(6.5, 6.0).buffer(1.4))  # a hole, to show interiors

    cov = np.zeros((n_lat, n_lon))
    for i, y in enumerate(lats):
        for j, x in enumerate(lons):
            cell = shapely.box(x - 0.5, y - 0.5, x + 0.5, y + 0.5)
            cov[i, j] = cell.intersection(poly).area

    fig, ax = plt.subplots(figsize=(7.4, 6.8))
    _draw_cells(ax, lats, lons, cov, vmax=1.0, lw=0.7)
    _outline(ax, poly, color=NAVY, lw=2.6)
    sm = plt.cm.ScalarMappable(cmap=HALO, norm=plt.Normalize(0, 1))
    cb = fig.colorbar(sm, ax=ax, fraction=0.046, pad=0.03)
    cb.set_label("fractional coverage  ×  cell area  (one stencil row)")
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title("A polygon's halo of weighted cells", fontsize=13, fontweight="bold", pad=8)
    _save(fig, "stencil-halo.png")


# --------------------------------------------- fig 4: linear operator ---------
def fig_linear_operator() -> None:
    """Schematic: sparse W (n_polygons x n_cells) @ flat grid = per-polygon vector."""
    rng = np.random.default_rng(7)
    n_poly, n_cells = 6, 64  # 8x8 grid flattened
    side = 8
    # Build a plausible band-sparse W: each polygon hits a small contiguous blob.
    W = np.zeros((n_poly, n_cells))
    for r in range(n_poly):
        cy, cx = rng.integers(1, side - 1), rng.integers(1, side - 1)
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                yy, xx = cy + dy, cx + dx
                if 0 <= yy < side and 0 <= xx < side and rng.random() > 0.25:
                    W[r, yy * side + xx] = rng.uniform(0.2, 1.0)

    fig = plt.figure(figsize=(13, 4.4))
    gs = fig.add_gridspec(1, 5, width_ratios=[3.4, 0.5, 1.1, 0.5, 1.0], wspace=0.32)

    axW = fig.add_subplot(gs[0, 0])
    axW.imshow(np.ma.masked_where(W == 0, W), cmap=HALO, aspect="auto", vmin=0, vmax=1)
    axW.set_title(r"$\mathbf{W}$  —  stencil  (sparse)", fontsize=12, fontweight="bold")
    axW.set_xlabel(r"$N_\mathrm{cells}$  (flattened grid)")
    axW.set_ylabel(r"$N_\mathrm{polygons}$")
    axW.set_xticks([]); axW.set_yticks(range(n_poly))

    axop = fig.add_subplot(gs[0, 1]); axop.axis("off")
    axop.text(0.5, 0.5, "@", ha="center", va="center", fontsize=30, color=AMBER_DARK, fontweight="bold")

    grid = rng.uniform(0, 1, (side, side))
    axx = fig.add_subplot(gs[0, 2])
    axx.imshow(grid, cmap="Blues", aspect="auto")
    axx.set_title(r"$\mathbf{x}$  —  grid", fontsize=12, fontweight="bold")
    axx.set_xticks([]); axx.set_yticks([])

    axeq = fig.add_subplot(gs[0, 3]); axeq.axis("off")
    axeq.text(0.5, 0.5, "=", ha="center", va="center", fontsize=30, color=NAVY, fontweight="bold")

    a = W @ grid.ravel()
    axa = fig.add_subplot(gs[0, 4])
    axa.barh(range(n_poly), a[::-1], color=AMBER, edgecolor=AMBER_DARK)
    axa.set_title(r"$\mathbf{a}$  —  per polygon", fontsize=12, fontweight="bold")
    axa.set_yticks([]); axa.set_xticks([])
    axa.spines[["top", "right", "left"]].set_visible(False)

    fig.suptitle(r"Aggregation is one matmul:   $\mathbf{a} = \mathbf{W}\,\mathbf{x}$",
                 fontsize=14, fontweight="bold", y=1.02)
    _save(fig, "linear-operator.png")


# ----------------------------------------- fig 5: 1-D mean-preserving ---------
def _resample_1d(values, factor, iterations):
    """1-D analogue of geohalo's resampler: B (linear), A (parent mean), P (broadcast)."""
    n_s = values.size
    n_t = n_s * factor
    s_centres = np.arange(n_s) + 0.5
    t_centres = (np.arange(n_t) + 0.5) / factor

    # B: linear interpolation target<-source with edge clamp
    idx = np.clip(np.searchsorted(s_centres, t_centres) - 1, 0, n_s - 2)
    frac = np.clip((t_centres - s_centres[idx]) / (s_centres[idx + 1] - s_centres[idx]), 0, 1)
    B = np.zeros((n_t, n_s))
    B[np.arange(n_t), idx] = 1 - frac
    B[np.arange(n_t), idx + 1] += frac

    parent = np.floor(t_centres).astype(int)
    P = np.zeros((n_t, n_s)); P[np.arange(n_t), parent] = 1.0
    counts = np.bincount(parent, minlength=n_s)
    A = np.zeros((n_s, n_t)); A[parent, np.arange(n_t)] = 1.0 / counts[parent]

    G = np.eye(n_t) - B @ A
    y_op = sum(np.linalg.matrix_power(G, j) for j in range(iterations)) @ B
    T = y_op + P @ (np.eye(n_s) - A @ y_op)
    return t_centres, T @ values


def fig_downscale_1d() -> None:
    src = np.array([2.0, 5.0, 1.5, 4.0, 3.0])
    factor = 16  # smooth curve
    s_edges = np.arange(src.size + 1)

    xb, yb = _resample_1d(src, factor, 1)  # bilinear-only baseline (iterations=1 w/o... )
    # raw bilinear (no correction): just B
    n_s = src.size; n_t = n_s * factor
    s_centres = np.arange(n_s) + 0.5
    t_centres = (np.arange(n_t) + 0.5) / factor
    idx = np.clip(np.searchsorted(s_centres, t_centres) - 1, 0, n_s - 2)
    frac = np.clip((t_centres - s_centres[idx]) / (s_centres[idx + 1] - s_centres[idx]), 0, 1)
    raw = (1 - frac) * src[idx] + frac * src[idx + 1]
    x1, y1 = _resample_1d(src, factor, 1)
    x3, y3 = _resample_1d(src, factor, 3)

    fig, ax = plt.subplots(figsize=(11, 5.2))
    # source step function
    ax.hlines(src, s_edges[:-1], s_edges[1:], color=NAVY, lw=3, label="source cell mean", zorder=5)
    for k in range(src.size):
        ax.add_patch(plt.Rectangle((k, 0), 1, src[k], facecolor=AMBER, alpha=0.07))
        ax.axvline(k, color="#e2e8f0", lw=1)
    ax.axvline(src.size, color="#e2e8f0", lw=1)

    ax.plot(t_centres, raw, color=SLATE_SOFT, lw=2, ls="--", label="raw bilinear (drifts off the mean)")
    ax.plot(x1, y1, color=AMBER_DARK, lw=2.4, label="mean-preserving, iterations=1")
    ax.plot(x3, y3, color=RUST, lw=2.4, label="mean-preserving, iterations=3 (smoother)")

    ax.set_xlim(0, src.size); ax.set_ylim(0, 6.6)
    ax.set_xlabel("source cell"); ax.set_ylabel("value")
    ax.set_xticks(np.arange(src.size) + 0.5, [f"cell {i}" for i in range(src.size)])
    ax.set_title("Mean-preserving downscaling (1-D): the average over each shaded "
                 "cell\nequals the source value — for every iteration count",
                 fontsize=12.5, fontweight="bold")
    ax.legend(loc="upper right", framealpha=0.95, fontsize=9.5)
    ax.spines[["top", "right"]].set_visible(False)
    _save(fig, "downscale-1d.png")


# ------------------------------------------ fig 6: fused operator size --------
def fig_fused_operator_size() -> None:
    """Materialised resampler T vs the fused ReduceOperator (cache blob sizes)."""
    fig, ax = plt.subplots(figsize=(9.5, 4.4))
    labels = ["Resampler matrix  T\n(materialised)", "ReduceOperator  M = W·T\n(fused, thin)"]
    sizes = [358.0, 0.40]  # MB, from the README headline example
    bars = ax.barh(labels, sizes, color=[SLATE_SOFT, AMBER],
                   edgecolor=[SLATE, AMBER_DARK], height=0.55)
    ax.set_xscale("log")
    ax.set_xlim(0.1, 1200)
    ax.set_xlabel("cache blob size (MB, log scale)")
    for b, s in zip(bars, sizes, strict=True):
        ax.text(s * 1.25, b.get_y() + b.get_height() / 2,
                f"{s:.2f} MB" if s < 1 else f"{s:.0f} MB",
                va="center", fontsize=12, fontweight="bold", color=NAVY)
    ax.text(358 * 1.25, 1.0, "  ✗ cannot even build at iterations=3", va="center",
            color=RUST, fontsize=9.5, style="italic")
    ax.set_title("0.25° → 0.05° refine, 500 polygons  —  same answer, ~900× smaller",
                 fontsize=12, fontweight="bold")
    ax.spines[["top", "right"]].set_visible(False)
    ax.invert_yaxis()
    _save(fig, "fused-operator-size.png")


if __name__ == "__main__":
    fig_coverage_methods()
    fig_latitude_area()
    fig_stencil_halo()
    fig_linear_operator()
    fig_downscale_1d()
    fig_fused_operator_size()
    print("all figures written to", OUT)
