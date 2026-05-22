<p align="center">
  <img src="docs/figures/logo.svg" alt="geohalo" width="180"/>
</p>

<h1 align="center">geohalo</h1>

<p align="center">
  <em>Exact-fractional-area zonal statistics over weather grids.</em>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/python-3.12%2B-3776ab.svg" alt="Python 3.12+"/>
  <img src="https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json" alt="Ruff"/>
  <img src="https://img.shields.io/badge/managed%20by-uv-de5fe9.svg" alt="Managed by uv"/>
</p>

---

Given a regular lat/lon mesh of weather values (loaded with `xarray`
from GRIB, NetCDF, Zarr, …) and an arbitrary set of polygons,
`geohalo` reduces the spatial dimensions of the mesh to one value per
polygon with **sub-cell precision** and **millisecond-scale aggregation**
in the hot path.

## Install

`geohalo` targets **Python ≥ 3.12**.

```bash
uv add geohalo            # or: pip install geohalo
```
