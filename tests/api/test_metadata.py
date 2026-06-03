import geopandas as gpd
import numpy as np
import pandas as pd
import xarray as xr
from shapely.geometry import box

import geohalo as ghl

LATS = np.arange(-25.0, -19.0, 0.5)
LONS = np.arange(-50.0, -44.0, 0.5)
GEOMS = gpd.GeoSeries(
    [box(-49, -24, -47, -22), box(-47, -24, -45, -22)], index=["A", "B"],
)


def _grid(values: np.ndarray | None = None) -> xr.DataArray:
    if values is None:
        values = np.arange(2 * LATS.size * LONS.size, dtype=float).reshape(2, LATS.size, LONS.size)
    da = xr.DataArray(
        values,
        dims=("time", "latitude", "longitude"),
        coords={"time": [0, 1], "latitude": LATS, "longitude": LONS},
        name="t2m",
        attrs={"units": "K"},
    )
    return da.assign_coords(
        model="ecmwf",  # scalar coord
        valid_time=("time", pd.to_datetime(["2020-01-01", "2020-01-02"])),  # aux coord on a batch dim
    )


def test_reduce_preserves_scalar_and_aux_coords() -> None:
    out = ghl.reduce(_grid(), GEOMS)
    assert out.coords["model"].item() == "ecmwf"
    assert list(out.coords["valid_time"].values) == list(
        pd.to_datetime(["2020-01-01", "2020-01-02"]),
    )


def test_reduce_preserves_name_and_attrs() -> None:
    out = ghl.reduce(_grid(), GEOMS)
    assert out.name == "t2m"
    assert out.attrs == {"units": "K"}


def test_reduce_drops_latitude_dependent_coord() -> None:
    da = _grid().assign_coords(lat_label=("latitude", [f"r{i}" for i in range(LATS.size)]))
    out = ghl.reduce(da, GEOMS)
    assert "lat_label" not in out.coords
    assert "model" in out.coords  # the rest still survive


def test_masked_path_preserves_coords() -> None:
    values = np.arange(2 * LATS.size * LONS.size, dtype=float).reshape(2, LATS.size, LONS.size)
    values[0, 0, 0] = np.nan  # force the masked reduce path
    out = ghl.reduce(_grid(values), GEOMS)
    assert out.coords["model"].item() == "ecmwf"
    assert "valid_time" in out.coords
    assert out.name == "t2m"


def test_aggregate_bias_preserves_coords_name_attrs() -> None:
    reduced = ghl.reduce(_grid(), GEOMS)
    edges = pd.DataFrame({"parent": ["P", "P"]}, index=pd.Index(["A", "B"], name="geom"))
    rolled = ghl.aggregate_bias(reduced, edges)
    assert rolled.coords["model"].item() == "ecmwf"
    assert "valid_time" in rolled.coords
    assert rolled.name == "t2m"
    assert rolled.attrs == {"units": "K"}


def test_resample_preserves_scalar_batch_coords_and_name() -> None:
    out = ghl.resample_grid(_grid(), target_resolution=0.25)
    assert out.coords["model"].item() == "ecmwf"
    assert "valid_time" in out.coords
    assert out.name == "t2m"
    assert out.attrs == {"units": "K"}


def test_dataset_reduce_preserves_var_and_dataset_attrs() -> None:
    da = _grid()
    ds = xr.Dataset({"t2m": da, "tp": da.rename("tp")})
    ds["t2m"].attrs["units"] = "K"
    ds["tp"].attrs["units"] = "mm"
    ds.attrs["source"] = "test"
    out = ghl.reduce(ds, GEOMS)
    assert out["t2m"].attrs == {"units": "K"}
    assert out["tp"].attrs == {"units": "mm"}
    assert out.attrs == {"source": "test"}
    assert out.coords["model"].item() == "ecmwf"
