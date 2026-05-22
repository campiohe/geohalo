import numpy as np
import pandas as pd
import xarray as xr

from geohalo.weights import Weights


def aggregate(
    da: xr.DataArray,
    weights: Weights,
    *,
    lat_dim: str = "latitude",
    lon_dim: str = "longitude",
    polygon_dim: str = "polygon",
) -> xr.DataArray:
    n_lat, n_lon = weights.native_shape
    if da.sizes[lat_dim] != n_lat or da.sizes[lon_dim] != n_lon:
        raise ValueError(
            f"DataArray spatial shape {(da.sizes[lat_dim], da.sizes[lon_dim])} "
            f"does not match weights native shape {weights.native_shape}",
        )

    lat_values = da[lat_dim].to_numpy()
    if lat_values[0] > lat_values[-1]:
        da = da.sortby(lat_dim)

    batch_dims = [d for d in da.dims if d not in (lat_dim, lon_dim)]
    arr = da.transpose(*batch_dims, lat_dim, lon_dim).to_numpy()
    batch_shape = arr.shape[:-2]
    flat = arr.reshape(-1, n_lat * n_lon)

    valid = ~np.isnan(flat)
    if valid.all():
        out = (weights.matrix @ flat.T).T
    else:
        filled = np.where(valid, flat, 0.0)
        numer = (weights.matrix @ filled.T).T
        denom = (weights.matrix @ valid.T.astype(np.float64)).T
        with np.errstate(invalid="ignore", divide="ignore"):
            out = np.where(denom > 0, numer / denom, np.nan)

    out = out.reshape(*batch_shape, len(weights.polygon_keys))
    polygon_index = pd.MultiIndex.from_tuples(
        weights.polygon_keys,
        names=list(weights.key_names),
    )
    return xr.DataArray(
        out,
        dims=(*batch_dims, polygon_dim),
        coords={
            **{d: da[d] for d in batch_dims},
            polygon_dim: polygon_index,
        },
    )
