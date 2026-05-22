import hashlib
from dataclasses import dataclass, field

import numpy as np
import xarray as xr

EARTH_RADIUS_M = 6_371_008.8


@dataclass
class GridSpec:
    lats: np.ndarray
    lons: np.ndarray
    lats_were_descending: bool = field(init=False)
    digest: bytes = field(init=False)
    shape: tuple[int, int] = field(init=False)
    cell_area: np.ndarray = field(init=False)

    def __post_init__(self) -> None:
        lats = np.asarray(self.lats, dtype=np.float64)
        lons = np.asarray(self.lons, dtype=np.float64)
        if lats.ndim != 1 or lons.ndim != 1:
            raise ValueError("lats and lons must be 1-D")
        descending = bool(lats.size > 1 and lats[0] > lats[-1])
        if descending:
            lats = lats[::-1].copy()
        self.lats = lats
        self.lons = lons
        self.lats_were_descending = descending
        self.shape = (lats.size, lons.size)
        self.cell_area = _spherical_cell_areas(lats, lons)
        h = hashlib.sha256()
        h.update(lats.tobytes())
        h.update(lons.tobytes())
        h.update(b"epsg:4326")
        self.digest = h.digest()

    @classmethod
    def from_dataarray(
        cls,
        da: xr.DataArray,
        *,
        lat_dim: str = "latitude",
        lon_dim: str = "longitude",
    ) -> "GridSpec":
        return cls(lats=da[lat_dim].values, lons=da[lon_dim].values)


def _spherical_cell_areas(lats: np.ndarray, lons: np.ndarray) -> np.ndarray:
    lat_edges = _midpoint_edges(lats)
    lon_edges = _midpoint_edges(lons)
    sin_top = np.sin(np.deg2rad(lat_edges[1:]))
    sin_bot = np.sin(np.deg2rad(lat_edges[:-1]))
    dlon_rad = np.deg2rad(np.diff(lon_edges))
    area_per_lat = (EARTH_RADIUS_M**2) * (sin_top - sin_bot)
    return area_per_lat[:, None] * dlon_rad[None, :]


def _midpoint_edges(centres: np.ndarray) -> np.ndarray:
    if centres.size < 2:
        raise ValueError("need >= 2 coordinates to derive edges")
    mids = (centres[:-1] + centres[1:]) / 2.0
    first = centres[0] - (mids[0] - centres[0])
    last = centres[-1] + (centres[-1] - mids[-1])
    return np.concatenate([[first], mids, [last]])
