"""Separable conservative application: never build a grid-to-grid Kronecker matrix."""

from dataclasses import dataclass
from functools import cached_property
from typing import Literal

import numpy as np
import scipy.sparse as sp

from geohalo._sparse import projection_dtype

type GridBounds = tuple[np.ndarray, np.ndarray]
type ResampleMethod = Literal["meanpreserving", "conservative"]
type Normalization = Literal["destination", "covered"]


def validate_method(
    method: ResampleMethod, iterations: int, normalization: Normalization,
    source_bounds: GridBounds | None, target_bounds: GridBounds | None,
) -> None:
    if method not in ("meanpreserving", "conservative"):
        raise ValueError("method must be 'meanpreserving' or 'conservative'")
    if normalization not in ("destination", "covered"):
        raise ValueError("normalization must be 'destination' or 'covered'")
    if method == "conservative":
        if iterations != 1:
            raise ValueError("iterations only applies to meanpreserving; use iterations=1 with conservative")
    elif normalization != "destination" or source_bounds is not None or target_bounds is not None:
        raise ValueError("normalization and explicit bounds options require method='conservative'")


def canonical_bounds(bounds: GridBounds | None, *, descending_lat: bool = False) -> GridBounds | None:
    if bounds is None:
        return None
    if len(bounds) != 2:
        raise ValueError("grid bounds must be a pair of latitude and longitude edge arrays")
    lat, lon = (np.asarray(axis, dtype=np.float64) for axis in bounds)
    return (lat[::-1] if descending_lat else lat), lon


@dataclass(frozen=True)
class ConservativeGrid:
    latitude: sp.csr_matrix
    longitude: sp.csr_matrix
    normalization: Normalization

    @cached_property
    def axis_coverage(self) -> tuple[np.ndarray, np.ndarray]:
        return tuple(np.asarray(axis.sum(axis=1)).ravel() for axis in (self.latitude, self.longitude))

    @property
    def coverage(self) -> np.ndarray:
        """Fraction of each full target cell covered by the source domain."""
        lat, lon = self.axis_coverage
        return lat[:, None] * lon[None, :]

    def _project(self, values: np.ndarray) -> np.ndarray:
        target_lat, source_lat = self.latitude.shape
        target_lon, source_lon = self.longitude.shape
        latitude_first = target_lat * source_lon
        longitude_first = source_lat * target_lon
        # Minimize the intermediate. On ties, refining ends on latitude so the
        # larger result is C-contiguous for output copies and normalization.
        if latitude_first < longitude_first or (
            latitude_first == longitude_first and target_lat * target_lon <= source_lat * source_lon
        ):
            return (self.longitude @ (self.latitude @ values).T).T
        return self.latitude @ (self.longitude @ values.T).T

    def apply(
        self, values: np.ndarray, *, descending: bool, skipna: bool, preserve_dtype: bool = False,
    ) -> np.ndarray:
        source_shape = self.latitude.shape[1], self.longitude.shape[1]
        if values.shape[-2:] != source_shape:
            raise ValueError(f"expected trailing source shape {source_shape}, got {values.shape}")
        batch_shape = values.shape[:-2]
        target_shape = self.latitude.shape[0], self.longitude.shape[0]
        dtype = projection_dtype(values.dtype, np.dtype(np.float64), None, preserve_dtype=preserve_dtype)
        out = np.empty((*batch_shape, *target_shape), dtype=dtype)
        coverage = self.coverage
        for index in np.ndindex(batch_shape):
            # Flip one slice as a view, not a contiguous copy of the whole batch.
            step = values[index][::-1] if descending else values[index]
            valid = ~np.isnan(step) if skipna else None
            if valid is not None and not valid.all():
                total = self._project(np.where(valid, step, 0))
                denominator = self._project(valid.astype(np.float64))
            else:
                total = self._project(step)
                denominator = coverage
            if skipna or self.normalization == "covered":
                out[index].fill(np.nan)
                np.divide(total, denominator, out=out[index], where=denominator > 0)
            else:
                out[index] = total
                out[index][coverage <= 0] = np.nan
        return out.reshape(*batch_shape, target_shape[0] * target_shape[1])
