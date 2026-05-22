from collections.abc import Callable

import numpy as np
import pandas as pd
import pytest
import xarray as xr


@pytest.fixture
def make_leaves_da() -> Callable[[dict[tuple, float]], xr.DataArray]:
    """Factory: dict of `{key: value}` -> 1-D DataArray over a polygon MultiIndex."""
    def _make(values: dict[tuple, float]) -> xr.DataArray:
        keys = sorted(values.keys())
        arr = np.array([values[k] for k in keys], dtype=np.float64)
        idx = pd.MultiIndex.from_tuples(keys, names=["polygon_id"])
        return xr.DataArray(arr, dims=("polygon",), coords={"polygon": idx})
    return _make
