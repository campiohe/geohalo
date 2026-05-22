"""compute_bias input validation."""

from collections.abc import Callable

import numpy as np
import pytest
import xarray as xr

from geohalo.bias import BiasHierarchy, compute_bias


def test_bad_on_nan_child_raises(
    make_leaves_da: Callable[[dict[tuple, float]], xr.DataArray],
) -> None:
    h = BiasHierarchy.build([(("p",), ("c",), 1.0)])
    da = make_leaves_da({("c",): 1.0})
    with pytest.raises(ValueError, match="on_nan_child must be one of"):
        compute_bias(da, h, on_nan_child="explode")  # type: ignore[arg-type]


def test_missing_polygon_dim_raises() -> None:
    h = BiasHierarchy.build([(("p",), ("c",), 1.0)])
    da = xr.DataArray(np.array([1.0]), dims=("other",))
    with pytest.raises(ValueError, match="has no dim named"):
        compute_bias(da, h)


def test_keynames_mismatch_raises(
    make_leaves_da: Callable[[dict[tuple, float]], xr.DataArray],
) -> None:
    h = BiasHierarchy.build(
        [(("p",), ("c",), 1.0)],
        key_names=("country",),
    )
    da = make_leaves_da({("c",): 1.0})  # named "polygon_id"
    with pytest.raises(ValueError, match="key_names"):
        compute_bias(da, h)


def test_missing_leaves_raises(
    make_leaves_da: Callable[[dict[tuple, float]], xr.DataArray],
) -> None:
    h = BiasHierarchy.build([
        (("p",), ("c1",), 1.0),
        (("p",), ("c2",), 1.0),
    ])
    da = make_leaves_da({("c1",): 1.0})  # missing c2
    with pytest.raises(ValueError, match="missing leaf"):
        compute_bias(da, h)
