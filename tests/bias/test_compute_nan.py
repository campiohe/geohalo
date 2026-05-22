"""compute_bias NaN handling: 'raise' lists parents; 'ignore' renormalizes."""

from collections.abc import Callable

import numpy as np
import pytest
import xarray as xr

from geohalo.bias import BiasHierarchy, compute_bias


def test_raise_mode_lists_parent_keys_not_leaves(
    make_leaves_da: Callable[[dict[tuple, float]], xr.DataArray],
) -> None:
    h = BiasHierarchy.build([
        (("p",), ("c1",), 1.0),
        (("p",), ("c2",), 1.0),
    ])
    da = make_leaves_da({("c1",): np.nan, ("c2",): 1.0})
    with pytest.raises(ValueError, match=r"\('p',\)"):
        compute_bias(da, h, on_nan_child="raise")


def test_ignore_mode_renormalizes_over_finite_children(
    make_leaves_da: Callable[[dict[tuple, float]], xr.DataArray],
) -> None:
    h = BiasHierarchy.build([
        (("p",), ("c1",), 1.0),
        (("p",), ("c2",), 1.0),
    ])
    da = make_leaves_da({("c1",): np.nan, ("c2",): 5.0})
    out = compute_bias(da, h, on_nan_child="ignore")
    out_values = dict(zip(out["polygon"].to_index(), out.values, strict=True))
    # Parent renormalizes to just c2 -> 5.0
    np.testing.assert_allclose(out_values[("p",)], 5.0)


def test_ignore_mode_all_nan_yields_nan_parent(
    make_leaves_da: Callable[[dict[tuple, float]], xr.DataArray],
) -> None:
    h = BiasHierarchy.build([
        (("p",), ("c1",), 1.0),
        (("p",), ("c2",), 1.0),
    ])
    da = make_leaves_da({("c1",): np.nan, ("c2",): np.nan})
    out = compute_bias(da, h, on_nan_child="ignore")
    out_values = dict(zip(out["polygon"].to_index(), out.values, strict=True))
    assert np.isnan(out_values[("p",)])
