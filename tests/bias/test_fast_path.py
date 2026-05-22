"""Fast (no-NaN) path agreement with masked path."""

from collections.abc import Callable

import numpy as np
import xarray as xr

from geohalo.bias import BiasHierarchy, compute_bias


def test_no_nan_fast_path_equals_masked_path_value(
    make_leaves_da: Callable[[dict[tuple, float]], xr.DataArray],
) -> None:
    """With no NaNs anywhere, compute_bias yields the plain weighted average."""
    h = BiasHierarchy.build([
        (("p",), ("c1",), 1.0),
        (("p",), ("c2",), 1.0),
    ])
    da = make_leaves_da({("c1",): 1.0, ("c2",): 3.0})
    out = compute_bias(da, h)
    out_values = dict(zip(out["polygon"].to_index(), out.values, strict=True))
    np.testing.assert_allclose(out_values[("p",)], 2.0)
