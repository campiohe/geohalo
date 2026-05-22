"""compute_bias produces the weighted average at internal nodes."""

from collections.abc import Callable

import numpy as np
import xarray as xr

from geohalo.bias import BiasHierarchy, compute_bias


def test_parent_equals_weighted_average(
    make_leaves_da: Callable[[dict[tuple, float]], xr.DataArray],
) -> None:
    h = BiasHierarchy.build([
        (("p",), ("c1",), 2.0),
        (("p",), ("c2",), 3.0),
    ])
    da = make_leaves_da({("c1",): 10.0, ("c2",): 20.0})
    out = compute_bias(da, h)
    out_idx = out["polygon"].to_index()
    out_values = dict(zip(out_idx, out.values, strict=True))
    np.testing.assert_allclose(out_values[("p",)], (2 * 10 + 3 * 20) / 5)
    assert out_values[("c1",)] == 10.0
    assert out_values[("c2",)] == 20.0
