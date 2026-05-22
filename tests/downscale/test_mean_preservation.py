"""Per-parent-cell mean preservation — the load-bearing downscale invariant.

`downscale_plane` must reproduce each parent cell's value when its
factor-by-factor children block is averaged.
"""

import numpy as np
from hypothesis import given, settings, strategies as st

from geohalo.downscale import downscale_plane


@given(
    data=st.lists(
        st.lists(
            st.floats(min_value=-100.0, max_value=100.0, allow_nan=False),
            min_size=3, max_size=8,
        ),
        min_size=3, max_size=8,
    ),
    factor=st.integers(min_value=2, max_value=4),
    iterations=st.integers(min_value=1, max_value=4),
)
@settings(max_examples=200, deadline=None)
def test_linear_preserves_per_parent_mean(
    data: list[list[float]],
    factor: int,
    iterations: int,
) -> None:
    rows_eq = all(len(r) == len(data[0]) for r in data)
    if not rows_eq:
        return
    arr = np.asarray(data, dtype=np.float64)
    out = downscale_plane(arr, factor, iterations=iterations)
    n_lat, n_lon = arr.shape
    out_5d = out.reshape(n_lat, factor, n_lon, factor)
    per_parent_mean = out_5d.mean(axis=(1, 3))
    np.testing.assert_allclose(per_parent_mean, arr, rtol=1e-9, atol=1e-9)
