"""_subdivide_centres: refine a 1D coordinate array."""

import numpy as np

from geohalo.downscale import _subdivide_centres


def test_known_factor_2() -> None:
    out = _subdivide_centres(np.array([0.0, 1.0, 2.0]), 2)
    np.testing.assert_allclose(out, [-0.25, 0.25, 0.75, 1.25, 1.75, 2.25])


def test_factor_1_passthrough() -> None:
    out = _subdivide_centres(np.array([0.0, 1.0]), 1)
    np.testing.assert_allclose(out, [0.0, 1.0])
