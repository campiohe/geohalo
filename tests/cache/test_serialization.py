"""_serialize / _deserialize: roundtrip + schema version enforcement."""

import pickle as pk

import numpy as np
import pytest

from geohalo.cache import _deserialize, _serialize
from geohalo.geometry import PolygonSet
from geohalo.grid import GridSpec
from geohalo.weights import compute_weights


def test_roundtrip_preserves_all_fields(
    simple_grid: GridSpec,
    simple_polygons: PolygonSet,
) -> None:
    weights = compute_weights(simple_polygons, simple_grid)
    blob = _serialize(weights)
    restored = _deserialize(blob)
    assert restored.polygon_keys == weights.polygon_keys
    assert restored.key_names == weights.key_names
    assert restored.native_shape == weights.native_shape
    assert restored.grid_digest == weights.grid_digest
    assert restored.polyset_digest == weights.polyset_digest
    assert restored.downscale_factor == weights.downscale_factor
    assert restored.target_resolution == weights.target_resolution
    assert restored.achieved_resolution == weights.achieved_resolution
    np.testing.assert_array_equal(restored.matrix.data, weights.matrix.data)
    np.testing.assert_array_equal(restored.matrix.indices, weights.matrix.indices)
    np.testing.assert_array_equal(restored.matrix.indptr, weights.matrix.indptr)
    assert restored.matrix.shape == weights.matrix.shape


def test_unsupported_version_raises() -> None:
    bad_blob = pk.dumps({"version": 4, "x": "stuff"}, protocol=pk.HIGHEST_PROTOCOL)
    with pytest.raises(ValueError, match="version"):
        _deserialize(bad_blob)


def test_missing_version_raises() -> None:
    bad_blob = pk.dumps({"x": "stuff"}, protocol=pk.HIGHEST_PROTOCOL)
    with pytest.raises(ValueError, match="version"):
        _deserialize(bad_blob)
