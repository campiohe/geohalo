import io
import json

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from geohalo import LocalCache, RedisCache, RestrictedOperator, reduce_with_restricted_operator
from geohalo.cache import _deser_restricted_op
from tests.restricted_operator._helpers import make_grid, make_operator


@pytest.mark.parametrize("descending", [False, True])
@pytest.mark.parametrize("multiindex", [False, True])
def test_local_cache_roundtrip_skips_build(tmp_path, monkeypatch, descending, multiindex):
    operator = make_operator(multiindex=multiindex)
    grid = make_grid(operator, descending=descending)
    cache = LocalCache(tmp_path)
    original = cache.get_or_compute_restricted_operator(operator, grid.latitude.values, 3, 4)

    def no_build(*_args, **_kwargs):
        pytest.fail("cache hit must not build the read plan")

    monkeypatch.setattr(RestrictedOperator, "compute", no_build)
    loaded = cache.get_or_compute_restricted_operator(operator, grid.latitude.values, (3, 3, 2), (4, 4, 2))
    assert loaded.digest == original.digest
    assert loaded.windows == original.windows
    for a, b in zip(loaded.gathers, original.gathers, strict=True):
        np.testing.assert_array_equal(a, b)
    pd.testing.assert_index_equal(loaded.keys, original.keys)
    xr.testing.assert_identical(
        reduce_with_restricted_operator(grid, loaded), reduce_with_restricted_operator(grid, original),
    )


def test_force_recompute(tmp_path, monkeypatch):
    operator = make_operator()
    cache = LocalCache(tmp_path)
    first = cache.get_or_compute_restricted_operator(operator, operator.source_lat, 2, 2)
    calls = []
    compute = RestrictedOperator.compute

    def record(*args, **kwargs):
        calls.append(True)
        return compute(*args, **kwargs)

    monkeypatch.setattr(RestrictedOperator, "compute", record)
    second = cache.get_or_compute_restricted_operator(operator, operator.source_lat, 2, 2, force_recompute=True)
    assert calls == [True]
    assert second.digest == first.digest


def test_unknown_payload_version_rejected():
    output = io.BytesIO()
    metadata = np.frombuffer(json.dumps({"format": "geohalo", "version": 100}).encode(), dtype=np.uint8)
    np.savez(output, metadata=metadata)
    with pytest.raises(ValueError, match="unsupported NPZ schema version"):
        _deser_restricted_op(output.getvalue())


@pytest.mark.redis
def test_redis_roundtrip(redis_client):
    operator = make_operator(multiindex=True)
    cache = RedisCache(redis_client)
    original = cache.get_or_compute_restricted_operator(operator, operator.source_lat, 2, 2)
    loaded = cache.get_or_compute_restricted_operator(operator, operator.source_lat, 2, 2)
    assert loaded.digest == original.digest
    pd.testing.assert_index_equal(loaded.keys, original.keys)
    np.testing.assert_array_equal(loaded.matrix.toarray(), original.matrix.toarray())
    assert loaded.windows == original.windows
