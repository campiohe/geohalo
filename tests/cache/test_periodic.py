import numpy as np
import pandas as pd
import pytest
import xarray as xr

import geohalo as ghl
from tests.api.test_periodic import seam_inputs


@pytest.fixture(params=["local", pytest.param("redis", marks=pytest.mark.redis)])
def cache(request, tmp_path):
    if request.param == "local":
        return ghl.LocalCache(tmp_path)
    return ghl.RedisCache(request.getfixturevalue("redis_client"))


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_period_cache_roundtrip_isolation_and_no_rebuild(cache, monkeypatch, dtype):
    grid, stencil, geoms = seam_inputs()
    lat, lon = grid.latitude.to_numpy(), grid.longitude.to_numpy()
    stencil = cache.get_or_compute_stencil(stencil.lats, stencil.lons, geoms, dtype=dtype)
    built = {}
    for period in (None, 360, 720):
        resampler = cache.get_or_compute_resampler(lat, lon, stencil.lats, stencil.lons, iterations=3, period=period)
        operator = cache.get_or_compute_reduce_operator(stencil, lat, lon, iterations=3, period=period, dtype=dtype)
        plan = cache.get_or_compute_restricted_operator(operator, lat[::-1], 1, 30)
        built[period] = resampler, operator, plan
    for i in range(3):
        assert len({objects[i].digest for objects in built.values()}) == 3
    assert not np.allclose(built[None][0].transform_matrix.toarray(), built[360][0].transform_matrix.toarray())
    # Forced rebuilding keeps the digest and transform stable.
    forced = cache.get_or_compute_resampler(lat, lon, stencil.lats, stencil.lons,
                                            iterations=3, period=360, force_recompute=True)
    np.testing.assert_array_equal(forced.transform_matrix.toarray(), built[360][0].transform_matrix.toarray())

    def no_build(*_args, **_kwargs):
        pytest.fail("period aliases, latitude orientation, and row-order changes must reuse cached builds")

    for cls in (ghl.Stencil, ghl.Resampler, ghl.ReduceOperator, ghl.RestrictedOperator):
        monkeypatch.setattr(cls, "compute", no_build)
    stencil = cache.get_or_compute_stencil(stencil.lats, stencil.lons, geoms.iloc[::-1], dtype=dtype)
    for period, (reference_resampler, reference_operator, reference_plan) in built.items():
        alias = None if period is None else np.float32(period)
        resampler = cache.get_or_compute_resampler(lat[::-1], lon, stencil.lats, stencil.lons,
                                                  iterations=3, period=alias)
        operator = cache.get_or_compute_reduce_operator(
            stencil, lat[::-1], lon, iterations=3, period=alias, dtype=dtype,
        )
        plan = cache.get_or_compute_restricted_operator(operator, lat[::-1], 1, 30)
        assert resampler.digest == reference_resampler.digest
        np.testing.assert_array_equal(
            resampler.transform_matrix.toarray(), reference_resampler.transform_matrix.toarray(),
        )
        for actual, reference in ((operator, reference_operator), (plan, reference_plan)):
            assert actual.digest == reference.digest
            assert actual.matrix.dtype == reference.matrix.dtype
            pd.testing.assert_index_equal(actual.keys, geoms.iloc[::-1].index)
            np.testing.assert_array_equal(actual.matrix.toarray(), reference.matrix.toarray()[::-1])
            np.testing.assert_array_equal(actual.row_sums, reference.row_sums[::-1])
        descending = grid.isel(latitude=slice(None, None, -1))
        xr.testing.assert_identical(ghl.reduce_with_restricted_operator(descending, plan),
                                    ghl.reduce_with_operator(descending, operator))
