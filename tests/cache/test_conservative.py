import io
import json

import numpy as np
import pytest

import geohalo as ghl
from geohalo.cache import _deser_resampler, _ser_resampler
from geohalo.resampler import resampler_digest


@pytest.fixture(params=["local", pytest.param("redis", marks=pytest.mark.redis)])
def cache(request, tmp_path):
    if request.param == "local":
        return ghl.LocalCache(tmp_path)
    return ghl.RedisCache(request.getfixturevalue("redis_client"))


@pytest.mark.parametrize("normalization", ["destination", "covered"])
@pytest.mark.parametrize("period", [None, 360])
def test_separable_cache_roundtrip_canonical_bounds_and_skip_build(cache, monkeypatch, normalization, period):
    lat, lon = np.array([-1., 1.]), np.arange(-135., 180., 90.)
    tlat, tlon = np.array([-.5, .5]), np.arange(-157.5, 180., 45.)
    bounds = np.array([-2., 0., 2.]), np.linspace(-180., 180., 5)
    options = {"method": "conservative", "normalization": normalization, "period": period}
    first = cache.get_or_compute_resampler(lat, lon, tlat, tlon, source_bounds=bounds, **options)
    forced = cache.get_or_compute_resampler(lat, lon, tlat, tlon, source_bounds=bounds,
                                            force_recompute=True, **options)
    assert first.digest == forced.digest
    assert first.digest == resampler_digest(lat, lon, tlat, tlon, 1, source_bounds=bounds, **options)

    def forbidden(*_args, **_kwargs):
        pytest.fail("conservative cache hits must not build either axis")

    monkeypatch.setattr(ghl.Resampler, "compute", forbidden)
    cached = cache.get_or_compute_resampler(lat[::-1], lon, tlat, tlon,
                                            source_bounds=(bounds[0][::-1], bounds[1]), **options)
    assert cached.digest == first.digest
    assert cached.method == "conservative"
    assert cached.normalization == normalization
    assert cached.transform_matrix is None
    np.testing.assert_array_equal(cached.source_lat, lat)
    np.testing.assert_array_equal(cached.coverage, first.coverage)
    values = np.arange(16.).reshape(2, 2, 4)
    values[0, 0, 0] = np.nan
    for skipna in (False, True):
        np.testing.assert_array_equal(cached.apply_grid(values, skipna=skipna), first.apply_grid(values, skipna=skipna))
    for actual, expected in zip(cached.axis_weights, first.axis_weights, strict=True):
        np.testing.assert_array_equal(actual.data, expected.data)
        np.testing.assert_array_equal(actual.indices, expected.indices)
        np.testing.assert_array_equal(actual.indptr, expected.indptr)


def test_method_normalization_bounds_and_period_distinguish_cache_keys(cache):
    lat, lon = np.array([0., 1.]), np.array([0., 90.])
    options = [
        {}, {"method": "conservative"}, {"method": "conservative", "normalization": "covered"},
        {"method": "conservative", "source_bounds": (np.array([-1., .5, 2.]), np.array([-45., 45., 135.]))},
        {"method": "conservative", "target_bounds": (np.array([-1., .5, 2.]), np.array([-45., 45., 135.]))},
        {"method": "conservative", "period": 360},
    ]
    objects = [cache.get_or_compute_resampler(lat, lon, lat, lon, **kwargs) for kwargs in options]
    assert len({obj.digest for obj in objects}) == len(options)
    default = cache.get_or_compute_resampler(lat, lon, lat, lon, method="meanpreserving")
    assert default.digest == objects[0].digest
    np.testing.assert_array_equal(default.transform_matrix.toarray(), objects[0].transform_matrix.toarray())


def test_classic_npz_payload_and_version_validation():
    coords = np.arange(2.)
    classic = ghl.Resampler.compute(coords, coords, coords, coords)
    blob = _ser_resampler(classic)
    with np.load(io.BytesIO(blob), allow_pickle=False) as archive:
        payload = dict(archive)
    metadata = json.loads(payload["metadata"].tobytes())
    assert metadata["version"] == 1
    assert metadata["method"] == "meanpreserving"
    assert "latitude_data" not in payload
    restored = _deser_resampler(blob)
    assert restored.method == "meanpreserving"
    assert restored.digest == classic.digest
    metadata["version"] = 999
    payload["metadata"] = np.frombuffer(json.dumps(metadata).encode(), dtype=np.uint8)
    output = io.BytesIO()
    np.savez(output, **payload)
    with pytest.raises(ValueError, match="unsupported NPZ schema version"):
        _deser_resampler(output.getvalue())


def test_malformed_grid_bounds_rejected():
    coords = np.arange(2.)
    for bounds in ((), (np.arange(3.),), (np.arange(3.),) * 3):
        with pytest.raises(ValueError, match="pair"):
            ghl.Resampler.compute(coords, coords, coords, coords, method="conservative", source_bounds=bounds)
