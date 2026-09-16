import hashlib

import numpy as np
import pytest
import scipy.sparse as sp

from geohalo.geometry import bilinear_matrix_1d, grid_digest
from geohalo.reduce_operator import reduce_operator_digest
from geohalo.resampler import FactoredResampler, Resampler, resampler_digest
from tests.geometry.test_periodic import unrolled_weights


@pytest.mark.parametrize("iterations", [1, 3, 6])
@pytest.mark.parametrize("descending_lat", [False, True])
@pytest.mark.parametrize("descending_lon", [False, True])
def test_periodic_factors_materialized_fusion_and_parent_means(iterations, descending_lat, descending_lon):
    lat = np.array([-1., 1.])
    lon = np.arange(-180., 180., 60.)
    if descending_lat:
        lat = lat[::-1]
    if descending_lon:
        lon = lon[::-1]
    target_lat = np.array([1.5, .5, -.5, -1.5])  # latitude remains clamped, not wrapped
    target_lon = np.roll(np.arange(180., 540., 15.)[::-1], 7)
    f = FactoredResampler.compute(lat, lon, target_lat, target_lon, iterations=iterations, period=360)
    r = Resampler.compute(lat, lon, target_lat, target_lon, iterations=iterations, period=360)
    expected_lon, parents = unrolled_weights(lon, target_lon, 360)
    expected_b = np.kron(bilinear_matrix_1d(np.sort(lat), target_lat).toarray(), expected_lon)
    np.testing.assert_allclose(f.b.toarray(), expected_b)
    flat_parent = (np.array([1, 1, 0, 0])[:, None] * lon.size + parents[None, :]).ravel()
    np.testing.assert_array_equal(f.p.toarray(), np.eye(lat.size * lon.size)[flat_parent])
    flat = np.random.default_rng(0).normal(size=(3, lat.size * lon.size))
    expected = flat @ r.transform_matrix.T
    np.testing.assert_allclose(f.apply_flat(flat), expected, atol=2e-14)
    np.testing.assert_allclose(expected @ f.a.T, flat, atol=2e-14)
    np.testing.assert_allclose(r.transform_matrix @ np.ones(flat.shape[1]), 1., atol=2e-14)
    w = sp.random(3, target_lat.size * target_lon.size, density=.1, format="csr", random_state=0)
    np.testing.assert_allclose(f.fuse_left(w).toarray(), (w @ r.transform_matrix).toarray(), atol=2e-14)
    assert f.digest == r.digest
    np.testing.assert_array_equal(r.source_lon, lon)
    np.testing.assert_array_equal(r.target_lon, target_lon)
    equivalent = Resampler.compute(lat, lon, target_lat, target_lon - 720, iterations=iterations, period=360)
    np.testing.assert_allclose(equivalent.transform_matrix.toarray(), r.transform_matrix.toarray(), atol=2e-14)


@pytest.mark.parametrize("builder", [Resampler, FactoredResampler])
def test_periodic_identity_with_descending_longitudes(builder):
    lat, lon = np.arange(3.), np.arange(350., -1., -10.)
    operator = builder.compute(lat, lon, lat, lon, period=360)
    actual = operator.b if builder is FactoredResampler else operator.transform_matrix
    np.testing.assert_array_equal(actual.toarray(), np.eye(lat.size * lon.size))


@pytest.mark.parametrize("target_lon", [np.arange(-180., 180., 120.), np.array([179.5, -179.5])])
def test_periodic_coarsening_and_partial_targets_preserve_occupied_parents(target_lon):
    lat, lon = np.arange(2.), np.arange(-180., 180., 60.)
    f = FactoredResampler.compute(lat, lon, lat, target_lon, period=360, iterations=3)
    r = Resampler.compute(lat, lon, lat, target_lon, period=360, iterations=3)
    occupied = np.asarray(f.a.sum(axis=1)).ravel() > 0
    np.testing.assert_allclose((f.a @ r.transform_matrix).toarray(), np.diag(occupied.astype(float)), atol=1e-14)


def test_digest_backward_compatibility_and_period_isolation():
    lat, lon, tlat, tlon = np.arange(2.), np.arange(-180., 180., 90.), np.arange(3.), np.arange(175., 185.)
    old = hashlib.sha256(b"".join(x.tobytes() for x in (lat, lon, tlat, tlon)) + b"3").digest()
    assert resampler_digest(lat, lon, tlat, tlon, 3) == old
    old_op = hashlib.sha256(b"stencil" + grid_digest(lat, lon) + b"3").digest()
    assert reduce_operator_digest(b"stencil", lat, lon, 3) == old_op
    digests, op_digests = [], []
    for period in (None, 360, 720):
        digests.append(resampler_digest(lat, lon, tlat, tlon, 3, period=period))
        op_digests.append(reduce_operator_digest(b"stencil", lat, lon, 3, period=period))
    assert len(set(digests)) == len(set(op_digests)) == 3
    assert resampler_digest(lat[::-1], lon, tlat, tlon, 3, period=np.float32(360)) == digests[1]
    assert reduce_operator_digest(b"stencil", lat[::-1], lon, 3, period=360.) == op_digests[1]


@pytest.mark.parametrize("builder", [Resampler, FactoredResampler])
def test_periodic_rejects_repeated_source_endpoint(builder):
    with pytest.raises(ValueError, match="omit the repeated endpoint"):
        builder.compute(np.arange(2.), np.array([-180., 0., 180.]), np.arange(2.), np.arange(2.), period=360)
