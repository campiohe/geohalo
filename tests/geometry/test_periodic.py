"""Periodic weights agree with independently unrolled, folded source axes."""

import numpy as np
import pytest

from geohalo.geometry import bilinear_matrix_1d, nearest_index, target_coords_from_resolution


def unrolled_weights(source, target, period):
    """Use the unchanged nonperiodic routines as an independent reference."""
    order = np.argsort(source)
    ascending = source[order]
    unrolled = np.concatenate([ascending - period, ascending, ascending + period])
    wrapped = ascending[0] + (target - ascending[0]) % period
    weights = bilinear_matrix_1d(unrolled, wrapped).toarray()
    folded = sum(np.split(weights, 3, axis=1))
    nearest = order[nearest_index(unrolled, wrapped) % source.size]
    return folded[:, np.argsort(order)], nearest


@pytest.mark.parametrize("descending", [False, True])
@pytest.mark.parametrize(("source", "period"), [
    (np.arange(-180., 180., 2.), 360), (np.arange(0., 360., 2.), 360.),
    (np.array([10., 13., 17.5]), 10.), (np.array([-3.]), 10.),
])
def test_periodic_matches_unrolling(source, period, descending):
    source = source[::-1] if descending else source
    target = np.r_[source, source.min() + np.array([-2.25, -.25, 0, .1, .5, .875, 1, 3.25]) * period]
    original = source.copy(), target.copy()
    expected, parents = unrolled_weights(source, target, period)
    matrix = bilinear_matrix_1d(source, target, period=period)
    np.testing.assert_allclose(matrix.toarray(), expected, atol=1e-14)
    np.testing.assert_array_equal(nearest_index(source, target, period=period), parents)
    np.testing.assert_allclose(np.asarray(matrix.sum(axis=1)).ravel(), 1.)
    assert np.all(matrix.data >= 0.)
    assert matrix.nnz <= 2 * target.size
    np.testing.assert_array_equal(source, original[0])
    np.testing.assert_array_equal(target, original[1])


def test_issue_22_seam_and_ties():
    source = np.arange(-180., 180., 2.)
    matrix = bilinear_matrix_1d(source, np.array([179.5]), period=360)
    np.testing.assert_array_equal(matrix.indices, [0, 179])
    np.testing.assert_array_equal(matrix.data, [.75, .25])
    target = np.array([179., 179.5, -181., -180.5, -179., 899.])
    np.testing.assert_array_equal(nearest_index(source, target, period=360), [179, 0, 179, 0, 0, 179])
    # None/default retains edge clamping and the original sparse entry order.
    clamped = bilinear_matrix_1d(source, target)
    explicit = bilinear_matrix_1d(source, target, period=None)
    np.testing.assert_array_equal(explicit.data, clamped.data)
    np.testing.assert_array_equal(explicit.indices, clamped.indices)
    np.testing.assert_array_equal(clamped.toarray()[1, -2:], [0., 1.])
    assert nearest_index(source, np.array([179.5]))[0] == 179


@pytest.mark.parametrize("builder", [bilinear_matrix_1d, nearest_index])
@pytest.mark.parametrize("period", [0, -360, np.nan, np.inf, -np.inf])
def test_invalid_period(builder, period):
    with pytest.raises(ValueError, match="period must be finite and > 0"):
        builder(np.array([0.]), np.array([1.]), period=period)


@pytest.mark.parametrize("builder", [bilinear_matrix_1d, nearest_index])
@pytest.mark.parametrize("source", [
    [], [[0., 1.]], [np.nan], [np.inf], [0., 0.], [0., 2., 1.],
    [-180., 0., 180.], [180., 0., -180.], [0., 361.],
])
def test_invalid_periodic_source(builder, source):
    with pytest.raises(ValueError, match="periodic source coordinates"):
        builder(np.array(source), np.array([1.]), period=360)


@pytest.mark.parametrize("builder", [bilinear_matrix_1d, nearest_index])
@pytest.mark.parametrize("target", [[np.nan], [np.inf], [[0.]]])
def test_invalid_periodic_target(builder, target):
    with pytest.raises(ValueError, match="periodic target coordinates"):
        builder(np.array([0., 2.]), np.array(target), period=360)


def test_empty_periodic_targets():
    assert bilinear_matrix_1d(np.array([0., 1.]), np.array([]), period=360).shape == (0, 2)
    assert nearest_index(np.array([0.]), np.array([]), period=360).shape == (0,)


@pytest.mark.parametrize("start", [-180., 0.])
@pytest.mark.parametrize("resolution", [.5, 2., 7.])
def test_periodic_generated_targets_cover_one_cycle(start, resolution):
    source_lat = np.array([2., 0.])
    source_lon = np.arange(start, start + 360., 2.)[::-1]
    target_lat, target_lon = target_coords_from_resolution(source_lat, source_lon, resolution, period=360)
    np.testing.assert_array_equal(target_lat, target_coords_from_resolution(source_lat, source_lon, resolution)[0])
    np.testing.assert_array_equal(target_lon, start + np.arange(0., 360., resolution))
    assert target_lon[-1] < start + 360
    assert start + 360 - target_lon[-1] <= resolution
