import numpy as np
import pytest

from geohalo.geometry import conservative_matrix_1d


@pytest.mark.parametrize("latitude", [False, True])
@pytest.mark.parametrize("reverse_source", [False, True])
@pytest.mark.parametrize("reverse_target", [False, True])
def test_irregular_explicit_bounds_match_dense_overlap(latitude, reverse_source, reverse_target):
    src_edges = np.array([-4., -2., 0., 1., 5.])
    dst_edges = np.array([-6., -3., -.5, 2., 3., 6., 8.])
    source, target = (src_edges[:-1] + src_edges[1:]) / 2, (dst_edges[:-1] + dst_edges[1:]) / 2
    measured_src = np.sin(np.deg2rad(src_edges)) if latitude else src_edges
    measured_dst = np.sin(np.deg2rad(dst_edges)) if latitude else dst_edges
    expected = np.maximum(0, np.minimum(measured_dst[1:, None], measured_src[None, 1:])
                          - np.maximum(measured_dst[:-1, None], measured_src[None, :-1]))
    expected /= np.diff(measured_dst)[:, None]
    if reverse_source:
        source, src_edges, expected = source[::-1], src_edges[::-1], expected[:, ::-1]
    if reverse_target:
        target, dst_edges, expected = target[::-1], dst_edges[::-1], expected[::-1]
    actual = conservative_matrix_1d(source, target, latitude=latitude,
                                     source_bounds=src_edges, target_bounds=dst_edges)
    np.testing.assert_allclose(actual.toarray(), expected, atol=1e-15)
    assert np.all(actual.data > 0)


@pytest.mark.parametrize("shift", [-720., 0., 1080.])
@pytest.mark.parametrize("reverse", [False, True])
def test_periodic_overlap_crosses_seam(shift, reverse):
    source = np.arange(-180., 180., 2.)
    target = np.array([178.5, 179.5, 180.5, 181.5]) + shift
    if reverse:
        source = source[::-1]
    matrix = conservative_matrix_1d(source, target, period=360)
    expected = np.zeros((4, 180))
    expected[np.arange(4), [179, 0, 0, 1]] = 1
    np.testing.assert_array_equal(matrix.toarray(), expected[:, ::-1] if reverse else expected)


def test_irregular_periodic_source_uses_cyclic_midpoints():
    source = np.array([0., 2., 5.])
    # Source cells are [-2.5, 1], [1, 3.5], [3.5, 7.5], repeating every 10.
    matrix = conservative_matrix_1d(source, np.array([7.5]), period=10,
                                     target_bounds=np.array([7., 8.]))
    np.testing.assert_array_equal(matrix.toarray(), [[.5, 0., .5]])


def test_single_periodic_source_and_whole_cycle_target():
    matrix = conservative_matrix_1d(np.array([10.]), np.array([190.]), period=360,
                                     target_bounds=np.array([10., 370.]))
    np.testing.assert_array_equal(matrix.toarray(), [[1.]])
    full = conservative_matrix_1d(np.arange(4.), np.array([1.5]), period=4,
                                  target_bounds=np.array([-.5, 3.5]))
    np.testing.assert_array_equal(full.toarray(), [[.25, .25, .25, .25]])


def test_latitude_midpoint_edges_clip_at_poles():
    source = np.array([-90., -60., 0., 60., 90.])
    target = np.array([-45., 45.])
    actual = conservative_matrix_1d(source, target, latitude=True).toarray()
    np.testing.assert_allclose(actual.sum(axis=1), 1.)
    np.testing.assert_allclose(actual[0], [1 - np.sin(np.deg2rad(75)),
                                         np.sin(np.deg2rad(75)) - .5, .5, 0., 0.])
    np.testing.assert_allclose(actual[1], actual[0, ::-1])


@pytest.mark.parametrize(("source", "kwargs", "match"), [
    ([], {}, "nonempty"), ([[0., 1.]], {}, "1-D"), ([np.nan, 1.], {}, "finite"),
    ([0., 0.], {}, "monotonic"), ([0., 2., 1.], {}, "monotonic"),
    ([91., 92.], {"latitude": True}, "latitude centres"),
    ([0.], {}, "single-cell"),
    ([0., 1.], {"source_bounds": np.array([-.5, .5])}, "length"),
    ([0., 1.], {"source_bounds": np.array([-.5, .5, np.inf])}, "finite"),
    ([0., 1.], {"source_bounds": np.array([1., 2., 3.])}, "within"),
    ([0., 1.], {"source_bounds": np.array([-.5, -.5, 2.])}, "monotonic"),
    ([0., 1.], {"source_bounds": np.array([1.5, .5, -.5])}, "monotonic"),
    ([0., 1.], {"latitude": True, "source_bounds": np.array([-91., .5, 2.])}, "latitude bounds"),
    ([0., 1.], {"latitude": True, "period": 360}, "latitude cannot"),
    ([0., 1.], {"period": 0}, "period"),
    ([0., 1.], {"period": 360, "source_bounds": np.array([-.5, .5, 1.5])}, "exactly one period"),
    ([0., 1.], {"period": 4, "target_bounds": np.array([-3., .5, 3.])}, "no more than one period"),
    ([90. - 1e-10, 90.], {"latitude": True}, "positive width"),
])
def test_invalid_conservative_geometry(source, kwargs, match):
    with pytest.raises(ValueError, match=match):
        conservative_matrix_1d(np.asarray(source), np.array([0., 1.]), **kwargs)
