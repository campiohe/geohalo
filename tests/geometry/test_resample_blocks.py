import numpy as np

from geohalo.geometry import bilinear_matrix_1d, nearest_index, parent_flat_2d


def test_bilinear_identity_when_same_grid() -> None:
    coords = np.array([0.0, 1.0, 2.0])
    m = bilinear_matrix_1d(coords, coords).toarray()
    np.testing.assert_allclose(m, np.eye(3), atol=1e-12)


def test_bilinear_midpoints_are_half_half() -> None:
    m = bilinear_matrix_1d(np.array([0.0, 1.0]), np.array([0.5])).toarray()
    np.testing.assert_allclose(m, [[0.5, 0.5]])


def test_bilinear_clamps_beyond_edges() -> None:
    m = bilinear_matrix_1d(np.array([0.0, 1.0]), np.array([-1.0, 2.0])).toarray()
    np.testing.assert_allclose(m, [[1.0, 0.0], [0.0, 1.0]])


def test_bilinear_rows_sum_to_one() -> None:
    m = bilinear_matrix_1d(np.linspace(0, 10, 6), np.linspace(-1, 11, 25))
    np.testing.assert_allclose(np.asarray(m.sum(axis=1)).ravel(), 1.0)


def test_nearest_index_basic() -> None:
    src = np.array([0.0, 1.0, 2.0])
    tgt = np.array([0.1, 0.9, 1.4, 1.6, 5.0])
    np.testing.assert_array_equal(nearest_index(src, tgt), [0, 1, 1, 2, 2])


def test_nearest_index_refine_ties_low() -> None:
    src = np.array([0.0, 1.0])
    tgt = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
    np.testing.assert_array_equal(nearest_index(src, tgt), [0, 0, 0, 1, 1])


def test_bilinear_descending_source_matches_ascending() -> None:
    # descending source must give the column-reversed weights of the ascending case.
    asc = np.array([0.0, 1.0, 2.0, 3.0])
    desc = asc[::-1].copy()
    tgt = np.array([0.4, 1.7, 2.9])
    m_asc = bilinear_matrix_1d(asc, tgt).toarray()
    m_desc = bilinear_matrix_1d(desc, tgt).toarray()
    np.testing.assert_allclose(m_desc, m_asc[:, ::-1], atol=1e-12)


def test_nearest_index_descending_source() -> None:
    asc = np.array([0.0, 1.0, 2.0, 3.0])
    desc = asc[::-1].copy()
    tgt = np.array([0.1, 1.4, 2.9])
    asc_idx = nearest_index(asc, tgt)
    desc_idx = nearest_index(desc, tgt)
    # descending source index = (n-1) - ascending index
    np.testing.assert_array_equal(desc_idx, (asc.size - 1) - asc_idx)


def test_bilinear_single_source_cell_maps_all_targets() -> None:
    # A single source coordinate is constant along the axis: every target reads it.
    m = bilinear_matrix_1d(np.array([5.0]), np.array([4.0, 5.0, 6.0])).toarray()
    np.testing.assert_allclose(m, [[1.0], [1.0], [1.0]])


def test_nearest_index_single_source_cell() -> None:
    idx = nearest_index(np.array([5.0]), np.array([4.0, 5.0, 6.0]))
    np.testing.assert_array_equal(idx, [0, 0, 0])


def test_parent_flat_2d_refine() -> None:
    s_lat = np.array([0.0, 1.0])
    s_lon = np.array([0.0, 1.0])
    t_lat = np.array([0.0, 0.4, 0.6, 1.0])
    t_lon = np.array([0.0, 0.4, 0.6, 1.0])
    # nearest parent per axis is [0, 0, 1, 1]; flat = lat_parent * n_s_lon + lon_parent
    expected = np.array(
        [
            [0, 0, 1, 1],
            [0, 0, 1, 1],
            [2, 2, 3, 3],
            [2, 2, 3, 3],
        ]
    ).ravel()
    np.testing.assert_array_equal(parent_flat_2d(s_lat, s_lon, t_lat, t_lon), expected)
