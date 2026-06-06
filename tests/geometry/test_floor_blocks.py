import numpy as np

from geohalo.geometry import floor_blocks


def test_clip_and_rescale_restores_block_means() -> None:
    parent_flat = np.array([0, 0, 0, 1, 1, 1])
    source = np.array([[1.0, 2.0]])
    # block 0 has a negative child; block 1 is clean
    resampled = np.array([[-1.0, 1.0, 3.0, 2.0, 2.0, 2.0]])
    out = floor_blocks(resampled, source, parent_flat, 0.0)
    assert out.min() >= 0.0
    np.testing.assert_allclose(out[0, :3].mean(), 1.0, atol=1e-12)
    np.testing.assert_allclose(out[0, 3:], [2.0, 2.0, 2.0], atol=1e-12)


def test_noop_when_above_floor_and_mean_matching() -> None:
    parent_flat = np.array([0, 0, 1, 1])
    source = np.array([[2.0, 4.0]])
    resampled = np.array([[1.5, 2.5, 3.0, 5.0]])  # block means already 2.0 and 4.0
    out = floor_blocks(resampled, source, parent_flat, 0.0)
    np.testing.assert_allclose(out, resampled, atol=1e-12)


def test_source_below_floor_fills_block_with_floor() -> None:
    # rule 2: mean preservation and the floor are mutually impossible -> floor wins
    parent_flat = np.array([0, 0, 1, 1])
    source = np.array([[-0.5, 2.0]])
    resampled = np.array([[0.2, -1.2, 2.0, 2.0]])
    out = floor_blocks(resampled, source, parent_flat, 0.0)
    np.testing.assert_array_equal(out[0, :2], [0.0, 0.0])
    np.testing.assert_allclose(out[0, 2:].mean(), 2.0, atol=1e-12)


def test_all_children_clipped_fills_block_with_parent() -> None:
    # rule 3: the rescale is 0/0 -> constant parent fill keeps mean and floor
    parent_flat = np.array([0, 0])
    source = np.array([[1.0]])
    resampled = np.array([[-2.0, -3.0]])
    out = floor_blocks(resampled, source, parent_flat, 0.0)
    np.testing.assert_array_equal(out, [[1.0, 1.0]])


def test_nan_block_stays_nan_neighbours_unaffected() -> None:
    # rule 1: one NaN child blanks its whole block (same footprint as the
    # linear path, where P(x - A@y) broadcasts the NaN residual block-wide)
    parent_flat = np.array([0, 0, 1, 1])
    source = np.array([[1.0, 3.0]])
    resampled = np.array([[np.nan, 0.5, -1.0, 5.0]])
    out = floor_blocks(resampled, source, parent_flat, 0.0)
    assert np.isnan(out[0, :2]).all()
    assert out[0, 2:].min() >= 0.0
    np.testing.assert_allclose(out[0, 2:].mean(), 3.0, atol=1e-12)


def test_nan_source_keeps_block_nan() -> None:
    parent_flat = np.array([0, 0])
    source = np.array([[np.nan]])
    resampled = np.array([[1.0, 2.0]])
    out = floor_blocks(resampled, source, parent_flat, 0.0)
    assert np.isnan(out).all()


def test_nonzero_floor() -> None:
    parent_flat = np.array([0, 0])
    source = np.array([[2.0]])
    resampled = np.array([[0.0, 3.0]])  # 0.0 is below floor=1.0
    out = floor_blocks(resampled, source, parent_flat, 1.0)
    assert out.min() >= 1.0
    np.testing.assert_allclose(out.mean(), 2.0, atol=1e-12)


def test_batched_rows_independent() -> None:
    parent_flat = np.array([0, 0, 1, 1])
    source = np.array([[1.0, 2.0], [3.0, 4.0]])
    resampled = np.array([[-1.0, 1.0, 2.0, 2.0], [3.0, 3.0, -4.0, 4.0]])
    out = floor_blocks(resampled, source, parent_flat, 0.0)
    row0 = floor_blocks(resampled[:1], source[:1], parent_flat, 0.0)
    row1 = floor_blocks(resampled[1:], source[1:], parent_flat, 0.0)
    np.testing.assert_allclose(out, np.vstack([row0, row1]), atol=1e-12)
